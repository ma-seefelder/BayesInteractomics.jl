# ═══════════════════════════════════════════════════════════════════════════════
# Docking integration: types, scoring functions, request generation, result parsing
# ═══════════════════════════════════════════════════════════════════════════════

import DataFrames: nrow, hasproperty, eachrow
import JSON3
import Downloads

# ─── Configuration ────────────────────────────────────────────────────────────

"""
    DockingConfig

Configuration for the AlphaFold docking integration pipeline.

# Fields
- `posterior_threshold::Float64`: Minimum posterior probability for candidate selection (default: 0.8)
- `pep_threshold::Float64`: Maximum PEP (Posterior Error Probability) for candidate selection (default: 0.01)
- `max_pairs::Int`: Maximum number of protein pairs to dock (default: 100)
- `max_tokens_per_job::Int`: Maximum token count per AlphaFold Server job (default: 5000)
- `max_jobs_per_batch::Int`: Maximum jobs per request batch (default: 30)
- `iptm_missing_threshold::Float64`: ipTM below this is treated as no interaction (default: 0.20)
- `disorder_threshold::Float64`: Fraction disordered above this sets BF=1 (default: 0.50)
- `parse_full_data::Bool`: Whether to parse full_data JSONs for pDockQ and pLDDT (default: true)
- `cache_dir::String`: Cache directory path (empty string uses default `.bayesinteractomics_cache/docking/`)
- `save_raw_zips::Bool`: Whether to save raw ZIP files (default: false)
- `request_output_dir::String`: Directory for generated request JSON files (default: "docking_requests")
- `verbose::Bool`: Enable verbose logging (default: true)
- `species::Int`: NCBI taxonomy ID for UniProt queries (default: 9606 = human)
"""
Base.@kwdef mutable struct DockingConfig
    # Candidate filtering
    posterior_threshold::Float64 = 0.8
    pep_threshold::Float64      = 0.01
    max_pairs::Int              = 100

    # Token limit
    max_tokens_per_job::Int     = 5000
    max_jobs_per_batch::Int     = 30

    # Scoring / calibration
    iptm_missing_threshold::Float64 = 0.20
    disorder_threshold::Float64     = 0.50
    parse_full_data::Bool           = true   # Parse full_data JSONs for pDockQ + pLDDT

    # Cache
    cache_dir::String           = ""        # Empty → .bayesinteractomics_cache/docking/
    save_raw_zips::Bool         = false
    plddt_cache_max_age_days::Int = 7   # AlphaFold DB pLDDT cache TTL; past this a version check is run
    plddt_force_refresh::Bool   = false # If true, ignore cache and refetch all pLDDT data

    # Output
    request_output_dir::String  = "docking_requests"
    verbose::Bool               = true

    # Species (for UniProt ID resolution)
    species::Int                = 9606   # NCBI taxonomy ID (9606 = human)

    # Candidate prioritization
    dockability_weight::Float64 = 0.3  # Weight for structural dockability in composite score (0=posterior only, 1=dockability only)
end

# ─── Calibration ──────────────────────────────────────────────────────────────

"""
    DockingCalibration

Calibration parameters for converting docking scores to Bayes factors.

# Fields
- `clamp_range::Tuple{Float64, Float64}`: Min/max bounds for the docking Bayes factor (default: (0.1, 20.0))
- `iptm_missing_threshold::Float64`: ipTM threshold below which the protein pair is considered non-interacting
"""
struct DockingCalibration
    clamp_range::Tuple{Float64, Float64}
    iptm_missing_threshold::Float64
end

# ─── Result types ─────────────────────────────────────────────────────────────

"""
    DockingPairResult

Docking results for a single protein pair, including structural scores and the computed Bayes factor.

# Fields
- `protein_a::String`, `protein_b::String`: Protein identifiers
- `uniprot_a::String`, `uniprot_b::String`: UniProt accession numbers
- `iptm_best::Float64`: Best ipTM score across models
- `iptm_all::Vector{Float64}`: ipTM scores from all models
- `iptm_std::Float64`: Standard deviation of ipTM across models
- `ranking_score::Float64`: AlphaFold ranking score
- `fraction_disordered::Float64`: Fraction of interface residues that are disordered
- `chain_pair_iptm::Float64`: Chain-pair ipTM from summary JSON
- `chain_pair_pae_min::Float64`: Minimum PAE for the chain pair
- `pdockq::Float64`: pDockQ confidence score (NaN if full_data not parsed)
- `mean_plddt_a::Float64`, `mean_plddt_b::Float64`: Mean pLDDT per chain (NaN if full_data not parsed)
- `n_interface_contacts::Int`: Number of interface contacts (0 if full_data not parsed)
- `bf_dock::Float64`: Computed docking Bayes factor
- `calibration_tier::String`: Scoring tier used (`"tier1_iptm"` or `"tier2_pdockq"`)
- `status::Symbol`: Processing status
- `token_count::Int`: Token count for this pair
"""
struct DockingPairResult
    protein_a::String
    protein_b::String
    uniprot_a::String
    uniprot_b::String
    iptm_best::Float64
    iptm_all::Vector{Float64}
    iptm_std::Float64
    ranking_score::Float64
    fraction_disordered::Float64
    chain_pair_iptm::Float64
    chain_pair_pae_min::Float64
    pdockq::Float64                    # NaN if full_data not parsed
    mean_plddt_a::Float64              # NaN if full_data not parsed
    mean_plddt_b::Float64              # NaN if full_data not parsed
    n_interface_contacts::Int          # 0 if full_data not parsed
    bf_dock::Float64
    calibration_tier::String           # "tier1_iptm", "tier2_pdockq", or "tier2_c2qscore"
    status::Symbol
    token_count::Int
    c2qscore::Float64              # C2Qscore value (NaN if not computed)
    ipae::Float64                  # Raw iPAE before normalization (NaN if not computed)
    iplddt_interface::Float64      # Interface ipLDDT / 100 (NaN if not computed)
end

"""
    DockingResult

Aggregate results from the docking integration pipeline.

# Fields
- `pairs::Vector{DockingPairResult}`: Per-pair docking results
- `config::DockingConfig`: Configuration used for this run
- `n_total::Int`: Total number of candidate pairs
- `n_docked::Int`: Number of pairs with docking results
- `n_cached::Int`: Number of pairs loaded from cache
- `n_pending::Int`: Number of pairs awaiting AlphaFold results
- `n_too_large::Int`: Number of pairs exceeding token limit
- `n_disordered::Int`: Number of pairs skipped due to high disorder
- `timestamp::DateTime`: When the docking run was performed
"""
struct DockingResult
    pairs::Vector{DockingPairResult}
    config::DockingConfig
    n_total::Int
    n_docked::Int
    n_cached::Int
    n_pending::Int
    n_too_large::Int
    n_disordered::Int
    timestamp::DateTime
end

"""
    DockingRequestBatch

Output from [`generate_docking_requests`](@ref), containing paths to generated AlphaFold Server request JSON files.

# Fields
- `batch_dirs::Vector{String}`: Directories containing request JSON files (one per batch)
- `n_requests::Int`: Total number of request files generated
- `n_batches::Int`: Number of batches
- `n_skipped_cached::Int`: Pairs skipped because results were already cached
- `n_skipped_too_large::Int`: Pairs skipped because they exceed the token limit
- `manifest_path::String`: Path to the manifest CSV listing all requests
- `guide_path::String`: Path to the user guide markdown file
"""
struct DockingRequestBatch
    batch_dirs::Vector{String}
    n_requests::Int
    n_batches::Int
    n_skipped_cached::Int
    n_skipped_too_large::Int
    manifest_path::String
    guide_path::String
end

# ─── Cache key ────────────────────────────────────────────────────────────────

"""
    docking_cache_key(id_a, id_b) -> String

Canonical pair key for docking cache. Order-independent.
"""
function docking_cache_key(id_a::AbstractString, id_b::AbstractString)::String
    sorted = sort([uppercase(strip(String(id_a))), uppercase(strip(String(id_b)))])
    return join(sorted, "__")
end

# ─── pDockQ computation (Bryant et al. 2022) ─────────────────────────────────

"""
    compute_pdockq(avg_interface_plddt, n_interface_contacts) -> Float64

Compute pDockQ score from interface pLDDT and contact count.
Uses the exact published sigmoid parameters from Bryant et al. 2022.
"""
function compute_pdockq(avg_interface_plddt::Float64, n_interface_contacts::Int)::Float64
    n_interface_contacts <= 0 && return 0.0
    x = avg_interface_plddt * log(n_interface_contacts)
    return 0.707 / (1.0 + exp(-0.03148 * (x - 388.06))) + 0.03138
end

# ─── Default calibration ─────────────────────────────────────────────────────

"""
    default_calibration() -> DockingCalibration

Returns default calibration based on published data from Bryant et al. 2022
and Burke et al. 2023.
"""
function default_calibration()::DockingCalibration
    return DockingCalibration((0.1, 20.0), 0.20)
end

# ─── C2Qscore (Genz et al. 2025) ─────────────────────────────────────────────

const C2QSCORE_AF3_WEIGHTS = [-0.036, 0.169, 0.335, 0.683]  # [ipLDDT, iPAE, pTM, ipTM]
const C2QSCORE_AF3_BIAS = -0.331
const C2QSCORE_LOGISTIC_COEF = 8.7486
const C2QSCORE_LOGISTIC_INTERCEPT = -2.3721

"""
    compute_c2qscore(iplddt_norm, ipae_norm, ptm, iptm) -> Float64

Compute C2Qscore (4-metric variant, no VoroIF) from normalized metrics.

Inputs should be pre-normalized:
- `iplddt_norm`: interface pLDDT / 100
- `ipae_norm`: 1 - (raw_iPAE / 31.75)
- `ptm`: pTM score (as-is)
- `iptm`: ipTM score (as-is)

Reference: Genz et al. 2025, AF3-specific coefficients.
"""
function compute_c2qscore(iplddt_norm::Float64, ipae_norm::Float64,
                          ptm::Float64, iptm::Float64)::Float64
    return C2QSCORE_AF3_BIAS +
           C2QSCORE_AF3_WEIGHTS[1] * iplddt_norm +
           C2QSCORE_AF3_WEIGHTS[2] * ipae_norm +
           C2QSCORE_AF3_WEIGHTS[3] * ptm +
           C2QSCORE_AF3_WEIGHTS[4] * iptm
end

"""
    compute_bf_from_c2qscore(c2qscore; base_rate=0.15) -> Float64

Compute Bayes factor from C2Qscore using logistic calibration fitted to
Genz et al. 2025 AF3 benchmark (1265 models, DockQ >= 0.23).

No clamping applied — the logistic calibration is well-calibrated.
"""
function compute_bf_from_c2qscore(c2qscore::Float64; base_rate::Float64 = 0.15)::Float64
    p_correct = 1.0 / (1.0 + exp(-(C2QSCORE_LOGISTIC_INTERCEPT + C2QSCORE_LOGISTIC_COEF * c2qscore)))
    odds_correct = p_correct / (1.0 - p_correct)
    odds_prior = base_rate / (1.0 - base_rate)
    return odds_correct / odds_prior  # No clamp for Tier 2 — logistic calibration is well-calibrated
end

# ─── BF computation from pDockQ (Tier 2) ─────────────────────────────────────

"""
    compute_bf_from_pdockq(pdockq; base_rate=0.15) -> Float64

Compute Bayes factor from pDockQ using logistic calibration fitted to
Burke et al. 2023 empirical data points:
  pDockQ=0.0 → P≈0.003, pDockQ=0.23 → P≈0.70, pDockQ=0.50 → P≈0.80

Clamped to [0.1, 20.0].
"""
function compute_bf_from_pdockq(pdockq::Float64; base_rate::Float64 = 0.15)::Float64
    # Logistic: P(correct | pDockQ) fitted to Burke et al. data
    p_correct = 1.0 / (1.0 + exp(-(-4.56 + 12.8 * pdockq)))

    # Convert to BF relative to base rate
    odds_correct = p_correct / (1.0 - p_correct)
    odds_prior = base_rate / (1.0 - base_rate)
    bf = odds_correct / odds_prior

    return clamp(bf, 0.1, 20.0)
end

# ─── BF computation from ipTM (Tier 1) ───────────────────────────────────────

"""
    compute_bf_from_iptm(iptm) -> Float64

Conservative step-function BF from ipTM (fallback when full_data unavailable).
Based on Yin et al. 2022 and Burke et al. 2023 threshold correspondence.
"""
function compute_bf_from_iptm(iptm::Float64)::Float64
    iptm < 0.20 && return 1.0   # Below noise floor
    iptm < 0.40 && return 0.7   # Weak evidence against
    iptm < 0.60 && return 1.5   # Ambiguous
    iptm < 0.80 && return 5.0   # Moderate confidence
    return 12.0                  # High confidence
end

# ─── Main BF computation with quality gates ───────────────────────────────────

"""
    compute_bf_dock(iptm, fraction_disordered; pdockq=NaN, iptm_std=0.0,
                    chain_pair_pae_min=NaN, mean_plddt_min=NaN,
                    calibration=default_calibration()) -> (Float64, String)

Compute clamped Bayes factor from docking scores. Returns `(bf_dock, calibration_tier)`.

Quality gates applied in order:
1. `fraction_disordered > 0.5` → BF=1.0, tier="disordered"
2. `mean_plddt_min < 50` → clamp to [0.7, 1.5] (Tier 1 only; Tier 2 scores already
   incorporate interface pLDDT via their published calibration)
3. `chain_pair_pae_min > 25` → clamp to [0.5, 2.0]
4. `iptm_std > 0.15` → BF dampened toward 1.0
5. If `c2qscore` available → Tier 2 C2Qscore logistic calibration
6. Else if `pdockq` available → Tier 2 pDockQ logistic calibration
7. Otherwise → Tier 1 ipTM step function
"""
function compute_bf_dock(iptm::Float64, fraction_disordered::Float64;
                         pdockq::Float64 = NaN,
                         c2qscore::Float64 = NaN,
                         iptm_std::Float64 = 0.0,
                         chain_pair_pae_min::Float64 = NaN,
                         mean_plddt_min::Float64 = NaN,
                         calibration::DockingCalibration = default_calibration())::Tuple{Float64, String}

    # Gate 1: Disorder
    if fraction_disordered > 0.50
        return (1.0, "disordered")
    end

    # Compute raw BF — prefer C2Qscore > pDockQ > ipTM
    if !isnan(c2qscore)
        bf_raw = compute_bf_from_c2qscore(c2qscore)
        tier = "tier2_c2qscore"
    elseif !isnan(pdockq) && pdockq >= 0.0
        bf_raw = compute_bf_from_pdockq(pdockq)
        tier = "tier2_pdockq"
    else
        bf_raw = compute_bf_from_iptm(iptm)
        tier = "tier1_iptm"
    end

    # Gate 2: pLDDT quality (Tier 1 only)
    # Tier 2 scores (C2Qscore, pDockQ) already incorporate interface pLDDT in their
    # published calibration, so an additional whole-chain pLDDT clamp is redundant
    # and unfairly penalizes large, partially disordered proteins with a confident
    # docked interface.
    if tier == "tier1_iptm" && !isnan(mean_plddt_min) && mean_plddt_min < 50.0
        bf_raw = clamp(bf_raw, 0.7, 1.5)
    end

    # Gate 3: PAE check
    if !isnan(chain_pair_pae_min) && chain_pair_pae_min > 25.0
        bf_raw = clamp(bf_raw, 0.5, 2.0)
    end

    # Gate 4: Model agreement
    if iptm_std > 0.15
        damping = max(0.0, 1.0 - (iptm_std - 0.10) / 0.15)
        bf_raw = 1.0 + (bf_raw - 1.0) * damping
    end

    # Final clamping — skip for C2Qscore (logistic calibration is well-calibrated)
    if tier != "tier2_c2qscore"
        bf_raw = clamp(bf_raw, calibration.clamp_range[1], calibration.clamp_range[2])
    end

    return (bf_raw, tier)
end

# ─── Two-stage Bayesian update ────────────────────────────────────────────────

"""
    _bayesian_update_log_odds(pp_ms::Float64, bf_dock::Float64, epsilon::Float64)::Float64

Perform Bayesian update in log-odds space for numerical stability.
Clamps posterior to [epsilon, 1-epsilon] before computing odds to handle P=0 and P=1.
"""
function _bayesian_update_log_odds(pp_ms::Float64, bf_dock::Float64, epsilon::Float64)::Float64
    pp_clamped = clamp(pp_ms, epsilon, 1.0 - epsilon)
    log_odds_prior = log(pp_clamped / (1.0 - pp_clamped))
    log_odds_post = log_odds_prior + log(bf_dock)
    return 1.0 / (1.0 + exp(-log_odds_post))
end

"""Derive epsilon from minimum non-zero q-value in the dataset. Fallback: 1e-10."""
function _derive_epsilon(q_values)::Float64
    min_q = Inf
    for qv in q_values
        if !ismissing(qv) && isfinite(qv) && qv > 0.0
            min_q = min(min_q, qv)
        end
    end
    return isinf(min_q) ? 1e-10 : min_q / 10.0
end

"""
    apply_docking_update(results, docking) -> DataFrame

Merge docking BF_dock into results as a two-stage Bayesian update.
Adds columns: `posterior_prob_ms`, `bf_docking`, `posterior_prob_combined`,
plus all docking score columns.

Safe to call without docking data (returns unmodified copy).
"""
function apply_docking_update(results::DataFrame, docking::DockingResult)::DataFrame
    df = copy(results)

    # Preserve original MS posterior
    df.posterior_prob_ms = copy(df.posterior_prob)

    # Initialize docking columns
    n = nrow(df)
    df.bf_docking              = ones(Float64, n)
    df.posterior_prob_combined  = copy(df.posterior_prob)
    df.iptm_best               = fill(NaN, n)
    df.iptm_std                = fill(NaN, n)
    df.pdockq                  = fill(NaN, n)
    df.ranking_score_dock      = fill(NaN, n)
    df.fraction_disordered     = fill(NaN, n)
    df.chain_pair_iptm         = fill(NaN, n)
    df.chain_pair_pae_min      = fill(NaN, n)
    df.mean_plddt_a            = fill(NaN, n)
    df.mean_plddt_b            = fill(NaN, n)
    df.n_interface_contacts    = zeros(Int, n)
    df.calibration_tier        = fill("", n)
    df.docking_status          = fill("no_result", n)
    df.c2qscore                = fill(NaN, n)
    df.ipae                    = fill(NaN, n)
    df.iplddt_interface        = fill(NaN, n)

    # Derive epsilon from minimum non-zero BFDR value
    epsilon = hasproperty(df, :BFDR) ? _derive_epsilon(df.BFDR) : 1e-10

    # Build protein lookup: name → row index
    protein_idx = Dict{String, Int}()
    for (i, name) in enumerate(df.Protein)
        protein_idx[name] = i
    end

    # Apply docking results
    for p in docking.pairs
        # Find the prey protein in the results table
        # protein_a is typically the bait, protein_b the prey
        idx = get(protein_idx, p.protein_b, nothing)
        if idx === nothing
            idx = get(protein_idx, p.protein_a, nothing)
        end
        idx === nothing && continue

        df.bf_docking[idx]           = p.bf_dock
        df.iptm_best[idx]            = p.iptm_best
        df.iptm_std[idx]             = p.iptm_std
        df.pdockq[idx]               = p.pdockq
        df.ranking_score_dock[idx]   = p.ranking_score
        df.fraction_disordered[idx]  = p.fraction_disordered
        df.chain_pair_iptm[idx]      = p.chain_pair_iptm
        df.chain_pair_pae_min[idx]   = p.chain_pair_pae_min
        df.mean_plddt_a[idx]         = p.mean_plddt_a
        df.mean_plddt_b[idx]         = p.mean_plddt_b
        df.n_interface_contacts[idx] = p.n_interface_contacts
        df.calibration_tier[idx]     = p.calibration_tier
        df.docking_status[idx]       = string(p.status)
        df.c2qscore[idx]            = p.c2qscore
        df.ipae[idx]                = p.ipae
        df.iplddt_interface[idx]    = p.iplddt_interface

        # Two-stage Bayesian update in log-odds space
        pp_ms = df.posterior_prob_ms[idx]
        if isfinite(pp_ms) && isfinite(p.bf_dock)
            df.posterior_prob_combined[idx] = _bayesian_update_log_odds(pp_ms, p.bf_dock, epsilon)
        end
    end

    # Recompute BFDR values globally after docking update
    df.BFDR_combined = bfdr(df.posterior_prob_combined, isBF=false)

    return df
end

# ─── Implementation files ─────────────────────────────────────────────────────

include("cache.jl")
include("sequence_retrieval.jl")
include("request_generator.jl")
include("alphafold_db.jl")
include("full_data_parser.jl")
include("result_parser.jl")
