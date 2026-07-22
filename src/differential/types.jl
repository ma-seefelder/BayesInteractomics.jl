# Differential Interaction Analysis — Type Definitions
# Compare interaction profiles between two experimental conditions

import DataFrames: DataFrame, nrow
using Dates

# ----------------------- InteractionClass Enum ----------------------- #

"""
    InteractionClass

Enum classifying the differential interaction status of a protein.

# Values
- `GAINED`: Interaction significantly stronger in condition A than B (PP_A ≥ threshold, PP_B < threshold, Δlog₂FC ≥ 0)
- `REDUCED`: Interaction significantly stronger in condition B than A (PP_B ≥ threshold, PP_A < threshold, Δlog₂FC ≤ 0)
- `UNCHANGED`: Both conditions show interaction; no significant directional difference, or ambiguous evidence
- `BOTH_NEGATIVE`: Neither condition detects interaction (PP below threshold), but differential evidence is significant
- `CONDITION_A_SPECIFIC`: Protein detected only in condition A results
- `CONDITION_B_SPECIFIC`: Protein detected only in condition B results
"""
@enum InteractionClass begin
    GAINED
    REDUCED
    UNCHANGED
    BOTH_NEGATIVE
    CONDITION_A_SPECIFIC
    CONDITION_B_SPECIFIC
end

# ----------------------- DifferentialConfig ----------------------- #

"""
    DifferentialConfig

Configuration for differential interaction analysis.

# Statistical Thresholds
- `posterior_threshold::Float64`: Minimum posterior probability to consider a protein
  as a true interactor in a given condition (default: 0.8)
- `bfdr_threshold::Float64`: Maximum BFDR (Bayesian FDR) for significance in differential
  analysis (default: 0.05)
- `delta_log2fc_threshold::Float64`: Minimum absolute difference in mean_log2FC
  to classify as gained/lost when both conditions show interaction (default: 1.0)
- `dbf_threshold::Float64`: Minimum absolute log10 differential Bayes factor to
  classify as gained/lost (default: 1.0, i.e., 10-fold BF difference)
- `classification_method::Symbol`: Method for classifying interactions:
  - `:posterior` (default): use posterior probability thresholds
  - `:dbf`: use differential Bayes factor thresholds
  - `:combined`: require both posterior and dBF criteria
- `standardize_log2fc::Bool`: If `true`, z-score standardize log2FC values within each
  condition before computing `delta_log2fc`. This removes confounding from protocol-level
  scale differences, but it also **destroys the bait-anchor**: z-scoring re-introduces
  per-condition variance, so the bait — whose raw Δlog2FC the anchor zeroes — is rescaled
  back to an extreme z-score in each condition and re-emerges as a spurious top differential
  hit. Default is `false` so that `bait_anchor` (default ON) keeps the bait Δlog2FC at zero;
  `delta_log2fc_threshold` is then a raw log2FC threshold. Set `true` only when no bait is
  defined and cross-protocol scale confounding dominates. When enabled, `delta_log2fc_threshold`
  is interpreted as a z-score threshold rather than a raw log2FC threshold.

# Output Paths (used by the pipeline method `differential_analysis(config_a, config_b)`)
- `results_file::String`: Path for the differential results Excel file (default: `"differential_results.xlsx"`)
- `volcano_file::String`: Path for the volcano plot image (default: `"differential_volcano.png"`)
- `evidence_file::String`: Path for the evidence plot image (default: `"differential_evidence.png"`)
- `scatter_file::String`: Path for the scatter plot image (default: `"differential_scatter.png"`)
- `classification_file::String`: Path for the classification bar chart (default: `"differential_classification.png"`)
- `ma_file::String`: Path for the MA plot (default: `"differential_ma.png"`)
- `generate_report_html::Bool`: Automatically generate an interactive HTML report after analysis (default: `true`)

# Bayesian Decision Risk
- `loss_matrix::Matrix{Float64}`: 4×4 asymmetric loss matrix used by
  `compute_decision_risk!` to compute the expected-risk-minimising optimal call
  per protein. Rows = actions, columns = truth states in
  `DECISION_RISK_ACTIONS` order (`:gained`, `:reduced`, `:unchanged`,
  `:both_negative`). Defaults to `DEFAULT_DIFFERENTIAL_LOSS`. Validated by the
  constructor: must be 4×4, have zero diagonal, all entries non-negative and
  finite. Throws `ArgumentError` on violation.
- `validation_candidates_top_n::Int`: Number of top hits surfaced by the
  Validation Candidates panel in the Decision Risk report tab. Default: `20`
  (locked). Validated `> 0` in the constructor.

# Sensitivity Tab
- `top_n_stability::Int`: Number of top proteins surfaced by the per-protein
  stability-change strip in the Sensitivity report tab. Default: `20`. Validated
  `> 0` in the constructor. Used by `_build_diff_stability_change_strip` (backend)
  and the JS strip renderer (template) for both k=2 and k≥3 reports.

# Examples
```julia
# Default configuration
config = DifferentialConfig()

# Stringent thresholds with custom output paths
config = DifferentialConfig(
    posterior_threshold = 0.95,
    bfdr_threshold = 0.01,
    classification_method = :combined,
    results_file = "results/diff.xlsx",
    volcano_file = "results/volcano.svg"
)
```
"""
Base.@kwdef struct DifferentialConfig
    # Statistical thresholds
    posterior_threshold::Float64     = 0.8
    bfdr_threshold::Float64          = 0.05
    delta_log2fc_threshold::Float64 = 1.0
    dbf_threshold::Float64          = 1.0
    classification_method::Symbol   = :posterior
    # Default OFF: z-scoring undoes the bait-anchor (see field docs above) and re-inflates the
    # bait to a spurious top hit. Raw Δlog2FC + bait_anchor is the correct default pairing.
    standardize_log2fc::Bool        = false

    # Regression-safe per-condition bait-anchor correction (per-condition raw-bait δ
    # subtracted from SAMPLE cells only; controls untouched so the regression dose axis
    # is preserved). Default ON — it is near-inert when bait levels are matched across
    # conditions and corrects bait-abundance differences when present. Set false to
    # disable the correction.
    bait_anchor::Bool               = true

    # Output paths
    results_file::String            = "differential_results.xlsx"
    volcano_file::String            = "differential_volcano.png"
    evidence_file::String           = "differential_evidence.png"
    scatter_file::String            = "differential_scatter.png"
    classification_file::String     = "differential_classification.png"
    ma_file::String                 = "differential_ma.png"

    # Report generation
    generate_report_html::Bool      = true

    # Decision Risk loss matrix override (default: DEFAULT_DIFFERENTIAL_LOSS)
    loss_matrix::Matrix{Float64}    = DEFAULT_DIFFERENTIAL_LOSS

    # Validation Candidates top-N (default: 20)
    validation_candidates_top_n::Int = 20

    # Sensitivity stability strip top-N (default: 20)
    top_n_stability::Int = 20

    function DifferentialConfig(posterior_threshold, bfdr_threshold,
                                delta_log2fc_threshold, dbf_threshold,
                                classification_method, standardize_log2fc,
                                bait_anchor,
                                results_file, volcano_file,
                                evidence_file, scatter_file,
                                classification_file, ma_file,
                                generate_report_html,
                                loss_matrix, validation_candidates_top_n,
                                top_n_stability)
        0.0 <= posterior_threshold <= 1.0 || throw(ArgumentError(
            "posterior_threshold must be in [0, 1], got $posterior_threshold"))
        0.0 < bfdr_threshold <= 1.0 || throw(ArgumentError(
            "bfdr_threshold must be in (0, 1], got $bfdr_threshold"))
        delta_log2fc_threshold >= 0.0 || throw(ArgumentError(
            "delta_log2fc_threshold must be non-negative, got $delta_log2fc_threshold"))
        dbf_threshold >= 0.0 || throw(ArgumentError(
            "dbf_threshold must be non-negative, got $dbf_threshold"))
        classification_method in (:posterior, :dbf, :combined) || throw(ArgumentError(
            "classification_method must be :posterior, :dbf, or :combined, got :$classification_method"))

        # loss_matrix validation
        size(loss_matrix) == (4, 4) || throw(ArgumentError(
            "loss_matrix must be 4×4 in shape, got size $(size(loss_matrix))"))
        all(loss_matrix[i, i] == 0.0 for i in 1:4) || throw(ArgumentError(
            "loss_matrix must have zero diagonal, got diagonal $([loss_matrix[i,i] for i in 1:4])"))
        all(loss_matrix .>= 0.0) || throw(ArgumentError(
            "loss_matrix entries must be non-negative, found negative entry at index $(findfirst(<(0.0), loss_matrix))"))
        all(isfinite, loss_matrix) || throw(ArgumentError(
            "loss_matrix entries must be finite (no Inf/NaN), found non-finite entry at index $(findfirst(!isfinite, loss_matrix))"))

        # validation_candidates_top_n validation
        validation_candidates_top_n > 0 || throw(ArgumentError(
            "validation_candidates_top_n must be > 0, got $validation_candidates_top_n"))

        # top_n_stability validation
        top_n_stability > 0 || throw(ArgumentError(
            "top_n_stability must be > 0, got $top_n_stability"))

        # bait_anchor is a Bool, no constraint needed

        new(posterior_threshold, bfdr_threshold, delta_log2fc_threshold,
            dbf_threshold, classification_method, standardize_log2fc,
            bait_anchor,
            results_file, volcano_file, evidence_file, scatter_file,
            classification_file, ma_file, generate_report_html,
            loss_matrix, validation_candidates_top_n, top_n_stability)
    end
end

function Base.show(io::IO, c::DifferentialConfig)
    println(io, "DifferentialConfig")
    println(io, "  posterior_threshold     : $(c.posterior_threshold)")
    println(io, "  bfdr_threshold         : $(c.bfdr_threshold)")
    println(io, "  delta_log2fc_threshold : $(c.delta_log2fc_threshold)")
    println(io, "  dbf_threshold          : $(c.dbf_threshold)")
    println(io, "  classification_method  : $(c.classification_method)")
    println(io, "  standardize_log2fc    : $(c.standardize_log2fc)")
    println(io, "  bait_anchor            : $(c.bait_anchor)")
    println(io, "  results_file           : $(c.results_file)")
    println(io, "  volcano_file           : $(c.volcano_file)")
    println(io, "  evidence_file          : $(c.evidence_file)")
    println(io, "  scatter_file           : $(c.scatter_file)")
    println(io, "  classification_file    : $(c.classification_file)")
    println(io, "  ma_file                : $(c.ma_file)")
    println(io, "  generate_report_html   : $(c.generate_report_html)")
    println(io, "  loss_matrix             : $(c.loss_matrix == DEFAULT_DIFFERENTIAL_LOSS ? "<default 4×4>" : "<custom 4×4>")")
    println(io, "  validation_candidates_top_n : $(c.validation_candidates_top_n)")
    println(io, "  top_n_stability            : $(c.top_n_stability)")
end

# ----------------------- DifferentialResult ----------------------- #

"""
    DifferentialResult

Complete results from a differential interaction analysis comparing two conditions.

# Fields
## Result DataFrame
- `results::DataFrame`: Per-protein differential statistics. Columns:
  `Protein`, `bf_A`, `bf_B`, `dbf`, `log10_dbf`, `posterior_A`, `posterior_B`,
  `delta_posterior`, `BFDR_A`, `BFDR_B`, `PEP_A`, `PEP_B`, `log2fc_A`, `log2fc_B`, `delta_log2fc`,
  `bf_enrichment_A`, `bf_enrichment_B`, `dbf_enrichment`,
  `bf_correlation_A`, `bf_correlation_B`, `dbf_correlation`,
  `bf_detected_A`, `bf_detected_B`, `dbf_detected`,
  `differential_posterior`, `differential_BFDR`, `diff_PEP`, `classification`

## Condition labels
- `condition_A::String`: Label for condition A
- `condition_B::String`: Label for condition B

## Configuration
- `config::DifferentialConfig`: Configuration used for this analysis

## Metadata
- `n_proteins_A::Int`: Number of proteins in condition A
- `n_proteins_B::Int`: Number of proteins in condition B
- `n_shared::Int`: Number of shared proteins (inner join)
- `n_condition_A_specific::Int`: Proteins only in condition A
- `n_condition_B_specific::Int`: Proteins only in condition B
- `timestamp::DateTime`: When analysis was performed

## Summary counts
- `n_gained::Int`: Interactions gained (stronger in condition A)
- `n_reduced::Int`: Interactions reduced (stronger in condition B)
- `n_unchanged::Int`: Interactions unchanged between conditions
- `n_both_negative::Int`: Proteins not detected in either condition but with significant differential evidence

# Iterator Interface
Iterates over `(protein_name, row_data)` tuples.

# Indexing
- `diff[i]`: Get row i from results (Integer indexing)
- `diff[protein]`: Get row for specific protein (String indexing)

# Examples
```julia
diff = differential_analysis(result_wt, result_mut,
    condition_A = "WT", condition_B = "Mutant")

# Summary
println(diff)

# Iterate
for (protein, row) in diff
    if row.classification == GAINED
        println("\$protein gained in WT")
    end
end

# Index
row = diff["MYC"]
println("dBF: ", row.dbf)
```

See also: [`differential_analysis`](@ref), [`DifferentialConfig`](@ref), [`InteractionClass`](@ref)
"""
struct DifferentialResult
    results::DataFrame
    condition_A::String
    condition_B::String
    config::DifferentialConfig
    n_proteins_A::Int
    n_proteins_B::Int
    n_shared::Int
    n_condition_A_specific::Int
    n_condition_B_specific::Int
    timestamp::DateTime
    n_gained::Int
    n_reduced::Int
    n_unchanged::Int
    n_both_negative::Int
    # per-condition AnalysisResult objects in [condition_A, condition_B]
    # order; nothing for path-based overloads. Populated only by the AR-based
    # differential_analysis(::AbstractAnalysisResult, ::AbstractAnalysisResult)
    # overload. Required by per-condition tab JSON builders (Plans 03/04) which
    # iterate diff.analyses without hardcoding A/B.
    analyses::Union{Nothing, Vector{AnalysisResult}}
    # per-condition calibration provenance (dataset-level ECE-gate
    # scope per AR). Mirrors AnalysisResult.is_calibrated for each side of the
    # differential comparison so the Methods tab and Pitfall-7 mixed-calibration
    # banner can report each condition independently. Both default to false on
    # the 14-arg and 15-arg backward-compat constructors below.
    is_calibrated_A::Bool
    is_calibrated_B::Bool
    # condition-pair similarity matrices (Spearman log10-BF, Pearson
    # log2FC, Pearson posterior, Jaccard@TopK) + n_shared_per_cell hover tooltip +
    # UPGMA dendrogram (extension-gated). `nothing` for path-based overloads or
    # when CONFIG.embeddings_config.run_embeddings == false. Mirrors
    # AnalysisResult.embeddings (single-bait) but at the differential level. The
    # 17-arg backward-compat ctor below forwards `condition_similarity = nothing`
    # so earlier callers keep working.
    condition_similarity::Union{Nothing, ConditionSimilarityResult}
    # ordered list of pairwise contrasts performed by the k-group
    # overload. Empty `Pair{Symbol, Symbol}[]` for legacy 2-group calls;
    # populated with [:c1 => :c2, ...] in NamedTuple-key insertion order for
    # k-group calls. The vector is the *canonical* contrast ordering — downstream
    # code MUST iterate this vector (not `keys(pairwise_results)`) to preserve
    # determinism across parallel pair execution.
    contrasts::Vector{Pair{Symbol, Symbol}}
    # per-pair DataFrame results in the existing 2-group shape.
    # `nothing` for legacy 2-group calls (the wide `results::DataFrame` IS the
    # pairwise result for k=2). For k≥2 NamedTuple calls, contains exactly
    # `length(contrasts)` entries, one per pair. The 18-arg backward-compat ctor
    # below forwards `pairwise_results = nothing`.
    pairwise_results::Union{Nothing, Dict{Pair{Symbol, Symbol}, DataFrame}}
end

# backward-compat constructor: earlier callers (14 positional args)
# keep working with analyses=nothing.
# also defaults is_calibrated_A=false, is_calibrated_B=false.
# also defaults condition_similarity=nothing.
# also defaults contrasts=Pair{Symbol,Symbol}[] and pairwise_results=nothing.
DifferentialResult(results, condition_A, condition_B, config,
                   n_proteins_A, n_proteins_B, n_shared,
                   n_condition_A_specific, n_condition_B_specific,
                   timestamp, n_gained, n_reduced, n_unchanged, n_both_negative) =
    DifferentialResult(results, condition_A, condition_B, config,
                       n_proteins_A, n_proteins_B, n_shared,
                       n_condition_A_specific, n_condition_B_specific,
                       timestamp, n_gained, n_reduced, n_unchanged, n_both_negative,
                       nothing)

# backward-compat constructor: earlier callers (15 positional args,
# i.e. the canonical signature including analyses) keep working with
# is_calibrated_A=false, is_calibrated_B=false defaults.
# also defaults condition_similarity=nothing.
# also defaults contrasts=Pair{Symbol,Symbol}[] and pairwise_results=nothing.
DifferentialResult(results, condition_A, condition_B, config,
                   n_proteins_A, n_proteins_B, n_shared,
                   n_condition_A_specific, n_condition_B_specific,
                   timestamp, n_gained, n_reduced, n_unchanged, n_both_negative,
                   analyses) =
    DifferentialResult(results, condition_A, condition_B, config,
                       n_proteins_A, n_proteins_B, n_shared,
                       n_condition_A_specific, n_condition_B_specific,
                       timestamp, n_gained, n_reduced, n_unchanged, n_both_negative,
                       analyses,
                       false, false, nothing)

# backward-compat constructor: earlier callers (17 positional args,
# i.e. the canonical signature including is_calibrated_A / is_calibrated_B)
# keep working with condition_similarity=nothing default.
# also defaults contrasts=Pair{Symbol,Symbol}[] and pairwise_results=nothing
# by forwarding to the 18-arg ctor below.
DifferentialResult(results, condition_A, condition_B, config,
                   n_proteins_A, n_proteins_B, n_shared,
                   n_condition_A_specific, n_condition_B_specific,
                   timestamp, n_gained, n_reduced, n_unchanged, n_both_negative,
                   analyses, is_calibrated_A, is_calibrated_B) =
    DifferentialResult(results, condition_A, condition_B, config,
                       n_proteins_A, n_proteins_B, n_shared,
                       n_condition_A_specific, n_condition_B_specific,
                       timestamp, n_gained, n_reduced, n_unchanged, n_both_negative,
                       analyses, is_calibrated_A, is_calibrated_B,
                       nothing)

# backward-compat constructor: earlier callers (18 positional args,
# i.e. the canonical signature including condition_similarity) keep working
# with contrasts=Pair{Symbol,Symbol}[] and pairwise_results=nothing defaults. The
# 14-/15-/17-arg ctors above eventually land here via the cascade. The canonical
# signature has 20 positional fields (struct fieldcount = 20).
DifferentialResult(results, condition_A, condition_B, config,
                   n_proteins_A, n_proteins_B, n_shared,
                   n_condition_A_specific, n_condition_B_specific,
                   timestamp, n_gained, n_reduced, n_unchanged, n_both_negative,
                   analyses, is_calibrated_A, is_calibrated_B,
                   condition_similarity) =
    DifferentialResult(results, condition_A, condition_B, config,
                       n_proteins_A, n_proteins_B, n_shared,
                       n_condition_A_specific, n_condition_B_specific,
                       timestamp, n_gained, n_reduced, n_unchanged, n_both_negative,
                       analyses, is_calibrated_A, is_calibrated_B,
                       condition_similarity,
                       Pair{Symbol, Symbol}[],   # contrasts default
                       nothing)                  # pairwise_results default

# ----------------------- Accessors ----------------------- #

"""
    getProteins(r::DifferentialResult) -> Vector{String}

Get vector of protein names from differential results.
"""
getProteins(r::DifferentialResult) = r.results.Protein

"""
    getDifferentialBayesFactors(r::DifferentialResult) -> Vector{Float64}

Get vector of differential Bayes factors (BF_A / BF_B).
"""
getDifferentialBayesFactors(r::DifferentialResult) = r.results.dbf

"""
    getDifferentialPosteriors(r::DifferentialResult) -> Vector{Float64}

Get vector of differential posterior probabilities.
"""
getDifferentialPosteriors(r::DifferentialResult) = r.results.differential_posterior

"""
    getDifferentialBFDR(r::DifferentialResult)

Get vector of differential BFDR (Bayesian FDR) values.
"""
getDifferentialBFDR(r::DifferentialResult) = r.results.differential_BFDR

"""
    getDifferentialQValues(r::DifferentialResult)

Deprecated: use `getDifferentialBFDR` instead.
"""
function getDifferentialQValues(r::DifferentialResult)
    @warn "getDifferentialQValues is deprecated, use getDifferentialBFDR instead" maxlog=1
    return getDifferentialBFDR(r)
end

"""
    getDifferentialPEP(r::DifferentialResult; class::Symbol=:alpha)
        -> Vector{Union{Missing, Float64}}

Return the per-protein differential PEP column.

- `class = :alpha` (default): returns `differential_pep` (α direct complement of `differential_posterior`).
- `class ∈ (:gained, :reduced, :unchanged, :both_negative)`: returns the matching
  γ class-conditional PEP column (`pep_<class>`).

γ-PEP is computed under a documented conditional-independence approximation:
P(state_A, state_B | data) ≈ P(state_A | data) · P(state_B | data). See the
Methods tab subsection for the full formula.

Throws `ArgumentError` for unknown `class` values.
"""
function getDifferentialPEP(r::DifferentialResult; class::Symbol = :alpha)
    col = class === :alpha ? :differential_pep : Symbol("pep_$class")
    hasproperty(r.results, col) || throw(ArgumentError(
        "Unknown PEP class :$class. Valid: :alpha, :gained, :reduced, :unchanged, :both_negative"))
    return r.results[!, col]
end

"""
    isCalibrated(r::DifferentialResult) -> NamedTuple{(:A, :B), Tuple{Bool, Bool}}

Per-condition Platt-calibration provenance. The two underlying `AnalysisResult`
objects (`r.analyses[1]` and `r.analyses[2]`) may have different calibration
outcomes; this NamedTuple reports each independently.
"""
isCalibrated(r::DifferentialResult) = (A = r.is_calibrated_A, B = r.is_calibrated_B)

"""
    getClassifications(r::DifferentialResult) -> Vector{InteractionClass}

Get vector of interaction classifications for all proteins.
"""
getClassifications(r::DifferentialResult) = r.results.classification

"""
    getDeltaLog2FC(r::DifferentialResult) -> Vector{Float64}

Get vector of delta log2 fold changes (log2FC_A - log2FC_B).
"""
getDeltaLog2FC(r::DifferentialResult) = r.results.delta_log2fc

"""
    condition_labels(r::DifferentialResult) -> Vector{String}

Return ordered condition labels for k-aware iteration.

- **Legacy 2-group path** (`isempty(r.contrasts)`): returns `[r.condition_A, r.condition_B]`.
  This is the legacy path; preserved verbatim for backward compatibility.

- **k-group path** (`!isempty(r.contrasts)`): derives the label vector from
  the NamedTuple-key insertion order as recorded in `r.contrasts`. Each `Symbol`
  appears in either `first(pair)` or `last(pair)`; the function deduplicates while
  preserving first-encounter order. Returns `Vector{String}` for downstream HTML
  / JSON consumers (the `Symbol` form lives only in `r.contrasts`).

This function is the canonical label source for the small-multiples report
header, the Decision Risk row labels, and the existing `differential_report.html`
condition-axis labels.
"""
function condition_labels(r::DifferentialResult)
    if isempty(r.contrasts)
        return [r.condition_A, r.condition_B]   # legacy path
    end
    # k-group: iterate r.contrasts (the canonical authoritative ordering;
    # NOT r.pairwise_results keys — Dict iteration order is insertion-order in
    # Julia 1.4+ but tests should not rely on this).
    seen = Set{Symbol}()
    out  = String[]
    for pair in r.contrasts
        for sym in (first(pair), last(pair))
            if !(sym in seen)
                push!(seen, sym)
                push!(out, String(sym))
            end
        end
    end
    return out
end

"""
    getAnalyses(r::DifferentialResult) -> Union{Nothing, Vector{AnalysisResult}}

Return per-condition AnalysisResult vector (or `nothing` for
path-based overloads).
"""
getAnalyses(r::DifferentialResult) = r.analyses

# ----------------------- Iterator Interface ----------------------- #

Base.length(r::DifferentialResult) = nrow(r.results)

Base.iterate(r::DifferentialResult, state=1) = state > length(r) ? nothing :
    ((r.results.Protein[state], r.results[state, :]), state + 1)

Base.getindex(r::DifferentialResult, protein::String) =
    r.results[findfirst(==(protein), r.results.Protein), :]

Base.getindex(r::DifferentialResult, i::Integer) = r.results[i, :]

# ----------------------- Display ----------------------- #

function Base.show(io::IO, r::DifferentialResult)
    println(io, "DifferentialResult")
    println(io, String(repeat(Char(0x2500), 40)))
    println(io, " Condition A           : $(r.condition_A)")
    println(io, " Condition B           : $(r.condition_B)")
    println(io, " Shared proteins       : $(r.n_shared)")
    println(io, " Condition A specific  : $(r.n_condition_A_specific)")
    println(io, " Condition B specific  : $(r.n_condition_B_specific)")
    println(io, " Gained interactions   : $(r.n_gained)")
    println(io, " Reduced interactions  : $(r.n_reduced)")
    println(io, " Unchanged             : $(r.n_unchanged)")
    println(io, " Both negative         : $(r.n_both_negative)")
    println(io, " Timestamp             : $(r.timestamp)")
end
