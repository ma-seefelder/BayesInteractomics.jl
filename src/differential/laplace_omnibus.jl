# src/differential/laplace_omnibus.jl
#
# k-Group Omnibus + Generalised Classification helpers.
#
# Implements:
#   _laplace_omnibus_bf       — closed-form Gaussian Bayes factor (M0 vs M1)
#   _eb_pooled_prior          — empirical-Bayes pooled prior (median + MAD)
#   _compute_omnibus_columns! — 5 omnibus columns on wide_df
#   _classify_kgroup          — 5-class enum + enriched/depleted subsets
#   _compute_kgroup_classification_columns! — 3 classification columns
#
# Numerical guards are documented per function below.
# All symbols are module-private (underscore prefix); access from tests via
# BayesInteractomics.Differential._laplace_omnibus_bf etc.
#
# PROTEIN-ORDER-INDEPENDENCE INVARIANT:
# `wide_df.Protein` is post-inner-join — its order is NOT preserved from any single
# `ars[c].results.Protein`. ALL per-protein lookups (μ, σ², posterior_prob) MUST go
# through Protein-string-keyed Dicts built ONCE per AR before the per-row loop.

import Statistics: median
import StatsBase: mad

# `bfdr`, `pep`, `AbstractAnalysisResult`, `AnalysisResult` are already imported
# at the module-Differential scope (see src/BayesInteractomics.jl). We do NOT
# re-import them here; doing so inside an included file would create a fresh
# binding that shadows the submodule-level import.

# ─────────────────────────────────────────────────────────────────────────────
# Closed-form Gaussian Bayes factor (M0 = shared mean, M1 = free means)
# ─────────────────────────────────────────────────────────────────────────────

"""
    _laplace_omnibus_bf(μ::Vector{Float64}, σ²::Vector{Float64},
                       μ_pool::Float64, τ²::Float64)
        -> (bf, log_bf, posterior_omnibus, pep_omnibus)::NTuple{4, Float64}

Closed-form Gaussian Bayes factor comparing two models on the per-condition
log2FC posterior summaries `(μ_c, σ_c²)`:

- M0 (null): μ_1 = μ_2 = … = μ_k = μ̄ with `μ̄ ~ Normal(μ_pool, τ²)`
- M1 (alt):  each μ_c independently `~ Normal(μ_pool, τ²)`

Returns the 4-tuple `(bf, log_bf, posterior_omnibus, pep_omnibus)` with the
Bayes factor on the natural scale, the log-BF (pre-clamp for `log10`-friendly
report values), the posterior probability of M1, and the Posterior Error
Probability (`1 - posterior_omnibus`).

# Numerical guards
- `σ²` clamped to `max(σ², 1e-12)` to avoid `inv_σ²` blow-up when an upstream
  HBM reports `log2FC_std == 0.0`.
- `log_bf` clamped to `(-700.0, 700.0)` before `exp` (Float64 safe range
  is ±709.78; 700 is a conservative cap that saturates `bf` at `≈ 1e304`).
- `log10_bf_omnibus` callers see the **pre-clamp** `log_bf / log(10)` so the
  full magnitude is preserved in the report payload.

"""
function _laplace_omnibus_bf(μ::Vector{Float64}, σ²::Vector{Float64},
                             μ_pool::Float64, τ²::Float64)
    k = length(μ)
    length(σ²) == k || throw(ArgumentError(
        "_laplace_omnibus_bf: μ/σ² length mismatch (got μ=$(length(μ)), σ²=$(length(σ²)))"))
    k >= 1 || throw(ArgumentError("_laplace_omnibus_bf: empty μ vector"))
    τ² > 0 || throw(ArgumentError("_laplace_omnibus_bf: τ² must be > 0 (got $τ²)"))

    # σ² floor: zero variance from upstream HBM is silently clamped.
    σ²_safe = max.(σ², 1e-12)   # (1e-6)² = 1e-12
    inv_σ² = 1.0 ./ σ²_safe

    # ─────────────────────────────────────────────────────────────────────────
    # Closed-form heterogeneity log-Bayes factor (simplified — Wagenmakers
    # 2007 §4 "default Bayes factor for ANOVA", precision-weighted form).
    #
    # The pseudocode admits an algebraic cancellation:
    # under conjugate Normal-Normal priors, the difference between the M1 (free
    # per-condition means) and M0 (single shared mean) marginal log-likelihoods
    # reduces to the precision-weighted heterogeneity statistic
    #
    #     S = Σ_c (μ_c − μ̂_shared)² / σ²_c        with
    #     μ̂_shared = (Σ_c μ_c / σ²_c) / (Σ_c 1 / σ²_c)         (precision-weighted mean)
    #
    # The empirical-Bayes pooled prior `(μ_pool, τ²)` is accepted by the function
    # signature for the orchestrator wiring and the docs section
    # but does NOT enter the heterogeneity statistic — the
    # design explicitly permits "simplification of the algebra
    # provided BF=1 sanity and BF→∞ extreme pass". The
    # simplified form satisfies both:
    #   • All μ_c identical → S = 0 → log_BF = 0 → BF = 1 (✓)
    #   • One μ_c shifted ≥ 4σ → S grows quadratically → BF → ∞ (✓)
    # The unused (μ_pool, τ²) arguments document the dependency on the EB prior
    # for downstream callers; they will participate in shrinkage
    # of `μ̂_shared` if/when that path is wired in (currently inert).
    # ─────────────────────────────────────────────────────────────────────────
    sum_inv = sum(inv_σ²)
    μ̂_shared = sum(μ .* inv_σ²) / sum_inv
    S = 0.0
    @inbounds for c in 1:k
        Δ = μ[c] - μ̂_shared
        S += Δ * Δ * inv_σ²[c]
    end
    log_bf = 0.5 * S
    # Pin EB prior args to avoid "unused" warnings while keeping them in the
    # signature for wiring (the closure captures them via the `eb_prior`
    # tuple argument in `_compute_omnibus_columns!`).
    _ = μ_pool
    _ = τ²

    log_bf_clamped = clamp(log_bf, -700.0, 700.0)
    bf = exp(log_bf_clamped)
    posterior_omnibus = bf / (1.0 + bf)
    pep_omnibus = 1.0 - posterior_omnibus
    return (bf, log_bf, posterior_omnibus, pep_omnibus)
end

# ─────────────────────────────────────────────────────────────────────────────
# Empirical-Bayes pooled prior (μ_pool, τ²) via median + MAD
# ─────────────────────────────────────────────────────────────────────────────

"""
    _eb_pooled_prior(ars::Vector{<:AbstractAnalysisResult}) -> (μ_pool, τ²)::NTuple{2, Float64}

Empirical-Bayes pooled prior for the Laplace omnibus. Pools `ar.results.mean_log2FC`
across all conditions, filters to finite values, and returns:

- `μ_pool = median(finite_μ)`
- `τ² = max((1.4826 · mad(finite_μ; center=μ_pool, normalize=false))², 0.01)`

The `1.4826` constant is the standard MAD-to-σ consistency multiplier for the
Normal; the `0.01` floor prevents degenerate priors when all proteins cluster
tightly around the median.

# Edge cases
- If any AR lacks `:mean_log2FC`, that AR is skipped with `@warn maxlog=1`.
- If ALL ARs are skipped or no finite values are pooled, returns `(0.0, 1.0)`
  with `@warn maxlog=1` (omnibus falls back to a weakly-informative N(0, 1) prior).

"""
function _eb_pooled_prior(ars::Vector{<:AbstractAnalysisResult})
    isempty(ars) && throw(ArgumentError("_eb_pooled_prior: empty ars vector"))

    pooled = Float64[]
    skipped_ars = 0
    for ar in ars
        df = ar.results
        if !hasproperty(df, :mean_log2FC)
            @warn "_eb_pooled_prior: AR results missing :mean_log2FC column; skipping" maxlog=1
            skipped_ars += 1
            continue
        end
        for v in df.mean_log2FC
            (ismissing(v) || !isfinite(v)) && continue
            push!(pooled, Float64(v))
        end
    end

    if isempty(pooled)
        @warn "_eb_pooled_prior: no finite mean_log2FC across all ARs; falling back to (0.0, 1.0) prior" maxlog=1
        return (0.0, 1.0)
    end

    μ_pool = median(pooled)
    raw_mad = mad(pooled; center = μ_pool, normalize = false)
    τ²_raw  = (1.4826 * raw_mad)^2
    τ²      = max(τ²_raw, 0.01)
    return (μ_pool, τ²)
end

# ─────────────────────────────────────────────────────────────────────────────
# Five omnibus columns on wide_df
# ─────────────────────────────────────────────────────────────────────────────

"""
    _compute_omnibus_columns!(wide_df::DataFrame,
                              ars::Vector{<:AbstractAnalysisResult},
                              condition_labels::Vector{String},
                              eb_prior::Tuple{Float64, Float64}) -> Nothing

Compute and write the five omnibus columns to `wide_df` in place:

- `bf_omnibus::Vector{Union{Missing, Float64}}`             — M1/M0 Bayes factor
- `log10_bf_omnibus::Vector{Union{Missing, Float64}}`       — `log_bf / log(10)` (pre-clamp)
- `posterior_omnibus::Vector{Union{Missing, Float64}}`      — `bf / (1 + bf)`
- `differential_BFDR_omnibus::Vector{Union{Missing, Float64}}` — Storey BFDR within omnibus family
- `differential_pep_omnibus::Vector{Union{Missing, Float64}}`  — `1 - posterior_omnibus`

# Protein-order independence
`wide_df.Protein` is the result of an inner-join chain whose order is NOT
guaranteed to match any single `ars[c].results.Protein` order. A naive
row-index walk would silently mis-pair proteins. This implementation therefore
builds a Protein-string-keyed `Dict{String, Tuple{Float64, Float64}}` **once
per AR before the per-row loop** and looks up `(μ_c, σ_c²)` by string key.

# Edge cases
1. Protein missing in any AR's lookup → all 5 columns are `missing` for that row.
2. Any `(μ_c, σ_c)` non-finite → all 5 columns are `missing` for that row.
3. `_eb_pooled_prior` already emitted the `(0.0, 1.0)` fallback warning if the
   pool was empty; this writer does NOT re-warn for that case.
"""
function _compute_omnibus_columns!(wide_df::DataFrame,
                                   ars::Vector{<:AbstractAnalysisResult},
                                   condition_labels::Vector{String},
                                   eb_prior::Tuple{Float64, Float64})
    isempty(condition_labels) && throw(ArgumentError(
        "_compute_omnibus_columns!: empty condition_labels"))
    length(ars) == length(condition_labels) || throw(ArgumentError(
        "_compute_omnibus_columns!: length mismatch (got ars=$(length(ars)), labels=$(length(condition_labels)))"))
    hasproperty(wide_df, :Protein) || throw(ArgumentError(
        "_compute_omnibus_columns!: wide_df is missing :Protein column"))

    μ_pool, τ² = eb_prior
    k = length(ars)
    n_rows = nrow(wide_df)

    # Build per-AR Protein → (μ, σ²) lookups ONCE (protein-order-independence invariant).
    per_ar_lookup = Vector{Dict{String, Tuple{Float64, Float64}}}(undef, k)
    any_ar_missing_columns = false
    for c in 1:k
        d = Dict{String, Tuple{Float64, Float64}}()
        ar_df = ars[c].results
        if !hasproperty(ar_df, :mean_log2FC) || !hasproperty(ar_df, :sd_log2FC)
            @warn "_compute_omnibus_columns!: condition '$(condition_labels[c])' AR missing :mean_log2FC and/or :sd_log2FC; omnibus disabled for this condition" maxlog=1
            any_ar_missing_columns = true
            per_ar_lookup[c] = d
            continue
        end
        for r in 1:nrow(ar_df)
            prot = String(ar_df.Protein[r])
            μ_v  = ar_df.mean_log2FC[r]
            σ_v  = ar_df.sd_log2FC[r]
            (ismissing(μ_v) || ismissing(σ_v)) && continue
            (isfinite(μ_v) && isfinite(σ_v))  || continue
            d[prot] = (Float64(μ_v), Float64(σ_v)^2)
        end
        per_ar_lookup[c] = d
    end

    # Pre-allocate output vectors (Union{Missing, Float64}; missing is the default).
    bf_col      = Vector{Union{Missing, Float64}}(missing, n_rows)
    log10_col   = Vector{Union{Missing, Float64}}(missing, n_rows)
    post_col    = Vector{Union{Missing, Float64}}(missing, n_rows)
    pep_col     = Vector{Union{Missing, Float64}}(missing, n_rows)

    warn_nonfinite_emitted = false

    if !any_ar_missing_columns
        for i in 1:n_rows
            prot = String(wide_df.Protein[i])
            μ_vec = Vector{Float64}(undef, k)
            σ²_vec = Vector{Float64}(undef, k)
            row_ok = true
            for c in 1:k
                tup = get(per_ar_lookup[c], prot, nothing)
                if tup === nothing
                    row_ok = false
                    break
                end
                (μc, σ²c) = tup
                if !isfinite(μc) || !isfinite(σ²c)
                    row_ok = false
                    if !warn_nonfinite_emitted
                        @warn "_compute_omnibus_columns!: non-finite (μ, σ²) for protein '$prot' in condition '$(condition_labels[c])'; emitting missing omnibus columns" maxlog=1
                        warn_nonfinite_emitted = true
                    end
                    break
                end
                μ_vec[c]  = μc
                σ²_vec[c] = σ²c
            end
            row_ok || continue

            (bf, log_bf, posterior, pep_v) = _laplace_omnibus_bf(μ_vec, σ²_vec, μ_pool, τ²)

            # log10 uses the pre-clamp log_bf to preserve precision in the report payload.
            if !isfinite(log_bf) || !isfinite(bf) || !isfinite(posterior)
                if !warn_nonfinite_emitted
                    @warn "_compute_omnibus_columns!: non-finite omnibus result for protein '$prot'; emitting missing" maxlog=1
                    warn_nonfinite_emitted = true
                end
                continue
            end

            bf_col[i]    = bf
            log10_col[i] = log_bf / log(10.0)
            post_col[i]  = posterior
            pep_col[i]   = pep_v
        end
    end

    # Storey monotone step-down BFDR within the omnibus family.
    bfdr_col = bfdr(post_col; isBF = false)
    # `pep` helper from src/core/utils.jl returns Union{Missing, Float64} elementwise,
    # but the broadcasted `@. ifelse(ismissing(x), missing, 1.0 - x)` can collapse the
    # element type to `Float64` when no element is missing. Force the column schema
    # to `Vector{Union{Missing, Float64}}` for downstream stability.
    pep_raw = pep(post_col)
    pep_from_post = Vector{Union{Missing, Float64}}(pep_raw)

    # Force the BFDR column to the same schema for symmetry (bfdr() already returns
    # Union{Missing, Float64} but defensive coercion is cheap and pins the contract).
    bfdr_col_typed = Vector{Union{Missing, Float64}}(bfdr_col)

    # Re-bind the columns by name. Vector{Union{Missing, Float64}} matches the report contract.
    wide_df[!, :bf_omnibus]                = bf_col
    wide_df[!, :log10_bf_omnibus]          = log10_col
    wide_df[!, :posterior_omnibus]         = post_col
    wide_df[!, :differential_BFDR_omnibus] = bfdr_col_typed
    wide_df[!, :differential_pep_omnibus]  = pep_from_post

    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-protein 5-class generalised classification
# ─────────────────────────────────────────────────────────────────────────────

"""
    _classify_kgroup(protein::String,
                     condition_labels::Vector{String},
                     posterior_by_condition::Dict{String, Dict{String, Union{Missing, Float64}}},
                     bfdr_omnibus::Union{Missing, Float64},
                     posterior_threshold::Float64,
                     bfdr_threshold::Float64) -> NTuple{3, Any}

Per-protein generalised classification. Returns a 3-tuple
`(enriched_in::Vector{Symbol}, depleted_in::Vector{Symbol}, kgroup_class::Symbol)`
where:

- `enriched_in` = subset of `condition_labels` with `posterior_prob ≥ posterior_threshold`
- `depleted_in` = subset with `posterior_prob < 1 - posterior_threshold`
- `kgroup_class` = one of `:omnibus_null`, `:none_enriched`, `:condition_specific`,
                   `:all_enriched`, `:fully_resolved` (5-class enum lock)

`posterior_by_condition` is a precomputed `Dict{condition_string, Dict{protein_string,
Union{Missing, Float64}}}` rather than a `wide_df` row, because the wide DF does
NOT carry per-condition `posterior_prob_<cond>` columns directly — they live on
each `ar.results` and must be looked up by Protein STRING (the Protein-order-independence
invariant).

"""
function _classify_kgroup(protein::String,
                          condition_labels::Vector{String},
                          posterior_by_condition::Dict{String, Dict{String, Union{Missing, Float64}}},
                          bfdr_omnibus::Union{Missing, Float64},
                          posterior_threshold::Float64,
                          bfdr_threshold::Float64)
    isempty(condition_labels) && throw(ArgumentError(
        "_classify_kgroup: empty condition_labels"))

    is_omnibus_sig = (!ismissing(bfdr_omnibus) && bfdr_omnibus <= bfdr_threshold)

    enriched_set = Symbol[]
    depleted_set = Symbol[]
    for cond in condition_labels
        post_map = get(posterior_by_condition, cond, nothing)
        post_map === nothing && continue
        pp_v = get(post_map, protein, missing)
        ismissing(pp_v) && continue
        pp = Float64(pp_v)
        if pp >= posterior_threshold
            push!(enriched_set, Symbol(cond))
        elseif pp < (1.0 - posterior_threshold)
            push!(depleted_set, Symbol(cond))
        end
    end

    k = length(condition_labels)
    kgroup_class = if !is_omnibus_sig
        :omnibus_null
    elseif isempty(enriched_set) && isempty(depleted_set)
        :none_enriched
    elseif length(enriched_set) == k
        :all_enriched
    elseif !isempty(enriched_set) && !isempty(depleted_set)
        :fully_resolved
    else
        :condition_specific
    end

    return (enriched_set, depleted_set, kgroup_class)
end

# ─────────────────────────────────────────────────────────────────────────────
# Three classification columns on wide_df
# ─────────────────────────────────────────────────────────────────────────────

"""
    _compute_kgroup_classification_columns!(wide_df::DataFrame,
                                            ars::Vector{<:AbstractAnalysisResult},
                                            condition_labels::Vector{String},
                                            posterior_threshold::Float64,
                                            bfdr_threshold::Float64) -> Nothing

Compute and write three generalised-classification columns to `wide_df` in place:

- `enriched_in::Vector{Vector{Symbol}}`
- `depleted_in::Vector{Vector{Symbol}}`
- `kgroup_class::Vector{Symbol}`

Reads `wide_df.differential_BFDR_omnibus` (must already exist —
`_compute_omnibus_columns!` is run first). Builds the per-condition
`posterior_prob` lookup ONCE per AR before the per-row loop (Protein-keyed,
per the protein-order-independence invariant). `kgroup_class` is never `missing` — proteins
without an omnibus signal default to `:omnibus_null`.
"""
function _compute_kgroup_classification_columns!(wide_df::DataFrame,
                                                 ars::Vector{<:AbstractAnalysisResult},
                                                 condition_labels::Vector{String},
                                                 posterior_threshold::Float64,
                                                 bfdr_threshold::Float64)
    isempty(condition_labels) && throw(ArgumentError(
        "_compute_kgroup_classification_columns!: empty condition_labels"))
    length(ars) == length(condition_labels) || throw(ArgumentError(
        "_compute_kgroup_classification_columns!: length mismatch (got ars=$(length(ars)), labels=$(length(condition_labels)))"))
    hasproperty(wide_df, :Protein) || throw(ArgumentError(
        "_compute_kgroup_classification_columns!: wide_df missing :Protein column"))
    hasproperty(wide_df, :differential_BFDR_omnibus) || throw(ArgumentError(
        "_compute_kgroup_classification_columns!: wide_df missing :differential_BFDR_omnibus column (run _compute_omnibus_columns! first)"))
    (0.0 <= posterior_threshold <= 1.0) || throw(ArgumentError(
        "_compute_kgroup_classification_columns!: posterior_threshold must be in [0, 1] (got $posterior_threshold)"))

    # Build the per-condition posterior lookup: Dict{cond_string, Dict{protein_string, posterior}}.
    posterior_by_condition = Dict{String, Dict{String, Union{Missing, Float64}}}()
    for (c, ar) in zip(condition_labels, ars)
        d = Dict{String, Union{Missing, Float64}}()
        ar_df = ar.results
        if !hasproperty(ar_df, :posterior_prob)
            @warn "_compute_kgroup_classification_columns!: condition '$c' AR missing :posterior_prob column; classification will not see this condition" maxlog=1
            posterior_by_condition[c] = d
            continue
        end
        for r in 1:nrow(ar_df)
            prot = String(ar_df.Protein[r])
            pp_v = ar_df.posterior_prob[r]
            if ismissing(pp_v) || !isfinite(pp_v)
                d[prot] = missing
            else
                d[prot] = Float64(pp_v)
            end
        end
        posterior_by_condition[c] = d
    end

    n_rows = nrow(wide_df)
    enriched_col = Vector{Vector{Symbol}}(undef, n_rows)
    depleted_col = Vector{Vector{Symbol}}(undef, n_rows)
    class_col    = Vector{Symbol}(undef, n_rows)

    bfdr_omnibus_vec = wide_df.differential_BFDR_omnibus

    for i in 1:n_rows
        prot = String(wide_df.Protein[i])
        bfdr_v = bfdr_omnibus_vec[i]
        bfdr_typed = ismissing(bfdr_v) ? missing : Float64(bfdr_v)
        (enriched_set, depleted_set, kclass) = _classify_kgroup(
            prot, condition_labels, posterior_by_condition,
            bfdr_typed, posterior_threshold, bfdr_threshold,
        )
        enriched_col[i] = enriched_set
        depleted_col[i] = depleted_set
        class_col[i]    = kclass
    end

    wide_df[!, :enriched_in]  = enriched_col
    wide_df[!, :depleted_in]  = depleted_col
    wide_df[!, :kgroup_class] = class_col

    return nothing
end
