# src/embeddings/protein_embedding.jl
# Similarity & Embeddings: protein-level UMAP compute.
# Pattern source: src/differential/analysis.jl::_compute_differential_statistics.

using Random
using Statistics
using DataFrames

# Reuse the numerical safety clamp from Differential — single source of truth.
# Differential.jl defines `_safe_log10(x::Real) = x > 0 ? log10(Float64(x)) : _LOG10_FLOOR`
# inside `module Differential` (not reachable from this flat-namespace file). We keep a local
# clone with the locked semantics (`log10(max(x, 1e-12))`) to avoid an
# inter-module import dependency. Both clamps map BF=0 to a finite floor; the only difference
# is the floor value (-12 here vs Differential's _LOG10_FLOOR). Equivalent for our use case
# since the feature is subsequently z-scored.
"""
    _safe_log10_emb(x::Real) -> Float64

Numerical safety clamp. Returns `log10(max(x, 1e-12))`. `missing` propagates as
`missing` so callers can detect drop-rows before constructing the feature matrix.
"""
_safe_log10_emb(x::Real) = log10(max(Float64(x), 1e-12))
_safe_log10_emb(::Missing) = missing

"""
    _zscore_columns(F::AbstractMatrix{Float64}) -> Matrix{Float64}

Standardise each column to mean 0, std 1. Zero-std columns (constant features) are
returned as zeros. Z-score per feature column BEFORE UMAP.
"""
function _zscore_columns(F::AbstractMatrix{Float64})
    F_z = similar(F)
    for j in 1:size(F, 2)
        col = view(F, :, j)
        m = mean(col)
        s = std(col)
        if s > 0 && isfinite(s)
            @inbounds for i in 1:size(F, 1)
                F_z[i, j] = (F[i, j] - m) / s
            end
        else
            @inbounds for i in 1:size(F, 1)
                F_z[i, j] = 0.0
            end
        end
    end
    return F_z
end

# Internal: locate the log2FC column under any of the three observed schemas.
# Repo canonical: `:mean_log2FC` (BMA wiring; see src/analysis/pipeline.jl:815).
# Fallbacks: `:log2FC_mean`, `:log2FC` (kept for forward/backward compat with
# any future schema realignment).
function _find_log2fc_column(copula_results::DataFrame)
    for sym in (:mean_log2FC, :log2FC_mean, :log2FC)
        hasproperty(copula_results, sym) && return sym
    end
    error("[Embeddings] copula_results must have :mean_log2FC, :log2FC_mean, or :log2FC column")
end

# Internal: locate each per-component Bayes-factor column. Repo canonical names are
# lowercase per src/analysis/pipeline.jl:816-818 (`bf_enrichment`, `bf_correlation`,
# `bf_detected`). Plan-body interfaces showed uppercase `:BF_*` — kept as a fallback so
# either schema works.
function _find_bf_columns(copula_results::DataFrame)
    bf_e = hasproperty(copula_results, :bf_enrichment) ? :bf_enrichment :
           hasproperty(copula_results, :BF_enrichment) ? :BF_enrichment :
           error("[Embeddings] copula_results must have :bf_enrichment column")
    bf_c = hasproperty(copula_results, :bf_correlation) ? :bf_correlation :
           hasproperty(copula_results, :BF_correlation) ? :BF_correlation :
           error("[Embeddings] copula_results must have :bf_correlation column")
    bf_d = hasproperty(copula_results, :bf_detected) ? :bf_detected :
           hasproperty(copula_results, :BF_detected) ? :BF_detected :
           error("[Embeddings] copula_results must have :bf_detected column")
    return (bf_e, bf_c, bf_d)
end

"""
    _assemble_protein_features(copula_results::DataFrame) ->
        (F::Matrix{Float64}, kept_indices::Vector{Int}, protein_ids::Vector{String})

Build the (n_proteins, 5) feature matrix:
`[log10(bf_enrichment), log10(bf_correlation), log10(bf_detected), posterior_prob, log2FC_mean]`.
All log10 features apply max(bf, 1e-12) clamp. Proteins with any missing entry across
the 5 features are dropped; `kept_indices` records the surviving row indices in `copula_results`.

Column names (lowercase canonical per repo schema, with uppercase fallback):
- `:Protein`
- `:bf_enrichment`, `:bf_correlation`, `:bf_detected`
- `:posterior_prob`
- `:mean_log2FC` (or `:log2FC_mean` / `:log2FC` fallback)
"""
function _assemble_protein_features(copula_results::DataFrame)
    n_total = nrow(copula_results)
    n_total == 0 && return (Matrix{Float64}(undef, 0, 5), Int[], String[])

    log2fc_col = _find_log2fc_column(copula_results)
    bf_e_col, bf_c_col, bf_d_col = _find_bf_columns(copula_results)

    kept = Int[]
    rows = Vector{Float64}[]
    ids  = String[]

    for i in 1:n_total
        bfe = copula_results[i, bf_e_col]
        bfc = copula_results[i, bf_c_col]
        bfd = copula_results[i, bf_d_col]
        pp  = copula_results[i, :posterior_prob]
        l2  = copula_results[i, log2fc_col]
        # Reduce vector-valued log2FC (multi-protocol case) to scalar via mean — robust to either schema.
        l2_scalar = (l2 isa AbstractVector) ? (isempty(l2) ? missing : mean(skipmissing(l2))) : l2
        if any(ismissing, (bfe, bfc, bfd, pp, l2_scalar))
            continue
        end
        push!(kept, i)
        push!(rows, [_safe_log10_emb(bfe), _safe_log10_emb(bfc), _safe_log10_emb(bfd),
                     Float64(pp), Float64(l2_scalar)])
        push!(ids, String(copula_results[i, :Protein]))
    end

    if isempty(rows)
        return (Matrix{Float64}(undef, 0, 5), Int[], String[])
    end

    F = Matrix{Float64}(undef, length(rows), 5)
    for (r, row) in enumerate(rows)
        F[r, :] .= row
    end
    return (F, kept, ids)
end

# Internal: derive a Symbol assignment vector aligned to all rows of copula_results.
# Resolution order: (1) hasproperty(lc_result, :assignments) — exact must_have semantic;
# (2) responsibilities argmax with `1→:H0, 2→:Agnostic, 3→:H1` mapping
# (per src/combination/latent_class.jl gamma ordering, lines 1206-1271); (3) nothing
# (caller falls through to 2-class posterior fallback).
function _lc_assignments(lc_result)
    lc_result === nothing && return nothing
    if hasproperty(lc_result, :assignments) && lc_result.assignments !== nothing
        return [Symbol(x) for x in lc_result.assignments]
    end
    if hasproperty(lc_result, :responsibilities) && lc_result.responsibilities !== nothing
        R = lc_result.responsibilities
        n, k = size(R)
        labels = Vector{Symbol}(undef, n)
        if k == 3
            class_map = (:H0, :Agnostic, :H1)
            @inbounds for i in 1:n
                labels[i] = class_map[argmax(@view R[i, :])]
            end
            return labels
        elseif k == 2
            # 2-component fallback: column 1 = H0, column 2 = H1.
            @inbounds for i in 1:n
                labels[i] = R[i, 2] > R[i, 1] ? :H1 : :H0
            end
            return labels
        end
    end
    return nothing
end

"""
    _derive_protein_class_labels(copula_results::DataFrame, lc_result, kept_indices::Vector{Int}) ->
        Vector{Symbol}

Class labels aligned to the surviving protein indices:

- If `copula_results` has a `:classification` column (differential context): use those
  4-class enum symbols (`:GAINED`, `:REDUCED`, `:UNCHANGED`, `:BOTH_NEGATIVE`, plus optional
  `:CONDITION_A_SPECIFIC` / `:CONDITION_B_SPECIFIC` — 6-class total).
- Else if `lc_result` exposes `.assignments` or `.responsibilities` (single-bait 3-class):
  use that vector indexed by `kept_indices` (values `:H0` / `:Agnostic` / `:H1`).
- Else: 2-class fallback — `:H1` if `posterior_prob > 0.5`, else `:H0`; @info emitted once.
"""
function _derive_protein_class_labels(copula_results::DataFrame, lc_result, kept_indices::Vector{Int})
    if hasproperty(copula_results, :classification)
        return [Symbol(copula_results[i, :classification]) for i in kept_indices]
    end
    lc_labels = _lc_assignments(lc_result)
    if lc_labels !== nothing && length(lc_labels) >= maximum(kept_indices; init=0)
        return [lc_labels[i] for i in kept_indices]
    end
    @info "[Embeddings] protein UMAP using 2-class fallback (no latent_class_result)" maxlog=1
    return [copula_results[i, :posterior_prob] > 0.5 ? :H1 : :H0 for i in kept_indices]
end

"""
    _compute_protein_embedding(copula_results::DataFrame, lc_result, cfg::EmbeddingsConfig) ->
        NamedTuple(protein_umap_coords, protein_classes, protein_ids)

Protein-level UMAP. Builds the 5-feature vector, z-scores per column, calls the
extension-gated `fit_protein_umap` stub. Determinism via `Random.seed!(cfg.seed)` BEFORE
the call (NEVER `seed=` kwarg per UMAP.jl 0.1.10 Risk-1 lock).

Returns a 3-key NamedTuple shaped to populate the `protein_*` fields of `EmbeddingsResult`:
- `protein_umap_coords::Union{Nothing, Matrix{Float64}}` — (n_kept, 2) or `nothing`
- `protein_classes::Vector{Symbol}`                     — length n_kept
- `protein_ids::Vector{String}`                         — length n_kept
"""
function _compute_protein_embedding(copula_results::DataFrame, lc_result, cfg::EmbeddingsConfig)
    if !cfg.run_embeddings
        return (protein_umap_coords = nothing,
                protein_classes     = Symbol[],
                protein_ids         = String[])
    end

    F_raw, kept, ids = _assemble_protein_features(copula_results)
    classes = _derive_protein_class_labels(copula_results, lc_result, kept)

    # Empty cohort: no proteins survived the missing-drop. Return empty result + classes.
    if isempty(kept)
        return (protein_umap_coords = nothing,
                protein_classes     = classes,
                protein_ids         = ids)
    end

    # z-score per feature column BEFORE UMAP.
    F_z = _zscore_columns(F_raw)

    umap_coords = nothing
    if cfg.method === :umap && _embeddings_extension_loaded()
        Random.seed!(cfg.seed)
        try
            umap_coords = fit_protein_umap(F_z, cfg.n_neighbors, cfg.min_dist;
                                            supervised = cfg.supervised,
                                            class_labels = classes)
        catch e
            @warn "[Embeddings] protein UMAP failed: $e" maxlog=1
            umap_coords = nothing
        end
    end
    # cfg.method === :tsne: protein-level t-SNE is NOT shipped in v1.2.0 (UMAP only); leave nothing.
    # cfg.method === :none: leave nothing.
    # cfg.method === :umap + ext not loaded: umap_coords stays nothing; report renders banner.

    return (protein_umap_coords = umap_coords,
            protein_classes     = classes,
            protein_ids         = ids)
end
