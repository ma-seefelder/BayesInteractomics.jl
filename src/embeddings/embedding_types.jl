# src/embeddings/embedding_types.jl
# Similarity & Embeddings: type + config definitions.

using Dates

"""
    EmbeddingsConfig

Configuration for sample-level + protein-level embeddings and condition-level similarity.

Defaults are locked values.

# Fields
- `method::Symbol = :umap` — non-linear method; one of `:umap`, `:tsne`, `:none`.
- `seed::Int = 42` — `Random.seed!(seed)` invoked before each UMAP / t-SNE call.
  UMAP.jl 0.1.10 has no `seed=` kwarg — determinism is injected via Random state only.
- `supervised::Bool = false` — supervised UMAP using protein_classes. UMAP.jl 0.1.x has no
  `y=` kwarg; when `true`, the extension method emits `@warn` once and proceeds unsupervised.
- `n_neighbors::Int = 15` — UMAP n_neighbors; clamped to `max(2, min(n_neighbors, n-1))` for sample-level.
- `min_dist::Float64 = 0.1` — UMAP min_dist default.
- `top_k_jaccard::Int = 50` — Top-K for the Jaccard@Top-50 secondary metric.
- `run_embeddings::Bool = true` — master toggle; `false` skips all computation.
"""
Base.@kwdef struct EmbeddingsConfig
    method::Symbol       = :umap
    seed::Int            = 42
    supervised::Bool     = false
    n_neighbors::Int     = 15
    min_dist::Float64    = 0.1
    top_k_jaccard::Int   = 50
    run_embeddings::Bool = true
end

"""
    _validate_embeddings_config(cfg::EmbeddingsConfig)

Throws `ArgumentError` for any out-of-range field. Returns `nothing` on success.
Call from `run_analysis(::CONFIG)` validation cascade.
"""
function _validate_embeddings_config(cfg::EmbeddingsConfig)
    cfg.method in (:umap, :tsne, :none) ||
        throw(ArgumentError("EmbeddingsConfig.method must be :umap, :tsne, or :none; got $(cfg.method)"))
    cfg.seed >= 0 ||
        throw(ArgumentError("EmbeddingsConfig.seed must be >= 0; got $(cfg.seed)"))
    cfg.n_neighbors >= 2 ||
        throw(ArgumentError("EmbeddingsConfig.n_neighbors must be >= 2; got $(cfg.n_neighbors)"))
    cfg.min_dist >= 0.0 ||
        throw(ArgumentError("EmbeddingsConfig.min_dist must be >= 0.0; got $(cfg.min_dist)"))
    cfg.top_k_jaccard >= 1 ||
        throw(ArgumentError("EmbeddingsConfig.top_k_jaccard must be >= 1; got $(cfg.top_k_jaccard)"))
    return nothing
end

"""
    _config_snapshot(cfg::EmbeddingsConfig) -> NamedTuple

Returns a NamedTuple snapshot used by `_should_recompute_embeddings` to detect
embedding-config drift and trigger partial cache invalidation without touching the full pipeline.
"""
function _config_snapshot(cfg::EmbeddingsConfig)
    return (
        method        = cfg.method,
        seed          = cfg.seed,
        n_neighbors   = cfg.n_neighbors,
        min_dist      = cfg.min_dist,
        supervised    = cfg.supervised,
        top_k_jaccard = cfg.top_k_jaccard,
    )
end

"""
    EmbeddingsResult

Sample-level + protein-level embeddings for a single `AnalysisResult`.

Round-trips through `save_result` / `load_result`; CACHE_VERSION bumped 23 → 24.

# Fields
- `sample_pca_scores::Matrix{Float64}` — (n_samples, 2) PC1/PC2 scores; always populated when `run_embeddings = true`.
- `sample_pca_var_explained::Vector{Float64}` — length 2, percent variance explained by PC1 and PC2.
- `sample_labels::NamedTuple` — `(condition, replicate, experiment, protocol)` label vectors; each length n_samples.
- `sample_filter_level::Symbol` — `:complete_case | :threshold_80 | :threshold_50 | :skipped` (per `filter_complete_case`).
- `sample_umap_coords::Union{Nothing, Matrix{Float64}}` — (n_samples, 2) or `nothing` when extension not loaded.
- `sample_tsne_coords::Union{Nothing, Matrix{Float64}}` — (n_samples, 2); only populated when `method = :tsne` + TSne loaded.
- `protein_umap_coords::Union{Nothing, Matrix{Float64}}` — (n_proteins, 2) or `nothing`.
- `protein_classes::Vector{Symbol}` — length n_proteins; `:H0|:Agnostic|:H1` (single-bait) or differential enum symbols.
- `protein_ids::Vector{String}` — length n_proteins; same row order as `protein_umap_coords`.
- `config_snapshot::NamedTuple` — the `_config_snapshot(cfg)` value used at compute time (partial invalidation key).
"""
mutable struct EmbeddingsResult
    sample_pca_scores::Matrix{Float64}
    sample_pca_var_explained::Vector{Float64}
    sample_labels::NamedTuple
    sample_filter_level::Symbol
    sample_umap_coords::Union{Nothing, Matrix{Float64}}
    sample_tsne_coords::Union{Nothing, Matrix{Float64}}
    protein_umap_coords::Union{Nothing, Matrix{Float64}}
    protein_classes::Vector{Symbol}
    protein_ids::Vector{String}
    config_snapshot::NamedTuple
end

"""
    ConditionSimilarityResult

Condition-level k×k similarity matrices + hclust dendrogram payload for a `DifferentialResult`.

Round-trips through `save_differential_result` / `load_differential_result`
(if those exist) or via JLD2 inside the differential struct.

# Fields
- `condition_labels::Vector{String}` — k condition names; order matches matrix rows/cols.
- `spearman_log10_bf::Matrix{Float64}` — (k, k); primary metric, Spearman ρ on log10(BF).
- `pearson_log2fc::Matrix{Float64}` — (k, k); secondary togglable view.
- `pearson_posterior::Matrix{Float64}` — (k, k); secondary togglable view.
- `jaccard_top_k::Matrix{Float64}` — (k, k); Jaccard@Top-K set overlap.
- `n_shared_per_cell::Matrix{Int}` — (k, k); pairwise intersection support count (hover tooltip).
- `top_k_used::Int` — nominal top_k_jaccard value at compute time.
- `dendrogram_merges::Matrix{Int}` — from Clustering.hclust; (k-1, 2).
- `dendrogram_heights::Vector{Float64}` — length k-1; hclust merge heights.
- `dendrogram_order::Vector{Int}` — length k; leaf permutation.
- `linkage::Symbol` — locked to `:average`.
"""
mutable struct ConditionSimilarityResult
    condition_labels::Vector{String}
    spearman_log10_bf::Matrix{Float64}
    pearson_log2fc::Matrix{Float64}
    pearson_posterior::Matrix{Float64}
    jaccard_top_k::Matrix{Float64}
    n_shared_per_cell::Matrix{Int}
    top_k_used::Int
    dendrogram_merges::Matrix{Int}
    dendrogram_heights::Vector{Float64}
    dendrogram_order::Vector{Int}
    linkage::Symbol
end
