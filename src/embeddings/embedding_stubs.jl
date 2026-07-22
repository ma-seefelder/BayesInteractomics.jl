# src/embeddings/embedding_stubs.jl
# Similarity & Embeddings: extension stub surface.
# Pattern source: src/data/imputation_stubs.jl + src/ml/metalearner_stubs.jl.

"""
    fit_sample_umap(X::AbstractMatrix{<:Real}, n_neighbors::Int, min_dist::Float64) -> Matrix{Float64}

Compute 2-D UMAP embedding from a sample matrix.

**Requires:** `using UMAP, Clustering`

Real implementation: `ext/BayesInteractomicsEmbeddingsExt/sample_umap.jl`.
"""
function fit_sample_umap end

"""
    fit_sample_tsne(X::AbstractMatrix{<:Real}, seed::Int) -> Matrix{Float64}

Compute 2-D t-SNE embedding from a sample matrix. Pinned to perplexity = min(30, (n-1)/3).

**Requires:** `using UMAP, Clustering, TSne` (TSne is optional within the extension).

Real implementation: `ext/BayesInteractomicsEmbeddingsExt/sample_tsne.jl`.
"""
function fit_sample_tsne end

"""
    fit_protein_umap(F::AbstractMatrix{<:Real}, n_neighbors::Int, min_dist::Float64;
                     supervised::Bool = false, class_labels = nothing) -> Matrix{Float64}

Compute 2-D UMAP embedding from a (n_proteins, 5) z-scored feature matrix.
`supervised=true` is a no-op-with-warning in v1.2.0 (UMAP.jl 0.1.x has no y= kwarg).

**Requires:** `using UMAP, Clustering`

Real implementation: `ext/BayesInteractomicsEmbeddingsExt/protein_umap.jl`.
"""
function fit_protein_umap end

"""
    fit_condition_clustering(D::AbstractMatrix{Float64}) -> NamedTuple

Compute hierarchical clustering (UPGMA / average linkage) on a (k, k) distance matrix.
Returns NamedTuple `(merges, heights, order)` ready for JSON serialisation.

**Requires:** `using UMAP, Clustering`

Real implementation: `ext/BayesInteractomicsEmbeddingsExt/hclust_helper.jl`.
"""
function fit_condition_clustering end

"""
    EMBEDDINGS_STATUS_VALUES

Allowed sentinel values for the embedding status surfaced in reports:
- `:not_run`            — `run_embeddings = false`
- `:pca_only`           — extension not loaded; PCA fallback rendered
- `:loaded`             — extension loaded and embeddings computed successfully
- `:failed`             — extension loaded but `fit_*` threw (caught + logged via @warn maxlog=1)
"""
const EMBEDDINGS_STATUS_VALUES = (:not_run, :pca_only, :loaded, :failed)

"""
    _embeddings_extension_loaded() -> Bool

`true` once `BayesInteractomicsEmbeddingsExt` has registered methods for `fit_sample_umap`.
"""
_embeddings_extension_loaded() = !isempty(methods(fit_sample_umap))

"""
    _require_embeddings_extension(method::Symbol)

For `method = :tsne` only: throw `ArgumentError` if TSne extension methods are not loaded.
For `method = :umap` or `:none`: returns nothing.

t-SNE is an explicit opt-in; the user requested it, so loud failure is honest.
UMAP default uses Variante B silent fallback (banner in report, no throw).
"""
function _require_embeddings_extension(method::Symbol)
    if method === :tsne && isempty(methods(fit_sample_tsne))
        throw(ArgumentError(
            "CONFIG.embeddings_config.method = :tsne requires `using TSne, UMAP, Clustering` " *
            "BEFORE `using BayesInteractomics`. Either load TSne.jl, switch to `:umap` (default), " *
            "or set `method = :none`."
        ))
    end
    return nothing
end
