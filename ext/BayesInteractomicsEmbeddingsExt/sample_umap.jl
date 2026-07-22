"""
    fit_sample_umap(X::AbstractMatrix{<:Real}, n_neighbors::Int, min_dist::Float64) -> Matrix{Float64}

UMAP.jl wrapper for sample-level embedding. Input `X` is (n_samples, n_features); UMAP.jl
expects column-major (n_features, n_observations), so we pass `Matrix(X')`. Returns
(n_samples, 2) coordinate matrix.

Determinism is injected via Random.seed!(seed) BEFORE this call — UMAP.jl 0.1.10 does NOT
accept a `seed=` kwarg.
"""
function BayesInteractomics.fit_sample_umap(X::AbstractMatrix{<:Real},
                                            n_neighbors::Int,
                                            min_dist::Float64)
    X_f = Matrix{Float64}(X)
    X_col_major = Matrix(X_f')  # (n_features, n_samples)
    # n_components is positional in UMAP.UMAP_ (UMAP.jl 0.1.x); passing as kwarg
    # raises MethodError.
    embed = UMAP.umap(X_col_major, 2;
                      n_neighbors = n_neighbors,
                      min_dist    = min_dist,
                      metric      = Distances.Euclidean(),
                      init        = :spectral)
    # UMAP returns (n_components, n_samples); transpose to (n_samples, 2).
    return Matrix(embed')
end
