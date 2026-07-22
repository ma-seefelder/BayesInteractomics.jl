"""
    fit_sample_tsne(X::AbstractMatrix{<:Real}, seed::Int) -> Matrix{Float64}

TSne.jl wrapper for sample-level embedding. Perplexity pinned to `min(30, (n-1) / 3)` per
the canonical rule perplexity ≤ (n-1)/3 (CONTEXT <specifics>).

TSne.jl `tsne(X; ndims=2, reduce_dims=0, max_iter=1000, perplexity=30, verbose=false)` —
no `seed=` kwarg; determinism via Random.seed!(seed) BEFORE the call.
"""
function BayesInteractomics.fit_sample_tsne(X::AbstractMatrix{<:Real}, seed::Int)
    Random.seed!(seed)
    X_f = Matrix{Float64}(X)
    n = size(X_f, 1)
    perplexity = max(2.0, min(30.0, (n - 1) / 3))
    coords = TSne.tsne(X_f, 2, 0, 1000, perplexity; verbose=false)
    return Matrix{Float64}(coords)
end
