"""
    fit_protein_umap(F::AbstractMatrix{<:Real}, n_neighbors::Int, min_dist::Float64;
                     supervised::Bool = false, class_labels = nothing) -> Matrix{Float64}

Protein-level UMAP on the (n_proteins, 5) z-scored feature matrix.
`supervised=true` emits `@warn` once and proceeds UNSUPERVISED — UMAP.jl 0.1.x has no `y=`
kwarg; supervised UMAP requires UMAP.jl >= 0.2 (unreleased). Tracked as v1.3 work.

Determinism via Random.seed!(seed) BEFORE this call.
"""
function BayesInteractomics.fit_protein_umap(F::AbstractMatrix{<:Real},
                                             n_neighbors::Int,
                                             min_dist::Float64;
                                             supervised::Bool = false,
                                             class_labels = nothing)
    if supervised
        @warn "[Embeddings] supervised UMAP requires UMAP.jl >= 0.2 (unreleased); " *
              "falling back to unsupervised for v1.2.0" maxlog=1
    end
    F_f = Matrix{Float64}(F)
    F_col_major = Matrix(F_f')  # (5, n_proteins)
    # n_components is positional in UMAP.UMAP_ (UMAP.jl 0.1.x); passing as kwarg
     # raises MethodError. Keep all other args as kwargs.
    embed = UMAP.umap(F_col_major, 2;
                      n_neighbors = n_neighbors,
                      min_dist    = min_dist,
                      metric      = Distances.Euclidean(),
                      init        = :spectral)
    return Matrix(embed')
end
