"""
    fit_condition_clustering(D::AbstractMatrix{Float64}) -> NamedTuple

UPGMA / average-linkage hclust on a (k, k) distance matrix. Returns NamedTuple
`(merges, heights, order)` ready for JSON serialisation. `branchorder = :optimal` keeps
the leaf order visually aligned with the heatmap row/column order.

Linkage = :average (UPGMA).
"""
function BayesInteractomics.fit_condition_clustering(D::AbstractMatrix{Float64})
    hc = Clustering.hclust(D; linkage = :average, branchorder = :optimal)
    return (merges = Matrix{Int}(hc.merges),
            heights = Vector{Float64}(hc.heights),
            order   = Vector{Int}(hc.order))
end
