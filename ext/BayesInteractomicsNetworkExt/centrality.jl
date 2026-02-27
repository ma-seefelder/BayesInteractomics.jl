# Centrality measures for network analysis

"""
    BayesInteractomics.centrality_measures(net::InteractionNetwork) -> CentralityMeasures

Compute multiple centrality measures for all nodes in the network.

Centrality measures identify "hub" proteins that are central to the interaction network.

# Arguments
- `net::InteractionNetwork`: Network to analyze

# Returns
- `CentralityMeasures`: Object containing centrality scores for each protein

# Centrality Measures
- **Degree**: Number of direct connections (in + out degree)
- **Weighted Degree**: Sum of edge weights for connections
- **Betweenness**: How often a node appears on shortest paths between other nodes
- **Closeness**: Average distance to all other nodes (higher = more central)
- **Eigenvector**: Importance based on connections to other important nodes
- **PageRank**: Probability of reaching node in random walk (like Google's PageRank)

# Example
```julia
cm = centrality_measures(net)
df = centrality_dataframe(cm)
top_hubs = sort(df, :PageRank, rev=true)[1:10, :]
```
"""
function BayesInteractomics.centrality_measures(net::InteractionNetwork)
    g = net.graph
    g_simple = SimpleDiGraph(g)
    n = nv(g)

    if n == 0
        return CentralityMeasures(
            String[],
            Int[],
            Float64[],
            Float64[],
            Float64[],
            Float64[],
            Float64[]
        )
    end

    # Degree centrality (in + out degree)
    degrees = [indegree(g_simple, v) + outdegree(g_simple, v) for v in vertices(g_simple)]

    # Weighted degree
    weighted_degrees = Float64[]
    for v in vertices(g)
        in_weight = sum((get_weight(g, u, v) for u in inneighbors(g, v)); init=0.0)
        out_weight = sum((get_weight(g, v, u) for u in outneighbors(g, v)); init=0.0)
        push!(weighted_degrees, in_weight + out_weight)
    end

    # Betweenness centrality
    betweenness = try
        betweenness_centrality(g_simple)
    catch e
        @warn "Could not compute betweenness centrality: $e"
        zeros(Float64, n)
    end

    # Closeness centrality
    closeness = try
        closeness_centrality(g_simple)
    catch e
        @warn "Could not compute closeness centrality: $e"
        zeros(Float64, n)
    end

    # Eigenvector centrality
    eigenvector = try
        eigenvector_centrality(g_simple)
    catch e
        @warn "Could not compute eigenvector centrality: $e"
        zeros(Float64, n)
    end

    # PageRank
    pr = try
        Graphs.pagerank(g_simple)
    catch e
        @warn "Could not compute PageRank: $e"
        fill(1.0/n, n)
    end

    return CentralityMeasures(
        copy(net.protein_names),
        degrees,
        weighted_degrees,
        betweenness,
        closeness,
        eigenvector,
        pr
    )
end

"""
    get_top_hubs(cm::CentralityMeasures; by=:pagerank, n=10) -> DataFrame

Get the top hub proteins by a specified centrality measure.

# Arguments
- `cm::CentralityMeasures`: Centrality measures object
- `by::Symbol=:pagerank`: Centrality measure to sort by
  (:degree, :weighted_degree, :betweenness, :closeness, :eigenvector, :pagerank)
- `n::Int=10`: Number of top hubs to return

# Returns
- `DataFrame`: Top n proteins with all their centrality measures
"""
function BayesInteractomics.get_top_hubs(cm::CentralityMeasures; by::Symbol=:pagerank, n::Int=10)
    df = BayesInteractomics.centrality_dataframe(cm)

    sort_col = if by == :degree
        :Degree
    elseif by == :weighted_degree
        :WeightedDegree
    elseif by == :betweenness
        :Betweenness
    elseif by == :closeness
        :Closeness
    elseif by == :eigenvector
        :Eigenvector
    elseif by == :pagerank
        :PageRank
    else
        error("Unknown centrality measure: $by")
    end

    sorted = sort(df, sort_col, rev=true)
    return first(sorted, min(n, nrow(sorted)))
end
