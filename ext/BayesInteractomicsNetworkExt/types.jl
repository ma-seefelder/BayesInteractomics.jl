# Network analysis types

"""
    InteractionNetwork

Represents a protein-protein interaction network derived from Bayesian analysis.

# Fields
- `graph::SimpleWeightedDiGraph`: The underlying graph structure with weighted edges
- `protein_names::Vector{String}`: Protein names/IDs for each node
- `node_attributes::DataFrame`: Node-level attributes (posterior_prob, bayes_factor, log2fc, etc.)
- `edge_attributes::DataFrame`: Edge-level attributes (weight, source_protein, target_protein)
- `bait_protein::Union{String, Nothing}`: Name of bait protein if specified
- `bait_index::Union{Int, Nothing}`: Node index of bait protein
- `threshold_used::NamedTuple`: Thresholds used in network construction
"""
struct InteractionNetwork <: BayesInteractomics.AbstractNetworkResult
    graph::SimpleWeightedDiGraph{Int64, Float64}
    protein_names::Vector{String}
    node_attributes::DataFrame
    edge_attributes::DataFrame
    bait_protein::Union{String, Nothing}
    bait_index::Union{Int, Nothing}
    threshold_used::NamedTuple
end

"""
    NetworkStatistics

Summary statistics for network topology.

# Fields
- `n_nodes::Int`: Number of nodes (proteins)
- `n_edges::Int`: Number of edges (interactions)
- `density::Float64`: Network density (actual edges / possible edges)
- `avg_degree::Float64`: Average node degree
- `avg_weighted_degree::Float64`: Average weighted degree
- `avg_clustering::Float64`: Average clustering coefficient
- `n_components::Int`: Number of connected components
- `largest_component_size::Int`: Size of largest connected component
- `diameter::Union{Int, Nothing}`: Network diameter (Nothing if disconnected)
- `avg_path_length::Union{Float64, Nothing}`: Average shortest path length
"""
struct NetworkStatistics <: BayesInteractomics.AbstractNetworkResult
    n_nodes::Int
    n_edges::Int
    density::Float64
    avg_degree::Float64
    avg_weighted_degree::Float64
    avg_clustering::Float64
    n_components::Int
    largest_component_size::Int
    diameter::Union{Int, Nothing}
    avg_path_length::Union{Float64, Nothing}
end

"""
    CentralityMeasures

Centrality measures for all proteins in the network.

# Fields
- `protein_names::Vector{String}`: Protein names/IDs
- `degree::Vector{Int}`: Degree centrality (number of connections)
- `weighted_degree::Vector{Float64}`: Weighted degree (sum of edge weights)
- `betweenness::Vector{Float64}`: Betweenness centrality
- `closeness::Vector{Float64}`: Closeness centrality
- `eigenvector::Vector{Float64}`: Eigenvector centrality
- `pagerank::Vector{Float64}`: PageRank scores
"""
struct CentralityMeasures <: BayesInteractomics.AbstractNetworkResult
    protein_names::Vector{String}
    degree::Vector{Int}
    weighted_degree::Vector{Float64}
    betweenness::Vector{Float64}
    closeness::Vector{Float64}
    eigenvector::Vector{Float64}
    pagerank::Vector{Float64}
end

"""
    CommunityResult

Community detection results.

# Fields
- `protein_names::Vector{String}`: Protein names/IDs
- `membership::Vector{Int}`: Community assignment for each protein
- `n_communities::Int`: Number of communities detected
- `modularity::Float64`: Modularity score
- `community_sizes::Vector{Int}`: Size of each community
"""
struct CommunityResult <: BayesInteractomics.AbstractNetworkResult
    protein_names::Vector{String}
    membership::Vector{Int}
    n_communities::Int
    modularity::Float64
    community_sizes::Vector{Int}
end

# Helper function to convert centrality measures to DataFrame
"""
    centrality_dataframe(cm::CentralityMeasures) -> DataFrame

Convert centrality measures to a DataFrame for easy inspection and export.
"""
function BayesInteractomics.centrality_dataframe(cm::CentralityMeasures)
    return DataFrame(
        Protein = cm.protein_names,
        Degree = cm.degree,
        WeightedDegree = cm.weighted_degree,
        Betweenness = cm.betweenness,
        Closeness = cm.closeness,
        Eigenvector = cm.eigenvector,
        PageRank = cm.pagerank
    )
end

# Helper function to convert community results to DataFrame
"""
    community_dataframe(cr::CommunityResult) -> DataFrame

Convert community detection results to a DataFrame for easy inspection and export.
"""
function BayesInteractomics.community_dataframe(cr::CommunityResult)
    return DataFrame(
        Protein = cr.protein_names,
        Community = cr.membership
    )
end

# Pretty printing
function Base.show(io::IO, net::InteractionNetwork)
    n_nodes = nv(net.graph)
    n_edges = ne(net.graph)
    bait_info = isnothing(net.bait_protein) ? "none" : net.bait_protein
    println(io, "InteractionNetwork:")
    println(io, "  Nodes: $n_nodes proteins")
    println(io, "  Edges: $n_edges interactions")
    println(io, "  Bait: $bait_info")
    print(io, "  Thresholds: posterior=$(net.threshold_used.posterior_threshold), q=$(net.threshold_used.q_threshold)")
end

function Base.show(io::IO, stats::NetworkStatistics)
    println(io, "NetworkStatistics:")
    println(io, "  Nodes: $(stats.n_nodes)")
    println(io, "  Edges: $(stats.n_edges)")
    println(io, "  Density: $(round(stats.density, digits=4))")
    println(io, "  Avg degree: $(round(stats.avg_degree, digits=2))")
    println(io, "  Avg clustering: $(round(stats.avg_clustering, digits=4))")
    println(io, "  Components: $(stats.n_components)")
    if !isnothing(stats.diameter)
        println(io, "  Diameter: $(stats.diameter)")
        print(io, "  Avg path length: $(round(stats.avg_path_length, digits=2))")
    else
        print(io, "  Network is disconnected")
    end
end

function Base.show(io::IO, cm::CentralityMeasures)
    n = length(cm.protein_names)
    println(io, "CentralityMeasures for $n proteins")
    println(io, "  Top 3 by degree:")
    perm = sortperm(cm.degree, rev=true)
    for i in 1:min(3, n)
        idx = perm[i]
        println(io, "    $(cm.protein_names[idx]): $(cm.degree[idx])")
    end
end

function Base.show(io::IO, cr::CommunityResult)
    println(io, "CommunityResult:")
    println(io, "  Proteins: $(length(cr.protein_names))")
    println(io, "  Communities: $(cr.n_communities)")
    println(io, "  Modularity: $(round(cr.modularity, digits=4))")
    println(io, "  Community sizes: $(cr.community_sizes)")
end
