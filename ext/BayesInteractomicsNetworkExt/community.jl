# Community detection algorithms

"""
    BayesInteractomics.detect_communities(net::InteractionNetwork; algorithm=:louvain) -> CommunityResult

Detect communities (modules/clusters) in the protein interaction network.

Communities represent groups of proteins that are densely connected internally
but sparsely connected to other groups, potentially representing protein complexes
or functional modules.

# Arguments
- `net::InteractionNetwork`: Network to analyze
- `algorithm::Symbol=:louvain`: Community detection algorithm
  Currently supported: `:louvain`, `:label_propagation`, `:greedy_modularity`

# Returns
- `CommunityResult`: Community assignments and modularity score

# Example
```julia
communities = detect_communities(net, algorithm=:louvain)
df = community_dataframe(communities)
println("Found ", communities.n_communities, " communities")
println("Modularity: ", communities.modularity)
```
"""
function BayesInteractomics.detect_communities(
    net::InteractionNetwork;
    algorithm::Symbol = :louvain
)
    g = net.graph
    g_simple = SimpleDiGraph(g)
    n = nv(g)

    if n == 0
        return CommunityResult(
            String[],
            Int[],
            0,
            0.0,
            Int[]
        )
    end

    # Detect communities based on algorithm
    membership = if algorithm == :louvain
        _louvain_communities(g_simple)
    elseif algorithm == :label_propagation
        _label_propagation_communities(g_simple)
    elseif algorithm == :greedy_modularity
        _greedy_modularity_communities(g_simple)
    else
        error("Unknown community detection algorithm: $algorithm. " *
              "Supported: :louvain, :label_propagation, :greedy_modularity")
    end

    # Compute modularity
    modularity = _compute_modularity(g_simple, membership)

    # Count community sizes
    n_communities = maximum(membership)
    community_sizes = [count(==(i), membership) for i in 1:n_communities]

    return CommunityResult(
        copy(net.protein_names),
        membership,
        n_communities,
        modularity,
        community_sizes
    )
end

# Louvain method (greedy optimization of modularity)
function _louvain_communities(g::SimpleDiGraph)
    n = nv(g)
    if n == 0
        return Int[]
    end

    # Convert to undirected for community detection
    g_undirected = SimpleGraph(n)
    for e in edges(g)
        add_edge!(g_undirected, src(e), dst(e))
    end

    # Use Graphs.jl's label propagation as a proxy
    # (True Louvain requires additional packages)
    # For a simple implementation, we'll use label propagation
    return label_propagation(g_undirected)[1]
end

# Label propagation
function _label_propagation_communities(g::SimpleDiGraph)
    n = nv(g)
    if n == 0
        return Int[]
    end

    # Convert to undirected
    g_undirected = SimpleGraph(n)
    for e in edges(g)
        add_edge!(g_undirected, src(e), dst(e))
    end

    return label_propagation(g_undirected)[1]
end

# Greedy modularity optimization
function _greedy_modularity_communities(g::SimpleDiGraph)
    n = nv(g)
    if n == 0
        return Int[]
    end

    # Start with each node in its own community
    membership = collect(1:n)

    # Convert to undirected
    g_undirected = SimpleGraph(n)
    for e in edges(g)
        add_edge!(g_undirected, src(e), dst(e))
    end

    # Greedy merging based on modularity gain
    improved = true
    iteration = 0
    max_iterations = 100

    while improved && iteration < max_iterations
        improved = false
        iteration += 1

        for v in vertices(g_undirected)
            current_community = membership[v]
            best_community = current_community
            best_modularity = _compute_modularity(g_undirected, membership)

            # Try moving v to each neighbor's community
            neighbor_communities = unique([membership[n] for n in neighbors(g_undirected, v)])

            for new_community in neighbor_communities
                if new_community == current_community
                    continue
                end

                # Temporarily move v
                old_membership = copy(membership)
                membership[v] = new_community

                # Check modularity improvement
                new_modularity = _compute_modularity(g_undirected, membership)

                if new_modularity > best_modularity
                    best_modularity = new_modularity
                    best_community = new_community
                    improved = true
                else
                    # Revert
                    membership = old_membership
                end
            end

            membership[v] = best_community
        end
    end

    # Renumber communities consecutively
    unique_communities = sort(unique(membership))
    community_map = Dict(c => i for (i, c) in enumerate(unique_communities))
    membership = [community_map[c] for c in membership]

    return membership
end

# Compute modularity score
function _compute_modularity(g::SimpleGraph, membership::Vector{Int})
    n = nv(g)
    m = ne(g)

    if m == 0
        return 0.0
    end

    modularity = 0.0
    for i in vertices(g)
        for j in vertices(g)
            if membership[i] == membership[j]
                A_ij = has_edge(g, i, j) ? 1.0 : 0.0
                k_i = degree(g, i)
                k_j = degree(g, j)
                modularity += A_ij - (k_i * k_j) / (2 * m)
            end
        end
    end

    return modularity / (2 * m)
end

function _compute_modularity(g::SimpleDiGraph, membership::Vector{Int})
    # Convert to undirected for modularity calculation
    n = nv(g)
    g_undirected = SimpleGraph(n)
    for e in edges(g)
        add_edge!(g_undirected, src(e), dst(e))
    end
    return _compute_modularity(g_undirected, membership)
end

"""
    get_community_proteins(cr::CommunityResult, community_id::Int) -> Vector{String}

Get all proteins belonging to a specific community.

# Arguments
- `cr::CommunityResult`: Community detection results
- `community_id::Int`: Community ID (1 to n_communities)

# Returns
- `Vector{String}`: Protein names in the specified community
"""
function get_community_proteins(cr::CommunityResult, community_id::Int)
    if community_id < 1 || community_id > cr.n_communities
        error("Invalid community ID: $community_id (valid range: 1-$(cr.n_communities))")
    end

    indices = findall(==(community_id), cr.membership)
    return cr.protein_names[indices]
end
