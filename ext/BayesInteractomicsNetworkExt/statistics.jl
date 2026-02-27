# Network topology statistics

"""
    BayesInteractomics.network_statistics(net::InteractionNetwork) -> NetworkStatistics

Compute comprehensive network topology statistics.

# Arguments
- `net::InteractionNetwork`: Network to analyze

# Returns
- `NetworkStatistics`: Object containing topology metrics

# Example
```julia
stats = network_statistics(net)
println("Network density: ", stats.density)
println("Average degree: ", stats.avg_degree)
```
"""
function BayesInteractomics.network_statistics(net::InteractionNetwork)
    g = net.graph
    n_nodes = nv(g)
    n_edges = ne(g)

    # Handle empty networks
    if n_nodes == 0
        return NetworkStatistics(0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, nothing, nothing)
    end

    # Basic metrics
    density = n_edges / (n_nodes * (n_nodes - 1))  # For directed graphs
    avg_degree = 2 * n_edges / n_nodes  # Average total degree (in + out)

    # Weighted degree
    weighted_degrees = Float64[]
    for v in vertices(g)
        in_weight = sum((get_weight(g, u, v) for u in inneighbors(g, v)); init=0.0)
        out_weight = sum((get_weight(g, v, u) for u in outneighbors(g, v)); init=0.0)
        push!(weighted_degrees, in_weight + out_weight)
    end
    avg_weighted_degree = mean(weighted_degrees)

    # Clustering coefficient (use undirected version)
    g_simple = SimpleDiGraph(g)
    clustering_coeffs = Float64[]
    for v in vertices(g_simple)
        neighbors_v = union(inneighbors(g_simple, v), outneighbors(g_simple, v))
        k = length(neighbors_v)
        if k < 2
            push!(clustering_coeffs, 0.0)
            continue
        end

        # Count edges between neighbors
        edges_between = 0
        for n1 in neighbors_v
            for n2 in neighbors_v
                if n1 < n2 && (has_edge(g_simple, n1, n2) || has_edge(g_simple, n2, n1))
                    edges_between += 1
                end
            end
        end

        coeff = 2 * edges_between / (k * (k - 1))
        push!(clustering_coeffs, coeff)
    end
    avg_clustering = mean(clustering_coeffs)

    # Connected components (treat as undirected)
    components = weakly_connected_components(g_simple)
    n_components = length(components)
    largest_component_size = maximum(length.(components))

    # Diameter and average path length (use undirected graph for reachability)
    diameter = nothing
    avg_path_length = nothing

    # Convert to undirected for path analysis (directed star topologies have
    # many unreachable pairs, producing infinite distances)
    g_undirected = SimpleGraph(n_nodes)
    for e in edges(g_simple)
        add_edge!(g_undirected, src(e), dst(e))
    end
    undirected_components = connected_components(g_undirected)

    if length(undirected_components) == 1
        # Network is connected (undirected)
        try
            diameter = Graphs.diameter(g_undirected)

            # Compute average path length
            all_paths = Float64[]
            for i in vertices(g_undirected)
                dists = gdistances(g_undirected, i)
                for d in dists
                    if d > 0 && d < typemax(Int)
                        push!(all_paths, Float64(d))
                    end
                end
            end
            if !isempty(all_paths)
                avg_path_length = mean(all_paths)
            end
        catch e
            @warn "Could not compute diameter/path length: $e"
        end
    end

    return NetworkStatistics(
        n_nodes,
        n_edges,
        density,
        avg_degree,
        avg_weighted_degree,
        avg_clustering,
        n_components,
        largest_component_size,
        diameter,
        avg_path_length
    )
end

"""
    edge_source_summary(net::InteractionNetwork) -> Dict{String, Int}

Count edges by source type (e.g., "experimental" vs "public_ppi").
Returns a dictionary mapping edge source labels to edge counts.
If no `edge_source` column exists, all edges are counted as "experimental".
"""
function BayesInteractomics.edge_source_summary(net::InteractionNetwork)
    if !hasproperty(net.edge_attributes, :edge_source) || nrow(net.edge_attributes) == 0
        return Dict("experimental" => ne(net.graph))
    end
    counts = Dict{String, Int}()
    for src in net.edge_attributes.edge_source
        label = ismissing(src) ? "unknown" : string(src)
        counts[label] = get(counts, label, 0) + 1
    end
    return counts
end
