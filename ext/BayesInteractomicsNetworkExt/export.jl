# Network export functions

"""
    BayesInteractomics.export_graphml(net::InteractionNetwork, filename::String)

Export network to GraphML format for use in Cytoscape, Gephi, or other network tools.

GraphML is an XML-based format that preserves node and edge attributes.

# Arguments
- `net::InteractionNetwork`: Network to export
- `filename::String`: Output file path (should end in .graphml)

# Example
```julia
export_graphml(net, "network.graphml")
# Open in Cytoscape: File > Import > Network from File
```
"""
function BayesInteractomics.export_graphml(net::InteractionNetwork, filename::String)
    if !endswith(filename, ".graphml")
        filename = filename * ".graphml"
    end

    open(filename, "w") do io
        # Write GraphML header
        write(io, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        write(io, "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\n")
        write(io, "         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n")
        write(io, "         xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns\n")
        write(io, "         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n")

        # Define node attributes
        write(io, "  <key id=\"protein\" for=\"node\" attr.name=\"protein\" attr.type=\"string\"/>\n")
        if hasproperty(net.node_attributes, :posterior_prob)
            write(io, "  <key id=\"posterior_prob\" for=\"node\" attr.name=\"posterior_prob\" attr.type=\"double\"/>\n")
        end
        if hasproperty(net.node_attributes, :bayes_factor)
            write(io, "  <key id=\"bayes_factor\" for=\"node\" attr.name=\"bayes_factor\" attr.type=\"double\"/>\n")
        end
        if hasproperty(net.node_attributes, :q_value)
            write(io, "  <key id=\"q_value\" for=\"node\" attr.name=\"q_value\" attr.type=\"double\"/>\n")
        end
        if hasproperty(net.node_attributes, :mean_log2fc)
            write(io, "  <key id=\"log2fc\" for=\"node\" attr.name=\"log2fc\" attr.type=\"double\"/>\n")
        end
        if hasproperty(net.node_attributes, :is_bait)
            write(io, "  <key id=\"is_bait\" for=\"node\" attr.name=\"is_bait\" attr.type=\"boolean\"/>\n")
        end

        # Define edge attributes
        write(io, "  <key id=\"weight\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>\n")
        if hasproperty(net.edge_attributes, :edge_source)
            write(io, "  <key id=\"edge_source\" for=\"edge\" attr.name=\"edge_source\" attr.type=\"string\"/>\n")
        end
        if hasproperty(net.edge_attributes, :string_score)
            write(io, "  <key id=\"string_score\" for=\"edge\" attr.name=\"string_score\" attr.type=\"int\"/>\n")
        end

        # Start graph
        write(io, "  <graph id=\"G\" edgedefault=\"directed\">\n")

        # Write nodes
        for i in 1:nv(net.graph)
            write(io, "    <node id=\"n$i\">\n")
            write(io, "      <data key=\"protein\">$(net.protein_names[i])</data>\n")

            if hasproperty(net.node_attributes, :posterior_prob)
                val = net.node_attributes.posterior_prob[i]
                if !ismissing(val)
                    write(io, "      <data key=\"posterior_prob\">$val</data>\n")
                end
            end

            if hasproperty(net.node_attributes, :bayes_factor)
                val = net.node_attributes.bayes_factor[i]
                if !ismissing(val)
                    write(io, "      <data key=\"bayes_factor\">$val</data>\n")
                end
            end

            if hasproperty(net.node_attributes, :q_value)
                val = net.node_attributes.q_value[i]
                if !ismissing(val)
                    write(io, "      <data key=\"q_value\">$val</data>\n")
                end
            end

            if hasproperty(net.node_attributes, :mean_log2fc)
                val = net.node_attributes.mean_log2fc[i]
                if !ismissing(val)
                    write(io, "      <data key=\"log2fc\">$val</data>\n")
                end
            end

            if hasproperty(net.node_attributes, :is_bait)
                val = net.node_attributes.is_bait[i]
                write(io, "      <data key=\"is_bait\">$(val ? "true" : "false")</data>\n")
            end

            write(io, "    </node>\n")
        end

        # Write edges
        edge_id = 0
        for e in edges(net.graph)
            edge_id += 1
            src_id = src(e)
            dst_id = dst(e)
            weight = get_weight(net.graph, src_id, dst_id)

            write(io, "    <edge id=\"e$edge_id\" source=\"n$src_id\" target=\"n$dst_id\">\n")
            write(io, "      <data key=\"weight\">$weight</data>\n")

            # Write edge_source and string_score if available
            if hasproperty(net.edge_attributes, :edge_source) && edge_id <= nrow(net.edge_attributes)
                edge_src = net.edge_attributes.edge_source[edge_id]
                if !ismissing(edge_src)
                    write(io, "      <data key=\"edge_source\">$edge_src</data>\n")
                end
            end
            if hasproperty(net.edge_attributes, :string_score) && edge_id <= nrow(net.edge_attributes)
                ss = net.edge_attributes.string_score[edge_id]
                if !ismissing(ss)
                    write(io, "      <data key=\"string_score\">$ss</data>\n")
                end
            end

            write(io, "    </edge>\n")
        end

        # Close graph and graphml
        write(io, "  </graph>\n")
        write(io, "</graphml>\n")
    end

    @info "Network exported to GraphML: $filename"
end

"""
    BayesInteractomics.export_edgelist(net::InteractionNetwork, filename::String; include_attributes=true)

Export network edge list to CSV format.

# Arguments
- `net::InteractionNetwork`: Network to export
- `filename::String`: Output file path (should end in .csv)
- `include_attributes::Bool=true`: Whether to include edge weights and attributes

# Example
```julia
export_edgelist(net, "edges.csv")
```
"""
function BayesInteractomics.export_edgelist(
    net::InteractionNetwork,
    filename::String;
    include_attributes::Bool = true
)
    if !endswith(filename, ".csv")
        filename = filename * ".csv"
    end

    # Create DataFrame from edge list
    edges_df = DataFrame(
        source = String[],
        target = String[],
        weight = Float64[]
    )

    for e in edges(net.graph)
        src_id = src(e)
        dst_id = dst(e)
        weight = get_weight(net.graph, src_id, dst_id)

        push!(edges_df, (
            source = net.protein_names[src_id],
            target = net.protein_names[dst_id],
            weight = weight
        ))
    end

    # Add additional attributes if available and requested
    if include_attributes && !isempty(net.edge_attributes)
        # Try to merge with edge_attributes if they match
        if nrow(edges_df) == nrow(net.edge_attributes)
            for col in names(net.edge_attributes)
                if !(col in names(edges_df))
                    edges_df[!, col] = net.edge_attributes[!, col]
                end
            end
        end
    end

    CSV.write(filename, edges_df)
    @info "Edge list exported to: $filename"
end

"""
    BayesInteractomics.export_node_attributes(net::InteractionNetwork, filename::String)

Export node attributes to CSV format.

# Arguments
- `net::InteractionNetwork`: Network to export
- `filename::String`: Output file path (should end in .csv)

# Example
```julia
export_node_attributes(net, "nodes.csv")
```
"""
function BayesInteractomics.export_node_attributes(net::InteractionNetwork, filename::String)
    if !endswith(filename, ".csv")
        filename = filename * ".csv"
    end

    # Use the node_attributes DataFrame directly
    CSV.write(filename, net.node_attributes)
    @info "Node attributes exported to: $filename"
end

"""
    export_network_bundle(net::InteractionNetwork, prefix::String)

Export network in multiple formats: GraphML, edge list CSV, and node attributes CSV.

# Arguments
- `net::InteractionNetwork`: Network to export
- `prefix::String`: Prefix for output files (e.g., "mynetwork" creates mynetwork.graphml, etc.)

# Example
```julia
export_network_bundle(net, "results/mynetwork")
# Creates:
#   results/mynetwork.graphml
#   results/mynetwork_edges.csv
#   results/mynetwork_nodes.csv
```
"""
function export_network_bundle(net::InteractionNetwork, prefix::String)
    export_graphml(net, "$(prefix).graphml")
    export_edgelist(net, "$(prefix)_edges.csv")
    export_node_attributes(net, "$(prefix)_nodes.csv")
    @info "Network bundle exported with prefix: $prefix"
end
