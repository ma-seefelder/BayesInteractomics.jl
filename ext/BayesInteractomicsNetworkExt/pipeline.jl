# Network analysis pipeline - orchestrates the complete network analysis workflow

using Dates

# ============================================================
# Pipeline entry point
# ============================================================

function BayesInteractomics.run_network_analysis(
    ar::BayesInteractomics.AbstractAnalysisResult,
    config::BayesInteractomics.NetworkConfig = BayesInteractomics.NetworkConfig()
)
    warnings = String[]
    export_paths = Dict{String, String}()

    # --- Validate config ---
    _validate_config(config)

    # --- Apply bait info from config (override AnalysisResult values) ---
    ar = _apply_bait_info(ar, config)

    # --- Create output directory ---
    if config.export_files || config.generate_report || config.plot
        mkpath(config.output_dir)
    end

    config.verbose && @info "Building network (posterior ≥ $(config.posterior_threshold), q ≤ $(config.q_threshold))"

    # --- 1. Build network ---
    net = BayesInteractomics.build_network(ar;
        posterior_threshold = config.posterior_threshold,
        bf_threshold = config.bf_threshold,
        q_threshold = config.q_threshold,
        log2fc_threshold = config.log2fc_threshold,
        include_bait = config.include_bait,
        weight_by = config.weight_by
    )

    # Early return if empty
    if nv(net.graph) == 0
        push!(warnings, "No interactions passed filtering criteria. Network is empty.")
        config.verbose && @warn "Empty network — no interactions passed thresholds"
        return BayesInteractomics.NetworkPipelineResult(
            net, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
            config, nothing, nothing, export_paths, warnings
        )
    end

    config.verbose && @info "Network: $(nv(net.graph)) nodes, $(ne(net.graph)) edges"

    # --- 2. Optional PPI enrichment ---
    enriched_net = nothing
    if config.enrich
        config.verbose && @info "Enriching network with STRING PPI data (species=$(config.species))"
        try
            enriched_net = BayesInteractomics.enrich_network(net;
                species = config.species,
                min_string_score = config.min_string_score,
                network_type = config.network_type,
                co_purification_prior = config.co_purification_prior,
                string_prior = config.string_prior,
                use_bayesian_weighting = config.use_bayesian_weighting,
                channels = config.channels,
                force_refresh = config.force_refresh,
                protein_mapping = config.protein_mapping,
                offline_file = config.offline_file,
                verbose = config.ppi_verbose
            )
            config.verbose && @info "Enriched network: $(nv(enriched_net.graph)) nodes, $(ne(enriched_net.graph)) edges"
        catch e
            msg = "PPI enrichment failed: $(sprint(showerror, e))"
            push!(warnings, msg)
            config.verbose && @warn msg
        end
    end

    active_net = isnothing(enriched_net) ? net : enriched_net

    # --- 3. Network statistics ---
    config.verbose && @info "Computing network statistics"
    stats = BayesInteractomics.network_statistics(active_net)

    # --- 4. Centrality measures ---
    cm = nothing
    top_hubs = nothing
    if config.compute_centrality
        config.verbose && @info "Computing centrality measures"
        cm = BayesInteractomics.centrality_measures(active_net)
        try
            top_hubs = BayesInteractomics.get_top_hubs(cm; by=config.top_hubs_by, n=config.top_hubs_n)
        catch e
            msg = "Top hubs computation failed: $(sprint(showerror, e))"
            push!(warnings, msg)
            config.verbose && @warn msg
        end
    end

    # --- 5. Community detection ---
    communities = nothing
    if config.detect_communities
        if nv(active_net.graph) >= 3
            config.verbose && @info "Detecting communities ($(config.community_algorithm))"
            communities = BayesInteractomics.detect_communities(active_net; algorithm=config.community_algorithm)
        else
            msg = "Network too small for community detection ($(nv(active_net.graph)) nodes, need ≥ 3)"
            push!(warnings, msg)
            config.verbose && @info msg
        end
    end

    # --- 6. Edge source summary ---
    edge_sources = BayesInteractomics.edge_source_summary(active_net)

    # --- 7. Visualization ---
    plot_obj = nothing
    if config.plot
        config.verbose && @info "Generating network plot"
        try
            plot_obj = BayesInteractomics.plot_network(active_net;
                layout = config.layout,
                node_size = config.node_size,
                node_color = config.node_color,
                edge_width = config.edge_width,
                edge_style = config.edge_style,
                show_labels = config.show_labels,
                show_bait = config.show_bait,
                highlight_bait = config.highlight_bait,
                show_legend = config.show_legend,
                figsize = config.figsize
            )

            # Save plot
            ext = string(config.plot_format)
            plot_filename = joinpath(config.output_dir, "$(config.file_prefix)_plot.$(ext)")
            BayesInteractomics.save_network_plot(plot_obj, plot_filename; figsize=config.figsize)
            export_paths["plot"] = plot_filename
            config.verbose && @info "Plot saved to $(plot_filename)"
        catch e
            msg = "Visualization failed: $(sprint(showerror, e))"
            push!(warnings, msg)
            config.verbose && @warn msg
        end
    end

    # --- 8. File exports ---
    if config.export_files
        prefix = joinpath(config.output_dir, config.file_prefix)

        if config.export_graphml
            graphml_path = "$(prefix).graphml"
            try
                BayesInteractomics.export_graphml(active_net, graphml_path)
                export_paths["graphml"] = graphml_path
            catch e
                push!(warnings, "GraphML export failed: $(sprint(showerror, e))")
            end
        end

        if config.export_edgelist
            edges_path = "$(prefix)_edges.csv"
            try
                BayesInteractomics.export_edgelist(active_net, edges_path)
                export_paths["edgelist"] = edges_path
            catch e
                push!(warnings, "Edge list export failed: $(sprint(showerror, e))")
            end
        end

        if config.export_node_attributes
            nodes_path = "$(prefix)_nodes.csv"
            try
                BayesInteractomics.export_node_attributes(active_net, nodes_path)
                export_paths["node_attributes"] = nodes_path
            catch e
                push!(warnings, "Node attributes export failed: $(sprint(showerror, e))")
            end
        end

        # Export centrality if computed
        if !isnothing(cm)
            centrality_path = "$(prefix)_centrality.csv"
            try
                cdf = BayesInteractomics.centrality_dataframe(cm)
                CSV.write(centrality_path, cdf)
                export_paths["centrality"] = centrality_path
                config.verbose && @info "Centrality exported to $(centrality_path)"
            catch e
                push!(warnings, "Centrality export failed: $(sprint(showerror, e))")
            end
        end

        # Export communities if detected
        if !isnothing(communities)
            communities_path = "$(prefix)_communities.csv"
            try
                comm_df = BayesInteractomics.community_dataframe(communities)
                CSV.write(communities_path, comm_df)
                export_paths["communities"] = communities_path
                config.verbose && @info "Communities exported to $(communities_path)"
            catch e
                push!(warnings, "Community export failed: $(sprint(showerror, e))")
            end
        end
    end

    # --- 9. Assemble result ---
    result = BayesInteractomics.NetworkPipelineResult(
        net, enriched_net, stats, cm, top_hubs, communities, edge_sources,
        plot_obj, config, nothing, nothing, export_paths, warnings
    )

    # --- 10. Generate report ---
    if config.generate_report
        config.verbose && @info "Generating Markdown report"
        report_path, report_content = BayesInteractomics.generate_network_report(result)
        # Reconstruct with report fields populated
        result = BayesInteractomics.NetworkPipelineResult(
            result.network, result.enriched_network, result.statistics,
            result.centrality, result.top_hubs, result.communities,
            result.edge_sources, result.plot_object, result.config,
            report_path, report_content, result.export_paths, result.warnings
        )
    end

    config.verbose && @info "Network analysis pipeline complete"
    return result
end

# ============================================================
# Report generation
# ============================================================

function BayesInteractomics.generate_network_report(
    result::BayesInteractomics.NetworkPipelineResult;
    filename::Union{String, Nothing} = nothing,
    title::Union{String, Nothing} = nothing
)
    cfg = result.config
    report_title = isnothing(title) ? cfg.report_title : title
    report_path = isnothing(filename) ? joinpath(cfg.output_dir, cfg.report_filename) : filename

    mkpath(dirname(report_path))

    io = IOBuffer()

    # Header
    println(io, "# ", report_title)
    println(io)
    println(io, "Generated: ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
    println(io, "Package: BayesInteractomics.jl")
    println(io)

    # Parameters
    _report_parameters!(io, cfg)

    # Topology
    _report_topology!(io, result.statistics)

    # Edge sources
    _report_edge_sources!(io, result.edge_sources, !isnothing(result.enriched_network))

    # Hub proteins
    _report_hubs!(io, result.top_hubs, cfg)

    # Communities
    _report_communities!(io, result.communities)

    # Bait protein info
    _report_bait_info!(io, result.network)

    # Visualization
    _report_visualization!(io, result.export_paths, cfg)

    # Exported files
    _report_exports!(io, result.export_paths)

    # Warnings
    _report_warnings!(io, result.warnings)

    content = String(take!(io))

    open(report_path, "w") do f
        write(f, content)
    end

    return report_path, content
end

# ============================================================
# Bait info application
# ============================================================

"""
Apply bait_protein / bait_index from NetworkConfig onto the AnalysisResult.
Config values override AnalysisResult values when set (not nothing).
For mutable AnalysisResult, mutates in place. For immutable NetworkAnalysisResult,
returns a new wrapper.
"""
function _apply_bait_info(ar::BayesInteractomics.AnalysisResult, config::BayesInteractomics.NetworkConfig)
    if !isnothing(config.bait_protein) || !isnothing(config.bait_index)
        BayesInteractomics.set_bait_info!(ar;
            bait_protein = config.bait_protein,
            bait_index = config.bait_index
        )
    end
    return ar
end

function _apply_bait_info(ar::BayesInteractomics.NetworkAnalysisResult, config::BayesInteractomics.NetworkConfig)
    bp = isnothing(config.bait_protein) ? ar.bait_protein : config.bait_protein
    bi = isnothing(config.bait_index) ? ar.bait_index : config.bait_index
    if bp !== ar.bait_protein || bi !== ar.bait_index
        return BayesInteractomics.NetworkAnalysisResult(ar.results; bait_protein=bp, bait_index=bi)
    end
    return ar
end

# Fallback for any other AbstractAnalysisResult subtypes
function _apply_bait_info(ar::BayesInteractomics.AbstractAnalysisResult, config::BayesInteractomics.NetworkConfig)
    return ar
end

# ============================================================
# Config validation
# ============================================================

function _validate_config(config::BayesInteractomics.NetworkConfig)
    valid_weight_by = (:posterior_prob, :bayes_factor, :log2fc)
    config.weight_by in valid_weight_by ||
        error("weight_by must be one of $valid_weight_by, got :$(config.weight_by)")

    valid_layouts = (:spring, :circular, :shell, :spectral)
    config.layout in valid_layouts ||
        error("layout must be one of $valid_layouts, got :$(config.layout)")

    valid_node_sizes = (:degree, :posterior_prob, :log2fc, :uniform)
    config.node_size in valid_node_sizes ||
        error("node_size must be one of $valid_node_sizes, got :$(config.node_size)")

    valid_node_colors = (:posterior_prob, :log2fc, :community, :uniform)
    config.node_color in valid_node_colors ||
        error("node_color must be one of $valid_node_colors, got :$(config.node_color)")

    valid_community_algos = (:louvain, :label_propagation, :greedy_modularity)
    config.community_algorithm in valid_community_algos ||
        error("community_algorithm must be one of $valid_community_algos, got :$(config.community_algorithm)")

    valid_hubs_by = (:degree, :weighted_degree, :betweenness, :closeness, :eigenvector, :pagerank)
    config.top_hubs_by in valid_hubs_by ||
        error("top_hubs_by must be one of $valid_hubs_by, got :$(config.top_hubs_by)")

    valid_plot_formats = (:png, :pdf, :svg)
    config.plot_format in valid_plot_formats ||
        error("plot_format must be one of $valid_plot_formats, got :$(config.plot_format)")

    config.top_hubs_n > 0 ||
        error("top_hubs_n must be positive, got $(config.top_hubs_n)")

    config.posterior_threshold >= 0.0 && config.posterior_threshold <= 1.0 ||
        error("posterior_threshold must be in [0, 1], got $(config.posterior_threshold)")

    config.q_threshold >= 0.0 && config.q_threshold <= 1.0 ||
        error("q_threshold must be in [0, 1], got $(config.q_threshold)")
end

# ============================================================
# Report section helpers
# ============================================================

function _report_parameters!(io::IOBuffer, cfg::BayesInteractomics.NetworkConfig)
    println(io, "## Analysis Parameters")
    println(io)
    println(io, "### Network Construction")
    println(io)
    println(io, "| Parameter | Value |")
    println(io, "|-----------|-------|")
    println(io, "| Posterior threshold | $(cfg.posterior_threshold) |")
    println(io, "| Bayes factor threshold | $(isnothing(cfg.bf_threshold) ? "none" : cfg.bf_threshold) |")
    println(io, "| Q-value threshold | $(cfg.q_threshold) |")
    println(io, "| log2FC threshold | $(isnothing(cfg.log2fc_threshold) ? "none" : cfg.log2fc_threshold) |")
    println(io, "| Include bait | $(cfg.include_bait) |")
    println(io, "| Edge weight | $(cfg.weight_by) |")
    println(io)

    if cfg.enrich
        println(io, "### PPI Enrichment")
        println(io)
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| Species (NCBI) | $(cfg.species) |")
        println(io, "| Min STRING score | $(cfg.min_string_score) |")
        println(io, "| Network type | $(cfg.network_type) |")
        println(io, "| Bayesian weighting | $(cfg.use_bayesian_weighting) |")
        println(io, "| Channels | $(join(string.(cfg.channels), ", ")) |")
        println(io)
    end
end

function _report_topology!(io::IOBuffer, stats)
    println(io, "## Network Topology")
    println(io)

    if isnothing(stats) || stats.n_nodes == 0
        println(io, "Network is empty (no interactions passed filtering criteria).")
        println(io)
        return
    end

    println(io, "| Metric | Value |")
    println(io, "|--------|-------|")
    println(io, "| Nodes | $(stats.n_nodes) |")
    println(io, "| Edges | $(stats.n_edges) |")
    println(io, "| Density | $(round(stats.density, digits=4)) |")
    println(io, "| Avg degree | $(round(stats.avg_degree, digits=2)) |")
    println(io, "| Avg weighted degree | $(round(stats.avg_weighted_degree, digits=2)) |")
    println(io, "| Avg clustering coefficient | $(round(stats.avg_clustering, digits=4)) |")
    println(io, "| Connected components | $(stats.n_components) |")
    println(io, "| Largest component | $(stats.largest_component_size) |")
    if !isnothing(stats.diameter)
        println(io, "| Diameter | $(stats.diameter) |")
    end
    if !isnothing(stats.avg_path_length)
        println(io, "| Avg path length | $(round(stats.avg_path_length, digits=2)) |")
    end
    println(io)
end

function _report_edge_sources!(io::IOBuffer, edge_sources, is_enriched::Bool)
    if isnothing(edge_sources) || isempty(edge_sources)
        return
    end

    if is_enriched || length(edge_sources) > 1
        println(io, "### Edge Sources")
        println(io)
        println(io, "| Source | Count |")
        println(io, "|--------|-------|")
        for (source, count) in sort(collect(edge_sources), by=x->x[2], rev=true)
            println(io, "| $(source) | $(count) |")
        end
        println(io)
    end
end

function _report_hubs!(io::IOBuffer, top_hubs, cfg::BayesInteractomics.NetworkConfig)
    if isnothing(top_hubs) || nrow(top_hubs) == 0
        return
    end

    println(io, "## Hub Proteins")
    println(io)
    println(io, "Top $(nrow(top_hubs)) proteins ranked by $(cfg.top_hubs_by):")
    println(io)

    # Table header
    println(io, "| Rank | Protein | Degree | PageRank | Betweenness | Eigenvector |")
    println(io, "|------|---------|--------|----------|-------------|-------------|")

    for (i, row) in enumerate(eachrow(top_hubs))
        println(io, "| $(i) | $(row.Protein) | $(row.Degree) | $(round(row.PageRank, digits=4)) | $(round(row.Betweenness, digits=4)) | $(round(row.Eigenvector, digits=4)) |")
    end
    println(io)
end

function _report_communities!(io::IOBuffer, communities)
    if isnothing(communities) || communities.n_communities == 0
        return
    end

    println(io, "## Community Structure")
    println(io)
    println(io, "| Metric | Value |")
    println(io, "|--------|-------|")
    println(io, "| Communities detected | $(communities.n_communities) |")
    println(io, "| Modularity | $(round(communities.modularity, digits=4)) |")
    println(io)

    # Community size table
    println(io, "### Community Sizes")
    println(io)
    println(io, "| Community | Size | Members (up to 5) |")
    println(io, "|-----------|------|-------------------|")

    for i in 1:communities.n_communities
        size = communities.community_sizes[i]
        members_idx = findall(==(i), communities.membership)
        member_names = communities.protein_names[members_idx]
        shown = length(member_names) > 5 ? join(member_names[1:5], ", ") * ", ..." : join(member_names, ", ")
        println(io, "| $(i) | $(size) | $(shown) |")
    end
    println(io)
end

function _report_bait_info!(io::IOBuffer, net)
    bait_name = net.bait_protein
    if isnothing(bait_name)
        return
    end

    println(io, "## Bait Protein")
    println(io)
    println(io, "- **Bait**: $(bait_name)")

    n_interactors = nv(net.graph)
    if !isnothing(net.bait_index)
        n_interactors -= 1  # exclude bait itself
    end
    println(io, "- **Interactors**: $(n_interactors)")
    println(io)
end

function _report_visualization!(io::IOBuffer, export_paths::Dict{String, String}, cfg::BayesInteractomics.NetworkConfig)
    if !haskey(export_paths, "plot")
        return
    end

    println(io, "## Visualization")
    println(io)

    plot_path = export_paths["plot"]
    # Use relative path from report location
    rel_path = basename(plot_path)
    println(io, "![Network plot]($(rel_path))")
    println(io)
    println(io, "Layout: $(cfg.layout), Node color: $(cfg.node_color), Node size: $(cfg.node_size)")
    println(io)
end

function _report_exports!(io::IOBuffer, export_paths::Dict{String, String})
    if isempty(export_paths)
        return
    end

    println(io, "## Exported Files")
    println(io)
    println(io, "| Type | File |")
    println(io, "|------|------|")
    for (type, path) in sort(collect(export_paths))
        println(io, "| $(type) | `$(basename(path))` |")
    end
    println(io)
end

function _report_warnings!(io::IOBuffer, warnings::Vector{String})
    if isempty(warnings)
        return
    end

    println(io, "## Warnings")
    println(io)
    for w in warnings
        println(io, "- $(w)")
    end
    println(io)
end

# Pretty printing for NetworkPipelineResult
function Base.show(io::IO, r::BayesInteractomics.NetworkPipelineResult)
    n_nodes = nv(r.network.graph)
    n_edges = ne(r.network.graph)
    println(io, "NetworkPipelineResult:")
    println(io, "  Network: $(n_nodes) nodes, $(n_edges) edges")
    if !isnothing(r.enriched_network)
        println(io, "  Enriched: $(nv(r.enriched_network.graph)) nodes, $(ne(r.enriched_network.graph)) edges")
    end
    if !isnothing(r.statistics)
        println(io, "  Density: $(round(r.statistics.density, digits=4))")
    end
    if !isnothing(r.communities)
        println(io, "  Communities: $(r.communities.n_communities)")
    end
    if !isnothing(r.top_hubs)
        println(io, "  Top hubs: $(nrow(r.top_hubs))")
    end
    if !isnothing(r.report_path)
        println(io, "  Report: $(r.report_path)")
    end
    n_exports = length(r.export_paths)
    if n_exports > 0
        println(io, "  Exports: $(n_exports) files")
    end
    n_warnings = length(r.warnings)
    if n_warnings > 0
        print(io, "  Warnings: $(n_warnings)")
    end
end
