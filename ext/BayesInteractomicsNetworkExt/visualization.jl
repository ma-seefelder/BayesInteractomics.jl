# Network visualization using GraphPlot.jl and Compose.jl

# ============================================================
# Main API
# ============================================================

"""
    BayesInteractomics.plot_network(net::InteractionNetwork; kwargs...)

Visualize the protein interaction network with an optional legend panel.

For enriched networks (from `enrich_network`), the bait protein and its edges are
hidden by default to reveal prey-prey community structure. Use `show_bait=true` to
display the full network including the bait.

# Arguments
- `net::InteractionNetwork`: Network to visualize
- `layout::Symbol=:spring`: Layout algorithm (:spring, :circular, :shell, :spectral)
- `node_size::Symbol=:degree`: Node size mapping (:degree, :posterior_prob, :log2fc, :uniform)
- `node_color::Symbol=:posterior_prob`: Node color mapping (:posterior_prob, :log2fc, :community, :uniform)
- `edge_width::Symbol=:weight`: Edge width mapping (:weight, :uniform)
- `edge_style::Symbol=:by_source`: Edge color by evidence source (:by_source, :uniform)
- `show_labels::Bool=true`: Whether to show protein name labels
- `show_bait::Union{Bool,Nothing}=nothing`: Bait visibility (nothing = auto: hide in enriched networks)
- `highlight_bait::Bool=true`: Highlight bait protein with gold color when visible
- `show_legend::Bool=true`: Display a legend panel on the right side
- `figsize::Tuple{Int,Int}=(800,800)`: Figure size in pixels

# Returns
- Compose canvas that can be displayed or saved

# Example
```julia
plt = plot_network(net)                          # auto legend, auto bait visibility
plt = plot_network(net, show_bait=true)          # force bait visible
plt = plot_network(net, show_legend=false)        # no legend panel
plt = plot_network(net, node_color=:log2fc, node_size=:posterior_prob)

# Save to file
save_network_plot(plt, "network.png")
```
"""
function BayesInteractomics.plot_network(
    net::InteractionNetwork;
    layout::Symbol = :spring,
    node_size::Symbol = :degree,
    node_color::Symbol = :posterior_prob,
    edge_width::Symbol = :weight,
    edge_style::Symbol = :by_source,
    show_labels::Bool = true,
    show_bait::Union{Bool, Nothing} = nothing,
    highlight_bait::Bool = true,
    show_legend::Bool = true,
    figsize::Tuple{Int,Int} = (800, 800)
)
    n = nv(net.graph)
    if n == 0
        @warn "Network is empty, cannot plot"
        return compose(context())
    end

    # Determine network type and bait visibility
    enriched = _is_enriched(net)
    hide_bait = _should_hide_bait(net, show_bait, enriched)

    # Prepare visualization network (optionally without bait)
    if hide_bait
        vis_net, bait_meta = _create_vis_network(net)
        if nv(vis_net.graph) == 0
            @warn "No prey nodes remain after hiding bait"
            return compose(context())
        end
        if enriched
            @info string(
                "Enriched network: bait '", bait_meta.name, "' hidden (",
                bait_meta.n_interactors, " interactors, ",
                nv(vis_net.graph), " prey shown). Use show_bait=true to display."
            )
        else
            @info string(
                "Bait '", bait_meta.name, "' hidden (",
                bait_meta.n_interactors, " interactors). ",
                "Use show_bait=true to display."
            )
        end
    else
        vis_net = net
        bait_meta = nothing
    end

    # Build undirected graph for layout and rendering
    g_undirected = _to_undirected(vis_net.graph)
    n_vis = nv(g_undirected)

    # Compute visual properties
    locs_x, locs_y = _compute_layout(g_undirected, layout, n_vis)
    nodesizes = _compute_node_sizes(vis_net, node_size)
    nodecolors = _compute_node_colors(vis_net, node_color, !hide_bait && highlight_bait)
    edgewidths = _compute_edge_widths_ordered(g_undirected, vis_net, edge_width)
    edgecolors = _compute_edge_colors_ordered(g_undirected, vis_net, edge_style)
    nodelabels = show_labels ? vis_net.protein_names : nothing

    # Build network plot
    network_plot = gplot(
        g_undirected,
        locs_x, locs_y,
        nodelabel = nodelabels,
        nodefillc = nodecolors,
        nodesize = nodesizes,
        edgestrokec = edgecolors,
        edgelinewidth = edgewidths
    )

    # Compose final canvas (in Compose.jl, first child renders on top)
    bg = (context(), Compose.rectangle(), fill("white"))

    if show_legend
        legend = _build_legend(
            node_color_by = node_color,
            node_size_by = node_size,
            edge_width_by = edge_width,
            edge_style_by = edge_style,
            is_enriched = enriched,
            bait_hidden = hide_bait,
            bait_name = hide_bait ? bait_meta.name : net.bait_protein,
            n_bait_interactors = hide_bait ? bait_meta.n_interactors : 0,
            bait_highlighted = !hide_bait && highlight_bait && !isnothing(net.bait_index)
        )
        return compose(context(),
            compose(context(0, 0, 0.78, 1.0), network_plot),
            compose(context(0.78, 0, 0.22, 1.0), legend),
            bg
        )
    else
        return compose(context(), network_plot, bg)
    end
end

"""
    BayesInteractomics.save_network_plot(plt, filename::String; figsize=(800,800))

Save a Compose plot (e.g. from `plot_network`) to file.

# Arguments
- `plt`: Compose context returned by `plot_network()`
- `filename::String`: Output file path (.png, .pdf, or .svg)
- `figsize::Tuple{Int,Int}=(800,800)`: Output size in pixels

# Example
```julia
plt = plot_network(net, layout=:spring, node_color=:posterior_prob)
save_network_plot(plt, "network.png")
save_network_plot(plt, "network.svg", figsize=(1200, 1200))
```
"""
function BayesInteractomics.save_network_plot(
    plt::Compose.Context,
    filename::String;
    figsize::Tuple{Int,Int} = (800, 800)
)
    w, h = figsize
    w_cm = w / 72 * 2.54
    h_cm = h / 72 * 2.54
    if endswith(filename, ".png")
        draw(PNG(filename, w, h), plt)
    elseif endswith(filename, ".pdf")
        draw(PDF(filename, w_cm * Compose.cm, h_cm * Compose.cm), plt)
    elseif endswith(filename, ".svg")
        draw(SVG(filename, w_cm * Compose.cm, h_cm * Compose.cm), plt)
    else
        error("Unsupported file format. Use .png, .pdf, or .svg")
    end
    @info "Network plot saved to $filename ($(w)x$(h))"
end

# ============================================================
# Bait visibility helpers
# ============================================================

"""Check whether the network contains public PPI edges from enrichment."""
function _is_enriched(net::InteractionNetwork)
    !isempty(net.edge_attributes) &&
    hasproperty(net.edge_attributes, :edge_source) &&
    any(net.edge_attributes.edge_source .== "public_ppi")
end

"""Decide whether to hide the bait node in the visualization."""
function _should_hide_bait(net::InteractionNetwork, show_bait::Union{Bool, Nothing}, enriched::Bool)
    if !isnothing(show_bait)
        return !show_bait && !isnothing(net.bait_index)
    end
    # Auto: hide bait only for enriched networks
    return enriched && !isnothing(net.bait_index)
end

"""
Create a visualization-only InteractionNetwork without the bait node and its edges.
Returns `(vis_net, bait_meta)` where `bait_meta` is a NamedTuple with `name` and
`n_interactors`.
"""
function _create_vis_network(net::InteractionNetwork)
    bait_idx = net.bait_index
    n = nv(net.graph)
    keep = [i for i in 1:n if i != bait_idx]
    old_to_new = Dict(old => new for (new, old) in enumerate(keep))
    new_n = length(keep)

    # Count unique bait neighbors
    g_di = SimpleDiGraph(net.graph)
    bait_neighbors = Set{Int}()
    for e in edges(g_di)
        s, d = src(e), dst(e)
        if s == bait_idx
            push!(bait_neighbors, d)
        elseif d == bait_idx
            push!(bait_neighbors, s)
        end
    end

    # Build filtered weighted directed graph (prey-prey edges only)
    new_graph = SimpleWeightedDiGraph(new_n)
    for e in edges(g_di)
        s, d = src(e), dst(e)
        if s != bait_idx && d != bait_idx
            w = net.graph.weights[d, s]  # SimpleWeightedGraphs stores transposed
            add_edge!(new_graph, old_to_new[s], old_to_new[d], w)
        end
    end

    # Filter node attributes
    new_node_attrs = copy(net.node_attributes[keep, :])

    # Filter and remap edge attributes
    if !isempty(net.edge_attributes) && hasproperty(net.edge_attributes, :source_node)
        mask = [(net.edge_attributes.source_node[i] != bait_idx &&
                 net.edge_attributes.target_node[i] != bait_idx)
                for i in 1:nrow(net.edge_attributes)]
        new_edge_attrs = copy(net.edge_attributes[mask, :])
        new_edge_attrs.source_node .= [old_to_new[s] for s in new_edge_attrs.source_node]
        new_edge_attrs.target_node .= [old_to_new[d] for d in new_edge_attrs.target_node]
    else
        new_edge_attrs = DataFrame()
    end

    vis_net = InteractionNetwork(
        new_graph, net.protein_names[keep], new_node_attrs, new_edge_attrs,
        nothing, nothing, net.threshold_used
    )

    bait_meta = (name = net.bait_protein, n_interactors = length(bait_neighbors))
    return vis_net, bait_meta
end

"""Convert a SimpleWeightedDiGraph to an undirected SimpleGraph."""
function _to_undirected(wdg::SimpleWeightedDiGraph)
    n = nv(wdg)
    g = SimpleGraph(n)
    for e in edges(SimpleDiGraph(wdg))
        add_edge!(g, src(e), dst(e))
    end
    return g
end

# ============================================================
# Edge properties (lookup-based for correct undirected ordering)
# ============================================================

"""
Compute edge widths ordered by `edges(g_undirected)` iteration.
Uses a lookup table to correctly map edge attributes regardless of insertion order.
"""
function _compute_edge_widths_ordered(g::SimpleGraph, net::InteractionNetwork, width_by::Symbol)
    n_e = ne(g)
    if n_e == 0 || width_by == :uniform
        return ones(Float64, n_e)
    end

    if isempty(net.edge_attributes) ||
       !hasproperty(net.edge_attributes, :weight) ||
       !hasproperty(net.edge_attributes, :source_node)
        return ones(Float64, n_e)
    end

    # Build (min_node, max_node) -> weight lookup
    lookup = Dict{Tuple{Int,Int}, Float64}()
    for row in eachrow(net.edge_attributes)
        s, d = row.source_node, row.target_node
        key = s < d ? (s, d) : (d, s)
        lookup[key] = row.weight
    end

    # src(e) < dst(e) guaranteed for SimpleGraph edges
    weights = [get(lookup, (src(e), dst(e)), 1.0) for e in edges(g)]

    max_w, min_w = maximum(weights), minimum(weights)
    if max_w > min_w
        return 0.5 .+ 2.5 .* (weights .- min_w) ./ (max_w - min_w)
    end
    return ones(Float64, n_e)
end

"""
Compute edge colors ordered by `edges(g_undirected)` iteration.
Uses a lookup table to correctly map edge source regardless of insertion order.
"""
function _compute_edge_colors_ordered(g::SimpleGraph, net::InteractionNetwork, edge_style::Symbol)
    n_e = ne(g)
    if n_e == 0
        return fill(colorant"gray", 0)
    end

    if edge_style != :by_source ||
       !hasproperty(net.edge_attributes, :edge_source) ||
       !hasproperty(net.edge_attributes, :source_node)
        return fill(colorant"gray", n_e)
    end

    # Build (min_node, max_node) -> edge_source lookup
    lookup = Dict{Tuple{Int,Int}, String}()
    for row in eachrow(net.edge_attributes)
        s, d = row.source_node, row.target_node
        key = s < d ? (s, d) : (d, s)
        lookup[key] = row.edge_source
    end

    return [begin
        source = get(lookup, (src(e), dst(e)), "experimental")
        source == "experimental" ? colorant"gray60" : colorant"#6baed6"
    end for e in edges(g)]
end

# ============================================================
# Node properties
# ============================================================

function _compute_node_sizes(net::InteractionNetwork, size_by::Symbol)
    n = nv(net.graph)

    if size_by == :uniform
        return ones(Float64, n)
    elseif size_by == :degree
        g_simple = SimpleDiGraph(net.graph)
        degrees = [indegree(g_simple, v) + outdegree(g_simple, v) for v in vertices(g_simple)]
        max_deg = maximum(degrees; init=0)
        return max_deg > 0 ? Float64.(degrees) ./ max_deg : ones(Float64, n)
    elseif size_by == :posterior_prob
        if hasproperty(net.node_attributes, :posterior_prob)
            probs = coalesce.(net.node_attributes.posterior_prob, 0.0)
            return Float64.(probs)
        else
            @warn "No posterior probabilities available, using uniform size"
            return ones(Float64, n)
        end
    elseif size_by == :log2fc
        if hasproperty(net.node_attributes, :mean_log2fc)
            lfc = coalesce.(net.node_attributes.mean_log2fc, 0.0)
            abs_lfc = abs.(Float64.(lfc))
            max_lfc = maximum(abs_lfc; init=0.0)
            return max_lfc > 0 ? abs_lfc ./ max_lfc : ones(Float64, n)
        else
            @warn "No log2FC values available, using uniform size"
            return ones(Float64, n)
        end
    else
        @warn "Unknown size mapping $size_by, using uniform size"
        return ones(Float64, n)
    end
end

function _compute_node_colors(net::InteractionNetwork, color_by::Symbol, highlight_bait::Bool)
    n = nv(net.graph)

    if color_by == :uniform
        colors = fill(colorant"steelblue", n)
    elseif color_by == :posterior_prob
        if hasproperty(net.node_attributes, :posterior_prob)
            probs = coalesce.(net.node_attributes.posterior_prob, 0.0)
            colors = [get_gradient_color(Float64(p)) for p in probs]
        else
            @warn "No posterior probabilities available, using uniform color"
            colors = fill(colorant"steelblue", n)
        end
    elseif color_by == :log2fc
        if hasproperty(net.node_attributes, :mean_log2fc)
            lfc = coalesce.(net.node_attributes.mean_log2fc, 0.0)
            colors = [get_diverging_color(Float64(fc)) for fc in lfc]
        else
            @warn "No log2FC values available, using uniform color"
            colors = fill(colorant"steelblue", n)
        end
    elseif color_by == :community
        @warn "Community coloring requires running detect_communities first. Using uniform color."
        colors = fill(colorant"steelblue", n)
    else
        @warn "Unknown color mapping $color_by, using uniform color"
        colors = fill(colorant"steelblue", n)
    end

    # Highlight bait if requested and present
    if highlight_bait && !isnothing(net.bait_index)
        colors[net.bait_index] = colorant"gold"
    end

    return colors
end

# ============================================================
# Layout
# ============================================================

function _compute_layout(g::SimpleGraph, layout::Symbol, n::Int)
    if layout == :spring
        return _spring_layout(g, n)
    elseif layout == :circular
        return _circular_layout(n)
    elseif layout == :shell
        return _circular_layout(n)
    elseif layout == :spectral
        return _spring_layout(g, n)
    else
        @warn "Unknown layout $layout, using spring layout"
        return _spring_layout(g, n)
    end
end

"""
Fruchterman-Reingold force-directed layout with adaptive cooling.
Produces well-spread node positions for star and general topologies.
"""
function _spring_layout(g::SimpleGraph, n::Int; iterations::Int=300)
    if n == 0
        return Float64[], Float64[]
    end
    if n == 1
        return [0.0], [0.0]
    end

    # Optimal spacing: k = C * sqrt(area / n)
    area = Float64(n)
    k = sqrt(area / n)

    # Initialize with circular + jitter to avoid symmetry traps
    locs_x = [cos(2π * i / n) + 0.05 * randn() for i in 1:n]
    locs_y = [sin(2π * i / n) + 0.05 * randn() for i in 1:n]

    # Displacement accumulators
    disp_x = zeros(n)
    disp_y = zeros(n)

    temp = 0.1 * sqrt(area)  # initial temperature

    for iter in 1:iterations
        fill!(disp_x, 0.0)
        fill!(disp_y, 0.0)

        # Repulsive forces between all pairs
        for i in 1:n
            for j in (i+1):n
                dx = locs_x[i] - locs_x[j]
                dy = locs_y[i] - locs_y[j]
                dist = sqrt(dx^2 + dy^2) + 1e-8
                # Repulsive force: k² / dist
                force = (k * k) / dist
                fx = force * dx / dist
                fy = force * dy / dist
                disp_x[i] += fx
                disp_y[i] += fy
                disp_x[j] -= fx
                disp_y[j] -= fy
            end
        end

        # Attractive forces along edges
        for e in edges(g)
            i, j = src(e), dst(e)
            dx = locs_x[i] - locs_x[j]
            dy = locs_y[i] - locs_y[j]
            dist = sqrt(dx^2 + dy^2) + 1e-8
            # Attractive force: dist² / k
            force = (dist * dist) / k
            fx = force * dx / dist
            fy = force * dy / dist
            disp_x[i] -= fx
            disp_y[i] -= fy
            disp_x[j] += fx
            disp_y[j] += fy
        end

        # Apply displacements clamped by temperature
        for i in 1:n
            disp_len = sqrt(disp_x[i]^2 + disp_y[i]^2) + 1e-8
            scale = min(disp_len, temp) / disp_len
            locs_x[i] += disp_x[i] * scale
            locs_y[i] += disp_y[i] * scale
        end

        # Cool down
        temp *= (1.0 - iter / iterations)
    end

    # Normalize to [-1, 1] range
    min_x, max_x = extrema(locs_x)
    min_y, max_y = extrema(locs_y)
    rx = max_x - min_x
    ry = max_y - min_y
    r = max(rx, ry, 1e-8)
    locs_x .= (locs_x .- (min_x + max_x) / 2) ./ r .* 2
    locs_y .= (locs_y .- (min_y + max_y) / 2) ./ r .* 2

    return locs_x, locs_y
end

function _circular_layout(n::Int)
    angles = range(0, 2π, length=n+1)[1:n]
    locs_x = cos.(angles)
    locs_y = sin.(angles)
    return locs_x, locs_y
end

# ============================================================
# Color mapping
# ============================================================

function get_gradient_color(value::Float64)
    # Blue (0.0) -> Red (1.0)
    r = value
    g = 0.0
    b = 1.0 - value
    return RGB(r, g, b)
end

function get_diverging_color(value::Float64)
    # Blue (negative) -> White (0) -> Red (positive)
    # Assume value is in range [-5, 5] for log2FC
    norm_value = clamp(value / 5.0, -1.0, 1.0)

    if norm_value < 0
        # Blue gradient
        intensity = abs(norm_value)
        return RGB(1.0 - intensity, 1.0 - intensity, 1.0)
    else
        # Red gradient
        intensity = norm_value
        return RGB(1.0, 1.0 - intensity, 1.0 - intensity)
    end
end

# ============================================================
# Legend
# ============================================================

"""
Build the legend panel as a Compose context.
Shows node color scale, node size mapping, edge style, and bait information.
"""
function _build_legend(;
    node_color_by::Symbol,
    node_size_by::Symbol,
    edge_width_by::Symbol,
    edge_style_by::Symbol,
    is_enriched::Bool,
    bait_hidden::Bool,
    bait_name::Union{String, Nothing},
    n_bait_interactors::Int,
    bait_highlighted::Bool
)
    elements = Compose.Context[]
    y = 0.03

    # --- Title ---
    push!(elements, compose(context(),
        (context(), Compose.text(0.5, y, "Legend", hcenter, vtop),
         Compose.fontsize(8pt), fill("gray30"))))
    y += 0.045

    push!(elements, _legend_separator(y))
    y += 0.025

    # --- Node Color ---
    push!(elements, _legend_section_header(y, "Node color"))
    y += 0.04

    if node_color_by == :posterior_prob
        # Gradient: blue -> red (5 circles)
        for (i, frac) in enumerate(range(0, 1, length=5))
            cx = 0.08 + (i - 1) * 0.15
            c = get_gradient_color(frac)
            push!(elements, compose(context(),
                (context(), Compose.circle(cx, y, 0.012),
                 fill(c), Compose.stroke("gray60"), Compose.linewidth(0.1mm))))
        end
        y += 0.03
        push!(elements, _legend_annotation(y, "0 \u2190 Posterior \u2192 1"))
        y += 0.035

    elseif node_color_by == :log2fc
        # Diverging: blue -> white -> red
        for (i, val) in enumerate([-4.0, -2.0, 0.0, 2.0, 4.0])
            cx = 0.08 + (i - 1) * 0.15
            c = get_diverging_color(val)
            push!(elements, compose(context(),
                (context(), Compose.circle(cx, y, 0.012),
                 fill(c), Compose.stroke("gray60"), Compose.linewidth(0.1mm))))
        end
        y += 0.03
        push!(elements, _legend_annotation(y, "-FC    0    +FC"))
        y += 0.035

    elseif node_color_by == :community
        palette = [colorant"#e41a1c", colorant"#377eb8", colorant"#4daf4a", colorant"#984ea3"]
        for (i, c) in enumerate(palette)
            cx = 0.08 + (i - 1) * 0.17
            push!(elements, compose(context(),
                (context(), Compose.circle(cx, y, 0.012),
                 fill(c), Compose.stroke("gray60"), Compose.linewidth(0.1mm))))
        end
        y += 0.03
        push!(elements, _legend_annotation(y, "Community ID"))
        y += 0.035

    else  # :uniform
        push!(elements, compose(context(),
            (context(), Compose.circle(0.12, y, 0.012),
             fill(colorant"steelblue"), Compose.stroke("gray60"), Compose.linewidth(0.1mm)),
            (context(), Compose.text(0.22, y, "Uniform", hleft, vcenter),
             Compose.fontsize(5.5pt), fill("gray55"))))
        y += 0.04
    end

    # --- Node Size ---
    push!(elements, _legend_section_header(y, "Node size"))
    y += 0.04

    size_labels = Dict(
        :degree => "Degree",
        :posterior_prob => "Posterior",
        :log2fc => "|log2FC|",
        :uniform => "Uniform"
    )
    size_label = get(size_labels, node_size_by, string(node_size_by))

    if node_size_by != :uniform
        # Small -> medium -> large circles
        for (i, r) in enumerate([0.006, 0.010, 0.016])
            cx = 0.08 + (i - 1) * 0.15
            push!(elements, compose(context(),
                (context(), Compose.circle(cx, y, r),
                 fill("gray60"), Compose.stroke("gray50"), Compose.linewidth(0.1mm))))
        end
        push!(elements, compose(context(),
            (context(), Compose.text(0.58, y, size_label, hleft, vcenter),
             Compose.fontsize(5.5pt), fill("gray55"))))
        y += 0.04
    else
        push!(elements, compose(context(),
            (context(), Compose.circle(0.12, y, 0.012),
             fill("gray60"), Compose.stroke("gray50"), Compose.linewidth(0.1mm)),
            (context(), Compose.text(0.22, y, "Uniform", hleft, vcenter),
             Compose.fontsize(5.5pt), fill("gray55"))))
        y += 0.04
    end

    # --- Edges ---
    push!(elements, _legend_section_header(y, "Edges"))
    y += 0.04

    if is_enriched && edge_style_by == :by_source
        # Experimental edges (gray)
        push!(elements, compose(context(),
            (context(), Compose.line([(0.06, y), (0.26, y)]),
             Compose.stroke(colorant"gray60"), Compose.linewidth(0.4mm)),
            (context(), Compose.text(0.30, y, "Experimental", hleft, vcenter),
             Compose.fontsize(5.5pt), fill("gray55"))))
        y += 0.035
        # Public PPI edges (blue)
        push!(elements, compose(context(),
            (context(), Compose.line([(0.06, y), (0.26, y)]),
             Compose.stroke(colorant"#6baed6"), Compose.linewidth(0.4mm)),
            (context(), Compose.text(0.30, y, "Public PPI", hleft, vcenter),
             Compose.fontsize(5.5pt), fill("gray55"))))
        y += 0.035
    else
        push!(elements, compose(context(),
            (context(), Compose.line([(0.06, y), (0.26, y)]),
             Compose.stroke(colorant"gray"), Compose.linewidth(0.4mm)),
            (context(), Compose.text(0.30, y, "Interaction", hleft, vcenter),
             Compose.fontsize(5.5pt), fill("gray55"))))
        y += 0.035
    end

    if edge_width_by == :weight
        push!(elements, _legend_annotation(y, "Width proportional to weight"))
        y += 0.03
    end

    # --- Bait info ---
    push!(elements, _legend_separator(y))
    y += 0.025

    if bait_highlighted
        push!(elements, compose(context(),
            (context(), Compose.circle(0.12, y, 0.012),
             fill(colorant"gold"), Compose.stroke("gray50"), Compose.linewidth(0.1mm)),
            (context(), Compose.text(0.22, y, "Bait protein", hleft, vcenter),
             Compose.fontsize(5.5pt), fill("gray55"))))
        y += 0.04
    end

    if bait_hidden && !isnothing(bait_name)
        push!(elements, compose(context(),
            (context(), Compose.text(0.06, y, "Bait: $bait_name", hleft, vtop),
             Compose.fontsize(6.5pt), fill("gray40"))))
        y += 0.035
        push!(elements, compose(context(),
            (context(), Compose.text(0.06, y,
                "$n_bait_interactors interactors (hidden)", hleft, vtop),
             Compose.fontsize(5.5pt), fill("gray55"))))
        y += 0.03
    elseif !isnothing(bait_name) && !bait_highlighted
        push!(elements, compose(context(),
            (context(), Compose.text(0.06, y, "Bait: $bait_name", hleft, vtop),
             Compose.fontsize(6.5pt), fill("gray40"))))
        y += 0.04
    end

    # Assemble legend with white background and border
    # In Compose, first child renders on top, so: content on top, rectangle behind
    legend_height = min(y + 0.01, 0.98)
    compose(context(),
        compose(context(0.02, 0.02, 0.96, legend_height),
            compose(context(), elements...),
            (context(), Compose.rectangle(),
             fill("white"), Compose.stroke("gray80"), Compose.linewidth(0.3mm))
        )
    )
end

# Legend helpers

function _legend_section_header(y::Float64, label::String)
    compose(context(),
        (context(), Compose.text(0.06, y, label, hleft, vtop),
         Compose.fontsize(6.5pt), fill("gray40")))
end

function _legend_annotation(y::Float64, label::String)
    compose(context(),
        (context(), Compose.text(0.06, y, label, hleft, vtop),
         Compose.fontsize(5pt), fill("gray55")))
end

function _legend_separator(y::Float64)
    compose(context(),
        (context(), Compose.line([(0.06, y), (0.94, y)]),
         Compose.stroke("gray85"), Compose.linewidth(0.15mm)))
end
