# Differential Interaction Analysis — Visualization
# Volcano plots, evidence plots, and scatter comparisons

"""
    differential_volcano_plot(diff::DifferentialResult; kwargs...)

Create a volcano plot for differential interaction analysis.

# Keywords
- `x_axis::Symbol = :log10_dbf`: X-axis metric (`:log10_dbf` or `:delta_log2fc`)
- `y_axis::Symbol = :differential_q`: Y-axis metric (`:differential_q` → -log10(q), or `:differential_posterior`)
- `legend_pos::Symbol = :topleft`: Legend position
- `x_clip::Union{Float64,Nothing} = nothing`: X-axis half-width. When `nothing` (default),
  limits are set to the 0.5th–99.5th percentile of the data for a readable range. When a
  `Float64`, limits are `(-x_clip, x_clip)`. Points outside the limits are shown with
  triangular markers at the boundary so they remain visible.

# Returns
A StatsPlots plot object. Points colored by classification:
green = GAINED, red = REDUCED, grey = UNCHANGED.

# Examples
```julia
diff = differential_analysis(result_wt, result_mut)
plt = differential_volcano_plot(diff)
StatsPlots.savefig(plt, "differential_volcano.png")

# Use delta log2FC on x-axis
plt2 = differential_volcano_plot(diff, x_axis = :delta_log2fc)

# Fixed ±4 range
plt3 = differential_volcano_plot(diff, x_clip = 4.0)
```
"""
function differential_volcano_plot(
    diff::DifferentialResult;
    x_axis::Symbol = :log10_dbf,
    y_axis::Symbol = :differential_q,
    legend_pos::Symbol = :topleft,
    x_clip::Union{Float64,Nothing} = nothing
)
    df = diff.results

    # Filter to shared proteins only (condition-specific have NaN coordinates)
    shared_idx = findall(row ->
        row.classification != CONDITION_A_SPECIFIC &&
        row.classification != CONDITION_B_SPECIFIC,
        eachrow(df)
    )
    df_s = df[shared_idx, :]

    # X-axis values
    if x_axis == :log10_dbf
        x_vals = Float64.(df_s.log10_dbf)
        x_label = "log\u2081\u2080(dBF) [$(diff.condition_A) / $(diff.condition_B)]"
    elseif x_axis == :delta_log2fc
        x_vals = Float64.(df_s.delta_log2fc)
        x_label = "\u0394 log\u2082FC [$(diff.condition_A) \u2212 $(diff.condition_B)]"
    else
        throw(ArgumentError("x_axis must be :log10_dbf or :delta_log2fc, got :$x_axis"))
    end

    # Y-axis values
    if y_axis == :differential_q
        y_vals = [ismissing(q) ? 0.0 : -log10(max(Float64(q), eps(Float64))) for q in df_s.differential_q]
        y_label = "-log\u2081\u2080(differential q)"
    elseif y_axis == :differential_posterior
        y_vals = Float64.(df_s.differential_posterior)
        y_label = "P(differential | data)"
    else
        throw(ArgumentError("y_axis must be :differential_q or :differential_posterior, got :$y_axis"))
    end

    # Filter valid (finite) entries
    valid = findall(i -> isfinite(x_vals[i]) && isfinite(y_vals[i]), eachindex(x_vals))
    x_vals = x_vals[valid]
    y_vals = y_vals[valid]
    classifications = df_s.classification[valid]

    # Compute x-axis limits
    if isnothing(x_clip)
        x_lo, x_hi = length(x_vals) >= 2 ?
            quantile(x_vals, [0.005, 0.995]) :
            (minimum(x_vals), maximum(x_vals))
    else
        x_lo, x_hi = -x_clip, x_clip
    end
    x_pad = 0.05 * max(x_hi - x_lo, 1.0)
    xlim = (x_lo - x_pad, x_hi + x_pad)

    # Y-axis limits with padding
    y_range = maximum(y_vals) - minimum(y_vals)
    y_pad = 0.05 * max(y_range, 1.0)
    ylim = (minimum(y_vals) - y_pad, maximum(y_vals) + y_pad)

    # Split points into in-range and clipped
    in_range  = findall(i -> x_lo <= x_vals[i] <= x_hi, eachindex(x_vals))
    clip_lo   = findall(i -> x_vals[i] < x_lo, eachindex(x_vals))
    clip_hi   = findall(i -> x_vals[i] > x_hi, eachindex(x_vals))

    # Winsorize x values for rendering (clipped points plotted at boundary)
    x_render = copy(x_vals)
    x_render[clip_lo] .= x_lo
    x_render[clip_hi] .= x_hi

    # Helper: indices by classification within a subset
    _cls_idx(subset, cls) = filter(i -> classifications[i] == cls, subset)

    gained_in   = _cls_idx(in_range, GAINED)
    lost_in     = _cls_idx(in_range, REDUCED)
    unchanged_in= _cls_idx(in_range, UNCHANGED)
    gained_lo   = _cls_idx(clip_lo, GAINED);  gained_hi  = _cls_idx(clip_hi, GAINED)
    lost_lo     = _cls_idx(clip_lo, REDUCED); lost_hi    = _cls_idx(clip_hi, REDUCED)
    unch_lo     = _cls_idx(clip_lo, UNCHANGED); unch_hi  = _cls_idx(clip_hi, UNCHANGED)

    # Base plot with unchanged in-range (background)
    plt = StatsPlots.scatter(
        x_render[unchanged_in], y_vals[unchanged_in],
        markerstrokewidth = 0, markersize = 2.0, markercolor = :grey70,
        label = "unchanged ($(count(==(UNCHANGED), classifications)))",
        xlabel = x_label, ylabel = y_label,
        xlims = xlim, ylims = ylim,
        size = (800, 600),
        legendposition = legend_pos,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        minorgrid = true
    )

    # Unchanged clipped (triangles)
    if !isempty(unch_lo)
        StatsPlots.scatter!(plt, x_render[unch_lo], y_vals[unch_lo],
            markershape = :dtriangle, markerstrokewidth = 0, markersize = 3.0,
            markercolor = :grey70, label = nothing)
    end
    if !isempty(unch_hi)
        StatsPlots.scatter!(plt, x_render[unch_hi], y_vals[unch_hi],
            markershape = :utriangle, markerstrokewidth = 0, markersize = 3.0,
            markercolor = :grey70, label = nothing)
    end

    # Gained
    if !isempty(gained_in)
        StatsPlots.scatter!(plt, x_render[gained_in], y_vals[gained_in],
            markerstrokewidth = 0, markersize = 3.5, markercolor = :forestgreen,
            label = "gained ($(count(==(GAINED), classifications)))")
    end
    if !isempty(gained_lo)
        StatsPlots.scatter!(plt, x_render[gained_lo], y_vals[gained_lo],
            markershape = :dtriangle, markerstrokewidth = 0, markersize = 4.0,
            markercolor = :forestgreen, label = nothing)
    end
    if !isempty(gained_hi)
        StatsPlots.scatter!(plt, x_render[gained_hi], y_vals[gained_hi],
            markershape = :utriangle, markerstrokewidth = 0, markersize = 4.0,
            markercolor = :forestgreen, label = nothing)
    end

    # Reduced
    if !isempty(lost_in)
        StatsPlots.scatter!(plt, x_render[lost_in], y_vals[lost_in],
            markerstrokewidth = 0, markersize = 3.5, markercolor = :firebrick,
            label = "reduced ($(count(==(REDUCED), classifications)))")
    end
    if !isempty(lost_lo)
        StatsPlots.scatter!(plt, x_render[lost_lo], y_vals[lost_lo],
            markershape = :dtriangle, markerstrokewidth = 0, markersize = 4.0,
            markercolor = :firebrick, label = nothing)
    end
    if !isempty(lost_hi)
        StatsPlots.scatter!(plt, x_render[lost_hi], y_vals[lost_hi],
            markershape = :utriangle, markerstrokewidth = 0, markersize = 4.0,
            markercolor = :firebrick, label = nothing)
    end

    # Reference lines
    if y_axis == :differential_q
        q_line = -log10(diff.config.q_threshold)
        StatsPlots.hline!(plt, [q_line], label = nothing, color = :black, linestyle = :dash, linewidth = 0.8)
    end

    if x_axis == :log10_dbf
        StatsPlots.vline!(plt, [diff.config.dbf_threshold, -diff.config.dbf_threshold],
            label = nothing, color = :black, linestyle = :dash, linewidth = 0.8)
    elseif x_axis == :delta_log2fc
        StatsPlots.vline!(plt, [diff.config.delta_log2fc_threshold, -diff.config.delta_log2fc_threshold],
            label = nothing, color = :black, linestyle = :dash, linewidth = 0.8)
    end

    return plt
end

"""
    differential_evidence_plot(diff::DifferentialResult)

Create a 4-panel plot showing per-evidence-type differential Bayes factors.

Panels:
1. log10(dBF_enrichment) vs log10(dBF_correlation)
2. log10(dBF_enrichment) vs log10(dBF_detected)
3. log10(dBF_correlation) vs log10(dBF_detected)
4. Density of log10(dBF_combined)

All scatter panels colored by classification (green/red/grey).
"""
function differential_evidence_plot(diff::DifferentialResult)
    df = diff.results

    shared_idx = findall(row ->
        row.classification != CONDITION_A_SPECIFIC &&
        row.classification != CONDITION_B_SPECIFIC,
        eachrow(df)
    )
    df_s = df[shared_idx, :]

    log_enr = _safe_log10.(df_s.dbf_enrichment)
    log_cor = _safe_log10.(df_s.dbf_correlation)
    log_det = _safe_log10.(df_s.dbf_detected)

    colors = map(df_s.classification) do c
        c == GAINED ? :forestgreen : c == REDUCED ? :firebrick : :grey70
    end

    plt1 = StatsPlots.scatter(log_enr, log_cor,
        markerstrokewidth = 0, ms = 2.0, markercolor = colors,
        xlabel = "log\u2081\u2080(dBF enrichment)",
        ylabel = "log\u2081\u2080(dBF correlation)",
        label = nothing, minorgrid = true)
    StatsPlots.hline!(plt1, [0.0], label = nothing, color = :black, linestyle = :dash, lw = 0.5)
    StatsPlots.vline!(plt1, [0.0], label = nothing, color = :black, linestyle = :dash, lw = 0.5)

    plt2 = StatsPlots.scatter(log_enr, log_det,
        markerstrokewidth = 0, ms = 2.0, markercolor = colors,
        xlabel = "log\u2081\u2080(dBF enrichment)",
        ylabel = "log\u2081\u2080(dBF detected)",
        label = nothing, minorgrid = true)
    StatsPlots.hline!(plt2, [0.0], label = nothing, color = :black, linestyle = :dash, lw = 0.5)
    StatsPlots.vline!(plt2, [0.0], label = nothing, color = :black, linestyle = :dash, lw = 0.5)

    plt3 = StatsPlots.scatter(log_cor, log_det,
        markerstrokewidth = 0, ms = 2.0, markercolor = colors,
        xlabel = "log\u2081\u2080(dBF correlation)",
        ylabel = "log\u2081\u2080(dBF detected)",
        label = nothing, minorgrid = true)
    StatsPlots.hline!(plt3, [0.0], label = nothing, color = :black, linestyle = :dash, lw = 0.5)
    StatsPlots.vline!(plt3, [0.0], label = nothing, color = :black, linestyle = :dash, lw = 0.5)

    # Density panel
    valid_dbf = filter(isfinite, Float64.(df_s.log10_dbf))
    plt4 = StatsPlots.density(valid_dbf,
        label = "log\u2081\u2080(dBF)",
        xlabel = "log\u2081\u2080(dBF)",
        fill = (0, 0.3, :steelblue),
        linewidth = 1.5)
    StatsPlots.vline!(plt4, [0.0], label = nothing, color = :black, linestyle = :dash, lw = 0.5)

    return StatsPlots.plot(plt1, plt2, plt3, plt4, layout = (2, 2), size = (1000, 1000))
end

"""
    differential_scatter_plot(diff::DifferentialResult; metric=:posterior_prob)

Scatter plot comparing a metric between conditions A and B.

# Keywords
- `metric::Symbol`: Which metric to compare:
  - `:posterior_prob` (default): Posterior probabilities
  - `:bf`: Combined Bayes factors (log10 scale)
  - `:log2fc`: Mean log2 fold change

Points colored by classification with identity line.
"""
function differential_scatter_plot(
    diff::DifferentialResult;
    metric::Symbol = :posterior_prob
)
    df = diff.results

    shared_idx = findall(row ->
        row.classification != CONDITION_A_SPECIFIC &&
        row.classification != CONDITION_B_SPECIFIC,
        eachrow(df)
    )
    df_s = df[shared_idx, :]

    if metric == :posterior_prob
        x_vals = Float64.(df_s.posterior_A)
        y_vals = Float64.(df_s.posterior_B)
        axis_label = "Posterior Probability"
    elseif metric == :bf
        x_vals = _safe_log10.(df_s.bf_A)
        y_vals = _safe_log10.(df_s.bf_B)
        axis_label = "log\u2081\u2080(BF)"
    elseif metric == :log2fc
        x_vals = Float64.(df_s.log2fc_A)
        y_vals = Float64.(df_s.log2fc_B)
        axis_label = "mean log\u2082FC"
    else
        throw(ArgumentError("metric must be :posterior_prob, :bf, or :log2fc, got :$metric"))
    end

    colors = map(df_s.classification) do c
        c == GAINED ? :forestgreen : c == REDUCED ? :firebrick : :grey70
    end

    lo = min(minimum(x_vals), minimum(y_vals))
    hi = max(maximum(x_vals), maximum(y_vals))
    pad = 0.05 * max(hi - lo, 1.0)

    plt = StatsPlots.scatter(x_vals, y_vals,
        markerstrokewidth = 0, markersize = 2.5, markercolor = colors,
        xlabel = "$(axis_label) [$(diff.condition_A)]",
        ylabel = "$(axis_label) [$(diff.condition_B)]",
        label = nothing,
        size = (600, 600),
        xlims = (lo - pad, hi + pad),
        ylims = (lo - pad, hi + pad),
        minorgrid = true,
        aspect_ratio = :equal
    )

    # Identity line
    StatsPlots.plot!(plt, [lo - pad, hi + pad], [lo - pad, hi + pad],
        label = nothing, color = :black, linestyle = :dash, linewidth = 0.8)

    return plt
end

"""
    differential_classification_plot(diff::DifferentialResult)

Horizontal bar chart showing the count of proteins in each interaction class.

Bars are grouped as: GAINED, REDUCED, UNCHANGED, BOTH_NEGATIVE, condition-A-specific, condition-B-specific.
Provides a quick sanity-check of the differential analysis at a glance.

# Returns
A StatsPlots plot object.
"""
function differential_classification_plot(diff::DifferentialResult)
    labels = [
        "Gained\n($(diff.condition_A))",
        "Reduced\n($(diff.condition_B))",
        "Unchanged",
        "Both\nnegative",
        "$(diff.condition_A)\nonly",
        "$(diff.condition_B)\nonly"
    ]
    counts = Float64[
        diff.n_gained,
        diff.n_reduced,
        diff.n_unchanged,
        diff.n_both_negative,
        diff.n_condition_A_specific,
        diff.n_condition_B_specific
    ]
    colors = [:forestgreen, :firebrick, :grey70, :orchid4, :steelblue, :darkorange]
    n = length(counts)

    plt = StatsPlots.bar(
        1:n, counts,
        color = colors,
        legend = false,
        ylabel = "Number of proteins",
        xticks = (1:n, labels),
        title = "Differential Interaction Summary: $(diff.condition_A) vs $(diff.condition_B)",
        size = (700, 450),
        bottom_margin = 10 * StatsPlots.Plots.mm,
        bar_width = 0.6
    )

    return plt
end

"""
    differential_ma_plot(diff::DifferentialResult)

MA-style plot for differential proteomics.

- **x-axis**: ½ (log2FC_A + log2FC_B) — mean log2 fold change (proxy for abundance)
- **y-axis**: Δlog2FC = log2FC_A − log2FC_B — differential effect size
- Horizontal reference line at y = 0
- Points colored by classification (green = GAINED, red = REDUCED, grey = UNCHANGED)

Useful for detecting systematic biases where differential enrichment correlates
with overall enrichment level.

# Returns
A StatsPlots plot object.
"""
function differential_ma_plot(diff::DifferentialResult)
    df = diff.results

    # Shared proteins only
    shared_idx = findall(row ->
        row.classification != CONDITION_A_SPECIFIC &&
        row.classification != CONDITION_B_SPECIFIC,
        eachrow(df)
    )
    df_s = df[shared_idx, :]

    lfc_A = Float64.(df_s.log2fc_A)
    lfc_B = Float64.(df_s.log2fc_B)
    x_vals = 0.5 .* (lfc_A .+ lfc_B)   # mean log2FC
    y_vals = lfc_A .- lfc_B             # delta log2FC

    colors = map(df_s.classification) do c
        c == GAINED ? :forestgreen : c == REDUCED ? :firebrick : :grey70
    end

    # Filter finite values
    valid = findall(i -> isfinite(x_vals[i]) && isfinite(y_vals[i]), eachindex(x_vals))
    x_vals = x_vals[valid]
    y_vals = y_vals[valid]
    colors  = colors[valid]

    x_range = maximum(x_vals) - minimum(x_vals)
    y_range = maximum(y_vals) - minimum(y_vals)
    x_pad = 0.05 * max(x_range, 1.0)
    y_pad = 0.05 * max(y_range, 1.0)

    plt = StatsPlots.scatter(
        x_vals, y_vals,
        markerstrokewidth = 0, markersize = 2.5, markercolor = colors,
        xlabel = "\u00BD (log\u2082FC_$(diff.condition_A) + log\u2082FC_$(diff.condition_B))",
        ylabel = "\u0394 log\u2082FC [$(diff.condition_A) \u2212 $(diff.condition_B)]",
        label = nothing,
        size = (700, 500),
        xlims = (minimum(x_vals) - x_pad, maximum(x_vals) + x_pad),
        ylims = (minimum(y_vals) - y_pad, maximum(y_vals) + y_pad),
        minorgrid = true
    )

    StatsPlots.hline!(plt, [0.0], label = nothing, color = :black, linestyle = :dash, linewidth = 0.8)

    return plt
end
