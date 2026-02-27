# Diagnostic Visualization
# Plots for posterior predictive checks, residuals, and calibration

"""
    ppc_density_plot(ppc::ProteinPPC; file="")

Overlay observed data density with simulated posterior predictive densities.

# Keywords
- `file::String`: If non-empty, save plot to this path
"""
function ppc_density_plot(ppc::ProteinPPC; file::String = "")
    observed = ppc.observed
    n_draws = size(ppc.simulated, 2)

    # Plot a subset of simulated draws as light density lines
    n_show = min(50, n_draws)
    plt = StatsPlots.plot(
        title = "PPC: $(ppc.protein_name) ($(ppc.model))",
        xlabel = "Value",
        ylabel = "Density",
        legend = :topright
    )

    for d in 1:n_show
        sim_vals = ppc.simulated[:, d]
        if length(sim_vals) > 2
            StatsPlots.density!(plt, sim_vals, color=:lightblue, alpha=0.1, label=(d == 1 ? "Simulated" : nothing))
        end
    end

    # Overlay observed data
    if length(observed) > 2
        StatsPlots.density!(plt, observed, color=:red, linewidth=2, label="Observed")
    end

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end

"""
    ppc_pvalue_histogram(dr::DiagnosticsResult; model=:all, file="")

Plot histogram of Bayesian PPC p-values. Well-calibrated models produce a uniform distribution.

# Keywords
- `model::Symbol`: Filter by model (`:all`, `:hbm`, `:regression`, `:betabernoulli`)
- `file::String`: If non-empty, save plot to this path
"""
function ppc_pvalue_histogram(dr::DiagnosticsResult; model::Symbol = :all, file::String = "")
    pvals = Float64[]

    if model in (:all, :hbm, :regression)
        for ppc in dr.protein_ppcs
            (model != :all && ppc.model != model) && continue
            push!(pvals, ppc.pvalue_mean)
        end
    end

    if model in (:all, :betabernoulli)
        for bb in dr.bb_ppcs
            push!(pvals, bb.pvalue_detection_diff)
        end
    end

    isempty(pvals) && return StatsPlots.plot(title="No PPC p-values available")

    model_label = model == :all ? "All Models" : string(model)
    plt = StatsPlots.histogram(
        pvals,
        bins = 20,
        normalize = :probability,
        xlabel = "Bayesian p-value",
        ylabel = "Proportion",
        title = "PPC P-value Distribution ($model_label)",
        label = "p-values (n=$(length(pvals)))",
        color = :steelblue,
        alpha = 0.7,
        legend = :topright
    )

    # Reference line for uniform distribution
    StatsPlots.hline!(plt, [1.0 / 20], color=:red, linestyle=:dash, label="Uniform reference", linewidth=2)

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end

"""
    residual_qq_plot(res::ResidualResult; file="")

Q-Q plot of standardized residuals against the standard Normal distribution.

# Keywords
- `file::String`: If non-empty, save plot to this path
"""
function residual_qq_plot(res::ResidualResult; file::String = "")
    pooled = sort(res.pooled_residuals)
    n = length(pooled)

    isempty(pooled) && return StatsPlots.plot(title="No residuals available")

    # Theoretical quantiles from standard Normal
    theoretical = [quantile(Normal(0.0, 1.0), (i - 0.5) / n) for i in 1:n]

    plt = StatsPlots.scatter(
        theoretical, pooled,
        xlabel = "Theoretical Quantiles (Normal)",
        ylabel = "Standardized Residuals",
        title = "Q-Q Plot: $(res.model) Residuals",
        label = "Residuals (n=$n)",
        color = :steelblue,
        markersize = 2,
        alpha = 0.5,
        legend = :topleft
    )

    # Reference line
    range_min = min(minimum(theoretical), minimum(pooled))
    range_max = max(maximum(theoretical), maximum(pooled))
    StatsPlots.plot!(plt, [range_min, range_max], [range_min, range_max],
        color=:red, linewidth=2, linestyle=:dash, label="Identity line")

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end

"""
    scale_location_plot(res::ResidualResult; file="", n_bins=30)

Scale-location plot: √|standardized residuals| vs fitted values.

A flat smoother line indicates homoscedasticity; an increasing trend reveals
variance that grows with the predicted value.

# Keywords
- `file::String`: If non-empty, save plot to this path
- `n_bins::Int`: Number of equal-width bins for the binned smoother (default: 30)
"""
function scale_location_plot(res::ResidualResult; file::String = "", n_bins::Int = 30)
    fitted = res.pooled_fitted
    resids = res.pooled_residuals
    n = length(resids)

    (isempty(resids) || isempty(fitted)) && return StatsPlots.plot(title="No residuals available")

    sqrt_abs_resid = sqrt.(abs.(resids))

    plt = StatsPlots.scatter(
        fitted, sqrt_abs_resid,
        xlabel = "Fitted Values",
        ylabel = "√|Standardized Residuals|",
        title = "Scale-Location: $(res.model) Residuals (n=$n)",
        label = nothing,
        color = :steelblue,
        markersize = 2,
        alpha = 0.3,
        legend = :topright
    )

    # Binned smoother: equal-width bins over fitted range, mean √|resid| per bin
    lo, hi = extrema(fitted)
    if hi > lo
        bin_edges = range(lo, hi, length = n_bins + 1)
        bin_mids = Float64[]
        bin_means = Float64[]
        for i in 1:n_bins
            mask = (fitted .>= bin_edges[i]) .& (fitted .< bin_edges[i + 1])
            # Include right edge in last bin
            if i == n_bins
                mask = mask .| (fitted .== bin_edges[i + 1])
            end
            vals = sqrt_abs_resid[mask]
            if !isempty(vals)
                push!(bin_mids, (bin_edges[i] + bin_edges[i + 1]) / 2)
                push!(bin_means, mean(vals))
            end
        end
        if length(bin_mids) >= 2
            StatsPlots.plot!(plt, bin_mids, bin_means,
                color=:red, linewidth=2, label="Binned mean")
        end
    end

    # Reference line at √(2/π) ≈ 0.798 (expected value of √|Z| for Z~N(0,1))
    StatsPlots.hline!(plt, [sqrt(2 / π)],
        color=:gray, linestyle=:dash, linewidth=1, label="√(2/π) reference")

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end

"""
    residual_distribution_plot(res::ResidualResult; file="")

Histogram of standardized residuals overlaid with a standard Normal density.

# Keywords
- `file::String`: If non-empty, save plot to this path
"""
function residual_distribution_plot(res::ResidualResult; file::String = "")
    pooled = res.pooled_residuals
    isempty(pooled) && return StatsPlots.plot(title="No residuals available")

    plt = StatsPlots.histogram(
        pooled,
        bins = 50,
        normalize = :pdf,
        xlabel = "Standardized Residual",
        ylabel = "Density",
        title = "Residual Distribution: $(res.model) (skew=$(round(res.skewness, digits=2)), kurt=$(round(res.kurtosis, digits=2)))",
        label = "Residuals (n=$(length(pooled)))",
        color = :steelblue,
        alpha = 0.7,
        legend = :topright
    )

    # Overlay standard Normal
    x_range = range(minimum(pooled) - 0.5, maximum(pooled) + 0.5, length=200)
    y_normal = [pdf(Normal(0.0, 1.0), x) for x in x_range]
    StatsPlots.plot!(plt, x_range, y_normal, color=:red, linewidth=2, label="N(0,1)")

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end

"""
    calibration_plot(cal::CalibrationResult; file="")

Predicted vs observed calibration plot with diagonal reference line.

# Keywords
- `file::String`: If non-empty, save plot to this path
"""
function calibration_plot(cal::CalibrationResult; file::String = "")
    # Filter bins with data
    mask = cal.bin_counts .> 0
    pred = cal.predicted_rate[mask]
    obs = cal.observed_rate[mask]
    counts = cal.bin_counts[mask]

    plt = StatsPlots.plot(
        xlims = (0, 1),
        ylims = (0, 1),
        xlabel = "Predicted Probability",
        ylabel = "Observed Rate",
        title = "Calibration Plot (ECE=$(round(cal.ece, digits=4)), MCE=$(round(cal.mce, digits=4)))",
        aspect_ratio = :equal,
        legend = :topleft
    )

    # Diagonal reference (perfect calibration)
    StatsPlots.plot!(plt, [0, 1], [0, 1],
        color=:gray, linewidth=1, linestyle=:dash, label="Perfect calibration")

    # Calibration curve
    if !isempty(pred)
        # Scale marker sizes by bin counts
        max_count = maximum(counts)
        marker_sizes = 3.0 .+ 8.0 .* counts ./ max(max_count, 1)

        StatsPlots.scatter!(plt, pred, obs,
            markersize=marker_sizes, color=:steelblue, label="Bins (size ~ count)")
        StatsPlots.plot!(plt, pred, obs,
            color=:steelblue, linewidth=1.5, label=nothing)
    end

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end

"""
    calibration_comparison_plot(dr::DiagnosticsResult; file="")

Overlay calibration curves from all available proxies (strict, relaxed, enrichment-only)
on a single plot for direct comparison.

# Keywords
- `file::String`: If non-empty, save plot to this path
"""
function calibration_comparison_plot(dr; file::String = "")
    plt = StatsPlots.plot(
        xlims = (0, 1),
        ylims = (0, 1),
        xlabel = "Predicted Probability",
        ylabel = "Observed Rate",
        title = "Calibration Comparison",
        aspect_ratio = :equal,
        legend = :topleft
    )

    # Diagonal reference (perfect calibration)
    StatsPlots.plot!(plt, [0, 1], [0, 1],
        color=:gray, linewidth=1, linestyle=:dash, label="Perfect calibration")

    # Plot each proxy
    proxies = [
        ("Strict 3/3", dr.calibration, :steelblue, :circle),
        ("Relaxed 2/3", dr.calibration_relaxed, :orange, :diamond),
        ("Enrichment", dr.calibration_enrichment_only, :green, :utriangle)
    ]

    for (label, cal, color, shape) in proxies
        isnothing(cal) && continue
        mask = cal.bin_counts .> 0
        pred = cal.predicted_rate[mask]
        obs = cal.observed_rate[mask]

        if !isempty(pred)
            ece_str = string(round(cal.ece, digits=3))
            StatsPlots.scatter!(plt, pred, obs,
                markersize=4, color=color, markershape=shape,
                label="$label (ECE=$ece_str)")
            StatsPlots.plot!(plt, pred, obs,
                color=color, linewidth=1.5, alpha=0.7, label=nothing)
        end
    end

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end

"""
    pit_histogram_plot(pit_values::Vector{Float64}; n_bins=10, file="")

Histogram of PIT (Probability Integral Transform) values with Uniform(0,1) reference line.

For a well-specified model, PIT values should be uniformly distributed. Deviations indicate:
- **U-shaped**: underdispersion (model variance too small)
- **Inverse-U (hump)**: overdispersion (model variance too large)
- **Skewed**: location bias (systematic over/under-prediction)

# Keywords
- `n_bins::Int`: Number of histogram bins (default: 10)
- `title::String`: Plot title
- `file::String`: If non-empty, save plot to this path
"""
function pit_histogram_plot(pit_values::Vector{Float64}; n_bins::Int = 10, title::String = "PIT Histogram", file::String = "")
    valid = filter(x -> !isnan(x) && isfinite(x), pit_values)
    isempty(valid) && return StatsPlots.plot(title="No PIT values available")

    plt = StatsPlots.histogram(
        valid,
        bins = n_bins,
        normalize = :probability,
        xlabel = "PIT Value",
        ylabel = "Proportion",
        title = title,
        label = "PIT values (n=$(length(valid)))",
        color = :steelblue,
        alpha = 0.7,
        legend = :topright,
        xlims = (0, 1)
    )

    # Uniform reference line
    StatsPlots.hline!(plt, [1.0 / n_bins], color=:red, linestyle=:dash, label="Uniform reference", linewidth=2)

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end

"""
    nu_optimization_plot(res::NuOptimizationResult; file="")

Plot the WAIC vs ν curve from Student-t ν optimization, showing the optimal ν
and the Normal model baseline.

# Keywords
- `file::String`: If non-empty, save plot to this path
"""
function nu_optimization_plot(res::NuOptimizationResult; file::String = "")
    # Sort trace by ν for a clean curve
    perm = sortperm(res.nu_trace)
    nu_sorted = res.nu_trace[perm]
    waic_sorted = res.waic_trace[perm]

    plt = StatsPlots.plot(
        title = "Student-t ν Optimization (Brent's Method)",
        xlabel = "Degrees of Freedom (ν)",
        ylabel = "WAIC",
        legend = :topright,
        yscale = :log10,
    )

    # Line connecting sorted points
    StatsPlots.plot!(plt, nu_sorted, waic_sorted,
        color=:steelblue, linewidth=1.5, label=nothing)

    # Scatter points for each evaluated ν
    StatsPlots.scatter!(plt, nu_sorted, waic_sorted,
        color=:steelblue, markersize=4, label="Evaluated ν (n=$(length(nu_sorted)))")

    # Vertical dashed line at optimal ν
    StatsPlots.vline!(plt, [res.optimal_nu],
        color=:red, linestyle=:dash, linewidth=2,
        label="Optimal ν = $(round(res.optimal_nu, digits=2))")

    # Horizontal dashed line at Normal model WAIC
    StatsPlots.hline!(plt, [res.normal_waic.waic],
        color=:gray, linestyle=:dash, linewidth=1.5,
        label="Normal model (WAIC = $(round(res.normal_waic.waic, digits=1)))")

    # Annotation with ΔWAIC
    delta_sign = res.delta_waic > 0 ? "+" : ""
    ann_text = "ΔWAIC = $(delta_sign)$(round(res.delta_waic, digits=1)) ± $(round(res.delta_se, digits=1))"
    StatsPlots.annotate!(plt, [(res.optimal_nu, res.optimal_waic.waic,
        StatsPlots.Plots.text(ann_text, 8, :left, :bottom, :red))])

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end

"""
    bb_ppc_summary_plot(bb_ppcs::Vector{BetaBernoulliPPC}; n_top=20, file="")

Bar chart showing observed detection differences alongside simulated median and range
for the top proteins by detection difference.

# Keywords
- `n_top::Int`: Number of proteins to display
- `file::String`: If non-empty, save plot to this path
"""
function bb_ppc_summary_plot(bb_ppcs::Vector{BetaBernoulliPPC}; n_top::Int = 20, file::String = "")
    isempty(bb_ppcs) && return StatsPlots.plot(title="No Beta-Bernoulli PPC results")

    # Sort by observed detection difference (descending)
    obs_diffs = [bb.observed_k_sample - bb.observed_k_control for bb in bb_ppcs]
    order = sortperm(obs_diffs, rev=true)
    n_show = min(n_top, length(bb_ppcs))
    top_indices = order[1:n_show]

    labels = [bb_ppcs[i].protein_name for i in top_indices]
    obs_vals = [obs_diffs[i] for i in top_indices]
    sim_medians = [median(bb_ppcs[i].simulated_k_sample .- bb_ppcs[i].simulated_k_control) for i in top_indices]

    # Reverse for horizontal bar chart (top at top)
    labels = reverse(labels)
    obs_vals = reverse(obs_vals)
    sim_medians = reverse(sim_medians)

    y = 1:n_show
    fig_height = max(300, 35 * n_show + 100)

    plt = StatsPlots.plot(
        yticks = (y, labels),
        xlabel = "Detection Difference (sample - control)",
        title = "Beta-Bernoulli PPC: Detection Differences",
        size = (700, fig_height),
        left_margin = 10 * StatsPlots.Plots.mm,
        legend = :bottomright
    )

    StatsPlots.scatter!(plt, obs_vals, y,
        color=:red, markersize=5, label="Observed", markershape=:diamond)
    StatsPlots.scatter!(plt, sim_medians, y,
        color=:steelblue, markersize=4, label="Simulated (median)")

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end
