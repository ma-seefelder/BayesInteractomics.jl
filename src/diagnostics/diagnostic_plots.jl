# Diagnostic Visualization
# Plots for posterior predictive checks, residuals, and calibration

"""
    compute_em_responsibilities(p_data, joint_H0, joint_H1, π0, π1) -> Vector{Float64}

Recompute per-protein EM responsibilities (H1 weights) from stored model parameters.
Returns w_i = P(H1 | x_i) for each protein, i.e. the posterior probability of belonging
to the H1 component given the fitted mixture model.
"""
function compute_em_responsibilities(p_data, joint_H0::SklarDist, joint_H1::SklarDist,
                                     π0::Float64, π1::Float64)
    p_mat = hcat(p_data.enrichment, p_data.correlation, p_data.detection)'
    lo, hi = -300.0, 300.0
    f0 = _safe_logpdf_vec(joint_H0, p_mat, lo, hi)
    f1 = _safe_logpdf_vec(joint_H1, p_mat, lo, hi)
    log_π0 = log(max(π0, 1e-300))
    log_π1 = log(max(π1, 1e-300))
    log_denom = logsumexp.(log_π0 .+ f0, log_π1 .+ f1)
    log_w = (log_π1 .+ f1) .- log_denom
    w = exp.(clamp.(log_w, -20.0, 0.0))
    return clamp.(w, 0.0, 1.0)
end

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

"""
    bma_weights_plot(bma_result::BMAResult; file="")

Two-panel diagnostic for BMA model averaging.
Left: Bar chart of LOO stacking weights (2 models: 3c-EM and Copula).
Right: Scatter of log10(copula BF) vs log10(3c-EM BF) with disagreement coloring.

# Keywords
- `file::String`: If non-empty, save plot to this path
"""
function bma_weights_plot(bma_result::BMAResult; file::String = "")
    plt = StatsPlots.plot(layout = (1, 2), size = (1100, 500),
        bottom_margin = 8 * StatsPlots.Plots.mm,
        top_margin = 5 * StatsPlots.Plots.mm)

    # Left panel: Stacking weight bar chart (always 2 models now)
    model_names = ["3c-EM", "Copula"]
    w = [bma_result.em_weight, bma_result.copula_weight]
    bar_colors = [:orange, :steelblue]

    StatsPlots.bar!(plt, model_names, w,
        color = bar_colors, alpha = 0.8,
        ylabel = "Stacking Weight", title = "Model Weights (LOO Stacking)",
        label = nothing, ylims = (0, 1.15),
        xrotation = 0,
        subplot = 1)
    for (i, wt) in enumerate(w)
        y_pos = max(wt / 2, 0.08)
        StatsPlots.annotate!(plt,
            [(i, y_pos, StatsPlots.Plots.text(
                "w=$(round(wt, digits=3))",
                9, :center, :center))],
            subplot = 1)
    end

    # Right panel: copula BF vs 3c-EM BF scatter with disagreement coloring
    log10_cop = log10.(max.(bma_result.copula_result.bf, 1e-300))
    log10_em = log10.(max.(bma_result.em3c_result.bf, 1e-300))

    n = length(log10_cop)
    colors = [bma_result.model_disagreement[i] ? :red : :gray60 for i in 1:n]

    StatsPlots.scatter!(plt, log10_cop, log10_em,
        color = colors, markersize = 2, alpha = 0.4,
        xlabel = "log10(BF) Copula", ylabel = "log10(BF) 3c-EM",
        title = "Model Agreement",
        label = nothing, legend = :bottomright,
        subplot = 2)

    bf_range = [min(minimum(log10_cop), minimum(log10_em)),
                max(maximum(log10_cop), maximum(log10_em))]
    StatsPlots.plot!(plt, bf_range, bf_range,
        color = :red, linewidth = 1.5, linestyle = :dash,
        label = "y = x", subplot = 2)

    r = cor(log10_cop, log10_em)
    n_disagree = count(bma_result.model_disagreement)
    x_ann = bf_range[2] - 0.05 * (bf_range[2] - bf_range[1])
    y_ann = bf_range[1] + 0.15 * (bf_range[2] - bf_range[1])
    StatsPlots.annotate!(plt,
        [(x_ann, y_ann,
          StatsPlots.Plots.text("r = $(round(r, digits=3))\ndisagree = $n_disagree / $n",
            9, :right, :bottom))],
        subplot = 2)

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end


# ============================================================================ #
# 3-Component Diagnostic Visualizations
# ============================================================================ #

"""
    component_assignment_plot(bf::BayesFactorTriplet, lc_result::LatentClassResult; file="")

Scatter plot of proteins in 2D log-BF space (enrichment vs correlation), colored by
dominant component assignment (H0=blue, Agnostic=gray, H1=red).
Marker alpha proportional to max responsibility, size scaled by log detection BF.
"""
function component_assignment_plot(bf::BayesFactorTriplet, lc_result::LatentClassResult;
                                    file::String = "")
    if lc_result.responsibilities === nothing
        plt = StatsPlots.plot(title="No responsibilities available for component assignment")
        !isempty(file) && StatsPlots.savefig(plt, file)
        return plt
    end

    log_e = log.(max.(bf.enrichment, 1e-300))
    log_c = log.(max.(bf.correlation, 1e-300))
    log_d = log.(max.(bf.detection, 1e-300))

    n = length(log_e)
    resp = lc_result.responsibilities

    # Determine dominant component and max responsibility for each protein
    assignments = [argmax(resp[i, :]) for i in 1:n]
    max_resp = [maximum(resp[i, :]) for i in 1:n]

    # Marker size scaled by log detection BF (clamped for visibility)
    log_d_clamped = clamp.(log_d, -5.0, 5.0)
    ms_range = log_d_clamped .- minimum(log_d_clamped)
    ms_scaled = 2.0 .+ 6.0 .* ms_range ./ max(maximum(ms_range), 1e-10)

    h0_idx = findall(assignments .== 1)
    ag_idx = findall(assignments .== 2)
    h1_idx = findall(assignments .== 3)

    plt = StatsPlots.plot(
        title="3-Component Assignment (n_H0=$(length(h0_idx)), n_ag=$(length(ag_idx)), n_H1=$(length(h1_idx)))",
        xlabel="log(BF enrichment)", ylabel="log(BF correlation)",
        size=(800, 600), legend=:topright
    )

    # Plot each component separately for legend
    if !isempty(h0_idx)
        StatsPlots.scatter!(plt, log_e[h0_idx], log_c[h0_idx],
            color=:steelblue, alpha=max_resp[h0_idx] .* 0.8,
            ms=ms_scaled[h0_idx], label="H0 (n=$(length(h0_idx)))",
            markerstrokewidth=0)
    end
    if !isempty(ag_idx)
        StatsPlots.scatter!(plt, log_e[ag_idx], log_c[ag_idx],
            color=:gray50, alpha=max_resp[ag_idx] .* 0.8,
            ms=ms_scaled[ag_idx], label="Agnostic (n=$(length(ag_idx)))",
            markerstrokewidth=0)
    end
    if !isempty(h1_idx)
        StatsPlots.scatter!(plt, log_e[h1_idx], log_c[h1_idx],
            color=:red, alpha=max_resp[h1_idx] .* 0.8,
            ms=ms_scaled[h1_idx], label="H1 (n=$(length(h1_idx)))",
            markerstrokewidth=0)
    end

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end

"""
    em_convergence_plot(lc_result::LatentClassResult; file="")

Two-panel layout:
Left: Log-likelihood trace from EM iterations.
Right: Final mixing weights bar chart [pi_H0, pi_ag, pi_H1].
"""
function em_convergence_plot(lc_result::LatentClassResult; file::String = "")
    plt = StatsPlots.plot(layout=(1, 2), size=(1000, 400),
        bottom_margin=8 * StatsPlots.Plots.mm,
        top_margin=5 * StatsPlots.Plots.mm)

    # Left panel: log-likelihood trace
    fe = lc_result.free_energy
    if isempty(fe)
        StatsPlots.annotate!(plt,
            [(0.5, 0.5, StatsPlots.Plots.text("No convergence data", 12, :center))],
            subplot=1)
        StatsPlots.plot!(plt, title="EM Convergence", subplot=1)
    else
        iters = 1:length(fe)
        StatsPlots.plot!(plt, collect(iters), fe,
            color=:steelblue, linewidth=2, label="Log-likelihood",
            xlabel="Iteration", ylabel="Log-Likelihood",
            title="EM Convergence (converged=$(lc_result.converged), n_iter=$(lc_result.n_iterations))",
            subplot=1)
        if lc_result.converged
            StatsPlots.vline!(plt, [lc_result.n_iterations],
                color=:red, linestyle=:dash, linewidth=1.5,
                label="Converged", subplot=1)
        end
    end

    # Right panel: final mixing weights
    w = lc_result.mixing_weights
    if length(w) >= 3
        labels = ["H0", "Agnostic", "H1"]
        colors = [:steelblue, :gray50, :red]
        StatsPlots.bar!(plt, labels, w[1:3],
            color=colors, alpha=0.8, label=nothing,
            ylabel="Weight", title="Final Mixing Weights",
            ylims=(0, 1.15), subplot=2)
        for (i, wt) in enumerate(w[1:3])
            y_pos = max(wt / 2, 0.05)
            StatsPlots.annotate!(plt,
                [(i, y_pos, StatsPlots.Plots.text(
                    "$(round(wt, digits=3))", 9, :center, :center))],
                subplot=2)
        end
    else
        StatsPlots.annotate!(plt,
            [(0.5, 0.5, StatsPlots.Plots.text("Weights unavailable", 12, :center))],
            subplot=2)
    end

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end
