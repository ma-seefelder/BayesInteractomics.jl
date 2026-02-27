# Sensitivity Analysis Visualization
# Tornado plot, heatmap, and rank correlation matrix for SensitivityResult

"""
    sensitivity_tornado_plot(sr::SensitivityResult; n_top=0, file="")

Horizontal bar chart showing the posterior probability range for the most sensitive proteins.
Red diamonds mark the baseline posterior for each protein.

# Keywords
- `n_top::Int`: Number of top proteins to display (default: `sr.config.n_top_proteins`)
- `file::String`: If non-empty, save the plot to this path

# Returns
- StatsPlots plot object
"""
function sensitivity_tornado_plot(sr::SensitivityResult; n_top::Int=0, file::String="")
    n_top = n_top > 0 ? n_top : sr.config.n_top_proteins
    n_top = min(n_top, length(sr.protein_names))

    sorted_summary = sort(sr.summary, :range, rev=true)
    top = first(sorted_summary, n_top)

    # Reverse so the most sensitive protein is at the top
    labels = reverse(top.Protein)
    mins = reverse(top.min_posterior)
    maxs = reverse(top.max_posterior)
    baselines = reverse(top.baseline_posterior)
    ranges = reverse(top.range)

    y = 1:n_top
    fig_height = max(300, 40 * n_top + 100)

    plt = StatsPlots.plot(
        xlims=(0, 1),
        yticks=(y, labels),
        xlabel="Posterior Probability",
        title="Sensitivity Tornado Plot (top $n_top)",
        size=(700, fig_height),
        left_margin=10 * StatsPlots.Plots.mm,
        legend=:bottomright
    )

    # Horizontal bars from min to max
    for i in eachindex(y)
        StatsPlots.plot!(plt, [mins[i], maxs[i]], [y[i], y[i]],
            linewidth=6, color=:steelblue, label=(i == 1 ? "Range" : nothing))
    end

    # Baseline markers
    StatsPlots.scatter!(plt, baselines, collect(y),
        marker=:diamond, markersize=5, color=:red, label="Baseline")

    if !isempty(file)
        mkpath(dirname(file))
        StatsPlots.savefig(plt, file)
    end

    return plt
end

"""
    sensitivity_heatmap(sr::SensitivityResult; n_top=0, file="")

Heatmap of posterior probabilities (proteins x prior settings) for the most sensitive proteins.

# Keywords
- `n_top::Int`: Number of top proteins to display (default: `sr.config.n_top_proteins`)
- `file::String`: If non-empty, save the plot to this path

# Returns
- StatsPlots plot object
"""
function sensitivity_heatmap(sr::SensitivityResult; n_top::Int=0, file::String="")
    n_top = n_top > 0 ? n_top : sr.config.n_top_proteins
    n_top = min(n_top, length(sr.protein_names))

    # Sort by range (most sensitive first) and pick top N
    sorted_idx = sortperm(vec(sr.summary.range), rev=true)
    top_idx = sorted_idx[1:n_top]

    # Reverse row order so most sensitive is at the top of the heatmap
    top_idx_reversed = reverse(top_idx)

    mat = sr.posterior_matrix[top_idx_reversed, :]
    ylabels = sr.protein_names[top_idx_reversed]
    xlabels = [s.label for s in sr.prior_settings]

    fig_height = max(400, 30 * n_top + 150)

    plt = StatsPlots.heatmap(
        mat,
        xticks=(1:length(xlabels), xlabels),
        yticks=(1:length(ylabels), ylabels),
        xlabel="Prior Setting",
        ylabel="Protein",
        title="Posterior Probability Heatmap (top $n_top)",
        color=:viridis,
        clims=(0, 1),
        xrotation=45,
        size=(max(600, 60 * length(xlabels)), fig_height),
        left_margin=10 * StatsPlots.Plots.mm,
        bottom_margin=10 * StatsPlots.Plots.mm
    )

    if !isempty(file)
        mkpath(dirname(file))
        StatsPlots.savefig(plt, file)
    end

    return plt
end

"""
    sensitivity_rank_correlation(sr::SensitivityResult; file="")

Heatmap of mean absolute posterior difference between each pair of prior settings.
Shows how much posterior probabilities diverge on average between any two settings.
Values near 0 indicate agreement; larger values indicate prior sensitivity.

# Keywords
- `file::String`: If non-empty, save the plot to this path

# Returns
- StatsPlots plot object
"""
function sensitivity_rank_correlation(sr::SensitivityResult; file::String="")
    n_settings = length(sr.prior_settings)
    labels = [s.label for s in sr.prior_settings]

    # Mean absolute difference matrix
    mad_matrix = zeros(n_settings, n_settings)
    for i in 1:n_settings, j in 1:n_settings
        mad_matrix[i, j] = mean(abs.(sr.posterior_matrix[:, i] .- sr.posterior_matrix[:, j]))
    end

    plt = StatsPlots.heatmap(
        mad_matrix,
        xticks=(1:n_settings, labels),
        yticks=(1:n_settings, labels),
        title="Mean |ΔPosterior| Between Settings",
        color=:YlOrRd,
        clims=(0, max(maximum(mad_matrix), 0.01)),
        aspect_ratio=:equal,
        xrotation=45,
        size=(max(600, 55 * n_settings + 150), max(550, 55 * n_settings + 100)),
        left_margin=10 * StatsPlots.Plots.mm,
        bottom_margin=10 * StatsPlots.Plots.mm,
        right_margin=5 * StatsPlots.Plots.mm,
        top_margin=5 * StatsPlots.Plots.mm
    )

    if !isempty(file)
        mkpath(dirname(file))
        StatsPlots.savefig(plt, file)
    end

    return plt
end
