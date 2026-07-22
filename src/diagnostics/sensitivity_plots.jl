# Sensitivity Analysis Visualization
# Rank correlation matrix for SensitivityResult

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
