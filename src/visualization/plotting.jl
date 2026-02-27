"""
    plot_analysis(copula_df, file = "results_copula.png")

Create and save a plot summarizing the results of the copula analysis.

The plot consists of four subplots:
1.  Scatter plot of log10(BF) vs log10(BF enrichment)
2.  Scatter plot of log10(BF) vs log10(BF correlation)
3.  Scatter plot of log10(BF) vs log10(BF detected)
4.  Histograms of log10(BF), log10(enrichment), log10(correlation), and log10(Detection)

# Arguments
- `copula_df`: A DataFrame containing the results of the copula analysis.
- `file`: The name of the file to save the plot to. Defaults to "results_copula.png".

# Returns
- A plot object.
"""
function plot_analysis(copula_df, file = "results_copula.png")
    log10_BF          = log10.(copula_df.BF)
    log10_enrichment  = log10.(copula_df.bf_enrichment)
    log10_correlation = log10.(copula_df.bf_correlation)
    log10_detected    = log10.(copula_df.bf_detected)

    plt1 = StatsPlots.plot(
        log10_BF, log10_enrichment;
        seriestype = :scatter, markersize = 2, label = nothing,
        xlabel = "log10(BF)", ylabel = "log10(BF enrichment)"
    )
    plt2 = StatsPlots.plot(
        log10_BF, log10_correlation;
        seriestype = :scatter, markersize = 2, label = nothing,
        xlabel = "log10(BF)", ylabel = "log10(BF correlation)"
    )
    plt3 = StatsPlots.plot(
        log10_BF, log10_detected;
        seriestype = :scatter, markersize = 2, label = nothing,
        xlabel = "log10(BF)", ylabel = "log10(BF detected)"
    )
    plt4 = StatsPlots.plot(
        StatsPlots.histogram(log10_BF;          xlabel = "log10(BF)",            normalize = :pdf, label = nothing),
        StatsPlots.histogram(log10_enrichment;  xlabel = "log10(BF enrichment)", normalize = :pdf, label = nothing),
        StatsPlots.histogram(log10_correlation; xlabel = "log10(BF correlation)", normalize = :pdf, label = nothing),
        StatsPlots.histogram(log10_detected;    xlabel = "log10(BF detected)",   normalize = :pdf, label = nothing),
    )

    plt = StatsPlots.plot(plt1, plt2, plt3, plt4; size = (1000, 1000))
    StatsPlots.savefig(plt, file)
    return plt
end

"""
    plot_results(df::DataFrame)

Create a plot summarizing the results of the analysis.

The plot consists of five subplots:
1.  Density plot of the posterior probability.
2.  Density plot of the meta-classifier.
3.  Density plot of the DNN.
4.  Scatter plot of the meta-classifier vs the posterior probability.
5.  Scatter plot of the DNN vs the posterior probability.

# Arguments
- `df`: A DataFrame containing the results of the analysis.

# Returns
- A plot object.
"""
function plot_results(df::DataFrame)
    cols = [df.posterior_prob, df.mean_log2FC, df.q, df.MetaClassifier, df.DNN]

    # filter missing values
    idx = reduce(∩, findall(x -> !ismissing(x), col) for col in cols)
    posterior_prob, empiric_log2FC, q_value, meta_classifier, dnn = map(
        col -> Float64.(col[idx]), cols
    )

    # filter NaN values
    vals = [posterior_prob, empiric_log2FC, q_value, meta_classifier, dnn]
    idx_nan = reduce(∩, findall(x -> !isnan(x), v) for v in vals)
    posterior_prob, empiric_log2FC, q_value, meta_classifier, dnn = map(
        v -> v[idx_nan], vals
    )

    plt1 = StatsPlots.density(posterior_prob;
        label = "Posterior probability",
        xlim = (0, 1), xlabel = "Posterior Probability", ylabel = "density"
    )
    plt2 = StatsPlots.density(meta_classifier;
        label = "Meta-Classifier",
        xlabel = "Prior Probability", ylabel = "density"
    )
    plt3 = StatsPlots.density(dnn;
        label = nothing,
        xlabel = "Prior Probability", ylabel = "density"
    )
    plt4 = StatsPlots.plot(meta_classifier, posterior_prob;
        seriestype = :scatter, markersize = 2, label = nothing,
        xlabel = "Meta-Classifier", ylabel = "Posterior probability"
    )
    plt5 = StatsPlots.plot(dnn, posterior_prob;
        seriestype = :scatter, markersize = 2, label = nothing,
        xlabel = "DNN", ylabel = "Posterior probability"
    )

    StatsPlots.plot(plt1, plt2, plt3, plt4, plt5; size = (600, 600))
end

"""
    evidence_plot(df)

Create a plot summarizing the evidence contribution.

The plot consists of four subplots:
1.  Scatter plot of log10(BF-enrichment) vs log10(BF-correlation). The color indicates the Bayesian false discovery rate.
2.  Scatter plot of log10(BF-enrichment) vs log10(BF-detection). The color indicates the Bayesian false discovery rate.
3.  Scatter plot of log10(BF-correlation) vs log10(BF-detection). The color indicates the Bayesian false discovery rate.
4.  Density plot of the Bayesian false discovery rate.

# Arguments
- `df`: A DataFrame containing the results of the analysis.

# Returns
- A plot object.
"""
function evidence_plot(df)
    numeric_cols  = [df.bf_enrichment, df.bf_correlation, df.bf_detected, df.mean_log2FC, df.q]
    protein_names = df.Protein

    # filter missing values
    idx = reduce(∩, findall(x -> !ismissing(x), col) for col in numeric_cols)
    bf_enrichment, bf_correlation, bf_detection, mean_log2FC, q = map(
        col -> col[idx], numeric_cols
    )
    protein_names = protein_names[idx]

    # filter NaN values
    vals = [bf_enrichment, bf_correlation, bf_detection, mean_log2FC, q]
    idx_nan = reduce(∩, findall(x -> !isnan(x), v) for v in vals)
    bf_enrichment, bf_correlation, bf_detection, mean_log2FC, q = map(
        v -> v[idx_nan], vals
    )
    protein_names = protein_names[idx_nan]

    # clamp q and compute color scale
    q = clamp.(q, eps(), 1.0)
    negative_decadiclog_q = .-log10.(q)

    # log-transform BF values
    log10_enrichment  = log10.(bf_enrichment)
    log10_correlation = log10.(bf_correlation)
    log10_detection   = log10.(bf_detection)

    # shared scatter plot kwargs
    scatter_kwargs = (
        seriestype            = :scatter,
        markerstrokewidth     = 0,
        ms                    = 2.0,
        size                  = (600, 600),
        hoverfontsize         = 8,
        hovertext             = protein_names,
        legendposition        = :topleft,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        zcolor                = negative_decadiclog_q,
        m                     = (:dense),
        label                 = nothing,
        colorbar_title        = "-log10(BFDR)",
        minorgrid             = true,
    )

    plt1 = StatsPlots.plot(log10_enrichment, log10_correlation;
        xlabel = "log10(BF-enrichment)", ylabel = "log10(BF-correlation)",
        xlim   = extrema(log10_enrichment), ylim = extrema(log10_correlation),
        scatter_kwargs...
    )
    plt2 = StatsPlots.plot(log10_enrichment, log10_detection;
        xlabel = "log10(BF-enrichment)", ylabel = "log10(BF-detection)",
        xlim   = extrema(log10_enrichment), ylim = extrema(log10_detection),
        scatter_kwargs...
    )
    plt3 = StatsPlots.plot(log10_correlation, log10_detection;
        xlabel = "log10(BF-correlation)", ylabel = "log10(BF-detection)",
        xlim   = extrema(log10_correlation), ylim = extrema(log10_detection),
        scatter_kwargs...
    )
    plt4 = StatsPlots.density(q;
        label = "q", legend = nothing,
        xlabel = "BFDR", fill = (0, 0.5, :darkblue), linewidth = 0
    )

    return StatsPlots.plot(plt1, plt2, plt3, plt4)
end

"""
    rank_rank_plot(df; legend_pos = :topleft)

Create a rank-rank plot to visualize interactome analysis results.

This function generates a scatter plot where each point represents a protein. The plot visualizes four different metrics:
1.  **x-axis**: `log10(BF)` - The decadic logarithm of the Bayes Factor, indicating the overall evidence for interaction.
2.  **y-axis**: `log2FC` - The mean log2 fold change, indicating the magnitude of enrichment.
3.  **Color**: `BF Correlation` - The winsorized decadic logarithm of the Bayes Factor for correlation between replicates. Values are clamped between the 1st and 99th percentile.
4.  **Size**: `BF Detected` - The rank of the decadic logarithm of the Bayes Factor for detection across replicates.

# Arguments
- `df::DataFrame`: A DataFrame containing the results of the analysis. It must include the columns `BF`, `bf_correlation`, `bf_detected`, `mean_log2FC`, and `Protein`.
- `legend_pos`: The position of the legend. Defaults to `:topleft`.

# Returns
- A plot object representing the rank-rank plot. 
"""
function rank_rank_plot(df; legend_pos = :topleft)
    numeric_cols  = [df.BF, df.mean_log2FC, df.bf_correlation, df.bf_detected]
    protein_names = df.Protein

    # filter missing and nothing values
    idx = reduce(∩, findall(v -> !ismissing(v) && !isnothing(v), col) for col in numeric_cols)
    bf, mean_log2FC, bf_correlation, bf_detected = map(col -> col[idx], numeric_cols)
    protein_names = protein_names[idx]

    # filter NaN values
    vals = [bf, mean_log2FC, bf_correlation, bf_detected]
    idx_nan = reduce(∩, findall(x -> !isnan(x), v) for v in vals)
    bf, mean_log2FC, bf_correlation, bf_detected = map(v -> v[idx_nan], vals)
    protein_names = protein_names[idx_nan]

    # clamp zeros to eps() and log-transform
    bf             = log10.(max.(bf,             eps()))
    bf_correlation = log10.(max.(bf_correlation, eps()))
    bf_detected    = log10.(max.(bf_detected,    eps()))

    # winsorize bf_correlation to [1st, 99th] percentile
    bf_correlation_clipped = clamp.(bf_correlation,
        quantile(bf_correlation, 0.01),
        quantile(bf_correlation, 0.99)
    )

    # normalize bf_detected ranks to marker sizes in [1, 5]
    ranks        = invperm(sortperm(bf_detected))
    norm_ranks   = (ranks .- minimum(ranks)) ./ (maximum(ranks) - minimum(ranks))
    marker_sizes = 1.0 .+ norm_ranks .* 4.0

    plt = StatsPlots.plot(bf, mean_log2FC;
        seriestype            = :scatter,
        markerstrokewidth     = 0,
        xlim                  = extrema(bf),
        ylim                  = extrema(mean_log2FC),
        xlabel                = "log10(BF)",
        ylabel                = "log2FC",
        size                  = (600, 600),
        hoverfontsize         = 8,
        hovertext             = protein_names,
        legendposition        = legend_pos,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        label                 = nothing,
        zcolor                = bf_correlation_clipped,
        m                     = (:dense),
        colorbar_title        = "log10(BF Correlation)",
        markersize            = marker_sizes,
        minorgrid             = true
    )

    StatsPlots.hline!([1.0]; label = nothing, color = "black", linestyle = :dash)
    StatsPlots.vline!([2.0]; label = nothing, color = "black", linestyle = :dash)

    return plt
end

"""
    volcano_plot(df; legend_pos = :topleft)

Create a volcano plot.

# Arguments
- `df`: A DataFrame containing the results of the analysis.
- `legend_pos`: The position of the legend. Defaults to `:topleft`.

# Returns
- A plot object.
"""
function volcano_plot(df; legend_pos = :topleft)
	mean_log2FC     = df.mean_log2FC
	q               = df.q
	protein_names   = df.Protein

	# remove missing values
	idx_non_missing = findall(x -> ismissing(x) == false, df.q)

	mean_log2FC, q, protein_names = map(
		x -> x[idx_non_missing], 
		[mean_log2FC, q, protein_names]
	)

	# define axis minima and maxima
	min_x = minimum(mean_log2FC)        * 1.05
	max_x = maximum(mean_log2FC)        * 1.05
	min_y = minimum(0.0 .- log10.(q))   * 1.05
	max_y = maximum(0.0 .- log10.(q))   * 1.05

	# define significant proteins
	idx_significant_FC  = findall(x -> x >= 1.0, mean_log2FC) 
	idx_significant_q   = findall(x -> x < 0.01, q) 
	idx_significant     = idx_significant_FC ∩ idx_significant_q
	idx_non_significant = setdiff(1:length(q),idx_significant)

	negative_decadiclog_q = 0.0 .- log10.(q)

	# generate plot	
	plt = StatsPlots.plot(
        mean_log2FC[idx_significant], negative_decadiclog_q[idx_significant],
		label = "significant", markersize = 2.0, 
		foreground_color_legend = nothing, 
		background_color_legend = nothing,
        xlabel = "log2(Fold Change)",  
		ylabel = "-log10(BFDR)",
        size = (800, 600), seriestype = :scatter, 
        xlims = (min_x, max_x), ylims = (min_y, max_y),
		markercolor = "green", markerstrokewidth = 0,
		hoverfontsize = 8, hovertext = protein_names,
        legendposition = legend_pos,
		minorgrid = true
        )
	
	StatsPlots.plot!(
		mean_log2FC[idx_non_significant], negative_decadiclog_q[idx_non_significant],
		seriestype = :scatter, markerstrokewidth = 0,
		markersize = 2.0, markercolor = "grey",
		label = "non significant"
	)

	StatsPlots.hline!([2.0], label = nothing, color = "black", linestyle = :dash)
	StatsPlots.vline!([1.0], label = nothing, color = "black", linestyle = :dash)

    return plt
end