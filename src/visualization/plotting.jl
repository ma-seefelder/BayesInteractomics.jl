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
    # Use calibrated posteriors when available
    posterior_col = hasproperty(df, :posterior_calibrated) ? df.posterior_calibrated : df.posterior_prob
    posterior_label = hasproperty(df, :posterior_calibrated) ? "Calibrated Posterior Probability" : "Posterior Probability"

    cols = [posterior_col, df.mean_log2FC, df.BFDR, df.MetaClassifier, df.DNN]

    # filter missing values
    idx = reduce(∩, findall(x -> !ismissing(x), col) for col in cols)
    posterior_prob, empiric_log2FC, bfdr_value, meta_classifier, dnn = map(
        col -> Float64.(col[idx]), cols
    )

    # filter NaN values
    vals = [posterior_prob, empiric_log2FC, bfdr_value, meta_classifier, dnn]
    idx_nan = reduce(∩, findall(x -> !isnan(x), v) for v in vals)
    posterior_prob, empiric_log2FC, bfdr_value, meta_classifier, dnn = map(
        v -> v[idx_nan], vals
    )

    plt1 = StatsPlots.density(posterior_prob;
        label = posterior_label,
        xlim = (0, 1), xlabel = posterior_label, ylabel = "density"
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
    numeric_cols  = [df.bf_enrichment, df.bf_correlation, df.bf_detected, df.mean_log2FC, df.BFDR]
    protein_names = df.Protein

    # filter missing values
    idx = reduce(∩, findall(x -> !ismissing(x), col) for col in numeric_cols)
    bf_enrichment, bf_correlation, bf_detection, mean_log2FC, bfdr_vals = map(
        col -> col[idx], numeric_cols
    )
    protein_names = protein_names[idx]

    # filter NaN values
    vals = [bf_enrichment, bf_correlation, bf_detection, mean_log2FC, bfdr_vals]
    idx_nan = reduce(∩, findall(x -> !isnan(x), v) for v in vals)
    bf_enrichment, bf_correlation, bf_detection, mean_log2FC, bfdr_vals = map(
        v -> v[idx_nan], vals
    )
    protein_names = protein_names[idx_nan]

    # clamp BFDR and compute color scale
    bfdr_vals = clamp.(bfdr_vals, eps(), 1.0)
    negative_decadiclog_bfdr = .-log10.(bfdr_vals)

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
        zcolor                = negative_decadiclog_bfdr,
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
    plt4 = StatsPlots.density(bfdr_vals;
        label = "BFDR", legend = nothing,
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
	protein_names   = df.Protein

	# Compute PEP = 1 - posterior_prob for y-axis
	pep_raw = 1.0 .- df.posterior_prob

	# remove missing values
	idx_non_missing = findall(x -> !ismissing(x), df.posterior_prob)

	mean_log2FC, pep_raw, protein_names = map(
		x -> x[idx_non_missing],
		[mean_log2FC, pep_raw, protein_names]
	)

	# Dynamic PEP floor: 10x below smallest non-zero PEP
	pep_nonzero = filter(x -> x > 0, pep_raw)
	pep_floor = isempty(pep_nonzero) ? 1e-20 : minimum(pep_nonzero) / 10
	pep_clamped = clamp.(pep_raw, pep_floor, 1.0)
	negative_decadiclog_pep = 0.0 .- log10.(pep_clamped)

	# Proteins with true PEP=0 (posterior=1.0) get distinct markers at axis top
	idx_pep_zero = findall(x -> x == 0.0, pep_raw)

	# define axis minima and maxima
	min_x = minimum(mean_log2FC)        * 1.05
	max_x = maximum(mean_log2FC)        * 1.05
	idx_nonzero_pep = findall(x -> x > 0.0, pep_raw)
	if !isempty(idx_nonzero_pep)
		min_y = minimum(negative_decadiclog_pep[idx_nonzero_pep]) * 0.95
		max_y = maximum(negative_decadiclog_pep[idx_nonzero_pep]) * 1.1
	else
		min_y = 0.0
		max_y = 10.0  # Sensible default when all PEP=0
	end

	# PEP-based significance bands
	# <=0.001 blue, <=0.01 dark blue, <=0.05 amber, >0.05 grey
	idx_pep_001  = findall(x -> x <= 0.001, pep_raw)
	idx_pep_01   = findall(x -> 0.001 < x <= 0.01, pep_raw)
	idx_pep_05   = findall(x -> 0.01 < x <= 0.05, pep_raw)
	idx_pep_ns   = findall(x -> x > 0.05, pep_raw)

	# Remove PEP=0 proteins from band sets (they get their own series)
	idx_pep_001 = setdiff(idx_pep_001, idx_pep_zero)

	# generate plot - start with non-significant (grey, background)
	plt = StatsPlots.plot(
        mean_log2FC[idx_pep_ns], negative_decadiclog_pep[idx_pep_ns],
		label = "PEP > 0.05", markersize = 2.0,
		foreground_color_legend = nothing,
		background_color_legend = nothing,
        xlabel = "log2(Fold Change)",
		ylabel = "-log\u2081\u2080(PEP)",
        size = (800, 600), seriestype = :scatter,
        xlims = (min_x, max_x), ylims = (min_y, max_y),
		markercolor = "#bdbdbd", markerstrokewidth = 0,
		hoverfontsize = 8, hovertext = protein_names,
        legendposition = legend_pos,
		minorgrid = true
        )

	# PEP <= 0.05 band (amber)
	if !isempty(idx_pep_05)
		StatsPlots.plot!(
			mean_log2FC[idx_pep_05], negative_decadiclog_pep[idx_pep_05],
			seriestype = :scatter, markerstrokewidth = 0,
			markersize = 2.0, markercolor = "#f9a825",
			label = "PEP <= 0.05"
		)
	end

	# PEP <= 0.01 band (dark blue)
	if !isempty(idx_pep_01)
		StatsPlots.plot!(
			mean_log2FC[idx_pep_01], negative_decadiclog_pep[idx_pep_01],
			seriestype = :scatter, markerstrokewidth = 0,
			markersize = 2.5, markercolor = "#0d47a1",
			label = "PEP <= 0.01"
		)
	end

	# PEP <= 0.001 band (blue)
	if !isempty(idx_pep_001)
		StatsPlots.plot!(
			mean_log2FC[idx_pep_001], negative_decadiclog_pep[idx_pep_001],
			seriestype = :scatter, markerstrokewidth = 0,
			markersize = 2.5, markercolor = "#1565c0",
			label = "PEP <= 0.001"
		)
	end

	# PEP=0 proteins rendered with distinct triangle markers at top of y-axis
	if !isempty(idx_pep_zero)
		StatsPlots.plot!(
			mean_log2FC[idx_pep_zero], fill(max_y, length(idx_pep_zero)),
			seriestype = :scatter, markershape = :utriangle,
			markersize = 4.0, markercolor = "red",
			markerstrokewidth = 0.5, markerstrokecolor = "darkred",
			label = "PEP = 0"
		)
	end

	# Threshold lines for PEP = 0.05, 0.01, 0.001
	StatsPlots.hline!([-log10(0.05)], label = nothing, color = "#f9a825", linestyle = :dash, linewidth = 0.8)
	StatsPlots.hline!([-log10(0.01)], label = nothing, color = "#0d47a1", linestyle = :dash, linewidth = 0.8)
	StatsPlots.hline!([-log10(0.001)], label = nothing, color = "#1565c0", linestyle = :dash, linewidth = 0.8)
	StatsPlots.vline!([1.0], label = nothing, color = "black", linestyle = :dash)

    return plt
end