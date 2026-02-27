# Prior Sensitivity Analysis
# Systematic sweep over Beta-Bernoulli and evidence combination priors

"""
    sensitivity_analysis(ar, data::InteractionData; kwargs...) -> SensitivityResult

Run prior sensitivity analysis by sweeping Beta-Bernoulli detection priors and
evidence combination priors (copula-EM or latent class), then summarizing how
posterior probabilities change across the grid.

HBM/regression priors are excluded because they are data-driven (fitted via `τ0()`/`μ0()`).

# Arguments
- `ar`: Completed analysis result (`AnalysisResult`), provides baseline BFs and combination method
- `data::InteractionData`: Raw data (needed for BB recomputation)

# Keywords
- `config::SensitivityConfig`: Grid specification (default: `SensitivityConfig()`)
- `n_controls::Int`: Number of control replicates
- `n_samples::Int`: Number of sample replicates
- `refID::Int`: Bait protein index
- `H0_file::String`: Path to H0 file for copula mode
- `combination_method::Symbol`: `:copula` or `:latent_class`
- `lc_n_iterations::Int`: VMP iterations for latent class
- `lc_convergence_tol::Float64`: Convergence tolerance for latent class
- `verbose::Bool`: Print progress info

# Returns
- `SensitivityResult` with posterior/BF/q matrices and summary statistics
"""
function sensitivity_analysis(
    ar,  # AnalysisResult — untyped to avoid world-age issues with include order
    data::InteractionData;
    config::SensitivityConfig = SensitivityConfig(),
    n_controls::Int = 0,
    n_samples::Int = 0,
    refID::Int = 1,
    H0_file::String = "copula_H0.xlsx",
    combination_method::Symbol = ar.combination_method,
    lc_n_iterations::Int = 100,
    lc_convergence_tol::Float64 = 1e-6,
    verbose::Bool = false
)
    # Extract baseline BFs from the analysis result (filtered protein subset)
    cr = ar.copula_results
    protein_names = Vector{String}(cr.Protein)
    n_proteins = length(protein_names)

    bf_enrichment = Vector{Float64}(cr.bf_enrichment)
    bf_correlation = Vector{Float64}(cr.bf_correlation)
    bf_detected_baseline = Vector{Float64}(cr.bf_detected)

    # Build lookup from full data protein IDs to indices for BB recomputation.
    # copula_results may have fewer proteins than data (HBM/regression filtering),
    # so we must align the recomputed BB BFs to the filtered protein set.
    all_protein_ids = getIDs(data)

    # Build the list of prior settings and compute posteriors for each
    prior_settings = PriorSetting[]
    posterior_columns = Vector{Float64}[]
    bf_columns = Vector{Float64}[]
    q_columns = Vector{Float64}[]
    baseline_index = 0

    # ------------------------------------------------------------------ #
    # 1. Beta-Bernoulli prior sweep
    # ------------------------------------------------------------------ #
    for (α, β) in config.bb_priors
        label = "BB($(α),$(β))"
        push!(prior_settings, PriorSetting(:betabernoulli, label, (α=α, β=β)))

        # Track baseline (default prior)
        if α == 3.0 && β == 3.0
            baseline_index = length(prior_settings)
        end

        # Recompute BB Bayes factors for ALL proteins, then filter to copula_results subset
        bf_d_full = _recompute_bb_bf(data, n_controls, n_samples; prior_alpha=α, prior_beta=β)
        bf_lookup = Dict(all_protein_ids[i] => bf_d_full[i] for i in eachindex(all_protein_ids))
        bf_d = [get(bf_lookup, name, 0.0) for name in protein_names]

        # Recombine evidence with the new detection BFs
        bf, posterior, q_vals = _recombine_evidence(
            bf_enrichment, bf_correlation, bf_d, refID;
            combination_method = combination_method,
            H0_file = H0_file,
            lc_n_iterations = lc_n_iterations,
            lc_convergence_tol = lc_convergence_tol,
            verbose = verbose
        )

        push!(posterior_columns, posterior)
        push!(bf_columns, bf)
        push!(q_columns, q_vals)
    end

    # ------------------------------------------------------------------ #
    # 2. Copula-EM prior sweep (only for copula mode)
    # ------------------------------------------------------------------ #
    if combination_method == :copula
        for em_prior in config.em_prior_grid
            expected = round(em_prior.α / (em_prior.α + em_prior.β), digits=3)
            label = "EM(α=$(em_prior.α),β=$(em_prior.β),E[π₁]=$(expected))"
            push!(prior_settings, PriorSetting(:copula_em, label, (α=em_prior.α, β=em_prior.β)))

            bf, posterior, q_vals = _recombine_evidence(
                bf_enrichment, bf_correlation, bf_detected_baseline, refID;
                combination_method = :copula,
                H0_file = H0_file,
                em_prior = em_prior,
                verbose = verbose
            )

            push!(posterior_columns, posterior)
            push!(bf_columns, bf)
            push!(q_columns, q_vals)
        end
    end

    # ------------------------------------------------------------------ #
    # 3. Latent class prior sweep (only for latent_class mode)
    # ------------------------------------------------------------------ #
    if combination_method == :latent_class
        for lc_prior in config.lc_alpha_prior_grid
            label = "LC(α=[$(join(lc_prior, ","))])"
            push!(prior_settings, PriorSetting(:latent_class, label, (alpha_prior=lc_prior,)))

            bf, posterior, q_vals = _recombine_evidence(
                bf_enrichment, bf_correlation, bf_detected_baseline, refID;
                combination_method = :latent_class,
                lc_alpha_prior = lc_prior,
                lc_n_iterations = lc_n_iterations,
                lc_convergence_tol = lc_convergence_tol,
                verbose = verbose
            )

            push!(posterior_columns, posterior)
            push!(bf_columns, bf)
            push!(q_columns, q_vals)
        end
    end

    # If no baseline found (BB(3,3) not in grid), use the first setting
    if baseline_index == 0
        baseline_index = 1
    end

    # ------------------------------------------------------------------ #
    # 4. Assemble matrices and compute summaries
    # ------------------------------------------------------------------ #
    n_settings = length(prior_settings)
    posterior_matrix = hcat(posterior_columns...)  # n_proteins × n_settings
    bf_matrix = hcat(bf_columns...)
    q_matrix = hcat(q_columns...)

    summary_df = _compute_sensitivity_summary(posterior_matrix, protein_names, baseline_index)
    stability_df = _compute_classification_stability(posterior_matrix, q_matrix, protein_names)

    return SensitivityResult(
        config,
        prior_settings,
        posterior_matrix,
        bf_matrix,
        q_matrix,
        protein_names,
        baseline_index,
        summary_df,
        stability_df,
        now()
    )
end


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

"""
    _recompute_bb_bf(data, n_controls, n_samples; prior_alpha, prior_beta) -> Vector{Float64}

Recompute Beta-Bernoulli Bayes factors for all proteins with custom prior parameters.
"""
function _recompute_bb_bf(
    data::InteractionData,
    n_controls::Int,
    n_samples::Int;
    prior_alpha::Float64 = 3.0,
    prior_beta::Float64 = 3.0
)
    n_proteins = length(getIDs(data))
    bf_detected = zeros(Float64, n_proteins)

    Threads.@threads for i in 1:n_proteins
        b, _, _ = betabernoulli(data, i, n_controls, n_samples;
                                prior_alpha = prior_alpha, prior_beta = prior_beta)
        bf_detected[i] = ismissing(b) ? 0.0 : b
    end

    return bf_detected
end

"""
    _recombine_evidence(bf_e, bf_c, bf_d, refID; kwargs...) -> (bf, posterior, q)

Re-run evidence combination with given BF vectors and optional prior overrides.
Returns vectors of combined BFs, posterior probabilities, and q-values.
"""
function _recombine_evidence(
    bf_enrichment::Vector{Float64},
    bf_correlation::Vector{Float64},
    bf_detected::Vector{Float64},
    refID::Int;
    combination_method::Symbol = :copula,
    H0_file::String = "copula_H0.xlsx",
    em_prior::Union{Nothing, NamedTuple} = nothing,
    lc_alpha_prior::Vector{Float64} = [10.0, 1.0],
    lc_n_iterations::Int = 100,
    lc_convergence_tol::Float64 = 1e-6,
    verbose::Bool = false
)
    triplet = BayesFactorTriplet(bf_enrichment, bf_correlation, bf_detected)

    if combination_method == :copula
        prior_arg = isnothing(em_prior) ? :default : em_prior
        result = combined_BF(
            triplet, refID;
            H0_file = H0_file,
            prior = prior_arg,
            n_restarts = 5,  # reduced restarts for sensitivity sweep
            verbose = verbose
        )
        bf = result.bf
        posterior = result.posterior_prob
    elseif combination_method == :latent_class
        result = combined_BF_latent_class(
            triplet, refID;
            alpha_prior = lc_alpha_prior,
            n_iterations = lc_n_iterations,
            convergence_tol = lc_convergence_tol,
            verbose = verbose
        )
        bf = result.bf
        posterior = result.posterior_prob
    elseif combination_method == :bma
        # For BMA sensitivity sweeps use copula as the base method (full BMA is too slow for a grid)
        prior_arg = isnothing(em_prior) ? :default : em_prior
        result = combined_BF(
            triplet, refID;
            H0_file = H0_file,
            prior = prior_arg,
            n_restarts = 5,
            verbose = verbose
        )
        bf = result.bf
        posterior = result.posterior_prob
    else
        error("Unknown combination_method: $combination_method")
    end

    q_vals = q(bf)
    return bf, posterior, q_vals
end

"""
    _compute_sensitivity_summary(posterior_matrix, protein_names, baseline_idx) -> DataFrame

Per-protein summary: baseline posterior, mean, std, min, max, range across all prior settings.
"""
function _compute_sensitivity_summary(
    posterior_matrix::Matrix{Float64},
    protein_names::Vector{String},
    baseline_idx::Int
)
    n_proteins = size(posterior_matrix, 1)
    baseline_col = posterior_matrix[:, baseline_idx]

    summary = DataFrame(
        Protein = protein_names,
        baseline_posterior = baseline_col,
        mean_posterior = vec(mean(posterior_matrix, dims=2)),
        std_posterior = vec(std(posterior_matrix, dims=2)),
        min_posterior = vec(minimum(posterior_matrix, dims=2)),
        max_posterior = vec(maximum(posterior_matrix, dims=2)),
        range = vec(maximum(posterior_matrix, dims=2) .- minimum(posterior_matrix, dims=2))
    )

    return summary
end

"""
    _compute_classification_stability(posterior_matrix, q_matrix, protein_names) -> DataFrame

Per-protein classification stability: fraction of settings where protein exceeds
P > 0.5, P > 0.8, P > 0.95, and q < 0.05, q < 0.01.
"""
function _compute_classification_stability(
    posterior_matrix::Matrix{Float64},
    q_matrix::Matrix{Float64},
    protein_names::Vector{String}
)
    n_settings = size(posterior_matrix, 2)

    stability = DataFrame(
        Protein = protein_names,
        frac_P_gt_0_5  = vec(sum(posterior_matrix .> 0.5, dims=2)) ./ n_settings,
        frac_P_gt_0_8  = vec(sum(posterior_matrix .> 0.8, dims=2)) ./ n_settings,
        frac_P_gt_0_95 = vec(sum(posterior_matrix .> 0.95, dims=2)) ./ n_settings,
        frac_q_lt_0_05 = vec(sum(q_matrix .< 0.05, dims=2)) ./ n_settings,
        frac_q_lt_0_01 = vec(sum(q_matrix .< 0.01, dims=2)) ./ n_settings
    )

    return stability
end


# ------------------------------------------------------------------ #
# Report generation
# ------------------------------------------------------------------ #

"""
    generate_sensitivity_report(sr::SensitivityResult; filename, title, tornado_file, heatmap_file, rankcorr_file) -> (filepath, content)

Generate a Markdown report summarizing the prior sensitivity analysis results.

# Keywords
- `filename::String`: Output file path (default: "sensitivity_report.md")
- `title::String`: Report title
- `tornado_file::String`: Path to tornado plot image. If non-empty, embedded in report.
- `heatmap_file::String`: Path to heatmap image. If non-empty, embedded in report.
- `rankcorr_file::String`: Path to rank correlation image. If non-empty, embedded in report.

# Returns
- Tuple of `(filepath, content)` where content is the Markdown string
"""
function generate_sensitivity_report(
    sr::SensitivityResult;
    filename::String = "sensitivity_report.md",
    title::String = "Prior Sensitivity Analysis Report",
    tornado_file::String = "",
    heatmap_file::String = "",
    rankcorr_file::String = ""
)
    n_proteins = length(sr.protein_names)
    n_settings = length(sr.prior_settings)
    n_top = min(sr.config.n_top_proteins, n_proteins)

    bb_settings = filter(s -> s.model == :betabernoulli, sr.prior_settings)
    em_settings = filter(s -> s.model == :copula_em, sr.prior_settings)
    lc_settings = filter(s -> s.model == :latent_class, sr.prior_settings)

    io = IOBuffer()

    # Header
    println(io, "# $title")
    println(io, "Generated: $(sr.timestamp) | Package: BayesInteractomics")
    println(io)

    # Summary
    println(io, "## Summary")
    println(io)
    println(io, "| Metric | Value |")
    println(io, "|--------|-------|")
    println(io, "| Proteins analyzed | $n_proteins |")
    println(io, "| Prior settings tested | $n_settings |")
    println(io, "| BB prior grid size | $(length(bb_settings)) |")
    if !isempty(em_settings)
        println(io, "| EM prior grid size | $(length(em_settings)) |")
    end
    if !isempty(lc_settings)
        println(io, "| LC prior grid size | $(length(lc_settings)) |")
    end
    println(io, "| Baseline setting | $(sr.prior_settings[sr.baseline_index].label) |")
    println(io)

    # Global robustness
    println(io, "## Global Robustness")
    println(io)
    mean_std = mean(sr.summary.std_posterior)
    max_std = maximum(sr.summary.std_posterior)
    mean_range = mean(sr.summary.range)
    max_range = maximum(sr.summary.range)
    n_range_gt_01 = sum(sr.summary.range .> 0.1)
    n_always_above_095 = sum(sr.classification_stability.frac_P_gt_0_95 .== 1.0)
    n_always_below_05 = sum(sr.classification_stability.frac_P_gt_0_5 .== 0.0)

    println(io, "| Metric | Value |")
    println(io, "|--------|-------|")
    println(io, "| Mean posterior std | $(round(mean_std, digits=4)) |")
    println(io, "| Max posterior std | $(round(max_std, digits=4)) |")
    println(io, "| Mean posterior range | $(round(mean_range, digits=4)) |")
    println(io, "| Max posterior range | $(round(max_range, digits=4)) |")
    println(io, "| Proteins with range > 0.1 | $n_range_gt_01 ($(round(100*n_range_gt_01/n_proteins, digits=1))%) |")
    println(io, "| Always above P > 0.95 | $n_always_above_095 ($(round(100*n_always_above_095/n_proteins, digits=1))%) |")
    println(io, "| Always below P < 0.5 | $n_always_below_05 ($(round(100*n_always_below_05/n_proteins, digits=1))%) |")
    println(io)

    # Embed tornado plot
    if !isempty(tornado_file) && isfile(tornado_file)
        rel = _relative_plot_path(filename, tornado_file)
        println(io, "![$title - Tornado Plot]($rel)")
        println(io)
    end

    # Classification stability
    println(io, "## Classification Stability")
    println(io)
    println(io, "| Threshold | Stable (100%) | Mostly (>80%) | Unstable (<50%) |")
    println(io, "|-----------|--------------|---------------|-----------------|")
    for (col, label) in [
        (:frac_P_gt_0_5, "P > 0.5"),
        (:frac_P_gt_0_8, "P > 0.8"),
        (:frac_P_gt_0_95, "P > 0.95"),
        (:frac_q_lt_0_05, "q < 0.05"),
        (:frac_q_lt_0_01, "q < 0.01")
    ]
        vals = sr.classification_stability[!, col]
        stable = sum(vals .== 1.0)
        mostly = sum(vals .> 0.8)
        unstable = sum((vals .> 0.0) .& (vals .< 0.5))
        println(io, "| $label | $stable | $mostly | $unstable |")
    end
    println(io)

    # Most sensitive proteins (top N by range)
    println(io, "## Most Sensitive Proteins (top $n_top by range)")
    println(io)
    sorted_summary = sort(sr.summary, :range, rev=true)
    top_sensitive = first(sorted_summary, n_top)
    println(io, "| Protein | Baseline P | Mean P | Std P | Min P | Max P | Range |")
    println(io, "|---------|-----------|--------|-------|-------|-------|-------|")
    for row in eachrow(top_sensitive)
        println(io, "| $(row.Protein) | $(round(row.baseline_posterior, digits=4)) | $(round(row.mean_posterior, digits=4)) | $(round(row.std_posterior, digits=4)) | $(round(row.min_posterior, digits=4)) | $(round(row.max_posterior, digits=4)) | $(round(row.range, digits=4)) |")
    end
    println(io)

    # Embed heatmap
    if !isempty(heatmap_file) && isfile(heatmap_file)
        rel = _relative_plot_path(filename, heatmap_file)
        println(io, "![$title - Posterior Heatmap]($rel)")
        println(io)
    end

    # Most robust high-confidence proteins
    high_conf = filter(r -> r.baseline_posterior > 0.8, sr.summary)
    if nrow(high_conf) > 0
        println(io, "## Most Robust High-Confidence Proteins (baseline P > 0.8, smallest range)")
        println(io)
        sorted_robust = sort(high_conf, :range)
        top_robust = first(sorted_robust, min(n_top, nrow(sorted_robust)))
        println(io, "| Protein | Baseline P | Std P | Range |")
        println(io, "|---------|-----------|-------|-------|")
        for row in eachrow(top_robust)
            println(io, "| $(row.Protein) | $(round(row.baseline_posterior, digits=4)) | $(round(row.std_posterior, digits=4)) | $(round(row.range, digits=4)) |")
        end
        println(io)
    end

    # Embed posterior divergence matrix
    if !isempty(rankcorr_file) && isfile(rankcorr_file)
        println(io, "## Posterior Divergence Across Prior Settings")
        println(io)
        rel = _relative_plot_path(filename, rankcorr_file)
        println(io, "![$title - Mean Absolute Posterior Difference]($rel)")
        println(io)
    end

    # Prior settings used
    println(io, "## Prior Settings Used")
    println(io)

    if !isempty(bb_settings)
        println(io, "### Beta-Bernoulli Priors")
        println(io)
        println(io, "| Label | α | β | E[θ] |")
        println(io, "|-------|---|---|------|")
        for s in bb_settings
            expected = round(s.params.α / (s.params.α + s.params.β), digits=3)
            println(io, "| $(s.label) | $(s.params.α) | $(s.params.β) | $expected |")
        end
        println(io)
    end

    if !isempty(em_settings)
        println(io, "### Copula-EM Priors")
        println(io)
        println(io, "| Label | α | β | E[π₁] |")
        println(io, "|-------|---|---|-------|")
        for s in em_settings
            expected = round(s.params.α / (s.params.α + s.params.β), digits=3)
            println(io, "| $(s.label) | $(s.params.α) | $(s.params.β) | $expected |")
        end
        println(io)
    end

    if !isempty(lc_settings)
        println(io, "### Latent Class Priors")
        println(io)
        println(io, "| Label | α_prior |")
        println(io, "|-------|---------|")
        for s in lc_settings
            println(io, "| $(s.label) | $(s.params.alpha_prior) |")
        end
        println(io)
    end

    content = String(take!(io))

    # Write to file
    open(filename, "w") do f
        write(f, content)
    end

    return (filename, content)
end

"""
    _relative_plot_path(report_file, plot_file) -> String

Compute the relative path from the report file's directory to the plot file,
normalizing backslashes to forward slashes for Markdown compatibility.
"""
function _relative_plot_path(report_file::String, plot_file::String)
    report_dir = dirname(abspath(report_file))
    plot_abs = abspath(plot_file)
    rel = relpath(plot_abs, report_dir)
    return replace(rel, "\\" => "/")
end
