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
    h0_cache_file::String = "",
    combination_method::Symbol = ar.combination_method,
    lc_n_iterations::Int = 100,
    lc_convergence_tol::Float64 = 1e-6,
    verbose::Bool = false
)
    # Extract baseline BFs from the analysis result (filtered protein subset)
    cr = ar.copula_results
    # Filter to detected proteins only AND drop rows with missing BFs.
    # `is_detected` reflects data-level detection; `bf_*` columns are only filled
    # for proteins that survived HBM/regression inference. A protein can be
    # detected but have missing BFs (inference failure), and Vector{Float64}(...)
    # would then throw `MethodError(convert, (Float64, missing))`.
    if hasproperty(cr, :is_detected)
        cr = filter(r -> coalesce(r.is_detected, false), cr)
    end
    cr = filter(
        r -> !ismissing(r.bf_enrichment) && !ismissing(r.bf_correlation) &&
             !ismissing(r.bf_detected) && !ismissing(r.posterior_prob) && !ismissing(r.BF),
        cr,
    )
    protein_names = Vector{String}(cr.Protein)
    n_proteins = length(protein_names)

    bf_enrichment = Vector{Float64}(cr.bf_enrichment)
    bf_correlation = Vector{Float64}(cr.bf_correlation)
    bf_detected_baseline = Vector{Float64}(cr.bf_detected)

    # Build lookup from full data protein IDs to indices for BB recomputation.
    # copula_results may have fewer proteins than data (HBM/regression filtering),
    # so we must align the recomputed BB BFs to the filtered protein set.
    all_protein_ids = getIDs(data)

    # Extract the baseline H1 family from the analysis result to fix it across all
    # sensitivity sweep iterations. Without this, different prior settings can cause
    # different EM restarts to win, selecting different H1 families (Gamma/LogNormal/
    # Weibull) which dominates the apparent prior sensitivity.
    baseline_h1_family = nothing
    if combination_method in (:latent_class, :bma) && ar.latent_class_result !== nothing
        baseline_h1_family = ar.latent_class_result.h1_enrichment_family
        verbose && @info "Fixing H1 family to baseline: $baseline_h1_family"
    elseif combination_method in (:latent_class, :bma) && ar.bma_result !== nothing
        baseline_h1_family = ar.bma_result.em3c_result.h1_enrichment_family
        verbose && @info "Fixing H1 family to BMA baseline: $baseline_h1_family"
    end

    # Run the stage-1 EM once (shared across all sweep iterations for copula/bma modes)
    phase1_precomputed = nothing
    h0_precomputed = nothing
    if combination_method in (:copula, :bma)
        phase1_precomputed = combined_BF_latent_class(
            BayesFactorTriplet(bf_enrichment, bf_correlation, bf_detected_baseline),
            refID; verbose=verbose, return_responsibilities=true)
        h0_precomputed = precompute_h0(
            BayesFactorTriplet(bf_enrichment, bf_correlation, bf_detected_baseline),
            phase1_precomputed; verbose=verbose
        )
    end

    # Build the list of prior settings and compute posteriors for each.
    # First entry is always the actual baseline from the completed analysis.
    prior_settings = PriorSetting[]
    posterior_columns = Vector{Float64}[]
    bf_columns = Vector{Float64}[]
    bfdr_columns = Vector{Float64}[]

    # Insert baseline from the analysis result as the first column
    baseline_posterior = Vector{Float64}(cr.posterior_prob)
    baseline_bf = Vector{Float64}(cr.BF)
    baseline_bfdr = bfdr(baseline_bf)
    push!(prior_settings, PriorSetting(:baseline, "Baseline", (;)))
    push!(posterior_columns, baseline_posterior)
    push!(bf_columns, baseline_bf)
    push!(bfdr_columns, baseline_bfdr)
    baseline_index = 1

    # ------------------------------------------------------------------ #
    # 1. Beta-Bernoulli prior sweep (optional — disabled by default)
    # ------------------------------------------------------------------ #
    # The main pipeline uses a fixed BB prior (3,3) that is not user-configurable,
    # so sweeping BB priors measures hypothetical sensitivity rather than actual
    # uncertainty. Enable by passing bb_priors in SensitivityConfig.
    bb_priors = config.bb_priors
    n_bb = length(bb_priors)

    if n_bb > 0
        bb_n_restarts = combination_method == :latent_class ? 10 : 20
        bb_results = Vector{Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}(undef, n_bb)
        bb_prior_settings = Vector{PriorSetting}(undef, n_bb)

        verbose && @info "BB prior sweep: running $n_bb settings in parallel..."

        # Pre-compute all BB detection BF vectors
        bb_bf_d_list = Vector{Vector{Float64}}(undef, n_bb)
        for (idx, (α, β)) in enumerate(bb_priors)
            bf_d_full = _recompute_bb_bf(data, n_controls, n_samples; prior_alpha=α, prior_beta=β)
            bf_lookup = Dict(all_protein_ids[i] => bf_d_full[i] for i in eachindex(all_protein_ids))
            bb_bf_d_list[idx] = [get(bf_lookup, name, 0.0) for name in protein_names]
        end

        # Parallel evidence recombination across BB settings
        bb_tasks = map(1:n_bb) do idx
            Threads.@spawn begin
                (α, β) = bb_priors[idx]
                bf_d = bb_bf_d_list[idx]
                _recombine_evidence(
                    bf_enrichment, bf_correlation, bf_d, refID;
                    combination_method = combination_method,
                    H0_file = H0_file,
                    lc_n_iterations = lc_n_iterations,
                    lc_convergence_tol = lc_convergence_tol,
                    verbose = false,
                    precomputed_h0 = h0_precomputed,
                    phase1_result = phase1_precomputed,
                    n_restarts = bb_n_restarts,
                    force_h1_family = baseline_h1_family
                )
            end
        end

        for (idx, (α, β)) in enumerate(bb_priors)
            label = "BB($(α),$(β))"
            bb_prior_settings[idx] = PriorSetting(:betabernoulli, label, (α=α, β=β))
            bb_results[idx] = fetch(bb_tasks[idx])
        end

        for idx in 1:n_bb
            push!(prior_settings, bb_prior_settings[idx])
            bf, posterior, bfdr_vals = bb_results[idx]
            push!(posterior_columns, posterior)
            push!(bf_columns, bf)
            push!(bfdr_columns, bfdr_vals)
        end
    end

    # ------------------------------------------------------------------ #
    # 2. Copula-EM prior sweep (only for copula mode, parallel)
    # ------------------------------------------------------------------ #
    if combination_method == :copula
        em_priors = config.em_prior_grid
        n_em = length(em_priors)
        em_results = Vector{Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}(undef, n_em)

        verbose && @info "Copula-EM prior sweep: running $n_em settings in parallel..."

        em_tasks = map(1:n_em) do idx
            em_prior = em_priors[idx]
            Threads.@spawn _recombine_evidence(
                bf_enrichment, bf_correlation, bf_detected_baseline, refID;
                combination_method = :copula,
                H0_file = H0_file,
                em_prior = em_prior,
                verbose = false,
                precomputed_h0 = h0_precomputed,
                phase1_result = phase1_precomputed
            )
        end

        for idx in 1:n_em
            em_prior = em_priors[idx]
            expected = round(em_prior.α / (em_prior.α + em_prior.β), digits=3)
            label = "EM(α=$(em_prior.α),β=$(em_prior.β),E[π₁]=$(expected))"
            push!(prior_settings, PriorSetting(:copula_em, label, (α=em_prior.α, β=em_prior.β)))
            em_results[idx] = fetch(em_tasks[idx])
            bf, posterior, bfdr_vals = em_results[idx]
            push!(posterior_columns, posterior)
            push!(bf_columns, bf)
            push!(bfdr_columns, bfdr_vals)
        end
    end

    # ------------------------------------------------------------------ #
    # 3. Latent class prior sweep (only for latent_class mode, parallel)
    # ------------------------------------------------------------------ #
    if combination_method == :latent_class
        lc_priors = config.lc_alpha_prior_grid
        n_lc = length(lc_priors)
        lc_results = Vector{Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}(undef, n_lc)

        verbose && @info "LC prior sweep: running $n_lc settings in parallel (n_restarts=20)..."

        lc_tasks = map(1:n_lc) do idx
            lc_prior = lc_priors[idx]
            Threads.@spawn _recombine_evidence(
                bf_enrichment, bf_correlation, bf_detected_baseline, refID;
                combination_method = :latent_class,
                lc_alpha_prior = lc_prior,
                lc_n_iterations = lc_n_iterations,
                lc_convergence_tol = lc_convergence_tol,
                verbose = false,
                n_restarts = 20,
                force_h1_family = baseline_h1_family
            )
        end

        for idx in 1:n_lc
            lc_prior = lc_priors[idx]
            label = "LC(α=[$(join(lc_prior, ","))])"
            push!(prior_settings, PriorSetting(:latent_class, label, (alpha_prior=lc_prior,)))
            lc_results[idx] = fetch(lc_tasks[idx])
            bf, posterior, bfdr_vals = lc_results[idx]
            push!(posterior_columns, posterior)
            push!(bf_columns, bf)
            push!(bfdr_columns, bfdr_vals)
        end
    end

    # ------------------------------------------------------------------ #
    # 3.5  BMA Cartesian sweep: LC grid x copula EM grid
    # ------------------------------------------------------------------ #
    if combination_method == :bma
        lc_priors = config.lc_alpha_prior_grid
        em_priors = config.em_prior_grid
        n_lc = length(lc_priors)
        n_em = length(em_priors)

        verbose && @info "BMA sensitivity sweep: $(n_lc) LC x $(n_em) EM = $(n_lc * n_em) grid points"

        # Stage 1: Run LC once per LC prior (reused across copula EM variants).
        lc_results_bma = Vector{Any}(undef, n_lc)
        lc_tasks_bma = map(1:n_lc) do idx
            lc_prior = lc_priors[idx]
            Threads.@spawn combined_BF_latent_class(
                BayesFactorTriplet(bf_enrichment, bf_correlation, bf_detected_baseline),
                refID;
                alpha_prior = lc_prior,
                n_iterations = lc_n_iterations,
                convergence_tol = lc_convergence_tol,
                verbose = false,
                n_restarts = 20,
                force_h1_family = baseline_h1_family
            )
        end
        for idx in 1:n_lc
            lc_results_bma[idx] = fetch(lc_tasks_bma[idx])
        end

        # Stage 2: For each (LC, EM) pair, run copula with that LC as the stage-1 result + EM prior override,
        # then compute fresh stacking weights.
        triplet = BayesFactorTriplet(bf_enrichment, bf_correlation, bf_detected_baseline)

        # Build flat task list for all (lc_idx, em_idx) combinations
        bma_task_list = Tuple{Int,Int}[]
        for lc_idx in 1:n_lc, em_idx in 1:n_em
            push!(bma_task_list, (lc_idx, em_idx))
        end

        bma_tasks = map(bma_task_list) do (lc_idx, em_idx)
            Threads.@spawn begin
                lc_result = lc_results_bma[lc_idx]
                em_prior = em_priors[em_idx]

                # Convert Beta(alpha,beta) on pi_1 to 3-component Dirichlet for copula EM.
                # Beta(alpha,beta) encodes E[pi_1] = alpha/(alpha+beta).
                # Map to Dirichlet[d_H0, d_ag, d_H1] preserving total strength S=alpha+beta:
                #   d_H1 = alpha  (pseudo-counts for H1)
                #   d_H0 = beta * 0.7  (70:30 split matches default [5,2,1] ratio)
                #   d_ag = beta * 0.3
                d_H1 = em_prior.α
                d_H0 = em_prior.β * 0.7
                d_ag = em_prior.β * 0.3
                copula_dir = [d_H0, d_ag, d_H1]

                cop_result = combined_BF(
                    triplet, refID;
                    phase1_result = lc_result,
                    n_restarts = 5,
                    verbose = false,
                    precomputed_h0 = h0_precomputed,
                    copula_dirichlet_prior = copula_dir
                )

                # Fresh stacking weights
                ll_em = pointwise_ll_em(lc_result, triplet; winsorize=false)
                ll_cop = pointwise_ll_copula(cop_result)
                w_em_raw, w_cop_raw = stacking_weights(ll_em, ll_cop)
                weight_floor = 0.05
                w_em = max(w_em_raw, weight_floor)
                w_cop = max(w_cop_raw, weight_floor)
                w_sum = w_em + w_cop
                w_em /= w_sum
                w_cop /= w_sum

                # Prior odds from LC mixing weights
                mw = lc_result.mixing_weights
                prior_odds = mw[end] / max(sum(mw[1:end-1]), 1e-300)

                # Merge posteriors
                P_avg, BF_avg, _ = merge_posteriors(
                    lc_result.bf, cop_result.bf, prior_odds, w_em, w_cop;
                    bf_triplet = triplet
                )
                bfdr_vals = bfdr(BF_avg)

                (BF_avg, P_avg, bfdr_vals, w_em, w_cop)
            end
        end

        # Collect results in order
        for (task_idx, (lc_idx, em_idx)) in enumerate(bma_task_list)
            lc_prior = lc_priors[lc_idx]
            em_prior = em_priors[em_idx]

            lc_label = "LC(α=[$(join(round.(lc_prior, digits=2), ","))])"
            expected_pi1 = round(em_prior.α / (em_prior.α + em_prior.β), digits=3)
            label = "$(lc_label) | E[π₁]=$(expected_pi1)"

            bf_vec, post_vec, bfdr_vec, w_em_val, w_cop_val = fetch(bma_tasks[task_idx])

            push!(prior_settings, PriorSetting(:bma, label, (
                alpha_prior = lc_prior,
                em_alpha = em_prior.α,
                em_beta = em_prior.β,
                w_em = w_em_val,
                w_cop = w_cop_val,
            )))
            push!(posterior_columns, post_vec)
            push!(bf_columns, bf_vec)
            push!(bfdr_columns, bfdr_vec)
        end
    end

    # baseline_index is always 1 (the actual analysis result)

    # ------------------------------------------------------------------ #
    # 4. Assemble matrices and compute summaries
    # ------------------------------------------------------------------ #
    n_settings = length(prior_settings)
    posterior_matrix = hcat(posterior_columns...)  # n_proteins × n_settings
    bf_matrix = hcat(bf_columns...)
    bfdr_matrix = hcat(bfdr_columns...)

    summary_df = _compute_sensitivity_summary(posterior_matrix, protein_names, baseline_index)
    stability_df = _compute_classification_stability(posterior_matrix, bfdr_matrix, protein_names)

    return SensitivityResult(
        config,
        prior_settings,
        posterior_matrix,
        bf_matrix,
        bfdr_matrix,
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
    verbose::Bool = false,
    precomputed_h0::Union{Nothing, PrecomputedH0} = nothing,
    phase1_result::Union{Nothing, LatentClassResult} = nothing,
    n_restarts::Int = 20,
    force_h1_family::Union{Nothing, Symbol} = nothing
)
    triplet = BayesFactorTriplet(bf_enrichment, bf_correlation, bf_detected)

    if combination_method == :copula
        # Compute the stage-1 result if not provided
        p1 = phase1_result
        if p1 === nothing
            p1 = combined_BF_latent_class(triplet, refID; verbose=verbose, return_responsibilities=true)
        end
        # Convert em_prior Beta(alpha,beta) to copula Dirichlet if provided
        copula_dir_kw = if em_prior !== nothing
            d_H1 = em_prior.α
            d_H0 = em_prior.β * 0.7
            d_ag = em_prior.β * 0.3
            [d_H0, d_ag, d_H1]
        else
            [5.0, 2.0, 1.0]  # default
        end
        result = combined_BF(
            triplet, refID;
            phase1_result = p1,
            n_restarts = 5,  # reduced restarts for sensitivity sweep
            verbose = verbose,
            precomputed_h0 = precomputed_h0,
            copula_dirichlet_prior = copula_dir_kw
        )
        bf = result.bf
        posterior = result.posterior_prob
    elseif combination_method == :latent_class
        result = combined_BF_latent_class(
            triplet, refID;
            alpha_prior = lc_alpha_prior,
            n_iterations = lc_n_iterations,
            convergence_tol = lc_convergence_tol,
            verbose = verbose,
            n_restarts = n_restarts,
            force_h1_family = force_h1_family
        )
        bf = result.bf
        posterior = result.posterior_prob
    elseif combination_method == :bma
        # Full BMA: run LC + copula + stacking
        lc_result = combined_BF_latent_class(
            triplet, refID;
            alpha_prior = lc_alpha_prior,
            n_iterations = lc_n_iterations,
            convergence_tol = lc_convergence_tol,
            verbose = verbose,
            n_restarts = n_restarts,
            force_h1_family = force_h1_family
        )

        # Convert em_prior Beta(alpha,beta) to copula Dirichlet if provided
        copula_dir_kw = if em_prior !== nothing
            d_H1 = em_prior.α
            d_H0 = em_prior.β * 0.7
            d_ag = em_prior.β * 0.3
            [d_H0, d_ag, d_H1]
        else
            [5.0, 2.0, 1.0]  # default
        end

        cop_result = combined_BF(
            triplet, refID;
            phase1_result = lc_result,
            n_restarts = 5,
            verbose = verbose,
            precomputed_h0 = precomputed_h0,
            copula_dirichlet_prior = copula_dir_kw
        )

        # Stacking weights
        ll_em = pointwise_ll_em(lc_result, triplet; winsorize=false)
        ll_cop = pointwise_ll_copula(cop_result)
        w_em_raw, w_cop_raw = stacking_weights(ll_em, ll_cop)
        w_em = max(w_em_raw, 0.05)
        w_cop = max(w_cop_raw, 0.05)
        w_sum = w_em + w_cop
        w_em /= w_sum; w_cop /= w_sum
        mw = lc_result.mixing_weights
        prior_odds = mw[end] / max(sum(mw[1:end-1]), 1e-300)
        P_avg, BF_avg, _ = merge_posteriors(
            lc_result.bf, cop_result.bf, prior_odds, w_em, w_cop;
            bf_triplet = triplet
        )
        bf = BF_avg
        posterior = P_avg
    else
        error("Unknown combination_method: $combination_method")
    end

    bfdr_vals = bfdr(bf)
    return bf, posterior, bfdr_vals
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
    _compute_classification_stability(posterior_matrix, bfdr_matrix, protein_names) -> DataFrame

Per-protein classification stability: fraction of settings where protein exceeds
P > 0.5, P > 0.8, P > 0.95, and BFDR < 0.05, BFDR < 0.01.
"""
function _compute_classification_stability(
    posterior_matrix::Matrix{Float64},
    bfdr_matrix::Matrix{Float64},
    protein_names::Vector{String}
)
    n_settings = size(posterior_matrix, 2)

    stability = DataFrame(
        Protein = protein_names,
        frac_P_gt_0_5  = vec(sum(posterior_matrix .> 0.5, dims=2)) ./ n_settings,
        frac_P_gt_0_8  = vec(sum(posterior_matrix .> 0.8, dims=2)) ./ n_settings,
        frac_P_gt_0_95 = vec(sum(posterior_matrix .> 0.95, dims=2)) ./ n_settings,
        frac_BFDR_lt_0_05 = vec(sum(bfdr_matrix .< 0.05, dims=2)) ./ n_settings,
        frac_BFDR_lt_0_01 = vec(sum(bfdr_matrix .< 0.01, dims=2)) ./ n_settings
    )

    # Threshold crossing detection: does this protein cross the boundary across settings?
    threshold_crossing_0_95 = [
        any(posterior_matrix[i, :] .>= 0.95) && any(posterior_matrix[i, :] .< 0.95)
        for i in 1:size(posterior_matrix, 1)
    ]
    threshold_crossing_0_5 = [
        any(posterior_matrix[i, :] .>= 0.5) && any(posterior_matrix[i, :] .< 0.5)
        for i in 1:size(posterior_matrix, 1)
    ]

    stability.threshold_crossing_0_95 = threshold_crossing_0_95
    stability.threshold_crossing_0_5 = threshold_crossing_0_5

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

    # Classification stability
    println(io, "## Classification Stability")
    println(io)
    println(io, "| Threshold | Stable (100%) | Mostly (>80%) | Unstable (<50%) |")
    println(io, "|-----------|--------------|---------------|-----------------|")
    for (col, label) in [
        (:frac_P_gt_0_5, "P > 0.5"),
        (:frac_P_gt_0_8, "P > 0.8"),
        (:frac_P_gt_0_95, "P > 0.95"),
        (:frac_BFDR_lt_0_05, "BFDR < 0.05"),
        (:frac_BFDR_lt_0_01, "BFDR < 0.01")
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
        Base.write(f, content)
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
