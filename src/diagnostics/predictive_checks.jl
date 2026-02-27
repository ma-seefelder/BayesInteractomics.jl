# Posterior Predictive Checks & Model Diagnostics
# Core PPC logic for HBM, regression, and Beta-Bernoulli models

using Random
import DataFrames: nrow, leftjoin
import Distributions: Binomial, Beta, mean as dist_mean

"""
    model_diagnostics(ar, data::InteractionData; kwargs...) -> DiagnosticsResult

Run posterior predictive checks (PPC) and model diagnostics on analysis results.

Re-runs inference for a subset of proteins to generate posterior predictive samples,
computes standardized residuals, and assesses calibration consistency.

# Arguments
- `ar`: Completed `AnalysisResult` from `run_analysis()`
- `data::InteractionData`: Raw data used in the analysis

# Keywords
- `config::DiagnosticsConfig`: Diagnostics configuration (default: `DiagnosticsConfig()`)
- `n_controls::Int`: Number of control replicates
- `n_samples::Int`: Number of sample replicates
- `refID::Int`: Bait protein index
- `verbose::Bool`: Print progress info

# Returns
- `DiagnosticsResult` with PPC results, residuals, calibration, and summary
"""
function model_diagnostics(
    ar,  # AnalysisResult — untyped to avoid world-age issues
    data::InteractionData;
    config::DiagnosticsConfig = DiagnosticsConfig(),
    n_controls::Int = 0,
    n_samples::Int = 0,
    refID::Int = 1,
    verbose::Bool = false,
    regression_likelihood::Symbol = :normal,
    student_t_nu::Float64 = 5.0,
    robust_tau_base::Float64 = NaN
)
    cr = ar.copula_results
    protein_names = Vector{String}(cr.Protein)
    n_proteins = length(protein_names)

    rng = Random.MersenneTwister(config.seed)

    # Select proteins for PPC
    selected_indices = _select_proteins_for_ppc(cr, config, rng)
    verbose && @info "Selected $(length(selected_indices)) proteins for PPC"

    # Map protein names to data indices
    all_ids = getIDs(data)
    name_to_idx = Dict(all_ids[i] => i for i in eachindex(all_ids))

    # Precompute priors for re-running inference
    τ_dist = τ0(data)
    a_0, b_0 = τ_dist.α, τ_dist.θ
    μ_0, σ_0 = μ0(data)

    if getNoProtocols(data) == 1
        cached_hbm_prior = precompute_HBM_single_protocol_prior(data, μ_0=μ_0, σ_0=σ_0, a_0=a_0, b_0=b_0)
        if regression_likelihood == :robust_t
            cached_regression_prior = precompute_regression_one_protocol_robust_prior(data, refID, μ_0, σ_0; nu=student_t_nu, τ_base=robust_tau_base)
        else
            cached_regression_prior = precompute_regression_one_protocol_prior(data, refID, μ_0, σ_0)
        end
    else
        cached_hbm_prior = precompute_HBM_prior(data, μ_0=μ_0, σ_0=σ_0, a_0=a_0, b_0=b_0)
        if regression_likelihood == :robust_t
            cached_regression_prior = precompute_regression_multi_protocol_robust_prior(data, refID, μ_0, σ_0; nu=student_t_nu, τ_base=robust_tau_base)
        else
            cached_regression_prior = precompute_regression_multi_protocol_prior(data, refID, μ_0, σ_0)
        end
    end

    # Run PPC for selected proteins (threaded — main() is thread-safe, see pipeline.jl:239)
    n_selected = length(selected_indices)

    # Pre-allocate per-slot storage for thread safety (no push! into shared collections)
    slot_hbm_ppcs       = Vector{Union{Nothing, ProteinPPC}}(nothing, n_selected)
    slot_reg_ppcs       = Vector{Union{Nothing, ProteinPPC}}(nothing, n_selected)
    slot_bb_ppcs        = Vector{Union{Nothing, BetaBernoulliPPC}}(nothing, n_selected)
    slot_hbm_results    = Vector{Union{Nothing, Pair{String, HBMResult}}}(nothing, n_selected)
    slot_reg_results    = Vector{Union{Nothing, Pair{String, AnyRegressionResult}}}(nothing, n_selected)

    ppc_progress = Progress(
        n_selected, desc="Diagnostics: PPC inference...",
        showspeed=true,
        barglyphs=BarGlyphs('|','█', [' ' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|'),
        barlen=20
    )

    Threads.@threads for slot in 1:n_selected
        sel_idx = selected_indices[slot]
        pname = protein_names[sel_idx]
        data_idx = get(name_to_idx, pname, nothing)
        if isnothing(data_idx)
            ProgressMeter.next!(ppc_progress)
            continue
        end

        # Per-slot deterministic RNG (thread-safe: each slot gets its own)
        slot_rng = Random.MersenneTwister(config.seed + sel_idx)

        try
            # Re-run inference for this protein
            bayes_result = main(
                data, data_idx, refID,
                plotHBMdists=false, plotlog2fc=false,
                plotregr=false, plotbayesrange=false,
                csv_file=tempname(), writecsv=false,
                verbose=false, computeHBM=true,
                μ_0=μ_0, σ_0=σ_0, a_0=a_0, b_0=b_0,
                cached_hbm_prior=cached_hbm_prior,
                cached_regression_prior=cached_regression_prior,
                regression_likelihood=regression_likelihood,
                student_t_nu=student_t_nu,
                robust_tau_base=robust_tau_base
            )

            # HBM PPC
            if !isnothing(bayes_result.hbm_result)
                slot_hbm_results[slot] = pname => bayes_result.hbm_result
                slot_hbm_ppcs[slot] = _ppc_hbm(data, data_idx, bayes_result.hbm_result;
                                                n_draws=config.n_ppc_draws, rng=slot_rng)
            end

            # Regression PPC
            if !isnothing(bayes_result.regression_result)
                slot_reg_results[slot] = pname => bayes_result.regression_result
                slot_reg_ppcs[slot] = _ppc_regression(data, data_idx, refID, bayes_result.regression_result;
                                                       n_draws=config.n_ppc_draws, rng=slot_rng)
            end

            # Beta-Bernoulli PPC
            slot_bb_ppcs[slot] = _ppc_betabernoulli(data, data_idx, n_controls, n_samples;
                                                     n_draws=config.n_ppc_draws, rng=slot_rng)
        catch e
            verbose && @warn "PPC failed for protein $pname: $e"
        end
        ProgressMeter.next!(ppc_progress)
    end
    finish!(ppc_progress)

    # Collect results from slots (filter out nothing entries)
    protein_ppcs = ProteinPPC[p for p in slot_hbm_ppcs if !isnothing(p)]
    append!(protein_ppcs, ProteinPPC[p for p in slot_reg_ppcs if !isnothing(p)])
    bb_ppcs = BetaBernoulliPPC[p for p in slot_bb_ppcs if !isnothing(p)]

    hbm_results_map = Dict{String, HBMResult}(
        kv.first => kv.second for kv in slot_hbm_results if !isnothing(kv))
    regression_results_map = Dict{String, AnyRegressionResult}(
        kv.first => kv.second for kv in slot_reg_results if !isnothing(kv))

    verbose && @info "Completed PPC for $(length(protein_ppcs)) model checks and $(length(bb_ppcs)) BB checks"

    # Compute residuals
    hbm_residuals = nothing
    regression_residuals = nothing

    if config.residual_model in (:both, :hbm) && !isempty(hbm_results_map)
        try
            hbm_residuals = _compute_hbm_residuals(data, hbm_results_map, name_to_idx)
        catch e
            verbose && @warn "HBM residual computation failed: $e"
        end
    end

    if config.residual_model in (:both, :regression) && !isempty(regression_results_map)
        try
            regression_residuals = _compute_regression_residuals(
                data, regression_results_map, name_to_idx, refID)
        catch e
            verbose && @warn "Regression residual computation failed: $e"
        end
    end

    # Compute calibration (strict: all 3 BFs > 1.0)
    calibration = nothing
    try
        calibration = _compute_calibration(ar; n_bins=config.calibration_bins)
    catch e
        verbose && @warn "Calibration computation failed: $e"
    end

    # Compute relaxed calibration (2-of-3 BFs > 1.0)
    calibration_relaxed = nothing
    try
        calibration_relaxed = _compute_calibration_relaxed(ar; n_bins=config.calibration_bins)
    catch e
        verbose && @warn "Relaxed calibration computation failed: $e"
    end

    # Compute enrichment-only calibration (BF_enrichment > 3.0)
    calibration_enrichment_only = nothing
    try
        calibration_enrichment_only = _compute_calibration_enrichment_only(ar; n_bins=config.calibration_bins)
    catch e
        verbose && @warn "Enrichment-only calibration computation failed: $e"
    end

    # Enhanced residuals (randomized quantile residuals + PIT values)
    enhanced_hbm = nothing
    enhanced_regression = nothing

    if config.enhanced_residuals
        n_enhanced_steps = (config.residual_model in (:both, :hbm) && !isempty(hbm_results_map) ? 1 : 0) +
                           (config.residual_model in (:both, :regression) && !isempty(regression_results_map) ? 1 : 0)
        enh_progress = Progress(
            n_enhanced_steps, desc="Diagnostics: Enhanced residuals...",
            showspeed=true,
            barglyphs=BarGlyphs('|','█', [' ' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|'),
            barlen=20,
            enabled=n_enhanced_steps > 0
        )

        if config.residual_model in (:both, :hbm) && !isempty(hbm_results_map)
            try
                enhanced_hbm = _compute_enhanced_hbm_residuals(
                    data, hbm_results_map, name_to_idx; rng=rng)
                verbose && @info "Computed enhanced HBM residuals ($(length(enhanced_hbm.pit_values)) PIT values)"
            catch e
                verbose && @warn "Enhanced HBM residual computation failed: $e"
            end
            ProgressMeter.next!(enh_progress)
        end

        if config.residual_model in (:both, :regression) && !isempty(regression_results_map)
            try
                enhanced_regression = _compute_enhanced_regression_residuals(
                    data, regression_results_map, name_to_idx, refID; rng=rng)
                verbose && @info "Computed enhanced regression residuals ($(length(enhanced_regression.pit_values)) PIT values)"
            catch e
                verbose && @warn "Enhanced regression residual computation failed: $e"
            end
            ProgressMeter.next!(enh_progress)
        end
        finish!(enh_progress)
    end

    # Extended PPC statistics (skewness, kurtosis, IQR/SD ratio)
    ppc_extended = PPCExtendedStatistics[]
    for ppc in protein_ppcs
        ext = _compute_extended_ppc_stats(ppc)
        !isnothing(ext) && push!(ppc_extended, ext)
    end
    ppc_extended = isempty(ppc_extended) ? nothing : ppc_extended

    # Per-protein diagnostic flags (computed for ALL proteins, not just PPC subset)
    protein_flags = nothing
    base_residuals = !isnothing(enhanced_hbm) ? enhanced_hbm.base :
                     !isnothing(hbm_residuals) ? hbm_residuals : nothing
    flags_progress = Progress(
        n_proteins, desc="Diagnostics: Protein flags...",
        showspeed=true,
        barglyphs=BarGlyphs('|','█', [' ' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|'),
        barlen=20
    )
    try
        protein_flags = _compute_protein_flags(
            data, protein_names, name_to_idx;
            ppc_residuals = base_residuals,
            progress = flags_progress
        )
    catch e
        verbose && @warn "Protein flag computation failed: $e"
    end
    finish!(flags_progress)

    # Build summary DataFrame
    summary_df = _build_diagnostics_summary(protein_ppcs, bb_ppcs)

    return DiagnosticsResult(
        config,
        protein_ppcs,
        bb_ppcs,
        hbm_residuals,
        regression_residuals,
        calibration,
        calibration_relaxed,
        calibration_enrichment_only,
        enhanced_hbm,
        enhanced_regression,
        ppc_extended,
        protein_flags,
        nothing,  # model_comparison (computed separately via compare_regression_models)
        nothing,  # nu_optimization (computed separately via optimize_nu)
        summary_df,
        now()
    )
end

# ============================================================================ #
# Protein selection
# ============================================================================ #

"""
    _select_proteins_for_ppc(cr, config, rng) -> Vector{Int}

Select protein indices (into copula_results) for PPC.
Strategy `:top_and_random` picks top N/2 by combined BF + N/2 random.
"""
function _select_proteins_for_ppc(cr::DataFrame, config::DiagnosticsConfig, rng::AbstractRNG)
    n_proteins = nrow(cr)
    n_check = min(config.n_proteins_to_check, n_proteins)

    if config.ppc_protein_selection == :top_and_random
        n_top = div(n_check, 2)
        n_random = n_check - n_top

        # Top by combined BF
        bf_order = sortperm(cr.BF, rev=true)
        top_indices = bf_order[1:min(n_top, n_proteins)]

        # Random from remaining
        remaining = setdiff(1:n_proteins, top_indices)
        n_random = min(n_random, length(remaining))
        random_indices = n_random > 0 ? randperm(rng, length(remaining))[1:n_random] : Int[]
        random_indices = remaining[random_indices]

        return vcat(top_indices, random_indices)

    elseif config.ppc_protein_selection == :stratified
        # Stratified selection: sample equally from posterior probability quartiles
        pp = cr.posterior_prob
        sorted_indices = sortperm(pp)
        quartile_size = div(n_proteins, 4)
        per_quartile = max(1, div(n_check, 4))

        selected = Int[]
        for q in 1:4
            lo = (q - 1) * quartile_size + 1
            hi = q == 4 ? n_proteins : q * quartile_size
            quartile_indices = sorted_indices[lo:hi]
            n_pick = min(per_quartile, length(quartile_indices))
            perm = randperm(rng, length(quartile_indices))
            append!(selected, quartile_indices[perm[1:n_pick]])
        end
        return unique(selected)[1:min(n_check, length(unique(selected)))]

    else
        # Fallback: random selection
        return randperm(rng, n_proteins)[1:n_check]
    end
end

# ============================================================================ #
# HBM PPC
# ============================================================================ #

"""
    _ppc_hbm(data, idx, hbm_result; n_draws, rng) -> Union{ProteinPPC, Nothing}

Generate posterior predictive samples from the HBM model for a single protein.
"""
function _ppc_hbm(
    data::InteractionData, idx::Int, hbm_result::HBMResult;
    n_draws::Int = 1000, rng::AbstractRNG = Random.GLOBAL_RNG
)
    protein = getProteinData(data, idx)
    protein_name = protein.name
    posterior = hbm_result.posterior

    # Extract observed sample data (flatten across protocols/experiments)
    sample_mat = getSampleMatrix(protein)
    observed = Float64[]
    for val in sample_mat
        !ismissing(val) && push!(observed, val)
    end
    isempty(observed) && return nothing

    n_obs = length(observed)
    simulated = Matrix{Float64}(undef, n_obs, n_draws)

    # Get posterior distributions for the first experiment-level parameter
    # For single protocol: posteriors[:μ_sample] is a vector, use index 2 (first experiment)
    # For multi protocol: use parameter_lookup to find the right index
    μ_posteriors = posterior.posteriors[:μ_sample]
    τ_posteriors = posterior.posteriors[:σ_sample]

    # Use the global-level posterior (index 1) as representative
    μ_dist = μ_posteriors[1]
    τ_dist = τ_posteriors[1]

    # Extract posterior mean and precision
    μ_mean = dist_mean(to_normal(μ_dist))
    τ_mean = dist_mean(τ_dist)
    τ_mean = max(τ_mean, 1e-10)

    for d in 1:n_draws
        # Sample from posterior predictive
        μ_draw = rand(rng, to_normal(μ_dist))
        τ_draw = rand(rng, τ_dist)
        τ_draw = max(τ_draw, 1e-10)
        sd_draw = 1.0 / sqrt(τ_draw)

        for i in 1:n_obs
            simulated[i, d] = rand(rng, Normal(μ_draw, sd_draw))
        end
    end

    # Compute Bayesian p-values
    obs_mean = mean(observed)
    obs_sd = length(observed) > 1 ? std(observed) : 0.0

    sim_means = vec(mean(simulated, dims=1))
    sim_sds = vec([length(observed) > 1 ? std(simulated[:, d]) : 0.0 for d in 1:n_draws])

    pvalue_mean = sum(sim_means .>= obs_mean) / n_draws
    pvalue_sd = sum(sim_sds .>= obs_sd) / n_draws

    # Log2FC p-value: compare observed vs simulated fold change relative to control
    control_mat = getControlMatrix(protein)
    control_vals = Float64[]
    for val in control_mat
        !ismissing(val) && push!(control_vals, val)
    end

    if !isempty(control_vals)
        obs_log2fc = log2(max(mean(observed), 1e-10)) - log2(max(mean(control_vals), 1e-10))
        sim_log2fcs = [log2(max(mean(simulated[:, d]), 1e-10)) - log2(max(mean(control_vals), 1e-10))
                       for d in 1:n_draws]
        pvalue_log2fc = sum(sim_log2fcs .>= obs_log2fc) / n_draws
    else
        pvalue_log2fc = NaN
    end

    return ProteinPPC(protein_name, :hbm, observed, simulated, pvalue_mean, pvalue_sd, pvalue_log2fc)
end

# ============================================================================ #
# Regression PPC
# ============================================================================ #

"""
    _ppc_regression(data, idx, refID, regression_result; n_draws, rng) -> Union{ProteinPPC, Nothing}

Generate posterior predictive samples from the regression model for a single protein.
"""
function _ppc_regression(
    data::InteractionData, idx::Int, refID::Int, regression_result::RegressionResult;
    n_draws::Int = 1000, rng::AbstractRNG = Random.GLOBAL_RNG
)
    protein = getProteinData(data, idx)
    protein_name = protein.name
    posterior = regression_result.posterior

    # Prepare regression data
    reg_data = prepare_regression_data(data, idx, refID)
    sample_flat = Float64[]
    reference_flat = Float64[]

    for val in reg_data.sample
        if !ismissing(val)
            push!(sample_flat, val)
        end
    end
    for val in reg_data.reference
        if !ismissing(val)
            push!(reference_flat, val)
        end
    end

    n_obs = min(length(sample_flat), length(reference_flat))
    n_obs == 0 && return nothing
    sample_flat = sample_flat[1:n_obs]
    reference_flat = reference_flat[1:n_obs]

    simulated = Matrix{Float64}(undef, n_obs, n_draws)

    # Extract posterior parameters
    if regression_result isa RegressionResultSingleProtocol
        α_dist = posterior.posteriors[:α]
        β_dist = posterior.posteriors[:β]
    else
        # Multi-protocol: use first protocol slopes
        α_dist = posterior.posteriors[:α][1]
        β_dist = posterior.posteriors[:β][1]
    end
    σ_dist = posterior.posteriors[:σ]

    for d in 1:n_draws
        α_draw = rand(rng, to_normal(α_dist))
        β_draw = rand(rng, to_normal(β_dist))
        τ_draw = rand(rng, σ_dist)
        τ_draw = max(τ_draw, 1e-10)
        sd_draw = 1.0 / sqrt(τ_draw)

        for i in 1:n_obs
            predicted = β_draw + α_draw * reference_flat[i]
            simulated[i, d] = rand(rng, Normal(predicted, sd_draw))
        end
    end

    # Bayesian p-values
    obs_mean = mean(sample_flat)
    obs_sd = length(sample_flat) > 1 ? std(sample_flat) : 0.0

    sim_means = vec(mean(simulated, dims=1))
    sim_sds = vec([length(sample_flat) > 1 ? std(simulated[:, d]) : 0.0 for d in 1:n_draws])

    pvalue_mean = sum(sim_means .>= obs_mean) / n_draws
    pvalue_sd = sum(sim_sds .>= obs_sd) / n_draws

    return ProteinPPC(protein_name, :regression, sample_flat, simulated, pvalue_mean, pvalue_sd, NaN)
end

"""
    _ppc_regression(data, idx, refID, regression_result::RobustRegressionResult; n_draws, rng) -> Union{ProteinPPC, Nothing}

Generate posterior predictive samples from the robust regression model (Empirical Bayes τ_base).
"""
function _ppc_regression(
    data::InteractionData, idx::Int, refID::Int, regression_result::RobustRegressionResult;
    n_draws::Int = 1000, rng::AbstractRNG = Random.GLOBAL_RNG
)
    protein = getProteinData(data, idx)
    protein_name = protein.name
    posterior = regression_result.posterior

    # Prepare regression data
    reg_data = prepare_regression_data(data, idx, refID)
    sample_flat = Float64[]
    reference_flat = Float64[]

    for val in reg_data.sample
        if !ismissing(val)
            push!(sample_flat, val)
        end
    end
    for val in reg_data.reference
        if !ismissing(val)
            push!(reference_flat, val)
        end
    end

    n_obs = min(length(sample_flat), length(reference_flat))
    n_obs == 0 && return nothing
    sample_flat = sample_flat[1:n_obs]
    reference_flat = reference_flat[1:n_obs]

    simulated = Matrix{Float64}(undef, n_obs, n_draws)

    # Extract posterior parameters
    is_single = regression_result isa RobustRegressionResultSingleProtocol
    if is_single
        α_dist = posterior.posteriors[:α]
        β_dist = posterior.posteriors[:β]
    else
        α_dist = posterior.posteriors[:α][1]
        β_dist = posterior.posteriors[:β][1]
    end

    # Robust model: sample from Student-t(ν, predicted, 1/√τ_base) marginal
    nu = regression_result.nu
    scale = 1.0 / sqrt(regression_result.τ_base)
    tdist = TDist(nu)

    for d in 1:n_draws
        α_draw = rand(rng, to_normal(α_dist))
        β_draw = rand(rng, to_normal(β_dist))

        for i in 1:n_obs
            predicted = β_draw + α_draw * reference_flat[i]
            simulated[i, d] = predicted + scale * rand(rng, tdist)
        end
    end

    # Bayesian p-values
    obs_mean = mean(sample_flat)
    obs_sd = length(sample_flat) > 1 ? std(sample_flat) : 0.0

    sim_means = vec(mean(simulated, dims=1))
    sim_sds = vec([length(sample_flat) > 1 ? std(simulated[:, d]) : 0.0 for d in 1:n_draws])

    pvalue_mean = sum(sim_means .>= obs_mean) / n_draws
    pvalue_sd = sum(sim_sds .>= obs_sd) / n_draws

    return ProteinPPC(protein_name, :regression, sample_flat, simulated, pvalue_mean, pvalue_sd, NaN)
end

# ============================================================================ #
# Beta-Bernoulli PPC
# ============================================================================ #

"""
    _ppc_betabernoulli(data, idx, n_controls, n_samples; n_draws, prior_alpha, prior_beta, rng) -> Union{BetaBernoulliPPC, Nothing}

Generate posterior predictive samples from the Beta-Bernoulli detection model.
"""
function _ppc_betabernoulli(
    data::InteractionData, idx::Int, n_controls::Int, n_samples::Int;
    n_draws::Int = 1000,
    prior_alpha::Float64 = 3.0,
    prior_beta::Float64 = 3.0,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    counts = count_detections(data, idx, n_samples, n_controls)
    k_s = counts.k_sample
    k_c = counts.k_control
    f_s = counts.f_sample
    f_c = counts.f_control

    (f_s < 0 || f_c < 0) && return nothing

    # Posterior Beta distributions
    θ_s_post = Beta(prior_alpha + k_s, prior_beta + f_s)
    θ_c_post = Beta(prior_alpha + k_c, prior_beta + f_c)

    n_total_s = k_s + f_s
    n_total_c = k_c + f_c

    protein_name = getProteinData(data, idx).name

    sim_k_s = Vector{Int}(undef, n_draws)
    sim_k_c = Vector{Int}(undef, n_draws)

    for d in 1:n_draws
        θ_s_draw = rand(rng, θ_s_post)
        θ_c_draw = rand(rng, θ_c_post)
        sim_k_s[d] = rand(rng, Binomial(n_total_s, θ_s_draw))
        sim_k_c[d] = rand(rng, Binomial(n_total_c, θ_c_draw))
    end

    # P-value for detection difference
    obs_diff = k_s - k_c
    sim_diffs = sim_k_s .- sim_k_c
    pvalue = sum(sim_diffs .>= obs_diff) / n_draws

    return BetaBernoulliPPC(protein_name, k_s, k_c, sim_k_s, sim_k_c, pvalue)
end

# ============================================================================ #
# Summary builder
# ============================================================================ #

function _build_diagnostics_summary(protein_ppcs::Vector{ProteinPPC}, bb_ppcs::Vector{BetaBernoulliPPC})
    names_vec = String[]
    model_vec = Symbol[]
    pval_mean_vec = Float64[]
    pval_sd_vec = Float64[]
    pval_log2fc_vec = Float64[]

    for ppc in protein_ppcs
        push!(names_vec, ppc.protein_name)
        push!(model_vec, ppc.model)
        push!(pval_mean_vec, ppc.pvalue_mean)
        push!(pval_sd_vec, ppc.pvalue_sd)
        push!(pval_log2fc_vec, ppc.pvalue_log2fc)
    end

    for bb in bb_ppcs
        push!(names_vec, bb.protein_name)
        push!(model_vec, :betabernoulli)
        push!(pval_mean_vec, bb.pvalue_detection_diff)
        push!(pval_sd_vec, NaN)
        push!(pval_log2fc_vec, NaN)
    end

    return DataFrame(
        Protein = names_vec,
        model = model_vec,
        pvalue_mean = pval_mean_vec,
        pvalue_sd = pval_sd_vec,
        pvalue_log2fc = pval_log2fc_vec
    )
end

# ============================================================================ #
# Extended PPC statistics
# ============================================================================ #

"""
    _compute_extended_ppc_stats(ppc::ProteinPPC) -> Union{PPCExtendedStatistics, Nothing}

Compute extended test statistics (skewness, kurtosis, IQR/SD ratio) p-values
for a single protein's PPC results.
"""
function _compute_extended_ppc_stats(ppc::ProteinPPC)
    observed = ppc.observed
    n_obs = length(observed)
    n_obs < 4 && return nothing
    n_draws = size(ppc.simulated, 2)
    n_draws == 0 && return nothing

    obs_sk = _compute_skewness(observed)
    obs_ku = _compute_kurtosis(observed)
    obs_sd = std(observed)
    obs_iqr = quantile(observed, 0.75) - quantile(observed, 0.25)
    obs_iqr_ratio = obs_sd > 1e-15 ? obs_iqr / obs_sd : NaN

    sim_sk = Float64[]
    sim_ku = Float64[]
    sim_iqr_ratio = Float64[]

    for d in 1:n_draws
        sim_col = ppc.simulated[:, d]
        push!(sim_sk, _compute_skewness(sim_col))
        push!(sim_ku, _compute_kurtosis(sim_col))
        s = std(sim_col)
        iqr = quantile(sim_col, 0.75) - quantile(sim_col, 0.25)
        push!(sim_iqr_ratio, s > 1e-15 ? iqr / s : NaN)
    end

    # Filter NaN from simulated stats for p-value computation
    valid_sk = filter(!isnan, sim_sk)
    valid_ku = filter(!isnan, sim_ku)
    valid_iqr = filter(!isnan, sim_iqr_ratio)

    pval_sk = isnan(obs_sk) || isempty(valid_sk) ? NaN : sum(abs.(valid_sk) .>= abs(obs_sk)) / length(valid_sk)
    pval_ku = isnan(obs_ku) || isempty(valid_ku) ? NaN : sum(abs.(valid_ku) .>= abs(obs_ku)) / length(valid_ku)
    pval_iqr = isnan(obs_iqr_ratio) || isempty(valid_iqr) ? NaN : sum(valid_iqr .>= obs_iqr_ratio) / length(valid_iqr)

    return PPCExtendedStatistics(ppc.protein_name, ppc.model, pval_sk, pval_ku, pval_iqr)
end

"""
    _ks_test_uniform(pvals::Vector{Float64}) -> Float64

Compute the Kolmogorov-Smirnov statistic for testing whether p-values are Uniform(0,1).

Returns the maximum absolute deviation between the empirical CDF and Uniform CDF.
Small values indicate p-values consistent with uniformity (well-calibrated model).
"""
function _ks_test_uniform(pvals::Vector{Float64})
    valid = filter(x -> !isnan(x) && isfinite(x), pvals)
    isempty(valid) && return NaN
    sorted = sort(valid)
    n = length(sorted)
    ecdf_vals = (1:n) ./ n
    return maximum(abs.(ecdf_vals .- sorted))
end

# ============================================================================ #
# Merge diagnostics into results DataFrame
# ============================================================================ #

"""
    _merge_diagnostics_to_results(results_df, dr; sensitivity=nothing) -> DataFrame

Left-join diagnostic and sensitivity columns onto a results DataFrame.

Merges:
- Protein flags from `DiagnosticsResult` (computed for ALL proteins)
- Extended PPC statistics (PPC subset only; `missing` for other proteins)
- Per-protein sensitivity summary (range, std across prior settings)
- Classification stability (fraction of settings exceeding thresholds)

# Arguments
- `results_df::DataFrame`: The final results DataFrame (must have a `:Protein` column)
- `dr::Union{DiagnosticsResult, Nothing}`: Completed diagnostics result (optional)

# Keywords
- `sensitivity::Union{SensitivityResult, Nothing}`: Sensitivity analysis result (optional)

# Returns
- `DataFrame`: Copy of `results_df` with diagnostic/sensitivity columns appended
"""
function _merge_diagnostics_to_results(
    results_df::DataFrame,
    dr::Union{DiagnosticsResult, Nothing} = nothing;
    sensitivity::Union{SensitivityResult, Nothing} = nothing
)
    merged = copy(results_df)

    # --- Diagnostics columns ---
    if !isnothing(dr)
        # Merge protein flags (computed for ALL proteins)
        if !isnothing(dr.protein_flags) && !isempty(dr.protein_flags)
            flag_df = DataFrame(
                Protein = [f.protein_name for f in dr.protein_flags],
                n_observations = [f.n_observations for f in dr.protein_flags],
                diagnostic_flag = [string(f.overall_flag) for f in dr.protein_flags],
                is_low_data = [f.is_low_data for f in dr.protein_flags],
                is_residual_outlier = [f.is_residual_outlier for f in dr.protein_flags],
                mean_residual = [f.mean_residual for f in dr.protein_flags],
                max_abs_residual = [f.max_abs_residual for f in dr.protein_flags]
            )
            merged = leftjoin(merged, flag_df, on = :Protein)
        end

        # Merge extended PPC stats (PPC subset only; missing for rest)
        if !isnothing(dr.ppc_extended) && !isempty(dr.ppc_extended)
            # Deduplicate: keep first entry per protein (HBM preferred over regression)
            seen = Set{String}()
            ext_protein = String[]
            ext_sk = Float64[]
            ext_ku = Float64[]
            ext_iqr = Float64[]
            for e in dr.ppc_extended
                e.protein_name in seen && continue
                push!(seen, e.protein_name)
                push!(ext_protein, e.protein_name)
                push!(ext_sk, e.pvalue_skewness)
                push!(ext_ku, e.pvalue_kurtosis)
                push!(ext_iqr, e.pvalue_iqr_ratio)
            end
            ext_df = DataFrame(
                Protein = ext_protein,
                ppc_pvalue_skewness = ext_sk,
                ppc_pvalue_kurtosis = ext_ku,
                ppc_pvalue_iqr_ratio = ext_iqr
            )
            merged = leftjoin(merged, ext_df, on = :Protein)
        end
    end

    # --- Sensitivity analysis columns (per-protein, ALL proteins) ---
    if !isnothing(sensitivity)
        # Merge summary stats (range, std across prior settings)
        if !isempty(sensitivity.summary)
            sens_cols = intersect(
                ["Protein", "std_posterior", "min_posterior", "max_posterior", "range"],
                names(sensitivity.summary)
            )
            if "Protein" in sens_cols && length(sens_cols) > 1
                sens_df = sensitivity.summary[:, sens_cols]
                # Rename to avoid ambiguity
                for col in names(sens_df)
                    col == "Protein" && continue
                    DataFrames.rename!(sens_df, col => "sensitivity_" * col)
                end
                merged = leftjoin(merged, sens_df, on = :Protein)
            end
        end

        # Merge classification stability
        if !isempty(sensitivity.classification_stability)
            stab_cols = intersect(
                ["Protein", "frac_P_gt_0_5", "frac_P_gt_0_8", "frac_P_gt_0_95",
                 "frac_q_lt_0_05", "frac_q_lt_0_01"],
                names(sensitivity.classification_stability)
            )
            if "Protein" in stab_cols && length(stab_cols) > 1
                stab_df = sensitivity.classification_stability[:, stab_cols]
                merged = leftjoin(merged, stab_df, on = :Protein)
            end
        end
    end

    return merged
end

# ============================================================================ #
# Report generation
# ============================================================================ #

"""
    generate_diagnostics_report(dr::DiagnosticsResult; kwargs...) -> String

Generate a Markdown report summarizing posterior predictive checks and model diagnostics.

# Keywords
- `filename::String`: Output file path (default: `"diagnostics_report.md"`)
- `title::String`: Report title
- `ppc_histogram_file::String`: Path to PPC histogram plot (for embedding)
- `qq_plot_file::String`: Path to HBM Q-Q plot (for embedding)
- `regression_qq_plot_file::String`: Path to Regression Q-Q plot (for embedding)
- `calibration_plot_file::String`: Path to single calibration plot (for embedding)
- `calibration_comparison_file::String`: Path to calibration comparison plot (preferred over single)
- `pit_histogram_file::String`: Path to PIT histogram plot (for embedding)
- `scale_location_hbm_file::String`: Path to HBM scale-location plot (for embedding)
- `scale_location_regression_file::String`: Path to regression scale-location plot (for embedding)
- `nu_optimization_file::String`: Path to ν optimization plot (for embedding)

# Returns
- `String`: The report content as a string
"""
function generate_diagnostics_report(
    dr::DiagnosticsResult;
    filename::String = "diagnostics_report.md",
    title::String = "Model Diagnostics Report",
    ppc_histogram_file::String = "",
    qq_plot_file::String = "",
    regression_qq_plot_file::String = "",
    calibration_plot_file::String = "",
    calibration_comparison_file::String = "",
    pit_histogram_file::String = "",
    scale_location_hbm_file::String = "",
    scale_location_regression_file::String = "",
    nu_optimization_file::String = ""
)
    io = IOBuffer()

    # Header
    println(io, "# $title")
    println(io, "Generated: $(dr.timestamp) | Package: BayesInteractomics")
    println(io)

    # Summary
    n_hbm = count(p -> p.model == :hbm, dr.protein_ppcs)
    n_reg = count(p -> p.model == :regression, dr.protein_ppcs)
    n_bb = length(dr.bb_ppcs)

    println(io, "## Summary")
    println(io)
    println(io, "| Metric | Value |")
    println(io, "|--------|-------|")
    println(io, "| PPC draws per protein | $(dr.config.n_ppc_draws) |")
    println(io, "| Proteins checked (HBM) | $n_hbm |")
    println(io, "| Proteins checked (Regression) | $n_reg |")
    println(io, "| Proteins checked (Beta-Bernoulli) | $n_bb |")
    println(io, "| Protein selection strategy | $(dr.config.ppc_protein_selection) |")
    println(io)

    # PPC p-value summary
    println(io, "## Posterior Predictive P-values")
    println(io)
    println(io, "Well-calibrated models produce approximately uniform p-value distributions.")
    println(io, "Extreme p-values (< 0.05 or > 0.95) indicate model misspecification.")
    println(io)

    for (model_label, model_sym) in [("HBM", :hbm), ("Regression", :regression)]
        ppcs = filter(p -> p.model == model_sym, dr.protein_ppcs)
        isempty(ppcs) && continue

        pvals_mean = [p.pvalue_mean for p in ppcs]
        pvals_sd = [p.pvalue_sd for p in ppcs]

        n_extreme_mean = count(p -> p < 0.05 || p > 0.95, pvals_mean)
        n_extreme_sd = count(p -> p < 0.05 || p > 0.95, pvals_sd)

        println(io, "### $model_label Model")
        println(io)
        println(io, "| Statistic | Mean p-value | Extreme (< 0.05 or > 0.95) |")
        println(io, "|-----------|-------------|---------------------------|")
        println(io, "| Mean | $(round(mean(pvals_mean), digits=4)) | $n_extreme_mean / $(length(pvals_mean)) ($(round(100*n_extreme_mean/length(pvals_mean), digits=1))%) |")
        println(io, "| Std Dev | $(round(mean(pvals_sd), digits=4)) | $n_extreme_sd / $(length(pvals_sd)) ($(round(100*n_extreme_sd/length(pvals_sd), digits=1))%) |")
        println(io)
    end

    # Beta-Bernoulli PPC
    if !isempty(dr.bb_ppcs)
        bb_pvals = [p.pvalue_detection_diff for p in dr.bb_ppcs]
        n_extreme_bb = count(p -> p < 0.05 || p > 0.95, bb_pvals)

        println(io, "### Beta-Bernoulli Detection Model")
        println(io)
        println(io, "| Metric | Value |")
        println(io, "|--------|-------|")
        println(io, "| Mean detection diff p-value | $(round(mean(bb_pvals), digits=4)) |")
        println(io, "| Extreme p-values | $n_extreme_bb / $(length(bb_pvals)) ($(round(100*n_extreme_bb/length(bb_pvals), digits=1))%) |")
        println(io)
    end

    # Embed PPC histogram
    if !isempty(ppc_histogram_file) && isfile(ppc_histogram_file)
        rel = _relative_plot_path(filename, ppc_histogram_file)
        println(io, "![PPC P-value Histogram]($rel)")
        println(io)
    end

    # Residuals — each model section gets its own Q-Q plot and scale-location plot
    for (label, res, plot_file, sl_file) in [
        ("HBM", dr.hbm_residuals, qq_plot_file, scale_location_hbm_file),
        ("Regression", dr.regression_residuals, regression_qq_plot_file, scale_location_regression_file)
    ]
        isnothing(res) && continue
        println(io, "## $label Residuals")
        println(io)
        println(io, "| Metric | Value |")
        println(io, "|--------|-------|")
        println(io, "| Proteins analyzed | $(length(res.protein_names)) |")
        println(io, "| Pooled residuals | $(length(res.pooled_residuals)) |")
        println(io, "| Skewness | $(round(res.skewness, digits=4)) |")
        println(io, "| Excess kurtosis | $(round(res.kurtosis, digits=4)) |")
        println(io, "| Outlier proteins (|mean r| > 2) | $(length(res.outlier_proteins)) |")
        println(io)

        if !isempty(res.outlier_proteins)
            println(io, "**Outlier proteins:** $(join(res.outlier_proteins, ", "))")
            println(io)
        end

        # Embed per-model Q-Q plot
        if !isempty(plot_file) && isfile(plot_file)
            rel = _relative_plot_path(filename, plot_file)
            println(io, "![$label Residual Q-Q Plot]($rel)")
            println(io)
        end

        # Embed per-model scale-location plot
        if !isempty(sl_file) && isfile(sl_file)
            rel = _relative_plot_path(filename, sl_file)
            println(io, "### Heteroscedasticity Check")
            println(io)
            println(io, "The scale-location plot shows √|standardized residuals| against fitted values.")
            println(io, "A flat red smoother indicates homoscedasticity (constant variance); an upward")
            println(io, "trend suggests variance increases with the predicted value.")
            println(io)
            println(io, "![$label Scale-Location Plot]($rel)")
            println(io)
        end
    end

    # Calibration
    _has_any_calibration = !isnothing(dr.calibration) || !isnothing(dr.calibration_relaxed) || !isnothing(dr.calibration_enrichment_only)

    if _has_any_calibration
        println(io, "## Calibration Assessment")
        println(io)
        println(io, "> **Note:** Without a gold-standard dataset, calibration is assessed using internal")
        println(io, "> consistency proxies based on individual Bayes factors. Three proxies are provided")
        println(io, "> to account for different interaction biology (constitutive vs dose-dependent).")
        println(io)
        println(io, "### Why calibration may deviate from the diagonal")
        println(io)
        println(io, "The calibration plot compares the model's predicted posterior probability against")
        println(io, "the observed fraction of proteins that are \"empirically positive\" in each bin.")
        println(io, "Several systematic factors can cause the observed rate to fall below the diagonal:")
        println(io)
        println(io, "1. **Conservative proxy for ground truth.** The strict proxy counts a protein as")
        println(io, "   empirically positive only when *all three* individual Bayes factors exceed 1.0.")
        println(io, "   The relaxed proxy (2-of-3) correctly classifies constitutive interactors that")
        println(io, "   have strong enrichment and detection but lack dose-response correlation.")
        println(io)
        println(io, "2. **Regression slope threshold (α > 0.3).** The regression Bayes factor tests")
        println(io, "   H₁: α > 0.3 (not α > 0). Proteins with real but modest positive slopes")
        println(io, "   (0 < α ≤ 0.3) receive BF_correlation < 1, even though they may be true")
        println(io, "   interactors. This conservative threshold systematically undercounts positives")
        println(io, "   in the strict proxy.")
        println(io)
        println(io, "3. **Constitutive interactors.** Proteins that interact regardless of bait")
        println(io, "   concentration (e.g., stable complex members) will not show dose-dependent")
        println(io, "   correlation. The strict proxy classifies them as negative despite strong")
        println(io, "   enrichment and detection evidence. The relaxed proxy (2-of-3) and the")
        println(io, "   enrichment-only proxy handle these correctly.")
        println(io)
        println(io, "4. **Self-referential limitation.** All internal proxies use the same Bayes")
        println(io, "   factors that feed into the posterior probability. This creates a circular")
        println(io, "   dependency: the observed rate reflects model agreement, not biological truth.")
        println(io, "   True calibration requires an external gold-standard dataset.")
        println(io)

        # Comparison table of ECE/MCE across proxies
        println(io, "### Calibration Proxy Comparison")
        println(io)
        println(io, "| Proxy | Definition | ECE | MCE |")
        println(io, "|-------|-----------|-----|-----|")
        if !isnothing(dr.calibration)
            println(io, "| Strict (3-of-3) | All 3 BFs > 1.0 | $(round(dr.calibration.ece, digits=4)) | $(round(dr.calibration.mce, digits=4)) |")
        end
        if !isnothing(dr.calibration_relaxed)
            println(io, "| Relaxed (2-of-3) | At least 2 BFs > 1.0 | $(round(dr.calibration_relaxed.ece, digits=4)) | $(round(dr.calibration_relaxed.mce, digits=4)) |")
        end
        if !isnothing(dr.calibration_enrichment_only)
            println(io, "| Enrichment-only | BF_enrichment > 3.0 | $(round(dr.calibration_enrichment_only.ece, digits=4)) | $(round(dr.calibration_enrichment_only.mce, digits=4)) |")
        end
        println(io)

        # Detailed bin tables for each proxy
        for (label, cal) in [
            ("Strict (all 3 BFs > 1.0)", dr.calibration),
            ("Relaxed (2-of-3 BFs > 1.0)", dr.calibration_relaxed),
            ("Enrichment-only (BF_enrichment > 3.0)", dr.calibration_enrichment_only)
        ]
            isnothing(cal) && continue
            println(io, "### $label")
            println(io)
            println(io, "| Metric | Value |")
            println(io, "|--------|-------|")
            println(io, "| Expected Calibration Error (ECE) | $(round(cal.ece, digits=4)) |")
            println(io, "| Maximum Calibration Error (MCE) | $(round(cal.mce, digits=4)) |")
            println(io, "| Number of bins | $(length(cal.bin_midpoints)) |")
            println(io)

            println(io, "| Bin Midpoint | Predicted | Observed | Count |")
            println(io, "|-------------|-----------|----------|-------|")
            for i in eachindex(cal.bin_midpoints)
                println(io, "| $(round(cal.bin_midpoints[i], digits=3)) | $(round(cal.predicted_rate[i], digits=4)) | $(round(cal.observed_rate[i], digits=4)) | $(cal.bin_counts[i]) |")
            end
            println(io)
        end
    end

    # Embed calibration plot (prefer comparison plot over single-line)
    if !isempty(calibration_comparison_file) && isfile(calibration_comparison_file)
        rel = _relative_plot_path(filename, calibration_comparison_file)
        println(io, "![Calibration Comparison]($rel)")
        println(io)
    elseif !isempty(calibration_plot_file) && isfile(calibration_plot_file)
        rel = _relative_plot_path(filename, calibration_plot_file)
        println(io, "![Calibration Plot]($rel)")
        println(io)
    end

    # Model Comparison & ν Optimization section
    _has_model_cmp = !isnothing(dr.model_comparison)
    _has_nu_opt = !isnothing(dr.nu_optimization)

    if _has_model_cmp || _has_nu_opt
        println(io, "## Model Comparison & ν Optimization")
        println(io)

        if _has_model_cmp
            mc = dr.model_comparison
            println(io, "### WAIC Comparison (Normal vs. Robust Regression)")
            println(io)
            println(io, "| Model | WAIC | lppd | p_WAIC | SE |")
            println(io, "|-------|------|------|--------|-----|")
            println(io, "| Normal | $(round(mc.normal_waic.waic, digits=1)) | $(round(mc.normal_waic.lppd, digits=1)) | $(round(mc.normal_waic.p_waic, digits=1)) | $(round(mc.normal_waic.se, digits=1)) |")
            if !isnothing(mc.robust_waic)
                println(io, "| Robust (Student-t) | $(round(mc.robust_waic.waic, digits=1)) | $(round(mc.robust_waic.lppd, digits=1)) | $(round(mc.robust_waic.p_waic, digits=1)) | $(round(mc.robust_waic.se, digits=1)) |")
            end
            println(io)
            delta_sign = mc.delta_waic > 0 ? "+" : ""
            println(io, "**ΔWAIC** (Normal − Robust) = $(delta_sign)$(round(mc.delta_waic, digits=1)) ± $(round(mc.delta_se, digits=1))")
            println(io)
            println(io, "**Preferred model:** $(mc.preferred_model == :robust ? "Robust (Student-t)" : "Normal")")
            println(io)
        end

        if _has_nu_opt
            nu = dr.nu_optimization
            println(io, "### Student-t ν Optimization")
            println(io)
            println(io, "| Metric | Value |")
            println(io, "|--------|-------|")
            println(io, "| Optimal ν | $(round(nu.optimal_nu, digits=3)) |")
            println(io, "| Search bounds | [$(nu.search_bounds[1]), $(nu.search_bounds[2])] |")
            println(io, "| WAIC at optimal ν | $(round(nu.optimal_waic.waic, digits=1)) |")
            println(io, "| Normal WAIC (baseline) | $(round(nu.normal_waic.waic, digits=1)) |")
            delta_sign = nu.delta_waic > 0 ? "+" : ""
            println(io, "| ΔWAIC (Normal − Robust) | $(delta_sign)$(round(nu.delta_waic, digits=1)) ± $(round(nu.delta_se, digits=1)) |")
            println(io, "| N evaluations | $(length(nu.nu_trace)) |")
            println(io)

            # Interpretation
            if nu.delta_waic > 2 * nu.delta_se && nu.delta_waic > 0
                println(io, "The robust Student-t model with ν = $(round(nu.optimal_nu, digits=2)) provides a")
                println(io, "significantly better fit than the Normal model (ΔWAIC > 2 SE), indicating")
                println(io, "the presence of outliers or heavy tails in the regression residuals.")
            elseif nu.delta_waic > 0
                println(io, "The robust model shows a modest improvement over Normal (ΔWAIC within 2 SE),")
                println(io, "suggesting mild heavy-tailed behaviour.")
            else
                println(io, "The Normal model fits as well or better than the robust model,")
                println(io, "suggesting no significant outlier contamination in the data.")
            end
            println(io)

            # Embed ν optimization plot
            if !isempty(nu_optimization_file) && isfile(nu_optimization_file)
                rel = _relative_plot_path(filename, nu_optimization_file)
                println(io, "![ν Optimization Plot]($rel)")
                println(io)
            end
        end
    end

    # PIT Histogram section (when enhanced residuals available)
    _has_pit = (!isnothing(dr.enhanced_hbm_residuals) && !isempty(dr.enhanced_hbm_residuals.pit_values)) ||
               (!isnothing(dr.enhanced_regression_residuals) && !isempty(dr.enhanced_regression_residuals.pit_values))
    if _has_pit
        println(io, "## PIT (Probability Integral Transform) Histogram")
        println(io)
        println(io, "PIT values are computed by averaging the CDF over posterior draws for each observation.")
        println(io, "A well-specified model produces PIT ∼ Uniform(0,1). Deviations indicate:")
        println(io, "- **U-shaped**: underdispersion (model variance too small)")
        println(io, "- **Inverse-U**: overdispersion (model variance too large)")
        println(io, "- **Skewed**: location bias (systematic over/under-prediction)")
        println(io)

        for (label, enh) in [("HBM", dr.enhanced_hbm_residuals), ("Regression", dr.enhanced_regression_residuals)]
            isnothing(enh) && continue
            isempty(enh.pit_values) && continue
            pit = enh.pit_values
            println(io, "### $label Model PIT Values")
            println(io)
            println(io, "| Metric | Value |")
            println(io, "|--------|-------|")
            println(io, "| N PIT values | $(length(pit)) |")
            println(io, "| Mean PIT | $(round(mean(pit), digits=4)) |")
            println(io, "| Std PIT | $(round(std(pit), digits=4)) |")
            ks = _ks_test_uniform(pit)
            println(io, "| KS statistic (vs Uniform) | $(round(ks, digits=4)) |")
            println(io)
        end

        # Embed PIT histogram
        if !isempty(pit_histogram_file) && isfile(pit_histogram_file)
            rel = _relative_plot_path(filename, pit_histogram_file)
            println(io, "![PIT Histogram]($rel)")
            println(io)
        end
    end

    # Extended PPC Statistics section
    if !isnothing(dr.ppc_extended) && !isempty(dr.ppc_extended)
        println(io, "## Extended PPC Statistics")
        println(io)
        println(io, "Additional test statistics beyond mean and standard deviation.")
        println(io, "Extreme p-values (< 0.05 or > 0.95) indicate the model fails to capture")
        println(io, "the corresponding distributional feature.")
        println(io)

        println(io, "| Protein | Model | p(skewness) | p(kurtosis) | p(IQR/SD) |")
        println(io, "|---------|-------|-------------|-------------|-----------|")
        n_show = min(dr.config.n_top_display, length(dr.ppc_extended))
        for ext in dr.ppc_extended[1:n_show]
            sk_str = isnan(ext.pvalue_skewness) ? "N/A" : string(round(ext.pvalue_skewness, digits=4))
            ku_str = isnan(ext.pvalue_kurtosis) ? "N/A" : string(round(ext.pvalue_kurtosis, digits=4))
            iqr_str = isnan(ext.pvalue_iqr_ratio) ? "N/A" : string(round(ext.pvalue_iqr_ratio, digits=4))
            println(io, "| $(ext.protein_name) | $(ext.model) | $sk_str | $ku_str | $iqr_str |")
        end
        println(io)

        # KS test on pooled p-values
        all_ext_pvals = Float64[]
        for ext in dr.ppc_extended
            !isnan(ext.pvalue_skewness) && push!(all_ext_pvals, ext.pvalue_skewness)
            !isnan(ext.pvalue_kurtosis) && push!(all_ext_pvals, ext.pvalue_kurtosis)
            !isnan(ext.pvalue_iqr_ratio) && push!(all_ext_pvals, ext.pvalue_iqr_ratio)
        end
        if !isempty(all_ext_pvals)
            ks_stat = _ks_test_uniform(all_ext_pvals)
            println(io, "**Pooled KS statistic** (extended p-values vs Uniform): $(round(ks_stat, digits=4))")
            println(io)
        end
    end

    # Per-Protein Diagnostic Flags section
    if !isnothing(dr.protein_flags) && !isempty(dr.protein_flags)
        flagged = filter(f -> f.overall_flag != :ok, dr.protein_flags)
        println(io, "## Per-Protein Diagnostic Flags")
        println(io)
        println(io, "| Metric | Value |")
        println(io, "|--------|-------|")
        println(io, "| Total proteins assessed | $(length(dr.protein_flags)) |")
        println(io, "| Flagged (warning or fail) | $(length(flagged)) |")
        n_fail = count(f -> f.overall_flag == :fail, dr.protein_flags)
        n_warn = count(f -> f.overall_flag == :warning, dr.protein_flags)
        println(io, "| Fail | $n_fail |")
        println(io, "| Warning | $n_warn |")
        println(io)

        if !isempty(flagged)
            n_show = min(dr.config.n_top_display, length(flagged))
            # Sort: fail first, then warning, by descending |mean residual|
            sort!(flagged, by=f -> (-Int(f.overall_flag == :fail), -abs(f.mean_residual)))
            println(io, "### Top Flagged Proteins")
            println(io)
            println(io, "| Protein | N obs | Mean Resid | Max |Resid| | Outlier | Low Data | Flag |")
            println(io, "|---------|-------|------------|--------------|---------|----------|------|")
            for f in flagged[1:n_show]
                mr_str = isnan(f.mean_residual) ? "N/A" : string(round(f.mean_residual, digits=3))
                mx_str = isnan(f.max_abs_residual) ? "N/A" : string(round(f.max_abs_residual, digits=3))
                println(io, "| $(f.protein_name) | $(f.n_observations) | $mr_str | $mx_str | $(f.is_residual_outlier) | $(f.is_low_data) | $(f.overall_flag) |")
            end
            println(io)
        end
    end

    content = String(take!(io))

    # Write to file
    open(filename, "w") do f
        write(f, content)
    end

    return content
end

# _relative_plot_path is defined in sensitivity.jl and reused here
