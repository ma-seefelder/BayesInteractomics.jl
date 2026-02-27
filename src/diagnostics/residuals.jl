# Standardized Residuals for HBM and Regression Models
# Computes residuals from posterior predictive distributions

"""
    _compute_hbm_residuals(data, hbm_results, name_to_idx) -> ResidualResult

Compute standardized residuals from the HBM model for all proteins with inference results.

For each observed value: `residual = (y - E[μ]) / sqrt(E[1/τ])`
where E[μ] is the posterior mean of the location parameter and E[1/τ] is the posterior
mean of the variance (computed from the Gamma posterior on precision τ).

Using E[1/τ] rather than 1/E[τ] correctly accounts for the posterior predictive variance
via Jensen's inequality: E[1/τ] ≥ 1/E[τ] for any non-degenerate distribution on τ.
For a Gamma(α, θ) posterior on τ, E[1/τ] = 1/(θ(α-1)) when α > 1.

Outlier proteins are those with |mean residual| > 2.
"""
function _compute_hbm_residuals(
    data::InteractionData,
    hbm_results::Dict{String, HBMResult},
    name_to_idx::Dict{String, Int}
)
    protein_names = collect(keys(hbm_results))
    all_residuals = Vector{Vector{Float64}}()
    all_fitted = Vector{Vector{Float64}}()
    mean_residuals = Float64[]

    for pname in protein_names
        data_idx = get(name_to_idx, pname, nothing)
        isnothing(data_idx) && continue

        hbm_result = hbm_results[pname]
        protein = getProteinData(data, data_idx)
        posterior = hbm_result.posterior

        # Extract observed sample data
        sample_mat = getSampleMatrix(protein)
        observed = Float64[]
        for val in sample_mat
            !ismissing(val) && push!(observed, val)
        end
        isempty(observed) && continue

        # Get posterior mean of location and predictive variance (global level, index 1)
        μ_posteriors = posterior.posteriors[:μ_sample]
        τ_posteriors = posterior.posteriors[:σ_sample]

        μ_mean = dist_mean(to_normal(μ_posteriors[1]))
        predictive_variance = _posterior_predictive_variance(τ_posteriors[1])

        # Standardized residuals using posterior predictive standard deviation
        resids = (observed .- μ_mean) ./ sqrt(predictive_variance)
        push!(all_residuals, resids)
        push!(all_fitted, fill(μ_mean, length(observed)))
        push!(mean_residuals, mean(resids))
    end

    pooled = reduce(vcat, all_residuals; init=Float64[])
    pooled_fitted = reduce(vcat, all_fitted; init=Float64[])

    sk = _compute_skewness(pooled)
    ku = _compute_kurtosis(pooled)

    outliers = protein_names[abs.(mean_residuals) .> 2.0]

    return ResidualResult(
        :hbm,
        protein_names,
        all_residuals,
        mean_residuals,
        pooled,
        pooled_fitted,
        sk,
        ku,
        outliers
    )
end

"""
    _compute_regression_residuals(data, regression_results, name_to_idx, refID) -> ResidualResult

Compute standardized residuals from the regression model.

For each observed value: `residual = (y - predicted) / sqrt(E[1/τ])`
where predicted = E[β] + E[α] * reference and E[1/τ] is the posterior mean
of the variance (from the Gamma posterior on precision τ).
"""
function _compute_regression_residuals(
    data::InteractionData,
    regression_results::Dict{String, <:AnyRegressionResult},
    name_to_idx::Dict{String, Int},
    refID::Int
)
    protein_names = collect(keys(regression_results))
    all_residuals = Vector{Vector{Float64}}()
    all_fitted = Vector{Vector{Float64}}()
    mean_residuals = Float64[]

    for pname in protein_names
        data_idx = get(name_to_idx, pname, nothing)
        isnothing(data_idx) && continue

        reg_result = regression_results[pname]
        posterior = reg_result.posterior

        # Prepare regression data
        reg_data = prepare_regression_data(data, data_idx, refID)
        sample_flat = Float64[]
        reference_flat = Float64[]

        for val in reg_data.sample
            !ismissing(val) && push!(sample_flat, val)
        end
        for val in reg_data.reference
            !ismissing(val) && push!(reference_flat, val)
        end

        n_obs = min(length(sample_flat), length(reference_flat))
        n_obs == 0 && continue
        sample_flat = sample_flat[1:n_obs]
        reference_flat = reference_flat[1:n_obs]

        # Extract posterior means
        is_single = reg_result isa RegressionResultSingleProtocol || reg_result isa RobustRegressionResultSingleProtocol
        if is_single
            α_mean = dist_mean(to_normal(posterior.posteriors[:α]))
            β_mean = dist_mean(to_normal(posterior.posteriors[:β]))
        else
            α_mean = dist_mean(to_normal(posterior.posteriors[:α][1]))
            β_mean = dist_mean(to_normal(posterior.posteriors[:β][1]))
        end
        # For robust models, use τ_base (Empirical Bayes) with Student-t variance inflation
        if reg_result isa RobustRegressionResult
            nu = reg_result.nu
            variance_inflation = nu > 2.0 ? nu / (nu - 2.0) : 2.0
            predictive_variance = (1.0 / reg_result.τ_base) * variance_inflation
        else
            predictive_variance = _posterior_predictive_variance(posterior.posteriors[:σ])
        end

        # Compute standardized residuals using posterior predictive standard deviation
        predicted = β_mean .+ α_mean .* reference_flat
        resids = (sample_flat .- predicted) ./ sqrt(predictive_variance)

        push!(all_residuals, resids)
        push!(all_fitted, predicted)
        push!(mean_residuals, mean(resids))
    end

    pooled = reduce(vcat, all_residuals; init=Float64[])
    pooled_fitted = reduce(vcat, all_fitted; init=Float64[])

    sk = _compute_skewness(pooled)
    ku = _compute_kurtosis(pooled)

    outliers = protein_names[abs.(mean_residuals) .> 2.0]

    return ResidualResult(
        :regression,
        protein_names,
        all_residuals,
        mean_residuals,
        pooled,
        pooled_fitted,
        sk,
        ku,
        outliers
    )
end

# ============================================================================ #
# Posterior predictive variance
# ============================================================================ #

"""
    _posterior_predictive_variance(τ_dist) -> Float64

Compute the posterior predictive variance E[1/τ] from a Gamma posterior on precision τ.

For a Gamma(α, θ) posterior (shape α, scale θ):
  - E[τ] = αθ
  - E[1/τ] = 1/(θ(α-1))  for α > 1

Using E[1/τ] instead of the naive 1/E[τ] corrects for Jensen's inequality bias,
which otherwise underestimates the predictive variance and inflates residual dispersion.

Falls back to 1/E[τ] when α ≤ 1 (the inverse mean does not exist).
"""
function _posterior_predictive_variance(τ_dist)
    α_post = Distributions.shape(τ_dist)
    θ_post = Distributions.scale(τ_dist)
    if α_post > 1.0 && θ_post > 0.0
        return 1.0 / (θ_post * (α_post - 1.0))
    else
        # Fallback: use 1/E[τ] when E[1/τ] is undefined
        τ_mean = dist_mean(τ_dist)
        return 1.0 / max(τ_mean, 1e-10)
    end
end

# ============================================================================ #
# Statistical helpers
# ============================================================================ #

function _compute_skewness(x::Vector{Float64})
    length(x) < 3 && return NaN
    μ = mean(x)
    s = std(x)
    s < 1e-15 && return NaN
    n = length(x)
    return (n / ((n-1)*(n-2))) * sum(((x .- μ) ./ s) .^ 3)
end

function _compute_kurtosis(x::Vector{Float64})
    length(x) < 4 && return NaN
    μ = mean(x)
    s = std(x)
    s < 1e-15 && return NaN
    n = length(x)
    k4 = sum(((x .- μ) ./ s) .^ 4) / n
    return k4 - 3.0  # Excess kurtosis
end

# ============================================================================ #
# Enhanced (randomized quantile) residuals
# ============================================================================ #

"""
    _compute_enhanced_hbm_residuals(data, hbm_results, name_to_idx; n_posterior_draws=200, rng) -> EnhancedResidualResult

Compute randomized quantile residuals from the HBM model.

For each protein and each observation yᵢ:
1. Draw `n_posterior_draws` samples (μ_d, τ_d) from the posterior
2. For each draw, compute F_d = cdf(Normal(μ_d, 1/√τ_d), yᵢ)
3. PIT value = mean(F_draws)
4. Quantile residual = Φ⁻¹(PIT value)

Well-specified models produce PIT ∼ Uniform(0,1) and quantile residuals ∼ N(0,1).
"""
function _compute_enhanced_hbm_residuals(
    data::InteractionData,
    hbm_results::Dict{String, HBMResult},
    name_to_idx::Dict{String, Int};
    n_posterior_draws::Int = 200,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    # First compute standard residuals
    base = _compute_hbm_residuals(data, hbm_results, name_to_idx)

    protein_names = collect(keys(hbm_results))
    all_quantile_residuals = Vector{Vector{Float64}}()
    all_pit_values = Float64[]

    for pname in protein_names
        data_idx = get(name_to_idx, pname, nothing)
        isnothing(data_idx) && continue

        hbm_result = hbm_results[pname]
        protein = getProteinData(data, data_idx)
        posterior = hbm_result.posterior

        # Extract observed sample data
        sample_mat = getSampleMatrix(protein)
        observed = Float64[]
        for val in sample_mat
            !ismissing(val) && push!(observed, val)
        end
        isempty(observed) && continue

        μ_dist = to_normal(posterior.posteriors[:μ_sample][1])
        τ_dist = posterior.posteriors[:σ_sample][1]

        protein_qresids = Float64[]
        for y in observed
            # Average CDF over posterior draws
            pit_sum = 0.0
            for _ in 1:n_posterior_draws
                μ_draw = rand(rng, μ_dist)
                τ_draw = max(rand(rng, τ_dist), 1e-10)
                sd_draw = 1.0 / sqrt(τ_draw)
                pit_sum += Distributions.cdf(Normal(μ_draw, sd_draw), y)
            end
            pit_val = pit_sum / n_posterior_draws
            push!(all_pit_values, pit_val)
            qresid = quantile(Normal(), clamp(pit_val, 1e-8, 1.0 - 1e-8))
            push!(protein_qresids, qresid)
        end
        push!(all_quantile_residuals, protein_qresids)
    end

    return EnhancedResidualResult(base, all_quantile_residuals, all_pit_values)
end

"""
    _compute_enhanced_regression_residuals(data, regression_results, name_to_idx, refID; n_posterior_draws=200, rng) -> EnhancedResidualResult

Compute randomized quantile residuals from the regression model.

For each protein and each observation yᵢ:
1. Draw (α_d, β_d, τ_d) from posterior
2. predicted_d = β_d + α_d * reference_i
3. F_d = cdf(Normal(predicted_d, 1/√τ_d), yᵢ)
4. PIT = mean(F_draws), quantile residual = Φ⁻¹(PIT)
"""
function _compute_enhanced_regression_residuals(
    data::InteractionData,
    regression_results::Dict{String, <:AnyRegressionResult},
    name_to_idx::Dict{String, Int},
    refID::Int;
    n_posterior_draws::Int = 200,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    # First compute standard residuals
    base = _compute_regression_residuals(data, regression_results, name_to_idx, refID)

    protein_names = collect(keys(regression_results))
    all_quantile_residuals = Vector{Vector{Float64}}()
    all_pit_values = Float64[]

    for pname in protein_names
        data_idx = get(name_to_idx, pname, nothing)
        isnothing(data_idx) && continue

        reg_result = regression_results[pname]
        posterior = reg_result.posterior

        # Prepare regression data
        reg_data = prepare_regression_data(data, data_idx, refID)
        sample_flat = Float64[]
        reference_flat = Float64[]

        for val in reg_data.sample
            !ismissing(val) && push!(sample_flat, val)
        end
        for val in reg_data.reference
            !ismissing(val) && push!(reference_flat, val)
        end

        n_obs = min(length(sample_flat), length(reference_flat))
        n_obs == 0 && continue
        sample_flat = sample_flat[1:n_obs]
        reference_flat = reference_flat[1:n_obs]

        # Extract posterior distributions
        is_single = reg_result isa RegressionResultSingleProtocol || reg_result isa RobustRegressionResultSingleProtocol
        if is_single
            α_dist = to_normal(posterior.posteriors[:α])
            β_dist = to_normal(posterior.posteriors[:β])
        else
            α_dist = to_normal(posterior.posteriors[:α][1])
            β_dist = to_normal(posterior.posteriors[:β][1])
        end
        # For robust models, use Student-t CDF; for normal, draw from σ posterior
        is_robust = reg_result isa RobustRegressionResult
        if is_robust
            nu = reg_result.nu
            sqrt_tau_base = sqrt(reg_result.τ_base)
            tdist = TDist(nu)
        else
            σ_dist = posterior.posteriors[:σ]
        end

        protein_qresids = Float64[]
        for (i, y) in enumerate(sample_flat)
            pit_sum = 0.0
            for _ in 1:n_posterior_draws
                α_draw = rand(rng, α_dist)
                β_draw = rand(rng, β_dist)
                predicted = β_draw + α_draw * reference_flat[i]
                if is_robust
                    # PIT via Student-t CDF: z = (y - μ) * √τ_base
                    z = (y - predicted) * sqrt_tau_base
                    pit_sum += Distributions.cdf(tdist, z)
                else
                    τ_draw = max(rand(rng, σ_dist), 1e-10)
                    sd_draw = 1.0 / sqrt(τ_draw)
                    pit_sum += Distributions.cdf(Normal(predicted, sd_draw), y)
                end
            end
            pit_val = pit_sum / n_posterior_draws
            push!(all_pit_values, pit_val)
            qresid = quantile(Normal(), clamp(pit_val, 1e-8, 1.0 - 1e-8))
            push!(protein_qresids, qresid)
        end
        push!(all_quantile_residuals, protein_qresids)
    end

    return EnhancedResidualResult(base, all_quantile_residuals, all_pit_values)
end

# ============================================================================ #
# Per-protein diagnostic flags
# ============================================================================ #

"""
    _compute_protein_flags(data, protein_names, name_to_idx; ppc_residuals=nothing, progress=nothing) -> Vector{ProteinDiagnosticFlag}

Compute diagnostic flags for ALL proteins using raw data for observation counts,
and optional PPC residuals for residual-based metrics.

For every protein: `n_observations` is counted from the raw sample data and
`is_low_data` is set when `n_observations < 4`.

For the PPC subset (proteins present in `ppc_residuals`): `mean_residual`,
`max_abs_residual`, and `is_residual_outlier` are computed from residuals.
For all other proteins these are `NaN` / `false`.

# Arguments
- `data::InteractionData`: Raw data used in the analysis
- `protein_names::Vector{String}`: Full list of protein identifiers (all proteins)
- `name_to_idx::Dict{String, Int}`: Map from protein name to data index

# Keywords
- `ppc_residuals::Union{ResidualResult, Nothing}`: Residuals from PPC subset (optional)
- `progress::Union{Progress, Nothing}`: Optional ProgressMeter progress bar to advance per protein
"""
function _compute_protein_flags(
    data::InteractionData,
    protein_names::Vector{String},
    name_to_idx::Dict{String, Int};
    ppc_residuals::Union{ResidualResult, Nothing} = nothing,
    progress::Union{Progress, Nothing} = nothing
)
    # Build lookup from PPC residuals: protein_name -> (residuals_vector,)
    ppc_lookup = Dict{String, Vector{Float64}}()
    if !isnothing(ppc_residuals)
        for (i, pname) in enumerate(ppc_residuals.protein_names)
            i > length(ppc_residuals.residuals) && break
            ppc_lookup[pname] = ppc_residuals.residuals[i]
        end
    end

    flags = ProteinDiagnosticFlag[]
    for pname in protein_names
        # Count non-missing sample observations from raw data
        data_idx = get(name_to_idx, pname, nothing)
        if !isnothing(data_idx)
            protein = getProteinData(data, data_idx)
            sample_mat = getSampleMatrix(protein)
            n_obs = count(v -> !ismissing(v), sample_mat)
        else
            n_obs = 0
        end

        is_low = n_obs < 4

        # Residual-based metrics: only available for PPC subset
        if haskey(ppc_lookup, pname)
            resids = ppc_lookup[pname]
            n_resid = length(resids)
            mean_r = n_resid > 0 ? mean(resids) : NaN
            max_abs_r = n_resid > 0 ? maximum(abs.(resids)) : NaN
            is_outlier = !isnan(mean_r) && abs(mean_r) > 2.0
        else
            mean_r = NaN
            max_abs_r = NaN
            is_outlier = false
        end

        flag = if is_outlier && is_low
            :fail
        elseif is_outlier || is_low
            :warning
        else
            :ok
        end

        push!(flags, ProteinDiagnosticFlag(pname, n_obs, mean_r, max_abs_r, is_outlier, is_low, flag))
        !isnothing(progress) && ProgressMeter.next!(progress)
    end
    return flags
end
