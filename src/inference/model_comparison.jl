# WAIC (Widely Applicable Information Criterion) for regression model comparison
# Computes WAIC from VMP posteriors following Gelman et al. (2013) and Watanabe (2010)

using LogExpFunctions: logsumexp
import Distributions: logpdf
using Optim: Optim
import Random: rand!, MersenneTwister

"""
    compute_waic(data, regression_results, name_to_idx, refID; n_draws=1000, rng=GLOBAL_RNG) -> WAICResult

Compute WAIC from VMP posteriors for the regression model.

WAIC = -2 * (lppd - p_waic), where:
- lppd = Σᵢ log(1/S × Σₛ p(yᵢ | θˢ)) is the log pointwise predictive density
- p_waic = Σᵢ var_s(log p(yᵢ | θˢ)) is the effective number of parameters

For each observation yᵢ with reference xᵢ, draws S samples from q(α), q(β), q(τ)
[and q(w) for robust model] and computes the pointwise log-likelihood.

# Arguments
- `data::InteractionData`: The interaction data
- `regression_results::Dict{String, <:Union{RegressionResult, RobustRegressionResult}}`: Per-protein regression results
- `name_to_idx::Dict{String, Int}`: Map from protein name to data index
- `refID::Int`: Reference protein index

# Keywords
- `n_draws::Int=1000`: Number of posterior draws for WAIC estimation
- `rng::AbstractRNG=GLOBAL_RNG`: Random number generator

# Returns
- `WAICResult`: WAIC statistics
"""
function compute_waic(
    data::InteractionData,
    regression_results::Dict{String, T},
    name_to_idx::Dict{String, Int},
    refID::Int;
    n_draws::Int = 1000,
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {T <: Union{RegressionResult, RobustRegressionResult}}

    lppd_total = 0.0
    p_waic_total = 0.0
    pointwise_waic = Float64[]
    log_n_draws = log(n_draws)

    # Pre-allocate reusable buffers (avoids per-observation allocation)
    log_liks = Vector{Float64}(undef, n_draws)
    α_draws = Vector{Float64}(undef, n_draws)
    β_draws = Vector{Float64}(undef, n_draws)
    τ_draws = Vector{Float64}(undef, n_draws)

    for (pname, reg_result) in regression_results
        data_idx = get(name_to_idx, pname, nothing)
        isnothing(data_idx) && continue

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

        posterior = reg_result.posterior

        # Extract posterior distributions for α, β, τ
        is_single = reg_result isa RegressionResultSingleProtocol || reg_result isa RobustRegressionResultSingleProtocol
        if is_single
            α_dist = to_normal(posterior.posteriors[:α])
            β_dist = to_normal(posterior.posteriors[:β])
        else
            α_dist = to_normal(posterior.posteriors[:α][1])
            β_dist = to_normal(posterior.posteriors[:β][1])
        end
        # For robust models, use τ_base (Empirical Bayes constant) instead of posterior σ
        is_robust = reg_result isa RobustRegressionResult
        if is_robust
            nu = reg_result.nu
            τ_base_val = reg_result.τ_base
        else
            σ_dist = posterior.posteriors[:σ]  # Gamma on precision τ
        end

        # Draw S samples from posteriors (reuse pre-allocated buffers)
        rand!(rng, α_dist, α_draws)
        rand!(rng, β_dist, β_draws)
        if !is_robust
            rand!(rng, σ_dist, τ_draws)
            @inbounds for j in 1:n_draws
                τ_draws[j] = max(τ_draws[j], 1e-10)
            end
        end

        # Precompute Student-t constants for robust model (avoids loggamma per draw)
        if is_robust
            sqrt_tau_base = sqrt(τ_base_val)
            # log_norm = loggamma((ν+1)/2) - loggamma(ν/2) - 0.5*log(ν*π), i.e. logpdf(TDist(ν), 0)
            log_norm = logpdf(TDist(nu), 0.0)
            neg_exp = -(nu + 1) / 2
            inv_nu = 1.0 / nu
            base_const = log_norm + 0.5 * log(τ_base_val)
        end

        for i in 1:n_obs
            y = sample_flat[i]
            x = reference_flat[i]

            @inbounds for s in 1:n_draws
                predicted = β_draws[s] + α_draws[s] * x
                if is_robust
                    z = (y - predicted) * sqrt_tau_base
                    log_liks[s] = base_const + neg_exp * log1p(z * z * inv_nu)
                else
                    variance = 1.0 / τ_draws[s]
                    sd = sqrt(max(variance, 1e-15))
                    log_liks[s] = logpdf(Normal(predicted, sd), y)
                end
            end

            # Compute pointwise lppd and p_waic immediately (no intermediate storage)
            lppd_i = logsumexp(log_liks) - log_n_draws
            p_waic_i = var(log_liks)
            lppd_total += lppd_i
            p_waic_total += p_waic_i
            push!(pointwise_waic, -2.0 * (lppd_i - p_waic_i))
        end
    end

    n_total = length(pointwise_waic)
    n_total == 0 && return WAICResult(NaN, NaN, NaN, Float64[], NaN)

    waic = -2.0 * (lppd_total - p_waic_total)

    # Standard error of WAIC (Vehtari et al. 2017)
    se = sqrt(n_total * var(pointwise_waic))

    return WAICResult(waic, lppd_total, p_waic_total, pointwise_waic, se)
end


"""
    compare_regression_models(data, normal_results, robust_results, name_to_idx, refID; n_draws=1000, rng=GLOBAL_RNG) -> ModelComparisonResult

Compare Normal vs. robust (Student-t) regression models via WAIC.

# Arguments
- `data::InteractionData`: The interaction data
- `normal_results::Dict{String, <:RegressionResult}`: Results from Normal regression
- `robust_results::Dict{String, <:RobustRegressionResult}`: Results from robust regression
- `name_to_idx::Dict{String, Int}`: Map from protein name to data index
- `refID::Int`: Reference protein index

# Keywords
- `n_draws::Int=1000`: Number of posterior draws for WAIC estimation
- `rng::AbstractRNG=GLOBAL_RNG`: Random number generator

# Returns
- `ModelComparisonResult`: WAIC comparison with preferred model indication
"""
function compare_regression_models(
    data::InteractionData,
    normal_results::Dict{String, T1},
    robust_results::Dict{String, T2},
    name_to_idx::Dict{String, Int},
    refID::Int;
    n_draws::Int = 1000,
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {T1 <: RegressionResult, T2 <: RobustRegressionResult}

    @info "Computing WAIC for Normal regression model..."
    normal_waic = compute_waic(data, normal_results, name_to_idx, refID; n_draws=n_draws, rng=rng)

    @info "Computing WAIC for Robust regression model..."
    robust_waic = compute_waic(data, robust_results, name_to_idx, refID; n_draws=n_draws, rng=rng)

    # Compute WAIC difference: positive delta means robust is better (lower WAIC)
    delta_waic = normal_waic.waic - robust_waic.waic

    # SE of difference (pointwise differences)
    n_normal = length(normal_waic.pointwise_waic)
    n_robust = length(robust_waic.pointwise_waic)
    if n_normal == n_robust && n_normal > 0
        pw_diff = normal_waic.pointwise_waic .- robust_waic.pointwise_waic
        delta_se = sqrt(n_normal * var(pw_diff))
    else
        delta_se = sqrt(normal_waic.se^2 + robust_waic.se^2)
    end

    preferred = delta_waic > 0 ? :robust : :normal

    return ModelComparisonResult(normal_waic, robust_waic, delta_waic, delta_se, preferred)
end


"""
    _fit_robust_and_compute_waic(data, refID, μ_0, σ_0, nu, τ_base; verbose=false) -> WAICResult

Fit robust (Student-t) regression for all proteins at a given ν and compute WAIC.
This is a reusable helper for both `_run_model_comparison` and `optimize_nu`.
"""
function _fit_robust_and_compute_waic(
    data::InteractionData, refID::Int,
    μ_0::Float64, σ_0::Float64,
    nu::Float64, τ_base::Float64;
    verbose::Bool = false,
    progress::Union{Progress, Nothing} = nothing,
    waic_rng::AbstractRNG = Random.GLOBAL_RNG
)
    n_proteins = length(getIDs(data))
    protein_names = getNames(data)

    # Precompute robust prior for this ν
    if getNoProtocols(data) == 1
        robust_prior = precompute_regression_one_protocol_robust_prior(data, refID, μ_0, σ_0; nu=nu, τ_base=τ_base)
    else
        robust_prior = precompute_regression_multi_protocol_robust_prior(data, refID, μ_0, σ_0; nu=nu, τ_base=τ_base)
    end

    # Build name → idx mapping
    name_to_idx = Dict(protein_names[i] => i for i in 1:n_proteins)

    # Fit robust model for all proteins (thread-safe)
    robust_thread_results = [Dict{String, RobustRegressionResult}() for _ in 1:Threads.nthreads()]

    Threads.@threads for i in 1:n_proteins
        tid = Threads.threadid()
        pname = protein_names[i]
        try
            if getNoProtocols(data) == 1
                rr = RegressionModel_one_protocol_robust(data, i, refID, μ_0, σ_0; nu=nu, τ_base=τ_base, cached_prior=robust_prior)
                robust_thread_results[tid][pname] = rr
            else
                rr = RegressionModelRobust(data, i, refID, μ_0, σ_0; nu=nu, τ_base=τ_base, cached_prior=robust_prior)
                robust_thread_results[tid][pname] = rr
            end
        catch e
            verbose && @warn "Robust regression (ν=$nu) failed for protein $i ($pname): $e"
        end
        !isnothing(progress) && ProgressMeter.next!(progress)
    end

    # Merge thread-local results
    robust_results = Dict{String, RobustRegressionResult}()
    for tid in 1:Threads.nthreads()
        merge!(robust_results, robust_thread_results[tid])
    end

    return compute_waic(data, robust_results, name_to_idx, refID; rng=waic_rng)
end


"""
    _fit_normal_and_compute_waic(data, refID, μ_0, σ_0; verbose=false) -> WAICResult

Fit Normal regression for all proteins and compute WAIC.
"""
function _fit_normal_and_compute_waic(
    data::InteractionData, refID::Int,
    μ_0::Float64, σ_0::Float64;
    verbose::Bool = false,
    progress::Union{Progress, Nothing} = nothing,
    waic_rng::AbstractRNG = Random.GLOBAL_RNG
)
    n_proteins = length(getIDs(data))
    protein_names = getNames(data)

    # Precompute normal prior
    if getNoProtocols(data) == 1
        normal_prior = precompute_regression_one_protocol_prior(data, refID, μ_0, σ_0)
    else
        normal_prior = precompute_regression_multi_protocol_prior(data, refID, μ_0, σ_0)
    end

    # Build name → idx mapping
    name_to_idx = Dict(protein_names[i] => i for i in 1:n_proteins)

    # Fit normal model for all proteins (thread-safe)
    normal_thread_results = [Dict{String, RegressionResult}() for _ in 1:Threads.nthreads()]

    Threads.@threads for i in 1:n_proteins
        tid = Threads.threadid()
        pname = protein_names[i]
        try
            if getNoProtocols(data) == 1
                nr = RegressionModel_one_protocol(data, i, refID, μ_0, σ_0; cached_prior=normal_prior)
                normal_thread_results[tid][pname] = nr
            else
                nr = RegressionModel(data, i, refID, μ_0, σ_0; cached_prior=normal_prior)
                normal_thread_results[tid][pname] = nr
            end
        catch e
            verbose && @warn "Normal regression failed for protein $i ($pname): $e"
        end
        !isnothing(progress) && ProgressMeter.next!(progress)
    end

    # Merge thread-local results
    normal_results = Dict{String, RegressionResult}()
    for tid in 1:Threads.nthreads()
        merge!(normal_results, normal_thread_results[tid])
    end

    return compute_waic(data, normal_results, name_to_idx, refID; rng=waic_rng)
end


"""
    optimize_nu(data::InteractionData, config; lower=3.0, upper=50.0) -> NuOptimizationResult

Optimize the Student-t degrees-of-freedom parameter ν over [lower, upper] using Brent's method,
minimizing WAIC for the robust regression model.

The Normal regression WAIC is computed once as a baseline for comparison.

# Arguments
- `data::InteractionData`: The interaction data
- `config`: Configuration object with fields `refID::Int` and `verbose::Bool` (e.g., `CONFIG`)

# Keywords
- `lower::Float64=3.0`: Lower bound for ν search
- `upper::Float64=50.0`: Upper bound for ν search

# Returns
- `NuOptimizationResult`: Optimization results including optimal ν, WAIC trace, and comparison
"""
function optimize_nu(
    data::InteractionData, config;
    lower::Float64 = 3.0, upper::Float64 = 50.0
)
    refID = config.refID
    verbose = config.verbose
    n_proteins = length(getIDs(data))

    # Compute hyperparameters (ν-independent)
    μ_0, σ_0 = μ0(data)
    τ_base = estimate_regression_tau_base(data, refID)
    @info "  ν-optimization: τ_base = $(round(τ_base, digits=4)), μ_0 = $(round(μ_0, digits=4)), σ_0 = $(round(σ_0, digits=4))"

    # Fixed-seed RNG for deterministic WAIC → smooth objective for Brent
    waic_seed = 42

    # Compute Normal WAIC once (baseline), with fixed RNG for fair comparison
    normal_prog = Progress(n_proteins; desc="  ν-opt: Normal baseline ", showspeed=true)
    normal_waic = _fit_normal_and_compute_waic(data, refID, μ_0, σ_0; verbose=verbose, progress=normal_prog, waic_rng=MersenneTwister(waic_seed))
    finish!(normal_prog)
    @info "  Normal WAIC = $(round(normal_waic.waic, digits=1))"

    # Trace vectors to record evaluations
    nu_trace = Float64[]
    waic_trace = Float64[]
    eval_count = Threads.Atomic{Int}(0)

    # Cache last WAICResult to avoid redundant final recomputation
    last_waic_result = Ref(WAICResult(NaN, NaN, NaN, Float64[], NaN))

    # Objective: minimize WAIC as a function of ν
    function f(nu::Float64)
        n_eval = Threads.atomic_add!(eval_count, 1) + 1
        # Fresh fixed-seed RNG per evaluation → deterministic, noise-free objective
        waic_result = _fit_robust_and_compute_waic(data, refID, μ_0, σ_0, nu, τ_base;
                                                    verbose=verbose, waic_rng=MersenneTwister(waic_seed))
        waic_val = waic_result.waic
        last_waic_result[] = waic_result

        push!(nu_trace, nu)
        push!(waic_trace, waic_val)

        @info "  ν = $(round(nu, digits=3)) → WAIC = $(round(waic_val, digits=1))  [eval $n_eval]"
        return waic_val
    end

    # Run Brent's method with coarser tolerance (ν precision beyond 0.5 is meaningless given WAIC uncertainty)
    @info "  Optimizing ν over [$lower, $upper] via Brent's method..."
    opt_result = Optim.optimize(f, lower, upper, Optim.Brent(); abs_tol=0.5)
    optimal_nu = Optim.minimizer(opt_result)
    @info "  Brent converged: optimal ν = $(round(optimal_nu, digits=3)) after $(eval_count[]) evaluations"

    # Use cached result from last Brent evaluation (avoids redundant recomputation)
    optimal_waic = last_waic_result[]

    # Compute ΔWAIC and SE
    delta_waic = normal_waic.waic - optimal_waic.waic
    n_normal = length(normal_waic.pointwise_waic)
    n_robust = length(optimal_waic.pointwise_waic)
    if n_normal == n_robust && n_normal > 0
        pw_diff = normal_waic.pointwise_waic .- optimal_waic.pointwise_waic
        delta_se = sqrt(n_normal * var(pw_diff))
    else
        delta_se = sqrt(normal_waic.se^2 + optimal_waic.se^2)
    end

    return NuOptimizationResult(
        optimal_nu, optimal_waic, normal_waic,
        nu_trace, waic_trace,
        delta_waic, delta_se,
        (lower, upper)
    )
end
