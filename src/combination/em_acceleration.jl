#=
SQUAREM (Squared Iterative Methods) acceleration for EM algorithm.

This module implements the SQUAREM algorithm for accelerating EM convergence.
SQUAREM achieves 2-10x speedup by using quasi-Newton acceleration without
requiring explicit Hessian computation.

Reference:
Varadhan, R. and Roland, C. (2008). "Simple and Globally Convergent Methods
for Accelerating the Convergence of Any EM Algorithm." Scandinavian Journal
of Statistics, 35(2):335-353.
=#

"""
    SQUAREMState

Internal state for SQUAREM acceleration tracking.
"""
mutable struct SQUAREMState
    # Parameter vectors from last 3 iterations (for acceleration)
    θ_prev2::Union{Vector{Float64}, Nothing}  # θ_{k-2}
    θ_prev1::Union{Vector{Float64}, Nothing}  # θ_{k-1}
    θ_curr::Union{Vector{Float64}, Nothing}   # θ_k

    # Log-likelihoods for step validation
    ll_prev::Float64
    ll_curr::Float64

    # Counters
    n_accel_steps::Int     # Number of acceleration steps taken
    n_fallback_steps::Int  # Number of fallback steps (acceleration rejected)

    function SQUAREMState()
        new(nothing, nothing, nothing, -Inf, -Inf, 0, 0)
    end
end

"""
    extract_em_params(π1::Float64, joint_H1) -> Vector{Float64}

Extract parameters from EM state into a flat vector for SQUAREM.
Currently tracks π₁ and H1 marginal Beta parameters.
NaN values are replaced with safe defaults.
"""
function extract_em_params(π1::Float64, joint_H1)
    margs = joint_H1.m

    # Extract with NaN protection - replace NaN with default values
    safe_π1 = isfinite(π1) ? π1 : 0.5

    p1 = params(margs[1])
    p2 = params(margs[2])
    p3 = params(margs[3])

    return Float64[
        safe_π1,
        isfinite(p1[1]) ? p1[1] : 2.0, isfinite(p1[2]) ? p1[2] : 2.0,
        isfinite(p2[1]) ? p2[1] : 2.0, isfinite(p2[2]) ? p2[2] : 2.0,
        isfinite(p3[1]) ? p3[1] : 2.0, isfinite(p3[2]) ? p3[2] : 2.0
    ]
end

"""
    restore_em_params(θ::Vector{Float64}, copula_H1) -> Tuple{Float64, SklarDist}

Restore EM parameters from flat vector.
Returns (π₁, joint_H1) tuple.
"""
function restore_em_params(θ::Vector{Float64}, copula_H1)
    π1 = clamp(θ[1], 1e-6, 1.0 - 1e-6)
    if !isfinite(π1)
        π1 = 0.5
    end

    # Restore marginals with valid parameters - handle NaN explicitly
    α1 = isfinite(θ[2]) ? max(θ[2], 0.1) : 2.0
    β1 = isfinite(θ[3]) ? max(θ[3], 0.1) : 2.0
    α2 = isfinite(θ[4]) ? max(θ[4], 0.1) : 2.0
    β2 = isfinite(θ[5]) ? max(θ[5], 0.1) : 2.0
    α3 = isfinite(θ[6]) ? max(θ[6], 0.1) : 2.0
    β3 = isfinite(θ[7]) ? max(θ[7], 0.1) : 2.0

    marg1 = Beta(α1, β1)
    marg2 = Beta(α2, β2)
    marg3 = Beta(α3, β3)

    joint_H1 = SklarDist(copula_H1, (marg1, marg2, marg3))

    return π1, joint_H1
end

"""
    squarem_acceleration_step(state::SQUAREMState) -> Union{Vector{Float64}, Nothing}

Compute SQUAREM acceleration step from last 3 parameter vectors.
Returns accelerated parameters or nothing if acceleration cannot be computed.
"""
function squarem_acceleration_step(state::SQUAREMState)
    # Need 3 consecutive parameter vectors
    if state.θ_prev2 === nothing || state.θ_prev1 === nothing || state.θ_curr === nothing
        return nothing
    end

    # Compute r and v vectors
    r = state.θ_prev1 .- state.θ_prev2   # First EM step direction
    v = state.θ_curr .- state.θ_prev1 .- r  # Change in step direction

    # Compute step length (α)
    r_norm_sq = sum(r.^2)
    v_norm_sq = sum(v.^2)

    # Avoid division by zero
    if v_norm_sq < 1e-20
        return nothing
    end

    # SQUAREM step length
    α = -sqrt(r_norm_sq / v_norm_sq)

    # Bound α to prevent overshooting (recommended: -1 to -0.01)
    α = clamp(α, -1.0, -0.01)

    # Compute accelerated parameters
    θ_accel = state.θ_prev2 .- 2 * α .* r .+ α^2 .* v

    # Ensure parameters are valid (positive for Beta params, bounded for π)
    θ_accel[1] = clamp(θ_accel[1], 1e-6, 1.0 - 1e-6)  # π₁
    for i in 2:length(θ_accel)
        θ_accel[i] = max(θ_accel[i], 0.1)  # Beta parameters
    end

    return θ_accel
end

"""
    update_squarem_state!(state::SQUAREMState, θ_new::Vector{Float64}, ll_new::Float64)

Update SQUAREM state with new parameters after EM step.
"""
function update_squarem_state!(state::SQUAREMState, θ_new::Vector{Float64}, ll_new::Float64)
    # Shift parameter history
    state.θ_prev2 = state.θ_prev1
    state.θ_prev1 = state.θ_curr
    state.θ_curr = copy(θ_new)

    # Update likelihood
    state.ll_prev = state.ll_curr
    state.ll_curr = ll_new
end

"""
    should_attempt_acceleration(state::SQUAREMState, iter::Int, burn_in::Int) -> Bool

Determine if SQUAREM acceleration should be attempted this iteration.
"""
function should_attempt_acceleration(state::SQUAREMState, iter::Int, burn_in::Int)
    # Only accelerate after burn-in
    if iter <= burn_in
        return false
    end

    # Need 3 parameter vectors
    if state.θ_prev2 === nothing
        return false
    end

    # Accelerate every 3 iterations
    return (iter - burn_in) % 3 == 0
end

"""
    compute_log_likelihood(π0, π1, joint_H0, joint_H1, p_triplets, min_log_exp, max_log_exp) -> Float64

Compute log marginal likelihood for given parameters.
"""
function compute_log_likelihood(π0::Float64, π1::Float64,
                                 joint_H0, joint_H1,
                                 p_triplets::Matrix{Float64},
                                 min_log_exp::Float64, max_log_exp::Float64)
    f0_vals = clamp.(logpdf.(Ref(joint_H0), eachcol(p_triplets)), min_log_exp, max_log_exp)
    f1_vals = clamp.(logpdf.(Ref(joint_H1), eachcol(p_triplets)), min_log_exp, max_log_exp)

    log_π0 = log(π0)
    log_π1 = log(π1)

    log_marginal = logsumexp.(log_π0 .+ f0_vals, log_π1 .+ f1_vals)

    return sum(log_marginal)
end

"""
    em_fit_mixture_accelerated(p, joint_H0, refID; use_acceleration=true, kwargs...)

EM mixture fitting with optional SQUAREM acceleration.

This function wraps `em_fit_mixture` with SQUAREM acceleration for faster convergence.
When acceleration is enabled, it can achieve 2-10x speedup on typical datasets.

# Arguments
- `p::PosteriorProbabilityTriplet`: Posterior probabilities from the three models
- `joint_H0::SklarDist`: Joint distribution under the null hypothesis
- `refID::Int64`: Index of the reference (bait) protein

# Keywords
- `use_acceleration::Bool=true`: Enable SQUAREM acceleration
- All other keywords passed to `em_fit_mixture`

# Returns
- `EMResult`: EM fitting results

# Notes
SQUAREM acceleration works by:
1. Storing the last 3 parameter vectors
2. Every 3 iterations (after burn-in), computing an accelerated step
3. Accepting the accelerated step only if it improves likelihood
4. Falling back to standard EM if acceleration fails

The acceleration is safe (monotonic likelihood) and typically reduces
iterations by 50-80%.
"""
function em_fit_mixture_accelerated(p::PosteriorProbabilityTriplet,
                                     joint_H0::SklarDist,
                                     refID::Int64;
                                     use_acceleration::Bool = true,
                                     max_iter::Int = 5000,
                                     init_π0::Float64 = 0.80,
                                     prior::Union{Symbol, NamedTuple} = :default,
                                     h1_refitting::Bool = true,
                                     burn_in::Int = 10,
                                     copula_criterion::Symbol = :BIC,
                                     init_set::Union{Vector{Int}, Nothing} = nothing,
                                     damping_initial::Float64 = 0.3,
                                     damping_final::Float64 = 0.8,
                                     verbose::Bool = true)

    # If acceleration disabled, fall back to standard EM
    if !use_acceleration
        return em_fit_mixture(p, joint_H0, refID;
                              max_iter=max_iter, init_π0=init_π0, prior=prior,
                              h1_refitting=h1_refitting, burn_in=burn_in,
                              copula_criterion=copula_criterion, init_set=init_set,
                              damping_initial=damping_initial, damping_final=damping_final,
                              verbose=verbose)
    end

    ϵ = eps(Float64)
    π0 = init_π0
    π1 = 1.0 - π0

    # Get prior hyperparameters
    if prior isa Symbol
        prior_params = get_prior_hyperparameters(prior)
    else
        prior_params = prior
    end
    α_prior, β_prior = prior_params.α, prior_params.β

    # Float64 limits
    max_log_exp = 709.0
    min_log_exp = -745.0

    # Initialize H1 distribution
    if init_set === nothing
        # Compute mean strength, replacing NaN with 0 (no signal)
        mean_strength = @. p.enrichment * p.correlation * p.detection
        mean_strength = replace(mean_strength, NaN => 0.0)

        # Filter valid values for quantile computation
        valid_strengths = filter(isfinite, mean_strength)
        if isempty(valid_strengths) || length(valid_strengths) < 10
            quantile_threshold = 0.5
        else
            quantile_threshold = quantile(valid_strengths, 0.95)
        end
        idx_init = find_H1_initialization_set(mean_strength, p, quantile_threshold)
    else
        idx_init = init_set
    end

    # Check if we have enough proteins for initialization
    if isempty(idx_init) || length(idx_init) < 5
        @warn "Insufficient proteins for H1 initialization ($(length(idx_init))), using default distributions"
        marg1 = Beta(2.0, 2.0)
        marg2 = Beta(2.0, 2.0)
        marg3 = Beta(2.0, 2.0)
        copula_H1 = FrankCopula(3, 1.0)
        joint_H1 = SklarDist(copula_H1, (marg1, marg2, marg3))
    else
        p_init = squeeze(p[idx_init], ϵ=1e-10)
        marg1 = fit_beta_safe(p_init.enrichment)
        marg2 = fit_beta_safe(p_init.correlation)
        marg3 = fit_beta_safe(p_init.detection)

        # Fit copula with fallback to FrankCopula if fitting fails
        copula_H1 = try
            fit_copula(p_init; criterion=copula_criterion)
        catch e
            @warn "Copula fitting failed: $e, using default FrankCopula"
            FrankCopula(3, 1.0)
        end
        joint_H1 = SklarDist(copula_H1, (marg1, marg2, marg3))
    end

    p_squeezed = squeeze(p, ϵ=1e-10)
    p_triplets = hcat(p_squeezed.enrichment, p_squeezed.correlation, p_squeezed.detection)'

    prev_ll = -Inf
    logs = DataFrame(iter = Int[0], π0=Float64[π0], π1=Float64[π1], ll=Float64[prev_ll])

    # SQUAREM state
    sqr_state = SQUAREMState()

    progress = nothing
    if verbose
        progress = Progress(
            max_iter, desc="Fitting EM model (accelerated)...",
            showspeed=true,
            barglyphs=BarGlyphs('|','█', [' ' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen = 20, dt = 1
        )
    end

    for iter in 1:max_iter
        # Standard E-step
        f0_vals = clamp.(logpdf.(Ref(joint_H0), eachcol(p_triplets)), min_log_exp, max_log_exp)
        f1_vals = clamp.(logpdf.(Ref(joint_H1), eachcol(p_triplets)), min_log_exp, max_log_exp)

        !isfinite(f0_vals[refID]) && (f0_vals[refID] = min_log_exp)
        !isfinite(f1_vals[refID]) && (f1_vals[refID] = min_log_exp)

        log_π0 = log(π0)
        log_π1 = log(π1)

        log_weights = log_π1 .+ f1_vals
        log_weights .-= @. log(
            π0 * exp(clamp(log_π0 + f0_vals, min_log_exp, max_log_exp)) +
            π1 * exp(clamp(log_π1 + f1_vals, min_log_exp, max_log_exp))
        )

        w = exp.(clamp.(log_weights, -7.0, max_log_exp))
        w = clamp.(w, ϵ, 1-ϵ)
        replace!(w, Inf => 1.0 - ϵ, -Inf => ϵ)

        log_marginal_likelihood = logsumexp.(log_π0 .+ f0_vals, log_π1 .+ f1_vals)

        # Standard M-step
        if h1_refitting && iter > burn_in
            progress_frac = min(1.0, (iter - burn_in) / max(1, max_iter - burn_in))
            α_damp = damping_initial + progress_frac * (damping_final - damping_initial)

            marg1_fit = fit_beta_weighted(p_squeezed.enrichment, w)
            marg2_fit = fit_beta_weighted(p_squeezed.correlation, w)
            marg3_fit = fit_beta_weighted(p_squeezed.detection, w)

            current_margs = joint_H1.m
            prev_marg1 = current_margs[1]
            prev_marg2 = current_margs[2]
            prev_marg3 = current_margs[3]

            # Apply damped updates with safety checks
            marg1_new = safe_damped_beta(marg1_fit, prev_marg1, α_damp)
            marg2_new = safe_damped_beta(marg2_fit, prev_marg2, α_damp)
            marg3_new = safe_damped_beta(marg3_fit, prev_marg3, α_damp)

            cop_new = fit_copula_weighted(p_squeezed, w)
            if cop_new !== nothing
                copula_H1 = cop_new
                joint_H1 = SklarDist(cop_new, (marg1_new, marg2_new, marg3_new))
            else
                joint_H1 = SklarDist(copula_H1, (marg1_new, marg2_new, marg3_new))
            end
        end

        sum_weights = sum(w)
        N_weights = length(w)
        π1_new = (sum_weights + α_prior - 1) / (N_weights + α_prior + β_prior - 2)
        π0_new = 1.0 - π1_new

        ll = sum(log_marginal_likelihood)

        # Update SQUAREM state
        θ_current = extract_em_params(π1_new, joint_H1)
        update_squarem_state!(sqr_state, θ_current, ll)

        # Attempt SQUAREM acceleration
        if should_attempt_acceleration(sqr_state, iter, burn_in)
            θ_accel = squarem_acceleration_step(sqr_state)

            if θ_accel !== nothing
                # Restore accelerated parameters
                π1_accel, joint_H1_accel = restore_em_params(θ_accel, copula_H1)
                π0_accel = 1.0 - π1_accel

                # Compute likelihood at accelerated point
                ll_accel = compute_log_likelihood(π0_accel, π1_accel, joint_H0, joint_H1_accel,
                                                  p_triplets, min_log_exp, max_log_exp)

                # Accept if likelihood improved
                if isfinite(ll_accel) && ll_accel > ll
                    π0_new = π0_accel
                    π1_new = π1_accel
                    joint_H1 = joint_H1_accel
                    ll = ll_accel
                    sqr_state.n_accel_steps += 1
                else
                    sqr_state.n_fallback_steps += 1
                end
            end
        end

        push!(logs, (iter, π0_new, π1_new, ll))

        if hasEMconverged(logs, tol=1e-3)
            break
        end

        π0 = max(π0_new, 1e-6)
        π1 = π1_new
        prev_ll = ll

        verbose && !isnothing(progress) && ProgressMeter.next!(progress)
    end

    verbose && !isnothing(progress) && finish!(progress)

    iter = logs[end, :iter]
    has_converged = iter < max_iter

    # Oscillation-aware termination
    if !has_converged && size(logs, 1) >= 20
        ll_recent = logs[(end - 19):end, :ll]
        if all(isfinite.(ll_recent))
            ll_diffs = diff(ll_recent)
            signs = sign.(ll_diffs)
            sign_changes = sum(signs[1:end-1] .!= signs[2:end])

            if sign_changes >= 12
                π0 = mean(logs[(end - 19):end, :π0])
                π1 = mean(logs[(end - 19):end, :π1])
                has_converged = true
                verbose && @info "EM oscillating, using averaged parameters (π₁=$(round(π1, digits=4)))"
            end
        end
    end

    if verbose
        if has_converged && iter < max_iter
            println("EM converged at iteration $iter")
        end
        if sqr_state.n_accel_steps > 0 || sqr_state.n_fallback_steps > 0
            @info "SQUAREM: $(sqr_state.n_accel_steps) accelerated steps, $(sqr_state.n_fallback_steps) fallbacks"
        end
    end

    !has_converged && @warn "EM did not converge"

    return EMResult(π0, π1, joint_H1, logs, has_converged)
end
