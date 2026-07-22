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
    extract_em_params_logbf(pi_H0, pi_ag, pi_H1, joint_H0, joint_ag, joint_H1) -> Vector{Float64}

Extract parameters from 3-component log-BF EM state into a flat vector for SQUAREM.

The parameter vector layout (21 elements):
- [1:3]   Mixing weights: pi_H0, pi_ag, pi_H1
- [4:9]   H0 marginals: mu_e, sigma_e, mu_c, sigma_c, mu_d, sigma_d
- [10:15] Agnostic marginals: mu_e, sigma_e, mu_c, sigma_c, mu_d, sigma_d
- [16:21] H1 marginals: mu_e, sigma_e, mu_c, sigma_c, mu_d, sigma_d

NaN values are replaced with safe defaults.
"""
function extract_em_params_logbf(pi_H0::Float64, pi_ag::Float64, pi_H1::Float64,
                                  joint_H0, joint_ag, joint_H1)
    theta = Float64[]
    push!(theta, isfinite(pi_H0) ? pi_H0 : 0.7,
                 isfinite(pi_ag) ? pi_ag : 0.15,
                 isfinite(pi_H1) ? pi_H1 : 0.15)
    for joint in (joint_H0, joint_ag, joint_H1)
        for marg in joint.m  # SklarDist marginals
            if marg isa LocationShifted
                # H1 enrichment: extract 2 params from LocationShifted{T}
                # Supports Gamma, LogNormal, Weibull (all have 2-param parameterizations)
                p = params(marg.dist)
                push!(theta, isfinite(p[1]) ? p[1] : 2.0,
                             isfinite(p[2]) ? p[2] : 2.0)
            elseif marg isa DiscreteEmpirical
                # Detection marginal: store weighted mean/std as Normal placeholders.
                # The actual DiscreteEmpirical is threaded separately through the
                # acceleration loop and restored via _replace_detection_marginal.
                sw = sum(marg.probs)
                if sw > 1e-10
                    mu = sum(marg.values .* marg.probs) / sw
                    var_val = sum(marg.probs .* (marg.values .- mu).^2) / sw
                    sigma = max(sqrt(max(var_val, 0.0)), 0.01)
                else
                    mu = 0.0
                    sigma = 1.0
                end
                push!(theta, isfinite(mu) ? mu : 0.0,
                             isfinite(sigma) ? sigma : 1.0)
            else
                mu = mean(marg)
                sigma = std(marg)
                push!(theta, isfinite(mu) ? mu : 0.0,
                             isfinite(sigma) ? sigma : 1.0)
            end
        end
    end
    return theta  # length = 3 + 3*6 = 21
end

"""
    _replace_detection_marginal(joint::SklarDist, disc::DiscreteEmpirical) -> SklarDist

Replace the third marginal (detection dimension) in a SklarDist with the given
DiscreteEmpirical, preserving the copula structure and the first two marginals.
Used by SQUAREM to restore discrete detection marginals after flat-vector acceleration.
"""
function _replace_detection_marginal(joint::SklarDist, disc::DiscreteEmpirical)
    margs = joint.m
    new_margs = (margs[1], margs[2], disc)
    return SklarDist(joint.C, new_margs)
end

"""
    restore_em_params_logbf(theta, cop_H0, cop_ag, cop_H1; h1_family=:gamma) -> (pi_H0, pi_ag, pi_H1, joint_H0, joint_ag, joint_H1)

Restore 3-component EM parameters from flat vector.
Copula structures are held fixed (only marginal parameters and mixing weights are restored).

`h1_family` controls which distribution is used for H1 enrichment (positions 16-17):
- `:gamma` (default): `LocationShifted(Gamma(p1, p2), JEFFREYS_SHIFT)`
- `:lognormal`: `LocationShifted(LogNormal(p1, p2), JEFFREYS_SHIFT)`
- `:weibull`: `LocationShifted(Weibull(p1, p2), JEFFREYS_SHIFT)`
"""
function restore_em_params_logbf(theta::Vector{Float64}, cop_H0, cop_ag, cop_H1;
                                  h1_family::Symbol=:gamma)
    # Weights: clamp to positive, normalize
    raw_w = clamp.(theta[1:3], 1e-6, Inf)
    s = sum(raw_w)
    pi_H0, pi_ag, pi_H1 = raw_w ./ s

    idx = 4
    joints = []
    for (comp_idx, cop) in enumerate((cop_H0, cop_ag, cop_H1))
        margs = []
        for d in 1:3
            p1 = isfinite(theta[idx]) ? theta[idx] : 0.0
            p2 = isfinite(theta[idx+1]) ? max(theta[idx+1], 0.01) : 1.0
            if comp_idx == 3 && d == 1  # H1 enrichment: LocationShifted{T} (family-dispatched)
                marg = if h1_family == :lognormal
                    LocationShifted(LogNormal(clamp(p1, -2.0, 5.0), clamp(p2, 0.05, 3.0)), JEFFREYS_SHIFT)
                elseif h1_family == :weibull
                    LocationShifted(Weibull(clamp(p1, 0.5, 20.0), clamp(p2, 0.05, 20.0)), JEFFREYS_SHIFT)
                else  # :gamma (default)
                    LocationShifted(Gamma(clamp(p1, 0.5, 50.0), clamp(p2, 0.05, 20.0)), JEFFREYS_SHIFT)
                end
                push!(margs, marg)
            else  # Normal (H0 enrichment, correlation, detection, agnostic enrichment)
                push!(margs, Normal(p1, max(p2, 0.01)))
            end
            idx += 2
        end
        push!(joints, SklarDist(cop, Tuple(margs)))
    end
    return pi_H0, pi_ag, pi_H1, joints[1], joints[2], joints[3]
end

"""
    squarem_acceleration_step(state::SQUAREMState) -> Union{Vector{Float64}, Nothing}

Compute SQUAREM acceleration step from last 3 parameter vectors.
Returns accelerated parameters or nothing if acceleration cannot be computed.
"""
function squarem_acceleration_step(state::SQUAREMState; step_scale::Float64 = 1.0)
    # Need 3 consecutive parameter vectors
    if state.θ_prev2 === nothing || state.θ_prev1 === nothing || state.θ_curr === nothing
        return nothing
    end

    # Compute r and v vectors
    r = state.θ_prev1 .- state.θ_prev2   # First EM step direction
    v = state.θ_curr .- state.θ_prev1 .- r  # Change in step direction

    # Compute step length (alpha)
    r_norm_sq = sum(r.^2)
    v_norm_sq = sum(v.^2)

    # Avoid division by zero
    if v_norm_sq < 1e-20
        return nothing
    end

    # SQUAREM step length
    alpha = -sqrt(r_norm_sq / v_norm_sq)

    # Bound alpha to prevent overshooting (recommended: -1 to -0.01)
    alpha = clamp(alpha, -1.0, -0.01)

    # Apply step scale for step-halving
    alpha_scaled = alpha * step_scale

    # Compute accelerated parameters
    theta_accel = state.θ_prev2 .- 2 * alpha_scaled .* r .+ alpha_scaled^2 .* v

    # Ensure mixing weights valid
    for i in 1:3
        theta_accel[i] = max(theta_accel[i], 1e-6)
    end
    # Ensure sigma params positive (indices 5,7,9,11,13,15,17,19,21)
    for i in 4:length(theta_accel)
        if mod(i, 2) == 1  # sigma positions (odd indices starting from 5)
            theta_accel[i] = max(theta_accel[i], 0.01)
        end
    end

    return theta_accel
end

"""
    update_squarem_state!(state::SQUAREMState, theta_new::Vector{Float64}, ll_new::Float64)

Update SQUAREM state with new parameters after EM step.
"""
function update_squarem_state!(state::SQUAREMState, theta_new::Vector{Float64}, ll_new::Float64)
    # Shift parameter history
    state.θ_prev2 = state.θ_prev1
    state.θ_prev1 = state.θ_curr
    state.θ_curr = copy(theta_new)

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
    compute_log_likelihood_3c(pi_H0, pi_ag, pi_H1, joint_H0, joint_ag, joint_H1,
                               log_bf_matrix, min_log_exp, max_log_exp) -> Float64

Compute 3-component log marginal likelihood for given parameters.
"""
function compute_log_likelihood_3c(pi_H0::Float64, pi_ag::Float64, pi_H1::Float64,
                                    joint_H0, joint_ag, joint_H1,
                                    log_bf_matrix::AbstractMatrix{Float64},
                                    min_log_exp::Float64, max_log_exp::Float64)
    ll_H0 = _safe_logpdf_vec(joint_H0, log_bf_matrix, min_log_exp, max_log_exp)
    ll_ag = _safe_logpdf_vec(joint_ag, log_bf_matrix, min_log_exp, max_log_exp)
    ll_H1 = _safe_logpdf_vec(joint_H1, log_bf_matrix, min_log_exp, max_log_exp)

    log_pi_H0 = log(max(pi_H0, 1e-300))
    log_pi_ag = log(max(pi_ag, 1e-300))
    log_pi_H1 = log(max(pi_H1, 1e-300))

    total = 0.0
    for j in eachindex(ll_H0)
        a = log_pi_H0 + ll_H0[j]
        b = log_pi_ag + ll_ag[j]
        c = log_pi_H1 + ll_H1[j]
        mx = max(a, b, c)
        total += mx + log(exp(a - mx) + exp(b - mx) + exp(c - mx))
    end

    return total
end

"""
    em_fit_mixture_accelerated_logbf(pi_H0, pi_ag, pi_H1, joint_H0, joint_ag, joint_H1,
                                      log_bf_matrix, refID; kwargs...) -> (pi_H0, pi_ag, pi_H1, joint_H0, joint_ag, joint_H1, converged, n_iter)

SQUAREM-accelerated continuation of 3-component copula EM on log-BF scale.
Takes the current EM state (mixing weights + 3 SklarDist components) and
accelerates using SQUAREM on a 21-param vector, with copula structures held fixed
(only mixing weights and Normal marginal params are accelerated).

DiscreteEmpirical detection marginals are threaded through the acceleration loop
separately via `disc_H0`, `disc_ag`, `disc_H1` keyword arguments. After each
SQUAREM accepted step, the restored joints have their detection dimension replaced
with the current DiscreteEmpirical via `_replace_detection_marginal`.

# Arguments
- Current EM state: mixing weights and 3 SklarDist joints
- `log_bf_matrix`: 3 x n matrix of log-BFs
- `refID::Int`: Reference protein index

# Keywords
- `max_iter::Int=500`: Maximum acceleration iterations
- `burn_in::Int=5`: Burn-in before acceleration starts
- `dirichlet_prior::Vector{Float64}`: Dirichlet prior for mixing weights
- `verbose::Bool=true`: Print info
- `disc_H0`, `disc_ag`, `disc_H1`: Optional DiscreteEmpirical for detection dimension
"""
function em_fit_mixture_accelerated_logbf(
    pi_H0::Float64, pi_ag::Float64, pi_H1::Float64,
    joint_H0::SklarDist, joint_ag::SklarDist, joint_H1::SklarDist,
    log_bf_matrix::AbstractMatrix{Float64},
    refID::Int;
    max_iter::Int = 500,
    burn_in::Int = 5,
    dirichlet_prior::Vector{Float64} = [5.0, 2.0, 1.0],
    verbose::Bool = true,
    h1_family::Union{Symbol, Nothing} = nothing,  # auto-detected from joint_H1 if nothing
    disc_H0::Union{Nothing, DiscreteEmpirical} = nothing,
    disc_ag::Union{Nothing, DiscreteEmpirical} = nothing,
    disc_H1::Union{Nothing, DiscreteEmpirical} = nothing
)
    min_log_exp = -700.0
    max_log_exp = 700.0
    n = size(log_bf_matrix, 2)

    cop_H0 = joint_H0.C
    cop_ag = joint_ag.C
    cop_H1 = joint_H1.C

    # Auto-detect h1_family from joint_H1 if not provided
    _h1_family = if h1_family !== nothing
        h1_family
    else
        # H1 enrichment marginal is joint_H1.m[1] (first marginal = enrichment)
        m1 = joint_H1.m[1]
        if m1 isa LocationShifted{<:LogNormal}
            :lognormal
        elseif m1 isa LocationShifted{<:Weibull}
            :weibull
        else
            :gamma  # default (Gamma or any other type)
        end
    end

    sqr_state = SQUAREMState()
    converged = false

    for iter in 1:max_iter
        # E-step: compute responsibilities
        ll_h0 = _safe_logpdf_vec(joint_H0, log_bf_matrix, min_log_exp, max_log_exp)
        ll_ag = _safe_logpdf_vec(joint_ag, log_bf_matrix, min_log_exp, max_log_exp)
        ll_h1 = _safe_logpdf_vec(joint_H1, log_bf_matrix, min_log_exp, max_log_exp)

        # Exclude bait
        ll_h0[refID] = min_log_exp
        ll_ag[refID] = min_log_exp
        ll_h1[refID] = min_log_exp

        log_pi_H0 = log(max(pi_H0, 1e-300))
        log_pi_ag = log(max(pi_ag, 1e-300))
        log_pi_H1 = log(max(pi_H1, 1e-300))

        # Responsibilities via log-sum-exp
        r_H0 = Vector{Float64}(undef, n)
        r_ag = Vector{Float64}(undef, n)
        r_H1 = Vector{Float64}(undef, n)
        total_ll = 0.0
        for j in 1:n
            a = log_pi_H0 + ll_h0[j]
            b = log_pi_ag + ll_ag[j]
            c = log_pi_H1 + ll_h1[j]
            mx = max(a, b, c)
            denom = mx + log(exp(a - mx) + exp(b - mx) + exp(c - mx))
            total_ll += denom
            r_H0[j] = exp(a - denom)
            r_ag[j] = exp(b - denom)
            r_H1[j] = exp(c - denom)
        end

        # M-step: mixing weights with Dirichlet prior
        n_h0 = sum(r_H0) + dirichlet_prior[1] - 1.0
        n_ag = sum(r_ag) + dirichlet_prior[2] - 1.0
        n_h1 = sum(r_H1) + dirichlet_prior[3] - 1.0
        total_n = n_h0 + n_ag + n_h1
        pi_H0_new = max(n_h0 / total_n, 1e-6)
        pi_ag_new = max(n_ag / total_n, 1e-6)
        pi_H1_new = max(n_h1 / total_n, 1e-6)
        s_w = pi_H0_new + pi_ag_new + pi_H1_new
        pi_H0_new /= s_w; pi_ag_new /= s_w; pi_H1_new /= s_w

        # M-step: refit Normal marginals for enrichment/correlation dimensions;
        # detection dimension (d=3) uses DiscreteEmpirical via separate storage.
        new_margs_h0 = _fit_normal_weighted_3(log_bf_matrix, r_H0)
        new_margs_ag = _fit_normal_weighted_3(log_bf_matrix, r_ag)
        new_margs_h1 = _fit_normal_weighted_3(log_bf_matrix, r_H1)
        joint_H0 = SklarDist(cop_H0, new_margs_h0)
        joint_ag = SklarDist(cop_ag, new_margs_ag)
        joint_H1 = SklarDist(cop_H1, new_margs_h1)

        # M-step: refit DiscreteEmpirical for detection dimension (if provided)
        det_data = @view log_bf_matrix[3, :]
        if disc_H0 !== nothing
            disc_H0 = _fit_discrete_empirical_weighted(det_data, r_H0)
            disc_ag = _fit_discrete_empirical_weighted(det_data, r_ag)
            disc_H1 = _fit_discrete_empirical_weighted(det_data, r_H1)
        end

        # Update SQUAREM state (DiscreteEmpirical stored as Normal placeholders)
        theta_current = extract_em_params_logbf(pi_H0_new, pi_ag_new, pi_H1_new,
                                                 joint_H0, joint_ag, joint_H1)
        update_squarem_state!(sqr_state, theta_current, total_ll)

        # Attempt SQUAREM acceleration
        if should_attempt_acceleration(sqr_state, iter, burn_in)
            theta_accel = squarem_acceleration_step(sqr_state)
            if theta_accel !== nothing
                pi_H0_a, pi_ag_a, pi_H1_a, jH0_a, jag_a, jH1_a = restore_em_params_logbf(
                    theta_accel, cop_H0, cop_ag, cop_H1; h1_family=_h1_family)
                # Restore DiscreteEmpirical detection marginals after SQUAREM step
                if disc_H0 !== nothing
                    jH0_a = _replace_detection_marginal(jH0_a, disc_H0)
                    jag_a  = _replace_detection_marginal(jag_a,  disc_ag)
                    jH1_a  = _replace_detection_marginal(jH1_a,  disc_H1)
                end
                ll_accel = compute_log_likelihood_3c(pi_H0_a, pi_ag_a, pi_H1_a,
                    jH0_a, jag_a, jH1_a, log_bf_matrix, min_log_exp, max_log_exp)
                if isfinite(ll_accel) && ll_accel > total_ll
                    pi_H0_new = pi_H0_a; pi_ag_new = pi_ag_a; pi_H1_new = pi_H1_a
                    joint_H0 = jH0_a; joint_ag = jag_a; joint_H1 = jH1_a
                    sqr_state.n_accel_steps += 1
                else
                    # Step-halving: try alpha/2, alpha/4, alpha/8 before falling back to vanilla EM
                    accepted_halving = false
                    for halving_power in 1:3
                        scale = 1.0 / (2^halving_power)
                        theta_half = squarem_acceleration_step(sqr_state; step_scale=scale)
                        if theta_half !== nothing
                            pi_H0_h, pi_ag_h, pi_H1_h, jH0_h, jag_h, jH1_h = restore_em_params_logbf(
                                theta_half, cop_H0, cop_ag, cop_H1; h1_family=_h1_family)
                            if disc_H0 !== nothing
                                jH0_h = _replace_detection_marginal(jH0_h, disc_H0)
                                jag_h  = _replace_detection_marginal(jag_h,  disc_ag)
                                jH1_h  = _replace_detection_marginal(jH1_h,  disc_H1)
                            end
                            ll_half = compute_log_likelihood_3c(pi_H0_h, pi_ag_h, pi_H1_h,
                                jH0_h, jag_h, jH1_h, log_bf_matrix, min_log_exp, max_log_exp)
                            if isfinite(ll_half) && ll_half > total_ll
                                pi_H0_new = pi_H0_h; pi_ag_new = pi_ag_h; pi_H1_new = pi_H1_h
                                joint_H0 = jH0_h; joint_ag = jag_h; joint_H1 = jH1_h
                                sqr_state.n_accel_steps += 1
                                accepted_halving = true
                                break
                            end
                        end
                    end
                    if !accepted_halving
                        sqr_state.n_fallback_steps += 1
                    end
                end
            end
        end

        # Check convergence
        if iter > 10
            ll_change = abs(total_ll - sqr_state.ll_prev) / max(abs(total_ll), 1.0)
            if ll_change < 1e-6
                converged = true
                pi_H0 = pi_H0_new; pi_ag = pi_ag_new; pi_H1 = pi_H1_new
                break
            end
        end

        pi_H0 = pi_H0_new; pi_ag = pi_ag_new; pi_H1 = pi_H1_new
    end

    if verbose && (sqr_state.n_accel_steps > 0 || sqr_state.n_fallback_steps > 0)
        @info "SQUAREM: $(sqr_state.n_accel_steps) accelerated steps, $(sqr_state.n_fallback_steps) fallbacks"
    end

    return pi_H0, pi_ag, pi_H1, joint_H0, joint_ag, joint_H1, converged, max_iter
end

"""
    _fit_normal_weighted_3(log_bf_matrix, weights) -> Tuple{Normal, Normal, Normal}

Fit 3 Normal marginals (one per dimension) using weighted data.
"""
function _fit_normal_weighted_3(log_bf_matrix::AbstractMatrix{Float64},
                                 weights::Vector{Float64})
    margs = []
    sw = sum(weights)
    for d in 1:3
        data_d = @view log_bf_matrix[d, :]
        if sw < 1e-10
            push!(margs, Normal(0.0, 1.0))
            continue
        end
        mu = sum(weights .* data_d) / sw
        var_val = sum(weights .* (data_d .- mu).^2) / sw
        sigma = max(sqrt(var_val), 0.01)
        push!(margs, Normal(mu, sigma))
    end
    return Tuple(margs)
end
