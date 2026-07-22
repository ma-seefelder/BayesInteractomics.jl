# ============================================================
# Empirical Bayes Core Functions
# ============================================================
# Minka (2000) fixed-point Dirichlet alpha estimation and
# BIC-weighted prior grid marginalization.
# Functions:
#   - inv_digamma: Newton solver for the inverse digamma function
#   - estimate_dirichlet_eb: outer fixed-point loop for Dirichlet MLE
#   - _perturb_proportions: simplex mass shift helper
#   - build_prior_grid: constant-strength proportion-varying Dirichlet prior grid
#   - _marginalize_over_priors: parallel EM across grid with BIC averaging
# ============================================================

using SpecialFunctions: digamma, trigamma
using Statistics: mean
using LogExpFunctions: logsumexp

"""
    inv_digamma(y::Float64; maxiter::Int=50, tol::Float64=1e-12) -> Float64

Compute the inverse of the digamma (psi) function using Newton's method.

Given `y`, find `x > 0` such that `digamma(x) = y`.

Two-regime initialization (Minka 2000):
- For `y >= -2.22`: `x0 = exp(y) + 0.5`
- For `y < -2.22`: `x0 = -1 / (y - digamma(1.0))`

Newton iteration: `x -= (digamma(x) - y) / trigamma(x)` with positivity guard.
"""
function inv_digamma(y::Float64; maxiter::Int=50, tol::Float64=1e-12)::Float64
    # Two-regime initialization
    if y >= -2.22
        x = exp(y) + 0.5
    else
        x = -1.0 / (y - digamma(1.0))
    end
    x = max(x, 1e-10)

    # Newton iteration
    for _ in 1:maxiter
        dx = (digamma(x) - y) / trigamma(x)
        x -= dx
        x = max(x, 1e-10)  # Positivity guard
        if abs(dx) < tol
            break
        end
    end

    return x
end

"""
    estimate_dirichlet_eb(gamma::Matrix{Float64}; maxiter::Int=100, tol::Float64=1e-8,
                          floor::Float64=0.5, sum_bounds::Tuple{Float64,Float64}=(3.0, 30.0))

Estimate Dirichlet concentration parameters from a responsibility matrix using
Minka's (2000) fixed-point iteration.

# Arguments
- `gamma`: N x K matrix of responsibilities (each row sums to ~1)
- `maxiter`: Maximum outer fixed-point iterations
- `tol`: Relative convergence tolerance
- `floor`: Minimum allowed alpha component value (post-convergence)
- `sum_bounds`: (min, max) bounds for sum(alpha) (post-convergence rescaling)

# Returns
A NamedTuple `(alpha, converged, iterations, was_clamped)`:
- `alpha::Vector{Float64}`: Estimated Dirichlet concentration parameters
- `converged::Bool`: Whether the fixed-point iteration converged
- `iterations::Int`: Number of iterations performed
- `was_clamped::Bool`: Whether post-convergence clamping was applied
"""
function estimate_dirichlet_eb(gamma::Matrix{Float64};
                                maxiter::Int=1000,
                                tol::Float64=1e-8,
                                floor::Float64=0.5,
                                sum_bounds::Tuple{Float64,Float64}=(3.0, 30.0))
    n, K = size(gamma)

    # Log-guard: protect against zeros in gamma
    log_gamma = log.(max.(gamma, eps(Float64)))

    # Mean log-responsibilities: K-vector
    mean_log = vec(mean(log_gamma, dims=1))

    # Initialize alpha from mean responsibilities
    mean_gamma = vec(mean(gamma, dims=1))
    alpha = K .* mean_gamma
    alpha = max.(alpha, 0.1)

    converged = false
    iter = 0

    # Outer fixed-point loop (Minka 2000)
    for i in 1:maxiter
        iter = i
        alpha_old = copy(alpha)

        s = sum(alpha)
        psi_s = digamma(s)

        for k in 1:K
            alpha[k] = inv_digamma(psi_s + mean_log[k])
            alpha[k] = max(alpha[k], 1e-10)  # Keep positive during iteration
        end

        # Convergence check: relative change
        rel_change = maximum(abs.(alpha .- alpha_old) ./ max.(alpha_old, 1e-10))
        if rel_change < tol
            converged = true
            break
        end
    end

    # Post-convergence clamping
    was_clamped = false

    # Rescale if sum out of bounds (before floor, so floor takes priority)
    s = sum(alpha)
    if s < sum_bounds[1]
        alpha .*= sum_bounds[1] / s
        was_clamped = true
    elseif s > sum_bounds[2]
        alpha .*= sum_bounds[2] / s
        was_clamped = true
    end

    # Floor individual components (after rescaling, so floor is guaranteed)
    for k in 1:K
        if alpha[k] < floor
            alpha[k] = floor
            was_clamped = true
        end
    end

    return (alpha = alpha, converged = converged, iterations = iter, was_clamped = was_clamped)
end

"""
    _perturb_proportions(base_props, target_indices, shift)

Shift `shift` total mass from non-target to target components on the simplex.
Each proportion is floored at 0.05 and renormalized to sum to 1.0.
"""
function _perturb_proportions(base_props::Vector{Float64}, target_indices::Vector{Int}, shift::Float64)
    p = copy(base_props)
    non_targets = setdiff(1:3, target_indices)
    total_removable = sum(p[non_targets])
    for i in non_targets
        p[i] -= shift * (p[i] / total_removable)
    end
    n_targets = length(target_indices)
    for i in target_indices
        p[i] += shift / n_targets
    end
    # Iterative floor + renormalize to guarantee all components >= 0.05
    for _ in 1:10
        any_below = false
        for i in 1:length(p)
            if p[i] < 0.05
                p[i] = 0.05
                any_below = true
            end
        end
        p ./= sum(p)
        if !any_below
            break
        end
    end
    return p
end

"""
    build_prior_grid(alpha_hat)

Build a 9-point constant-strength Dirichlet prior grid. Total strength `S = sum(alpha_hat)`
is fixed; component proportions are varied across the 2-simplex:
EB center, Uniform, 3 single-component pushes (25%), 3 two-component pushes (20%),
1 strong H0 corner (30%). Degenerate components are floored at 5% of S.
Duplicates (from symmetric inputs) are removed via isapprox.
"""
function build_prior_grid(alpha_hat::Vector{Float64})
    S = sum(alpha_hat)
    base_props = alpha_hat ./ S

    grid = Vector{Float64}[]

    # 1. EB center
    push!(grid, copy(alpha_hat))

    # 2. Uniform
    push!(grid, fill(S / 3, 3))

    # 3-5. Single-component pushes (~25% shift)
    for target in 1:3
        p = _perturb_proportions(base_props, [target], 0.25)
        push!(grid, p .* S)
    end

    # 6-8. Two-component pushes (~20% shift)
    for (t1, t2) in [(1, 2), (1, 3), (2, 3)]
        p = _perturb_proportions(base_props, [t1, t2], 0.20)
        push!(grid, p .* S)
    end

    # 9. Strong H0 corner (~30% shift)
    p = _perturb_proportions(base_props, [1], 0.30)
    push!(grid, p .* S)

    # Deduplicate via isapprox
    unique_grid = Vector{Float64}[]
    for g in grid
        if isempty(unique_grid) || !any(ug -> isapprox(ug, g, atol=1e-10), unique_grid)
            push!(unique_grid, g)
        end
    end

    return unique_grid
end

"""
    compute_bic_weights(log_likelihoods)

Compute normalized BIC weights from final log-likelihoods.
Since all models have the same parameter count and data size,
BIC differences reduce to log-likelihood differences.
Falls back to uniform weights if all log-likelihoods are -Inf.
"""
function compute_bic_weights(log_likelihoods::Vector{Float64})
    n = length(log_likelihoods)
    if all(isinf.(log_likelihoods))
        return fill(1.0 / n, n)
    end
    # Filter out -Inf for logsumexp, assign zero weight to -Inf entries
    finite_mask = isfinite.(log_likelihoods)
    if !any(finite_mask)
        return fill(1.0 / n, n)
    end
    finite_lls = log_likelihoods[finite_mask]
    log_norm = logsumexp(finite_lls)
    weights = zeros(n)
    for i in 1:n
        if isfinite(log_likelihoods[i])
            weights[i] = exp(log_likelihoods[i] - log_norm)
        end
    end
    return weights
end

"""
    _marginalize_over_priors(y_e_win, y_c_win, y_p_win, y_e_orig, y_c_orig, y_p_orig,
                              grid; force_h1_family, n_restarts=10, n_iterations=200,
                              convergence_tol=1e-6)

Run parallel multi-restart EMs across a Dirichlet prior grid, compute BIC weights,
and average posteriors (probability scale) and BFs (geometric mean via log scale).

# Arguments
- `y_e_win, y_c_win, y_p_win`: Winsorized log-BF vectors for EM fitting
- `y_e_orig, y_c_orig, y_p_orig`: Original log-BF vectors for posterior computation
- `grid`: Vector of alpha prior vectors from `build_prior_grid`
- `force_h1_family`: H1 enrichment family (:gamma, :lognormal, :weibull) locked from baseline EM
- `n_restarts`: Number of EM restarts per grid point (default: 10)
- `n_iterations`: Max EM iterations per restart (default: 200)
- `convergence_tol`: EM convergence tolerance (default: 1e-6)

# Returns
NamedTuple with:
- `posterior_prob`: Averaged posteriors (probability scale)
- `combined_bf`: Averaged BFs (geometric mean via log scale)
- `bic_weights`: Normalized BIC weights per grid point
- `grid`: The alpha vectors used
- `per_grid_ll`: Final log-likelihood per grid point (best restart)
- `per_grid_posteriors`: Per-grid posterior vectors for diagnostics
- `dominant_grid_index`: Index of highest-weight grid point
- `dominant_weight`: Weight of dominant grid point (flag if > 0.95)
- `n_distinct_points`: Number of grid points
"""
function _marginalize_over_priors(y_e_win::Vector{Float64}, y_c_win::Vector{Float64}, y_p_win::Vector{Float64},
                                   y_e_orig::Vector{Float64}, y_c_orig::Vector{Float64}, y_p_orig::Vector{Float64},
                                   grid::Vector{Vector{Float64}};
                                   force_h1_family::Symbol,
                                   n_restarts::Int = 10,
                                   n_iterations::Int = 200,
                                   convergence_tol::Float64 = 1e-6)
    n_grid = length(grid)
    n_proteins = length(y_e_win)

    # Parallel multi-restart EM per grid point (standalone parallel loop)
    tasks = map(1:n_grid) do idx
        Threads.@spawn begin
            best_em_local = nothing
            best_ll_local = -Inf
            for r in 1:n_restarts
                init_params = initialize_3c_em(y_e_win, y_c_win, y_p_win, r; n_restarts=n_restarts)
                em_result = fit_gaussian_mixture_em_3c(y_e_win, y_c_win, y_p_win;
                                                       n_iterations = n_iterations,
                                                       alpha_prior = grid[idx],
                                                       tol = convergence_tol,
                                                       init_params = init_params,
                                                       h1_family = force_h1_family,
                                                       skip_bic_selection = true)
                final_ll = isempty(em_result.log_likelihood) ? -Inf : em_result.log_likelihood[end]
                if final_ll > best_ll_local
                    best_ll_local = final_ll
                    best_em_local = em_result
                end
            end
            (best_em = best_em_local, best_ll = best_ll_local)
        end
    end

    grid_results = [fetch(t) for t in tasks]

    # Extract best log-likelihoods and compute BIC weights
    per_grid_ll = [gr.best_ll for gr in grid_results]
    bic_weights = compute_bic_weights(per_grid_ll)

    # Compute posteriors per grid point using compute_robust_posteriors_3c
    # Use original data for enrichment/correlation, winsorized for detection
    per_grid_posteriors = Vector{Vector{Float64}}(undef, n_grid)
    per_grid_log_bfs = Vector{Vector{Float64}}(undef, n_grid)

    for idx in 1:n_grid
        em = grid_results[idx].best_em
        if em === nothing
            per_grid_posteriors[idx] = fill(0.0, n_proteins)
            per_grid_log_bfs[idx] = fill(0.0, n_proteins)
        else
            posteriors = compute_robust_posteriors_3c(y_e_orig, y_c_orig, y_p_win, em)
            per_grid_posteriors[idx] = copy(posteriors.p_h1)
            per_grid_log_bfs[idx] = clamp.(log.(max.(posteriors.joint_bf, 1e-300)), -46.0, 46.0)
        end
    end

    # Posterior averaging on probability scale
    avg_posterior = zeros(n_proteins)
    for (w, p) in zip(bic_weights, per_grid_posteriors)
        avg_posterior .+= w .* p
    end

    # Geometric mean of BFs via log-scale weighted average
    avg_log_bf = zeros(n_proteins)
    for (w, lb) in zip(bic_weights, per_grid_log_bfs)
        avg_log_bf .+= w .* lb
    end
    # Clamp log-BFs to [-46, 46] to match the explicit-alpha path in combined_BF_latent_class
    # Without this, compute_robust_posteriors_3c can produce BFs of 1e300+ when p_h1 ≈ 1.0
    clamp!(avg_log_bf, -46.0, 46.0)
    avg_bf = exp.(avg_log_bf)

    # Dominant weight diagnostic
    dominant_idx = argmax(bic_weights)
    dominant_w = bic_weights[dominant_idx]
    if dominant_w > 0.95
        @warn "Grid marginalization: dominant BIC weight $(round(dominant_w, digits=3)) at grid point $dominant_idx -- averaging provides limited robustness"
    end

    return (
        posterior_prob = avg_posterior,
        combined_bf = avg_bf,
        bic_weights = bic_weights,
        grid = grid,
        per_grid_ll = per_grid_ll,
        per_grid_posteriors = per_grid_posteriors,
        dominant_grid_index = dominant_idx,
        dominant_weight = dominant_w,
        n_distinct_points = n_grid
    )
end
