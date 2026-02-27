# ============================================================
# AP-MS Latent Class Model - Manual EM Implementation
# ============================================================
#
# Combination of three evidence arms (Enrichment, Correlation,
# Presence/Absence) via a shared latent mixture model.
#
# Core assumption: The three scores are CONDITIONALLY INDEPENDENT given
# the true interaction status z ∈ {Background, Interaction}.
#
# This implementation uses a manual EM algorithm instead of RxInfer
# to avoid complexity with mixture node APIs.
# ============================================================

# ============================================================
# 1. Data Preprocessing
# ============================================================

"""
    _winsorize(x::Vector{Float64}, lower::Float64, upper::Float64)

Clamp values in `x` to the `[lower, upper]` quantile range.
Returns a new vector with extreme values replaced by the quantile bounds.
"""
function _winsorize(x::Vector{Float64}, lower::Float64, upper::Float64)
    lo = quantile(x, lower)
    hi = quantile(x, upper)
    return clamp.(x, lo, hi)
end

"""
    prepare_lc_scores(bf_enrich, bf_corr, bf_pres; log_transform=true,
                      winsorize=true, winsorize_quantiles=(0.01, 0.99))

Prepares Bayes factors for the latent class model.

Returns a 6-tuple: `(y_e_win, y_c_win, y_p_win, y_e_orig, y_c_orig, y_p_orig)`
where `_win` are winsorized (for EM fitting) and `_orig` are original log-BFs
(for final BF computation).

The scores should be input as log-BFs because:
1. log-BFs are approximately normally distributed (CLT)
2. The scale is symmetric: log(BF)=0 means no evidence
3. Matches the Normal mixture assumption of the model
"""
function prepare_lc_scores(bf_enrich::Vector{<:Real},
                        bf_corr::Vector{<:Real},
                        bf_pres::Vector{<:Real};
                        log_transform::Bool = true,
                        winsorize::Bool = true,
                        winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99))
    n = length(bf_enrich)
    @assert length(bf_corr) == n && length(bf_pres) == n "All score vectors must have equal length"

    if log_transform
        # log-transformation; small epsilon correction against log(0)
        ε = 1e-300
        y_e = log.(max.(bf_enrich, ε))
        y_c = log.(max.(bf_corr, ε))
        y_p = log.(max.(bf_pres, ε))
    else
        y_e = float.(bf_enrich)
        y_c = float.(bf_corr)
        y_p = float.(bf_pres)
    end

    # Keep original (non-winsorized) values
    y_e_orig = copy(y_e)
    y_c_orig = copy(y_c)
    y_p_orig = copy(y_p)

    # Winsorize for EM fitting
    if winsorize
        lo, hi = winsorize_quantiles
        y_e_win = _winsorize(y_e, lo, hi)
        y_c_win = _winsorize(y_c, lo, hi)
        y_p_win = _winsorize(y_p, lo, hi)
    else
        y_e_win = copy(y_e)
        y_c_win = copy(y_c)
        y_p_win = copy(y_p)
    end

    return y_e_win, y_c_win, y_p_win, y_e_orig, y_c_orig, y_p_orig
end

# ============================================================
# 2. EM Algorithm for Gaussian Mixture
# ============================================================

"""
    fit_gaussian_mixture_em(y_enrich, y_corr, y_pres;
                            n_iterations=100, alpha_prior=[10.0, 1.0], tol=1e-6,
                            sigma_floor=0.1)

Fit a 2-component Gaussian mixture model using EM algorithm.

Includes label ordering constraint: the interaction component enrichment mean
is always >= the background enrichment mean. If violated after M-step,
all parameters and responsibilities are swapped.

Returns a NamedTuple with fitted parameters and responsibilities.
"""
function fit_gaussian_mixture_em(y_enrich::Vector{Float64}, y_corr::Vector{Float64}, y_pres::Vector{Float64};
                                  n_iterations::Int=100, alpha_prior::Vector{Float64}=[10.0, 1.0],
                                  tol::Float64=1e-6, sigma_floor::Float64=0.1)
    n = length(y_enrich)

    # Initialize parameters
    # Background component (centered around 0)
    μ_e0, σ_e0 = 0.0, 1.5
    μ_c0, σ_c0 = 0.0, 1.2
    μ_p0, σ_p0 = 0.0, 1.3

    # Interaction component (positive values)
    μ_e1, σ_e1 = 4.0, 1.0
    μ_c1, σ_c1 = 3.0, 0.9
    μ_p1, σ_p1 = 3.5, 0.8

    # Mixing weights (incorporate prior)
    π = [0.9, 0.1]  # Start with prior expectation

    # Storage for responsibilities and log-likelihood
    γ = zeros(n, 2)  # Responsibilities
    log_liks = Float64[]

    for iter in 1:n_iterations
        # E-step: Compute responsibilities
        for i in 1:n
            # Log-likelihood for background component
            ll_bg = logpdf(Normal(μ_e0, σ_e0), y_enrich[i]) +
                    logpdf(Normal(μ_c0, σ_c0), y_corr[i]) +
                    logpdf(Normal(μ_p0, σ_p0), y_pres[i]) +
                    log(π[1] + 1e-300)

            # Log-likelihood for interaction component
            ll_int = logpdf(Normal(μ_e1, σ_e1), y_enrich[i]) +
                     logpdf(Normal(μ_c1, σ_c1), y_corr[i]) +
                     logpdf(Normal(μ_p1, σ_p1), y_pres[i]) +
                     log(π[2] + 1e-300)

            # Normalize responsibilities (log-sum-exp trick for numerical stability)
            max_ll = max(ll_bg, ll_int)
            ll_bg_norm = ll_bg - max_ll
            ll_int_norm = ll_int - max_ll

            denom = exp(ll_bg_norm) + exp(ll_int_norm)
            γ[i, 1] = exp(ll_bg_norm) / denom
            γ[i, 2] = exp(ll_int_norm) / denom
        end

        # M-step: Update parameters
        N_bg = sum(γ[:, 1])
        N_int = sum(γ[:, 2])

        # Update mixing weights (with Dirichlet prior)
        π[1] = (N_bg + alpha_prior[1] - 1) / (n + sum(alpha_prior) - 2)
        π[2] = (N_int + alpha_prior[2] - 1) / (n + sum(alpha_prior) - 2)

        # Guard against empty components: skip parameter updates if effective count is too small
        # (division by near-zero N produces NaN means, which propagate to sigmas)
        if N_bg > 1e-10
            μ_e0 = sum(γ[:, 1] .* y_enrich) / N_bg
            μ_c0 = sum(γ[:, 1] .* y_corr) / N_bg
            μ_p0 = sum(γ[:, 1] .* y_pres) / N_bg
            σ_e0 = max(sqrt(sum(γ[:, 1] .* (y_enrich .- μ_e0).^2) / N_bg), sigma_floor)
            σ_c0 = max(sqrt(sum(γ[:, 1] .* (y_corr .- μ_c0).^2) / N_bg), sigma_floor)
            σ_p0 = max(sqrt(sum(γ[:, 1] .* (y_pres .- μ_p0).^2) / N_bg), sigma_floor)
        end

        if N_int > 1e-10
            μ_e1 = sum(γ[:, 2] .* y_enrich) / N_int
            μ_c1 = sum(γ[:, 2] .* y_corr) / N_int
            μ_p1 = sum(γ[:, 2] .* y_pres) / N_int
            σ_e1 = max(sqrt(sum(γ[:, 2] .* (y_enrich .- μ_e1).^2) / N_int), sigma_floor)
            σ_c1 = max(sqrt(sum(γ[:, 2] .* (y_corr .- μ_c1).^2) / N_int), sigma_floor)
            σ_p1 = max(sqrt(sum(γ[:, 2] .* (y_pres .- μ_p1).^2) / N_int), sigma_floor)
        end

        # Label ordering constraint: interaction enrichment mean must be >= background
        if μ_e1 < μ_e0
            # Swap ALL parameters between components
            μ_e0, μ_e1 = μ_e1, μ_e0
            σ_e0, σ_e1 = σ_e1, σ_e0
            μ_c0, μ_c1 = μ_c1, μ_c0
            σ_c0, σ_c1 = σ_c1, σ_c0
            μ_p0, μ_p1 = μ_p1, μ_p0
            σ_p0, σ_p1 = σ_p1, σ_p0
            π[1], π[2] = π[2], π[1]
            γ[:, 1], γ[:, 2] = γ[:, 2], γ[:, 1]
        end

        # Compute log-likelihood
        ll = 0.0
        for i in 1:n
            ll_bg = logpdf(Normal(μ_e0, σ_e0), y_enrich[i]) +
                    logpdf(Normal(μ_c0, σ_c0), y_corr[i]) +
                    logpdf(Normal(μ_p0, σ_p0), y_pres[i])

            ll_int = logpdf(Normal(μ_e1, σ_e1), y_enrich[i]) +
                     logpdf(Normal(μ_c1, σ_c1), y_corr[i]) +
                     logpdf(Normal(μ_p1, σ_p1), y_pres[i])

            ll += log(π[1] * exp(ll_bg) + π[2] * exp(ll_int) + 1e-300)
        end
        push!(log_liks, ll)

        # Check convergence
        if iter > 10
            rel_change = abs(log_liks[end] - log_liks[end-1]) / abs(log_liks[end-1] + 1e-300)
            if rel_change < tol
                return (
                    mixing_weights = π,
                    means = Dict(
                        "background" => (enrichment=μ_e0, correlation=μ_c0, presence=μ_p0),
                        "interaction" => (enrichment=μ_e1, correlation=μ_c1, presence=μ_p1)
                    ),
                    precisions = Dict(
                        "background" => (enrichment=1/σ_e0^2, correlation=1/σ_c0^2, presence=1/σ_p0^2),
                        "interaction" => (enrichment=1/σ_e1^2, correlation=1/σ_c1^2, presence=1/σ_p1^2)
                    ),
                    std_devs = Dict(
                        "background" => (enrichment=σ_e0, correlation=σ_c0, presence=σ_p0),
                        "interaction" => (enrichment=σ_e1, correlation=σ_c1, presence=σ_p1)
                    ),
                    responsibilities = γ,
                    log_likelihood = log_liks,
                    converged = true,
                    n_iterations = iter
                )
            end
        end
    end

    # Did not converge within iterations
    return (
        mixing_weights = π,
        means = Dict(
            "background" => (enrichment=μ_e0, correlation=μ_c0, presence=μ_p0),
            "interaction" => (enrichment=μ_e1, correlation=μ_c1, presence=μ_p1)
        ),
        precisions = Dict(
            "background" => (enrichment=1/σ_e0^2, correlation=1/σ_c0^2, presence=1/σ_p0^2),
            "interaction" => (enrichment=1/σ_e1^2, correlation=1/σ_c1^2, presence=1/σ_p1^2)
        ),
        std_devs = Dict(
            "background" => (enrichment=σ_e0, correlation=σ_c0, presence=σ_p0),
            "interaction" => (enrichment=σ_e1, correlation=σ_c1, presence=σ_p1)
        ),
        responsibilities = γ,
        log_likelihood = log_liks,
        converged = false,
        n_iterations = n_iterations
    )
end

# ============================================================
# 3. Post-Processing: Posterior Probabilities & BFs
# ============================================================

"""
    compute_robust_posteriors(y_e, y_c, y_p, em_result)

Compute posteriors on original (non-winsorized) data using fitted parameters,
with a monotonicity correction: for each dimension, if a protein's value exceeds
the interaction mean in the positive direction, floor the per-dimension
log-likelihood ratio at 0 (i.e., don't penalize for having "too strong" evidence).

Returns `(p_interact, joint_bf, π_mean, prior_odds)`.
"""
function compute_robust_posteriors(y_e::Vector{Float64}, y_c::Vector{Float64}, y_p::Vector{Float64},
                                   em_result)
    n = length(y_e)
    π = em_result.mixing_weights

    means_bg = em_result.means["background"]
    means_int = em_result.means["interaction"]
    std_bg = em_result.std_devs["background"]
    std_int = em_result.std_devs["interaction"]

    p_interact = Vector{Float64}(undef, n)

    for i in 1:n
        # Per-dimension log-likelihood ratios: log(p(y|int) / p(y|bg))
        llr_e = logpdf(Normal(means_int.enrichment, std_int.enrichment), y_e[i]) -
                logpdf(Normal(means_bg.enrichment, std_bg.enrichment), y_e[i])
        llr_c = logpdf(Normal(means_int.correlation, std_int.correlation), y_c[i]) -
                logpdf(Normal(means_bg.correlation, std_bg.correlation), y_c[i])
        llr_p = logpdf(Normal(means_int.presence, std_int.presence), y_p[i]) -
                logpdf(Normal(means_bg.presence, std_bg.presence), y_p[i])

        # Monotonicity correction: if value exceeds interaction mean in the
        # positive direction, don't let the LLR go negative (which would penalize
        # extreme positive evidence)
        if y_e[i] > means_int.enrichment
            llr_e = max(llr_e, 0.0)
        end
        if y_c[i] > means_int.correlation
            llr_c = max(llr_c, 0.0)
        end
        if y_p[i] > means_int.presence
            llr_p = max(llr_p, 0.0)
        end

        # Total log-likelihood ratio
        total_llr = llr_e + llr_c + llr_p

        # Log posterior odds = log prior odds + total LLR
        log_prior_odds = log(π[2] + 1e-300) - log(π[1] + 1e-300)
        log_posterior_odds = log_prior_odds + total_llr

        # Convert to probability via logistic function
        p_interact[i] = 1.0 / (1.0 + exp(-log_posterior_odds))
    end

    # Prior odds from mixing weights
    prior_odds = π[2] / max(π[1], 1e-300)

    # Joint BF for each protein
    joint_bf = Vector{Float64}(undef, n)
    for i in 1:n
        posterior_odds_i = p_interact[i] / max(1.0 - p_interact[i], 1e-300)
        joint_bf[i] = posterior_odds_i / max(prior_odds, 1e-300)
    end

    return (p_interact = p_interact,
            joint_bf   = joint_bf,
            π_mean     = π,
            prior_odds = prior_odds)
end

"""
    extract_lc_posteriors(em_result)

Extracts posterior interaction probabilities and
computes joint Bayes factors from EM results.
"""
function extract_lc_posteriors(em_result)
    # Responsibilities are already posterior probabilities
    p_interact = em_result.responsibilities[:, 2]

    # Prior odds from mixing weights
    π_mean = em_result.mixing_weights
    prior_odds = π_mean[2] / max(π_mean[1], 1e-300)

    # Joint BF for each protein: posterior odds / prior odds
    n = length(p_interact)
    joint_bf = Vector{Float64}(undef, n)

    for i in 1:n
        posterior_odds_i = p_interact[i] / max(1.0 - p_interact[i], 1e-300)
        joint_bf[i] = posterior_odds_i / max(prior_odds, 1e-300)
    end

    return (p_interact = p_interact,
            joint_bf   = joint_bf,
            π_mean     = π_mean,
            prior_odds = prior_odds)
end

"""
    extract_lc_class_parameters(em_result)

Returns aggregated class-specific parameters for LatentClassResult.
"""
function extract_lc_class_parameters(em_result)
    means_bg = em_result.means["background"]
    means_int = em_result.means["interaction"]
    prec_bg = em_result.precisions["background"]
    prec_int = em_result.precisions["interaction"]
    std_bg = em_result.std_devs["background"]
    std_int = em_result.std_devs["interaction"]

    # Average across dimensions
    bg_avg_mu = mean([means_bg.enrichment, means_bg.correlation, means_bg.presence])
    bg_avg_sigma = mean([std_bg.enrichment, std_bg.correlation, std_bg.presence])
    bg_avg_precision = mean([prec_bg.enrichment, prec_bg.correlation, prec_bg.presence])

    int_avg_mu = mean([means_int.enrichment, means_int.correlation, means_int.presence])
    int_avg_sigma = mean([std_int.enrichment, std_int.correlation, std_int.presence])
    int_avg_precision = mean([prec_int.enrichment, prec_int.correlation, prec_int.presence])

    return Dict(
        "background" => (mu = bg_avg_mu, sigma = bg_avg_sigma, precision = bg_avg_precision),
        "interaction" => (mu = int_avg_mu, sigma = int_avg_sigma, precision = int_avg_precision)
    )
end

# ============================================================
# 4. Main Entry Point
# ============================================================

"""
    combined_BF_latent_class(bf::BayesFactorTriplet, refID::Int;
                              n_iterations=100, alpha_prior=[10.0, 1.0],
                              convergence_tol=1e-6, verbose=true,
                              winsorize=true, winsorize_quantiles=(0.01, 0.99),
                              kwargs...)

Main entry point for latent class-based evidence combination.

This is a drop-in alternative to the copula-based `combined_BF()` function.
It uses an EM algorithm to fit a 2-component Gaussian mixture model on log-Bayes factors.

# Arguments
- `bf::BayesFactorTriplet`: Triplet of Bayes factors (enrichment, correlation, detection)
- `refID::Int`: Index of the bait protein (will be clamped to max interaction probability)

# Keyword Arguments
- `n_iterations::Int=100`: Number of EM iterations
- `alpha_prior::Vector{Float64}=[10.0, 1.0]`: Dirichlet prior for mixing weights (background, interaction)
- `convergence_tol::Float64=1e-6`: Convergence tolerance for log-likelihood
- `verbose::Bool=true`: Print convergence diagnostics
- `winsorize::Bool=true`: Whether to winsorize log-BFs before EM fitting
- `winsorize_quantiles::Tuple{Float64,Float64}=(0.01, 0.99)`: Quantile range for winsorization

# Returns
`LatentClassResult` with combined Bayes factors and posterior probabilities.
"""
function combined_BF_latent_class(bf::BayesFactorTriplet, refID::Int;
                                   n_iterations::Int = 100,
                                   alpha_prior::Vector{Float64} = [10.0, 1.0],
                                   convergence_tol::Float64 = 1e-6,
                                   verbose::Bool = true,
                                   winsorize::Bool = true,
                                   winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99),
                                   kwargs...)
    # 1. Prepare scores (log-transform BFs, with winsorization)
    y_e_win, y_c_win, y_p_win, y_e_orig, y_c_orig, y_p_orig = prepare_lc_scores(
        bf.enrichment, bf.correlation, bf.detection;
        log_transform = true,
        winsorize = winsorize,
        winsorize_quantiles = winsorize_quantiles
    )

    # 2. Run EM algorithm on winsorized data
    if verbose
        @info "Running EM algorithm for latent class model with $n_iterations iterations..."
        if winsorize
            @info "Winsorization enabled with quantiles $(winsorize_quantiles)"
        end
    end

    em_result = fit_gaussian_mixture_em(y_e_win, y_c_win, y_p_win;
                                        n_iterations = n_iterations,
                                        alpha_prior = alpha_prior,
                                        tol = convergence_tol)

    # 3. Check convergence
    if verbose
        if em_result.converged
            @info "Model converged after $(em_result.n_iterations) iterations"
        else
            @warn "Model did not converge within $n_iterations iterations"
        end
        @info "Fitted means - Background enrichment: $(round(em_result.means["background"].enrichment, digits=3)), " *
              "Interaction enrichment: $(round(em_result.means["interaction"].enrichment, digits=3))"
    end

    # 4. Compute robust posteriors on ORIGINAL (non-winsorized) data
    posteriors = compute_robust_posteriors(y_e_orig, y_c_orig, y_p_orig, em_result)
    params = extract_lc_class_parameters(em_result)

    # 5. Handle bait protein: clamp to max interaction probability
    posterior_prob = copy(posteriors.p_interact)
    joint_bf = copy(posteriors.joint_bf)

    if 1 <= refID <= length(posterior_prob)
        max_prob = maximum(posterior_prob)
        posterior_prob[refID] = max_prob
        # Recompute BF for bait: BF = P/(1-P) / prior_odds
        joint_bf[refID] = (max_prob / max(1.0 - max_prob, 1e-300)) / max(posteriors.prior_odds, 1e-300)

        if verbose
            @info "Bait protein (index $refID) clamped to maximum posterior probability: $(round(max_prob, digits=4))"
        end
    end

    # 6. Build and return result
    return LatentClassResult(
        joint_bf,
        posterior_prob,
        params,
        [posteriors.π_mean[1], posteriors.π_mean[2]],
        em_result.log_likelihood,
        em_result.converged,
        em_result.n_iterations
    )
end

# ============================================================
# 5. Visualization
# ============================================================

"""
    plot_lc_convergence(result::LatentClassResult; kwargs...)

Plots the log-likelihood convergence trajectory.
"""
function plot_lc_convergence(result::LatentClassResult;
                              title::String = "Latent Class Model Convergence",
                              xlabel::String = "Iteration",
                              ylabel::String = "Log-Likelihood",
                              kwargs...)
    p = StatsPlots.plot(1:length(result.free_energy), result.free_energy,
             xlabel = xlabel,
             ylabel = ylabel,
             title = title,
             label = "Log-Likelihood",
             linewidth = 2,
             legend = :bottomright;
             kwargs...)

    # Add convergence marker if converged
    if result.converged
        StatsPlots.scatter!(p, [result.n_iterations], [result.free_energy[end]],
                 label = "Converged",
                 markersize = 6,
                 markercolor = :green)
    end

    return p
end
