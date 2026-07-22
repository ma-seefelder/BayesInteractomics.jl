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

# JEFFREYS_SHIFT is defined in src/core/types.jl (single definition site)
# Jeffreys' "substantial evidence" threshold: BF > sqrt(10) ~ 3.16
# On natural-log scale: ln(sqrt(10)) ~ 1.151
# Proteins with ln(BF_enrichment) below this get exactly zero H1 density.

using LogExpFunctions: logistic
using StatsBase: skewness, kurtosis

"""
    _h1_enrichment_logdensity(y::Float64, h1_dist, shift::Float64, k::Float64)

Compute H1 enrichment log-density with smooth sigmoid transition at `shift`.
Returns logpdf of the shifted distribution plus a sigmoid penalty that smoothly
suppresses density below the shift point. At `shift`: penalty = log(0.5) = -0.69 nats.
"""
function _h1_enrichment_logdensity(y::Float64, h1_dist, shift::Float64, k::Float64)::Float64
    shifted_val = max(y - shift, 1e-6)
    return logpdf(h1_dist, shifted_val) + log(logistic(k * (y - shift)))
end

"""
    _sarles_bimodality_coefficient(x::Vector{Float64}) -> Float64

Compute Sarle's bimodality coefficient: BC = (g1^2 + 1) / (g2 + 3)
where g1 = skewness, g2 = excess kurtosis. BC > 5/9 (~0.555) suggests bimodality.
"""
function _sarles_bimodality_coefficient(x::Vector{Float64})::Float64
    n = length(x)
    if n < 4
        return 0.0
    end
    g1 = skewness(x)
    g2 = kurtosis(x)  # StatsBase returns excess kurtosis
    if !isfinite(g1) || !isfinite(g2)
        return 0.0  # constant data produces NaN moments
    end
    denom = g2 + 3.0
    if denom <= 0.0
        return 0.0
    end
    return (g1^2 + 1.0) / denom
end

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
    _validate_bf_inputs(bf_vec::AbstractVector{<:Real})

Sanitize BF inputs before log-transform: NaN -> BF=1 (agnostic),
zero/negative -> tiny positive, Inf -> finite cap.
"""
function _validate_bf_inputs(bf_vec::AbstractVector{<:Real})
    out = copy(Float64.(bf_vec))
    for i in eachindex(out)
        if isnan(out[i])
            out[i] = 1.0  # NaN -> BF=1 (agnostic, log(1)=0)
        elseif out[i] <= 0.0
            out[i] = 1e-300  # Zero/negative BF -> tiny positive (avoids log(0)=-Inf)
        elseif isinf(out[i])
            out[i] = 1e6  # Inf BF -> large finite (matches existing BF cap)
        end
    end
    return out
end

"""
    prepare_lc_scores(bf_enrich, bf_corr, bf_pres; log_transform=true,
                      winsorize=true, winsorize_quantiles=(0.01, 0.99))

Prepares Bayes factors for the latent class model. Validates inputs (NaN, zero, Inf),
log-transforms, and optionally winsorizes.

Returns a 6-tuple: `(y_e_win, y_c_win, y_p_win, y_e_orig, y_c_orig, y_p_orig)`
where `_win` are winsorized (for EM fitting) and `_orig` are original log-BFs
(for final BF computation).
"""
function prepare_lc_scores(bf_enrich::Vector{<:Real},
                        bf_corr::Vector{<:Real},
                        bf_pres::Vector{<:Real};
                        log_transform::Bool = true,
                        winsorize::Bool = true,
                        winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99))
    n = length(bf_enrich)
    @assert length(bf_corr) == n && length(bf_pres) == n "All score vectors must have equal length"

    # Validate and sanitize inputs before log-transform
    bf_enrich = _validate_bf_inputs(bf_enrich)
    bf_corr = _validate_bf_inputs(bf_corr)
    bf_pres = _validate_bf_inputs(bf_pres)

    if log_transform
        # log-transformation with post-log clamp for residual edge cases
        y_e = clamp.(log.(bf_enrich), -10.0, 50.0)
        y_c = clamp.(log.(bf_corr), -10.0, 50.0)
        y_p = clamp.(log.(bf_pres), -10.0, 50.0)
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

    # DiscreteEmpirical for detection dimension
    disc_bg = DiscreteEmpirical(y_pres)
    disc_int = DiscreteEmpirical(y_pres)

    for iter in 1:n_iterations
        # E-step: Compute responsibilities
        for i in 1:n
            # Detection log-likelihoods via DiscreteEmpirical
            ll_det_bg = log(max(pdf(disc_bg, y_pres[i]), 1e-300))
            ll_det_int = log(max(pdf(disc_int, y_pres[i]), 1e-300))

            # Log-likelihood for background component
            ll_bg = logpdf(Normal(μ_e0, σ_e0), y_enrich[i]) +
                    logpdf(Normal(μ_c0, σ_c0), y_corr[i]) +
                    ll_det_bg +
                    log(π[1] + 1e-300)

            # Log-likelihood for interaction component
            ll_int = logpdf(Normal(μ_e1, σ_e1), y_enrich[i]) +
                     logpdf(Normal(μ_c1, σ_c1), y_corr[i]) +
                     ll_det_int +
                     log(π[2] + 1e-300)

            # Normalize responsibilities (log-sum-exp trick for numerical stability)
            max_ll = max(ll_bg, ll_int)
            ll_bg_norm = ll_bg - max_ll
            ll_int_norm = ll_int - max_ll

            denom = exp(ll_bg_norm) + exp(ll_int_norm)
            if !isfinite(denom) || denom < 1e-300
                γ[i, 1] = 0.5
                γ[i, 2] = 0.5
            else
                γ[i, 1] = exp(ll_bg_norm) / denom
                γ[i, 2] = exp(ll_int_norm) / denom
            end
        end

        # M-step: Update parameters
        N_bg = sum(γ[:, 1])
        N_int = sum(γ[:, 2])

        # Update mixing weights (Dirichlet posterior mean — more stable than MAP)
        π[1] = (N_bg + alpha_prior[1]) / (n + sum(alpha_prior))
        π[2] = (N_int + alpha_prior[2]) / (n + sum(alpha_prior))

        # Guard against empty components: skip parameter updates if effective count is too small
        # (division by near-zero N produces NaN means, which propagate to sigmas)
        if N_bg > 1e-10
            μ_e0 = sum(γ[:, 1] .* y_enrich) / N_bg
            μ_c0 = sum(γ[:, 1] .* y_corr) / N_bg
            μ_p0 = sum(γ[:, 1] .* y_pres) / N_bg
            σ_e0 = max(sqrt(sum(γ[:, 1] .* (y_enrich .- μ_e0).^2) / N_bg), sigma_floor)
            σ_c0 = max(sqrt(sum(γ[:, 1] .* (y_corr .- μ_c0).^2) / N_bg), sigma_floor)
            σ_p0 = max(sqrt(sum(γ[:, 1] .* (y_pres .- μ_p0).^2) / N_bg), sigma_floor)
            # Re-fit detection DiscreteEmpirical for background component
            disc_bg = _fit_discrete_empirical_weighted(y_pres, γ[:, 1])
        end

        if N_int > 1e-10
            μ_e1 = sum(γ[:, 2] .* y_enrich) / N_int
            μ_c1 = sum(γ[:, 2] .* y_corr) / N_int
            μ_p1 = sum(γ[:, 2] .* y_pres) / N_int
            σ_e1 = max(sqrt(sum(γ[:, 2] .* (y_enrich .- μ_e1).^2) / N_int), sigma_floor)
            σ_c1 = max(sqrt(sum(γ[:, 2] .* (y_corr .- μ_c1).^2) / N_int), sigma_floor)
            σ_p1 = max(sqrt(sum(γ[:, 2] .* (y_pres .- μ_p1).^2) / N_int), sigma_floor)
            # Re-fit detection DiscreteEmpirical for interaction component
            disc_int = _fit_discrete_empirical_weighted(y_pres, γ[:, 2])
        end

        # Variance ratio constraint: prevent interaction component from becoming
        # much wider than background (causes fat-tail crossover at extremes)
        max_sigma_ratio = 2.0
        σ_e1 = min(σ_e1, max_sigma_ratio * σ_e0)
        σ_c1 = min(σ_c1, max_sigma_ratio * σ_c0)
        σ_p1 = min(σ_p1, max_sigma_ratio * σ_p0)

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
            disc_bg, disc_int = disc_int, disc_bg
        end

        # Compute log-likelihood (using DiscreteEmpirical for detection)
        ll = 0.0
        for i in 1:n
            ll_bg = logpdf(Normal(μ_e0, σ_e0), y_enrich[i]) +
                    logpdf(Normal(μ_c0, σ_c0), y_corr[i]) +
                    log(max(pdf(disc_bg, y_pres[i]), 1e-300))

            ll_int = logpdf(Normal(μ_e1, σ_e1), y_enrich[i]) +
                     logpdf(Normal(μ_c1, σ_c1), y_corr[i]) +
                     log(max(pdf(disc_int, y_pres[i]), 1e-300))

            ll_bg_full = ll_bg + log(π[1] + 1e-300)
            ll_int_full = ll_int + log(π[2] + 1e-300)
            max_ll_i = max(ll_bg_full, ll_int_full)
            ll += max_ll_i + log(exp(ll_bg_full - max_ll_i) + exp(ll_int_full - max_ll_i))
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
                    n_iterations = iter,
                    disc_bg = disc_bg,
                    disc_int = disc_int
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
        n_iterations = n_iterations,
        disc_bg = disc_bg,
        disc_int = disc_int
    )
end

# ============================================================
# 3. Post-Processing: Posterior Probabilities & BFs
# ============================================================

"""
    compute_robust_posteriors(y_e, y_c, y_p, em_result)

Compute posteriors using fitted EM parameters. The input data should be
winsorized to match the EM training range (prevents fat-tail crossover at
extreme negative values). Includes a monotonicity correction: for each
dimension, if a protein's value exceeds the interaction mean in the positive
direction, floor the per-dimension LLR at 0 (don't penalize extreme positive
evidence). The total log-likelihood ratio is clamped to [-46, 46] to match
the copula model's dynamic range.

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

    # Use DiscreteEmpirical for detection if available
    has_disc = hasproperty(em_result, :disc_bg) && em_result.disc_bg !== nothing
    disc_bg_det = has_disc ? em_result.disc_bg : nothing
    disc_int_det = has_disc ? em_result.disc_int : nothing

    p_interact = Vector{Float64}(undef, n)

    for i in 1:n
        # Per-dimension log-likelihood ratios: log(p(y|int) / p(y|bg))
        llr_e = logpdf(Normal(means_int.enrichment, std_int.enrichment), y_e[i]) -
                logpdf(Normal(means_bg.enrichment, std_bg.enrichment), y_e[i])
        llr_c = logpdf(Normal(means_int.correlation, std_int.correlation), y_c[i]) -
                logpdf(Normal(means_bg.correlation, std_bg.correlation), y_c[i])
        # Detection: use DiscreteEmpirical if fitted, else fall back to Normal
        llr_p = if has_disc
            log(max(pdf(disc_int_det, y_p[i]), 1e-300)) -
            log(max(pdf(disc_bg_det, y_p[i]), 1e-300))
        else
            logpdf(Normal(means_int.presence, std_int.presence), y_p[i]) -
            logpdf(Normal(means_bg.presence, std_bg.presence), y_p[i])
        end

        # Monotonicity correction: if value exceeds interaction mean in the
        # positive direction, don't let the LLR go negative (which would penalize
        # extreme positive evidence due to the narrow interaction component)
        if y_e[i] > means_int.enrichment
            llr_e = max(llr_e, 0.0)
        end
        if y_c[i] > means_int.correlation
            llr_c = max(llr_c, 0.0)
        end
        if y_p[i] > means_int.presence
            llr_p = max(llr_p, 0.0)
        end

        # Total log-likelihood ratio, clamped to match copula model dynamic range
        # Bounds of [-46, 46] correspond to BF range ~[1e-20, 1e20]
        total_llr = clamp(llr_e + llr_c + llr_p, -46.0, 46.0)

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

Returns sum-of-Gaussians class-specific parameters for 2-component LatentClassResult.
Uses sum of means and RSS of sigmas (independence assumption) to match log(combined_BF) histogram.
"""
function extract_lc_class_parameters(em_result)
    means_bg = em_result.means["background"]
    means_int = em_result.means["interaction"]
    std_bg = em_result.std_devs["background"]
    std_int = em_result.std_devs["interaction"]

    # Sum-of-Gaussians: sum means, RSS sigmas (independence assumption)
    bg_sum_mu = means_bg.enrichment + means_bg.correlation + means_bg.presence
    bg_sum_sigma = sqrt(std_bg.enrichment^2 + std_bg.correlation^2 + std_bg.presence^2)
    bg_sum_precision = 1.0 / bg_sum_sigma^2

    int_sum_mu = means_int.enrichment + means_int.correlation + means_int.presence
    int_sum_sigma = sqrt(std_int.enrichment^2 + std_int.correlation^2 + std_int.presence^2)
    int_sum_precision = 1.0 / int_sum_sigma^2

    return Dict(
        "background" => (mu = bg_sum_mu, sigma = bg_sum_sigma, precision = bg_sum_precision),
        "interaction" => (mu = int_sum_mu, sigma = int_sum_sigma, precision = int_sum_precision)
    )
end

# ============================================================
# 3b. 3-Component EM Algorithm
# ============================================================

"""
    _restart_family(restart_idx, n_restarts) -> Symbol

Allocate EM restarts across three H1 enrichment families for diversification.
- First third: :gamma
- Second third: :lognormal
- Last third: :weibull
"""
function _restart_family(restart_idx::Int, n_restarts::Int)::Symbol
    n = max(n_restarts, 3)
    third = ceil(Int, n / 3)
    if restart_idx <= third
        return :gamma
    elseif restart_idx <= 2 * third
        return :lognormal
    else
        return :weibull
    end
end

"""
    initialize_3c_em(y_e, y_c, y_p, restart_idx; n_restarts=20)

Generate initial parameters for the 3-component EM. Mixes initialization strategies:
- Restarts 1-5: Quantile-based splits on enrichment dimension
- Restarts 6-10: K-means-inspired clustering on enrichment
- Restarts 11+: Random initialization

Returns a NamedTuple of initial parameters.
"""
function initialize_3c_em(y_e::Vector{Float64}, y_c::Vector{Float64}, y_p::Vector{Float64},
                           restart_idx::Int; n_restarts::Int=20)
    n = length(y_e)
    rng = Random.MersenneTwister(restart_idx * 7919 + 42)

    if restart_idx <= 5
        # Quantile-based splits
        frac_lo = 0.05 + 0.08 * restart_idx  # 0.13..0.45
        frac_hi = 0.95 - 0.08 * restart_idx  # 0.87..0.55
        frac_lo = clamp(frac_lo, 0.05, 0.5)
        frac_hi = clamp(frac_hi, 0.5, 0.95)
        q_lo = quantile(y_e, frac_lo)
        q_hi = quantile(y_e, frac_hi)

        mask_h0 = y_e .<= q_lo
        mask_ag = (y_e .> q_lo) .& (y_e .<= q_hi)
        mask_h1 = y_e .> q_hi

        # Compute means from data splits (with fallbacks)
        mu_e0 = sum(mask_h0) > 0 ? mean(y_e[mask_h0]) : -1.0
        mu_c0 = sum(mask_h0) > 0 ? mean(y_c[mask_h0]) : 0.0
        mu_p0 = sum(mask_h0) > 0 ? mean(y_p[mask_h0]) : 0.0

        mu_ea = sum(mask_ag) > 0 ? mean(y_e[mask_ag]) : 0.0
        mu_ca = sum(mask_ag) > 0 ? mean(y_c[mask_ag]) : 0.0
        mu_pa = sum(mask_ag) > 0 ? mean(y_p[mask_ag]) : 0.0

        mu_c1 = sum(mask_h1) > 0 ? mean(y_c[mask_h1]) : 3.0
        mu_p1 = sum(mask_h1) > 0 ? mean(y_p[mask_h1]) : 3.5

        sig_e0, sig_c0, sig_p0 = 1.5, 1.2, 1.3
        sig_ea, sig_ca, sig_pa = 1.0, 0.8, 0.9
        alpha_e1, theta_e1 = 2.0, 1.5  # Shifted Gamma: mode=(alpha-1)*theta=1.5, mean=alpha*theta=3.0
        sig_c1, sig_p1 = 0.9, 0.8

    elseif restart_idx <= 10
        # K-means-inspired on enrichment dimension
        # Random initial centers
        sorted_e = sort(y_e)
        c0 = sorted_e[max(1, round(Int, n * (0.15 + 0.05 * randn(rng))))]
        c1 = sorted_e[max(1, min(n, round(Int, n * (0.5 + 0.1 * randn(rng)))))]
        c2 = sorted_e[max(1, min(n, round(Int, n * (0.9 + 0.05 * randn(rng)))))]
        centers = [c0, c1, c2]

        # Run 5 iterations of k-means on enrichment
        for _ in 1:5
            assignments = [argmin([abs(y_e[i] - centers[k]) for k in 1:3]) for i in 1:n]
            for k in 1:3
                members = findall(assignments .== k)
                if length(members) > 0
                    centers[k] = mean(y_e[members])
                end
            end
        end

        # Sort centers
        perm = sortperm(centers)
        centers = centers[perm]

        assignments = [argmin([abs(y_e[i] - centers[k]) for k in 1:3]) for i in 1:n]

        mu_e0 = centers[1]
        mu_ea = centers[2]

        # Compute means for other dimensions based on assignments
        m0 = findall(assignments .== 1)
        m1 = findall(assignments .== 3)
        ma = findall(assignments .== 2)

        mu_c0 = length(m0) > 0 ? mean(y_c[m0]) : 0.0
        mu_p0 = length(m0) > 0 ? mean(y_p[m0]) : 0.0
        mu_ca = length(ma) > 0 ? mean(y_c[ma]) : 0.0
        mu_pa = length(ma) > 0 ? mean(y_p[ma]) : 0.0
        mu_c1 = length(m1) > 0 ? mean(y_c[m1]) : 3.0
        mu_p1 = length(m1) > 0 ? mean(y_p[m1]) : 3.5

        sig_e0 = length(m0) > 1 ? std(y_e[m0]) : 1.5
        sig_c0 = length(m0) > 1 ? std(y_c[m0]) : 1.2
        sig_p0 = length(m0) > 1 ? std(y_p[m0]) : 1.3
        sig_ea = length(ma) > 1 ? std(y_e[ma]) : 1.0
        sig_ca = length(ma) > 1 ? std(y_c[ma]) : 0.8
        sig_pa = length(ma) > 1 ? std(y_p[ma]) : 0.9
        alpha_e1, theta_e1 = 2.0, 1.5  # Shifted Gamma for enrichment H1
        sig_c1 = length(m1) > 1 ? std(y_c[m1]) : 0.9
        sig_p1 = length(m1) > 1 ? std(y_p[m1]) : 0.8

    else
        # Random initialization
        mu_e0 = randn(rng) * 0.5 - 1.0
        mu_c0 = randn(rng) * 0.5 - 0.5
        mu_p0 = randn(rng) * 0.5 - 0.5

        mu_ea = randn(rng) * 0.3
        mu_ca = randn(rng) * 0.3
        mu_pa = randn(rng) * 0.3

        mu_c1 = randn(rng) * 0.5 + 2.5
        mu_p1 = randn(rng) * 0.5 + 3.0

        sig_e0 = 0.5 + 2.5 * rand(rng)
        sig_c0 = 0.5 + 2.5 * rand(rng)
        sig_p0 = 0.5 + 2.5 * rand(rng)
        sig_ea = 0.5 + 2.0 * rand(rng)
        sig_ca = 0.5 + 2.0 * rand(rng)
        sig_pa = 0.5 + 2.0 * rand(rng)
        alpha_e1 = 1.0 + 3.0 * rand(rng)  # Shifted Gamma alpha in [1, 4]
        theta_e1 = 0.5 + 3.0 * rand(rng)  # Shifted Gamma theta in [0.5, 3.5]
        sig_c1 = 0.5 + 2.0 * rand(rng)
        sig_p1 = 0.5 + 2.0 * rand(rng)
    end

    # COMP-02: Anchor agnostic enrichment mean at 0.0 for all initialization branches
    mu_ea = 0.0

    # Ensure ordering: mu_e0 < mu_ea (H1 enrichment uses shifted Gamma, not a Normal mean)
    # Only order H0 and agnostic — mu_ea is 0.0, so this checks mu_e0 > 0.0
    if mu_e0 > mu_ea
        mu_e0, mu_ea = mu_ea, mu_e0
        mu_c0, mu_ca = mu_ca, mu_c0
        mu_p0, mu_pa = mu_pa, mu_p0
        sig_e0, sig_ea = sig_ea, sig_e0
        sig_c0, sig_ca = sig_ca, sig_c0
        sig_p0, sig_pa = sig_pa, sig_p0
        # Re-anchor mu_ea after swap
        mu_ea = 0.0
    end

    # ---- Sum-of-means ordering: sum(H0) < sum(agnostic) < sum(H1 effective) ----
    sum_h0_init = mu_e0 + mu_c0 + mu_p0
    sum_ag_init = mu_ea + mu_ca + mu_pa
    sum_h1_init = (alpha_e1 * theta_e1 + JEFFREYS_SHIFT) + mu_c1 + mu_p1
    if sum_h0_init >= sum_ag_init
        # Full label swap between H0 and agnostic (same as enrichment swap above)
        mu_e0, mu_ea = mu_ea, mu_e0
        mu_c0, mu_ca = mu_ca, mu_c0
        mu_p0, mu_pa = mu_pa, mu_p0
        sig_e0, sig_ea = sig_ea, sig_e0
        sig_c0, sig_ca = sig_ca, sig_c0
        sig_p0, sig_pa = sig_pa, sig_p0
        # Re-anchor mu_ea after swap
        mu_ea = 0.0
    end
    # Recompute after potential swap
    sum_ag_init = mu_ea + mu_ca + mu_pa
    sum_h1_init = (alpha_e1 * theta_e1 + JEFFREYS_SHIFT) + mu_c1 + mu_p1
    if sum_ag_init >= sum_h1_init
        excess = sum_ag_init - sum_h1_init + 0.1
        # Distribute excess into free agnostic dimensions only (mu_ea is anchored at 0.0)
        mu_ca -= excess / 2.0
        mu_pa -= excess / 2.0
        mu_ea = 0.0  # Re-anchor after any sum-of-means adjustment
        # Re-apply agnostic clamp for free dims
        mu_ca = clamp(mu_ca, -0.5, 0.5)
        mu_pa = clamp(mu_pa, -0.5, 0.5)
    end

    # Gamma mode for backward-compat enrichment mean display
    _gamma_mode_e1 = alpha_e1 >= 1.0 ? (alpha_e1 - 1.0) * theta_e1 + JEFFREYS_SHIFT : JEFFREYS_SHIFT

    # Data-space SD for H1 enrichment
    # At initialization, family is always :gamma (BIC selection happens at iter 5)
    _init_h1_sd = sqrt(alpha_e1) * theta_e1

    return (
        pi = [0.85, 0.10, 0.05],
        means = Dict(
            "background"  => (enrichment=mu_e0, correlation=mu_c0, presence=mu_p0),
            "agnostic"    => (enrichment=mu_ea, correlation=mu_ca, presence=mu_pa),
            "interaction" => (enrichment=_gamma_mode_e1, correlation=mu_c1, presence=mu_p1)
        ),
        std_devs = Dict(
            "background"  => (enrichment=max(sig_e0, 0.1), correlation=max(sig_c0, 0.1), presence=max(sig_p0, 0.1)),
            "agnostic"    => (enrichment=max(sig_ea, 0.1), correlation=max(sig_ca, 0.1), presence=max(sig_pa, 0.1)),
            "interaction" => (enrichment=_init_h1_sd, correlation=max(sig_c1, 0.1), presence=max(sig_p1, 0.1))
        ),
        alpha_e1 = alpha_e1,
        theta_e1 = theta_e1,
        h1_enrichment_sd = _init_h1_sd
    )
end

"""
    _compute_ll_3c(y_enrich, y_corr, y_pres, disc_H0, disc_ag, disc_H1,
                   h0_enrich_dist, mu_c0, sig_c0, mu_ea, sig_ea, mu_ca, sig_ca,
                   mu_c1, sig_c1, pi_k, h1_enrich_dist, JEFFREYS_SHIFT_val)

Compute the observed-data log-likelihood for the 3-component mixture model.
Uses log-sum-exp trick for numerical stability. Extracted as helper to avoid duplication
between E-step LL tracking and end-of-iteration LL computation.

`h0_enrich_dist` is a distribution object: either `Normal(mu, sigma)` or
`LocationScale(mu, sigma, TDist(nu))` depending on BIC selection at iter 5.
"""
function _compute_ll_3c(y_enrich::Vector{Float64}, y_corr::Vector{Float64}, y_pres::Vector{Float64},
                         disc_H0, disc_ag, disc_H1,
                         h0_enrich_dist, mu_c0::Float64, sig_c0::Float64,
                         mu_ea::Float64, sig_ea::Float64, mu_ca::Float64, sig_ca::Float64,
                         mu_c1::Float64, sig_c1::Float64, pi_k::Vector{Float64},
                         h1_enrich_dist, JEFFREYS_SHIFT_val::Float64)
    ll = 0.0
    n = length(y_enrich)
    for i in 1:n
        ll_det_h0 = log(max(pdf(disc_H0, y_pres[i]), 1e-300))
        ll_det_ag = log(max(pdf(disc_ag, y_pres[i]), 1e-300))
        ll_det_h1 = log(max(pdf(disc_H1, y_pres[i]), 1e-300))
        ll_h0 = logpdf(h0_enrich_dist, y_enrich[i]) +
                logpdf(Normal(mu_c0, sig_c0), y_corr[i]) + ll_det_h0 + log(pi_k[1] + 1e-300)
        ll_ag = logpdf(Normal(mu_ea, sig_ea), y_enrich[i]) +
                logpdf(Normal(mu_ca, sig_ca), y_corr[i]) + ll_det_ag + log(pi_k[2] + 1e-300)
        ll_h1_e = _h1_enrichment_logdensity(y_enrich[i], h1_enrich_dist, JEFFREYS_SHIFT_val, SIGMOID_STEEPNESS)
        ll_h1 = ll_h1_e + logpdf(Normal(mu_c1, sig_c1), y_corr[i]) + ll_det_h1 + log(pi_k[3] + 1e-300)
        mx = max(ll_h0, ll_ag, ll_h1)
        if isfinite(mx)
            ll += mx + log(exp(ll_h0 - mx) + exp(ll_ag - mx) + exp(ll_h1 - mx))
        end
    end
    return ll
end

"""
    _snapshot_params(mu_e0, sig_e0, mu_c0, sig_c0, mu_p0, sig_p0,
                     mu_ea, sig_ea, mu_ca, sig_ca, mu_pa, sig_pa,
                     alpha_e1, theta_e1, mu_c1, sig_c1, mu_p1, sig_p1,
                     pi_k, gamma, disc_H0, disc_ag, disc_H1)

Create an immutable snapshot of all EM parameters for step-halving guard.
Scalars are captured by value. Mutable arrays (`pi_k`, `gamma`) are `copy()`-ed.
`DiscreteEmpirical` objects are immutable structs — direct reference is safe
because the M-step creates NEW objects via `_fit_discrete_empirical_weighted`.
"""
function _snapshot_params(
    mu_e0, sig_e0, mu_c0, sig_c0, mu_p0, sig_p0,
    mu_ea, sig_ea, mu_ca, sig_ca, mu_pa, sig_pa,
    alpha_e1, theta_e1, mu_c1, sig_c1, mu_p1, sig_p1,
    pi_k, gamma, disc_H0, disc_ag, disc_H1
)
    return (
        mu_e0=mu_e0, sig_e0=sig_e0, mu_c0=mu_c0, sig_c0=sig_c0, mu_p0=mu_p0, sig_p0=sig_p0,
        mu_ea=mu_ea, sig_ea=sig_ea, mu_ca=mu_ca, sig_ca=sig_ca, mu_pa=mu_pa, sig_pa=sig_pa,
        alpha_e1=alpha_e1, theta_e1=theta_e1, mu_c1=mu_c1, sig_c1=sig_c1, mu_p1=mu_p1, sig_p1=sig_p1,
        pi_k=copy(pi_k), gamma=copy(gamma),
        disc_H0=disc_H0, disc_ag=disc_ag, disc_H1=disc_H1
    )
end

"""
    _apply_constraints!(mu_e0, sig_e0, ..., pi_k, gamma, sigma_floor, max_sigma_e, max_sigma_c, max_sigma_p, current_family)

Apply all post-M-step constraints in canonical order (EM-03):
1. Sigma floors (all components)
2. Sigma caps (data-dependent absolute caps, all components)
3. Mean constraints — agnostic anchoring [-0.5, 0.5]
4. Agnostic variance cap (≤ 1.5× H0)
5. H1 shape/scale clamps (alpha_e1, theta_e1)
6. H1 data-space SD cap
7. Label ordering (mu_e0 < mu_ea)
8. Sum-of-means ordering (sum(H0) < sum(agnostic) < sum(H1 effective))

Scalars are passed by value and returned in a NamedTuple.
`pi_k` and `gamma` are mutated in-place (label swap).
"""
function _apply_constraints!(
    mu_e0, sig_e0, mu_c0, sig_c0, mu_p0, sig_p0,
    mu_ea, sig_ea, mu_ca, sig_ca, mu_pa, sig_pa,
    alpha_e1, theta_e1, mu_c1, sig_c1, mu_p1, sig_p1,
    pi_k, gamma,
    sigma_floor, max_sigma_e, max_sigma_c, max_sigma_p, current_family
)
    # --- 1. Sigma floors (all components) ---
    sig_e0 = max(sig_e0, sigma_floor)
    sig_c0 = max(sig_c0, sigma_floor)
    sig_p0 = max(sig_p0, sigma_floor)
    sig_ea = max(sig_ea, sigma_floor)
    sig_ca = max(sig_ca, sigma_floor)
    sig_pa = max(sig_pa, sigma_floor)
    sig_c1 = max(sig_c1, sigma_floor)
    sig_p1 = max(sig_p1, sigma_floor)

    # --- 2. Sigma caps (data-dependent absolute caps, all components) ---
    sig_e0 = min(sig_e0, max_sigma_e)
    sig_c0 = min(sig_c0, max_sigma_c)
    sig_p0 = min(sig_p0, max_sigma_p)
    sig_ea = min(sig_ea, max_sigma_e)
    sig_ca = min(sig_ca, max_sigma_c)
    sig_pa = min(sig_pa, max_sigma_p)
    sig_c1 = min(sig_c1, max_sigma_c)
    sig_p1 = min(sig_p1, max_sigma_p)

    # --- 3. Mean constraints — agnostic anchoring (COMP-02) ---
    mu_ea = 0.0  # COMP-02: Agnostic enrichment mean anchored at zero
    mu_ca = clamp(mu_ca, -0.5, 0.5)
    mu_pa = clamp(mu_pa, -0.5, 0.5)

    # --- 4. Agnostic variance cap: ≤ 1.5× H0 ---
    sig_ea = min(sig_ea, 1.5 * sig_e0)
    sig_ca = min(sig_ca, 1.5 * sig_c0)
    sig_pa = min(sig_pa, 1.5 * sig_p0)

    # --- 5. H1 shape/scale clamps ---
    alpha_e1 = clamp(alpha_e1, 0.5, 50.0)
    theta_e1 = clamp(theta_e1, 0.05, 20.0)

    # --- 6. H1 data-space SD cap ---
    _h1_sd_raw = if current_family == :lognormal
        std(LogNormal(alpha_e1, theta_e1))
    elseif current_family == :weibull
        std(Weibull(alpha_e1, theta_e1))
    else  # :gamma
        sqrt(alpha_e1) * theta_e1
    end
    _sd_target = clamp(_h1_sd_raw, 2.0, max_sigma_e)
    if abs(_h1_sd_raw - _sd_target) > 1e-10
        theta_e1 *= (_sd_target / _h1_sd_raw)
        theta_e1 = clamp(theta_e1, 0.05, 20.0)
    end

    # --- 7. Label ordering: mu_e0 < 0.0 (mu_ea is fixed at 0.0) ---
    if mu_e0 > 0.0
        # H0 mean drifted positive; swap H0 <-> Agnostic
        mu_e0, mu_ea = mu_ea, mu_e0
        mu_c0, mu_ca = mu_ca, mu_c0
        mu_p0, mu_pa = mu_pa, mu_p0
        sig_e0, sig_ea = sig_ea, sig_e0
        sig_c0, sig_ca = sig_ca, sig_c0
        sig_p0, sig_pa = sig_pa, sig_p0
        pi_k[1], pi_k[2] = pi_k[2], pi_k[1]
        gamma[:, 1], gamma[:, 2] = gamma[:, 2], gamma[:, 1]
        # Re-anchor mu_ea after swap
        mu_ea = 0.0
    end

    # --- 8. Sum-of-means ordering: sum(H0) < sum(agnostic) < sum(H1 effective) ---
    _h1_eff_mean = if current_family == :lognormal
        mean(LogNormal(alpha_e1, theta_e1)) + JEFFREYS_SHIFT
    elseif current_family == :weibull
        mean(Weibull(alpha_e1, theta_e1)) + JEFFREYS_SHIFT
    else
        alpha_e1 * theta_e1 + JEFFREYS_SHIFT  # Gamma mean
    end
    sum_h0 = mu_e0 + mu_c0 + mu_p0
    sum_ag = mu_ea + mu_ca + mu_pa
    sum_h1 = _h1_eff_mean + mu_c1 + mu_p1
    if sum_h0 >= sum_ag
        # Full label swap between H0 and agnostic
        mu_e0, mu_ea = mu_ea, mu_e0
        mu_c0, mu_ca = mu_ca, mu_c0
        mu_p0, mu_pa = mu_pa, mu_p0
        sig_e0, sig_ea = sig_ea, sig_e0
        sig_c0, sig_ca = sig_ca, sig_c0
        sig_p0, sig_pa = sig_pa, sig_p0
        pi_k[1], pi_k[2] = pi_k[2], pi_k[1]
        gamma[:, 1], gamma[:, 2] = gamma[:, 2], gamma[:, 1]
        # Re-anchor mu_ea after swap
        mu_ea = 0.0
    end
    # Recompute after potential swap
    sum_ag = mu_ea + mu_ca + mu_pa
    sum_h1 = _h1_eff_mean + mu_c1 + mu_p1
    if sum_ag >= sum_h1
        # Distribute excess into free agnostic dimensions only (mu_ea is anchored at 0.0)
        excess = sum_ag - sum_h1 + 0.1
        mu_ca -= excess / 2.0
        mu_pa -= excess / 2.0
        mu_ea = 0.0  # Re-anchor after any sum-of-means adjustment
        # Re-apply agnostic clamp for free dims [-0.5, 0.5]
        mu_ca = clamp(mu_ca, -0.5, 0.5)
        mu_pa = clamp(mu_pa, -0.5, 0.5)
    end

    return (
        mu_e0=mu_e0, sig_e0=sig_e0, mu_c0=mu_c0, sig_c0=sig_c0, mu_p0=mu_p0, sig_p0=sig_p0,
        mu_ea=mu_ea, sig_ea=sig_ea, mu_ca=mu_ca, sig_ca=sig_ca, mu_pa=mu_pa, sig_pa=sig_pa,
        alpha_e1=alpha_e1, theta_e1=theta_e1, mu_c1=mu_c1, sig_c1=sig_c1, mu_p1=mu_p1, sig_p1=sig_p1
    )
end

"""
    fit_gaussian_mixture_em_3c(y_enrich, y_corr, y_pres;
                                n_iterations=200, alpha_prior=[5.0, 2.0, 1.0],
                                tol=1e-6, sigma_floor=0.1, init_params=nothing,
                                h1_family=:gamma)

Fit a 3-component Gaussian mixture model (H0/agnostic/H1) using EM algorithm.

The agnostic component absorbs proteins with log-BF near 0, preventing
H1 contamination. Includes enrichment gate (after iter 5), BIC-based family
selection at iteration 5, label ordering constraint, variance ratio caps,
and agnostic mean anchoring.

`h1_family` controls the initial H1 enrichment marginal family (:gamma, :lognormal, :weibull).
At iteration 5, BIC selection may switch to the best-fitting family.

Returns a NamedTuple with fitted parameters, responsibilities (n x 3),
convergence info, and h1_family/h1_bic_scores from BIC selection.
"""
function fit_gaussian_mixture_em_3c(y_enrich::Vector{Float64}, y_corr::Vector{Float64}, y_pres::Vector{Float64};
                                     n_iterations::Int=200, alpha_prior::Vector{Float64}=[5.0, 2.0, 1.0],
                                     tol::Float64=1e-6, sigma_floor::Float64=0.1,
                                     init_params=nothing, h1_family::Symbol=:gamma,
                                     skip_bic_selection::Bool=false)
    n = length(y_enrich)

    # Track current H1 family and BIC scores
    current_family = h1_family
    bic_scores = Dict{Symbol, Float64}(:gamma => Inf, :lognormal => Inf, :weibull => Inf)
    bic_selection_done = skip_bic_selection

    # H0 Student-t tracking (COMP-01)
    nu_h0 = 0.0          # 0.0 = Normal (default)
    use_student_t = false
    h0_bic_done = false

    # Initialize parameters
    if init_params !== nothing
        pi_k = copy(init_params.pi)
        mu_e0 = init_params.means["background"].enrichment
        mu_c0 = init_params.means["background"].correlation
        mu_p0 = init_params.means["background"].presence
        sig_e0 = init_params.std_devs["background"].enrichment
        sig_c0 = init_params.std_devs["background"].correlation
        sig_p0 = init_params.std_devs["background"].presence

        mu_ea = init_params.means["agnostic"].enrichment
        mu_ca = init_params.means["agnostic"].correlation
        mu_pa = init_params.means["agnostic"].presence
        sig_ea = init_params.std_devs["agnostic"].enrichment
        sig_ca = init_params.std_devs["agnostic"].correlation
        sig_pa = init_params.std_devs["agnostic"].presence

        # H1 enrichment: params from init (family-agnostic: param1, param2)
        alpha_e1 = hasproperty(init_params, :alpha_e1) ? init_params.alpha_e1 : 2.0
        theta_e1 = hasproperty(init_params, :theta_e1) ? init_params.theta_e1 : 1.5
        mu_c1 = init_params.means["interaction"].correlation
        mu_p1 = init_params.means["interaction"].presence
        sig_c1 = init_params.std_devs["interaction"].correlation
        sig_p1 = init_params.std_devs["interaction"].presence
    else
        # Default initialization
        pi_k = [0.85, 0.10, 0.05]
        mu_e0, sig_e0 = -0.5, 1.5
        mu_c0, sig_c0 = -0.3, 1.2
        mu_p0, sig_p0 = -0.3, 1.3
        mu_ea, sig_ea = 0.0, 1.0
        mu_ca, sig_ca = 0.0, 0.8
        mu_pa, sig_pa = 0.0, 0.9
        alpha_e1, theta_e1 = 2.0, 1.5  # H1 enrichment params (family-agnostic)
        mu_c1, sig_c1 = 3.0, 0.9
        mu_p1, sig_p1 = 3.5, 0.8
    end

    # COMP-02: Anchor agnostic enrichment mean at 0.0
    mu_ea = 0.0
    # Data-driven initial sigma_ea: robust IQR of proteins near zero enrichment
    near_zero = y_enrich[abs.(y_enrich) .< 1.0]
    if length(near_zero) >= 5
        # Use explicit quantile computation (no StatsBase.iqr dependency)
        sig_ea = max((quantile(near_zero, 0.75) - quantile(near_zero, 0.25)) / 1.349, sigma_floor)
    else
        sig_ea = 1.0
    end

    # Storage for responsibilities and log-likelihood
    gamma = zeros(n, 3)  # Responsibilities: H0, agnostic, H1
    log_liks = Float64[]
    # Per-step LL tracking
    ll_trace_e = Float64[]
    ll_trace_m = Float64[]
    n_step_halving_reverts = 0
    # BIC soft restart tracking
    family_switch_iter = nothing
    convergence_skip_until = 0
    # H1 effective mean for ordering (updated each iteration, accessible after loop)
    _h1_eff_mean::Float64 = alpha_e1 * theta_e1 + JEFFREYS_SHIFT

    # DiscreteEmpirical for detection dimension
    # Initialize from full data (uniform weights); updated each M-step per component
    disc_H0 = DiscreteEmpirical(y_pres)
    disc_ag = DiscreteEmpirical(y_pres)
    disc_H1 = DiscreteEmpirical(y_pres)

    # ---- Data-dependent sigma caps (computed once from input data) ----
    _iqr_e = quantile(y_enrich, 0.75) - quantile(y_enrich, 0.25)
    _iqr_c = quantile(y_corr, 0.75) - quantile(y_corr, 0.25)
    _iqr_p = quantile(y_pres, 0.75) - quantile(y_pres, 0.25)
    max_sigma_e = clamp(_iqr_e * 2.0, 2.0, 20.0)
    max_sigma_c = clamp(_iqr_c * 2.0, 2.0, 20.0)
    max_sigma_p = clamp(_iqr_p * 2.0, 2.0, 20.0)

    for iter in 1:n_iterations
        # ---- E-step: Compute responsibilities ----
        # Construct H1 enrichment distribution once per iteration (family-dispatched)
        h1_enrich_dist = if current_family == :lognormal
            LogNormal(alpha_e1, theta_e1)
        elseif current_family == :weibull
            Weibull(alpha_e1, theta_e1)
        else  # :gamma (default)
            Gamma(alpha_e1, theta_e1)
        end

        # H0 enrichment distribution: Student-t or Normal (COMP-01)
        h0_enrich_dist = use_student_t ?
            LocationScale(mu_e0, sig_e0, TDist(nu_h0)) :
            Normal(mu_e0, sig_e0)

        for i in 1:n
            # Detection log-likelihoods via DiscreteEmpirical
            ll_det_h0 = log(max(pdf(disc_H0, y_pres[i]), 1e-300))
            ll_det_ag = log(max(pdf(disc_ag, y_pres[i]), 1e-300))
            ll_det_h1 = log(max(pdf(disc_H1, y_pres[i]), 1e-300))

            # Compute log-unnormalized responsibilities (all in log-space)
            log_gamma_h0 = logpdf(h0_enrich_dist, y_enrich[i]) +
                           logpdf(Normal(mu_c0, sig_c0), y_corr[i]) +
                           ll_det_h0 + log(pi_k[1] + 1e-300)
            log_gamma_ag = logpdf(Normal(mu_ea, sig_ea), y_enrich[i]) +
                           logpdf(Normal(mu_ca, sig_ca), y_corr[i]) +
                           ll_det_ag + log(pi_k[2] + 1e-300)
            ll_h1_e = _h1_enrichment_logdensity(y_enrich[i], h1_enrich_dist, JEFFREYS_SHIFT, SIGMOID_STEEPNESS)
            log_gamma_h1 = ll_h1_e +
                           logpdf(Normal(mu_c1, sig_c1), y_corr[i]) +
                           ll_det_h1 + log(pi_k[3] + 1e-300)

            # Monotonicity correction for E-step:
            # If enrichment value exceeds H1 effective mean and H1 density < H0 density,
            # floor H1 enrichment density at H0 level to prevent thin-tail penalty
            if y_enrich[i] > _h1_eff_mean && ll_h1_e > -100.0
                ld_h0_e = logpdf(h0_enrich_dist, y_enrich[i])
                if ll_h1_e < ld_h0_e
                    log_gamma_h1 = log_gamma_h1 - ll_h1_e + ld_h0_e
                end
            end

            # Monotonicity correction for correlation dimension
            if y_corr[i] > mu_c1
                ld_h0_c = logpdf(Normal(mu_c0, sig_c0), y_corr[i])
                ld_h1_c = logpdf(Normal(mu_c1, sig_c1), y_corr[i])
                if ld_h1_c < ld_h0_c
                    log_gamma_h1 = log_gamma_h1 - ld_h1_c + ld_h0_c
                end
            end

            # Log-sum-exp normalization: subtract max for numerical stability, then exponentiate
            max_log_gamma = max(log_gamma_h0, log_gamma_ag, log_gamma_h1)

            if !isfinite(max_log_gamma)
                # All components return -Inf (zero density) -- assign uniform
                gamma[i, 1] = 1.0/3.0
                gamma[i, 2] = 1.0/3.0
                gamma[i, 3] = 1.0/3.0
            else
                # Subtract max, exponentiate, normalize -- standard log-sum-exp trick
                exp_h0 = exp(log_gamma_h0 - max_log_gamma)
                exp_ag = exp(log_gamma_ag - max_log_gamma)
                exp_h1 = exp(log_gamma_h1 - max_log_gamma)
                log_sum_exp = max_log_gamma + log(exp_h0 + exp_ag + exp_h1)

                # Responsibilities = exp(log_gamma_k - log_sum_exp)
                gamma[i, 1] = exp(log_gamma_h0 - log_sum_exp)
                gamma[i, 2] = exp(log_gamma_ag - log_sum_exp)
                gamma[i, 3] = exp(log_gamma_h1 - log_sum_exp)

                # Guard: ensure valid probability simplex
                row_sum = gamma[i, 1] + gamma[i, 2] + gamma[i, 3]
                if !isfinite(row_sum) || row_sum < 1e-300
                    gamma[i, 1] = 1.0/3.0
                    gamma[i, 2] = 1.0/3.0
                    gamma[i, 3] = 1.0/3.0
                end
            end
        end

        # Enrichment gate REMOVED: LocationShifted structural zero
        # (y_enrich[i] > JEFFREYS_SHIFT ? logpdf(...) : -Inf) in E-step above
        # provides the same constraint without depleting H1 membership.

        # Per-step LL tracking: LL after E-step
        ll_after_e_val = _compute_ll_3c(y_enrich, y_corr, y_pres, disc_H0, disc_ag, disc_H1,
            h0_enrich_dist, mu_c0, sig_c0, mu_ea, sig_ea, mu_ca, sig_ca,
            mu_c1, sig_c1, pi_k, h1_enrich_dist, JEFFREYS_SHIFT)
        push!(ll_trace_e, ll_after_e_val)

        # ---- BIC selection at iteration 5 ----
        # Run once after the enrichment gate activates to pick the best H1 family
        if iter == 5 && !bic_selection_done
            above_c = findall(y_enrich .> JEFFREYS_SHIFT)
            if length(above_c) >= 5
                w_above = gamma[above_c, 3]
                shifted = max.(y_enrich[above_c] .- JEFFREYS_SHIFT, 1e-10)
                selected_family, bic_scores = _select_h1_family_bic(shifted, w_above, current_family)
                if selected_family != current_family
                    # Reinitialize H1 enrichment params for new family
                    p1, p2 = _reinit_h1_params_for_family(selected_family, shifted, w_above)
                    alpha_e1 = clamp(p1, 0.5, 50.0)
                    theta_e1 = clamp(p2, 0.05, 20.0)
                    # Data-space SD cap after BIC switch
                    _h1_sd_bic = if selected_family == :lognormal
                        std(LogNormal(alpha_e1, theta_e1))
                    elseif selected_family == :weibull
                        std(Weibull(alpha_e1, theta_e1))
                    else
                        sqrt(alpha_e1) * theta_e1
                    end
                    _sd_target_bic = clamp(_h1_sd_bic, 2.0, max_sigma_e)
                    if abs(_h1_sd_bic - _sd_target_bic) > 1e-10
                        theta_e1 *= (_sd_target_bic / _h1_sd_bic)
                        theta_e1 = clamp(theta_e1, 0.05, 20.0)
                    end
                    current_family = selected_family
                    @debug "BIC family switch at iter 5: $(h1_family) -> $(current_family)"
                    # Soft restart: reset LL baseline
                    _h0_dist_bic = use_student_t ?
                        LocationScale(mu_e0, sig_e0, TDist(nu_h0)) :
                        Normal(mu_e0, sig_e0)
                    ll_baseline = _compute_ll_3c(y_enrich, y_corr, y_pres, disc_H0, disc_ag, disc_H1,
                        _h0_dist_bic, mu_c0, sig_c0, mu_ea, sig_ea, mu_ca, sig_ca,
                        mu_c1, sig_c1, pi_k,
                        (current_family == :lognormal ? LogNormal(alpha_e1, theta_e1) :
                         current_family == :weibull ? Weibull(alpha_e1, theta_e1) :
                         Gamma(alpha_e1, theta_e1)),
                        JEFFREYS_SHIFT)
                    push!(log_liks, ll_baseline)
                    push!(ll_trace_e, ll_baseline)
                    push!(ll_trace_m, ll_baseline)
                    family_switch_iter = iter
                    convergence_skip_until = iter + 3
                end
            end
            bic_selection_done = true
        end

        # ---- H0 enrichment Student-t BIC selection at iter 5 (COMP-01) ----
        # Guard: if n_iterations < 5, this block never executes and Normal is used (correct fallback)
        if iter == 5 && !h0_bic_done
            h0_mask = findall(gamma[:, 1] .> 0.5)
            if length(h0_mask) >= 10
                h0_data = y_enrich[h0_mask]
                w_h0 = gamma[h0_mask, 1]
                # Weighted Normal fit (baseline)
                mu_fit = sum(w_h0 .* h0_data) / sum(w_h0)
                sig_fit = sqrt(sum(w_h0 .* (h0_data .- mu_fit).^2) / sum(w_h0))
                sig_fit = max(sig_fit, sigma_floor)
                ll_normal = sum(w_h0 .* logpdf.(Normal(mu_fit, sig_fit), h0_data))
                bic_normal = -2.0 * ll_normal + 2.0 * log(length(h0_mask))  # k=2 free params
                best_nu = 0  # 0 = Normal
                best_bic = bic_normal
                for nu_cand in [3, 5, 7, 10]
                    cand_dist = LocationScale(mu_fit, sig_fit, TDist(Float64(nu_cand)))
                    ll_t = sum(w_h0 .* logpdf.(cand_dist, h0_data))
                    bic_t = -2.0 * ll_t + 2.0 * log(length(h0_mask))  # same k=2 (nu is grid, not free)
                    if bic_t < best_bic - 2.0  # BIC margin > 2
                        best_bic = bic_t
                        best_nu = nu_cand
                    end
                end
                nu_h0 = best_nu > 0 ? Float64(best_nu) : 0.0
                use_student_t = nu_h0 > 0.0
                @debug "H0 BIC: Normal=$(round(bic_normal,digits=1)), best_nu=$best_nu, use_student_t=$use_student_t"
            end
            h0_bic_done = true
        end

        # ---- Step-halving guard: snapshot before M-step ----
        guard_active = iter > 10 && iter > convergence_skip_until
        ll_before_mstep = ll_after_e_val  # LL from E-step (correct baseline per Pitfall 1)
        if guard_active
            snap = _snapshot_params(
                mu_e0, sig_e0, mu_c0, sig_c0, mu_p0, sig_p0,
                mu_ea, sig_ea, mu_ca, sig_ca, mu_pa, sig_pa,
                alpha_e1, theta_e1, mu_c1, sig_c1, mu_p1, sig_p1,
                pi_k, gamma, disc_H0, disc_ag, disc_H1
            )
        end

        # ---- M-step: Update parameters ----
        N_h0 = sum(gamma[:, 1])
        N_ag = sum(gamma[:, 2])
        N_h1 = sum(gamma[:, 3])

        # Update mixing weights (Dirichlet posterior mean — more stable than MAP)
        pi_k[1] = (N_h0 + alpha_prior[1]) / (n + sum(alpha_prior))
        pi_k[2] = (N_ag + alpha_prior[2]) / (n + sum(alpha_prior))
        pi_k[3] = (N_h1 + alpha_prior[3]) / (n + sum(alpha_prior))

        # Normalize to ensure sum = 1 (posterior mean is exact but guard rounding)
        pi_sum = sum(pi_k)
        pi_k ./= pi_sum

        # Update H0 (background)
        if N_h0 > 1e-10
            mu_e0 = sum(gamma[:, 1] .* y_enrich) / N_h0
            mu_c0 = sum(gamma[:, 1] .* y_corr) / N_h0
            mu_p0 = sum(gamma[:, 1] .* y_pres) / N_h0
            sig_e0 = sqrt(sum(gamma[:, 1] .* (y_enrich .- mu_e0).^2) / N_h0)
            sig_c0 = sqrt(sum(gamma[:, 1] .* (y_corr .- mu_c0).^2) / N_h0)
            sig_p0 = sqrt(sum(gamma[:, 1] .* (y_pres .- mu_p0).^2) / N_h0)
            # Re-fit detection DiscreteEmpirical for H0 component
            disc_H0 = _fit_discrete_empirical_weighted(y_pres, gamma[:, 1])
        end

        # Update agnostic
        if N_ag > 1e-10
            mu_ea = 0.0  # COMP-02: Agnostic enrichment mean anchored at zero — DO NOT update from data
            mu_ca = sum(gamma[:, 2] .* y_corr) / N_ag
            mu_pa = sum(gamma[:, 2] .* y_pres) / N_ag
            # sigma_ea computed relative to fixed mean of 0.0
            sig_ea = sqrt(sum(gamma[:, 2] .* (y_enrich .- 0.0).^2) / N_ag)
            sig_ca = sqrt(sum(gamma[:, 2] .* (y_corr .- mu_ca).^2) / N_ag)
            sig_pa = sqrt(sum(gamma[:, 2] .* (y_pres .- mu_pa).^2) / N_ag)
            # Re-fit detection DiscreteEmpirical for agnostic component
            disc_ag = _fit_discrete_empirical_weighted(y_pres, gamma[:, 2])
        end

        # Update H1 (interaction)
        if N_h1 > 1e-10
            # H1 enrichment: fit LocationShifted{T} using family-dispatched weighted MLE
            above_c = findall(y_enrich .> JEFFREYS_SHIFT)
            w_above = gamma[above_c, 3]
            if sum(w_above) > 1e-10 && length(above_c) >= 5
                shifted = max.(y_enrich[above_c] .- JEFFREYS_SHIFT, 1e-10)
                p1_new, p2_new = _reinit_h1_params_for_family(current_family, shifted, w_above)
                alpha_e1 = p1_new
                theta_e1 = p2_new
            else
                # Fallback: use defaults for the current family
                if current_family == :lognormal
                    alpha_e1, theta_e1 = 0.5, 1.0
                elseif current_family == :weibull
                    alpha_e1, theta_e1 = 1.5, 2.0
                else
                    alpha_e1, theta_e1 = 2.0, 2.0
                end
            end

            # Correlation: symmetric Normal (free mean)
            mu_c1 = sum(gamma[:, 3] .* y_corr) / N_h1
            mu_p1 = sum(gamma[:, 3] .* y_pres) / N_h1
            sig_c1 = sqrt(sum(gamma[:, 3] .* (y_corr .- mu_c1).^2) / N_h1)
            sig_p1 = sqrt(sum(gamma[:, 3] .* (y_pres .- mu_p1).^2) / N_h1)
            # Re-fit detection DiscreteEmpirical for H1 component
            disc_H1 = _fit_discrete_empirical_weighted(y_pres, gamma[:, 3])
        end

        # ---- Apply all constraints in canonical order ----
        constrained = _apply_constraints!(
            mu_e0, sig_e0, mu_c0, sig_c0, mu_p0, sig_p0,
            mu_ea, sig_ea, mu_ca, sig_ca, mu_pa, sig_pa,
            alpha_e1, theta_e1, mu_c1, sig_c1, mu_p1, sig_p1,
            pi_k, gamma,
            sigma_floor, max_sigma_e, max_sigma_c, max_sigma_p, current_family
        )
        # Unpack constrained scalar values
        mu_e0, sig_e0, mu_c0, sig_c0, mu_p0, sig_p0 = constrained.mu_e0, constrained.sig_e0, constrained.mu_c0, constrained.sig_c0, constrained.mu_p0, constrained.sig_p0
        mu_ea, sig_ea, mu_ca, sig_ca, mu_pa, sig_pa = constrained.mu_ea, constrained.sig_ea, constrained.mu_ca, constrained.sig_ca, constrained.mu_pa, constrained.sig_pa
        alpha_e1, theta_e1, mu_c1, sig_c1, mu_p1, sig_p1 = constrained.alpha_e1, constrained.theta_e1, constrained.mu_c1, constrained.sig_c1, constrained.mu_p1, constrained.sig_p1

        # Update _h1_eff_mean after constraints (used by E-step monotonicity correction)
        _h1_eff_mean = if current_family == :lognormal
            mean(LogNormal(alpha_e1, theta_e1)) + JEFFREYS_SHIFT
        elseif current_family == :weibull
            mean(Weibull(alpha_e1, theta_e1)) + JEFFREYS_SHIFT
        else
            alpha_e1 * theta_e1 + JEFFREYS_SHIFT  # Gamma mean
        end

        # ---- Compute log-likelihood (after M-step + constraints) ----
        h1_enrich_dist_ll = if current_family == :lognormal
            LogNormal(alpha_e1, theta_e1)
        elseif current_family == :weibull
            Weibull(alpha_e1, theta_e1)
        else
            Gamma(alpha_e1, theta_e1)
        end
        h0_enrich_dist_ll = use_student_t ?
            LocationScale(mu_e0, sig_e0, TDist(nu_h0)) :
            Normal(mu_e0, sig_e0)
        ll = _compute_ll_3c(y_enrich, y_corr, y_pres, disc_H0, disc_ag, disc_H1,
            h0_enrich_dist_ll, mu_c0, sig_c0, mu_ea, sig_ea, mu_ca, sig_ca,
            mu_c1, sig_c1, pi_k, h1_enrich_dist_ll, JEFFREYS_SHIFT)

        # Step-halving guard: revert if LL decreased
        if guard_active && (ll - ll_before_mstep) < -1e-6
            # Full revert to pre-M-step snapshot
            mu_e0, sig_e0 = snap.mu_e0, snap.sig_e0
            mu_c0, sig_c0 = snap.mu_c0, snap.sig_c0
            mu_p0, sig_p0 = snap.mu_p0, snap.sig_p0
            mu_ea, sig_ea = snap.mu_ea, snap.sig_ea
            mu_ca, sig_ca = snap.mu_ca, snap.sig_ca
            mu_pa, sig_pa = snap.mu_pa, snap.sig_pa
            alpha_e1, theta_e1 = snap.alpha_e1, snap.theta_e1
            mu_c1, sig_c1 = snap.mu_c1, snap.sig_c1
            mu_p1, sig_p1 = snap.mu_p1, snap.sig_p1
            pi_k .= snap.pi_k       # in-place restore
            gamma .= snap.gamma      # in-place restore (Pitfall 4)
            disc_H0 = snap.disc_H0   # safe: immutable struct
            disc_ag = snap.disc_ag
            disc_H1 = snap.disc_H1
            ll = ll_before_mstep      # LL stays at pre-M-step value
            n_step_halving_reverts += 1
            @debug "Step-halving reverted M-step at iter $iter: delta=$(ll - ll_before_mstep)"
        end
        push!(log_liks, ll)
        push!(ll_trace_m, ll)

        # ---- Check convergence ----
        if iter > 10 && iter > convergence_skip_until
            rel_change = abs(log_liks[end] - log_liks[end-1]) / abs(log_liks[end-1] + 1e-300)
            if rel_change < tol
                # Report enrichment mean for display (family-agnostic)
                _enrich_disp = _h1_eff_mean
                # Compute data-space SD for H1 enrichment
                _h1_sd_final = if current_family == :lognormal
                    std(LogNormal(alpha_e1, theta_e1))
                elseif current_family == :weibull
                    std(Weibull(alpha_e1, theta_e1))
                else
                    sqrt(alpha_e1) * theta_e1
                end
                return (
                    mixing_weights = copy(pi_k),
                    means = Dict(
                        "background"  => (enrichment=mu_e0, correlation=mu_c0, presence=mu_p0),
                        "agnostic"    => (enrichment=mu_ea, correlation=mu_ca, presence=mu_pa),
                        "interaction" => (enrichment=_enrich_disp, correlation=mu_c1, presence=mu_p1)
                    ),
                    precisions = Dict(
                        "background"  => (enrichment=1/sig_e0^2, correlation=1/sig_c0^2, presence=1/sig_p0^2),
                        "agnostic"    => (enrichment=1/sig_ea^2, correlation=1/sig_ca^2, presence=1/sig_pa^2),
                        "interaction" => (enrichment=1/_h1_sd_final^2, correlation=1/sig_c1^2, presence=1/sig_p1^2)
                    ),
                    std_devs = Dict(
                        "background"  => (enrichment=sig_e0, correlation=sig_c0, presence=sig_p0),
                        "agnostic"    => (enrichment=sig_ea, correlation=sig_ca, presence=sig_pa),
                        "interaction" => (enrichment=_h1_sd_final, correlation=sig_c1, presence=sig_p1)
                    ),
                    responsibilities = copy(gamma),
                    log_likelihood = log_liks,
                    converged = true,
                    n_iterations = iter,
                    alpha_e1 = alpha_e1,
                    theta_e1 = theta_e1,
                    h1_enrichment_sd = _h1_sd_final,
                    h1_family = current_family,
                    h1_bic_scores = copy(bic_scores),
                    disc_H0 = disc_H0,
                    disc_ag = disc_ag,
                    disc_H1 = disc_H1,
                    ll_trace_e_step = copy(ll_trace_e),
                    ll_trace_m_step = copy(ll_trace_m),
                    n_step_halving_reverts = n_step_halving_reverts,
                    family_switch_iter = family_switch_iter,
                    nu_h0 = nu_h0,
                    use_student_t = use_student_t
                )
            end
        end
    end

    # Did not converge within iterations
    _enrich_disp = _h1_eff_mean
    # Compute data-space SD for H1 enrichment
    _h1_sd_final_nc = if current_family == :lognormal
        std(LogNormal(alpha_e1, theta_e1))
    elseif current_family == :weibull
        std(Weibull(alpha_e1, theta_e1))
    else
        sqrt(alpha_e1) * theta_e1
    end
    return (
        mixing_weights = copy(pi_k),
        means = Dict(
            "background"  => (enrichment=mu_e0, correlation=mu_c0, presence=mu_p0),
            "agnostic"    => (enrichment=mu_ea, correlation=mu_ca, presence=mu_pa),
            "interaction" => (enrichment=_enrich_disp, correlation=mu_c1, presence=mu_p1)
        ),
        precisions = Dict(
            "background"  => (enrichment=1/sig_e0^2, correlation=1/sig_c0^2, presence=1/sig_p0^2),
            "agnostic"    => (enrichment=1/sig_ea^2, correlation=1/sig_ca^2, presence=1/sig_pa^2),
            "interaction" => (enrichment=1/_h1_sd_final_nc^2, correlation=1/sig_c1^2, presence=1/sig_p1^2)
        ),
        std_devs = Dict(
            "background"  => (enrichment=sig_e0, correlation=sig_c0, presence=sig_p0),
            "agnostic"    => (enrichment=sig_ea, correlation=sig_ca, presence=sig_pa),
            "interaction" => (enrichment=_h1_sd_final_nc, correlation=sig_c1, presence=sig_p1)
        ),
        responsibilities = copy(gamma),
        log_likelihood = log_liks,
        converged = false,
        n_iterations = n_iterations,
        alpha_e1 = alpha_e1,
        theta_e1 = theta_e1,
        h1_enrichment_sd = _h1_sd_final_nc,
        h1_family = current_family,
        h1_bic_scores = copy(bic_scores),
        disc_H0 = disc_H0,
        disc_ag = disc_ag,
        disc_H1 = disc_H1,
        ll_trace_e_step = copy(ll_trace_e),
        ll_trace_m_step = copy(ll_trace_m),
        n_step_halving_reverts = n_step_halving_reverts,
        family_switch_iter = family_switch_iter,
        nu_h0 = nu_h0,
        use_student_t = use_student_t
    )
end

"""
    compute_robust_posteriors_3c(y_e, y_c, y_p, em_result)

Compute H1-vs-rest posteriors using fitted 3-component EM parameters.
Uses the same winsorized data as EM training to avoid fat-tail crossover.

For each protein:
- P_H1 = pi_H1 * f_H1 / (pi_H0 * f_H0 + pi_ag * f_ag + pi_H1 * f_H1)
- Combined BF = (f_H1 / pi_H1) / ((f_H0 + f_ag) / (pi_H0 + pi_ag)) [density ratio]
- Prior odds = pi_H1 / (pi_H0 + pi_ag) [from EM mixing weights]

Includes monotonicity correction and LLR clamping.

Returns `(p_h1, p_h0, p_agnostic, joint_bf, pi_mean, prior_odds)`.
"""
function compute_robust_posteriors_3c(y_e::Vector{Float64}, y_c::Vector{Float64}, y_p::Vector{Float64},
                                       em_result)
    n = length(y_e)
    pi_k = em_result.mixing_weights

    means_bg = em_result.means["background"]
    means_ag = em_result.means["agnostic"]
    means_int = em_result.means["interaction"]
    std_bg = em_result.std_devs["background"]
    std_ag = em_result.std_devs["agnostic"]
    std_int = em_result.std_devs["interaction"]

    # H1 enrichment distribution (family-dispatched)
    alpha_h1 = em_result.alpha_e1
    theta_h1 = em_result.theta_e1
    h1_family_sel = hasproperty(em_result, :h1_family) ? em_result.h1_family : :gamma
    h1_enrich_dist = if h1_family_sel == :lognormal
        LogNormal(alpha_h1, theta_h1)
    elseif h1_family_sel == :weibull
        Weibull(alpha_h1, theta_h1)
    else
        Gamma(alpha_h1, theta_h1)
    end

    # H0 enrichment distribution (Student-t or Normal, from EM result)
    h0_nu = hasproperty(em_result, :nu_h0) ? em_result.nu_h0 : 0.0
    h0_use_t = hasproperty(em_result, :use_student_t) ? em_result.use_student_t : false
    h0_enrich_dist = h0_use_t && h0_nu > 0.0 ?
        LocationScale(means_bg.enrichment, std_bg.enrichment, TDist(h0_nu)) :
        Normal(means_bg.enrichment, std_bg.enrichment)

    # Use DiscreteEmpirical for detection if available
    has_disc = hasproperty(em_result, :disc_H0) && em_result.disc_H0 !== nothing
    disc_H0_det = has_disc ? em_result.disc_H0 : nothing
    disc_ag_det = has_disc ? em_result.disc_ag : nothing
    disc_H1_det = has_disc ? em_result.disc_H1 : nothing

    p_h1 = Vector{Float64}(undef, n)
    p_h0 = Vector{Float64}(undef, n)
    p_agnostic = Vector{Float64}(undef, n)

    for i in 1:n
        # H1 enrichment uses LocationShifted distribution (family-dispatched)
        ll_h1_e = _h1_enrichment_logdensity(y_e[i], h1_enrich_dist, JEFFREYS_SHIFT, SIGMOID_STEEPNESS)
        ll_h0_e = logpdf(h0_enrich_dist, y_e[i])

        # Detection log-likelihoods: use DiscreteEmpirical if fitted, else fall back to Normal
        ll_det_h0 = has_disc ?
            log(max(pdf(disc_H0_det, y_p[i]), 1e-300)) :
            logpdf(Normal(means_bg.presence, std_bg.presence), y_p[i])
        ll_det_ag = has_disc ?
            log(max(pdf(disc_ag_det, y_p[i]), 1e-300)) :
            logpdf(Normal(means_ag.presence, std_ag.presence), y_p[i])
        ll_det_h1 = has_disc ?
            log(max(pdf(disc_H1_det, y_p[i]), 1e-300)) :
            logpdf(Normal(means_int.presence, std_int.presence), y_p[i])

        # Per-dimension log-likelihood ratios: log(p(y|int) / p(y|bg))
        llr_e = ll_h1_e - ll_h0_e
        llr_c = logpdf(Normal(means_int.correlation, std_int.correlation), y_c[i]) -
                logpdf(Normal(means_bg.correlation, std_bg.correlation), y_c[i])
        llr_p = ll_det_h1 - ll_det_h0

        # Monotonicity correction: if value exceeds interaction mean, floor LLR at 0
        if y_e[i] > means_int.enrichment
            llr_e = max(llr_e, 0.0)
        end
        if y_c[i] > means_int.correlation
            llr_c = max(llr_c, 0.0)
        end
        # For detection: monotonicity based on DiscreteEmpirical mode or Normal mean
        det_int_center = has_disc ?
            disc_H1_det.values[argmax(disc_H1_det.probs)] :
            means_int.presence
        if y_p[i] > det_int_center
            llr_p = max(llr_p, 0.0)
        end

        # Compute full posteriors via densities for accurate P_H0/P_ag/P_H1
        ll_h0 = ll_h0_e +
                 logpdf(Normal(means_bg.correlation, std_bg.correlation), y_c[i]) +
                 ll_det_h0 +
                 log(pi_k[1] + 1e-300)

        ll_ag = logpdf(Normal(means_ag.enrichment, std_ag.enrichment), y_e[i]) +
                logpdf(Normal(means_ag.correlation, std_ag.correlation), y_c[i]) +
                ll_det_ag +
                log(pi_k[2] + 1e-300)

        ll_h1 = ll_h1_e +
                 logpdf(Normal(means_int.correlation, std_int.correlation), y_c[i]) +
                 ll_det_h1 +
                 log(pi_k[3] + 1e-300)

        # Monotonicity correction for posteriors:
        # Apply same correction as E-step to prevent thin-tail penalty
        # means_int.enrichment is _h1_eff_mean (verified from EM return structure)
        if y_e[i] > means_int.enrichment && ll_h1_e > -100.0
            ld_h0_e = logpdf(h0_enrich_dist, y_e[i])
            if ll_h1_e < ld_h0_e
                ll_h1 = ll_h1 - ll_h1_e + ld_h0_e
            end
        end

        if y_c[i] > means_int.correlation
            ld_h0_c = logpdf(Normal(means_bg.correlation, std_bg.correlation), y_c[i])
            ld_h1_c = logpdf(Normal(means_int.correlation, std_int.correlation), y_c[i])
            if ld_h1_c < ld_h0_c
                ll_h1 = ll_h1 - ld_h1_c + ld_h0_c
            end
        end

        # Log-sum-exp for numerical stability
        max_ll = max(ll_h0, ll_ag, ll_h1)
        denom = exp(ll_h0 - max_ll) + exp(ll_ag - max_ll) + exp(ll_h1 - max_ll)

        p_h0[i] = exp(ll_h0 - max_ll) / denom
        p_agnostic[i] = exp(ll_ag - max_ll) / denom
        p_h1[i] = exp(ll_h1 - max_ll) / denom
    end

    # Prior odds from mixing weights: pi_H1 / (pi_H0 + pi_ag)
    prior_odds = pi_k[3] / max(pi_k[1] + pi_k[2], 1e-300)

    # Joint BF for each protein: posterior_odds / prior_odds
    joint_bf = Vector{Float64}(undef, n)
    for i in 1:n
        posterior_odds_i = p_h1[i] / max(1.0 - p_h1[i], 1e-300)
        joint_bf[i] = posterior_odds_i / max(prior_odds, 1e-300)
    end

    return (p_h1 = p_h1,
            p_h0 = p_h0,
            p_agnostic = p_agnostic,
            joint_bf = joint_bf,
            pi_mean = pi_k,
            prior_odds = prior_odds)
end

"""
    extract_lc_class_parameters_3c(em_result)

Returns sum-of-Gaussians class-specific parameters for 3-component LatentClassResult.
The histogram plots `log(combined_BF)` which is the log of the product of per-dimension BFs.
On log scale, product becomes sum, so:
- `mu_sum = mu_enrichment + mu_correlation + mu_presence`
- `sigma_sum = sqrt(sigma_enrichment^2 + sigma_correlation^2 + sigma_presence^2)` (RSS)
- `precision_sum = 1 / sigma_sum^2`
"""
function extract_lc_class_parameters_3c(em_result)
    means_bg = em_result.means["background"]
    means_ag = em_result.means["agnostic"]
    means_int = em_result.means["interaction"]
    std_bg = em_result.std_devs["background"]
    std_ag = em_result.std_devs["agnostic"]
    std_int = em_result.std_devs["interaction"]

    # Sum-of-Gaussians: sum means, RSS sigmas (independence assumption)
    bg_sum_mu = means_bg.enrichment + means_bg.correlation + means_bg.presence
    bg_sum_sigma = sqrt(std_bg.enrichment^2 + std_bg.correlation^2 + std_bg.presence^2)
    bg_sum_precision = 1.0 / bg_sum_sigma^2

    ag_sum_mu = means_ag.enrichment + means_ag.correlation + means_ag.presence
    ag_sum_sigma = sqrt(std_ag.enrichment^2 + std_ag.correlation^2 + std_ag.presence^2)
    ag_sum_precision = 1.0 / ag_sum_sigma^2

    # H1 enrichment: compute proper data-space mean from LocationShifted distribution
    h1_family = hasproperty(em_result, :h1_family) ? em_result.h1_family :
                (hasproperty(em_result, :h1_enrichment_family) ? em_result.h1_enrichment_family : :gamma)
    alpha_h1 = hasproperty(em_result, :alpha_e1) ? em_result.alpha_e1 :
               (hasproperty(em_result, :alpha_enrichment_h1) ? em_result.alpha_enrichment_h1 : 2.0)
    theta_h1 = hasproperty(em_result, :theta_e1) ? em_result.theta_e1 :
               (hasproperty(em_result, :theta_enrichment_h1) ? em_result.theta_enrichment_h1 : 2.0)
    h1_enrich_mean = if h1_family == :lognormal
        mean(LogNormal(alpha_h1, theta_h1)) + JEFFREYS_SHIFT
    elseif h1_family == :weibull
        mean(Weibull(alpha_h1, theta_h1)) + JEFFREYS_SHIFT
    else  # :gamma
        alpha_h1 * theta_h1 + JEFFREYS_SHIFT
    end
    # Use stored h1_enrichment_sd for proper data-space SD
    h1_enrich_sd = hasproperty(em_result, :h1_enrichment_sd) ? em_result.h1_enrichment_sd : std_int.enrichment

    int_sum_mu = h1_enrich_mean + means_int.correlation + means_int.presence
    int_sum_sigma = sqrt(h1_enrich_sd^2 + std_int.correlation^2 + std_int.presence^2)
    int_sum_precision = 1.0 / int_sum_sigma^2

    return Dict(
        "background" => (mu = bg_sum_mu, sigma = bg_sum_sigma, precision = bg_sum_precision),
        "agnostic" => (mu = ag_sum_mu, sigma = ag_sum_sigma, precision = ag_sum_precision),
        "interaction" => (mu = int_sum_mu, sigma = int_sum_sigma, precision = int_sum_precision)
    )
end

# ============================================================
# 3b. KL Divergence Helper (COMP-03)
# ============================================================

"""
    _kl_divergence_enrichment(h0_dist, agnostic_dist; n_sigma=10)

Compute KL divergence KL(h0_dist || agnostic_dist) via numerical integration.
Uses QuadGK adaptive quadrature. Returns non-negative KL value.

For TDist(3) where `std` is infinite, uses `scale(h0_dist) * 20` for bounds.
"""
function _kl_divergence_enrichment(h0_dist, agnostic_dist; n_sigma::Int=10)
    # Compute integration bounds covering both distributions
    mu_h0 = mean(h0_dist)
    mu_ag = mean(agnostic_dist)
    # For Student-t with nu<=2, std is Inf; use scale parameter * 20 as fallback
    sig_h0 = isfinite(std(h0_dist)) ? std(h0_dist) : (hasproperty(h0_dist, :σ) ? h0_dist.σ * 20.0 : 20.0)
    sig_ag = std(agnostic_dist)
    span = Float64(n_sigma) * max(sig_h0, sig_ag)
    lo = min(mu_h0, mu_ag) - span
    hi = max(mu_h0, mu_ag) + span

    kl, _ = quadgk(lo, hi; rtol=1e-8) do x
        p = pdf(h0_dist, x)
        p < 1e-300 && return 0.0
        q = pdf(agnostic_dist, x)
        q < 1e-300 && return p * (log(p) - log(1e-300))
        return p * log(p / q)
    end
    return max(kl, 0.0)  # guard against tiny negative from numerical error
end

# ============================================================
# 4. Main Entry Point
# ============================================================

"""
    combined_BF_latent_class(bf::BayesFactorTriplet, refID::Int;
                              n_iterations=200, alpha_prior=[5.0, 2.0, 1.0],
                              convergence_tol=1e-6, verbose=true,
                              winsorize=true, winsorize_quantiles=(0.01, 0.99),
                              use_3c=true, n_restarts=20, kwargs...)

Main entry point for latent class-based evidence combination.

This is a drop-in alternative to the copula-based `combined_BF()` function.
By default uses a 3-component Gaussian mixture (H0/agnostic/H1) on log-Bayes factors
with multi-restart initialization. Set `use_3c=false` for legacy 2-component mode.

# Arguments
- `bf::BayesFactorTriplet`: Triplet of Bayes factors (enrichment, correlation, detection)
- `refID::Int`: Index of the bait protein (will be clamped to max interaction probability)

# Keyword Arguments
- `n_iterations::Int=200`: Number of EM iterations per restart
- `alpha_prior::Vector{Float64}=[5.0, 2.0, 1.0]`: Dirichlet prior for mixing weights
   (H0, agnostic, H1) for 3-component, or [10.0, 1.0] for 2-component
- `convergence_tol::Float64=1e-6`: Convergence tolerance for log-likelihood
- `verbose::Bool=true`: Print convergence diagnostics
- `winsorize::Bool=true`: Whether to winsorize log-BFs before EM fitting
- `winsorize_quantiles::Tuple{Float64,Float64}=(0.01, 0.99)`: Quantile range for winsorization
- `use_3c::Bool=true`: Use 3-component model (default) or 2-component (legacy)
- `n_restarts::Int=20`: Number of EM restarts (3-component only; best selected by log-likelihood)

# Returns
`LatentClassResult` with combined Bayes factors, posterior probabilities,
and (for 3-component) responsibilities matrix.
"""
function combined_BF_latent_class(bf::BayesFactorTriplet, refID::Int;
                                   n_iterations::Int = 200,
                                   alpha_prior::Union{Symbol, Vector{Float64}} = :auto,
                                   convergence_tol::Float64 = 1e-6,
                                   verbose::Bool = true,
                                   winsorize::Bool = true,
                                   winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99),
                                   use_3c::Bool = true,
                                   n_restarts::Int = 20,
                                   force_h1_family::Union{Nothing, Symbol} = nothing,
                                   protein_names::Union{Nothing, Vector{String}} = nothing,
                                   kwargs...)
    # 1. Prepare scores (log-transform BFs, with winsorization and input validation)
    y_e_win, y_c_win, y_p_win, y_e_orig, y_c_orig, y_p_orig = prepare_lc_scores(
        bf.enrichment, bf.correlation, bf.detection;
        log_transform = true,
        winsorize = winsorize,
        winsorize_quantiles = winsorize_quantiles
    )

    if !use_3c
        # ---- Legacy 2-component path ----
        alpha_2c = length(alpha_prior) == 2 ? alpha_prior : [10.0, 1.0]

        if verbose
            @info "Running 2-component EM algorithm for latent class model with $n_iterations iterations..."
        end

        em_result = fit_gaussian_mixture_em(y_e_win, y_c_win, y_p_win;
                                            n_iterations = n_iterations,
                                            alpha_prior = alpha_2c,
                                            tol = convergence_tol)

        if verbose
            if em_result.converged
                @info "Model converged after $(em_result.n_iterations) iterations"
            else
                @warn "Model did not converge within $n_iterations iterations"
            end
        end

        posteriors = compute_robust_posteriors(y_e_win, y_c_win, y_p_win, em_result)
        params = extract_lc_class_parameters(em_result)

        posterior_prob = copy(posteriors.p_interact)
        joint_bf = copy(posteriors.joint_bf)

        # Monotonicity constraint
        for i in 1:length(joint_bf)
            if bf.enrichment[i] < 1.0 && bf.correlation[i] < 1.0
                max_individual = max(bf.enrichment[i], bf.correlation[i], bf.detection[i])
                if joint_bf[i] > max_individual
                    joint_bf[i] = max_individual
                    capped_odds = max_individual * posteriors.prior_odds
                    posterior_prob[i] = capped_odds / (1.0 + capped_odds)
                end
            end
        end

        # Bait handling
        if 1 <= refID <= length(posterior_prob)
            max_prob = maximum(posterior_prob)
            posterior_prob[refID] = max_prob
            joint_bf[refID] = (max_prob / max(1.0 - max_prob, 1e-300)) / max(posteriors.prior_odds, 1e-300)
        end

        # Final BF clamping
        min_log_bf, max_log_bf = -46.0, 46.0
        for i in 1:length(joint_bf)
            log_bf_i = log(max(joint_bf[i], 1e-300))
            log_bf_i = clamp(log_bf_i, min_log_bf, max_log_bf)
            joint_bf[i] = exp(log_bf_i)
            capped_odds = joint_bf[i] * posteriors.prior_odds
            posterior_prob[i] = capped_odds / (1.0 + capped_odds)
        end

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

    # ---- 3-component path (default) ----

    # Auto-extend 2-element alpha_prior for backward compatibility
    if isa(alpha_prior, Vector) && length(alpha_prior) == 2
        alpha_prior = [alpha_prior[1], 2.0, alpha_prior[2]]
    end

    # ---- :auto path: EB estimation + grid marginalization ----
    if isa(alpha_prior, Symbol) && alpha_prior === :auto
        if verbose
            @info "EB auto-prior: running baseline EM with flat prior [1,1,1]..."
        end

        # Step 1: Baseline EM with flat [1,1,1] prior to get responsibilities
        # Recursive call triggers the explicit-alpha path (flat_alpha is Vector{Float64})
        flat_alpha = [1.0, 1.0, 1.0]
        baseline_result = combined_BF_latent_class(bf, refID;
            n_iterations = n_iterations,
            alpha_prior = flat_alpha,
            convergence_tol = convergence_tol,
            verbose = verbose,
            winsorize = winsorize,
            winsorize_quantiles = winsorize_quantiles,
            use_3c = true,
            n_restarts = n_restarts,
            force_h1_family = force_h1_family,
            kwargs...)

        # Step 2: EB estimation from baseline responsibilities
        gamma = baseline_result.responsibilities
        if gamma === nothing
            @warn "EB auto-prior: baseline EM did not return responsibilities, falling back to [5.0, 2.0, 1.0]"
            eb_alpha = [5.0, 2.0, 1.0]
            eb_conv = false
        else
            eb_result = estimate_dirichlet_eb(gamma)
            eb_alpha = eb_result.alpha
            eb_conv = eb_result.converged
            if !eb_conv
                @warn "EB auto-prior: Dirichlet estimation did not converge ($(eb_result.iterations) iterations), using clamped estimate"
            end
            if verbose
                @info "EB auto-prior: estimated alpha = $(round.(eb_alpha, digits=3)), converged = $eb_conv"
            end
        end

        # Step 3: Build prior grid centered on EB estimate
        grid = build_prior_grid(eb_alpha)
        if verbose
            @info "EB auto-prior: grid has $(length(grid)) points"
        end

        # Step 4: Lock H1 family from baseline
        locked_family = baseline_result.h1_enrichment_family
        if verbose
            @info "EB auto-prior: locking H1 family = :$(locked_family) from baseline"
        end

        # Step 5: Marginalize over prior grid
        marg = _marginalize_over_priors(
            y_e_win, y_c_win, y_p_win,
            y_e_orig, y_c_orig, y_p_orig,
            grid;
            force_h1_family = locked_family,
            n_restarts = n_restarts,
            n_iterations = n_iterations,
            convergence_tol = convergence_tol
        )

        # Step 6: Build LatentClassResult from marginalized results
        # Use baseline_result for structural fields, override bf/posterior with marginalized values

        # Bait handling: _marginalize_over_priors averages the raw per-grid p_h1 and
        # applies NO bait special treatment, so the refID protein would be scored like
        # any background protein. Clamp the bait to the maximum posterior (and BF) here,
        # consistent with the legacy-2c and explicit-3c paths in this function (E-5 fix).
        auto_posterior = copy(marg.posterior_prob)
        auto_bf = copy(marg.combined_bf)
        if 1 <= refID <= length(auto_posterior)
            auto_posterior[refID] = maximum(auto_posterior)
            auto_bf[refID] = maximum(auto_bf)
        end

        return LatentClassResult(
            auto_bf,                                # averaged BFs (bait clamped to max)
            auto_posterior,                         # averaged posteriors (bait clamped to max)
            baseline_result.class_parameters,      # from baseline EM
            baseline_result.mixing_weights,         # from baseline EM
            baseline_result.free_energy,            # from baseline EM
            baseline_result.converged,              # from baseline EM
            baseline_result.n_iterations,           # from baseline EM
            baseline_result.responsibilities,       # from baseline EM
            baseline_result.all_restart_traces,     # from baseline EM
            baseline_result.alpha_enrichment_h1,    # from baseline EM
            baseline_result.theta_enrichment_h1,    # from baseline EM
            baseline_result.h1_enrichment_sd,       # from baseline EM
            baseline_result.h1_enrichment_family,   # locked family
            baseline_result.h1_bic_scores,          # from baseline EM
            baseline_result.em_diagnostics,         # from baseline EM
            baseline_result.disc_detection_H0,      # from baseline EM
            baseline_result.disc_detection_ag,      # from baseline EM
            baseline_result.disc_detection_H1,      # from baseline EM
            baseline_result.per_step_ll_traces,     # from baseline EM
            baseline_result.n_step_halving_reverts, # from baseline EM
            baseline_result.per_dimension_params,   # from baseline EM
            baseline_result.nu_h0,                  # from baseline EM
            baseline_result.kl_divergence,          # from baseline EM
            baseline_result.merged,                 # from baseline EM
            baseline_result.annealing_schedule,     # from baseline EM
            baseline_result.bimodality_coefficient, # from baseline EM
            eb_alpha,                               # effective_alpha_prior
            marg.bic_weights,                       # prior_grid_weights
            marg.per_grid_posteriors,               # prior_grid_posteriors
            eb_conv,                                # eb_converged
            protein_names                           # protein_names
        )
    end

    # ---- Explicit alpha path (existing code) ----

    if verbose
        @info "Running 3-component EM with $n_restarts restarts, $n_iterations iterations each..."
        if winsorize
            @info "Winsorization enabled with quantiles $(winsorize_quantiles)"
        end
    end

    # Multi-restart: run EM with different initializations, select best by log-likelihood
    # Restarts are diversified across H1 families: first third Gamma, second third LogNormal, last third Weibull
    best_em = nothing
    best_ll = -Inf
    all_final_lls = Float64[]
    all_restart_traces = Vector{Vector{Float64}}()

    # Per-step LL traces and violation counts
    all_ll_traces = NamedTuple{(:ll_after_e, :ll_after_m), Tuple{Vector{Float64}, Vector{Float64}}}[]
    all_violations = Int[]

    # Per-restart diagnostics accumulators
    diag_restart     = Int[]
    diag_init_pi0    = Float64[]
    diag_init_method = String[]
    diag_final_pi0   = Float64[]
    diag_final_pi1   = Float64[]
    diag_ll          = Float64[]
    diag_iterations  = Int[]
    diag_converged   = Bool[]
    diag_status      = String[]
    diag_bic_gamma   = Union{Float64, Missing}[]
    diag_bic_lnorm   = Union{Float64, Missing}[]
    diag_bic_weib    = Union{Float64, Missing}[]
    diag_fam         = Union{String, Missing}[]
    diag_n_violations = Int[]
    diag_family_switch_iter = Union{Int, Missing}[]

    for r in 1:n_restarts
        restart_fam = force_h1_family !== nothing ? force_h1_family : _restart_family(r, n_restarts)
        init_params = initialize_3c_em(y_e_win, y_c_win, y_p_win, r; n_restarts=n_restarts)
        em_result = fit_gaussian_mixture_em_3c(y_e_win, y_c_win, y_p_win;
                                                n_iterations = n_iterations,
                                                alpha_prior = alpha_prior,
                                                tol = convergence_tol,
                                                init_params = init_params,
                                                h1_family = restart_fam,
                                                skip_bic_selection = force_h1_family !== nothing)

        final_ll = isempty(em_result.log_likelihood) ? -Inf : em_result.log_likelihood[end]
        push!(all_final_lls, final_ll)
        push!(all_restart_traces, copy(em_result.log_likelihood))

        # Collect per-step LL traces and violations
        push!(all_ll_traces, (ll_after_e = copy(em_result.ll_trace_e_step),
                              ll_after_m = copy(em_result.ll_trace_m_step)))
        push!(all_violations, em_result.n_step_halving_reverts)

        # Collect per-restart diagnostics
        push!(diag_restart, r)
        push!(diag_init_pi0, init_params.pi[1])
        push!(diag_init_method, r <= 5 ? "quantile" : (r <= 10 ? "kmeans" : "random"))
        push!(diag_final_pi0, em_result.mixing_weights[1])
        push!(diag_final_pi1, em_result.mixing_weights[3])
        push!(diag_ll, final_ll)
        push!(diag_iterations, em_result.n_iterations)
        push!(diag_converged, em_result.converged)
        push!(diag_status, isfinite(final_ll) ? "success" : "failed")

        # BIC columns from iteration 5 selection (missing if restart didn't reach iter 5)
        bic = em_result.h1_bic_scores
        has_bic = any(v -> isfinite(v) && v != 0.0, values(bic))
        push!(diag_bic_gamma, has_bic ? get(bic, :gamma, Inf) : missing)
        push!(diag_bic_lnorm, has_bic ? get(bic, :lognormal, Inf) : missing)
        push!(diag_bic_weib, has_bic ? get(bic, :weibull, Inf) : missing)
        push!(diag_fam, has_bic ? string(em_result.h1_family) : missing)
        push!(diag_n_violations, em_result.n_step_halving_reverts)
        push!(diag_family_switch_iter, em_result.family_switch_iter === nothing ? missing : em_result.family_switch_iter)

        if final_ll > best_ll
            best_ll = final_ll
            best_em = em_result
        end
    end

    # Build per-restart diagnostics DataFrame
    em_diagnostics = DataFrame(
        restart = diag_restart,
        init_pi0 = diag_init_pi0,
        init_method = diag_init_method,
        final_pi0 = diag_final_pi0,
        final_pi1 = diag_final_pi1,
        log_likelihood = diag_ll,
        iterations = diag_iterations,
        converged = diag_converged,
        status = diag_status,
        h1_bic_gamma = diag_bic_gamma,
        h1_bic_lognormal = diag_bic_lnorm,
        h1_bic_weibull = diag_bic_weib,
        h1_family_selected = diag_fam,
        n_step_halving_reverts = diag_n_violations,
        family_switch_iter = diag_family_switch_iter
    )

    # Extract winning family (from best restart's BIC selection at iteration 5)
    winning_family = hasproperty(best_em, :h1_family) ? best_em.h1_family : :gamma
    winning_bic_table = hasproperty(best_em, :h1_bic_scores) ? best_em.h1_bic_scores : Dict{Symbol,Float64}(:gamma => 0.0, :lognormal => Inf, :weibull => Inf)

    if verbose
        @info "Best restart: log-likelihood = $(round(best_ll, digits=2)) out of $n_restarts restarts"
        @info "H1 family selected: $winning_family (BIC: gamma=$(round(get(winning_bic_table, :gamma, Inf), digits=1)), lognormal=$(round(get(winning_bic_table, :lognormal, Inf), digits=1)), weibull=$(round(get(winning_bic_table, :weibull, Inf), digits=1)))"
        if best_em.converged
            @info "Best model converged after $(best_em.n_iterations) iterations"
        else
            @warn "Best model did not converge within $n_iterations iterations"
        end
        @info "Fitted enrichment - Background mu: $(round(best_em.means["background"].enrichment, digits=3)), " *
              "Agnostic mu: $(round(best_em.means["agnostic"].enrichment, digits=3)), " *
              "H1 $(winning_family)(param1=$(round(best_em.alpha_e1, digits=3)), param2=$(round(best_em.theta_e1, digits=3))) + shift=$(round(JEFFREYS_SHIFT, digits=3))"
        @info "Mixing weights: pi_H0=$(round(best_em.mixing_weights[1], digits=3)), " *
              "pi_ag=$(round(best_em.mixing_weights[2], digits=3)), " *
              "pi_H1=$(round(best_em.mixing_weights[3], digits=3))"
    end

    # COMP-04: Mixing weight separation diagnostic
    weight_diff = abs(best_em.mixing_weights[1] - best_em.mixing_weights[2])
    comp04_pass = weight_diff > 0.05
    # COMP-05: Mean separation diagnostic
    mu_e0_best = best_em.means["background"].enrichment
    sig_e0_best = best_em.std_devs["background"].enrichment
    sig_ea_best = best_em.std_devs["agnostic"].enrichment
    mean_sep = abs(mu_e0_best - 0.0)  # mu_ea is 0.0
    sigma_ref = max(sig_e0_best, sig_ea_best)
    comp05_pass = mean_sep > sigma_ref
    if verbose
        @info "COMP-04 weight separation: $(round(weight_diff, digits=3)) (>0.05: $comp04_pass)"
        @info "COMP-05 mean separation: $(round(mean_sep, digits=3)) vs 1*sigma=$(round(sigma_ref, digits=3)) (pass: $comp05_pass)"
        _best_nu_h0 = hasproperty(best_em, :nu_h0) ? best_em.nu_h0 : 0.0
        @info "H0 Student-t: nu=$(_best_nu_h0) (0.0 = Normal fallback)"
    end

    # Add h0_nu column to em_diagnostics
    em_diagnostics[!, :h0_nu] = fill(hasproperty(best_em, :nu_h0) ? best_em.nu_h0 : 0.0, nrow(em_diagnostics))

    # ---- Post-EM deterministic annealing for bimodal sharpening ----
    anneal_schedule = [0.9, 0.8, 0.7]
    gamma_annealed = copy(best_em.responsibilities)
    for T in anneal_schedule
        for i in axes(gamma_annealed, 1)
            gamma_annealed[i, 1] = gamma_annealed[i, 1] ^ (1.0 / T)
            gamma_annealed[i, 2] = gamma_annealed[i, 2] ^ (1.0 / T)
            gamma_annealed[i, 3] = gamma_annealed[i, 3] ^ (1.0 / T)
            row_sum = gamma_annealed[i, 1] + gamma_annealed[i, 2] + gamma_annealed[i, 3]
            if row_sum > 1e-300
                gamma_annealed[i, :] ./= row_sum
            else
                gamma_annealed[i, :] .= 1.0 / 3.0
            end
        end
    end

    # Compute bimodality coefficient on annealed H1 responsibilities
    bc = _sarles_bimodality_coefficient(gamma_annealed[:, 3])
    if verbose
        @info "Post-EM annealing: schedule=$(anneal_schedule), bimodality BC=$(round(bc, digits=3)) (>0.555 = bimodal)"
    end

    # Add annealing metadata to EM diagnostics
    em_diagnostics[!, :anneal_T_final] = fill(anneal_schedule[end], nrow(em_diagnostics))
    em_diagnostics[!, :bimodality_bc] = fill(bc, nrow(em_diagnostics))

    # Use original (non-winsorized) data for enrichment and correlation
    # to preserve extreme evidence (Root Cause 4). Keep winsorized detection data
    # because DiscreteEmpirical uses exact float lookup and non-winsorized values
    # would return pdf=0.0 for values outside the lookup table.
    posteriors = compute_robust_posteriors_3c(y_e_orig, y_c_orig, y_p_win, best_em)
    params = extract_lc_class_parameters_3c(best_em)

    posterior_prob = copy(posteriors.p_h1)
    joint_bf = copy(posteriors.joint_bf)

    # ---- COMP-03: Post-EM KL divergence merge check (best restart only) ----
    winning_nu_h0 = hasproperty(best_em, :nu_h0) ? best_em.nu_h0 : 0.0
    winning_use_student_t = hasproperty(best_em, :use_student_t) ? best_em.use_student_t : false

    mu_e0_kl = best_em.means["background"].enrichment
    sig_e0_kl = best_em.std_devs["background"].enrichment
    sig_ea_kl = best_em.std_devs["agnostic"].enrichment

    h0_dist_kl = winning_use_student_t ?
        LocationScale(mu_e0_kl, sig_e0_kl, TDist(winning_nu_h0)) :
        Normal(mu_e0_kl, sig_e0_kl)
    ag_dist_kl = Normal(0.0, sig_ea_kl)  # mu_ea is 0.0

    kl_val = _kl_divergence_enrichment(h0_dist_kl, ag_dist_kl)
    merged_flag = false
    final_pi = copy(best_em.mixing_weights)

    if verbose
        @info "KL(H0 || Agnostic) enrichment = $(round(kl_val, digits=4))"
    end

    if kl_val < 0.1
        # Merge H0 and Agnostic into single background
        merged_flag = true
        w_h0_merge = best_em.mixing_weights[1] / (best_em.mixing_weights[1] + best_em.mixing_weights[2])
        w_ag_merge = best_em.mixing_weights[2] / (best_em.mixing_weights[1] + best_em.mixing_weights[2])

        # Weighted average parameters
        merged_mu_e = w_h0_merge * mu_e0_kl + w_ag_merge * 0.0
        merged_sig_e = sqrt(w_h0_merge * sig_e0_kl^2 + w_ag_merge * sig_ea_kl^2 +
                            w_h0_merge * w_ag_merge * (mu_e0_kl - 0.0)^2)

        mu_c0_kl = best_em.means["background"].correlation
        mu_ca_kl = best_em.means["agnostic"].correlation
        sig_c0_kl = best_em.std_devs["background"].correlation
        sig_ca_kl = best_em.std_devs["agnostic"].correlation
        merged_mu_c = w_h0_merge * mu_c0_kl + w_ag_merge * mu_ca_kl
        merged_sig_c = sqrt(w_h0_merge * sig_c0_kl^2 + w_ag_merge * sig_ca_kl^2 +
                            w_h0_merge * w_ag_merge * (mu_c0_kl - mu_ca_kl)^2)

        mu_p0_kl = best_em.means["background"].presence
        mu_pa_kl = best_em.means["agnostic"].presence
        sig_p0_kl = best_em.std_devs["background"].presence
        sig_pa_kl = best_em.std_devs["agnostic"].presence
        merged_mu_p = w_h0_merge * mu_p0_kl + w_ag_merge * mu_pa_kl
        merged_sig_p = sqrt(w_h0_merge * sig_p0_kl^2 + w_ag_merge * sig_pa_kl^2 +
                            w_h0_merge * w_ag_merge * (mu_p0_kl - mu_pa_kl)^2)

        # Update mixing weights: absorb agnostic into H0
        final_pi[1] = best_em.mixing_weights[1] + best_em.mixing_weights[2]
        final_pi[2] = 0.0  # Agnostic weight zeroed (backward compat: 3-component structure retained)

        if verbose
            @info "KL < 0.1: merging H0 and Agnostic into single background (pi_bg=$(round(final_pi[1], digits=3)))"
        end

        # Recompute posteriors with merged parameters
        # CRITICAL: merged_em must include ALL fields accessed by compute_robust_posteriors_3c
        # AND extract_lc_class_parameters_3c. See <interfaces> section for complete field list.
        merged_em = (
            mixing_weights = final_pi,
            means = Dict(
                "background" => (enrichment=merged_mu_e, correlation=merged_mu_c, presence=merged_mu_p),
                "agnostic" => (enrichment=0.0, correlation=mu_ca_kl, presence=mu_pa_kl),
                "interaction" => best_em.means["interaction"]
            ),
            std_devs = Dict(
                "background" => (enrichment=merged_sig_e, correlation=merged_sig_c, presence=merged_sig_p),
                "agnostic" => (enrichment=sig_ea_kl, correlation=sig_ca_kl, presence=sig_pa_kl),
                "interaction" => best_em.std_devs["interaction"]
            ),
            # Fields for compute_robust_posteriors_3c:
            alpha_e1 = best_em.alpha_e1,
            theta_e1 = best_em.theta_e1,
            h1_family = winning_family,
            disc_H0 = hasproperty(best_em, :disc_H0) ? best_em.disc_H0 : nothing,
            disc_ag = hasproperty(best_em, :disc_ag) ? best_em.disc_ag : nothing,
            disc_H1 = hasproperty(best_em, :disc_H1) ? best_em.disc_H1 : nothing,
            nu_h0 = winning_nu_h0,
            use_student_t = winning_use_student_t,
            # Fields for extract_lc_class_parameters_3c:
            h1_enrichment_sd = hasproperty(best_em, :h1_enrichment_sd) ? best_em.h1_enrichment_sd :
                _compute_h1_enrichment_sd(Float64(best_em.alpha_e1), Float64(best_em.theta_e1), winning_family),
            # Fields for LatentClassResult constructor (responsibilities used downstream):
            responsibilities = hasproperty(best_em, :responsibilities) ? best_em.responsibilities : nothing
        )
        posteriors = compute_robust_posteriors_3c(y_e_orig, y_c_orig, y_p_win, merged_em)
        posterior_prob = copy(posteriors.p_h1)
        joint_bf = copy(posteriors.joint_bf)
        # Update params from merged result
        params = extract_lc_class_parameters_3c(merged_em)

        # Build post-merge responsibilities: P(agn) = 0, P(H0) absorbs agnostic mass
        n_proteins_local = length(posteriors.p_h1)
        merged_responsibilities = Matrix{Float64}(undef, n_proteins_local, 3)
        merged_responsibilities[:, 1] = posteriors.p_h0    # H0 absorbs agnostic
        merged_responsibilities[:, 2] .= 0.0               # Agnostic zeroed
        merged_responsibilities[:, 3] = posteriors.p_h1    # H1 unchanged
        # Defensive row renormalisation. On real data p_h0 + p_h1 sums to ~1, but
        # synthetic extremes (BF collapse, saturating evidence) can leave a residual
        # agnostic mass or underflow one arm, so a hard equality assert is too brittle.
        # Renormalise each row to a proper 2-class posterior; a fully-degenerate row
        # (both arms underflowed) falls back to an uninformative split.
        for i in 1:n_proteins_local
            s = merged_responsibilities[i, 1] + merged_responsibilities[i, 3]
            if s > 1e-300
                merged_responsibilities[i, 1] /= s
                merged_responsibilities[i, 3] /= s
            else
                merged_responsibilities[i, 1] = 0.5
                merged_responsibilities[i, 3] = 0.5
            end
        end
    end

    # Post-hoc monotonicity constraint (strengthened)
    # When both enrichment AND correlation favor H0, detection alone cannot drive H1 classification.
    # Detection BF is damped to sqrt(BFd) when it is the only positive arm.
    for i in 1:length(joint_bf)
        if bf.enrichment[i] < 1.0 && bf.correlation[i] < 1.0
            # Cap at max of the two primary arms (enrichment, correlation)
            max_primary = max(bf.enrichment[i], bf.correlation[i])
            if bf.detection[i] > 1.0
                # Detection alone: damp to sqrt to prevent override
                det_damped = sqrt(bf.detection[i])
                max_individual = max(max_primary, det_damped)
            else
                max_individual = max(max_primary, bf.detection[i])
            end
            if joint_bf[i] > max_individual
                joint_bf[i] = max_individual
                capped_odds = max_individual * posteriors.prior_odds
                posterior_prob[i] = capped_odds / (1.0 + capped_odds)
            end
        end
    end

    # Handle bait protein: clamp to max interaction probability
    if 1 <= refID <= length(posterior_prob)
        max_prob = maximum(posterior_prob)
        posterior_prob[refID] = max_prob
        joint_bf[refID] = (max_prob / max(1.0 - max_prob, 1e-300)) / max(posteriors.prior_odds, 1e-300)

        if verbose
            @info "Bait protein (index $refID) clamped to maximum posterior probability: $(round(max_prob, digits=4))"
        end
    end

    # Final BF clamping
    min_log_bf, max_log_bf = -46.0, 46.0
    for i in 1:length(joint_bf)
        log_bf_i = log(max(joint_bf[i], 1e-300))
        log_bf_i = clamp(log_bf_i, min_log_bf, max_log_bf)
        joint_bf[i] = exp(log_bf_i)
        capped_odds = joint_bf[i] * posteriors.prior_odds
        posterior_prob[i] = capped_odds / (1.0 + capped_odds)
    end

    # Store convergence trace: best restart's full trace + all final LLs
    convergence_trace = copy(best_em.log_likelihood)
    append!(convergence_trace, all_final_lls)

    # Extract DiscreteEmpirical detection marginals from best restart
    winning_disc_H0 = hasproperty(best_em, :disc_H0) ? best_em.disc_H0 : nothing
    winning_disc_ag = hasproperty(best_em, :disc_ag) ? best_em.disc_ag : nothing
    winning_disc_H1 = hasproperty(best_em, :disc_H1) ? best_em.disc_H1 : nothing

    # Extract h1_enrichment_sd from best EM restart
    winning_h1_sd = hasproperty(best_em, :h1_enrichment_sd) ? best_em.h1_enrichment_sd :
        _compute_h1_enrichment_sd(Float64(best_em.alpha_e1), Float64(best_em.theta_e1), winning_family)

    # Build per-dimension params from best EM restart
    _pdp = Dict{String, NamedTuple{(:mu_e, :sigma_e, :mu_c, :sigma_c, :mu_p, :sigma_p), NTuple{6, Float64}}}()
    for comp_key in ["background", "agnostic", "interaction"]
        if haskey(best_em.means, comp_key) && haskey(best_em.std_devs, comp_key)
            m = best_em.means[comp_key]
            s = best_em.std_devs[comp_key]
            _pdp[comp_key] = (mu_e=Float64(m.enrichment), sigma_e=Float64(s.enrichment),
                              mu_c=Float64(m.correlation), sigma_c=Float64(s.correlation),
                              mu_p=Float64(m.presence), sigma_p=Float64(s.presence))
        end
    end

    # Build and return 3-component result (with responsibilities, family params, BIC table, diagnostics, discrete detection, per-step LL traces, and per-dimension params)
    return LatentClassResult(
        joint_bf,
        posterior_prob,
        params,
        merged_flag ? final_pi : copy(best_em.mixing_weights),
        convergence_trace,
        best_em.converged,
        best_em.n_iterations,
        merged_flag ? merged_responsibilities : best_em.responsibilities,  # use post-merge responsibilities
        all_restart_traces,
        best_em.alpha_e1,
        best_em.theta_e1,
        winning_h1_sd,
        winning_family,
        winning_bic_table,
        em_diagnostics,
        winning_disc_H0,
        winning_disc_ag,
        winning_disc_H1,
        all_ll_traces,
        all_violations,
        _pdp,
        winning_nu_h0,    # nu_h0
        kl_val,           # kl_divergence
        merged_flag,      # merged
        anneal_schedule,  # annealing_schedule
        bc,               # bimodality_coefficient
        copy(alpha_prior),  # effective_alpha_prior
        nothing,            # prior_grid_weights
        nothing,            # prior_grid_posteriors
        false,              # eb_converged
        protein_names       # protein_names
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
