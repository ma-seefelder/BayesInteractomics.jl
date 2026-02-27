# ============================================================
# Bayesian Model Averaging (BMA) for Evidence Combination
# ============================================================
#
# Combines copula-based and latent class evidence combination methods
# using BIC-approximated marginal likelihoods as model weights.
#
# The model-averaged posterior probability for each protein is:
#   p_avg = w_copula * p_copula + w_lc * p_lc
#
# where weights are derived from BIC:
#   w_m = exp(-0.5 * ΔBIC_m) / Σ exp(-0.5 * ΔBIC_j)
# ============================================================

# ============================================================
# 1. BIC Computation
# ============================================================

"""
    compute_bic(log_likelihood::Float64, n_params::Int, n_observations::Int) -> Float64

Compute the Bayesian Information Criterion (BIC).

BIC = -2 * log_likelihood + n_params * log(n_observations)

Lower BIC indicates better model fit (penalized by complexity).
"""
function compute_bic(log_likelihood::Float64, n_params::Int, n_observations::Int)
    return -2.0 * log_likelihood + n_params * log(n_observations)
end

"""
    bma_weights(bic_values::Vector{Float64}) -> Vector{Float64}

Convert a vector of BIC values to BMA model weights using the standard formula:

    w_m = exp(-0.5 * ΔBIC_m) / Σ_j exp(-0.5 * ΔBIC_j)

where ΔBIC_m = BIC_m - min(BIC).

# Arguments
- `bic_values::Vector{Float64}`: BIC values for each model

# Returns
- `Vector{Float64}`: Normalized model weights summing to 1.0
"""
function bma_weights(bic_values::Vector{Float64})
    min_bic = minimum(bic_values)
    delta_bic = bic_values .- min_bic
    raw_weights = exp.(-0.5 .* delta_bic)
    return raw_weights ./ sum(raw_weights)
end

# ============================================================
# 2. Parameter Counting
# ============================================================

"""
    copula_model_nparams(result::CombinedBayesResult) -> Int

Count the total number of estimated parameters in the copula combination model.

Components:
- Copula dependence parameters (1 for Archimedean, 3 for Gaussian)
- H0 marginals: 3 Beta distributions × 2 params = 6
- H1 marginals: 3 Beta distributions × 2 params = 6
- Mixing weight: 1

Total = copula_nparams + 13
"""
function copula_model_nparams(result::CombinedBayesResult)
    cop_type = typeof(result.joint_H1.C)
    n_cop = copula_nparams(cop_type)
    # 3 H0 Beta marginals (2 params each) + 3 H1 Beta marginals (2 params each) + 1 mixing weight
    return n_cop + 13
end

const LATENT_CLASS_NPARAMS = 13  # 2 classes × 3 dims × 2 params (mu, sigma) + 1 mixing weight

# ============================================================
# 3. Log-Likelihood Extraction
# ============================================================

"""
    copula_log_likelihood(result::CombinedBayesResult) -> Float64

Compute the log-likelihood of the fitted copula mixture model.

Uses the fitted mixture: π₀ · f₀(x) + π₁ · f₁(x) evaluated at all data points,
where f₀ and f₁ are the joint H0 and H1 densities.
"""
function copula_log_likelihood(result::CombinedBayesResult)
    em = result.em_result
    π0, π1 = em.π0, em.π1
    joint_H0 = result.joint_H0
    joint_H1 = result.joint_H1

    # Extract the final log-likelihood from the EM convergence logs
    # The EM logs DataFrame has a :ll column tracking per-iteration log-likelihood
    if hasproperty(em.logs, :ll) && nrow(em.logs) > 0
        return em.logs.ll[end]
    end

    # Fallback: reconstruct mixture log-likelihood from BFs and mixing weights
    # log p(x_i) ≈ log(f₀(x_i)) + log(π₀ + π₁ · BF_i)
    # The absolute f₀ term cancels in the ΔBIC, so we use the relative quantity
    n = length(result.bf)
    ll = 0.0
    for i in 1:n
        bf_i = max(result.bf[i], 1e-300)
        ll += log(π0 + π1 * bf_i)
    end
    return ll
end

"""
    latent_class_log_likelihood(result::LatentClassResult) -> Float64

Extract the final log-likelihood from the latent class EM fitting.

Returns the last element of `result.free_energy`, which stores per-iteration log-likelihoods.
"""
function latent_class_log_likelihood(result::LatentClassResult)
    return result.free_energy[end]
end

# ============================================================
# 4. Main Entry Point
# ============================================================

"""
    combined_BF_bma(bf::BayesFactorTriplet, refID::Int;
                     H0_file="copula_H0.xlsx", verbose=true,
                     copula_kwargs..., lc_kwargs...) -> BMAResult

Perform Bayesian Model Averaging over copula and latent class evidence combination.

This function:
1. Runs copula-based combination (`combined_BF`)
2. Runs latent class combination (`combined_BF_latent_class`)
3. Computes BIC for each model
4. Computes BMA weights from BIC differences
5. Model-averages posterior probabilities: p_avg = w_copula * p_copula + w_lc * p_lc
6. Derives averaged BFs from averaged posteriors

# Arguments
- `bf::BayesFactorTriplet`: Triplet of individual-model Bayes factors
- `refID::Int`: Index of the bait protein

# Keyword Arguments
## Copula parameters
- `H0_file`: Path to null hypothesis Bayes factors file
- `prior`: Prior specification for copula EM (`:default` or NamedTuple)
- `n_restarts::Int=20`: EM random restarts
- `copula_criterion::Symbol=:BIC`: Copula model selection criterion
- `h1_refitting::Bool=true`: Refit H1 after EM
- `burn_in::Int=10`: EM burn-in iterations

## Latent class parameters
- `lc_n_iterations::Int=100`: Max EM iterations
- `lc_alpha_prior::Vector{Float64}=[10.0, 1.0]`: Dirichlet prior
- `lc_convergence_tol::Float64=1e-6`: Convergence tolerance
- `lc_winsorize::Bool=true`: Winsorize log-BFs
- `lc_winsorize_quantiles::Tuple{Float64,Float64}=(0.01, 0.99)`: Winsorization quantiles

## General
- `verbose::Bool=true`: Print diagnostic information

# Returns
- `BMAResult`: Model-averaged result with individual model results and BIC weights
"""
function combined_BF_bma(bf::BayesFactorTriplet, refID::Int;
                          H0_file::String = "copula_H0.xlsx",
                          # Copula parameters
                          prior::Union{Symbol, NamedTuple} = :default,
                          n_restarts::Int = 20,
                          copula_criterion::Symbol = :BIC,
                          h1_refitting::Bool = true,
                          burn_in::Int = 10,
                          # Latent class parameters
                          lc_n_iterations::Int = 100,
                          lc_alpha_prior::Vector{Float64} = [10.0, 1.0],
                          lc_convergence_tol::Float64 = 1e-6,
                          lc_winsorize::Bool = true,
                          lc_winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99),
                          # General
                          verbose::Bool = true,
                          kwargs...)
    n = length(bf.enrichment)

    # ---- Step 1: Run copula combination ----
    if verbose
        @info "BMA: Running copula-based evidence combination..."
    end

    copula_result = combined_BF(bf, refID;
        H0_file = H0_file,
        max_iter = 5_000,
        prior = prior,
        n_restarts = n_restarts,
        copula_criterion = copula_criterion,
        h1_refitting = h1_refitting,
        burn_in = burn_in,
        verbose = verbose
    )

    # ---- Step 2: Run latent class combination ----
    if verbose
        @info "BMA: Running latent class evidence combination..."
    end

    lc_result = combined_BF_latent_class(bf, refID;
        n_iterations = lc_n_iterations,
        alpha_prior = lc_alpha_prior,
        convergence_tol = lc_convergence_tol,
        verbose = verbose,
        winsorize = lc_winsorize,
        winsorize_quantiles = lc_winsorize_quantiles
    )

    # ---- Step 3: Compute BIC for each model ----
    copula_ll = copula_log_likelihood(copula_result)
    lc_ll = latent_class_log_likelihood(lc_result)

    n_copula_params = copula_model_nparams(copula_result)
    n_lc_params = LATENT_CLASS_NPARAMS

    copula_bic = compute_bic(copula_ll, n_copula_params, n)
    lc_bic = compute_bic(lc_ll, n_lc_params, n)

    if verbose
        @info "BMA: Copula  — log-lik = $(round(copula_ll, digits=2)), k = $n_copula_params, BIC = $(round(copula_bic, digits=2))"
        @info "BMA: Latent class — log-lik = $(round(lc_ll, digits=2)), k = $n_lc_params, BIC = $(round(lc_bic, digits=2))"
    end

    # ---- Step 4: Compute BMA weights ----
    weights = bma_weights([copula_bic, lc_bic])
    w_copula = weights[1]
    w_lc = weights[2]

    if verbose
        @info "BMA: Model weights — copula = $(round(w_copula, digits=4)), latent class = $(round(w_lc, digits=4))"
    end

    # ---- Step 5: Model-average posterior probabilities ----
    p_avg = w_copula .* copula_result.posterior_prob .+ w_lc .* lc_result.posterior_prob

    # ---- Step 6: Derive averaged BFs from averaged posteriors ----
    # Use the copula mixing weight as prior (π₁ from copula EM)
    prior_odds = copula_result.em_result.π1 / max(copula_result.em_result.π0, 1e-300)
    bf_avg = Vector{Float64}(undef, n)
    for i in 1:n
        posterior_odds_i = p_avg[i] / max(1.0 - p_avg[i], 1e-300)
        bf_avg[i] = posterior_odds_i / max(prior_odds, 1e-300)
    end

    if verbose
        @info "BMA: Model averaging complete"
    end

    return BMAResult(
        bf_avg,
        p_avg,
        copula_result,
        lc_result,
        copula_bic,
        lc_bic,
        w_copula,
        w_lc
    )
end
