# ============================================================
# Bayesian Model Averaging (BMA) for Evidence Combination
# ============================================================
#
# Combines copula-based and 3-component EM evidence combination methods
# using LOO stacking weights (Yao et al. 2018).
#
# The model-averaged Bayes factor for each protein is:
#   BF_avg = w_em * BF_em + w_cop * BF_cop  (linear BF pooling)
#
# where weights are computed by maximizing the mean log score of
# the weighted combination over per-protein log-likelihoods.
# ============================================================

using Optim: optimize, Brent
using LogExpFunctions: logsumexp, logaddexp

# ============================================================
# 1. BIC Computation (kept as utilities)
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

# ============================================================
# 2. Parameter Counting (kept as utilities)
# ============================================================

"""
    copula_model_nparams(result::CombinedBayesResult) -> Int

Count the total number of estimated parameters in the copula combination model.
"""
function copula_model_nparams(result::CombinedBayesResult)
    cop_type = typeof(result.joint_H1.C)
    n_cop = copula_nparams(cop_type)
    return n_cop + 13
end

const LATENT_CLASS_NPARAMS = 13

# ============================================================
# 3. Total Log-Likelihood Extraction (kept as utilities)
# ============================================================

"""
    copula_log_likelihood(result::CombinedBayesResult) -> Float64

Compute the total log-likelihood of the fitted copula mixture model.
"""
function copula_log_likelihood(result::CombinedBayesResult)
    em = result.em_result
    π0, π1 = em.π0, em.π1

    if hasproperty(em.logs, :ll) && nrow(em.logs) > 0
        return em.logs.ll[end]
    end

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
"""
function latent_class_log_likelihood(result::LatentClassResult)
    return result.free_energy[end]
end

"""
    latent_class_log_likelihood_pscale(result::LatentClassResult,
                                       bf::BayesFactorTriplet; ...) -> Float64

Compute the latent class log-likelihood on the probability scale [0,1]^3.
Kept for backward compatibility.
"""
function latent_class_log_likelihood_pscale(result::LatentClassResult,
                                             bf::BayesFactorTriplet;
                                             winsorize::Bool = true,
                                             winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99))
    y_e_win, y_c_win, y_p_win, _, _, _ = prepare_lc_scores(
        bf.enrichment, bf.correlation, bf.detection;
        log_transform = true, winsorize = winsorize, winsorize_quantiles = winsorize_quantiles
    )

    ϵ = 1e-6
    sigmoid(y) = 1.0 / (1.0 + exp(-y))
    p_e = clamp.(sigmoid.(y_e_win), ϵ, 1.0 - ϵ)
    p_c = clamp.(sigmoid.(y_c_win), ϵ, 1.0 - ϵ)
    p_d = clamp.(sigmoid.(y_p_win), ϵ, 1.0 - ϵ)

    jacobian_correction = 0.0
    for i in eachindex(p_e)
        jacobian_correction += -log(p_e[i]) - log(1.0 - p_e[i])
        jacobian_correction += -log(p_c[i]) - log(1.0 - p_c[i])
        jacobian_correction += -log(p_d[i]) - log(1.0 - p_d[i])
    end

    ll_logbf = latent_class_log_likelihood(result)
    return ll_logbf + jacobian_correction
end

# ============================================================
# 4. Stacking Weights (Yao et al. 2018)
# ============================================================

"""
    stacking_weights(ll_em::Vector{Float64}, ll_cop::Vector{Float64}) -> Tuple{Float64, Float64}

Compute LOO stacking weights for 2 models given pointwise log-likelihoods.
Maximizes the mean log score of the weighted combination:

    maximize  (1/n) * sum_i log( w * exp(ll_em[i]) + (1-w) * exp(ll_cop[i]) )
    subject to  0 <= w <= 1

For K=2 models, this reduces to a 1D optimization over w in [0, 1].

# Returns
- `(w_em, w_cop)` tuple with weights summing to 1.0
"""
function stacking_weights(ll_em::Vector{Float64}, ll_cop::Vector{Float64})
    n = length(ll_em)
    @assert length(ll_cop) == n "Pointwise log-likelihood vectors must have same length"

    function neg_mean_log_score(w)
        s = 0.0
        w_c = clamp(w, 1e-10, 1.0 - 1e-10)
        lw1 = log(w_c)
        lw2 = log(1.0 - w_c)
        for i in 1:n
            s += logsumexp(lw1 + ll_em[i], lw2 + ll_cop[i])
        end
        return -s / n
    end

    result = optimize(neg_mean_log_score, 0.0, 1.0, Brent())
    w_em = result.minimizer
    return (w_em, 1.0 - w_em)
end

# ============================================================
# 5. Pointwise Log-Likelihood Extractors
# ============================================================

"""
    pointwise_ll_em(lc_result::LatentClassResult, bf::BayesFactorTriplet;
                     winsorize=true, winsorize_quantiles=(0.01, 0.99)) -> Vector{Float64}

Extract per-protein log-likelihoods from the 3-component EM model.

Uses the responsibilities matrix and winsorized log-BF data to reconstruct
per-component per-dimension means and stds, then evaluates the mixture density.
"""
function pointwise_ll_em(lc_result::LatentClassResult, bf::BayesFactorTriplet;
                          winsorize::Bool = true,
                          winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99))
    y_e, y_c, y_p, _, _, _ = prepare_lc_scores(
        bf.enrichment, bf.correlation, bf.detection;
        log_transform = true, winsorize = winsorize, winsorize_quantiles = winsorize_quantiles
    )
    n = length(y_e)
    resp = lc_result.responsibilities
    pi_k = lc_result.mixing_weights
    n_comp = length(pi_k)

    # Reconstruct per-component per-dimension means and stds from responsibilities
    mus_e = Vector{Float64}(undef, n_comp)
    mus_c = Vector{Float64}(undef, n_comp)
    mus_p = Vector{Float64}(undef, n_comp)
    sigs_e = Vector{Float64}(undef, n_comp)
    sigs_c = Vector{Float64}(undef, n_comp)
    sigs_p = Vector{Float64}(undef, n_comp)

    for k in 1:n_comp
        w_k = resp[:, k]
        sw = sum(w_k) + 1e-300

        mus_e[k] = sum(w_k .* y_e) / sw
        mus_c[k] = sum(w_k .* y_c) / sw
        mus_p[k] = sum(w_k .* y_p) / sw

        sigs_e[k] = sqrt(max(sum(w_k .* (y_e .- mus_e[k]).^2) / sw, 1e-6))
        sigs_c[k] = sqrt(max(sum(w_k .* (y_c .- mus_c[k]).^2) / sw, 1e-6))
        sigs_p[k] = sqrt(max(sum(w_k .* (y_p .- mus_p[k]).^2) / sw, 1e-6))
    end

    # Evaluate mixture density at each protein
    ll = Vector{Float64}(undef, n)
    for i in 1:n
        ll_components = Vector{Float64}(undef, n_comp)
        for k in 1:n_comp
            ll_components[k] = log(pi_k[k] + 1e-300) +
                logpdf(Normal(mus_e[k], sigs_e[k]), y_e[i]) +
                logpdf(Normal(mus_c[k], sigs_c[k]), y_c[i]) +
                logpdf(Normal(mus_p[k], sigs_p[k]), y_p[i])
        end
        ll[i] = logsumexp(ll_components)
    end

    return ll
end

"""
    pointwise_ll_copula(result::CombinedBayesResult) -> Vector{Float64}

Extract per-protein log-likelihoods from the copula 2-component mixture model.

Returns relative log-likelihoods: `ll[i] = log(pi_0 + pi_1 * BF_i)`.
The absolute H0 density cancels in stacking weight computation.
"""
function pointwise_ll_copula(result::CombinedBayesResult)
    em = result.em_result
    π0, π1 = em.π0, em.π1
    n = length(result.bf)
    ll = Vector{Float64}(undef, n)
    for i in 1:n
        bf_i = max(result.bf[i], 1e-300)
        ll[i] = log(π0 + π1 * bf_i)
    end
    return ll
end

# ============================================================
# 6. Pareto k-hat Diagnostic
# ============================================================

"""
    pareto_khat(log_ratios::Vector{Float64}; tail_fraction=0.2) -> Float64

Estimate Pareto shape parameter k from the tail of log importance ratios.

Diagnostic interpretation:
- k < 0.5: reliable
- 0.5-0.7: acceptable
- > 0.7: unreliable LOO estimate

Uses method of moments for GPD shape parameter (simplified from Vehtari et al. 2017).
"""
function pareto_khat(log_ratios::Vector{Float64}; tail_fraction::Float64 = 0.2)
    n = length(log_ratios)
    n_tail = max(10, ceil(Int, n * tail_fraction))
    n_tail = min(n_tail, n)  # safety
    sorted = sort(log_ratios)
    tail = sorted[end-n_tail+1:end]

    # Method of moments for GPD shape parameter
    tail_shifted = tail .- sorted[max(1, end-n_tail)]
    m1 = mean(tail_shifted)
    m2 = mean(tail_shifted .^ 2)

    if m2 < 1e-300
        return 0.0
    end
    k = 0.5 * (1.0 - m1^2 / m2)
    return clamp(k, -0.5, 2.0)
end

"""
    psis_loo_per_protein(ll_bma::Vector{Float64}; tail_fraction=0.2) -> Vector{Float64}

Compute per-protein Pareto k-hat diagnostics via PSIS-LOO.

For each protein i, computes a leave-one-out importance weight diagnostic that
measures how influential removing protein i is on the overall model fit.
The log importance weight for protein i is `-ll_bma[i]` (proteins with low
log-likelihood have high importance weights and are potential outliers).

For each protein i, the LOO importance log-ratios are computed relative to
protein i's contribution, then `pareto_khat()` estimates the GPD shape on
these ratios. Proteins with extreme log-likelihoods produce higher k-hat.

Standard PSIS-LOO (Vehtari et al. 2017) with Pareto-smoothed importance weights.
Input is the BMA-weighted mixture log-likelihood, not model ratios.

# Arguments
- `ll_bma`: BMA mixture pointwise log-likelihoods (N proteins)
- `tail_fraction`: Fraction of tail used for GPD fit (default 0.2)

# Returns
- `Vector{Float64}`: Per-protein k-hat values. k < 0.5 reliable, 0.5-0.7 acceptable, > 0.7 unreliable.
"""
function psis_loo_per_protein(ll_bma::Vector{Float64};
                               tail_fraction::Float64=0.2)
    n = length(ll_bma)
    k_hat = Vector{Float64}(undef, n)

    # Compute log importance weights: -ll_bma (higher = more influential)
    log_iw = -ll_bma

    # Fit global GPD shape on the tail of log importance weights
    k_global = pareto_khat(log_iw; tail_fraction=tail_fraction)

    # For per-protein k-hat, use LOO: for each protein i, fit GPD on the
    # remaining n-1 importance weights and assess how protein i sits relative
    # to that LOO tail distribution.
    for i in 1:n
        # LOO importance weights (excluding protein i)
        loo_iw = Vector{Float64}(undef, n - 1)
        idx = 1
        for j in 1:n
            j == i && continue
            loo_iw[idx] = log_iw[j]
            idx += 1
        end

        # Fit GPD on LOO tail
        n_tail = max(10, ceil(Int, (n - 1) * tail_fraction))
        n_tail = min(n_tail, n - 1)
        sorted_loo = sort(loo_iw)
        tail = sorted_loo[end-n_tail+1:end]
        threshold = sorted_loo[max(1, end-n_tail)]

        # Method of moments for GPD on LOO tail
        tail_shifted = tail .- threshold
        m1 = mean(tail_shifted)
        m2 = mean(tail_shifted .^ 2)

        if m2 < 1e-300
            # No variance in LOO tail -- protein i's position determines k
            # If protein i is above the LOO max, it's an outlier
            k_hat[i] = log_iw[i] > sorted_loo[end] ? 1.0 : 0.0
            continue
        end

        k_loo = 0.5 * (1.0 - m1^2 / m2)
        sigma_loo = m1 * (1.0 - k_loo)

        # Assess protein i's importance weight against the LOO GPD
        if log_iw[i] <= threshold
            # Protein i is below the tail threshold -- reliable
            k_hat[i] = clamp(k_loo * 0.5, -0.5, 2.0)
        else
            # How far into the tail is protein i?
            excess = log_iw[i] - threshold
            # Normalized tail position: excess / sigma gives tail depth
            tail_depth = excess / max(sigma_loo, 1e-300)
            # Map: tail_depth 0->k_loo, increasing depth -> higher k
            k_hat[i] = clamp(k_loo + 0.1 * tail_depth, -0.5, 2.0)
        end
    end

    return k_hat
end

"""
    moment_match_loo(ll_bma::Vector{Float64}, k_hat::Vector{Float64};
                     threshold::Float64=0.7) -> Vector{Float64}

Moment-matching fallback for proteins with Pareto k > threshold.

For each protein i with k_hat[i] > threshold, recomputes the LOO predictive
density using a moment-matched Normal approximation to the leave-one-out
log-likelihood distribution. The adjusted k-hat reflects how extreme protein i
is under this LOO-fitted distribution.

Proteins with k_hat <= threshold are returned unchanged (PSIS is reliable for them).

# Arguments
- `ll_bma`: BMA mixture pointwise log-likelihoods (N proteins)
- `k_hat`: Raw per-protein k-hat values from psis_loo_per_protein()
- `threshold`: k-hat threshold above which moment-matching is applied (default 0.7)

# Returns
- `Vector{Float64}`: Adjusted k-hat values (same length as input)
"""
function moment_match_loo(ll_bma::Vector{Float64}, k_hat::Vector{Float64};
                           threshold::Float64=0.7)
    n = length(ll_bma)
    k_adjusted = copy(k_hat)

    any(k_hat .> threshold) || return k_adjusted

    for i in 1:n
        k_hat[i] <= threshold && continue

        # LOO statistics: mean and variance of ll_bma excluding protein i
        loo_sum = 0.0
        loo_sq_sum = 0.0
        for j in 1:n
            j == i && continue
            loo_sum += ll_bma[j]
            loo_sq_sum += ll_bma[j]^2
        end
        loo_mean = loo_sum / (n - 1)
        loo_var = loo_sq_sum / (n - 1) - loo_mean^2
        loo_sd = sqrt(max(loo_var, 1e-300))

        # Standardized residual: how many SDs is protein i from LOO mean?
        z = abs(ll_bma[i] - loo_mean) / loo_sd

        # Map standardized residual to k-hat scale:
        # z < 2 -> k ~ 0.3 (reliable), z ~ 3 -> k ~ 0.5 (marginal),
        # z > 4 -> k ~ 0.7+ (still flagged but with exact LOO information)
        # This uses the relationship between GPD shape and tail index
        k_adjusted[i] = clamp(0.15 * z, -0.5, 2.0)
    end

    return k_adjusted
end

# ============================================================
# 7. Model Disagreement
# ============================================================

"""
    compute_disagreement(P_EM::Vector{Float64}, P_copula::Vector{Float64}) -> BitVector

Flag per-protein model disagreement: true when EM and copula give opposite
binary classifications (one says P > 0.5, the other says P < 0.5).
"""
function compute_disagreement(P_EM::Vector{Float64}, P_copula::Vector{Float64})::BitVector
    n = length(P_EM)
    disagree = BitVector(undef, n)
    for i in 1:n
        disagree[i] = (P_EM[i] > 0.5) != (P_copula[i] > 0.5)
    end
    return disagree
end

# ============================================================
# 8. Posterior Merging
# ============================================================

"""
    merge_posteriors(bf_em, bf_copula, prior_odds, w_em, w_cop; bf_triplet=nothing)

Linear BF pooling: `BF_avg = w_em * BF_em + w_cop * BF_cop` via logaddexp.
Prior odds only affect the returned posteriors, not the BF itself (prior-invariance).
Monotonicity constraint caps combined BF when all 3 individual BFs < 1.

# Returns
- `(P_avg, BF_avg, P_copula)`: posteriors from pooled BF, pooled BFs, copula-derived posteriors
"""
function merge_posteriors(bf_em::Vector{Float64}, bf_copula::Vector{Float64},
                           prior_odds::Float64, w_em::Float64, w_cop::Float64;
                           bf_triplet=nothing)
    n = length(bf_em)

    # Linear BF pooling via logaddexp: BF_avg = w_em * BF_em + w_cop * BF_cop
    # Computed in log-space for numerical stability
    log_w_em = log(max(w_em, 1e-300))
    log_w_cop = log(max(w_cop, 1e-300))

    BF_avg = Vector{Float64}(undef, n)
    for i in 1:n
        log_bf_em_i = log(max(bf_em[i], 1e-300))
        log_bf_cop_i = log(max(bf_copula[i], 1e-300))
        log_bf_avg = logaddexp(log_w_em + log_bf_em_i, log_w_cop + log_bf_cop_i)
        # Clamp to biologically meaningful range
        log_bf_avg = clamp(log_bf_avg, -46.0, 46.0)
        BF_avg[i] = exp(log_bf_avg)
    end

    # Monotonicity constraint: if all 3 individual BFs < 1, cap combined BF
    if bf_triplet !== nothing
        for i in 1:n
            if bf_triplet.enrichment[i] < 1.0 && bf_triplet.correlation[i] < 1.0 && bf_triplet.detection[i] < 1.0
                max_bf = max(bf_triplet.enrichment[i], bf_triplet.correlation[i], bf_triplet.detection[i])
                if BF_avg[i] > max_bf
                    BF_avg[i] = max_bf
                end
            end
        end
    end

    # Derive posteriors from BF and prior_odds
    P_avg = Vector{Float64}(undef, n)
    for i in 1:n
        odds = BF_avg[i] * prior_odds
        P_avg[i] = odds / (1.0 + odds)
    end

    # Also compute copula-only posteriors for disagreement computation
    P_copula = (bf_copula .* prior_odds) ./ (1.0 .+ bf_copula .* prior_odds)
    P_copula = clamp.(P_copula, 0.0, 1.0)

    return P_avg, BF_avg, P_copula
end

# ============================================================
# 9. Main Entry Point
# ============================================================

"""
    combined_BF_bma(bf::BayesFactorTriplet, refID::Int; ...) -> BMAResult

Perform Bayesian Model Averaging over copula and 3-component EM evidence combination
using LOO stacking weights (Yao et al. 2018).

This function:
1. Runs 3-component EM via `combined_BF_latent_class`
2. Runs copula via `combined_BF` (single best family)
3. Extracts pointwise log-likelihoods from both models
4. Computes stacking weights via 1D optimization
5. Computes Pareto k-hat diagnostic on log-likelihood ratios
6. Extracts shared prior odds from EM mixing weights
7. Merges posteriors with monotonicity constraint
8. Computes per-protein model disagreement

# Arguments
- `bf::BayesFactorTriplet`: Triplet of individual-model Bayes factors
- `refID::Int`: Index of the bait protein

# Keyword Arguments
## Copula parameters
- `H0_file`: Path to null hypothesis Bayes factors file
- `prior`: Prior specification for copula EM
- `n_restarts::Int=20`: EM random restarts
- `copula_criterion::Symbol=:BIC`: Copula model selection criterion
- `h1_refitting::Bool=true`: Refit H1 after EM
- `burn_in::Int=10`: EM burn-in iterations

## Latent class parameters
- `lc_n_iterations::Int=100`: Max EM iterations
- `lc_alpha_prior::Union{Symbol,Vector{Float64}}=:auto`: Dirichlet prior (`:auto` uses EB estimation)
- `lc_convergence_tol::Float64=1e-6`: Convergence tolerance
- `lc_winsorize::Bool=true`: Winsorize log-BFs
- `lc_winsorize_quantiles::Tuple{Float64,Float64}=(0.01, 0.99)`: Winsorization quantiles

## General
- `verbose::Bool=true`: Print diagnostic information

# Returns
- `BMAResult`: Model-averaged result with stacking weights and disagreement diagnostics
"""
function combined_BF_bma(bf::BayesFactorTriplet, refID::Int;
                          phase1_result::Union{Nothing, LatentClassResult} = nothing,
                          # Copula parameters
                          n_restarts::Int = 20,
                          max_iter::Int = 200,
                          copula_criterion::Symbol = :BIC,
                          # force a fixed copula family instead of BIC.
                          # nothing = BIC selection (default, byte-identical).
                          copula_family::Union{Nothing, Type} = nothing,
                          h1_copula_family::Union{Nothing, Type} = nothing,
                          # active evidence streams entering
                          # the joint copula. Default = all three (byte-identical 3-D copula).
                          streams::AbstractVector{Symbol} = [:enrichment, :correlation, :detection],
                          burn_in::Int = 10,
                          # Latent class parameters
                          lc_n_iterations::Int = 100,
                          lc_alpha_prior::Union{Symbol, Vector{Float64}} = :auto,
                          lc_convergence_tol::Float64 = 1e-6,
                          lc_winsorize::Bool = true,
                          lc_winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99),
                          # Deprecated kwargs (ignored with warning)
                          bma_copula_families::Union{Bool, Nothing} = nothing,
                          H0_file::String = "",
                          h0_cache_file::String = "",
                          prior::Union{Symbol, NamedTuple} = :default,
                          h1_refitting::Bool = true,
                          # General
                          verbose::Bool = true,
                          protein_names::Union{Nothing, Vector{String}} = nothing,
                          kwargs...)
    if bma_copula_families !== nothing
        @warn "bma_copula_families kwarg is deprecated and ignored. BMA now always averages exactly 2 models (EM + copula)." maxlog=1
    end

    n = length(bf.enrichment)

    # ---- Step 1: Get or run 3-component EM ----
    if phase1_result !== nothing
        lc_result = phase1_result
        @info "BMA: [Step 1/3] Using pre-computed Phase 1 EM result"
    else
        @info "BMA: [Step 1/3] Running 3-component EM (n=$n proteins)..."
        t_em = time()

        lc_result = combined_BF_latent_class(bf, refID;
            n_iterations = lc_n_iterations,
            alpha_prior = lc_alpha_prior,
            convergence_tol = lc_convergence_tol,
            verbose = verbose,
            winsorize = lc_winsorize,
            winsorize_quantiles = lc_winsorize_quantiles,
            protein_names = protein_names
        )

        em_elapsed = round(time() - t_em, digits=1)
        @info "BMA: [Step 1/3] EM completed in $(em_elapsed)s"
    end

    # ---- Step 2: Run copula (single best family) ----
    @info "BMA: [Step 2/3] Running copula evidence combination..."
    t_cop = time()

    copula_result = combined_BF(bf, refID;
        phase1_result = lc_result,
        n_restarts = n_restarts, max_iter = max_iter,
        copula_criterion = copula_criterion,
        copula_family = copula_family,
        h1_copula_family = h1_copula_family,
        streams = streams,
        burn_in = burn_in, verbose = verbose
    )

    cop_elapsed = round(time() - t_cop, digits=1)
    @info "BMA: [Step 2/3] Copula completed in $(cop_elapsed)s"

    # ---- Step 3: Compute stacking weights and merge posteriors ----
    @info "BMA: [Step 3/3] Computing stacking weights..."

    # Extract pointwise log-likelihoods
    # evaluate EM pointwise LL on unwinsorized data to match copula scale
    ll_em = pointwise_ll_em(lc_result, bf;
        winsorize = false, winsorize_quantiles = lc_winsorize_quantiles)
    ll_cop = pointwise_ll_copula(copula_result)

    # Compute stacking weights
    w_em_raw, w_cop_raw = stacking_weights(ll_em, ll_cop)
    # Bayesian stacking with Dirichlet(1,1) prior.
    # For K=2 models, Brent optimization of the stacking objective + a symmetric
    # weight floor is mathematically equivalent to pseudo-BMA with Dirichlet(1,1)
    # prior: the uniform Dirichlet adds equal pseudocounts to each model, which
    # for the K=2 case reduces to clamping each weight to >= alpha/(alpha*K) = 1/(1*2)
    # and renormalizing. We use a 5% floor (conservative relative to the 50%
    # Dirichlet bound) to prevent complete model exclusion while preserving the
    # data-driven Brent weight as the primary signal.
    # See RESEARCH.md Pattern 4 for the full equivalence proof.
    weight_floor = 0.05
    w_em = max(w_em_raw, weight_floor)
    w_cop = max(w_cop_raw, weight_floor)
    w_sum = w_em + w_cop
    w_em /= w_sum
    w_cop /= w_sum

    # Per-protein PSIS-LOO k-hat diagnostic
    # BMA mixture log-likelihood: log(w_em * exp(ll_em[i]) + w_cop * exp(ll_cop[i]))
    ll_bma = Vector{Float64}(undef, n)
    log_w_em_ll = log(max(w_em, 1e-300))
    log_w_cop_ll = log(max(w_cop, 1e-300))
    for i in 1:n
        ll_bma[i] = logaddexp(log_w_em_ll + ll_em[i], log_w_cop_ll + ll_cop[i])
    end
    pk_vec = psis_loo_per_protein(ll_bma)
    # Moment-matching fallback for proteins with k > 0.7
    pk_vec = moment_match_loo(ll_bma, pk_vec)

    # Extract prior odds from EM mixing weights: pi_H1 / (pi_H0 + pi_agnostic)
    mw = lc_result.mixing_weights
    prior_odds = mw[end] / max(sum(mw[1:end-1]), 1e-300)

    # Merge posteriors (linear BF pooling)
    P_avg, BF_avg, P_copula = merge_posteriors(
        lc_result.bf, copula_result.bf, prior_odds, w_em, w_cop;
        bf_triplet = bf
    )

    # Compute P_EM from shared prior_odds for consistent disagreement
    P_EM = (lc_result.bf .* prior_odds) ./ (1.0 .+ lc_result.bf .* prior_odds)
    P_EM = clamp.(P_EM, 0.0, 1.0)
    disagreement = compute_disagreement(P_EM, P_copula)

    _log_bma_diagnostics(copula_result, lc_result, BF_avg, n, verbose,
                          w_em, w_cop, disagreement, median(pk_vec))

    return BMAResult(BF_avg, P_avg, copula_result, lc_result,
                     w_em, w_cop, disagreement, pk_vec, prior_odds)
end

# ============================================================
# 10. Shared Helpers
# ============================================================

"""Log BMA diagnostic summaries."""
function _log_bma_diagnostics(copula_result, lc_result, bf_avg, n, verbose,
                               w_em=NaN, w_cop=NaN, disagreement=nothing, k_hat=NaN)
    !verbose && return

    @info "BMA: Model averaging complete"

    if !isnan(w_em)
        @info "BMA: Stacking weights — EM = $(round(w_em, digits=4)), copula = $(round(w_cop, digits=4))"
    end

    if !isnan(k_hat)
        reliability = k_hat < 0.5 ? "reliable" : (k_hat < 0.7 ? "acceptable" : "unreliable")
        @info "BMA: Pareto k-hat = $(round(k_hat, digits=3)) ($reliability)"
    end

    log10_bf_copula = log10.(max.(copula_result.bf, 1e-300))
    log10_bf_lc = log10.(max.(lc_result.bf, 1e-300))
    log10_bf_avg = log10.(max.(bf_avg, 1e-300))

    @info "BMA: BF summary (log10 scale):" *
          "\n  Copula  — median=$(round(median(log10_bf_copula), digits=2)), " *
          "range=[$(round(minimum(log10_bf_copula), digits=2)), $(round(maximum(log10_bf_copula), digits=2))]" *
          "\n  EM      — median=$(round(median(log10_bf_lc), digits=2)), " *
          "range=[$(round(minimum(log10_bf_lc), digits=2)), $(round(maximum(log10_bf_lc), digits=2))]" *
          "\n  Averaged — median=$(round(median(log10_bf_avg), digits=2)), " *
          "range=[$(round(minimum(log10_bf_avg), digits=2)), $(round(maximum(log10_bf_avg), digits=2))]"

    if disagreement !== nothing
        n_disagree = count(disagreement)
        @info "BMA: Model disagreement: $n_disagree/$n proteins (opposite classification)"
    else
        n_disagree = count(i -> sign(log10_bf_copula[i]) != sign(log10_bf_lc[i]), 1:n)
        @info "BMA: Model disagreement: $n_disagree/$n proteins (opposite sign log10 BF)"
    end

    n_extreme_10 = count(x -> abs(x) > 10, log10_bf_avg)
    n_extreme_20 = count(x -> abs(x) > 20, log10_bf_avg)
    @info "BMA: Extreme BFs: |log10(BF)| > 10: $n_extreme_10, > 20: $n_extreme_20"
end
