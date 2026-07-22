# ============================================================
# Parametric Simulation Engine for FDR Calibration
# ============================================================
#
# Draws synthetic BF triplets from a fitted LatentClassResult,
# sweeps a 5x5 grid of (pi_H1, effect_scale) scenarios with
# 10 replicates each, computes FDR/sensitivity/specificity/AUC/
# reliability calibration curves, and caches results to JLD2.
#
# BayesInteractomics v1.1
# ============================================================

# JEFFREYS_SHIFT is defined in src/core/types.jl
# CALIBRATION_CACHE_VERSION is defined in src/core/intermediate_cache.jl (per-cache version split)
# and governs both the simulation grid cache and the Platt calibration cache below.

import Dates: now, DateTime
import Statistics: quantile, mean, median, std
import Random
import Optim

using Distributions: Gamma, LogNormal, Weibull, Normal, Categorical
using LogExpFunctions: logsumexp, logit, logistic

# ============================================================
# 1. Types
# ============================================================

"""
    ScenarioResult

Pre-summarized calibration data for one (pi_H1, effect_scale) scenario.
All curve arrays have length = n_thresholds (e.g., 200).

Confidence bands are 2.5th/97.5th percentile across replicates (nonparametric).
"""
struct ScenarioResult
    pi_h1::Float64
    effect_scale::Float64
    thresholds::Vector{Float64}                # n_thresholds-point grid [0, 1]
    fdr_median::Vector{Float64}
    fdr_lo::Vector{Float64}                    # 2.5th percentile across replicates
    fdr_hi::Vector{Float64}                    # 97.5th percentile
    sensitivity_median::Vector{Float64}
    sensitivity_lo::Vector{Float64}
    sensitivity_hi::Vector{Float64}
    specificity_median::Vector{Float64}
    specificity_lo::Vector{Float64}
    specificity_hi::Vector{Float64}
    auc_median::Float64                        # scalar AUC (median across replicates)
    auc_lo::Float64
    auc_hi::Float64
    # Reliability diagram (binned)
    reliability_bin_centers::Vector{Float64}
    reliability_observed_median::Vector{Float64}
    reliability_observed_lo::Vector{Float64}
    reliability_observed_hi::Vector{Float64}
end

"""
    SimulationResult

Complete output of `run_simulation()`.

Contains one `ScenarioResult` per (pi_H1, effect_scale) combination in the
5x5 grid, plus global metadata.
"""
struct SimulationResult
    scenarios::Vector{ScenarioResult}
    pi_h1_grid::Vector{Float64}             # [0.02, 0.05, 0.10, 0.15, 0.20]
    effect_grid::Vector{Float64}            # [0.5, 0.75, 1.0, 1.25, 1.5]
    n_synthetic::Int
    n_replicates::Int
    h1_enrichment_family::Symbol            # from lc_result
    fdr_at_p95_range::Tuple{Float64, Float64}  # (min, max) FDR at P>0.95 across scenarios
    # Isotonic calibration additions
    calibration_model::Union{Nothing, Any}        # CalibrationModel or nothing
    fdr_calibration_model::Union{Nothing, Any}    # FDRCalibrationModel or nothing
    calibration_cv::Union{Nothing, Any}           # CalibrationCVMetrics or nothing
    # Pre-computed empirical FDR curves from simulation
    fdr_curve_empirical::Vector{Float64}          # Empirical FDR from simulation ground truth (100 points)
    fdr_curve_declared_bfdr::Vector{Float64}      # Declared BFDR from simulation posteriors (100 points)
end

# Backward-compatible constructor (7 positional args, fills calibration fields with nothing)
# 7-arg convenience constructor (original)
function SimulationResult(scenarios, pi_h1_grid, effect_grid, n_synthetic,
                           n_replicates, h1_enrichment_family, fdr_at_p95_range)
    SimulationResult(scenarios, pi_h1_grid, effect_grid, n_synthetic,
                     n_replicates, h1_enrichment_family, fdr_at_p95_range,
                     nothing, nothing, nothing, Float64[], Float64[])
end

# 10-arg backward-compat constructor (with calibration fields)
function SimulationResult(scenarios, pi_h1_grid, effect_grid, n_synthetic,
                           n_replicates, h1_enrichment_family, fdr_at_p95_range,
                           calibration_model, fdr_calibration_model, calibration_cv)
    SimulationResult(scenarios, pi_h1_grid, effect_grid, n_synthetic,
                     n_replicates, h1_enrichment_family, fdr_at_p95_range,
                     calibration_model, fdr_calibration_model, calibration_cv,
                     Float64[], Float64[])
end

# ============================================================
# 1b. Calibration Types
# ============================================================

"""
    CalibrationModel

Platt scaling calibration model: calibrated = logistic(a * logit(raw) + b).
Parameters a, b are fitted by minimizing cross-entropy loss via Optim.jl.
Falls back to identity mapping when fitting fails.
"""
struct CalibrationModel
    a::Float64           # slope in logit space
    b::Float64           # intercept in logit space
    n_training::Int
    converged::Bool      # whether Optim converged
end

"""
    FDRCalibrationModel

Platt scaling FDR calibration: calibrated_fdr = logistic(a * threshold + b).
FDR decreases as threshold increases, so the mapping is applied directly to thresholds.
"""
struct FDRCalibrationModel
    a::Float64
    b::Float64
    n_training::Int
    converged::Bool
end

"""
    CalibrationCVMetrics

Results from 10-fold stratified cross-validation of calibration quality.
"""
struct CalibrationCVMetrics
    n_folds::Int
    posterior_ece_per_fold::Vector{Float64}
    posterior_ece_mean::Float64
    posterior_ece_std::Float64
    fdr_ece_per_fold::Vector{Float64}
    fdr_ece_mean::Float64
    fdr_ece_std::Float64
    raw_rel_bins::Vector{Float64}
    raw_rel_observed::Vector{Float64}
    cal_rel_bins::Vector{Float64}
    cal_rel_observed::Vector{Float64}
    passes_ece_threshold::Bool
    ece_badge_color::String
end

# ============================================================
# 2. Sampling Helpers
# ============================================================

"""
    _draw_h1_enrichment(family::Symbol, alpha::Float64, theta::Float64)::Float64

Draw a single value from the BIC-selected H1 enrichment distribution
and add JEFFREYS_SHIFT to place it on the log-BF scale.

Dispatches on `family`:
- `:gamma`    — `Gamma(alpha, theta)` where alpha=shape, theta=scale
- `:lognormal`— `LogNormal(alpha, theta)` where alpha=mu_log, theta=sigma_log
                (parameters are on the log-scale, as fitted by _fit_lognormal_weighted)
- `:weibull`  — `Weibull(alpha, theta)` where alpha=shape, theta=scale

The stored alpha/theta represent the distribution of `(log_BF_enrichment - JEFFREYS_SHIFT)`,
so JEFFREYS_SHIFT is added back to samples to obtain log-BF scale values.
"""
function _draw_h1_enrichment(family::Symbol, alpha::Float64, theta::Float64)::Float64
    dist = if family == :lognormal
        LogNormal(alpha, theta)       # alpha = mu_log, theta = sigma_log
    elseif family == :weibull
        Weibull(alpha, theta)         # alpha = shape, theta = scale
    else  # :gamma (default)
        Gamma(alpha, theta)           # alpha = shape, theta = scale
    end
    return rand(dist) + JEFFREYS_SHIFT
end

"""
    _draw_synthetic_triplet(label::Int, lc_result::LatentClassResult,
                            effect_scale::Float64)

Draw a log-BF triplet (enrichment, correlation, detection) for a synthetic protein
given its component assignment.

- label=1 : H0 (background)
- label=2 : Agnostic
- label=3 : H1 (interaction)

For H0/Agnostic: all three dimensions drawn from Normal(class.mu, class.sigma).
For H1:
  - enrichment: drawn from BIC-selected shifted family, then multiplied by effect_scale
  - correlation/detection: Normal(int.mu * effect_scale, int.sigma)

Uses dimension-averaged mu/sigma from `class_parameters` (consistent with EM fitting).
"""
function _draw_synthetic_triplet(label::Int, lc_result::LatentClassResult,
                                  effect_scale::Float64)
    pdp = lc_result.per_dimension_params
    cp = lc_result.class_parameters  # fallback when pdp is nothing

    if label == 1  # H0 / background
        if pdp !== nothing && haskey(pdp, "background")
            p = pdp["background"]
            lbf_e = rand(Normal(p.mu_e, p.sigma_e))
            lbf_c = rand(Normal(p.mu_c, p.sigma_c))
            lbf_d = rand(Normal(p.mu_p, p.sigma_p))
        else
            bg = cp["background"]
            lbf_e = rand(Normal(bg.mu, bg.sigma))
            lbf_c = rand(Normal(bg.mu, bg.sigma))
            lbf_d = rand(Normal(bg.mu, bg.sigma))
        end
    elseif label == 2  # Agnostic
        if pdp !== nothing && haskey(pdp, "agnostic")
            p = pdp["agnostic"]
            lbf_e = rand(Normal(p.mu_e, p.sigma_e))
            lbf_c = rand(Normal(p.mu_c, p.sigma_c))
            lbf_d = rand(Normal(p.mu_p, p.sigma_p))
        else
            ag = haskey(cp, "agnostic") ? cp["agnostic"] : cp["background"]
            lbf_e = rand(Normal(ag.mu, ag.sigma))
            lbf_c = rand(Normal(ag.mu, ag.sigma))
            lbf_d = rand(Normal(ag.mu, ag.sigma))
        end
    else  # H1 / interaction (label == 3)
        # Enrichment: always use BIC-selected family (unchanged)
        lbf_e = _draw_h1_enrichment(lc_result.h1_enrichment_family,
                                    lc_result.alpha_enrichment_h1,
                                    lc_result.theta_enrichment_h1) * effect_scale
        # Correlation and detection: use per-dimension if available
        if pdp !== nothing && haskey(pdp, "interaction")
            p = pdp["interaction"]
            lbf_c = rand(Normal(p.mu_c * effect_scale, p.sigma_c))
            lbf_d = rand(Normal(p.mu_p * effect_scale, p.sigma_p))
        else
            int = cp["interaction"]
            lbf_c = rand(Normal(int.mu * effect_scale, int.sigma))
            lbf_d = rand(Normal(int.mu * effect_scale, int.sigma))
        end
    end

    return (lbf_e, lbf_c, lbf_d)
end

# ============================================================
# 3. Posterior Computation
# ============================================================

"""
    _logpdf_h1_enrich(family::Symbol, alpha::Float64, theta::Float64,
                       lbf_e_shifted::Float64)::Float64

Compute log-PDF of H1 enrichment for a shifted log-BF value.
Returns -Inf if the shifted value is non-positive (below JEFFREYS_SHIFT).
"""
function _logpdf_h1_enrich(family::Symbol, alpha::Float64, theta::Float64,
                             lbf_e_shifted::Float64)::Float64
    lbf_e_shifted <= 0.0 && return -Inf
    if family == :lognormal
        return logpdf(LogNormal(alpha, theta), lbf_e_shifted)
    elseif family == :weibull
        return logpdf(Weibull(alpha, theta), lbf_e_shifted)
    else  # :gamma
        return logpdf(Gamma(alpha, theta), lbf_e_shifted)
    end
end

"""
    _synthetic_posterior(lbf_e::Float64, lbf_c::Float64, lbf_d::Float64,
                          lc_result::LatentClassResult)::Float64

Compute P(H1 | data) for a synthetic protein using the fitted LatentClassResult
responsibility formula directly (no EM re-fitting).

For H1 enrichment: applies JEFFREYS_SHIFT threshold — values below get ll_h1_enrich=-Inf.
For all other dimensions (H0/Agnostic enrichment, correlation, detection): Normal.
"""
function _synthetic_posterior(lbf_e::Float64, lbf_c::Float64, lbf_d::Float64,
                               lc_result::LatentClassResult)::Float64
    pdp = lc_result.per_dimension_params
    cp = lc_result.class_parameters
    pi = lc_result.mixing_weights

    # H0 log-likelihood
    if pdp !== nothing && haskey(pdp, "background")
        p = pdp["background"]
        ll_h0 = log(max(pi[1], 1e-300)) +
                logpdf(Normal(p.mu_e, p.sigma_e), lbf_e) +
                logpdf(Normal(p.mu_c, p.sigma_c), lbf_c) +
                logpdf(Normal(p.mu_p, p.sigma_p), lbf_d)
    else
        bg = cp["background"]
        ll_h0 = log(max(pi[1], 1e-300)) +
                logpdf(Normal(bg.mu, bg.sigma), lbf_e) +
                logpdf(Normal(bg.mu, bg.sigma), lbf_c) +
                logpdf(Normal(bg.mu, bg.sigma), lbf_d)
    end

    # Agnostic log-likelihood
    pi_ag = length(pi) >= 3 ? pi[2] : 0.0
    if pdp !== nothing && haskey(pdp, "agnostic")
        p = pdp["agnostic"]
        ll_ag = log(max(pi_ag, 1e-300)) +
                logpdf(Normal(p.mu_e, p.sigma_e), lbf_e) +
                logpdf(Normal(p.mu_c, p.sigma_c), lbf_c) +
                logpdf(Normal(p.mu_p, p.sigma_p), lbf_d)
    else
        ag = haskey(cp, "agnostic") ? cp["agnostic"] : cp["background"]
        ll_ag = log(max(pi_ag, 1e-300)) +
                logpdf(Normal(ag.mu, ag.sigma), lbf_e) +
                logpdf(Normal(ag.mu, ag.sigma), lbf_c) +
                logpdf(Normal(ag.mu, ag.sigma), lbf_d)
    end

    # H1 log-likelihood (enrichment uses BIC family, correlation/detection per-dimension)
    pi_h1 = length(pi) >= 3 ? pi[3] : pi[2]
    lbf_e_shifted = lbf_e - JEFFREYS_SHIFT
    ll_h1_enrich = _logpdf_h1_enrich(lc_result.h1_enrichment_family,
                                      lc_result.alpha_enrichment_h1,
                                      lc_result.theta_enrichment_h1,
                                      lbf_e_shifted)
    if pdp !== nothing && haskey(pdp, "interaction")
        p = pdp["interaction"]
        ll_h1 = log(max(pi_h1, 1e-300)) +
                 ll_h1_enrich +
                 logpdf(Normal(p.mu_c, p.sigma_c), lbf_c) +
                 logpdf(Normal(p.mu_p, p.sigma_p), lbf_d)
    else
        int = cp["interaction"]
        ll_h1 = log(max(pi_h1, 1e-300)) +
                 ll_h1_enrich +
                 logpdf(Normal(int.mu, int.sigma), lbf_c) +
                 logpdf(Normal(int.mu, int.sigma), lbf_d)
    end

    # Aggregate via logsumexp
    all_lls = if length(pi) >= 3
        [ll_h0, ll_ag, ll_h1]
    else
        [ll_h0, ll_h1]
    end

    log_sum = logsumexp(all_lls)
    return isfinite(log_sum) ? exp(ll_h1 - log_sum) : 0.0
end

# ============================================================
# 4. Calibration Curves
# ============================================================

"""
    _compute_calibration_curves(posteriors::Vector{Float64},
                                 ground_truth::BitVector,
                                 thresholds::Vector{Float64})

Compute FDR, sensitivity, and specificity at each threshold in `thresholds`.

- FDR = FP / (FP + TP). Returns 0.0 when no predicted positives (conservative NaN handling).
- Sensitivity = TP / n_pos. Returns NaN when n_pos = 0.
- Specificity = TN / n_neg. Returns NaN when n_neg = 0.
"""
function _compute_calibration_curves(posteriors::Vector{Float64},
                                      ground_truth::BitVector,
                                      thresholds::Vector{Float64})
    n = length(thresholds)
    fdr = Vector{Float64}(undef, n)
    sensitivity = Vector{Float64}(undef, n)
    specificity = Vector{Float64}(undef, n)

    n_pos = sum(ground_truth)
    n_neg = length(ground_truth) - n_pos

    for (j, t) in enumerate(thresholds)
        predicted_pos = posteriors .>= t
        tp = sum(predicted_pos .& ground_truth)
        fp = sum(predicted_pos .& .!ground_truth)
        tn = sum(.!predicted_pos .& .!ground_truth)

        # FDR: 0.0 when no predicted positives (conservative interpretation)
        denom_fdr = tp + fp
        fdr[j] = denom_fdr > 0 ? fp / denom_fdr : 0.0

        sensitivity[j] = n_pos > 0 ? tp / n_pos : NaN
        specificity[j] = n_neg > 0 ? tn / n_neg : NaN
    end

    return fdr, sensitivity, specificity
end

"""
    _compute_auc(posteriors::Vector{Float64}, ground_truth::BitVector)::Float64

Compute the standard ROC AUC (area under the ROC curve) using the trapezoidal rule.
Returns a value in [0.0, 1.0].
"""
function _compute_auc(posteriors::Vector{Float64}, ground_truth::BitVector)::Float64
    n = length(posteriors)
    n_pos = sum(ground_truth)
    n_neg = n - n_pos

    (n_pos == 0 || n_neg == 0) && return 0.5  # degenerate case

    # Sort by descending posterior (higher threshold = fewer predicted positives)
    order = sortperm(posteriors; rev=true)
    sorted_gt = ground_truth[order]

    # Accumulate TPR/FPR for ROC curve
    tpr = Vector{Float64}(undef, n + 2)
    fpr = Vector{Float64}(undef, n + 2)
    tpr[1] = 0.0; fpr[1] = 0.0
    cum_tp = 0
    cum_fp = 0
    for i in 1:n
        if sorted_gt[i]
            cum_tp += 1
        else
            cum_fp += 1
        end
        tpr[i+1] = cum_tp / n_pos
        fpr[i+1] = cum_fp / n_neg
    end
    tpr[n+2] = 1.0; fpr[n+2] = 1.0

    # Trapezoidal AUC
    auc = 0.0
    for i in 1:(n+1)
        auc += (fpr[i+1] - fpr[i]) * (tpr[i+1] + tpr[i]) / 2.0
    end
    return clamp(auc, 0.0, 1.0)
end

"""
    _compute_reliability(posteriors::Vector{Float64}, ground_truth::BitVector,
                          bins::Vector{Float64})

Bin posteriors by predicted probability and compute observed H1 fraction per bin.

Returns `(bin_centers::Vector{Float64}, observed_fractions::Vector{Float64})`.
`observed_fractions[j]` = fraction of true H1 among proteins whose posterior
falls in the j-th bin. Returns `NaN` for empty bins.
"""
function _compute_reliability(posteriors::Vector{Float64}, ground_truth::BitVector,
                               bins::Vector{Float64})
    n_bins = length(bins) - 1
    bin_centers = Vector{Float64}(undef, n_bins)
    observed_fractions = Vector{Float64}(undef, n_bins)

    for j in 1:n_bins
        lo = bins[j]
        hi = bins[j+1]
        bin_centers[j] = (lo + hi) / 2.0

        # Assign to bin: [lo, hi) except last bin includes hi
        if j < n_bins
            mask = (posteriors .>= lo) .& (posteriors .< hi)
        else
            mask = posteriors .>= lo
        end

        n_in_bin = sum(mask)
        if n_in_bin > 0
            observed_fractions[j] = sum(ground_truth[mask]) / n_in_bin
        else
            observed_fractions[j] = NaN
        end
    end

    return bin_centers, observed_fractions
end

# ============================================================
# 5. Cache Hash
# ============================================================

"""
    _simulation_param_hash(lc_result::LatentClassResult,
                            pi_h1_grid, effect_grid,
                            n_synthetic::Int, n_replicates::Int)::UInt64

Deterministic hash over all parameters that determine the simulation output.
Any change in these parameters triggers cache invalidation.
"""
function _simulation_param_hash(lc_result::LatentClassResult,
                                 pi_h1_grid,
                                 effect_grid,
                                 n_synthetic::Int,
                                 n_replicates::Int)::UInt64
    cp = lc_result.class_parameters
    mw = lc_result.mixing_weights

    return hash((
        collect(Float64, pi_h1_grid),
        collect(Float64, effect_grid),
        n_synthetic,
        n_replicates,
        mw,
        cp["background"].mu, cp["background"].sigma,
        haskey(cp, "agnostic") ? cp["agnostic"].mu : 0.0,
        haskey(cp, "agnostic") ? cp["agnostic"].sigma : 0.0,
        cp["interaction"].mu, cp["interaction"].sigma,
        lc_result.alpha_enrichment_h1,
        lc_result.theta_enrichment_h1,
        lc_result.h1_enrichment_family
    ))
end

# ============================================================
# 6. JLD2 Cache I/O
# ============================================================

"""
    save_simulation_cache(result::SimulationResult, param_hash::UInt64, filepath::String)

Save simulation results to JLD2 with compression. Follows the H0Cache pattern.
Uses CALIBRATION_CACHE_VERSION for compatibility checking.
Timestamp stored as String (JLD2 DateTime serialization note).
"""
function save_simulation_cache(result::SimulationResult, param_hash::UInt64,
                                 filepath::String)
    # Pre-serialize ScenarioResult fields as plain arrays for JLD2 compatibility
    n_scenarios = length(result.scenarios)
    pi_h1_vec        = [sc.pi_h1 for sc in result.scenarios]
    effect_scale_vec = [sc.effect_scale for sc in result.scenarios]
    thresholds_mat   = hcat([sc.thresholds for sc in result.scenarios]...)     # n_thresh x n_scen
    fdr_med_mat      = hcat([sc.fdr_median for sc in result.scenarios]...)
    fdr_lo_mat       = hcat([sc.fdr_lo for sc in result.scenarios]...)
    fdr_hi_mat       = hcat([sc.fdr_hi for sc in result.scenarios]...)
    sens_med_mat     = hcat([sc.sensitivity_median for sc in result.scenarios]...)
    sens_lo_mat      = hcat([sc.sensitivity_lo for sc in result.scenarios]...)
    sens_hi_mat      = hcat([sc.sensitivity_hi for sc in result.scenarios]...)
    spec_med_mat     = hcat([sc.specificity_median for sc in result.scenarios]...)
    spec_lo_mat      = hcat([sc.specificity_lo for sc in result.scenarios]...)
    spec_hi_mat      = hcat([sc.specificity_hi for sc in result.scenarios]...)
    auc_med_vec      = [sc.auc_median for sc in result.scenarios]
    auc_lo_vec       = [sc.auc_lo for sc in result.scenarios]
    auc_hi_vec       = [sc.auc_hi for sc in result.scenarios]
    rel_bins_mat     = hcat([sc.reliability_bin_centers for sc in result.scenarios]...)
    rel_obs_med_mat  = hcat([sc.reliability_observed_median for sc in result.scenarios]...)
    rel_obs_lo_mat   = hcat([sc.reliability_observed_lo for sc in result.scenarios]...)
    rel_obs_hi_mat   = hcat([sc.reliability_observed_hi for sc in result.scenarios]...)

    # Serialize calibration fields
    has_cal = !isnothing(result.calibration_model)
    has_fdr_cal = !isnothing(result.fdr_calibration_model)
    has_cv = !isnothing(result.calibration_cv)

    jldsave(filepath; compress=true,
        cache_version        = CALIBRATION_CACHE_VERSION,
        cache_type           = "simulation",
        n_scenarios          = n_scenarios,
        pi_h1_vec            = pi_h1_vec,
        effect_scale_vec     = effect_scale_vec,
        thresholds_mat       = thresholds_mat,
        fdr_med_mat          = fdr_med_mat,
        fdr_lo_mat           = fdr_lo_mat,
        fdr_hi_mat           = fdr_hi_mat,
        sens_med_mat         = sens_med_mat,
        sens_lo_mat          = sens_lo_mat,
        sens_hi_mat          = sens_hi_mat,
        spec_med_mat         = spec_med_mat,
        spec_lo_mat          = spec_lo_mat,
        spec_hi_mat          = spec_hi_mat,
        auc_med_vec          = auc_med_vec,
        auc_lo_vec           = auc_lo_vec,
        auc_hi_vec           = auc_hi_vec,
        rel_bins_mat         = rel_bins_mat,
        rel_obs_med_mat      = rel_obs_med_mat,
        rel_obs_lo_mat       = rel_obs_lo_mat,
        rel_obs_hi_mat       = rel_obs_hi_mat,
        pi_h1_grid           = result.pi_h1_grid,
        effect_grid          = result.effect_grid,
        n_synthetic          = result.n_synthetic,
        n_replicates         = result.n_replicates,
        h1_enrichment_family = string(result.h1_enrichment_family),
        fdr_p95_min          = result.fdr_at_p95_range[1],
        fdr_p95_max          = result.fdr_at_p95_range[2],
        param_hash           = param_hash,
        timestamp            = string(now()),
        # Calibration fields
        has_cal              = has_cal,
        cal_a                = has_cal ? result.calibration_model.a : 1.0,
        cal_b                = has_cal ? result.calibration_model.b : 0.0,
        cal_n_training       = has_cal ? result.calibration_model.n_training : 0,
        cal_converged        = has_cal ? result.calibration_model.converged : false,
        has_fdr_cal          = has_fdr_cal,
        fdr_cal_a            = has_fdr_cal ? result.fdr_calibration_model.a : 1.0,
        fdr_cal_b            = has_fdr_cal ? result.fdr_calibration_model.b : 0.0,
        fdr_cal_n_training   = has_fdr_cal ? result.fdr_calibration_model.n_training : 0,
        fdr_cal_converged    = has_fdr_cal ? result.fdr_calibration_model.converged : false,
        has_cv               = has_cv,
        cv_n_folds           = has_cv ? result.calibration_cv.n_folds : 0,
        cv_post_ece_folds    = has_cv ? result.calibration_cv.posterior_ece_per_fold : Float64[],
        cv_post_ece_mean     = has_cv ? result.calibration_cv.posterior_ece_mean : 0.0,
        cv_post_ece_std      = has_cv ? result.calibration_cv.posterior_ece_std : 0.0,
        cv_fdr_ece_folds     = has_cv ? result.calibration_cv.fdr_ece_per_fold : Float64[],
        cv_fdr_ece_mean      = has_cv ? result.calibration_cv.fdr_ece_mean : 0.0,
        cv_fdr_ece_std       = has_cv ? result.calibration_cv.fdr_ece_std : 0.0,
        cv_raw_rel_bins      = has_cv ? result.calibration_cv.raw_rel_bins : Float64[],
        cv_raw_rel_obs       = has_cv ? result.calibration_cv.raw_rel_observed : Float64[],
        cv_cal_rel_bins      = has_cv ? result.calibration_cv.cal_rel_bins : Float64[],
        cv_cal_rel_obs       = has_cv ? result.calibration_cv.cal_rel_observed : Float64[],
        cv_passes            = has_cv ? result.calibration_cv.passes_ece_threshold : false,
        cv_badge_color       = has_cv ? result.calibration_cv.ece_badge_color : "red",
        # FDR curve fields
        has_fdr_curves       = !isempty(result.fdr_curve_empirical),
        fdr_curve_emp        = result.fdr_curve_empirical,
        fdr_curve_decl       = result.fdr_curve_declared_bfdr,
    )
end

"""
    load_simulation_cache(filepath::String, expected_hash::UInt64)::Union{SimulationResult, Nothing}

Load simulation cache from JLD2. Returns `nothing` if:
- File doesn't exist
- Cache version mismatch
- Cache type mismatch
- param_hash mismatch (parameter change)
- Loading fails due to corruption
"""
function load_simulation_cache(filepath::String,
                                 expected_hash::UInt64)::Union{SimulationResult, Nothing}
    !isfile(filepath) && return nothing
    try
        data = load(filepath)
        get(data, "cache_version", 0) != CALIBRATION_CACHE_VERSION && return nothing
        get(data, "cache_type", "") != "simulation" && return nothing
        get(data, "param_hash", UInt64(0)) != expected_hash && return nothing

        n_sc        = data["n_scenarios"]
        pi_h1_vec   = data["pi_h1_vec"]
        eff_vec     = data["effect_scale_vec"]
        thr_mat     = data["thresholds_mat"]
        fdr_med     = data["fdr_med_mat"]
        fdr_lo      = data["fdr_lo_mat"]
        fdr_hi      = data["fdr_hi_mat"]
        sens_med    = data["sens_med_mat"]
        sens_lo     = data["sens_lo_mat"]
        sens_hi     = data["sens_hi_mat"]
        spec_med    = data["spec_med_mat"]
        spec_lo     = data["spec_lo_mat"]
        spec_hi     = data["spec_hi_mat"]
        auc_med     = data["auc_med_vec"]
        auc_lo      = data["auc_lo_vec"]
        auc_hi      = data["auc_hi_vec"]
        rel_bins    = data["rel_bins_mat"]
        rel_obs_med = data["rel_obs_med_mat"]
        rel_obs_lo  = data["rel_obs_lo_mat"]
        rel_obs_hi  = data["rel_obs_hi_mat"]

        scenarios = [ScenarioResult(
            pi_h1_vec[i],
            eff_vec[i],
            thr_mat[:, i],
            fdr_med[:, i], fdr_lo[:, i], fdr_hi[:, i],
            sens_med[:, i], sens_lo[:, i], sens_hi[:, i],
            spec_med[:, i], spec_lo[:, i], spec_hi[:, i],
            auc_med[i], auc_lo[i], auc_hi[i],
            rel_bins[:, i], rel_obs_med[:, i], rel_obs_lo[:, i], rel_obs_hi[:, i]
        ) for i in 1:n_sc]

        # Load calibration fields (backward compat: default to nothing)
        cal_model = nothing
        if get(data, "has_cal", false)
            cal_model = CalibrationModel(
                data["cal_a"],
                data["cal_b"],
                data["cal_n_training"],
                get(data, "cal_converged", true)
            )
        end

        fdr_cal_model = nothing
        if get(data, "has_fdr_cal", false)
            fdr_cal_model = FDRCalibrationModel(
                data["fdr_cal_a"],
                data["fdr_cal_b"],
                data["fdr_cal_n_training"],
                get(data, "fdr_cal_converged", true)
            )
        end

        cal_cv = nothing
        if get(data, "has_cv", false)
            cal_cv = CalibrationCVMetrics(
                data["cv_n_folds"],
                data["cv_post_ece_folds"],
                data["cv_post_ece_mean"],
                data["cv_post_ece_std"],
                data["cv_fdr_ece_folds"],
                data["cv_fdr_ece_mean"],
                data["cv_fdr_ece_std"],
                data["cv_raw_rel_bins"],
                data["cv_raw_rel_obs"],
                data["cv_cal_rel_bins"],
                data["cv_cal_rel_obs"],
                data["cv_passes"],
                data["cv_badge_color"]
            )
        end

        # Load FDR curve fields (backward compat: default to empty)
        fdr_curve_emp = get(data, "has_fdr_curves", false) ? data["fdr_curve_emp"] : Float64[]
        fdr_curve_decl = get(data, "has_fdr_curves", false) ? data["fdr_curve_decl"] : Float64[]

        return SimulationResult(
            scenarios,
            data["pi_h1_grid"],
            data["effect_grid"],
            data["n_synthetic"],
            data["n_replicates"],
            Symbol(data["h1_enrichment_family"]),
            (data["fdr_p95_min"], data["fdr_p95_max"]),
            cal_model,
            fdr_cal_model,
            cal_cv,
            fdr_curve_emp,
            fdr_curve_decl
        )
    catch e
        @warn "Failed to load simulation cache" exception=e
        return nothing
    end
end

# ============================================================
# 6b. Calibration Cache I/O (CAL-05: independent invalidation)
# ============================================================

"""
    save_calibration_cache(result::SimulationResult, filepath::String; imputation_method::Symbol = :mnar)

Save calibration data (CalibrationModel, FDRCalibrationModel, CalibrationCVMetrics)
to a separate JLD2 file for independent invalidation from simulation cache.
Uses CALIBRATION_CACHE_VERSION for compatibility checking.

The `imputation_method` kwarg is round-tripped via string-coercion in the
JLD2 payload (matching the existing `timestamp = string(now())` pattern) so MICE
and MNAR calibration caches refuse to alias when both are loaded in one session.
"""
function save_calibration_cache(result::SimulationResult, filepath::String; imputation_method::Symbol = :mnar)
    has_cal = !isnothing(result.calibration_model)
    has_fdr_cal = !isnothing(result.fdr_calibration_model)
    has_cv = !isnothing(result.calibration_cv)

    jldsave(filepath; compress=true,
        cache_version        = CALIBRATION_CACHE_VERSION,
        cache_type           = "calibration",
        timestamp            = string(now()),
        imputation_method    = string(imputation_method),
        has_cal              = has_cal,
        cal_a                = has_cal ? result.calibration_model.a : 1.0,
        cal_b                = has_cal ? result.calibration_model.b : 0.0,
        cal_n_training       = has_cal ? result.calibration_model.n_training : 0,
        cal_converged        = has_cal ? result.calibration_model.converged : false,
        has_fdr_cal          = has_fdr_cal,
        fdr_cal_a            = has_fdr_cal ? result.fdr_calibration_model.a : 1.0,
        fdr_cal_b            = has_fdr_cal ? result.fdr_calibration_model.b : 0.0,
        fdr_cal_n_training   = has_fdr_cal ? result.fdr_calibration_model.n_training : 0,
        fdr_cal_converged    = has_fdr_cal ? result.fdr_calibration_model.converged : false,
        has_cv               = has_cv,
        cv_n_folds           = has_cv ? result.calibration_cv.n_folds : 0,
        cv_post_ece_folds    = has_cv ? result.calibration_cv.posterior_ece_per_fold : Float64[],
        cv_post_ece_mean     = has_cv ? result.calibration_cv.posterior_ece_mean : 0.0,
        cv_post_ece_std      = has_cv ? result.calibration_cv.posterior_ece_std : 0.0,
        cv_fdr_ece_folds     = has_cv ? result.calibration_cv.fdr_ece_per_fold : Float64[],
        cv_fdr_ece_mean      = has_cv ? result.calibration_cv.fdr_ece_mean : 0.0,
        cv_fdr_ece_std       = has_cv ? result.calibration_cv.fdr_ece_std : 0.0,
        cv_raw_rel_bins      = has_cv ? result.calibration_cv.raw_rel_bins : Float64[],
        cv_raw_rel_obs       = has_cv ? result.calibration_cv.raw_rel_observed : Float64[],
        cv_cal_rel_bins      = has_cv ? result.calibration_cv.cal_rel_bins : Float64[],
        cv_cal_rel_obs       = has_cv ? result.calibration_cv.cal_rel_observed : Float64[],
        cv_passes            = has_cv ? result.calibration_cv.passes_ece_threshold : false,
        cv_badge_color       = has_cv ? result.calibration_cv.ece_badge_color : "red",
    )
end

"""
    load_calibration_cache(filepath::String; imputation_method::Symbol = :mnar)::Union{Tuple, Nothing}

Load calibration cache from JLD2. Returns `nothing` if file missing, version mismatch,
type mismatch, or imputation_method mismatch. Returns tuple of
(CalibrationModel, FDRCalibrationModel, CalibrationCVMetrics).

Old caches without `imputation_method` default to `:mar` (RESEARCH Risk #7); pre-Phase-67
caches were MICE-derived.
"""
function load_calibration_cache(filepath::String; imputation_method::Symbol = :mnar)::Union{Tuple, Nothing}
    !isfile(filepath) && return nothing
    try
        data = load(filepath)
        get(data, "cache_version", 0) != CALIBRATION_CACHE_VERSION && return nothing
        get(data, "cache_type", "") != "calibration" && return nothing

        # imputation_method must match
        cached_method = Symbol(get(data, "imputation_method", "mar"))
        cached_method != imputation_method && return nothing

        cal_model = if get(data, "has_cal", false)
            CalibrationModel(data["cal_a"], data["cal_b"], data["cal_n_training"], data["cal_converged"])
        else
            nothing
        end

        fdr_model = if get(data, "has_fdr_cal", false)
            FDRCalibrationModel(data["fdr_cal_a"], data["fdr_cal_b"], data["fdr_cal_n_training"], data["fdr_cal_converged"])
        else
            nothing
        end

        cv_metrics = if get(data, "has_cv", false)
            CalibrationCVMetrics(
                data["cv_n_folds"],
                data["cv_post_ece_folds"], data["cv_post_ece_mean"], data["cv_post_ece_std"],
                data["cv_fdr_ece_folds"], data["cv_fdr_ece_mean"], data["cv_fdr_ece_std"],
                data["cv_raw_rel_bins"], data["cv_raw_rel_obs"],
                data["cv_cal_rel_bins"], data["cv_cal_rel_obs"],
                data["cv_passes"], data["cv_badge_color"]
            )
        else
            nothing
        end

        return (cal_model, fdr_model, cv_metrics)
    catch e
        @warn "Failed to load calibration cache from $filepath" exception=e
        return nothing
    end
end

# ============================================================
# 7. JSON Builder
# ============================================================

"""
    _build_simulation_json(sim::Union{SimulationResult, Nothing})::String

Build a JSON string for report generation. Returns `"null"` if `sim` is nothing.
Pre-summarizes calibration curves to ~50KB — no raw per-protein data included.

Uses `json_object`, `json_array`, `json_number` from `src/reports/json_utils.jl`.
NaN/Inf values are rendered as `null` by `json_number`.
"""
function _build_simulation_json(sim::Union{SimulationResult, Nothing};
                                declared_bfdr::Vector{Float64} = Float64[])::String
    isnothing(sim) && return "null"

    scenario_jsons = String[]
    for sc in sim.scenarios
        push!(scenario_jsons, json_object(
            "pi_h1"              => json_number(sc.pi_h1),
            "effect_scale"       => json_number(sc.effect_scale),
            "thresholds"         => json_array(json_number.(sc.thresholds)),
            "fdr_median"         => json_array(json_number.(sc.fdr_median)),
            "fdr_lo"             => json_array(json_number.(sc.fdr_lo)),
            "fdr_hi"             => json_array(json_number.(sc.fdr_hi)),
            "sensitivity_median" => json_array(json_number.(sc.sensitivity_median)),
            "sensitivity_lo"     => json_array(json_number.(sc.sensitivity_lo)),
            "sensitivity_hi"     => json_array(json_number.(sc.sensitivity_hi)),
            "specificity_median" => json_array(json_number.(sc.specificity_median)),
            "auc_median"         => json_number(sc.auc_median),
            "auc_lo"             => json_number(sc.auc_lo),
            "auc_hi"             => json_number(sc.auc_hi),
            "rel_bins"           => json_array(json_number.(sc.reliability_bin_centers)),
            "rel_median"         => json_array(json_number.(sc.reliability_observed_median)),
            "rel_lo"             => json_array(json_number.(sc.reliability_observed_lo)),
            "rel_hi"             => json_array(json_number.(sc.reliability_observed_hi)),
        ))
    end

    return json_object(
        "scenarios"    => json_array(scenario_jsons),
        "pi_h1_grid"   => json_array(json_number.(sim.pi_h1_grid)),
        "effect_grid"  => json_array(json_number.(sim.effect_grid)),
        "n_synthetic"  => json_number(sim.n_synthetic),
        "n_replicates" => json_number(sim.n_replicates),
        "fdr_p95_min"  => json_number(sim.fdr_at_p95_range[1]),
        "fdr_p95_max"  => json_number(sim.fdr_at_p95_range[2]),
        "calibration"  => _build_calibration_json(sim; declared_bfdr=declared_bfdr),
    )
end

"""
    _build_calibration_json(sim::Union{SimulationResult, Nothing})::String

Build a JSON object with all calibration data for the HTML report.
Returns `"null"` if sim is nothing or calibration_cv is nothing.
"""
function _build_calibration_json(sim::Union{SimulationResult, Nothing};
                                 declared_bfdr::Vector{Float64} = Float64[])::String
    isnothing(sim) && return "null"
    isnothing(sim.calibration_cv) && return "null"

    cv = sim.calibration_cv
    cal = sim.calibration_model
    fdr_cal = sim.fdr_calibration_model

    # Badge color: map string to hex
    badge_color_hex = if cv.ece_badge_color == "green"
        "#198754"
    elseif cv.ece_badge_color == "yellow"
        "#ffc107"
    else
        "#dc3545"
    end

    # Platt calibration curve (generate smooth points for plotting)
    if !isnothing(cal)
        eps_c = 1e-6
        cal_curve_x = collect(range(eps_c, 1.0 - eps_c, length=100))
        cal_curve_y = [clamp(logistic(cal.a * logit(x) + cal.b), eps_c, 1.0 - eps_c) for x in cal_curve_x]
    else
        cal_curve_x = Float64[]
        cal_curve_y = Float64[]
    end
    n_training = isnothing(cal) ? 0 : cal.n_training

    # FDR calibration curve — prefer pre-computed empirical curves from simulation
    if !isempty(sim.fdr_curve_empirical)
        # Self-consistent: both axes from simulation data
        fdr_thresholds = collect(range(0.0, 1.0, length=length(sim.fdr_curve_empirical)))
        fdr_calibrated = sim.fdr_curve_empirical
    elseif !isnothing(fdr_cal)
        # Fallback to logistic model for old SimulationResult without curves
        fdr_thresholds = collect(range(0.0, 1.0, length=100))
        fdr_calibrated = [clamp(logistic(fdr_cal.a * t + fdr_cal.b), 0.0, 1.0) for t in fdr_thresholds]
    else
        fdr_thresholds = Float64[]
        fdr_calibrated = Float64[]
    end

    # Use simulation-derived declared BFDR (NaN becomes null via json_number)
    sim_declared = !isempty(sim.fdr_curve_declared_bfdr) ? sim.fdr_curve_declared_bfdr : declared_bfdr

    return json_object(
        "ece_mean"         => json_number(cv.posterior_ece_mean),
        "ece_std"          => json_number(cv.posterior_ece_std),
        "ece_per_fold"     => json_array(json_number.(cv.posterior_ece_per_fold)),
        "fdr_ece_mean"     => json_number(cv.fdr_ece_mean),
        "fdr_ece_per_fold" => json_array(json_number.(cv.fdr_ece_per_fold)),
        "passes_gate"      => cv.passes_ece_threshold ? "true" : "false",
        "badge_color"      => json_string(badge_color_hex),
        "cal_curve_x"      => json_array(json_number.(cal_curve_x)),
        "cal_curve_y"      => json_array(json_number.(cal_curve_y)),
        "raw_rel_bins"     => json_array(json_number.(cv.raw_rel_bins)),
        "raw_rel_observed" => json_array(json_number.(cv.raw_rel_observed)),
        "cal_rel_bins"     => json_array(json_number.(cv.cal_rel_bins)),
        "cal_rel_observed" => json_array(json_number.(cv.cal_rel_observed)),
        "fdr_thresholds"   => json_array(json_number.(fdr_thresholds)),
        "fdr_calibrated"   => json_array(json_number.(fdr_calibrated)),
        "declared_bfdr"    => json_array(json_number.(sim_declared)),
        "n_training"       => json_number(n_training),
    )
end

# ============================================================
# 7b. Platt Scaling Calibration Functions
# ============================================================

"""
    _fit_posterior_calibration(raw_posteriors::Vector{Float64},
                                ground_truth::BitVector)::CalibrationModel

Fit Platt scaling: calibrated = logistic(a * logit(raw) + b).
Minimizes binary cross-entropy loss using Optim.jl LBFGS.
Clamps raw posteriors to [epsilon, 1-epsilon] before logit to avoid Inf.
"""
function _fit_posterior_calibration(raw_posteriors::Vector{Float64},
                                     ground_truth::BitVector)::CalibrationModel
    n = length(raw_posteriors)
    n == 0 && return CalibrationModel(1.0, 0.0, 0, false)

    eps_clamp = 1e-6
    # Clamp to avoid logit(0) = -Inf, logit(1) = Inf
    x_logit = [logit(clamp(p, eps_clamp, 1.0 - eps_clamp)) for p in raw_posteriors]
    y = Float64.(ground_truth)

    # Binary cross-entropy loss: -sum(y*log(sigma) + (1-y)*log(1-sigma))
    function loss(params)
        a, b = params
        total = 0.0
        for i in 1:n
            z = a * x_logit[i] + b
            # Use log-sum-exp stable form: log(sigma(z)) = -log(1 + exp(-z))
            # log(1 - sigma(z)) = -log(1 + exp(z))
            if z >= 0
                total += y[i] * (-log(1.0 + exp(-z))) + (1.0 - y[i]) * (-z - log(1.0 + exp(-z)))
            else
                total += y[i] * (z - log(1.0 + exp(z))) + (1.0 - y[i]) * (-log(1.0 + exp(z)))
            end
        end
        return -total / n  # minimize negative log-likelihood
    end

    # Initial: a=1 (identity in logit space), b=0
    result = Optim.optimize(loss, [1.0, 0.0], Optim.LBFGS(),
                            Optim.Options(iterations=200, g_tol=1e-8))

    a_fit, b_fit = Optim.minimizer(result)
    return CalibrationModel(a_fit, b_fit, n, Optim.converged(result))
end

"""
    _apply_calibration(raw_post::Float64, model::CalibrationModel;
                       epsilon::Float64=1e-6)::Float64

Apply Platt scaling calibration to a raw posterior probability.
Returns calibrated = logistic(a * logit(raw) + b), clamped to [epsilon, 1-epsilon].
"""
function _apply_calibration(raw_post::Float64, model::CalibrationModel;
                              epsilon::Float64=1e-6)::Float64
    clamped = clamp(raw_post, epsilon, 1.0 - epsilon)
    z = model.a * logit(clamped) + model.b
    calibrated = logistic(z)
    return clamp(calibrated, epsilon, 1.0 - epsilon)
end

"""
    _fit_fdr_calibration(thresholds::Vector{Float64},
                          empirical_fdr::Vector{Float64})::FDRCalibrationModel

Fit Platt scaling for FDR calibration.
FDR = logistic(a * threshold + b). Since FDR decreases as threshold increases,
we expect a < 0. Uses least-squares loss.
"""
function _fit_fdr_calibration(thresholds::Vector{Float64},
                               empirical_fdr::Vector{Float64})::FDRCalibrationModel
    n = length(thresholds)
    n == 0 && return FDRCalibrationModel(1.0, 0.0, 0, false)

    # FDR = logistic(a * threshold + b)
    # Since FDR decreases as threshold increases, we expect a < 0
    function loss(params)
        a, b = params
        total = 0.0
        for i in 1:n
            fdr_pred = logistic(a * thresholds[i] + b)
            total += (fdr_pred - empirical_fdr[i])^2
        end
        return total / n
    end

    # Initial: a=-5 (decreasing), b=0
    result = Optim.optimize(loss, [-5.0, 0.0], Optim.LBFGS(),
                            Optim.Options(iterations=200, g_tol=1e-8))
    a_fit, b_fit = Optim.minimizer(result)
    return FDRCalibrationModel(a_fit, b_fit, n, Optim.converged(result))
end

"""
    _apply_fdr_calibration(threshold::Float64, model::FDRCalibrationModel)::Float64

Apply Platt scaling FDR calibration. Returns logistic(a * threshold + b).
"""
function _apply_fdr_calibration(threshold::Float64, model::FDRCalibrationModel)::Float64
    fdr = logistic(model.a * threshold + model.b)
    return clamp(fdr, 0.0, 1.0)
end

"""
    _stratified_kfold_indices(labels::Vector{Int}, n_folds::Int,
                               rng::Random.AbstractRNG)::Vector{Int}

Stratified k-fold: assign each index to a fold (1:n_folds) preserving class proportions.
"""
function _stratified_kfold_indices(labels::Vector{Int}, n_folds::Int,
                                    rng::Random.AbstractRNG)::Vector{Int}
    n = length(labels)
    fold_assignments = Vector{Int}(undef, n)
    unique_classes = sort(unique(labels))

    for cls in unique_classes
        cls_indices = findall(==(cls), labels)
        # Shuffle within class
        shuffled = cls_indices[randperm(rng, length(cls_indices))]
        # Assign cyclically to folds
        for (j, idx) in enumerate(shuffled)
            fold_assignments[idx] = mod1(j, n_folds)
        end
    end

    return fold_assignments
end

"""
    _compute_calibration_ece(predicted::Vector{Float64}, observed::BitVector;
                              n_bins::Int=15)::Float64

Expected Calibration Error: weighted mean absolute difference between
mean predicted probability and observed fraction per bin.
"""
function _compute_calibration_ece(predicted::Vector{Float64}, observed::BitVector;
                                   n_bins::Int=15)::Float64
    n = length(predicted)
    n == 0 && return 0.0

    bin_edges = range(0.0, 1.0; length=n_bins + 1)
    ece = 0.0

    for j in 1:n_bins
        lo = bin_edges[j]
        hi = bin_edges[j + 1]
        mask = if j < n_bins
            (predicted .>= lo) .& (predicted .< hi)
        else
            predicted .>= lo
        end
        n_bin = sum(mask)
        n_bin == 0 && continue
        mean_pred = mean(predicted[mask])
        mean_obs  = mean(Float64.(observed[mask]))
        ece += (n_bin / n) * abs(mean_pred - mean_obs)
    end

    return ece
end

"""
    _compute_reliability_curve(predicted::Vector{Float64}, observed::BitVector;
                                n_bins::Int=15)

Compute reliability diagram data: (bin_centers, observed_fractions).
Empty bins are skipped.
"""
function _compute_reliability_curve(predicted::Vector{Float64}, observed::BitVector;
                                     n_bins::Int=15)
    bin_edges = range(0.0, 1.0; length=n_bins + 1)
    bin_centers = Float64[]
    obs_fracs   = Float64[]

    for j in 1:n_bins
        lo = bin_edges[j]
        hi = bin_edges[j + 1]
        mask = if j < n_bins
            (predicted .>= lo) .& (predicted .< hi)
        else
            predicted .>= lo
        end
        n_bin = sum(mask)
        n_bin == 0 && continue
        push!(bin_centers, (lo + hi) / 2.0)
        push!(obs_fracs, mean(Float64.(observed[mask])))
    end

    return bin_centers, obs_fracs
end

"""
    _compute_pooled_fdr(posteriors::Vector{Float64}, ground_truth::BitVector,
                         thresholds::Vector{Float64})::Vector{Float64}

Compute empirical FDR at each threshold.
FDR = FP / (FP + TP). Returns 0.0 when no predicted positives.
"""
function _compute_pooled_fdr(posteriors::Vector{Float64}, ground_truth::BitVector,
                              thresholds::Vector{Float64})::Vector{Float64}
    fdr = Vector{Float64}(undef, length(thresholds))
    for (j, t) in enumerate(thresholds)
        predicted_pos = posteriors .>= t
        tp = sum(predicted_pos .& ground_truth)
        fp = sum(predicted_pos .& .!ground_truth)
        denom = tp + fp
        fdr[j] = denom > 0 ? fp / denom : 0.0
    end
    return fdr
end

"""
    _compute_simulation_fdr_curves(posteriors, ground_truth; n_points=100)

Compute self-consistent FDR curves from simulation data:
- `fdr_empirical`: empirical FDR (FP/(FP+TP)) at each threshold (0.0 when no positives)
- `declared_bfdr`: cumulative mean of (1-p_i) for proteins with posterior >= threshold (NaN when empty)

Both axes come from simulation data, avoiding domain mismatch with real data.
"""
function _compute_simulation_fdr_curves(posteriors::Vector{Float64},
                                         ground_truth::BitVector;
                                         n_points::Int=100)
    thresholds = collect(range(0.0, 1.0, length=n_points))
    # Empirical FDR: reuse _compute_pooled_fdr (FP/(FP+TP), 0.0 when no positives)
    fdr_empirical = _compute_pooled_fdr(posteriors, ground_truth, thresholds)
    # Declared BFDR: cumulative mean of (1-p_i) for proteins with posterior >= threshold
    declared_bfdr = map(thresholds) do t
        above = filter(p -> p >= t, posteriors)
        isempty(above) ? NaN : sum(1.0 .- above) / length(above)
    end
    return fdr_empirical, declared_bfdr
end

"""
    _run_calibration_cv(raw_posteriors::Vector{Float64}, ground_truth::BitVector,
                         labels::Vector{Int}; n_folds::Int=10, seed::Int=42)::CalibrationCVMetrics

Run stratified k-fold CV to estimate calibration quality.
"""
function _run_calibration_cv(raw_posteriors::Vector{Float64}, ground_truth::BitVector,
                              labels::Vector{Int};
                              n_folds::Int=10, seed::Int=42)::CalibrationCVMetrics
    n = length(raw_posteriors)
    rng = Random.MersenneTwister(seed)

    fold_assignments = _stratified_kfold_indices(labels, n_folds, rng)

    posterior_ece_per_fold = Vector{Float64}(undef, n_folds)
    fdr_ece_per_fold       = Vector{Float64}(undef, n_folds)

    # For calibrated reliability curve: collect leave-one-fold-out predictions
    cal_predictions = Vector{Float64}(undef, n)
    thresholds_200 = collect(range(0.0, 1.0; length=200))

    for k in 1:n_folds
        train_mask = fold_assignments .!= k
        test_mask  = fold_assignments .== k

        train_post = raw_posteriors[train_mask]
        train_gt   = ground_truth[train_mask]
        test_post  = raw_posteriors[test_mask]
        test_gt    = ground_truth[test_mask]

        # Fit posterior calibration on training fold
        cal_model_k = _fit_posterior_calibration(train_post, train_gt)

        # Apply to test fold
        test_cal = [_apply_calibration(p, cal_model_k) for p in test_post]
        cal_predictions[test_mask] .= test_cal

        # Posterior ECE on held-out data
        posterior_ece_per_fold[k] = _compute_calibration_ece(test_cal, test_gt)

        # FDR ECE: fit FDR calibration on training fold, evaluate on test fold
        if length(train_post) > 0 && sum(train_gt) > 0
            train_fdr = _compute_pooled_fdr(train_post, train_gt, thresholds_200)
            fdr_model_k = _fit_fdr_calibration(thresholds_200, train_fdr)

            # For test fold: compute actual FDR at each threshold
            test_fdr_actual = _compute_pooled_fdr(test_post, test_gt, thresholds_200)
            # Calibrated FDR at each threshold
            test_fdr_cal = [_apply_fdr_calibration(t, fdr_model_k) for t in thresholds_200]
            # ECE on FDR: weighted mean abs diff
            fdr_ece_per_fold[k] = mean(abs.(test_fdr_cal .- test_fdr_actual))
        else
            fdr_ece_per_fold[k] = 0.0
        end
    end

    # Raw reliability curve (full data, before calibration)
    raw_bins, raw_obs = _compute_reliability_curve(raw_posteriors, ground_truth)

    # Calibrated reliability curve (leave-one-fold-out predictions)
    cal_bins, cal_obs = _compute_reliability_curve(cal_predictions, ground_truth)

    ece_mean = mean(posterior_ece_per_fold)
    ece_std  = length(posterior_ece_per_fold) > 1 ? std(posterior_ece_per_fold) : 0.0
    fdr_ece_mean = mean(fdr_ece_per_fold)

    passes = ece_mean < 0.05
    badge_color = if ece_mean < 0.02
        "green"
    elseif ece_mean < 0.05
        "yellow"
    else
        "red"
    end

    return CalibrationCVMetrics(
        n_folds,
        posterior_ece_per_fold,
        ece_mean,
        ece_std,
        fdr_ece_per_fold,
        fdr_ece_mean,
        std(fdr_ece_per_fold),
        raw_bins,
        raw_obs,
        cal_bins,
        cal_obs,
        passes,
        badge_color,
    )
end

# ============================================================
# 8. Main Entry Point
# ============================================================

"""
    run_simulation(lc_result::LatentClassResult;
                   n_synthetic::Int = 10_000,
                   n_replicates::Int = 10,
                   pi_h1_grid = [0.02, 0.05, 0.10, 0.15, 0.20],
                   effect_grid = [0.5, 0.75, 1.0, 1.25, 1.5],
                   n_thresholds::Int = 200,
                   cache_file::String = "")::SimulationResult

Parametric simulation engine for empirical FDR calibration.

Draws synthetic BF triplets from the fitted `LatentClassResult` parameters
and computes FDR/sensitivity/specificity calibration curves, ROC AUC, and
reliability diagram data across a 5x5 grid of (pi_H1, effect_scale) scenarios.

## Arguments
- `lc_result`: Fitted LatentClassResult (requires mixing_weights, class_parameters,
               alpha/theta_enrichment_h1, h1_enrichment_family)
- `n_synthetic`: Number of synthetic proteins per replicate (default 10,000)
- `n_replicates`: Number of replicates per scenario (default 10)
- `pi_h1_grid`: H1 prevalence values to sweep (default [0.02, 0.05, 0.10, 0.15, 0.20])
- `effect_grid`: Effect size scaling factors (default [0.5, 0.75, 1.0, 1.25, 1.5])
- `n_thresholds`: Number of threshold grid points (default 200)
- `cache_file`: JLD2 cache path. If non-empty, attempts load before computing,
                saves result after computing.

## Returns
`SimulationResult` with 25 `ScenarioResult` entries (one per scenario).

## Notes
- Sequential loop (not @threads) for reproducibility and simpler RNG management.
- Each replicate seeded deterministically: `hash(scenario_idx * 10_000 + rep_idx)`.
- Agnostic-drawn proteins count as negatives (conservative metric).
- Effect scaling applies to H1 means only; variances unchanged.
- Progress bar displayed with ETA using ProgressMeter.
"""
function run_simulation(lc_result::LatentClassResult;
                         n_synthetic::Int = 10_000,
                         n_replicates::Int = 10,
                         pi_h1_grid = [0.02, 0.05, 0.10, 0.15, 0.20],
                         effect_grid = [0.5, 0.75, 1.0, 1.25, 1.5],
                         n_thresholds::Int = 200,
                         cache_file::String = "")::SimulationResult

    pi_h1_grid_f  = collect(Float64, pi_h1_grid)
    effect_grid_f = collect(Float64, effect_grid)

    # Check cache
    param_hash = _simulation_param_hash(lc_result, pi_h1_grid_f, effect_grid_f,
                                         n_synthetic, n_replicates)
    if !isempty(cache_file)
        cached = load_simulation_cache(cache_file, param_hash)
        if !isnothing(cached)
            @info "Simulation cache hit — loaded from $cache_file"
            return cached
        end
    end

    # Grid setup
    n_pi_h1   = length(pi_h1_grid_f)
    n_effect  = length(effect_grid_f)
    n_scenarios = n_pi_h1 * n_effect

    # Original mixing weights
    pi_orig = lc_result.mixing_weights
    pi_h0_orig  = pi_orig[1]
    pi_ag_orig  = length(pi_orig) >= 3 ? pi_orig[2] : 0.0
    pi_h1_orig  = length(pi_orig) >= 3 ? pi_orig[3] : pi_orig[2]

    # Threshold grid
    thresholds = collect(range(0.0, 1.0; length=n_thresholds))

    # Reliability bins: [0, 0.1, 0.2, ..., 0.9, 0.95, 1.0]
    reliability_bins = vcat(collect(0.0:0.1:0.9), [0.95, 1.0])
    n_rel_bins = length(reliability_bins) - 1

    # Progress bar
    prog = Progress(n_scenarios * n_replicates;
                    desc="Simulation: ", showspeed=true, color=:cyan)

    # Results storage: per scenario, per replicate
    scenario_results = Vector{ScenarioResult}(undef, n_scenarios)
    scenario_idx = 0

    # Accumulate pooled posteriors + ground truth for calibration fitting
    pooled_capacity = n_scenarios * n_replicates * n_synthetic
    all_raw_posteriors = Float64[]; sizehint!(all_raw_posteriors, pooled_capacity)
    all_ground_truth_v = Bool[];    sizehint!(all_ground_truth_v, pooled_capacity)
    all_labels_int     = Int[];     sizehint!(all_labels_int,     pooled_capacity)

    # Single-scenario calibration vectors (avoid distribution mismatch)
    # Train calibration on the scenario closest to real data (effect_scale=1.0, pi_h1 = EM estimate)
    cal_raw_posteriors = Float64[]; sizehint!(cal_raw_posteriors, n_replicates * n_synthetic)
    cal_ground_truth_v = Bool[];    sizehint!(cal_ground_truth_v, n_replicates * n_synthetic)
    cal_labels_int     = Int[];     sizehint!(cal_labels_int,     n_replicates * n_synthetic)
    best_pi_idx   = argmin(abs.(collect(pi_h1_grid_f) .- pi_h1_orig))
    best_eff_idx  = argmin(abs.(collect(effect_grid_f) .- 1.0))
    best_scenario = (best_pi_idx - 1) * length(effect_grid_f) + best_eff_idx

    for (pi_idx, pi_h1_target) in enumerate(pi_h1_grid_f)
        for (eff_idx, effect_scale) in enumerate(effect_grid_f)
            scenario_idx += 1

            # Renormalize H0/Agnostic to accommodate pi_h1_target
            orig_h0_ag = pi_h0_orig + pi_ag_orig
            scale_factor = orig_h0_ag > 1e-10 ? (1.0 - pi_h1_target) / orig_h0_ag : 0.0
            pi_h0_new = pi_h0_orig * scale_factor
            pi_ag_new = pi_ag_orig * scale_factor
            # Ensure weights sum to 1
            pi_weights = length(pi_orig) >= 3 ?
                [pi_h0_new, pi_ag_new, pi_h1_target] :
                [pi_h0_new + pi_ag_new, pi_h1_target]
            pi_weights ./= sum(pi_weights)  # normalize for numerical safety

            cat_dist = Categorical(pi_weights)

            # Per-replicate arrays
            fdr_reps         = Matrix{Float64}(undef, n_thresholds, n_replicates)
            sens_reps        = Matrix{Float64}(undef, n_thresholds, n_replicates)
            spec_reps        = Matrix{Float64}(undef, n_thresholds, n_replicates)
            auc_reps         = Vector{Float64}(undef, n_replicates)
            rel_obs_reps     = Matrix{Float64}(undef, n_rel_bins, n_replicates)

            for rep in 1:n_replicates
                # Deterministic seed per (scenario, replicate)
                seed = hash(scenario_idx * 100_000 + rep)
                rng = Random.MersenneTwister(seed % typemax(UInt32))

                # Draw component labels
                labels = rand(rng, cat_dist, n_synthetic)
                # Ground truth: H1-drawn proteins are true positives
                if length(pi_orig) >= 3
                    ground_truth = BitVector(labels .== 3)
                else
                    ground_truth = BitVector(labels .== 2)
                end

                # Draw log-BF triplets
                posteriors = Vector{Float64}(undef, n_synthetic)
                for i in 1:n_synthetic
                    lbf_e, lbf_c, lbf_d = _draw_synthetic_triplet(labels[i], lc_result, effect_scale)
                    posteriors[i] = _synthetic_posterior(lbf_e, lbf_c, lbf_d, lc_result)
                end

                # Accumulate for report (all scenarios x replicates pooled)
                append!(all_raw_posteriors, posteriors)
                append!(all_ground_truth_v, Vector{Bool}(ground_truth))
                append!(all_labels_int,     labels)

                # Accumulate for calibration ONLY from representative scenario
                if scenario_idx == best_scenario
                    append!(cal_raw_posteriors, posteriors)
                    append!(cal_ground_truth_v, Vector{Bool}(ground_truth))
                    append!(cal_labels_int,     labels)
                end

                # Compute calibration curves
                fdr_rep, sens_rep, spec_rep = _compute_calibration_curves(
                    posteriors, ground_truth, thresholds)
                fdr_reps[:, rep]  = fdr_rep
                sens_reps[:, rep] = sens_rep
                spec_reps[:, rep] = spec_rep

                # Compute AUC
                auc_reps[rep] = _compute_auc(posteriors, ground_truth)

                # Compute reliability diagram
                bin_centers, obs_fracs = _compute_reliability(posteriors, ground_truth,
                                                               reliability_bins)
                rel_obs_reps[:, rep] = obs_fracs

                ProgressMeter.next!(prog)
            end

            # Aggregate across replicates: median, 2.5th, 97.5th percentile
            _agg(mat, q) = [quantile(filter(isfinite, mat[j, :]), q)
                            for j in 1:size(mat, 1)]

            fdr_med  = _agg(fdr_reps, 0.5)
            fdr_lo   = _agg(fdr_reps, 0.025)
            fdr_hi   = _agg(fdr_reps, 0.975)
            sens_med = _agg(sens_reps, 0.5)
            sens_lo  = _agg(sens_reps, 0.025)
            sens_hi  = _agg(sens_reps, 0.975)
            spec_med = _agg(spec_reps, 0.5)
            spec_lo  = _agg(spec_reps, 0.025)
            spec_hi  = _agg(spec_reps, 0.975)

            # AUC scalar aggregation
            finite_aucs = filter(isfinite, auc_reps)
            auc_med_val = isempty(finite_aucs) ? 0.5 : quantile(finite_aucs, 0.5)
            auc_lo_val  = isempty(finite_aucs) ? 0.5 : quantile(finite_aucs, 0.025)
            auc_hi_val  = isempty(finite_aucs) ? 0.5 : quantile(finite_aucs, 0.975)

            # Reliability aggregation
            rel_med = Vector{Float64}(undef, n_rel_bins)
            rel_lo  = Vector{Float64}(undef, n_rel_bins)
            rel_hi  = Vector{Float64}(undef, n_rel_bins)
            reliability_bin_centers_vec = Vector{Float64}(undef, n_rel_bins)
            for j in 1:n_rel_bins
                reliability_bin_centers_vec[j] = (reliability_bins[j] + reliability_bins[j+1]) / 2.0
                finite_obs = filter(isfinite, rel_obs_reps[j, :])
                if isempty(finite_obs)
                    rel_med[j] = NaN
                    rel_lo[j]  = NaN
                    rel_hi[j]  = NaN
                else
                    rel_med[j] = quantile(finite_obs, 0.5)
                    rel_lo[j]  = quantile(finite_obs, 0.025)
                    rel_hi[j]  = quantile(finite_obs, 0.975)
                end
            end

            scenario_results[scenario_idx] = ScenarioResult(
                pi_h1_target,
                effect_scale,
                thresholds,
                fdr_med, fdr_lo, fdr_hi,
                sens_med, sens_lo, sens_hi,
                spec_med, spec_lo, spec_hi,
                auc_med_val, auc_lo_val, auc_hi_val,
                reliability_bin_centers_vec, rel_med, rel_lo, rel_hi
            )
        end
    end

    ProgressMeter.finish!(prog)

    # Compute fdr_at_p95 summary: FDR at threshold closest to 0.95 across all scenarios
    p95_thresh_idx = argmin(abs.(thresholds .- 0.95))
    fdr_at_p95_vals = [sc.fdr_median[p95_thresh_idx] for sc in scenario_results]
    finite_fdr_p95 = filter(isfinite, fdr_at_p95_vals)
    fdr_p95_min = isempty(finite_fdr_p95) ? 0.0 : minimum(finite_fdr_p95)
    fdr_p95_max = isempty(finite_fdr_p95) ? 1.0 : maximum(finite_fdr_p95)

    # Fit isotonic calibration on pooled synthetic data
    cal_model     = nothing
    fdr_cal_model = nothing
    cal_cv        = nothing
    try
        # Fit calibration on single representative scenario (not pooled)
        cal_gt_bv = BitVector(cal_ground_truth_v)
        @info "Fitting calibration on $(length(cal_raw_posteriors)) single-scenario synthetic data points (scenario $(best_scenario): pi_h1=$(pi_h1_grid_f[best_pi_idx]), effect=$(effect_grid_f[best_eff_idx]))..."

        cal_cv = _run_calibration_cv(cal_raw_posteriors, cal_gt_bv, cal_labels_int)
        cal_model = _fit_posterior_calibration(cal_raw_posteriors, cal_gt_bv)

        pooled_thresholds = collect(range(0.0, 1.0; length=200))
        pooled_fdr_vals   = _compute_pooled_fdr(cal_raw_posteriors, cal_gt_bv, pooled_thresholds)
        fdr_cal_model = _fit_fdr_calibration(pooled_thresholds, pooled_fdr_vals)

        @info "Calibration complete: ECE=$(round(cal_cv.posterior_ece_mean, digits=4)), badge=$(cal_cv.ece_badge_color)"
    catch e
        @warn "Calibration fitting failed — continuing without calibration" exception=e
    end

    # Compute self-consistent FDR curves from simulation data
    fdr_emp, fdr_decl = Float64[], Float64[]
    if !isempty(cal_raw_posteriors)
        try
            cal_gt_bv_fdr = BitVector(cal_ground_truth_v)
            fdr_emp, fdr_decl = _compute_simulation_fdr_curves(cal_raw_posteriors, cal_gt_bv_fdr)
        catch e
            @warn "FDR curve computation failed — continuing without pre-computed curves" exception=e
        end
    end

    result = SimulationResult(
        scenario_results,
        pi_h1_grid_f,
        effect_grid_f,
        n_synthetic,
        n_replicates,
        lc_result.h1_enrichment_family,
        (fdr_p95_min, fdr_p95_max),
        cal_model,
        fdr_cal_model,
        cal_cv,
        fdr_emp,
        fdr_decl,
    )

    # Save cache
    if !isempty(cache_file)
        try
            mkpath(dirname(isempty(dirname(cache_file)) ? "." : dirname(cache_file)))
            save_simulation_cache(result, param_hash, cache_file)
            @info "Simulation cache saved to $cache_file"
        catch e
            @warn "Failed to save simulation cache" exception=e
        end
    end

    return result
end
