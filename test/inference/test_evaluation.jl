"""
    test_evaluation.jl

Tests for Bayes factor calculation and evaluation functions.
"""

@testitem "calculate_bayes_factor with equal posterior and prior" begin
    using BayesInteractomics
    using BayesInteractomics: calculate_bayes_factor
    using Distributions

    # When posterior equals prior, BF should be approximately 1
    prior = [Normal(0.0, 1.0), Normal(0.0, 1.0)]
    posterior = [Normal(0.0, 1.0), Normal(0.0, 1.0)]

    bf, p_post, p_prior = calculate_bayes_factor(posterior, prior; threshold=0.0)

    @test all(isapprox.(bf, 1.0, atol=0.01))
    @test p_post ≈ p_prior
end

@testitem "calculate_bayes_factor with shifted posterior (H1 support)" begin
    using BayesInteractomics
    using BayesInteractomics: calculate_bayes_factor
    using Distributions

    # Posterior shifted right -> higher probability above threshold -> BF > 1
    prior = [Normal(0.0, 1.0)]
    posterior = [Normal(2.0, 1.0)]  # Mean shifted to 2.0

    bf, p_post, p_prior = calculate_bayes_factor(posterior, prior; threshold=0.0)

    @test bf[1] > 1.0  # Evidence for H1
    @test p_post[1] > p_prior[1]
end

@testitem "calculate_bayes_factor with shifted posterior (H0 support)" begin
    using BayesInteractomics
    using BayesInteractomics: calculate_bayes_factor
    using Distributions

    # Posterior shifted left -> lower probability above threshold -> BF < 1
    prior = [Normal(0.0, 1.0)]
    posterior = [Normal(-2.0, 1.0)]  # Mean shifted to -2.0

    bf, p_post, p_prior = calculate_bayes_factor(posterior, prior; threshold=0.0)

    @test bf[1] < 1.0  # Evidence for H0
    @test p_post[1] < p_prior[1]
end

@testitem "calculate_bayes_factor with custom threshold" begin
    using BayesInteractomics
    using BayesInteractomics: calculate_bayes_factor
    using Distributions

    prior = [Normal(0.0, 1.0)]
    posterior = [Normal(1.0, 1.0)]

    # With threshold=0, posterior has higher prob above threshold
    bf_0, _, _ = calculate_bayes_factor(posterior, prior; threshold=0.0)

    # With higher threshold, both have lower prob but relationship preserved
    bf_1, _, _ = calculate_bayes_factor(posterior, prior; threshold=1.0)

    @test bf_0[1] > 1.0
    # Both should show evidence for H1, but magnitude differs with threshold
end

@testitem "calculate_bayes_factor — extreme posteriors capped, no overflow" begin
    using BayesInteractomics
    using BayesInteractomics: calculate_bayes_factor
    using Distributions

    # Very tight posterior far from 0: previously overflowed to 2^52 ≈ 4.5e15
    prior     = [Normal(0.0, 1.0)]
    posterior = [Normal(10.0, 0.01)]   # P(slope > 0) ≈ 1.0 exactly in Float64

    bf, p_post, p_prior = calculate_bayes_factor(posterior, prior; threshold=0.0)

    # BF should be capped at max_bf (default 1e6), not overflow to 4.5e15
    @test bf[1] ≈ 1e6
    @test isfinite(bf[1])

    # Custom cap
    bf2, _, _ = calculate_bayes_factor(posterior, prior; threshold=0.0, max_bf=1e4)
    @test bf2[1] ≈ 1e4

    # Very tight posterior on the negative side — BF should be tiny but not zero
    posterior_neg = [Normal(-10.0, 0.01)]
    bf_neg, _, _ = calculate_bayes_factor(posterior_neg, prior; threshold=0.0)
    @test bf_neg[1] ≈ 1e-6   # 1/max_bf
    @test bf_neg[1] > 0.0
end

@testitem "probability_of_direction with positive draws" begin
    using BayesInteractomics
    using BayesInteractomics: probability_of_direction

    # Mostly positive draws
    draws = [0.5, 1.2, 0.8, -0.1, 0.9, 1.5, 0.3, 2.0]

    pd, direction = probability_of_direction(draws)

    @test pd > 0.5
    @test direction == "+"
end

@testitem "probability_of_direction with negative draws" begin
    using BayesInteractomics
    using BayesInteractomics: probability_of_direction

    # Mostly negative draws
    draws = [-0.5, -1.2, -0.8, 0.1, -0.9, -1.5, -0.3, -2.0]

    pd, direction = probability_of_direction(draws)

    @test pd > 0.5
    @test direction == "-"
end

@testitem "probability_of_direction with balanced draws" begin
    using BayesInteractomics
    using BayesInteractomics: probability_of_direction

    # Exactly balanced (equal positive and negative)
    draws = [1.0, -1.0, 2.0, -2.0]

    pd, direction = probability_of_direction(draws)

    @test pd == 0.5
    @test direction == "~"
end

@testitem "probability_of_direction with Normal distribution" begin
    using BayesInteractomics
    using BayesInteractomics: probability_of_direction
    using Distributions

    # Normal centered at positive value
    dist = Normal(2.0, 1.0)
    pd, direction = probability_of_direction(dist)

    @test pd > 0.5
    @test direction == "+"

    # Normal centered at negative value
    dist_neg = Normal(-2.0, 1.0)
    pd_neg, direction_neg = probability_of_direction(dist_neg)

    @test pd_neg > 0.5
    @test direction_neg == "-"
end

@testitem "probability_of_direction with vector of draws" begin
    using BayesInteractomics
    using BayesInteractomics: probability_of_direction

    draws_list = [
        [1.0, 2.0, 0.5, 0.8],      # Positive
        [-1.0, -2.0, -0.5, -0.8],   # Negative
    ]

    pd, direction = probability_of_direction(draws_list)

    @test length(pd) == 2
    @test pd[1] > 0.5
    @test pd[2] > 0.5
    @test direction[1] == "+"
    @test direction[2] == "-"
end

@testitem "pd_to_p_value two-sided" begin
    using BayesInteractomics
    using BayesInteractomics: pd_to_p_value

    # pd = 0.975 -> p = 2*(1-0.975) = 0.05
    @test pd_to_p_value(0.975, true) ≈ 0.05

    # pd = 0.95 -> p = 2*(1-0.95) = 0.10
    @test pd_to_p_value(0.95, true) ≈ 0.10

    # pd = 0.5 -> p = 2*(1-0.5) = 1.0
    @test pd_to_p_value(0.5, true) ≈ 1.0

    # pd = 1.0 -> p = 2*(1-1.0) = 0.0
    @test pd_to_p_value(1.0, true) ≈ 0.0
end

@testitem "pd_to_p_value one-sided" begin
    using BayesInteractomics
    using BayesInteractomics: pd_to_p_value

    # pd = 0.975 -> p = 1-0.975 = 0.025
    @test pd_to_p_value(0.975, false) ≈ 0.025

    # pd = 0.95 -> p = 1-0.95 = 0.05
    @test pd_to_p_value(0.95, false) ≈ 0.05
end

@testitem "pd_to_p_value invalid input" begin
    using BayesInteractomics
    using BayesInteractomics: pd_to_p_value

    # pd must be in [0.5, 1.0]
    @test_throws AssertionError pd_to_p_value(0.3, true)
    @test_throws AssertionError pd_to_p_value(1.1, true)
end

@testitem "log2FCStatistics with Normal distributions" begin
    using BayesInteractomics
    using BayesInteractomics: log2FCStatistics
    using Distributions

    # Create mock log2FC distributions
    log2fc = [Normal(2.0, 0.5), Normal(-1.0, 0.3)]

    stats = log2FCStatistics(log2fc)

    @test haskey(stats, :mean_log2FC)
    @test haskey(stats, :sd_log2FC)
    @test haskey(stats, :pd)
    @test haskey(stats, :pd_direction)

    # Check mean values
    @test stats[:mean_log2FC][1] ≈ 2.0 atol=0.1
    @test stats[:mean_log2FC][2] ≈ -1.0 atol=0.1

    # Check standard deviations
    @test stats[:sd_log2FC][1] ≈ 0.5 atol=0.1
    @test stats[:sd_log2FC][2] ≈ 0.3 atol=0.1

    # Check directions
    @test stats[:pd_direction][1] == "+"
    @test stats[:pd_direction][2] == "-"
end

# =====================================================================
# BayesFactorRegression tests
# =====================================================================

@testitem "BayesFactorRegression single-protocol — threshold=0.0" begin
    using BayesInteractomics
    using BayesInteractomics: RegressionResultSingleProtocol, calculate_bayes_factor
    using Distributions
    using RxInfer

    # --- positive posterior mean → BF > 1 ---
    post_ir = RxInfer.InferenceResult(
        Dict(:α => Normal(0.3, 0.1), :β => Normal(0.0, 1.0), :σ => Gamma(2.0, 0.5)),
        nothing, nothing, nothing, nothing
    )
    prior_ir = RxInfer.InferenceResult(
        Dict(:α => Normal(0.0, 0.1), :β => Normal(0.0, 1.0), :σ => Gamma(2.0, 0.5)),
        nothing, nothing, nothing, nothing
    )
    result = RegressionResultSingleProtocol(post_ir, prior_ir)
    bf, p_post, p_prior = BayesFactorRegression(result; threshold=0.0)

    @test bf[1] > 1.0  # positive slope → evidence for H1
    @test p_post[1] > p_prior[1]

    # --- negative posterior mean → BF < 1 ---
    post_neg = RxInfer.InferenceResult(
        Dict(:α => Normal(-0.3, 0.1), :β => Normal(0.0, 1.0), :σ => Gamma(2.0, 0.5)),
        nothing, nothing, nothing, nothing
    )
    result_neg = RegressionResultSingleProtocol(post_neg, prior_ir)
    bf_neg, _, _ = BayesFactorRegression(result_neg; threshold=0.0)

    @test bf_neg[1] < 1.0  # negative slope → evidence for H0

    # --- symmetric prior with threshold=0.0 → prior_odds ≈ 1.0 ---
    # Prior N(0,σ²) is symmetric around 0 → P(x>0) = 0.5 → prior_odds = 1.0
    @test isapprox(p_prior[1] / (1 - p_prior[1]), 1.0, atol=0.01)

    # --- identical prior and posterior → BF ≈ 1 ---
    result_equal = RegressionResultSingleProtocol(prior_ir, prior_ir)
    bf_eq, _, _ = BayesFactorRegression(result_equal; threshold=0.0)

    @test isapprox(bf_eq[1], 1.0, atol=0.01)
end

@testitem "BayesFactorRegression multi-protocol — μ_α fix verification" begin
    using BayesInteractomics
    using BayesInteractomics: RegressionResultMultipleProtocols, calculate_bayes_factor
    using Distributions
    using RxInfer

    # Create mock multi-protocol result with :α (vector) and :μ_α (single dist)
    # Set μ_α to a DIFFERENT distribution than any α[i] to verify the global BF
    # comes from μ_α, not from α
    post_ir = RxInfer.InferenceResult(
        Dict(
            :α => [Normal(0.2, 0.1), Normal(0.4, 0.1)],   # per-protocol slopes
            :μ_α => Normal(0.8, 0.05),                      # global slope (intentionally different)
            :β => Normal(0.0, 1.0),
            :σ => Gamma(2.0, 0.5)
        ),
        nothing, nothing, nothing, nothing
    )
    prior_ir = RxInfer.InferenceResult(
        Dict(
            :α => [Normal(0.0, 0.1), Normal(0.0, 0.1)],
            :μ_α => Normal(0.0, 0.1),
            :β => Normal(0.0, 1.0),
            :σ => Gamma(2.0, 0.5)
        ),
        nothing, nothing, nothing, nothing
    )

    result = RegressionResultMultipleProtocols(post_ir, prior_ir)
    bf, p_post, p_prior = BayesFactorRegression(result; threshold=0.0)

    # bf[1] is global slope (from μ_α), bf[2:] are per-protocol (from α[i])
    @test length(bf) == 3  # 1 global + 2 per-protocol

    # Global BF should differ from per-protocol BFs
    @test !isapprox(bf[1], bf[2], atol=0.1)
    @test !isapprox(bf[1], bf[3], atol=0.1)

    # bf[1] (global slope) should match what we'd get from calculate_bayes_factor on μ_α
    bf_global_expected, _, _ = calculate_bayes_factor(
        [post_ir.posteriors[:μ_α]], [prior_ir.posteriors[:μ_α]]; threshold=0.0
    )
    @test isapprox(bf[1], bf_global_expected[1], rtol=1e-10)

    # All BFs should be > 1 (all posteriors shifted positive)
    @test all(bf .> 1.0)
end

@testitem "BayesFactorRegression threshold effect" begin
    using BayesInteractomics
    using BayesInteractomics: RegressionResultSingleProtocol
    using Distributions
    using RxInfer

    # Prior with σ=0.153 ≈ sqrt(0.0234), matching the real regression prior width
    σ_prior = sqrt(0.0234)
    post_ir = RxInfer.InferenceResult(
        Dict(:α => Normal(0.15, 0.05), :β => Normal(0.0, 1.0), :σ => Gamma(2.0, 0.5)),
        nothing, nothing, nothing, nothing
    )
    prior_ir = RxInfer.InferenceResult(
        Dict(:α => Normal(0.0, σ_prior), :β => Normal(0.0, 1.0), :σ => Gamma(2.0, 0.5)),
        nothing, nothing, nothing, nothing
    )

    result = RegressionResultSingleProtocol(post_ir, prior_ir)

    # threshold=0.0: standard comparison
    bf_t0, _, _ = BayesFactorRegression(result; threshold=0.0)

    # threshold=0.3: requires slope > 0.3 for H1
    # With posterior mean=0.15 (below 0.3) and small σ, BF should be much smaller
    bf_t03, _, _ = BayesFactorRegression(result; threshold=0.3)

    @test bf_t0[1] > bf_t03[1]  # threshold=0.3 penalizes weak slopes

    # The threshold=0.3 BF should be dramatically smaller (the documented 40x handicap)
    # With prior N(0, 0.153) and posterior N(0.15, 0.05):
    # P(prior > 0.3) is tiny; P(posterior > 0.3) is also small for mean=0.15
    # This creates a huge penalty compared to threshold=0.0
    @test bf_t0[1] / bf_t03[1] > 5.0  # At least 5x reduction
end

@testitem "BayesFactorRegression robust dispatch" begin
    using BayesInteractomics
    using BayesInteractomics: RegressionResultSingleProtocol, RegressionResultMultipleProtocols
    using Distributions
    using RxInfer

    # Single protocol: robust should delegate to non-robust
    post_ir = RxInfer.InferenceResult(
        Dict(:α => Normal(0.3, 0.1), :β => Normal(0.0, 1.0), :σ => Gamma(2.0, 0.5)),
        nothing, nothing, nothing, nothing
    )
    prior_ir = RxInfer.InferenceResult(
        Dict(:α => Normal(0.0, 0.1), :β => Normal(0.0, 1.0), :σ => Gamma(2.0, 0.5)),
        nothing, nothing, nothing, nothing
    )

    normal_result = RegressionResultSingleProtocol(post_ir, prior_ir)
    robust_result = RobustRegressionResultSingleProtocol(post_ir, prior_ir, 5.0, 1.0)

    bf_normal, _, _ = BayesFactorRegression(normal_result; threshold=0.0)
    bf_robust, _, _ = BayesFactorRegression(robust_result; threshold=0.0)

    @test isapprox(bf_normal[1], bf_robust[1], rtol=1e-10)

    # Multi protocol: robust should delegate to non-robust
    post_multi = RxInfer.InferenceResult(
        Dict(
            :α => [Normal(0.2, 0.1), Normal(0.3, 0.1)],
            :μ_α => Normal(0.25, 0.05),
            :β => Normal(0.0, 1.0),
            :σ => Gamma(2.0, 0.5)
        ),
        nothing, nothing, nothing, nothing
    )
    prior_multi = RxInfer.InferenceResult(
        Dict(
            :α => [Normal(0.0, 0.1), Normal(0.0, 0.1)],
            :μ_α => Normal(0.0, 0.1),
            :β => Normal(0.0, 1.0),
            :σ => Gamma(2.0, 0.5)
        ),
        nothing, nothing, nothing, nothing
    )

    normal_multi = RegressionResultMultipleProtocols(post_multi, prior_multi)
    robust_multi = RobustRegressionResultMultipleProtocols(post_multi, prior_multi, 5.0, 1.0)

    bf_nm, _, _ = BayesFactorRegression(normal_multi; threshold=0.0)
    bf_rm, _, _ = BayesFactorRegression(robust_multi; threshold=0.0)

    @test length(bf_nm) == length(bf_rm)
    @test all(isapprox.(bf_nm, bf_rm, rtol=1e-10))
end

@testitem "BayesFactorRegression CONFIG threading — regression_bf_threshold field" begin
    using BayesInteractomics

    # Verify the regression_bf_threshold field exists on CONFIG with default 0.0
    config = CONFIG(
        datafile = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2])],
        sample_cols = [Dict(1 => [3,4])],
        poi = "test"
    )
    @test config.regression_bf_threshold == 0.1

    # Verify it can be changed
    config.regression_bf_threshold = 0.3
    @test config.regression_bf_threshold == 0.3

    # Verify other regression-related fields exist with expected defaults
    @test config.regression_likelihood == :robust_t
    @test config.student_t_nu == 5.0
end

# =====================================================================
# Regression BF gradation with posterior variance floor
# =====================================================================

@testitem "Regression BF gradation — min_posterior_var backward compatibility" begin
    using BayesInteractomics
    using BayesInteractomics: calculate_bayes_factor
    using Distributions

    # With min_posterior_var=0.0 (default), behavior is identical to original
    prior = [Normal(0.0, 1.0), Normal(0.0, 0.5)]
    posterior = [Normal(1.5, 0.3), Normal(-0.8, 0.2)]

    bf_default, p_post_default, p_prior_default = calculate_bayes_factor(posterior, prior; threshold=0.0)
    bf_zero, p_post_zero, p_prior_zero = calculate_bayes_factor(posterior, prior; threshold=0.0, min_posterior_var=0.0)

    @test bf_default == bf_zero
    @test p_post_default == p_post_zero
    @test p_prior_default == p_prior_zero
end

@testitem "Regression BF gradation — narrow posterior regularized by variance floor" begin
    using BayesInteractomics
    using BayesInteractomics: calculate_bayes_factor
    using Distributions

    # Very narrow posterior (std=1e-6): without floor, BF saturates at max_bf
    prior = [Normal(0.0, 1.0)]
    narrow_posterior = [Normal(0.5, 1e-6)]

    bf_no_floor, _, _ = calculate_bayes_factor(narrow_posterior, prior; threshold=0.0, min_posterior_var=0.0)
    @test isapprox(bf_no_floor[1], 1e6, rtol=1e-6)  # saturated at max_bf

    # With min_posterior_var=1.0 (min std=1.0), BF should NOT saturate
    # (posterior becomes Normal(0.5, 1.0), same width as prior)
    bf_with_floor, _, _ = calculate_bayes_factor(narrow_posterior, prior; threshold=0.0, min_posterior_var=1.0)
    @test bf_with_floor[1] < 1e6  # no longer at clamp boundary
    @test bf_with_floor[1] > 1.0  # still evidence for H1 (mean > threshold)
    # BF should show meaningful gradation (well below saturation)
    @test bf_with_floor[1] < 1e4
end

@testitem "Regression BF gradation — wide posterior unaffected by variance floor" begin
    using BayesInteractomics
    using BayesInteractomics: calculate_bayes_factor
    using Distributions

    # Posterior already has std=1.0, which is >> sqrt(0.01)=0.1
    prior = [Normal(0.0, 1.0)]
    wide_posterior = [Normal(1.0, 1.0)]

    bf_no_floor, _, _ = calculate_bayes_factor(wide_posterior, prior; threshold=0.0, min_posterior_var=0.0)
    bf_with_floor, _, _ = calculate_bayes_factor(wide_posterior, prior; threshold=0.0, min_posterior_var=0.01)

    # Wide posterior is unaffected since std(1.0) > sqrt(0.01)=0.1
    @test isapprox(bf_no_floor[1], bf_with_floor[1], rtol=1e-10)
end

@testitem "Regression BF gradation — posterior at threshold gives BF near 1" begin
    using BayesInteractomics
    using BayesInteractomics: calculate_bayes_factor
    using Distributions

    # Posterior centered at threshold=0.0 → P(x > 0) ≈ 0.5 → BF ≈ 1 regardless of variance
    prior = [Normal(0.0, 1.0)]
    posterior_at_zero = [Normal(0.0, 1e-6)]

    # Without floor: P(x>0) = 0.5 exactly → BF = 1
    bf_no_floor, _, _ = calculate_bayes_factor(posterior_at_zero, prior; threshold=0.0, min_posterior_var=0.0)
    @test isapprox(bf_no_floor[1], 1.0, rtol=0.01)

    # With floor: still BF ≈ 1 since mean is at threshold
    bf_with_floor, _, _ = calculate_bayes_factor(posterior_at_zero, prior; threshold=0.0, min_posterior_var=0.01)
    @test isapprox(bf_with_floor[1], 1.0, rtol=0.01)
end

@testitem "Regression BF gradation — CONFIG has regression_min_posterior_var field" begin
    using BayesInteractomics

    config = CONFIG(
        datafile = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2])],
        sample_cols = [Dict(1 => [3,4])],
        poi = "test"
    )
    # Default value is 0.01
    @test config.regression_min_posterior_var == 0.01

    # Can be changed
    config.regression_min_posterior_var = 0.05
    @test config.regression_min_posterior_var == 0.05

    # Can be disabled
    config.regression_min_posterior_var = 0.0
    @test config.regression_min_posterior_var == 0.0
end

@testitem "Regression BF gradation — multiple posteriors with different widths" begin
    using BayesInteractomics
    using BayesInteractomics: calculate_bayes_factor
    using Distributions

    prior = [Normal(0.0, 1.0), Normal(0.0, 1.0), Normal(0.0, 1.0)]
    # Mix of narrow, moderate, and wide posteriors — all positive mean
    posteriors = [Normal(0.5, 1e-8), Normal(0.5, 0.05), Normal(0.5, 1.0)]

    bf_no_floor, _, _ = calculate_bayes_factor(posteriors, prior; threshold=0.0, min_posterior_var=0.0)
    bf_with_floor, _, _ = calculate_bayes_factor(posteriors, prior; threshold=0.0, min_posterior_var=0.01)

    # Without floor: narrow saturates
    @test isapprox(bf_no_floor[1], 1e6, rtol=1e-6)  # narrow saturates
    # With floor (min_std=0.1): narrow no longer saturates; wide unchanged
    @test bf_with_floor[1] < 1e6  # regularized
    @test isapprox(bf_no_floor[3], bf_with_floor[3], rtol=1e-10)  # wide unaffected (std=1.0 >> 0.1)
    # Narrow (floored to std=0.1) should differ from wide (std=1.0)
    @test bf_with_floor[1] != bf_with_floor[3]
end
