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
    @test ismissing(direction)
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
