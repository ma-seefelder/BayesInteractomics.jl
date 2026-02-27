"""
    test_utils.jl

Tests for utility functions used throughout BayesInteractomics.
"""

@testitem "log2FC computation between distributions" begin
    using BayesInteractomics
    using BayesInteractomics: log2FC
    using Distributions

    sample_dist = Normal(10.0, 1.5)
    control_dist = Normal(8.0, 1.2)

    result = log2FC(sample_dist, control_dist)

    @test isa(result, Normal)
    @test mean(result) ≈ 2.0 atol=0.01
    # Variance should be sum of variances: 1.5^2 + 1.2^2
    expected_var = 1.5^2 + 1.2^2
    @test var(result) ≈ expected_var atol=0.01
end

@testitem "log2FC with equal sample and control" begin
    using BayesInteractomics
    using BayesInteractomics: log2FC
    using Distributions

    dist = Normal(8.0, 1.0)

    result = log2FC(dist, dist)

    @test mean(result) ≈ 0.0 atol=0.01
    @test var(result) ≈ 2.0 atol=0.01
end

@testitem "q-value calculation with Bayes factors" begin
    using BayesInteractomics
    using BayesInteractomics: q

    # BF vector: higher values should have lower q-values
    bf = [10.0, 5.0, 2.0, 0.5, 0.1]
    q_vals = q(bf, isBF=true)

    @test !any(ismissing.(q_vals))
    @test all(x -> 0.0 <= x <= 1.0, q_vals)
    # Q-values should be sorted (high BF -> low q, low BF -> high q)
    @test q_vals[1] <= q_vals[5]
end

@testitem "q-value calculation with posterior probabilities" begin
    using BayesInteractomics
    using BayesInteractomics: q

    # Posterior probabilities directly
    pp = [0.8, 0.5, 0.3, 0.1, 0.05]
    q_vals = q(pp, isBF=false)

    @test !any(ismissing.(q_vals))
    @test all(x -> 0.0 <= x <= 1.0, q_vals)
end

@testitem "q-value with missing values" begin
    using BayesInteractomics
    using BayesInteractomics: q

    # Vector with missing values
    bf = [10.0, missing, 5.0, missing, 2.0]
    q_vals = q(bf, isBF=true)

    @test ismissing(q_vals[2])
    @test ismissing(q_vals[4])
    @test !ismissing(q_vals[1])
    @test !ismissing(q_vals[3])
    @test !ismissing(q_vals[5])
end

@testitem "q-value with zero posterior probabilities" begin
    using BayesInteractomics
    using BayesInteractomics: q

    # BF = 0 converts to posterior probability = 0
    bf = [10.0, 0.0, 5.0]
    q_vals = q(bf, isBF=true)

    @test !any(ismissing.(q_vals))
    # BF=0 should have q-value = 1.0
    @test q_vals[2] == 1.0
end

@testitem "q-value with all missing values" begin
    using BayesInteractomics
    using BayesInteractomics: q

    bf = [missing, missing, missing]
    q_vals = q(bf, isBF=true)

    @test all(ismissing.(q_vals))
end

@testitem "cdf_log2FC computation" begin
    using BayesInteractomics
    using BayesInteractomics: cdf_log2FC
    using Distributions

    log2fc = Normal(2.0, 1.0)

    # CDF at threshold 0 should be < 0.5 since mean is 2.0
    cdf_at_zero = cdf_log2FC(log2fc, threshold=0.0)
    @test 0.0 < cdf_at_zero < 0.5

    # CDF at threshold equal to mean should be ~0.5
    cdf_at_mean = cdf_log2FC(log2fc, threshold=2.0)
    @test 0.4 < cdf_at_mean < 0.6

    # CDF at high threshold should be close to 1
    cdf_at_high = cdf_log2FC(log2fc, threshold=10.0)
    @test cdf_at_high > 0.95
end

@testitem "append_unique! merges vectors correctly" begin
    using BayesInteractomics
    using BayesInteractomics: append_unique!

    v1 = [1, 2, 3]
    v2 = [3, 4, 5]

    result = append_unique!(v1, v2)

    @test result == [1, 2, 3, 4, 5]
    @test v1 == [1, 2, 3, 4, 5]  # Modified in place
end

@testitem "append_unique! with no overlaps" begin
    using BayesInteractomics
    using BayesInteractomics: append_unique!

    v1 = [1, 2, 3]
    v2 = [4, 5, 6]

    result = append_unique!(v1, v2)

    @test result == [1, 2, 3, 4, 5, 6]
end

@testitem "append_unique! with complete overlap" begin
    using BayesInteractomics
    using BayesInteractomics: append_unique!

    v1 = [1, 2, 3]
    v2 = [1, 2, 3]

    result = append_unique!(v1, v2)

    @test result == [1, 2, 3]
end

@testitem "check_file validates file existence" begin
    using BayesInteractomics
    using BayesInteractomics: check_file

    # Non-existent file should throw
    @test_throws ArgumentError check_file("this_file_does_not_exist.txt")
end

@testitem "to_normal converts distributions correctly" begin
    using BayesInteractomics
    using BayesInteractomics: to_normal
    using Distributions

    # Test with Normal distribution
    normal_dist = Normal(5.0, 2.0)
    converted = to_normal(normal_dist)

    @test isa(converted, Normal)
    @test mean(converted) == 5.0
    @test std(converted) == 2.0
end
