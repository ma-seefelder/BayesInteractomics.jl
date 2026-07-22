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

@testitem "BFDR calculation with Bayes factors" begin
    using BayesInteractomics
    using BayesInteractomics: bfdr

    # BF vector: higher values should have lower BFDR values
    bf = [10.0, 5.0, 2.0, 0.5, 0.1]
    bfdr_vals = bfdr(bf, isBF=true)

    @test !any(ismissing.(bfdr_vals))
    @test all(x -> 0.0 <= x <= 1.0, bfdr_vals)
    # BFDR values should be sorted (high BF -> low BFDR, low BF -> high BFDR)
    @test bfdr_vals[1] <= bfdr_vals[5]
end

@testitem "BFDR calculation with posterior probabilities" begin
    using BayesInteractomics
    using BayesInteractomics: bfdr

    # Posterior probabilities directly
    pp = [0.8, 0.5, 0.3, 0.1, 0.05]
    bfdr_vals = bfdr(pp, isBF=false)

    @test !any(ismissing.(bfdr_vals))
    @test all(x -> 0.0 <= x <= 1.0, bfdr_vals)
end

@testitem "BFDR with missing values" begin
    using BayesInteractomics
    using BayesInteractomics: bfdr

    # Vector with missing values
    bf = [10.0, missing, 5.0, missing, 2.0]
    bfdr_vals = bfdr(bf, isBF=true)

    @test ismissing(bfdr_vals[2])
    @test ismissing(bfdr_vals[4])
    @test !ismissing(bfdr_vals[1])
    @test !ismissing(bfdr_vals[3])
    @test !ismissing(bfdr_vals[5])
end

@testitem "BFDR with zero posterior probabilities" begin
    using BayesInteractomics
    using BayesInteractomics: bfdr

    # BF = 0 converts to posterior probability = 0
    bf = [10.0, 0.0, 5.0]
    bfdr_vals = bfdr(bf, isBF=true)

    @test !any(ismissing.(bfdr_vals))
    # BF=0 should have BFDR = 1.0
    @test bfdr_vals[2] == 1.0
end

@testitem "BFDR with all missing values" begin
    using BayesInteractomics
    using BayesInteractomics: bfdr

    bf = [missing, missing, missing]
    bfdr_vals = bfdr(bf, isBF=true)

    @test all(ismissing.(bfdr_vals))
end

@testitem "pep computation" begin
    using BayesInteractomics
    using BayesInteractomics: pep

    pp = [0.95, 0.8, 0.5, 0.1, 0.0]
    pep_vals = pep(pp)

    @test pep_vals[1] ≈ 0.05
    @test pep_vals[2] ≈ 0.20
    @test pep_vals[3] ≈ 0.50
    @test pep_vals[4] ≈ 0.90
    @test pep_vals[5] ≈ 1.0

    # Missing handling
    pp_missing = [0.9, missing, 0.5]
    pep_missing = pep(pp_missing)
    @test pep_missing[1] ≈ 0.1
    @test ismissing(pep_missing[2])
    @test pep_missing[3] ≈ 0.5
end

@testitem "deprecated q() alias warns" begin
    using BayesInteractomics
    using BayesInteractomics: q, bfdr

    bf = [10.0, 5.0, 2.0]
    # q() should produce same result as bfdr() but emit deprecation warning
    bfdr_vals = bfdr(bf, isBF=true)
    q_vals = q(bf, isBF=true)

    @test q_vals == bfdr_vals
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

@testitem "BFDR monotonicity with posterior probabilities" begin
    using BayesInteractomics
    using BayesInteractomics: bfdr

    pp = [0.95, 0.8, 0.5, 0.3, 0.1, 0.05]
    bfdr_vals = bfdr(pp, isBF=false)

    # All BFDR values must be in [0,1]
    @test all(x -> 0.0 <= x <= 1.0, skipmissing(bfdr_vals))

    # When sorted by decreasing posterior, BFDR values must be non-increasing
    sorted_order = sortperm(pp, rev=true)
    bfdr_sorted = bfdr_vals[sorted_order]
    for i in 1:(length(bfdr_sorted)-1)
        @test bfdr_sorted[i] <= bfdr_sorted[i+1]
    end
end

@testitem "BFDR monotonicity property-based" begin
    using BayesInteractomics
    using BayesInteractomics: bfdr
    using Random

    rng = MersenneTwister(42)
    for _ in 1:100
        n = rand(rng, 5:50)
        pp = rand(rng, n)
        bfdr_vals = bfdr(pp, isBF=false)

        # All valid BFDR values in [0,1]
        valid = collect(skipmissing(bfdr_vals))
        @test all(x -> 0.0 <= x <= 1.0, valid)

        # When sorted by decreasing posterior, BFDR values must be non-increasing
        sorted_order = sortperm(pp, rev=true)
        bfdr_sorted = bfdr_vals[sorted_order]
        for i in 1:(length(bfdr_sorted)-1)
            @test bfdr_sorted[i] <= bfdr_sorted[i+1]
        end
    end
end

@testitem "BFDR EM posteriors differ from flat-prior BF/(1+BF)" begin
    using BayesInteractomics
    using BayesInteractomics: bfdr

    # Simulate the real pipeline scenario: EM posteriors differ from BF/(1+BF)
    # because EM uses mixture model weights, not flat 0.5 prior.
    combined_bf = [100.0, 20.0, 5.0, 1.0, 0.2]
    em_posteriors = [0.99, 0.92, 0.70, 0.25, 0.03]

    bfdr_from_em = bfdr(em_posteriors, isBF=false)
    bfdr_from_bf = bfdr(combined_bf, isBF=true)

    # BFDR values must differ because EM posteriors != BF/(1+BF) flat-prior posteriors
    @test any(i -> !isapprox(bfdr_from_em[i], bfdr_from_bf[i]; atol=1e-10), eachindex(bfdr_from_em))

    # Both should still be valid BFDR values
    @test all(x -> 0.0 <= x <= 1.0, skipmissing(bfdr_from_em))
    @test all(x -> 0.0 <= x <= 1.0, skipmissing(bfdr_from_bf))
end
