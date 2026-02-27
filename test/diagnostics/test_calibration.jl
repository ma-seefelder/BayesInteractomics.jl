"""
    test_calibration.jl

Tests for the calibration assessment.
"""

@testitem "CalibrationResult construction" begin
    using BayesInteractomics

    midpoints = [0.1, 0.3, 0.5, 0.7, 0.9]
    predicted = [0.1, 0.3, 0.5, 0.7, 0.9]
    observed = [0.08, 0.28, 0.52, 0.72, 0.88]
    counts = [20, 30, 25, 15, 10]

    cal = CalibrationResult(midpoints, predicted, observed, counts, 0.02, 0.03)

    @test length(cal.bin_midpoints) == 5
    @test length(cal.predicted_rate) == 5
    @test length(cal.observed_rate) == 5
    @test length(cal.bin_counts) == 5
    @test cal.ece ≈ 0.02
    @test cal.mce ≈ 0.03
end

@testitem "CalibrationResult perfect calibration" begin
    using BayesInteractomics

    # Perfect calibration: predicted == observed -> ECE = 0, MCE = 0
    n = 5
    rates = collect(range(0.1, 0.9, length=n))
    counts = fill(10, n)

    cal = CalibrationResult(rates, rates, rates, counts, 0.0, 0.0)
    @test cal.ece ≈ 0.0
    @test cal.mce ≈ 0.0
end

@testitem "ECE computation logic" begin
    using BayesInteractomics
    using DataFrames

    # Create a mock AnalysisResult-like structure to test _compute_calibration
    # We test the logic by constructing CalibrationResult directly
    midpoints = [0.25, 0.75]
    predicted = [0.2, 0.8]
    observed = [0.3, 0.7]
    counts = [50, 50]

    # ECE = (50/100)*|0.2-0.3| + (50/100)*|0.8-0.7| = 0.5*0.1 + 0.5*0.1 = 0.1
    ece = sum(counts ./ sum(counts) .* abs.(predicted .- observed))
    mce = maximum(abs.(predicted .- observed))

    cal = CalibrationResult(midpoints, predicted, observed, counts, ece, mce)
    @test cal.ece ≈ 0.1
    @test cal.mce ≈ 0.1
end

@testitem "CalibrationResult bin counts validation" begin
    using BayesInteractomics

    # Verify that bin counts are non-negative integers
    midpoints = collect(range(0.05, 0.95, length=10))
    predicted = midpoints
    observed = midpoints .+ 0.01 .* randn(10)
    counts = rand(0:100, 10)

    cal = CalibrationResult(midpoints, predicted, observed, counts, 0.01, 0.05)
    @test all(c -> c >= 0, cal.bin_counts)
    @test length(cal.bin_counts) == 10
end
