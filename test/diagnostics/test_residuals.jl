"""
    test_residuals.jl

Tests for the standardized residual computation.
"""

@testitem "ResidualResult construction" begin
    using BayesInteractomics

    names = ["P1", "P2", "P3"]
    resids = [randn(10) for _ in 1:3]
    mean_r = [0.1, -0.2, 0.5]
    pooled = reduce(vcat, resids)

    fitted = reduce(vcat, [fill(Float64(i), 10) for i in 1:3])

    rr = ResidualResult(
        :hbm,
        names,
        resids,
        mean_r,
        pooled,
        fitted,
        0.15,   # skewness
        0.3,    # kurtosis
        String[]  # no outliers
    )

    @test rr.model == :hbm
    @test length(rr.protein_names) == 3
    @test length(rr.residuals) == 3
    @test length(rr.pooled_residuals) == 30
    @test length(rr.pooled_fitted) == 30
    @test rr.skewness ≈ 0.15
    @test rr.kurtosis ≈ 0.3
    @test isempty(rr.outlier_proteins)
end

@testitem "ResidualResult with outliers" begin
    using BayesInteractomics

    names = ["P1", "P2", "P3"]
    resids = [randn(10) for _ in 1:3]
    mean_r = [0.5, -3.0, 2.5]  # P2 and P3 are outliers
    pooled = reduce(vcat, resids)

    outliers = names[abs.(mean_r) .> 2.0]

    fitted = reduce(vcat, [fill(Float64(i), 10) for i in 1:3])
    rr = ResidualResult(:regression, names, resids, mean_r, pooled, fitted, 0.0, 0.0, outliers)

    @test length(rr.outlier_proteins) == 2
    @test "P2" in rr.outlier_proteins
    @test "P3" in rr.outlier_proteins
    @test !("P1" in rr.outlier_proteins)
end

@testitem "Skewness computation" begin
    using BayesInteractomics
    using Random

    # Symmetric distribution should have near-zero skewness
    rng = MersenneTwister(42)
    x = randn(rng, 10000)
    sk = BayesInteractomics._compute_skewness(x)
    @test abs(sk) < 0.1

    # Edge cases
    @test isnan(BayesInteractomics._compute_skewness(Float64[]))
    @test isnan(BayesInteractomics._compute_skewness([1.0, 2.0]))
end

@testitem "Kurtosis computation" begin
    using BayesInteractomics
    using Random

    # Normal distribution should have near-zero excess kurtosis
    rng = MersenneTwister(42)
    x = randn(rng, 10000)
    ku = BayesInteractomics._compute_kurtosis(x)
    @test abs(ku) < 0.2

    # Edge cases
    @test isnan(BayesInteractomics._compute_kurtosis(Float64[]))
    @test isnan(BayesInteractomics._compute_kurtosis([1.0, 2.0, 3.0]))
end
