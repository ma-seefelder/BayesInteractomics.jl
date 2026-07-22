"""
    test_qc.jl

Tests for input data quality control types and checks using TestItemRunner.
Each @testitem runs in an isolated environment.
"""

# ============================================================================
# Type and helper tests (can run now)
# ============================================================================

@testitem "worst_flag aggregation" begin
    using BayesInteractomics
    using BayesInteractomics: worst_flag

    # Single flags
    @test worst_flag(:ok) == :ok
    @test worst_flag(:warning) == :warning
    @test worst_flag(:fail) == :fail

    # Pairs
    @test worst_flag(:ok, :ok) == :ok
    @test worst_flag(:ok, :warning) == :warning
    @test worst_flag(:ok, :fail) == :fail
    @test worst_flag(:warning, :fail) == :fail
    @test worst_flag(:warning, :warning) == :warning

    # Empty
    @test worst_flag() == :ok

    # Multiple mixed
    @test worst_flag(:ok, :ok, :warning, :ok) == :warning
    @test worst_flag(:ok, :warning, :fail, :ok) == :fail
end

@testitem "InputQCResult construction and show" begin
    using BayesInteractomics
    using BayesInteractomics: InputQCResult, ScaleCheckResult, ProtocolScaleCheck, worst_flag

    # Construct with only scale check
    scale = ScaleCheckResult(
        [ProtocolScaleCheck(1, 25.0, :ok)],
        :ok
    )
    result = InputQCResult(scale, nothing, nothing, nothing, nothing, :ok)

    @test result.scale !== nothing
    @test result.replicate_correlation === nothing
    @test result.missingness === nothing
    @test result.intensity_shape === nothing
    @test result.overall_flag == :ok

    # Verify show method
    s = sprint(show, result)
    @test occursin("InputQCResult", s)
    @test occursin("ok", s)
    @test occursin("scale=ok", s)

    # Construct with warning scale
    scale_warn = ScaleCheckResult(
        [ProtocolScaleCheck(1, 5000.0, :warning)],
        :warning
    )
    result_warn = InputQCResult(scale_warn, nothing, nothing, nothing, nothing, :warning)
    @test result_warn.overall_flag == :warning
    s2 = sprint(show, result_warn)
    @test occursin("warning", s2)
end

@testitem "ProtocolScaleCheck construction" begin
    using BayesInteractomics
    using BayesInteractomics: ProtocolScaleCheck

    check = ProtocolScaleCheck(1, 25.0, :ok)
    @test check.protocol_index == 1
    @test check.max_value == 25.0
    @test check.flag == :ok

    check_warn = ProtocolScaleCheck(2, 1500.0, :warning)
    @test check_warn.flag == :warning
end

@testitem "ReplicateCorrelation construction" begin
    using BayesInteractomics
    using BayesInteractomics: ReplicateCorrelation, ReplicateCorrelationResult

    cor_mat = [1.0 0.9; 0.9 1.0]
    shared = [100 95; 95 100]
    check = ReplicateCorrelation(1, 1, :sample, cor_mat, shared, 2, 0.9, :ok)

    @test check.protocol_index == 1
    @test check.group == :sample
    @test check.min_correlation == 0.9
    @test check.flag == :ok

    result = ReplicateCorrelationResult([check], :ok)
    @test result.flag == :ok
    @test length(result.checks) == 1
end

@testitem "ReplicateMissingness construction" begin
    using BayesInteractomics
    using BayesInteractomics: ReplicateMissingness, MissingnessResult

    check = ReplicateMissingness(1, 1, :sample, [0.1, 0.12, 0.11], 0.11, 1.09, :ok)
    @test check.median_fraction == 0.11
    @test check.max_ratio == 1.09
    @test check.flag == :ok

    result = MissingnessResult([check], :ok)
    @test result.flag == :ok
end

@testitem "IntensityShapeCheck construction" begin
    using BayesInteractomics
    using BayesInteractomics: IntensityShapeCheck, IntensityShapeResult

    check = IntensityShapeCheck(1, 1, :sample, 1, 500, 0.5, 0.3, 0.01, :ok, :ok, :ok, :ok)
    @test check.n_values == 500
    @test check.bimodality_flag == :ok
    @test check.spike_flag == :ok
    @test check.tail_flag == :ok
    @test check.flag == :ok

    result = IntensityShapeResult([check], :ok)
    @test result.flag == :ok
end

# ============================================================================
# Functional tests
# ============================================================================

@testitem "Scale detection - log2 data passes" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, check_scale
    using Random
    Random.seed!(42)

    # Typical log2 AP-MS data: values in [10, 30] range
    n_proteins = 100
    ids = ["P$i" for i in 1:n_proteins]
    names = ["Protein$i" for i in 1:n_proteins]
    mat = Matrix{Union{Missing, Float64}}(10.0 .+ 20.0 .* rand(n_proteins, 3))

    sp = Protocol(1, ids, Dict(1 => mat))
    cp = Protocol(1, ids, Dict(1 => copy(mat)))
    data = InteractionData(
        ids, names, Dict(1 => sp), Dict(1 => cp),
        1, Dict(1 => 1), 2 + n_proteins, 1 + n_proteins,
        [1], [1], collect(1:n_proteins), trues(n_proteins)
    )

    result = check_scale(data)
    @test result.flag == :ok
    @test result.protocols[1].max_value <= 1000.0
end

@testitem "Scale detection - linear data warns" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, check_scale
    using Random
    Random.seed!(42)

    n_proteins = 100
    ids = ["P$i" for i in 1:n_proteins]
    names = ["Protein$i" for i in 1:n_proteins]
    mat = Matrix{Union{Missing, Float64}}(hcat(
        1e3 .+ 1e3 .* rand(n_proteins),
        1e5 .+ 1e5 .* rand(n_proteins),
        1e7 .+ 1e7 .* rand(n_proteins)
    ))

    sp = Protocol(1, ids, Dict(1 => mat))
    cp = Protocol(1, ids, Dict(1 => copy(mat)))
    data = InteractionData(
        ids, names, Dict(1 => sp), Dict(1 => cp),
        1, Dict(1 => 1), 2 + n_proteins, 1 + n_proteins,
        [1], [1], collect(1:n_proteins), trues(n_proteins)
    )

    result = check_scale(data)
    @test result.flag == :warning
    @test result.protocols[1].max_value > 1000.0
end

@testitem "Replicate correlation - good replicates" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, check_replicate_correlation
    using Random
    Random.seed!(42)

    n_proteins = 100
    ids = ["P$i" for i in 1:n_proteins]
    names = ["Protein$i" for i in 1:n_proteins]
    base_signal = 10.0 .+ 15.0 .* rand(n_proteins)
    noise_scale = 0.5
    mat = Matrix{Union{Missing, Float64}}(hcat(
        base_signal .+ noise_scale .* randn(n_proteins),
        base_signal .+ noise_scale .* randn(n_proteins),
        base_signal .+ noise_scale .* randn(n_proteins)
    ))

    sp = Protocol(1, ids, Dict(1 => mat))
    cp = Protocol(1, ids, Dict(1 => copy(mat)))
    data = InteractionData(
        ids, names, Dict(1 => sp), Dict(1 => cp),
        1, Dict(1 => 1), 2 + n_proteins, 1 + n_proteins,
        [1], [1], collect(1:n_proteins), trues(n_proteins)
    )

    result = check_replicate_correlation(data)
    @test result.flag == :ok
    # Both sample and control checks should be :ok
    for c in result.checks
        @test c.flag == :ok
        @test c.min_correlation >= 0.80
    end
end

@testitem "Replicate correlation - bad replicate" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, check_replicate_correlation
    using Random
    Random.seed!(42)

    n_proteins = 100
    ids = ["P$i" for i in 1:n_proteins]
    names = ["Protein$i" for i in 1:n_proteins]
    base_signal = 10.0 .+ 15.0 .* rand(n_proteins)
    mat_bad = Matrix{Union{Missing, Float64}}(hcat(
        base_signal .+ 0.5 .* randn(n_proteins),
        base_signal .+ 0.5 .* randn(n_proteins),
        10.0 .+ 15.0 .* rand(n_proteins)  # uncorrelated noise
    ))
    mat_good = Matrix{Union{Missing, Float64}}(hcat(
        base_signal .+ 0.5 .* randn(n_proteins),
        base_signal .+ 0.5 .* randn(n_proteins),
        base_signal .+ 0.5 .* randn(n_proteins)
    ))

    sp = Protocol(1, ids, Dict(1 => mat_bad))
    cp = Protocol(1, ids, Dict(1 => mat_good))
    data = InteractionData(
        ids, names, Dict(1 => sp), Dict(1 => cp),
        1, Dict(1 => 1), 2 + n_proteins, 1 + n_proteins,
        [1], [1], collect(1:n_proteins), trues(n_proteins)
    )

    result = check_replicate_correlation(data)
    # Overall should be :fail because sample group has uncorrelated replicate
    @test result.flag == :fail
    # Find the sample check
    sample_check = first(c for c in result.checks if c.group == :sample)
    @test sample_check.flag == :fail
    @test sample_check.min_correlation < 0.60
end

@testitem "Replicate correlation - shared count tracking" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, check_replicate_correlation
    using Random
    Random.seed!(42)

    n_proteins = 100
    ids = ["P$i" for i in 1:n_proteins]
    names = ["Protein$i" for i in 1:n_proteins]
    base_signal = 10.0 .+ 15.0 .* rand(n_proteins)
    mat = Matrix{Union{Missing, Float64}}(hcat(
        base_signal .+ 0.5 .* randn(n_proteins),
        base_signal .+ 0.5 .* randn(n_proteins),
        base_signal .+ 0.5 .* randn(n_proteins)
    ))

    # Introduce missing values
    mat[1:10, 1] .= missing    # 10 missing in col 1
    mat[5:20, 2] .= missing    # 16 missing in col 2
    mat[15:25, 3] .= missing   # 11 missing in col 3

    ctrl_mat = Matrix{Union{Missing, Float64}}(10.0 .+ 15.0 .* rand(n_proteins, 3))
    sp = Protocol(1, ids, Dict(1 => mat))
    cp = Protocol(1, ids, Dict(1 => ctrl_mat))
    data = InteractionData(
        ids, names, Dict(1 => sp), Dict(1 => cp),
        1, Dict(1 => 1), 2 + n_proteins, 1 + n_proteins,
        [1], [1], collect(1:n_proteins), trues(n_proteins)
    )

    result = check_replicate_correlation(data)
    sample_check = first(c for c in result.checks if c.group == :sample)
    # Cols 1,2: 100 - |union 1:10 and 5:20| = 100 - 20 = 80
    @test sample_check.shared_counts[1, 2] == 80
end

@testitem "Missingness asymmetry - balanced" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, check_missingness
    using Random
    Random.seed!(42)

    n_proteins = 100
    ids = ["P$i" for i in 1:n_proteins]
    names = ["Protein$i" for i in 1:n_proteins]
    mat = Matrix{Union{Missing, Float64}}(10.0 .+ 15.0 .* rand(n_proteins, 3))

    # Add ~10% missing to each column
    for col in 1:3
        idx = randperm(n_proteins)[1:10]
        mat[idx, col] .= missing
    end

    ctrl_mat = Matrix{Union{Missing, Float64}}(10.0 .+ 15.0 .* rand(n_proteins, 3))
    sp = Protocol(1, ids, Dict(1 => mat))
    cp = Protocol(1, ids, Dict(1 => ctrl_mat))
    data = InteractionData(
        ids, names, Dict(1 => sp), Dict(1 => cp),
        1, Dict(1 => 1), 2 + n_proteins, 1 + n_proteins,
        [1], [1], collect(1:n_proteins), trues(n_proteins)
    )

    result = check_missingness(data)
    # Balanced missingness should not trigger warning
    sample_check = first(c for c in result.checks if c.group == :sample)
    @test sample_check.flag == :ok
    @test sample_check.max_ratio <= 2.0
end

@testitem "Missingness asymmetry - asymmetric" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, check_missingness
    using Random
    Random.seed!(42)

    n_proteins = 100
    ids = ["P$i" for i in 1:n_proteins]
    names = ["Protein$i" for i in 1:n_proteins]
    mat = Matrix{Union{Missing, Float64}}(10.0 .+ 15.0 .* rand(n_proteins, 3))

    # Col 1: 5% missing
    mat[randperm(n_proteins)[1:5], 1] .= missing
    # Col 2: 10% missing
    mat[randperm(n_proteins)[1:10], 2] .= missing
    # Col 3: 50% missing
    mat[randperm(n_proteins)[1:50], 3] .= missing

    ctrl_mat = Matrix{Union{Missing, Float64}}(10.0 .+ 15.0 .* rand(n_proteins, 3))
    sp = Protocol(1, ids, Dict(1 => mat))
    cp = Protocol(1, ids, Dict(1 => ctrl_mat))
    data = InteractionData(
        ids, names, Dict(1 => sp), Dict(1 => cp),
        1, Dict(1 => 1), 2 + n_proteins, 1 + n_proteins,
        [1], [1], collect(1:n_proteins), trues(n_proteins)
    )

    result = check_missingness(data)
    sample_check = first(c for c in result.checks if c.group == :sample)
    @test sample_check.flag == :fail
    @test sample_check.max_ratio > 3.0
end

@testitem "Intensity shape - normal data" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, check_intensity_shape
    using Random
    Random.seed!(42)

    # Single replicate with 500 normally-distributed values (log2 scale)
    n_proteins = 500
    ids = ["P$i" for i in 1:n_proteins]
    names = ["Protein$i" for i in 1:n_proteins]
    values = 20.0 .+ 3.0 .* randn(n_proteins)
    mat = Matrix{Union{Missing, Float64}}(reshape(values, n_proteins, 1))
    ctrl_mat = Matrix{Union{Missing, Float64}}(20.0 .+ 3.0 .* randn(n_proteins, 1))

    sp = Protocol(1, ids, Dict(1 => mat))
    cp = Protocol(1, ids, Dict(1 => ctrl_mat))
    data = InteractionData(
        ids, names, Dict(1 => sp), Dict(1 => cp),
        1, Dict(1 => 1), 2 + n_proteins, 1 + n_proteins,
        [1], [1], collect(1:n_proteins), trues(n_proteins)
    )

    result = check_intensity_shape(data)
    @test result.flag == :ok
    # At least one check should exist (the sample replicate)
    @test length(result.checks) >= 1
    sample_check = first(c for c in result.checks if c.group == :sample)
    @test sample_check.bimodality_flag == :ok
    @test sample_check.spike_flag == :ok
    @test sample_check.tail_flag == :ok
    @test sample_check.flag == :ok
end

@testitem "Intensity shape - bimodal data" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, check_intensity_shape
    using Random
    Random.seed!(42)

    # Mixture of two well-separated normals: N(5,1) and N(25,1)
    n_each = 250
    n_proteins = 2 * n_each
    ids = ["P$i" for i in 1:n_proteins]
    names = ["Protein$i" for i in 1:n_proteins]
    values = vcat(5.0 .+ randn(n_each), 25.0 .+ randn(n_each))
    mat = Matrix{Union{Missing, Float64}}(reshape(values, n_proteins, 1))
    ctrl_mat = Matrix{Union{Missing, Float64}}(15.0 .+ 3.0 .* randn(n_proteins, 1))

    sp = Protocol(1, ids, Dict(1 => mat))
    cp = Protocol(1, ids, Dict(1 => ctrl_mat))
    data = InteractionData(
        ids, names, Dict(1 => sp), Dict(1 => cp),
        1, Dict(1 => 1), 2 + n_proteins, 1 + n_proteins,
        [1], [1], collect(1:n_proteins), trues(n_proteins)
    )

    result = check_intensity_shape(data)
    sample_check = first(c for c in result.checks if c.group == :sample)
    @test sample_check.bimodality_flag == :warning
end

@testitem "Intensity shape - spike at zero" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, check_intensity_shape
    using Random
    Random.seed!(42)

    # 25% of values are exactly 0.0 (common artifact in AP-MS data)
    n_proteins = 500
    ids = ["P$i" for i in 1:n_proteins]
    names = ["Protein$i" for i in 1:n_proteins]
    n_zeros = div(n_proteins, 4)  # 125 zeros = 25%
    values = vcat(zeros(n_zeros), 10.0 .+ 5.0 .* randn(n_proteins - n_zeros))
    mat = Matrix{Union{Missing, Float64}}(reshape(values, n_proteins, 1))
    ctrl_mat = Matrix{Union{Missing, Float64}}(10.0 .+ 5.0 .* randn(n_proteins, 1))

    sp = Protocol(1, ids, Dict(1 => mat))
    cp = Protocol(1, ids, Dict(1 => ctrl_mat))
    data = InteractionData(
        ids, names, Dict(1 => sp), Dict(1 => cp),
        1, Dict(1 => 1), 2 + n_proteins, 1 + n_proteins,
        [1], [1], collect(1:n_proteins), trues(n_proteins)
    )

    result = check_intensity_shape(data)
    sample_check = first(c for c in result.checks if c.group == :sample)
    @test sample_check.spike_flag == :warning
    @test sample_check.spike_fraction >= 0.20
end
