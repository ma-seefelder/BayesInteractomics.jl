"""
    test_models.jl

Tests for Bayesian model interfaces and utility functions.
Focus on testable components without running full inference.
"""

@testitem "getParameterLookup with single protocol" begin
    using BayesInteractomics
    using BayesInteractomics: getParameterLookup

    n_protocols = 1
    n_experiments = Dict(1 => 3)

    lookup = getParameterLookup(n_protocols, n_experiments)

    @test size(lookup) == (1, 3)
    @test lookup[1, 1] == 3
    @test lookup[1, 2] == 4
    @test lookup[1, 3] == 5
end

@testitem "getParameterLookup with multiple protocols" begin
    using BayesInteractomics
    using BayesInteractomics: getParameterLookup

    n_protocols = 2
    n_experiments = Dict(1 => 2, 2 => 3)

    lookup = getParameterLookup(n_protocols, n_experiments)

    @test size(lookup) == (2, 3)
    # Protocol 1: experiments at positions 3, 4
    @test lookup[1, 1] == 3
    @test lookup[1, 2] == 4
    # Protocol 2: experiments at positions 6, 7, 8
    @test lookup[2, 1] == 6
    @test lookup[2, 2] == 7
    @test lookup[2, 3] == 8
end

@testitem "getParameterLookup with unequal experiments" begin
    using BayesInteractomics
    using BayesInteractomics: getParameterLookup

    n_protocols = 3
    n_experiments = Dict(1 => 1, 2 => 4, 3 => 2)

    lookup = getParameterLookup(n_protocols, n_experiments)

    @test size(lookup) == (3, 4)  # Max experiments = 4
    # First row should have only one non-zero entry
    @test lookup[1, 1] == 3
    @test lookup[1, 2] == 0
end

@testitem "checkForDuplicates detects duplicates" begin
    using BayesInteractomics
    using BayesInteractomics: checkForDuplicates

    # Vector with duplicates
    v_dups = ["A", "B", "A", "C"]
    @test_throws AssertionError checkForDuplicates(v_dups)

    # Vector without duplicates - should return nothing
    v_no_dups = ["A", "B", "C", "D"]
    @test isnothing(checkForDuplicates(v_no_dups))
end

@testitem "compute_σ2 variance computation" begin
    using BayesInteractomics
    using BayesInteractomics: compute_σ2

    # Create 3D arrays: (protocols, experiments, replicates)
    # Sample: 1 protocol, 2 experiments, 3 replicates each
    sample = Array{Union{Missing, Float64}, 3}(undef, 1, 2, 3)
    sample[1, 1, :] = [1.0, 2.0, 3.0]
    sample[1, 2, :] = [4.0, 5.0, 6.0]

    control = Array{Union{Missing, Float64}, 3}(undef, 1, 2, 3)
    control[1, 1, :] = [2.0, 3.0, 4.0]
    control[1, 2, :] = [5.0, 6.0, 7.0]

    σ2 = compute_σ2(sample, control)

    @test σ2 > 0  # Variance must be positive
end

@testitem "Protein data extraction" begin
    using BayesInteractomics
    using BayesInteractomics: getProteinData, getSampleMatrix, getControlMatrix, Protocol, InteractionData

    m_sample::Matrix{Union{Missing, Float64}} = [1.0 2.0; 3.0 4.0]
    m_control::Matrix{Union{Missing, Float64}} = [5.0 6.0; 7.0 8.0]

    p_sample = Protocol(1, ["P1", "P2"], Dict(1 => m_sample))
    p_control = Protocol(1, ["P1", "P2"], Dict(1 => m_control))

    data = InteractionData(
        ["P1", "P2"], ["Protein1", "Protein2"],
        Dict(1 => p_sample),
        Dict(1 => p_control),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1]
    )

    protein = getProteinData(data, 1)
    sample_matrix = getSampleMatrix(protein)
    control_matrix = getControlMatrix(protein)

    @test size(sample_matrix, 3) == 2  # 2 replicates
    @test size(control_matrix, 3) == 2
end

@testitem "HBM model position assertions" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, getProtocolPositions, getExperimentPositions

    m::Matrix{Union{Missing, Float64}} = [1.0 2.0; 3.0 4.0]
    p = Protocol(1, ["P1", "P2"], Dict(1 => m))

    data = InteractionData(
        ["P1", "P2"], ["Protein1", "Protein2"],
        Dict(1 => p),
        Dict(1 => p),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1]
    )

    proto_pos = getProtocolPositions(data)
    exp_pos = getExperimentPositions(data)

    # Protocol positions and experiment positions should not overlap
    @test isempty(intersect(proto_pos, exp_pos))
end

@testitem "InteractionData position consistency" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, getMatchedPositions

    m::Matrix{Union{Missing, Float64}} = [1.0 2.0; 3.0 4.0]
    p = Protocol(2, ["P1", "P2"], Dict(1 => m, 2 => m))

    # 2 protocols, each with 2 experiments
    data = InteractionData(
        ["P1", "P2"], ["Protein1", "Protein2"],
        Dict(1 => p, 2 => p),
        Dict(1 => p, 2 => p),
        2, Dict(1 => 2, 2 => 2),
        8, 4,
        [2, 5], [3, 4, 6, 7], [1, 1, 2, 2]
    )

    matched = getMatchedPositions(data)

    # Matched positions should indicate which protocol each experiment belongs to
    @test length(matched) == 4  # 4 experiments total
end

@testitem "Model parameter count validation" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData

    m::Matrix{Union{Missing, Float64}} = [1.0 2.0; 3.0 4.0]
    p = Protocol(2, ["P1", "P2"], Dict(1 => m, 2 => m))

    # Create data with known parameter counts
    n_protocols = 2
    n_experiments_per_protocol = 2

    data = InteractionData(
        ["P1", "P2"], ["Protein1", "Protein2"],
        Dict(1 => p, 2 => p),
        Dict(1 => p, 2 => p),
        n_protocols, Dict(1 => n_experiments_per_protocol, 2 => n_experiments_per_protocol),
        8, 4,  # HBM params, regression params
        [2, 5], [3, 4, 6, 7], [1, 1, 2, 2]
    )

    # Verify parameter counts are reasonable
    @test data.no_parameters_HBM >= n_protocols + 1
    @test data.no_parameters_Regression >= n_protocols + 1
end

@testitem "compute_log2FC basic computation" begin
    using BayesInteractomics
    using BayesInteractomics: compute_log2FC, Protocol, InteractionData

    # Create simple interaction data with known values
    m_sample::Matrix{Union{Missing, Float64}} = [10.0 11.0 12.0; 8.0 9.0 10.0]
    m_control::Matrix{Union{Missing, Float64}} = [8.0 9.0 10.0; 8.0 9.0 10.0]

    p_sample = Protocol(1, ["P1", "P2"], Dict(1 => m_sample))
    p_control = Protocol(1, ["P1", "P2"], Dict(1 => m_control))

    data = InteractionData(
        ["P1", "P2"], ["Protein1", "Protein2"],
        Dict(1 => p_sample),
        Dict(1 => p_control),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1]
    )

    log2fc = compute_log2FC(data, 1)

    # compute_log2FC returns a vector (one value per protocol×experiment)
    @test log2fc isa Vector
    @test length(log2fc) > 0
    # First protein: sample mean ~11, control mean ~9 -> positive log2FC
    @test any(x -> !ismissing(x) && x > 0, log2fc)
end

@testitem "prepare_regression_data structure" begin
    using BayesInteractomics
    using BayesInteractomics: prepare_regression_data, Protocol, InteractionData

    # Create interaction data with 3 proteins, 1 protocol
    m_sample::Matrix{Union{Missing, Float64}} = [10.0 11.0 12.0; 8.0 9.0 10.0; 5.0 6.0 7.0]
    m_control::Matrix{Union{Missing, Float64}} = [8.0 9.0 10.0; 8.0 9.0 10.0; 5.0 6.0 7.0]

    p_sample = Protocol(1, ["P1", "P2", "P3"], Dict(1 => m_sample))
    p_control = Protocol(1, ["P1", "P2", "P3"], Dict(1 => m_control))

    data = InteractionData(
        ["P1", "P2", "P3"], ["Protein1", "Protein2", "Protein3"],
        Dict(1 => p_sample),
        Dict(1 => p_control),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1]
    )

    # Prepare regression data for protein 1 with reference protein 2
    x, y = prepare_regression_data(data, 1, 2)

    @test length(x) == length(y)
    @test length(x) > 0
end

@testitem "τ0 computation for precision prior" begin
    using BayesInteractomics
    using BayesInteractomics: τ0, Protocol, InteractionData
    using Distributions
    using Random

    Random.seed!(42)
    # Create data with more variation to avoid NaN in gamma fitting
    # Need enough proteins with varying data
    m_sample::Matrix{Union{Missing, Float64}} = [
        10.0  11.0  12.0  13.0  14.0;
        8.0   9.0   10.0  11.0  12.0;
        15.0  16.0  17.0  18.0  19.0;
        5.0   6.0   7.0   8.0   9.0;
        20.0  21.0  22.0  23.0  24.0
    ]
    m_control::Matrix{Union{Missing, Float64}} = [
        8.0   9.0   10.0  11.0  12.0;
        8.0   9.0   10.0  11.0  12.0;
        10.0  11.0  12.0  13.0  14.0;
        5.0   6.0   7.0   8.0   9.0;
        15.0  16.0  17.0  18.0  19.0
    ]

    protein_ids = ["P1", "P2", "P3", "P4", "P5"]
    p_sample = Protocol(1, protein_ids, Dict(1 => m_sample))
    p_control = Protocol(1, protein_ids, Dict(1 => m_control))

    data = InteractionData(
        protein_ids, protein_ids,
        Dict(1 => p_sample),
        Dict(1 => p_control),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1]
    )

    # τ0 returns a fitted Gamma distribution
    tau_dist = τ0(data)

    @test tau_dist isa Gamma
    @test shape(tau_dist) > 0
    @test scale(tau_dist) > 0
end

@testitem "μ0 computation for mean prior" begin
    using BayesInteractomics
    using BayesInteractomics: μ0, Protocol, InteractionData

    m_sample::Matrix{Union{Missing, Float64}} = [10.0 11.0 12.0; 8.0 9.0 10.0]
    m_control::Matrix{Union{Missing, Float64}} = [8.0 9.0 10.0; 8.0 9.0 10.0]

    p_sample = Protocol(1, ["P1", "P2"], Dict(1 => m_sample))
    p_control = Protocol(1, ["P1", "P2"], Dict(1 => m_control))

    data = InteractionData(
        ["P1", "P2"], ["Protein1", "Protein2"],
        Dict(1 => p_sample),
        Dict(1 => p_control),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1]
    )

    # μ0 returns a tuple: (median_of_means, max_variance)
    mu_median, max_sigma2 = μ0(data)

    # μ0 median should be close to the overall mean of the data
    @test 7.0 < mu_median < 12.0
    # max_sigma2 should be positive
    @test max_sigma2 > 0.0
end
