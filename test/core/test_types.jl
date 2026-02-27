"""
    test_types.jl

Tests for core data structure types: Protocol, InteractionData, and evidence containers.
"""

@testitem "Protocol creation and basic accessors" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, getNoExperiments, getExperiment, getIDs

    m::Matrix{Union{Missing, Float64}} = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    p = Protocol(2, ["P1", "P2", "P3"], Dict(1 => m, 2 => m))

    @test getNoExperiments(p) == 2
    @test getIDs(p) == ["P1", "P2", "P3"]
    @test size(getExperiment(p, 1)) == (3, 2)
end

@testitem "Protocol invalid experiment access" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, getExperiment

    m::Matrix{Union{Missing, Float64}} = [1.0 2.0; 3.0 4.0]
    p = Protocol(2, ["P1"], Dict(1 => m, 2 => m))

    @test_throws BoundsError getExperiment(p, 0)
    @test_throws BoundsError getExperiment(p, 3)
end

@testitem "Protocol iterator interface" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol

    m1::Matrix{Union{Missing, Float64}} = [1.0 2.0; 3.0 4.0]
    m2::Matrix{Union{Missing, Float64}} = [5.0 6.0; 7.0 8.0]
    p = Protocol(2, ["P1"], Dict(1 => m1, 2 => m2))

    matrices = collect(p)
    @test length(matrices) == 2
    @test matrices[1] == m1
    @test matrices[2] == m2
end

@testitem "InteractionData with multiple protocols" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, getNoProtocols, getControls, getSamples

    m::Matrix{Union{Missing, Float64}} = [1.0 2.0; 3.0 4.0]
    p1 = Protocol(1, ["P1"], Dict(1 => m))
    p2 = Protocol(1, ["P1"], Dict(1 => m))

    # InteractionData field order:
    # protein_IDs, protein_names, samples, controls,
    # no_protocols, no_experiments, no_parameters_HBM, no_parameters_Regression,
    # protocol_positions, experiment_positions, matched_positions
    data = InteractionData(
        ["P1"], ["Protein1"],
        Dict(1 => p1, 2 => p2),
        Dict(1 => p2, 2 => p1),
        2, Dict(1 => 1, 2 => 1),
        4, 3,
        [2, 4], [3, 5], [1, 2]
    )

    @test getNoProtocols(data) == 2
    @test isa(getControls(data, 1), Protocol)
    @test isa(getSamples(data, 1), Protocol)
end

@testitem "InteractionData protocol mismatch error" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, getControls

    m::Matrix{Union{Missing, Float64}} = [1.0 2.0; 3.0 4.0]
    p = Protocol(1, ["P1"], Dict(1 => m))

    data = InteractionData(
        ["P1"], ["Protein1"],
        Dict(1 => p),
        Dict(1 => p),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1]
    )

    @test_throws ArgumentError getControls(data, 2)
end

@testitem "BayesFactorTriplet construction" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet

    enrichment = [10.0, 5.0, 0.5]
    correlation = [8.0, 3.0, 0.2]
    detection = [15.0, 7.0, 0.1]

    bf = BayesFactorTriplet(enrichment, correlation, detection)

    @test bf.enrichment == enrichment
    @test bf.correlation == correlation
    @test bf.detection == detection
    @test length(bf) == 3
end

@testitem "BayesFactorTriplet unequal length error" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet

    enrichment = [10.0, 5.0]
    correlation = [8.0, 3.0, 1.0]
    detection = [15.0, 7.0]

    # This should throw an error due to unequal length
    @test_throws Exception BayesFactorTriplet(enrichment, correlation, detection)
end

@testitem "PosteriorProbabilityTriplet construction" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet

    enrichment = [0.8, 0.5, 0.2]
    correlation = [0.7, 0.4, 0.1]
    detection = [0.9, 0.6, 0.15]

    pp = PosteriorProbabilityTriplet(enrichment, correlation, detection)

    @test pp.enrichment == enrichment
    @test pp.correlation == correlation
    @test pp.detection == detection
    @test length(pp.enrichment) == 3
end

@testitem "BayesResult basic construction" begin
    using BayesInteractomics
    using BayesInteractomics: BayesResult, getProteinName, getbfHBM, getbfRegression

    bf_hbm = [5.0 3.0; 10.0 2.0]
    bf_reg = [8.0, 4.0]
    stats_hbm = Dict{Symbol, Union{Vector{Vector{Float64}}, Vector{Float64}, Vector{String}}}(
        :mean_log2FC => [1.0, 2.0]
    )
    name = "TestProtein"

    result = BayesResult(
        bf_hbm,
        bf_reg,
        stats_hbm,
        nothing,
        nothing,
        nothing,
        name
    )

    @test getProteinName(result) == name
    @test getbfHBM(result) == bf_hbm
    @test getbfRegression(result) == bf_reg
end

@testitem "BayesResult with missing models" begin
    using BayesInteractomics
    using BayesInteractomics: BayesResult

    stats_hbm = Dict{Symbol, Union{Vector{Vector{Float64}}, Vector{Float64}, Vector{String}}}(
        :mean_log2FC => [2.0]
    )

    result = BayesResult(
        nothing,          # No HBM Bayes factors
        nothing,          # No regression BF
        stats_hbm,
        nothing,
        nothing,
        nothing,
        "TestProtein"
    )

    @test result.bfHBM === nothing
    @test result.bfRegression === nothing
    @test result.HBM_stats[:mean_log2FC] == [2.0]
end

@testitem "Evidence triplet length and indexing" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, PosteriorProbabilityTriplet

    bf = BayesFactorTriplet([10.0, 5.0, 0.5], [8.0, 3.0, 0.2], [15.0, 7.0, 0.1])
    @test length(bf) == 3

    pp = PosteriorProbabilityTriplet([0.8, 0.5, 0.2], [0.7, 0.4, 0.1], [0.9, 0.6, 0.15])
    # PosteriorProbabilityTriplet doesn't have length defined, check field lengths
    @test length(pp.enrichment) == 3
end

@testitem "InteractionData position tracking" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, getProtocolPositions, getExperimentPositions

    m::Matrix{Union{Missing, Float64}} = [1.0 2.0; 3.0 4.0]
    p = Protocol(1, ["P1"], Dict(1 => m))

    data = InteractionData(
        ["P1"], ["Protein1"],
        Dict(1 => p, 2 => p),
        Dict(1 => p, 2 => p),
        2, Dict(1 => 1, 2 => 1),
        5, 3,
        [2, 4], [3, 5], [1, 2]
    )

    protocol_pos = getProtocolPositions(data)
    exp_pos = getExperimentPositions(data)

    @test isa(protocol_pos, Vector)
    @test isa(exp_pos, Vector)
end
