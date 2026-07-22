"""
    data_structure_fixtures.jl

Hosts the `DataStructureFixtures` @testsetup module.

This setup previously lived at the tail of `test_fixtures.jl` as an `@testsetup module`. In this
TestItemRunner version `@testsetup` modules referenced via `setup=[...]` are not reliably registered
(test/combination/test_h0_sampling.jl errored with "Test setup DataStructureFixtures is not defined"
and ran 0 testitems), whereas `@testmodule` fixtures (e.g. DifferentialFixtures, 86 usages) register
fine. It is therefore declared as an `@testmodule` here — the proven-working pattern — and consumed via
`setup=[DataStructureFixtures]` exactly as before. No fixture logic changed.
"""

@testmodule DataStructureFixtures begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, getNoExperiments, getExperiment, getIDs,
        getNoProtocols, getControls, getSamples, getProtocolPositions, getPositions
    using Random

    """
        create_mock_protocol(n_proteins::Int, n_experiments::Int, n_replicates::Int)

    Create a mock Protocol with random data for testing.
    """
    function create_mock_protocol(n_proteins::Int, n_experiments::Int, n_replicates::Int)
        Random.seed!(42)

        protein_ids = ["P$i" for i in 1:n_proteins]
        data_dict = Dict{Int, Matrix{Union{Missing, Float64}}}()

        for exp in 1:n_experiments
            # Create data matrix: rows=proteins, cols=samples
            data_dict[exp] = randn(n_proteins, n_replicates) .+ 8.0
        end

        return Protocol(n_experiments, protein_ids, data_dict)
    end

    """
        create_mock_interaction_data(n_proteins::Int, n_protocols::Int; n_experiments_per_protocol::Int = 3)

    Create a mock InteractionData with multiple protocols for testing.
    """
    function create_mock_interaction_data(n_proteins::Int, n_protocols::Int; n_experiments_per_protocol::Int = 3)
        Random.seed!(42)

        protein_ids = ["P$i" for i in 1:n_proteins]
        protein_names = ["Protein_$i" for i in 1:n_proteins]

        # Concrete Protocol type params required: the InteractionData ctor is
        # `(...; samples::Dict{I, Protocol{F, I}}, ...)` — an abstract `Dict{Int, Protocol}`
        # eltype does not match the parametric method signature (types.jl:1332).
        samples_dict = Dict{Int, Protocol{Float64, Int64}}()
        controls_dict = Dict{Int, Protocol{Float64, Int64}}()
        no_experiments_dict = Dict{Int, Int}()

        # Create protocols with same structure
        for proto in 1:n_protocols
            samples_dict[proto] = create_mock_protocol(n_proteins, n_experiments_per_protocol, 3)
            controls_dict[proto] = create_mock_protocol(n_proteins, n_experiments_per_protocol, 3)
            no_experiments_dict[proto] = n_experiments_per_protocol
        end

        # Calculate HBM and Regression parameters
        # HBM: 1 (intercept) + n_protocols (protocol means) + n_protocols*n_experiments (experiment means)
        no_parameters_HBM = 1 + n_protocols + n_protocols * n_experiments_per_protocol

        # Regression: 1 (intercept) + n_protocols (slopes)
        no_parameters_Regression = 1 + n_protocols

        # Get position vectors
        protocol_positions, experiment_positions, matched_positions =
            getPositions(no_experiments_dict, no_parameters_HBM)

        return InteractionData(
            protein_ids,
            protein_names,
            samples_dict,
            controls_dict,
            n_protocols,
            no_experiments_dict,
            no_parameters_HBM,
            no_parameters_Regression,
            experiment_positions,
            protocol_positions,
            matched_positions,
            trues(n_proteins)
        )
    end
end
