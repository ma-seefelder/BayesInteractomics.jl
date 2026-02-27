"""
    test_fixtures.jl

Shared test fixtures and helper functions for BayesInteractomics tests.

This file provides @testsetup modules that generate common test data
used across multiple test files.
"""

using BayesInteractomics
using BayesInteractomics: Protocol, InteractionData, BayesFactorTriplet, PosteriorProbabilityTriplet,
    BayesResult, getNoExperiments, getExperiment, getIDs, getNoProtocols, getControls, getSamples
using Distributions
using Random
using Statistics

@testsetup module StatisticalFixtures
    using BayesInteractomics
    using Distributions
    using Random

    """
        create_enriched_data(enrichment_fc::Float64, bait_index::Int, n_proteins::Int = 5,
                            n_experiments::Int = 3, n_replicates::Int = 3)

    Create synthetic protein abundance data with known enrichment pattern.

    The bait protein (at index bait_index) and a candidate protein (at index 1, unless bait_index=1)
    show enrichment in samples vs controls.
    """
    function create_enriched_data(enrichment_fc::Float64, bait_index::Int;
                                  n_proteins::Int = 5, n_experiments::Int = 3, n_replicates::Int = 3)
        Random.seed!(42)

        # Create control data (baseline)
        control_data = Dict{Int, Matrix{Union{Missing, Float64}}}()
        sample_data = Dict{Int, Matrix{Union{Missing, Float64}}}()

        for exp in 1:n_experiments
            # Controls: all proteins have similar abundance
            control_mat = randn(n_proteins, n_replicates) .+ 8.0  # log2 scale, mean ~8

            # Samples: bait and enriched protein have higher abundance
            sample_mat = copy(control_mat)
            sample_mat[bait_index, :] .+= enrichment_fc  # Bait is enriched

            # Enrich a second protein if not the bait
            enriched_idx = (bait_index == 1) ? 2 : 1
            sample_mat[enriched_idx, :] .+= enrichment_fc / 2

            control_data[exp] = control_mat
            sample_data[exp] = sample_mat
        end

        return control_data, sample_data
    end

    """
        mock_normal_inference_result(means::Vector, sds::Vector)

    Create a mock Normal distribution for testing BF calculations.
    """
    function mock_normal_inference_result(means::Vector, sds::Vector)
        @assert length(means) == length(sds) "means and sds must have same length"
        return [Normal(means[i], sds[i]) for i in eachindex(means)]
    end

    """
        create_synthetic_bayes_factor_triplet(n_proteins::Int)

    Create a synthetic triplet of Bayes factors for testing copula functions.
    """
    function create_synthetic_bayes_factor_triplet(n_proteins::Int = 5)
        Random.seed!(42)

        # Create Bayes factors > 1 (supporting H1) for some proteins
        enrichment_bf = vcat(10.0 .+ rand(2) .* 40, 0.2 .+ rand(3) .* 0.3)
        correlation_bf = vcat(5.0 .+ rand(2) .* 25, 0.1 .+ rand(3) .* 0.4)
        detection_bf = vcat(20.0 .+ rand(2) .* 80, 0.05 .+ rand(3) .* 0.15)

        return BayesFactorTriplet(enrichment_bf, correlation_bf, detection_bf)
    end
end

@testsetup module DataStructureFixtures
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

        samples_dict = Dict{Int, Protocol}()
        controls_dict = Dict{Int, Protocol}()
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
            matched_positions
        )
    end
end
