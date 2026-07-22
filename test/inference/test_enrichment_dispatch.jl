@testitem "Type hierarchy" begin
    using BayesInteractomics

    # HBMResult is the abstract parent
    @test HBMResultSingleProtocol <: HBMResult
    @test HBMResultMultipleProtocols <: HBMResult

    # Both are concrete (not abstract)
    @test isconcretetype(HBMResultSingleProtocol)
    @test isconcretetype(HBMResultMultipleProtocols)
end

@testitem "BayesFactorHBM concrete dispatch - single protocol" begin
    using BayesInteractomics
    using RxInfer
    using Distributions

    # Create mock InferenceResult with :μ_sample and :μ_control
    mock_posterior = RxInfer.InferenceResult(
        Dict(:μ_sample => [Normal(2.0, 1.0)], :μ_control => [Normal(0.0, 1.0)]),
        nothing, nothing, nothing, nothing
    )
    mock_prior = RxInfer.InferenceResult(
        Dict(:μ_sample => [Normal(0.0, 2.0)], :μ_control => [Normal(0.0, 2.0)]),
        nothing, nothing, nothing, nothing
    )

    result_single = HBMResultSingleProtocol(mock_posterior, mock_prior)

    bf, post_prob, prior_prob = BayesFactorHBM(result_single)

    @test bf isa Vector{Float64}
    @test post_prob isa Vector{Float64}
    @test prior_prob isa Vector{Float64}
    @test all(bf .> 0)
    @test length(bf) == 1
end

@testitem "BayesFactorHBM concrete dispatch - multiple protocols" begin
    using BayesInteractomics
    using RxInfer
    using Distributions

    # Create mock with multiple protocol entries
    mock_posterior = RxInfer.InferenceResult(
        Dict(:μ_sample => [Normal(3.0, 0.5), Normal(2.5, 0.8)],
             :μ_control => [Normal(0.0, 1.0), Normal(0.1, 0.9)]),
        nothing, nothing, nothing, nothing
    )
    mock_prior = RxInfer.InferenceResult(
        Dict(:μ_sample => [Normal(0.0, 2.0), Normal(0.0, 2.0)],
             :μ_control => [Normal(0.0, 2.0), Normal(0.0, 2.0)]),
        nothing, nothing, nothing, nothing
    )

    result_multi = HBMResultMultipleProtocols(mock_posterior, mock_prior)

    bf, post_prob, prior_prob = BayesFactorHBM(result_multi)

    @test bf isa Vector{Float64}
    @test post_prob isa Vector{Float64}
    @test prior_prob isa Vector{Float64}
    @test all(bf .> 0)
    @test length(bf) == 2
end

@testitem "BayesFactorHBM concrete vs abstract equivalence" begin
    using BayesInteractomics
    using RxInfer
    using Distributions

    mock_posterior = RxInfer.InferenceResult(
        Dict(:μ_sample => [Normal(2.0, 1.0)], :μ_control => [Normal(0.0, 1.0)]),
        nothing, nothing, nothing, nothing
    )
    mock_prior = RxInfer.InferenceResult(
        Dict(:μ_sample => [Normal(0.0, 2.0)], :μ_control => [Normal(0.0, 2.0)]),
        nothing, nothing, nothing, nothing
    )

    result_single = HBMResultSingleProtocol(mock_posterior, mock_prior)
    result_multi = HBMResultMultipleProtocols(mock_posterior, mock_prior)

    bf_s, pp_s, prp_s = BayesFactorHBM(result_single)
    bf_m, pp_m, prp_m = BayesFactorHBM(result_multi)

    # Both concrete types with identical data should produce identical results
    @test bf_s ≈ bf_m
    @test pp_s ≈ pp_m
    @test prp_s ≈ prp_m
end

@testitem "enrichment wrapper exists" begin
    using BayesInteractomics

    @test isdefined(BayesInteractomics, :enrichment)
    @test isdefined(BayesInteractomics, :precompute_enrichment_prior)

    # Verify they are functions
    @test enrichment isa Function
    @test precompute_enrichment_prior isa Function
end
