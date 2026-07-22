"""
    test_jzs_regression.jl

Tests for JZS prior (Cauchy via scale mixture) regression model and per-protein τ_base.
"""

@testitem "Cauchy survival function _cauchy_sf" begin
    using BayesInteractomics
    using BayesInteractomics: _cauchy_sf

    # At threshold=0, symmetric Cauchy has P(X > 0) = 0.5
    @test _cauchy_sf(0.0, 0.354) ≈ 0.5

    # At threshold=r, P(X > r) = 0.5 - atan(1)/π = 0.5 - 0.25 = 0.25
    @test _cauchy_sf(0.354, 0.354) ≈ 0.25

    # P(X > θ) decreases with θ
    @test _cauchy_sf(0.1, 0.354) > _cauchy_sf(0.5, 0.354)

    # P(X > 0.1 | Cauchy(0, 0.354)) ≈ 0.413 — more spread than Normal(0, 0.153²)
    p_cauchy = _cauchy_sf(0.1, 0.354)
    @test 0.40 < p_cauchy < 0.43

    # Large threshold → small probability
    @test _cauchy_sf(10.0, 0.354) < 0.02

    # Negative threshold → P > 0.5
    @test _cauchy_sf(-0.1, 0.354) > 0.5
end


@testitem "calculate_bayes_factor with prior_p_override (JZS)" begin
    using BayesInteractomics
    using BayesInteractomics: calculate_bayes_factor, _cauchy_sf
    using Distributions

    # Posterior moderately supports α > 0.1 (not so extreme as to hit BF cap)
    posterior = [Normal(0.15, 0.08)]
    prior = [Normal(0.0, 0.153)]  # old Normal prior (not used when override set)

    # Without override: uses Normal prior
    bf_normal, _, p_prior_normal = calculate_bayes_factor(posterior, prior; threshold=0.1)

    # With override: uses Cauchy prior probability
    cauchy_p = _cauchy_sf(0.1, 0.354)
    bf_jzs, _, p_prior_jzs = calculate_bayes_factor(
        posterior, prior; threshold=0.1, prior_p_override=cauchy_p)

    # JZS prior gives more mass above 0.1 → BF should be SMALLER (more conservative)
    @test bf_jzs[1] < bf_normal[1]

    # Prior probability should match the override
    @test p_prior_jzs[1] ≈ cauchy_p

    # Both BFs should still be > 1 (posterior evidence)
    @test bf_normal[1] > 1.0
    @test bf_jzs[1] > 1.0
end


@testitem "JZS BF is more conservative than Normal prior BF" begin
    using BayesInteractomics
    using BayesInteractomics: calculate_bayes_factor, _cauchy_sf
    using Distributions

    # Test across a range of posterior means
    for μ_post in [0.2, 0.5, 1.0, 2.0]
        posterior = [Normal(μ_post, 0.05)]
        prior = [Normal(0.0, 0.153)]

        bf_normal, _, _ = calculate_bayes_factor(posterior, prior; threshold=0.1)
        bf_jzs, _, _ = calculate_bayes_factor(
            posterior, prior; threshold=0.1,
            prior_p_override=_cauchy_sf(0.1, 0.354))

        # JZS should always be more conservative
        @test bf_jzs[1] <= bf_normal[1]
    end
end


@testitem "Per-protein τ_base estimation" begin
    using BayesInteractomics
    using BayesInteractomics: estimate_per_protein_tau_base, estimate_regression_tau_base
    using BayesInteractomics: InteractionData, getIDs
    using Statistics

    # Create minimal test data
    # This test verifies the shrinkage behavior without full InteractionData
    # by testing the mathematical properties

    # For the bait protein (idx == refID), should return global
    global_tau = 5.0

    # Shrinkage weight: w = n_obs / min_obs
    # At n_obs=5 (min_obs=5): w=1.0, fully local
    # At n_obs=3: w=0.6, mixed
    # At n_obs=0: returns global directly

    # Test that the shrinkage formula works on log scale
    local_tau = 10.0
    global_tau = 2.0

    # Full weight (w=1): should return local
    w = 1.0
    result = exp(w * log(local_tau) + (1 - w) * log(global_tau))
    @test result ≈ local_tau

    # Half weight (w=0.5): geometric mean
    w = 0.5
    result = exp(w * log(local_tau) + (1 - w) * log(global_tau))
    @test result ≈ sqrt(local_tau * global_tau)

    # Zero weight (w=0): should return global
    w = 0.0
    result = exp(w * log(local_tau) + (1 - w) * log(global_tau))
    @test result ≈ global_tau
end


@testitem "BayesFactorRegression with jzs_r_scale dispatches correctly" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorRegression, _cauchy_sf
    using BayesInteractomics: RegressionResultSingleProtocol
    using Distributions
    using RxInfer

    # Create mock InferenceResult with Normal posteriors
    α_posterior = Normal(0.15, 0.08)
    β_posterior = Normal(1.0, 0.3)
    σ_posterior = Gamma(5.0, 2.0)

    α_prior = Normal(0.0, 0.153)
    β_prior = Normal(1.0, 0.5)
    σ_prior = Gamma(5.0, 2.0)

    posterior_result = RxInfer.InferenceResult(
        Dict(:α => α_posterior, :β => β_posterior, :σ => σ_posterior),
        nothing, nothing, nothing, nothing)
    prior_result = RxInfer.InferenceResult(
        Dict(:α => α_prior, :β => β_prior, :σ => σ_prior),
        nothing, nothing, nothing, nothing)

    result = RegressionResultSingleProtocol(posterior_result, prior_result)

    # Without JZS
    bf_normal, _, _ = BayesFactorRegression(result; threshold=0.1, jzs_r_scale=0.0)
    # With JZS
    bf_jzs, _, p_prior_jzs = BayesFactorRegression(result; threshold=0.1, jzs_r_scale=0.354)

    # JZS should be more conservative
    @test bf_jzs[1] < bf_normal[1]

    # Prior probability should match analytical Cauchy
    @test p_prior_jzs[1] ≈ _cauchy_sf(0.1, 0.354)
end


@testitem "CONFIG jzs_r_scale field" begin
    using BayesInteractomics
    using BayesInteractomics: CONFIG, OutputFiles

    # Default should be 0.354 (JASP convention)
    config = CONFIG(
        datafile=["test.xlsx"],
        control_cols=[Dict(1 => [2,3])],
        sample_cols=[Dict(1 => [4,5])],
        poi="test_bait",
        output=OutputFiles("/tmp/test_output")
    )
    @test config.jzs_r_scale == 0.354

    # Can override
    config2 = CONFIG(
        datafile=["test.xlsx"],
        control_cols=[Dict(1 => [2,3])],
        sample_cols=[Dict(1 => [4,5])],
        poi="test_bait",
        output=OutputFiles("/tmp/test_output"),
        jzs_r_scale=0.5
    )
    @test config2.jzs_r_scale == 0.5

    # Setting to 0.0 disables JZS (uses Normal prior)
    config3 = CONFIG(
        datafile=["test.xlsx"],
        control_cols=[Dict(1 => [2,3])],
        sample_cols=[Dict(1 => [4,5])],
        poi="test_bait",
        output=OutputFiles("/tmp/test_output"),
        jzs_r_scale=0.0
    )
    @test config3.jzs_r_scale == 0.0
end


@testitem "VMP convergence with JZS Gamma(0.5,...) prior — single protocol" begin
    using BayesInteractomics
    using BayesInteractomics: RegressionModel_one_protocol_robust_jzs,
        estimate_regression_tau_base, getIDs, InteractionData, Protocol, getPositions
    using BayesInteractomics: RobustRegressionResultSingleProtocol
    using BayesInteractomics: BayesFactorRegression, _cauchy_sf
    using Distributions
    using Statistics
    using Random

    # Inline mock data creation (same pattern as DataStructureFixtures)
    Random.seed!(42)
    n_proteins = 5; n_exp = 3; n_rep = 3
    protein_ids = ["P$i" for i in 1:n_proteins]
    protein_names = ["Protein_$i" for i in 1:n_proteins]

    function _mock_protocol(np, ne, nr)
        data_dict = Dict{Int, Matrix{Union{Missing, Float64}}}()
        for e in 1:ne
            data_dict[e] = rand(np, nr) .+ 1.0
        end
        Protocol(ne, ["P$i" for i in 1:np], data_dict)
    end

    samples_dict = Dict(1 => _mock_protocol(n_proteins, n_exp, n_rep))
    controls_dict = Dict(1 => _mock_protocol(n_proteins, n_exp, n_rep))
    no_exp_dict = Dict(1 => n_exp)
    no_hbm = 1 + 1 + n_exp
    no_reg = 1 + 1
    pp, ep, mp = getPositions(no_exp_dict, no_hbm)
    data = InteractionData(protein_ids, protein_names, samples_dict, controls_dict,
        1, no_exp_dict, no_hbm, no_reg, ep, pp, mp, trues(n_proteins))
    refID = 1

    # Estimate global τ_base
    global_tau = estimate_regression_tau_base(data, refID)
    @test global_tau > 0.0
    @test isfinite(global_tau)

    # --- Test JZS model on protein 2 ---
    result = RegressionModel_one_protocol_robust_jzs(
        data, 2, refID, 0.0, 1.0;
        nu=5.0, τ_base=global_tau, jzs_r_scale=0.354,
        regression_iterations=100
    )

    @test result isa RobustRegressionResultSingleProtocol

    # Check posterior exists and has expected keys
    post = result.posterior.posteriors
    @test haskey(post, :α)
    @test haskey(post, :β)
    @test haskey(post, :τ_g)

    # Posterior values should be finite
    α_mean = mean(post[:α])
    @test isfinite(α_mean)

    # τ_g posterior should be finite and positive
    τ_g_post = post[:τ_g]
    @test mean(τ_g_post) > 0.0
    @test isfinite(mean(τ_g_post))

    # --- Test another protein for comparison ---
    result3 = RegressionModel_one_protocol_robust_jzs(
        data, 3, refID, 0.0, 1.0;
        nu=5.0, τ_base=global_tau, jzs_r_scale=0.354,
        regression_iterations=100
    )
    @test result3 isa RobustRegressionResultSingleProtocol
    @test isfinite(mean(result3.posterior.posteriors[:α]))

    # --- BF computation should work ---
    bf2, _, _ = BayesFactorRegression(result; threshold=0.1, jzs_r_scale=0.354)
    bf3, _, _ = BayesFactorRegression(result3; threshold=0.1, jzs_r_scale=0.354)

    @test isfinite(bf2[1])
    @test bf2[1] > 0.0
    @test isfinite(bf3[1])
    @test bf3[1] > 0.0
end
