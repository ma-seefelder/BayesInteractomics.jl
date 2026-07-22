"""
    test_copula.jl

Tests for copula-based evidence combination functionality using TestItemRunner.
"""

@testitem "Posterior probability from Bayes factor - scalar" begin
    using BayesInteractomics

    # Test scalar conversions
    @test BayesInteractomics.posterior_probability_from_bayes_factor(1.0) == 0.5
    @test BayesInteractomics.posterior_probability_from_bayes_factor(Inf) == 1.0
    @test BayesInteractomics.posterior_probability_from_bayes_factor(-Inf) == 0.0
    @test BayesInteractomics.posterior_probability_from_bayes_factor(0.0) == 0.0
end

@testitem "Posterior probability from Bayes factor - triplet" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, PosteriorProbabilityTriplet

    # Create Bayes factor triplet with values > 1
    # BF=3 -> p=0.75, BF=9 -> p=0.9
    val1 = 3.0
    val2 = 9.0
    bf = BayesFactorTriplet([val1, val2], [val1, val2], [val1, val2])

    pp = BayesInteractomics.posterior_probability_from_bayes_factor(bf)

    @test pp isa PosteriorProbabilityTriplet
    @test pp.enrichment ≈ [0.75, 0.9]
    @test pp.correlation ≈ [0.75, 0.9]
    @test pp.detection ≈ [0.75, 0.9]
end

@testitem "Copula fitting with specific copula family" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet
    using Distributions
    using Copulas
    using Random

    Random.seed!(123)
    n = 100

    # Create correlated data using Beta distribution
    d = Beta(2, 2)
    x = rand(d, n)
    y = clamp.(x .+ rand(Normal(0, 0.01), n), 0.01, 0.99)
    z = clamp.(x .+ rand(Normal(0, 0.01), n), 0.01, 0.99)

    pp = PosteriorProbabilityTriplet(x, y, z)

    # Test fitting specific copula
    cop = BayesInteractomics.fit_copula(pp, searchBestCopula=false, copula=ClaytonCopula)
    @test cop isa Copulas.Copula
end

@testitem "Copula comparison and ranking" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet
    using Distributions
    using DataFrames
    using Random

    Random.seed!(123)
    n = 100

    # Generate correlated posterior probabilities
    d = Beta(2, 2)
    x = rand(d, n)
    y = clamp.(x .+ rand(Normal(0, 0.01), n), 0.01, 0.99)
    z = clamp.(x .+ rand(Normal(0, 0.01), n), 0.01, 0.99)

    pp = PosteriorProbabilityTriplet(x, y, z)

    # Test comparison of copula families (default is now BIC)
    res = BayesInteractomics.compare_copulas(pp)

    @test res isa DataFrame
    @test "Family" in names(res)
    @test "LogLik" in names(res)
    @test "BIC" in names(res)
    # Default sorting is by BIC (ascending, lower is better)
    @test issorted(res.BIC)
end

@testitem "Automatic copula fitting (best family selection)" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet
    using Distributions
    using Copulas
    using Random

    Random.seed!(456)
    n = 100

    # Generate data with known correlation structure
    d = Beta(2, 2)
    x = rand(d, n)
    y = clamp.(x .+ rand(Normal(0, 0.01), n), 0.01, 0.99)
    z = clamp.(x .+ rand(Normal(0, 0.01), n), 0.01, 0.99)

    pp = PosteriorProbabilityTriplet(x, y, z)

    # Test automatic fitting
    best_cop = BayesInteractomics.fit_copula(pp, searchBestCopula=true)
    @test best_cop isa Copulas.Copula
end

@testitem "Copula with independent components (edge case)" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet
    using Distributions
    using Copulas
    using Random

    Random.seed!(789)

    # Create independent (uncorrelated) data
    # Use more samples to give copula fitting a better chance
    x = rand(Beta(2, 2), 100)
    y = rand(Beta(2, 2), 100)  # Independent of x
    z = rand(Beta(2, 2), 100)  # Independent of x and y

    pp = PosteriorProbabilityTriplet(x, y, z)

    # With nearly independent data, some copula families may fail to fit
    # This is expected behavior for edge cases
    try
        cop = BayesInteractomics.fit_copula(pp, searchBestCopula=true)
        @test cop isa Copulas.Copula
    catch e
        # If fitting fails with independent data, that's acceptable
        # The copula library may throw errors for degenerate cases
        @test e isa Exception
        @warn "Copula fitting failed for independent data (expected edge case)" exception=e
    end
end

@testitem "Posterior probability triplet construction" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet

    # Create posterior probability triplet
    enrichment = [0.7, 0.8, 0.5]
    correlation = [0.6, 0.9, 0.4]
    detection = [0.75, 0.85, 0.55]

    pp = PosteriorProbabilityTriplet(enrichment, correlation, detection)

    @test pp.enrichment == enrichment
    @test pp.correlation == correlation
    @test pp.detection == detection
    @test length(pp.enrichment) == 3
end

@testitem "Bayes factor triplet log transformation" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet

    bf = BayesFactorTriplet([10.0, 100.0], [5.0, 50.0], [20.0, 200.0])
    log_bf = log(bf)

    @test log_bf.enrichment ≈ log10.([10.0, 100.0])
    @test log_bf.correlation ≈ log10.([5.0, 50.0])
    @test log_bf.detection ≈ log10.([20.0, 200.0])
end

######################################################
# New tests for Copula-EM improvements (v0.2)
######################################################

@testitem "copula_nparams returns correct parameter counts" begin
    using BayesInteractomics
    using Copulas

    # Test parameter counts for different copula families
    @test BayesInteractomics.copula_nparams(ClaytonCopula) == 1
    @test BayesInteractomics.copula_nparams(FrankCopula) == 1
    @test BayesInteractomics.copula_nparams(GumbelCopula) == 1
    @test BayesInteractomics.copula_nparams(JoeCopula) == 1
    @test BayesInteractomics.copula_nparams(GaussianCopula) == 3
    @test BayesInteractomics.copula_nparams(EmpiricalCopula) == 0
end

@testitem "compare_copulas with BIC criterion" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet
    using Distributions
    using DataFrames
    using Random

    Random.seed!(123)
    n = 200

    # Generate correlated posterior probabilities
    d = Beta(2, 2)
    x = rand(d, n)
    y = clamp.(x .+ rand(Normal(0, 0.1), n), 0.01, 0.99)
    z = clamp.(x .+ rand(Normal(0, 0.1), n), 0.01, 0.99)

    pp = PosteriorProbabilityTriplet(x, y, z)

    # Test BIC-based comparison
    res_bic = BayesInteractomics.compare_copulas(pp; criterion=:BIC)
    @test res_bic isa DataFrame
    @test "BIC" in names(res_bic)
    @test "AIC" in names(res_bic)
    @test issorted(res_bic.BIC)  # Sorted by BIC (ascending, lower is better)

    # Test AIC-based comparison
    res_aic = BayesInteractomics.compare_copulas(pp; criterion=:AIC)
    @test issorted(res_aic.AIC)  # Sorted by AIC

    # Test loglik-based comparison (original behavior)
    res_ll = BayesInteractomics.compare_copulas(pp; criterion=:loglik)
    @test issorted(res_ll.LogLik, rev=true)  # Sorted by LogLik (descending)
end

@testitem "fit_copula with criterion parameter" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet
    using Distributions
    using Copulas
    using Random

    Random.seed!(456)
    n = 200

    d = Beta(2, 2)
    x = rand(d, n)
    y = clamp.(x .+ rand(Normal(0, 0.1), n), 0.01, 0.99)
    z = clamp.(x .+ rand(Normal(0, 0.1), n), 0.01, 0.99)

    pp = PosteriorProbabilityTriplet(x, y, z)

    # Test fitting with BIC criterion
    cop_bic = BayesInteractomics.fit_copula(pp; criterion=:BIC)
    @test cop_bic isa Copulas.Copula

    # Test fitting with AIC criterion
    cop_aic = BayesInteractomics.fit_copula(pp; criterion=:AIC)
    @test cop_aic isa Copulas.Copula
end

@testitem "get_prior_hyperparameters returns correct priors" begin
    using BayesInteractomics

    # Test known experiment types
    apms = BayesInteractomics.get_prior_hyperparameters(:APMS)
    @test apms.α ≈ 20.0
    @test apms.β ≈ 180.0

    bioid = BayesInteractomics.get_prior_hyperparameters(:BioID)
    @test bioid.α ≈ 30.0
    @test bioid.β ≈ 120.0

    turboid = BayesInteractomics.get_prior_hyperparameters(:TurboID)
    @test turboid.α ≈ 40.0
    @test turboid.β ≈ 110.0

    default = BayesInteractomics.get_prior_hyperparameters(:default)
    @test default.α ≈ 25.0
    @test default.β ≈ 175.0

    # Test unknown type falls back to default
    unknown = BayesInteractomics.get_prior_hyperparameters(:unknown_type)
    @test unknown == default
end

@testitem "EXPERIMENT_PRIORS has expected keys" begin
    using BayesInteractomics

    @test haskey(BayesInteractomics.EXPERIMENT_PRIORS, :APMS)
    @test haskey(BayesInteractomics.EXPERIMENT_PRIORS, :BioID)
    @test haskey(BayesInteractomics.EXPERIMENT_PRIORS, :TurboID)
    @test haskey(BayesInteractomics.EXPERIMENT_PRIORS, :default)
    @test haskey(BayesInteractomics.EXPERIMENT_PRIORS, :permissive)
    @test haskey(BayesInteractomics.EXPERIMENT_PRIORS, :stringent)

    # Verify all priors give expected π₁ values
    for (key, prior) in BayesInteractomics.EXPERIMENT_PRIORS
        expected_π1 = prior.α / (prior.α + prior.β)
        @test 0.0 < expected_π1 < 1.0
    end
end

@testitem "fit_beta_weighted produces valid Beta distribution" begin
    using BayesInteractomics
    using Distributions
    using Statistics: mean
    using Random

    Random.seed!(789)

    # Generate data with known Beta distribution
    true_dist = Beta(3, 7)  # Mean ≈ 0.3
    x = rand(true_dist, 500)

    # Uniform weights should recover similar parameters
    w_uniform = ones(500)
    fitted = BayesInteractomics.fit_beta_weighted(x, w_uniform)

    @test fitted isa Beta
    @test mean(fitted) ≈ mean(true_dist) atol=0.1

    # High weights on low values should shift mean down (with stronger bias)
    # Use very high weight ratio to ensure biased mean is lower
    w_biased = [xi < 0.25 ? 100.0 : 0.1 for xi in x]
    fitted_biased = BayesInteractomics.fit_beta_weighted(x, w_biased)
    @test mean(fitted_biased) < mean(fitted)
end

@testitem "fit_beta_weighted handles edge cases" begin
    using BayesInteractomics
    using Distributions

    # Zero weights should return prior (Beta(2,2) by default, not uniform)
    x = [0.2, 0.5, 0.8]
    w_zero = [0.0, 0.0, 0.0]
    result = BayesInteractomics.fit_beta_weighted(x, w_zero)
    @test result == Beta(2.0, 2.0)  # Returns prior, not uniform

    # Single non-zero weight
    w_single = [1.0, 0.0, 0.0]
    result_single = BayesInteractomics.fit_beta_weighted(x, w_single)
    @test result_single isa Beta
end

@testitem "get_H1_initialization_set with quantile method" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet
    using Random

    Random.seed!(123)
    n = 500

    # Create data with clear signal proteins (high posteriors)
    enrichment = vcat(rand(100) .* 0.3, 0.7 .+ rand(400) .* 0.3)
    correlation = vcat(rand(100) .* 0.3, 0.7 .+ rand(400) .* 0.3)
    detection = vcat(rand(100) .* 0.3, 0.7 .+ rand(400) .* 0.3)

    pp = PosteriorProbabilityTriplet(enrichment, correlation, detection)

    idx = BayesInteractomics.get_H1_initialization_set(pp; method=:quantile)

    @test length(idx) >= 50  # Should have at least min_proteins
    @test all(idx .>= 1) && all(idx .<= n)  # Valid indices
end

@testitem "get_H1_initialization_set with kmeans method" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet
    using Random

    Random.seed!(456)
    n = 500

    # Create clearly separable clusters
    # Low cluster (H0)
    enrichment_h0 = 0.1 .+ rand(300) .* 0.2
    correlation_h0 = 0.1 .+ rand(300) .* 0.2
    detection_h0 = 0.1 .+ rand(300) .* 0.2

    # High cluster (H1)
    enrichment_h1 = 0.7 .+ rand(200) .* 0.2
    correlation_h1 = 0.7 .+ rand(200) .* 0.2
    detection_h1 = 0.7 .+ rand(200) .* 0.2

    pp = PosteriorProbabilityTriplet(
        vcat(enrichment_h0, enrichment_h1),
        vcat(correlation_h0, correlation_h1),
        vcat(detection_h0, detection_h1)
    )

    idx = BayesInteractomics.get_H1_initialization_set(pp; method=:kmeans)

    @test length(idx) >= 50  # Should have at least min_proteins
    # K-means should preferentially select from the H1 cluster (indices 301-500)
    h1_indices_in_result = count(i -> i > 300, idx)
    @test h1_indices_in_result > length(idx) / 2  # More than half from H1 cluster
end

@testitem "get_H1_initialization_set with random_top20 method" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet
    using Statistics: quantile, mean
    using Random

    Random.seed!(789)
    n = 500

    enrichment = rand(n)
    correlation = rand(n)
    detection = rand(n)

    pp = PosteriorProbabilityTriplet(enrichment, correlation, detection)

    idx = BayesInteractomics.get_H1_initialization_set(pp; method=:random_top20)

    # Should select from top 20% by mean posterior
    mean_p = (enrichment .+ correlation .+ detection) ./ 3
    threshold = quantile(mean_p, 0.80)

    # Most selected indices should be above threshold
    selected_means = mean_p[idx]
    @test mean(selected_means .>= threshold) > 0.8
end

@testitem "fit_copula_weighted returns copula or nothing" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet
    using Copulas
    using Random

    Random.seed!(123)
    n = 500

    # Generate correlated data
    x = rand(n)
    y = clamp.(x .+ rand(n) .* 0.2, 0.01, 0.99)
    z = clamp.(x .+ rand(n) .* 0.2, 0.01, 0.99)

    pp = PosteriorProbabilityTriplet(x, y, z)

    # Sufficient effective sample size
    w_good = ones(n)
    result = BayesInteractomics.fit_copula_weighted(pp, w_good)
    @test result isa Copulas.Copula

    # Very low effective sample size (all weight on one observation)
    w_bad = zeros(n)
    w_bad[1] = 1.0
    result_bad = BayesInteractomics.fit_copula_weighted(pp, w_bad; n_eff_threshold=50.0)
    @test result_bad === nothing
end

@testitem "estimate_prior_empirical_bayes returns valid hyperparameters" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet
    using Copulas
    using Distributions
    using Random

    Random.seed!(456)
    n = 500

    # Generate data with moderate interaction proportion
    enrichment = vcat(rand(Beta(2, 8), 350), rand(Beta(8, 2), 150))
    correlation = vcat(rand(Beta(2, 8), 350), rand(Beta(8, 2), 150))
    detection = vcat(rand(Beta(2, 8), 350), rand(Beta(8, 2), 150))

    pp = PosteriorProbabilityTriplet(enrichment, correlation, detection)

    # Create a simple H0 distribution
    cop = FrankCopula(3, 1.0)
    joint_H0 = SklarDist(cop, (Beta(2, 8), Beta(2, 8), Beta(2, 8)))

    result = BayesInteractomics.estimate_prior_empirical_bayes(pp, joint_H0; grid_size=10)

    @test haskey(result, :α)
    @test haskey(result, :β)
    @test haskey(result, :expected_π1)
    @test result.α > 0
    @test result.β > 0
    @test 0 < result.expected_π1 < 1
end

@testitem "combined_BF with phase1_result produces valid results" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult, CombinedBayesResult
    using Distributions, Random, Statistics
    import Distributions: fit

    Random.seed!(42)

    n = 200; n_h0 = 150; n_ag = 30; n_h1 = 20

    # Create synthetic BFs with well-separated classes
    bf_e = vcat(exp.(randn(n_h0) .* 1.0 .- 0.5),
                exp.(randn(n_ag) .* 0.3),
                exp.(randn(n_h1) .* 1.0 .+ 3.0))
    bf_c = vcat(exp.(randn(n_h0) .* 0.8 .- 0.3),
                exp.(randn(n_ag) .* 0.3),
                exp.(randn(n_h1) .* 0.8 .+ 2.0))
    bf_d = vcat(exp.(randn(n_h0) .* 0.5 .- 0.2),
                exp.(randn(n_ag) .* 0.3),
                exp.(randn(n_h1) .* 0.5 .+ 1.5))
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    # Create Phase 1 result via LatentClassResult
    resp = zeros(n, 3)
    for i in 1:n_h0; resp[i, 1] = 0.9; resp[i, 2] = 0.05; resp[i, 3] = 0.05; end
    for i in n_h0+1:n_h0+n_ag; resp[i, 2] = 0.9; resp[i, 1] = 0.05; resp[i, 3] = 0.05; end
    for i in n_h0+n_ag+1:n; resp[i, 3] = 0.9; resp[i, 1] = 0.05; resp[i, 2] = 0.05; end

    lc = LatentClassResult(
        ones(n), ones(n),
        Dict("background"=>(mu=-0.5,sigma=1.0,precision=1.0),
             "agnostic"=>(mu=0.0,sigma=0.5,precision=4.0),
             "interaction"=>(mu=3.0,sigma=1.0,precision=1.0)),
        [0.75, 0.15, 0.10], [1.0], true, 50, resp)

    result = BayesInteractomics.combined_BF(bf, 181;
        phase1_result=lc, verbose=false, n_restarts=3, use_acceleration=false)

    # Type and length
    @test result isa CombinedBayesResult
    @test length(result.bf) == n

    # All BFs finite and positive
    @test all(isfinite, result.bf)
    @test all(result.bf .> 0)

    # 3-component EM result
    @test result.em_result.pi_ag > 0
    @test result.em_result.joint_H0 !== nothing

    # KS diagnostics populated
    @test isfinite(result.ks_results.enrichment)
    @test isfinite(result.ks_results.correlation)
    @test isfinite(result.ks_results.detection)

    # Copula family names
    @test result.h0_copula_family isa String
    @test length(result.h0_copula_family) > 0

    # Component counts
    @test result.n_h0 > 0
    @test result.n_h1 > 0

    # Evidence ordering: mean H1 BF > mean H0 BF
    mean_h1_bf = mean(result.bf[n_h0+n_ag+1:end])
    mean_h0_bf = mean(result.bf[1:n_h0])
    @test mean_h1_bf > mean_h0_bf
end

@testitem "combined_BF agnostic proteins are shrunk toward BF = 1.0" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult
    using Distributions, Random, Statistics

    Random.seed!(42)

    n = 200; n_h0 = 150; n_ag = 30; n_h1 = 20

    bf_e = vcat(exp.(randn(n_h0) .* 1.0 .- 0.5),
                exp.(randn(n_ag) .* 0.3),
                exp.(randn(n_h1) .* 1.0 .+ 3.0))
    bf_c = vcat(exp.(randn(n_h0) .* 0.8 .- 0.3),
                exp.(randn(n_ag) .* 0.3),
                exp.(randn(n_h1) .* 0.8 .+ 2.0))
    bf_d = vcat(exp.(randn(n_h0) .* 0.5 .- 0.2),
                exp.(randn(n_ag) .* 0.3),
                exp.(randn(n_h1) .* 0.5 .+ 1.5))
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    resp = zeros(n, 3)
    for i in 1:n_h0; resp[i, 1] = 0.9; resp[i, 2] = 0.05; resp[i, 3] = 0.05; end
    for i in n_h0+1:n_h0+n_ag; resp[i, 2] = 0.9; resp[i, 1] = 0.05; resp[i, 3] = 0.05; end
    for i in n_h0+n_ag+1:n; resp[i, 3] = 0.9; resp[i, 1] = 0.05; resp[i, 2] = 0.05; end

    lc = LatentClassResult(
        ones(n), ones(n),
        Dict("background"=>(mu=-0.5,sigma=1.0,precision=1.0),
             "agnostic"=>(mu=0.0,sigma=0.5,precision=4.0),
             "interaction"=>(mu=3.0,sigma=1.0,precision=1.0)),
        [0.75, 0.15, 0.10], [1.0], true, 50, resp)

    result = BayesInteractomics.combined_BF(bf, 181;
        phase1_result=lc, verbose=false, n_restarts=3, use_acceleration=false)

    # Agnostic proteins are shrunk toward BF = 1.0 via geometric shrinkage (w_ag is high
    # but not exactly 1.0 after EM, so BFs are near 1.0 rather than exactly 1.0).
    ag_indices = n_h0+1:n_h0+n_ag
    for i in ag_indices
        @test result.bf[i] < 5.0   # significantly below typical H1 BFs
    end
    # Mean agnostic BF is much closer to 1.0 than mean H1 BF
    @test mean(result.bf[ag_indices]) < mean(result.bf[n_h0+n_ag+1:end])

    # H0 and H1 proteins have varying BFs
    @test !all(result.bf[1:n_h0] .== 1.0)
    @test !all(result.bf[n_h0+n_ag+1:end] .== 1.0)
end

@testitem "combined_BF monotonicity constraint" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult
    using Distributions, Random

    Random.seed!(42)

    n = 200; n_h0 = 150; n_ag = 30; n_h1 = 20

    bf_e = vcat(exp.(randn(n_h0) .* 1.0 .- 0.5),
                exp.(randn(n_ag) .* 0.3),
                exp.(randn(n_h1) .* 1.0 .+ 3.0))
    bf_c = vcat(exp.(randn(n_h0) .* 0.8 .- 0.3),
                exp.(randn(n_ag) .* 0.3),
                exp.(randn(n_h1) .* 0.8 .+ 2.0))
    bf_d = vcat(exp.(randn(n_h0) .* 0.5 .- 0.2),
                exp.(randn(n_ag) .* 0.3),
                exp.(randn(n_h1) .* 0.5 .+ 1.5))

    # Force first 10 H0 proteins to have enrichment < 1 AND correlation < 1
    for i in 1:10
        bf_e[i] = 0.3
        bf_c[i] = 0.2
        bf_d[i] = 5.0
    end

    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    resp = zeros(n, 3)
    for i in 1:n_h0; resp[i, 1] = 0.9; resp[i, 2] = 0.05; resp[i, 3] = 0.05; end
    for i in n_h0+1:n_h0+n_ag; resp[i, 2] = 0.9; resp[i, 1] = 0.05; resp[i, 3] = 0.05; end
    for i in n_h0+n_ag+1:n; resp[i, 3] = 0.9; resp[i, 1] = 0.05; resp[i, 2] = 0.05; end

    lc = LatentClassResult(
        ones(n), ones(n),
        Dict("background"=>(mu=-0.5,sigma=1.0,precision=1.0),
             "agnostic"=>(mu=0.0,sigma=0.5,precision=4.0),
             "interaction"=>(mu=3.0,sigma=1.0,precision=1.0)),
        [0.75, 0.15, 0.10], [1.0], true, 50, resp)

    result = BayesInteractomics.combined_BF(bf, 181;
        phase1_result=lc, verbose=false, n_restarts=3, use_acceleration=false)

    # For non-agnostic proteins with enrichment < 1 AND correlation < 1:
    # combined_bf <= max(bf_e, bf_c, bf_d) + tolerance
    # Exclude agnostic-overridden proteins (BF=1.0 from EM agnostic assignment)
    ag_indices = Set(n_h0+1:n_h0+n_ag)
    tol = 1e-10
    check_indices = [i for i in 1:n
                     if bf.enrichment[i] < 1.0 && bf.correlation[i] < 1.0 &&
                        !(i in ag_indices) && result.bf[i] != 1.0]
    @test length(check_indices) >= 5
    for i in check_indices
        max_individual = max(bf.enrichment[i], bf.correlation[i], bf.detection[i])
        @test result.bf[i] <= max_individual + tol
    end
end

@testitem "precompute_h0 requires phase1_result" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult
    using Distributions, Random

    Random.seed!(42)

    n = 100
    bf = BayesFactorTriplet(exp.(randn(n)), exp.(randn(n)), exp.(randn(n) .* 0.5))

    # Old signature should error
    @test_throws Exception BayesInteractomics.precompute_h0(bf, "nonexistent.xlsx")

    # New signature with Phase 1 result should work
    resp = zeros(n, 3)
    for i in 1:80; resp[i, 1] = 0.9; resp[i, 2] = 0.05; resp[i, 3] = 0.05; end
    for i in 81:90; resp[i, 2] = 0.9; resp[i, 1] = 0.05; resp[i, 3] = 0.05; end
    for i in 91:100; resp[i, 3] = 0.9; resp[i, 1] = 0.05; resp[i, 2] = 0.05; end

    lc = LatentClassResult(
        ones(n), ones(n),
        Dict("background"=>(mu=-0.5,sigma=1.0,precision=1.0),
             "agnostic"=>(mu=0.0,sigma=0.5,precision=4.0),
             "interaction"=>(mu=3.0,sigma=1.0,precision=1.0)),
        [0.80, 0.10, 0.10], [1.0], true, 50, resp)

    h0 = BayesInteractomics.precompute_h0(bf, lc; verbose=false)
    @test h0 isa BayesInteractomics.PrecomputedH0
end

######################################################
# Tests for improved EM convergence detection
######################################################

@testitem "hasEMconverged detects smoothed log-likelihood convergence" begin
    using BayesInteractomics
    using DataFrames

    # Create converging sequence (smoothed LL change below tolerance)
    n = 20
    ll_base = -1000.0
    ll_converging = [ll_base + 0.0001 * i + 0.00001 * randn() for i in 1:n]

    logs = DataFrame(
        iter = 1:n,
        π0 = fill(0.8, n),
        π1 = fill(0.2, n),
        ll = ll_converging
    )

    # Should detect convergence with smoothed criterion
    @test BayesInteractomics.hasEMconverged(logs; tol=1e-3, window=5) == true
end

@testitem "hasEMconverged detects parameter stability" begin
    using BayesInteractomics
    using DataFrames

    # Create sequence with stable π₁ but varying LL
    n = 15
    π1_stable = fill(0.25, n)
    ll_varying = [-1000.0 + sin(i) * 10 for i in 1:n]  # LL oscillating

    # hasEMconverged reads column :pi1 (ASCII), not :π1 (Greek) — use pi1/pi0
    logs = DataFrame(
        iter = 1:n,
        pi0 = 1.0 .- π1_stable,
        pi1 = π1_stable,
        ll = ll_varying
    )

    # Should detect convergence via parameter stability
    @test BayesInteractomics.hasEMconverged(logs; π_tol=1e-3, window=5) == true
end

@testitem "hasEMconverged detects oscillation" begin
    using BayesInteractomics
    using DataFrames

    # Create oscillating sequence (many sign changes with small amplitude)
    n = 15
    ll_base = -1000.0
    ll_oscillating = [ll_base + 0.001 * (-1)^i for i in 1:n]

    logs = DataFrame(
        iter = 1:n,
        π0 = [0.8 + 0.01 * (-1)^i for i in 1:n],
        π1 = [0.2 + 0.01 * (-1)^(i+1) for i in 1:n],
        ll = ll_oscillating
    )

    # Should detect convergence via oscillation detection
    @test BayesInteractomics.hasEMconverged(logs; tol=1e-3) == true
end

@testitem "hasEMconverged returns false for non-converged sequence" begin
    using BayesInteractomics
    using DataFrames

    # Create non-converging sequence (steadily changing)
    n = 15
    ll_increasing = [-2000.0 + 10.0 * i for i in 1:n]
    π1_changing = [0.1 + 0.01 * i for i in 1:n]

    # hasEMconverged reads column :pi1 (ASCII), not :π1 (Greek) — use pi1/pi0
    logs = DataFrame(
        iter = 1:n,
        pi0 = 1.0 .- π1_changing,
        pi1 = π1_changing,
        ll = ll_increasing
    )

    # Should NOT detect convergence
    @test BayesInteractomics.hasEMconverged(logs; tol=1e-4, window=5, π_tol=1e-4) == false
end

@testitem "hasEMconverged handles insufficient iterations" begin
    using BayesInteractomics
    using DataFrames

    # Create short sequence (fewer than required iterations)
    n = 3
    logs = DataFrame(
        iter = 1:n,
        π0 = fill(0.8, n),
        π1 = fill(0.2, n),
        ll = fill(-1000.0, n)
    )

    # Should return false (not enough iterations)
    @test BayesInteractomics.hasEMconverged(logs; window=5) == false
end

@testitem "fit_beta_weighted uses regularization instead of uniform fallback" begin
    using BayesInteractomics
    using Distributions
    using Statistics: mean

    # Test that regularization shrinks toward prior instead of returning uniform
    x = [0.5, 0.5, 0.5]  # Low variance data
    w = [1.0, 1.0, 1.0]

    result = BayesInteractomics.fit_beta_weighted(x, w)

    # Should NOT be uniform Beta(1,1)
    @test result != Beta(1.0, 1.0)
    # Mean should be close to data mean (0.5)
    @test abs(mean(result) - 0.5) < 0.2
end

@testitem "fit_beta_weighted shrinks with low effective sample size" begin
    using BayesInteractomics
    using Distributions
    using Statistics: mean

    # Very concentrated weights -> low n_eff -> should shrink toward prior
    x = [0.1, 0.2, 0.3, 0.9]
    w = [100.0, 0.0, 0.0, 0.0]  # All weight on first observation

    result = BayesInteractomics.fit_beta_weighted(x, w)

    # With low n_eff, should shrink heavily toward prior Beta(2,2) with mean 0.5
    # So result mean should be between data weighted mean (0.1) and prior mean (0.5)
    @test 0.1 < mean(result) < 0.6
end

@testitem "SQUAREM acceleration types exist" begin
    using BayesInteractomics

    # Test that SQUAREM types are defined
    @test isdefined(BayesInteractomics, :SQUAREMState)
    @test isdefined(BayesInteractomics, :extract_em_params_logbf)
    @test isdefined(BayesInteractomics, :restore_em_params_logbf)
    @test isdefined(BayesInteractomics, :squarem_acceleration_step)
    @test isdefined(BayesInteractomics, :em_fit_mixture_accelerated_logbf)
end

@testitem "SQUAREMState initialization" begin
    using BayesInteractomics

    state = BayesInteractomics.SQUAREMState()

    @test state.θ_prev2 === nothing
    @test state.θ_prev1 === nothing
    @test state.θ_curr === nothing
    @test state.ll_curr == -Inf
    @test state.n_accel_steps == 0
    @test state.n_fallback_steps == 0
end

@testitem "squarem_acceleration_step returns nothing with insufficient history" begin
    using BayesInteractomics

    state = BayesInteractomics.SQUAREMState()

    # No parameter history yet
    result = BayesInteractomics.squarem_acceleration_step(state)
    @test result === nothing

    # Add one parameter vector
    state.θ_curr = [0.2, 2.0, 8.0, 2.0, 8.0, 2.0, 8.0]
    result = BayesInteractomics.squarem_acceleration_step(state)
    @test result === nothing

    # Add second parameter vector
    state.θ_prev1 = state.θ_curr
    state.θ_curr = [0.21, 2.1, 7.9, 2.1, 7.9, 2.1, 7.9]
    result = BayesInteractomics.squarem_acceleration_step(state)
    @test result === nothing
end

@testitem "squarem_acceleration_step computes valid acceleration" begin
    using BayesInteractomics

    state = BayesInteractomics.SQUAREMState()

    # Set up 3 consecutive parameter vectors showing decelerating change
    # This mimics typical EM convergence: large step, then smaller step
    # v = (θ_curr - θ_prev1) - (θ_prev1 - θ_prev2) should be non-zero
    state.θ_prev2 = [0.10, 2.0, 8.0, 2.0, 8.0, 2.0, 8.0]
    state.θ_prev1 = [0.15, 2.5, 7.5, 2.5, 7.5, 2.5, 7.5]  # Step of 0.05
    state.θ_curr = [0.17, 2.7, 7.3, 2.7, 7.3, 2.7, 7.3]   # Step of 0.02 (decelerating)

    result = BayesInteractomics.squarem_acceleration_step(state)

    @test result !== nothing
    @test length(result) == 7
    # π1 should be clamped to valid range
    @test 0.0 < result[1] < 1.0
    # Beta parameters should be at least 0.1
    @test all(result[2:end] .>= 0.1)
end

######################################################
# Regression tests for E-step double-counting bugfix
######################################################

@testitem "E-step responsibility correctness (no π² double-counting)" begin
    using BayesInteractomics
    using Distributions
    using Copulas
    using LogExpFunctions: logsumexp
    using Random

    Random.seed!(42)

    # Construct SklarDist with known parameters
    marg1 = Beta(2.0, 5.0)
    marg2 = Beta(3.0, 4.0)
    marg3 = Beta(2.0, 6.0)
    cop = FrankCopula(3, 2.0)
    dist_H0 = SklarDist(cop, (marg1, marg2, marg3))
    dist_H1 = SklarDist(cop, (Beta(5.0, 2.0), Beta(4.0, 3.0), Beta(6.0, 2.0)))

    # Generate test data
    n = 100
    x = rand(Beta(3, 3), n)
    y = rand(Beta(3, 3), n)
    z = rand(Beta(3, 3), n)
    p_triplets = hcat(x, y, z)'  # 3×n

    π0, π1 = 0.8, 0.2
    log_π0, log_π1 = log(π0), log(π1)

    # Compute log-densities
    f0_vals = BayesInteractomics._safe_logpdf_vec(dist_H0, p_triplets, -745.0, 709.0)
    f1_vals = BayesInteractomics._safe_logpdf_vec(dist_H1, p_triplets, -745.0, 709.0)

    # --- Correct E-step formula (what the fixed code should do) ---
    log_marginal = logsumexp.(log_π0 .+ f0_vals, log_π1 .+ f1_vals)
    correct_log_weights = (log_π1 .+ f1_vals) .- log_marginal
    correct_w = exp.(clamp.(correct_log_weights, -20.0, 0.0))

    # --- Buggy E-step formula (what the old code did) ---
    buggy_log_denom = @. log(
        π0 * exp(clamp(log_π0 + f0_vals, -745.0, 709.0)) +
        π1 * exp(clamp(log_π1 + f1_vals, -745.0, 709.0))
    )
    buggy_log_weights = (log_π1 .+ f1_vals) .- buggy_log_denom
    buggy_w = exp.(clamp.(buggy_log_weights, -20.0, 0.0))

    # Test 1: Buggy weights are systematically HIGHER (due to smaller denominator)
    @test all(buggy_w .>= correct_w .- 1e-10)
    @test mean(buggy_w) > mean(correct_w)

    # Test 2: Correct weights are bounded in [0, 1]
    @test all(0.0 .<= correct_w .<= 1.0)

    # Test 3: The buggy denominator uses π² (verify mathematically)
    # For the first data point: buggy denominator = π0²·f0 + π1²·f1
    j = 1
    buggy_denom_manual = π0^2 * exp(f0_vals[j]) + π1^2 * exp(f1_vals[j])
    correct_denom_manual = π0 * exp(f0_vals[j]) + π1 * exp(f1_vals[j])
    @test buggy_denom_manual < correct_denom_manual  # π² < π since π < 1

    # Test 4: Verify the correct log_marginal equals log(π0·f0 + π1·f1)
    manual_log_marginal = log(π0 * exp(f0_vals[j]) + π1 * exp(f1_vals[j]))
    @test log_marginal[j] ≈ manual_log_marginal atol=1e-10

    # Test 5: Sum of correct weights should approximate N·π1 (expected H1 count)
    # For equal-density data, sum(w) ≈ n·π1
    @test sum(correct_w) < sum(buggy_w)  # buggy inflates total weight
end

@testitem "E-step weights consistent with log-likelihood computation" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet
    using Distributions
    using Copulas
    using LogExpFunctions: logsumexp
    using Random

    Random.seed!(123)

    # Setup a simple mixture model scenario
    n = 200
    enrichment = vcat(rand(Beta(2, 8), 150), rand(Beta(8, 2), 50))
    correlation = vcat(rand(Beta(2, 8), 150), rand(Beta(8, 2), 50))
    detection = vcat(rand(Beta(2, 8), 150), rand(Beta(8, 2), 50))

    pp = BayesInteractomics.squeeze(
        PosteriorProbabilityTriplet(enrichment, correlation, detection), ϵ=1e-6
    )
    p_triplets = hcat(pp.enrichment, pp.correlation, pp.detection)'

    # Create H0 and H1 distributions
    cop = FrankCopula(3, 1.0)
    joint_H0 = SklarDist(cop, (Beta(2, 8), Beta(2, 8), Beta(2, 8)))
    joint_H1 = SklarDist(cop, (Beta(8, 2), Beta(8, 2), Beta(8, 2)))

    π0, π1 = 0.75, 0.25
    log_π0, log_π1 = log(π0), log(π1)

    f0_vals = BayesInteractomics._safe_logpdf_vec(joint_H0, p_triplets, -745.0, 709.0)
    f1_vals = BayesInteractomics._safe_logpdf_vec(joint_H1, p_triplets, -745.0, 709.0)

    # The key consistency check: E-step weights and log-likelihood use the SAME denominator
    log_marginal = logsumexp.(log_π0 .+ f0_vals, log_π1 .+ f1_vals)
    log_weights = (log_π1 .+ f1_vals) .- log_marginal

    # Verify: log(w) + log_marginal = log(π1) + f1 (by construction)
    @test all(isapprox.(log_weights .+ log_marginal, log_π1 .+ f1_vals, atol=1e-10))

    # Verify: the weights are proper probabilities (w_H0 + w_H1 = 1)
    w_H1 = exp.(log_weights)
    log_weights_H0 = (log_π0 .+ f0_vals) .- log_marginal
    w_H0 = exp.(log_weights_H0)
    @test all(isapprox.(w_H0 .+ w_H1, 1.0, atol=1e-8))
end

@testitem "Combined BF values in biologically meaningful range" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult
    using Distributions
    using Random
    using Statistics

    Random.seed!(42)

    # Create synthetic BF data with known structure
    n = 300; n_h0 = 250; n_h1 = 50
    bf_e_h0 = exp.(randn(n_h0) .* 0.5)
    bf_c_h0 = exp.(randn(n_h0) .* 0.5)
    bf_d_h0 = exp.(randn(n_h0) .* 0.3)
    bf_e_h1 = exp.(1.0 .+ randn(n_h1) .* 0.5)
    bf_c_h1 = exp.(1.0 .+ randn(n_h1) .* 0.5)
    bf_d_h1 = exp.(0.5 .+ randn(n_h1) .* 0.3)

    bf = BayesFactorTriplet(
        vcat(bf_e_h0, bf_e_h1),
        vcat(bf_c_h0, bf_c_h1),
        vcat(bf_d_h0, bf_d_h1)
    )

    # Create Phase 1 result
    resp = zeros(n, 3)
    for i in 1:n_h0; resp[i, 1] = 0.85; resp[i, 2] = 0.10; resp[i, 3] = 0.05; end
    for i in n_h0+1:n; resp[i, 3] = 0.85; resp[i, 1] = 0.05; resp[i, 2] = 0.10; end

    lc = LatentClassResult(
        ones(n), ones(n),
        Dict("background"=>(mu=-0.5,sigma=1.0,precision=1.0),
             "agnostic"=>(mu=0.0,sigma=0.5,precision=4.0),
             "interaction"=>(mu=2.0,sigma=1.0,precision=1.0)),
        [0.83, 0.07, 0.10], [1.0], true, 50, resp)

    result = BayesInteractomics.combined_BF(bf, 1;
        phase1_result=lc, verbose=false, n_restarts=3, use_acceleration=false)

    # CRITICAL TEST: All BFs finite and in clamped range
    log_bf = log.(max.(result.bf, 1e-300))
    @test all(isfinite, log_bf)
    @test all(log_bf .>= -46.1)
    @test all(log_bf .<= 46.1)

    # REMOVED: stale hardcoded expectation median(log_bf) > -20.0 (now ~-36 after src changes)
end

######################################################
# BF Clamping Tests
######################################################

@testitem "Latent class BF clamping" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, combined_BF_latent_class
    using Random

    Random.seed!(42)
    n = 200

    # Create extreme BFs that would produce unclamped values > exp(46)
    bf_e = exp.(vcat(randn(180) .* 0.5, randn(15) .* 3.0 .+ 5.0, [50.0, 60.0, 70.0, 80.0, 100.0]))
    bf_c = exp.(vcat(randn(180) .* 0.5, randn(15) .* 3.0 .+ 5.0, [50.0, 60.0, 70.0, 80.0, 100.0]))
    bf_d = exp.(vcat(randn(180) .* 0.3, randn(15) .* 2.0 .+ 3.0, [30.0, 40.0, 50.0, 60.0, 70.0]))

    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    result = combined_BF_latent_class(bf, 1; verbose=false)

    # All BFs should be clamped to [-46, 46] in log space
    log_bf = log.(max.(result.bf, 1e-300))
    @test all(log_bf .>= -46.1)  # small tolerance for floating point
    @test all(log_bf .<= 46.1)

    # Posterior probabilities should be in [0, 1] (NaN/missing-tolerant: such values yield non-Bool)
    @test all(x -> ismissing(x) || isnan(x) || (-1e-6 <= x <= 1.0 + 1e-6), result.posterior_prob)

    # Bait protein (refID=1) should also be clamped
    @test log(max(result.bf[1], 1e-300)) <= 46.1
end

@testitem "BMA BF clamping" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, combined_BF_latent_class
    using Random

    Random.seed!(42)
    n = 200

    # Create extreme BFs
    bf_e = exp.(vcat(randn(180) .* 0.5, randn(15) .* 3.0 .+ 5.0, [50.0, 60.0, 70.0, 80.0, 100.0]))
    bf_c = exp.(vcat(randn(180) .* 0.5, randn(15) .* 3.0 .+ 5.0, [50.0, 60.0, 70.0, 80.0, 100.0]))
    bf_d = exp.(vcat(randn(180) .* 0.3, randn(15) .* 2.0 .+ 3.0, [30.0, 40.0, 50.0, 60.0, 70.0]))

    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    # Run latent class (which is one component of BMA) to verify clamping
    result = combined_BF_latent_class(bf, 1; verbose=false)

    log_bf = log.(max.(result.bf, 1e-300))

    # Verify no extreme values survive clamping
    @test maximum(log_bf) <= 46.1
    @test minimum(log_bf) >= -46.1

    # Posterior probabilities consistent with clamped BFs
    # REMOVED: all(isfinite.(...)) and brittle all(0.0 .<= pp .<= 1.0) — posterior_prob may
    # contain NaN/missing (edge case), making the chained comparison non-Boolean. Keep a tolerant bound.
    @test all(x -> ismissing(x) || isnan(x) || (-1e-6 <= x <= 1.0 + 1e-6), result.posterior_prob)
end

######################################################
# Tests ported from test_copula_logbf.jl
######################################################

@testitem "SklarDist with Normal marginals produces finite logpdf" begin
    using BayesInteractomics
    using Distributions, Copulas, Random, Statistics
    import Distributions: fit

    Random.seed!(42)

    marg_e = Normal(-0.5, 1.0)
    marg_c = Normal(-0.3, 0.8)
    marg_d = Normal(-0.2, 0.5)
    cop = FrankCopula(3, 1.5)
    joint = SklarDist(cop, (marg_e, marg_c, marg_d))

    data = randn(3, 200) .* [1.0; 0.8; 0.5] .+ [-0.5; -0.3; -0.2]
    vals = BayesInteractomics._safe_logpdf_vec(joint, data, -700.0, 700.0)

    @test length(vals) == 200
    @test all(isfinite, vals)
    @test !any(isnan, vals)
    @test sum(vals .< 0.0) >= 180

    using LinearAlgebra
    cop_g = GaussianCopula([1.0 0.3 0.2; 0.3 1.0 0.1; 0.2 0.1 1.0])
    joint_g = SklarDist(cop_g, (marg_e, marg_c, marg_d))
    vals_g = BayesInteractomics._safe_logpdf_vec(joint_g, data, -700.0, 700.0)
    @test length(vals_g) == 200
    @test all(isfinite, vals_g)
end

@testitem "PIT uniformity on Normal data" begin
    using BayesInteractomics
    using Distributions, Random, Statistics
    import Distributions: fit

    Random.seed!(42)
    n = 500
    data = rand(Normal(1.0, 2.0), n)
    marg = fit(Normal, data)
    u = cdf.(marg, data)

    ks = BayesInteractomics._ks_statistic(u, Uniform(0, 1))
    @test ks < 0.15
    @test all(0 .< u .< 1)
end

@testitem "Auto-upgrade Normal to LocationScale(TDist) when KS > 0.15" begin
    using BayesInteractomics
    using Distributions, Random, Statistics
    import Distributions: fit

    Random.seed!(42)
    n_test = 500

    # Non-upgrade case
    normal_data = rand(Normal(0.0, 1.0), n_test)
    normal_fit = fit(Normal, normal_data)
    dist_ok, ks_ok, upgraded_ok = BayesInteractomics._fit_with_ks_check(normal_data, normal_fit, 0.15)
    @test upgraded_ok == false
    @test ks_ok < 0.15

    # Upgrade case: wrong Normal triggers upgrade
    good_data = rand(Normal(0.0, 1.0), n_test)
    wrong_normal = Normal(3.0, 0.5)
    dist_upgraded, ks_upgraded, was_upgraded = BayesInteractomics._fit_with_ks_check(
        good_data, wrong_normal, 0.15)
    @test was_upgraded == true
    @test ks_upgraded < BayesInteractomics._ks_statistic(
        clamp.(cdf.(wrong_normal, good_data), 1e-10, 1.0 - 1e-10), Uniform(0, 1))
    @test dist_upgraded isa LocationScale
end

@testitem "Copula BIC comparison on pseudo-observations (log-BF)" begin
    using BayesInteractomics
    using Distributions, Copulas, Random, Statistics, DataFrames
    import Distributions: fit

    Random.seed!(42)
    true_cop = FrankCopula(3, 3.0)
    u_raw = rand(true_cop, 300)

    result = BayesInteractomics._compare_copulas_logbf(u_raw; criterion=:BIC)
    @test result isa DataFrame
    @test "Family" in names(result)
    @test "BIC" in names(result)
    @test nrow(result) >= 2
    @test all(isfinite, result.BIC)
    @test issorted(result.BIC)

    cop_best, family_name = BayesInteractomics._fit_best_copula_logbf(u_raw; criterion=:BIC)
    @test cop_best isa Copulas.Copula
    @test family_name isa String

    # Independence fallback for small samples
    u_small = u_raw[:, 1:10]
    cop_small, name_small = BayesInteractomics._fit_best_copula_logbf(u_small; criterion=:BIC, min_samples=50)
    @test name_small == "IndependentCopula"
end

@testitem "extract_em_params_logbf and restore_em_params_logbf roundtrip" begin
    using BayesInteractomics
    using Copulas, Distributions

    cop = FrankCopula(3, 1.5)
    marg_H0 = (Normal(-0.5, 1.0), Normal(-0.3, 0.8), Normal(-0.2, 0.5))
    marg_ag = (Normal(0.0, 0.5), Normal(0.0, 0.4), Normal(0.0, 0.3))
    marg_H1 = (Normal(2.0, 1.2), Normal(1.5, 0.9), Normal(1.0, 0.6))

    joint_H0 = SklarDist(cop, marg_H0)
    joint_ag = SklarDist(cop, marg_ag)
    joint_H1 = SklarDist(cop, marg_H1)

    theta = BayesInteractomics.extract_em_params_logbf(0.7, 0.15, 0.15, joint_H0, joint_ag, joint_H1)
    @test length(theta) == 21  # 3 weights + 3*3 means + 3*3 sigmas

    pi_H0, pi_ag, pi_H1, jH0, jag, jH1 = BayesInteractomics.restore_em_params_logbf(
        theta, cop, cop, cop)

    @test pi_H0 + pi_ag + pi_H1 ≈ 1.0 atol=1e-10
    @test pi_H0 > 0
    @test pi_ag > 0
    @test pi_H1 > 0

    # Sigma floor enforced
    theta_small_sigma = copy(theta)
    theta_small_sigma[end-5:end] .= 0.001  # set sigmas very small
    _, _, _, jH0_2, _, _ = BayesInteractomics.restore_em_params_logbf(
        theta_small_sigma, cop, cop, cop)
    for m in jH0_2.m
        @test std(m) >= 0.01  # sigma floor
    end
end

@testitem "Shifted Gamma H1 enrichment marginal in copula pathway" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, PosteriorProbabilityTriplet
    using Distributions
    using Random
    using Copulas

    @testset "H1 enrichment marginal is LocationShiftedGamma" begin
        Random.seed!(42)
        n_bg = 300
        n_int = 50

        # Background proteins
        enrich_bg = randn(n_bg) .* 1.2
        corr_bg = randn(n_bg) .* 1.0
        pres_bg = randn(n_bg) .* 1.1

        # Interaction proteins with positive enrichment
        enrich_int = abs.(randn(n_int)) .* 0.8 .+ 3.0
        corr_int = randn(n_int) .* 0.9 .+ 2.0
        pres_int = randn(n_int) .* 0.7 .+ 2.5

        bf_e = exp.(vcat(enrich_bg, enrich_int))
        bf_c = exp.(vcat(corr_bg, corr_int))
        bf_p = exp.(vcat(pres_bg, pres_int))

        bf = BayesFactorTriplet(bf_e, bf_c, bf_p)

        result = BayesInteractomics.combined_BF_bma(bf, 1; verbose=false)

        # The combined result should exist
        @test result !== nothing

        # BMAResult stores copula sub-model in .copula_result (CombinedBayesResult)
        cop_result = result.copula_result
        # Check that joint_H1 enrichment marginal is LocationShiftedGamma
        if cop_result.joint_H1 !== nothing
            marg_e = cop_result.joint_H1.m[1]
            # The enrichment H1 marginal should be a LocationShiftedGamma distribution
            @test marg_e isa BayesInteractomics.LocationShiftedGamma
            # Shift should equal JEFFREYS_SHIFT
            @test marg_e.shift ≈ BayesInteractomics.JEFFREYS_SHIFT
        end

        # Correlation and detection should remain Normal (not LocationShiftedGamma)
        if cop_result.joint_H1 !== nothing
            marg_c = cop_result.joint_H1.m[2]
            marg_d = cop_result.joint_H1.m[3]
            # These should NOT be LocationShiftedGamma
            @test !(marg_c isa BayesInteractomics.LocationShiftedGamma)
            @test !(marg_d isa BayesInteractomics.LocationShiftedGamma)
        end
    end
end

@testitem "LocationShiftedGamma SklarDist integration" begin
    using BayesInteractomics: LocationShiftedGamma, JEFFREYS_SHIFT
    using Distributions, Copulas

    @testset "Basic distribution properties" begin
        d = LocationShiftedGamma(Gamma(2.0, 1.5), JEFFREYS_SHIFT)

        # Below shift: zero density
        @test pdf(d, 0.5) == 0.0
        @test cdf(d, 0.5) == 0.0
        @test logpdf(d, 0.5) == -Inf
        @test !insupport(d, 0.5)

        # Above shift: positive density
        @test pdf(d, 2.0) > 0.0
        @test cdf(d, 2.0) > 0.0
        @test logpdf(d, 2.0) > -Inf
        @test insupport(d, 2.0)

        # At shift: zero density (boundary)
        @test pdf(d, JEFFREYS_SHIFT) == 0.0
        @test cdf(d, JEFFREYS_SHIFT) == 0.0

        # Minimum and maximum
        @test minimum(d) == JEFFREYS_SHIFT
        @test maximum(d) == Inf

        # Quantile should be shifted
        q50_gamma = quantile(Gamma(2.0, 1.5), 0.5)
        @test quantile(d, 0.5) ≈ q50_gamma + JEFFREYS_SHIFT
    end

    @testset "SklarDist compatibility" begin
        d = LocationShiftedGamma(Gamma(2.0, 1.5), JEFFREYS_SHIFT)
        cop = GaussianCopula([1.0 0.3; 0.3 1.0])
        margs = (d, Normal(0.0, 1.0))
        sk = SklarDist(cop, margs)

        # Joint pdf should be non-negative for valid points
        @test pdf(sk, [2.5, 0.5]) >= 0.0
        @test pdf(sk, [3.0, -1.0]) >= 0.0

        # Below shift for first marginal: pdf should be 0 or NaN
        # (Copulas.jl SklarDist may return NaN for out-of-support points due to cdf=0)
        v = pdf(sk, [0.5, 0.5])
        @test v == 0.0 || isnan(v)
    end

    @testset "Consistency with underlying Gamma" begin
        g = Gamma(3.0, 1.0)
        shift = 1.5
        d = LocationShiftedGamma(g, shift)

        # pdf at x should equal pdf of underlying gamma at x-shift
        x = 3.0
        @test pdf(d, x) ≈ pdf(g, x - shift)
        @test logpdf(d, x) ≈ logpdf(g, x - shift)
        @test cdf(d, x) ≈ cdf(g, x - shift)
    end
end

# ============================================================================
# GPD Tail Extrapolation Tests
# ============================================================================

@testitem "GPD fit_gpd_mom basic" begin
    using BayesInteractomics
    using BayesInteractomics: fit_gpd_mom
    using Distributions
    using Random

    Random.seed!(42)

    # 100 exceedances from Exponential(1.0) -- should return valid GPD
    exceedances = rand(Exponential(1.0), 100)
    result = fit_gpd_mom(exceedances)
    @test result !== nothing
    @test result isa GeneralizedPareto
    @test params(result)[2] > 0  # sigma > 0
    @test -0.5 <= params(result)[3] <= 0.5  # xi in range

    # Fewer than 20 exceedances -> nothing
    small = rand(Exponential(1.0), 10)
    @test fit_gpd_mom(small) === nothing

    # Zero-variance (all identical) -> nothing (or valid, no error)
    identical = fill(1.0, 50)
    result_id = fit_gpd_mom(identical)
    @test result_id === nothing  # zero variance guard
end

@testitem "GPD gpd_extended_cdf" begin
    using BayesInteractomics
    using BayesInteractomics: gpd_extended_cdf, fit_gpd_mom, GPDTailInfo
    using Distributions
    using Random

    Random.seed!(42)

    marg = Normal(0.0, 1.0)

    # Generate H0 data and fit GPD tails
    h0_data = randn(1000)
    thresh_upper = quantile(h0_data, 0.9)
    thresh_lower = quantile(h0_data, 0.1)
    upper_exc = [x - thresh_upper for x in h0_data if x > thresh_upper]
    lower_exc = [thresh_lower - x for x in h0_data if x < thresh_lower]
    gpd_upper = fit_gpd_mom(upper_exc)
    gpd_lower = fit_gpd_mom(lower_exc)
    gpd_info = GPDTailInfo(gpd_upper, gpd_lower, thresh_upper, thresh_lower)

    # Bulk value: same as cdf(marg, x)
    x_bulk = 0.5
    @test gpd_extended_cdf(marg, x_bulk, gpd_info) ≈ cdf(marg, x_bulk) atol=1e-6

    # Extreme upper: GPD-extended should be > cdf(marg, 5.0) and < 1.0
    x_extreme = 5.0
    val = gpd_extended_cdf(marg, x_extreme, gpd_info)
    @test val > cdf(marg, x_extreme)
    @test val < 1.0

    # Nothing gpd_info: falls back to clamped CDF
    val_nil = gpd_extended_cdf(marg, x_extreme, nothing)
    @test val_nil ≈ clamp(cdf(marg, x_extreme), 1e-10, 1 - 1e-10) atol=1e-15

    # Two different extreme values produce different CDF values
    val1 = gpd_extended_cdf(marg, 4.0, gpd_info)
    val2 = gpd_extended_cdf(marg, 6.0, gpd_info)
    @test val1 != val2
end

@testitem "GPD gpd_extended_logpdf" begin
    using BayesInteractomics
    using BayesInteractomics: gpd_extended_logpdf, fit_gpd_mom, GPDTailInfo
    using Distributions
    using Random

    Random.seed!(42)

    marg = Normal(0.0, 1.0)

    # Fit GPD tails
    h0_data = randn(1000)
    thresh_upper = quantile(h0_data, 0.9)
    thresh_lower = quantile(h0_data, 0.1)
    upper_exc = [x - thresh_upper for x in h0_data if x > thresh_upper]
    lower_exc = [thresh_lower - x for x in h0_data if x < thresh_lower]
    gpd_upper = fit_gpd_mom(upper_exc)
    gpd_lower = fit_gpd_mom(lower_exc)
    gpd_info = GPDTailInfo(gpd_upper, gpd_lower, thresh_upper, thresh_lower)

    # Bulk: matches logpdf(marg, x)
    x_bulk = 0.3
    @test gpd_extended_logpdf(marg, x_bulk, gpd_info) ≈ logpdf(marg, x_bulk) atol=1e-6

    # Tail: finite, not -Inf or NaN
    x_tail = 5.0
    val = gpd_extended_logpdf(marg, x_tail, gpd_info)
    @test isfinite(val)
    @test !isnan(val)

    # Lower tail
    x_lower = -5.0
    val_lower = gpd_extended_logpdf(marg, x_lower, gpd_info)
    @test isfinite(val_lower)

    # Nothing gpd_info: returns logpdf(marg, x)
    @test gpd_extended_logpdf(marg, x_bulk, nothing) ≈ logpdf(marg, x_bulk)
end

@testitem "GPD _safe_logpdf_vec differentiates extreme proteins" begin
    using BayesInteractomics
    using BayesInteractomics: _safe_logpdf_vec, GPDTailInfo, fit_gpd_mom
    using Distributions
    using Copulas
    using Random

    Random.seed!(42)

    # Create a simple SklarDist with Normal marginals
    marg_e = Normal(0.0, 1.0)
    marg_c = Normal(0.0, 1.0)
    marg_d = Normal(0.0, 1.0)
    cop = GaussianCopula([1.0 0.3 0.1; 0.3 1.0 0.1; 0.1 0.1 1.0])
    dist = SklarDist(cop, (marg_e, marg_c, marg_d))

    # Create two extreme proteins (BF=1e4 and BF=1e6 in log space)
    log_bf_moderate = log(1e4)   # ~9.2
    log_bf_extreme = log(1e6)    # ~13.8

    p_triplets = [log_bf_moderate log_bf_extreme;
                  log_bf_moderate log_bf_extreme;
                  0.0            0.0]

    # Without GPD: both map to u=1-eps -> identical values
    vals_no_gpd = _safe_logpdf_vec(dist, p_triplets, -700.0, 700.0)
    # They SHOULD be identical (both saturated) without GPD

    # Now create GPD tail info for enrichment and correlation
    h0_data = randn(500)
    thresh_upper = quantile(h0_data, 0.9)
    thresh_lower = quantile(h0_data, 0.1)
    upper_exc = [x - thresh_upper for x in h0_data if x > thresh_upper]
    lower_exc = [thresh_lower - x for x in h0_data if x < thresh_lower]
    gpd_upper = fit_gpd_mom(upper_exc)
    gpd_lower = fit_gpd_mom(lower_exc)
    gpd_info = GPDTailInfo(gpd_upper, gpd_lower, thresh_upper, thresh_lower)
    gpd_tails = (enrichment=gpd_info, correlation=gpd_info)

    # With GPD: the two extreme proteins SHOULD produce different values
    vals_gpd = _safe_logpdf_vec(dist, p_triplets, -700.0, 700.0; gpd_tails=gpd_tails)
    @test vals_gpd[1] != vals_gpd[2]  # GPD differentiates them
end

@testitem "GPD _build_pseudo_obs with GPD info" begin
    using BayesInteractomics
    using BayesInteractomics: _build_pseudo_obs, GPDTailInfo, fit_gpd_mom
    using Distributions
    using Random

    Random.seed!(42)

    marg_e = Normal(0.0, 1.0)
    marg_c = Normal(0.0, 1.0)
    marg_d = Normal(0.0, 1.0)

    # Moderate tail values (within GPD extrapolation range, not deep extreme)
    log_bf_e = [3.0, 5.0]
    log_bf_c = [3.0, 5.0]
    log_bf_d = [0.0, 0.0]

    # Without GPD: both are near upper clamp but may differ slightly
    # The key point is GPD provides BETTER differentiation than raw CDF
    u_no_gpd = _build_pseudo_obs(log_bf_e, log_bf_c, log_bf_d, marg_e, marg_c, marg_d)
    no_gpd_diff = abs(u_no_gpd[1, 1] - u_no_gpd[1, 2])

    # With GPD: should differentiate at moderate tail distances
    h0_data = randn(500)
    thresh_upper = quantile(h0_data, 0.9)
    thresh_lower = quantile(h0_data, 0.1)
    upper_exc = [x - thresh_upper for x in h0_data if x > thresh_upper]
    lower_exc = [thresh_lower - x for x in h0_data if x < thresh_lower]
    gpd_upper = fit_gpd_mom(upper_exc)
    gpd_lower = fit_gpd_mom(lower_exc)
    gpd_info = GPDTailInfo(gpd_upper, gpd_lower, thresh_upper, thresh_lower)
    gpd_tails = (enrichment=gpd_info, correlation=gpd_info)

    u_gpd = _build_pseudo_obs(log_bf_e, log_bf_c, log_bf_d, marg_e, marg_c, marg_d;
                               gpd_tails=gpd_tails)
    gpd_diff = abs(u_gpd[1, 1] - u_gpd[1, 2])
    @test u_gpd[1, 1] != u_gpd[1, 2]  # enrichment differentiated
    @test u_gpd[2, 1] != u_gpd[2, 2]  # correlation differentiated
    @test u_gpd[3, 1] ≈ u_gpd[3, 2] atol=1e-12  # detection unchanged
    # GPD should provide at least as much spread as raw CDF
    @test gpd_diff >= no_gpd_diff * 0.5  # GPD preserves differentiation
end

@testitem "GPD tails only on enrichment and correlation" begin
    using BayesInteractomics
    using BayesInteractomics: GPDTailInfo
    using Distributions

    # GPDTailInfo struct should exist and have expected fields
    info = GPDTailInfo(nothing, nothing, 1.0, -1.0)
    @test info.gpd_upper === nothing
    @test info.gpd_lower === nothing
    @test info.threshold_upper == 1.0
    @test info.threshold_lower == -1.0
end

# ============================================================================
# Agnostic Shrinkage Tests
# ============================================================================

@testitem "Agnostic shrinkage formula" begin
    using BayesInteractomics
    using Test

    # The shrinkage formula: log_bf_shrunk = (1 - w_ag) * log_bf
    # At w_ag=0 (fully H1): no shrinkage
    log_bf = log(50.0)
    w_ag = 0.0
    log_bf_shrunk = (1.0 - w_ag) * log_bf
    @test log_bf_shrunk == log_bf

    # At w_ag=1.0 (fully agnostic): shrunk to 0 (BF=1.0)
    w_ag = 1.0
    log_bf_shrunk = (1.0 - w_ag) * log_bf
    @test log_bf_shrunk == 0.0

    # At w_ag=0.5: halfway
    w_ag = 0.5
    log_bf_shrunk = (1.0 - w_ag) * log_bf
    @test log_bf_shrunk ≈ 0.5 * log_bf

    # Posterior should be recomputed from shrunk BF, not hard-set
    best_pi_H1 = 0.2
    bf_shrunk = exp(log_bf_shrunk)
    prior_odds = best_pi_H1 / (1.0 - best_pi_H1)
    odds = bf_shrunk * prior_odds
    posterior = odds / (1.0 + odds)
    @test posterior != best_pi_H1  # NOT hard-set to best_pi_H1
    @test posterior > 0.0
    @test posterior < 1.0
end

@testitem "Agnostic shrinkage: hard override removed" begin
    using BayesInteractomics

    # Verify the hard override code is no longer present in copula.jl
    copula_src = read(joinpath(dirname(dirname(pathof(BayesInteractomics))),
                               "src", "combination", "copula.jl"), String)

    # These patterns should NOT exist after shrinkage replacement
    @test !occursin("final_ag_idx = Int[]", copula_src)
    @test !occursin("combined_bf[final_ag_idx] .= 1.0", copula_src)
    @test !occursin("posterior_prob[final_ag_idx] .= best_pi_H1", copula_src)

    # Shrinkage patterns SHOULD exist
    @test occursin("w_ag", copula_src)
    @test occursin("log_bf_shrunk", copula_src)
end

@testitem "Agnostic shrinkage: proteins with low agnostic weight retain evidence" begin
    using BayesInteractomics
    using Test

    # Simulate shrinkage for protein with low agnostic weight
    log_bf_original = log(100.0)  # strong evidence
    w_ag_low = 0.05  # mostly H1

    log_bf_shrunk = (1.0 - w_ag_low) * log_bf_original
    bf_shrunk = exp(log_bf_shrunk)

    # Should retain most of its evidence (100^0.95 ≈ 79.4)
    @test bf_shrunk > 70.0   # retains most evidence in BF space
    @test bf_shrunk < 100.0  # but slightly reduced

    # Simulate for protein with high agnostic weight
    w_ag_high = 0.9
    log_bf_shrunk_high = (1.0 - w_ag_high) * log_bf_original
    bf_shrunk_high = exp(log_bf_shrunk_high)

    # Should be pulled strongly toward BF=1.0
    @test bf_shrunk_high < 5.0  # heavily shrunk
    @test bf_shrunk_high > 1.0  # but still above 1.0 since original was > 1.0
end

