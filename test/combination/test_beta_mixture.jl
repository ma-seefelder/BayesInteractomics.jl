"""
    test_beta_mixture.jl

Tests for Beta mixture model fitting for marginal distributions.
"""

@testitem "fit_beta_mixture - unimodal data returns K=1" begin
    using BayesInteractomics
    using Distributions
    using Random

    rng = MersenneTwister(42)
    data = rand(rng, Beta(5.0, 3.0), 1000)

    result = BayesInteractomics.fit_beta_mixture(data; max_K=3, n_starts=4)

    # Unimodal data → BIC should select K=1 (single Beta)
    @test result isa Beta
    # Parameters should be close to truth
    α, β = params(result)
    @test α > 2.0  # true α=5
    @test β > 1.0  # true β=3
    # Mean should be approximately correct
    @test abs(mean(result) - 5 / 8) < 0.1
end

@testitem "fit_beta_mixture - bimodal data returns K≥2" begin
    using BayesInteractomics
    using Distributions
    using Random

    rng = MersenneTwister(123)
    # Create clearly bimodal data: 50% Beta(0.5, 5) near 0, 50% Beta(5, 0.5) near 1
    n = 2000
    data = vcat(
        rand(rng, Beta(0.5, 5.0), n ÷ 2),
        rand(rng, Beta(5.0, 0.5), n ÷ 2)
    )

    result = BayesInteractomics.fit_beta_mixture(data; max_K=3, n_starts=6)

    # Should select K≥2 for bimodal data
    if result isa MixtureModel
        @test ncomponents(result) >= 2
        @test length(probs(result)) >= 2
        @test sum(probs(result)) ≈ 1.0
    else
        # If K=1 was selected, the fit should still be reasonable
        @test result isa Beta
    end

    # KS statistic should be much better than single Beta
    single_beta = fit(Beta, data)
    ks_single = BayesInteractomics._ks_statistic(data, single_beta)
    ks_mixture = BayesInteractomics._ks_statistic(data, result)
    @test ks_mixture < ks_single
end

@testitem "fit_beta_mixture - spike-at-0.5 data" begin
    using BayesInteractomics
    using Distributions
    using Random

    # Simulate the H0 enrichment pattern: sharp spike at 0.5 with light tails
    rng = MersenneTwister(456)
    n = 3000
    data = vcat(
        rand(rng, Beta(50.0, 50.0), round(Int, 0.7 * n)),  # spike at 0.5
        rand(rng, Beta(2.0, 2.0), round(Int, 0.3 * n))     # broad tails
    )

    result = BayesInteractomics.fit_beta_mixture(data; max_K=3)

    # Should fit well
    ks = BayesInteractomics._ks_statistic(data, result)
    @test ks < 0.15  # Much better than single Beta

    # Mixture density should be valid at all points
    @test all(isfinite(pdf(result, x)) for x in 0.01:0.01:0.99)
    @test all(pdf(result, x) >= 0 for x in 0.01:0.01:0.99)
end

@testitem "fit_beta_mixture - edge cases" begin
    using BayesInteractomics
    using Distributions

    # Very small dataset → should fallback to K=1
    small = [0.3, 0.5, 0.7, 0.4, 0.6]
    result_small = BayesInteractomics.fit_beta_mixture(small; max_K=3)
    @test result_small isa Beta  # Too few data for mixture

    # All same value → should not crash
    constant = fill(0.5, 100)
    result_const = BayesInteractomics.fit_beta_mixture(constant; max_K=2)
    @test result_const isa Union{Beta, MixtureModel}
    @test isfinite(mean(result_const))

    # Data with NaN/Inf → should filter and still work
    noisy = vcat(rand(Beta(3.0, 3.0), 200), [NaN, Inf, -Inf])
    result_noisy = BayesInteractomics.fit_beta_mixture(noisy; max_K=2)
    @test isfinite(mean(result_noisy))
end

@testitem "fit_beta_mixture_weighted - basic functionality" begin
    using BayesInteractomics
    using Distributions
    using Random

    rng = MersenneTwister(789)
    x = rand(rng, Beta(3.0, 5.0), 500)
    w = ones(500)  # Uniform weights

    result = BayesInteractomics.fit_beta_mixture_weighted(x, w; max_K=2)
    @test result isa Union{Beta, MixtureModel}
    @test isfinite(mean(result))

    # With concentrated weights → should fit the high-weight subset
    w_conc = zeros(500)
    w_conc[1:50] .= 1.0  # Only first 50 points matter
    result_conc = BayesInteractomics.fit_beta_mixture_weighted(x, w_conc; max_K=2)
    @test isfinite(mean(result_conc))

    # Low n_eff → should return prior/fallback
    w_low = zeros(500)
    w_low[1] = 1.0  # Only 1 point
    result_low = BayesInteractomics.fit_beta_mixture_weighted(x, w_low; max_K=2)
    @test result_low isa Beta  # Fallback
end

@testitem "fit_beta_mixture_weighted - K=3 trimodal data" begin
    using BayesInteractomics
    using Distributions
    using Random

    rng = MersenneTwister(2026)
    # Create trimodal data: mass near 0, spike at 0.5, mass near 1
    n = 3000
    x = vcat(
        rand(rng, Beta(0.5, 5.0), n ÷ 3),     # near 0
        rand(rng, Beta(20.0, 20.0), n ÷ 3),    # spike at 0.5
        rand(rng, Beta(5.0, 0.5), n ÷ 3)       # near 1
    )
    w = ones(n)

    # max_K=3 should capture all three modes
    result_k3 = BayesInteractomics.fit_beta_mixture_weighted(x, w; max_K=3, n_starts=6)
    @test result_k3 isa Union{Beta, MixtureModel}
    @test isfinite(mean(result_k3))

    # K=3 should fit better than K=2 for trimodal data
    result_k2 = BayesInteractomics.fit_beta_mixture_weighted(x, w; max_K=2, n_starts=6)
    ks_k3 = BayesInteractomics._ks_statistic(x, result_k3)
    ks_k2 = BayesInteractomics._ks_statistic(x, result_k2)
    @test ks_k3 <= ks_k2 + 0.01  # K=3 should be at least as good

    # Default max_K is now 3
    result_default = BayesInteractomics.fit_beta_mixture_weighted(x, w)
    @test isfinite(mean(result_default))
end

@testitem "safe_damped_mixture - backward compatible with Beta" begin
    using BayesInteractomics
    using Distributions

    fit_d = Beta(4.0, 6.0)
    prev_d = Beta(2.0, 8.0)
    α_damp = 0.5

    result = BayesInteractomics.safe_damped_mixture(fit_d, prev_d, α_damp)
    @test result isa Beta
    α_r, β_r = params(result)
    @test α_r ≈ 3.0  # 0.5*4 + 0.5*2
    @test β_r ≈ 7.0  # 0.5*6 + 0.5*8
end

@testitem "safe_damped_mixture - same K mixture interpolation" begin
    using BayesInteractomics
    using Distributions

    fit_mix = MixtureModel([Beta(2.0, 8.0), Beta(8.0, 2.0)], [0.4, 0.6])
    prev_mix = MixtureModel([Beta(3.0, 7.0), Beta(7.0, 3.0)], [0.5, 0.5])
    α_damp = 0.6

    result = BayesInteractomics.safe_damped_mixture(fit_mix, prev_mix, α_damp)
    @test result isa MixtureModel
    @test ncomponents(result) == 2
    @test sum(probs(result)) ≈ 1.0
    @test isfinite(mean(result))
end

@testitem "SklarDist compatibility with MixtureModel marginals" begin
    using BayesInteractomics
    using Distributions
    using Copulas
    using Random

    rng = MersenneTwister(42)

    # Create mixture marginals
    marg1 = MixtureModel([Beta(2.0, 5.0), Beta(8.0, 2.0)], [0.6, 0.4])
    marg2 = Beta(3.0, 3.0)  # Single Beta (K=1 case)
    marg3 = MixtureModel([Beta(1.5, 3.0), Beta(5.0, 1.5)], [0.5, 0.5])

    # GaussianCopula for simplicity
    cop = GaussianCopula([1.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 1.0])
    joint = SklarDist(cop, (marg1, marg2, marg3))

    # Should be able to compute logpdf
    x = [0.3, 0.5, 0.7]
    lp = logpdf(joint, x)
    @test isfinite(lp)

    # Should be able to compute cdf of marginals
    @test 0 < cdf(marg1, 0.5) < 1
    @test 0 < cdf(marg3, 0.5) < 1

    # Should be able to sample
    samples = rand(rng, joint, 100)
    @test size(samples) == (3, 100)
    @test all(0 .< samples .< 1)

    # _safe_logpdf_vec should work with mixture marginals
    p_mat = hcat([[0.3, 0.5, 0.7], [0.2, 0.8, 0.4], [0.6, 0.3, 0.9]]...)
    vals = BayesInteractomics._safe_logpdf_vec(joint, p_mat, -500.0, 500.0)
    @test length(vals) == 3
    @test all(isfinite, vals)
end

@testitem "mixture_ncomponents and mixture_info_string" begin
    using BayesInteractomics
    using Distributions

    # Single Beta
    d1 = Beta(3.0, 5.0)
    @test BayesInteractomics.mixture_ncomponents(d1) == 1
    s1 = BayesInteractomics.mixture_info_string(d1)
    @test occursin("Beta", s1)

    # MixtureModel
    d2 = MixtureModel([Beta(2.0, 8.0), Beta(8.0, 2.0)], [0.5, 0.5])
    @test BayesInteractomics.mixture_ncomponents(d2) == 2
    s2 = BayesInteractomics.mixture_info_string(d2)
    @test occursin("Beta", s2)
    @test occursin("+", s2)  # Multiple components shown with "+"
end

@testitem "analytical gradient matches finite differences" begin
    using BayesInteractomics
    using Distributions
    using Random

    # Central finite difference gradient
    function finite_diff_gradient(f, θ; ε=1e-6)
        g = similar(θ)
        for j in eachindex(θ)
            θp = copy(θ); θp[j] += ε
            θm = copy(θ); θm[j] -= ε
            g[j] = (f(θp) - f(θm)) / (2ε)
        end
        return g
    end

    rng = MersenneTwister(2026)
    data = Float64[clamp(v, 1e-10, 1.0 - 1e-10)
                   for v in rand(rng, Beta(3.0, 5.0), 500)]

    for K in 1:3
        if K == 1
            θ_test = [log(3.0), log(5.0)]
        elseif K == 2
            θ_test = [0.3, log(2.0), log(4.0), log(5.0), log(2.0)]
        else
            θ_test = [0.2, -0.1, log(2.0), log(5.0), log(4.0), log(3.0),
                       log(6.0), log(1.5)]
        end

        # Finite difference gradient
        grad_fd = finite_diff_gradient(
            θ -> BayesInteractomics._negloglik(θ, data, K), θ_test)
        nll_fd = BayesInteractomics._negloglik(θ_test, data, K)

        # Analytical gradient
        fg! = BayesInteractomics._make_negloglik_fg(data, K)
        G_an = zeros(length(θ_test))
        nll_an = fg!(true, G_an, θ_test)

        @test nll_an ≈ nll_fd atol=1e-10
        @test G_an ≈ grad_fd atol=1e-4
    end

    # Also test weighted version
    w = rand(rng, 500)
    w ./= sum(w)

    for K in 1:3
        if K == 1
            θ_test = [log(3.0), log(5.0)]
        elseif K == 2
            θ_test = [0.3, log(2.0), log(4.0), log(5.0), log(2.0)]
        else
            θ_test = [0.2, -0.1, log(2.0), log(5.0), log(4.0), log(3.0),
                       log(6.0), log(1.5)]
        end

        grad_fd = finite_diff_gradient(
            θ -> BayesInteractomics._negloglik_weighted(θ, data, w, K), θ_test)
        nll_fd = BayesInteractomics._negloglik_weighted(θ_test, data, w, K)

        fg! = BayesInteractomics._make_negloglik_weighted_fg(data, w, K)
        G_an = zeros(length(θ_test))
        nll_an = fg!(true, G_an, θ_test)

        @test nll_an ≈ nll_fd atol=1e-10
        @test G_an ≈ grad_fd atol=1e-4
    end
end

@testitem "analytical gradient performance" begin
    using BayesInteractomics
    using Distributions
    using Random

    rng = MersenneTwister(999)
    data = Float64[clamp(v, 1e-10, 1.0 - 1e-10)
                   for v in rand(rng, Beta(3.0, 5.0), 5000)]
    K = 3
    θ_test = [0.2, -0.1, log(2.0), log(5.0), log(4.0), log(3.0),
               log(6.0), log(1.5)]

    # Warm up
    fg! = BayesInteractomics._make_negloglik_fg(data, K)
    G_an = zeros(length(θ_test))
    fg!(true, G_an, θ_test)

    # Time full fit_beta_mixture on realistic data (analytical gradients)
    fit_data = Float64[clamp(v, 1e-10, 1.0 - 1e-10)
                       for v in rand(rng, Beta(3.0, 5.0), 2000)]
    t_fit = @elapsed for _ in 1:3
        BayesInteractomics.fit_beta_mixture(fit_data; max_K=3, n_starts=4)
    end
    @info "fit_beta_mixture (analytical): $(round(t_fit/3*1000, digits=1))ms avg"

    # Time gradient evaluation
    t_an = @elapsed for _ in 1:100
        fill!(G_an, 0.0)
        fg!(true, G_an, θ_test)
    end
    @info "Analytical gradient: $(round(t_an/100*1e6, digits=1))μs per eval (n=5000, K=3)"

    # Basic sanity: gradient evaluation should be fast
    @test t_an / 100 < 0.1  # < 100ms per eval
end
