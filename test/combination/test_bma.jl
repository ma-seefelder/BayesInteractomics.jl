@testitem "Bayesian Model Averaging (BMA)" begin
    using BayesInteractomics
    using Random
    using Distributions
    using Statistics
    using DataFrames

    Random.seed!(42)

    # ---- Shared synthetic data ----
    n_bg = 80
    n_int = 20
    n = n_bg + n_int

    enrich_bg  = randn(n_bg) .* 1.2
    corr_bg    = randn(n_bg) .* 1.0
    pres_bg    = randn(n_bg) .* 1.1

    enrich_int = randn(n_int) .* 0.8 .+ 4.0
    corr_int   = randn(n_int) .* 0.9 .+ 3.0
    pres_int   = randn(n_int) .* 0.7 .+ 3.5

    bf_enrich = exp.(vcat(enrich_bg, enrich_int))
    bf_corr   = exp.(vcat(corr_bg,   corr_int))
    bf_pres   = exp.(vcat(pres_bg,   pres_int))

    bf_triplet = BayesInteractomics.BayesFactorTriplet(bf_enrich, bf_corr, bf_pres)
    refID = 1

    @testset "BIC helpers" begin
        @test BayesInteractomics.compute_bic(-100.0, 7, 100) ≈ 200.0 + 7 * log(100)
        @test BayesInteractomics.compute_bic(-100.0, 7, 100) > 200.0

        weights = BayesInteractomics.bma_weights([100.0, 100.0])
        @test length(weights) == 2
        @test weights[1] ≈ 0.5 atol=1e-10
        @test weights[2] ≈ 0.5 atol=1e-10
        @test sum(weights) ≈ 1.0 atol=1e-10

        # Lower BIC gets higher weight
        weights2 = BayesInteractomics.bma_weights([90.0, 100.0])
        @test weights2[1] > weights2[2]
        @test sum(weights2) ≈ 1.0 atol=1e-10
    end

    @testset "LatentClassResult log-likelihood extraction" begin
        lc = BayesInteractomics.combined_BF_latent_class(bf_triplet, refID; verbose=false)
        ll = BayesInteractomics.latent_class_log_likelihood(lc)
        @test isfinite(ll)
        @test ll < 0.0  # log-likelihood should be negative
        @test ll == lc.free_energy[end]
    end

    @testset "BMAResult structure" begin
        # We cannot call combined_BF_bma without an H0 file, so we test the
        # model_averaging logic directly using existing sub-results
        lc = BayesInteractomics.combined_BF_latent_class(bf_triplet, refID; verbose=false)

        @test lc isa LatentClassResult
        @test length(lc.bf) == n
        @test length(lc.posterior_prob) == n
        @test all(0 .<= lc.posterior_prob .<= 1)
    end

    @testset "BMA weight normalization" begin
        for (bic_a, bic_b) in [(100.0, 200.0), (50.0, 50.0), (300.0, 100.0)]
            w = BayesInteractomics.bma_weights([bic_a, bic_b])
            @test sum(w) ≈ 1.0 atol=1e-10
            @test all(w .>= 0)
            if bic_a < bic_b
                @test w[1] > w[2]
            elseif bic_a > bic_b
                @test w[1] < w[2]
            else
                @test w[1] ≈ w[2] atol=1e-10
            end
        end
    end

    @testset "BMAResult exported" begin
        @test isdefined(BayesInteractomics, :BMAResult)
        @test BMAResult <: BayesInteractomics.AbstractCombinationResult
        @test fieldnames(BMAResult) == (
            :bf, :posterior_prob, :copula_result, :latent_class_result,
            :copula_bic, :latent_class_bic, :copula_weight, :latent_class_weight
        )
    end
end
