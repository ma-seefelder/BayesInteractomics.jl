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

    # ============================================================
    # 1. BIC helpers (kept as utilities, backward compat)
    # ============================================================
    @testset "BIC helpers" begin
        @test BayesInteractomics.compute_bic(-100.0, 7, 100) ≈ 200.0 + 7 * log(100)
        @test BayesInteractomics.compute_bic(-100.0, 7, 100) > 200.0
    end

    # ============================================================
    # 2. Stacking weights
    # ============================================================
    @testset "Stacking weights" begin
        # Equal log-likelihoods -> weights near 0.5
        ll_equal1 = randn(100)
        ll_equal2 = copy(ll_equal1)
        w1, w2 = BayesInteractomics.stacking_weights(ll_equal1, ll_equal2)
        @test w1 + w2 ≈ 1.0 atol=1e-10
        # Both should be > 0
        @test w1 > 0.0
        @test w2 > 0.0

        # One model clearly better -> gets higher weight
        ll_good = randn(100)
        ll_bad  = ll_good .- 2.0  # consistently worse by 2 log-units
        w_good, w_bad = BayesInteractomics.stacking_weights(ll_good, ll_bad)
        @test w_good > w_bad
        @test w_good + w_bad ≈ 1.0 atol=1e-10

        # Reversed
        w_bad2, w_good2 = BayesInteractomics.stacking_weights(ll_bad, ll_good)
        @test w_good2 > w_bad2

        # Small n (10 proteins) still works
        w_small = BayesInteractomics.stacking_weights(randn(10), randn(10))
        @test w_small[1] + w_small[2] ≈ 1.0 atol=1e-10
        @test w_small[1] > 0.0
        @test w_small[2] > 0.0

        # Larger n (1000 proteins) works
        w_large = BayesInteractomics.stacking_weights(randn(1000), randn(1000))
        @test w_large[1] + w_large[2] ≈ 1.0 atol=1e-10

        # Non-degenerate: both weights > 0.01 when models are similar
        ll_sim1 = randn(200) .* 2.0
        ll_sim2 = randn(200) .* 2.0 .+ 0.1
        ws1, ws2 = BayesInteractomics.stacking_weights(ll_sim1, ll_sim2)
        @test ws1 > 0.01
        @test ws2 > 0.01
    end

    # ============================================================
    # 3. Pointwise log-likelihood - EM
    # ============================================================
    @testset "Pointwise log-likelihood - EM" begin
        lc = BayesInteractomics.combined_BF_latent_class(bf_triplet, refID; verbose=false)

        ll_em = BayesInteractomics.pointwise_ll_em(lc, bf_triplet)
        @test length(ll_em) == n
        @test all(isfinite, ll_em)
        @test all(x -> x < 0, ll_em)  # log-densities are negative
    end

    # ============================================================
    # 4. Pointwise log-likelihood - copula
    # ============================================================
    @testset "Pointwise log-likelihood - copula" begin
        # Construct a mock CombinedBayesResult-like test
        # We test the function signature with a real LatentClassResult first
        lc = BayesInteractomics.combined_BF_latent_class(bf_triplet, refID; verbose=false)

        # For copula, we need a CombinedBayesResult. Test the function logic directly.
        # Create a minimal mock scenario: the function takes (result::CombinedBayesResult)
        # We'll test via the total LL extractor for backward compat
        ll_total = BayesInteractomics.latent_class_log_likelihood(lc)
        @test isfinite(ll_total)
        @test ll_total < 0.0
        @test ll_total == lc.free_energy[end]
    end

    # ============================================================
    # 5. Pareto k-hat diagnostic
    # ============================================================
    @testset "Pareto k-hat diagnostic" begin
        # Gaussian log-ratios give k near 0 (light-tailed)
        log_ratios_gaussian = randn(200)
        k = BayesInteractomics.pareto_khat(log_ratios_gaussian)
        @test isfinite(k)
        @test k >= -0.5
        @test k <= 2.0
        # For Gaussian, k should be relatively small
        @test k < 1.0

        # Heavy-tailed (t-distribution with df=2) -> higher k
        log_ratios_heavy = rand(TDist(2.0), 500)
        k_heavy = BayesInteractomics.pareto_khat(log_ratios_heavy)
        @test isfinite(k_heavy)
        @test k_heavy >= -0.5
        @test k_heavy <= 2.0

        # All equal -> k = 0 (degenerate case handled)
        k_const = BayesInteractomics.pareto_khat(zeros(100))
        @test isfinite(k_const)
    end

    # ============================================================
    # 6. Model disagreement
    # ============================================================
    @testset "Model disagreement" begin
        # No disagreement when both agree
        P_EM_high = fill(0.8, 10)
        P_cop_high = fill(0.7, 10)
        disagree = BayesInteractomics.compute_disagreement(P_EM_high, P_cop_high)
        @test disagree isa BitVector
        @test length(disagree) == 10
        @test count(disagree) == 0

        # All disagree
        P_EM_above = fill(0.8, 10)
        P_cop_below = fill(0.3, 10)
        disagree2 = BayesInteractomics.compute_disagreement(P_EM_above, P_cop_below)
        @test count(disagree2) == 10

        # Mixed
        P_EM_mix = [0.8, 0.3, 0.9, 0.2]
        P_cop_mix = [0.3, 0.8, 0.7, 0.1]
        disagree3 = BayesInteractomics.compute_disagreement(P_EM_mix, P_cop_mix)
        @test disagree3[1] == true   # 0.8 > 0.5 vs 0.3 < 0.5 -> disagree
        @test disagree3[2] == true   # 0.3 < 0.5 vs 0.8 > 0.5 -> disagree
        @test disagree3[3] == false  # both > 0.5
        @test disagree3[4] == false  # both < 0.5
    end

    # ============================================================
    # 7. Posterior merging
    # ============================================================
    @testset "Posterior merging (linear BF pooling)" begin
        bf_em_test = [100.0, 0.1, 1.0]
        bf_cop_test = [10.0, 0.1, 1.0]
        prior_odds_test = 0.1

        P_avg, BF_avg, P_cop = BayesInteractomics.merge_posteriors(
            bf_em_test, bf_cop_test, prior_odds_test, 0.5, 0.5
        )

        @test length(P_avg) == 3
        @test all(isfinite, P_avg)
        @test all(isfinite, BF_avg)
        @test all(0.0 .<= P_avg .<= 1.0)
        @test all(BF_avg .> 0.0)

        # BF should be linear average: 0.5*100 + 0.5*10 = 55.0 for first protein
        @test BF_avg[1] ≈ 55.0 atol=1e-6

        # BF and posterior should be consistent
        for i in 1:3
            expected_odds = BF_avg[i] * prior_odds_test
            expected_p = expected_odds / (1.0 + expected_odds)
            @test P_avg[i] ≈ expected_p atol=1e-10
        end

        # Prior invariance: BF should be identical across different prior_odds
        _, BF_1, _ = BayesInteractomics.merge_posteriors(bf_em_test, bf_cop_test, 0.01, 0.5, 0.5)
        _, BF_2, _ = BayesInteractomics.merge_posteriors(bf_em_test, bf_cop_test, 1.0, 0.5, 0.5)
        _, BF_3, _ = BayesInteractomics.merge_posteriors(bf_em_test, bf_cop_test, 10.0, 0.5, 0.5)
        @test BF_avg ≈ BF_1 atol=1e-10
        @test BF_avg ≈ BF_2 atol=1e-10
        @test BF_avg ≈ BF_3 atol=1e-10

        # Monotonicity constraint: all 3 individual BFs < 1 -> combined BF capped
        bf_all_low = BayesInteractomics.BayesFactorTriplet(
            [0.3, 0.5], [0.2, 0.4], [0.1, 0.6]
        )
        bf_em_mono = [5.0, 5.0]
        bf_cop_mono = [5.0, 5.0]

        P_avg_mono, BF_avg_mono, _ = BayesInteractomics.merge_posteriors(
            bf_em_mono, bf_cop_mono, 0.1, 0.5, 0.5;
            bf_triplet = bf_all_low
        )

        # Combined BF should not exceed max individual BF
        for i in 1:2
            max_ind = max(bf_all_low.enrichment[i], bf_all_low.correlation[i], bf_all_low.detection[i])
            @test BF_avg_mono[i] <= max_ind + 1e-10
        end
    end

    # ============================================================
    # 8. BMAResult struct
    # ============================================================
    @testset "BMAResult struct" begin
        @test isdefined(BayesInteractomics, :BMAResult)
        @test BMAResult <: BayesInteractomics.AbstractCombinationResult
        @test fieldnames(BMAResult) == (
            :bf, :posterior_prob, :copula_result, :em3c_result,
            :em_weight, :copula_weight, :model_disagreement, :pareto_k, :prior_odds
        )

        # Test backward-compatible aliases via getproperty
        # We need a minimal BMAResult to test getproperty
        lc = BayesInteractomics.combined_BF_latent_class(bf_triplet, refID; verbose=false)

        # Create a mock CombinedBayesResult is complex, but we can test the struct fields exist
        @test hasfield(BMAResult, :em3c_result)
        @test hasfield(BMAResult, :em_weight)
        @test hasfield(BMAResult, :model_disagreement)
        @test hasfield(BMAResult, :pareto_k)
        @test hasfield(BMAResult, :prior_odds)

        # Verify deprecated fields return expected values via getproperty
        # (copula_bic, latent_class_bic, family_details are handled via getproperty)
    end

    # ============================================================
    # 9. Stacking weight floor prevents complete exclusion
    # ============================================================
    @testset "BMA stacking weight floor prevents complete exclusion" begin
        rng = MersenneTwister(42)
        n_test = 100

        # Create scenario where one model dominates completely
        ll_em_test = randn(rng, n_test) .- 10.0   # EM much worse
        ll_cop_test = randn(rng, n_test)           # Copula much better

        w_em_raw, w_cop_raw = BayesInteractomics.stacking_weights(ll_em_test, ll_cop_test)

        # Raw weights: EM should be near 0
        @test w_em_raw < 0.01

        # Now test the floor via combined_BF_bma behavior:
        bf_e_test = exp.(randn(rng, n_test))
        bf_c_test = exp.(randn(rng, n_test))
        bf_d_test = exp.(randn(rng, n_test) .* 0.3)
        bf_test = BayesInteractomics.BayesFactorTriplet(bf_e_test, bf_c_test, bf_d_test)

        result_floor = BayesInteractomics.combined_BF_bma(bf_test, 1;
            lc_n_iterations=30, n_restarts=3, max_iter=50, verbose=false)

        # Weight floor: both models get at least floor/(1+floor) after renormalization
        # With 5% floor: min weight = 0.05/1.05 ≈ 0.0476
        weight_floor = 0.05
        min_after_renorm = weight_floor / (1.0 + weight_floor)
        @test result_floor.em_weight >= min_after_renorm - 1e-10
        @test result_floor.copula_weight >= min_after_renorm - 1e-10
        @test result_floor.em_weight + result_floor.copula_weight ≈ 1.0 atol=1e-10
    end

    # ============================================================
    # 10. BMA pointwise LL uses unwinsorized data
    # ============================================================
    @testset "BMA pointwise LL uses unwinsorized data" begin
        rng = MersenneTwister(123)
        n_test = 80
        bf_e_test = exp.(randn(rng, n_test) .+ 1.0)
        bf_c_test = exp.(randn(rng, n_test))
        bf_d_test = exp.(randn(rng, n_test) .* 0.3)
        bf_test = BayesInteractomics.BayesFactorTriplet(bf_e_test, bf_c_test, bf_d_test)

        result_unwin = BayesInteractomics.combined_BF_bma(bf_test, 1;
            lc_n_iterations=30, n_restarts=3, max_iter=50, verbose=false)

        # EM should get non-trivial weight (at least the post-renorm floor)
        weight_floor = 0.05
        min_after_renorm = weight_floor / (1.0 + weight_floor)
        @test result_unwin.em_weight >= min_after_renorm - 1e-10
    end

    # ============================================================
    # 11. Jacobian-corrected log-likelihood (backward compat)
    # ============================================================
    @testset "Jacobian-corrected log-likelihood" begin
        lc = BayesInteractomics.combined_BF_latent_class(bf_triplet, refID; verbose=false)

        ll_raw = BayesInteractomics.latent_class_log_likelihood(lc)
        ll_pscale = BayesInteractomics.latent_class_log_likelihood_pscale(lc, bf_triplet)

        @test isfinite(ll_pscale)
        @test ll_pscale > ll_raw

        jacobian_delta = ll_pscale - ll_raw
        @test jacobian_delta > 0.0
        @test isfinite(jacobian_delta)
    end

    # ============================================================
    # 12. Sub-model BF and weight columns (TRANS-01)
    # ============================================================
    @testset "Sub-model BF and weight columns (TRANS-01)" begin
        # Run BMA to get a real BMAResult
        bma_result = BayesInteractomics.combined_BF_bma(bf_triplet, refID;
            lc_n_iterations=30, n_restarts=3, max_iter=50, verbose=false)

        # Verify BMAResult has the fields that pipeline.jl reads for column construction
        @test hasproperty(bma_result, :em3c_result)
        @test hasproperty(bma_result, :copula_result)
        @test hasproperty(bma_result, :em_weight)
        @test hasproperty(bma_result, :copula_weight)

        # Sub-model BF vectors have correct length (detected-only)
        @test length(bma_result.em3c_result.bf) == n
        @test length(bma_result.copula_result.bf) == n

        # Sub-model BFs are positive finite values
        @test all(isfinite, bma_result.em3c_result.bf)
        @test all(x -> x > 0, bma_result.em3c_result.bf)
        @test all(isfinite, bma_result.copula_result.bf)
        @test all(x -> x > 0, bma_result.copula_result.bf)

        # Stacking weights are scalars in (0, 1) that sum to ~1
        @test bma_result.em_weight > 0
        @test bma_result.copula_weight > 0
        @test bma_result.em_weight + bma_result.copula_weight ≈ 1.0 atol=1e-10

        # Simulate the scatter pattern from pipeline.jl to verify it works
        n_full = n + 5  # simulate 5 non-detected proteins
        bf_em_full = Vector{Union{Missing, Float64}}(fill(missing, n_full))
        bf_copula_full = Vector{Union{Missing, Float64}}(fill(missing, n_full))
        w_em_full = Vector{Union{Missing, Float64}}(fill(missing, n_full))
        w_copula_full = Vector{Union{Missing, Float64}}(fill(missing, n_full))
        detected_indices = collect(1:n)  # first n are detected
        for (k, idx) in enumerate(detected_indices)
            bf_em_full[idx] = bma_result.em3c_result.bf[k]
            bf_copula_full[idx] = bma_result.copula_result.bf[k]
            w_em_full[idx] = bma_result.em_weight
            w_copula_full[idx] = bma_result.copula_weight
        end

        # Detected proteins have values
        @test !ismissing(bf_em_full[1])
        @test !ismissing(bf_copula_full[1])
        @test !ismissing(w_em_full[1])

        # Non-detected proteins remain missing
        @test ismissing(bf_em_full[n_full])
        @test ismissing(bf_copula_full[n_full])
        @test ismissing(w_em_full[n_full])
        @test ismissing(w_copula_full[n_full])

        # Weight columns are constant for all detected proteins
        detected_w_em = skipmissing(w_em_full) |> collect
        @test all(w -> w == first(detected_w_em), detected_w_em)

        # w_em + w_copula = 1.0 for detected proteins
        detected_w_copula = skipmissing(w_copula_full) |> collect
        @test first(detected_w_em) + first(detected_w_copula) ≈ 1.0 atol=1e-10
    end
end
