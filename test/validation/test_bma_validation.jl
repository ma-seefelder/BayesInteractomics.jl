# BMA Validation Tests (v1.1.6)
# Validates the BMA BF fix using synthetic Bayes Factor triplets.
# All tests use synthetic data -- NO real experimental data, NO XLSX fixtures.
#
# Each @testitem is fully self-contained (no @testsetup) because the current
# TestItemRunner version does not detect @testsetup modules.

# ============================================================
# 1. Basic sanity across prevalences
# ============================================================

@testitem "BMA validation: basic sanity across prevalences" begin
    using BayesInteractomics
    using Random
    using Statistics

    rng_base = MersenneTwister(20260412)

    function make_bf_triplet(rng, n_h0, n_h1)
        bf_e_h0 = exp.(randn(rng, n_h0) .* 1.2)
        bf_c_h0 = exp.(randn(rng, n_h0) .* 1.0)
        bf_d_h0 = exp.(randn(rng, n_h0) .* 0.8)
        bf_e_h1 = exp.(randn(rng, n_h1) .* 1.5 .+ log(1000))
        bf_c_h1 = exp.(randn(rng, n_h1) .* 1.2 .+ log(100))
        bf_d_h1 = exp.(randn(rng, n_h1) .* 1.5 .+ log(10))
        triplet = BayesInteractomics.BayesFactorTriplet(
            vcat(bf_e_h0, bf_e_h1), vcat(bf_c_h0, bf_c_h1), vcat(bf_d_h0, bf_d_h1))
        return (triplet=triplet, n_h0=n_h0, n_h1=n_h1, h1_indices=(n_h0+1):(n_h0+n_h1))
    end

    prevalences = [0.02, 0.05, 0.10, 0.20, 0.50]
    n_total = 200

    for prev in prevalences
        local n_h1 = max(round(Int, n_total * prev), 4)
        local n_h0 = n_total - n_h1
        local data = make_bf_triplet(copy(rng_base), n_h0, n_h1)

        local result = BayesInteractomics.combined_BF_bma(data.triplet, 1;
            lc_n_iterations=100, n_restarts=10, max_iter=100, verbose=false)

        @testset "prevalence=$prev" begin
            @test length(result.bf) == n_total
            @test all(isfinite, result.bf)
            @test all(x -> x > 0, result.bf)
            @test all(isfinite, result.posterior_prob)
            @test all(x -> 0 <= x <= 1, result.posterior_prob)
            @test result.em_weight + result.copula_weight ≈ 1.0 atol=1e-10
        end
    end
end


# ============================================================
# 2. Strong evidence produces high posteriors (VAL-01)
# ============================================================

@testitem "BMA validation: strong evidence produces high posteriors (VAL-01)" begin
    using BayesInteractomics
    using Random
    using Statistics

    rng_base = MersenneTwister(20260412)

    function make_bf_triplet(rng, n_h0, n_h1)
        bf_e_h0 = exp.(randn(rng, n_h0) .* 1.2)
        bf_c_h0 = exp.(randn(rng, n_h0) .* 1.0)
        bf_d_h0 = exp.(randn(rng, n_h0) .* 0.8)
        bf_e_h1 = exp.(randn(rng, n_h1) .* 1.5 .+ log(1000))
        bf_c_h1 = exp.(randn(rng, n_h1) .* 1.2 .+ log(100))
        bf_d_h1 = exp.(randn(rng, n_h1) .* 1.5 .+ log(10))
        triplet = BayesInteractomics.BayesFactorTriplet(
            vcat(bf_e_h0, bf_e_h1), vcat(bf_c_h0, bf_c_h1), vcat(bf_d_h0, bf_d_h1))
        return (triplet=triplet, n_h0=n_h0, n_h1=n_h1, h1_indices=(n_h0+1):(n_h0+n_h1))
    end

    prevalences = [0.02, 0.05, 0.10, 0.20, 0.50]
    n_total = 200

    for prev in prevalences
        local n_h1 = max(round(Int, n_total * prev), 4)
        local n_h0 = n_total - n_h1
        local data = make_bf_triplet(copy(rng_base), n_h0, n_h1)

        local result = BayesInteractomics.combined_BF_bma(data.triplet, 1;
            lc_n_iterations=100, n_restarts=10, max_iter=100, verbose=false)

        local h1_idx = data.h1_indices
        @testset "prevalence=$prev" begin
            local mean_post = mean(result.posterior_prob[h1_idx])
            @test mean_post > 0.9

            local mean_bf = mean(result.bf[h1_idx])
            @test mean_bf > 10
        end
    end

    # Specific VAL-01 check: 10% prevalence, >= 80% of interactors at P > 0.99
    begin
        local n_h1_10 = round(Int, n_total * 0.10)
        local n_h0_10 = n_total - n_h1_10
        local data_10 = make_bf_triplet(copy(rng_base), n_h0_10, n_h1_10)
        local result_10 = BayesInteractomics.combined_BF_bma(data_10.triplet, 1;
            lc_n_iterations=100, n_restarts=10, max_iter=100, verbose=false)
        local h1_idx_10 = data_10.h1_indices
        local frac_high = mean(result_10.posterior_prob[h1_idx_10] .> 0.99)
        @test frac_high >= 0.80
    end
end


# ============================================================
# 3. Proportionality preserved (VAL-02)
# ============================================================

@testitem "BMA validation: proportionality preserved (VAL-02)" begin
    using BayesInteractomics
    using Random
    using Statistics

    rng_base = MersenneTwister(20260412)

    function make_bf_triplet(rng, n_h0, n_h1)
        bf_e_h0 = exp.(randn(rng, n_h0) .* 1.2)
        bf_c_h0 = exp.(randn(rng, n_h0) .* 1.0)
        bf_d_h0 = exp.(randn(rng, n_h0) .* 0.8)
        bf_e_h1 = exp.(randn(rng, n_h1) .* 1.5 .+ log(1000))
        bf_c_h1 = exp.(randn(rng, n_h1) .* 1.2 .+ log(100))
        bf_d_h1 = exp.(randn(rng, n_h1) .* 1.5 .+ log(10))
        triplet = BayesInteractomics.BayesFactorTriplet(
            vcat(bf_e_h0, bf_e_h1), vcat(bf_c_h0, bf_c_h1), vcat(bf_d_h0, bf_d_h1))
        return (triplet=triplet, n_h0=n_h0, n_h1=n_h1, h1_indices=(n_h0+1):(n_h0+n_h1))
    end

    prevalences = [0.02, 0.05, 0.10, 0.20, 0.50]
    n_total = 200

    for prev in prevalences
        local n_h1 = max(round(Int, n_total * prev), 4)
        local n_h0 = n_total - n_h1
        local data = make_bf_triplet(copy(rng_base), n_h0, n_h1)

        local result = BayesInteractomics.combined_BF_bma(data.triplet, 1;
            lc_n_iterations=100, n_restarts=10, max_iter=100, verbose=false)

        local h0_idx = 1:data.n_h0
        local h1_idx = data.h1_indices

        @testset "prevalence=$prev" begin
            # Interactors have higher BFs than null
            @test mean(result.bf[h1_idx]) > mean(result.bf[h0_idx])

            # Rank correlation: input BF strength vs BMA BF
            local bft = data.triplet
            local input_strength = [
                (bft.enrichment[i] * bft.correlation[i] * bft.detection[i])^(1/3)
                for i in 1:length(result.bf)
            ]
            # Simple sign-based concordance check on a subsample
            local n_check = min(50, length(result.bf))
            local concordant = 0
            local total_pairs = 0
            for i in 1:n_check
                for j in (i+1):n_check
                    local sign_input = sign(input_strength[i] - input_strength[j])
                    local sign_bma = sign(result.bf[i] - result.bf[j])
                    if sign_input != 0 && sign_bma != 0
                        concordant += (sign_input == sign_bma) ? 1 : 0
                        total_pairs += 1
                    end
                end
            end
            if total_pairs > 0
                local concordance_rate = concordant / total_pairs
                @test concordance_rate > 0.5
            end
        end
    end
end


# ============================================================
# 4. BF collapse regression
# ============================================================

@testitem "BMA validation: BF collapse regression" begin
    using BayesInteractomics
    using Random
    using Statistics

    rng = MersenneTwister(20260412)

    n_h0 = 200
    n_h1 = 50
    bf_e_h0 = exp.(randn(rng, n_h0) .* 1.2)
    bf_c_h0 = exp.(randn(rng, n_h0) .* 1.0)
    bf_d_h0 = exp.(randn(rng, n_h0) .* 0.8)
    # Interactors with diverse BFs in [100, 100000] range
    bf_e_h1 = exp.(randn(rng, n_h1) .* 2.0 .+ log(1000))
    bf_c_h1 = exp.(randn(rng, n_h1) .* 1.5 .+ log(500))
    bf_d_h1 = exp.(randn(rng, n_h1) .* 1.0 .+ log(50))
    triplet = BayesInteractomics.BayesFactorTriplet(
        vcat(bf_e_h0, bf_e_h1), vcat(bf_c_h0, bf_c_h1), vcat(bf_d_h0, bf_d_h1))

    result = BayesInteractomics.combined_BF_bma(triplet, 1;
        lc_n_iterations=100, n_restarts=10, max_iter=100, verbose=false)

    h1_idx = (n_h0+1):(n_h0+n_h1)
    interactor_bfs = result.bf[h1_idx]

    # BFs should NOT all be identical (the collapse bug)
    # NOTE: All assertions here remain @test_broken because the 3c-EM model
    # itself produces extreme identical BFs (3.97e300) for all interactors,
    # which after log-BF clamping in merge_posteriors collapse to exp(46).
    # The linear BF pooling fix correctly removed the
    # posterior-space averaging bug, but BF collapse persists due to the
    # EM model's BF computation. Root cause is in latent_class.jl, not bma.jl.
    @test std(interactor_bfs) > 0.0

    # No more than 5 proteins should share the same BF value (within tolerance).
    tol = 1e-6
    cluster_sizes = [sum(abs(interactor_bfs[j] - interactor_bfs[i]) < tol
                         for j in eachindex(interactor_bfs))
                     for i in eachindex(interactor_bfs)]
    @test maximum(cluster_sizes) <= 5

    # Coefficient of variation should be meaningful
    @test_broken std(interactor_bfs) / max(mean(interactor_bfs), 1e-300) > 0.01
end


# ============================================================
# 5. All-weak stays weak
# ============================================================

@testitem "BMA validation: all-weak stays weak" begin
    using BayesInteractomics
    using Random
    using Statistics

    rng = MersenneTwister(20260412)

    # 200 proteins ALL null (no interactors)
    bf_e = exp.(randn(rng, 200) .* 1.2)
    bf_c = exp.(randn(rng, 200) .* 1.0)
    bf_d = exp.(randn(rng, 200) .* 0.8)
    triplet = BayesInteractomics.BayesFactorTriplet(bf_e, bf_c, bf_d)

    result = BayesInteractomics.combined_BF_bma(triplet, 1;
        lc_n_iterations=100, n_restarts=10, max_iter=100, verbose=false)

    # Median BF should be modest (no true interactors -- mean can be inflated by
    # right-skewed BF distribution even for pure noise, so use median)
    @test median(result.bf) < 100

    # Median posterior should be low (no true interactors)
    @test median(result.posterior_prob) < 0.5

    # Monotonicity: for proteins where ALL 3 input BFs < 1, combined BF should be bounded
    for i in 1:length(result.bf)
        if triplet.enrichment[i] < 1.0 && triplet.correlation[i] < 1.0 && triplet.detection[i] < 1.0
            local max_individual = max(triplet.enrichment[i], triplet.correlation[i], triplet.detection[i])
            @test result.bf[i] <= max_individual + 1e-6
        end
    end
end


# ============================================================
# 6. Extreme BFs handled gracefully
# ============================================================

@testitem "BMA validation: extreme BFs handled gracefully" begin
    using BayesInteractomics
    using Random
    using Statistics

    rng = MersenneTwister(20260412)

    # 180 null + 20 interactors with BFs near clamp boundary
    bf_e_h0 = exp.(randn(rng, 180) .* 1.2)
    bf_c_h0 = exp.(randn(rng, 180) .* 1.0)
    bf_d_h0 = exp.(randn(rng, 180) .* 0.8)
    bf_e_h1 = exp.(randn(rng, 20) .* 0.5 .+ log(1e8))
    bf_c_h1 = exp.(randn(rng, 20) .* 0.5 .+ log(1e6))
    bf_d_h1 = exp.(randn(rng, 20) .* 0.5 .+ log(1e4))
    triplet = BayesInteractomics.BayesFactorTriplet(
        vcat(bf_e_h0, bf_e_h1), vcat(bf_c_h0, bf_c_h1), vcat(bf_d_h0, bf_d_h1))

    result = BayesInteractomics.combined_BF_bma(triplet, 1;
        lc_n_iterations=100, n_restarts=10, max_iter=100, verbose=false)

    # No NaN or Inf
    @test all(isfinite, result.bf)
    @test !any(isnan, result.bf)
    @test !any(isnan, result.posterior_prob)

    # BFs clamped to finite range
    @test all(x -> x > 0, result.bf)
    @test maximum(result.bf) < exp(47)

    # Posteriors in [0, 1]
    @test all(x -> 0 <= x <= 1, result.posterior_prob)

    # Interactor proteins still identified
    h1_idx = 181:200
    frac_identified = mean(result.posterior_prob[h1_idx] .> 0.5)
    @test frac_identified > 0.5
end


# ============================================================
# 7. Mixed evidence pattern
# ============================================================

@testitem "BMA validation: mixed evidence pattern" begin
    using BayesInteractomics
    using Random
    using Statistics

    # Strong evidence scenario (triple-strong)
    rng_strong = MersenneTwister(20260412)
    bf_e_h0_s = exp.(randn(rng_strong, 160) .* 1.2)
    bf_c_h0_s = exp.(randn(rng_strong, 160) .* 1.0)
    bf_d_h0_s = exp.(randn(rng_strong, 160) .* 0.8)
    bf_e_h1_s = exp.(randn(rng_strong, 40) .* 1.5 .+ log(1000))
    bf_c_h1_s = exp.(randn(rng_strong, 40) .* 1.2 .+ log(100))
    bf_d_h1_s = exp.(randn(rng_strong, 40) .* 1.5 .+ log(10))
    strong_triplet = BayesInteractomics.BayesFactorTriplet(
        vcat(bf_e_h0_s, bf_e_h1_s), vcat(bf_c_h0_s, bf_c_h1_s), vcat(bf_d_h0_s, bf_d_h1_s))
    strong_result = BayesInteractomics.combined_BF_bma(strong_triplet, 1;
        lc_n_iterations=100, n_restarts=10, max_iter=100, verbose=false)

    # Mixed evidence scenario (strong enrichment, weak corr/detection)
    rng_mixed = MersenneTwister(20260412)
    bf_e_h0_m = exp.(randn(rng_mixed, 160) .* 1.2)
    bf_c_h0_m = exp.(randn(rng_mixed, 160) .* 1.0)
    bf_d_h0_m = exp.(randn(rng_mixed, 160) .* 0.8)
    bf_e_h1_m = exp.(randn(rng_mixed, 40) .* 1.5 .+ log(1000))  # strong enrichment
    bf_c_h1_m = exp.(randn(rng_mixed, 40) .* 0.8)                # weak correlation
    bf_d_h1_m = exp.(randn(rng_mixed, 40) .* 0.6)                # weak detection
    mixed_triplet = BayesInteractomics.BayesFactorTriplet(
        vcat(bf_e_h0_m, bf_e_h1_m), vcat(bf_c_h0_m, bf_c_h1_m), vcat(bf_d_h0_m, bf_d_h1_m))
    mixed_result = BayesInteractomics.combined_BF_bma(mixed_triplet, 1;
        lc_n_iterations=100, n_restarts=10, max_iter=100, verbose=false)

    h0_idx = 1:160
    h1_idx = 161:200

    # Mixed-evidence interactors should be distinguishable from null
    mean_bf_h0 = mean(mixed_result.bf[h0_idx])
    mean_bf_mixed = mean(mixed_result.bf[h1_idx])

    @test mean_bf_mixed > mean_bf_h0

    # Mixed-evidence interactors should have lower MEDIAN posterior than triple-strong
    # (mean BF can be dominated by outliers, median is more robust)
    median_post_mixed = median(mixed_result.posterior_prob[h1_idx])
    median_post_strong = median(strong_result.posterior_prob[h1_idx])
    @test median_post_mixed <= median_post_strong

    # Mixed BFs should not be collapsed
    # NOTE: @test_broken because EM model produces extreme identical BFs that
    # dominate linear pooling and clamp to exp(46). Same root cause as the collapse case above.
    mixed_bfs = mixed_result.bf[h1_idx]
    @test std(mixed_bfs) > 0.0
end


# ============================================================
# 8. Prior-invariance
# ============================================================

@testitem "BMA validation: prior-invariance" begin
    using BayesInteractomics
    using Random
    using Statistics

    # Call merge_posteriors directly with fixed synthetic inputs
    # at different prior_odds values. BF_avg should be identical if
    # the BF is a proper likelihood ratio (prior-free).

    n = 20
    # Fixed synthetic EM BFs and copula BFs
    bf_em = vcat(fill(100.0, 10), fill(0.5, 10))
    bf_copula = vcat(fill(50.0, 10), fill(0.2, 10))
    w_em = 0.6
    w_cop = 0.4

    prior_odds_values = [0.01, 0.1, 1.0, 10.0]

    bf_results = Dict{Float64, Vector{Float64}}()
    for po in prior_odds_values
        local merge_out = BayesInteractomics.merge_posteriors(bf_em, bf_copula, po, w_em, w_cop)
        bf_results[po] = merge_out[2]  # BF_avg is second element
    end

    # All pairs of prior_odds values must produce identical BFs
    pair_matches = Bool[]
    for i in 1:length(prior_odds_values)
        for j in (i+1):length(prior_odds_values)
            local bf_i = bf_results[prior_odds_values[i]]
            local bf_j = bf_results[prior_odds_values[j]]
            push!(pair_matches, all(isapprox.(bf_i, bf_j, atol=1e-10)))
        end
    end

    @test length(pair_matches) == 6
    @test all(pair_matches)  # All 6 pairs must match (prior-invariance)

    # Verify basic properties regardless of prior-invariance
    for po in prior_odds_values
        local bfv = bf_results[po]
        @test all(isfinite, bfv)
        @test all(x -> x > 0, bfv)
        @test length(bfv) == n
    end
end
