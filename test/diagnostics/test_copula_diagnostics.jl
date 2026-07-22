"""
    test_copula_diagnostics.jl

Tests for copula diagnostic functions: KL divergence, within-class correlation,
agnostic zone analysis, copula bootstrap CI, and discordant protein analysis.
Updated for log-BF scale with 3-component assignments.
"""

@testitem "KL divergence diagnostic (log-BF scale)" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult
    using Distributions, Random, Test

    Random.seed!(42)
    n = 200

    # Create BFs with clear H0/H1 separation
    bf_e = [exp.(randn(150) .- 1.0); exp.(randn(50) .+ 3.0)]
    bf_c = [exp.(randn(150) .- 1.0); exp.(randn(50) .+ 3.0)]
    bf_d = [exp.(randn(150) .- 1.0); exp.(randn(50) .+ 3.0)]
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    # Mock LatentClassResult with responsibilities
    resp = zeros(n, 3)
    resp[1:120, 1] .= 0.9     # H0
    resp[1:120, 2] .= 0.05
    resp[1:120, 3] .= 0.05
    resp[121:150, 2] .= 0.9   # Agnostic
    resp[121:150, 1] .= 0.05
    resp[121:150, 3] .= 0.05
    resp[151:200, 3] .= 0.97  # H1 (pure)
    resp[151:200, 1] .= 0.015
    resp[151:200, 2] .= 0.015

    class_params = Dict(
        "background" => (mu=-1.0, sigma=1.0, precision=1.0),
        "agnostic"   => (mu=0.0, sigma=0.5, precision=4.0),
        "interaction" => (mu=3.0, sigma=0.5, precision=4.0)
    )
    lc = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [0.6, 0.15, 0.25], collect(1.0:10.0), true, 10, resp
    )

    kl_result = kl_h1_divergence(bf, lc; pure_threshold=0.95)
    @test kl_result.kl_enrichment >= 0.0
    @test kl_result.kl_correlation >= 0.0
    @test kl_result.kl_detection >= 0.0
    @test kl_result.kl_total >= 0.0
    @test kl_result.kl_total ≈ kl_result.kl_enrichment + kl_result.kl_correlation + kl_result.kl_detection
    @test length(kl_result.top_indices) > 0
    @test length(kl_result.pure_marginals) == 3
    @test length(kl_result.full_marginals) == 3
end

@testitem "Within-class correlation diagnostic (3-component)" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult
    using Random, Test

    Random.seed!(42)
    n = 300

    bf_e = [rand(100) .* 0.5; rand(100) .+ 0.5; rand(100) .* 20 .+ 2]
    bf_c = [rand(100) .* 0.5; rand(100) .+ 0.5; rand(100) .* 20 .+ 2]
    bf_d = [rand(100) .* 0.5; rand(100) .+ 0.5; rand(100) .* 20 .+ 2]
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    resp = zeros(n, 3)
    resp[1:100, 1] .= 0.9
    resp[1:100, 2] .= 0.05
    resp[1:100, 3] .= 0.05
    resp[101:200, 2] .= 0.9
    resp[101:200, 1] .= 0.05
    resp[101:200, 3] .= 0.05
    resp[201:300, 3] .= 0.9
    resp[201:300, 1] .= 0.05
    resp[201:300, 2] .= 0.05

    class_params = Dict(
        "background" => (mu=-1.0, sigma=1.0, precision=1.0),
        "agnostic"   => (mu=0.0, sigma=0.5, precision=4.0),
        "interaction" => (mu=3.0, sigma=0.5, precision=4.0)
    )
    lc = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [0.33, 0.33, 0.34], collect(1.0:10.0), true, 10, resp
    )

    wc_result = within_class_correlation(bf, lc)
    # Now returns 3 correlation matrices
    @test size(wc_result.h0_corr) == (3, 3)
    @test size(wc_result.agnostic_corr) == (3, 3)
    @test size(wc_result.h1_corr) == (3, 3)
    @test wc_result.h0_n == 100
    @test wc_result.agnostic_n == 100
    @test wc_result.h1_n == 100
    # Diagonal should be 1.0
    @test all(wc_result.h0_corr[i, i] ≈ 1.0 for i in 1:3)
    @test all(wc_result.agnostic_corr[i, i] ≈ 1.0 for i in 1:3)
    @test all(wc_result.h1_corr[i, i] ≈ 1.0 for i in 1:3)
end

@testitem "Agnostic zone diagnostic (responsibility-based)" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult
    using Distributions, Random, Test

    Random.seed!(42)
    n = 200

    bf_e = exp.(randn(n))
    bf_c = exp.(randn(n))
    bf_d = exp.(randn(n))
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    # 40 proteins are agnostic (high responsibility in component 2)
    resp = zeros(n, 3)
    resp[1:80, 1] .= 0.9
    resp[1:80, 2] .= 0.05
    resp[1:80, 3] .= 0.05
    resp[81:120, 2] .= 0.9   # Agnostic
    resp[81:120, 1] .= 0.05
    resp[81:120, 3] .= 0.05
    resp[121:200, 3] .= 0.9
    resp[121:200, 1] .= 0.05
    resp[121:200, 2] .= 0.05

    class_params = Dict(
        "background" => (mu=-1.0, sigma=1.0, precision=1.0),
        "agnostic"   => (mu=0.0, sigma=0.5, precision=4.0),
        "interaction" => (mu=3.0, sigma=0.5, precision=4.0)
    )
    lc = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [0.4, 0.2, 0.4], collect(1.0:10.0), true, 10, resp
    )

    az_result = agnostic_zone_analysis(bf, lc)
    @test az_result.n_total == n
    @test az_result.n_zone >= 0
    @test 0.0 <= az_result.zone_fraction <= 1.0
    @test az_result.n_zone == length(az_result.zone_indices)
    # The 40 agnostic proteins should be identified
    @test az_result.n_zone >= 30  # at least most
end

@testitem "Copula bootstrap CI (log-BF scale)" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult
    using Random, Test

    Random.seed!(42)
    n = 200

    bf_e = exp.(randn(n) .+ 1.0)
    bf_c = exp.(randn(n) .+ 1.0)
    bf_d = exp.(randn(n) .+ 1.0)
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    resp = zeros(n, 3)
    resp[1:100, 1] .= 0.9
    resp[1:100, 2] .= 0.05
    resp[1:100, 3] .= 0.05
    resp[101:200, 3] .= 0.9
    resp[101:200, 1] .= 0.05
    resp[101:200, 2] .= 0.05

    class_params = Dict(
        "background" => (mu=-1.0, sigma=1.0, precision=1.0),
        "agnostic"   => (mu=0.0, sigma=0.5, precision=4.0),
        "interaction" => (mu=3.0, sigma=0.5, precision=4.0)
    )
    lc = LatentClassResult(
        bf_e .* bf_c .* bf_d, fill(0.5, n), class_params,
        [0.5, 0.0, 0.5], collect(1.0:5.0), true, 5, resp
    )

    boot_result = copula_bootstrap_ci(bf, lc; n_bootstrap=10, seed=42)
    @test haskey(boot_result, :ks_ci)
    @test haskey(boot_result, :tau_ci)
    @test boot_result.n_bootstrap_completed >= 0
end

@testitem "Discordant protein diagnostic (log-BF scale)" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult
    using Random, Test

    Random.seed!(42)
    n = 200

    # Marginals: first 20 have all BFs < 1 (marginal contribution < 0)
    bf_e = [fill(0.3, 20); rand(180) .* 5 .+ 0.1]
    bf_c = [fill(0.4, 20); rand(180) .* 5 .+ 0.1]
    bf_d = [fill(0.5, 20); rand(180) .* 5 .+ 0.1]
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    # Combined BFs: first 20 are > 1 despite marginals < 1
    combined = [fill(3.0, 20); rand(180) .* 10]

    resp = zeros(n, 3)
    resp[1:100, 1] .= 0.9
    resp[1:100, 2] .= 0.05
    resp[1:100, 3] .= 0.05
    resp[101:200, 3] .= 0.9
    resp[101:200, 1] .= 0.05
    resp[101:200, 2] .= 0.05

    class_params = Dict(
        "background" => (mu=-1.0, sigma=1.0, precision=1.0),
        "agnostic"   => (mu=0.0, sigma=0.5, precision=4.0),
        "interaction" => (mu=3.0, sigma=0.5, precision=4.0)
    )
    lc = LatentClassResult(
        combined, combined ./ (1.0 .+ combined), class_params,
        [0.5, 0.0, 0.5], collect(1.0:5.0), true, 5, resp
    )

    disc_result = discordant_protein_analysis(bf, lc)
    @test length(disc_result.marginal_contrib) == n
    @test length(disc_result.copula_contrib) == n
    @test length(disc_result.combined_log_bf) == n
    @test disc_result.n_discordant >= 15
    @test 0.0 <= disc_result.discordant_fraction <= 1.0
end

@testitem "Deprecated signature backward compatibility" begin
    using BayesInteractomics
    using BayesInteractomics: PosteriorProbabilityTriplet, BayesFactorTriplet,
        CombinedBayesResult, EMResult
    using Distributions, Copulas, DataFrames, Random, Test

    Random.seed!(42)
    n = 100

    p_e = clamp.(rand(Beta(2, 8), n), 1e-6, 1 - 1e-6)
    p_c = clamp.(rand(Beta(2, 8), n), 1e-6, 1 - 1e-6)
    p_d = clamp.(rand(Beta(2, 8), n), 1e-6, 1 - 1e-6)
    ppt = PosteriorProbabilityTriplet(p_e, p_c, p_d)

    bf_e = p_e ./ (1 .- p_e)
    bf_c = p_c ./ (1 .- p_c)
    bf_d = p_d ./ (1 .- p_d)
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    marg_e = fit(Beta, p_e)
    marg_c = fit(Beta, p_c)
    marg_d = fit(Beta, p_d)
    cop = FrankCopula(3, 2.0)
    joint = SklarDist(cop, (marg_e, marg_c, marg_d))
    logs_df = DataFrame(iter=1:5, loglikelihood=cumsum(randn(5)),
        pi0=fill(0.7, 5), pi1=fill(0.3, 5))
    em = EMResult(0.7, 0.3, joint, logs_df, true)
    combined_bf = rand(n) .* 5
    posterior_prob = combined_bf ./ (1 .+ combined_bf)
    cbr = CombinedBayesResult(combined_bf, posterior_prob, joint, joint, em, nothing)

    # Old KL divergence signature should still work
    kl_old = kl_h1_divergence(cbr, ppt; top_n=30)
    @test kl_old.kl_total >= 0.0

    # Old agnostic zone signature should still work
    az_old = agnostic_zone_analysis(bf, ppt, cbr)
    @test az_old.n_total == n

    # Old discordant protein signature should still work
    disc_old = discordant_protein_analysis(bf, cbr, ppt)
    @test length(disc_old.marginal_contrib) == n
end

@testitem "Quality gate with soft-assignment responsibilities returns distinct KS values" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult,
        run_quality_gates, QualityGateResult
    using Distributions, Random

    Random.seed!(42)
    n = 200

    # Create BFs with different distributions per component
    # H0 proteins: log-BF centered at -1 (evidence against)
    # Agnostic: log-BF centered at 0
    # H1 proteins: log-BF centered at +3 (evidence for)
    bf_e = vcat(exp.(randn(120) .- 1.0), exp.(randn(40) .* 0.5), exp.(randn(40) .+ 3.0))
    bf_c = vcat(exp.(randn(120) .- 1.0), exp.(randn(40) .* 0.5), exp.(randn(40) .+ 3.0))
    bf_d = vcat(exp.(randn(120) .- 1.0), exp.(randn(40) .* 0.5), exp.(randn(40) .+ 3.0))
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    # SOFT assignments: no protein exceeds 0.5 in any component
    # H0-leaning: 0.45/0.30/0.25
    # Agnostic-leaning: 0.25/0.45/0.30
    # H1-leaning: 0.20/0.35/0.45
    resp = zeros(n, 3)
    resp[1:120, :] .= [0.45 0.30 0.25]   # H0-leaning
    resp[121:160, :] .= [0.25 0.45 0.30]  # Agnostic-leaning
    resp[161:200, :] .= [0.20 0.35 0.45]  # H1-leaning (all < 0.5!)

    class_params = Dict(
        "background" => (mu=-1.0, sigma=1.0, precision=1.0),
        "agnostic"   => (mu=0.0, sigma=0.5, precision=4.0),
        "interaction" => (mu=3.0, sigma=0.5, precision=4.0)
    )
    lc = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [0.6, 0.2, 0.2], collect(1.0:10.0), true, 10, resp
    )

    qg = run_quality_gates(bf, lc)
    @test qg isa QualityGateResult

    # Collect all 9 KS values
    ks_values = [qg.cells[m, k].ks_statistic for m in 1:3, k in 1:3]
    # Should have at least 3 distinct values (not all identical)
    @test length(unique(round.(ks_values, digits=3))) >= 3

    # No cell should be the default 0.0 (which indicates skipped test)
    # Since we have 200 proteins and min_effective_n=5, all should run
    for m in 1:3, k in 1:3
        @test qg.cells[m, k].n_effective > 0
    end
end
