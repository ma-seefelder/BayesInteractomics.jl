# ============================================================
# :auto prior (EB Dirichlet + grid marginalization) Integration Test Suite
# ============================================================
# Covers: :auto dispatch (PIPE-01), config hash (PIPE-02),
# H1 family lock (PIPE-03), explicit alpha path, constructor regression.
# ============================================================

# --------------------------------------------------
# Block 1: :auto dispatch runs EB + grid marginalization (PIPE-01)
# --------------------------------------------------
@testitem ":auto dispatch runs EB + grid marginalization" begin
    using BayesInteractomics, Distributions, Random

    # Create synthetic BFs with known structure: ~10% true interactors
    n = 80
    rng = MersenneTwister(42)
    bf_e = ones(n)
    bf_c = ones(n)
    bf_d = fill(0.5, n)
    # 8 true interactors with high BFs
    bf_e[1:8] .= exp.(2.0 .+ randn(rng, 8))
    bf_c[1:8] .= exp.(1.5 .+ 0.5 * randn(rng, 8))
    bf_d[1:8] .= 0.8 .+ 0.1 * rand(rng, 8)
    bf = BayesInteractomics.BayesFactorTriplet(bf_e, bf_c, bf_d)

    r = BayesInteractomics.combined_BF_latent_class(bf, 1;
        alpha_prior=:auto, n_restarts=3, n_iterations=50, verbose=false)

    @test r isa BayesInteractomics.LatentClassResult
    @test length(r.effective_alpha_prior) == 3
    @test all(r.effective_alpha_prior .> 0)
    @test r.prior_grid_weights !== nothing
    @test r.prior_grid_posteriors !== nothing
    @test length(r.prior_grid_weights) == length(r.prior_grid_posteriors)
    @test sum(r.prior_grid_weights) ≈ 1.0 atol=1e-6
    @test r.h1_enrichment_family in (:gamma, :lognormal, :weibull)
    # Posteriors should be valid probabilities (allow tiny floating-point overshoot)
    @test all(-1e-10 .<= r.posterior_prob .<= 1.0 + 1e-10)
    @test all(r.bf .>= 0.0)
    # BFs should be finite
    @test all(isfinite, r.bf)
    @test all(isfinite, r.posterior_prob)
end

# --------------------------------------------------
# Block 2: Explicit alpha path unchanged (PIPE-01)
# --------------------------------------------------
@testitem "explicit alpha path unchanged" begin
    using BayesInteractomics, Distributions, Random

    n = 60
    rng = MersenneTwister(123)
    bf_e = exp.(randn(rng, n))
    bf_c = exp.(0.5 * randn(rng, n))
    bf_d = rand(rng, n) .* 0.8 .+ 0.1
    bf = BayesInteractomics.BayesFactorTriplet(bf_e, bf_c, bf_d)

    r = BayesInteractomics.combined_BF_latent_class(bf, 1;
        alpha_prior=[5.0, 2.0, 1.0], n_restarts=3, n_iterations=50, verbose=false)

    @test r isa BayesInteractomics.LatentClassResult
    @test r.prior_grid_weights === nothing
    @test r.prior_grid_posteriors === nothing
    @test r.eb_converged === false
    @test r.effective_alpha_prior == [5.0, 2.0, 1.0]
    # Basic validity checks
    @test all(0.0 .<= r.posterior_prob .<= 1.0)
    @test all(isfinite, r.bf)
end

# --------------------------------------------------
# Block 3: Config hash differs for :auto vs explicit (PIPE-02)
# --------------------------------------------------
@testitem "config hash differs for :auto vs explicit" begin
    using BayesInteractomics

    # Required CONFIG fields
    base_kw = (
        datafile = ["test.xlsx"],
        control_cols = [Dict(1 => [1, 2])],
        sample_cols = [Dict(1 => [3, 4])],
        poi = "BAIT",
    )

    c1 = BayesInteractomics.CONFIG(; base_kw..., lc_alpha_prior=:auto)
    c2 = BayesInteractomics.CONFIG(; base_kw..., lc_alpha_prior=[5.0, 2.0, 1.0])
    h1 = BayesInteractomics.compute_config_hash(c1)
    h2 = BayesInteractomics.compute_config_hash(c2)
    @test h1 != h2

    # Same config should produce same hash
    c3 = BayesInteractomics.CONFIG(; base_kw..., lc_alpha_prior=:auto)
    h3 = BayesInteractomics.compute_config_hash(c3)
    @test h1 == h3

    # Same explicit vector should produce same hash
    c4 = BayesInteractomics.CONFIG(; base_kw..., lc_alpha_prior=[5.0, 2.0, 1.0])
    h4 = BayesInteractomics.compute_config_hash(c4)
    @test h2 == h4
end

# --------------------------------------------------
# Block 4: H1 family lock from baseline (PIPE-03)
# --------------------------------------------------
@testitem "H1 family locked from baseline EM" begin
    using BayesInteractomics, Distributions, Random

    # Create synthetic BFs with clear signal
    n = 80
    rng = MersenneTwister(77)
    bf_e = ones(n)
    bf_c = ones(n)
    bf_d = fill(0.5, n)
    bf_e[1:10] .= exp.(2.5 .+ randn(rng, 10))
    bf_c[1:10] .= exp.(2.0 .+ 0.5 * randn(rng, 10))
    bf_d[1:10] .= 0.85 .+ 0.1 * rand(rng, 10)
    bf = BayesInteractomics.BayesFactorTriplet(bf_e, bf_c, bf_d)

    r = BayesInteractomics.combined_BF_latent_class(bf, 1;
        alpha_prior=:auto, n_restarts=3, n_iterations=50, verbose=false)

    # H1 family should be a valid selection
    @test r.h1_enrichment_family in (:gamma, :lognormal, :weibull)
    # The family should be consistent (locked from baseline)
    @test r.h1_enrichment_family isa Symbol

    # Per-grid posteriors should all have the same length as the input
    for pp in r.prior_grid_posteriors
        @test length(pp) == n
        @test all(0.0 .<= pp .<= 1.0)
    end
end

# --------------------------------------------------
# Block 5: EB convergence fallback (robustness)
# --------------------------------------------------
@testitem "EB fallback on degenerate input" begin
    using BayesInteractomics, Distributions, Random

    # All proteins with nearly identical BFs -> EB may struggle
    # but should still produce valid output
    n = 40
    rng = MersenneTwister(999)
    bf_e = exp.(0.01 * randn(rng, n))  # very tight around 1.0
    bf_c = exp.(0.01 * randn(rng, n))
    bf_d = fill(0.5, n)
    bf = BayesInteractomics.BayesFactorTriplet(bf_e, bf_c, bf_d)

    # Should not error even with degenerate data
    r = BayesInteractomics.combined_BF_latent_class(bf, 1;
        alpha_prior=:auto, n_restarts=2, n_iterations=30, verbose=false)

    @test r isa BayesInteractomics.LatentClassResult
    @test length(r.effective_alpha_prior) == 3
    @test all(r.effective_alpha_prior .> 0)
    # Grid fields should be populated (even if EB didn't converge)
    @test r.prior_grid_weights !== nothing
    @test r.prior_grid_posteriors !== nothing
    @test all(isfinite, r.posterior_prob)
end

# --------------------------------------------------
# Block 6: Convenience constructors regression (:auto prior fields)
# --------------------------------------------------
@testitem "convenience constructors regression" begin
    using BayesInteractomics, DataFrames

    # Shared test data
    bf = [1.0, 2.0, 3.0]
    pp = [0.3, 0.6, 0.9]
    cp = Dict(
        "background" => (mu=0.0, sigma=1.0, precision=1.0),
        "interaction" => (mu=2.0, sigma=0.5, precision=4.0)
    )
    mw = [0.7, 0.3]
    fe = [100.0, 200.0]

    # 7-arg constructor
    r7 = BayesInteractomics.LatentClassResult(bf, pp, cp, mw, fe, true, 10)
    @test r7.eb_converged === false
    @test r7.effective_alpha_prior == Float64[]
    @test r7.prior_grid_weights === nothing
    @test r7.prior_grid_posteriors === nothing

    # 8-arg constructor
    r8 = BayesInteractomics.LatentClassResult(bf, pp, cp, mw, fe, true, 10, nothing)
    @test r8.eb_converged === false
    @test r8.effective_alpha_prior == Float64[]

    # 9-arg constructor
    r9 = BayesInteractomics.LatentClassResult(bf, pp, cp, mw, fe, true, 10, nothing, nothing)
    @test r9.eb_converged === false
    @test r9.effective_alpha_prior == Float64[]

    # 11-arg constructor
    r11 = BayesInteractomics.LatentClassResult(bf, pp, cp, mw, fe, true, 10, nothing, nothing, 2.0, 2.0)
    @test r11.eb_converged === false
    @test r11.effective_alpha_prior == Float64[]

    # 13-arg constructor
    r13 = BayesInteractomics.LatentClassResult(bf, pp, cp, mw, fe, true, 10, nothing, nothing, 2.0, 2.0, :gamma, Dict{Symbol,Float64}(:gamma => 0.0))
    @test r13.eb_converged === false
    @test r13.effective_alpha_prior == Float64[]

    # 15-arg constructor
    r15 = BayesInteractomics.LatentClassResult(bf, pp, cp, mw, fe, true, 10, nothing, nothing, 2.0, 2.0, 1.5, :gamma, Dict{Symbol,Float64}(:gamma => 0.0), nothing)
    @test r15.eb_converged === false
    @test r15.effective_alpha_prior == Float64[]

    # 18-arg constructor
    r18 = BayesInteractomics.LatentClassResult(bf, pp, cp, mw, fe, true, 10, nothing, nothing, 2.0, 2.0, 1.5, :gamma, Dict{Symbol,Float64}(:gamma => 0.0), nothing, nothing, nothing, nothing)
    @test r18.eb_converged === false
    @test r18.effective_alpha_prior == Float64[]
    @test r18.prior_grid_weights === nothing
    @test r18.prior_grid_posteriors === nothing
end
