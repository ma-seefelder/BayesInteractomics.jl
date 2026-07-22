# test/simulation/test_simulation.jl
# Tests for the parametric simulation engine
#
# Requirements covered: SIM-01 through SIM-06
# Run command: julia --project=. -e 'using TestItemRunner; @run_package_tests filter=ti->occursin("test_simulation", ti.filename)'

# ============================================================
# Shared mock LatentClassResult
# ============================================================
# Each @testitem runs in isolated module, so we construct the mock inside each test.
# Parameters:
#   H0 (background):  mu=-1.0, sigma=1.0
#   Agnostic:         mu=0.0,  sigma=0.5
#   H1 (interaction): mu=2.0,  sigma=0.8  (corr+det); Gamma(2.0, 1.5) for enrichment
#   mixing_weights:   [0.7, 0.15, 0.15]
# JEFFREYS_SHIFT ~ 1.151, so H1 enrichment mean = 2.0*1.5 + 1.151 = 4.151

@testitem "Test 1: _draw_h1_enrichment for H1 component returns enrichment > JEFFREYS_SHIFT on average" begin
    using BayesInteractomics
    using BayesInteractomics: _draw_h1_enrichment, JEFFREYS_SHIFT
    using Random
    using Statistics

    Random.seed!(42)
    n_draws = 200
    # Gamma(2.0, 1.5): mean = 3.0, so drawn values + JEFFREYS_SHIFT should average > JEFFREYS_SHIFT
    samples_gamma = [_draw_h1_enrichment(:gamma, 2.0, 1.5) for _ in 1:n_draws]
    @test mean(samples_gamma) > JEFFREYS_SHIFT
    @test all(samples_gamma .> JEFFREYS_SHIFT)  # All samples > JEFFREYS_SHIFT (positive support)

    # LogNormal: alpha=mu_log=0.5, theta=sigma_log=0.8
    # mean(LogNormal(0.5, 0.8)) + JEFFREYS_SHIFT > JEFFREYS_SHIFT
    samples_ln = [_draw_h1_enrichment(:lognormal, 0.5, 0.8) for _ in 1:n_draws]
    @test mean(samples_ln) > JEFFREYS_SHIFT

    # Weibull(2.0, 2.0): mean ~ 1.77; mean + JEFFREYS_SHIFT > JEFFREYS_SHIFT
    samples_wb = [_draw_h1_enrichment(:weibull, 2.0, 2.0) for _ in 1:n_draws]
    @test mean(samples_wb) > JEFFREYS_SHIFT

    # Unknown family falls back to Gamma
    samples_fallback = [_draw_h1_enrichment(:unknown_family, 2.0, 1.5) for _ in 1:n_draws]
    @test mean(samples_fallback) > JEFFREYS_SHIFT
end

@testitem "Test 2: _draw_synthetic_triplet for H0 component returns enrichment near background mu" begin
    using BayesInteractomics
    using BayesInteractomics: _draw_synthetic_triplet, LatentClassResult, JEFFREYS_SHIFT
    using Random, Statistics

    mock_lc = LatentClassResult(
        zeros(10), zeros(10),
        Dict("background"  => (mu=-1.0, sigma=1.0, precision=1.0),
             "agnostic"    => (mu=0.0,  sigma=0.5, precision=4.0),
             "interaction" => (mu=2.0,  sigma=0.8, precision=1.5625)),
        [0.7, 0.15, 0.15],
        [0.0], true, 50, nothing, nothing,
        2.0, 1.5, :gamma, Dict(:gamma => 100.0, :lognormal => 110.0, :weibull => 120.0)
    )

    Random.seed!(1234)
    n_draws = 300
    enrichment_vals = [_draw_synthetic_triplet(1, mock_lc, 1.0)[1] for _ in 1:n_draws]
    # H0 enrichment should center near bg.mu = -1.0
    @test abs(mean(enrichment_vals) - (-1.0)) < 0.3
    # Should be mostly well below JEFFREYS_SHIFT (~1.151)
    @test mean(enrichment_vals .< JEFFREYS_SHIFT) > 0.8
end

@testitem "Test 3: _synthetic_posterior returns higher posterior for high-scoring protein" begin
    using BayesInteractomics
    using BayesInteractomics: _synthetic_posterior, LatentClassResult, JEFFREYS_SHIFT

    mock_lc = LatentClassResult(
        zeros(10), zeros(10),
        Dict("background"  => (mu=-1.0, sigma=1.0, precision=1.0),
             "agnostic"    => (mu=0.0,  sigma=0.5, precision=4.0),
             "interaction" => (mu=2.0,  sigma=0.8, precision=1.5625)),
        [0.7, 0.15, 0.15],
        [0.0], true, 50, nothing, nothing,
        2.0, 1.5, :gamma, Dict(:gamma => 100.0, :lognormal => 110.0, :weibull => 120.0)
    )

    # High-evidence protein: enrichment well above shift, correlation high, detection high
    post_high = _synthetic_posterior(4.0, 3.5, 3.0, mock_lc)
    # Low-evidence protein: enrichment below shift, correlation near 0, detection near 0
    post_low  = _synthetic_posterior(-2.0, -1.0, -1.0, mock_lc)

    @test post_high > post_low
    @test post_high > 0.5   # should be classified as H1 with high confidence
    @test post_low  < 0.1   # should be classified as H0/Agnostic
    @test 0.0 <= post_high <= 1.0
    @test 0.0 <= post_low  <= 1.0
end

@testitem "Test 4: _compute_calibration_curves boundary conditions" begin
    using BayesInteractomics
    using BayesInteractomics: _compute_calibration_curves

    n = 100
    # Create clear separation: first 20 are H1, rest H0
    gt = BitVector(vcat(trues(20), falses(80)))
    post = vcat(ones(20) .* 0.95, zeros(80) .* 0.0 .+ 0.05)

    thresholds = collect(range(0.0, stop=1.0, length=200))
    fdr, sens, spec = _compute_calibration_curves(post, gt, thresholds)

    # At threshold 0.0: everything predicted positive, so all H1 are caught → sensitivity = 1
    @test isapprox(sens[1], 1.0, atol=1e-10)

    # At threshold > max posterior (1.0+eps not possible, but at 1.0): no predicted positives
    # last threshold is 1.0, so only proteins with posterior == 1.0 are predicted positive
    # Our H1 proteins have posterior 0.95 < 1.0, so at threshold 1.0, FDR = 0.0 (no predicted positives)
    @test fdr[end] == 0.0

    # FDR should be 0.0 or low at very high thresholds (conservative NaN handling)
    @test all(fdr .>= 0.0)
    @test all(sens .>= 0.0)
end

@testitem "Test 5: _compute_auc returns value in [0.5, 1.0] for non-trivial scenario" begin
    using BayesInteractomics
    using BayesInteractomics: _compute_auc
    using Random

    Random.seed!(99)
    n = 500
    n_h1 = 50
    gt = BitVector(vcat(trues(n_h1), falses(n - n_h1)))
    # H1 proteins have higher posteriors (good discriminator)
    post = vcat(rand(n_h1) .* 0.5 .+ 0.5, rand(n - n_h1) .* 0.3)
    auc = _compute_auc(post, gt)

    @test 0.5 <= auc <= 1.0
    @test auc > 0.7  # should be quite good with this separation

    # Degenerate case: all positive
    gt_all_pos = BitVector(trues(n))
    auc_deg = _compute_auc(post, gt_all_pos)
    @test auc_deg == 0.5  # degenerate
end

@testitem "Test 6: _compute_reliability returns bin centers and observed fractions" begin
    using BayesInteractomics
    using BayesInteractomics: _compute_reliability
    using Random, Statistics

    # Perfect calibration case: ground truth probability ~ posterior
    Random.seed!(777)
    n = 2000
    # Simulate calibrated posteriors
    post = rand(n)
    gt = BitVector([rand() < post[i] for i in 1:n])
    bins = vcat(collect(0.0:0.1:0.9), [0.95, 1.0])

    bin_centers, obs = _compute_reliability(post, gt, bins)

    @test length(bin_centers) == length(bins) - 1
    @test length(obs) == length(bins) - 1
    # Check bin centers are between 0 and 1
    @test all(0.0 .<= bin_centers .<= 1.0)
    # Perfect calibration: observed ≈ predicted (bin center) — within 2 * sigma
    # Only check finite bins
    finite_mask = .!isnan.(obs)
    @test sum(finite_mask) >= 5  # at least some bins have data
    for j in findall(finite_mask)
        @test abs(obs[j] - bin_centers[j]) < 0.25  # loose bound for stochastic test
    end
end

@testitem "Test 7: run_simulation returns SimulationResult with exactly 25 scenarios" begin
    using BayesInteractomics
    using BayesInteractomics: run_simulation, SimulationResult, LatentClassResult

    mock_lc = LatentClassResult(
        zeros(10), zeros(10),
        Dict("background"  => (mu=-1.0, sigma=1.0, precision=1.0),
             "agnostic"    => (mu=0.0,  sigma=0.5, precision=4.0),
             "interaction" => (mu=2.0,  sigma=0.8, precision=1.5625)),
        [0.7, 0.15, 0.15],
        [0.0], true, 50, nothing, nothing,
        2.0, 1.5, :gamma, Dict(:gamma => 100.0, :lognormal => 110.0, :weibull => 120.0)
    )

    result = run_simulation(mock_lc; n_synthetic=500, n_replicates=3)

    @test result isa SimulationResult
    @test length(result.scenarios) == 25  # 5 pi_h1 x 5 effect_scale
    @test result.n_synthetic == 500
    @test result.n_replicates == 3
    @test result.h1_enrichment_family == :gamma
    @test length(result.pi_h1_grid) == 5
    @test length(result.effect_grid) == 5
end

@testitem "Test 8: Each ScenarioResult has correct pi_h1 and effect_scale fields" begin
    using BayesInteractomics
    using BayesInteractomics: run_simulation, LatentClassResult

    mock_lc = LatentClassResult(
        zeros(10), zeros(10),
        Dict("background"  => (mu=-1.0, sigma=1.0, precision=1.0),
             "agnostic"    => (mu=0.0,  sigma=0.5, precision=4.0),
             "interaction" => (mu=2.0,  sigma=0.8, precision=1.5625)),
        [0.7, 0.15, 0.15],
        [0.0], true, 50, nothing, nothing,
        2.0, 1.5, :gamma, Dict(:gamma => 100.0, :lognormal => 110.0, :weibull => 120.0)
    )

    pi_h1_grid = [0.02, 0.05, 0.10, 0.15, 0.20]
    effect_grid = [0.5, 0.75, 1.0, 1.25, 1.5]
    result = run_simulation(mock_lc; n_synthetic=300, n_replicates=2,
                             pi_h1_grid=pi_h1_grid, effect_grid=effect_grid)

    # Verify grid values are correctly assigned
    all_pi_h1 = [sc.pi_h1 for sc in result.scenarios]
    all_effects = [sc.effect_scale for sc in result.scenarios]

    # Each pi_h1 value should appear in n_effect = 5 scenarios
    for pi in pi_h1_grid
        @test count(isapprox.(all_pi_h1, pi, atol=1e-10)) == length(effect_grid)
    end
    # Each effect value should appear in n_pi = 5 scenarios
    for eff in effect_grid
        @test count(isapprox.(all_effects, eff, atol=1e-10)) == length(pi_h1_grid)
    end

    # Calibration curve arrays should have correct length
    for sc in result.scenarios
        @test length(sc.thresholds) == 200
        @test length(sc.fdr_median) == 200
        @test length(sc.sensitivity_median) == 200
        @test length(sc.reliability_bin_centers) == 11  # bins: [0,0.1,...,0.9,0.95,1.0] → 11 bins
    end
end

@testitem "Test 9: H1-drawn proteins have higher mean posterior than H0-drawn" begin
    using BayesInteractomics
    using BayesInteractomics: run_simulation, LatentClassResult
    using Statistics

    mock_lc = LatentClassResult(
        zeros(10), zeros(10),
        Dict("background"  => (mu=-1.5, sigma=0.8, precision=1.5625),
             "agnostic"    => (mu=0.0,  sigma=0.4, precision=6.25),
             "interaction" => (mu=2.5,  sigma=0.6, precision=2.778)),
        [0.7, 0.15, 0.15],
        [0.0], true, 100, nothing, nothing,
        3.0, 1.0, :gamma, Dict(:gamma => 100.0, :lognormal => 110.0, :weibull => 120.0)
    )

    # Use effect_scale=1.0, pi_h1=0.15 (the "canonical" scenario)
    # Find scenario with effect_scale=1.0 and pi_h1=0.15
    result = run_simulation(mock_lc; n_synthetic=1000, n_replicates=3,
                             pi_h1_grid=[0.15], effect_grid=[1.0])

    @test length(result.scenarios) == 1
    sc = result.scenarios[1]

    # At low threshold (0.0), all proteins predicted positive → sensitivity = 1
    @test isapprox(sc.sensitivity_median[1], 1.0, atol=1e-6)

    # AUC should be well above 0.5 (H1 proteins should be discriminable)
    @test sc.auc_median > 0.6
end

@testitem "Test 10: Cache round-trip — save and load returns equivalent SimulationResult" begin
    using BayesInteractomics
    using BayesInteractomics: run_simulation, LatentClassResult, _simulation_param_hash
    using BayesInteractomics: save_simulation_cache, load_simulation_cache
    import Dates: now

    mock_lc = LatentClassResult(
        zeros(10), zeros(10),
        Dict("background"  => (mu=-1.0, sigma=1.0, precision=1.0),
             "agnostic"    => (mu=0.0,  sigma=0.5, precision=4.0),
             "interaction" => (mu=2.0,  sigma=0.8, precision=1.5625)),
        [0.7, 0.15, 0.15],
        [0.0], true, 50, nothing, nothing,
        2.0, 1.5, :gamma, Dict(:gamma => 100.0, :lognormal => 110.0, :weibull => 120.0)
    )

    pi_h1_grid = [0.05, 0.10]
    effect_grid = [1.0, 1.5]

    # Compute result
    result = run_simulation(mock_lc; n_synthetic=200, n_replicates=2,
                             pi_h1_grid=pi_h1_grid, effect_grid=effect_grid)

    # Save to temp file
    cache_path = tempname() * "_sim_test.jld2"
    param_hash = _simulation_param_hash(mock_lc, pi_h1_grid, effect_grid, 200, 2)
    save_simulation_cache(result, param_hash, cache_path)
    @test isfile(cache_path)

    # Load back
    loaded = load_simulation_cache(cache_path, param_hash)
    @test !isnothing(loaded)
    @test length(loaded.scenarios) == 4  # 2 x 2 grid
    @test loaded.n_synthetic == 200
    @test loaded.n_replicates == 2
    @test loaded.h1_enrichment_family == :gamma
    @test isapprox(loaded.scenarios[1].pi_h1, result.scenarios[1].pi_h1, atol=1e-10)
    @test isapprox(loaded.scenarios[1].auc_median, result.scenarios[1].auc_median, atol=1e-10)

    # Cleanup
    rm(cache_path, force=true)
end

@testitem "Test 11: Cache invalidation — different mixing_weights produces different hash" begin
    using BayesInteractomics
    using BayesInteractomics: LatentClassResult, _simulation_param_hash

    cp = Dict("background"  => (mu=-1.0, sigma=1.0, precision=1.0),
              "agnostic"    => (mu=0.0,  sigma=0.5, precision=4.0),
              "interaction" => (mu=2.0,  sigma=0.8, precision=1.5625))

    lc1 = LatentClassResult(
        zeros(5), zeros(5), cp, [0.7, 0.15, 0.15],
        [0.0], true, 50, nothing, nothing,
        2.0, 1.5, :gamma, Dict(:gamma => 100.0, :lognormal => 110.0, :weibull => 120.0)
    )
    lc2 = LatentClassResult(
        zeros(5), zeros(5), cp, [0.6, 0.2, 0.2],  # different mixing_weights
        [0.0], true, 50, nothing, nothing,
        2.0, 1.5, :gamma, Dict(:gamma => 100.0, :lognormal => 110.0, :weibull => 120.0)
    )

    pi_h1_grid = [0.05, 0.10]
    effect_grid = [1.0]
    h1 = _simulation_param_hash(lc1, pi_h1_grid, effect_grid, 1000, 5)
    h2 = _simulation_param_hash(lc2, pi_h1_grid, effect_grid, 1000, 5)
    @test h1 != h2

    # Same params → same hash (deterministic)
    h1b = _simulation_param_hash(lc1, pi_h1_grid, effect_grid, 1000, 5)
    @test h1 == h1b
end

@testitem "Test 12: _build_simulation_json(nothing) returns \"null\"" begin
    using BayesInteractomics
    using BayesInteractomics: _build_simulation_json

    result = _build_simulation_json(nothing)
    @test result == "null"
end

@testitem "Test 13: _build_simulation_json(sim) produces valid JSON with all 25 scenarios" begin
    using BayesInteractomics
    using BayesInteractomics: run_simulation, LatentClassResult, _build_simulation_json

    mock_lc = LatentClassResult(
        zeros(10), zeros(10),
        Dict("background"  => (mu=-1.0, sigma=1.0, precision=1.0),
             "agnostic"    => (mu=0.0,  sigma=0.5, precision=4.0),
             "interaction" => (mu=2.0,  sigma=0.8, precision=1.5625)),
        [0.7, 0.15, 0.15],
        [0.0], true, 50, nothing, nothing,
        2.0, 1.5, :gamma, Dict(:gamma => 100.0, :lognormal => 110.0, :weibull => 120.0)
    )

    result = run_simulation(mock_lc; n_synthetic=200, n_replicates=2)
    json_str = _build_simulation_json(result)

    @test isa(json_str, String)
    @test length(json_str) > 100   # non-trivial JSON

    # Must contain structural keys
    @test occursin("\"scenarios\"", json_str)
    @test occursin("\"pi_h1_grid\"", json_str)
    @test occursin("\"effect_grid\"", json_str)
    @test occursin("\"n_synthetic\"", json_str)
    @test occursin("\"fdr_median\"", json_str)
    @test occursin("\"sensitivity_median\"", json_str)
    @test occursin("\"auc_median\"", json_str)
    @test occursin("\"rel_bins\"", json_str)
    @test occursin("\"fdr_p95_min\"", json_str)

    # Should contain 25 scenario entries (check for "pi_h1" key occurrences)
    n_pi_h1_keys = count(x -> x, [startswith(json_str[i:min(i+8, end)], "\"pi_h1\"")
                                   for i in 1:length(json_str)])
    @test n_pi_h1_keys >= 25   # at least one per scenario

    # Should NOT be extremely large (no raw per-protein data)
    @test length(json_str) < 5_000_000   # well under 5MB
end

@testitem "Test 14: _simulation_param_hash is deterministic for same inputs" begin
    using BayesInteractomics
    using BayesInteractomics: LatentClassResult, _simulation_param_hash

    mock_lc = LatentClassResult(
        zeros(10), zeros(10),
        Dict("background"  => (mu=-1.0, sigma=1.0, precision=1.0),
             "agnostic"    => (mu=0.0,  sigma=0.5, precision=4.0),
             "interaction" => (mu=2.0,  sigma=0.8, precision=1.5625)),
        [0.7, 0.15, 0.15],
        [0.0], true, 50, nothing, nothing,
        2.0, 1.5, :gamma, Dict(:gamma => 100.0, :lognormal => 110.0, :weibull => 120.0)
    )

    pi_h1_grid = [0.02, 0.05, 0.10, 0.15, 0.20]
    effect_grid = [0.5, 0.75, 1.0, 1.25, 1.5]

    h1 = _simulation_param_hash(mock_lc, pi_h1_grid, effect_grid, 10_000, 10)
    h2 = _simulation_param_hash(mock_lc, pi_h1_grid, effect_grid, 10_000, 10)
    @test h1 == h2

    # Different n_synthetic → different hash
    h3 = _simulation_param_hash(mock_lc, pi_h1_grid, effect_grid, 5_000, 10)
    @test h1 != h3

    # Different family → different hash
    mock_lc2 = LatentClassResult(
        zeros(10), zeros(10),
        Dict("background"  => (mu=-1.0, sigma=1.0, precision=1.0),
             "agnostic"    => (mu=0.0,  sigma=0.5, precision=4.0),
             "interaction" => (mu=2.0,  sigma=0.8, precision=1.5625)),
        [0.7, 0.15, 0.15],
        [0.0], true, 50, nothing, nothing,
        2.0, 1.5, :lognormal, Dict(:gamma => 100.0, :lognormal => 110.0, :weibull => 120.0)
    )
    h4 = _simulation_param_hash(mock_lc2, pi_h1_grid, effect_grid, 10_000, 10)
    @test h1 != h4
end
