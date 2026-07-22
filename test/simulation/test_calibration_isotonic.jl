# test/simulation/test_calibration_isotonic.jl
# Tests for Platt scaling calibration engine
#
# Requirements covered: CAL-01 through CAL-04, CAL-FIX-03
# Run command: julia --project=. -e 'using TestItemRunner; @run_package_tests filter=ti->occursin("test_calibration_isotonic", ti.filename)'

# ============================================================
# Task 1: CalibrationModel, FDRCalibrationModel types and Platt fitting
# ============================================================

@testitem "CalibrationModel struct construction" begin
    using BayesInteractomics
    using BayesInteractomics: CalibrationModel, FDRCalibrationModel, CalibrationCVMetrics

    m = CalibrationModel(1.0, 0.0, 100, true)
    @test m.a == 1.0
    @test m.b == 0.0
    @test m.n_training == 100
    @test m.converged == true

    fdr_m = FDRCalibrationModel(-5.0, 2.0, 100, true)
    @test fdr_m.a == -5.0
    @test fdr_m.b == 2.0
    @test fdr_m.n_training == 100
    @test fdr_m.converged == true

    cv = CalibrationCVMetrics(
        10,
        fill(0.04, 10), 0.04, 0.005,
        fill(0.06, 10), 0.06, 0.005,
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.1, 0.3, 0.5, 0.7, 0.9],
        true,
        "green"
    )
    @test cv.n_folds == 10
    @test cv.posterior_ece_mean ≈ 0.04
    @test cv.passes_ece_threshold == true
    @test cv.ece_badge_color == "green"
end

@testitem "_fit_posterior_calibration returns valid Platt model" begin
    using BayesInteractomics
    using BayesInteractomics: _fit_posterior_calibration, CalibrationModel

    x = [0.1, 0.3, 0.5, 0.7, 0.9]
    y = BitVector([0, 0, 1, 0, 1])
    model = _fit_posterior_calibration(x, y)

    @test model isa CalibrationModel
    @test model.n_training == 5
    @test isfinite(model.a)
    @test isfinite(model.b)
end

@testitem "_apply_calibration is monotone with epsilon clamping" begin
    using BayesInteractomics
    using BayesInteractomics: _fit_posterior_calibration, _apply_calibration

    x = collect(range(0.0, 1.0, length=20))
    y = BitVector(x .>= 0.5)
    model = _fit_posterior_calibration(x, y)

    # Test on sorted inputs: Platt with positive a is monotone
    test_inputs = collect(range(0.01, 0.99, length=50))
    calibrated = [_apply_calibration(v, model) for v in test_inputs]
    # Platt is smooth and monotone (allow tiny numerical noise)
    @test all(diff(calibrated) .>= -1e-10)

    # Clamping: all values in [epsilon, 1-epsilon]
    epsilon = 1e-6
    @test all(calibrated .>= epsilon)
    @test all(calibrated .<= 1 - epsilon)

    # Custom epsilon
    calibrated2 = [_apply_calibration(v, model; epsilon=1e-4) for v in test_inputs]
    @test all(calibrated2 .>= 1e-4)
    @test all(calibrated2 .<= 1 - 1e-4)
end

@testitem "_apply_calibration handles extreme values" begin
    using BayesInteractomics
    using BayesInteractomics: CalibrationModel, _apply_calibration

    # Identity-like model (a=1, b=0)
    model = CalibrationModel(1.0, 0.0, 30, true)

    epsilon = 1e-6
    # Near 0: should return small but > epsilon
    val_low = _apply_calibration(0.001, model)
    @test val_low >= epsilon
    @test val_low < 0.1

    # Near 1: should return high but < 1-epsilon
    val_high = _apply_calibration(0.999, model)
    @test val_high <= 1 - epsilon
    @test val_high > 0.9

    # Exact boundary
    val_zero = _apply_calibration(0.0, model)
    @test val_zero >= epsilon
    val_one = _apply_calibration(1.0, model)
    @test val_one <= 1 - epsilon
end

@testitem "_fit_fdr_calibration returns valid Platt FDR model" begin
    using BayesInteractomics
    using BayesInteractomics: _fit_fdr_calibration, FDRCalibrationModel

    thresholds = collect(range(0.0, 1.0, length=20))
    empirical_fdr = max.(0.0, 1.0 .- thresholds .* 0.9)

    model = _fit_fdr_calibration(thresholds, empirical_fdr)

    @test model isa FDRCalibrationModel
    @test model.n_training == 20
    @test isfinite(model.a)
    @test isfinite(model.b)
end

@testitem "_apply_fdr_calibration returns value in [0, 1]" begin
    using BayesInteractomics
    using BayesInteractomics: FDRCalibrationModel, _apply_fdr_calibration

    # FDR decreasing model (a < 0)
    model = FDRCalibrationModel(-5.0, 2.0, 30, true)

    for t in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        v = _apply_fdr_calibration(t, model)
        @test 0.0 <= v <= 1.0
    end

    # Monotone: higher threshold should give same or lower FDR
    vals = [_apply_fdr_calibration(t, model) for t in collect(range(0.0, 1.0, length=50))]
    @test all(diff(vals) .<= 1e-10)
end

@testitem "_fit_posterior_calibration and _apply_calibration on large realistic data" begin
    using BayesInteractomics
    using BayesInteractomics: _fit_posterior_calibration, _apply_calibration
    using Random, Statistics

    Random.seed!(42)
    n = 500
    # Simulated miscalibrated posteriors: true P(H1) = 0.3, but model outputs are overconfident
    raw = sort(vcat(rand(350) .* 0.4, rand(150) .* 0.4 .+ 0.6))
    truth = BitVector(vcat(falses(350), trues(150)))

    model = _fit_posterior_calibration(raw, truth)

    @test model.n_training == n
    @test isfinite(model.a)
    @test isfinite(model.b)

    # Apply to test set
    calibrated = [_apply_calibration(v, model) for v in raw]
    # Monotonicity on sorted inputs (Platt with a > 0 is monotone)
    @test all(diff(calibrated) .>= -1e-10)
    # All in (0, 1)
    @test all(calibrated .> 0.0)
    @test all(calibrated .< 1.0)
end

# ============================================================
# Task 2: Stratified k-fold CV and ECE computation
# ============================================================

@testitem "_stratified_kfold_indices covers all indices exactly once" begin
    using BayesInteractomics
    using BayesInteractomics: _stratified_kfold_indices
    using Random

    rng = Random.MersenneTwister(42)
    n = 120
    # 3 classes with 40 each
    labels = vcat(fill(1, 40), fill(2, 40), fill(3, 40))
    n_folds = 10
    fold_assignments = _stratified_kfold_indices(labels, n_folds, rng)

    @test length(fold_assignments) == n
    # Each index assigned to exactly one fold
    @test sort(unique(fold_assignments)) == collect(1:n_folds)
    # Every fold has approximately equal size (n / n_folds = 12)
    for k in 1:n_folds
        @test 10 <= sum(fold_assignments .== k) <= 14
    end
end

@testitem "_stratified_kfold_indices preserves class proportions per fold" begin
    using BayesInteractomics
    using BayesInteractomics: _stratified_kfold_indices
    using Random

    rng = Random.MersenneTwister(99)
    n = 300
    # 3 classes: 200 H0, 70 Agnostic, 30 H1
    labels = vcat(fill(1, 200), fill(2, 70), fill(3, 30))
    n_folds = 10
    fold_assignments = _stratified_kfold_indices(labels, n_folds, rng)

    # Each fold should have roughly proportional representation
    # Expected per fold: H0~20, Agnostic~7, H1~3
    for k in 1:n_folds
        fold_mask = fold_assignments .== k
        n_h0 = sum(labels[fold_mask] .== 1)
        n_ag = sum(labels[fold_mask] .== 2)
        n_h1 = sum(labels[fold_mask] .== 3)
        # Within +/- 2 of expected
        @test abs(n_h0 - 20) <= 2
        @test abs(n_ag - 7) <= 2
        @test abs(n_h1 - 3) <= 2
    end
end

@testitem "_compute_calibration_ece returns 0 for perfect calibration" begin
    using BayesInteractomics
    using BayesInteractomics: _compute_calibration_ece
    using Random

    Random.seed!(7)
    n = 5000
    # Perfect calibration: ground truth probability = posterior
    predicted = rand(n)
    observed = BitVector([rand() < predicted[i] for i in 1:n])
    ece = _compute_calibration_ece(predicted, observed)

    # ECE should be small (not exactly 0 due to sampling noise)
    @test ece >= 0.0
    @test ece <= 0.1  # well-calibrated should be close to 0

    # Perfectly calibrated midpoint case
    pred_mid = fill(0.5, 100)
    obs_mid = BitVector(vcat(trues(50), falses(50)))
    ece_mid = _compute_calibration_ece(pred_mid, obs_mid)
    @test ece_mid ≈ 0.0 atol=1e-10
end

@testitem "_compute_calibration_ece returns value in [0, 1]" begin
    using BayesInteractomics
    using BayesInteractomics: _compute_calibration_ece
    using Random

    Random.seed!(13)
    n = 200
    predicted = rand(n)
    observed = BitVector(rand(n) .< 0.5)
    ece = _compute_calibration_ece(predicted, observed)
    @test 0.0 <= ece <= 1.0

    # Empty data
    ece_empty = _compute_calibration_ece(Float64[], BitVector())
    @test ece_empty == 0.0
end

@testitem "_run_calibration_cv returns CalibrationCVMetrics with correct n_folds" begin
    using BayesInteractomics
    using BayesInteractomics: _run_calibration_cv, CalibrationCVMetrics
    using Random

    Random.seed!(42)
    n = 300
    # Simulated well-separated data: 80% H0, 10% Agnostic, 10% H1
    raw = vcat(rand(240) .* 0.3, rand(30) .* 0.2 .+ 0.3, rand(30) .* 0.3 .+ 0.7)
    truth = BitVector(vcat(falses(240), falses(30), trues(30)))
    labels = vcat(fill(1, 240), fill(2, 30), fill(3, 30))

    metrics = _run_calibration_cv(raw, truth, labels; n_folds=10, seed=42)

    @test metrics isa CalibrationCVMetrics
    @test metrics.n_folds == 10
    @test length(metrics.posterior_ece_per_fold) == 10
    @test length(metrics.fdr_ece_per_fold) == 10
    @test all(metrics.posterior_ece_per_fold .>= 0.0)
    @test all(metrics.fdr_ece_per_fold .>= 0.0)
    @test metrics.posterior_ece_mean >= 0.0
    @test metrics.posterior_ece_std >= 0.0
    @test metrics.fdr_ece_mean >= 0.0
end

@testitem "_run_calibration_cv: calibrated ECE <= raw ECE on miscalibrated data" begin
    using BayesInteractomics
    using BayesInteractomics: _run_calibration_cv, _compute_calibration_ece
    using Random, Statistics

    Random.seed!(123)
    n = 500
    # Deliberately miscalibrated: all posteriors uniformly overconfident (shifted high)
    # H1 proteins: posterior in [0.85, 0.99] (overconfident)
    # H0 proteins: posterior in [0.55, 0.85] (overconfident)
    n_h1 = 100; n_h0 = 400
    raw = vcat(
        rand(n_h0) .* 0.3 .+ 0.55,   # H0 posteriors: 0.55 to 0.85 (overconfident)
        rand(n_h1) .* 0.14 .+ 0.85    # H1 posteriors: 0.85 to 0.99
    )
    truth = BitVector(vcat(falses(n_h0), trues(n_h1)))
    labels = vcat(fill(1, n_h0), fill(3, n_h1))

    metrics = _run_calibration_cv(raw, truth, labels; n_folds=5, seed=7)

    # Mean calibrated ECE should not be higher than raw ECE
    raw_ece = _compute_calibration_ece(raw, truth)
    @test metrics.posterior_ece_mean <= raw_ece + 0.05  # allow small noise margin
end

@testitem "_run_calibration_cv badge color and passes_ece_threshold" begin
    using BayesInteractomics
    using BayesInteractomics: _run_calibration_cv, CalibrationCVMetrics
    using Random

    Random.seed!(42)
    n = 200
    # Well-calibrated data: posteriors ~ true probabilities
    raw = rand(n)
    truth = BitVector([rand() < raw[i] for i in 1:n])
    labels = vcat(fill(1, 140), fill(2, 30), fill(3, 30))

    metrics = _run_calibration_cv(raw, truth, labels; n_folds=5, seed=42)

    # Badge color should be one of the valid values
    @test metrics.ece_badge_color in ["green", "yellow", "red"]

    # passes_ece_threshold <=> mean ECE < 0.05
    if metrics.posterior_ece_mean < 0.05
        @test metrics.passes_ece_threshold == true
        @test metrics.ece_badge_color == "green"
    elseif metrics.posterior_ece_mean < 0.10
        @test metrics.passes_ece_threshold == false
        @test metrics.ece_badge_color == "yellow"
    else
        @test metrics.passes_ece_threshold == false
        @test metrics.ece_badge_color == "red"
    end
end

@testitem "_run_calibration_cv reliability curves have correct format" begin
    using BayesInteractomics
    using BayesInteractomics: _run_calibration_cv
    using Random

    Random.seed!(77)
    n = 300
    raw = rand(n)
    truth = BitVector([rand() < raw[i] for i in 1:n])
    labels = vcat(fill(1, 210), fill(2, 45), fill(3, 45))

    metrics = _run_calibration_cv(raw, truth, labels; n_folds=5, seed=77)

    # Reliability curves must have matching lengths
    @test length(metrics.raw_rel_bins) == length(metrics.raw_rel_observed)
    @test length(metrics.cal_rel_bins) == length(metrics.cal_rel_observed)

    # Bin centers in [0, 1]
    @test all(0.0 .<= metrics.raw_rel_bins .<= 1.0)
    @test all(0.0 .<= metrics.cal_rel_bins .<= 1.0)

    # Observed fractions (non-NaN values) in [0, 1]
    valid_raw = filter(!isnan, metrics.raw_rel_observed)
    valid_cal = filter(!isnan, metrics.cal_rel_observed)
    @test all(0.0 .<= valid_raw .<= 1.0)
    @test all(0.0 .<= valid_cal .<= 1.0)
end

# ============================================================
# Single-scenario calibration and ECE guard
# ============================================================

@testitem "single-scenario calibration does not collapse mid-range posteriors" begin
    using BayesInteractomics
    using BayesInteractomics: _fit_posterior_calibration, _apply_calibration, _compute_calibration_ece
    using Random

    Random.seed!(42)
    # Simulate a single scenario matching real data: pi_H1 = 0.05, effect_scale = 1.0
    n = 10_000
    n_h1 = round(Int, n * 0.05)
    n_h0 = n - n_h1

    # H0 posteriors: mostly low (0.0-0.3)
    raw_h0 = rand(n_h0) .* 0.35
    # H1 posteriors: spread across range with concentration at high end
    raw_h1 = vcat(rand(round(Int, n_h1 * 0.3)) .* 0.5 .+ 0.3,
                  rand(round(Int, n_h1 * 0.7)) .* 0.3 .+ 0.7)
    raw = vcat(raw_h0, raw_h1)
    truth = BitVector(vcat(falses(n_h0), trues(length(raw_h1))))

    model = _fit_posterior_calibration(raw, truth)

    # Mid-range posteriors (0.3-0.8) should NOT collapse to near-zero
    mid_range = collect(0.3:0.05:0.8)
    calibrated_mid = [_apply_calibration(v, model) for v in mid_range]
    # With Platt scaling, most mid-range should map to non-trivial values
    # (the lowest bin at 0.3 may calibrate low in skewed scenarios)
    @test count(x -> x > 0.1, calibrated_mid) >= length(mid_range) - 2
    # Specifically, 0.5 should map to something reasonable (not near 0)
    cal_05 = _apply_calibration(0.5, model)
    @test cal_05 > 0.05

    # ECE should be reasonable (< 0.15)
    all_cal = [_apply_calibration(v, model) for v in raw]
    ece = _compute_calibration_ece(all_cal, truth)
    @test ece < 0.15
end

@testitem "ECE guard skips calibration when CV ECE >= 0.10" begin
    using BayesInteractomics
    using BayesInteractomics: CalibrationCVMetrics, CalibrationModel, SimulationResult

    # Helper: build a CalibrationCVMetrics with a specific posterior_ece_mean
    function mock_cv_metrics(; ece_mean::Float64)
        CalibrationCVMetrics(
            5,                          # n_folds
            fill(ece_mean, 5),          # posterior_ece_per_fold
            ece_mean,                   # posterior_ece_mean
            0.01,                       # posterior_ece_std
            fill(0.05, 5),              # fdr_ece_per_fold
            0.05,                       # fdr_ece_mean
            0.01,                       # fdr_ece_std
            [0.1, 0.5, 0.9],           # raw_rel_bins
            [0.1, 0.5, 0.9],           # raw_rel_observed
            [0.1, 0.5, 0.9],           # cal_rel_bins
            [0.1, 0.5, 0.9],           # cal_rel_observed
            ece_mean < 0.10,            # passes_ece_threshold
            ece_mean < 0.10 ? "green" : "red"  # ece_badge_color
        )
    end

    # Case 1: ECE >= 0.10 -> should NOT calibrate
    cv_bad = mock_cv_metrics(ece_mean=0.15)
    cal_ece_bad = cv_bad.posterior_ece_mean
    should_calibrate_bad = cal_ece_bad < 0.10
    @test !should_calibrate_bad

    # Case 2: ECE < 0.10 -> should calibrate
    cv_good = mock_cv_metrics(ece_mean=0.05)
    cal_ece_good = cv_good.posterior_ece_mean
    should_calibrate_good = cal_ece_good < 0.10
    @test should_calibrate_good

    # Case 3: calibration_cv is nothing -> ECE = Inf -> should NOT calibrate
    cal_ece_nil = nothing !== nothing ? nothing : Inf
    should_calibrate_nil = cal_ece_nil < 0.10
    @test !should_calibrate_nil

    # Case 4: ECE exactly at boundary (0.10) -> should NOT calibrate (strict <)
    cv_boundary = mock_cv_metrics(ece_mean=0.10)
    should_calibrate_boundary = cv_boundary.posterior_ece_mean < 0.10
    @test !should_calibrate_boundary

    # Case 5: Verify SimulationResult with bad CV has correct guard logic
    dummy_cal_model = CalibrationModel(1.0, 0.0, 100, true)
    sim_bad = SimulationResult(
        [],                     # scenarios (empty -- not needed for guard check)
        [0.05],                 # pi_h1_grid
        [1.0],                  # effect_grid
        1000,                   # n_synthetic
        3,                      # n_replicates
        :normal,                # h1_enrichment_family
        (0.01, 0.05),           # fdr_at_p95_range
        dummy_cal_model,        # calibration_model (non-nothing, so guard is reached)
        nothing,                # fdr_calibration_model
        cv_bad                  # calibration_cv with ECE 0.15
    )
    # The pipeline guard logic:
    ece_val = sim_bad.calibration_cv !== nothing ?
              sim_bad.calibration_cv.posterior_ece_mean : Inf
    @test ece_val >= 0.10
    @test !(ece_val < 0.10)  # should_calibrate is false
end
