# Tests for empirical FDR curve computation in simulation engine
# FDR calibration fix

@testitem "fdr curve: _compute_simulation_fdr_curves returns two length-100 vectors" begin
    using BayesInteractomics
    # 20 proteins: first 10 are true interactors
    posteriors = vcat(fill(0.9, 10), fill(0.1, 10))
    ground_truth = BitVector(vcat(fill(true, 10), fill(false, 10)))
    fdr_emp, declared_bfdr = BayesInteractomics._compute_simulation_fdr_curves(posteriors, ground_truth)
    @test length(fdr_emp) == 100
    @test length(declared_bfdr) == 100
    @test eltype(fdr_emp) == Float64
    @test eltype(declared_bfdr) == Float64
end

@testitem "fdr curve: perfect separation gives correct values" begin
    using BayesInteractomics
    # Perfect case: interactors have posterior=1.0, non-interactors have 0.0
    posteriors = vcat(fill(1.0, 10), fill(0.0, 10))
    ground_truth = BitVector(vcat(fill(true, 10), fill(false, 10)))
    fdr_emp, declared_bfdr = BayesInteractomics._compute_simulation_fdr_curves(posteriors, ground_truth)

    # At threshold 0.5 (index ~51 of 100): only interactors are above, so FDR=0
    idx_05 = findfirst(x -> x >= 0.5, collect(range(0.0, 1.0, length=100)))
    @test fdr_emp[idx_05] == 0.0

    # At threshold 0.0 (index 1): all proteins called positive
    # declared_bfdr = mean(1-p) for all proteins = mean([0.0,...,1.0,...]) = 0.5
    @test declared_bfdr[1] ≈ 0.5
end

@testitem "fdr curve: declared_bfdr is NaN when no proteins exceed threshold" begin
    using BayesInteractomics
    # All posteriors are low
    posteriors = fill(0.1, 20)
    ground_truth = BitVector(fill(false, 20))
    _, declared_bfdr = BayesInteractomics._compute_simulation_fdr_curves(posteriors, ground_truth)

    # At high thresholds (near 1.0), no proteins exceed -> NaN
    @test isnan(declared_bfdr[end])  # threshold ≈ 1.0
    # At low threshold (0.0), all proteins are included -> not NaN
    @test !isnan(declared_bfdr[1])
end

@testitem "fdr curve: fdr_empirical is 0.0 when no predicted positives" begin
    using BayesInteractomics
    # All posteriors are 0.0, so at any threshold > 0, no predicted positives
    posteriors = fill(0.0, 20)
    ground_truth = BitVector(vcat(fill(true, 10), fill(false, 10)))
    fdr_emp, _ = BayesInteractomics._compute_simulation_fdr_curves(posteriors, ground_truth)

    # At high thresholds, FDR should be 0.0 (no positives means 0 FP, FDR=0)
    @test fdr_emp[end] == 0.0
end

@testitem "fdr curve: SimulationResult backward-compat constructors" begin
    using BayesInteractomics
    # 7-arg constructor (original convenience)
    sr7 = BayesInteractomics.SimulationResult(
        BayesInteractomics.ScenarioResult[],
        Float64[0.1], Float64[1.0],
        100, 5, :Gamma, (0.01, 0.1)
    )
    @test sr7.fdr_curve_empirical == Float64[]
    @test sr7.fdr_curve_declared_bfdr == Float64[]

    # 10-arg constructor (old full constructor with cal fields)
    sr10 = BayesInteractomics.SimulationResult(
        BayesInteractomics.ScenarioResult[],
        Float64[0.1], Float64[1.0],
        100, 5, :Gamma, (0.01, 0.1),
        nothing, nothing, nothing
    )
    @test sr10.fdr_curve_empirical == Float64[]
    @test sr10.fdr_curve_declared_bfdr == Float64[]

    # 12-arg constructor (new full constructor with FDR curves)
    fdr_emp = collect(range(0.0, 0.5, length=100))
    fdr_decl = collect(range(0.0, 1.0, length=100))
    sr12 = BayesInteractomics.SimulationResult(
        BayesInteractomics.ScenarioResult[],
        Float64[0.1], Float64[1.0],
        100, 5, :Gamma, (0.01, 0.1),
        nothing, nothing, nothing,
        fdr_emp, fdr_decl
    )
    @test sr12.fdr_curve_empirical == fdr_emp
    @test sr12.fdr_curve_declared_bfdr == fdr_decl
end

@testitem "fdr curve: save/load round-trip preserves FDR curve fields" begin
    using BayesInteractomics
    using JLD2

    fdr_emp = collect(range(0.0, 0.5, length=100))
    fdr_decl = collect(range(0.0, 1.0, length=100))

    # Create a minimal ScenarioResult
    sc = BayesInteractomics.ScenarioResult(
        0.1, 1.0,
        collect(range(0.0, 1.0, length=200)),
        zeros(200), zeros(200), zeros(200),  # fdr
        zeros(200), zeros(200), zeros(200),  # sensitivity
        zeros(200), zeros(200), zeros(200),  # specificity
        0.5, 0.4, 0.6,                       # auc
        [0.1, 0.5, 0.9], [0.1, 0.5, 0.9], [0.05, 0.4, 0.8], [0.15, 0.6, 0.95]  # reliability
    )

    sr = BayesInteractomics.SimulationResult(
        [sc],
        Float64[0.1], Float64[1.0],
        100, 5, :Gamma, (0.01, 0.1),
        nothing, nothing, nothing,
        fdr_emp, fdr_decl
    )

    tmpdir = mktempdir()
    filepath = joinpath(tmpdir, "test_sim_fdr.jld2")
    param_hash = UInt64(12345)

    BayesInteractomics.save_simulation_cache(sr, param_hash, filepath)
    loaded = BayesInteractomics.load_simulation_cache(filepath, param_hash)

    @test !isnothing(loaded)
    @test loaded.fdr_curve_empirical ≈ fdr_emp
    @test loaded.fdr_curve_declared_bfdr ≈ fdr_decl

    # Also test loading old cache without FDR curves (backward compat)
    # Load with wrong hash to get nothing (simulates missing field path)
    loaded_wrong = BayesInteractomics.load_simulation_cache(filepath, UInt64(99999))
    @test isnothing(loaded_wrong)

    rm(tmpdir; recursive=true)
end
