"""
    test_differential.jl

Tests for differential interaction analysis: types, core logic, convenience functions, and visualization.
"""

# =========================================================
# Type Tests (no mock needed)
# =========================================================

@testitem "DifferentialConfig construction and validation" begin
    using BayesInteractomics

    # Default construction
    config = DifferentialConfig()
    @test config.posterior_threshold == 0.8
    @test config.bfdr_threshold == 0.05
    @test config.delta_log2fc_threshold == 1.0
    @test config.dbf_threshold == 1.0
    @test config.classification_method == :posterior

    # Default output paths
    @test config.results_file == "differential_results.xlsx"
    @test config.volcano_file == "differential_volcano.png"
    @test config.evidence_file == "differential_evidence.png"
    @test config.scatter_file == "differential_scatter.png"

    # Custom construction
    config2 = DifferentialConfig(
        posterior_threshold = 0.9,
        bfdr_threshold = 0.01,
        classification_method = :dbf,
        volcano_file = "out/volcano.svg"
    )
    @test config2.posterior_threshold == 0.9
    @test config2.bfdr_threshold == 0.01
    @test config2.classification_method == :dbf
    @test config2.volcano_file == "out/volcano.svg"

    # Invalid parameters
    @test_throws ArgumentError DifferentialConfig(posterior_threshold = -0.1)
    @test_throws ArgumentError DifferentialConfig(posterior_threshold = 1.5)
    @test_throws ArgumentError DifferentialConfig(bfdr_threshold = 0.0)
    @test_throws ArgumentError DifferentialConfig(bfdr_threshold = 1.5)
    @test_throws ArgumentError DifferentialConfig(delta_log2fc_threshold = -1.0)
    @test_throws ArgumentError DifferentialConfig(dbf_threshold = -0.5)
    @test_throws ArgumentError DifferentialConfig(classification_method = :invalid)
end

@testitem "InteractionClass enum values" begin
    using BayesInteractomics

    @test GAINED isa InteractionClass
    @test REDUCED isa InteractionClass
    @test UNCHANGED isa InteractionClass
    @test BOTH_NEGATIVE isa InteractionClass
    @test CONDITION_A_SPECIFIC isa InteractionClass
    @test CONDITION_B_SPECIFIC isa InteractionClass

    # All six values are distinct
    all_values = [GAINED, REDUCED, UNCHANGED, BOTH_NEGATIVE, CONDITION_A_SPECIFIC, CONDITION_B_SPECIFIC]
    @test length(unique(all_values)) == 6
end

# =========================================================
# Core Analysis Tests
# =========================================================

@testitem "Differential analysis with identical conditions" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))

    result = _make_mock_result(seed = 42)

    diff = differential_analysis(result, result,
        condition_A = "WT", condition_B = "WT_copy")

    @test diff isa DifferentialResult
    @test diff.condition_A == "WT"
    @test diff.condition_B == "WT_copy"
    @test diff.n_shared == 10
    @test diff.n_condition_A_specific == 0
    @test diff.n_condition_B_specific == 0

    # dBF should be exactly 1.0 (log10_dbf == 0)
    @test all(x -> isapprox(x, 0.0, atol = 1e-10), diff.results.log10_dbf)

    # delta_log2fc should be exactly 0
    @test all(x -> isapprox(x, 0.0, atol = 1e-10), diff.results.delta_log2fc)

    # All should be UNCHANGED
    @test all(==(UNCHANGED), diff.results.classification)
    @test diff.n_gained == 0
    @test diff.n_reduced == 0
    @test diff.n_unchanged == 10
end

@testitem "Differential analysis with non-overlapping proteins" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))

    # Use high BFs so that q-values (BFDR) are low enough for classification
    result_A = _make_mock_result(proteins = ["P1", "P2", "P3"],
        bfs = [50.0, 80.0, 90.0], seed = 1)
    result_B = _make_mock_result(proteins = ["P2", "P3", "P4"],
        bfs = [80.0, 90.0, 50.0], seed = 2)

    diff = differential_analysis(result_A, result_B,
        condition_A = "A", condition_B = "B")

    @test diff.n_shared == 2  # P2, P3
    @test diff.n_condition_A_specific == 1  # P1
    @test diff.n_condition_B_specific == 1  # P4
    @test nrow(diff.results) == 4

    # Check condition-specific classifications
    p1_row = diff["P1"]
    @test p1_row.classification == CONDITION_A_SPECIFIC
    @test isnan(p1_row.bf_B)
    @test !isnan(p1_row.bf_A)

    p4_row = diff["P4"]
    @test p4_row.classification == CONDITION_B_SPECIFIC
    @test isnan(p4_row.bf_A)
    @test !isnan(p4_row.bf_B)
end

@testitem "Differential analysis with completely disjoint proteins" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))

    result_A = _make_mock_result(proteins = ["P1", "P2"], seed = 1)
    result_B = _make_mock_result(proteins = ["P3", "P4"], seed = 2)

    diff = differential_analysis(result_A, result_B)

    @test diff.n_shared == 0
    @test nrow(diff.results) == 4
    # Condition-specific proteins with low PP now get BOTH_NEGATIVE instead of
    # CONDITION_A/B_SPECIFIC (Issue 2 fix), so valid classes include BOTH_NEGATIVE
    @test all(c -> c in (CONDITION_A_SPECIFIC, CONDITION_B_SPECIFIC, BOTH_NEGATIVE),
              diff.results.classification)
end

@testitem "Differential analysis classifies gained/reduced correctly" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))

    proteins = ["P_gained", "P_reduced", "P_unchanged"]

    result_A = _make_mock_result(
        proteins = proteins,
        bfs = [100.0, 0.5, 50.0],
        posteriors = [0.99, 0.1, 0.95],
        log2fcs = [3.0, 0.2, 2.0],
        seed = 1
    )

    result_B = _make_mock_result(
        proteins = proteins,
        bfs = [0.5, 100.0, 50.0],
        posteriors = [0.1, 0.99, 0.95],
        log2fcs = [0.2, 3.0, 2.0],
        seed = 2
    )

    config = DifferentialConfig(
        posterior_threshold = 0.8,
        bfdr_threshold = 1.0  # Loose to ensure significance
    )

    diff = differential_analysis(result_A, result_B, config = config)

    p_gained  = diff["P_gained"]
    p_reduced = diff["P_reduced"]

    @test p_gained.classification == GAINED
    @test p_reduced.classification == REDUCED

    # dBF direction
    @test p_gained.log10_dbf > 0
    @test p_reduced.log10_dbf < 0

    # delta_log2fc direction
    @test p_gained.delta_log2fc > 0
    @test p_reduced.delta_log2fc < 0
end

@testitem "differential_posterior is direction-symmetric (REDUCED not suppressed)" begin
    # Regression test for the direction-bias bug: differential_posterior was computed as
    # |dBF| / (1 + |dBF|). Since dBF = 10^log10_dbf is always > 0, a protein reduced in A
    # (dBF << 1) collapsed to differential_posterior ~ 0, never reached significance, and
    # REDUCED was never called. The fix uses the SYMMETRIC two-sided fold 10^|log10_dbf|.
    # NOTE: a realistic bfdr_threshold (0.05) is essential here — the older gained/reduced
    # test used bfdr_threshold = 1.0, which forced significance and masked the bug.
    include(joinpath(@__DIR__, "mock_helper.jl"))

    proteins = ["P_gained", "P_reduced"]
    result_A = _make_mock_result(proteins = proteins,
        bfs = [1.0e4, 1.0e-2], posteriors = [0.999, 0.01],
        log2fcs = [4.0, -4.0], seed = 1)
    result_B = _make_mock_result(proteins = proteins,
        bfs = [1.0e-2, 1.0e4], posteriors = [0.01, 0.999],
        log2fcs = [-4.0, 4.0], seed = 2)

    diff = differential_analysis(result_A, result_B,
        config = DifferentialConfig(bfdr_threshold = 0.05,
                                    classification_method = :dbf))

    g = diff["P_gained"]
    r = diff["P_reduced"]

    # Core of the bug: evidence magnitude is symmetric — both gained AND reduced get a
    # HIGH differential_posterior (the reduced protein no longer collapses below 0.5).
    @test g.differential_posterior > 0.9
    @test r.differential_posterior > 0.9
    @test isapprox(g.differential_posterior, r.differential_posterior; atol = 1e-6)

    # Direction is still resolved correctly from the sign of log10_dbf.
    @test g.log10_dbf > 0
    @test r.log10_dbf < 0

    # End-to-end symptom fixed: REDUCED is now reachable at a realistic threshold.
    @test g.classification == GAINED
    @test r.classification == REDUCED
    @test diff.n_reduced >= 1
end

@testitem "Differential analysis with :dbf classification method" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))

    proteins = ["P_high_dbf", "P_low_dbf"]

    result_A = _make_mock_result(
        proteins = proteins,
        bfs = [1000.0, 2.0],
        posteriors = [0.99, 0.6],
        log2fcs = [4.0, 0.5],
        seed = 1
    )

    result_B = _make_mock_result(
        proteins = proteins,
        bfs = [1.0, 1.0],
        posteriors = [0.5, 0.5],
        log2fcs = [0.0, 0.0],
        seed = 2
    )

    config = DifferentialConfig(
        classification_method = :dbf,
        dbf_threshold = 1.0,
        bfdr_threshold = 1.0
    )

    diff = differential_analysis(result_A, result_B, config = config)

    # P_high_dbf: BF ratio = 1000, log10(dBF) = 3 > 1
    @test diff["P_high_dbf"].classification == GAINED

    # P_low_dbf: BF ratio = 2, log10(dBF) ~ 0.3 < 1
    @test diff["P_low_dbf"].classification == UNCHANGED
end

# =========================================================
# Accessors and Interface Tests
# =========================================================

@testitem "DifferentialResult accessors and iteration" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))
    using BayesInteractomics: getDifferentialBayesFactors, getDifferentialPosteriors,
        getDifferentialQValues, getClassifications, getDeltaLog2FC, getProteins

    diff = differential_analysis(
        _make_mock_result(seed = 1),
        _make_mock_result(seed = 2)
    )

    # Length
    @test length(diff) == 10

    # Iteration
    items = collect(diff)
    @test length(items) == 10
    @test items[1][1] isa String
    @test haskey(items[1][2], :dbf)
    @test haskey(items[1][2], :classification)

    # Integer indexing
    row1 = diff[1]
    @test haskey(row1, :dbf)
    @test haskey(row1, :classification)
    @test haskey(row1, :differential_posterior)

    # String indexing
    row_p1 = diff["P1"]
    @test row_p1.Protein == "P1"

    # Accessors
    @test length(getDifferentialBayesFactors(diff)) == 10
    @test length(getDifferentialPosteriors(diff)) == 10
    @test length(getDifferentialQValues(diff)) == 10
    @test length(getClassifications(diff)) == 10
    @test length(getDeltaLog2FC(diff)) == 10
    @test length(getProteins(diff)) == 10
end

@testitem "DifferentialResult display" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))

    diff = differential_analysis(
        _make_mock_result(seed = 1),
        _make_mock_result(seed = 2),
        condition_A = "WT", condition_B = "KO"
    )

    io = IOBuffer()
    show(io, diff)
    output = String(take!(io))

    @test occursin("DifferentialResult", output)
    @test occursin("WT", output)
    @test occursin("KO", output)
    @test occursin("Shared proteins", output)
    @test occursin("Gained", output)
    @test occursin("Reduced", output)
end

# =========================================================
# Convenience Function Tests
# =========================================================

@testitem "Convenience filter functions" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))

    proteins = ["P_gained", "P_lost", "P_unchanged"]

    result_A = _make_mock_result(
        proteins = proteins,
        bfs = [100.0, 0.5, 50.0],
        posteriors = [0.99, 0.1, 0.95],
        log2fcs = [3.0, 0.2, 2.0]
    )
    result_B = _make_mock_result(
        proteins = proteins,
        bfs = [0.5, 100.0, 50.0],
        posteriors = [0.1, 0.99, 0.95],
        log2fcs = [0.2, 3.0, 2.0],
        seed = 43
    )

    config = DifferentialConfig(posterior_threshold = 0.8, bfdr_threshold = 1.0)
    diff = differential_analysis(result_A, result_B, config = config)

    @test nrow(gained_interactions(diff)) >= 1
    @test nrow(lost_interactions(diff)) >= 1

    unch = unchanged_interactions(diff)
    @test all(==(UNCHANGED), unch.classification)

    sig = significant_differential(diff, bfdr_threshold = 1.0)
    @test nrow(sig) >= 0
end

# =========================================================
# Visualization Tests
# =========================================================

@testitem "Differential volcano plot does not error" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))
    using StatsPlots

    diff = differential_analysis(
        _make_mock_result(; proteins = ["P$i" for i in 1:20], seed = 1),
        _make_mock_result(; proteins = ["P$i" for i in 1:20], seed = 2)
    )

    plt1 = differential_volcano_plot(diff)
    @test plt1 !== nothing

    plt2 = differential_volcano_plot(diff, x_axis = :delta_log2fc)
    @test plt2 !== nothing

    plt3 = differential_volcano_plot(diff, y_axis = :differential_posterior)
    @test plt3 !== nothing

    @test_throws ArgumentError differential_volcano_plot(diff, x_axis = :invalid)
    @test_throws ArgumentError differential_volcano_plot(diff, y_axis = :invalid)
end

@testitem "Differential evidence and scatter plots do not error" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))
    using StatsPlots

    diff = differential_analysis(
        _make_mock_result(; proteins = ["P$i" for i in 1:20], seed = 1),
        _make_mock_result(; proteins = ["P$i" for i in 1:20], seed = 2)
    )

    plt_ev = differential_evidence_plot(diff)
    @test plt_ev !== nothing

    plt_s1 = differential_scatter_plot(diff, metric = :posterior_prob)
    @test plt_s1 !== nothing

    plt_s2 = differential_scatter_plot(diff, metric = :bf)
    @test plt_s2 !== nothing

    plt_s3 = differential_scatter_plot(diff, metric = :log2fc)
    @test plt_s3 !== nothing

    @test_throws ArgumentError differential_scatter_plot(diff, metric = :invalid)
end

# =========================================================
# Export Test
# =========================================================

@testitem "Export differential results to XLSX" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))

    diff = differential_analysis(
        _make_mock_result(seed = 1),
        _make_mock_result(seed = 2),
        condition_A = "WT", condition_B = "KO"
    )

    mktempdir() do tmpdir
        filepath = joinpath(tmpdir, "diff_results.xlsx")
        export_differential(diff, filepath)
        @test isfile(filepath)
    end
end

# =========================================================
# Metalearner Stripping Tests (Issue 3)
# =========================================================

@testitem "_extract_copula_df strips metalearner and returns BF/(1+BF) posteriors" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))
    using BayesInteractomics: _extract_copula_df

    # Create a mock result where posterior_prob differs from BF/(1+BF)
    # (simulating metalearner-adjusted posteriors)
    bfs = [10.0, 100.0, 0.5]
    metalearner_posteriors = [0.99, 0.999, 0.8]  # NOT BF/(1+BF)
    result = _make_mock_result(
        proteins = ["P1", "P2", "P3"],
        bfs = bfs,
        posteriors = metalearner_posteriors,
        seed = 42
    )

    df = _extract_copula_df(result)

    # posterior_prob should be recomputed as BF/(1+BF), NOT the metalearner values
    for i in 1:3
        expected_pp = bfs[i] / (1.0 + bfs[i])
        @test isapprox(df.posterior_prob[i], expected_pp, atol=1e-10)
    end

    # BFDR values should also be recomputed (not the original random values)
    @test length(df.BFDR) == 3
    @test all(x -> !ismissing(x), df.BFDR)
    # PEP column should also be present
    @test length(df.PEP) == 3
end

# =========================================================
# Condition-Specific Labeling Tests (Issue 2)
# =========================================================

@testitem "Condition-specific protein with low PP gets BOTH_NEGATIVE" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))

    # P1 only in A with LOW BF (posterior well below 0.8)
    # P2 only in B with LOW BF
    # P3 only in A with HIGH BF (posterior above 0.8)
    result_A = _make_mock_result(
        proteins = ["P1", "P3"],
        bfs = [0.1, 100.0],           # P1: pp=0.09, P3: pp=0.99
        posteriors = [0.09, 0.99],
        log2fcs = [0.5, 3.0],
        seed = 1
    )
    result_B = _make_mock_result(
        proteins = ["P2"],
        bfs = [0.2],                   # P2: pp=0.17
        posteriors = [0.17],
        log2fcs = [0.3],
        seed = 2
    )

    config = DifferentialConfig(posterior_threshold = 0.8)
    diff = differential_analysis(result_A, result_B, config = config)

    # P1 is in A only but has low PP → BOTH_NEGATIVE (not CONDITION_A_SPECIFIC)
    @test diff["P1"].classification == BOTH_NEGATIVE

    # P2 is in B only but has low PP → BOTH_NEGATIVE (not CONDITION_B_SPECIFIC)
    @test diff["P2"].classification == BOTH_NEGATIVE

    # P3 is in A only with high PP → still CONDITION_A_SPECIFIC
    @test diff["P3"].classification == CONDITION_A_SPECIFIC
end

# =========================================================
# Z-Score Standardization Tests (Issue 5)
# =========================================================

@testitem "Z-score standardization produces zero delta for equal-rank proteins" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))

    # Same rank/distribution but different raw scales
    proteins = ["P1", "P2", "P3"]
    result_A = _make_mock_result(
        proteins = proteins,
        bfs = [10.0, 10.0, 10.0],
        posteriors = [0.9, 0.9, 0.9],
        log2fcs = [2.0, 4.0, 6.0],     # scale A: mean=4, sd=2
        seed = 1
    )
    result_B = _make_mock_result(
        proteins = proteins,
        bfs = [10.0, 10.0, 10.0],
        posteriors = [0.9, 0.9, 0.9],
        log2fcs = [20.0, 40.0, 60.0],  # scale B: mean=40, sd=20 (10x scale)
        seed = 2
    )

    # With z-score standardization: delta_z should be ~0 for all
    config_z = DifferentialConfig(standardize_log2fc = true, bfdr_threshold = 1.0)
    diff_z = differential_analysis(result_A, result_B, config = config_z)

    for i in 1:3
        @test isapprox(diff_z.results.delta_log2fc[i], 0.0, atol = 1e-10)
    end

    # Without standardization: delta should reflect raw difference
    config_raw = DifferentialConfig(standardize_log2fc = false, bfdr_threshold = 1.0)
    diff_raw = differential_analysis(result_A, result_B, config = config_raw)

    # Raw delta = log2fc_A - log2fc_B (e.g., 2.0 - 20.0 = -18.0)
    @test diff_raw.results.delta_log2fc[1] < -10.0
end

# =========================================================
# DifferentialConfig standardize_log2fc Tests
# =========================================================

@testitem "DifferentialConfig standardize_log2fc field" begin
    using BayesInteractomics

    # Default is false (z-scoring would undo the bait-anchor; raw Δlog2FC is the correct default)
    config = DifferentialConfig()
    @test config.standardize_log2fc == false

    # Can set to true
    config2 = DifferentialConfig(standardize_log2fc = true)
    @test config2.standardize_log2fc == true

    # Show method includes the field
    io = IOBuffer()
    show(io, config)
    output = String(take!(io))
    @test occursin("standardize_log2fc", output)
end

# =========================================================
# dBF==1 minimum check test (Issue 3, BFDR pooling)
# =========================================================

@testitem "dBF==1 proteins classified as UNCHANGED not directional" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))

    # Both conditions have identical BFs → dBF = 1, log10_dbf = 0
    proteins = ["P_identical"]
    bfs = [50.0]
    result_A = _make_mock_result(
        proteins = proteins,
        bfs = bfs,
        posteriors = [0.98],    # Strong interactor in A
        log2fcs = [3.0],
        seed = 1
    )
    result_B = _make_mock_result(
        proteins = proteins,
        bfs = bfs,
        posteriors = [0.1],     # Not interactor in B
        log2fcs = [0.5],
        seed = 2
    )

    config = DifferentialConfig(
        posterior_threshold = 0.8,
        bfdr_threshold = 1.0    # Loose to ensure significance
    )
    diff = differential_analysis(result_A, result_B, config = config)

    # dBF == 1 (identical BFs), so despite pp_A > threshold and pp_B < threshold,
    # the protein should be UNCHANGED (not GAINED) due to the dBF minimum check
    @test isapprox(diff["P_identical"].log10_dbf, 0.0, atol = 1e-10)
    @test diff["P_identical"].classification == UNCHANGED
end

# =========================================================
# Per-condition AnalysisResult plumbing
# =========================================================

@testitem "AR-based differential populates analyses" begin
    include(joinpath(@__DIR__, "mock_helper.jl"))
    using BayesInteractomics
    using BayesInteractomics: AnalysisResult, DifferentialResult, DifferentialConfig,
                              condition_labels, getAnalyses

    # Two minimal synthetic AnalysisResult objects (uses 20-arg backward-compat
    # constructor; simulation_result + config default to nothing).
    ar_A = _make_mock_result(seed = 1)
    ar_B = _make_mock_result(seed = 2)

    # Sanity: the two new fields exist on the constructed AR and default to nothing
    # via the backward-compat 20-arg constructor exercised by _make_mock_result.
    @test ar_A.simulation_result === nothing
    @test ar_A.config === nothing

    dcfg = DifferentialConfig()
    diff = differential_analysis(ar_A, ar_B;
        condition_A = "WT", condition_B = "Mut", config = dcfg)

    @test diff.analyses isa Vector{AnalysisResult}
    @test length(diff.analyses) == 2
    @test diff.analyses[1] === ar_A
    @test diff.analyses[2] === ar_B
    @test condition_labels(diff) == ["WT", "Mut"]
    @test getAnalyses(diff) === diff.analyses
end

@testitem "path-based differential leaves analyses=nothing" begin
    using BayesInteractomics
    using BayesInteractomics: DifferentialResult, DifferentialConfig, InteractionClass,
                              condition_labels, getAnalyses
    using DataFrames, Dates

    # Construct via the 14-arg backward-compat constructor (mirrors the path-based
    # overload behaviour where `analyses` is not threaded through).
    results_df = DataFrame(
        Protein = ["P1"],
        dbf = [2.0],
        delta_log2fc = [1.0],
        classification = InteractionClass[BayesInteractomics.UNCHANGED],
    )
    diff = DifferentialResult(
        results_df, "WT", "Mut", DifferentialConfig(),
        1, 1, 1, 0, 0,
        now(), 0, 0, 1, 0,
    )
    @test diff.analyses === nothing
    @test getAnalyses(diff) === nothing
    @test condition_labels(diff) == ["WT", "Mut"]
end

# =========================================================
# dbf_diagnostic column tests
# =========================================================

@testitem "dbf_diagnostic column on differential results" begin
    using BayesInteractomics

    # _compute_dbf_diagnostic is internal — access via Differential submodule namespace
    # (the submodule forwards internals like _safe_log10, _safe_ratio for test access).
    # If not forwarded, qualify as BayesInteractomics.Differential._compute_dbf_diagnostic.
    compute = isdefined(BayesInteractomics, :_compute_dbf_diagnostic) ?
              BayesInteractomics._compute_dbf_diagnostic :
              BayesInteractomics.Differential._compute_dbf_diagnostic

    # Branch :saturated — |log10(bf_A)| > 18
    @test compute(1e19, 1.0, 1.0, 1.0, 1.0, 19.0,
                  missing, missing, missing, missing) === :saturated
    @test compute(1.0, 1e19, 1.0, 1.0, 1.0, -19.0,
                  missing, missing, missing, missing) === :saturated

    # Branch :single_component — dbf_enrichment dominates log10_dbf
    # log10_dbf = 2.0; log10(dbf_e) = 2.0 → 100% > 90%
    @test compute(100.0, 1.0, 100.0, 1.001, 1.001, 2.0,
                  missing, missing, missing, missing) === :single_component

    # Branch :model_disagreement — log10_dbf_em - log10_dbf_cop = 1.0 - (-1.0) = 2.0 > 1.0
    @test compute(10.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  10.0, 1.0, 1.0, 10.0) === :model_disagreement

    # Branch :ok — balanced inputs, no sub-model BFs, no single dominance
    @test compute(2.0, 1.0, 1.4, 1.4, 1.0, 0.3,
                  missing, missing, missing, missing) === :ok

    # Threshold edge: |log10(bf_A)| = 18 exactly → NOT saturated (strict >)
    @test compute(1e18, 1.0, 1.0, 1.0, 1.0, 18.0,
                  missing, missing, missing, missing) !== :saturated
end

# =========================================================
# kgroup_legacy_parity_2group: byte-equality from the LEGACY
# 2-group AR entry point angle. The existing `kgroup_legacy_parity`
# in test_kgroup_pairwise.jl asserts byte-equality at the aggregator
# level. This testitem asserts it end-to-end: the legacy 2-group
# `differential_analysis(ar_A, ar_B; ...)` call produces a DataFrame
# byte-identical to the k=2 NamedTuple call modulo the new BH column.
# Lives in test_differential.jl so it runs alongside the pre-existing
# 2-group testitems for natural regression coverage.
# =========================================================

@testitem "kgroup_legacy_parity_2group: legacy 2-group ≡ k=2 NamedTuple call" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using BayesInteractomics: condition_labels
    using DataFrames

    fx = DifferentialFixtures.create_three_condition_result()

    # ---- Legacy 2-group entry point ----
    d_legacy = differential_analysis(fx.ar_wt, fx.ar_mut1;
                                     condition_A = "wt", condition_B = "mut1")
    df_legacy = d_legacy.results
    @test d_legacy.contrasts == Pair{Symbol, Symbol}[]   # legacy default
    @test d_legacy.pairwise_results === nothing          # legacy default

    # ---- k=2 NamedTuple entry point ----
    d_kgroup = differential_analysis(conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1))
    df_kgroup = d_kgroup.results
    @test d_kgroup.contrasts == [:wt => :mut1]
    @test d_kgroup.pairwise_results !== nothing
    @test haskey(d_kgroup.pairwise_results, :wt => :mut1)

    # ---- byte-equality: results::DataFrame is identical modulo the new BH column
    # and k-group aggregate columns (decision_risk_min + optimal_call_min;
    # only present on k-group wide table per schema uniformity) ----
    @test hasproperty(df_kgroup, :differential_BFDR_pairwise_BH)
    @test !hasproperty(df_legacy, :differential_BFDR_pairwise_BH)
    # k-group aggregate columns are also k-group-only.
    @test hasproperty(df_kgroup, :decision_risk_min)
    @test hasproperty(df_kgroup, :optimal_call_min)
    @test !hasproperty(df_legacy, :decision_risk_min)
    @test !hasproperty(df_legacy, :optimal_call_min)
    df_kgroup_minus_bh = select(df_kgroup, Not([:differential_BFDR_pairwise_BH,
                                                 :decision_risk_min, :optimal_call_min]))
    @test isequal(df_kgroup_minus_bh, df_legacy)

    # ---- Summary fields equal ----
    @test d_legacy.n_gained == d_kgroup.n_gained
    @test d_legacy.n_reduced == d_kgroup.n_reduced
    @test d_legacy.n_unchanged == d_kgroup.n_unchanged
    @test d_legacy.n_both_negative == d_kgroup.n_both_negative
    @test d_legacy.n_shared == d_kgroup.n_shared

    # ---- condition_labels equivalence (both return ["wt", "mut1"]) ----
    @test condition_labels(d_legacy) == ["wt", "mut1"]   # legacy: from condition_A/B
    @test condition_labels(d_kgroup) == ["wt", "mut1"]   # k-group: from contrasts
end
