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
    @test config.q_threshold == 0.05
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
        q_threshold = 0.01,
        classification_method = :dbf,
        volcano_file = "out/volcano.svg"
    )
    @test config2.posterior_threshold == 0.9
    @test config2.q_threshold == 0.01
    @test config2.classification_method == :dbf
    @test config2.volcano_file == "out/volcano.svg"

    # Invalid parameters
    @test_throws ArgumentError DifferentialConfig(posterior_threshold = -0.1)
    @test_throws ArgumentError DifferentialConfig(posterior_threshold = 1.5)
    @test_throws ArgumentError DifferentialConfig(q_threshold = 0.0)
    @test_throws ArgumentError DifferentialConfig(q_threshold = 1.5)
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

    result_A = _make_mock_result(proteins = ["P1", "P2", "P3"], seed = 1)
    result_B = _make_mock_result(proteins = ["P2", "P3", "P4"], seed = 2)

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
    @test diff.n_condition_A_specific == 2
    @test diff.n_condition_B_specific == 2
    @test nrow(diff.results) == 4
    @test all(c -> c in (CONDITION_A_SPECIFIC, CONDITION_B_SPECIFIC), diff.results.classification)
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
        q_threshold = 1.0  # Loose to ensure significance
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
        q_threshold = 1.0
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
    @test occursin("Lost", output)
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

    config = DifferentialConfig(posterior_threshold = 0.8, q_threshold = 1.0)
    diff = differential_analysis(result_A, result_B, config = config)

    @test nrow(gained_interactions(diff)) >= 1
    @test nrow(lost_interactions(diff)) >= 1

    unch = unchanged_interactions(diff)
    @test all(==(UNCHANGED), unch.classification)

    sig = significant_differential(diff, q_threshold = 1.0)
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
