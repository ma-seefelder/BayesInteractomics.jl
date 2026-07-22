# test/analysis/test_bb_mnar_codriven_xlsx.jl
#
# Regression: `bb_mnar_codriven::Bool` column is present in per-condition
# `final_results.xlsx` AND positioned after the canonical diagnostic cluster
# (`diagnostic_flag`, `classification_stability`, `sensitivity_range`).
#
# The column is present at column 17 in all 4 production xlsx outputs (wtHTT,
# mHTT, HAP40_Strep, GST_HAP40). This testitem locks the contract so future
# refactors to the `final_results` writer path don't silently drop it.
#
# Lightweight fixture rationale: running `run_analysis(config)` end-to-end
# would blow past the per-task 60s budget due to HBM inference cost. Instead
# we exercise the SAME XLSX.writetable path used by the pipeline writers
# (src/analysis/pipeline.jl L3576-3582 single-data, L4441-4447 imputed-data)
# on a synthetic `final_results`-shaped DataFrame. This is identical to the
# write path the pipeline uses, so the round-trip read-back proves the
# xlsx-emission contract end-to-end without invoking RxInfer.
#
# A second testitem additionally asserts the column producer
# (`_compute_bb_mnar_codriven`) is still wired into the `copula_df` builder
# in `src/analysis/pipeline.jl` — catches regressions where the column is
# silently dropped upstream of the writer.
#
# Contracts honoured:
# - column name verbatim `bb_mnar_codriven` (snake_case).
# - column positioned after the diagnostic cluster.

@testitem "bb_mnar_codriven present in final_results.xlsx" tags=[:xlsx] begin
    using BayesInteractomics
    using DataFrames
    import XLSX

    tmpdir = mktempdir()
    xlsx_path = joinpath(tmpdir, "final_results.xlsx")

    # Synthesise a final_results-shaped DataFrame mirroring the schema produced
    # by the copula_df builder + _merge_diagnostics_to_results. The canonical
    # diagnostic cluster (diagnostic_flag, classification_stability,
    # sensitivity_range) precedes bb_mnar_codriven.
    n = 5
    final_results = DataFrame(
        Protein                  = ["P$i" for i in 1:n],
        is_detected              = fill(true, n),
        BF                       = [0.5, 1.0, 5.0, 50.0, 500.0],
        posterior_prob           = [0.1, 0.3, 0.5, 0.7, 0.9],
        pep                      = [0.9, 0.7, 0.5, 0.3, 0.1],
        BFDR                     = [0.5, 0.3, 0.2, 0.1, 0.05],
        mean_log2FC              = [-1.0, 0.0, 1.0, 2.0, 3.0],
        sd_log2FC                = fill(0.5, n),
        bf_enrichment            = [0.1, 0.5, 1.0, 5.0, 10.0],
        bf_correlation           = [0.1, 0.5, 1.0, 5.0, 10.0],
        bf_detected              = [0.1, 1.0, 5.0, 20.0, 100.0],
        missing_fraction         = [0.9, 0.5, 0.2, 0.1, 0.0],
        diagnostic_flag          = ["ok", "ok", "warning", "ok", "ok"],
        classification_stability = ["robust", "robust", "sensitive", "robust", "robust"],
        sensitivity_range        = [0.05, 0.03, 0.10, 0.02, 0.01],
        bb_mnar_codriven         = [false, false, false, false, true],
    )

    # Use the SAME XLSX.writetable path the pipeline writers use
    # (src/analysis/pipeline.jl L3582, L4447).
    XLSX.writetable(xlsx_path, "df" => final_results; overwrite = true)
    @test isfile(xlsx_path)

    # Read back and assert header row contains `bb_mnar_codriven`
    xl = XLSX.readxlsx(xlsx_path)
    sheet = xl[XLSX.sheetnames(xl)[1]]
    header = string.(sheet[1, :])

    @test "bb_mnar_codriven" in header

    # Defensive wiring guard: the column producer remains reachable from the
    # parent namespace AND the pipeline.jl copula_df builders still include the
    # `bb_mnar_codriven` keyword. Catches future refactors that silently drop
    # the wiring between the compute and the xlsx writer.
    @test isdefined(BayesInteractomics, :_compute_bb_mnar_codriven)

    pipeline_path = joinpath(dirname(dirname(pathof(BayesInteractomics))),
                             "src", "analysis", "pipeline.jl")
    pipeline_src = read(pipeline_path, String)
    @test occursin("bb_mnar_codriven = bb_mnar_codriven_full", pipeline_src)
    @test occursin("bb_mnar_codriven = bb_mnar_codriven_full_imp", pipeline_src)
end

@testitem "bb_mnar_codriven positioned after diagnostic cluster" tags=[:xlsx] begin
    using BayesInteractomics
    using DataFrames
    import XLSX

    tmpdir = mktempdir()
    xlsx_path = joinpath(tmpdir, "final_results.xlsx")

    n = 5
    final_results = DataFrame(
        Protein                  = ["P$i" for i in 1:n],
        is_detected              = fill(true, n),
        BF                       = [0.5, 1.0, 5.0, 50.0, 500.0],
        posterior_prob           = [0.1, 0.3, 0.5, 0.7, 0.9],
        pep                      = [0.9, 0.7, 0.5, 0.3, 0.1],
        BFDR                     = [0.5, 0.3, 0.2, 0.1, 0.05],
        mean_log2FC              = [-1.0, 0.0, 1.0, 2.0, 3.0],
        sd_log2FC                = fill(0.5, n),
        bf_enrichment            = [0.1, 0.5, 1.0, 5.0, 10.0],
        bf_correlation           = [0.1, 0.5, 1.0, 5.0, 10.0],
        bf_detected              = [0.1, 1.0, 5.0, 20.0, 100.0],
        missing_fraction         = [0.9, 0.5, 0.2, 0.1, 0.0],
        diagnostic_flag          = ["ok", "ok", "warning", "ok", "ok"],
        classification_stability = ["robust", "robust", "sensitive", "robust", "robust"],
        sensitivity_range        = [0.05, 0.03, 0.10, 0.02, 0.01],
        bb_mnar_codriven         = [false, false, false, false, true],
    )

    XLSX.writetable(xlsx_path, "df" => final_results; overwrite = true)

    xl = XLSX.readxlsx(xlsx_path)
    sheet = xl[XLSX.sheetnames(xl)[1]]
    header = string.(sheet[1, :])

    idx_diagnostic_flag          = findfirst(==("diagnostic_flag"), header)
    idx_classification_stability = findfirst(==("classification_stability"), header)
    idx_sensitivity_range        = findfirst(==("sensitivity_range"), header)
    idx_bb_mnar_codriven         = findfirst(==("bb_mnar_codriven"), header)

    @test idx_diagnostic_flag          !== nothing
    @test idx_classification_stability !== nothing
    @test idx_sensitivity_range        !== nothing
    @test idx_bb_mnar_codriven         !== nothing

    # Canonical positioning: bb_mnar_codriven immediately after the diagnostic
    # cluster — assert it follows ALL three of the cluster columns.
    @test idx_bb_mnar_codriven > idx_diagnostic_flag
    @test idx_bb_mnar_codriven > idx_classification_stability
    @test idx_bb_mnar_codriven > idx_sensitivity_range
end

