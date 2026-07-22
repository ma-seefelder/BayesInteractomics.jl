# test/differential/test_xlsx_layout.jl
#
# k-group xlsx layout contract.
#
# Three @testitem blocks:
#   1. k=4 / :all_pairs → xlsx has 7 sheets: Sheet1 (wide DF) + 6 per-pair sheets
#      named `<a>_vs_<b>`.
#   2. k=2 legacy path → xlsx is single-sheet (`_write_kgroup_companion_files`
#      no-ops; the legacy 2-group writer is the single source of xlsx emission
#      for k=2 and uses `export_differential` with two sheets "differential" +
#      "summary"). Note: this testitem only asserts that the new helper does NOT
#      contribute extra `<a>_vs_<b>` sheets — the legacy single-sheet contract
#      is owned by the byte-equality lock.
#   3. Wide Sheet1 contains suffixed columns like `bf_A_<a>_vs_<b>`.
#
# BMA terminology "Copula" + "3c-EM"; FDR terminology BFDR / PEP / local_fdr.

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 1 — k=4 xlsx layout: Sheet1 + 6 pair sheets
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=4 xlsx has Sheet1 + 6 pair sheets" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics
    using BayesInteractomics: DifferentialConfig
    using DataFrames
    import XLSX

    tmpdir = mktempdir()
    cfg = DifferentialConfig(
        volcano_file        = joinpath(tmpdir, "vol.svg"),
        evidence_file       = joinpath(tmpdir, "ev.svg"),
        scatter_file        = joinpath(tmpdir, "sc.svg"),
        classification_file = joinpath(tmpdir, "cl.svg"),
        ma_file             = joinpath(tmpdir, "ma.svg"),
        results_file        = joinpath(tmpdir, "diff_results.xlsx"),
        generate_report_html = false,
    )

    fx = DifferentialFixturesK4.create_four_condition_result(; tmpdir = tmpdir)
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,
        config     = cfg,
    )

    @test isfile(cfg.results_file)
    xl = XLSX.readxlsx(cfg.results_file)
    sheet_names = XLSX.sheetnames(xl)

    # k=4 / :all_pairs → 6 pairs → 1 wide + 6 per-pair = 7 sheets.
    @test length(sheet_names) == 7
    @test "Sheet1" in sheet_names

    expected_pair_sheets = [
        "wt_vs_mut1", "wt_vs_mut2", "wt_vs_mut3",
        "mut1_vs_mut2", "mut1_vs_mut3", "mut2_vs_mut3",
    ]
    for s in expected_pair_sheets
        @test s in sheet_names
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 2 — k=2 xlsx layout: helper no-ops; no `<a>_vs_<b>` sheet leaks
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=2 xlsx contains no leaked per-pair sheets" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics
    using BayesInteractomics: DifferentialConfig
    using DataFrames

    tmpdir = mktempdir()
    cfg = DifferentialConfig(
        volcano_file        = joinpath(tmpdir, "vol.svg"),
        evidence_file       = joinpath(tmpdir, "ev.svg"),
        scatter_file        = joinpath(tmpdir, "sc.svg"),
        classification_file = joinpath(tmpdir, "cl.svg"),
        ma_file             = joinpath(tmpdir, "ma.svg"),
        results_file        = joinpath(tmpdir, "diff_results.xlsx"),
        generate_report_html = false,
    )

    # k=2 path via the keyword overload — helper must no-op.
    fx = DifferentialFixturesK4.create_four_condition_result(; tmpdir = tmpdir)
    diff_k2 = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1),
        contrasts  = :all_pairs,
        config     = cfg,
    )

    @test length(diff_k2.contrasts) == 1
    # The k-group helper does NOT write the xlsx for k=2 (byte-
    # equality lock). The keyword overload also does NOT invoke the legacy
    # 2-group writer block. Therefore no xlsx is created via this code path
    # — the file should not exist.
    @test !isfile(cfg.results_file)
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 3 — wide Sheet1 carries suffixed columns
# ─────────────────────────────────────────────────────────────────────────────
@testitem "wide Sheet1 contains suffixed columns" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics
    using BayesInteractomics: DifferentialConfig
    using DataFrames
    import XLSX

    tmpdir = mktempdir()
    cfg = DifferentialConfig(
        volcano_file        = joinpath(tmpdir, "vol.svg"),
        evidence_file       = joinpath(tmpdir, "ev.svg"),
        scatter_file        = joinpath(tmpdir, "sc.svg"),
        classification_file = joinpath(tmpdir, "cl.svg"),
        ma_file             = joinpath(tmpdir, "ma.svg"),
        results_file        = joinpath(tmpdir, "diff_results.xlsx"),
        generate_report_html = false,
    )

    fx = DifferentialFixturesK4.create_four_condition_result(; tmpdir = tmpdir)
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,
        config     = cfg,
    )

    @test isfile(cfg.results_file)
    xl = XLSX.readxlsx(cfg.results_file)
    @test "Sheet1" in XLSX.sheetnames(xl)

    # Read the header row of Sheet1; the wide DF carries
    # suffixed columns (`<col>_<a>_vs_<b>`). At least one such suffixed
    # numerical column from the first contrast `wt_vs_mut1` must be present
    # (e.g. `bf_A_wt_vs_mut1` or `log2fc_A_wt_vs_mut1` or
    # `posterior_A_wt_vs_mut1`).
    sheet1 = xl["Sheet1"]
    headers = String[string(h) for h in sheet1[1, :] if !ismissing(h)]

    # Tolerant check: at least one header ends with `_wt_vs_mut1`.
    @test any(endswith(h, "_wt_vs_mut1") for h in headers)
end
