# test/differential/test_companion_files.jl
#
# k-group companion file emission.
#
# Two @testitem blocks:
#   1. k=4 emits 30 svg files (6 pairs × 5 plot types) + xlsx (suffixed path scheme).
#   2. k=2 legacy single-file emission preserved (byte-equality):
#      the legacy 2-group writer emits unsuffixed plot files; the
#      `_write_kgroup_companion_files` helper no-ops via
#      `length(diff.contrasts) <= 1 && return nothing`. This testitem
#      asserts that no suffixed file leaks onto the k=2 path.
#
# BMA terminology "Copula" + "3c-EM"; FDR terminology BFDR / PEP / local_fdr.
#
# Filter command:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("test_companion_files", ti.filename)'

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 1 — k=4 / :all_pairs emits 30 svg files + multi-sheet xlsx
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=4 emits 30 svg + xlsx" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics
    using BayesInteractomics: DifferentialConfig
    using BayesInteractomics.Differential: _suffix_plot_path
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

    fx = DifferentialFixturesK4.create_four_condition_result(; tmpdir = tmpdir)
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,
        config     = cfg,
    )

    # k=4 / :all_pairs → 6 pairs × 5 plot types = 30 svg files.
    @test length(diff.contrasts) == 6

    for pair in diff.contrasts
        a = String(first(pair))
        b = String(last(pair))
        for base in (cfg.volcano_file, cfg.evidence_file, cfg.scatter_file,
                     cfg.classification_file, cfg.ma_file)
            @test isfile(_suffix_plot_path(base, a, b))
        end
    end

    # xlsx is written.
    @test isfile(cfg.results_file)
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 2 — k=2 legacy path: helper no-ops; no suffixed files leak
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=2 legacy single-file emission preserved" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics
    using BayesInteractomics: DifferentialConfig
    using BayesInteractomics.Differential: _suffix_plot_path,
                                           _write_kgroup_companion_files
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

    # k=2 path via the keyword overload (NamedTuple with two conditions). The
    # k-group code path runs with length(diff.contrasts) == 1 — the helper
    # `_write_kgroup_companion_files` early-returns under the
    # byte-equality guard.
    fx = DifferentialFixturesK4.create_four_condition_result(; tmpdir = tmpdir)
    diff_k2 = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1),
        contrasts  = :all_pairs,
        config     = cfg,
    )

    @test length(diff_k2.contrasts) == 1

    # The helper must NOT emit suffixed files for k=2 (byte-equality guard).
    @test !isfile(_suffix_plot_path(cfg.volcano_file,        "wt", "mut1"))
    @test !isfile(_suffix_plot_path(cfg.evidence_file,       "wt", "mut1"))
    @test !isfile(_suffix_plot_path(cfg.scatter_file,        "wt", "mut1"))
    @test !isfile(_suffix_plot_path(cfg.classification_file, "wt", "mut1"))
    @test !isfile(_suffix_plot_path(cfg.ma_file,             "wt", "mut1"))

    # Direct invocation of the helper is also a no-op for k=2 (defence in depth).
    @test _write_kgroup_companion_files(diff_k2, cfg) === nothing
end
