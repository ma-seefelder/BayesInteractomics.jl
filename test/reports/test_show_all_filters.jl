# test/reports/test_show_all_filters.jl
#
# DOM-smoke coverage for the Validation Candidates "show all proteins"
# expandable <details> panel. DataTables.js is wired into this panel with
# the same config object as the Results-tab init (paging, lengthMenu,
# searching, ordering) plus a pair-selector subscription for k≥3.
#
# Contracts asserted (all three target the rendered template HTML — DOM
# smoke, no JS execution required):
#
#   1. DataTables config object contains `searching: true`, `paging: true`,
#      and `lengthMenu` (Results-tab parity).
#   2. The show-all DataTable subscribes to `results-pair-select` change events
#      and calls `.column(3).search(...)` to filter rows by active pair.
#   3. Re-init-safe pattern: `.DataTable().destroy()` is invoked before the
#      construction of a new DataTable instance.
#
# Filter command:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("show_all_filters", ti.filename)'
#
# Locks honoured: __vc_all_wired sentinel; k=2 byte-equality (k=2 path is
# hidden dropdown, no row filter, single pair only — no special k=2
# conditional in the test or the template wiring).

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 1 — show-all panel DataTable config has search + paging + lengthMenu
# (Results-tab parity).
# ─────────────────────────────────────────────────────────────────────────────
@testitem "show-all panel DataTable config has search + paging" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics

    fx = DifferentialFixturesK4.create_four_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html_str = read(out, String)

    # The DataTables config object lives inside the <details> toggle handler.
    # Confirm the Results-tab-parity fields are present (verbatim port).
    @test occursin(r"searching\s*:\s*true", html_str)
    @test occursin(r"paging\s*:\s*true", html_str)
    @test occursin("lengthMenu", html_str)
    # The show-all-specific DataTable target id is stable for the
    # pair-selector subscription downstream. Confirm it is wired.
    @test occursin("vc-all-table", html_str)
    # Confirm at least one new DataTable(...) construction call exists in
    # the template (Results tab + show-all panel both construct one).
    n_init = length(collect(eachmatch(r"new DataTable\(", html_str)))
    @test n_init >= 2
end

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 2 — show-all DataTable subscribes to pair-selector dropdown.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "show-all DataTable subscribes to pair-selector" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics

    fx = DifferentialFixturesK4.create_four_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html_str = read(out, String)

    # The dropdown id is `results-pair-select`. The show-all
    # panel must reference it (lookup) AND subscribe a change handler that
    # invokes `.column(3).search(...)` on the show-all DataTable (column 3 =
    # the Pair column in the show-all table header order: Protein, Optimal
    # call, MAP class, Pair, Decision risk).
    @test occursin("results-pair-select", html_str)
    # A .column(...).search(...) call wired off the active pair must exist
    # near the show-all wiring. Anchor on `vcAllDT.column` to disambiguate
    # from the Results-tab `dt.search()` (which is the legacy global search,
    # not a per-column filter).
    @test occursin(r"vcAllDT\.column\(\s*3\s*\)\.search\(", html_str)
    # The dropdown change subscriber must exist in the template (the
    # show-all panel attaches a second listener; the Results tab attaches
    # the first one for `_renderResultsTableForPair`).
    @test occursin(r"pairSel\.addEventListener\('change'", html_str)
end

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 3 — re-init-safe pattern: destroy() before .DataTable() construction.
# (Sentinel guard alone is not sufficient; a stale DataTables instance must be
# torn down before re-init or jQuery throws.)
# ─────────────────────────────────────────────────────────────────────────────
@testitem "re-init-safe pattern uses destroy() before .DataTable()" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics

    fx = DifferentialFixturesK4.create_four_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html_str = read(out, String)

    # The show-all panel wraps its construction in an isDataTable() check +
    # destroy() before the new DataTable() call. Anchor on the vc-all-table
    # id so we are checking the show-all wiring (not the Results-tab init,
    # which does not need to destroy() because it only runs once at page
    # load).
    @test occursin(r"isDataTable\(\s*['\"]#vc-all-table['\"]\s*\)", html_str)
    @test occursin(r"#vc-all-table['\"]\s*\)\.DataTable\(\)\.destroy\(\)", html_str)
    # Sentinel is still in place (the destroy() pattern
    # complements but does not replace the lazy-render guard).
    @test occursin("__vc_all_wired", html_str)
end
