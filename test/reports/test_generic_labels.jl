# test/reports/test_generic_labels.jl
#
# Generic labels + k=2 legacy byte-equality regression.
#
# Three @testitems:
#   1. Zero literal `wtHTT_specific` / `mHTT_specific` / `condition_A_specific`
#      / `condition_B_specific` string hardcodes in the template source.
#   2. k=2 legacy InteractionClass labels (`${condA} + ' specific'`) are
#      template-literal interpolations at JS runtime and are preserved
#      verbatim (byte-equality lock).
#   3. k=2 dashboard card phrasing ("Gained (stronger in {A})" / "Reduced
#      (stronger in {B})") is preserved byte-equal to the baseline.
#
# The user-observed `wtHTT_specific` was a JS-runtime output of
# `${condA}_specific` (template-literal interpolation), not a hardcoded
# string in the template source, so this file is a documentation +
# regression guard only — no source-code label rewrites required.
#
# Filter command:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("generic_labels", ti.filename)'

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — zero literal `_specific` hardcodes in template
# ─────────────────────────────────────────────────────────────────────────────
@testitem "zero literal _specific hardcodes in template source" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports",
                             "templates", "differential_report.html")
    @test isfile(template_path)
    src = read(template_path, String)

    # The four hardcoded strings the user observed in the rendered report
    # MUST NOT appear in the template source. The strings show up at JS
    # runtime via template-literal interpolation `${condA}_specific` /
    # `${condB}_specific`, but never as static string literals in the source.
    @test count(_ -> true,
                eachmatch(r"\"wtHTT_specific\"|\"mHTT_specific\"|\"condition_A_specific\"|\"condition_B_specific\"",
                          src)) == 0
end

# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — k=2 legacy InteractionClass labels byte-equal preserved
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=2 legacy InteractionClass labels preserved (template-literal interpolation)" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports",
                             "templates", "differential_report.html")
    src = read(template_path, String)

    # The legacy CLS_LABEL.CONDITION_A_SPECIFIC / CONDITION_B_SPECIFIC entries
    # are emitted via `condA + ' specific'` / `condB + ' specific'`
    # concatenation at JS runtime. Assert both source lines exist verbatim
    # — byte-equality lock for k=2.
    @test occursin("condA + ' specific'", src)
    @test occursin("condB + ' specific'", src)

    # The two CLS_LABEL keys themselves remain in the template (locked
    # InteractionClass enum keys — §7a). They are CONSUMED for k=2
    # rendering only; the k>=3 path renders kgroup_class enum labels.
    @test occursin("'CONDITION_A_SPECIFIC':", src)
    @test occursin("'CONDITION_B_SPECIFIC':", src)
end

# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — k=2 Gained/Reduced phrasing preserved byte-equal
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=2 Gained/Reduced phrasing preserved" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports",
                             "templates", "differential_report.html")
    src = read(template_path, String)

    # The legacy 4-card dashboard layout for k=2 carries the literal
    # phrasing `'Gained (stronger in '` and `'Reduced (stronger in '`.
    # These strings MUST stay byte-equal to the baseline (the k=2 dashboard
    # branch in `initDashboard` was not touched here; this is a regression
    # guard).
    @test occursin("Gained (stronger in ", src)
    @test occursin("Reduced (stronger in ", src)
end
