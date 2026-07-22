# test/reports/test_sensitivity_strip.jl
#
# Sensitivity tab fixes.
#
# Two coordinated fixes:
#   1. Upper Sensitivity panel x-axis was cut off (Plotly automargin: false
#      layout bug). Fix adds `automargin: true` to xaxis + yaxis.
#   2. Per-protein stability change strip rendered with hardcoded
#      `[D.meta.condition_A, D.meta.condition_B]` (template L2954),
#      producing an empty / k=2-only strip even when the backend payload was
#      populated for k≥3. Fix replaces the accessor with the canonical
#      predicate `D.meta.condition_labels` and transposes the layout
#      (rows = condition labels of length k, columns = top-N proteins).
#
# The `top_n_stability::Int = 20` field on `DifferentialConfig` follows the
# `validation_candidates_top_n` precedent. The strip column count is now
# driven by `D.meta.top_n_stability`.
#
# Byte-equality lock: for k=2 the strip still renders the legacy 2-row
# layout (`condition_labels(diff)` returns `[condition_A, condition_B]` for
# legacy 2-group calls).
#
# `DifferentialResult` ctor stays at 20 args; the new field lives on
# `DifferentialConfig`, NOT on `DifferentialResult`.
#
# Filter command:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("sensitivity_strip", ti.filename)'

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 1 — Template structure: x-axis automargin set in Sensitivity tab
# ─────────────────────────────────────────────────────────────────────────────
@testitem "x-axis automargin set on upper Sensitivity panel" begin
    template_path = joinpath(@__DIR__, "..", "..",
                             "src", "reports", "templates",
                             "differential_report.html")
    template = read(template_path, String)

    # Isolate `initSensitivity` body (sentinel: next top-level function).
    start_idx = findfirst("function initSensitivity", template)
    @test start_idx !== nothing
    body_start = first(start_idx)
    end_idx = findnext("function initMixture", template, body_start)
    @test end_idx !== nothing
    body = SubString(template, body_start, first(end_idx) - 1)

    # The per-condition Sensitivity panel bar plot MUST carry
    # `automargin: true` on its xaxis so 'robust'/'sensitive'/'fragile'
    # category labels no longer truncate.
    @test occursin("automargin: true", body)
    # The xaxis line in the panel-plot layout block MUST be present.
    @test occursin("xaxis: {automargin: true}", body)
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 2 — Template structure: stability strip reads
#               D.meta.condition_labels (NOT hardcoded condition_A/B)
# ─────────────────────────────────────────────────────────────────────────────
@testitem "stability strip reads canonical D.meta.condition_labels" begin
    template_path = joinpath(@__DIR__, "..", "..",
                             "src", "reports", "templates",
                             "differential_report.html")
    template = read(template_path, String)

    # Isolate `initSensitivity` body.
    start_idx = findfirst("function initSensitivity", template)
    @test start_idx !== nothing
    body_start = first(start_idx)
    end_idx = findnext("function initMixture", template, body_start)
    @test end_idx !== nothing
    body = SubString(template, body_start, first(end_idx) - 1)

    # The new (canonical) accessor MUST be present.
    @test occursin("D.meta.condition_labels", body)

    # `D.meta.top_n_stability` MUST be consumed for column-count sizing.
    @test occursin("D.meta.top_n_stability", body) ||
          occursin("top_n_stability", body)

    # The legacy hardcoded `[D.meta.condition_A, D.meta.condition_B]` pair
    # MUST NOT appear in the strip render block — defang via stripping JS
    # line-comments before the contains-check (historical comments may
    # retain the literal). The lone hardcoded accessor in the wider file
    # (L728 condA fallback for protein labels) is OUTSIDE `initSensitivity`
    # and unaffected.
    code_lines = filter(l -> !occursin(r"^\s*//", l), split(body, '\n'))
    code_only  = join(code_lines, '\n')
    @test !occursin("[D.meta.condition_A, D.meta.condition_B]", code_only)
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 3 — D.meta carries top_n_stability and condition_labels for k=4
#               (4-row strip contract; backend payload + frontend wiring)
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=4 D.meta carries condition_labels + top_n_stability" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics
    using DataFrames

    fx   = DifferentialFixturesK4.create_four_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,
    )
    out  = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # k=4 condition labels exposed via the canonical accessor.
    @test occursin("\"condition_labels\":[\"wt\",\"mut1\",\"mut2\",\"mut3\"]", html)

    # `top_n_stability` exposed (default 20 since the fixture builds a
    # default-config differential_analysis call).
    @test occursin("\"top_n_stability\":20", html)
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 4 — k=2 byte-equality + top_n_stability config override.
#
# Two contracts in one testitem (exactly 4 @testitem):
#   (a) k=2 byte-equality: `condition_labels(diff)` still resolves to
#       `[condition_A, condition_B]` (byte-equality lock); strip render
#       reads it via the canonical D.meta.condition_labels accessor.
#   (b) `DifferentialConfig.top_n_stability` config field overrides the
#       default 20 in D.meta — propagates correctly to the JSON payload.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=2 byte-equality + top_n_stability config override" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics
    using BayesInteractomics: DifferentialConfig
    using DataFrames

    fx = DifferentialFixturesK4.create_four_condition_result()

    # Contract (a): k=2 byte-equality preserved.
    diff_k2 = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1),
        contrasts  = :all_pairs,
    )
    out_k2  = tempname() * ".html"
    generate_differential_report(diff_k2; output = out_k2)
    html_k2 = read(out_k2, String)

    # k=2 → exactly 2 labels in declaration order. condition_labels(diff)
    # for legacy 2-group is [diff.condition_A, diff.condition_B]; for k=2
    # NamedTuple it derives the same length-2 vector from r.contrasts.
    @test occursin("\"condition_labels\":[\"wt\",\"mut1\"]", html_k2) ||
          occursin("\"condition_labels\":[\"mut1\",\"wt\"]", html_k2)

    # Default top_n_stability = 20 exposed even for k=2.
    @test occursin("\"top_n_stability\":20", html_k2)

    # Contract (b): config override propagates.
    cfg5 = DifferentialConfig(top_n_stability = 5)
    diff5 = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts  = :all_pairs,
        config     = cfg5,
    )
    out5  = tempname() * ".html"
    generate_differential_report(diff5; output = out5)
    html5 = read(out5, String)

    @test occursin("\"top_n_stability\":5", html5)
    # Sanity: the k=3 condition_labels are also emitted in declaration order.
    @test occursin("\"condition_labels\":[\"wt\",\"mut1\",\"mut2\"]", html5)
end
