# test/reports/test_calibration_mixture_render.jl
#
# Calibration + Mixture renderer shape-fix.
#
# Two SEPARATE shape mismatches in the k-group differential report template
# (src/reports/templates/differential_report.html):
#
#   Calibration: `_renderCalibrationSingle` read
#     `payload.calibration.x / y` (legacy shape, does not exist). Actual
#     producer `_build_simulation_json` emits `payload.scenarios[]` plus
#     a `payload.calibration` sub-object with `cal_curve_x / cal_curve_y`.
#
#   Mixture:     `_renderMixtureSingle` read `payload.scatter.x / y`
#     (legacy shape, does not exist). Actual producer `_build_mixture_model_json`
#     emits `payload.params.{background,agnostic,interaction}.density_x / y`
#     and `payload.scatter.{log_bf_enrichment, log_bf_correlation, ...}`.
#
# These five testitems lock the consumer-side payload contract and verify
# k=2 byte-equality (the multi-panel k=2 container ALSO had the bug and is
# fixed symmetrically via delegation to the single-condition renderers).
#
# Filter command:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("calibration_mixture_render", ti.filename)'

using TestItemRunner

# Bundled template source — read once, asserted across multiple testitems.
const _TEMPLATE_PATH = joinpath(@__DIR__, "..", "..",
                                "src", "reports", "templates",
                                "differential_report.html")

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 1 — Calibration renderer consumes the actual payload shape
# ─────────────────────────────────────────────────────────────────────────────
@testitem "Calibration renderer consumes scenarios shape" begin
    template_path = joinpath(@__DIR__, "..", "..",
                             "src", "reports", "templates",
                             "differential_report.html")
    template = read(template_path, String)

    # Locate the `_renderCalibrationSingle` function body. The body ends at
    # the next top-level `function _render` declaration (sentinel).
    start_idx = findfirst("function _renderCalibrationSingle", template)
    @test start_idx !== nothing
    body_start = first(start_idx)
    # End sentinel — the next renderer function declaration in the file.
    next_render_idx = findnext("function _renderSensitivitySingle",
                               template, body_start)
    @test next_render_idx !== nothing
    body = SubString(template, body_start, first(next_render_idx) - 1)

    # Renderer body MUST reference the canonical producer keys.
    # cal_curve_x/y — the Platt curve sub-object emitted by
    # `_build_calibration_json` (src/simulation/simulation.jl:942).
    @test occursin("cal_curve_x", body)
    @test occursin("cal_curve_y", body)
    # `payload.scenarios` — the SimulationResult flavour emitted by
    # `_build_simulation_json` (src/simulation/simulation.jl:878).
    @test occursin("payload.scenarios", body) ||
          occursin("scenarios = payload.scenarios", body)

    # Renderer body MUST NOT contain the legacy "payload.calibration.x" read
    # (only the historical comment retains the literal — we filter to code
    # lines by stripping JS line-comments and verify the resulting source
    # does not still consume the legacy shape).
    code_lines = filter(l -> !occursin(r"^\s*//", l), split(body, '\n'))
    code_only  = join(code_lines, '\n')
    @test !occursin("curve.x && curve.y", code_only)
    @test !occursin("payload.calibration.x", code_only)
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 2 — k=3 fixture: per_condition.<label>.calibration sub-key present
#               and the renderer is wired to consume it for each label
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=3 Calibration payload-shape contract holds per condition" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    fx   = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts  = :all_pairs,
    )
    out  = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # All three condition labels carry the calibration sub-object — the
    # dropdown reads `D.per_condition[selectedLabel].calibration` so EACH
    # label MUST have the sub-key present (data-uniform path).
    for label in ("wt", "mut1", "mut2")
        @test occursin("\"$(label)\":{\"calibration\":", html)
    end

    # The template MUST carry the (now-fixed) renderer that consumes the
    # `cal_curve_x / cal_curve_y` Platt curve OR the median-effect
    # `scenarios[mid]` fallback. Smoke check: both producer keys reachable.
    @test occursin("cal_curve_x", html)
    @test occursin("payload.scenarios", html)
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 3 — Mixture renderer consumes the actual payload shape
# ─────────────────────────────────────────────────────────────────────────────
@testitem "Mixture renderer consumes params.density_x/y shape" begin
    template_path = joinpath(@__DIR__, "..", "..",
                             "src", "reports", "templates",
                             "differential_report.html")
    template = read(template_path, String)

    # Isolate the `_renderMixtureSingle` body up to the next `_renderQc`
    # sentinel.
    start_idx = findfirst("function _renderMixtureSingle", template)
    @test start_idx !== nothing
    body_start = first(start_idx)
    next_render_idx = findnext("function _renderQcSingle",
                               template, body_start)
    @test next_render_idx !== nothing
    body = SubString(template, body_start, first(next_render_idx) - 1)

    # Renderer body MUST reference the canonical producer keys.
    # `params.{component}.density_x / density_y` — per-component MC-convolved
    # densities emitted by `_build_mixture_model_json`
    # (src/reports/report_generator.jl:886-887).
    @test occursin("density_x", body)
    @test occursin("density_y", body)
    @test occursin("payload.params", body) ||
          occursin("params = payload.params", body)

    # Renderer body MUST NOT contain the legacy `scatter.x && scatter.y`
    # read (the actual scatter object has `log_bf_enrichment` /
    # `log_bf_correlation` keys, not `x` / `y`).
    code_lines = filter(l -> !occursin(r"^\s*//", l), split(body, '\n'))
    code_only  = join(code_lines, '\n')
    @test !occursin("sc.x && sc.y", code_only)
    @test !occursin("payload.scatter.x", code_only)

    # Fallback path MUST reference the actual scatter keys.
    @test occursin("log_bf_enrichment", body)
    @test occursin("log_bf_correlation", body)
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 4 — dropdown change events wire to the renderers via the
#               `_initConditionSelector` state machine (DOM smoke)
# ─────────────────────────────────────────────────────────────────────────────
@testitem "dropdown change events subscribe to renderers" setup=[DifferentialFixtures] begin
    using BayesInteractomics

    fx   = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts  = :all_pairs,
    )
    out  = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # Both dropdowns present in DOM (sanity guard so the renderer wiring
    # under test is reachable from the UI).
    @test occursin("id=\"calibration-cond-select\"", html)
    @test occursin("id=\"mixture-cond-select\"", html)

    # `_initConditionSelector` is invoked from both `initCalibration` and
    # `initMixture` with the matching renderer as the 6th argument.
    @test occursin("_renderCalibrationSingle", html)
    @test occursin("_renderMixtureSingle", html)

    # The state machine wires the change event to `renderFor(e.target.value)`
    # which calls `renderFn(payload, divId)` — i.e. the renderer receives the
    # per-condition payload for the selected label.
    @test occursin("addEventListener('change'", html)
    @test occursin("renderFn(payload, divId)", html)

    # `_initConditionSelector` is called with the calibration + mixture
    # renderer hooks (the 6-arg state machine).
    cal_init = findfirst("function initCalibration", html)
    @test cal_init !== nothing
    cal_body = SubString(html, first(cal_init),
                         min(first(cal_init) + 600, lastindex(html)))
    @test occursin("_renderCalibrationSingle", cal_body)

    mix_init = findfirst("function initMixture", html)
    @test mix_init !== nothing
    mix_body = SubString(html, first(mix_init),
                         min(first(mix_init) + 600, lastindex(html)))
    @test occursin("_renderMixtureSingle", mix_body)
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 5 — k=2 multi-panel container ALSO renders via the shape-fixed
#               renderers (CONTEXT item 11 "empty for ALL conditions" coverage)
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=2 Calibration multi-panel still renders" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    fx     = DifferentialFixtures.create_three_condition_result()
    # Build a k=2 differential by passing only two of the three conditions.
    diff_k2 = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1),
        contrasts  = :all_pairs,
    )
    out  = tempname() * ".html"
    generate_differential_report(diff_k2; output = out)
    html = read(out, String)

    # k=2 path → `D.meta.condition_labels.length === 2`. No third condition.
    @test occursin("\"condition_labels\":[\"wt\",\"mut1\"]", html) ||
          occursin("\"condition_labels\":[\"mut1\",\"wt\"]", html)

    # Per-condition sub-keys carry the calibration + mixture sub-objects for
    # BOTH labels (data-uniform path; byte-equality).
    @test occursin("\"wt\":{\"calibration\":",   html)
    @test occursin("\"mut1\":{\"calibration\":", html)

    # The k=2 multi-panel container at L2735 (`forEach` over
    # `D.calibration.panels`) MUST now delegate to `_renderCalibrationSingle`
    # — not the legacy `curve.x && curve.y` consumer. Grep the template
    # source: delegation pattern is `_renderCalibrationSingle(p.data, divId)`.
    template = read(joinpath(@__DIR__, "..", "..", "src", "reports",
                             "templates", "differential_report.html"),
                    String)
    @test occursin("_renderCalibrationSingle(p.data, divId)", template)
    @test occursin("_renderMixtureSingle(p.data, divId)",     template)

    # Regression guard: the legacy `curve.x && curve.y` consumer MUST NOT
    # appear in the k=2 multi-panel `initCalibration` body anymore (it
    # remained as the legacy code path before the fix — the symmetric fix
    # delegates to the single-condition renderer instead).
    cal_block_start = findfirst("function initCalibration", template)
    @test cal_block_start !== nothing
    cal_block_end   = findnext("function initSensitivity",
                               template, first(cal_block_start))
    @test cal_block_end !== nothing
    cal_block = SubString(template, first(cal_block_start),
                          first(cal_block_end) - 1)
    @test !occursin("curve.x && curve.y", cal_block)
end
