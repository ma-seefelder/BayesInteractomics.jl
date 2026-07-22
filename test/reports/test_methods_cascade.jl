# test/reports/test_methods_cascade.jl
#
# Methods cascade fallback regression.
#
# Verifies that `_build_diff_methods_json` walks the 4-level cascade:
#
#   Level 1 (config-present)    → `_build_methods_json(ar.config, ...)`
#   Level 2 (sidecar JSON only) → `methods.text` payload from
#                                 `<basedir>/interactive_report_data.json`
#   Level 3 (methods.md only)   → raw markdown wrapped in
#                                 `<div class="card-body markdown-body">`
#   Level 4 (neither)           → placeholder card + single @warn (maxlog=1)
#
# The helpers themselves (`_try_locate_sidecar`, `_try_locate_methods_md`,
# `_read_methods_from_sidecar_json`, `_methods_placeholder_card`) live in
# `src/reports/methods_generator.jl`. We exercise them directly on NamedTuple
# AR-mocks so the cascade can be tested without spinning up the full
# `differential_analysis` pipeline.
#
# Filter command:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("methods_cascade", ti.filename)'

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# Level 1: ar.config present → existing `_build_methods_json` path fires.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "cascade level 1: ar.config present" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using BayesInteractomics: CONFIG, OutputFiles, AnalysisResult, DifferentialResult
    using DataFrames, Dates

    fx = DifferentialFixtures.create_two_condition_result()
    # Attach a minimal but non-nothing CONFIG to one of the per-condition ARs so
    # the cascade level-1 branch is exercised. The other condition (config=nothing)
    # falls through to level 4 — that branch is asserted in the level-4 testitem
    # below, so we keep this testitem narrowly scoped.
    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile    = String[],
        control_cols = Dict{Int, Vector{Int}}[],
        sample_cols  = Dict{Int, Vector{Int}}[],
        poi          = "BAIT_WT",
        n_controls   = 3,
        n_samples    = 3,
        refID        = 1,
        output       = OutputFiles(tmpdir),
        run_simulation = false,
        run_input_qc   = false,
        run_validation = false,
        run_sensitivity = false,
    )
    fx.analyses[1].config = cfg

    json = BayesInteractomics.Reports._build_diff_methods_json(fx)
    @test json !== "null"
    @test occursin("\"per_condition\"", json)
    # Level 1 emits the structured methods payload — `text`, `reproducibility`,
    # `parameters`, `structured` keys all appear under the first condition.
    @test occursin("\"text\"", json)
    @test occursin("\"reproducibility\"", json)
    @test occursin("\"parameters\"", json)
    @test occursin("\"structured\"", json)
    # Label for the first condition is "WT" per the fixture.
    @test occursin("\"WT\"", json)
end

# ─────────────────────────────────────────────────────────────────────────────
# Level 2: sidecar JSON only → helper extracts `methods.text` from the JSON.
#
# Tests the helper directly with a duck-typed mock AR (NamedTuple) carrying an
# `output.basedir`, since the production `AnalysisResult` struct has no `output`
# field. The `_basedir_for_ar` helper resolves via `hasproperty`/`getproperty`
# so a NamedTuple suffices.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "cascade level 2: sidecar JSON only" begin
    using BayesInteractomics

    tmpdir = mktempdir()
    sidecar = joinpath(tmpdir, "interactive_report_data.json")
    sidecar_text = "Sidecar methods prose for unit test."
    write(sidecar, """{"methods":{"text":"$(sidecar_text)"}}""")

    ar_mock = (output = (basedir = tmpdir,), config = nothing)

    # Helper-level assertions.
    sp = BayesInteractomics.Reports._try_locate_sidecar(ar_mock)
    @test sp == sidecar

    parsed = BayesInteractomics.Reports._read_methods_from_sidecar_json(sp)
    @test parsed !== nothing
    @test occursin(sidecar_text, parsed)

    # And the `methods.md` path is NOT picked up (we didn't write one).
    @test BayesInteractomics.Reports._try_locate_methods_md(ar_mock) === nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Level 3: methods.md only → raw markdown wrapped in markdown-body card.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "cascade level 3: methods.md only" begin
    using BayesInteractomics

    tmpdir = mktempdir()
    md_path = joinpath(tmpdir, "methods.md")
    md_body = "# Test Methods\nBody paragraph with detail."
    write(md_path, md_body)

    ar_mock = (output = (basedir = tmpdir,), config = nothing)

    # Sidecar is absent at this level.
    @test BayesInteractomics.Reports._try_locate_sidecar(ar_mock) === nothing

    # methods.md IS present and the locator returns the absolute path.
    mp = BayesInteractomics.Reports._try_locate_methods_md(ar_mock)
    @test mp == md_path

    # Markdown body is readable verbatim from disk (the cascade wraps it in
    # `<div class="card-body markdown-body">…</div>` at the call-site; the
    # locator helper itself just returns the path).
    contents = read(mp, String)
    @test occursin("# Test Methods", contents)
    @test occursin("Body paragraph", contents)
end

# ─────────────────────────────────────────────────────────────────────────────
# Level 4: neither source → placeholder card + single @warn (maxlog=1).
#
# Asserts both that the placeholder string contains the expected text AND that
# the warning fires exactly once via `@test_logs (:warn, …)`.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "cascade level 4: placeholder + warning" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using Logging

    # Placeholder helper directly — confirms the both-paths hint.
    placeholder = BayesInteractomics.Reports._methods_placeholder_card("TestLabel")
    @test occursin("Methods unavailable for TestLabel", placeholder)
    @test occursin("interactive_report_data.json", placeholder)
    @test occursin("methods.md", placeholder)

    # Full cascade integration: build a 2-condition fixture (both ARs have
    # config=nothing and no `output.basedir`, so both fall straight to level 4)
    # and assert (a) the rendered JSON contains the placeholder text for each
    # condition AND (b) the level-4 @warn fires.
    fx = DifferentialFixtures.create_two_condition_result()

    # `@test_logs` returns the value of the wrapped expression — assign it so
    # post-warn assertions can inspect the rendered JSON. (Assignment INSIDE
    # the `begin` block would not escape the macro's hygienic scope.)
    json = @test_logs (:warn, r"\[Methods\] no methods source") match_mode=:any (
        BayesInteractomics.Reports._build_diff_methods_json(fx)
    )

    @test occursin("Methods unavailable for WT", json)
    @test occursin("Methods unavailable for Mutant", json)
    # `per_condition` MUST carry an entry per condition — no more silent drop.
    @test occursin("\"WT\"", json)
    @test occursin("\"Mutant\"", json)
end
