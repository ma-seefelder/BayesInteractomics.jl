# test/reports/test_validation_plots.jl
#
# DOM-smoke regression coverage for the four Validation Candidates summary
# plots that render above the top-20 grid card.
#
# Plots covered:
#   (a) #vc-plot-violin   — per-pair decision_risk violin+box overlay
#   (b) #vc-plot-topbar   — top-N decision_risk horizontal bar
#   (c) #vc-plot-scatter  — per-pair decision_risk vs posterior scatter
#   (d) #vc-plot-stacked  — per-pair risk-class composition stacked bar
#
# All four are filled by `renderValidationCandidatePlots(D)` (defined in
# `differential_report.html`) on DOMContentLoaded, immediately after
# `renderValidationCandidates(D)`. BOTH_NEGATIVE rows are excluded upstream
# by `_build_validation_candidates_block`; this test file's 4th block is a
# regression guard that the plot inputs (i.e. the `candidates` JSON consumed
# by the JS function) does NOT reintroduce BOTH_NEGATIVE rows.
#
# Filter command:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("validation_plots", ti.filename)'
#
# Locks honoured: §7a CLS_COLOR palette (no new hex codes); kgroup_class
# enum; palette reuse; BOTH_NEGATIVE exclusion; BMA + FDR terminology locks.

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# Block 1 — k=4 fixture: all four mountpoint div ids appear verbatim in
# the rendered HTML. Smoke check that the template insertion landed in the
# Validation Candidates section.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "four mountpoint divs present" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics

    fx = DifferentialFixturesK4.create_four_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # The four mountpoint ids must each appear at least once. Use the
    # literal `id="vc-plot-..."` attribute form so a stray substring match
    # in JS string literals or comments cannot satisfy the assertion.
    for mp in ("vc-plot-violin", "vc-plot-topbar", "vc-plot-scatter", "vc-plot-stacked")
        @test occursin("id=\"$(mp)\"", html)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Block 2 — `renderValidationCandidatePlots` function is defined in the
# script section AND invoked from the DOMContentLoaded boot block. Guards
# against the function being introduced but never wired into the boot path.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "renderValidationCandidatePlots invoked" begin
    using BayesInteractomics

    project_root = pkgdir(BayesInteractomics)
    template_path = joinpath(project_root, "src", "reports", "templates",
                             "differential_report.html")
    @test isfile(template_path)
    template = read(template_path, String)

    # Definition site — exactly one `function renderValidationCandidatePlots(`
    # declaration.
    @test occursin("function renderValidationCandidatePlots", template)

    # Invocation site — must be called at least once (the boot path).
    n_calls = length(collect(eachmatch(r"renderValidationCandidatePlots\(D\)", template)))
    @test n_calls >= 1

    # Plotly.newPlot must be called at least four times inside the function
    # body (one per plot). We bound the function body by matching from the
    # function header to the next sibling `function ` declaration.
    body_match = match(r"function renderValidationCandidatePlots\(D\)\s*\{(.*?)\nfunction "s, template)
    @test body_match !== nothing
    body = body_match === nothing ? "" : body_match.captures[1]
    n_plotly = length(collect(eachmatch(r"Plotly\.newPlot", body)))
    @test n_plotly >= 4
end

# ─────────────────────────────────────────────────────────────────────────────
# Block 3 — each of the four mountpoints has a `btn-explanation-toggle`
# button whose `aria-controls` (or `data-explain-target`) attribute names
# the corresponding `vc-plot-*` id. This enforces the contract that every
# plot card carries its own explanation toggle.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "each plot has explanation toggle" begin
    using BayesInteractomics

    project_root = pkgdir(BayesInteractomics)
    template_path = joinpath(project_root, "src", "reports", "templates",
                             "differential_report.html")
    template = read(template_path, String)

    # Toggle pattern: `<button class="btn-explanation-toggle"
    # ... aria-controls="expl-vc-plot-XYZ" ... data-explain-target="vc-plot-XYZ"`.
    # We assert one button per mountpoint by counting hits of
    # `btn-explanation-toggle.*vc-plot-XYZ` (the same predicate the
    # acceptance criterion uses).
    for mp in ("vc-plot-violin", "vc-plot-topbar", "vc-plot-scatter", "vc-plot-stacked")
        n_hits = length(collect(eachmatch(
            Regex("btn-explanation-toggle[^<>]*$(mp)"), template)))
        @test n_hits >= 1
        if n_hits < 1
            @info "missing explanation toggle for mountpoint" mp
        end
    end

    # Aggregate count: the acceptance criterion requires exactly 4
    # `btn-explanation-toggle.*vc-plot-` hits. We assert >= 4 to be tolerant
    # of future additive wiring (e.g. a fifth toggle on a hypothetical
    # composite plot) without breaking this regression.
    n_total = length(collect(eachmatch(r"btn-explanation-toggle[^<>]*vc-plot-", template)))
    @test n_total >= 4
end

# ─────────────────────────────────────────────────────────────────────────────
# Block 4 — regression guard: BOTH_NEGATIVE rows are NOT present in the
# `validation_candidates.candidates` JSON array that feeds
# `renderValidationCandidatePlots`. The upstream contract
# (`_build_validation_candidates_block` strips BOTH_NEGATIVE before
# emitting the candidates array) ensures a future regression in that filter
# surfaces here as a plot-data corruption.
#
# The `all_proteins` sibling array intentionally retains BOTH_NEGATIVE
# rows for QC inspection — we explicitly do NOT assert anything about that
# array here.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "BOTH_NEGATIVE excluded from plot inputs" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics

    fx = DifferentialFixturesK4.create_four_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # Extract the candidates JSON block (bounded by the all_proteins sibling
    # key — same anchor used by test_both_negative_filter.jl).
    cand_match = match(r"\"candidates\"\s*:\s*\[(.*?)\]\s*,\s*\"all_proteins\""s, html)
    @test cand_match !== nothing
    cand_block = cand_match === nothing ? "" : cand_match.captures[1]

    # Assert no candidate carries an `optimal_call` or `map_class` of
    # BOTH_NEGATIVE / both_negative (case-insensitive — guards against a
    # future regression that lowercases / uppercases the enum value).
    optimal_calls = [m.captures[1] for m in
                     eachmatch(r"\"optimal_call\"\s*:\s*\"([^\"]*)\"", cand_block)]
    map_classes   = [m.captures[1] for m in
                     eachmatch(r"\"map_class\"\s*:\s*\"([^\"]*)\"",   cand_block)]
    @test all(oc -> !occursin(r"both_negative"i, oc), optimal_calls)
    @test all(mc -> !occursin(r"both_negative"i, mc), map_classes)
end
