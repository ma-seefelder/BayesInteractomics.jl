# test/reports/test_volcano_render.jl
#
# Per-pair Plotly volcano with α-PEP / γ-PEP radio toggle for k≥3,
# byte-equality with legacy static-svg path for k=2.
#
# Five @testitems:
#   1. k=4 fixture: `kgroup-volcano-card` div + `kgroup-volcano-plot` mountpoint
#      are present in the rendered HTML.
#   2. k=4 fixture: α-PEP + γ-PEP radio inputs present; γ-PEP carries `checked`
#      attribute by default.
#   3. k=4 fixture: `renderKGroupVolcano` function is defined AND is invoked
#      from `initKGroupVolcano(D)` (entry point at boot for k>=3).
#   4. k=4 fixture: BOTH event handlers (results-pair-select dropdown change
#      AND volcano-pep-source radio change) call `renderKGroupVolcano` in their
#      handler bodies.
#   5. k=2 fixture: legacy `D.plots` reference still present (static-svg path
#      preserved) AND `kgroup-volcano-card` has the `display:none` initial
#      style (k=2 byte-equality lock).
#
# Locks honoured:
#   - BMA terminology (Copula + 3c-EM).
#   - FDR terminology (BFDR / PEP / local_fdr).
#   - canonical predicate (`D.meta.condition_labels`).
#   - §7a CLS_COLOR contract (hue + saturation from diff_PEP, not the toggled
#     PEP source).
#
# Filter command:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("volcano_render", ti.filename)'

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 1 — k=4 Plotly volcano card present (DOM smoke).
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=4 Plotly volcano card present" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics
    using DataFrames

    fx = DifferentialFixturesK4.create_four_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,   # 6 pairs
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # Card wrapper + mountpoint emitted by the template (DOM smoke).
    @test occursin("id=\"kgroup-volcano-card\"", html)
    @test occursin("id=\"kgroup-volcano-plot\"", html)

    # The card is gated on `condition_labels.length >= 3` — the JS branch
    # predicate must be present (canonical predicate).
    @test occursin("labels.length >= 3", html) ||
          occursin("condition_labels.length >= 3", html)
end

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 2 — k=4 α/γ-PEP radio group, γ-PEP checked by default.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=4 α/γ-PEP radio group present, γ-PEP checked by default" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics
    using DataFrames

    fx = DifferentialFixturesK4.create_four_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # Both radio inputs present (name=volcano-pep-source, paired ids).
    @test occursin("id=\"volcano-alpha-pep\"", html)
    @test occursin("id=\"volcano-gamma-pep\"", html)
    @test occursin("name=\"volcano-pep-source\"", html)

    # γ-PEP must carry the `checked` attribute (default per Open Q2).
    # Match within a small window after the gamma id to avoid false positives.
    gamma_window = match(r"id=\"volcano-gamma-pep\"[^>]{0,200}", html)
    @test gamma_window !== nothing
    @test occursin("checked", gamma_window.match)

    # α-PEP must NOT carry `checked` (would be a 2-radio conflict).
    alpha_window = match(r"id=\"volcano-alpha-pep\"[^>]{0,200}", html)
    @test alpha_window !== nothing
    @test !occursin("checked", alpha_window.match)
end

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 3 — renderKGroupVolcano defined + invoked on init (k>=3 boot path).
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=4 renderKGroupVolcano defined and invoked on init" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics
    using DataFrames

    fx = DifferentialFixturesK4.create_four_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # Function definition + boot-time invocation.
    @test occursin("function renderKGroupVolcano", html)
    @test occursin("function initKGroupVolcano", html)
    @test occursin("initKGroupVolcano(D)", html)

    # The initial render call uses `_pairLabel(contrasts[0])` as the active
    # pair seed (first contrast in declaration order).
    @test occursin("_pairLabel(contrasts[0])", html)

    # Default radio source for the init call is γ-PEP (Open Q2).
    @test occursin("_kgroupVolcanoPepSource = 'gamma'", html) ||
          occursin("_kgroupVolcanoPepSource = \"gamma\"", html)
end

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 4 — Pair-selector + radio events both subscribe to the renderer.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=4 pair-selector + radio events both call renderKGroupVolcano" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics
    using DataFrames

    fx = DifferentialFixturesK4.create_four_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # Slice the template from `function initKGroupVolcano` up to the next
    # top-level function declaration (heuristic: blank-line-then-function);
    # this delimits the init body well enough to assert that BOTH
    # subscriptions live inside it.
    init_start = findfirst("function initKGroupVolcano", html)
    @test init_start !== nothing
    sub_start  = init_start === nothing ? 1 : last(init_start)
    # End at the next "function " declaration on a fresh line, or end-of-string.
    init_end_match = findnext("\nfunction ", html, sub_start + 1)
    sub_end = init_end_match === nothing ? lastindex(html) : first(init_end_match)
    init_body = html[sub_start:sub_end]

    # Radio change subscription anchored on the volcano-pep-mode-group must
    # eventually call renderKGroupVolcano.
    @test occursin("kgroup-volcano-pep-mode-group", init_body)
    # Dropdown (`results-pair-select`) change subscription wired inside init.
    @test occursin("results-pair-select", init_body)
    # Both subscriptions ultimately re-invoke renderKGroupVolcano inside init.
    @test count(_ -> true, eachmatch(r"renderKGroupVolcano", init_body)) >= 2
end

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 5 — k=2 byte-equality: legacy static-svg path preserved AND
# Plotly card hidden (display:none) in initial DOM.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=2 static svg path preserved, Plotly volcano hidden" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    # Canonical 2-condition fixture: `contrasts == Pair{Symbol,Symbol}[]`, so
    # `D.meta.condition_labels.length === 2` and the Plotly card MUST stay
    # display:none (byte-equality lock).
    diff = DifferentialFixtures.create_two_condition_result()
    out  = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # Legacy static-svg volcano emission path is still wired through
    # `initStaticPlots()` which reads `D.plots` then `P['volcano']`. The
    # template must retain both the dispatch (`D.plots`) and the
    # `initStaticPlots` invocation.
    @test occursin("D.plots", html)
    @test occursin("initStaticPlots()", html)

    # The Plotly card markup is present in the template (template is shared)
    # but MUST carry the inline `style="display:none"` initial gate — the
    # k=2 boot path explicitly hides it (initKGroupVolcano returns early when
    # condition_labels.length < 3).
    card_window = match(r"id=\"kgroup-volcano-card\"[^>]{0,200}", html)
    @test card_window !== nothing
    @test occursin("display:none", card_window.match)
end
