# test/reports/test_report_phase_number_strip.jl
#
# Publication-cleanliness acceptance guard: a freshly generated report must
# not leak internal-process references into its output. This test regenerates
# BOTH report.html and differential_report.html from in-memory fixtures and
# greps the whole rendered DOM for the full internal-process token family
# (version-tag literals, planning-path references, internal decision/label
# IDs, assistant-attribution strings), asserting zero hits after subtracting a
# small, documented carve-out set.
#
# The narrower legacy testitems below scan the differential report's tooltip
# `title=` attributes and Validation-Candidates panel text for version-tag
# literals, plus a sanity guard that the Methods tab retains its legitimate
# domain terminology so an over-eager strip pass cannot silently wipe it.
#
# Carve-outs (legitimately allowed to remain, NOT asserted against):
#   - HTML/JS/CSS comments and Julia `#` / `"""..."""` source comments
#     (these never reach the rendered DOM text)
#   - Methods-tab content where domain terminology citations are legitimate
#   - The AlphaFold-Server docking workflow "Phase 1/2/3" step language
#   - This file's own detection-regex literal (a gate self-carve-out — the
#     acceptance regex string must appear here for the test to encode it)
#
# Filter command:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("report_phase_number_strip", ti.filename)'
#
# Terminology that MUST survive: BMA sub-models "Copula" + "3c-EM"; FDR terms
# BFDR / PEP / local_fdr.

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# testitem 1 — no version-tag literals inside any `data-bs-toggle="tooltip"`
# attribute's title= value (post-render). The tooltip body is the most
# likely accidental leak channel because tooltip strings are often pasted
# from internal development notes.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "no version-tag literals in tooltips (k=4)" setup=[DifferentialFixturesK4] begin
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

    # Scan every `data-bs-toggle="tooltip"` element's title= attribute.
    # The regex is intentionally tolerant of attribute ordering
    # (title= may precede or follow data-bs-toggle on the same element)
    # but tightly bounded by the tag's `>` close so we don't sweep into
    # other elements' content.
    bad_titles = String[]
    # Forward order: data-bs-toggle="tooltip" ... title="..."
    for m in eachmatch(r"data-bs-toggle=\"tooltip\"[^>]*?title=\"([^\"]*)\""s, html)
        title_val = m.captures[1]
        occursin(r"Phase 7[0-5]"i, title_val) && push!(bad_titles, title_val)
    end
    # Reverse order: title="..." ... data-bs-toggle="tooltip"
    for m in eachmatch(r"title=\"([^\"]*)\"[^>]*?data-bs-toggle=\"tooltip\""s, html)
        title_val = m.captures[1]
        occursin(r"Phase 7[0-5]"i, title_val) && push!(bad_titles, title_val)
    end
    @test isempty(bad_titles)
    if !isempty(bad_titles)
        @info "version-tag literal leaked into tooltip title=" bad_titles
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# testitem 2 — no version-tag literals inside the Validation Candidates panel's
# rendered text (post-comment-strip). We extract the `<div ... id="validation-
# candidates" ...>...</div>` panel, strip HTML/JS/CSS comments + Julia-style
# docstring blocks, then assert no version-tag literal remains in the visible-text payload.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "no version-tag literals in Validation Candidates panel text (k=4)" setup=[DifferentialFixturesK4] begin
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

    # Anchor at `id="validation-candidates"` and walk forward until the
    # next sibling `<div class="tab-pane`. Using a non-greedy capture
    # bounded by the next tab-pane open is more robust than DOM parsing.
    panel_match = match(
        r"id=\"validation-candidates\"[^>]*>(.*?)<div class=\"tab-pane"s,
        html,
    )
    @test panel_match !== nothing
    panel_html = panel_match === nothing ? "" : panel_match.captures[1]

    # Strip HTML comments, JS line/block comments, and CSS block comments
    # before searching — those are allowed to retain version-tag
    # citations (the Methods-subsection exemption applies at the
    # source-code level; runtime DOM text is what the user sees).
    stripped = replace(panel_html, r"<!--.*?-->"s => "")
    stripped = replace(stripped, r"/\*.*?\*/"s => "")
    # Strip `// ...` JS line comments (avoid hitting `https://` URLs by
    # requiring the `//` to not be preceded by `:`).
    stripped = replace(stripped, r"(?<![:/])//[^\n]*" => "")

    # Now assert NO `Phase 7[0-5]` literal survives.
    @test !occursin(r"Phase 7[0-5]"i, stripped)
end

# ─────────────────────────────────────────────────────────────────────────────
# testitem 3 — sanity guard: Methods tab MUST retain its domain terminology.
# This is the inverse contract — we must not over-strip. The Methods tab is
# the authoritative in-report reference for the analysis method.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "Methods tab retains domain terminology (sanity guard, k=4)" setup=[DifferentialFixturesK4] begin
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

    # The report's Methods tab carries text that legitimately cites the
    # locked domain terminology (BMA linear BF pooling, BFDR rename, the
    # kgroup_class enum). At minimum the report must mention at least ONE
    # of the locked terminology strings so the Methods text is not
    # accidentally wiped by an over-eager strip pass. We look for the
    # locked BMA + FDR terms rather than any version-tag string (the
    # user-visible content is the locked terminology, not version numbers).
    @test occursin("Copula", html)
    @test occursin("3c-EM", html)
    @test occursin("BFDR", html)
    @test occursin("DEFAULT_DIFFERENTIAL_LOSS", html)
end

# ─────────────────────────────────────────────────────────────────────────────
# testitem 4 — whole-DOM publication-cleanliness guard across BOTH reports.
# For EACH of report.html (single-condition) and differential_report.html we
# read the ENTIRE rendered file body (not just tooltip title= / headings) and
# apply the FULL internal-process token detection family, asserting zero hits
# after subtracting the documented carve-out set. The only carve-out is the
# AlphaFold-Server docking / stage-1-EM workflow "Phase 1/2/3" step language,
# which is legitimate domain language and may remain.
#
# This is the permanent PUB-02 guard. It is INTENTIONALLY RED until the
# report-generator string cleanup + HTML template comment cleanup land — that
# is precisely the point of an acceptance test.
#
# Self-carve-out: this file necessarily contains the detection-regex literal
# (the token classes appear here as regex source), so this test file itself is
# NOT expected to be grep-clean of its own regex string.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "no internal-process tokens in generated report DOM (both reports)" setup=[DifferentialFixturesK4] begin
    using BayesInteractomics
    using DataFrames

    # Full internal-process token detection family (case-insensitive). Version
    # numbers are captured whole so the "Phase 1/2/3" carve-out below is applied
    # precisely — a bare "Phase 1" is a workflow step; "Phase 17" is a hit.
    detect_re = Regex(
        join([
            raw"\b(?:phase|spike|plan|wave)\s*[0-9]+(?:\.[0-9]+)?",  # version-tag literals
            raw"\bStream\s*[A-E]\b",                                  # Stream A-E (not evidence_streams)
            raw"\bD-[0-9]+[a-z]?\b",                                  # decision-lock IDs
            raw"\b(?:SAM|ABL|DMSK|MNAR|METAL|PRIOR|NORM|EMB|PEP|DRPT|DIAG|MOD|DRISK|AUTH)-[0-9]+",  # req-IDs (dash+digit, never bare PEP)
            raw"\.planning/",                                        # planning-dir path/prose ref
            raw"\bCLAUDE\.md\b",                                     # attribution
            raw"\bRFC\s*§?\s*[0-9]",                                 # internal design RFC
            raw"\bNyquist\b",                                        # validation-gate term
            raw"\bWARNING\s*#?[0-9]",                                # invariant-lock label
            raw"\b(?:claude|anthropic|copilot|genai|gsd)\b",        # assistant hints
            raw"Co-Authored-By",                                     # commit-trailer attribution
        ], "|"),
        "i",
    )

    # Documented carve-out (D-03): the docking / stage-1-EM workflow steps
    # "Phase 1/2/3" are legitimate domain language and are allowed to remain.
    is_carveout(s) = begin
        n = replace(lowercase(strip(String(s))), r"\s+" => " ")
        n in ("phase 1", "phase 2", "phase 3")
    end

    scan_dom(body) = begin
        v = String[]
        for m in eachmatch(detect_re, body)
            is_carveout(m.match) && continue
            push!(v, m.match)
        end
        v
    end

    # ---- Report 1: single-condition report.html (pure in-memory fixture) ----
    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "MYC",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )
    results = DataFrame(
        Protein        = ["ACTB", "TP53", "BRCA1", "EGFR", "MYC"],
        BF             = [200.0, 50.0, 15.0, 3.0, 0.8],
        posterior_prob = [0.99, 0.97, 0.85, 0.60, 0.35],
        PEP            = 1.0 .- [0.99, 0.97, 0.85, 0.60, 0.35],
        BFDR           = [0.001, 0.005, 0.04, 0.12, 0.50],
        mean_log2FC    = [4.0, 2.5, 1.8, 0.8, -0.2],
        sd_log2FC      = [0.3, 0.4, 0.5, 0.6, 0.7],
        bf_enrichment  = [150.0, 40.0, 10.0, 2.5, 0.6],
        bf_correlation = [50.0, 10.0, 5.0, 0.5, 0.2],
        bf_detected    = [20.0, 5.0, 2.0, 0.8, 0.1],
    )
    report_path = joinpath(tmpdir, "report.html")
    generate_report(results, cfg; output = report_path)
    @test isfile(report_path)
    report_html = read(report_path, String)
    report_violations = scan_dom(report_html)

    # ---- Report 2: differential_report.html (k=4 fixture) ----
    fx = DifferentialFixturesK4.create_four_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,
    )
    diff_out = tempname() * ".html"
    generate_differential_report(diff; output = diff_out)
    diff_html = read(diff_out, String)
    diff_violations = scan_dom(diff_html)

    if !isempty(report_violations)
        @info "internal-process tokens leaked into report.html" unique(report_violations)
    end
    if !isempty(diff_violations)
        @info "internal-process tokens leaked into differential_report.html" unique(diff_violations)
    end

    @test isempty(report_violations)
    @test isempty(diff_violations)
end
