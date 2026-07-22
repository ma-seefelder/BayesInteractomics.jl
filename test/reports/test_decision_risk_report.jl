# test/reports/test_decision_risk_report.jl
#
# Decision Risk report-level testitems.
# 3 report-regex items + 1 synthetic-badge spec lock.
#
# Quick run:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("test_decision_risk_report", ti.filename)'

using TestItemRunner

# -----------------------------------------------------------------------------
# Testitem 1 - badge HTML present in rendered template (structural check)
# -----------------------------------------------------------------------------
@testitem "badge HTML present in rendered template (structural check)" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    fx = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (a = fx.ar_wt, b = fx.ar_mut1),
        contrasts = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)
    @test occursin("text-bg-warning", html)
    @test occursin("MAP:", html)
    @test occursin("Optimal:", html)
end

# -----------------------------------------------------------------------------
# Testitem 2 - Validation Candidates pill rendered in Results tab
# -----------------------------------------------------------------------------
@testitem "Validation Candidates pill rendered in Results tab" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    fx = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (a = fx.ar_wt, b = fx.ar_mut1),
        contrasts = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)
    @test occursin("Validation Candidates", html)
    @test occursin("validation-candidates-tab", html)
    @test occursin("validation-candidates-table", html)
    @test occursin("validation-candidates-cards", html)
    @test occursin("validation_candidates", html)
end

# -----------------------------------------------------------------------------
# Testitem 3 - Decision Risk heatmap div present for k>=3; suppressed for k=2
# -----------------------------------------------------------------------------
@testitem "Decision Risk heatmap div present for k>=3; suppressed for k=2" setup=[DifferentialFixtures] begin
    using BayesInteractomics

    fx = DifferentialFixtures.create_three_condition_result()
    diff_k2 = differential_analysis(
        conditions = (a = fx.ar_wt, b = fx.ar_mut1),
        contrasts = :all_pairs,
    )
    out2 = tempname() * ".html"
    generate_differential_report(diff_k2; output = out2)
    html_k2 = read(out2, String)
    @test !occursin("\"empty_state\":false", html_k2)

    diff_k3 = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts = :all_pairs,
    )
    out3 = tempname() * ".html"
    generate_differential_report(diff_k3; output = out3)
    html_k3 = read(out3, String)
    @test occursin("decision-risk-heatmap", html_k3)
    @test occursin("decision_risk_heatmap", html_k3)
    @test occursin("reversescale", html_k3)
    @test occursin("Viridis", html_k3)
end

# -----------------------------------------------------------------------------
# Testitem 4 - synthetic badge spec-lock (Example 2 forces MAP != Optimal)
# -----------------------------------------------------------------------------
@testitem "synthetic Example 2 forces MAP != Optimal; badge tooltip text present" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using BayesInteractomics: GAINED
    using DataFrames

    nt = compute_decision_risk([0.40], [0.55], [0.60], [0.90], [GAINED])
    @test nt.optimal_call[1] == :unchanged
    @test nt.optimal_call[1] != :gained
    @test isapprox(nt.decision_risk[1], 3.4516129032258065, atol=1e-12)

    df_synthetic = DataFrame(
        Protein           = ["P_DEMO"],
        pep_gained        = [0.40],
        pep_reduced       = [0.55],
        pep_unchanged     = [0.60],
        pep_both_negative = [0.90],
        classification    = [GAINED],
    )
    compute_decision_risk!(df_synthetic)
    @test df_synthetic.optimal_call[1] == :unchanged
    @test lowercase(string(df_synthetic.classification[1])) == "gained"

    fx = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (a = fx.ar_wt, b = fx.ar_mut1),
        contrasts = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    @test occursin("class=\"badge text-bg-warning\"", html)
    @test occursin("data-bs-toggle=\"tooltip\"", html)
    @test occursin("(risk saved:", html)
end

# -----------------------------------------------------------------------------
# Testitem 5 - spec-drift trap: Methods tab worked-example numbers must be
# byte-equal to compute_decision_risk helper output (W-4 pinned form).
# -----------------------------------------------------------------------------
@testitem "Methods tab worked-example numbers byte-equal to helper output (W-4 pinned form)" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using BayesInteractomics: GAINED

    # Step 1: run the implemented helper on the fixture gamma-PEPs
    nt1 = compute_decision_risk([0.05], [0.60], [0.85], [0.95], [GAINED])
    nt2 = compute_decision_risk([0.40], [0.55], [0.60], [0.90], [GAINED])

    # Step 2: format each risk via Julia's default `string(round(x; digits=4))`.
    # This is the SAME formatter the Methods tab worked-example numbers are
    # generated from -- `string(Float64)` strips trailing zeros, so
    # round(3.870967741935484; digits=4) prints as "3.871" not "3.8710".
    fmt(x) = (isnan(x) ? "NaN" : string(round(x; digits=4)))
    ex1_risks = (
        rg = fmt(nt1.risk_gained[1]),
        rr = fmt(nt1.risk_reduced[1]),
        ru = fmt(nt1.risk_unchanged[1]),
        rbn = fmt(nt1.risk_both_negative[1]),
        dr = fmt(nt1.decision_risk[1]),
    )
    ex2_risks = (
        rg = fmt(nt2.risk_gained[1]),
        rr = fmt(nt2.risk_reduced[1]),
        ru = fmt(nt2.risk_unchanged[1]),
        rbn = fmt(nt2.risk_both_negative[1]),
        dr = fmt(nt2.decision_risk[1]),
    )

    # Step 3: assert helper output matches the W-4 pinned values baked into
    # the Methods tab HTML. If a future helper change shifts the arithmetic,
    # this testitem will fail against the OLD Methods HTML, forcing the
    # Methods writer to update both in lockstep (spec-drift trap).
    @test ex1_risks.rg  == "2.9677"
    @test ex1_risks.rr  == "6.5161"
    @test ex1_risks.ru  == "4.3871"
    @test ex1_risks.rbn == "4.4516"
    @test ex1_risks.dr  == "2.9677"
    # W-4 PIN: Julia's default `string(Float64)` strips trailing zeros.
    # risk_gained Ex2 = 3.8709677... -> round(_, digits=4) = 3.871 (a Float64
    # whose string representation has no trailing zero). Other Example 2
    # values happen to have non-zero 4th decimal digits.
    @test ex2_risks.rg  == "3.871"     # NO trailing zero (W-4 pinned form)
    @test ex2_risks.rr  == "4.8387"
    @test ex2_risks.ru  == "3.4516"
    @test ex2_risks.rbn == "3.6452"
    @test ex2_risks.dr  == "3.4516"

    # Step 4: assert each formatted value appears verbatim in the rendered HTML.
    fx = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (a = fx.ar_wt, b = fx.ar_mut1),
        contrasts = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)
    @test occursin("2.9677", html)
    @test occursin("6.5161", html)
    @test occursin("4.3871", html)
    @test occursin("4.4516", html)
    @test occursin("3.871", html)    # W-4 pinned form for risk_gained Ex 2
    @test !occursin("3.8710", html)  # Anti-assert: trailing-zero form forbidden
    @test occursin("4.8387", html)
    @test occursin("3.4516", html)
    @test occursin("3.6452", html)

    # Step 5: assert the Optimal-call labels match verbatim Methods tab claims.
    @test occursin(":gained</code>", html)
    @test occursin(":unchanged</code>", html)
    @test occursin("lowest risk = 2.9677", html)
    @test occursin("lowest risk = 3.4516", html)
end
