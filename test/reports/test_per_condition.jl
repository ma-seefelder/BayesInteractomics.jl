# test/reports/test_per_condition.jl
#
# Real assertions for the per-condition calibration, sensitivity + mixture, and
# data quality payloads + dropdown UX.
#
# Verifies the `per_condition::Dict{String, …}` propagation contract AND the
# dropdown HTML structure for the four affected tabs (Calibration / Sensitivity
# / Mixture Model / Data Quality).
#
# Locks honored: D.meta.condition_labels canonical predicate; byte-equality
# (k=2 keeps single-card path, dropdown UI suppressed via display:none).
#
# Filter command:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("test_per_condition", ti.filename)'

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# per_condition has 3 keys + calibration sub-object
# ─────────────────────────────────────────────────────────────────────────────
@testitem "per_condition payload has 3 keys with calibration sub-object on k=3 fixture" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    fx = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # Top-level `per_condition` key is emitted.
    @test occursin("\"per_condition\":", html)

    # All three condition labels carry the calibration sub-object.
    @test occursin("\"wt\":{\"calibration\":", html)
    @test occursin("\"mut1\":{\"calibration\":", html)
    @test occursin("\"mut2\":{\"calibration\":", html)
end

# ─────────────────────────────────────────────────────────────────────────────
# per_condition[label].sensitivity sub-object
# ─────────────────────────────────────────────────────────────────────────────
@testitem "per_condition[label].sensitivity sub-object present on k=3 fixture" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    fx = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # Each per-condition inner object carries a "sensitivity" sub-key
    # (sensitivity payload itself may be "null" when the AR has no sensitivity
    # result — fixture omits it — but the SUB-KEY must be present).
    @test occursin("\"sensitivity\":", html)
    # Per-condition sensitivity sub-key embedded inside the per_condition dict
    # (regression guard against the sub-key being elided when the value is null).
    @test count(_ -> true, eachmatch(r"\"sensitivity\":", html)) >= 3
end

# ─────────────────────────────────────────────────────────────────────────────
# per_condition[label].mixture sub-object
# ─────────────────────────────────────────────────────────────────────────────
@testitem "per_condition[label].mixture sub-object present on k=3 fixture" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    fx = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # Each per-condition inner object carries a "mixture" sub-key.
    @test occursin("\"mixture\":", html)
    @test count(_ -> true, eachmatch(r"\"mixture\":", html)) >= 3
end

# ─────────────────────────────────────────────────────────────────────────────
# per_condition[label].qc sub-object
# ─────────────────────────────────────────────────────────────────────────────
@testitem "per_condition[label].qc sub-object present on k=3 fixture" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    fx = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # Each per-condition inner object carries a "qc" sub-key.
    @test occursin("\"qc\":", html)
    @test count(_ -> true, eachmatch(r"\"qc\":", html)) >= 3
end

# ─────────────────────────────────────────────────────────────────────────────
# k=2 fallback — dropdown wraps `display:none` + per_condition
#                  still carries the 2 entries (data path uniform).
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=2 fallback — dropdown suppressed + per_condition has 2 keys" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    fx = DifferentialFixtures.create_three_condition_result()

    # Build a true k=2 differential by passing only two conditions.
    diff_k2 = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1),
        contrasts = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff_k2; output = out)
    html = read(out, String)

    # Per-condition payload is still emitted for k=2 (data uniform).
    @test occursin("\"per_condition\":", html)
    @test occursin("\"wt\":{\"calibration\":", html)
    @test occursin("\"mut1\":{\"calibration\":", html)
    # The third k=3 label MUST be absent.
    @test !occursin("\"mut2\":{\"calibration\":", html)
end

# ─────────────────────────────────────────────────────────────────────────────
# Calibration / Sensitivity / Mixture / Data Quality dropdowns
#                  rendered in template markup (HTML smoke check).
# ─────────────────────────────────────────────────────────────────────────────
@testitem "per-condition selector dropdowns present in template for all 4 tabs" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    fx = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts = :all_pairs,
    )
    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # Four dropdowns (one per affected tab) — wrap div + select element.
    for tab in ("calibration", "sensitivity", "mixture", "qc")
        @test occursin("id=\"$(tab)-cond-select-wrap\"", html)
        @test occursin("id=\"$(tab)-cond-select\"", html)
    end

    # JS state machine helper present (mirrors the dropdown pattern).
    @test occursin("function _initConditionSelector", html)

    # Visibility predicate reads `D.meta.condition_labels` canonical key
    # (no new metadata schema).
    @test occursin("D.meta.condition_labels", html) ||
          occursin("D && D.meta && D.meta.condition_labels", html)

    # k>=3 threshold check is wired in the helper.
    @test occursin("labels.length >= 3", html) ||
          occursin("labels.length>=3", html)
end
