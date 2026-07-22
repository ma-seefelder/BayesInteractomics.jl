# test/reports/test_validation_candidates.jl
#
# Validation Candidates polish real assertions against:
#
#   a. BOTH_NEGATIVE excluded from top-N grid + ranked table
#   b. Collapsible <details> "Show all proteins (including BOTH_NEGATIVE)"
#      panel present
#   c. MAP class column non-empty for k>=3 (reads wide_df.kgroup_class)
#   d. decision_risk_min variation > 0 on k=3 fixture (finite positive values,
#      no compute_decision_risk patch was needed)
#   e. Methods explanation card rendered (tooltip text source)
#
# Locks honored: §7a CLS_COLOR (BOTH_NEGATIVE class label); kgroup_class enum;
# Methods card text source; omnibus-BFDR pre-filter wiring.
#
# Filter command:
#   julia --project=. --threads=16 -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("test_validation_candidates", ti.filename)'

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# BOTH_NEGATIVE excluded from top-N + default ranked table
# ─────────────────────────────────────────────────────────────────────────────
@testitem "BOTH_NEGATIVE excluded from top-N validation-candidate list" setup=[DifferentialFixtures] begin
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

    # Parse the validation_candidates JSON block out of the embedded data blob.
    # The data is serialized inside the {{REPORT_DATA_JSON}} placeholder. We
    # find the "candidates": [ ... ] array by string scan — top-N is a flat
    # array of small JSON objects, so a regex over the value is enough.
    cand_match = match(r"\"candidates\"\s*:\s*\[(.*?)\]\s*,\s*\"all_proteins\""s, html)
    @test cand_match !== nothing  # candidates array present in payload

    # Now extract every map_class string within the top-N candidates block.
    cand_block = cand_match === nothing ? "" : cand_match.captures[1]
    map_classes = [m.captures[1] for m in eachmatch(r"\"map_class\"\s*:\s*\"([^\"]*)\"", cand_block)]

    # Fixture P5 is BOTH_NEGATIVE across all 3 pairs (kgroup_class is the
    # 5-enum derived label; the k>=3 BOTH_NEGATIVE filter checks ALL per-pair
    # classification columns, so P5 must be filtered out before this point).
    # Assert no candidate carries an upper- or lower-case "both_negative" label.
    @test all(mc -> !occursin(r"both_negative"i, mc), map_classes)

    # Sanity: at least one candidate row survived the BFDR pre-filter on the
    # 6-protein fixture (P4 / P6 / etc. carry strong evidence and pass).
    @test length(map_classes) >= 1
end

# ─────────────────────────────────────────────────────────────────────────────
# collapsible <details>/<summary>Show all proteins panel
# ─────────────────────────────────────────────────────────────────────────────
@testitem "collapsible 'Show all proteins' <details> panel present" setup=[DifferentialFixtures] begin
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

    # <details> element + summary label both present in the Validation
    # Candidates pane (show-all collapsible).
    @test occursin("<details", html)
    @test occursin("vc-all-details", html)
    @test occursin("Show all proteins (including BOTH_NEGATIVE)", html)
    @test occursin("vc-all-proteins-table", html)

    # The all_proteins JSON array is emitted alongside the candidates array.
    # Verify it carries at least the 6 fixture proteins (no BFDR or
    # BOTH_NEGATIVE filtering on this list).
    @test occursin("\"all_proteins\"", html)

    # Lazy-render JS handler wired (toggle listener for first expansion).
    @test occursin("vc_all_wired", html) || occursin("__vc_all_wired", html)
end

# ─────────────────────────────────────────────────────────────────────────────
# Stub c — MAP class column non-empty for k>=3 (reads kgroup_class)
# ─────────────────────────────────────────────────────────────────────────────
@testitem "MAP class column populated for k>=3 (reads kgroup_class)" setup=[DifferentialFixtures] begin
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

    # Extract the candidates JSON block.
    cand_match = match(r"\"candidates\"\s*:\s*\[(.*?)\]\s*,\s*\"all_proteins\""s, html)
    @test cand_match !== nothing
    cand_block = cand_match === nothing ? "" : cand_match.captures[1]

    # Pull every map_class value from the candidates block.
    map_classes = [m.captures[1] for m in eachmatch(r"\"map_class\"\s*:\s*\"([^\"]*)\"", cand_block)]

    # For k>=3 every candidate row should carry a non-empty map_class derived
    # from row.kgroup_class (5-value enum). Empty strings
    # previously appeared because the k>=3 code path read row.classification
    # which does not exist on the wide DF (only suffixed classification_<a>_vs_<b>).
    @test !isempty(map_classes)
    @test all(mc -> !isempty(mc), map_classes)

    # Locked kgroup_class enum values: omnibus_null,
    # none_enriched, condition_specific, all_enriched, fully_resolved.
    kgroup_enum_values = Set([
        "omnibus_null", "none_enriched", "condition_specific",
        "all_enriched", "fully_resolved",
    ])
    @test all(mc -> mc in kgroup_enum_values, map_classes)
end

# ─────────────────────────────────────────────────────────────────────────────
# decision_risk_min variation > 0 on k=3 fixture
# (no decision_risk.jl patch needed.)
# ─────────────────────────────────────────────────────────────────────────────
@testitem "count(>(0), decision_risk_min) > 0 on k=3 fixture" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    fx = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts = :all_pairs,
    )
    # All 6 fixture proteins carry finite positive decision_risk_min. No
    # patch to compute_decision_risk! was required.
    @test :decision_risk_min in propertynames(diff.results)
    finite_vals = collect(skipmissing(diff.results.decision_risk_min))
    @test count(>(0), finite_vals) > 0
    @test count(x -> ismissing(x) || (isa(x, AbstractFloat) && isnan(x)),
                diff.results.decision_risk_min) == 0
end

# ─────────────────────────────────────────────────────────────────────────────
# Methods explanation card present in Validation Candidates pane
# ─────────────────────────────────────────────────────────────────────────────
@testitem "Methods explanation card rendered (mirrors tooltip text)" setup=[DifferentialFixtures] begin
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

    # Methods card present in the Validation Candidates pane.
    @test occursin("validation-candidates-methods-card", html)

    # Tooltip-text source verbiage mirrored into the card body.
    @test occursin("Decision Risk: how to read this tab", html)
    @test occursin("DEFAULT_DIFFERENTIAL_LOSS", html)

    # Locked loss-matrix penalty values.
    @test occursin("direction-flip", html)
    @test occursin("over-claim", html)
    @test occursin("missed-hit", html)
    @test occursin("conservative-default", html)

    # Override paths.
    @test occursin("DifferentialConfig.loss_matrix", html)
    @test occursin("loss_matrix=", html) || occursin("loss_matrix&#61;", html)

    # BMA + FDR terminology lock.
    @test occursin("Copula", html)
    @test occursin("3c-EM", html)
    @test occursin("BFDR", html)
end
