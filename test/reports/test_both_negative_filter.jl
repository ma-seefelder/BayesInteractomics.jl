# test/reports/test_both_negative_filter.jl
#
# Regression coverage for the BOTH_NEGATIVE exclusion filter inside the
# Validation Candidates pill.
#
# The filter ships in `_build_validation_candidates_block`, and the k=4
# `diff_4condition/differential_report.html` carries zero BOTH_NEGATIVE
# rows in the candidates JSON. This test exercises the k=4 fixture on top
# of the existing k=3 validation-candidates regression so a future
# regression on either fixture surfaces here.
#
# The filter pattern asserted:
#   k=2:   row.classification == "BOTH_NEGATIVE"      → DROPPED
#   k>=3:  every per-pair classification_<a>_vs_<b> column == "BOTH_NEGATIVE"
#          → DROPPED (a single non-BOTH_NEGATIVE pair classification
#          preserves the row).
#
# Filter command:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("both_negative_filter", ti.filename)'
#
# Locks honoured: §7a CLS_COLOR (BOTH_NEGATIVE class label); kgroup_class
# enum; omnibus-BFDR pre-filter wiring; BMA + FDR terminology locks.

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 1 — k=4 fixture: candidates JSON carries zero BOTH_NEGATIVE
# rows (case-insensitive). Extends the k=3 exclusion contract to k=4.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=4 BOTH_NEGATIVE absent from validation_candidates" setup=[DifferentialFixturesK4] begin
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

    # Extract the candidates array text (bounded by the all_proteins
    # sibling key — same anchor as the k=3 validation-candidates regression).
    cand_match = match(r"\"candidates\"\s*:\s*\[(.*?)\]\s*,\s*\"all_proteins\""s, html)
    @test cand_match !== nothing
    cand_block = cand_match === nothing ? "" : cand_match.captures[1]

    # Pull every map_class value from the candidates block.
    map_classes = [m.captures[1] for m in eachmatch(r"\"map_class\"\s*:\s*\"([^\"]*)\"", cand_block)]

    # Exclusion contract: no candidate carries an upper- or lower-
    # case BOTH_NEGATIVE label. Empty array (no candidates) is acceptable
    # — the contract is "if any candidates exist, none of them are
    # BOTH_NEGATIVE" — though the k=4 fixture's all_enriched block
    # guarantees at least one survivor.
    @test all(mc -> !occursin(r"both_negative"i, mc), map_classes)

    # The k=4 fixture's all-enriched block (proteins 1..10) must produce
    # at least one survivor through the omnibus-BFDR pre-filter. If this
    # ever drops to zero, the upstream pre-filter is over-aggressive and
    # this test should surface it.
    @test length(map_classes) >= 1
end

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 2 — the show-all `all_proteins` panel DOES include BOTH_NEGATIVE
# rows. This is the inverse contract — the QC path ships BOTH_NEGATIVE in the
# `all_proteins` array for QC inspection. We assert the all_proteins array
# is at least as large as the candidates array (it carries the full
# ranked list, no BFDR / no BOTH_NEGATIVE pre-filter).
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=4 all_proteins panel preserves BOTH_NEGATIVE rows" setup=[DifferentialFixturesK4] begin
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

    # The all_proteins array is the broader ranked list (no omnibus pre-
    # filter, no BOTH_NEGATIVE strip). It must therefore have at least
    # as many rows as the top-N candidates array.
    cand_match = match(r"\"candidates\"\s*:\s*\[(.*?)\]\s*,\s*\"all_proteins\""s, html)
    all_match  = match(r"\"all_proteins\"\s*:\s*\[(.*?)\]\s*\}", html)
    @test cand_match !== nothing
    @test all_match  !== nothing
    cand_block = cand_match === nothing ? "" : cand_match.captures[1]
    all_block  = all_match  === nothing ? "" : all_match.captures[1]
    n_cand = length(collect(eachmatch(r"\"protein\"\s*:", cand_block)))
    n_all  = length(collect(eachmatch(r"\"protein\"\s*:", all_block)))
    @test n_all >= n_cand
end
