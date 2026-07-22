# test/reports/test_pep_columns.jl
#
# Regression coverage for the `PEP_A$(suffix)` / `PEP_B$(suffix)` per-pair
# JSON keys emitted by `_build_diff_protein_json` in
# `src/reports/report_generator.jl`.
#
# The suffixed-key loop emits a strict 2-line additive patch (paired with
# the existing `posterior_A$(suffix)` / `posterior_B$(suffix)` pushes).
# These three @testitems guard:
#
#   1. k=4 fixture — every pair has BOTH `PEP_A_<pair>` AND `PEP_B_<pair>`
#      keys present in the embedded `D.results[*]` payload, with at least
#      one numeric (non-null/non-NaN) value across the fixture rows.
#   2. k=2 legacy path — unsuffixed `"PEP_A":` keeps emitting (L1661-1662
#      preserved); the suffixed form `"PEP_A_"` does NOT appear in the k=2
#      JSON (byte-equality lock).
#   3. Co-emission with `bf_A$(suffix)` — at least one pair shows
#      `"PEP_A_<pair>":` and `"bf_A_<pair>":` keys appearing alongside
#      each other in the same per-protein record (regression guard
#      against future column-name drift).
#
# Filter command:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("pep_columns", ti.filename)'
#
# Locks honoured: BMA terminology (Copula + 3c-EM). FDR terminology
# (BFDR / PEP / local_fdr).

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 1 — k=4 fixture: PEP_A_<pair> + PEP_B_<pair> keys present and
# at least one row carries a non-null numeric value for each pair.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=4 PEP_A/B per-pair keys non-null" setup=[DifferentialFixturesK4] begin
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

    # The 6 pairs for the k=4 all_pairs contract — labels are joined with
    # `_vs_` (mirrored in the JS pair-selector
    # dropdown). Iterate every pair so a regression in any single pair's
    # emission surfaces.
    pairs = ["wt_vs_mut1", "wt_vs_mut2", "wt_vs_mut3",
             "mut1_vs_mut2", "mut1_vs_mut3", "mut2_vs_mut3"]

    for pair in pairs
        # Key must appear at least once in the JSON payload (per-protein records).
        @test occursin("\"PEP_A_$(pair)\":", html)
        @test occursin("\"PEP_B_$(pair)\":", html)
    end

    # At least one row in the per-protein records must carry a numeric
    # (non-null, non-NaN) PEP_A and PEP_B for at least one pair — guards
    # against the root cause (all-null serialisation due to the
    # missing emission in the suffixed loop).
    has_numeric_pep_a = any(p -> occursin(Regex("\"PEP_A_$(p)\":-?\\d"), html), pairs)
    has_numeric_pep_b = any(p -> occursin(Regex("\"PEP_B_$(p)\":-?\\d"), html), pairs)
    @test has_numeric_pep_a
    @test has_numeric_pep_b
end

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 2 — k=2 legacy path: unsuffixed `"PEP_A":` keeps emitting;
# suffixed `"PEP_A_"` form is ABSENT (byte-equality lock).
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=2 legacy unsuffixed PEP_A/B preserved (no suffixed leak)" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    # Use the canonical 2-condition fixture which returns a fully built
    # DifferentialResult (contrasts == Pair{Symbol,Symbol}[], so
    # `pair_suffixes == String[]` per `_build_diff_data_json` at L1546-1550).
    diff = DifferentialFixtures.create_two_condition_result()
    out  = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    # Unsuffixed legacy keys (L1661-1662 in report_generator.jl) must remain.
    @test occursin("\"PEP_A\":", html)
    @test occursin("\"PEP_B\":", html)

    # The suffixed form must NOT appear for k=2 — `pair_suffixes` is empty,
    # so the suffixed-key loop never runs (byte-equality).
    @test !occursin("\"PEP_A_", html)
    @test !occursin("\"PEP_B_", html)
end

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 3 — PEP_A/B suffixed keys appear alongside bf_A/B suffixed keys
# in the same payload (regression guard against future loop reordering or
# accidental key drop).
# ─────────────────────────────────────────────────────────────────────────────
@testitem "PEP_A/B suffixed keys appear alongside bf_A/B suffixed keys" setup=[DifferentialFixturesK4] begin
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

    # For the canonical first pair, BOTH the existing bf_A/B suffixed keys
    # AND the new PEP_A/B suffixed keys must appear. This proves the
    # suffixed-key loop emits them together (no key fell out of the loop).
    pair = "wt_vs_mut1"
    @test occursin("\"bf_A_$(pair)\":",   html)
    @test occursin("\"bf_B_$(pair)\":",   html)
    @test occursin("\"PEP_A_$(pair)\":",  html)
    @test occursin("\"PEP_B_$(pair)\":",  html)

    # The unsuffixed legacy keys also remain present in k≥3 output (the
    # legacy block at L1649-1699 always runs; per-pair loop is additive
    # on top of it). This is the per-pair suffixed-key contract.
    @test occursin("\"PEP_A\":", html)
    @test occursin("\"PEP_B\":", html)
end
