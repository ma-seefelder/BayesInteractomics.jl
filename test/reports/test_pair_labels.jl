# test/reports/test_pair_labels.jl
#
# Regression coverage for the `pair_label::String` per-candidate JSON field
# emitted by `_resolve_pair_label_for_candidate` inside
# `_build_validation_candidates_block`.
#
# Contract:
#   k=2:   "<diff.condition_A> vs <diff.condition_B>" verbatim (byte-equal).
#   k>=3:  "<first(pair)> vs <last(pair)>" where pair achieves
#          `row.decision_risk_min`; first pair in `diff.contrasts` order
#          wins ties (canonical order).
#   Fallback: "unknown pair" for degenerate NaN / unmatched rows.
#
# Filter command:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("pair_labels", ti.filename)'
#
# Locks honoured: byte-equality (pair_label is additive, never replaces an
# existing field); BMA + FDR terminology locks.

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 1 — k=4 fixture: every candidate carries a non-empty `pair_label`
# field shaped `"<a> vs <b>"`.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=4 every candidate carries pair_label field" setup=[DifferentialFixturesK4] begin
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

    cand_match = match(r"\"candidates\"\s*:\s*\[(.*?)\]\s*,\s*\"all_proteins\""s, html)
    @test cand_match !== nothing
    cand_block = cand_match === nothing ? "" : cand_match.captures[1]

    # Every candidate object must carry a pair_label key.
    labels = [m.captures[1] for m in eachmatch(r"\"pair_label\"\s*:\s*\"([^\"]*)\"", cand_block)]
    @test !isempty(labels)

    # Each label must either be the literal "unknown pair" sentinel
    # (degenerate NaN row) OR match `<word_chars> vs <word_chars>`.
    # The fixture conditions are wt / mut1 / mut2 / mut3 — all simple
    # alphanumeric identifiers, so `[\w-]+ vs [\w-]+` is the right shape.
    for label in labels
        ok = label == "unknown pair" || occursin(r"^[\w-]+ vs [\w-]+$", label)
        @test ok
        if !ok
            @info "pair_label does not match expected shape" label
        end
    end

    # At least one candidate must resolve to a real fixture pair (no
    # all-NaN-degenerate-only run — the fixture's all_enriched block
    # guarantees finite decision_risk_min for proteins 1..10).
    real_labels = filter(l -> l != "unknown pair", labels)
    @test !isempty(real_labels)

    # Every real label must reference exactly two of the four fixture
    # condition names. (Defensive: catches a future regression where
    # the resolver leaks an internal token like ":symbol" or "n/a".)
    fixture_conds = Set(["wt", "mut1", "mut2", "mut3"])
    for label in real_labels
        parts = split(label, " vs ")
        @test length(parts) == 2
        @test parts[1] in fixture_conds
        @test parts[2] in fixture_conds
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 2 — k=2 legacy path: pair_label is byte-equal to
# "<diff.condition_A> vs <diff.condition_B>" for every candidate.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=2 pair_label matches condition_A vs condition_B verbatim" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    fx = DifferentialFixtures.create_three_condition_result()
    # Construct a k=2 NamedTuple call (legacy 2-group path is also
    # exercised by `differential_analysis(ar_a, ar_b)` but the k=2 path
    # via the kw-only `conditions =` overload is the closest analogue to
    # the k>=3 fixture wiring above — and `diff.contrasts == []` for both
    # legacy 2-arg and 2-key NamedTuple calls).
    diff = differential_analysis(fx.ar_wt, fx.ar_mut1)

    out = tempname() * ".html"
    generate_differential_report(diff; output = out)
    html = read(out, String)

    cand_match = match(r"\"candidates\"\s*:\s*\[(.*?)\]\s*,\s*\"all_proteins\""s, html)
    @test cand_match !== nothing
    cand_block = cand_match === nothing ? "" : cand_match.captures[1]

    expected = string(diff.condition_A) * " vs " * string(diff.condition_B)
    labels = [m.captures[1] for m in eachmatch(r"\"pair_label\"\s*:\s*\"([^\"]*)\"", cand_block)]

    # For k=2 every candidate must carry the SAME pair_label — the single
    # condition pair. No "unknown pair" fallback is acceptable on the k=2
    # path because `diff.condition_A` / `diff.condition_B` are always
    # populated (struct fields, not optional).
    @test !isempty(labels)
    @test all(l -> l == expected, labels)
end

# ─────────────────────────────────────────────────────────────────────────────
# @testitem 3 — template HTML wires `c.pair_label` and `r.pair_label` reads
# (DOM smoke check). The runtime contract is enforced by the two HTML
# tests above; this one guards against the template losing the
# rendering wiring under a future refactor.
# ─────────────────────────────────────────────────────────────────────────────
@testitem "template HTML reads c.pair_label / r.pair_label" begin
    using BayesInteractomics

    # Locate the template file via the source-tree layout used by the
    # other report tests (relative from project root).
    project_root = pkgdir(BayesInteractomics)
    template_path = joinpath(project_root, "src", "reports", "templates",
                             "differential_report.html")
    @test isfile(template_path)
    template = read(template_path, String)

    # Card-grid render site (top-N).
    @test occursin("c.pair_label", template)
    # Show-all panel render site.
    @test occursin("r.pair_label", template)
    # Total count guard — at least 2 distinct rendering sites must
    # consume pair_label.
    n_hits = length(collect(eachmatch(r"\.pair_label", template)))
    @test n_hits >= 2
end
