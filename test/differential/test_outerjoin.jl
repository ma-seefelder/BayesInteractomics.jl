# test/differential/test_outerjoin.jl
#
# Outer-join behaviour in `_aggregate_pairwise_results`.
#
# Five @testitem blocks covering:
#   1. nrow(wide_df) >= max(nrow(per_pair))
#   2. :n_pairs_with_data semantics
#   3. Identifier dedup (no _<a>_vs_<b>)
#   4. Missing (not NaN) for absent cells
#   5. k=2 byte-equality preserved
#
# BMA terminology: "Copula" + "3c-EM".
# FDR terminology: BFDR / PEP / local_fdr.
#
# Filter command:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("test_outerjoin", ti.filename)'

using TestItemRunner

# Helper: build a per-AR copula DataFrame matching the canonical `analyse()`
# pipeline shape (so the 2-group `differential_analysis(::AR, ::AR)` recursion
# invoked by `_runpair!` finds the columns it needs after `_rename_columns`).
# Inlined inside each @testitem (each testitem runs in an isolated module).
const _MK_DF_HELPER = """
    function _mk_copula_df(proteins::Vector{String}, bf::Vector{Float64}, log2fc::Vector{Float64})
        n = length(proteins)
        return DataFrame(
            Protein         = proteins,
            BF              = bf,
            bf_enrichment   = bf,
            bf_correlation  = ones(n),
            bf_detected     = ones(n),
            mean_log2FC     = log2fc,
            sd_log2FC       = fill(0.5, n),
            posterior_prob  = bf ./ (1 .+ bf),
            Component       = fill(1, n),
        )
    end

    function _mk_ar(label::String, df::DataFrame)
        AnalysisResult(
            df, DataFrame(),
            nothing, nothing, nothing,
            nothing, nothing, :bma,
            nothing, nothing,
            UInt64(0), UInt64(0),
            now(), "test",
            label, 1,
            nothing, nothing, nothing, :loaded,
            nothing, nothing,
            false,
            nothing,
        )
    end
"""

# ─────────────────────────────────────────────────────────────────────────────
# Stub 1 — nrow(wide_df) >= max(nrow(per_pair_df))
# ─────────────────────────────────────────────────────────────────────────────
@testitem "outerjoin row count >= max(per-pair) on disjoint protein sets" begin
    using BayesInteractomics
    using BayesInteractomics: AnalysisResult
    using DataFrames, Dates, Random

    function _mk_copula_df(proteins::Vector{String}, bf::Vector{Float64}, log2fc::Vector{Float64})
        n = length(proteins)
        return DataFrame(
            Protein         = proteins,
            BF              = bf,
            bf_enrichment   = bf,
            bf_correlation  = ones(n),
            bf_detected     = ones(n),
            mean_log2FC     = log2fc,
            sd_log2FC       = fill(0.5, n),
            posterior_prob  = bf ./ (1 .+ bf),
            Component       = fill(1, n),
        )
    end

    function _mk_ar(label::String, df::DataFrame)
        AnalysisResult(
            df, DataFrame(),
            nothing, nothing, nothing,
            nothing, nothing, :bma,
            nothing, nothing,
            UInt64(0), UInt64(0),
            now(), "test",
            label, 1,
            nothing, nothing, nothing, :loaded,
            nothing, nothing,
            false,
            nothing,
        )
    end

    Random.seed!(75)
    # Pairwise-disjoint AR design forces the union (legacy 2-group already
    # unions proteins within a pair) to differ across pairs. Concretely:
    #   pair (a,b) = union(ar1, ar2) = [P1,P2,P3,P4]
    #   pair (a,c) = union(ar1, ar3) = [P1,P2,P5,P6]
    #   pair (b,c) = union(ar2, ar3) = [P3,P4,P5,P6]
    # Outer-join across all three pairs yields [P1..P6] (6 proteins),
    # while max(nrow(per_pair)) = 4 — so the success criterion
    # 6 ≥ 4 holds strictly (not by equality).
    ar1 = _mk_ar("BAIT", _mk_copula_df(["P1","P2"], [200.0, 1.0], [3.0, 0.2]))
    ar2 = _mk_ar("BAIT", _mk_copula_df(["P3","P4"], [1.0, 200.0], [0.3, 3.0]))
    ar3 = _mk_ar("BAIT", _mk_copula_df(["P5","P6"], [1.0, 200.0], [0.2, 3.1]))

    diff = differential_analysis(
        conditions = (a = ar1, b = ar2, c = ar3),
        contrasts  = :all_pairs,
    )

    max_per_pair = maximum(nrow(df) for df in values(diff.pairwise_results))
    @test nrow(diff.results) >= max_per_pair             # success criterion
    @test nrow(diff.results) == 6                        # strict equality (designed)
    @test max_per_pair == 4                              # strict equality (designed)
    @test Set(String.(diff.results.Protein)) ==
          Set(["P1", "P2", "P3", "P4", "P5", "P6"])      # outer-join union of all pairs
end

# ─────────────────────────────────────────────────────────────────────────────
# Stub 2 — :n_pairs_with_data column exists and is in [1, n_contrasts]
# ─────────────────────────────────────────────────────────────────────────────
@testitem "n_pairs_with_data semantics (1 <= n_pairs <= n_contrasts)" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    fx = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts  = :all_pairs,
    )

    @test :n_pairs_with_data in propertynames(diff.results)
    n_contrasts = length(diff.contrasts)
    @test all(1 .<= diff.results.n_pairs_with_data .<= n_contrasts)
    # Fixture proteins (P1..P6) are present in every AR → every pair-DF has
    # the union (same protein set) → every protein lands in all n_contrasts pairs.
    @test all(diff.results.n_pairs_with_data .== n_contrasts)
end

@testitem "n_pairs_with_data on disjoint sets — proper subset counts" begin
    using BayesInteractomics
    using BayesInteractomics: AnalysisResult
    using DataFrames, Dates, Random

    function _mk_copula_df(proteins::Vector{String}, bf::Vector{Float64}, log2fc::Vector{Float64})
        n = length(proteins)
        return DataFrame(
            Protein = proteins, BF = bf,
            bf_enrichment = bf, bf_correlation = ones(n), bf_detected = ones(n),
            mean_log2FC = log2fc, sd_log2FC = fill(0.5, n),
            posterior_prob = bf ./ (1 .+ bf), Component = fill(1, n),
        )
    end
    function _mk_ar(label::String, df::DataFrame)
        AnalysisResult(df, DataFrame(),
            nothing, nothing, nothing, nothing, nothing, :bma, nothing, nothing,
            UInt64(0), UInt64(0), now(), "test",
            label, 1, nothing, nothing, nothing, :loaded, nothing, nothing,
            false, nothing)
    end

    Random.seed!(75)
    # Same disjoint design as Stub 1 — every protein lands in EXACTLY 2 of 3 pairs.
    ar1 = _mk_ar("BAIT", _mk_copula_df(["P1","P2"], [200.0, 1.0], [3.0, 0.2]))
    ar2 = _mk_ar("BAIT", _mk_copula_df(["P3","P4"], [1.0, 200.0], [0.3, 3.0]))
    ar3 = _mk_ar("BAIT", _mk_copula_df(["P5","P6"], [1.0, 200.0], [0.2, 3.1]))

    diff = differential_analysis(
        conditions = (a = ar1, b = ar2, c = ar3),
        contrasts  = :all_pairs,
    )
    @test :n_pairs_with_data in propertynames(diff.results)
    @test all(diff.results.n_pairs_with_data .== 2)      # exact count
end

# ─────────────────────────────────────────────────────────────────────────────
# Stub 3 — identifier columns are unsuffixed (no _<a>_vs_<b>)
# ─────────────────────────────────────────────────────────────────────────────
@testitem "identifier deduplication — no _<a>_vs_<b> suffix on shared columns" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    fx = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts  = :all_pairs,
    )

    cols = names(diff.results)
    # Identifier columns that should be present and UNSUFFIXED on the wide DF.
    # `:uniprot_id` / `:gene_name` are absent from the synthetic fixtures (no
    # curation step) — the forward-looking whitelist no-ops harmlessly. The k≥3
    # classification columns (`:enriched_in`, `:depleted_in`, `:kgroup_class`)
    # are added downstream by `_compute_kgroup_classification_columns!` and
    # MUST land unsuffixed on the wide DF.
    for ident in ("enriched_in", "depleted_in", "kgroup_class")
        @test ident in cols
        # Must NOT have a `<ident>_<a>_vs_<b>` variant.
        @test !any(c -> startswith(c, ident * "_") && occursin("_vs_", c), cols)
    end

    # Per-pair-suffixed numerical columns MUST be present (the whitelist
    # closes everything-not-on-ID_COLS to the `_<a>_vs_<b>` form).
    @test any(c -> startswith(c, "bf_A_") && occursin("_vs_", c), cols)
    @test any(c -> startswith(c, "classification_") && occursin("_vs_", c), cols)
    @test any(c -> startswith(c, "decision_risk_") && occursin("_vs_", c), cols)
end

# ─────────────────────────────────────────────────────────────────────────────
# Stub 4 — absent outerjoin cells land as Missing (not NaN)
# ─────────────────────────────────────────────────────────────────────────────
@testitem "missing cells render as Missing (not NaN) after outerjoin" begin
    using BayesInteractomics
    using BayesInteractomics: AnalysisResult
    using DataFrames, Dates, Random

    function _mk_copula_df(proteins::Vector{String}, bf::Vector{Float64}, log2fc::Vector{Float64})
        n = length(proteins)
        return DataFrame(
            Protein = proteins, BF = bf,
            bf_enrichment = bf, bf_correlation = ones(n), bf_detected = ones(n),
            mean_log2FC = log2fc, sd_log2FC = fill(0.5, n),
            posterior_prob = bf ./ (1 .+ bf), Component = fill(1, n),
        )
    end
    function _mk_ar(label::String, df::DataFrame)
        AnalysisResult(df, DataFrame(),
            nothing, nothing, nothing, nothing, nothing, :bma, nothing, nothing,
            UInt64(0), UInt64(0), now(), "test",
            label, 1, nothing, nothing, nothing, :loaded, nothing, nothing,
            false, nothing)
    end

    Random.seed!(75)
    # Same disjoint design — P1 is in pair(a,b) and pair(a,c) but NOT in pair(b,c).
    ar1 = _mk_ar("BAIT", _mk_copula_df(["P1","P2"], [200.0, 1.0], [3.0, 0.2]))
    ar2 = _mk_ar("BAIT", _mk_copula_df(["P3","P4"], [1.0, 200.0], [0.3, 3.0]))
    ar3 = _mk_ar("BAIT", _mk_copula_df(["P5","P6"], [1.0, 200.0], [0.2, 3.1]))

    diff = differential_analysis(
        conditions = (a = ar1, b = ar2, c = ar3),
        contrasts  = :all_pairs,
    )

    idx_P1 = findfirst(==("P1"), diff.results.Protein)
    @test idx_P1 !== nothing
    # P1 IS present in pair (a,b) (via union with ar1 ∋ P1): bf_A_a_vs_b is a real number.
    @test !ismissing(diff.results.bf_A_a_vs_b[idx_P1])
    # P1 is NOT present in pair (b,c) (neither ar2 nor ar3 has P1): bf_A_b_vs_c
    # must be Missing — this is the outer-join contract: ABSENT rows land
    # as Missing, not NaN-filled.
    @test ismissing(diff.results.bf_A_b_vs_c[idx_P1])
    # Type contract: the column must accept Missing (outer-join widens to Union{Missing, T}).
    @test eltype(diff.results.bf_A_b_vs_c) <: Union{Missing, Real}

    # The contract only governs the outer-join-introduced absent rows. NaN values
    # that flow IN from the per-pair DF (e.g. legacy 2-group `bf_A = NaN` for
    # proteins detected only in condition B) are upstream behaviour and
    # outside the scope of this fix. We verify the contract by checking the SET of
    # rows where outerjoin introduced Missing equals the set of proteins
    # absent from that pair's union of ARs.
    pair_bc_proteins = Set(diff.pairwise_results[:b => :c].Protein)  # proteins seen by pair (b,c)
    for i in 1:nrow(diff.results)
        protein = diff.results.Protein[i]
        cell = diff.results.bf_A_b_vs_c[i]
        if protein in pair_bc_proteins
            # Protein WAS in pair (b,c)'s source DF — outer-join did NOT fill it.
            # The cell may be a real number OR an upstream NaN (legacy 2-group
            # behaviour). What matters: it is NOT Missing.
            @test !ismissing(cell)
        else
            # Protein was NOT in pair (b,c) — outer-join filled with Missing.
            # The cell MUST be Missing (NOT NaN — that would mean outer-join
            # filled with a numeric sentinel, violating the contract).
            @test ismissing(cell)
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Stub 5 — byte-equality preserved (k=2 path unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@testitem "k=2 byte-equality preserved (legacy 2-group vs k-group with 2 conds)" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    fx = DifferentialFixtures.create_three_condition_result()

    # Legacy 2-group call. Use the SAME pair as the k=2 NamedTuple form below.
    diff_2g  = differential_analysis(fx.ar_wt, fx.ar_mut1;
                                     condition_A = "wt", condition_B = "mut1")
    diff_kg2 = differential_analysis(conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1))

    # Row count must match exactly — byte-equality.
    @test nrow(diff_2g.results) == nrow(diff_kg2.results)

    # The k-group call adds three columns not present in the legacy 2-group output
    # (schema-uniformity): :differential_BFDR_pairwise_BH,
    # :decision_risk_min + :optimal_call_min. All other common columns
    # MUST match value-for-value.
    extra_in_kg = setdiff(names(diff_kg2.results), names(diff_2g.results))
    @test Set(extra_in_kg) == Set(["differential_BFDR_pairwise_BH",
                                   "decision_risk_min", "optimal_call_min"])
    # The legacy 2-group output has no columns the k-group output lacks.
    @test isempty(setdiff(names(diff_2g.results), names(diff_kg2.results)))

    # byte-equality: every column present in both DFs must be
    # value-equal (handles Missing semantics via `isequal`).
    common_cols = intersect(names(diff_2g.results), names(diff_kg2.results))
    for c in common_cols
        @test isequal(diff_2g.results[!, c], diff_kg2.results[!, c])
    end

    # k=2 path short-circuits BEFORE the outer-join branch,
    # so `:n_pairs_with_data` is NOT added (preserves byte-equality).
    @test !(:n_pairs_with_data in propertynames(diff_kg2.results))
end
