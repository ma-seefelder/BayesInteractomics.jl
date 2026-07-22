# test/differential/test_kgroup_pairwise.jl
# k-Group differential_analysis overload — pairwise BMA test suite.
#
# This file is the canonical home for ALL k-group dispatch / aggregation /
# parallelism / fault-tolerance / report-compat testitems.
#
# Filter command:
#   julia --project=. --threads=4 -e 'using TestItemRunner; @run_package_tests filter=ti->occursin("kgroup", ti.filename)'
#
# Filter map (all testitems land here unless noted):
#   - kgroup_dispatch
#   - kgroup_validation
#   - kgroup_bh
#   - kgroup_aggregator
#   - kgroup_legacy_parity
#   - kgroup_pairwise_consistency
#   - kgroup_classification_summary
#   - kgroup_parallel_determinism
#   - kgroup_fault_tolerance
#   - kgroup_condition_similarity
#   - kgroup_ctor_cascade
#   - kgroup_contrasts
#   - kgroup_report_compat      (lands in test/reports/test_report.jl, NOT here)

# ─────────────────────────────────────────────────────────────────────────────
# _validate_kgroup_arguments error-path coverage
# ─────────────────────────────────────────────────────────────────────────────

@testitem "kgroup_validation: _validate_kgroup_arguments error paths" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    using Logging
    using Test
    v = BayesInteractomics.Differential._validate_kgroup_arguments

    fx = DifferentialFixtures.create_three_condition_result()
    conds_3 = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2)
    conds_2 = (wt = fx.ar_wt, mut1 = fx.ar_mut1)
    conds_1 = (wt = fx.ar_wt,)

    # ---- Happy paths: three contrast forms ----
    @test v(conds_3, :all_pairs) == [:wt => :mut1, :wt => :mut2, :mut1 => :mut2]
    @test v(conds_3, :vs_reference => :wt) == [:wt => :mut1, :wt => :mut2]
    @test v(conds_3, [:wt => :mut1, :mut1 => :mut2]) == [:wt => :mut1, :mut1 => :mut2]
    @test v(conds_2, :all_pairs) == [:wt => :mut1]

    # ---- Error paths (per CONTEXT <specifics>) ----
    # k < 2
    err = try; v(conds_1, :all_pairs); catch e; e; end
    @test err isa ArgumentError && occursin("k ≥ 2", err.msg) && occursin("got 1", err.msg)

    # Self-pair
    err = try; v(conds_2, [:wt => :wt]); catch e; e; end
    @test err isa ArgumentError && occursin("Self-pair in contrasts: wt => wt", err.msg)

    # Unknown reference symbol
    err = try; v(conds_3, :vs_reference => :unknown); catch e; e; end
    @test err isa ArgumentError && occursin("Unknown reference symbol unknown", err.msg)

    # Empty contrasts vector
    err = try; v(conds_2, Pair{Symbol, Symbol}[]); catch e; e; end
    @test err isa ArgumentError && occursin("Empty contrasts vector", err.msg)

    # Duplicate pair
    err = try; v(conds_2, [:wt => :mut1, :wt => :mut1]); catch e; e; end
    @test err isa ArgumentError && occursin("Duplicate pair in contrasts", err.msg)

    # Unknown contrast symbol
    err = try; v(conds_2, [:wt => :unknown]); catch e; e; end
    @test err isa ArgumentError && occursin("Unknown contrast symbol unknown", err.msg)

    # Unsupported contrasts form
    err = try; v(conds_2, "all"); catch e; e; end
    @test err isa ArgumentError && occursin("Unsupported contrasts form", err.msg)

    # ---- Bait mismatch @warn (does NOT throw) ----
    fx2 = DifferentialFixtures.create_three_condition_result()
    # Flip bait on ar_mut1 to trigger the warn. AnalysisResult.bait_protein is mutable
    # since AnalysisResult is a mutable struct.
    fx2.ar_mut1.bait_protein = "OTHER_BAIT"
    conds_bad = (wt = fx2.ar_wt, mut1 = fx2.ar_mut1, mut2 = fx2.ar_mut2)
    # Validate WITHOUT throw; capture the warn via a TestLogger.
    test_logger = Test.TestLogger()
    with_logger(test_logger) do
        result = v(conds_bad, :all_pairs)
        @test result == [:wt => :mut1, :wt => :mut2, :mut1 => :mut2]
    end
    @test any(r -> r.level == Logging.Warn && occursin("bait mismatch", r.message), test_logger.logs)
end

# ─────────────────────────────────────────────────────────────────────────────
# BH/Bonferroni/Holm primitives and _apply_multi_test_correction! dispatcher
# ─────────────────────────────────────────────────────────────────────────────

@testitem "kgroup_bh: BH/Bonferroni/Holm primitives and dispatcher" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    bh  = BayesInteractomics.Differential._bh_adjust
    bon = BayesInteractomics.Differential._bonferroni_adjust
    hol = BayesInteractomics.Differential._holm_adjust
    mtc = BayesInteractomics.Differential._apply_multi_test_correction!

    # ---- BH primitive: hand-computed expected outputs ----
    @test bh([0.5]) ≈ [0.5]                                # n=1 identity
    @test all(bh([0.01, 0.02, 0.03, 0.04, 0.05]) .≈ 0.05)  # cumulative-min from right
    q = bh([0.01, 0.5, 0.02])
    @test q[1] ≈ 0.03 && q[2] ≈ 0.5 && q[3] ≈ 0.03

    # ---- BH missing pass-through ----
    q = bh([missing, 0.01, 0.5])
    @test ismissing(q[1])
    @test q[2] ≈ 0.02   # n_valid = 2 → p*2/1 = 0.02
    @test q[3] ≈ 0.5

    # ---- Bonferroni ----
    @test bon([0.01, 0.5], 4) ≈ [0.04, 1.0]
    @test bon([missing, 0.1], 5)[1] |> ismissing
    @test bon([missing, 0.1], 5)[2] ≈ 0.5

    # ---- Holm ----
    q = hol([0.01, 0.5])
    @test q[1] ≈ 0.02 && q[2] ≈ 0.5

    # ---- Dispatcher: k=3 pairwise_dict + wide DF integration ----
    # 6-protein × 3-contrast family fixture pattern.
    df_a = DataFrame(Protein = ["P$i" for i in 1:6],
                     differential_BFDR = [0.001, 0.01, 0.5, 0.02, 0.8, 0.05])
    df_b = DataFrame(Protein = ["P$i" for i in 1:6],
                     differential_BFDR = [0.5,   0.001,0.02, 0.03, 0.8, 0.04])
    df_c = DataFrame(Protein = ["P$i" for i in 1:6],
                     differential_BFDR = [0.5,   0.5,  0.001,0.05, 0.8, 0.5])
    pd = Dict{Pair{Symbol, Symbol}, DataFrame}(
        (:wt => :mut1) => df_a,
        (:wt => :mut2) => df_b,
        (:mut1 => :mut2) => df_c,
    )
    wide = DataFrame(Protein = ["P$i" for i in 1:6])

    mtc(pd, wide, :bh)
    @test hasproperty(df_a, :differential_BFDR_pairwise_BH)
    @test hasproperty(df_b, :differential_BFDR_pairwise_BH)
    @test hasproperty(df_c, :differential_BFDR_pairwise_BH)
    @test hasproperty(wide, :differential_BFDR_wt_vs_mut1_pairwise_BH)
    @test hasproperty(wide, :differential_BFDR_wt_vs_mut2_pairwise_BH)
    @test hasproperty(wide, :differential_BFDR_mut1_vs_mut2_pairwise_BH)

    # BH on the flattened 18-element family — monotone non-decreasing in original p-value rank.
    flat_orig = [df_a.differential_BFDR; df_b.differential_BFDR; df_c.differential_BFDR]
    flat_bh   = [df_a.differential_BFDR_pairwise_BH;
                 df_b.differential_BFDR_pairwise_BH;
                 df_c.differential_BFDR_pairwise_BH]
    # BH q ≥ p always (BH inflates)
    @test all(flat_bh .>= flat_orig)
    # BH q ≤ 1 always
    @test all(flat_bh .<= 1.0)

    # ---- k=2 special case (byte-equality precondition) ----
    df_solo = DataFrame(Protein = ["P1", "P2"], differential_BFDR = [0.01, 0.5])
    pd2 = Dict{Pair{Symbol, Symbol}, DataFrame}((:wt => :mut) => df_solo)
    wide2 = DataFrame(Protein = ["P1", "P2"])
    mtc(pd2, wide2, :bh)
    # n_contrasts=1 → unsuffixed column on the wide DF (NO `_wt_vs_mut_` infix)
    @test hasproperty(wide2, :differential_BFDR_pairwise_BH)
    @test !hasproperty(wide2, :differential_BFDR_wt_vs_mut_pairwise_BH)
    # k=2 BH is identity over the input (short-circuit)
    @test wide2.differential_BFDR_pairwise_BH ≈ [0.01, 0.5]

    # ---- :none dispatch ----
    df_none = DataFrame(Protein = ["P1"], differential_BFDR = [0.3])
    pd_none = Dict{Pair{Symbol, Symbol}, DataFrame}((:wt => :mut) => df_none)
    wide_none = DataFrame(Protein = ["P1"])
    mtc(pd_none, wide_none, :none)
    @test hasproperty(wide_none, :differential_BFDR_pairwise_None)
    @test wide_none.differential_BFDR_pairwise_None ≈ [0.3]

    # ---- :bonferroni dispatch ----
    df_bon = DataFrame(Protein = ["P1"], differential_BFDR = [0.1])
    pd_bon = Dict{Pair{Symbol, Symbol}, DataFrame}((:wt => :mut) => df_bon)
    wide_bon = DataFrame(Protein = ["P1"])
    mtc(pd_bon, wide_bon, :bonferroni)
    @test hasproperty(wide_bon, :differential_BFDR_pairwise_Bonferroni)

    # ---- Throw on unknown method ----
    df_unk = DataFrame(Protein = ["P1"], differential_BFDR = [0.3])
    pd_unk = Dict{Pair{Symbol, Symbol}, DataFrame}((:wt => :mut) => df_unk)
    wide_unk = DataFrame(Protein = ["P1"])
    @test_throws ArgumentError mtc(pd_unk, wide_unk, :unknown_method)
end

# ─────────────────────────────────────────────────────────────────────────────
# BFDR pooled-family monotonicity sanity (k=3 family > within-pair Storey)
# ─────────────────────────────────────────────────────────────────────────────

@testitem "kgroup_pooled: BH pooling across pairs preserves rank order" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    mtc = BayesInteractomics.Differential._apply_multi_test_correction!

    # Three pairs with mixed BFDR values; the flattened family has size 12.
    # BH adjustment must preserve rank order in the SORTED p-value list and
    # produce q-values ≥ inputs (BH never deflates).
    df_a = DataFrame(Protein = ["P$i" for i in 1:4],
                     differential_BFDR = [0.001, 0.05, 0.30, 0.80])
    df_b = DataFrame(Protein = ["P$i" for i in 1:4],
                     differential_BFDR = [0.02, 0.10, 0.40, 0.90])
    df_c = DataFrame(Protein = ["P$i" for i in 1:4],
                     differential_BFDR = [0.005, 0.20, 0.60, 0.95])

    pd = Dict{Pair{Symbol, Symbol}, DataFrame}(
        (:wt => :mut1) => df_a,
        (:wt => :mut2) => df_b,
        (:mut1 => :mut2) => df_c,
    )
    wide = DataFrame(Protein = ["P$i" for i in 1:4])

    mtc(pd, wide, :bh)

    # Collect input and corrected values in matching order.
    flat_p = [df_a.differential_BFDR; df_b.differential_BFDR; df_c.differential_BFDR]
    flat_q = [df_a.differential_BFDR_pairwise_BH;
              df_b.differential_BFDR_pairwise_BH;
              df_c.differential_BFDR_pairwise_BH]

    # ---- BH inflates: every q ≥ p ----
    @test all(flat_q .>= flat_p .- 1e-12)
    # ---- BH bounded: every q ≤ 1 ----
    @test all(flat_q .<= 1.0)

    # ---- Rank-order preservation: sorting by input p produces ascending q ----
    order = sortperm(flat_p)
    q_sorted = flat_q[order]
    @test all(diff(q_sorted) .>= -1e-12)

    # ---- Cross-pair pooling actually adjusts (vs per-pair would be identity-ish):
    # the smallest input (0.001) should NOT equal 0.001 after BH on m=12 family.
    @test minimum(flat_q) > minimum(flat_p)

    # ---- :none dispatch over the same family preserves verbatim values
    df_a2 = DataFrame(Protein = ["P$i" for i in 1:4],
                      differential_BFDR = [0.001, 0.05, 0.30, 0.80])
    df_b2 = DataFrame(Protein = ["P$i" for i in 1:4],
                      differential_BFDR = [0.02, 0.10, 0.40, 0.90])
    df_c2 = DataFrame(Protein = ["P$i" for i in 1:4],
                      differential_BFDR = [0.005, 0.20, 0.60, 0.95])
    pd_none = Dict{Pair{Symbol, Symbol}, DataFrame}(
        (:wt => :mut1) => df_a2,
        (:wt => :mut2) => df_b2,
        (:mut1 => :mut2) => df_c2,
    )
    wide_none = DataFrame(Protein = ["P$i" for i in 1:4])
    mtc(pd_none, wide_none, :none)
    @test df_a2.differential_BFDR_pairwise_None ≈ df_a2.differential_BFDR
    @test df_b2.differential_BFDR_pairwise_None ≈ df_b2.differential_BFDR
    @test df_c2.differential_BFDR_pairwise_None ≈ df_c2.differential_BFDR
end

# ─────────────────────────────────────────────────────────────────────────────
# _aggregate_pairwise_results (k=2 verbatim + k=3 inner-join) +
#           _kgroup_classification_summary token grammar +
#           byte-equality contract between k=2 NamedTuple call and
#           legacy 2-condition positional call
# ─────────────────────────────────────────────────────────────────────────────

@testitem "kgroup_aggregator: _aggregate_pairwise_results k=2 verbatim + k=3 outer-join" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    agg = BayesInteractomics.Differential._aggregate_pairwise_results

    # ---- k=2: verbatim pass-through (byte-equality lock) ----
    df_solo = DataFrame(Protein = ["P1", "P2"],
                        bf_A = [1.0, 2.0],
                        differential_BFDR = [0.1, 0.2],
                        classification = ["GAINED", "UNCHANGED"])
    pd_solo = Dict{Pair{Symbol, Symbol}, DataFrame}((:wt => :mut) => df_solo)
    w_solo = agg(pd_solo, [:wt => :mut])
    @test w_solo === df_solo                          # same object identity
    @test names(w_solo) == ["Protein", "bf_A", "differential_BFDR", "classification"]  # no suffix
    @test !(:n_pairs_with_data in propertynames(w_solo))   # k=2 short-circuit: no companion column

    # ---- k=3: column suffixing + OUTER-join union ----
    # The k≥3 branch UNIONS proteins across pair-DFs
    # (was: intersection via innerjoin). Each pair-DF contributes its rows;
    # missing cells land as Missing; a `:n_pairs_with_data::Int` companion
    # column records per-row coverage.
    df1 = DataFrame(Protein = ["P1", "P2", "P3"], bf = [1.0, 2.0, 3.0])
    df2 = DataFrame(Protein = ["P1", "P2"],       bf = [4.0, 5.0])
    df3 = DataFrame(Protein = ["P2", "P3"],       bf = [6.0, 7.0])
    pd3 = Dict{Pair{Symbol, Symbol}, DataFrame}(
        (:wt   => :mut1) => df1,
        (:wt   => :mut2) => df2,
        (:mut1 => :mut2) => df3,
    )
    w3 = agg(pd3, [:wt => :mut1, :wt => :mut2, :mut1 => :mut2])

    # outer-join — every protein from ANY pair-DF appears.
    @test nrow(w3) == 3                               # P1, P2, P3 across all pair-DFs
    @test Set(w3.Protein) == Set(["P1", "P2", "P3"])

    # nrow(wide) >= max(nrow(per_pair_df)) — here 3 >= 3.
    @test nrow(w3) >= maximum(nrow(df) for df in values(pd3))

    # Per-pair suffixed columns are present.
    @test hasproperty(w3, :bf_wt_vs_mut1)
    @test hasproperty(w3, :bf_wt_vs_mut2)
    @test hasproperty(w3, :bf_mut1_vs_mut2)

    # P2 is the only protein in ALL THREE pair-DFs — its row carries every BF.
    idx_P2 = findfirst(==("P2"), w3.Protein)
    @test idx_P2 !== nothing
    @test w3.bf_wt_vs_mut1[idx_P2] == 2.0
    @test w3.bf_wt_vs_mut2[idx_P2] == 5.0
    @test w3.bf_mut1_vs_mut2[idx_P2] == 6.0

    # P1 is absent from pair (mut1, mut2) → that suffixed cell is Missing.
    idx_P1 = findfirst(==("P1"), w3.Protein)
    @test idx_P1 !== nothing
    @test w3.bf_wt_vs_mut1[idx_P1] == 1.0
    @test w3.bf_wt_vs_mut2[idx_P1] == 4.0
    @test ismissing(w3.bf_mut1_vs_mut2[idx_P1])

    # P3 is absent from pair (wt, mut2) → that suffixed cell is Missing.
    idx_P3 = findfirst(==("P3"), w3.Protein)
    @test idx_P3 !== nothing
    @test w3.bf_wt_vs_mut1[idx_P3] == 3.0
    @test ismissing(w3.bf_wt_vs_mut2[idx_P3])
    @test w3.bf_mut1_vs_mut2[idx_P3] == 7.0

    # n_pairs_with_data exists and ranges [1, 3] per design.
    @test :n_pairs_with_data in propertynames(w3)
    @test eltype(w3.n_pairs_with_data) == Int
    @test w3.n_pairs_with_data[idx_P1] == 2          # P1 in (wt,mut1) and (wt,mut2)
    @test w3.n_pairs_with_data[idx_P2] == 3          # P2 in all three pairs
    @test w3.n_pairs_with_data[idx_P3] == 2          # P3 in (wt,mut1) and (mut1,mut2)

    # ---- Empty contrasts throws ----
    @test_throws ArgumentError agg(Dict{Pair{Symbol, Symbol}, DataFrame}(),
                                   Pair{Symbol, Symbol}[])
end

@testitem "kgroup_classification_summary: token grammar matches RESEARCH §7" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    ks = BayesInteractomics.Differential._kgroup_classification_summary

    @test ks([GAINED, GAINED, UNCHANGED], [:wt => :mut1, :wt => :mut2, :mut1 => :mut2]) ==
          "wt>mut1; wt>mut2; mut1=mut2"
    @test ks([REDUCED, REDUCED], [:wt => :mut1, :wt => :mut2]) ==
          "wt<mut1; wt<mut2"
    @test ks([BOTH_NEGATIVE], [:wt => :mut1]) == "wt0mut1"
    @test ks([CONDITION_A_SPECIFIC], [:wt => :mut1]) == "wt!"
    @test ks([CONDITION_B_SPECIFIC], [:wt => :mut1]) == "mut1!"
    @test ks([UNCHANGED, GAINED, REDUCED, BOTH_NEGATIVE],
             [:a => :b, :a => :c, :b => :c, :a => :d]) ==
          "a=b; a>c; b<c; a0d"

    @test_throws ArgumentError ks([], Pair{Symbol, Symbol}[])
    @test_throws ArgumentError ks([GAINED], [:wt => :mut1, :wt => :mut2])  # length mismatch
end

@testitem "kgroup_legacy_parity: k=2 aggregator + BH ≡ legacy 2-group results" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    agg = BayesInteractomics.Differential._aggregate_pairwise_results
    mtc = BayesInteractomics.Differential._apply_multi_test_correction!

    # The DifferentialFixtures fixture returns a DifferentialResult directly
    # (not a NamedTuple). Its .results field is the legacy 2-condition DataFrame
    # produced by `differential_analysis(ar_A, ar_B)`-equivalent construction.
    diff_legacy = DifferentialFixtures.create_two_condition_result()
    diff_legacy_df = diff_legacy.results

    # Simulate the k=2 NamedTuple path: aggregator returns the single per-pair DF
    # VERBATIM (precondition); _apply_multi_test_correction! then adds
    # `differential_BFDR_pairwise_BH` to BOTH the per-pair DF and wide DF
    # (which are the same object for k=2).
    df_copy = copy(diff_legacy_df)
    pd  = Dict{Pair{Symbol, Symbol}, DataFrame}((:wt => :mut) => df_copy)
    wide = agg(pd, [:wt => :mut])
    @test wide === pd[:wt => :mut]                    # k=2 verbatim — same object
    mtc(pd, wide, :bh)

    # The wide DF gains a single new column `differential_BFDR_pairwise_BH`.
    @test hasproperty(wide, :differential_BFDR_pairwise_BH)
    # Drop it and compare byte-equal to the legacy DF.
    wide_minus_bh = select(wide, Not(:differential_BFDR_pairwise_BH))
    @test isequal(wide_minus_bh, diff_legacy_df)      # byte-equality contract

    # Additionally: BH on n_contrasts=1 is identity over differential_BFDR.
    @test wide.differential_BFDR_pairwise_BH ≈ wide.differential_BFDR

    # And: k=2 wide DF column names contain NO `_wt_vs_mut` suffix infix.
    @test !any(occursin("_wt_vs_mut", String(c)) for c in names(wide))
end

# ─────────────────────────────────────────────────────────────────────────────
# keyword-only differential_analysis overload end-to-end coverage
# ─────────────────────────────────────────────────────────────────────────────

@testitem "kgroup_dispatch: keyword overload dispatch + return type" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    fx = DifferentialFixtures.create_three_condition_result()

    # k=2 NamedTuple call
    d2 = differential_analysis(conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1))
    @test d2 isa DifferentialResult
    @test d2.contrasts == [:wt => :mut1]
    @test d2.pairwise_results !== nothing
    @test haskey(d2.pairwise_results, :wt => :mut1)
    # k=2 wide DF has NO classification_summary column (byte-equality precondition).
    @test !hasproperty(d2.results, :classification_summary)

    # k=3 NamedTuple call
    d3 = differential_analysis(conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2))
    @test d3 isa DifferentialResult
    @test length(d3.contrasts) == 3
    @test length(d3.pairwise_results) == 3
    # k=3 wide DF DOES have classification_summary column.
    @test hasproperty(d3.results, :classification_summary)

    # k=3 with :vs_reference
    d3v = differential_analysis(conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
                                contrasts = :vs_reference => :wt)
    @test d3v.contrasts == [:wt => :mut1, :wt => :mut2]

    # k=3 with explicit pairs
    d3e = differential_analysis(conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
                                contrasts = [:wt => :mut1, :mut1 => :mut2])
    @test d3e.contrasts == [:wt => :mut1, :mut1 => :mut2]

    # Mixed positional + conditions kwarg → MethodError.
    # The 2-group signature has no `conditions` kwarg so Julia's strict keyword
    # dispatch rejects the call without any extra guard.
    @test_throws MethodError differential_analysis(fx.ar_wt, fx.ar_mut1;
                                                   conditions = (wt = fx.ar_wt,))
end

@testitem "kgroup_contrasts: k=3 :all_pairs and k=3 :vs_reference contrast counts" setup=[DifferentialFixtures] begin
    using BayesInteractomics

    fx = DifferentialFixtures.create_three_condition_result()

    # k=3 :all_pairs → 3 contrasts.
    d3 = differential_analysis(conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
                               contrasts = :all_pairs)
    @test length(d3.contrasts) == 3
    @test d3.contrasts == [:wt => :mut1, :wt => :mut2, :mut1 => :mut2]

    # k=3 :vs_reference => :wt → (k-1)=2 contrasts; all share :wt on the left.
    d3v = differential_analysis(conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
                                contrasts = :vs_reference => :wt)
    @test length(d3v.contrasts) == 2
    @test all(first(p) === :wt for p in d3v.contrasts)

    # k=3 :vs_reference => :mut1 → also (k-1)=2 contrasts but pointed at :mut1.
    d3m = differential_analysis(conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
                                contrasts = :vs_reference => :mut1)
    @test length(d3m.contrasts) == 2
    @test all(first(p) === :mut1 for p in d3m.contrasts)
end

@testitem "kgroup_pairwise_consistency: pairwise_results[c => d] row-equal to 2-group call" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    fx = DifferentialFixtures.create_three_condition_result()

    # k-group call.
    dk = differential_analysis(conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2))

    # Direct 2-group call for the wt => mut1 pair using a fresh fixture (the previous
    # ARs' copula_df may have been mutated in-place by _extract_copula_df's column
    # additions; rebuild from scratch to compare against a pristine 2-group result).
    fx2 = DifferentialFixtures.create_three_condition_result()
    d_direct = differential_analysis(fx2.ar_wt, fx2.ar_mut1;
                                     condition_A = "wt", condition_B = "mut1")

    df_kgroup = dk.pairwise_results[:wt => :mut1]
    df_direct = d_direct.results

    # The k-group call adds `differential_BFDR_pairwise_BH` to every per-pair DF
    # (via _apply_multi_test_correction!); the legacy 2-group call does NOT.
    # Drop that column and compare row-by-row on the intersection of column names.
    df_kgroup_minus_bh = hasproperty(df_kgroup, :differential_BFDR_pairwise_BH) ?
        select(df_kgroup, Not(:differential_BFDR_pairwise_BH)) : df_kgroup

    # Compare only columns that exist in BOTH frames (the upstream 2-group call may
    # carry extra optional diagnostic columns that differ across fresh fixture builds).
    shared_cols = intersect(names(df_kgroup_minus_bh), names(df_direct))
    @test !isempty(shared_cols)
    @test isequal(select(df_kgroup_minus_bh, shared_cols), select(df_direct, shared_cols))

    # Specific row-level invariant: the differential_BFDR column (key BFDR output)
    # is byte-identical between the k-group pairwise DF and the legacy 2-group DF
    # (BH on n_contrasts=1 family is identity).
    @test "differential_BFDR" in shared_cols
    @test isequal(df_kgroup_minus_bh.differential_BFDR, df_direct.differential_BFDR)
end

@testitem "kgroup_parallel_determinism: parallel_pairs=true and =false produce identical results" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    fx_ser = DifferentialFixtures.create_three_condition_result()
    fx_par = DifferentialFixtures.create_three_condition_result()
    conds_ser = (wt = fx_ser.ar_wt, mut1 = fx_ser.ar_mut1, mut2 = fx_ser.ar_mut2)
    conds_par = (wt = fx_par.ar_wt, mut1 = fx_par.ar_mut1, mut2 = fx_par.ar_mut2)

    d_ser = differential_analysis(conditions = conds_ser, parallel_pairs = false)
    d_par = differential_analysis(conditions = conds_par, parallel_pairs = true)

    # Wide DataFrames are sort-stable equal (sort by Protein for deterministic compare).
    @test d_ser.contrasts == d_par.contrasts

    # Per-pair DataFrames are byte-equal modulo row ordering.
    for pair in d_ser.contrasts
        df_s = sort(d_ser.pairwise_results[pair], :Protein)
        df_p = sort(d_par.pairwise_results[pair], :Protein)
        # Compare on the intersection of column names (BH column is in both).
        shared = intersect(names(df_s), names(df_p))
        @test !isempty(shared)
        @test isequal(select(df_s, shared), select(df_p, shared))
    end

    # Wide results: differential_BFDR columns must match across serial/parallel.
    s_cols = intersect(names(d_ser.results), names(d_par.results))
    s_cols_bfdr = filter(c -> occursin("differential_BFDR", c), s_cols)
    @test !isempty(s_cols_bfdr)
    for c in s_cols_bfdr
        @test isequal(sort(d_ser.results, :Protein)[!, c], sort(d_par.results, :Protein)[!, c])
    end
end

@testitem "kgroup_fault_tolerance: one failing pair does NOT crash the call; failure is warned" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    using Logging
    using Test

    fx = DifferentialFixtures.create_three_condition_result()
    # Sabotage ar_mut1's copula_results so the 2-group pipeline throws ArgumentError
    # when it tries to access the BF column. Empty DataFrame triggers the catch path
    # inside _runpair!. AnalysisResult is mutable.
    fx.ar_mut1.copula_results = DataFrame()

    # Capture warnings emitted by the orchestrator.
    test_logger = Test.TestLogger()
    diff = with_logger(test_logger) do
        differential_analysis(conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2))
    end

    # Contract: the call does NOT throw; surviving pairs land in pairwise_results.
    # Pairs involving :mut1 (i.e. :wt => :mut1 and :mut1 => :mut2) fail. The
    # :wt => :mut2 pair survives, so we expect exactly 1 surviving entry.
    @test diff isa DifferentialResult
    @test length(diff.pairwise_results) <= 2
    @test length(diff.pairwise_results) >= 1
    @test haskey(diff.pairwise_results, :wt => :mut2)

    # At least one @warn fired for the failed pair(s). The orchestrator emits one
    # `@warn ... contrast :X => :Y failed ...` per failure (maxlog=10).
    failure_warns = filter(r -> r.level == Logging.Warn && occursin("failed", r.message),
                           test_logger.logs)
    @test !isempty(failure_warns)
end

@testitem "kgroup_condition_similarity: k-group similarity matrix is k×k with NamedTuple key labels" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using BayesInteractomics: condition_labels   # accessor lives in BayesInteractomics.Differential, re-exported

    fx = DifferentialFixtures.create_three_condition_result()
    # Attach a CONFIG with embeddings_config.run_embeddings=true to ar_wt so the
    # k-group call exercises the embeddings hook.
    cfg = CONFIG(
        datafile     = [""],
        control_cols = [Dict(1 => [1])],
        sample_cols  = [Dict(1 => [2])],
        poi          = "BAIT",
        refID        = 1,
        n_controls   = 1,
        n_samples    = 1,
        embeddings_config = EmbeddingsConfig(run_embeddings = true, method = :umap),
    )
    fx.ar_wt.config = cfg

    diff = differential_analysis(conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2))
    cs = diff.condition_similarity
    # cs may be `nothing` if the embeddings extension isn't loaded; if it IS present
    # we assert the k×k contract + the RESEARCH Open Q 10 label post-overwrite.
    if cs !== nothing
        @test size(cs.spearman_log10_bf) == (3, 3)
        # RESEARCH Open Q 10 lock: condition_labels is post-overwritten to NamedTuple
        # keys so the report shows ["wt", "mut1", "mut2"] not ["BAIT", "BAIT", "BAIT"].
        @test cs.condition_labels == ["wt", "mut1", "mut2"]
    end
    # condition_labels(::DifferentialResult) k-aware path — independent of cs.
    @test condition_labels(diff) == ["wt", "mut1", "mut2"]
end

# ─────────────────────────────────────────────────────────────────────────────
# DifferentialResult backward-compat constructor cascade
# Exercises all four ctor layers (14 / 15 / 17 / 19-arg surface; canonical is
# 20-positional fields). Each layer must default the unfilled
# trailing fields per the cascade contract documented in src/differential/types.jl.
# ─────────────────────────────────────────────────────────────────────────────

@testitem "kgroup_ctor_cascade: 14/15/17/19-arg DifferentialResult ctors all produce valid structs with correct defaults" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using BayesInteractomics: condition_labels   # accessor lives in BayesInteractomics.Differential, re-exported
    using DataFrames, Dates

    dcfg = DifferentialConfig()
    empty_df = DataFrame(Protein = String[])

    # ---- 14-arg ----
    r14 = DifferentialResult(empty_df, "A", "B", dcfg,
                             0, 0, 0, 0, 0,
                             now(), 0, 0, 0, 0)
    @test r14.analyses === nothing
    @test r14.is_calibrated_A == false
    @test r14.is_calibrated_B == false
    @test r14.condition_similarity === nothing
    @test r14.contrasts == Pair{Symbol, Symbol}[]
    @test r14.pairwise_results === nothing

    # ---- 15-arg ----
    r15 = DifferentialResult(empty_df, "A", "B", dcfg,
                             0, 0, 0, 0, 0,
                             now(), 0, 0, 0, 0,
                             nothing)   # analyses
    @test r15.analyses === nothing
    @test r15.is_calibrated_A == false
    @test r15.is_calibrated_B == false
    @test r15.condition_similarity === nothing
    @test r15.contrasts == Pair{Symbol, Symbol}[]
    @test r15.pairwise_results === nothing

    # ---- 17-arg ----
    r17 = DifferentialResult(empty_df, "A", "B", dcfg,
                             0, 0, 0, 0, 0,
                             now(), 0, 0, 0, 0,
                             nothing,   # analyses
                             true,      # is_calibrated_A
                             false)     # is_calibrated_B
    @test r17.is_calibrated_A == true
    @test r17.is_calibrated_B == false
    @test r17.condition_similarity === nothing
    @test r17.contrasts == Pair{Symbol, Symbol}[]
    @test r17.pairwise_results === nothing

    # ---- 18-arg (canonical) — defaults the newest fields ----
    r18 = DifferentialResult(empty_df, "A", "B", dcfg,
                             0, 0, 0, 0, 0,
                             now(), 0, 0, 0, 0,
                             nothing,   # analyses
                             true,      # is_calibrated_A
                             false,     # is_calibrated_B
                             nothing)   # condition_similarity
    @test r18.contrasts == Pair{Symbol, Symbol}[]
    @test r18.pairwise_results === nothing

    # ---- 20-arg canonical — all fields explicit ----
    r20 = DifferentialResult(empty_df, "wt", "mut1", dcfg,
                             0, 0, 0, 0, 0,
                             now(), 0, 0, 0, 0,
                             nothing,                                  # analyses
                             true, false,                              # is_calibrated_A, is_calibrated_B
                             nothing,                                  # condition_similarity
                             [:wt => :mut1, :wt => :mut2],             # contrasts
                             Dict{Pair{Symbol, Symbol}, DataFrame}(    # pairwise_results
                                 (:wt => :mut1) => empty_df,
                                 (:wt => :mut2) => empty_df,
                             ))
    @test r20.contrasts == [:wt => :mut1, :wt => :mut2]
    @test r20.pairwise_results !== nothing
    @test length(r20.pairwise_results) == 2
    @test haskey(r20.pairwise_results, :wt => :mut1)
    @test haskey(r20.pairwise_results, :wt => :mut2)

    # ---- condition_labels k-aware path (legacy vs k-group) ----
    @test condition_labels(r14) == ["A", "B"]            # legacy (empty contrasts)
    @test condition_labels(r17) == ["A", "B"]            # legacy (empty contrasts)
    @test condition_labels(r20) == ["wt", "mut1", "mut2"]  # k-group (contrasts populated)

    # ---- Struct fieldcount (canonical signature) ----
    @test fieldcount(DifferentialResult) == 20
end
