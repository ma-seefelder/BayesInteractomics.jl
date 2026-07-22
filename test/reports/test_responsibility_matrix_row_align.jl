# test/reports/test_responsibility_matrix_row_align.jl
# Responsibility-matrix row-alignment closure regression tests.
#
# Background:
#   The legacy @warn "responsibility matrix row count (X) != results_df rows (Y);
#   falling back to P_H0/P_agnostic/P_H1" at `report_generator.jl:~4091` fired on
#   every condition of the HTT/HAP40 run. The gap pattern (off-by-one for
#   wtHTT/mHTT, +1083 for HAP40_Strep, +210 for GST_HAP40) did NOT match the
#   BetaBernoulli `n_sample_obs < 2 || n_control_obs < 2` exclusion count,
#   ruling out "suppress on expected branch" as a clean fit. The warning was
#   then structurally resolved by reconstructing the responsibility matrix
#   row-wise so `size(responsibilities, 1) == nrow(results_df)` by construction.
#
# These tests pin three contracts:
#   1. `_count_betabernoulli_excluded` returns a non-negative count when given a
#      results_df carrying `n_sample_obs` + `n_control_obs`, and -1 (sentinel
#      "unknown") otherwise.
#   2. `_reconstruct_lc_responsibilities` always returns a matrix with
#      `nrow(results_df)` rows even when LC was fitted on a strict subset (the
#      row-count contract; regression for the original warning).
#   3. The legacy `responsibility matrix row count` @warn text does NOT live in
#      `_add_marginal_fit_json!` source as an @warn call — only as the @debug /
#      @warn entries the instrumentation anchor block introduced. This prevents
#      anyone from re-introducing the warning without going through the
#      BB-exclusion-aware branch.

@testitem "_count_betabernoulli_excluded returns count when results_df carries observation columns" begin
    using BayesInteractomics
    using DataFrames

    # 10 proteins; rows failing `n_sample_obs >= 2 && n_control_obs >= 2`:
    #   P2 (n_sample=1), P4 (n_sample=0), P7 (n_control=1)  →  3 excluded.
    results_df = DataFrame(
        Protein       = ["P$i" for i in 1:10],
        n_sample_obs  = [3, 1, 3, 0, 4, 5, 3, 4, 3, 3],
        n_control_obs = [2, 3, 3, 2, 2, 3, 1, 2, 4, 3],
    )

    n_excluded = BayesInteractomics.Reports._count_betabernoulli_excluded(nothing, results_df)
    @test n_excluded == 3
end

@testitem "_count_betabernoulli_excluded returns -1 (unknown) when neither results_df nor analysis_result carries exclusion data" begin
    using BayesInteractomics
    using DataFrames

    results_df = DataFrame(
        Protein         = ["P1", "P2", "P3"],
        bf_enrichment   = [1.0, 2.0, 3.0],
        bf_correlation  = [1.0, 2.0, 3.0],
        bf_detected     = [1.0, 2.0, 3.0],
    )

    n_excluded = BayesInteractomics.Reports._count_betabernoulli_excluded(nothing, results_df)
    @test n_excluded == -1
end

@testitem "_reconstruct_lc_responsibilities preserves the row-count contract: size(M, 1) == nrow(results_df) for LC-fitted-on-subset case (row-count regression)" begin
    using BayesInteractomics
    using DataFrames

    # 10 proteins on results_df; LC was fitted on only 7 (mimics the
    # responsibility-matrix gap pattern). Prior to the row-wise fix, this is
    # exactly the topology that triggered the W-02 warning.
    results_df = DataFrame(
        Protein         = ["P$i" for i in 1:10],
        bf_enrichment   = collect(1.0:10.0),
        bf_correlation  = collect(1.0:10.0),
        bf_detected     = collect(1.0:10.0),
        P_H0            = fill(0.5, 10),
        P_agnostic      = fill(0.3, 10),
        P_H1            = fill(0.2, 10),
    )

    # LC fitted on P1..P7 only — the 3 missing rows (P8..P10) are precisely what
    # used to make `size(lc.responsibilities, 1) (=7) != nrow(results_df) (=10)`
    # and trigger the obsolete @warn.
    lc_like = (
        protein_names    = ["P$i" for i in 1:7],
        responsibilities = hcat(fill(0.8, 7), fill(0.1, 7), fill(0.1, 7)),
    )

    mat = BayesInteractomics._reconstruct_lc_responsibilities(results_df, lc_like)

    # CONTRACT: row count MUST equal nrow(results_df), not size(lc.responsibilities, 1).
    @test size(mat) == (10, 3)
    # P1..P7: rows copied from LC.
    @test all(mat[i, 1] ≈ 0.8 for i in 1:7)
    # P8..P10: rows fall back to [P_H0, P_agnostic, P_H1] = [0.5, 0.3, 0.2].
    @test all(mat[i, :] ≈ [0.5, 0.3, 0.2] for i in 8:10)
end

@testitem "_add_marginal_fit_json! source no longer carries the prior unconditional @warn text" begin
    using BayesInteractomics  # for pkgdir(BayesInteractomics)

    # Source-text regression: prevent anyone from re-introducing the obsolete
    # `@warn "_add_marginal_fit_json!: responsibility matrix row count (...) != results_df rows (...); falling back to P_H0/P_agnostic/P_H1 columns"`
    # without going through the BB-exclusion-aware instrumentation block.
    src = read(joinpath(pkgdir(BayesInteractomics), "src", "reports", "report_generator.jl"), String)

    # The prior unconditional @warn looked like:
    #   @warn "_add_marginal_fit_json!: responsibility matrix row count ($(...)) != results_df rows ($n); falling back to P_H0/P_agnostic/P_H1 columns"
    # The current code path keeps the same text fragment but only inside the
    # BB-exclusion-aware branch, where the message always carries the
    # `BetaBernoulli-excluded=` diagnostic context. Any bare / unconditional
    # fallback warn would lack that context — assert every such warn has it.
    warn_matches = collect(eachmatch(r"@warn\s+\"_add_marginal_fit_json!:[^\"]*\"", src))
    @test !isempty(warn_matches)
    for m in warn_matches
        @test occursin("BetaBernoulli-excluded", m.match)
    end

    # And the BB-exclusion-aware instrumentation must exist (positive guard so
    # this test fails loudly if someone removes the instrumentation block).
    @test occursin("_count_betabernoulli_excluded", src)
end
