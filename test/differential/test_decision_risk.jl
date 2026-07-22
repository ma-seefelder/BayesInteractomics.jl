# test/differential/test_decision_risk.jl
#
# Bayesian Decision Risk helper-level testitems.
#
# Quick run:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("test_decision_risk", ti.filename)'
#
# Testitem coverage:
#   (1) DEFAULT_DIFFERENTIAL_LOSS shape + values
#   (2) Example 1: MAP == Optimal
#   (3) Example 2: MAP != Optimal
#   (4) CONDITION_A/B_SPECIFIC -> NaN risks
#   (5) Degenerate posterior -> uniform + @warn
#   (6) loss_matrix override -> default flag false
#
# Integration coverage (kwarg override threading, k-group double-write,
# byte-equality) follows in the integration testitems below.

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 1 — DEFAULT_DIFFERENTIAL_LOSS shape, eltype, values
# ─────────────────────────────────────────────────────────────────────────────

@testitem "DEFAULT_DIFFERENTIAL_LOSS shape and values" begin
    using BayesInteractomics
    L = DEFAULT_DIFFERENTIAL_LOSS
    @test size(L) == (4, 4)
    @test eltype(L) == Float64
    @test all(L[i, i] == 0.0 for i in 1:4)
    @test L[1, 2] == 10.0 && L[2, 1] == 10.0   # direction-flip
    @test L[1, 3] == 3.0 && L[2, 3] == 3.0     # over-claim
    @test L[3, 1] == 5.0 && L[3, 2] == 5.0     # missed-hit
    @test L[3, 4] == 1.0 && L[4, 3] == 1.0     # conservative-default
    @test all(L .>= 0.0)
    @test all(isfinite.(L))
    @test DECISION_RISK_ACTIONS == [:gained, :reduced, :unchanged, :both_negative]
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 2 — Example 1: MAP == Optimal (gained)
# ─────────────────────────────────────────────────────────────────────────────

@testitem "Example 1: MAP == Optimal (gained)" begin
    using BayesInteractomics
    using BayesInteractomics: GAINED
    nt = compute_decision_risk([0.05], [0.60], [0.85], [0.95], [GAINED])
    @test nt.risk_gained[1]        ≈ 2.967741935483871   atol=1e-12
    @test nt.risk_reduced[1]       ≈ 6.516129032258064   atol=1e-12
    @test nt.risk_unchanged[1]     ≈ 4.387096774193548   atol=1e-12
    @test nt.risk_both_negative[1] ≈ 4.451612903225806   atol=1e-12
    @test nt.optimal_call[1]       == :gained
    @test nt.decision_risk[1]      ≈ 2.967741935483871   atol=1e-12
    @test nt.loss_matrix_default   == [true]
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 3 — Example 2: MAP != Optimal (badge fires)
# ─────────────────────────────────────────────────────────────────────────────

@testitem "Example 2: MAP != Optimal (badge fires; unchanged wins)" begin
    using BayesInteractomics
    using BayesInteractomics: GAINED
    nt = compute_decision_risk([0.40], [0.55], [0.60], [0.90], [GAINED])
    @test nt.risk_gained[1]        ≈ 3.870967741935484   atol=1e-12
    @test nt.risk_reduced[1]       ≈ 4.838709677419354   atol=1e-12
    @test nt.risk_unchanged[1]     ≈ 3.4516129032258065  atol=1e-12
    @test nt.risk_both_negative[1] ≈ 3.6451612903225805  atol=1e-12
    # The badge-fires smoking gun: MAP is :gained (argmin γ-PEP = idx 1)
    # but optimal is :unchanged (argmin risks = idx 3).
    @test nt.optimal_call[1] == :unchanged
    @test nt.optimal_call[1] != :gained
    @test nt.decision_risk[1] ≈ 3.4516129032258065 atol=1e-12
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 4 — CONDITION_A/B_SPECIFIC rows -> NaN risks,
# eltype stays Float64 (no Union{Missing,Float64} cascade)
# ─────────────────────────────────────────────────────────────────────────────

@testitem "CONDITION_A/B_SPECIFIC rows produce NaN risks, eltype Float64" begin
    using BayesInteractomics
    using BayesInteractomics: GAINED, CONDITION_A_SPECIFIC, CONDITION_B_SPECIFIC
    using DataFrames
    df = DataFrame(
        pep_gained        = [0.05, 0.5, 0.5],
        pep_reduced       = [0.60, 0.5, 0.5],
        pep_unchanged     = [0.85, 0.5, 0.5],
        pep_both_negative = [0.95, 0.5, 0.5],
        classification    = [GAINED, CONDITION_A_SPECIFIC, CONDITION_B_SPECIFIC],
    )
    compute_decision_risk!(df)
    # Row 1 (GAINED) — finite
    @test isfinite(df.decision_risk[1])
    @test df.optimal_call[1] == :gained
    # Row 2 (CONDITION_A_SPECIFIC) — NaN
    @test isnan(df.decision_risk[2])
    @test isnan(df.risk_gained[2])
    @test isnan(df.risk_reduced[2])
    @test isnan(df.risk_unchanged[2])
    @test isnan(df.risk_both_negative[2])
    @test df.optimal_call[2] == :condition_a_specific
    # Row 3 (CONDITION_B_SPECIFIC) — NaN
    @test isnan(df.decision_risk[3])
    @test df.optimal_call[3] == :condition_b_specific
    # eltype: Float64 NOT Union{Missing, Float64} (NaN is the chosen sentinel)
    @test eltype(df.decision_risk) == Float64
    @test eltype(df.risk_gained) == Float64
    @test eltype(df.risk_reduced) == Float64
    @test eltype(df.risk_unchanged) == Float64
    @test eltype(df.risk_both_negative) == Float64
    @test eltype(df.optimal_call) == Symbol
    @test eltype(df.loss_matrix_default) == Bool
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 5 — degenerate posterior (Σ < 1e-12) -> uniform
# fallback + single @warn (maxlog=1)
# ─────────────────────────────────────────────────────────────────────────────

@testitem "degenerate posterior triggers uniform fallback and @warn (maxlog=1)" begin
    using BayesInteractomics
    using BayesInteractomics: GAINED
    using Logging, Test
    # All γ-PEPs = 1.0 → Σ(1 - pep) = 0 → degenerate
    res = @test_logs (:warn, r"degenerate"i) match_mode=:any begin
        compute_decision_risk([1.0], [1.0], [1.0], [1.0], [GAINED])
    end
    # Uniform fallback: each P=0.25.
    # risk_gained    = 0.25 * (0 + 10 + 3 + 3) = 4.0
    # risk_reduced   = 0.25 * (10 + 0 + 3 + 3) = 4.0
    # risk_unchanged = 0.25 * (5 + 5 + 0 + 1)  = 2.75
    # risk_both_neg  = 0.25 * (5 + 5 + 1 + 0)  = 2.75
    @test res.risk_gained[1] ≈ 4.0 atol=1e-12
    @test res.risk_reduced[1] ≈ 4.0 atol=1e-12
    @test res.risk_unchanged[1] ≈ 2.75 atol=1e-12
    @test res.risk_both_negative[1] ≈ 2.75 atol=1e-12
    # tied min → argmin returns first occurrence among the tied minima
    @test res.optimal_call[1] in (:unchanged, :both_negative)
    @test res.decision_risk[1] ≈ 2.75 atol=1e-12
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 6 — kwarg override flips
# loss_matrix_default to false; off-diagonal-1 matrix yields risk_a = 1 - P[a]
# ─────────────────────────────────────────────────────────────────────────────

@testitem "kwarg override flips loss_matrix_default to false; off-diagonal-1 matrix satisfies risk_a = 1 - P[a]" begin
    using BayesInteractomics
    using BayesInteractomics: GAINED
    custom = Float64[0 1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0]
    nt = compute_decision_risk([0.05], [0.60], [0.85], [0.95], [GAINED]; loss_matrix = custom)
    @test nt.loss_matrix_default == [false]
    # Per the off-diagonal-1 matrix: risk_a = Σ_{k≠a} P[k] = 1 - P[a]
    # P = [0.95/1.55, 0.40/1.55, 0.15/1.55, 0.05/1.55]
    P = [0.95/1.55, 0.40/1.55, 0.15/1.55, 0.05/1.55]
    @test nt.risk_gained[1]        ≈ 1.0 - P[1] atol=1e-12
    @test nt.risk_reduced[1]       ≈ 1.0 - P[2] atol=1e-12
    @test nt.risk_unchanged[1]     ≈ 1.0 - P[3] atol=1e-12
    @test nt.risk_both_negative[1] ≈ 1.0 - P[4] atol=1e-12
    # Validation: malformed loss_matrix throws ArgumentError
    bad1 = Float64[0 1 1; 1 0 1; 1 1 0]                            # 3×3
    @test_throws ArgumentError compute_decision_risk([0.05], [0.60], [0.85], [0.95], [GAINED]; loss_matrix = bad1)
    bad2 = Float64[1 1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0]            # nonzero diagonal
    @test_throws ArgumentError compute_decision_risk([0.05], [0.60], [0.85], [0.95], [GAINED]; loss_matrix = bad2)
    bad3 = Float64[0 -1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0]           # negative entry
    @test_throws ArgumentError compute_decision_risk([0.05], [0.60], [0.85], [0.95], [GAINED]; loss_matrix = bad3)
end

# ─────────────────────────────────────────────────────────────────────────────
# Integration testitems (kwarg override threading, k-group double-write, byte-equality).
# ─────────────────────────────────────────────────────────────────────────────

# Testitem 7 — kwarg override threads through k-group differential_analysis
@testitem "kwarg override threads through k-group differential_analysis" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    fx = DifferentialFixtures.create_three_condition_result()
    custom = Float64[0 1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0]
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts = :all_pairs,
        loss_matrix = custom,
    )
    # All per-pair sub-DataFrames carry loss_matrix_default
    for pair in diff.contrasts
        sub = diff.pairwise_results[pair]
        @test :loss_matrix_default in propertynames(sub)
        non_cs_rows = .!isnan.(sub.decision_risk)
        @test all(sub.loss_matrix_default[non_cs_rows] .== false)
    end
    # Default path: loss_matrix_default == true on non-NaN rows
    diff_default = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts = :all_pairs,
    )
    for pair in diff_default.contrasts
        sub = diff_default.pairwise_results[pair]
        non_cs_default = .!isnan.(sub.decision_risk)
        @test all(sub.loss_matrix_default[non_cs_default] .== true)
    end
end

# Testitem 8 — k-group double-write + decision_risk_min aggregation
@testitem "k-group writes Decision Risk to pairwise_results + decision_risk_min aggregation" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    fx = DifferentialFixtures.create_four_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts  = :all_pairs,
    )
    for pair in diff.contrasts
        sub = diff.pairwise_results[pair]
        @test :optimal_call         in propertynames(sub)
        @test :decision_risk        in propertynames(sub)
        @test :risk_gained          in propertynames(sub)
        @test :risk_reduced         in propertynames(sub)
        @test :risk_unchanged       in propertynames(sub)
        @test :risk_both_negative   in propertynames(sub)
        @test :loss_matrix_default  in propertynames(sub)
    end
    @test :decision_risk_min in propertynames(diff.results)
    @test :optimal_call_min  in propertynames(diff.results)
    proteins_wide = diff.results.Protein
    for (i, protein_name) in enumerate(proteins_wide)
        per_pair_vals = Float64[]
        for pair in diff.contrasts
            sub = diff.pairwise_results[pair]
            row_idx = findfirst(==(protein_name), sub.Protein)
            row_idx === nothing && continue
            v = sub.decision_risk[row_idx]
            isnan(v) && continue
            push!(per_pair_vals, v)
        end
        expected_min = isempty(per_pair_vals) ? NaN : minimum(per_pair_vals)
        actual_min   = diff.results.decision_risk_min[i]
        if isnan(expected_min)
            @test isnan(actual_min)
        else
            @test isapprox(actual_min, expected_min; atol=1e-12)
        end
    end
end

# Testitem 9 — byte-equality holds after dropping new columns
@testitem "byte-equality holds after dropping new columns" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    fx = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (a = fx.ar_wt, b = fx.ar_mut1),
        contrasts = :all_pairs,
    )
    omnibus_cols = [:bf_omnibus, :log10_bf_omnibus, :posterior_omnibus,
                    :differential_BFDR_omnibus, :differential_pep_omnibus,
                    :enriched_in, :depleted_in, :kgroup_class]
    decision_risk_cols = [:optimal_call, :decision_risk, :risk_gained, :risk_reduced,
                    :risk_unchanged, :risk_both_negative, :loss_matrix_default,
                    :decision_risk_min, :optimal_call_min]
    new_cols = vcat(omnibus_cols, decision_risk_cols)
    new_cols_present = intersect(new_cols, propertynames(diff.results))
    stripped = select(diff.results, Not(new_cols_present))
    for c in new_cols_present
        @test !(c in propertynames(stripped))
    end
    @test :Protein              in propertynames(stripped)
    @test :classification       in propertynames(stripped)
    @test :differential_BFDR    in propertynames(stripped)
    @test :differential_posterior in propertynames(stripped)
    @test :diff_PEP             in propertynames(stripped)
end
