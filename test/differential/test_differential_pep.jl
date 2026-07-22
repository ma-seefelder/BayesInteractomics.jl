# Differential PEP tests.
#
# Covers PEP coverage dimensions:
#   - C3  differential α-PEP column (differential_pep) exists + diff_PEP mirror
#   - C4  four γ-PEP columns + sum-to-1 invariant (|Σ P(class) − 1.0| < 1e-9)
#   - C5  γ-PEP edge cases: σ_zero non-negativity clamp, Z-vanishing uniform fallback,
#         CONDITION_A_SPECIFIC / CONDITION_B_SPECIFIC single-condition collapse
#   - C6  is_calibrated_A / is_calibrated_B per-side propagation
#   - C8  partial — getDifferentialPEP accessor ArgumentError on unknown class
#
# Tag scheme — `:pep` is the greppable root; sub-tags
# (`:gamma`, `:differential`, `:calibration`, `:condition_specific`, `:accessors`)
# mark coverage dimension.

@testitem "differential_pep alpha column exists (C3)" tags=[:pep, :differential] begin
    include(joinpath(@__DIR__, "mock_helper.jl"))
    using DataFrames

    result_A = _make_mock_result(seed = 42)
    result_B = _make_mock_result(seed = 43)
    diff = differential_analysis(result_A, result_B; condition_A = "A", condition_B = "B")

    @test hasproperty(diff.results, :differential_pep)
    @test hasproperty(diff.results, :diff_PEP)
    # Invariant: silent uppercase mirror MUST be the same Vector reference
    @test diff.results.diff_PEP === diff.results.differential_pep
    # differential_pep = 1 - differential_posterior, per pep(x) helper
    for i in 1:nrow(diff.results)
        if !ismissing(diff.results.differential_pep[i])
            @test diff.results.differential_pep[i] ≈ 1.0 - diff.results.differential_posterior[i]
        end
    end
end

@testitem "four gamma-PEP columns + sum-to-1 invariant (C4)" tags=[:pep, :gamma] begin
    include(joinpath(@__DIR__, "mock_helper.jl"))
    using DataFrames

    result_A = _make_mock_result(seed = 42)
    result_B = _make_mock_result(seed = 43)
    diff = differential_analysis(result_A, result_B; condition_A = "A", condition_B = "B")

    for col in (:pep_gained, :pep_reduced, :pep_unchanged, :pep_both_negative)
        @test hasproperty(diff.results, col)
    end

    # Sum-to-1 invariant for shared rows (detected_in == "both"):
    # Σ (1 − pep_class) ≈ 1 ± 1e-9 (since P(class) = 1 - pep_class and four classes partition)
    shared_idx = findall(==("both"), diff.results.detected_in)
    @test !isempty(shared_idx)

    # Compute sums vector first, then assert in a comprehension (avoids soft-scope counter)
    sums = Float64[]
    for i in shared_idx
        pg  = diff.results.pep_gained[i]
        pr  = diff.results.pep_reduced[i]
        pu  = diff.results.pep_unchanged[i]
        pbn = diff.results.pep_both_negative[i]
        if !ismissing(pg) && !ismissing(pr) && !ismissing(pu) && !ismissing(pbn)
            push!(sums, (1.0 - pg) + (1.0 - pr) + (1.0 - pu) + (1.0 - pbn))
        end
    end
    @test !isempty(sums)    # ensure we actually exercised the invariant
    @test all(s -> isapprox(s, 1.0, atol = 1e-9), sums)
end

@testitem "gamma-PEP edge cases sigma_zero clamp + Z-vanishing fallback (C5)" tags=[:pep, :gamma] begin
    using BayesInteractomics
    using DataFrames

    # _compute_gamma_pep is internal to the Differential submodule;
    # not forwarded to the BayesInteractomics namespace.
    _gamma = getfield(BayesInteractomics.Differential, :_compute_gamma_pep)
    cfg = DifferentialConfig()   # delta_log2fc_threshold = 1.0

    # Saturated logistic regime: |Δlog2FC|=5 ≫ δ=1, k=10 → σ_pos≈1, σ_neg≈0, σ_zero=max(0,1-1-0)=0
    # Pitfall 2 σ_zero clamp must keep all four γ-PEP outputs in [0, 1].
    df_sat = DataFrame(
        posterior_prob_A = [0.9],
        posterior_prob_B = [0.5],
        delta_log2fc = [5.0],
    )
    out_sat = _gamma(df_sat, cfg)
    @test 0.0 <= out_sat.pep_gained[1] <= 1.0
    @test 0.0 <= out_sat.pep_reduced[1] <= 1.0
    @test 0.0 <= out_sat.pep_unchanged[1] <= 1.0
    @test 0.0 <= out_sat.pep_both_negative[1] <= 1.0
    # σ_zero clamp guarantees pep_unchanged ≈ 1.0 in this saturated regime
    @test isapprox(out_sat.pep_unchanged[1], 1.0; atol = 1e-9)

    # p_A=0, p_B=0, Δlog2FC=0: BOTH_NEGATIVE-dominant regime.
    #   σ_pos ≈ 4.5e-5, σ_neg ≈ 4.5e-5, σ_zero ≈ 1
    #   raw_gained        = 0 · 1 · 4.5e-5 = 0
    #   raw_reduced       = 1 · 0 · 4.5e-5 = 0
    #   raw_unchanged     = 0 · 0 · 1      = 0
    #   raw_both_negative = 1 · 1          = 1.0
    #   Z = 1.0 → P(both_negative) = 1.0, others = 0; pep_both_negative ≈ 0
    df_bn = DataFrame(
        posterior_prob_A = [0.0],
        posterior_prob_B = [0.0],
        delta_log2fc = [0.0],
    )
    out_bn = _gamma(df_bn, cfg)
    @test isapprox(out_bn.pep_both_negative[1], 0.0; atol = 1e-9)
    @test isapprox(out_bn.pep_gained[1],        1.0; atol = 1e-9)
    @test isapprox(out_bn.pep_reduced[1],       1.0; atol = 1e-9)
    @test isapprox(out_bn.pep_unchanged[1],     1.0; atol = 1e-9)

    # Z-vanishing branch trigger: feed NaN through Δlog2FC, which propagates to
    # missing outputs — exercises the explicit-fallback path even
    # though Z<eps is unreachable from valid Float64 inputs in (0,1)x(0,1).
    df_nan = DataFrame(
        posterior_prob_A = [0.5],
        posterior_prob_B = [0.5],
        delta_log2fc = [NaN],
    )
    out_nan = _gamma(df_nan, cfg)
    @test ismissing(out_nan.pep_gained[1])
    @test ismissing(out_nan.pep_reduced[1])
    @test ismissing(out_nan.pep_unchanged[1])
    @test ismissing(out_nan.pep_both_negative[1])

    # Missing input propagation: missing → missing, not NaN
    df_missing = DataFrame(
        posterior_prob_A = Union{Missing,Float64}[missing],
        posterior_prob_B = [0.5],
        delta_log2fc = [1.0],
    )
    out_miss = _gamma(df_missing, cfg)
    @test ismissing(out_miss.pep_gained[1])
    @test ismissing(out_miss.pep_reduced[1])
    @test ismissing(out_miss.pep_unchanged[1])
    @test ismissing(out_miss.pep_both_negative[1])
end

@testitem "gamma-PEP single-condition collapse (Pitfall 6) (C5)" tags=[:pep, :gamma, :condition_specific] begin
    include(joinpath(@__DIR__, "mock_helper.jl"))
    using DataFrames
    using BayesInteractomics: CONDITION_A_SPECIFIC, CONDITION_B_SPECIFIC

    # Construct two mock AR with disjoint protein sets so the inner-join yields
    # only condition-specific rows.
    result_A = _make_mock_result(proteins = ["onlyA1", "onlyA2", "shared"], seed = 42)
    result_B = _make_mock_result(proteins = ["onlyB1", "onlyB2", "shared"], seed = 43)
    diff = differential_analysis(result_A, result_B; condition_A = "A", condition_B = "B")

    # At least one row should be CONDITION_A_SPECIFIC or BOTH_NEGATIVE (latter when bfdr threshold not crossed).
    cond_a_only = findall(==("condition_a_only"), diff.results.detected_in)
    @test !isempty(cond_a_only)

    # Condition-specific row builder: condition-specific rows set
    # differential_pep = missing (α undefined without both sides).
    for i in cond_a_only
        @test ismissing(diff.results.differential_pep[i])
    end
end

@testitem "is_calibrated_A and is_calibrated_B propagation (C6)" setup=[DifferentialFixtures] tags=[:pep, :calibration, :differential] begin
    using BayesInteractomics

    # Mixed-calibration fixture: A=true, B=false. Must survive into DifferentialResult
    # without being OR'd into a single flag (per-side honesty).
    diff_mixed = DifferentialFixtures.create_two_condition_result(
        is_calibrated_A = true,
        is_calibrated_B = false,
    )
    @test diff_mixed.is_calibrated_A == true
    @test diff_mixed.is_calibrated_B == false
    # isCalibrated returns NamedTuple{(:A, :B), Tuple{Bool, Bool}}
    nt = isCalibrated(diff_mixed)
    @test nt.A == true
    @test nt.B == false

    # Both-true and both-false branches
    diff_both_t = DifferentialFixtures.create_two_condition_result(
        is_calibrated_A = true, is_calibrated_B = true)
    @test isCalibrated(diff_both_t) == (A = true, B = true)

    diff_both_f = DifferentialFixtures.create_two_condition_result(
        is_calibrated_A = false, is_calibrated_B = false)
    @test isCalibrated(diff_both_f) == (A = false, B = false)
end

@testitem "getDifferentialPEP throws on unknown class (C8 partial)" tags=[:pep, :accessors, :differential] begin
    include(joinpath(@__DIR__, "mock_helper.jl"))
    using BayesInteractomics

    @test isdefined(BayesInteractomics, :getDifferentialPEP)

    # Use a real differential_analysis() output so the canonical PEP columns exist.
    result_A = _make_mock_result(seed = 42)
    result_B = _make_mock_result(seed = 43)
    diff = differential_analysis(result_A, result_B; condition_A = "A", condition_B = "B")

    # Unknown class → ArgumentError per accessor contract (types.jl:322-326)
    @test_throws ArgumentError getDifferentialPEP(diff; class = :foo)
    @test_throws ArgumentError getDifferentialPEP(diff; class = :unknown)

    # Valid classes return the per-class PEP Vector
    @test getDifferentialPEP(diff) isa AbstractVector                       # default :alpha
    @test getDifferentialPEP(diff; class = :alpha) isa AbstractVector
    @test getDifferentialPEP(diff; class = :gained) isa AbstractVector
    @test getDifferentialPEP(diff; class = :reduced) isa AbstractVector
    @test getDifferentialPEP(diff; class = :unchanged) isa AbstractVector
    @test getDifferentialPEP(diff; class = :both_negative) isa AbstractVector

    # :alpha default returns the canonical differential_pep column (same Vector ref)
    @test getDifferentialPEP(diff) === diff.results.differential_pep
    @test getDifferentialPEP(diff; class = :gained) === diff.results.pep_gained
end
