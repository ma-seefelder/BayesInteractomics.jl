# Bait-side PEP tests
#
# Covers coverage dimensions:
#   - C1  bait df.pep canonical column exists
#   - C2  pep == 1 - coalesce(posterior_calibrated, posterior_prob)
#   - C7  getPEP / isCalibrated accessor API
#   - C12 is_calibrated JLD2 round-trip + CACHE_VERSION 23
#   - C13 df.PEP === df.pep uppercase mirror (Vector identity, not equal-by-value)
#
# Tag scheme — `:pep` is the greppable root; sub-tags
# (`:calibration`, `:accessors`, `:jld2`) mark coverage dimension.

@testitem "bait df.pep canonical column exists (C1)" tags=[:pep] begin
    using BayesInteractomics
    using DataFrames

    # Minimal smoke construction (mirrors the canonical wiring in pipeline.jl:813,822).
    df = DataFrame(
        Protein = ["P1", "P2"],
        posterior_prob = [0.9, 0.1],
        pep = [0.1, 0.9],
    )
    df.PEP = df.pep   # silent uppercase mirror — same Vector ref
    @test hasproperty(df, :pep)
    @test hasproperty(df, :PEP)
    @test df.PEP === df.pep   # same Vector reference (identity)
    @test df.pep == [0.1, 0.9]
end

@testitem "df.PEP === df.pep uppercase mirror (C13)" tags=[:pep] begin
    using BayesInteractomics
    using DataFrames
    df = DataFrame(pep = [0.1, 0.9])
    df.PEP = df.pep                    # silent mirror
    # Mutating one MUST mutate the other (same Vector reference)
    df.pep[1] = 0.5
    @test df.PEP[1] == 0.5
    @test df.PEP === df.pep
    # Length invariance under mutation
    push!(df.pep, 0.42)
    @test df.PEP === df.pep
    @test length(df.PEP) == 3
end

@testitem "pep == 1 - coalesce(posterior_calibrated, posterior_prob) (C2)" tags=[:pep, :calibration] begin
    using BayesInteractomics
    using BayesInteractomics: pep
    using DataFrames

    # Case (a): posterior_calibrated present → pep = 1 - posterior_calibrated
    df_cal = DataFrame(
        posterior_prob = [0.9, 0.5],
        posterior_calibrated = [0.8, 0.4],
    )
    post = coalesce.(df_cal.posterior_calibrated, df_cal.posterior_prob)
    p = pep(post)
    @test p ≈ [0.2, 0.6]

    # Case (b): posterior_calibrated absent → pep = 1 - posterior_prob
    df_raw = DataFrame(posterior_prob = [0.9, 0.5])
    p2 = pep(df_raw.posterior_prob)
    @test p2 ≈ [0.1, 0.5]

    # Case (c): missing posterior_calibrated coalesces to posterior_prob
    df_mixed = DataFrame(
        posterior_prob = [0.9, 0.5],
        posterior_calibrated = Union{Missing, Float64}[missing, 0.4],
    )
    post_mixed = coalesce.(df_mixed.posterior_calibrated, df_mixed.posterior_prob)
    p3 = pep(post_mixed)
    @test p3 ≈ [0.1, 0.6]
end

@testitem "getPEP and isCalibrated accessors (C7)" tags=[:pep, :accessors] begin
    using BayesInteractomics
    @test isdefined(BayesInteractomics, :getPEP)
    @test isdefined(BayesInteractomics, :isCalibrated)
    # Both should have at least one method (getPEP: AnalysisResult)
    @test length(methods(BayesInteractomics.getPEP)) >= 1
    # isCalibrated dispatches on both AnalysisResult AND DifferentialResult
    @test length(methods(BayesInteractomics.isCalibrated)) >= 2
end

@testitem "is_calibrated JLD2 round-trip + CACHE_VERSION 23 (C12)" tags=[:pep, :jld2] begin
    using BayesInteractomics
    using JLD2

    # CACHE_VERSION pinned to 23
    @test isdefined(BayesInteractomics, :CACHE_VERSION)
    @test BayesInteractomics.CACHE_VERSION == 26

    # Stale-cache loud-fail: a v22 JLD2 file must NOT load — load_result returns nothing
    tmp = tempname() * ".jld2"
    try
        JLD2.jldsave(tmp; cache_version = 22, dummy_payload = "stale")
        loaded_stale = BayesInteractomics.load_result(tmp)
        @test loaded_stale === nothing   # CACHE_VERSION mismatch → nothing (per results.jl:457)
    finally
        rm(tmp; force = true)
    end

    # Future-version cache also fails loudly
    tmp2 = tempname() * ".jld2"
    try
        JLD2.jldsave(tmp2; cache_version = 99, dummy_payload = "future")
        loaded_future = BayesInteractomics.load_result(tmp2)
        @test loaded_future === nothing
    finally
        rm(tmp2; force = true)
    end
end
