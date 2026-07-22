"""
    test_sd_log2FC_smoke.jl

Per-factory smoke test asserting `:sd_log2FC` is present on
the `.results` DataFrame of every AnalysisResult produced by ALL THREE
DifferentialFixtures factories. Covers the fixture spec +
the k=2 legacy parity requirement (needs ar.results.sd_log2FC).
"""

@testitem "three_factory_sd_log2FC_smoke" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    using Test

    # Factory 1/3: create_two_condition_result (returns DifferentialResult)
    diff2 = DifferentialFixtures.create_two_condition_result()
    @test :sd_log2FC in propertynames(diff2.analyses[1].results)
    @test :sd_log2FC in propertynames(diff2.analyses[2].results)
    # Column-name lock: omnibus reads ar.results.mean_log2FC
    @test :mean_log2FC in propertynames(diff2.analyses[1].results)

    # Factory 2/3: create_three_condition_result (returns NamedTuple)
    fx3 = DifferentialFixtures.create_three_condition_result()
    @test :sd_log2FC in propertynames(fx3.ar_wt.results)
    @test :sd_log2FC in propertynames(fx3.ar_mut1.results)
    @test :sd_log2FC in propertynames(fx3.ar_mut2.results)

    # Factory 3/3: create_four_condition_result (NEW)
    fx4 = DifferentialFixtures.create_four_condition_result()
    @test length(propertynames(fx4)) == 4
    @test :ar_wt in propertynames(fx4)
    @test :ar_mut1 in propertynames(fx4)
    @test :ar_mut2 in propertynames(fx4)
    @test :ar_mut3 in propertynames(fx4)
    @test :sd_log2FC in propertynames(fx4.ar_wt.results)
    @test :sd_log2FC in propertynames(fx4.ar_mut3.results)

    # Per-protein σ variation: zero-σ clamp driver (P1) and BF→∞ driver (P11).
    @test fx4.ar_wt.results.sd_log2FC[1] ≈ 1e-6
    @test fx4.ar_wt.results.sd_log2FC[11] ≈ 0.05

    # Shared bait label across all 4 conditions — no bait-mismatch @warn fires.
    @test fx4.ar_wt.bait_protein == "BAIT"
    @test fx4.ar_mut3.bait_protein == "BAIT"

    # Ground-truth ≥50 rows (block structure 10+10+10+20).
    @test nrow(fx4.ar_wt.results) >= 50
end
