# bb_mnar_codriven diagnostic tests
#
# Coverage:
#   - C10 bb_mnar_codriven Boolean rule truth-table (2^3 = 8 corner cases)
#         (bf_detected > cfg.bb_bf_threshold) ∧ (bf_combined > cfg.hbm_bf_threshold)
#                                              ∧ (missing_fraction > cfg.missing_fraction_threshold)
#   - C11 missing_fraction provenance: helper exists and is callable
#         (full pipeline identity across imputation methods deferred to integration coverage)
#
# Rule (strict `>`, NOT `>=`):
#   bb_mnar_codriven_i := (bf_detected_i  > cfg.bb_bf_threshold)
#                      ∧ (bf_combined_i   > cfg.hbm_bf_threshold)
#                      ∧ (missing_fraction_i > cfg.missing_fraction_threshold)
# Defaults: bb=10.0, hbm=10.0, missing=0.5

@testitem "BBMnarCodrivenConfig defaults" tags=[:diag, :bb_mnar] begin
    using BayesInteractomics

    @test isdefined(BayesInteractomics, :BBMnarCodrivenConfig)

    cfg = BBMnarCodrivenConfig()
    @test cfg.bb_bf_threshold            == 10.0
    @test cfg.hbm_bf_threshold           == 10.0
    @test cfg.missing_fraction_threshold == 0.5

    # Kwarg overrides preserve other defaults
    cfg2 = BBMnarCodrivenConfig(bb_bf_threshold = 20.0)
    @test cfg2.bb_bf_threshold            == 20.0
    @test cfg2.hbm_bf_threshold           == 10.0
    @test cfg2.missing_fraction_threshold == 0.5

    cfg3 = BBMnarCodrivenConfig(missing_fraction_threshold = 0.75)
    @test cfg3.missing_fraction_threshold == 0.75
    @test cfg3.bb_bf_threshold            == 10.0
    @test cfg3.hbm_bf_threshold           == 10.0
end

@testitem "_compute_bb_mnar_codriven truth-table 2^3 corners (C10)" tags=[:diag, :bb_mnar] begin
    using BayesInteractomics

    # _compute_bb_mnar_codriven lives in src/diagnostics/ and is parent-namespace
    # accessible (no submodule wrap, unlike Differential).
    _flag = getfield(BayesInteractomics, :_compute_bb_mnar_codriven)
    cfg = BBMnarCodrivenConfig()   # 10.0, 10.0, 0.5

    low_bb, high_bb = 5.0, 50.0
    low_hbm, high_hbm = 5.0, 50.0
    low_mf, high_mf = 0.2, 0.8

    # All 8 corners (L, L, L) through (H, H, H). Only (H, H, H) → true.
    @test _flag([low_bb],  [low_hbm],  [low_mf],  cfg) == [false]   # (L, L, L)
    @test _flag([low_bb],  [low_hbm],  [high_mf], cfg) == [false]   # (L, L, H)
    @test _flag([low_bb],  [high_hbm], [low_mf],  cfg) == [false]   # (L, H, L)
    @test _flag([low_bb],  [high_hbm], [high_mf], cfg) == [false]   # (L, H, H)
    @test _flag([high_bb], [low_hbm],  [low_mf],  cfg) == [false]   # (H, L, L)
    @test _flag([high_bb], [low_hbm],  [high_mf], cfg) == [false]   # (H, L, H)
    @test _flag([high_bb], [high_hbm], [low_mf],  cfg) == [false]   # (H, H, L)
    @test _flag([high_bb], [high_hbm], [high_mf], cfg) == [true]    # (H, H, H) ← only true corner

    # Boundary: exactly threshold values → false (strict `>`, NOT `>=`)
    @test _flag([10.0], [10.0], [0.5], cfg) == [false]
    # Slightly above all three → true
    @test _flag([10.001], [10.001], [0.501], cfg) == [true]
    # One side at boundary while the other two are above → still false
    @test _flag([10.0],   [50.0], [0.8], cfg) == [false]
    @test _flag([50.0],   [10.0], [0.8], cfg) == [false]
    @test _flag([50.0],   [50.0], [0.5], cfg) == [false]

    # Vectorised: mixed-length 4-element vectors
    bf_det = [50.0,  5.0, 50.0, 10.001]
    bf_cmb = [50.0, 50.0,  5.0, 10.001]
    mfrac  = [0.8,   0.8,  0.8, 0.501]
    @test _flag(bf_det, bf_cmb, mfrac, cfg) == [true, false, false, true]
end

@testitem "missing_fraction provenance pre-imputation (C11)" tags=[:diag, :missingness] begin
    using BayesInteractomics

    # Helper lives in src/data/missingness.jl; not exported but reachable.
    @test isdefined(BayesInteractomics, :_compute_per_protein_missingness)
    _mf = getfield(BayesInteractomics, :_compute_per_protein_missingness)
    @test length(methods(_mf)) >= 1

    # Note: full provenance contract (identical Vector{Float64} for imputation_method ∈
    # (:mnar, :mar, :none) on the same raw InteractionData) is exercised end-to-end
    # in the integration testitems — running analyse() three times here would
    # blow past the per-task feedback budget (60s). This testitem validates the
    # surface API; integration coverage is deferred.
end

@testitem "missing_fraction = 1.0 protein never flags codriven (Pitfall 3)" tags=[:diag, :bb_mnar] begin
    using BayesInteractomics

    _flag = getfield(BayesInteractomics, :_compute_bb_mnar_codriven)
    cfg = BBMnarCodrivenConfig()

    # Protein with zero sample detections: bf_detected ≈ 0 (BB prior collapse),
    # missing_fraction = 1.0. The first AND clause (bf_detected > 10) fails → false.
    # Pitfall 3 contract: high missingness ALONE must NEVER flag.
    @test _flag([0.0],     [100.0], [1.0], cfg) == [false]
    @test _flag([missing], [100.0], [1.0], cfg) == [false]   # missing → coalesce(0.0)
    @test _flag([0.5],     [100.0], [1.0], cfg) == [false]   # below bb threshold still
    @test _flag([5.0],     [100.0], [1.0], cfg) == [false]   # just below
    @test _flag([10.0],    [100.0], [1.0], cfg) == [false]   # exactly at threshold (strict >)

    # Only when bf_detected ALSO crosses → true
    @test _flag([10.0001], [100.0], [1.0], cfg) == [true]

    # Per-clause missing propagation through coalesce → 0.0
    @test _flag([50.0], [missing], [1.0],     cfg) == [false]   # hbm coalesces to 0.0
    @test _flag([50.0], [50.0],    [missing], cfg) == [false]   # mfrac coalesces to 0.0
end
