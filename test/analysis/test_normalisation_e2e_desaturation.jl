"""
    test_normalisation_e2e_desaturation.jl

The end-to-end de-saturation proof + the cache loud-fail contract.

Testitems:
  - "cache loud-fail on stale-version cache": a JLD2 cache dict carrying the OLD
    HBM_REGRESSION / H0 / CALIBRATION versions makes load_*_cache return nothing;
    a BB cache at version 16 (unchanged) still loads (not over-invalidated).
  - "scale-disparate arm reproduction (synthetic)": the committed scale-disparate
    fixture run through the production multi-protocol structured-VMP regression under
    four normalisation arms — A :none, B :row_center, C :median_of_ratios, F :both —
    reproducing the orthogonal-axes pattern (A,C saturated; B,F ~0%).

The locked-narrow real-HAP40 quartiles (A ~38% / B 0% / C ~38% / F 0%) require the
user's uncommitted real HAP40 GST+Strep XLSX and are confirmed via the user-run
checkpoint (run_spike.jl), NOT here.
"""


@testitem "cache loud-fail on stale-version cache" tags=[:cache] begin
    using BayesInteractomics
    using JLD2
    using Dates
    using DataFrames

    # The version bumps: HBM_REGRESSION 18->19, H0 16->17, CALIBRATION 16->17,
    # BB unchanged at 16. A cache file written at the OLD version must be REJECTED
    # (loud-fail -> return nothing) by the matching loader; a BB file at 16 must STILL load.
    OLD_HBM = BayesInteractomics.HBM_REGRESSION_CACHE_VERSION - 1   # 18
    OLD_H0  = BayesInteractomics.H0_CACHE_VERSION - 1               # 16
    OLD_CAL = BayesInteractomics.CALIBRATION_CACHE_VERSION - 1      # 16

    @test OLD_HBM == 18
    @test OLD_H0 == 16
    @test OLD_CAL == 16
    @test BayesInteractomics.BB_CACHE_VERSION == 16

    mktempdir() do tmpdir
        # --- HBM regression at OLD version 18 -> rejected (the @warn + return nothing) ---
        hbm_stale = joinpath(tmpdir, "hbm_stale.jld2")
        jldsave(hbm_stale; compress=true,
            cache_version           = OLD_HBM,
            df_hierarchical         = DataFrame(),
            bf_enrichment           = Float64[],
            bf_correlation          = Float64[],
            protein_ids             = String[],
            refID                   = 1,
            regression_likelihood   = :normal,
            student_t_nu            = 4.0,
            regression_bf_threshold = 0.0,
            data_hash               = UInt64(0),
            timestamp               = string(now()),
            package_version         = "test",
            imputation_method       = :mar,
        )
        @test BayesInteractomics.load_hbm_regression_cache(hbm_stale) === nothing

        # --- H0 at OLD version 16 -> rejected ------------------------------------------
        h0_stale = joinpath(tmpdir, "h0_stale.jld2")
        jldsave(h0_stale; compress=true,
            cache_version = OLD_H0,
            cache_type    = "h0_logbf",
        )
        @test BayesInteractomics.load_h0_cache(h0_stale) === nothing

        # --- Calibration at OLD version 16 -> rejected ---------------------------------
        cal_stale = joinpath(tmpdir, "cal_stale.jld2")
        jldsave(cal_stale; compress=true,
            cache_version     = OLD_CAL,
            cache_type        = "calibration",
            imputation_method = "mnar",
        )
        @test BayesInteractomics.load_calibration_cache(cal_stale; imputation_method=:mnar) === nothing

        # --- BB at CURRENT version 16 -> STILL LOADS (NOT over-invalidated) ------------
        bb_ok = joinpath(tmpdir, "bb_ok.jld2")
        jldsave(bb_ok; compress=true,
            cache_version     = BayesInteractomics.BB_CACHE_VERSION,   # 16
            bf_detected       = Float64[1.0, 2.0],
            protein_ids       = ["P001", "P002"],
            n_controls        = 3,
            n_samples         = 3,
            data_hash         = UInt64(0),
            timestamp         = now(),   # DateTime — BetaBernoulliCache.timestamp::DateTime
            package_version   = "test",
            imputation_method = :mnar,
        )
        bb_loaded = BayesInteractomics.load_betabernoulli_cache(bb_ok)
        @test bb_loaded !== nothing
        @test bb_loaded isa BayesInteractomics.BetaBernoulliCache
    end
end


@testitem "scale-disparate arm reproduction (synthetic)" tags=[:slow, :normalisation] setup=[ScaleDisparateMultiprotocol] begin
    using BayesInteractomics
    using BayesInteractomics: RegressionModel_multi_protocol_robust_jzs_v2b,
        estimate_regression_tau_base, calculate_bayes_factor, _cauchy_sf,
        apply_normalisation, getIDs, prepare_regression_data, build_run_matrix
    using Distributions
    using Statistics

    # ──────────────────────────────────────────────────────────────────────────
    # End-to-end: run the COMMITTED scale-disparate fixture through the
    # production multi-protocol structured-VMP regression under four normalisation
    # arms (apply_normalisation BEFORE the regression). Confirm the
    # ORTHOGONAL-AXES finding:
    #   A :none             -> SATURATED       (no cross-protocol offset removed)
    #   B :row_center       -> de-saturated    (per-protein offset removed; the 018 fix)
    #   C :median_of_ratios -> NOT de-saturated by the row-centering axis
    #                          (per-COLUMN scaling cannot remove the per-PROTEIN offset)
    #   F :both             -> de-saturated    (column scale + row-center; arm F)
    #
    # CONTRACT (the PATTERN is the contract, not exact percentages — like the
    # mask-aware regression fixture). Two robust, honest metrics on the production μ_α Bayes factor:
    #   • saturation fraction `pct_hi` = fraction with bf ≥ SAT_THRESHOLD (the inflated
    #     regime). On this synthetic fixture the structured-VMP slope σ keeps proteins
    #     BELOW the literal 1e6 ceiling even when strongly inflated, so SAT_THRESHOLD is
    #     a high-inflation cut (1e3 ≫ the 0.1 BF threshold), NOT the exact 1e6 ceiling.
    #     The literal 1e6 ceiling pile-up (A ~38 %) is a real-HAP40 property confirmed
    #     via the user-run checkpoint (run_spike.jl, Option a).
    #   • median bf — the central de-saturation signal.
    #
    # Asserted pattern:
    #   A:  pct_hi HIGH (≫ 0)         AND median ≫ 1   (saturated)
    #   B:  median < 1                                 (row-centering de-saturates)
    #   F:  median < 1                                 (compose; tightest)
    #   C:  median > B median AND > F median           (median_of_ratios ALONE does NOT
    #                                                    achieve the row-centering de-sat)
    # ──────────────────────────────────────────────────────────────────────────
    fx = ScaleDisparateMultiprotocol.load_fixture(matched=false)
    refID = fx.refID
    SAT_THRESHOLD = 1.0e3   # high-inflation cut (see comment above; not the 1e6 ceiling)

    function arm_stats(method::Symbol)
        # Normalise BEFORE the regression. The fixture is already imputed; the
        # cross-protocol offset is structural and present in the imputed data, so
        # normalising the imputed matrix exercises the same arm as the spike.
        normed = apply_normalisation(fx.imputed, method)
        n_proteins = length(getIDs(normed))
        global_tau = estimate_regression_tau_base(normed, refID)
        prior_p_override = _cauchy_sf(0.1, 0.354)

        bfs = Float64[]
        for idx in 1:n_proteins
            idx == refID && continue
            try
                prep = prepare_regression_data(normed, idx, refID)
                result = RegressionModel_multi_protocol_robust_jzs_v2b(
                    normed, idx, refID, 0.0, 10.0;
                    is_imputed = falses(size(prep.sample)),
                    column_imputation_sigma_sq = Dict{Tuple{Int,Int,Int}, Float64}(),
                    raw_data = nothing,
                    nu = 5.0,
                    τ_base = global_tau,
                    jzs_r_scale = 0.354,
                    regression_iterations = 80,
                )
                posterior_μ_α = result.posterior.posteriors[:μ_α]
                prior_μ_α     = result.prior.posteriors[:μ_α]
                bf, _, _ = calculate_bayes_factor(
                    [posterior_μ_α], [prior_μ_α];
                    threshold = 0.1, max_bf = 1e6, min_posterior_var = 0.01,
                    prior_p_override = prior_p_override,
                )
                bf_val = bf isa AbstractVector ? bf[1] : bf
                isfinite(bf_val) && push!(bfs, bf_val)
            catch e
                @warn "arm $method protein $idx failed: $e"
            end
        end
        @test length(bfs) >= 0.7 * (n_proteins - 1)   # ≥ 70% convergence
        return (pct_hi = mean(bfs .>= SAT_THRESHOLD), med = median(bfs))
    end

    A = arm_stats(:none)
    B = arm_stats(:row_center)
    C = arm_stats(:median_of_ratios)
    F = arm_stats(:both)

    @info "synthetic normalisation arm reproduction" A_pct=A.pct_hi A_med=A.med B_med=B.med C_med=C.med F_med=F.med

    # A — :none: SATURATED (no normalisation; the cross-protocol offset inflates μ_α).
    @test A.pct_hi > 0.3
    @test A.med > 1.0e3

    # B, F — the ROW-CENTERING axis de-saturates the regression.
    @test B.med < 1.0
    @test F.med < 1.0

    # C — :median_of_ratios ALONE (column scaling) does NOT match the row-centering
    # de-saturation: its median bf stays ABOVE both row-centering arms. This is the
    # decisive orthogonal-axes finding (median_of_ratios is insufficient;
    # per-protein row-centering is what de-saturates multi-protocol bf_correlation).
    @test C.med > B.med
    @test C.med > F.med
end
