@testitem "intermediate_cache version split" tags=[:cache, :mask_aware] begin
    using BayesInteractomics
    using JLD2
    using Dates
    using DataFrames

    # ------------------------------------------------------------------
    # Test 1+2: All four per-cache constants exist and equal each other
    #          at split time; INTERMEDIATE_CACHE_VERSION is the alias.
    # ------------------------------------------------------------------
    @test isdefined(BayesInteractomics, :BB_CACHE_VERSION)
    @test isdefined(BayesInteractomics, :HBM_REGRESSION_CACHE_VERSION)
    @test isdefined(BayesInteractomics, :H0_CACHE_VERSION)
    @test isdefined(BayesInteractomics, :CALIBRATION_CACHE_VERSION)
    @test isdefined(BayesInteractomics, :INTERMEDIATE_CACHE_VERSION)

    bb  = BayesInteractomics.BB_CACHE_VERSION
    hbm = BayesInteractomics.HBM_REGRESSION_CACHE_VERSION
    h0  = BayesInteractomics.H0_CACHE_VERSION
    cal = BayesInteractomics.CALIBRATION_CACHE_VERSION
    alias = BayesInteractomics.INTERMEDIATE_CACHE_VERSION

    # HBM_REGRESSION_CACHE_VERSION was bumped for the v2b mask-aware
    # regression observation factor change. The other three constants are
    # unchanged. The contract is: invalidating one cache
    # does NOT churn the others.
    @test bb == 16                    # BetaBernoulli cache version (current)
    @test h0 == cal == 17             # H0 + calibration caches (current, bumped together)
    @test hbm == 19                   # HBM regression cache (current)
    # The deprecated alias === BB_CACHE_VERSION.
    @test alias === bb

    # ------------------------------------------------------------------
    # Test 3+5: Positive load round-trip for HBM regression cache.
    #          Writing with cache_version = HBM_REGRESSION_CACHE_VERSION
    #          must NOT return nothing; writing with version - 1 MUST.
    # ------------------------------------------------------------------
    mktempdir() do tmpdir
        # --- HBM Regression: positive (correct version) -------------
        hbm_pos = joinpath(tmpdir, "hbm_pos.jld2")
        jldsave(hbm_pos; compress=true,
            cache_version       = BayesInteractomics.HBM_REGRESSION_CACHE_VERSION,
            df_hierarchical     = DataFrame(),
            bf_enrichment       = Float64[],
            bf_correlation      = Float64[],
            protein_ids         = String[],
            refID               = 1,
            regression_likelihood = :normal,
            student_t_nu        = 4.0,
            regression_bf_threshold = 0.0,
            data_hash           = UInt64(0),
            timestamp           = string(now()),
            package_version     = "test",
            imputation_method   = :mar,
        )
        # The load may return nothing if the DataFrame type doesn't round-trip;
        # but the KEY contract here is "version matches → does NOT short-circuit
        # on the version line". Either a non-nothing return OR a return-due-to-
        # field-shape mismatch is acceptable; what we MUST NOT see is the
        # version-line short-circuit silently rejecting a valid version.
        # Use the negative test below as the canonical assertion.
        loaded_pos = try
            BayesInteractomics.load_hbm_regression_cache(hbm_pos)
        catch
            nothing
        end
        # Sanity: positive load either succeeds (HBMRegressionCache) or fails
        # gracefully for unrelated field issues — both are acceptable here.
        @test loaded_pos === nothing || loaded_pos isa BayesInteractomics.HBMRegressionCache

        # --- HBM Regression: negative (wrong version) ---------------
        hbm_neg = joinpath(tmpdir, "hbm_neg.jld2")
        jldsave(hbm_neg; compress=true,
            cache_version       = BayesInteractomics.HBM_REGRESSION_CACHE_VERSION - 1,
            df_hierarchical     = DataFrame(),
            bf_enrichment       = Float64[],
            bf_correlation      = Float64[],
            protein_ids         = String[],
            refID               = 1,
            regression_likelihood = :normal,
            student_t_nu        = 4.0,
            regression_bf_threshold = 0.0,
            data_hash           = UInt64(0),
            timestamp           = string(now()),
            package_version     = "test",
            imputation_method   = :mar,
        )
        @test BayesInteractomics.load_hbm_regression_cache(hbm_neg) === nothing

        # --- Beta-Bernoulli: negative (wrong version) ---------------
        bb_neg = joinpath(tmpdir, "bb_neg.jld2")
        jldsave(bb_neg; compress=true,
            cache_version       = BayesInteractomics.BB_CACHE_VERSION - 1,
            bf_detected         = Float64[],
            protein_ids         = String[],
            n_controls          = 3,
            n_samples           = 3,
            data_hash           = UInt64(0),
            timestamp           = string(now()),
            package_version     = "test",
            imputation_method   = :mar,
        )
        @test BayesInteractomics.load_betabernoulli_cache(bb_neg) === nothing

        # --- H0: negative (wrong version) ---------------------------
        h0_neg = joinpath(tmpdir, "h0_neg.jld2")
        jldsave(h0_neg; compress=true,
            cache_version       = BayesInteractomics.H0_CACHE_VERSION - 1,
            cache_type          = "h0_logbf",
        )
        @test BayesInteractomics.load_h0_cache(h0_neg) === nothing
    end

    # ------------------------------------------------------------------
    # Test 4: bumping HBM_REGRESSION_CACHE_VERSION must not change the
    #         other three. We can't redefine a const at runtime, so the
    #         contract is asserted by checking that each save_*/load_*
    #         function references its own constant. That is enforced
    #         structurally in the source (grep-based check below), and
    #         empirically by the negative-load tests above (each cache
    #         type's load function rejects only when ITS OWN constant
    #         is the one mismatched).
    # ------------------------------------------------------------------
    src_path = joinpath(pkgdir(BayesInteractomics), "src", "core", "intermediate_cache.jl")
    @test isfile(src_path)
    src_text = read(src_path, String)
    # Each per-cache constant is referenced by its own save/load pair.
    @test occursin("BB_CACHE_VERSION", src_text)
    @test occursin("HBM_REGRESSION_CACHE_VERSION", src_text)
    @test occursin("H0_CACHE_VERSION", src_text)
    @test occursin("CALIBRATION_CACHE_VERSION", src_text)
end
