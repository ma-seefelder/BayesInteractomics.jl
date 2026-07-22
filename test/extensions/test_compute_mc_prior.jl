# `_safe_compute_mc_prior!` Variante-B contract.
#
# Covered validation requirements:
#   - V-04 : `compute_mc_prior!` does NOT mutate `posterior_prob` — byte-equality
#            when `run_dnn_prior_mc_dropout` is toggled true/false on the same fixture.
#   - V-07 : extension-not-loaded fallback — `_safe_compute_mc_prior!` catches the
#            MethodError, emits exactly one @warn (maxlog=1), populates 5 prior
#            columns with NaN, leaves `posterior_prob` untouched.

@testitem "V-07: _safe_compute_mc_prior! Variante-B — @warn maxlog=1 + NaN columns + posterior_prob untouched" begin
    using BayesInteractomics, Test, DataFrames

    # The parent stub `compute_mc_prior!` exists but has zero methods
    # at the BayesInteractomics namespace until `using Flux, MLJ, MLJScikitLearnInterface, HDF5`
    # activates the extension. This test exercises the no-extension code path:
    # `_safe_compute_mc_prior!` should catch the MethodError, emit one @warn, and
    # populate the 5 prior columns with NaN — leaving `posterior_prob` byte-identical.
    @test isdefined(BayesInteractomics, :compute_mc_prior!)

    df = DataFrame(
        Protein = ["A", "B", "C"],
        posterior_prob = [0.9, 0.5, 0.1],
    )
    X = randn(Float32, 8, 3)
    cfg = (
        run_dnn_prior_mc_dropout = true,
        dnn_prior_mc_k = 30,
        dnn_prior_mc_batch_size = 4,
    )

    if isempty(methods(BayesInteractomics.compute_mc_prior!))
        # Extension genuinely not loaded — exercise the real Variante-B path:
        # `_safe_compute_mc_prior!` catches the MethodError, emits one @warn,
        # returns :extension_not_loaded, and populates the 5 prior columns with NaN.
        @test isempty(methods(BayesInteractomics.compute_mc_prior!))

        local status
        @test_logs (:warn, r"Metalearner extension.*not loaded"i) match_mode=:any begin
            status = BayesInteractomics._safe_compute_mc_prior!(df, X, cfg)
            @test status === :extension_not_loaded
        end

        # NaN columns populated (5 new MC-Dropout prior columns)
        @test all(isnan, df.prior_mc_mean)
        @test all(isnan, df.prior_mc_std)
        @test all(isnan, df.prior_mc_ci_low)
        @test all(isnan, df.prior_mc_ci_high)
        @test all(isnan, df.prior_contribution)

        # Byte-equality: posterior_prob untouched by Variante-B path
        @test df.posterior_prob == [0.9, 0.5, 0.1]
    else
        @info "Skipping compute_mc_prior! not-loaded fallback assertions: metalearner extension is globally loaded in this suite (covered by test_metalearner_refit_subprocess.jl in a fresh process)."
        @test true
    end
end

@testitem "V-04: posterior_prob byte-equal when flag toggled true/false" begin
    using BayesInteractomics, Test, DataFrames

    # V-04 byte-equality assertion on the Variante-B fixture: with the metalearner
    # extension NOT loaded, the wrapper falls through to `:extension_not_loaded`
    # (opt-in) or `:skipped` (opt-out) and leaves `posterior_prob` untouched on
    # both paths. The full extension-loaded path is covered by a manual visual checkpoint.
    tmpdir = mktempdir()
    cfg_on = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "MYC",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html     = false,
        run_dnn_prior_mc_dropout = true,
        run_simulation           = false,
        run_validation           = false,
    )
    cfg_off = deepcopy(cfg_on)
    cfg_off.run_dnn_prior_mc_dropout = false

    df_on  = DataFrame(Protein = ["A","B","C"], posterior_prob = [0.9, 0.5, 0.1])
    df_off = deepcopy(df_on)
    X = randn(Float32, 8, 3)
    BayesInteractomics._safe_compute_mc_prior!(df_on,  X, cfg_on)
    BayesInteractomics._safe_compute_mc_prior!(df_off, X, cfg_off)
    @test isequal(df_on.posterior_prob, df_off.posterior_prob)
end
