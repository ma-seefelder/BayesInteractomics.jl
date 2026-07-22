# Metalearner Variante B fallback tests.
#
# These testitems run WITHOUT `using Flux, MLJ, MLJScikitLearnInterface, HDF5`,
# so the metalearner extension (`BayesInteractomicsMetalearnerExt`) is NOT
# loaded and `predict_metalearner` has zero methods. The Variante B helper
# `_safe_predict_metalearner` must:
#   1. Catch the resulting MethodError (covers both direct and Core.kwcall forms),
#   2. Emit exactly ONE @warn per session (maxlog=1),
#   3. Return `(nothing, nothing, :extension_not_loaded)`.
#
# The AnalysisResult struct must also carry the `metalearner_status`
# field of type `Symbol`, and the cache version must reflect the schema change.

@testitem "Metalearner stub MethodError caught by _safe_predict_metalearner" begin
    using BayesInteractomics
    using Test

    # Construct a minimal CONFIG that exercises _safe_predict_metalearner.
    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "DUMMY",
        n_controls   = 3,
        n_samples    = 3,
        refID        = 1,
        output       = OutputFiles(tmpdir),
    )

    if isempty(methods(BayesInteractomics.predict_metalearner))
        # Extension genuinely not loaded — exercise the real Variante B fallback.
        # Without `using Flux, MLJ, MLJScikitLearnInterface, HDF5`, the stub has zero methods.
        @test isempty(methods(BayesInteractomics.predict_metalearner))

        result = BayesInteractomics._safe_predict_metalearner(cfg)
        @test isa(result, Tuple)
        @test length(result) == 3
        @test result[1] === nothing
        @test result[2] === nothing
        @test result[3] === :extension_not_loaded
    else
        @info "Skipping metalearner not-loaded fallback assertions: extension is globally loaded in this suite (covered by test_metalearner_refit_subprocess.jl in a fresh process)."
        @test true
    end
end

@testitem "Metalearner fallback emits @warn matching extension-not-loaded message" begin
    using BayesInteractomics
    using Test

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "DUMMY",
        n_controls   = 3,
        n_samples    = 3,
        refID        = 1,
        output       = OutputFiles(tmpdir),
    )

    if isempty(methods(BayesInteractomics.predict_metalearner))
        # Extension genuinely not loaded — the Variante B helper emits exactly one
        # @warn matching the "Metalearner extension … not loaded" message on its
        # first call (maxlog=1; subsequent calls within the same session stay
        # silent). `@test_logs match_mode=:any` verifies the warn fires at least once.
        @test_logs (:warn, r"Metalearner extension.*not loaded"i) match_mode=:any begin
            BayesInteractomics._safe_predict_metalearner(cfg)
        end
    else
        @info "Skipping metalearner not-loaded @warn assertion: extension is globally loaded in this suite (covered by test_metalearner_refit_subprocess.jl in a fresh process)."
        @test true
    end
end

@testitem "AnalysisResult.metalearner_status field and CACHE_VERSION current" begin
    using BayesInteractomics
    using Test

    # The AnalysisResult struct carries a `metalearner_status::Symbol`
    # field, and the cache version was bumped 20 -> 21 to invalidate any earlier JLD2
    # caches that lack the new field. Subsequent versions continued the bump trail
    # (22, 23, 24, 25); the 24 -> 25 bump accompanied the TR+DDI
    # metalearner schema. The field-presence assertion is the durable invariant;
    # the version assertion tracks current value.
    @test :metalearner_status in fieldnames(BayesInteractomics.AnalysisResult)
    @test fieldtype(BayesInteractomics.AnalysisResult, :metalearner_status) === Symbol
    @test BayesInteractomics.CACHE_VERSION == 26

    # The METALEARNER_STATUS_VALUES tuple must enumerate the three sentinels.
    @test BayesInteractomics.METALEARNER_STATUS_VALUES ==
          (:extension_not_loaded, :loaded, :prediction_failed)
end
