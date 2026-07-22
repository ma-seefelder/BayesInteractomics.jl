# Imputation extension explicit-error fallback tests.
#
# Unlike the metalearner (which uses Variante B soft fallback), the imputation
# extension uses an EXPLICIT-ERROR gate: the user opted in to
# :mar / :mnar imputation, so a missing GLM extension MUST throw a loud
# ArgumentError pointing them at `using GLM`. `imputation_method = :none`
# (the silent path) is preserved.

@testitem "imputation=:none silently skips imputation stubs even when GLM is not loaded" begin
    using BayesInteractomics
    using Test

    if isempty(methods(BayesInteractomics.fit_dropout_curves))
        # Extension genuinely not loaded — verify the stubs have zero methods.
        @test isempty(methods(BayesInteractomics.fit_dropout_curves))
        @test isempty(methods(BayesInteractomics.impute_mnar))
    else
        @info "Skipping imputation not-loaded stub assertions: GLM extension is globally loaded in this suite (some other testitem does `using GLM`)."
        @test true
    end
    # _require_imputation_extension(:none) must return nothing without throwing —
    # this holds regardless of whether the extension is loaded.
    @test BayesInteractomics._require_imputation_extension(:none) === nothing
end

@testitem "imputation=:mnar without GLM throws ArgumentError mentioning 'using GLM'" begin
    using BayesInteractomics
    using Test

    # In this testitem the imputation extension is not loaded (we did not
    # `using GLM`). The preflight helper MUST throw a loud ArgumentError
    # naming the package the user needs to load.
    @test_throws ArgumentError BayesInteractomics._require_imputation_extension(:mnar)

    try
        BayesInteractomics._require_imputation_extension(:mnar)
        @test false  # unreachable
    catch e
        @test isa(e, ArgumentError)
        @test occursin("using GLM", e.msg)
    end
end

@testitem "imputation=:mar without GLM throws ArgumentError mentioning 'using GLM'" begin
    using BayesInteractomics
    using Test

    @test_throws ArgumentError BayesInteractomics._require_imputation_extension(:mar)

    try
        BayesInteractomics._require_imputation_extension(:mar)
        @test false  # unreachable
    catch e
        @test isa(e, ArgumentError)
        @test occursin("using GLM", e.msg)
    end
end
