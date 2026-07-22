# Imputation extension ACTIVATION tests.
#
# The imputation extension (`BayesInteractomicsImputationExt`) activates when
# the user runs `using GLM` before `using BayesInteractomics`. We verify the
# trigger via subprocess so the testitem result is independent of any other
# testitem in the same TestItemRunner session.
#
# Falls back to a graceful @info+@test true skip when GLM is not discoverable
# in the active project (e.g., direct @run_package_tests outside the Pkg.test
# sandbox). Under Pkg.test() the [extras]+[targets] entries make GLM available.

@testitem "With using GLM, fit_dropout_curves resolves to extension method (subprocess)" begin
    using Test
    proj = Base.active_project()
    has_glm = Base.find_package("GLM") !== nothing
    if !has_glm
        @info "Skipping imputation-loaded test: GLM not discoverable in active project ($proj). Run via Pkg.test() to exercise the full check."
        @test true   # placeholder pass
    else
        script = """
            using GLM
            using BayesInteractomics
            n = length(methods(BayesInteractomics.fit_dropout_curves))
            m = length(methods(BayesInteractomics.impute_mnar))
            ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsImputationExt)
            println("FIT_METHODS=", n)
            println("IMPUTE_METHODS=", m)
            println("EXT_LOADED=", ext !== nothing)
        """
        cmd = `julia --project=$proj --threads=4 -e $script`
        out = read(cmd, String)
        @test occursin("EXT_LOADED=true", out)
        @test occursin(r"FIT_METHODS=[1-9]", out)
        @test occursin(r"IMPUTE_METHODS=[1-9]", out)
    end
end

@testitem "DropoutFit struct is concrete and has 8 fields (always-available)" begin
    using BayesInteractomics
    using Test

    # The DropoutFit struct lives in src/data/imputation_stubs.jl (always loaded)
    # so the type itself is constructible and round-trips through JLD2 even when
    # the imputation extension is NOT loaded. The constructor producer
    # `fit_dropout_curves` lives in the extension and requires `using GLM`.
    @test isconcretetype(BayesInteractomics.DropoutFit)
    @test fieldcount(BayesInteractomics.DropoutFit) == 8
    # Confirm field names match the DropoutFit schema (promotion preserved layout).
    expected_fields = (:rho, :zeta, :column_names, :n_proteins,
                       :n_detections_per_column, :fit_timestamp,
                       :software_version, :dataset_hash)
    @test fieldnames(BayesInteractomics.DropoutFit) == expected_fields
end
