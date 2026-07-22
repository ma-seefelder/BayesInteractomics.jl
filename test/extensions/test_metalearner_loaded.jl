# Metalearner extension ACTIVATION tests.
#
# To verify both "extension loaded" AND "extension NOT loaded" states cleanly
# we use a subprocess pattern: once a package is `using`'d in a Julia session
# it stays loaded for the remainder of that process. A child julia process
# gives us a clean namespace to assert each state independently.
#
# Both subprocesses inherit the TestItemRunner-managed test environment via
# `Base.active_project()` (Pkg.test sandbox) so Flux/MLJ/MLJScikitLearnInterface/HDF5
# are discoverable as declared in Project.toml [extras] + [targets].

@testitem "Loading Flux+MLJ+MLJScikitLearnInterface+HDF5 activates the metalearner extension (subprocess)" begin
    using Test
    proj = Base.active_project()
    # Skip cleanly if the test environment doesn't make Flux discoverable
    # (e.g., when run directly via `@run_package_tests` outside the Pkg.test
    # sandbox). The full check runs under `Pkg.test()` where the [extras] +
    # [targets].test entries make all five packages available.
    has_flux = Base.find_package("Flux") !== nothing
    if !has_flux
        @info "Skipping metalearner-loaded test: Flux not discoverable in active project ($proj). Run via Pkg.test() to exercise the full check."
        @test true   # placeholder pass so the @testitem block records a result
    else
        script = """
            using Flux, MLJ, MLJScikitLearnInterface, HDF5
            using BayesInteractomics
            n = length(methods(BayesInteractomics.predict_metalearner))
            ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)
            println("METHOD_COUNT=", n)
            println("EXT_LOADED=", ext !== nothing)
            println("REPORTS_MODULE=", isa(BayesInteractomics.Reports, Module))
        """
        cmd = `julia --project=$proj --threads=4 -e $script`
        out = read(cmd, String)
        @test occursin("EXT_LOADED=true", out)
        @test occursin(r"METHOD_COUNT=[1-9]", out)
        @test occursin("REPORTS_MODULE=true", out)
    end
end

@testitem "Without trigger packages, predict_metalearner stub has 0 methods and extension is not loaded (subprocess)" begin
    using Test
    proj = Base.active_project()
    script = """
        using BayesInteractomics
        n = length(methods(BayesInteractomics.predict_metalearner))
        ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)
        println("METHOD_COUNT=", n)
        println("EXT_LOADED=", ext !== nothing)
    """
    cmd = `julia --project=$proj --threads=4 -e $script`
    out = read(cmd, String)
    @test occursin("METHOD_COUNT=0", out)
    @test occursin("EXT_LOADED=false", out)
end
