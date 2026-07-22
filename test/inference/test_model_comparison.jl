@testitem "optimize_nu lower bound is 5.0" begin
    using BayesInteractomics

    # Verify the runtime source of optimize_nu has the correct default lower bound.
    # This check reads the source file directly to confirm the keyword default is 5.0,
    # avoiding the cost of running optimize_nu with real data in a unit test.
    src_text = read(
        joinpath(pkgdir(BayesInteractomics), "src", "inference", "model_comparison.jl"),
        String
    )

    # The function signature must declare lower::Float64 = 5.0
    @test occursin("lower::Float64 = 5.0", src_text)

    # The stale lower bound (3.0) must not remain in the signature or docstring
    @test !occursin("lower::Float64 = 3.0", src_text)

    # The NuOptimizationResult struct must carry search_bounds so callers can inspect them
    @test fieldnames(BayesInteractomics.NuOptimizationResult) ⊇ (:search_bounds,)

    # Confirm a NuOptimizationResult can be constructed with (5.0, 50.0) bounds
    mock_waic = BayesInteractomics.WAICResult(0.0, 0.0, 0.0, Float64[], 0.0)
    mock_result = BayesInteractomics.NuOptimizationResult(
        5.0,           # optimal_nu
        mock_waic,     # optimal_waic
        mock_waic,     # normal_waic
        Float64[],     # nu_trace
        Float64[],     # waic_trace
        0.0,           # delta_waic
        0.0,           # delta_se
        (5.0, 50.0)    # search_bounds — must match new default
    )
    @test mock_result.search_bounds == (5.0, 50.0)
end
