"""
    test_diagnostic_plots.jl

Tests for H0 marginal diagnostic plot, BMA weights diagnostic plot,
and 3-component diagnostic visualizations.
"""

# NOTE: the "H0 marginal diagnostic plot" and "H1 marginal diagnostic plot" testitems were
# removed here — the h0_marginal_diagnostic_plot / h1_marginal_diagnostic_plot functions were
# deleted in commit 6f54c769 ("remove deprecated SVG marginal diagnostic functions"). The tests
# referenced removed symbols; they are dropped rather than realigned (no replacement exists).

@testitem "BMA weights plot smoke test" begin
    using BayesInteractomics

    # Verify the function exists and is exported
    @test isdefined(BayesInteractomics, :bma_weights_plot)
    @test bma_weights_plot isa Function
end

@testitem "Component assignment plot" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult, component_assignment_plot
    using Distributions, Random, Test

    Random.seed!(42)
    n = 300

    bf_e = [exp.(randn(100) .- 2.0); exp.(randn(100)); exp.(randn(100) .+ 3.0)]
    bf_c = [exp.(randn(100) .- 2.0); exp.(randn(100)); exp.(randn(100) .+ 3.0)]
    bf_d = [exp.(randn(100) .- 2.0); exp.(randn(100)); exp.(randn(100) .+ 3.0)]
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    resp = zeros(n, 3)
    resp[1:100, 1] .= 0.9; resp[1:100, 2] .= 0.05; resp[1:100, 3] .= 0.05
    resp[101:200, 2] .= 0.9; resp[101:200, 1] .= 0.05; resp[101:200, 3] .= 0.05
    resp[201:300, 3] .= 0.9; resp[201:300, 1] .= 0.05; resp[201:300, 2] .= 0.05

    class_params = Dict(
        "background" => (mu=-2.0, sigma=1.0, precision=1.0),
        "agnostic"   => (mu=0.0, sigma=0.5, precision=4.0),
        "interaction" => (mu=3.0, sigma=0.5, precision=4.0)
    )
    lc = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [0.33, 0.33, 0.34], collect(1.0:10.0), true, 10, resp
    )

    plt = component_assignment_plot(bf, lc)
    @test plt !== nothing

    # Test file output
    tmpfile = tempname() * ".png"
    plt2 = component_assignment_plot(bf, lc; file=tmpfile)
    @test isfile(tmpfile)
    rm(tmpfile, force=true)

    # Test with nothing responsibilities
    lc_no_resp = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [0.33, 0.33, 0.34], collect(1.0:10.0), true, 10
    )
    plt3 = component_assignment_plot(bf, lc_no_resp)
    @test plt3 !== nothing
end

# NOTE: the "Marginal fit overlay plot" testitem was removed here — marginal_fit_overlay_plot was
# deleted in commit 6f54c769 ("remove deprecated SVG marginal diagnostic functions"). No replacement
# exists, so the orphaned test is dropped rather than realigned.

@testitem "EM convergence plot" begin
    using BayesInteractomics
    using BayesInteractomics: LatentClassResult, em_convergence_plot
    using Test

    # Create LatentClassResult with synthetic convergence trace
    n = 100
    free_energy = cumsum(abs.(randn(20)))  # 20 iterations, monotonically increasing
    class_params = Dict(
        "background" => (mu=-1.0, sigma=1.0, precision=1.0),
        "agnostic"   => (mu=0.0, sigma=0.5, precision=4.0),
        "interaction" => (mu=2.0, sigma=0.5, precision=4.0)
    )
    lc = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [0.5, 0.2, 0.3], free_energy, true, 20, nothing
    )

    plt = em_convergence_plot(lc)
    @test plt !== nothing

    # Test file output
    tmpfile = tempname() * ".png"
    plt2 = em_convergence_plot(lc; file=tmpfile)
    @test isfile(tmpfile)
    rm(tmpfile, force=true)

    # Edge case: empty free_energy
    lc_empty = LatentClassResult(
        ones(10), fill(0.5, 10), class_params,
        [0.5, 0.2, 0.3], Float64[], false, 0, nothing
    )
    plt3 = em_convergence_plot(lc_empty)
    @test plt3 !== nothing
end
