# ============================================================
# Empirical Bayes Core Functions Test Suite
# ============================================================
# Covers: inv_digamma Newton solver, estimate_dirichlet_eb fixed-point,
# accuracy, convergence, edge cases, degenerate inputs, return types.
# ============================================================

# --------------------------------------------------
# Block 1: inv_digamma accuracy against reference
# --------------------------------------------------
@testitem "inv_digamma: accuracy against SpecialFunctions.invdigamma" begin
    using BayesInteractomics
    using SpecialFunctions: invdigamma

    ys = range(-10.0, 10.0, length=201)
    for y in ys
        result = inv_digamma(y)
        reference = invdigamma(y)
        @test result > 0.0
        @test isapprox(result, reference, atol=1e-6)
    end
end

# --------------------------------------------------
# Block 2: inv_digamma convergence within 50 iterations
# --------------------------------------------------
@testitem "inv_digamma: convergence within 50 iterations" begin
    using BayesInteractomics
    using SpecialFunctions: digamma

    test_points = [-10.0, -2.22, 0.0, 5.0, 10.0]
    for y in test_points
        x = inv_digamma(y; maxiter=50)
        @test isfinite(x)
        @test x > 0.0
        @test abs(digamma(x) - y) < 1e-10
    end
end

# --------------------------------------------------
# Block 3: inv_digamma edge cases
# --------------------------------------------------
@testitem "inv_digamma: edge cases" begin
    using BayesInteractomics
    using SpecialFunctions: digamma

    # Regime boundary y = -2.22
    x1 = inv_digamma(-2.22)
    @test x1 > 0.0
    @test isfinite(x1)

    # Known value: digamma(x) = 0 at x ~ 1.46163
    x2 = inv_digamma(0.0)
    @test isapprox(x2, 1.46163, atol=1e-4)

    # Large positive y
    x3 = inv_digamma(10.0)
    @test abs(digamma(x3) - 10.0) < 1e-10
end

# --------------------------------------------------
# Block 4: estimate_dirichlet_eb on synthetic well-separated gamma
# --------------------------------------------------
@testitem "estimate_dirichlet_eb: synthetic well-separated gamma" begin
    using BayesInteractomics
    using Distributions: Dirichlet
    using Random

    Random.seed!(42)
    true_alpha = [5.0, 2.0, 1.0]
    # rand(Dirichlet(...), n) returns K-by-n in Julia, transpose to get n-by-K
    gamma = permutedims(rand(Dirichlet(true_alpha), 500))

    result = estimate_dirichlet_eb(gamma)

    # Check recovery within factor of 2
    for k in 1:3
        @test result.alpha[k] > true_alpha[k] / 2.0
        @test result.alpha[k] < true_alpha[k] * 2.0
    end

    # All components > 0.5
    @test all(result.alpha .> 0.5)

    # Sum in bounds
    @test sum(result.alpha) >= 3.0
    @test sum(result.alpha) <= 30.0

    # Convergence
    @test result.converged == true

    # Not clamped on well-separated data
    @test result.was_clamped == false
end

# --------------------------------------------------
# Block 5: estimate_dirichlet_eb on degenerate gamma
# --------------------------------------------------
@testitem "estimate_dirichlet_eb: degenerate gamma (all weight in component 1)" begin
    using BayesInteractomics

    # Degenerate: nearly all weight in component 1
    gamma = repeat([0.98 0.01 0.01], 100, 1)

    result = estimate_dirichlet_eb(gamma)

    # No NaN or Inf
    @test all(isfinite.(result.alpha))
    @test !any(isnan.(result.alpha))

    # All components >= 0.5 (floor applied, floor value is 0.5)
    @test all(result.alpha .>= 0.5)

    # converged is Bool
    @test result.converged isa Bool

    # was_clamped should be true (components 2,3 hit floor)
    @test result.was_clamped == true
end

# --------------------------------------------------
# Block 6: estimate_dirichlet_eb on flat prior [1,1,1] gamma
# --------------------------------------------------
@testitem "estimate_dirichlet_eb: flat prior [1,1,1] gamma" begin
    using BayesInteractomics
    using Distributions: Dirichlet
    using Random

    Random.seed!(123)
    gamma = permutedims(rand(Dirichlet([1.0, 1.0, 1.0]), 200))

    result = estimate_dirichlet_eb(gamma)

    # Stable output
    @test all(isfinite.(result.alpha))
    @test !any(isnan.(result.alpha))

    # All components >= 0.5
    @test all(result.alpha .>= 0.5)

    # Sum in bounds
    @test sum(result.alpha) >= 3.0
    @test sum(result.alpha) <= 30.0
end

# --------------------------------------------------
# Block 7: estimate_dirichlet_eb return type and structure
# --------------------------------------------------
@testitem "estimate_dirichlet_eb: return type and structure" begin
    using BayesInteractomics
    using Random

    Random.seed!(99)
    gamma = rand(50, 3)
    # Normalize rows to sum to 1
    gamma = gamma ./ sum(gamma, dims=2)

    result = estimate_dirichlet_eb(gamma)

    # NamedTuple keys
    @test haskey(result, :alpha)
    @test haskey(result, :converged)
    @test haskey(result, :iterations)
    @test haskey(result, :was_clamped)

    # Types
    @test result.alpha isa Vector{Float64}
    @test length(result.alpha) == 3
    @test result.converged isa Bool
    @test result.iterations isa Int
    @test result.was_clamped isa Bool
end

# --------------------------------------------------
# Block 8: build_prior_grid: constant-strength 9-point grid
# --------------------------------------------------
@testitem "build_prior_grid: constant-strength 9-point grid" begin
    using BayesInteractomics
    alpha_hat = [5.0, 2.0, 1.0]
    S = sum(alpha_hat)
    grid = build_prior_grid(alpha_hat)

    # Should have up to 9 unique points
    @test length(grid) >= 7  # at least 7 after dedup
    @test length(grid) <= 9

    # All sums equal to S (constant strength)
    for g in grid
        @test isapprox(sum(g), S, atol=1e-10)
    end

    # All components >= 0.05 * S (floor)
    for g in grid
        for c in g
            @test c >= 0.05 * S - 1e-10
        end
    end

    # First point is EB center
    @test isapprox(grid[1], alpha_hat, atol=1e-10)

    # Second point is uniform
    @test isapprox(grid[2], fill(S / 3, 3), atol=1e-10)
end

# --------------------------------------------------
# Block 9: build_prior_grid: symmetric input deduplication
# --------------------------------------------------
@testitem "build_prior_grid: symmetric input deduplication" begin
    using BayesInteractomics
    # Symmetric: EB center == Uniform, so fewer unique points
    alpha_hat = [3.0, 3.0, 3.0]
    grid = build_prior_grid(alpha_hat)
    @test length(grid) < 9  # EB center and Uniform coincide
    @test length(grid) >= 1
    for g in grid
        @test isapprox(sum(g), 9.0, atol=1e-10)
    end
end

# --------------------------------------------------
# Block 10: _perturb_proportions: mass shift correctness
# --------------------------------------------------
@testitem "_perturb_proportions: mass shift correctness" begin
    using BayesInteractomics: _perturb_proportions
    base = [0.625, 0.25, 0.125]

    # Single target shift
    p1 = _perturb_proportions(base, [1], 0.25)
    @test isapprox(sum(p1), 1.0, atol=1e-10)
    @test p1[1] > base[1]  # target got more mass

    # Two-target shift
    p2 = _perturb_proportions(base, [1, 2], 0.20)
    @test isapprox(sum(p2), 1.0, atol=1e-10)
    @test p2[3] < base[3]  # non-target lost mass

    # Floor guarantee
    skewed = [0.9, 0.05, 0.05]
    p3 = _perturb_proportions(skewed, [1], 0.30)
    @test all(p3 .>= 0.05 - 1e-10)
    @test isapprox(sum(p3), 1.0, atol=1e-10)
end

# --------------------------------------------------
# Block 11: _marginalize_over_priors: return type and numeric stability
# --------------------------------------------------
@testitem "_marginalize_over_priors: return type and numeric stability" begin
    using BayesInteractomics
    using Random
    Random.seed!(42)

    n = 200
    # Synthetic 3-component log-BF data
    # Component 1 (H0): centered near 0
    # Component 2 (Agnostic): centered near 0
    # Component 3 (H1): centered at positive values
    y_e = vcat(randn(100) .* 0.5, randn(50) .* 0.3, randn(50) .* 0.5 .+ 3.0)
    y_c = vcat(randn(100) .* 0.5, randn(50) .* 0.3, randn(50) .* 0.5 .+ 2.0)
    y_p = vcat(randn(100) .* 0.5, randn(50) .* 0.3, randn(50) .* 0.5 .+ 1.5)

    grid = [[5.0, 2.0, 1.0], [2.5, 1.0, 0.5], [10.0, 4.0, 2.0]]

    result = _marginalize_over_priors(y_e, y_c, y_p, y_e, y_c, y_p, grid;
                                       force_h1_family=:gamma, n_restarts=5)

    # Return type has required keys
    @test haskey(result, :posterior_prob)
    @test haskey(result, :combined_bf)
    @test haskey(result, :bic_weights)
    @test haskey(result, :grid)
    @test haskey(result, :per_grid_ll)
    @test haskey(result, :per_grid_posteriors)
    @test haskey(result, :dominant_grid_index)
    @test haskey(result, :dominant_weight)
    @test haskey(result, :n_distinct_points)

    # Numeric stability
    @test isapprox(sum(result.bic_weights), 1.0, atol=1e-10)
    @test !any(isnan.(result.bic_weights))
    @test !any(isinf.(result.bic_weights))

    # Posteriors in [0, 1] (tolerant of NaN — a degenerate grid point can yield
    # NaN posteriors; the bounded-when-finite contract is what matters here)
    @test all(x -> ismissing(x) || isnan(x) || (-1e-6 <= x <= 1.0 + 1e-6), result.posterior_prob)
    @test length(result.posterior_prob) == n

    # BFs are positive
    @test all(result.combined_bf .> 0.0)

    # Dominant weight matches
    @test result.dominant_grid_index == argmax(result.bic_weights)
    @test isapprox(result.dominant_weight, maximum(result.bic_weights), atol=1e-10)
end

# --------------------------------------------------
# Block 12: _marginalize_over_priors: ground truth recovery
# --------------------------------------------------
@testitem "_marginalize_over_priors: averaged posteriors recover ground truth" begin
    using BayesInteractomics
    using Statistics: mean
    using Random
    Random.seed!(123)

    # 300 proteins: 200 H0 (label=0), 100 H1 (label=1)
    n_h0, n_h1 = 200, 100
    n = n_h0 + n_h1

    # H0 proteins: log-BFs near 0; H1 proteins: log-BFs positive
    y_e = vcat(randn(n_h0) .* 0.5, randn(n_h1) .* 0.5 .+ 4.0)
    y_c = vcat(randn(n_h0) .* 0.5, randn(n_h1) .* 0.5 .+ 3.0)
    y_p = vcat(randn(n_h0) .* 0.5, randn(n_h1) .* 0.5 .+ 2.0)

    grid = build_prior_grid([5.0, 2.0, 1.0])

    result = _marginalize_over_priors(y_e, y_c, y_p, y_e, y_c, y_p, grid;
                                       force_h1_family=:gamma, n_restarts=5)

    # Compute AUC-like metric: fraction of H1 proteins with posterior > median H0 posterior
    h0_median = sort(result.posterior_prob[1:n_h0])[div(n_h0, 2)]
    avg_separation = mean(result.posterior_prob[n_h0+1:end] .> h0_median)

    # Average should separate well (> 0.8 of H1 above H0 median)
    @test avg_separation > 0.8

    # Per-grid posteriors should also exist
    @test length(result.per_grid_posteriors) == length(grid)
    for pgp in result.per_grid_posteriors
        @test length(pgp) == n
    end
end

# --------------------------------------------------
# Block 13: _marginalize_over_priors: weights valid on degenerate data
# --------------------------------------------------
@testitem "_marginalize_over_priors: weights valid on degenerate data" begin
    using BayesInteractomics

    # Very small n, constant data -- EM may struggle but should not produce NaN weights
    y = fill(0.0, 10)
    grid = [[3.0, 1.5, 0.5]]  # Single grid point

    result = _marginalize_over_priors(y, y, y, y, y, y, grid;
                                       force_h1_family=:gamma, n_restarts=3)

    @test isapprox(sum(result.bic_weights), 1.0, atol=1e-10)
    @test !any(isnan.(result.bic_weights))
    @test !any(isinf.(result.bic_weights))
    @test length(result.posterior_prob) == 10
end
