@testitem "DiscreteEmpirical constructor" begin
    import BayesInteractomics: DiscreteEmpirical

    # Basic construction from raw values (uniform weights)
    raw = [1.0, 2.0, 3.0, 2.0, 1.0]
    d = DiscreteEmpirical(raw)

    @test d.values == [1.0, 2.0, 3.0]
    @test length(d.probs) == 3
    @test sum(d.probs) ≈ 1.0 atol=1e-12
    @test d.probs[1] ≈ 2/5
    @test d.probs[2] ≈ 2/5
    @test d.probs[3] ≈ 1/5
    @test issorted(d.values)

    # Lookup dict correctness
    @test d.lookup[1.0] ≈ 2/5
    @test d.lookup[2.0] ≈ 2/5
    @test d.lookup[3.0] ≈ 1/5

    # Single unique value
    d2 = DiscreteEmpirical([5.0, 5.0, 5.0])
    @test d2.values == [5.0]
    @test d2.probs ≈ [1.0]
end

@testitem "DiscreteEmpirical pdf" begin
    import BayesInteractomics: DiscreteEmpirical
    import Distributions: pdf

    raw = [1.0, 2.0, 3.0, 2.0, 1.0]
    d = DiscreteEmpirical(raw)

    # Seen values return correct probability
    @test pdf(d, 1.0) ≈ 2/5
    @test pdf(d, 2.0) ≈ 2/5
    @test pdf(d, 3.0) ≈ 1/5

    # Unseen values return 0.0
    @test pdf(d, 0.0) == 0.0
    @test pdf(d, 4.0) == 0.0
    @test pdf(d, 1.5) == 0.0
    @test pdf(d, -100.0) == 0.0
end

@testitem "DiscreteEmpirical logpdf" begin
    import BayesInteractomics: DiscreteEmpirical
    import Distributions: logpdf

    raw = [1.0, 2.0, 3.0, 2.0, 1.0]
    d = DiscreteEmpirical(raw)

    # Seen values return log of correct probability
    @test logpdf(d, 1.0) ≈ log(2/5)
    @test logpdf(d, 2.0) ≈ log(2/5)
    @test logpdf(d, 3.0) ≈ log(1/5)

    # Unseen values return floor at log(1e-300)
    @test logpdf(d, 0.0) == log(1e-300)
    @test logpdf(d, 4.0) == log(1e-300)
    @test logpdf(d, 1.5) == log(1e-300)
end

@testitem "DiscreteEmpirical cdf" begin
    import BayesInteractomics: DiscreteEmpirical
    import Distributions: cdf

    raw = [1.0, 2.0, 3.0, 2.0, 1.0]
    d = DiscreteEmpirical(raw)

    # Boundary values
    @test cdf(d, 0.9) == 0.0   # below minimum
    @test cdf(d, 3.0) ≈ 1.0   # at maximum

    # Step function behavior
    @test cdf(d, 1.0) ≈ 2/5
    @test cdf(d, 1.5) ≈ 2/5   # between 1.0 and 2.0
    @test cdf(d, 2.0) ≈ 4/5
    @test cdf(d, 2.5) ≈ 4/5   # between 2.0 and 3.0

    # Monotonicity: cdf is non-decreasing
    xs = range(0.0, 4.0, length=100)
    cdfs = [cdf(d, x) for x in xs]
    @test all(diff(cdfs) .>= 0.0)

    # cdf just below minimum is 0
    @test cdf(d, 1.0 - eps()) == 0.0 || cdf(d, 0.5) == 0.0
end

@testitem "DiscreteEmpirical rand" begin
    import BayesInteractomics: DiscreteEmpirical
    import Random: MersenneTwister

    raw = [1.0, 2.0, 3.0, 2.0, 1.0]
    d = DiscreteEmpirical(raw)

    rng = MersenneTwister(42)
    samples = [rand(rng, d) for _ in 1:1000]

    # All samples must be in support
    @test all(s in d.values for s in samples)

    # Rough frequency check (should be close to 2/5, 2/5, 1/5)
    @test count(==(1.0), samples) / 1000 ≈ 2/5 atol=0.05
    @test count(==(2.0), samples) / 1000 ≈ 2/5 atol=0.05
    @test count(==(3.0), samples) / 1000 ≈ 1/5 atol=0.05
end

@testitem "DiscreteEmpirical insupport and bounds" begin
    import BayesInteractomics: DiscreteEmpirical
    import Distributions: insupport, minimum, maximum

    raw = [1.0, 2.0, 3.0, 2.0, 1.0]
    d = DiscreteEmpirical(raw)

    @test insupport(d, 1.0) == true
    @test insupport(d, 2.0) == true
    @test insupport(d, 3.0) == true
    @test insupport(d, 0.0) == false
    @test insupport(d, 4.0) == false
    @test insupport(d, 1.5) == false

    @test minimum(d) == 1.0
    @test maximum(d) == 3.0
end

@testitem "DiscreteEmpirical weighted constructor" begin
    import BayesInteractomics: DiscreteEmpirical
    import Distributions: pdf

    # Weighted construction
    values = [1.0, 2.0, 3.0]
    weights = [2.0, 3.0, 1.0]
    d = DiscreteEmpirical(values, weights)

    total = sum(weights)
    @test pdf(d, 1.0) ≈ 2.0 / total
    @test pdf(d, 2.0) ≈ 3.0 / total
    @test pdf(d, 3.0) ≈ 1.0 / total
    @test sum(d.probs) ≈ 1.0 atol=1e-12

    # Empty-weight guard: sum(weights) < 1e-10 returns stub
    d_stub = DiscreteEmpirical([1.0, 2.0], [0.0, 0.0])
    @test d_stub.values == [0.0]
    @test d_stub.probs == [1.0]
    @test pdf(d_stub, 0.0) == 1.0

    # Tiny weights (near zero but > 1e-10) still work
    d_tiny = DiscreteEmpirical([1.0, 2.0], [1e-11, 1e-11])
    # sum = 2e-11 < 1e-10: should be a stub
    @test d_tiny.values == [0.0]
    @test d_tiny.probs == [1.0]
end

@testitem "_fit_discrete_empirical_weighted" begin
    import BayesInteractomics: _fit_discrete_empirical_weighted
    import Distributions: pdf

    # Basic weighted fit
    values = [1.0, 2.0, 2.0, 3.0]
    weights = [1.0, 1.0, 1.0, 1.0]
    d = _fit_discrete_empirical_weighted(values, weights)

    @test d.values == [1.0, 2.0, 3.0]
    @test sum(d.probs) ≈ 1.0 atol=1e-12
    @test pdf(d, 1.0) ≈ 1/4
    @test pdf(d, 2.0) ≈ 2/4
    @test pdf(d, 3.0) ≈ 1/4

    # Non-uniform weights with grouping
    values2 = [1.0, 1.0, 2.0]
    weights2 = [3.0, 1.0, 2.0]
    d2 = _fit_discrete_empirical_weighted(values2, weights2)
    total = 6.0
    @test pdf(d2, 1.0) ≈ 4.0 / total
    @test pdf(d2, 2.0) ≈ 2.0 / total

    # Empty guard via helper
    d_empty = _fit_discrete_empirical_weighted([1.0, 2.0], [0.0, 0.0])
    @test d_empty.values == [0.0]
    @test d_empty.probs == [1.0]
end

@testitem "DiscreteEmpirical SQUAREM preservation" begin
    import BayesInteractomics: DiscreteEmpirical, _replace_detection_marginal
    import Distributions: pdf, Normal
    using Copulas: SklarDist, ClaytonCopula

    # Build a known DiscreteEmpirical with 3 support points
    disc = DiscreteEmpirical([0.5, 1.0, 2.0], [0.3, 0.5, 0.2])

    # Build a minimal 3D SklarDist with Normal marginals (including detection in position 3)
    cop = ClaytonCopula(3, 1.0)
    marg_e = Normal(0.0, 1.0)
    marg_c = Normal(0.0, 1.0)
    marg_d_normal = Normal(1.0, 0.5)  # placeholder that will be replaced
    joint = SklarDist(cop, (marg_e, marg_c, marg_d_normal))

    # Replace the detection marginal (position 3) with DiscreteEmpirical
    joint_new = _replace_detection_marginal(joint, disc)

    # Copula and first two marginals must be preserved
    @test joint_new.C === joint.C
    @test joint_new.m[1] === marg_e
    @test joint_new.m[2] === marg_c

    # Third marginal must be the DiscreteEmpirical
    result_disc = joint_new.m[3]
    @test result_disc isa DiscreteEmpirical

    # Probabilities for support points must match original
    @test pdf(result_disc, 0.5) ≈ 0.3
    @test pdf(result_disc, 1.0) ≈ 0.5
    @test pdf(result_disc, 2.0) ≈ 0.2

    # Unseen value must return 0.0 (discrete support is preserved exactly)
    @test pdf(result_disc, 999.0) == 0.0
    @test pdf(result_disc, 0.0) == 0.0

    # Round-trip: replacing twice with different distributions uses the latest one
    disc2 = DiscreteEmpirical([3.0, 4.0], [0.6, 0.4])
    joint_new2 = _replace_detection_marginal(joint_new, disc2)
    @test pdf(joint_new2.m[3], 3.0) ≈ 0.6
    @test pdf(joint_new2.m[3], 0.5) == 0.0  # old support gone
end
