# Regression test for the HierarchicalBayesianModelSingle nparameters fix
# (src/inference/models.jl:193).
#
# The single-protocol HBM allocates one experiment-level σ/μ node per experiment
# (+ a global node), and the likelihood indexes μ_*[experiment+1] over
# experiments = size(samples, 1). nparameters used to be size(samples, 2) + 1
# (the REPLICATE count), which only coincided with the experiment count when the
# data was square (e.g. HAP40_Strep 3×3). On the single-experiment HAP40 mutants:
#   - experiments > replicates  → σ_*[experiment+1] under-allocated → BoundsError
#   - replicates > experiments  → extra σ_* nodes unused → RxInfer half-edge error
# The fix sets nparameters = size(samples, 1) + 1 (the experiment count).
#
# The failure is a graph-construction error driven purely by array SHAPE, so an
# all-missing prior infer (the exact precompute_HBM_single_protocol_prior path)
# reproduces it deterministically without depending on data values.

using TestItemRunner

@testitem "HierarchicalBayesianModelSingle: nparameters tracks experiments, not replicates" begin
    using BayesInteractomics
    using RxInfer

    constraints = @constraints begin
        q(μ_control, σ_control) = q(μ_control)q(σ_control)
        q(μ_sample, σ_sample) = q(μ_sample)q(σ_sample)
    end
    init = @initialization begin
        q(μ_control) = vague(NormalMeanPrecision)
        q(σ_control) = vague(GammaShapeRate)
        q(μ_sample) = vague(NormalMeanPrecision)
        q(σ_sample) = vague(GammaShapeRate)
    end

    # (n_experiments, n_replicates). Data is passed as experiments × replicates.
    #   (3, 2): experiments > replicates  → pre-fix BoundsError
    #   (1, 3): replicates > experiments  → pre-fix half-edge (the single-experiment mutant)
    #   (2, 2): square control case (worked pre- and post-fix)
    for (n_exp, n_rep) in [(3, 2), (1, 3), (2, 2)]
        data = fill(missing, n_exp, n_rep)
        res = infer(
            model = BayesInteractomics.HierarchicalBayesianModelSingle(μ = 25.0, σ = 1.0, a = 1.0, b = 1.0),
            data = (samples = data, controls = data),
            initialization = init,
            constraints = constraints,
            iterations = 20,
            returnvars = KeepLast(),
        )
        @test res isa RxInfer.InferenceResult
        # One experiment-level posterior per experiment (+ global node) ⇒ n_exp + 1.
        @test length(res.posteriors[:σ_sample]) == n_exp + 1
        @test length(res.posteriors[:μ_control]) == n_exp + 1
    end
end
