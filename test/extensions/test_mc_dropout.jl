# Stubs for `mc_dropout_batch` round-trip + variance scaling.
#
# These @testitem blocks are RED baseline scaffolding. They MUST fail with
# explicit "NOT IMPLEMENTED YET" assertions. Landing
# `BayesInteractomicsMetalearnerExt.mc_dropout_batch` per the reference
# verbatim port flips these blocks green.
#
# Covered validation requirements:
#   - V-01  : mc_dropout_batch returns a NamedTuple with finite samples::Matrix{Float32}
#             of shape (K, n_pairs); baseline mean restored after K passes
#   - V-02  : K-sample variance scales monotonically with dropout rate on the mock
#             2-layer model with known seed-1 weights
#   - V-01b : Pitfall 7 — baseline pass restores `testmode!` on the model after MC

@testitem "V-01: mc_dropout_batch returns NamedTuple with correct shape" begin
    using BayesInteractomics, Test, Random, Flux, MLJ, MLJScikitLearnInterface, HDF5, DataFrames, Statistics
    include(joinpath(@__DIR__, "..", "fixtures", "dnn_prior_minimal.jl"))

    # The extension method `BayesInteractomicsMetalearnerExt.mc_dropout_batch`
    # asserts the return-shape contract (Pattern 2 / Pitfall 2):
    #   out::NamedTuple = (samples::Matrix{Float32}(K, n_pairs), mean, var, baseline)
    # with mean ≈ mean(samples; dims=1) and var Bessel-corrected.
    model = build_mock_dropout_model(0.3)
    X = build_mock_embedding_matrix(n_pairs=5, feature_dim=8)
    @test isa(model, Flux.Chain)
    @test size(X) == (8, 5)

    ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)
    out = ext.mc_dropout_batch(model, X; K=30, rng=MersenneTwister(42))
    @test size(out.samples) == (30, 5)
    @test eltype(out.samples) == Float32
    @test all(out.var .>= 0)
    @test length(out.baseline) == 5
end

@testitem "V-02: K-sample variance scales monotonically with dropout rate" begin
    using BayesInteractomics, Test, Random, Flux, MLJ, MLJScikitLearnInterface, HDF5, DataFrames, Statistics
    include(joinpath(@__DIR__, "..", "fixtures", "dnn_prior_minimal.jl"))

    # `mean(out_hi.var) > mean(out_lo.var)` holds on the mock
    # 2-layer model — higher dropout rate → larger MC predictive variance.
    model_lo = build_mock_dropout_model(0.1)
    model_hi = build_mock_dropout_model(0.5)
    X = build_mock_embedding_matrix(n_pairs=5, feature_dim=8)
    @test isa(model_lo, Flux.Chain)
    @test isa(model_hi, Flux.Chain)

    ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)
    out_lo = ext.mc_dropout_batch(model_lo, X; K=30, rng=MersenneTwister(42))
    out_hi = ext.mc_dropout_batch(model_hi, X; K=30, rng=MersenneTwister(42))
    @test mean(out_hi.var) > mean(out_lo.var)
end

@testitem "V-01b: mc_dropout_batch baseline pass restores testmode! (Pitfall 7)" begin
    using BayesInteractomics, Test, Random, Flux, MLJ, MLJScikitLearnInterface, HDF5, DataFrames, Statistics
    include(joinpath(@__DIR__, "..", "fixtures", "dnn_prior_minimal.jl"))

    # After `mc_dropout_batch` returns, the model MUST be in
    # `testmode!` (Pitfall 7). The implementation calls
    # `Flux.testmode!(model)` immediately before the baseline pass; this stub
    # locks the post-condition.
    model = build_mock_dropout_model(0.3)
    X = build_mock_embedding_matrix(n_pairs=5, feature_dim=8)
    @test isa(model, Flux.Chain)

    ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)
    _ = ext.mc_dropout_batch(model, X; K=30, rng=MersenneTwister(42))
    # After mc_dropout_batch returns, the model is in testmode! — two
    # successive deterministic passes must produce identical outputs.
    y1 = model(X)
    y2 = model(X)
    @test y1 ≈ y2
end
