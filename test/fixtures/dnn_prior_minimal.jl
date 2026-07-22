"""Shared test fixture. Plain-include file (no module). Provides mock model + embedding + DataFrame builders for the MC-Dropout test stubs. Helpers are deterministic by seed for reproducibility tests."""

using Random
using DataFrames
using Flux

"""
    build_mock_dropout_model(p::Float64; in_dim=8, hidden=4)

Build a deterministic 2-layer `Flux.Chain` with a `Dropout(p)` layer in the
middle, suitable for MC-Dropout round-trip stubs (V-01, V-02). Weights are
seeded via `Random.seed!(1)` immediately before `Dense(...)` construction so
that for any given `p` the returned Chain has identical weights every call.

The architecture is `Dense(in_dim => hidden, relu) → Dropout(p) → Dense(hidden => 1) → σ`.
"""
function build_mock_dropout_model(p::Float64; in_dim::Int=8, hidden::Int=4)
    Random.seed!(1)
    d1 = Dense(in_dim => hidden, relu)
    Random.seed!(1)
    d2 = Dense(hidden => 1)
    return Chain(d1, Dropout(p), d2, σ)
end

"""
    build_mock_embedding_matrix(; n_pairs=5, feature_dim=8, seed=42)

Returns a `Matrix{Float32}` of shape `(feature_dim, n_pairs)` constructed from
`MersenneTwister(seed)`. Matches the `(features, batch)` shape that
`mc_dropout_batch` expects per RESEARCH.md Pitfall 2.
"""
function build_mock_embedding_matrix(; n_pairs::Int=5, feature_dim::Int=8, seed::Int=42)
    rng = MersenneTwister(seed)
    return Float32.(randn(rng, feature_dim, n_pairs))
end

"""
    build_mock_results_df(n::Int=5; posterior_prob=fill(0.5, n))

Returns a `DataFrames.DataFrame` with columns `Protein::Vector{String}`
(`["P1","P2",...]`), `posterior_prob`, `BF::Vector{Float64} = fill(1.0, n)`.
This is the input shape `compute_mc_prior!` mutates.
"""
function build_mock_results_df(n::Int=5; posterior_prob::Vector{Float64}=fill(0.5, n))
    return DataFrame(
        Protein = ["P$i" for i in 1:n],
        posterior_prob = posterior_prob,
        BF = fill(1.0, n),
    )
end

"""
    build_two_condition_different_bait_fixture()

Returns NamedTuple `(X_A::Matrix{Float32}, X_B::Matrix{Float32}, shared_prey_idx::Int)`
where `X_A[:, shared_prey_idx]` and `X_B[:, shared_prey_idx]` are DIFFERENT
vectors (simulating the different-bait per-condition contract).
Use deterministic seeds (`MersenneTwister(101)` for A, `MersenneTwister(202)`
for B); `shared_prey_idx = 3`. Fixture is the synthetic different-bait contract —
different baits produce different embeddings even for the same prey.
"""
function build_two_condition_different_bait_fixture()
    X_A = Float32.(randn(MersenneTwister(101), 8, 5))
    X_B = Float32.(randn(MersenneTwister(202), 8, 5))
    return (X_A=X_A, X_B=X_B, shared_prey_idx=3)
end
