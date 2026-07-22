# MC-Dropout inference for ProteinPairTransformerModel.
#
# The `module MCDropoutPredict ... end` wrapper is stripped because the
# functions live directly inside the parent extension module
# `BayesInteractomicsMetalearnerExt`.
#
# Flux.testmode! disables BOTH Dropout and BatchNorm. For MC-Dropout we want
# BatchNorm in test mode (uses frozen running statistics) but Dropout active.
# The trick: testmode! first, then walk the model graph and selectively
# re-enable Dropout / AlphaDropout / the internal Dropout inside
# MultiHeadAttention.

using Flux
using Statistics
using Random

"""
    _collect_dropouts!(out, x, seen=IdSet())

Recursively descends through struct fields and tuple elements of `x`,
collecting any `Flux.Dropout`/`Flux.AlphaDropout` instances into `out`.

Flux's `fmap` only visits *leaves* (Float32 arrays etc.); the Dropout layers
themselves are intermediate nodes in the functor tree, so a leaf-only visitor
never sees them. This walker visits every node and stops at the layers we
care about.
"""
function _collect_dropouts!(out, x, seen=Base.IdSet{Any}())
    x in seen && return
    push!(seen, x)
    if x isa Flux.Dropout || x isa Flux.AlphaDropout
        push!(out, x)
        return
    end
    if x isa Tuple || x isa NamedTuple
        for c in x
            _collect_dropouts!(out, c, seen)
        end
        return
    end
    if x isa AbstractArray || x isa Number || x isa Symbol || x isa Function ||
       x isa Nothing || x isa AbstractString
        return
    end
    if isstructtype(typeof(x))
        for fname in fieldnames(typeof(x))
            isdefined(x, fname) || continue
            v = getfield(x, fname)
            _collect_dropouts!(out, v, seen)
        end
    end
    return
end

"""
    set_mc_dropout_mode!(model; verbose::Bool=false)

Put `model` into MC-Dropout inference mode: BatchNorm uses running stats
(test mode), but every Dropout-flavoured layer has `.active = true`.

Returns `model` (mutated in place). When `verbose=true`, logs how many
Dropout layers were flipped.
"""
function set_mc_dropout_mode!(model; verbose::Bool=false)
    Flux.testmode!(model)
    dropouts = []
    _collect_dropouts!(dropouts, model)
    n_flipped = 0
    for layer in dropouts
        try
            layer.active = true
            n_flipped += 1
        catch e
            verbose && @warn "Could not mutate .active on dropout layer" layer exception=e
        end
    end
    if verbose
        @info "MC-Dropout mode set" n_dropout_layers=length(dropouts) n_flipped
        n_off = sum(d -> d.active !== true, dropouts; init=0)
        n_off > 0 && @warn "$n_off Dropout layer(s) did not flip to active=true"
    end
    return model
end

"""
    mc_dropout_predict(model, x::AbstractMatrix; K::Int=30, rng=Random.default_rng())

Run `K` forward passes through `model` with Dropout active, on input `x`
shaped `(features, batch)`. Returns a `K × batch` matrix of sigmoid outputs
(one row per MC sample).

Caller is responsible for putting model into MC-Dropout mode first via
`set_mc_dropout_mode!`. This is intentional: the function does not flip mode
each call, because resetting mode mid-batch would defeat the purpose of
stable stochastic inference.
"""
function mc_dropout_predict(model, x::AbstractMatrix; K::Int=30,
                            rng::AbstractRNG=Random.default_rng())
    n_batch = size(x, 2)
    samples = Matrix{Float32}(undef, K, n_batch)
    for k in 1:K
        # Flux's Dropout layer pulls randomness from its `rng` field (default
        # `TaskLocalRNG()`). Reseed BOTH the passed `rng` (for any downstream
        # consumers) AND the task-local RNG (the one Flux actually consumes)
        # using the same deterministic per-K hash. Without the task-local
        # reseed, two consecutive calls with identical `MersenneTwister(s)`
        # would still produce different samples — a determinism-contract violation.
        # The passed-rng reseed retains the established behaviour for
        # any caller chaining additional randomness off `rng`.
        seed_k = hash((k, n_batch))
        Random.seed!(rng, seed_k)
        Random.seed!(seed_k)
        y = model(x)  # output shape: (1, batch)
        samples[k, :] = vec(y)
    end
    return samples
end

"""
    mc_dropout_batch(model, X::AbstractMatrix; K::Int=30, batch_size::Int=256,
                     rng=Random.default_rng())

Run MC-Dropout inference over a large input matrix `X` shape `(features, n)`
in mini-batches. Returns a NamedTuple with:

- `samples::Matrix{Float32}` — K × n full sample matrix (memory: K * n * 4 B)
- `mean::Vector{Float32}`    — posterior mean per pair, length n
- `var::Vector{Float32}`     — posterior variance per pair, length n
- `baseline::Vector{Float32}` — single-shot deterministic prediction
  (testmode, all dropout off) for direct comparison

The baseline pass runs once after the MC samples and restores test mode for
the deterministic prediction.
"""
function mc_dropout_batch(model, X::AbstractMatrix; K::Int=30,
                          batch_size::Int=256,
                          rng::AbstractRNG=Random.default_rng())
    n_total = size(X, 2)
    samples = Matrix{Float32}(undef, K, n_total)

    # 1. MC-Dropout passes
    set_mc_dropout_mode!(model; verbose=true)
    for start_idx in 1:batch_size:n_total
        end_idx = min(start_idx + batch_size - 1, n_total)
        x_batch = X[:, start_idx:end_idx]
        s = mc_dropout_predict(model, x_batch; K=K, rng=rng)
        samples[:, start_idx:end_idx] = s
    end

    # 2. Deterministic baseline pass
    Flux.testmode!(model)
    baseline = Vector{Float32}(undef, n_total)
    for start_idx in 1:batch_size:n_total
        end_idx = min(start_idx + batch_size - 1, n_total)
        x_batch = X[:, start_idx:end_idx]
        baseline[start_idx:end_idx] = vec(model(x_batch))
    end

    mean_vec = vec(mean(samples; dims=1))
    var_vec = vec(var(samples; dims=1, corrected=true))

    return (samples=samples, mean=mean_vec, var=var_vec, baseline=baseline)
end

# ---------------------------------------------------------------------------- #
# compute_mc_prior! extension entry point
# ---------------------------------------------------------------------------- #
#
# Extends the parent stub `BayesInteractomics.compute_mc_prior!` declared in
# `src/ml/metalearner_stubs.jl`. Loads model-473 fresh (mirroring the
# `predict_DNN` load pattern from `metalearner.jl`), runs MC-Dropout via
# `mc_dropout_batch`, computes empirical 95% CI per pair (empirical quantile,
# not a Beta fit), and writes the 5 prior columns onto `df`.
#
# The K-sample matrix is NEVER cached — only the 5 derived stats land
# on the DataFrame.

import DataFrames: AbstractDataFrame, nrow

function BayesInteractomics.compute_mc_prior!(
    df::AbstractDataFrame,
    embedding_matrix::AbstractMatrix,
    config;
    K::Int = config.dnn_prior_mc_k,
    batch_size::Int = config.dnn_prior_mc_batch_size,
    model_path::AbstractString = _dnn_model_path(),
    rng::AbstractRNG = Random.default_rng(),
    protein_names = nothing,
)
    # Load model fresh (mirror predict_DNN line 79 load pattern).
    model = getDNNModel(11, _define_layers(512, 11),
                        _define_activations("relu", 11),
                        0.6641025641025641)
    Flux.loadmodel!(model, JLD2.load(model_path, "model_state"))
    model = model |> Flux.cpu

    return BayesInteractomics._compute_mc_prior_with_model!(df, model, embedding_matrix,
                                                            K, batch_size, rng;
                                                            protein_names = protein_names)
end

# ---------------------------------------------------------------------------- #
# _compute_mc_prior_with_model! test entry point
# ---------------------------------------------------------------------------- #
#
# Internal MC + CI + write helper. Accepts an already-constructed Flux Chain
# (rather than loading model-473 from disk) so that test fixtures can exercise
# the prior-column write path on the mock 2-layer model without a checkpoint
# file. Production callers go through `compute_mc_prior!`, which loads model-473
# and delegates here.

function BayesInteractomics._compute_mc_prior_with_model!(
    df::AbstractDataFrame,
    model,
    X::AbstractMatrix,
    K::Int,
    batch_size::Int,
    rng::AbstractRNG;
    protein_names = nothing,   # names for X's COLUMNS, in column order (or nothing)
)
    out = mc_dropout_batch(model, X; K=K, batch_size=batch_size, rng=rng)

    # Empirical-quantile 95% CI (empirical quantile, not a Beta fit).
    n = size(out.samples, 2)
    ci_low  = Vector{Float64}(undef, n)
    ci_high = Vector{Float64}(undef, n)
    for j in 1:n
        q = Statistics.quantile(view(out.samples, :, j), [0.025, 0.975])
        ci_low[j]  = q[1]
        ci_high[j] = q[2]
    end
    mean_v = Float64.(out.mean)
    std_v  = sqrt.(Float64.(out.var))

    nrows = nrow(df)
    if n == nrows
        # Positional: X columns line up 1:1 with df rows (metalearner scored every row).
        df.prior_mc_mean    = mean_v
        df.prior_mc_std     = std_v
        df.prior_mc_ci_low  = ci_low
        df.prior_mc_ci_high = ci_high
    elseif protein_names !== nothing && length(protein_names) == n && hasproperty(df, :Protein)
        # Name-aligned: X columns (= metalearner-scored subset) must be mapped onto
        # df.Protein by name; unscored rows get NaN. (The metalearner scores only the
        # feature-available subset, so n < nrow(df) — a positional assign throws
        # "New columns must have the same length as old columns".)
        nm   = string.(protein_names)
        mmap = Dict{String, Int}()
        for j in 1:n
            get!(mmap, nm[j], j)   # first occurrence wins on duplicate names
        end
        pm = fill(NaN, nrows); ps = fill(NaN, nrows)
        pl = fill(NaN, nrows); ph = fill(NaN, nrows)
        @inbounds for i in 1:nrows
            j = get(mmap, string(df.Protein[i]), 0)
            if j != 0
                pm[i] = mean_v[j]; ps[i] = std_v[j]; pl[i] = ci_low[j]; ph[i] = ci_high[j]
            end
        end
        df.prior_mc_mean = pm; df.prior_mc_std = ps
        df.prior_mc_ci_low = pl; df.prior_mc_ci_high = ph
    else
        throw(ArgumentError(
            "MC-Dropout prior: cannot align $n scored proteins to $nrows result rows " *
            "(protein_names $(protein_names === nothing ? "not supplied" : "len=$(length(protein_names))"); " *
            "df has Protein column: $(hasproperty(df, :Protein)))."))
    end
    df.prior_contribution = coalesce.(df.posterior_prob, NaN) .- df.prior_mc_mean
    return df
end
