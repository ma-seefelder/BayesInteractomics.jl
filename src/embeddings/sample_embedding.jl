# src/embeddings/sample_embedding.jl
# Similarity & Embeddings: sample-level PCA + UMAP/t-SNE compute.
# Pattern source: src/qc/pca_separation.jl::build_data_matrix + _pca_for_matrix + compute_pca_scores.

using Random
using Statistics
using LinearAlgebra

"""
    prepare_sample_label_matrix(data::InteractionData) ->
        (data_matrix::Matrix{Union{Missing, Float64}}, labels::NamedTuple)

Extracts an `(n_samples × n_proteins)` data matrix from `data` and assembles four aligned
label vectors:

- `condition`  — `"sample"` or `"control"`
- `replicate`  — 1-based replicate index within the experiment column block
- `experiment` — 1-based experiment index within the protocol
- `protocol`   — 1-based protocol index

Returns a NamedTuple with keys `(condition, replicate, experiment, protocol)`. Each label
vector has length `n_samples` and aligns row-for-row with the data matrix.

Mirrors `src/qc/pca_separation.jl::build_data_matrix` but emits four labels (replicate +
experiment are NEW; the existing helper only emits condition + protocol).
"""
function prepare_sample_label_matrix(data::InteractionData)
    all_columns       = Vector{Vector{Union{Missing, Float64}}}()
    condition_labels  = String[]
    replicate_labels  = Int[]
    experiment_labels = Int[]
    protocol_labels   = Int[]

    n_protocols = getNoProtocols(data)
    for p in 1:n_protocols
        samples_proto  = getSamples(data, p)
        controls_proto = getControls(data, p)

        # Sample replicates
        for exp_idx in 1:getNoExperiments(samples_proto)
            mat = getExperiment(samples_proto, exp_idx)  # n_proteins × n_replicates
            for col in 1:size(mat, 2)
                push!(all_columns, mat[:, col])
                push!(condition_labels, "sample")
                push!(replicate_labels, col)
                push!(experiment_labels, exp_idx)
                push!(protocol_labels, p)
            end
        end

        # Control replicates
        for exp_idx in 1:getNoExperiments(controls_proto)
            mat = getExperiment(controls_proto, exp_idx)
            for col in 1:size(mat, 2)
                push!(all_columns, mat[:, col])
                push!(condition_labels, "control")
                push!(replicate_labels, col)
                push!(experiment_labels, exp_idx)
                push!(protocol_labels, p)
            end
        end
    end

    labels = (condition  = condition_labels,
              replicate  = replicate_labels,
              experiment = experiment_labels,
              protocol   = protocol_labels)

    if isempty(all_columns)
        return (Matrix{Union{Missing, Float64}}(undef, 0, 0), labels)
    end

    # Stack: rows = n_samples, cols = n_proteins
    data_matrix = Matrix{Union{Missing, Float64}}(reduce(hcat, all_columns)')
    return (data_matrix, labels)
end

"""
    _filter_and_impute(X::AbstractMatrix{Union{Missing, Float64}}) ->
        (X_float::Matrix{Float64}, level::Symbol)

Cascading complete-case filter (100% → 80% → 50%) with column-mean imputation for residual
missings. Reuses `filter_complete_case` from `src/qc/pca_separation.jl`. Returns level
`:skipped` when fewer than 20 proteins survive the 50% threshold.
"""
function _filter_and_impute(X::AbstractMatrix{Union{Missing, Float64}})
    if size(X, 1) == 0 || size(X, 2) == 0
        return (Matrix{Float64}(undef, size(X, 1), 0), :skipped)
    end

    mask, level = filter_complete_case(Matrix{Union{Missing, Float64}}(X))
    if level === :skipped
        return (Matrix{Float64}(undef, size(X, 1), 0), :skipped)
    end

    X_filtered = X[:, mask]
    X_float = Matrix{Float64}(undef, size(X_filtered))
    for j in 1:size(X_filtered, 2)
        col = X_filtered[:, j]
        non_missing = collect(skipmissing(col))
        col_mean = isempty(non_missing) ? 0.0 : mean(non_missing)
        for i in 1:size(X_filtered, 1)
            X_float[i, j] = ismissing(X_filtered[i, j]) ? col_mean : Float64(X_filtered[i, j])
        end
    end
    return (X_float, level)
end

"""
    _compute_sample_embedding(data, cfg::EmbeddingsConfig) -> NamedTuple

Compute sample-level PCA + (optionally) UMAP / t-SNE. Returns a NamedTuple with keys
`(sample_pca_scores, sample_pca_var_explained, sample_labels, sample_filter_level,
  sample_umap_coords, sample_tsne_coords)`. Caller (pipeline integration) wraps
into an `EmbeddingsResult`.

Contract highlights:
- `cfg.method = :tsne` without TSne loaded throws `ArgumentError` BEFORE PCA work begins.
- `cfg.method = :umap` without the embeddings extension loaded silently falls back to
  PCA-only (no throw, downstream banner). `sample_umap_coords` stays `nothing`.
- `cfg.seed` is injected via `Random.seed!(cfg.seed)` IMMEDIATELY before the UMAP /
  t-SNE call — NEVER passed as a `seed=` kwarg (UMAP.jl 0.1.10 has no such kwarg).
- Pitfall 1: `n_neighbors` clamped to `max(2, min(cfg.n_neighbors, n_samples - 1))`; an
  `@info` is logged when the clamp fires.
- Cascade: 100% / 80% / 50% non-missing threshold via `filter_complete_case`; if even the
  50% tier yields fewer than 20 proteins, the function emits `@warn` once and returns the
  skipped NamedTuple (PCA scores empty, both coord matrices `nothing`).
"""
function _compute_sample_embedding(data, cfg::EmbeddingsConfig)
    # Build the (n_samples, n_proteins) matrix + four label vectors.
    X, labels = prepare_sample_label_matrix(data)

    # Master toggle: run_embeddings=false skips everything.
    if !cfg.run_embeddings
        return (sample_pca_scores       = zeros(0, 2),
                sample_pca_var_explained = Float64[],
                sample_labels            = labels,
                sample_filter_level      = :skipped,
                sample_umap_coords       = nothing,
                sample_tsne_coords       = nothing)
    end

    # Explicit-error path: t-SNE opt-in without TSne loaded → loud failure before compute.
    if cfg.method === :tsne
        _require_embeddings_extension(:tsne)
    end

    # Dense fast-path — when X carries no actual missing cells (e.g. a pooled-imputed input),
    # skip _filter_and_impute and jump straight to compute_pca_scores. eltype(X) is EXPECTED
    # to be Union{Missing, Float64} by construction (see prepare_sample_label_matrix line 75);
    # we test for presence of actual missing values, NOT the container eltype.
    #
    # Defensive eltype shielding. Earlier the fast-path silently
    # trusted the eltype contract from prepare_sample_label_matrix. If a future revision
    # of that helper returns a different container eltype (e.g. `Matrix{Float64}` or
    # `Matrix{Union{Missing, Float32}}`), `Matrix{Float64}(X)` still succeeds but the
    # downstream cascade-filter path would then fail on a separate code path. Make the
    # contract explicit: warn-and-fall-through when the eltype is unexpected, so the
    # fast-path does not silently shadow an upstream contract change.
    is_dense = size(X, 1) > 0 && size(X, 2) > 0 && !any(ismissing, X)
    _expected_eltype = Union{Missing, Float64}
    if is_dense && eltype(X) !== _expected_eltype
        @warn "[Embeddings] dense fast-path: unexpected eltype $(eltype(X)) (expected $_expected_eltype). Falling back to cascade filter for safety." maxlog=1
        is_dense = false
    end
    if is_dense
        @info "[Embeddings] sample embedding on pooled-imputed matrix (no cascade filter)" maxlog=1
        X_float = Matrix{Float64}(X)
        # Sentinel name `:complete_post_imputation` would disambiguate from the cascade-tier-1
        # `:complete_case` sentinel (src/qc/pca_separation.jl). Keep `:complete` here to preserve
        # the locked test contract (test/embeddings/test_sample_embedding_dense_fastpath.jl
        # asserts `:complete`); rename deferred.
        level = :complete
    else
        # Cascading complete-case + column-mean impute.
        X_float, level = _filter_and_impute(X)
    end

    if level === :skipped
        @warn "[Embeddings] sample embeddings skipped — fewer than 20 proteins at 50% non-missing threshold" maxlog=1
        return (sample_pca_scores       = zeros(0, 2),
                sample_pca_var_explained = Float64[],
                sample_labels            = labels,
                sample_filter_level      = :skipped,
                sample_umap_coords       = nothing,
                sample_tsne_coords       = nothing)
    end

    # PCA — always populated (core dep; MultivariateStats stays in [deps]).
    scores, var_explained = compute_pca_scores(X_float)  # from src/qc/pca_separation.jl

    n_samples = size(X_float, 1)

    # UMAP / t-SNE branch.
    umap_coords = nothing
    tsne_coords = nothing

    if cfg.method === :umap && _embeddings_extension_loaded()
        # n_neighbors clamp (Pitfall 1).
        n_eff = max(2, min(cfg.n_neighbors, n_samples - 1))
        if n_eff != cfg.n_neighbors
            @info "[Embeddings] sample UMAP: n_neighbors clamped from $(cfg.n_neighbors) to $n_eff (n_samples=$n_samples)"
        end
        # Determinism via Random state, NEVER seed= kwarg.
        Random.seed!(cfg.seed)
        try
            umap_coords = fit_sample_umap(X_float, n_eff, cfg.min_dist)
        catch e
            @warn "[Embeddings] sample UMAP failed: $e; PCA only" maxlog=1
            umap_coords = nothing
        end
    elseif cfg.method === :tsne
        # _require_embeddings_extension(:tsne) was already called above; reaching here means TSne is loaded.
        Random.seed!(cfg.seed)
        try
            tsne_coords = fit_sample_tsne(X_float, cfg.seed)
        catch e
            @warn "[Embeddings] sample t-SNE failed: $e; PCA only" maxlog=1
            tsne_coords = nothing
        end
    end
    # cfg.method === :none → both stay nothing; PCA already populated.

    return (sample_pca_scores       = scores,
            sample_pca_var_explained = var_explained,
            sample_labels            = labels,
            sample_filter_level      = level,
            sample_umap_coords       = umap_coords,
            sample_tsne_coords       = tsne_coords)
end

"""
    _pool_imputed_matrix(imputed::AbstractVector{<:InteractionData}) -> InteractionData{F,I}

Pool a vector of `M` imputed `InteractionData`s element-wise (mean per
cell) into a single `InteractionData` whose per-experiment matrices contain the cell-wise
mean across the `M` imputations. The pooled result is consumed exclusively by
`prepare_sample_label_matrix` → `_compute_sample_embedding` for the Sample Similarity card.

Accepts any `AbstractVector{<:InteractionData}` (including the abstract
`Vector{InteractionData}` container produced by `load_data` in the
`run_analysis(::CONFIG, ::Vector{InteractionData}, ::InteractionData{F,I})` overload).
Type parameters `F`, `I` are recovered from the first element via inner dispatch on
`_pool_imputed_matrix_impl`.

This helper enforces the lock ("Sample-level PCA / UMAP operate on the
post-imputation log-intensity matrix") for the imputed-vector path at `pipeline.jl:4158`.

# Scope

DO NOT route the pooled output into HBM, regression, or Beta-Bernoulli — those stages
already consume the per-imputation data inside their loops and applying the mean-pool would
discard the between-imputation variance the MI workflow depends on.

# Eltype landmine

`Protocol{F,I}` declares `data::Dict{I, Matrix{Union{Missing, F}}}` (src/core/types.jl:1214)
and the type parameter is INVARIANT — `Matrix{Float64}` is NOT a subtype of
`Matrix{Union{Missing, Float64}}` in Julia. The allocation below therefore MUST be
`Matrix{Union{Missing, F}}` even though no actual missings are written, otherwise the
`Protocol` constructor throws `MethodError`. Downstream `_compute_sample_embedding`'s dense
fast-path tests for `any(ismissing, X)` (NOT for the eltype) so this is the only correct
allocation strategy.

# Assumptions (RESEARCH.md A1 + A2)

- Every `imputed[m]` shares identical scaffolding with `imputed[1]` (same protocols, same
  per-protocol experiment counts, same per-experiment matrix shapes, same protein_IDs,
  same `.detected` mask, same parameter-position vectors). The traversal mirrors
  `prepare_sample_label_matrix` so downstream label alignment is preserved.

  The seven scaffolding fields (`no_protocols`, `no_experiments`, `no_parameters_HBM`,
  `no_parameters_Regression`, `protocol_positions`, `experiment_positions`,
  `matched_positions`, `detected`, `protein_IDs`, `protein_names`) are forwarded verbatim
  from `imputed[1]` into the pooled InteractionData constructor — they are NOT
  reconstructed from the pooled dicts. Downstream consumers of the pooled output MUST
  NOT route it into HBM / regression / Beta-Bernoulli inference (those need per-
  imputation rather than pooled scaffolding). `_compute_sample_embedding` only reads
  `samples`/`controls` so it is safe; future helpers reading `.detected` from the pooled
  output would silently see only imputation #1's mask.

  When `JULIA_DEBUG=BayesInteractomics` is set, a dev-time
  assertion verifies the A1 assumption (no_protocols, no_experiments, protein_IDs,
  detected equal across imputed[1..M]). Production runs pay zero cost.
- Imputed cells are guaranteed non-missing by the imputation construction. If a `missing` is
  encountered the helper THROWS `ArgumentError` (explicit-error path, matching the
  `_require_imputation_extension` pattern) rather than silently
  propagating the missing into the pooled matrix. The latter would defeat the dense
  fast-path contract in `_compute_sample_embedding` (line 153) and silently cascade into
  `_filter_and_impute`, potentially dropping below the 20-protein floor with no log
  message indicating the proximate cause was a rogue per-imputation missing.
"""
function _pool_imputed_matrix(imputed::AbstractVector{<:InteractionData})
    isempty(imputed) && throw(ArgumentError(
        "_pool_imputed_matrix: imputed vector cannot be empty"
    ))
    # Dispatch to the type-narrowed worker via the first element so F and I
    # are recovered as method type parameters. Pipeline callers pass the
    # abstractly typed `Vector{InteractionData}` produced by `load_data`,
    # while tests construct `Vector{InteractionData{Float64,Int64}}` directly;
    # both routes converge here.
    return _pool_imputed_matrix_impl(imputed[1], imputed)
end

function _pool_imputed_matrix_impl(
    ::InteractionData{F,I},
    imputed::AbstractVector{<:InteractionData},
) where {F<:AbstractFloat, I<:Integer}
    template = imputed[1]
    M = length(imputed)

    # JULIA_DEBUG-gated scaffolding-equality assertion.
    # The constructor at the bottom of this function forwards seven scaffolding
    # fields verbatim from `template = imputed[1]` (no_protocols,
    # no_experiments, no_parameters_HBM, no_parameters_Regression,
    # protocol_positions, experiment_positions, matched_positions, detected,
    # protein_IDs, protein_names). The docstring's A1 assumption mandates these
    # are identical across `imputed[1..M]`; this debug-only check verifies it
    # without paying the cost in production.
    #
    # If a future helper reads `.detected` from the pooled output (currently
    # safe — `_compute_sample_embedding` iterates samples/controls directly),
    # an A1 violation would silently mislead. The dev-time assertion surfaces
    # the violation LOUDLY.
    #
    # Activation: `JULIA_DEBUG=BayesInteractomics julia ...` or
    # `ENV["JULIA_DEBUG"] = "BayesInteractomics"` in the REPL.
    if get(ENV, "JULIA_DEBUG", "") != "" && occursin("BayesInteractomics", ENV["JULIA_DEBUG"])
        for m in 2:M
            o = imputed[m]
            @assert o.no_protocols == template.no_protocols  "_pool_imputed_matrix A1 violation: imputed[$m].no_protocols != imputed[1]"
            @assert o.no_experiments == template.no_experiments  "_pool_imputed_matrix A1 violation: imputed[$m].no_experiments != imputed[1]"
            @assert o.protein_IDs == template.protein_IDs  "_pool_imputed_matrix A1 violation: imputed[$m].protein_IDs != imputed[1]"
            @assert o.detected == template.detected  "_pool_imputed_matrix A1 violation: imputed[$m].detected != imputed[1]"
        end
        @debug "_pool_imputed_matrix: scaffolding OK across $M imputations"
    end

    function _pool_protocol_block(block_selector::Function)
        # block_selector(id) returns the relevant Dict{I, Protocol{F,I}} (samples or controls)
        pooled_protocols = Dict{I, Protocol{F,I}}()
        for p in 1:template.no_protocols
            template_proto = block_selector(template)[p]
            n_exp = getNoExperiments(template_proto)
            pooled_dict = Dict{I, Matrix{Union{Missing, F}}}()
            for e in 1:n_exp
                m1_mat = getExperiment(template_proto, e)
                pooled_mat = Matrix{Union{Missing, F}}(undef, size(m1_mat)...)
                for i in axes(m1_mat, 1), j in axes(m1_mat, 2)
                    acc = zero(F)
                    for m in 1:M
                        cell = block_selector(imputed[m])[p].data[e][i, j]
                        if ismissing(cell)
                            # Explicit-error path. The imputation
                            # guarantees imputed cells are non-missing; a missing
                            # here indicates a corrupt InteractionData upstream
                            # (e.g., a per-imputation detection floor leaked
                            # missings through, or the wrong vector was passed).
                            # Silently propagating would defeat the dense
                            # fast-path contract documented in the docstring.
                            throw(ArgumentError(
                                "_pool_imputed_matrix: imputation $m has missing cell at " *
                                "protocol=$p, experiment=$e, row=$i, col=$j. " *
                                "the imputation guarantees imputed cells are non-missing; " *
                                "this indicates a corrupt InteractionData upstream."
                            ))
                        end
                        acc += F(cell)
                    end
                    pooled_mat[i, j] = acc / M
                end
                pooled_dict[I(e)] = pooled_mat
            end
            pooled_protocols[I(p)] = Protocol{F,I}(
                I(n_exp),
                getIDs(template_proto),
                pooled_dict,
            )
        end
        return pooled_protocols
    end

    pooled_samples  = _pool_protocol_block(d -> d.samples)
    pooled_controls = _pool_protocol_block(d -> d.controls)

    return InteractionData{F,I}(
        template.protein_IDs,
        template.protein_names,
        pooled_samples,
        pooled_controls,
        template.no_protocols,
        template.no_experiments,
        template.no_parameters_HBM,
        template.no_parameters_Regression,
        template.protocol_positions,
        template.experiment_positions,
        template.matched_positions,
        template.detected,
    )
end
