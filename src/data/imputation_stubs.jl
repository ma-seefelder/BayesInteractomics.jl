# Imputation stubs — extended by BayesInteractomicsImputationExt
#
# When the extension is NOT loaded (i.e., the user has not run `using GLM`),
# `fit_dropout_curves` and the `impute_mnar*` family have NO methods. Behaviour
# in that state, per ROADMAP §"Optional Features and Extensions" lock:
#
#   * If the user did NOT explicitly opt into MNAR imputation (i.e.
#     CONFIG.imputation_method == :none or kwarg unset on load_data), no stub
#     is invoked. Raw data flows through unchanged.
#   * If the user explicitly requested :mar or :mnar via CONFIG or load_data
#     kwarg, the dispatch site MUST detect the absence of GLM and call
#     `_require_imputation_extension(method)` below, which throws a clear
#     ArgumentError pointing the user at `using GLM`. (This is louder than the
#     metalearner's Variante B fallback because the user explicitly opted in.)

"""
    fit_dropout_curves(intensity_matrix, column_names; kwargs...) -> DropoutFit

Fit per-column logistic dropout curves `o_{i,c} ~ Bernoulli(σ(ρ_c + ζ_c · ȳ_i))`.

**Requires:** `using GLM`

Real implementation: `ext/BayesInteractomicsImputationExt/dropout.jl`
(moved from `src/data/dropout.jl`).
"""
function fit_dropout_curves end

"""
    save_dropout_fit(fit::DropoutFit, path::String) -> String

Persist a `DropoutFit` to JSON.

**Requires:** `using GLM` (because the producer of `DropoutFit` lives in the imputation extension).
"""
function save_dropout_fit end

"""
    load_dropout_fit(path::String) -> DropoutFit

Load a previously-persisted `DropoutFit` from JSON.

**Requires:** `using GLM`
"""
function load_dropout_fit end

"""
    impute_mnar(data, fit::DropoutFit; kwargs...) -> Matrix

Single MNAR-aware imputation via tilted-Gaussian sampler driven by the fitted dropout curves.

**Requires:** `using GLM`

Real implementation: `ext/BayesInteractomicsImputationExt/imputation_mnar.jl`
(moved from `src/data/imputation_mnar.jl`).
"""
function impute_mnar end

"""
    impute_mnar_from_paths(xlsx_path, fit_path, output_path; kwargs...) -> NamedTuple

CLI-style wrapper around `impute_mnar` that reads XLSX, applies the curves, writes XLSX.

**Requires:** `using GLM`
"""
function impute_mnar_from_paths end

"""
    column_imputation_sigma(fit::DropoutFit, col::Int,
                            intensity_matrix::AbstractMatrix) -> Float64

Empirical σ (sqrt-variance) of the finite-numeric values in column `col` of
the post-imputation `intensity_matrix`. Returns 0.0 when fewer than 2 finite
values are available.

Used by the v2b mask-aware regression to source σ_imp per source-
matrix column. `fit` is consulted for boundscheck only (col ∈ 1:length(fit.rho));
the actual σ value comes from the intensity matrix to match the
post-imputation variance pattern (src/analysis/pipeline.jl:513-518).

**Requires:** `using GLM`

Real implementation: `ext/BayesInteractomicsImputationExt/dropout.jl`.
"""
function column_imputation_sigma end

"""
    DropoutFit

Container for per-column dropout-curve parameters fit by `fit_dropout_curves`.

The struct is defined in core so the symbol is always available and the type
can appear in function signatures + JLD2-cache serialisation round-trips even
when the imputation extension is NOT loaded. The CONSTRUCTOR `fit_dropout_curves`
and the persistence helpers `save_dropout_fit` / `load_dropout_fit` live in
`BayesInteractomicsImputationExt` and require `using GLM`.

This is a concrete struct (previously an `abstract type DropoutFit end` stub).
The field layout preserves JLD2-cache schema compatibility with earlier releases.

Fields
------
- `rho::Vector{Float64}`              — intercept ρ̂_c per column (NaN if excluded)
- `zeta::Vector{Float64}`             — slope ζ̂_c per column (NaN if excluded)
- `column_names::Vector{String}`      — source column identifiers (length = n_columns)
- `n_proteins::Int`                   — proteins included in the fit (post-filter)
- `n_detections_per_column::Vector{Int}` — count of detections per column (diagnostic)
- `fit_timestamp::String`             — ISO8601 UTC timestamp at fit time
- `software_version::String`          — package version at fit time (e.g. "1.1.6")
- `dataset_hash::String`              — "sha256:<64-hex>" of the input matrix
"""
struct DropoutFit
    rho::Vector{Float64}
    zeta::Vector{Float64}
    column_names::Vector{String}
    n_proteins::Int
    n_detections_per_column::Vector{Int}
    fit_timestamp::String
    software_version::String
    dataset_hash::String
end

"""
    _imputation_extension_loaded() -> Bool

Returns `true` when `BayesInteractomicsImputationExt` is loaded (which happens
when the user has called `using GLM` before `using BayesInteractomics`).
Detection works by checking whether `fit_dropout_curves` has at least one
method registered — the stub has zero methods, and the extension's
`function BayesInteractomics.fit_dropout_curves(...)` adds the first method.
Used by `load_data` (and downstream callers) to gate the preflight
`_require_imputation_extension` check: when the extension IS loaded we skip
the loud error and let the real implementation run.
"""
function _imputation_extension_loaded()
    return !isempty(methods(fit_dropout_curves))
end

"""
    _require_imputation_extension(method::Symbol)

Internal helper. Throws an `ArgumentError` with actionable guidance when the
user explicitly requests `:mar` or `:mnar` imputation without `using GLM`.
Called from `load_data` / `CONFIG` validation paths.

Note: callers should normally gate this with `_imputation_extension_loaded()`
so the loud error fires ONLY when the extension is genuinely missing.
"""
function _require_imputation_extension(method::Symbol)
    if method === :mar || method === :mnar
        throw(ArgumentError(
            "Imputation method `:$(method)` requires the BayesInteractomicsImputationExt extension. " *
            "Load it by running `using GLM` BEFORE `load_data` / `run_analysis`. " *
            "Alternatively, set `imputation_method = :none` to skip imputation."
        ))
    end
    return nothing
end
