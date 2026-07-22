# Metalearner stubs — extended by BayesInteractomicsMetalearnerExt
#
# When the extension is NOT loaded (i.e., the user has not run
# `using Flux, MLJ, MLJScikitLearnInterface, HDF5`), calling
# `predict_metalearner(...)` raises a MethodError that is caught by
# `_safe_predict_metalearner` in src/analysis/pipeline.jl. That helper
# emits a one-shot @warn and sets `metalearner_status = :extension_not_loaded`
# on the AnalysisResult; downstream code falls back to the BF-derived
# posterior already populated by the EM/copula stage.

"""
    predict_metalearner(poi::String; kwargs...)

Predict metalearner-adjusted posterior probabilities for protein-of-interest `poi`.

**Requires:** `using Flux, MLJ, MLJScikitLearnInterface, HDF5`

When the metalearner extension is not loaded this function has no methods
and any call raises a `MethodError`. Callers should wrap invocations in
`_safe_predict_metalearner` (see `src/analysis/pipeline.jl`) to translate
the error into the Variante B graceful-fallback path: emit one `@warn`,
return `(nothing, nothing, :extension_not_loaded)`, and let the BF-derived
posterior already on `AnalysisResult.copula_results.posterior_prob` flow
through unchanged.

Real implementation: `ext/BayesInteractomicsMetalearnerExt/metalearner.jl`
(moved from `src/ml/metalearner.jl`).
"""
function predict_metalearner end

# ---------------------------------------------------------------------------- #
# MetalearnerStatus sentinel values
#
# Stored as a `Symbol` on `AnalysisResult.metalearner_status`. Used by the
# interactive HTML report to render the status banner, Methods
# subsection, and `posterior_prob` column tooltip.
# ---------------------------------------------------------------------------- #
#
#   :extension_not_loaded   — extension not loaded; posterior_prob is BF-derived
#                             (this is the default value at AnalysisResult
#                             construction time; flipped by _safe_predict_metalearner)
#   :loaded                 — extension loaded AND predict_metalearner returned
#                             non-nothing predictions; posterior_prob has been
#                             updated via update_posterior_prob!
#   :prediction_failed      — extension loaded but predict_metalearner raised
#                             an exception or returned nothing (e.g., POI
#                             renamed by STRING curation); posterior_prob
#                             falls back to BF-derived

const METALEARNER_STATUS_VALUES = (:extension_not_loaded, :loaded, :prediction_failed)

# ---------------------------------------------------------------------------- #
# DNN Prior tab MC-Dropout stubs
#
# Parent-level stubs for the MC-Dropout prior pipeline. Extended by
# `BayesInteractomicsMetalearnerExt/mc_dropout.jl` when the trigger packages
# `Flux, MLJ, MLJScikitLearnInterface, HDF5` are loaded. With the extension
# inactive the stubs have zero methods; any call raises a `MethodError` that
# `_safe_compute_mc_prior!` catches → one-shot `@warn` → 5 NaN columns
# via `_populate_mc_prior_nan_columns!`.
# ---------------------------------------------------------------------------- #

import DataFrames: nrow

"""
    compute_mc_prior!(df, embedding_matrix, config; K, batch_size, model_path, rng)

Parent stub — extended by
`BayesInteractomicsMetalearnerExt/mc_dropout.jl`. When the extension is not
loaded the function has zero methods and any call raises a `MethodError`.
Callers MUST wrap invocations in `_safe_compute_mc_prior!` (see
`src/analysis/pipeline.jl`) to translate the error into the Variante B
graceful-fallback path: emit one `@warn`, populate the 5 prior columns with
NaN, and let the BF-derived posterior_prob flow through unchanged.

**Requires:** `using Flux, MLJ, MLJScikitLearnInterface, HDF5`

Real implementation: `ext/BayesInteractomicsMetalearnerExt/mc_dropout.jl`.
"""
function compute_mc_prior! end

"""
    _compute_mc_prior_with_model!(df, model, X, K, batch_size, rng) -> df

Test-entry helper for the model-loaded MC-Dropout path.
Extended by `BayesInteractomicsMetalearnerExt/mc_dropout.jl`. With the metalearner
extension not loaded, the function has zero methods. Test fixtures call this
directly to avoid loading model-473 from disk.

**Requires:** `using Flux, MLJ, MLJScikitLearnInterface, HDF5`

Production code goes through `compute_mc_prior!`, which loads model-473 and
delegates here.
"""
function _compute_mc_prior_with_model! end

"""
    _populate_mc_prior_nan_columns!(df) -> df

Fallback helper. Populates the 5 MC-Dropout prior columns with
NaN on the results DataFrame. Used by `_safe_compute_mc_prior!` in the
Variante B and opt-out code paths to keep the result schema uniform.
"""
function _populate_mc_prior_nan_columns!(df::AbstractDataFrame)
    n = nrow(df)
    df.prior_mc_mean      = fill(NaN, n)
    df.prior_mc_std       = fill(NaN, n)
    df.prior_mc_ci_low    = fill(NaN, n)
    df.prior_mc_ci_high   = fill(NaN, n)
    df.prior_contribution = fill(NaN, n)
    return df
end
