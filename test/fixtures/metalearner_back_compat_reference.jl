# test/fixtures/metalearner_back_compat_reference.jl
#
# Byte-identical contract baseline.
#
# REFERENCE_PROB_LEGACY_8FEAT is the P(interaction) value produced by the
# legacy predict_metalearner path against `metalearners/HistGradientBoosting_tune.jld2`
# on the deterministic 1-row input `DataFrame(neighborhood=[0.5], fusion=[0.5],
# phylogenetic=[0.5], coexpression=[0.5], experimental=[0.5], database=[0.5],
# textmining=[0.5], DNN=[0.5])`. Captured BEFORE the schema-aware predict_metalearner change.
#
# The downstream test asserts `isapprox(new_prob, REFERENCE_PROB_LEGACY_8FEAT; atol=1e-9)`
# to prove the :legacy_8feat branch dispatch preserves byte-identical output.
#
# Capture procedure:
#   1. Activated examples/ project (carries Flux + MLJ + MLJScikitLearnInterface + HDF5 + JLD2 triggers).
#   2. `mach = ext.load_metalearner("metalearners/HistGradientBoosting_tune.jld2")` using the
#      canonical legacy loader `load_metalearner(path) = MLJ.machine(path)`
#      (ext/BayesInteractomicsMetalearnerExt/metalearner.jl:433).
#   3. `preds = MLJ.predict(mach, fixed)` returns a
#      `CategoricalDistributions.UnivariateFiniteVector{OrderedFactor{2}, Float64, UInt32, Float64}`.
#   4. Scalar extracted via the production recipe (matches metalearner.jl:385):
#        `prob_scalar = MLJ.pdf.(preds, Ref(1.0))[1]`
#      This pulls the P(label = 1.0) component of the UnivariateFinite distribution.
#
# Downstream comparison MUST extract the post-change
# prediction using the SAME recipe, then assert isapprox within 1e-9.

const REFERENCE_PROB_LEGACY_8FEAT::Float64 = 0.633000588611009

# Deterministic 1-row input — recorded as a NamedTuple so the test can rebuild
# the DataFrame in any column order the new schema chooses.
const REFERENCE_INPUT_ROW = (
    neighborhood = 0.5, fusion = 0.5, phylogenetic = 0.5,
    coexpression = 0.5, experimental = 0.5, database = 0.5,
    textmining = 0.5, DNN = 0.5,
)

# Artefact under test — the test loads the same JLD2 via the new
# `load_metalearner_with_schema` and re-evaluates against this input.
#
# Stored as a REPO-RELATIVE path. The large model files now live OUTSIDE the git
# tree (in the lazily-downloaded `bayesinteractomics_models` Pkg artifact) on CI,
# so consumers MUST resolve to an absolute on-disk path via
# `reference_artefact_path()` (which routes through the metalearner extension's
# `resolve_metalearner_path`), NOT assume this relative string exists on disk.
const REFERENCE_ARTEFACT_RELPATH = "metalearners/HistGradientBoosting_tune.jld2"

# Lazily resolve REFERENCE_ARTEFACT_RELPATH to an absolute path that exists in
# EITHER the developer repo tree OR the downloaded models artifact. Returns
# `nothing` when the metalearner extension is not loaded, or when the artefact is
# genuinely absent from both locations (callers should skip gracefully then).
# Must be called in a process where `using BayesInteractomics` (+ the Flux / MLJ /
# MLJScikitLearnInterface / HDF5 triggers) has activated
# BayesInteractomicsMetalearnerExt.
function reference_artefact_path()
    ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)
    ext === nothing ? nothing : ext.resolve_metalearner_path(REFERENCE_ARTEFACT_RELPATH)
end

# Provenance — ISO 8601 wall-clock of the capture command.
const REFERENCE_CAPTURED_AT = "2026-05-22T20:06:34Z"

# String form of typeof(preds) so the test can confirm the extraction recipe
# still applies to the post-change MLJ output (must match here byte-for-byte
# if the underlying machine is unchanged).
const REFERENCE_MLJ_OUTPUT_TYPE = "CategoricalDistributions.UnivariateFiniteVector{OrderedFactor{2}, Float64, UInt32, Float64}"

# Extraction recipe — the downstream test MUST use this exact code to derive
# the scalar that gets compared against REFERENCE_PROB_LEGACY_8FEAT.
const REFERENCE_EXTRACTION_RECIPE = "MLJ.pdf.(preds, Ref(1.0))[1]  # pulls P(label = 1.0) from UnivariateFinite"
