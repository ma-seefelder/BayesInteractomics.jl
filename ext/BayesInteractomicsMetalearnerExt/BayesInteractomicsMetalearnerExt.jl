module BayesInteractomicsMetalearnerExt

using BayesInteractomics
using Flux
using HDF5
using MLJ
using MLJScikitLearnInterface
using LazyArtifacts   # brings the lazy-capable `@artifact_str` into scope for the models artifact

# Re-imports of core deps the moved files use
using DataFrames
import DataFrames: innerjoin, leftjoin
import JLD2
import JLD2: load
import JLD2  # full module reference for JLD2.jldopen append-mode in save_metalearner_with_schema
import StatsPlots

# CSV and XLSX accessors used by the moved metalearner.jl
import CSV
import XLSX: readtable, writetable

include("group_disjoint_resampling.jl")  # GroupDisjointFolds ResamplingStrategy — must be in this stable namespace so a saved Stack's `resampling` field deserialises in a fresh process
include("dnn_model.jl")      # DNN architecture
include("metalearner_calibration.jl")  # post-hoc Stack output calibration (Platt/isotonic)
include("embedding_combine.jl")  # combine_embeddings group-method (must precede metalearner.jl so the symbol is in scope at the prediction_data call site)
include("metalearner.jl")    # schema-aware metalearner prediction + save/load
include("feature_lookup_cache.jl")  # TR+DDI feature lookup cache
include("mc_dropout.jl")     # MC-Dropout inference for DNN Prior tab

end # module
