###########################################################################################
# Training of and XGBoost classifier with isotonic regression
# to combined predictions from the DNN with the other features 
# (see below) 

# Author: Dr. rer. nat. Manuel Seefelder
# Date: 27th June 2025

###########################################################################################################

# This file hosts the metalearner in ext/BayesInteractomicsMetalearnerExt/.
#
# Eager imports (DataFrames: innerjoin/leftjoin, HDF5, Flux, JLD2: load, MLJ,
# MLJScikitLearnInterface, StatsPlots) are now provided by the extension module
# shell at ext/BayesInteractomicsMetalearnerExt/BayesInteractomicsMetalearnerExt.jl.
#
# The `predict_metalearner` function below is a method on the parent stub
# declared in `src/ml/metalearner_stubs.jl` (`function predict_metalearner end`)
# and is therefore qualified as `function BayesInteractomics.predict_metalearner`.

# Note: getDNNModel and helpers are loaded by dnn_model.jl (included before
# this file in BayesInteractomicsMetalearnerExt.jl).

# Environment variables
dtrain_path = "encodings/train_data.h5";
dval_path   = "encodings/val_data.h5";
dtest_path  = "encodings/test_data.h5";
const MODELPATH   = "encodings/model-473-0.5414302830201915.jld2"

# ---------------------------------------------------------------------------- #
# Shipped-model resolution (Pkg artifact vs. developer git-clone)
# ---------------------------------------------------------------------------- #
# The large model artefacts (metalearner Stacks, the production DNN checkpoint,
# the feature lookup) are shipped as a lazily-downloaded Pkg artifact
# (`bayesinteractomics_models`), because they exceed the size that belongs in a
# registered package's git repo. A developer git-clone keeps them in the repo
# tree under `metalearners/` + `encodings/`, so prefer that local copy when
# present and fall back to the downloaded artifact otherwise.
"""
    _models_root() -> String

Root directory containing the shipped `metalearners/` + `encodings/` model
artefacts: the package repo tree for a developer clone, else the lazily-fetched
`bayesinteractomics_models` artifact for a registry install.
"""
function _models_root()
    local_root = dirname(dirname(@__DIR__))
    isfile(joinpath(local_root, "metalearners", "metalearner_tr_ddi.jld2")) && return local_root
    return artifact"bayesinteractomics_models"
end

# Resolve a shipped model file given as a repo-relative subpath
# (e.g. "metalearners/feature_lookup.jld2" or MODELPATH) against `_models_root()`.
_shipped_model_path(subpath::AbstractString) = joinpath(_models_root(), subpath)

# Production DNN checkpoint, resolved against `_models_root()`.
_dnn_model_path() = _shipped_model_path(MODELPATH)

# ---------------------------------------------------------------------------- #
# Meta-Learner with STRING scores and deep-neural network
# ---------------------------------------------------------------------------- #

"""
    resolve_metalearner_path(path::String) -> Union{String, Nothing}

Attempt to resolve metalearner path by checking multiple locations:
1. Exact path as provided (if absolute or relative to CWD)
2. Relative to package root directory
3. Relative to package root/metalearners/ directory

Returns absolute path if found, nothing otherwise.
"""
function resolve_metalearner_path(path::String)
    # Try path as-is (absolute or relative to CWD)
    if isfile(path)
        return abspath(path)
    end

    # Try relative to the models root (repo tree for a dev clone, else the artifact)
    pkg_root = _models_root()
    pkg_relative = joinpath(pkg_root, path)
    if isfile(pkg_relative)
        return abspath(pkg_relative)
    end

    # Try in package root/metalearners/ directory (common case)
    if !startswith(path, "metalearners")
        pkg_ml_path = joinpath(pkg_root, "metalearners", basename(path))
        if isfile(pkg_ml_path)
            return abspath(pkg_ml_path)
        end
    end

    # Not found
    return nothing
end

"""
    resolve_metalearner_path(::Nothing; use_mc_dropout::Bool) -> Union{String, Nothing}

Schema-aware default resolver.

When `CONFIG.metalearner_path === nothing`, the runtime selects the schema-matching
default artefact based on `metalearner_use_mc_dropout` (default `false`):
- `use_mc_dropout = false` (default)          → `metalearners/metalearner_tr_ddi.jld2` (14-feat TR+DDI)
- `use_mc_dropout = true`  (deprecated opt-in) → `metalearners/metalearner_tr_ddi_mc.jld2` (15-feat, +mc_std)

Delegates to the existing `resolve_metalearner_path(::String)` for filesystem
resolution (CWD → pkg root → pkg root/metalearners). Returns `nothing` if the
default artefact is not on disk — callers are responsible for the fallback path.
"""
function resolve_metalearner_path(::Nothing; use_mc_dropout::Bool)
    default_name = use_mc_dropout ? "metalearner_tr_ddi_mc.jld2" : "metalearner_tr_ddi.jld2"
    return resolve_metalearner_path(joinpath("metalearners", default_name))
end

# The `_require_mc_dropout_pipeline` hard-error guard was REMOVED.
# Rationale: both `:tr_ddi` and `:tr_ddi_mc` already
# call `predict_DNN(embedding_matrix, model_path)` to produce the `DNN` feature,
# so model-473 is a hard dependency of EVERY metalearner schema. MC-Dropout is just
# K=30 stochastic passes through that SAME model. There is no runtime state where
# the 14-feat path works but MC cannot: if the extension+model are available the
# metalearner runs (and MC runs); if absent, the whole metalearner falls back to
# `bf/(1+bf)` (Variante B). The guard protected a dependency split that doesn't
# exist. MC-Dropout is a DEPRECATED OPT-IN as of v1.2.1 (non-MC :tr_ddi is the default).

# load model definitions
function predict_DNN(DATASET::String, model_path::String = _dnn_model_path(); device = Flux.cpu)
    model = getDNNModel(
      11, _define_layers(512, 11),
      _define_activations("relu", 11),
      0.6641025641025641
      )

    # load the model
    model_state = load(model_path, "model_state")
    Flux.loadmodel!(model, model_state)
    # move model to device
    model = model |> device

    # load data
    data = HDF5.h5open(DATASET, "r") do file
      HDF5.read(file, "features_labels")
    end

    data = data[:, 1:end-1]
    data = data|> device

    # predict
    predictions = model(data')[1,:]
    return predictions
end

function predict_DNN(
  d::Matrix{F}, model_path::String = _dnn_model_path(); 
  device = Flux.cpu) where F <: AbstractFloat
   
  # load model
    model = getDNNModel(
      11, _define_layers(512, 11),
      _define_activations("relu", 11),
      0.6641025641025641
      )

    # load the model
    model_state = load(model_path, "model_state")
    Flux.loadmodel!(model, model_state)
    # move model to device
    model = model |> device

    # load data
    d = d|> device

    # predict
    predictions = model(d) |> Flux.cpu
    predictions = predictions[1,:]

    return predictions
end

function getMetaLearnerDataset(DATASET::String, model_path::String; device = Flux.cpu, embedding_dim::Int64 = 3072)
  # Get predictions from the DNN model first
  dnn_predictions = predict_DNN(DATASET, model_path, device = device)

  # Load the HDF5 file
  scores, protein_names, features_and_labels = HDF5.h5open(DATASET, "r") do file
    s = HDF5.read(file, "scores")
    p = HDF5.read(file, "proteins")
    fl = HDF5.read(file, "features_labels")
    (s, p, fl)
  end

  # Extract labels from the combined feature/label matrix
  local labels
  if size(features_and_labels, 2) == embedding_dim + 1
    labels = features_and_labels[:, embedding_dim + 1]
  else
    num_rows = size(features_and_labels, 1)
    labels = [missing for _ in 1:num_rows]
  end

  # Construct the final DataFrame for the meta-learner
  data = DataFrame(
    Protein1 = protein_names[:, 1],
    Protein2 = protein_names[:, 2],
    neighborhood = scores[:, 1],
    fusion = scores[:, 2],
    phylogenetic = scores[:, 3],
    coexpression = scores[:, 4],
    experimental = scores[:, 5],
    database = scores[:, 6],
    textmining = scores[:, 7],
    DNN = dnn_predictions,
    label = labels
  )

  return data
end


# Instructions to download required files from the STRING database
# 1. Go to string database https://string-db.org/
# 2. Got to Download
# 3. Enter species and download the following files:
#       [XXX].protein.links.detailed.v12.0.onlyAB.txt.gz
#       [XXX].protein.sequence.embeddings.v12.0.h5
#       [XXX].protein.network.embeddings.v12.0.h5
#       [XXX].protein.info.v12.0.txt.gz (
# Search a databse for the ENSEMBL ID of your bait protein and provide that info
# to as a function argument

function prediction_data(embeddings_sequence, embeddings_network, links, protein_info, poi)
    # protein_info
    protein_info_df = CSV.read(protein_info, DataFrame)
    proteome_size = size(protein_info_df, 1)
    # generate new data frame with three columns
    # 1. poi: identifier of the protein of interest, aka bait protein
    # 2. STRING-ID of the all proteins
    # 3. preferred_name
    prediction_df = DataFrame(
        protein1        = [poi for _ in 1:proteome_size],
        protein2        = protein_info_df[:,1],
        preferred_name  = protein_info_df[:,2],
        neighborhood    = zeros(Float64, proteome_size),
        fusion          = zeros(Float64, proteome_size),
        phylogenetic    = zeros(Float64, proteome_size),
        coexpression    = zeros(Float64, proteome_size),
        experimental    = zeros(Float64, proteome_size),
        database        = zeros(Float64, proteome_size),
        textmining      = zeros(Float64, proteome_size)
    )

    # load links
    links_scores = CSV.read(links, DataFrame)
    links_scores = links_scores[links_scores.protein1 .== poi .|| links_scores.protein2 .== poi, :]

    # Create a dictionary for faster lookups
    protein_to_row = Dict(protein => i for (i, protein) in enumerate(prediction_df.protein2))

    # replace the initial scores with the ones from STRING db
    for row in eachrow(links_scores)
        other_protein = row.protein1 == poi ? row.protein2 : row.protein1
        position = get(protein_to_row, other_protein, 0)

        if position > 0
            prediction_df[position, :neighborhood] = row.neighborhood / 1000
            prediction_df[position, :fusion]       = row.fusion / 1000
            prediction_df[position, :phylogenetic] = row.cooccurence / 1000
            prediction_df[position, :coexpression] = row.coexpression / 1000
            prediction_df[position, :database]     = row.database / 1000
            prediction_df[position, :textmining]   = row.textmining / 1000
            prediction_df[position, :experimental] = row.experimental / 1000
        end
    end

    # load embeddings
    species_id = split(links, "/")[2]
    species_id = split(species_id, ".")[1]

    #  combine embeddings if they don't exist already in the encodings directory
    if !isfile("encodings/emb_$species_id.h5")
      file_1 = HDF5.h5open(embeddings_sequence, "r+")
      file_2 = HDF5.h5open(embeddings_network, "r+")

      file_1_embedding  = HDF5.read(file_1, "embeddings")
      file_1_proteins   = HDF5.read(file_1, "proteins")
      file_2_embedding  = HDF5.read(file_2, "embeddings")
      file_2_proteins   = HDF5.read(file_2, "proteins")

      tmp_file = "encodings/tmp_file.h5"
      f = HDF5.h5open(tmp_file, "w")

      group_1 = HDF5.create_group(f, "$(species_id)_seq")
      group_2 = HDF5.create_group(f, "$(species_id)_net")

      group_1["embeddings"] = file_1_embedding
      group_1["proteins"]   = file_1_proteins
      group_2["embeddings"] = file_2_embedding
      group_2["proteins"]   = file_2_proteins

      combine_embeddings(
        f["$(species_id)_seq"], f["$(species_id)_net"], String(species_id),
        output_file = "encodings/emb_$species_id.h5"
        )

      close(f)
      close(file_1)
      close(file_2)

      # delete the temporary file
      rm(tmp_file)

    end

    # load embeddings and protein names
    embedding = HDF5.h5open("encodings/emb_$species_id.h5", "r") do file 
      HDF5.read(file, "$species_id/embeddings")
    end

    protein_names = HDF5.h5open("encodings/emb_$species_id.h5", "r") do file 
        HDF5.read(file, "$species_id/proteins")
    end

    # --- Reorder embeddings to match the prediction_data DataFrame ---
    protein_to_embed_idx = Dict(name => i for (i, name) in enumerate(protein_names))
    target_protein_order = prediction_df.protein2
    source_indices = [get(protein_to_embed_idx, protein_id, nothing) for protein_id in target_protein_order]

    # Create a mask to identify which proteins were successfully found and filter the data accordingly
    found_mask = .!isnothing.(source_indices)
    valid_source_indices = Int.(filter(!isnothing, source_indices))
    prediction_data_filtered = prediction_df[found_mask, :]
    embedding_reordered = embedding[:, valid_source_indices]

    return prediction_data_filtered, embedding_reordered
end

"""
    predict_metalearner(poi::String; kwargs...)

Trains a meta-learner and uses it to predict interaction probabilities for a given protein of interest (POI) against a proteome.

The function first trains a meta-learner (NGBoost-Classifier) using pre-defined training, validation, and test datasets. It then prepares a prediction dataset for the specified POI by combining STRING database features (neighborhood, fusion, co-expression, etc.) with predictions from a pre-trained Deep Neural Network (DNN) based on sequence embeddings. The trained meta-learner then uses this combined dataset to generate final interaction probabilities.

The results, including the intermediate DNN predictions and the final meta-learner predictions, are saved to an Excel file.

# Arguments
- `poi::String`: The identifier for the protein of interest (bait protein), typically an ENSEMBL ID.

# Keywords
- `model_path::String = MODELPATH`: Path to the pre-trained production DNN checkpoint (`encodings/model-473-…jld2`), resolved via `_dnn_model_path()` against the shipped models artifact or the developer repo tree.
- `embeddings_seq::String = `MODELPATH`: Path to the STRING database protein sequence embeddings file.
- `embeddings_net::String = "encodings/9606.protein.network.embeddings.v12.0.h5"`: Path to the STRING database protein network embeddings file.
- `links::String = "encodings/9606.protein.links.detailed.v12.0.onlyAB.txt"`: Path to the STRING database detailed protein links file.
- `protein_info::String = "encodings/9606.protein.info.v12.0.txt"`: Path to the STRING database protein information file.
- `device = Flux.cpu`: The device (e.g., `Flux.cpu` or `Flux.gpu`) on which to run the DNN model.
- `output_file::String = "prior.xlsx"`: The path for the output Excel file containing the prediction results.
- `metalearner_file::String = "metalearners/HistGradienBossting_tune.jld2"`: Path to the pre-trained meta-learner file.

# Returns
- `Tuple{DataFrame, Any}`: A tuple containing:
  - `data::DataFrame`: A DataFrame with the combined features and final predictions (`MetaClassifier` column).
  - `meta`: The fitted meta-learner model object.

# Side Effects
- Trains a new meta-learner model in each call.
- Writes the prediction results to `output_file` if it does not already exist. If the file exists, a warning is printed to the console, and the file is not overwritten.

# Example
```julia
# Assuming the necessary data files are in the "encodings/" directory
# and the POI is "9606.ENSP00000479624"
final_predictions, meta_model = predict_metalearner("9606.ENSP00000479624")
```
"""
# ---------------------------------------------------------------------------- #
# On-the-fly TR/DDI featurisation for lookup misses.
#
# When the prebuilt feature lookup cache has no entry for a (poi, prey) pair AND
# the caller staged per-species STRING/UniProt/DDI sources under `source_dir`,
# featurise the pair on-the-fly via the `featurise_pair_onthefly` builder
# so a non-human bait reads REAL transferred + DDI channels instead of a silent
# zero. Returns `nothing` (→ caller zero-imputes) when no `source_dir` was given
# or featurisation fails (missing source file, novel ID) — degrade gracefully,
# never crash the inference. The result NamedTuple carries the 6 TR/DDI fields the
# caller reads (`neighborhood_tr`, `experiments_tr`, `database_tr`,
# `textmining_tr`, `ddi_n_known`, `ddi_has_known`).
# ---------------------------------------------------------------------------- #
function _onthefly_feature_entry(a::AbstractString, b::AbstractString,
                                 sp::AbstractString,
                                 source_dir::Union{Nothing, AbstractString})
    source_dir === nothing && return nothing
    # Path-safety: when `species === nothing`, `sp` is the
    # user-controlled POI prefix. Reject a non-all-digits taxon id before it
    # reaches the on-the-fly featuriser (mirrors `_validate_species` in
    # download_species_data.jl); prevents path traversal via a crafted POI.
    (isempty(sp) || !all(isdigit, sp)) && return nothing
    try
        return featurise_pair_onthefly(String(a), String(b);
                                       source_dir = String(source_dir),
                                       species    = String(sp))
    catch e
        @debug "predict_metalearner: on-the-fly featurisation failed for ($a, $b); zero-imputing." exception = e
        return nothing
    end
end

function BayesInteractomics.predict_metalearner(
    poi::String;
    species::Union{Nothing, Integer} = nothing,   # NCBI taxonomy ID; nothing ⇒ derive from the poi prefix
    model_path        = _dnn_model_path(),
    embeddings_seq    = nothing,
    embeddings_net    = nothing,
    links             = nothing,
    protein_info      = nothing,
    device            = Flux.cpu,
    output_file       = "prior.xlsx",
    metalearner_file::Union{Nothing, String} = "metalearners/HistGradientBoosting_tune.jld2",
    use_mc_dropout::Bool = true,   # MC-Dropout toggle (opt-out)
    feature_source_dir::Union{Nothing, AbstractString} = nothing,  # when set, the TR/DDI lookup-miss branch featurises novel pairs on-the-fly (real channels for non-human baits); nothing ⇒ zero-impute (byte-identical 9606 default)
    )

    # ---- Species-derived defaults ----
    # The four STRING input filenames are derived from `species` by interpolating
    # the taxonomy ID into the SAME templates that were previously hardcoded at the
    # call signature. For species == 9606 (or a 9606.* poi) the derived
    # strings are byte-identical to the legacy literals. An explicit
    # non-nothing kwarg still wins (back-compat for callers that pin a path).
    sp = species === nothing ? String(split(poi, ".")[1]) : string(species)
    embeddings_seq = embeddings_seq === nothing ? "encodings/$(sp).protein.sequence.embeddings.v12.0.h5" : embeddings_seq
    embeddings_net = embeddings_net === nothing ? "encodings/$(sp).protein.network.embeddings.v12.0.h5" : embeddings_net
    links          = links          === nothing ? "encodings/$(sp).protein.links.detailed.v12.0.onlyAB.txt" : links
    protein_info   = protein_info   === nothing ? "encodings/$(sp).protein.info.v12.0.txt" : protein_info

    # ---- Path resolution ----
    # When the caller (typically `_safe_predict_metalearner` forwarding
    # `config.metalearner_path === nothing`) requests default resolution,
    # dispatch to the schema-matching artefact via the
    # `resolve_metalearner_path(::Nothing; use_mc_dropout)` overload.
    if metalearner_file === nothing
        resolved_path = resolve_metalearner_path(nothing; use_mc_dropout = use_mc_dropout)
        if isnothing(resolved_path)
            default_name = use_mc_dropout ? "metalearner_tr_ddi_mc.jld2" : "metalearner_tr_ddi.jld2"
            error("""
            Default metalearner artefact not found: metalearners/$(default_name)

            This artefact ships with the package — ensure the metalearners/
            directory was installed intact, or pin an explicit path in CONFIG:
              metalearner_path = "/path/to/metalearner.jld2"
            """)
        end
        metalearner_file = resolved_path
    else
        resolved_path = resolve_metalearner_path(metalearner_file)
        if isnothing(resolved_path)
            error("""
            Metalearner file not found: $(metalearner_file)

            Searched locations:
            1. Exact path: $(abspath(metalearner_file))
            2. Package root: $(joinpath(dirname(dirname(@__DIR__)), metalearner_file))
            3. Metalearners dir: $(joinpath(dirname(dirname(@__DIR__)), "metalearners", basename(metalearner_file)))

            Solutions:
            1. Ensure the shipped metalearners/ artefacts are present (reinstall
               the package if the directory is missing or incomplete).

            2. Fix the path in CONFIG:
               metalearner_path = joinpath(@__DIR__, "..", "metalearners", "HistGradientBoosting_tune.jld2")

            3. Use absolute path:
               metalearner_path = "/full/path/to/HistGradientBoosting_tune.jld2"
            """)
        end
        metalearner_file = resolved_path
    end

    # ---- Schema-aware load + dispatch ----
    loaded         = load_metalearner_with_schema(metalearner_file)
    meta           = loaded.mach
    schema_tag     = loaded.schema_tag
    schema_columns = loaded.schema_columns
    # Post-hoc calibrator (identity for legacy artefacts → the
    # :legacy_8feat byte-identical contract is untouched; Platt/isotonic for the
    # two new Stack artefacts so the shipped P(true) is calibrated).
    calibrator     = loaded.calibrator

    if schema_tag === :legacy_8feat
        # DO NOT REFACTOR — byte-identical contract.
        # The body below MUST remain bit-for-bit identical to the legacy path
        # so that `metalearners/HistGradientBoosting_tune.jld2` continues to load
        # and predict byte-identically.

        # Get dataset for prediction
        data, embeddings_data = prediction_data(embeddings_seq, embeddings_net, links, protein_info, poi)

        # generate embedding_matrix by concenating the poi with the prey proteins
        embedding_matrix = zeros(Float32, 2*size(embeddings_data, 1) , size(embeddings_data, 2))
        findpoi = findall(x -> x == poi, data.protein2)

        # Guard: if the POI was renamed by STRING curation (e.g., ENSP* → gene symbol)
        # findpoi will be empty and the broadcast below crashes with DimensionMismatch.
        # Bail out with a warning; the downstream report falls back to bf/(1+bf).
        if isempty(findpoi)
            @warn "predict_metalearner: POI '$poi' not found in prediction data (likely renamed by STRING curation). Skipping metalearner prediction; downstream code falls back to bf/(1+bf)."
            # 3-tuple return (atomic update)
            return nothing, nothing, nothing
        end

        embedding_matrix[1:size(embeddings_data, 1), :]       .= embeddings_data[:, findpoi]
        embedding_matrix[size(embeddings_data, 1) + 1:end, :] .= embeddings_data


        # predict interaction probability based on the deep neural network
        prediction_dnn = predict_DNN(embedding_matrix, model_path, device = device)

        # add prediction_dnn to the data[1] dataframe
        data = hcat(data, prediction_dnn)
        rename!(data, :x1 => :DNN)

        # predict interaction probability based on the meta-learner
        metalearner_prediction = MLJ.predict(meta, data[:, 4:11])
        metalearner_prediction = MLJ.pdf.(metalearner_prediction, Ref(1.0))

        # add ngb_prediction to the data[1] dataframe
        data.MetaClassifier = metalearner_prediction


        if !isfile(output_file)
            writetable(
                output_file,
                "prior" => data
            )
        else
            @warn "$output_file already exists. Data is not replaced. Please delete $output_file first if you want to overwrite it. Then, rerun this command"
        end
        # Expose embedding_matrix as 3rd tuple slot so
        # downstream `_safe_compute_mc_prior!` can reuse it for MC-Dropout passes
        # without re-loading prediction_data (~6 s + 1.5 GB HDF5).
        return data, meta, embedding_matrix
        # END :legacy_8feat — DO NOT REFACTOR

    elseif schema_tag === :tr_ddi
        # ---- :tr_ddi branch (14-feature) ----
        # Build the 8-baseline (7 STRING + DNN) feature slice exactly as the legacy
        # path did, then append the 6 TR+DDI columns from the lookup cache.
        # Production schema:
        #   [neighborhood, fusion, phylogenetic, coexpression, experimental,
        #    database, textmining, DNN,
        #    neighborhood_tr, experiments_tr, database_tr, textmining_tr,
        #    ddi_n_known, ddi_has_known]

        data, embeddings_data = prediction_data(embeddings_seq, embeddings_net, links, protein_info, poi)

        embedding_matrix = zeros(Float32, 2*size(embeddings_data, 1), size(embeddings_data, 2))
        findpoi = findall(x -> x == poi, data.protein2)
        if isempty(findpoi)
            @warn "predict_metalearner: POI '$poi' not found in prediction data (likely renamed by STRING curation). Skipping metalearner prediction; downstream code falls back to bf/(1+bf)."
            return nothing, nothing, nothing
        end

        embedding_matrix[1:size(embeddings_data, 1), :]       .= embeddings_data[:, findpoi]
        embedding_matrix[size(embeddings_data, 1) + 1:end, :] .= embeddings_data

        prediction_dnn = predict_DNN(embedding_matrix, model_path, device = device)
        data = hcat(data, prediction_dnn)
        rename!(data, :x1 => :DNN)

        # ---- Append 6 TR+DDI columns from the lookup cache ----
        # Build / load the lookup once (cached on disk after first call).
        lookup = get_or_build_feature_lookup()

        # The cached lookup above is HUMAN-ONLY (built from the
        # 9606 sources). For a NON-HUMAN bait with staged per-species
        # data, build a per-species TR/DDI lookup ONCE over the candidate pairs
        # (a single file load — NOT the per-pair slow path) and merge it on, so
        # every non-human pair hits real channels instead of zero-imputing. Gated
        # on `sp != "9606"` to keep the human path byte-identical; the
        # per-pair `_onthefly_feature_entry` below stays as the rare fallback for
        # truly-novel pairs.
        if feature_source_dir !== nothing && sp != "9606" && all(isdigit, sp)
            pairs_mat = hcat(String.(data.protein1), String.(data.protein2))
            species_lookup = try
                featurise_pairs_onthefly(pairs_mat, sp;
                    source_dir    = String(feature_source_dir),
                    spike_src_dir = abspath(_default_spike_src_dir()))  # abspath: nested-Base.include trap (same as Step 4)
            catch e
                @warn "predict_metalearner (:tr_ddi): per-species TR/DDI featurisation failed for species $sp; falling back to human lookup + per-pair/zero-impute." exception=e
                Dict{Tuple{String,String},NamedTuple}()
            end
            isempty(species_lookup) || (lookup = merge(lookup, species_lookup))
        end

        n_rows = nrow(data)
        neighborhood_tr = zeros(Float64, n_rows)
        experiments_tr  = zeros(Float64, n_rows)
        database_tr     = zeros(Float64, n_rows)
        textmining_tr   = zeros(Float64, n_rows)
        ddi_n_known     = zeros(Int,     n_rows)
        ddi_has_known   = falses(n_rows)

        n_missing = 0
        for i in 1:n_rows
            a = String(data.protein1[i])
            b = String(data.protein2[i])
            entry = get(lookup, (a, b), nothing)
            if entry === nothing
                entry = get(lookup, (b, a), nothing)
            end
            if entry === nothing
                # Lookup miss. When the caller staged
                # per-species STRING/UniProt/DDI sources, featurise this novel pair
                # on-the-fly so a non-human bait gets REAL TR/DDI channels instead of
                # a silent zero. Otherwise fall back to the zero-impute default
                # (byte-identical to the human path).
                entry = _onthefly_feature_entry(a, b, sp, feature_source_dir)
                if entry === nothing
                    n_missing += 1
                    continue
                end
            end
            neighborhood_tr[i] = getproperty(entry, :neighborhood_tr)
            experiments_tr[i]  = getproperty(entry, :experiments_tr)
            database_tr[i]     = getproperty(entry, :database_tr)
            textmining_tr[i]   = getproperty(entry, :textmining_tr)
            ddi_n_known[i]     = getproperty(entry, :ddi_n_known)
            ddi_has_known[i]   = getproperty(entry, :ddi_has_known)
        end
        if n_missing > 0
            @info "predict_metalearner (:tr_ddi): $(n_missing) of $(n_rows) pairs missing from feature lookup cache; imputed with zero TR + no DDI defaults."
        end

        data.neighborhood_tr = neighborhood_tr
        data.experiments_tr  = experiments_tr
        data.database_tr     = database_tr
        data.textmining_tr   = textmining_tr
        data.ddi_n_known     = ddi_n_known
        data.ddi_has_known   = ddi_has_known

        # ---- Build the 14-column feature slice in canonical order ----
        feature_cols = [:neighborhood, :fusion, :phylogenetic, :coexpression,
                        :experimental, :database, :textmining, :DNN,
                        :neighborhood_tr, :experiments_tr, :database_tr,
                        :textmining_tr, :ddi_n_known, :ddi_has_known]
        data_14 = data[:, feature_cols]
        @assert ncol(data_14) == 14 "expected 14 columns, got $(ncol(data_14))"

        # ---- Schema-column validation ----
        present_cols = Set(string.(names(data_14)))
        expected_cols = Set(schema_columns)
        missing_cols = setdiff(expected_cols, present_cols)
        if !isempty(missing_cols)
            throw(ArgumentError(
                "Schema mismatch: expected columns $(schema_columns), got $(names(data_14)). " *
                "Missing: $(collect(missing_cols))."
            ))
        end

        # `meta` is a self-contained MLJ.Stack
        # whose base-learner packages (EvoTrees, DecisionTree, NearestNeighborModels)
        # were `Base.require`d at runtime by `load_metalearner_with_schema` — which
        # extends the world AFTER this branch was JIT-compiled. A direct
        # `MLJ.predict` would dispatch in the stale world and raise
        # `MethodError: no method matching predict(::EvoTreeClassifier, …)`.
        # `Base.invokelatest` re-resolves dispatch in the current world.
        metalearner_prediction = Base.invokelatest(MLJ.predict, meta, data_14)
        metalearner_prediction = MLJ.pdf.(metalearner_prediction, Ref(1.0))
        # Apply the post-hoc calibrator so the shipped P(true)
        # is calibrated (identity → no-op when the artefact has no calibrator).
        metalearner_prediction = apply_calibrator(calibrator, metalearner_prediction)
        data.MetaClassifier = metalearner_prediction

        if !isfile(output_file)
            writetable(output_file, "prior" => data)
        else
            @warn "$output_file already exists. Data is not replaced. Please delete $output_file first if you want to overwrite it. Then, rerun this command"
        end
        return data, meta, embedding_matrix

    elseif schema_tag === :tr_ddi_mc
        # ---- :tr_ddi_mc branch (15-feature) ----
        # Production schema (MC-Dropout column ON):
        #   [neighborhood, fusion, phylogenetic, coexpression, experimental,
        #    database, textmining, DNN,
        #    neighborhood_tr, experiments_tr, database_tr, textmining_tr,
        #    ddi_n_known, ddi_has_known, mc_std]
        #
        # No hard-error guard. MC-Dropout
        # rides the metalearner's existing DNN dependency (model-473 is loaded below
        # for the `DNN` feature anyway, then reused for the K=30 MC passes). If the
        # extension/model were absent, `predict_metalearner` would never reach this
        # branch with a real artefact — Variante-B fallback handles that upstream.
        data, embeddings_data = prediction_data(embeddings_seq, embeddings_net, links, protein_info, poi)

        embedding_matrix = zeros(Float32, 2*size(embeddings_data, 1), size(embeddings_data, 2))
        findpoi = findall(x -> x == poi, data.protein2)
        if isempty(findpoi)
            @warn "predict_metalearner: POI '$poi' not found in prediction data (likely renamed by STRING curation). Skipping metalearner prediction; downstream code falls back to bf/(1+bf)."
            return nothing, nothing, nothing
        end

        embedding_matrix[1:size(embeddings_data, 1), :]       .= embeddings_data[:, findpoi]
        embedding_matrix[size(embeddings_data, 1) + 1:end, :] .= embeddings_data

        # Load DNN model once + reuse for both deterministic prediction AND
        # MC-Dropout K-pass batch (avoids double model load and double HDF5 read).
        dnn_model = getDNNModel(11, _define_layers(512, 11),
                                _define_activations("relu", 11),
                                0.6641025641025641)
        Flux.loadmodel!(dnn_model, JLD2.load(model_path, "model_state"))
        dnn_model = dnn_model |> device

        # Deterministic prediction for the DNN column (testmode, all dropout off).
        Flux.testmode!(dnn_model)
        prediction_dnn = vec(dnn_model(embedding_matrix |> device) |> Flux.cpu)
        data = hcat(data, prediction_dnn)
        rename!(data, :x1 => :DNN)

        # ---- Append 6 TR+DDI columns from the lookup cache ----
        lookup = get_or_build_feature_lookup()

        # Build + merge a per-species TR/DDI lookup for a
        # non-human bait with staged data (see the :tr_ddi branch above for the
        # full rationale). Gated on `sp != "9606"` to keep the human path
        # byte-identical.
        if feature_source_dir !== nothing && sp != "9606" && all(isdigit, sp)
            pairs_mat = hcat(String.(data.protein1), String.(data.protein2))
            species_lookup = try
                featurise_pairs_onthefly(pairs_mat, sp;
                    source_dir    = String(feature_source_dir),
                    spike_src_dir = abspath(_default_spike_src_dir()))  # abspath: nested-Base.include trap (same as Step 4)
            catch e
                @warn "predict_metalearner (:tr_ddi_mc): per-species TR/DDI featurisation failed for species $sp; falling back to human lookup + per-pair/zero-impute." exception=e
                Dict{Tuple{String,String},NamedTuple}()
            end
            isempty(species_lookup) || (lookup = merge(lookup, species_lookup))
        end

        n_rows = nrow(data)
        neighborhood_tr = zeros(Float64, n_rows)
        experiments_tr  = zeros(Float64, n_rows)
        database_tr     = zeros(Float64, n_rows)
        textmining_tr   = zeros(Float64, n_rows)
        ddi_n_known     = zeros(Int,     n_rows)
        ddi_has_known   = falses(n_rows)

        n_missing = 0
        for i in 1:n_rows
            a = String(data.protein1[i])
            b = String(data.protein2[i])
            entry = get(lookup, (a, b), nothing)
            if entry === nothing
                entry = get(lookup, (b, a), nothing)
            end
            if entry === nothing
                # On-the-fly featurise novel pairs when
                # per-species sources are staged; else zero-impute (9606 default).
                entry = _onthefly_feature_entry(a, b, sp, feature_source_dir)
                if entry === nothing
                    n_missing += 1
                    continue
                end
            end
            neighborhood_tr[i] = getproperty(entry, :neighborhood_tr)
            experiments_tr[i]  = getproperty(entry, :experiments_tr)
            database_tr[i]     = getproperty(entry, :database_tr)
            textmining_tr[i]   = getproperty(entry, :textmining_tr)
            ddi_n_known[i]     = getproperty(entry, :ddi_n_known)
            ddi_has_known[i]   = getproperty(entry, :ddi_has_known)
        end
        if n_missing > 0
            @info "predict_metalearner (:tr_ddi_mc): $(n_missing) of $(n_rows) pairs missing from feature lookup cache; imputed with zero TR + no DDI defaults."
        end

        data.neighborhood_tr = neighborhood_tr
        data.experiments_tr  = experiments_tr
        data.database_tr     = database_tr
        data.textmining_tr   = textmining_tr
        data.ddi_n_known     = ddi_n_known
        data.ddi_has_known   = ddi_has_known

        # ---- Append 15th column: mc_std via mc_dropout_batch ----
        # K=30 passes. Note that mc_dropout_batch returns
        # `var` (not `std`) per its actual signature.
        # We derive `std = sqrt.(var)` at the call site.
        mc_result = mc_dropout_batch(dnn_model, embedding_matrix |> device; K = 30)
        mc_std    = sqrt.(Float64.(mc_result.var))
        data.mc_std = mc_std

        # ---- Build the 15-column feature slice in canonical order ----
        feature_cols = [:neighborhood, :fusion, :phylogenetic, :coexpression,
                        :experimental, :database, :textmining, :DNN,
                        :neighborhood_tr, :experiments_tr, :database_tr,
                        :textmining_tr, :ddi_n_known, :ddi_has_known, :mc_std]
        data_15 = data[:, feature_cols]
        @assert ncol(data_15) == 15 "expected 15 columns, got $(ncol(data_15))"

        # ---- Schema-column validation ----
        present_cols = Set(string.(names(data_15)))
        expected_cols = Set(schema_columns)
        missing_cols = setdiff(expected_cols, present_cols)
        if !isempty(missing_cols)
            throw(ArgumentError(
                "Schema mismatch: expected columns $(schema_columns), got $(names(data_15)). " *
                "Missing: $(collect(missing_cols))."
            ))
        end

        # METAL-INFER-FIX world-age guard (see :tr_ddi branch above) — Stack
        # base-learner packages were `Base.require`d at load time.
        metalearner_prediction = Base.invokelatest(MLJ.predict, meta, data_15)
        metalearner_prediction = MLJ.pdf.(metalearner_prediction, Ref(1.0))
        # Apply the post-hoc calibrator (identity → no-op).
        metalearner_prediction = apply_calibrator(calibrator, metalearner_prediction)
        data.MetaClassifier = metalearner_prediction

        if !isfile(output_file)
            writetable(output_file, "prior" => data)
        else
            @warn "$output_file already exists. Data is not replaced. Please delete $output_file first if you want to overwrite it. Then, rerun this command"
        end
        return data, meta, embedding_matrix

    else
        throw(ArgumentError(
            "Unknown schema_tag: $schema_tag in $(metalearner_file). " *
            "Expected one of (:legacy_8feat, :tr_ddi, :tr_ddi_mc)."
        ))
    end
end


function get_data(dtrain_path, dval_path, dtest_path, model_path)
    if isfile("encodings/meta_learner_data.xlsx")
        data = readtable("encodings/meta_learner_data.xlsx", "data") |> DataFrame
        test_data = readtable("encodings/meta_learner_data.xlsx", "test_data") |> DataFrame
    else
        val_data   = getMetaLearnerDataset(dval_path, model_path)
        train_data = getMetaLearnerDataset(dtrain_path, model_path)
        test_data  = getMetaLearnerDataset(dtest_path, model_path)

        data = val_data#vcat(train_data, val_data)

        writetable(
            "encodings/meta_learner_data.xlsx",
            "data" => data,
            "test_data" => test_data
        )
    end
    return data, test_data
end

function preprocess_data(df::DataFrame)
    df = df[:, 3:end]
    for col in [:neighborhood, :fusion, :phylogenetic, :coexpression, :database, :textmining, :experimental, :DNN]
        df[!, col] = Float64.(df[!, col])
    end
    # Removed OrderedFactor coercion to avoid CategoricalArrays serialization issues with JLD2
    # Labels remain as Float64 (0.0, 1.0) which works with MLJ classifiers and predictions
    return df
end

load_metalearner(path) = MLJ.machine(path)

# ---------------------------------------------------------------------------- #
# Schema-aware metalearner save / load
# ---------------------------------------------------------------------------- #

"""
    save_metalearner_with_schema(mach, path::String; schema_tag::Symbol, schema_columns::Vector{String})

Persist an MLJ machine to `path` together with three sibling metadata keys:

- `schema_tag::Symbol`   — one of `:tr_ddi`, `:tr_ddi_mc` (the legacy 8-feature
  artefact uses the implicit `:legacy_8feat` sentinel on load — saving that
  sentinel is rejected here so callers cannot accidentally produce a stale-schema
  artefact).
- `schema_columns::Vector{String}` — the feature column names in fit order.
- `cache_version::Int` — local artefact format version (independent of
  `AnalysisResult.CACHE_VERSION`); pinned to 1 at introduction.

Internally calls `MLJ.save(path, mach)` and then persists metadata.

Empirical testing on
Julia 1.12 + MLJ.jl current shows that `MLJ.save` writes Julia's native
`Serialization` format (magic bytes `0x37 0x4a 0x4c 0x1e`) at the `.jld2`
path, NOT a JLD2 superblock. Therefore `JLD2.jldopen(path, "a")` raises
`InvalidDataException("Did not find a Superblock.")`. As a fallback,
this implementation persists metadata to a sidecar
`<path>.meta.jld2` file alongside the MLJ machine artefact. The `load_metalearner_with_schema`
counterpart reads the sidecar transparently.

Returns `path` (the MLJ machine path, NOT the sidecar) to preserve back-compat
with callers that expect the primary artefact location.

Metalearner-level recalibration: an optional
`calibrator::MetalearnerCalibrator` is persisted into the sidecar so the shipped
artefact emits CALIBRATED P(true). Defaults to the identity calibrator (no-op).
`MetalearnerCalibrator` is a plain-field struct → JLD2-serialisable directly.

Raises `ArgumentError` if `schema_tag` is not one of the two allowed values.
"""
function save_metalearner_with_schema(mach, path::String; schema_tag::Symbol,
        schema_columns::Vector{String},
        calibrator::MetalearnerCalibrator = _identity_calibrator())
    schema_tag ∈ (:tr_ddi, :tr_ddi_mc) || throw(ArgumentError(
        "save_metalearner_with_schema: schema_tag must be :tr_ddi or :tr_ddi_mc; got $(schema_tag). " *
        "The :legacy_8feat sentinel is reserved for load-side detection of legacy artefacts and must not be written."))
    MLJ.save(path, mach)
    sidecar = path * ".meta.jld2"
    JLD2.jldopen(sidecar, "w") do f
        f["schema_tag"]     = schema_tag
        f["schema_columns"] = schema_columns
        f["cache_version"]  = 2          # bumped for `calibrator` key
        f["calibrator"]     = calibrator # MetalearnerCalibrator (Platt/isotonic/identity)
    end
    return path
end

"""
    load_metalearner_with_schema(path::String) -> NamedTuple{(:mach, :schema_tag, :schema_columns)}

Load an MLJ machine together with its schema metadata.

When a sidecar `<path>.meta.jld2` exists alongside the MLJ machine artefact
(sidecar fallback; see `save_metalearner_with_schema` for rationale),
the returned `schema_tag` reflects the sidecar's on-disk value (`:tr_ddi` or
`:tr_ddi_mc`). When the sidecar is absent (legacy artefacts such as
`metalearners/HistGradientBoosting_tune.jld2`),
the loader falls back to `schema_tag = :legacy_8feat` and the canonical 8-column
list `["neighborhood", "fusion", "phylogenetic", "coexpression", "experimental",
"database", "textmining", "DNN"]` so byte-identical back-compat is preserved.

Reconstructs the MLJ machine via `MLJ.machine(path)` — same call path as the
legacy `load_metalearner(path)` helper.
"""
function load_metalearner_with_schema(path::String)
    # Canonical legacy schema (matches the 8-column list in
    # preprocess_data line 425 and predict_metalearner's legacy code path).
    legacy_columns = String["neighborhood", "fusion", "phylogenetic", "coexpression",
                            "experimental", "database", "textmining", "DNN"]
    schema_tag::Symbol = :legacy_8feat
    schema_columns::Vector{String} = legacy_columns
    # Optional post-hoc calibrator. Defaults to identity (no-op)
    # so legacy artefacts (no sidecar, or cache_version 1 with no `calibrator`
    # key) are byte-identical — the :legacy_8feat contract is never touched.
    calibrator::MetalearnerCalibrator = _identity_calibrator()
    sidecar = path * ".meta.jld2"
    if isfile(sidecar)
        JLD2.jldopen(sidecar, "r") do f
            if haskey(f, "schema_tag")
                schema_tag = f["schema_tag"]
            end
            if haskey(f, "schema_columns")
                schema_columns = f["schema_columns"]
            end
            if haskey(f, "calibrator")
                calibrator = f["calibrator"]
            end
        end
    end

    # The :tr_ddi / :tr_ddi_mc production
    # artefacts are self-contained `MLJ.Stack` machines that embed the 6
    # base-learner model objects (HistGB + EvoTrees + Logistic + KNN +
    # RandomForest + ExtraTrees). `MLJ.machine(path)` uses Julia native
    # `Serialization`, which can only reconstruct a model struct when its
    # DEFINING package is loaded in the current world. The extension's own
    # `using` block only pulls Flux + MLJ + MLJScikitLearnInterface + HDF5, so
    # EvoTrees / DecisionTree / NearestNeighborModels are absent → deserialise
    # throws `KeyError: key Base.PkgId(... "EvoTrees") not found`. Pre-load the
    # 6 base-learner model types via `MLJ.@load` BEFORE deserialising so all
    # embedded struct types resolve. `:legacy_8feat` is a bare LR machine and
    # needs none of this (MLJScikitLearnInterface already loaded).
    if schema_tag === :tr_ddi || schema_tag === :tr_ddi_mc
        _ensure_stack_base_learners_loaded()
    end

    mach = MLJ.machine(path)
    return (mach = mach, schema_tag = schema_tag, schema_columns = schema_columns,
            calibrator = calibrator)
end

# Pre-load the Stack base-learner packages so a serialised `MLJ.Stack`
# deserialises cleanly in a fresh process.
#
# `MLJ.machine(path)` deserialises via Julia native `Serialization`, which
# reconstructs each embedded model struct by looking its DEFINING package up in
# `Base.loaded_modules` keyed by `PkgId`. If the package is absent the
# deserialiser throws `KeyError: key Base.PkgId(... "EvoTrees") not found`.
# `MLJ.@load` cannot be used here — it is a parse-time macro that `import`s into
# the surrounding MODULE scope and does nothing useful when expanded inside a
# function body at runtime. We therefore call `Base.require(PkgId)` directly,
# which is exactly what `import`/`using` lower to and brings the package module
# into `Base.loaded_modules`. UUIDs are pinned from `examples/Project.toml`
# (HistGB/Logistic/ExtraTrees come from MLJScikitLearnInterface, already loaded
# by the extension's `using`). Idempotent + cheap once loaded.
const _STACK_BASE_LEARNER_PKGS = (
    Base.PkgId(Base.UUID("f6006082-12f8-11e9-0c9c-0d5d367ab1e5"), "EvoTrees"),
    Base.PkgId(Base.UUID("636a865e-7cf4-491e-846c-de09b730eb36"), "NearestNeighborModels"),
    Base.PkgId(Base.UUID("c6f25543-311c-4c74-83dc-3ea6d1015661"), "MLJDecisionTreeInterface"),
    Base.PkgId(Base.UUID("7806a523-6efd-50cb-b5f6-3fa6f1930dbb"), "DecisionTree"),
)

function _ensure_stack_base_learners_loaded()
    for pkgid in _STACK_BASE_LEARNER_PKGS
        try
            Base.require(pkgid)
        catch e
            @warn "Could not load Stack base-learner package $(pkgid.name); a \
                   self-contained Stack artefact may fail to deserialise. Ensure \
                   EvoTrees, NearestNeighborModels, MLJDecisionTreeInterface, and \
                   DecisionTree are in the active project." exception=e
        end
    end
    return nothing
end


function validate_metalearner(mach, test_data)
  ŷ = MLJ.predict(mach, test_data[:, 1:8])

    @info "Test accuracy: $(MLJ.accuracy(mode.(ŷ), test_data.label))"
    @info "Test auc: $(MLJ.auc(ŷ, test_data.label))"
    @info "Test brier loss: $(mean(MLJ.brier_loss(ŷ, test_data.label)))"
    @info "MCC: $(MLJ.mcc(mode.(ŷ), test_data.label))"


    # ROC curve
    curve = MLJ.roc_curve(ŷ, test_data.label)
    roc_plot = StatsPlots.plot(
      curve, title = "ROC Curve", 
      label = "ROC", 
      xlabel = "False Positive Rate", 
      ylabel = "True Positive Rate"
      )

    StatsPlots.plot!(
      collect(0:0.01:1), 
      collect(0:0.01:1), 
      label = "Random", 
      linestyle = :dash, 
      linewidth = 2
    )
    return roc_plot

end



# ---------------------------------------------------------------------------- #
# Schema-aware column slicing helper
#
# The metalearner training tooling builds a 14-column DataFrame for the TR+DDI
# production schema and a 15-column DataFrame for the optional MC-Dropout
# variant. The
# existing `:legacy_8feat` slice `data[:, 1:8]` is preserved byte-identically for
# back-compat with the existing artefacts (`HistGradientBoosting_tune.jld2`,
# `LogisticClassifier_tune.jld2`, `GaussianNBC_tune.jld2`, `ensemble.jld2`).
# ---------------------------------------------------------------------------- #

"""
    _columns_for_schema(data, schema_tag::Symbol)

Return the feature-column slice of `data` matching the requested `schema_tag`:

- `:legacy_8feat` → `data[:, 1:8]`   (legacy production schema)
- `:tr_ddi`       → `data[:, 1:14]`  (TR+DDI production schema)
- `:tr_ddi_mc`    → `data[:, 1:15]`  (optional MC-Dropout schema; gated)

Throws `ArgumentError` for any other tag. The `:legacy_8feat` default is preserved
so callers that do NOT pass `schema_tag` get byte-identical behaviour to the
legacy fit functions (back-compat contract).
"""
function _columns_for_schema(data, schema_tag::Symbol)
    if schema_tag === :legacy_8feat
        return data[:, 1:8]
    elseif schema_tag === :tr_ddi
        return data[:, 1:14]
    elseif schema_tag === :tr_ddi_mc
        return data[:, 1:15]
    else
        throw(ArgumentError(
            "Unknown schema_tag: $schema_tag — expected one of (:legacy_8feat, :tr_ddi, :tr_ddi_mc)."
        ))
    end
end


function fit_HistGradientBoostingClassifier(
    dtrain_path = "encodings/train_data.h5",
    dval_path   = "encodings/val_data.h5",
    dtest_path  = "encodings/test_data.h5",
    model_path   = _dnn_model_path();
    schema_tag::Symbol = :legacy_8feat,
    measure = MLJ.BrierLoss(),
    nfolds::Int = 5,
    )


    # ---------------- Data handling ----------------- #
    data, test_data = get_data(dtrain_path, dval_path, dtest_path, model_path)
    data = preprocess_data(data)
    test_data = preprocess_data(test_data)

    # ---------------- Model set up ----------------- #
    HGB = MLJ.@load HistGradientBoostingClassifier pkg=MLJScikitLearnInterface
    model = HGB(max_iter = 100)

    # ---------------- hyperparameter tuning ----------------- #
    # define hyperparameter ranges
    lr_range = MLJ.range(
      model, :learning_rate,
      lower = 1e-6, upper = 1e-1
      )

    max_leaf_nodes = MLJ.range(
      model, :max_leaf_nodes,
      lower = 10, upper = 1000
    )

    max_depth = MLJ.range(
      model, :max_depth,
      lower = 1, upper = 10
      )

    l2_regularization = MLJ.range(
      model, :l2_regularization,
      lower = 1e-6, upper = 1e-1
      )

    # define latin hypercube
    latin = MLJ.LatinHypercube(gens = 5, popsize = 120)

    self_tuning_model = MLJ.TunedModel(
        model,
        resampling = MLJ.StratifiedCV(nfolds = nfolds, shuffle = true),
        tuning = latin,
        range = [lr_range, max_leaf_nodes, max_depth, l2_regularization],
        measure = measure,
        n = 1000 # maximum number of models to evaluate
    )

    mach = MLJ.machine(self_tuning_model, _columns_for_schema(data, schema_tag), data.label)
    @info "Tuning $(nameof(typeof(model)))..."
    MLJ.fit!(mach; verbosity = 1)

    StatsPlots.plot(mach)

    # ---------------- Model evaluation ----------------- #
    validate_metalearner(mach, test_data)

    # ---------------- Model saving ----------------- #
    out_path = schema_tag === :legacy_8feat ?
        "metalearners/HistGradientBoosting_tune.jld2" :
        "metalearners/HistGradientBoosting_tune_$(string(schema_tag)).jld2"
    MLJ.save(out_path, mach)

    return mach
end


function fit_LogisticClassifier(
  dtrain_path = "encodings/train_data.h5",
  dval_path   = "encodings/val_data.h5",
  dtest_path  = "encodings/test_data.h5",
  model_path   = _dnn_model_path();
  schema_tag::Symbol = :legacy_8feat,
  measure = MLJ.BrierLoss(),
  nfolds::Int = 5,
)

  # ---------------- Data handling ----------------- #
  data, test_data = get_data(dtrain_path, dval_path, dtest_path, model_path)
  data = preprocess_data(data)
  test_data = preprocess_data(test_data)

  # ---------------- Model set up ----------------- #
  LR = MLJ.@load LogisticClassifier pkg=MLJScikitLearnInterface
  model = LR()

  # ---------------- hyperparameter tuning ----------------- #
  # define hyperparameter ranges
  range_max_iter= MLJ.range(
    model, :max_iter,
    lower = 50, upper = 10_000
  )


  latin = MLJ.LatinHypercube(gens = 5, popsize = 120)

  self_tuning_model = MLJ.TunedModel(
    model,
    resampling = MLJ.StratifiedCV(nfolds = nfolds, shuffle = true),
    tuning = latin,
    range = [range_max_iter],
    measure = measure,
    n = 1000
  )

  mach = MLJ.machine(self_tuning_model, _columns_for_schema(data, schema_tag), data.label)
  @info "Tuning $(nameof(typeof(model)))..."
  MLJ.fit!(mach; verbosity = 0)

  StatsPlots.plot(mach)

  # ---------------- Model evaluation ----------------- #
  validate_metalearner(mach, test_data)

  # ---------------- Model saving ----------------- #
  out_path = schema_tag === :legacy_8feat ?
      "metalearners/LogisticClassifier_tune.jld2" :
      "metalearners/LogisticClassifier_tune_$(string(schema_tag)).jld2"
  MLJ.save(out_path, mach)

  return mach

end

function fit_GaussianNBClassifier(
  dtrain_path = "encodings/train_data.h5",
  dval_path   = "encodings/val_data.h5",
  dtest_path  = "encodings/test_data.h5",
  model_path   = _dnn_model_path();
  schema_tag::Symbol = :legacy_8feat,
  measure = MLJ.BrierLoss(),
  nfolds::Int = 5,
)
  # ---------------- Data handling ----------------- #
  data, test_data = get_data(dtrain_path, dval_path, dtest_path, model_path)
  data = preprocess_data(data)
  test_data = preprocess_data(test_data)

  # ---------------- Model set up ----------------- #
  m = MLJ.@load GaussianNBClassifier pkg=MLJScikitLearnInterface
  model = m()

  range_var_smoothing = MLJ.range(
    model, :var_smoothing, lower = 1e-12, upper = 0.25
    )

  latin = MLJ.LatinHypercube(gens = 5, popsize = 120)

  self_tuning_model = MLJ.TunedModel(
    model,
    resampling = MLJ.StratifiedCV(nfolds = nfolds, shuffle = true),
    tuning = latin,
    range = [range_var_smoothing],
    measure = measure,
    n = 1000
  )

  mach = MLJ.machine(self_tuning_model, _columns_for_schema(data, schema_tag), data.label)
  @info "Tuning $(nameof(typeof(model)))..."
  MLJ.fit!(mach; verbosity = 1)

  StatsPlots.plot(mach)

  # ---------------- Model evaluation ----------------- #
  validate_metalearner(mach, test_data)

  # ---------------- Model saving ----------------- #
  out_path = schema_tag === :legacy_8feat ?
      "metalearners/GaussianNBC_tune.jld2" :
      "metalearners/GaussianNBC_tune_$(string(schema_tag)).jld2"
  MLJ.save(out_path, mach)
  return mach

end


function fit_Ensemble(
  dtrain_path = "encodings/train_data.h5",
  dval_path   = "encodings/val_data.h5",
  dtest_path  = "encodings/test_data.h5",
  model_path   = _dnn_model_path()
  )


  # ---------------- Data handling ----------------- #
  data, test_data = get_data(dtrain_path, dval_path, dtest_path, model_path)
  data = preprocess_data(data)
  test_data = preprocess_data(test_data)

  # ---------------- Model set up ----------------- #

  # Logistic classifier
  LR = MLJ.@load LogisticClassifier pkg=MLJScikitLearnInterface
  lr = LR(max_iter = 100)

  # GaussianNBClassifier
  gaussian_nbc = MLJ.@load GaussianNBClassifier pkg=MLJScikitLearnInterface
  gaussian = gaussian_nbc(var_smoothing = 1.0e-9)

  # HistGradienBossting
  HGB = MLJ.@load HistGradientBoostingClassifier pkg=MLJScikitLearnInterface
  hgb = HGB(
    max_iter            = 100, 
    learning_rate       = 0.1, 
    max_leaf_nodes      = 31, 
    l2_regularization   = 0.0
    )

  # Stack
  stack = MLJ.Stack(;
    metalearner = LR(),
    lr          = lr,
    gaussian    = gaussian,
    hgb         = hgb,
    measure     = MLJ.BrierLoss(),
    resampling  = MLJ.StratifiedCV(nfolds = 5, shuffle = true)
  )
    
  mach = MLJ.machine(stack, data[:, 1:8], data.label)


  # ---------------- Model evaluation ----------------- #
  MLJ.fit!(mach; verbosity = 1)

  # --------------- Model ----------------------------- #
  validate_metalearner(mach, test_data)

    # ---------------- Model saving ----------------- #
  MLJ.save("metalearners/ensemble.jld2", mach)
end


# ---------------------------------------------------------------------------- #
# fit_* helpers for the 6-candidate pool
#
# These four helpers (EvoTrees, kNN, RandomForest, ExtraTrees) plus the
# existing fit_HistGradientBoostingClassifier + fit_LogisticClassifier form the
# 6-candidate level-1 base-learner pool that replaces the earlier AdaBoost
# + GaussianNB pool (AdaBoost ECE 0.12, GaussianNB ECE 0.22 —
# structural mis-calibration that survives hyperparameter tuning).
#
# Signature contract (uniform across the four helpers):
#
#     fit_<Name>(data, target, output_path::String;
#                schema_tag::Symbol = :tr_ddi,
#                measure = MLJ.BrierLoss(),
#                nfolds::Int = 5)
#
# `data` is a DataFrame whose first 14 columns are the TR+DDI production schema
# (or 15 for `:tr_ddi_mc`); `target` is the binary label vector. The function
# slices the feature matrix via `_columns_for_schema(data, schema_tag)`, runs an
# MLJ.TunedModel LatinHypercube search with the provided `measure`, fits via
# `MLJ.fit!`, persists the trained `mach` via `MLJ.save(output_path, mach)`, and
# returns the fitted machine.
#
# The `measure` kwarg is threaded explicitly so a caller can flip to
# `MLJ.LogLoss()` deterministically — no hardcoded
# BrierLoss inside the body.
# ---------------------------------------------------------------------------- #

"""
    fit_EvoTreesClassifier(data, target, output_path; schema_tag, measure, nfolds)

Level-1 base learner: gradient-boosted decision trees via
EvoTrees.jl. Modelled on `fit_HistGradientBoostingClassifier`.
"""
function fit_EvoTreesClassifier(data, target, output_path::String;
        schema_tag::Symbol = :tr_ddi,
        measure = MLJ.BrierLoss(),
        nfolds::Int = 5,
    )
    ET = MLJ.@load EvoTreeClassifier pkg=EvoTrees
    model = ET()

    eta_range       = MLJ.range(model, :eta,       lower = 1e-3, upper = 1.0)
    max_depth_range = MLJ.range(model, :max_depth, lower = 2,    upper = 10)
    nrounds_range   = MLJ.range(model, :nrounds,   lower = 50,   upper = 500)

    latin = MLJ.LatinHypercube(gens = 5, popsize = 120)

    self_tuning_model = MLJ.TunedModel(
        model,
        resampling = MLJ.StratifiedCV(nfolds = nfolds, shuffle = true),
        tuning     = latin,
        range      = [eta_range, max_depth_range, nrounds_range],
        measure    = measure,
        n          = 1000,
    )

    mach = MLJ.machine(self_tuning_model, _columns_for_schema(data, schema_tag), target)
    @info "Tuning $(nameof(typeof(model)))..."
    MLJ.fit!(mach; verbosity = 1)

    MLJ.save(output_path, mach)
    return mach
end


"""
    fit_KNNClassifier(data, target, output_path; schema_tag, measure, nfolds)

Level-1 base learner: k-nearest-neighbours via
NearestNeighborModels.jl. Modelled on `fit_HistGradientBoostingClassifier`.
"""
function fit_KNNClassifier(data, target, output_path::String;
        schema_tag::Symbol = :tr_ddi,
        measure = MLJ.BrierLoss(),
        nfolds::Int = 5,
    )
    KNN = MLJ.@load KNNClassifier pkg=NearestNeighborModels
    model = KNN()

    k_range      = MLJ.range(model, :K,         lower = 3,   upper = 50)
    leafsize_rng = MLJ.range(model, :leafsize,  lower = 5,   upper = 60)

    latin = MLJ.LatinHypercube(gens = 5, popsize = 120)

    self_tuning_model = MLJ.TunedModel(
        model,
        resampling = MLJ.StratifiedCV(nfolds = nfolds, shuffle = true),
        tuning     = latin,
        range      = [k_range, leafsize_rng],
        measure    = measure,
        n          = 1000,
    )

    mach = MLJ.machine(self_tuning_model, _columns_for_schema(data, schema_tag), target)
    @info "Tuning $(nameof(typeof(model)))..."
    MLJ.fit!(mach; verbosity = 1)

    MLJ.save(output_path, mach)
    return mach
end


"""
    fit_RandomForestClassifier(data, target, output_path; schema_tag, measure, nfolds)

Level-1 base learner: random forest via DecisionTree.jl
(MLJDecisionTreeInterface). Modelled on `fit_HistGradientBoostingClassifier`.
"""
function fit_RandomForestClassifier(data, target, output_path::String;
        schema_tag::Symbol = :tr_ddi,
        measure = MLJ.BrierLoss(),
        nfolds::Int = 5,
    )
    RF = MLJ.@load RandomForestClassifier pkg=MLJDecisionTreeInterface
    model = RF()

    n_trees_range    = MLJ.range(model, :n_trees,           lower = 50,  upper = 500)
    max_depth_range  = MLJ.range(model, :max_depth,         lower = -1,  upper = 30)
    min_samples_rng  = MLJ.range(model, :min_samples_split, lower = 2,   upper = 20)

    latin = MLJ.LatinHypercube(gens = 5, popsize = 120)

    self_tuning_model = MLJ.TunedModel(
        model,
        resampling = MLJ.StratifiedCV(nfolds = nfolds, shuffle = true),
        tuning     = latin,
        range      = [n_trees_range, max_depth_range, min_samples_rng],
        measure    = measure,
        n          = 1000,
    )

    mach = MLJ.machine(self_tuning_model, _columns_for_schema(data, schema_tag), target)
    @info "Tuning $(nameof(typeof(model)))..."
    MLJ.fit!(mach; verbosity = 1)

    MLJ.save(output_path, mach)
    return mach
end


"""
    fit_ExtraTreesClassifier(data, target, output_path; schema_tag, measure, nfolds)

Level-1 base learner: extremely-randomised trees via
MLJScikitLearnInterface. Modelled on `fit_HistGradientBoostingClassifier`.
"""
function fit_ExtraTreesClassifier(data, target, output_path::String;
        schema_tag::Symbol = :tr_ddi,
        measure = MLJ.BrierLoss(),
        nfolds::Int = 5,
    )
    XT = MLJ.@load ExtraTreesClassifier pkg=MLJScikitLearnInterface
    model = XT()

    n_estimators_rng = MLJ.range(model, :n_estimators,     lower = 50,  upper = 500)
    max_depth_rng    = MLJ.range(model, :max_depth,        lower = 2,   upper = 30)
    min_samples_rng  = MLJ.range(model, :min_samples_split, lower = 2,  upper = 20)

    latin = MLJ.LatinHypercube(gens = 5, popsize = 120)

    self_tuning_model = MLJ.TunedModel(
        model,
        resampling = MLJ.StratifiedCV(nfolds = nfolds, shuffle = true),
        tuning     = latin,
        range      = [n_estimators_rng, max_depth_rng, min_samples_rng],
        measure    = measure,
        n          = 1000,
    )

    mach = MLJ.machine(self_tuning_model, _columns_for_schema(data, schema_tag), target)
    @info "Tuning $(nameof(typeof(model)))..."
    MLJ.fit!(mach; verbosity = 1)

    MLJ.save(output_path, mach)
    return mach
end


# ---------------------------------------------------------------------------- #
# Sill-2009 feature-weighted stacking blender
#
# LogisticClassifier with L2 penalty as the level-2 blender,
# fed BOTH the OOF predictions of the level-1 base learners AND the original
# 14-column feature matrix (Sill et al. 2009 feature-weighted stacking; ECE
# 0.0259 on the C6_TR_DDI test fold). The plain "predictions-only" stacking
# variant lost on calibration.
#
# Input contract:
#   - `oof_predictions::AbstractMatrix` — n_rows × n_base columns of OOF
#     probabilities from the 6 level-1 base learners (training script computes
#     these via 5-fold CV).
#   - `original_features` — DataFrame or Matrix with the 14-column TR+DDI input
#     used at level-1 (15 columns when `schema_tag === :tr_ddi_mc`). Sliced via
#     `_columns_for_schema(original_features, schema_tag)`.
#   - `target` — the binary label vector.
#   - `output_path` — JLD2 path; `MLJ.save` writes the legacy MLJ-Serialization
#     format. The schema-aware sidecar is written by the caller via
#     `save_metalearner_with_schema(...)`.
# ---------------------------------------------------------------------------- #

"""
    fit_LR_L2_blender(oof_predictions, original_features, target, output_path; schema_tag, measure, nfolds)

Level-2 LogisticClassifier with L2 penalty
fitted on `[oof_predictions  original_features]` (Sill-2009 feature-weighted
stacking).

Returns the fitted MLJ machine.
"""
function fit_LR_L2_blender(oof_predictions, original_features, target, output_path::String;
        schema_tag::Symbol = :tr_ddi,
        measure = MLJ.BrierLoss(),
        nfolds::Int = 5,
    )
    feat_slice = _columns_for_schema(original_features, schema_tag)

    # Build the level-2 input matrix: [oof_predictions  original_features].
    oof_df = DataFrame(oof_predictions, :auto)
    feat_df = feat_slice isa DataFrame ? feat_slice : DataFrame(feat_slice, :auto)
    blender_input = hcat(oof_df, feat_df; makeunique = true)

    LR = MLJ.@load LogisticClassifier pkg=MLJScikitLearnInterface
    # L2 penalty. The scikit-learn LogisticRegression default
    # is :l2 already; pin it here for self-documenting code.
    model = LR(penalty = "l2", max_iter = 1_000)

    range_C        = MLJ.range(model, :C,        lower = 1e-3,  upper = 1e3, scale = :log10)
    range_max_iter = MLJ.range(model, :max_iter, lower = 100,   upper = 10_000)

    latin = MLJ.LatinHypercube(gens = 5, popsize = 120)

    self_tuning_model = MLJ.TunedModel(
        model,
        resampling = MLJ.StratifiedCV(nfolds = nfolds, shuffle = true),
        tuning     = latin,
        range      = [range_C, range_max_iter],
        measure    = measure,
        n          = 1000,
    )

    mach = MLJ.machine(self_tuning_model, blender_input, target)
    @info "Tuning LR_L2 feature-weighted-stacking blender..."
    MLJ.fit!(mach; verbosity = 1)

    MLJ.save(output_path, mach)
    return mach
end