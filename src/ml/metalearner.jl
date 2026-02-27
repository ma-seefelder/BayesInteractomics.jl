###########################################################################################
# Training of and XGBoost classifier with isotonic regression
# to combined predictions from the DNN with the other features 
# (see below) 

# Author: Dr. rer. nat. Manuel Seefelder
# Date: 27th June 2025

###########################################################################################################

# Additional imports needed (AbstractDataFrame, DataFrame, rename! already imported in main module)
import DataFrames: innerjoin, leftjoin
import HDF5, Flux
import JLD2: load
import MLJ
import MLJScikitLearnInterface
import StatsPlots

# Note: Required functions from utils.jl are already loaded by main module
# include("../core/utils.jl")
# include("../dnn/model.jl")

# Environment variables
dtrain_path = "encodings/train_data.h5";
dval_path   = "encodings/val_data.h5";
dtest_path  = "encodings/test_data.h5";
const MODELPATH   = "encodings/model-473-0.5414302830201915.jld2"

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

    # Try relative to package root (assuming metalearner.jl is in src/ml/)
    pkg_root = dirname(dirname(@__DIR__))
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

# load model definitions
function predict_DNN(DATASET::String, model_path::String = MODELPATH; device = Flux.cpu)
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
  d::Matrix{F}, model_path::String = MODELPATH; 
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
    protein_info_df = read(protein_info, DataFrame)
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
    links_scores = read(links, DataFrame)
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
- `model_path::String = "encodings/model-927-0.4975538112736281.jld2"`: Path to the pre-trained DNN model file.
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
function predict_metalearner(
    poi::String;
    model_path        = MODELPATH,
    embeddings_seq    = "encodings/9606.protein.sequence.embeddings.v12.0.h5",
    embeddings_net    = "encodings/9606.protein.network.embeddings.v12.0.h5",
    links             = "encodings/9606.protein.links.detailed.v12.0.onlyAB.txt",
    protein_info      = "encodings/9606.protein.info.v12.0.txt",
    device            = Flux.cpu,
    output_file       = "prior.xlsx",
    metalearner_file  = "metalearners/HistGradientBoosting_tune.jld2"
    )

    # load metalearner with smart path resolution
    resolved_path = resolve_metalearner_path(metalearner_file)
    if isnothing(resolved_path)
        error("""
        Metalearner file not found: $(metalearner_file)

        Searched locations:
        1. Exact path: $(abspath(metalearner_file))
        2. Package root: $(joinpath(dirname(dirname(@__DIR__)), metalearner_file))
        3. Metalearners dir: $(joinpath(dirname(dirname(@__DIR__)), "metalearners", basename(metalearner_file)))

        Solutions:
        1. Train metalearner first:
           julia> include("metalearners/train_all_metalearners.jl")

        2. Fix the path in CONFIG:
           metalearner_path = joinpath(@__DIR__, "..", "metalearners", "HistGradientBoosting_tune.jld2")

        3. Use absolute path:
           metalearner_path = "/full/path/to/HistGradientBoosting_tune.jld2"
        """)
    end
    metalearner_file = resolved_path

    meta = load_metalearner(metalearner_file)
    
    # Get dataset for prediction
    data, embeddings_data = prediction_data(embeddings_seq, embeddings_net, links, protein_info, poi)

    # generate embedding_matrix by concenating the poi with the prey proteins
    embedding_matrix = zeros(Float32, 2*size(embeddings_data, 1) , size(embeddings_data, 2))
    findpoi = findall(x -> x == poi, data.protein2)

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
    return data, meta
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



function fit_HistGradientBoostingClassifier(
    dtrain_path = "encodings/train_data.h5",
    dval_path   = "encodings/val_data.h5",
    dtest_path  = "encodings/test_data.h5",
    model_path   = MODELPATH
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
        resampling = MLJ.StratifiedCV(nfolds = 5, shuffle = true),
        tuning = latin,
        range = [lr_range, max_leaf_nodes, max_depth, l2_regularization],
        measure = MLJ.BrierLoss(),
        n = 1000 # maximum number of models to evaluate
    )

    mach = MLJ.machine(self_tuning_model, data[:, 1:8], data.label)
    @info "Tuning $(nameof(typeof(model)))..."
    MLJ.fit!(mach; verbosity = 1)
    
    StatsPlots.plot(mach)

    # ---------------- Model evaluation ----------------- #
    validate_metalearner(mach, test_data)

    # ---------------- Model saving ----------------- #
    MLJ.save("metalearners/HistGradientBoosting_tune.jld2", mach)

    return mach
end


function fit_LogisticClassifier(
  dtrain_path = "encodings/train_data.h5",
  dval_path   = "encodings/val_data.h5",
  dtest_path  = "encodings/test_data.h5",
  model_path   = MODELPATH
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
    resampling = MLJ.StratifiedCV(nfolds = 5, shuffle = true),
    tuning = latin,
    range = [range_max_iter],
    measure = MLJ.BrierLoss(),
    n = 1000
  )

  mach = MLJ.machine(self_tuning_model, data[:, 1:8], data.label)
  @info "Tuning $(nameof(typeof(model)))..."
  MLJ.fit!(mach; verbosity = 0)
    
  StatsPlots.plot(mach)

  # ---------------- Model evaluation ----------------- #
  validate_metalearner(mach, test_data)

  # ---------------- Model saving ----------------- #
  MLJ.save("metalearners/LogisticClassifier_tune.jld2", mach)

  return mach

end

function fit_GaussianNBClassifier(
  dtrain_path = "encodings/train_data.h5",
  dval_path   = "encodings/val_data.h5",
  dtest_path  = "encodings/test_data.h5",
  model_path   = MODELPATH
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
    resampling = MLJ.StratifiedCV(nfolds = 5, shuffle = true),
    tuning = latin,
    range = [range_var_smoothing],
    measure = MLJ.BrierLoss(),
    n = 1000
  )
  
  mach = MLJ.machine(self_tuning_model, data[:, 1:8], data.label)
  @info "Tuning $(nameof(typeof(model)))..."
  MLJ.fit!(mach; verbosity = 1)
    
  StatsPlots.plot(mach)

  # ---------------- Model evaluation ----------------- #
  validate_metalearner(mach, test_data)

  # ---------------- Model saving ----------------- #
  MLJ.save("metalearners/GaussianNBC_tune.jld2", mach)
  return mach

end


function fit_Ensemble(
  dtrain_path = "encodings/train_data.h5",
  dval_path   = "encodings/val_data.h5",
  dtest_path  = "encodings/test_data.h5",
  model_path   = MODELPATH
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