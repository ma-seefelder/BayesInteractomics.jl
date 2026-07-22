import Pkg; Pkg.activate("./src/dnn/"); 
# General imports
import Dates, Random
import JLD2: jldsave, jldopen
import ProgressMeter
import LinearAlgebra: dot, norm
import Statistics: mean
import Distributions: Beta
import ThreadsX
import HTTP

# DL imports
import Flux
using CUDA
CUDA.allowscalar(false)
# Data Handling
using DataFrames
import CSV
import HDF5
# Weights & Biases logging
using Wandb, Logging
# Hyperparameter Sweep
using Hyperopt

include("model.jl")
# ---------------------------------------------------------------------------- #
# Function to query the STRING database for biological informed scores
# ---------------------------------------------------------------------------- #
"""
    callSTRINGAPI(proteins::Matrix{String}; outfile::Union{String, Nothing} = nothing, tmp_file::String = "string_tmp.tsv")

Queries the STRING database for interaction partners for a list of proteins,
processes the results, and saves them to an HDF5 file.

This function takes a vector of protein identifiers, calls the STRING API to
fetch interaction data, parses the resulting TSV file, and stores the
biologically relevant scores in a structured HDF5 file.

# Arguments
- `proteins::Vector{String}`: A vector of protein identifiers (e.g., STRING or ENSEMBL IDs) to query.

# Keyword Arguments
- `outfile::String`: The path for the output HDF5 file. If the file already exists, it will be deleted and recreated. Defaults to `nothing`. If `nothing*, the function will return the processed data as a `DataFrame`.
- `tmp_file::String = "string_tmp.tsv"`: Path for the temporary TSV file downloaded from STRING. It will be deleted and recreated if it already exists.

# Returns
- `DataFrame`: A DataFrame containing the processed interaction data with columns for protein pairs and various interaction scores (`:neighborhood`, `:fusion`, etc.).

# Details
The function performs the following steps:
1.  Constructs a URL to query the STRING API's `interaction_partners` endpoint.
2.  Downloads the result as a TSV file to `tmp_file`.
3.  Parses the TSV into a `DataFrame`, selecting and renaming columns for clarity.
4.  Saves the data to an HDF5 file specified by `outfile`. The HDF5 file contains a group named `scores` with two datasets:
    - `scores/scores`: A `Matrix{Float32}` containing the numeric interaction scores.
    - `scores/annotation`: A `Matrix{String}` containing the corresponding protein pairs (`Protein1`, `Protein2`).

"""
function callSTRINGAPI(identifiers::Vector{String}; tmp_file::String = "string_tmp.tsv")
    # If the input list is empty, no need to make a call.
    if isempty(identifiers)
        @warn "Received an empty list of identifiers. Skipping API call."
        return DataFrame()
    end

    # --- FIX: Join identifiers first, then URL-escape the entire payload ---
    # This prevents special characters in protein IDs from corrupting the URL.
    identifier_string = join(identifiers, "%0d")
    escaped_identifiers = HTTP.escapeuri(identifier_string)
    
    url = raw"https://string-db.org/api/tsv/interaction_partners?identifiers=" * escaped_identifiers

    try
        # Use a temporary file for the download to handle large responses
        HTTP.download(url, tmp_file)
    catch e
        if e isa HTTP.StatusError && e.status == 400
            @error "HTTP Error 400 (Bad Request). The URL may be malformed or too long. Length: $(length(url)). Skipping this chunk."
            return DataFrame() # Return empty DataFrame on failure
        else
            @error "An unexpected HTTP error occurred: $e"
            rethrow(e) # Rethrow other, unexpected errors
        end
    end

    # Handle cases where the API returns no interactions for the given identifiers.
    if filesize(tmp_file) == 0
        @warn "API returned no interaction data for this chunk. Returning empty DataFrame."
        isfile(tmp_file) && rm(tmp_file)
        return DataFrame()
    end

    # Process the downloaded TSV file
    df = CSV.read(
        tmp_file, DataFrame,
        types = [String, String, String, String, Int32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32],
    )[:, [1,2,7,8,9,10,11,12,13]]

    rename!(
        df, 
        :stringId_A => :Protein1, :stringId_B => :Protein2, 
        :nscore => :neighborhood, :fscore => :fusion,
        :pscore => :phylogenetic, :ascore => :coexpression,
        :escore => :experimental, :dscore => :database,
        :tscore => :textmining    
    )
    
    # Clean up the temporary file
    isfile(tmp_file) && rm(tmp_file)

    return df
end

"""
    matchSTRINGscores(name_df::DataFrame, full_data::Matrix{F}) where {F<:AbstractFloat}

Matches protein pairs from `name_df` against a comprehensive set of STRING scores,
returning a DataFrame containing only the pairs for which scores were found. Furtheremore, 
 not identified pairs are removed from the emebdding matrix `full_data`.

This high-performance version avoids slow, nested loops by using a hash-based
`leftjoin`. It first creates a canonical, order-independent key for each protein
pair (e.g., "ID1_ID2" is the same as "ID2_ID1") and then performs a highly
optimized join on this key. This changes the algorithmic complexity from a slow
O(N*M) to a much faster O(N+M).

Returns
- `DataFrame`: A DataFrame containing the interaction data with additional columns for STRING scores.
- `Matrix{F}`: A matrix of protein embeddings after removing pairs without STRING scores.
"""
function matchSTRINGscores(name_df::DataFrame, full_data::Matrix{F}) where {F<:AbstractFloat}
    string_scores_h5_file = "encodings/STRINGscores.h5"

    if !isfile(string_scores_h5_file)
        @warn "STRING scores file not found. Calling API to generate it. This may take some time."

        # --- FIX: Implement dynamic chunking based on URL length ---
        all_unique_ids = union(name_df.Protein1, name_df.Protein2) |> collect
        @info "Found $(length(all_unique_ids)) unique protein identifiers to query."

        # Define constants for chunking
        base_url_len = length(raw"https://string-db.org/api/tsv/interaction_partners?identifiers=")
        max_url_len = 280 # Using 280 as a safe buffer below the 300 limit
        
        identifier_chunks = Vector{String}[]
        current_chunk = String[]

        for id in all_unique_ids
            # Check if adding the next ID would exceed the max length
            # We join with "%0d" which is 3 characters.
            potential_chunk_str = join(vcat(current_chunk, id), "%0d")
            
            # The length check must be on the URL-escaped string
            if base_url_len + length(HTTP.escapeuri(potential_chunk_str)) > max_url_len
                # Finalize the current chunk and start a new one
                push!(identifier_chunks, current_chunk)
                current_chunk = [id]
            else
                # Add the ID to the current chunk
                push!(current_chunk, id)
            end
        end
        # Don't forget to add the last chunk!
        if !isempty(current_chunk)
            push!(identifier_chunks, current_chunk)
        end
        
        @info "Dynamically split into $(length(identifier_chunks)) chunks to respect URL length limit."

        # Call API for each dynamically sized chunk and collect results
        all_scores_dfs = DataFrame[]
        ProgressMeter.@showprogress "Fetching STRING scores..." for (i, chunk) in enumerate(identifier_chunks)
            tmp_f = "string_tmp_$(rand(UInt32)).tsv"
            df_chunk = callSTRINGAPI(chunk; tmp_file=tmp_f)
            if !isempty(df_chunk)
                push!(all_scores_dfs, df_chunk)
            end
        end

        if isempty(all_scores_dfs)
             @error "Failed to retrieve any scores from STRING API. Cannot proceed."
             return full_data, DataFrame()
        end
        
        string_scores = vcat(all_scores_dfs...)
        @info "Retrieved a total of $(nrow(string_scores)) interactions. Caching to $string_scores_h5_file"

        values_matrix = Matrix{Float32}(string_scores[:, 3:end])
        annotations_matrix = Matrix{String}(string_scores[:, 1:2])
        HDF5.h5open(string_scores_h5_file, "w") do fid
            g = HDF5.create_group(fid, "scores")
            HDF5.write(g, "scores", values_matrix)
            HDF5.write(g, "annotations", annotations_matrix)
        end
    else
        @info "Loading cached STRING scores from $string_scores_h5_file"
        h5_data = HDF5.h5open(string_scores_h5_file, "r") do file
            (
                annotations = HDF5.read(file, "scores/annotations"),
                scores = HDF5.read(file, "scores/scores")
            )
        end
        string_scores = hcat(
            DataFrame(h5_data.annotations, ["Protein1", "Protein2"]),
            DataFrame(h5_data.scores, [:neighborhood, :fusion, :phylogenetic, :coexpression, :experimental, :database, :textmining]),
        )  

    end
    @info "STRING scores loaded with $(nrow(string_scores)) pairs."

    # --- Perform the high-performance join
    @info "Creating canonical join keys and performing join..."
    name_df.join_key = [join(sort([r.Protein1, r.Protein2]), "_") for r in eachrow(name_df)]
    string_scores.join_key = [join(sort([r.Protein1, r.Protein2]), "_") for r in eachrow(string_scores)]
    unique!(string_scores, :join_key)

    name_df.idx = 1:size(name_df,1)
    
    joined_df = leftjoin(name_df, select(string_scores, Not([:Protein1, :Protein2])), on = :join_key)
    sort!(joined_df, :idx)
    
    # replace missing values with zeros
    score_cols = [:neighborhood, :fusion, :phylogenetic, :coexpression, :experimental, :database, :textmining]
    for col in score_cols
        joined_df[:, col] = coalesce.(joined_df[!, col], 0.0f0)
    end
 
    # remove join_key and idx columns
    full_data = full_data[joined_df.idx,:]
    select!(joined_df, Not([:join_key, :idx]))
    
    return joined_df, full_data
end

# ---------------------------------------------------------------------------- #
# Function to split the dataset into train, val, and test sets
# ---------------------------------------------------------------------------- #

function _cosine_similarity(v1::AbstractVector, v2::AbstractVector)
    norm_v1 = norm(v1)
    norm_v2 = norm(v2)
    return (norm_v1 == 0 || norm_v2 == 0) ? 0.0 : dot(v1, v2) / (norm_v1 * norm_v2)
end

function _cosine_similarity_rows(row_vec1::AbstractMatrix, row_vec2::AbstractMatrix)
    # Assumes row_vec1 and row_vec2 are 1xN matrices (output of mean(..., dims=1))
    return _cosine_similarity(view(row_vec1, 1, :), view(row_vec2, 1, :))
end

function _validate_ratios(train_ratio, val_ratio, num_encoding_cols)
    
    if !(0.0 <= train_ratio <= 1.0)
        throw(ArgumentError("train_ratio must be between 0.0 and 1.0"))
    end

    if !(0.0 <= val_ratio <= 1.0)
        throw(ArgumentError("val_ratio must be between 0.0 and 1.0"))
    end

    if train_ratio + val_ratio > 1.0
        throw(ArgumentError("Sum of train_ratio and val_ratio cannot exceed 1.0"))
    end

    if num_encoding_cols <= 0
        throw(ArgumentError("num_encoding_cols must be positive."))
    end

    return true # Explicitly return true if all checks pass
end

function load_topological_negatives(encodings_path::String = "encodings/9606.protein.network.embeddings.v12.0.h5")
    file_1 = "encodings/Topological_Negatives_TPPNI_1.csv"
    file_2 = "encodings/Topological_Negatives_TPPNI_2.csv"
    file_3 = "encodings/Topological_Negatives_TPPNI_3.csv"
    file_4 = "encodings/Topological_Negatives_TPPNI_4.csv"
    # Load the CSV files into DataFrames
    df_1 = CSV.read(file_1, DataFrame)[:, [2, 3]]
    df_2 = CSV.read(file_2, DataFrame)[:, [2, 3]]
    df_3 = CSV.read(file_3, DataFrame)[:, [2, 3]]
    df_4 = CSV.read(file_4, DataFrame)[:, [2, 3]]
    # Concatenate the DataFrames
    df_concatenated = vcat(df_1, df_2, df_3, df_4)
    # Rename the columns
    rename!(df_concatenated, :SymbolA => :Protein1, :SymbolB => :Protein2)
    # shuffle the rows
    df_concatenated = df_concatenated[Random.shuffle(1:size(df_concatenated,1)), :]

    # free up memory by deleting the not-needed DataFrames [df_1, ..., df_4]
    empty!(df_1)
    empty!(df_2)
    empty!(df_3)
    empty!(df_4)

    # load the encodings from the HDF5 file
    @info "Loading full dataset from: $input_hdf5_filepath, dataset: $dataset_name_in_hdf5"
    full_data = HDF5.h5open(encodings_path, "r") do file
        HDF5.read(file, "embeddings")
    end

    # names
    protein_names = HDF5.h5open(encodings_path, "r") do file
        HDF5.read(file, "proteins")
    end

    # ---------------------------------- #
    # load converted ids 
    # conversion with tool on UniProt (uniprot.org,idmapping_2025_06_23.tsv)
    names_converted = CSV.read("encodings/idmapping_2025_06_23.tsv", DataFrame, types = String)[:, [1, end]]
    names_converted = names_converted[.! ismissing.(names_converted[:, 2]), :]
    names_converted[:, 2] = string.(names_converted[:, 2])
    names_converted[:, 2] = map(x -> split(x, " ")[1], names_converted[:, 2]) # keep only first gene name

    # remove all pairs from full_data whose ids are not in names_converted
    # before 3,063,605 pairs
    # after  2,608,037 pairs (85.13%)
    df_concatenated = df_concatenated[df_concatenated.Protein1 .∈ Ref(names_converted[:, 2]), :]
    df_concatenated = df_concatenated[df_concatenated.Protein2 .∈ Ref(names_converted[:, 2]), :]

    # replace ids in df_concatenated with the ids from names_converted
    df_concatenated.Protein1 = map(x -> names_converted[names_converted[:, 2] .== x, 1][1], df_concatenated.Protein1)
    df_concatenated.Protein2 = map(x -> names_converted[names_converted[:, 2] .== x, 1][1], df_concatenated.Protein2)

    # write df_concatenated to a csv file
    CSV.write("encodings/topological_negatives.csv", df_concatenated, compress = true)
    # ---------------------------------- #
    # create final matrix
    output = zeros(Float32, size(df_concatenated, 1), 1025)

    # fill the matrix with the encodings
    protein_to_index = Dict(name => i for (i, name) in enumerate(protein_names))

    # Use ThreadsX.foreach for a clear, parallel, in-place update in a single pass
    ThreadsX.foreach(eachindex(df_concatenated.Protein1)) do i
        # For Protein1
        p1_id = df_concatenated.Protein1[i]
        idx1 = get(protein_to_index, p1_id, nothing)
        if idx1 !== nothing
            # Use view to avoid allocating a new vector for the slice
            output[i, 1:512] = view(full_data, :, idx1, )
        end

        # For Protein2
        p2_id = df_concatenated.Protein2[i]
        idx2 = get(protein_to_index, p2_id, nothing)
        if idx2 !== nothing
            output[i, 513:1024] = view(full_data, :, idx2)
        end
    end
    # write output to a hdf5 file
    HDF5.h5write("encodings/topological_negatives.h5", "embeddings", output)
    return nothing
end

"""
    split_and_save_datasets(
    experimental_hdf5_filepath::String = "encodings/full_dataset.h5",
    dataset_name_in_hdf5::String = "features_labels";
    output_train_filepath::String = "train_data.h5",
    output_val_filepath::String   = "val_data.h5",
    output_test_filepath::String  = "test_data.h5",
    train_ratio::Float64          = 0.7,
    val_ratio::Float64            = 0.2,
    random_seed::Union{Nothing, Int} = nothing,
    num_encoding_cols::Int        = 1024,
    num_attempts_for_optimal_split::Int = 1000, 
    topological_negatives_filepath::String = "encodings/topological_negatives.h5",
    # --- NEW OPTIMIZATION PARAMETERS ---
    optimization_sample_size::Int = 10000,
    use_parallel::Bool = true,
    usetopological::Bool = false
)

Loads experimental and topological data, combines them, and performs a split
optimized for cosine distance. It ensures the validation and test sets contain
*only* experimental data to prevent data leakage and provide a gold-standard evaluation,
while the training set is augmented with topological data.

The logic is as follows:
1.  Load experimental ("gold standard") and topological data.
2.  Combine them into a single dataset. A boolean vector tracks the origin of each sample.
3.  Perform an optimized split on this entire combined dataset to find training, validation,
    and test partitions that are maximally dissimilar in the embedding space.
4.  The final training set consists of all samples (experimental and topological) that
    landed in the training partition.
5.  The final validation and test sets are constructed by taking their respective partitions
    and *filtering out* any samples that originated from the topological dataset, leaving
    only pure, gold-standard experimental data for evaluation.

# Arguments
- `experimental_hdf5_filepath::String`: Path to the HDF5 file containing the experimental (gold standard) dataset.
- `dataset_name_in_hdf5::String`: Name of the dataset within the HDF5 files (both experimental and generated train/val/test).

# Keyword Arguments
- `output_train_filepath::String`: Filename for the training dataset HDF5 file (saved in `encodings/`).
- `output_val_filepath::String`: Filename for the validation dataset HDF5 file (saved in `encodings/`).
- `output_test_filepath::String`: Filename for the testing dataset HDF5 file (saved in `encodings/`).
- `train_ratio::Float64 = 0.7`: Proportion of the combined data to allocate for the training set (0.0 to 1.0).
- `val_ratio::Float64 = 0.2`: Proportion of the combined data to allocate for the validation set (0.0 to 1.0).
   The test set proportion is calculated as `1.0 - train_ratio - val_ratio`.
- `random_seed::Union{Nothing, Int} = nothing`: Optional random seed for shuffling, ensuring reproducibility.
- `num_encoding_cols::Int = 1024`: The number of initial columns in the dataset that represent the encodings
  to be used for calculating cosine similarity for the split optimization.
- `num_attempts_for_optimal_split::Int = 1000`: Number of random splits to attempt to find one that minimizes
  the sum of pairwise cosine similarities between the average encodings of the sets. If `<= 0` or if
  fewer than two sets are active (i.e., expected to have samples based on ratios), a single random split is performed.
- `topological_negatives_filepath::String`: Path to the HDF5 file containing the topological negative samples.
- `optimization_sample_size::Int = 10000`: Number of samples to use for optimization. If <= 0, all samples are used.
- `use_parallel::Bool = true`: Whether to use parallel processing for optimization.
- `usetopological::Bool = false`: Whether to use topological negatives for training.

# Returns
- `nothing`

# Details
The function first validates the provided ratios. It then loads the experimental and topological datasets.
These are vertically concatenated, and a boolean vector `is_topological` is created to track the origin
of each sample (false for experimental, true for topological).

If `num_attempts_for_optimal_split` is greater than 0 and at least two dataset splits (train, val, test) are
expected to be non-empty, the function will try multiple random shuffles of the *combined* dataset. For each shuffle,
it calculates the average encoding (from the first `num_encoding_cols` columns) for each subset and then computes the
sum of pairwise cosine similarities between these average encodings. The shuffle that *minimizes* this sum
(i.e., maximizes average cosine distance) is chosen. Otherwise, a single random shuffle is used.

After determining the optimal indices for the combined dataset, the final training set is formed by taking
all samples corresponding to the `train_indices`. For the validation and test sets, only samples that
originated from the experimental dataset (i.e., `is_topological` is `false`) are included. This ensures
that the validation and test sets serve as a true "gold standard" for evaluation, free from potentially
noisy topological data.

Each resulting subset (training, validation, testing) is saved into its own HDF5 file within the `encodings/` directory.
The dataset within each new HDF5 file will have the same name as `dataset_name_in_hdf5`.

# Throws
- `ArgumentError`: If `train_ratio` or `val_ratio` are not between 0 and 1, if their sum exceeds 1, or if `num_encoding_cols` is not positive.
- Errors from `HDF5.h5open` or `HDF5.h5write` if file operations fail.
- `ArgumentError`: If `num_encoding_cols` is greater than the actual number of columns in the loaded data.

"""
function split_and_save_datasets(
    experimental_hdf5_filepath::String = "encodings/full_dataset.h5";
    dataset_name_in_hdf5::String = "features_labels",
    output_train_filepath::String = "train_data.h5",
    output_val_filepath::String   = "val_data.h5",
    output_test_filepath::String  = "test_data.h5",
    train_ratio::Float64          = 0.7,
    val_ratio::Float64            = 0.2,
    random_seed::Union{Nothing, Int} = nothing,
    num_encoding_cols::Int        = 1024,
    num_attempts_for_optimal_split::Int = 1000, 
    topological_negatives_filepath::String = "encodings/topological_negatives.h5",
    # --- OPTIMIZATION PARAMETERS ---
    optimization_sample_size::Int = 10000,
    use_parallel::Bool = true,
    usetopological::Bool = false
)
    # --- 1. Validation and Setup ---
    !_validate_ratios(train_ratio, val_ratio, num_encoding_cols) && return nothing
    !isnothing(random_seed) && Random.seed!(random_seed)

    if use_parallel && Threads.nthreads() > 1
        @info "Parallel execution enabled on $(Threads.nthreads()) threads."
    elseif use_parallel
        @info "Parallelism requested, but Julia was started with only 1 thread. Running serially. For parallel speedup, start Julia with multiple threads (e.g., `julia -t auto`)."
        use_parallel = false
    else
        @info "Serial execution."
    end
    
    # --- 2. Load and Combine Data ---
    @info "Loading experimental data from: $experimental_hdf5_filepath"
    experimental = HDF5.h5open(experimental_hdf5_filepath, "r") do file
        (
            experimental_data = HDF5.read(file, dataset_name_in_hdf5),
            proteins = HDF5.read(file, "proteins")
        )
    end
    protein_names_experimental = experimental.proteins
    experimental_data = experimental.experimental_data


    if usetopological
        @error "Currently not implemented for topological data!"
        return nothing

        @info "Loading topological negatives from: $topological_negatives_filepath"
        topological_negatives_data = HDF5.h5open(topological_negatives_filepath, "r") do file
            HDF5.read(file, "embeddings")
        end
        full_data = vcat(experimental_data, topological_negatives_data)
        is_topological = vcat(falses(size(experimental_data, 1)), trues(size(topological_negatives_data, 1)))
    else
        @info "No topological negatives loaded."
        full_data = experimental_data
        protein_names = protein_names_experimental
        is_topological = falses(size(experimental_data, 1))
    end

    n_samples = size(full_data, 1)
    @info "Combined dataset created. Total samples: $n_samples"

    # --- clean not needed variables
    experimental_data = nothing
    topological_negatives_data = nothing
    protein_names_experimental = nothing
    protein_names = DataFrame(protein_names, [:Protein1, :Protein2])

    # --- Load String scores via API calls ---
    @info "Loading string scores from STRING API..."
    protein_names, full_data = matchSTRINGscores(protein_names, full_data)

    # --- 3. Perform Split on the ENTIRE Combined Dataset ---
    n_train = floor(Int, train_ratio * n_samples)
    n_val = floor(Int, val_ratio * n_samples)
    idx_train_end = n_train
    idx_val_end = n_train + n_val
    n_test_actual = n_samples - idx_val_end

    best_split_indices_tuple = (train=Int[], val=Int[], test=Int[])
    
    num_active_sets = (n_train > 0) + (n_val > 0) + (n_test_actual > 0)

    if num_active_sets < 2 || num_attempts_for_optimal_split <= 0
         @info "Performing a single random split."
         # ... single split logic ...
         if !isnothing(random_seed) Random.seed!(random_seed) end
         shuffled_indices_final = Random.shuffle(1:n_samples)
         best_split_indices_tuple = (
            train=collect(view(shuffled_indices_final, 1:idx_train_end)),
            val=collect(view(shuffled_indices_final, (idx_train_end + 1):idx_val_end)),
            test=collect(view(shuffled_indices_final, (idx_val_end + 1):n_samples))
         )

    else
        @info "Optimizing split of combined dataset over $num_attempts_for_optimal_split attempts..."
        if optimization_sample_size > 0
             @info "Using subsampling for optimization: $optimization_sample_size samples per partition."
        else
             @info "Using full dataset for optimization (can be slow)."
        end

        all_encodings = view(full_data, :, 1:num_encoding_cols)
        
        # Low level function for one attempt: core functionality
        function perform_one_attempt(rng)
            shuffled_indices = Random.shuffle(rng, 1:n_samples)
            train_v = view(shuffled_indices, 1:idx_train_end)
            val_v   = view(shuffled_indices, (idx_train_end + 1):idx_val_end)
            test_v  = view(shuffled_indices, (idx_val_end + 1):n_samples)

            # --- SUBSAMPLING LOGIC ---
            sample_size = optimization_sample_size
            train_sample_v = sample_size > 0 && length(train_v) > sample_size ? view(train_v, rand(rng, 1:length(train_v), sample_size)) : train_v
            val_sample_v   = sample_size > 0 && length(val_v) > sample_size ? view(val_v, rand(rng, 1:length(val_v), sample_size)) : val_v
            test_sample_v  = sample_size > 0 && length(test_v) > sample_size ? view(test_v, rand(rng, 1:length(test_v), sample_size)) : test_v

            # --- MEAN CALCULATION ON SUBSAMPLES ---
            avg_enc_train = !isempty(train_sample_v) ? mean(view(all_encodings, train_sample_v, :), dims=1) : nothing
            avg_enc_val   = !isempty(val_sample_v) ? mean(view(all_encodings, val_sample_v, :), dims=1) : nothing
            avg_enc_test  = !isempty(test_sample_v) ? mean(view(all_encodings, test_sample_v, :), dims=1) : nothing

            similarity = 0.0
            !isnothing(avg_enc_train) && !isnothing(avg_enc_val)  && (similarity += _cosine_similarity_rows(avg_enc_train, avg_enc_val))
            !isnothing(avg_enc_train) && !isnothing(avg_enc_test) && (similarity += _cosine_similarity_rows(avg_enc_train, avg_enc_test))
            !isnothing(avg_enc_val)   && !isnothing(avg_enc_test) && (similarity += _cosine_similarity_rows(avg_enc_val, avg_enc_test))
            
            return similarity, (train=collect(train_v), val=collect(val_v), test=collect(test_v))
        end

        min_total_similarity = Inf
        
        if use_parallel
            # --- PARALLEL IMPLEMENTATION ---
            best_per_thread = Vector{Tuple{Float64, NamedTuple{(:train, :val, :test), Tuple{Vector{Int}, Vector{Int}, Vector{Int}}}}}(undef, Threads.nthreads())
            for i in 1:Threads.nthreads()
                best_per_thread[i] = (Inf, (train=Int[], val=Int[], test=Int[]))
            end

            # Create a separate, thread-safe RNG for each thread to avoid state conflicts
            rngs = [Random.MersenneTwister(isnothing(random_seed) ? rand(UInt) : random_seed + i) for i in 1:Threads.nthreads()]

            Threads.@threads for i in 1:num_attempts_for_optimal_split
                tid = Threads.threadid()
                rng = rngs[tid] 
                current_similarity, current_indices = perform_one_attempt(rng)

                if current_similarity < best_per_thread[tid][1]
                    best_per_thread[tid] = (current_similarity, current_indices)
                end
            end

            # Reduction step: find the best result among all threads
            for (similarity, indices) in best_per_thread
                if similarity < min_total_similarity
                    min_total_similarity = similarity
                    best_split_indices_tuple = indices
                end
            end

        else
            # --- SERIAL IMPLEMENTATION ---
            rng = Random.MersenneTwister(isnothing(random_seed) ? rand(UInt) : random_seed)
            for i in 1:num_attempts_for_optimal_split
                @info "Attempt #$i"
                similarity, indices = perform_one_attempt(rng)
                if similarity < min_total_similarity
                    min_total_similarity = similarity
                    best_split_indices_tuple = indices
                end
            end
        end
        @info "Optimal split found with estimated total similarity: $(round(min_total_similarity, digits=4))"
    end
    
    # --- 4. Assemble Final Datasets with Filtering ---
    train_indices = best_split_indices_tuple.train
    val_indices   = best_split_indices_tuple.val
    test_indices  = best_split_indices_tuple.test
    
    train_data = full_data[train_indices, :]
    train_scores = protein_names[train_indices, :]

    if usetopological
        val_indices = val_indices[.!is_topological[val_indices]]
        test_indices = test_indices[.!is_topological[test_indices]]
    end

    val_data  = full_data[val_indices, :]
    val_scores = protein_names[val_indices, :]

    test_data = full_data[test_indices, :]
    test_scores = protein_names[test_indices, :]

    @info "Final datasets assembled."
    usetopological ?  (@info " -> Training set size: $(size(train_data, 1)) (contains experimental + topological)") : (@info " -> Training set size: $(size(train_data, 1)) (contains ONLY experimental)")
    @info " -> Validation set size: $(size(val_data, 1)) (purified, contains ONLY experimental)"
    @info " -> Test set size: $(size(test_data, 1)) (purified, contains ONLY experimental)"

    # --- 5. Save Final Datasets ---
    # ... (save logic remains the same) ...
    output_dir = "encodings"
    !isdir(output_dir) && mkdir(output_dir)

    HDF5.h5write(joinpath(output_dir, output_train_filepath), dataset_name_in_hdf5, train_data)
    HDF5.h5write(joinpath(output_dir, output_train_filepath), "scores", Matrix{Float32}(train_scores[:,3:end]))
    HDF5.h5write(joinpath(output_dir, output_train_filepath), "proteins", Matrix{String}(train_scores[:,1:2]))
    @info "Training data saved to: $(joinpath(output_dir, output_train_filepath))"
    
    HDF5.h5write(joinpath(output_dir, output_val_filepath), dataset_name_in_hdf5, val_data)
    HDF5.h5write(joinpath(output_dir, output_val_filepath), "scores", Matrix{Float32}(val_scores[:,3:end]))
    HDF5.h5write(joinpath(output_dir, output_val_filepath), "proteins", Matrix{String}(val_scores[:,1:2]))
    @info "Validation data saved to: $(joinpath(output_dir, output_val_filepath))"
    
    HDF5.h5write(joinpath(output_dir, output_test_filepath), dataset_name_in_hdf5, test_data)
    HDF5.h5write(joinpath(output_dir, output_test_filepath), "scores", Matrix{Float32}(test_scores[:,3:end]))
    HDF5.h5write(joinpath(output_dir, output_test_filepath), "proteins", Matrix{String}(test_scores[:,1:2]))
    @info "Test data saved to: $(joinpath(output_dir, output_test_filepath))"
    
    return nothing
end

#split_and_save_datasets(usetopological = false, use_parallel = true)

# ------------------------------------------------------------------------------ #
# Training loop
# ------------------------------------------------------------------------------ #

"""
    cosine_annealing_lr(current_epoch; η_max, η_min, total_epochs, warmup_epochs)

Calculates a learning rate based on a cosine annealing schedule with a linear warmup phase.

# Arguments
- `current_epoch::Int`: The current epoch number (should be 1-based).
- `η_max::Float64`: The maximum learning rate, reached at the end of warmup. (default: 1e-3)
- `η_min::Float64`: The minimum learning rate, reached at the end of the schedule. (default: 1e-6)
- `total_epochs::Int`: The total number of epochs for the entire schedule. (default: 100)
- `warmup_epochs::Int`: The number of epochs for the linear warmup phase. (default: 10)

# Returns
- `Float64`: The calculated learning rate for the given epoch.
"""
function cosine_annealing_lr(
    current_epoch::Int;
    η_min::Float64 = 1e-12, η_max::Float64 = 1e-3, 
    total_epochs::Int = 100,
    warmup_epochs::Int = 10 
    )

    # --- Error Checking --- # 
    current_epoch < 1 && throw(ArgumentError("current_epoch must be at least 1"))
    warmup_epochs < 1 && throw(ArgumentError("warmup_epochs must be at least 1"))
    total_epochs < 1 && throw(ArgumentError("total_epochs must be at least 1"))

    # --- Pre-computation and Edge Case Handling --- # 
    if warmup_epochs >= total_epochs
        return η_min + (η_max - η_min) * (current_epoch / total_epochs)
    end

    t = current_epoch - 1 # convert to 0-based indexing

    # --- warmup phase --- #
    if t < warmup_epochs
        # Linear warmup from η_min to η_max
        # (t + 1) ensures that at t = warmup_epochs - 1, the fraction is 1.0
        lr = η_min + (η_max - η_min) * (t + 1) / warmup_epochs

    # --- cosine annealing phase --- #
    else
        # Time within the cosine annealing phase
        T_cur = t - warmup_epochs
        # total duration of the annealing phase
        T_total = total_epochs - warmup_epochs

        # Standard cosine annealing
        lr = (η_max - η_min) * (1 + cos(pi * T_cur / T_total)) / 2
        lr += η_min
    end

    return lr
end

"""
    log_batch_metrics!(
        model,
        x_batch::AbstractMatrix{Float32},
        y_batch_true::AbstractMatrix{Float32},
        loss_val::Float32
    )

Calculates various performance metrics for a given batch of predictions.

The model is temporarily set to test mode (`Flux.testmode!`) for predictions
and then restored to train mode (`Flux.trainmode!`).

# Arguments
- `model`: The Flux model being evaluated.
- `x_batch`: The input features for the batch.
- `y_batch_true`: The true labels for the batch (expected to be 0.0f0 or 1.0f0).
- `loss_val::Float32`: The pre-calculated loss value for this batch.

# Returns
- `NamedTuple`: A named tuple containing the following metrics: `loss`, `accuracy`, `precision`, `recall`, and `ba` (balanced accuracy).
"""
function log_batch_metrics!(
    model,
    x_batch::AbstractMatrix{F},
    y_batch_true::AbstractMatrix{F},
    loss_val::Float32
    ) where {F<:AbstractFloat}
    # Calculate metrics for the current batch
    Flux.testmode!(model)
    y_pred_proba_current = model(x_batch) # Get predictions
    Flux.trainmode!(model)
        
    # Threshold predictions to get binary outcomes
    predictions_binary = y_pred_proba_current .> 0.5f0 
    targets_binary = y_batch_true .== 1.0f0

    acc = mean(predictions_binary .== targets_binary)
        
    tp = sum(predictions_binary .& targets_binary)
    fp = sum(predictions_binary .& .!targets_binary) 
    fn = sum((.!predictions_binary) .& targets_binary)
    tn = sum((.!predictions_binary) .& .!targets_binary)

    prec = tp / (tp + fp + eps(Float32)) 
    rec = tp / (tp + fn + eps(Float32))

    # balanced accuracy
    specificity = tn / (tn + fp + eps(Float32))
    ba = (specificity + rec) / 2

    return (loss = loss_val, accuracy = acc, precision = prec, recall = rec, ba = ba, TP = tp, FP = fp, FN = fn, TN = tn)
end

"""
    mcc(tp, tn, fp, fn)

Calculates the Matthews Correlation Coefficient (MCC).

The MCC is a measure of the quality of binary (two-class) classifications.
It takes into account true and false positives and negatives and is generally
regarded as a balanced measure which can be used even if the classes are of
very different sizes.

# Arguments
- `tp::Real`: Number of true positives.
- `tn::Real`: Number of true negatives.
- `fp::Real`: Number of false positives.
- `fn::Real`: Number of false negatives.

# Returns
- `Float64`: The Matthews Correlation Coefficient. The value ranges from -1 to +1.
  A coefficient of +1 represents a perfect prediction, 0 an average random
  prediction, and -1 an inverse prediction. Returns `NaN` if the denominator is zero
  (e.g., if any of the sums `tp+fp`, `tp+fn`, `tn+fp`, `tn+fn` are zero).

# Formula
`MCC = (tp*tn - fp*fn) / sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))`
"""
function mcc(tp, tn, fp, fn)
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / (denominator + eps(Float64))
end


"""
    generate_input_features(X::AbstractMatrix{F}) where {F <: AbstractFloat}

Generates augmented features from an input matrix `X`. It assumes `X` contains
concatenated feature vectors for two entities (e.g., proteins) per sample,
with samples as rows and features as columns.

The function expects `X` to have its features structured as follows:
- Columns `1:1024`: Features for the first entity.
- Columns `1025:2048` (implicitly, `X[:, 1025:end]`): Features for the second entity.

It computes two new feature sets:
1. The Hadamard product (element-wise product) of the two entities' feature sets.
2. The element-wise difference (features_entity1 - features_entity2).

The output matrix is a horizontal concatenation of the original and derived feature sets
for each sample, in the order:
`[features_entity1, features_entity2, hadamard_product, element_wise_difference]`.

# Arguments
- `X::AbstractMatrix`: A 2D matrix where each row represents a sample and columns
  represent features. It is expected to have `2048` columns in total,
  corresponding to 1024 features for a first entity followed by 1024 features
  for a second entity.

# Returns
- `AbstractMatrix`: A new matrix with the same number of rows (samples) as `X`,
  but with `4096` columns. The columns are ordered as described above.
"""
function generate_input_features(X::AbstractMatrix{F}) where {F <: AbstractFloat}
    # Hadamard product
    hadamard_product = view(X, 1:512, :) .* view(X, 513:1024, :)
    # element-wise difference Δ
    Δ = view(X, 1:512, :) .- view(X, 513:1024, :)

    # Concatenate
    return cat(X[1:512,:], X[513:end,:], hadamard_product, Δ, dims=1)
end


"""
    augment_data_mixup(X::AbstractMatrix{F}, Y::AbstractVector{F}, α::F) where {F <: AbstractFloat}

Applies Mixup data augmentation to a batch of data.

This function creates synthetic training samples by forming convex combinations
of pairs of examples and their labels. It helps improve model generalization
and robustness.

# Arguments
- `X::AbstractMatrix`: The input feature matrix of shape `(feature_dim, batch_size)`.
- `Y::AbstractVecOrMat`: The input labels of shape `(batch_size,)` or `(1, batch_size)`.
- `α::Real = 0.2`: The alpha hyperparameter for the Beta distribution, controlling
  the strength of the interpolation. A common range is 0.1 to 0.4.

# Returns
- `Tuple{AbstractMatrix, AbstractVecOrMat}`: A tuple containing the new, augmented
  batch of features and labels `(X_mixed, Y_mixed)`.
"""
function augment_data_mixup(X::AbstractMatrix{F}, Y::AbstractMatrix{F}, α::F = 0.1) where {F <: AbstractFloat}
    batch_size = size(X, 2)
    # permute close

    shuffled_indices = Random.randperm(batch_size)
    X_shuffled  = X[:, shuffled_indices]
    Y_shuffled  = Y[:,shuffled_indices]

    if α == 0.0
        return X_shuffled, Y_shuffled
    end
    
    # sample from beta distribution
    λ = rand(Beta(α, α),1, batch_size)
    # mixup
    X_aug = @. λ * X + (1 - λ) * X_shuffled
    Y_aug = @. λ * Y + (1 - λ) * Y_shuffled

    return X_aug, Y_aug
end


"""
    train_dnn!(
    model,
    train_hdf5_file::String,
    dataset_name::String,
    val_hdf5_file::Union{Nothing, String} = nothing,
    val_dataset_name::Union{Nothing, String} = nothing; # Note: semicolon was missing in original request, added for consistency with style
    optimizer::String = "Adam",
    η_min::Float64 = 1e-12,
    η_max::Float64 = 1e-3,
    warmup_epochs::Int = 10,
    loss_fn = Flux.Losses.binary_focal_loss,
    epochs::Int = 10,
    batch_size::Int = 64,
    device = Flux.cpu,
    gamma::Float64 = 2.0,
    apply_feature_engineering::Bool = true,
    α::Float64 = 0.1
)


Trains the given Flux model using data from an HDF5 file.

The model and optimizer state are modified in place.

# Arguments
- `model`: The Flux model (e.g., a `Flux.Chain`) to be trained.
- `hdf5_filepath::String`: Path to the HDF5 file containing the training data.
- `dataset_name::String`: Name of the dataset within the HDF5 file. It's assumed
  that the last column of this dataset contains the labels, and all preceding
  columns are features.
- `val_hdf5_file::Union{Nothing, String}`: Optional path to an HDF5 file containing
  validation data. If provided along with `val_dataset_name`, validation will be
  performed after each epoch.
- `val_dataset_name::Union{Nothing, String}`: Optional name of the dataset within the
  validation HDF5 file.

# Keyword Arguments
- `optimizer::String = "Adam"`: The name of the optimizer to use (e.g., "Adam", "RMSProp").
- `η_min::Float64 = 1e-12`: Minimum learning rate for the cosine annealing scheduler.
- `η_max::Float64 = 1e-3`: Maximum learning rate for the cosine annealing scheduler.
- `lr_cycles::Int = 4`: Number of learning rate cycles for the cosine annealing scheduler.
- `warmup_epochs::Int = 10`: The number of epochs for linear learning rate warmup.
- `loss_fn`: The loss function to use (default: `Flux.Losses.binary_focal_loss`).
  If using `binary_focal_loss`, the `gamma` parameter is also relevant.
- `epochs::Int = 10`: The number of training epochs.
- `batch_size::Int = 64`: The size of mini-batches for training.
- `device`: The device to use for training (e.g., `Flux.cpu` or `Flux.gpu`).
  The model and data will be moved to this device.
- `gamma::Float64 = 2.0`: The `gamma` focusing parameter for `binary_focal_loss`. Ignored if a different loss function is used.
- `apply_feature_engineering::Bool = true`: If `true`, `generate_input_features` is applied to the input data.
  Set to `false` for models (like MHA) that expect raw concatenated embeddings.
- `l2_reg::Float64 = 0.0`: L2 regularization strength
- `α::Float64 = 0.1`: Mixup alpha parameter

# Returns
- `nothing`

# Details
The function reads the entire specified training (and optionally validation) dataset(s)
from the HDF5 file(s) into memory. Features are transposed to match Flux's expected
column-major format (features as columns, samples as rows after transposition).
Labels are expected to be binary (0.0f0 or 1.0f0) and are formatted as a row vector.
A `Flux.DataLoader` is used for batching and shuffling the training data (validation data is not shuffled).
Training progress is displayed using `ProgressMeter`. If a `logger` is provided (e.g., for Weights & Biases),
batch-level training metrics (logged via `@info` to the global logger) and epoch-level validation metrics (if applicable) are logged.
"""
function train_dnn!(
    model,
    train_hdf5_file::String,
    dataset_name::String,
    val_hdf5_file::Union{Nothing, String} = nothing,
    val_dataset_name::Union{Nothing, String} = nothing;
    optimizer::String = "Adam",
    η_min = 1e-12, 
    η_max = 1e-3,
    lr_cycles::Int = 4,
    warmup_epochs::Int = 10,
    loss_fn = Flux.Losses.binary_focal_loss,
    epochs::Int = 10,
    batch_size::Int = 64,
    device = Flux.cpu,
    gamma::Float64 = 2.0,
    apply_feature_engineering::Bool = true,
    l2_reg::Float64 = 0.0,
    α::Float64 = 0.1
)
    η_min > η_max && return nothing

    # learning rate scheduler
    lr = cosine_annealing_lr.(
        1:epochs,
        η_min = 1e-12, 
        η_max = 1e-3, 
        total_epochs = Int64(floor(epochs / lr_cycles)),
        warmup_epochs = warmup_epochs
    )

    # define optimizer
     if optimizer == "Adam"
        opt_rule = Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam(lr[1]))
    elseif optimizer == "RMSProp"
        opt_rule = Flux.OptimiserChain(Flux.ClipNorm(), Flux.RMSProp(lr[1]))
    elseif optimizer == "AdamW"
        opt_rule = Flux.OptimiserChain(Flux.ClipNorm(), Flux.AdamW(eta = lr[1], lambda = l2_reg))
    else
        @error "Unsupported optimizer: $optimizer"
        return nothing
    end

    current_learning_rate = lr[1]

    @info "Loading data from HDF5 file: $train_hdf5_file, dataset: $dataset_name"
    data = HDF5.h5open(train_hdf5_file, "r") do file
        HDF5.read(file, dataset_name) # Reads the entire dataset
    end

    # --------------------------------- #
    # Load training data from HDF5 file #
    # --------------------------------- #

    # Features: all columns except the last, transposed for Flux (features as columns)
    X = Float32.(data[:, 1:end-1]')
    protein_feature_dim = 512
    if apply_feature_engineering
        X = generate_input_features(X)
        protein_feature_dim = 2048    
    end

    # Labels: the last column, as a row vector [1, N]
    y = reshape(data[:, end], 1, :)
    
    # Ensure labels are Float32 if not already, as expected by binarycrossentropy
    if eltype(y) != Float32
        y = Float32.(y)
    end

    @info "Data loaded. X_size: $(size(X)), y_size: $(size(y)). Moving model to device: $device"
    model = model |> device
    loader = Flux.DataLoader((X, y), batchsize=batch_size, shuffle=true, parallel=true)

    # --------------------------------- #
    # Load val data from HDF5 file ---- #
    # --------------------------------- #
    val_loader = nothing
    X_val_full_size = 0 
    if !isnothing(val_hdf5_file) && !isnothing(val_dataset_name) && isfile(val_hdf5_file)
        @info "Loading validation data from HDF5 file: $val_hdf5_file, dataset: $val_dataset_name"
        val_data_h5 = HDF5.h5open(val_hdf5_file, "r") do file
            HDF5.read(file, val_dataset_name)
        end

        X_val = val_data_h5[:, 1:end-1]' 
        if apply_feature_engineering
            X_val = generate_input_features(X_val)
        end
        X_val = X_val |> device
        y_val = reshape(val_data_h5[:, end], 1, :) |> device

        if eltype(y_val) != Float32
            y_val = Float32.(y_val)
        end
        val_loader = Flux.DataLoader((X_val, y_val), batchsize=batch_size, shuffle=false)
        X_val_full_size = size(X_val, 2)
        @info "Validation data loaded. X_val_size: $(size(X_val)), y_val_size: $(size(y_val))."
    end

    # --------------------------------- #
    # Define progress bar          ---- #
    # --------------------------------- #
    actual_batches_per_epoch = ceil(Int, size(X, 2) / batch_size)
    num_total_batches = actual_batches_per_epoch * epochs

    progress = ProgressMeter.Progress(
        num_total_batches, desc="Starting training...",
        showspeed=true,
        barglyphs= ProgressMeter.BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
        barlen = 40, dt = 5
        )

    min_val_loss = Inf
    # --------------------------------- #
    # Training loop                     #
    # --------------------------------- #
    # early stopping
    current_val_mcc = -Inf32
    best_val_mcc = - Inf32
    patience = Int64(ceil(epochs / lr_cycles * 2) + 5)
    get_best_val_mcc = () -> current_val_mcc * (-1)
    
    early_stopping = Flux.early_stopping(
        get_best_val_mcc, patience,
        init_score = Inf32, min_dist = 1e-6 
    )

    # Create optimizer
    opt_state = Flux.setup(opt_rule, model)

    t = Dates.now() |> string 
    t = replace(t, ":" => "", "." => "_")
    path = joinpath(pwd(), "checkpoints", "model_$t")
    mkpath(path)

   
    @info "Starting training for $epochs epochs with batch size $batch_size. Metrics logged to W&B"
    total_batches_processed = 0
    for epoch in 1:epochs
        epoch_losses = Float32[]
        batch_in_epoch_counter = 0
        for (x_batch, y_batch_true) in loader # y_batch_true contains true labels (0.0f0 or 1.0f0)
            # randomly swap protein A and B for gamma with a probability of 0.5
            if apply_feature_engineering && rand() < 0.5
                protein_a = x_batch[1:protein_feature_dim, :]
                protein_b = x_batch[protein_feature_dim+1:end, :]
                # swap protein A and B
                x_batch[1:protein_feature_dim, :] = protein_b
                x_batch[protein_feature_dim+1:end, :] = protein_a
            end

            # add mixup augmentation and concatenate with original data
            if α > 0.0
                X_aug, Y_aug = augment_data_mixup(x_batch, y_batch_true, Float32(α))           
                x_batch = cat(x_batch, X_aug, dims = 2)
                y_batch_true = cat(y_batch_true, Y_aug, dims = 2)
            end

            # move data and labels to GPU
            x_batch = x_batch |> device 
            y_batch_true = y_batch_true |> device
               
            #set counters for progress bar
            total_batches_processed += 1
            batch_in_epoch_counter += 1

            Flux.trainmode!(model) # ensure that model is in training mode

            loss_val, grads = Flux.withgradient(model) do m
                y_pred_proba = m(x_batch)
                # clamp the probabilities to avoid log(0) instability in the loss-function
                y_pred_proba_clamped = clamp.(y_pred_proba, eps(Float32), 1.0f0 - eps(Float32))
                loss_fn(y_pred_proba_clamped, y_batch_true, gamma = gamma)
            end

            # only update the model parameters if the loss is finite
            infinite_counter = 0
            if isfinite(loss_val) 
                Flux.update!(opt_state, model, grads[1])
            else
                @warn "Loss is not finite. Skipping update."
                infinite_counter += 1
                if infinite_counter >= 10
                    @warn "Loss is not finite for 10 consecutive batches. Stopping training."
                    break
                end
            end

            push!(epoch_losses, loss_val)

            metrics = log_batch_metrics!(
                model, x_batch, 
                y_batch_true, Float32(loss_val)
            )

            batch = epoch + (batch_in_epoch_counter - 1) / actual_batches_per_epoch

            # compute MCC: Matthews Correlation Coefficient (MCC)
            train_mcc = mcc(metrics.TP, metrics.TN, metrics.FP, metrics.FN)

            # log to Weights & Biases
            @info "Metrics" loss = metrics.loss accuracy = metrics.accuracy precision = metrics.precision recall = metrics.recall epoch = batch balanced_accuracy = metrics.ba lr = lr[epoch] MCC = train_mcc

            ProgressMeter.next!(progress)
        end

        if epoch < epochs
            Flux.adjust!(opt_state, lr[epoch + 1])
        end

        # validation loss
        if !isnothing(val_loader)
            @info "Validating epoch $epoch..."
            Flux.testmode!(model)
            total_val_loss_epoch = 0.0f0
            all_val_preds_binary_epoch = Bool[]
            all_val_targets_binary_epoch = Bool[]

            for (x_batch_val, y_batch_true_val) in val_loader
                y_pred_proba_val = model(x_batch_val)
                y_pred_proba_val = clamp.(y_pred_proba_val, eps(Float32), 1.0f0 - eps(Float32))
                batch_loss_val = loss_fn(y_pred_proba_val, y_batch_true_val, gamma = gamma)
                total_val_loss_epoch += batch_loss_val * size(x_batch_val, 2)

                append!(all_val_preds_binary_epoch, vec(y_pred_proba_val .> 0.5f0))
                append!(all_val_targets_binary_epoch, vec(y_batch_true_val .== 1.0f0))
            end       

            mean_val_loss_epoch = X_val_full_size > 0 ? total_val_loss_epoch / X_val_full_size : 0.0f0

            val_tp_epoch = sum(all_val_preds_binary_epoch .& all_val_targets_binary_epoch)::Int
            val_fp_epoch = sum(all_val_preds_binary_epoch .& .!all_val_targets_binary_epoch)::Int
            val_fn_epoch = sum((.!all_val_preds_binary_epoch) .& all_val_targets_binary_epoch)::Int
            val_tn_epoch = sum((.!all_val_preds_binary_epoch) .& .!all_val_targets_binary_epoch)::Int
            total = val_tp_epoch + val_fp_epoch + val_fn_epoch + val_tn_epoch
            val_acc_epoch = (val_tp_epoch + val_tn_epoch) / total
            
            if mean_val_loss_epoch < min_val_loss
                min_val_loss = mean_val_loss_epoch
            end

            # MCC: Matthews Correlation Coefficient (MCC)
            val_mcc = mcc(val_tp_epoch, val_tn_epoch, val_fp_epoch, val_fn_epoch)

            # specificity
            val_prec_epoch = val_tp_epoch / (val_tp_epoch + val_fp_epoch + eps(Float32)) 
            val_rec_epoch  = val_tp_epoch / (val_tp_epoch + val_fn_epoch + eps(Float32))

            # balanced accuracy
            val_specificity = val_tn_epoch / (val_tn_epoch + val_fp_epoch)
            val_ba = (val_specificity + val_rec_epoch) / 2

            current_val_mcc = val_mcc

            # update best_val_mcc and save checkpoint
            if val_mcc > best_val_mcc
                best_val_mcc = val_mcc
                if best_val_mcc > 0.45
                    model = model |> Flux.cpu
                    jldsave(joinpath(path, "model-$epoch-$(best_val_mcc).jld2"), compress = true; model_state = Flux.state(model))
                    model = model |> Flux.gpu
                end
            end

            # log tp, fp, fn, tn, total and other metrics to Weights & Biases
            @info "Metrics" epoch = epoch val_loss = mean_val_loss_epoch val_acc = val_acc_epoch val_prec = val_prec_epoch val_rec = val_rec_epoch min_val_loss = min_val_loss val_balanced_accuracy = val_ba val_TP = val_tp_epoch val_FP = val_fp_epoch val_FN = val_fn_epoch val_TN = val_tn_epoch val_total = total val_mcc = val_mcc best_val_mcc = best_val_mcc
        
            early_stopping() ? break : nothing
        
        end
    end
    @info "Training complete."
    ProgressMeter.finish!(progress)
    return best_val_mcc
end


# ---------------------------------------------------------------------------------------------- #
# Main training function:
# ---------------------------------------------------------------------------------------------- #
"""
    main_training(; 
        optimizer::String = "Adam",
        η_min::Float64 = 1e-12, 
        η_max::Float64 = 1e-3, 
        lr_cycles::Int = 4,
        warmup_epochs::Int = 10,
        n_layers::Int = 7, 
        n_neurons::Vector{Int} = [1024, 512, 256, 128, 64, 32, 16], 
        activation::Vector{Function} = [Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.sigmoid],
        epochs::Int = 100, 
        batch_size::Int = 128,
        dropout_rate::Float64 = 0.1,
        gamma::Float64 = 2.0,
        α:: Float64 = 0.1,
        apply_feature_engineering::Bool = false
    )

Orchestrates the DNN training process, including Weights & Biases setup,
model creation, training execution, and model saving.

# Keyword Arguments
- `optimizer::String = "Adam"`: The name of the optimizer to use (e.g., "Adam", "RMSProp").
- `n_layers::Int = 7`: The number of layers in the DNN model.
- `n_neurons::Vector{Int} = [...]`: A vector specifying the number of neurons in each layer.
  The first element is the input dimension to the first layer, and subsequent elements
  define output dimensions for intermediate layers. The last user-defined layer's output
  is implicitly connected to a final single-neuron output layer (handled by `getDNNModel`).
- `activation::Vector{Union{Any,Function}} = [...]`: A vector of activation functions for each layer.
  The length must match `n_layers`.
- `epochs::Int = 100`: The total number of training epochs.
- `batch_size::Int = 128`: The batch size for training.
- `η_min::Float64 = 1e-12`: The minimum learning rate for the cosine annealing scheduler.
- `η_max::Float64 = 1e-3`: The maximum learning rate for the cosine annealing scheduler.
- `lr_cycles::Int = 4`: The number of learning rate cycles for the cosine annealing scheduler.
- `warmup_epochs::Int = 10`: The number of epochs for linear learning rate warmup.
- `dropout_rate::Float64 = 0.1`: The dropout rate to be used in the DNN model (passed to `getDNNModel`).
- `gamma::Float64 = 2.0`: The `gamma` focusing parameter for `binary_focal_loss` used in `train_dnn!`.
- `α::Float64 = 0.1`: The `alpha` smoothing parameter for `binary_focal_loss` used in `train_dnn!`.

# Returns
- `Float64` or `Infinity`: The minimum validation loss achieved during training. Returns `Inf` if validation is not performed or if an error occurs early.

# Details
The function performs the following steps:
1. Defines paths for training (`train_data.h5`) and validation (`val_data.h5`) HDF5 files.
   It assumes these files have been previously created (e.g., by `split_and_save_datasets`).
2. Checks for the existence of the training data file and exits if not found.
3. Initializes and configures a `WandbLogger` for Weights & Biases, logging hyperparameters.
4. Sets the global logger to the `WandbLogger` instance.
5. Selects the computation device (GPU if available via `CUDA.functional()`, otherwise CPU).
6. Creates the DNN model using `getDNNModel` with the specified architecture parameters.
7. Calls `train_dnn!` to perform the actual training, passing the model, data paths,
   optimizer name, learning rate schedule parameters, and other training parameters.
8. The trained model itself is not saved by this function in its current form (the save line is commented out).
9. Closes the `WandbLogger`.
10. Returns the minimum validation loss recorded during training.

The dataset name within the HDF5 files is assumed to be `"features_labels"`.

# Example
```julia
model = main_training(
    opt_rule = "Adam", 
    λ = 0.0001,
    epochs = 50,
    batch_size = 64
)
```
"""
function main_training(; 
    optimizer::String = "Adam",  
    n_layers::Int = 7, 
    n_neurons::Vector{Int} = [1024, 512, 256, 128, 64, 32, 16], 
    activation::Vector{Union{Any,Function}} = [Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.sigmoid],
    epochs::Int = 100, 
    batch_size::Int = 128, 
    η_min::Float64 = 1e-12, 
    η_max::Float64 = 1e-3, 
    lr_cycles::Int = 4,
    warmup_epochs::Int64 = 10,
    dropout_rate::Float64 = 0.1,
    gamma::Float64 = 2.0,
    α::Float64 = 0.1,
    apply_feature_engineering::Bool = false,
    lg = nothing,
    train_hdf5_file::String = "encodings/train_data.h5",
    val_hdf5_file::String = "encodings/val_data.h5"
    )

    # skip training if η_min >= η_max
    η_min >= η_max && return Inf

    # Define datasets
    dataset_name_in_h5 = "features_labels"

    if !isfile(train_hdf5_file)
        @error "HDF5 training data file '$train_hdf5_file' not found. Please generate it first using split_and_save_datasets."
        @info "Example: split_and_save_datasets(\"encodings/full_dataset.h5\", \"$dataset_name_in_h5\", output_train_filepath=\"$train_hdf5_file\", output_val_filepath=\"$val_hdf5_file\")"
        return
    end

    # --- Weights & Biases Logging Setup ---
    # Extract learning rate from the optimizer rule
    # This assumes opt_rule is an optimizer like Adam, RMSProp, etc., which has an .eta field
    if isnothing(lg)
        lg = WandbLogger(
            project = "DeepPPIPred2", name = "run-$(Dates.now())",
            reinit = "finish_previous",
            config = Dict(
                "min_learning_rate" => η_min,
                "max_learning_rate" => η_max,
                "warmup_epochs" => warmup_epochs,
                "lr_cycles" => lr_cycles, 
                "batch_size" => batch_size,
                "epochs" => epochs,
                "n_layers" => n_layers,
                "n_neurons" => n_neurons[2],
                "activation_functions" => activation_to_string(activation[1]),
                "optimizer" => optimizer,
                "architecture" => "Dense",
                "dropout_rate" => dropout_rate,
                "gamma" => gamma,
                "alpha" => α,
                "apply_feature_engineering" => apply_feature_engineering
            )
    )
    global_logger(lg)
    end
    # --- End of W&B Setup ---

   # Select device
   dev = CUDA.functional() ? Flux.gpu : Flux.cpu 

   # Create model
   model = getDNNModel(n_layers, n_neurons, activation, dropout_rate)
   model = model |> dev

   # Create optimizer

   @info "Starting DNN training example..."
   
   min_val_loss = train_dnn!(
         model, train_hdf5_file, dataset_name_in_h5, 
         val_hdf5_file, dataset_name_in_h5,
         optimizer = optimizer, η_min = η_min, η_max = η_max, warmup_epochs = warmup_epochs,
         lr_cycles = lr_cycles,
         epochs = epochs, batch_size = batch_size, device = dev,
         gamma = gamma, 
         α = α, 
         apply_feature_engineering = apply_feature_engineering
     )

    @info "DNN training example finished."

    close(lg) # Close the logger at the end of training

    return min_val_loss
end

# --------------------------------------------------------------------------- #
# Hyperparameter Sweep
# --------------------------------------------------------------------------- #

model = main_training(
    activation = _define_activations("relu", 10),
    optimizer = "Adam",
    n_layers = 10,
    n_neurons = _define_layers(2048, 10),
    epochs = 1000, 
    batch_size = 128,
    η_max = 0.00000323745754281764,
    η_min = 0.00000008685113737514,
    warmup_epochs = 90,
    dropout_rate = 0.175,
    gamma = 2.0,
    α = 0.1020408163265306 
)

#=
# old data 
model = main_training(
    activation = _define_activations("relu", 11),
    optimizer = "Adam",
    n_layers = 11,
    n_neurons = _define_layers(512, 11, input_size = 1024),
    epochs = 1000, batch_size = 64,
    η_min = 0.00000000000167683294,
    η_max = 0.00000000017575106249,
    warmup_epochs = 36,
    dropout_rate = 0.6641025641025641,
    gamma = 2.0,
    α = 0.25,
    train_hdf5_file = "encodings/old/train_data.h5",
    val_hdf5_file = "encodings/old/val_data.h5"    
)

model = main_training(
    optimizer = "Adam",
    n_layers = 7, n_neurons = [1024, 1024, 512, 256, 128, 64, 32],
    activation = Any[Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.sigmoid],
    epochs = 1000, batch_size = 128,
    η_min = 1e-7, η_max = 1e-4, warmup_epochs = 50,
    dropout_rate = 0.2, gamma = 3.0, α = 0.0, lr_cycles = 3
    )


model = main_training(
    optimizer = "Adam",
    n_layers = 7, n_neurons = [1024, 1024, 512, 256, 128, 64, 32],
    activation = Any[Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.sigmoid],
    epochs = 1000, batch_size = 128,
    η_min = 1e-6, η_max = 1e-5, warmup_epochs = 50,
    dropout_rate = 0.2, gamma = 3.0, lr_cycles = 3, α = 0.4
    )

model = main_training(
    optimizer = "Adam",
    n_layers = 7, n_neurons = [1024, 1024, 512, 256, 128, 64, 32],
    activation = Any[Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.sigmoid],
    epochs = 1000, batch_size = 128,
    η_min = 1e-6, η_max = 1e-5, warmup_epochs = 50,
    dropout_rate = 0.2, gamma = 3.0, lr_cycles = 3, α = 1.0
)
    

model = main_training(
    optimizer = "Adam",
    n_layers = 7, n_neurons = [1024, 1024, 512, 256, 128, 64, 32],
    activation = Any[Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.sigmoid],
    epochs = 1000, batch_size = 128,
    η_min = 1e-6, η_max = 1e-5, warmup_epochs = 50,
    dropout_rate = 0.2, gamma = 3.0, lr_cycles = 3, α = 0.0
)
    
model = main_training(
    optimizer = "Adam",
    n_layers = 9, n_neurons = [2048, 2048, 1024, 512, 256, 128, 64, 32, 16],
    activation = _define_activations("relu", 9),
    epochs = 1000, batch_size = 128,
    η_min = 1e-12, η_max = 1e-5, warmup_epochs = 50,
    dropout_rate = 0.2
    )

model = main_training(
    optimizer = "Adam",
    n_layers = 10, n_neurons = [2048, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16],
    activation = _define_activations("relu", 10),
    epochs = 1000, batch_size = 128,
    η_min = 1e-12, η_max = 1e-5, warmup_epochs = 50,
    dropout_rate = 0.2
    )
=#
# Systematic search

function hyperparameter_search()

    activation_functions = ["relu", "celu", "elu", "leakyrelu", "swish", "selu"]

    for i in 1:500, 
        # model defintion
        n_layers        = rand(collect(2:5));
        n_neurons       = rand([128,256,512,1024,2048,4096]);
        activation      = rand(activation_functions);
        # training parameters
        opt_rule        = "Adam";
        batch_size      = rand([16,32,64,128,256,512]);
        η_min           = rand(collect(logrange(1e-12, 1e-1, length = 50)));
        η_max           = rand(collect(logrange(1e-12, 1e-1, length = 50)));
        warmup_epochs   = rand(collect(1:100));
        dropout_rate    = rand(collect(0.0:0.025:0.7));
        gamma           = rand([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
        lr_cycles       = rand(collect(1:12));
        α               = rand(collect(range(0.0, 0.5, length = 50)))

        η_min >= η_max && @info "Skipping because η_min >= η_max"

        lg = WandbLogger(
            project = "DeepPPIPred2", name = "run-$(Dates.now())",
            reinit = "finish_previous",
            config = Dict(
                "min_learning_rate" => η_min,
                "max_learning_rate" => η_max,
                "warmup_epochs" => warmup_epochs,
                "lr_cycles" => lr_cycles, 
                "batch_size" => batch_size,
                "epochs" => 1000,
                "n_layers" => n_layers,
                "n_neurons" => n_neurons,
                "activation_functions" => activation_to_string(activation),
                "optimizer" => opt_rule,
                "architecture" => "Dense",
                "dropout_rate" => dropout_rate,
                "gamma" => gamma,
                "alpha" => α,
                "apply_feature_engineering" => false
            )
        )
        global_logger(lg)

        try 
            main_training(
                optimizer = opt_rule,
                n_layers = n_layers, 
                n_neurons = _define_layers(n_neurons, n_layers),
                activation = _define_activations(activation, n_layers),
                batch_size = batch_size,
                epochs = 1000,
                η_min = η_min,
                η_max = η_max, 
                warmup_epochs = warmup_epochs,
                dropout_rate = dropout_rate,
                gamma = gamma,
                lr_cycles = lr_cycles,
                lg = lg
            )
        catch e
            @info e
            @warn "Skipping due to error: $e"
            close(lg)
        end
    end

end

hyperparameter_search()


## --------------------------------------------------------------- #
### Multi-head attention model ------------------------------------ #
### --------------------------------------------------------------- #

"""
    main_mha_training(; kwargs...)

Orchestrates the training process for a Multi-Head Attention (MHA) model.

This function initializes Weights & Biases logging, constructs an MHA model
via `getMultiHeadAttentionModel`, and subsequently trains it using `train_dnn!`.
A key distinction from `main_training` is the model architecture and the omission
of the `generate_input_features` step within `train_dnn!`, as the MHA model
processes raw concatenated protein embeddings directly.

It is essential that `post_attention_n_neurons[1]` aligns with `protein_feature_dim`.
The default for `post_attention_n_neurons` is configured to automatically satisfy
this based on the chosen `protein_feature_dim`.

# Keyword Arguments
- `protein_feature_dim::Int = 512`: Dimensionality of individual protein encodings.
- `n_attention_layers::Int = 4`: Number of Transformer encoder blocks.
- `n_heads::Int = 8`: Number of attention heads.
- `attention_dropout_rate::Float64 = 0.1`: Dropout rate for the MHA layer.
- `post_attention_n_layers::Int = 4`: Number of dense layers after attention.
- `post_attention_n_neurons::Vector{Int} = [protein_feature_dim, 256, 128, 64]`: Neuron counts for post-attention dense layers. The first element *must* equal `protein_feature_dim`.
- `post_attention_activation::Vector{Union{Any,Function}} = [Flux.relu, Flux.relu, Flux.relu, Flux.sigmoid]`: Activation functions for post-attention dense layers.
- `post_attention_dropout_rate::Float64 = 0.1`: Dropout rate for post-attention dense layers.
- `optimizer::String = "AdamW"`: Optimizer name (options include "Adam", "RMSProp", and "AdamW").
- `η_min::Float64 = 1e-12`: Minimum learning rate for cosine annealing.
- `η_max::Float64 = 1e-3`: Maximum learning rate for cosine annealing.
- `lr_cycles::Int = 4`: Number of learning rate cycles for cosine annealing.
- `warmup_epochs::Int = 10`: Number of warmup epochs for the learning rate scheduler.
- `epochs::Int = 100`: Total number of training epochs.
- `batch_size::Int = 128`: Batch size for training.
- `gamma::Float64 = 2.0`: Gamma parameter for focal loss, if used.
- `l2_reg::Float64 = 0.0`: L2 regularization strength.

# Returns
- `Float64`: The minimum validation loss achieved during training. Returns `Inf` if training is skipped (e.g., `η_min >= η_max`) or if data files are missing.

# Example
```julia
min_loss = main_mha_training(
    protein_feature_dim=256,
    n_heads=4,
    epochs=50,
    batch_size=64
)
println("Minimum validation loss: \$min_loss")

# Notes:
The use of the activation functions `Flux.selu`, `Flux.celu`, and `Flux.elu` causes NaNs in the gradient during training due to a conflict between the activation function and the LayerNorm and BatchNorm layers. 
```
"""
function main_mha_training(;
    protein_feature_dim::Int = 512,
    n_attention_layers::Int = 2,
    n_heads::Int = 8,
    attention_dropout_rate::Float64 = 0.1,
    post_attention_n_layers::Int = 4,
    post_attention_n_neurons::Vector{Int} = [protein_feature_dim, 256, 128, 64],
    post_attention_activation::Vector{Union{Any,Function}} = Any[Flux.relu, Flux.relu, Flux.relu, Flux.sigmoid],
    post_attention_dropout_rate::Float64 = 0.1,
    optimizer::String = "AdamW",
    η_min::Float64 = 1e-12,
    η_max::Float64 = 1e-3,
    lr_cycles::Int = 4,
    warmup_epochs::Int = 50,
    epochs::Int = 1000,
    batch_size::Int = 128,
    gamma::Float64 = 2.0,
    l2_reg::Float64 = 0.0, 
    logger = nothing,
    α::Float64 = 0.1
    )

    # Ensure the first element of post_attention_n_neurons matches protein_feature_dim
    if isempty(post_attention_n_neurons) || post_attention_n_neurons[1] != protein_feature_dim
        @warn "Defaulting or adjusting post_attention_n_neurons[1] to match protein_feature_dim = $protein_feature_dim."
        _neurons = copy(post_attention_n_neurons) 
        if isempty(_neurons) && post_attention_n_layers > 0
            _neurons = [protein_feature_dim] 
        elseif !isempty(_neurons)
            _neurons[1] = protein_feature_dim
        end
    end

    η_min >= η_max && return Inf

    dataset_name_in_h5 = "features_labels"
    train_hdf5_file = "encodings/train_data.h5"
    val_hdf5_file = "encodings/val_data.h5"

    if !isfile(train_hdf5_file)
        @error "HDF5 training data file '$train_hdf5_file' not found."
        return Inf # Return Inf for hyperopt compatibility
    end

    if logger === nothing
    lg = WandbLogger(
        project = "DeepPPIPredMHA2", name = "mha-run-$(Dates.now())",
        reinit = "finish_previous",
        config = Dict(
            "protein_feature_dim" => protein_feature_dim,
            "n_attention_layers" => n_attention_layers,
            "n_heads" => n_heads,
            "attention_dropout_rate" => attention_dropout_rate,
            "post_attention_n_layers" => post_attention_n_layers,
            "post_attention_n_neurons_start" => post_attention_n_neurons[1],
            "post_attention_activation_first" => isempty(post_attention_activation) ? "N/A" : activation_to_string(post_attention_activation[1]),
            "post_attention_dropout_rate" => post_attention_dropout_rate,
            "min_learning_rate" => η_min,
            "max_learning_rate" => η_max,
            "lr_cycles" => lr_cycles, 
            "warmup_epochs" => warmup_epochs,
            "batch_size" => batch_size,
            "epochs" => epochs,
            "optimizer" => optimizer,
            "architecture" => "MultiHeadAttention",
            "gamma" => gamma,
            "l2_reg" => l2_reg,
            "alpha" => α
        )
    )
    global_logger(lg)
    end


    dev = CUDA.functional() ? Flux.gpu : Flux.cpu

    model = getProteinTransformerModel(
        protein_feature_dim, 
        n_attention_layers,
        n_heads,
        attention_dropout_rate,
        post_attention_n_layers,
        post_attention_n_neurons,
        post_attention_activation,
        post_attention_dropout_rate
    )
    model = model |> dev

    @info "Starting MHA model training..."

    min_val_loss = train_dnn!(
        model, train_hdf5_file, dataset_name_in_h5,
        val_hdf5_file, dataset_name_in_h5,
        optimizer = optimizer, 
        η_min = η_min, η_max = η_max, 
        lr_cycles = lr_cycles, 
        warmup_epochs = warmup_epochs,
        epochs = epochs, 
        batch_size = batch_size, device = dev,
        gamma = gamma,
        apply_feature_engineering = false,
        l2_reg = l2_reg,
        α = α
    )

    @info "MHA model training finished."
    # @save "sweep/trained_mha_model_$(Dates.now()).bson" model # Consider saving MHA model
    close(lg)
    return min_val_loss
end

#=
activation_functions = ["relu", "gelu", "leakyrelu", "swish"]

@hyperopt for i=1000, sampler=RandomSampler(),
    opt_rule = ["Adam", "RMSProp", "AdamW"],
    l2_reg = LinRange(0.0, 1e-5, 12),
    n_layers = Int64.(LinRange(3, 12, 10)),
    n_neurons = Int64.(round.(2 .^LinRange(5,12,8), digits = 0)),
    activation = activation_functions,
    # batch sizes of 64 showed reduced val metrics in first sweep
    batch_size = Int64[128, 256, 512],
    dropout_rate = Float64[0.1,0.2,0.3,0.4,0.5,0.6,0.7],

    #  Attention parameters
    n_attention_layers = Int64.(LinRange(4, 12, 9)),
    n_heads = Int64[16,32,64,128], # four heads had worst performance in first sweep
    attention_dropout_rate = Float64[0.1,0.2,0.3,0.4,0.5,0.6,0.7],

    # cosine annealing parameters
    η_min = 10 .^LinRange(-12, -1, 50), 
    η_max = 10 .^LinRange(-6, -2, 50),   
    warmup_epochs = Int64.(round.(LinRange(5, 50, 46), digits = 0)),
    lr_cycles = Int64.(round.(LinRange(2, 20, 19), digits = 0)),

    # loss parameters
    gamma = Float64.(LinRange(0.0, 5.0, 51)),   

    # data augmentation
    α = Float64[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # skipping conditions
    η_min >= η_max && @info "Skipping because η_min >= η_max"

    if n_attention_layers >=6 && η_max > 1e-4
        # the first runs of the hyperparameter search indicated that the combination of 
        # a high learning rate and high n_attention_layers caused the model to diverge
        @info "Setting η_max = 1e-4 because n_attention_layers >= 6 and η_max > 1e-4"
        η_max = 1e-4
    end

    # define logger
    lg = WandbLogger(
        project = "DeepPPIPredMHA2", name = "mha-run-$(Dates.now())",
        reinit = "finish_previous",
        config = Dict(
            "protein_feature_dim" => 512,
            "n_attention_layers" => n_attention_layers,
            "n_heads" => n_heads,
            "attention_dropout_rate" => attention_dropout_rate,
            "post_attention_n_layers" => n_layers,
            "post_attention_n_neurons_start" => _define_layers(n_neurons, n_layers)[2],
            "post_attention_activation_first" => activation,
            "post_attention_dropout_rate" => dropout_rate,
            "min_learning_rate" => η_min,
            "max_learning_rate" => η_max,
            "lr_cycles" => lr_cycles, 
            "warmup_epochs" => warmup_epochs,
            "batch_size" => batch_size,
            "epochs" => 1000,
            "optimizer" => opt_rule,
            "architecture" => "MultiHeadAttention",
            "gamma" => gamma,
            "l2_reg" => l2_reg,
            "alpha" => α
        )
    )
    global_logger(lg)

    try
        main_mha_training(
            protein_feature_dim = 512,
            n_attention_layers = n_attention_layers,
            n_heads = n_heads,
            attention_dropout_rate = attention_dropout_rate,
            post_attention_n_layers = n_layers,
            post_attention_n_neurons = _define_layers(n_neurons, n_layers),
            post_attention_activation = _define_activations(activation, n_layers),
            post_attention_dropout_rate = dropout_rate,
            optimizer = opt_rule,
            η_min = η_min,
            η_max = η_max,
            warmup_epochs = warmup_epochs,
            lr_cycles = lr_cycles,
            epochs = 1000,
            batch_size = batch_size,
            gamma = gamma,
            l2_reg = l2_reg,
            logger = lg,
            α = α
    )    
    catch e
        @warn "Skipping due to error: $e"
        close(lg)
    end
end


main_mha_training(
    protein_feature_dim = 512,
    n_attention_layers = 12,
    n_heads = 128,
    attention_dropout_rate = 0.3,
    post_attention_n_layers = 7,
    post_attention_n_neurons = [2048, 1024, 512, 256, 128, 64, 32],
    post_attention_activation = Any[Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.sigmoid],
    post_attention_dropout_rate = 0.3,
    optimizer = "AdamW",
    η_min = 1e-8,
    η_max = 1e-4,
    warmup_epochs = 50,
    lr_cycles = 4,
    epochs = 1000,
    batch_size = 512,
    gamma = 2.0,
    l2_reg = 0.0,
    logger = nothing,
    α = 0.4
)

=#