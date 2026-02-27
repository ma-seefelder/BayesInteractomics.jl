import CSV
import DataFrames: DataFrame, shuffle!, combine, groupby, nrow
import HDF5
using ProgressMeter

"""
    readHDF5(file::AbstractString = raw"encodings/9606.protein.network.embeddings.v12.0.h5")

Reads embeddings from an HDF5 file.

# Arguments
- `file`: Path to the HDF5 file containing the embeddings.
"""
function readHDF5(file::S) where {S<:AbstractString}
    fid = HDF5.h5open(file, "r") 
    data::Matrix{Float32} = HDF5.read(fid, "embeddings")  
    close(fid)
    return data
end

"""
    get_species_ids(; file_name::AbstractString = "encodings/interresidue_distance.csv")

Extracts unique species identifiers from a CSV file containing protein pairs.

The function reads a CSV file where each row represents a pair of proteins. It assumes
protein identifiers are in the format `species_id.protein_id`. It parses these identifiers
from two columns (`Protein1` and `Protein2`) to compile a comprehensive list of unique
species IDs present in the dataset.

# Arguments
- `file_name::AbstractString`: The path to the input CSV file. The file is expected to
  have columns named `Protein1` and `Protein2` containing protein identifiers. Defaults
  to `"encodings/interresidue_distance.csv"`.

# Returns
- `Vector{String}`: A vector containing the unique species identifiers found in the file.
"""
function get_species_ids(; file_name::S =  "encodings/interresidue_distance.csv") where {S<:AbstractString}
    df = CSV.File(
        file_name, header=[:Protein1, :Protein2, :MinDistance], 
        types = [String, String, Float16], delim = ',', stringtype = String
    ) |> DataFrame

    # revtrieve all species ids
    ids_protein_1 = map(x -> split(x, ".")[1], df.Protein1)
    ids_protein_2 = map(x -> split(x, ".")[1], df.Protein2)

    unique!(ids_protein_1)
    unique!(ids_protein_2)

    species_ids = unique(ids_protein_1 ∪ ids_protein_2)

    return species_ids
end


"""
    combine_embeddings(group_1::HDF5.Group, group_2::HDF5.Group, species::String; output_file::AbstractString)

Combines protein embeddings from two HDF5 groups and saves them to a new file.

This function takes two HDF5 groups, each assumed to contain a 'proteins' dataset (a vector of
protein identifiers) and an 'embeddings' dataset (a matrix of embedding vectors). It vertically
concatenates the embeddings for each corresponding protein.

If the protein identifiers in the two groups are not in the same order, the function attempts
to align them by sorting. An error is thrown if the sets of proteins in the two groups are
not identical.

The combined embeddings and the unified list of proteins are then written to a new group,
named after the `species` parameter, within the specified `output_file`.

# Arguments
- `group_1::HDF5.Group`: The first HDF5 group, containing 'proteins' and 'embeddings' datasets.
- `group_2::HDF5.Group`: The second HDF5 group, with the same structure as `group_1`.
- `species::String`: The name for the new group to be created in the output HDF5 file.

# Keyword Arguments
- `output_file::AbstractString`: Path to the output HDF5 file. Defaults to `"encodings/combined_embeddings.h5"`. The file will be created if it does not exist.

# Side Effects
- Creates or modifies the `output_file` by adding a new group for the specified `species` containing the combined embeddings.

# Throws
- `ErrorException`: If the protein lists in `group_1` and `group_2` do not contain the same set of proteins and cannot be aligned.
"""
function combine_embeddings(group_1::HDF5.Group, group_2::HDF5.Group, species::String; output_file::AbstractString = "encodings/combined_embeddings.h5")
    # check that the order of the proteins is the same
    proteins_1 = HDF5.read(group_1, "proteins")
    proteins_2 = HDF5.read(group_2, "proteins")

    # load the embeddings and concatenate them
    embedding_1 = HDF5.read(group_1, "embeddings")
    embedding_2 = HDF5.read(group_2, "embeddings")
    
    if proteins_1 != proteins_2
        # reorder proteins_2 to match proteins_1
        reorder_idx_1 = sortperm(proteins_1)
        reorder_idx_2 = sortperm(proteins_2)
        proteins_1 = proteins_1[reorder_idx_1]
        proteins_2 = proteins_2[reorder_idx_2]

        # confirm that the order is the same
        if proteins_1 != proteins_2
            error("Proteins in the two files are not in the same order.")
        end

        # reorder the embeddings
        embedding_1 = embedding_1[:, reorder_idx_1]
        embedding_2 = embedding_2[:, reorder_idx_2]
    end
    embedding = vcat(embedding_1, embedding_2)
    
    HDF5.h5open(output_file, "cw") do output_file_open
        group = HDF5.create_group(output_file_open, species)
        
        embedding_dset = HDF5.create_dataset(group, "embeddings", Float32, size(embedding))
        HDF5.write(embedding_dset, embedding)
        
        protein_dset = HDF5.create_dataset(group, "proteins", String, size(proteins_1))
        HDF5.write(protein_dset, proteins_1)
    end
end

"""
    combine_embeddings(
        file_1::AbstractString,
        file_2::AbstractString;
        output_file::AbstractString = "encodings/combined_embeddings.h5"
    )

Combines protein embeddings from two HDF5 files on a per-species basis.

This function orchestrates the combination of embeddings from two separate HDF5
files (`file_1` and `file_2`). It assumes both files have a similar structure,
with a main group "species" containing subgroups for each species ID.

It determines the list of species to process by taking all species from
`file_1` and combining them with a list of species derived from an external CSV
file (in that case the whole dataset of interresidue_distance from PDB) (via `get_species_ids()`). 
It then iterates over this combined list of species, combines their respective embeddings using 
the `combine_embeddings` method for HDF5 groups, and writes the result to a single new HDF5 file. The 
format of is file is comparable to the original file.

# Arguments
- `file_1::AbstractString`: Path to the first HDF5 file containing embeddings.
- `file_2::AbstractString`: Path to the second HDF5 file containing embeddings.

# Keyword Arguments
- `output_file::AbstractString`: Path for the new HDF5 file that will store the
  combined embeddings. Defaults to `"encodings/combined_embeddings.h5"`.

# Throws
- `ErrorException`: If `output_file` already exists, or if `file_1` or `file_2`
  cannot be found.
"""
function combine_embeddings(file_1::AbstractString, file_2::AbstractString; output_file::AbstractString = "encodings/combined_embeddings.h5")
    # check requirements
    isfile(output_file) && error("Output file already exists: $output_file")
    isfile(file_1) || error("File 1 does not exist: $file_1")
    isfile(file_2) || error("File 2 does not exist: $file_2")

    HDF5.h5open(file_1, "r") do file_1
        HDF5.h5open(file_2, "r") do file_2
            # --- Determine species to process
            file_1_group = file_1["species"]
            file_2_group = file_2["species"]

            species_ids_from_file = keys(file_1_group)
            species_ids_in_dataset = get_species_ids()
            species_to_process = unique(species_ids_from_file ∩ species_ids_in_dataset)
            # --- process each species ---
            @showprogress "Combining embeddings for species..." for species in species_to_process
                combine_embeddings(file_1_group[species], file_2_group[species], species, output_file = output_file)
            end
        end
    end
    return nothing
end

"""
    generate_dataset()

Generates a feature/label dataset from inter-residue distances and protein embeddings.

This high-performance version avoids common bottlenecks by:
1.  **Aggregating Duplicate Pairs:** It first identifies all observations for each unique
    protein pair and aggregates them by taking the **minimum distance**. This resolves
    ambiguity and ensures each protein pair has only one row and one ground-truth label.
2.  **Grouping by Species:** It processes the aggregated data one species at a time.
3.  **Hash-Based Lookups:** For each species, it builds a `Dict` for near-instantaneous
    embedding lookups.
4.  **Batched & Extendable HDF5 Writing:** It writes data in efficient batches.
"""
function generate_dataset()
    # ----------------------------------------------------------------------- #
    # --- 1. Generate combined embedding dataset
    # ----------------------------------------------------------------------- #
    @info "Generating combined embedding dataset..."
    output_file = "encodings/combined_embeddings.h5"
    file_1 = "encodings/protein.sequence.embeddings.v12.0.h5"
    file_2 = "encodings/protein.network.embeddings.v12.0.h5"
    combine_embeddings(file_1, file_2; output_file = output_file)
    @info "Combined embedding dataset generated."

    # ----------------------------------------------------------------------- #
    # --- 2. Load and PRE-PROCESS Input Data 
    # ----------------------------------------------------------------------- #
    @info "Reading inter-residue distance data..."

    df_raw = CSV.File(
        "encodings/interresidue_distance.csv", header=[:Protein1, :Protein2, :MinDistance], 
        types = [String, String, Float16], delim = ',', stringtype = String
    ) |> DataFrame

    @info "Aggregating duplicate protein pairs by minimum distance..."
    # Create a canonical, order-independent key for each pair
    df_raw.join_key = [join(sort([r.Protein1, r.Protein2]), "_") for r in eachrow(df_raw)]


    # Group by the unique key and find the minimum distance for each group
    df = combine(groupby(df_raw, :join_key), 
                 :MinDistance => minimum => :MinDistance, # Take the minimum distance
                 :Protein1 => first => :Protein1,       # Take the first protein name (they are consistent within a key)
                 :Protein2 => first => :Protein2)       # Take the second protein name
    
    @info "Aggregation complete. Reduced $(nrow(df_raw)) observations to $(nrow(df)) unique pairs."
    
    df.species_id = [split(p, '.')[1] for p in df.Protein1]
    gdf = groupby(df, :species_id)
    @info "Found $(length(gdf)) species groups in the unique pairs data."

    # ----------------------------------------------------------------------- #
    # --- 3. Setup Output HDF5 File with Extendable Datasets
    # ----------------------------------------------------------------------- #
    EMBEDDING_DIM = 512 + 1024
    FEATURE_VECTOR_LENGTH = 2 * EMBEDDING_DIM + 1
    output_h5_filename = "encodings/full_dataset_optimized.h5"
    output_dataset_name = "features_labels"
    output_datatype = Float32
    
    max_size = size(df, 1)

    isfile(output_h5_filename) && rm(output_h5_filename)

    HDF5.h5open(output_h5_filename, "w") do out_fid
        dset = HDF5.create_dataset(
            out_fid, output_dataset_name, output_datatype, 
            ((0, FEATURE_VECTOR_LENGTH), (max_size, FEATURE_VECTOR_LENGTH)),
            chunk=(1024, FEATURE_VECTOR_LENGTH)
        ) 
        proteins_dset = HDF5.create_dataset(
            out_fid, "proteins", String, 
            ((0, 2), (max_size, 2)),
            chunk=(1024, 2)
        )

        # save the join keys for save matching
        join_key_dset = HDF5.create_dataset(
            out_fid, "join_keys", String,
            ((0,), (max_size,)),
            chunk=(2048,)
        )

        current_rows_written = 0
        # ----------------------------------------------------------------------- #
        # --- 4. Process Data in Batches (Species by Species)
        # ----------------------------------------------------------------------- #
        @info "Opening embeddings HDF5 file for reading..."
        HDF5.h5open(output_file, "r") do in_fid
            @showprogress "Processing species..." for species_group in gdf
                species_id = species_group.species_id[1]
                
                local species_embeddings
                try
                    species_embeddings = HDF5.read(in_fid, "$species_id")
                catch e
                    @error "Failed to read embeddings for species '$species_id': $e. Skipping $(nrow(species_group)) rows."
                    continue
                end

                protein_list = species_embeddings["proteins"]
                embedding_matrix = species_embeddings["embeddings"]
                protein_to_idx = Dict(protein_id => i for (i, protein_id) in enumerate(protein_list))

                n_rows_in_batch = nrow(species_group)
                batch_features = Matrix{output_datatype}(undef, n_rows_in_batch, FEATURE_VECTOR_LENGTH)
                batch_proteins = Matrix{String}(undef, n_rows_in_batch, 2)
                batch_join_keys = Vector{String}(undef, n_rows_in_batch)
                valid_rows_in_batch = 0

                for row in eachrow(species_group)
                    idx1 = get(protein_to_idx, row.Protein1, 0)
                    idx2 = get(protein_to_idx, row.Protein2, 0)
                    (idx1 == 0 || idx2 == 0) && continue

                    valid_rows_in_batch += 1
                    embedding1 = view(embedding_matrix, :, idx1)
                    embedding2 = view(embedding_matrix, :, idx2)
                    is_interacting = row.MinDistance <= 5.0 ? 1.0f0 : 0.0f0

                    batch_features[valid_rows_in_batch, :] = vcat(embedding1, embedding2, is_interacting)
                    batch_proteins[valid_rows_in_batch, :] = [row.Protein1, row.Protein2]
                    batch_join_keys[valid_rows_in_batch] = row.join_key
                end

                if valid_rows_in_batch > 0
                    new_total_rows = current_rows_written + valid_rows_in_batch
                    HDF5.set_extent_dims(dset, (new_total_rows, FEATURE_VECTOR_LENGTH))
                    HDF5.set_extent_dims(proteins_dset, (new_total_rows, 2))
                    HDF5.set_extent_dims(join_key_dset, (new_total_rows,))
                    
                    start_idx = current_rows_written + 1
                    end_idx = new_total_rows
                    
                    dset[start_idx:end_idx, :] = view(batch_features, 1:valid_rows_in_batch, :)
                    proteins_dset[start_idx:end_idx, :] = view(batch_proteins, 1:valid_rows_in_batch, :)
                    join_key_dset[start_idx:end_idx] = view(batch_join_keys, 1:valid_rows_in_batch)
                    
                    current_rows_written = new_total_rows
                end
            end
        end 

        @info "Processing complete. A total of $current_rows_written valid unique pairs were written."
    end
    @info "Successfully generated aggregated dataset at: $output_h5_filename"
end
