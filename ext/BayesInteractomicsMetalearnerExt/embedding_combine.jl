# combine_embeddings extracted into the Metalearner extension include graph.
#
# This is the HDF5.Group group-method extracted VERBATIM from the offline builder
# scripts/dnn_training/generate_dataset.jl:88-124. Only the group-method is lifted
# here — NOT the AbstractString file-path overload (generate_dataset.jl:158), which
# drags ProgressMeter (@showprogress), CSV, and get_species_ids() (Pitfall 4). The
# extension already imports HDF5 (used throughout metalearner.jl), so no new import
# is needed.
#
# `prediction_data` (metalearner.jl) calls this method on the first non-human
# inference to build encodings/emb_<sp>.h5 by vcat-ing the 1024-dim sequence
# embedding with the 512-dim network embedding → a 1536-row matrix written under an
# HDF5 group named by the species string. Before this extraction the symbol lived
# only in the offline builder and resolved at the call site by accident (the human
# emb_9606.h5 pre-existed, so the `if !isfile(...)` branch never ran); the first
# non-human inference threw UndefVarError(:combine_embeddings).

"""
    combine_embeddings(group_1::HDF5.Group, group_2::HDF5.Group, species::String; output_file)

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
