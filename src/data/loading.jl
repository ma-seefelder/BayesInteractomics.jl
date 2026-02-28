#=
BayesInteractomics: A Julia package for the analysis of protein interactome data from Affinity-purification mass spectrometry (AP-MS) and proximity labelling experiments
# Version: 0.1.0

Copyright (C) 2024  Dr. rer. nat. Manuel Seefelder
E-Mail: manuel.seefelder@uni-ulm.de
Postal address: Department of Gene Therapy, University of Ulm, Helmholzstr. 8/1, 89081 Ulm, Germany

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
=#

"""
    InteractionData(
    protein_IDs::Vector{String},
    protein_names::Vector{String},
    samples::Dict{I, Protocol{F, I}},
    controls::Dict{I, Protocol{F, I}}
    ) where {I<:Integer, F<:AbstractFloat}

    This function returns an InteractionData object with the following fields:

        - protein_IDs: A vector of protein IDs.
        - protein_names: A vector of protein names.
        - samples: A dictionary with protocol indices as keys and Protocol objects as values.
        - controls: A dictionary with protocol indices as keys and Protocol objects as values.
        - no_protocols: The number of protocols.
        - no_experiments: A dictionary with protocol indices as keys and the number of experiments as values.
        - no_parameters_HBM: The number of parameters for the HierarchicalBayesianModel.
        - no_parameters_Regression: The number of parameters for the Regression model.
        - experiment_positions: A vector of positions where parameters for individual experiments are stored.
        - protocol_positions: A vector of positions where parameters for individual protocols are stored.
        - matched_positions: A vector of positions where the value shows the protocol index for each experiment.
"""
function InteractionData(
    protein_IDs::Vector{String},
    protein_names::Vector{String},
    samples::Dict{I,Protocol{F,I}},
    controls::Dict{I,Protocol{F,I}}
) where {F<:AbstractFloat,I<:Integer}

    # Sanity checkForDuplicates
    isempty(protein_IDs) && throw(ArgumentError("protein_IDs cannot be empty"))
    isempty(protein_names) && throw(ArgumentError("protein_names cannot be empty"))
    keys(controls) != keys(samples) && throw(ArgumentError("Keys and length of controls and samples must match"))

    num_protocols = length(samples)
    all(1:num_protocols .∈ Ref(keys(samples))) || throw(ArgumentError("Samples dictionary is missing protocol indices"))
    all(1:num_protocols .∈ Ref(keys(controls))) || throw(ArgumentError("Controls dictionary is missing protocol indices"))

    # Function body
    num_protocols = length(samples)
    num_experiments_per_protocol = Dict(i => getNoExperiments(samples[i]) for i in 1:num_protocols)
    num_params_regression = 1 + num_protocols
    # number of parameters for the HierarchicalBayesianModel
    num_params_HBM = 1 + sum(values(num_experiments_per_protocol)) + num_protocols

    # get positions
    protocol_positions, experiment_positions, matched_positions = getPositions(num_experiments_per_protocol, num_params_HBM)

    data = InteractionData(
        protein_IDs, protein_names,
        samples, controls,
        num_protocols, num_experiments_per_protocol,
        num_params_HBM, num_params_regression,
        protocol_positions, experiment_positions, matched_positions
    )
    validate(data) == false && @warn "Protein names of protocols do not match! DO NOT PROCEED with the analysis"
    return data
end

"""
    create_protocol(data, cols::Dict{I, Vector{I}}, no_experiments::I, protein_ids) where I<:Integer

    Create a `Protocol` object from a data matrix and a mapping of column indices for each experiment.

    # Arguments
    - `data::DataFrame`: The input data matrix, where rows correspond to proteins and columns to samples or controls.
    - `cols::Dict{I, Vector{I}}`: A dictionary mapping experiment indices to vectors of column indices in `data`. Each entry specifies the columns belonging to that experiment.
    - `no_experiments::I`: The total number of experiments to include in the protocol.
    - `protein_ids::Vector{String}`: A vector of protein identifiers corresponding to the rows in `data`.

    # Returns
    - A `Protocol{Float64, I}` object containing the extracted experiment matrices.

    # Notes
    - Each data matrix in the protocol has dimensions `(num_proteins, num_samples_per_experiment)`.
    - This function assumes all columns specified in `cols` exist in `data`.

"""
function create_protocol(data, cols::Dict{I,Vector{I}}, no_experiments::I, protein_ids) where I<:Integer
    return Protocol{Float64,I}(
        no_experiments, protein_ids,
        Dict([i => Matrix(data[:, cols[i]]) for i ∈ 1:no_experiments])
    )
end


flatten_rows(x::M) where {M<:AbstractMatrix} = vec(permutedims(deepcopy(x)))
custom_mean(x::Vector) = all(ismissing, x) ? missing : mean(skipmissing(x))
custom_mean(x::M) where {M<:AbstractMatrix} = custom_mean(flatten_rows(x))
custom_var(x::Vector) = all(ismissing, x) ? missing : var(skipmissing(x))
custom_var(x::M) where {M<:AbstractMatrix} = custom_var(flatten_rows(x))

sample_cols = Dict(1 => [2, 3, 4, 162], 2 => [5, 6, 7, 162], 3 => [162, 163, 164, 165], 4 => [162, 163, 164, 165])
control_cols = Dict(1 => [2, 3, 4, 162], 2 => [5, 6, 7, 162], 3 => [162, 163, 164, 165], 4 => [162, 163, 164, 165])

"""
    impute_missing_values!(
    data::DataFrame, sample_cols::Dict{I, Vector{I}}, 
    control_cols::Dict{I, Vector{I}}
    ) where I<:Integer

Impute missing values in a data matrix using row-wise or global means and variances.

# Arguments
- `data::DataFrame`: Input matrix with rows as proteins and columns as samples/controls.
- `sample_cols::Dict{I, Vector{I}}`: Dictionary mapping experiment index to sample column indices.
- `control_cols::Dict{I, Vector{I}}`: Dictionary mapping experiment index to control column indices.
- `impute::Bool`: Whether to actually perform imputation (vs. just preparing stats).

# Behavior
- Missing values are imputed with values drawn from a normal distribution.
- Mean and variance are computed per row; if unavailable, global values are used.
- Default variance fallback is 3.0 when none is computable.

# Returns
- The modified `data` DataFrame (in-place).

"""
function impute_missing_values!(
    data::DataFrame, sample_cols::Dict{I,Vector{I}},
    control_cols::Dict{I,Vector{I}}
) where I<:Integer

    sample_cols_unwrapped = vcat([sample_cols[i] for i in 1:length(sample_cols)]...)
    control_cols_unwrapped = vcat([control_cols[i] for i in 1:length(control_cols)]...)

    global_mean_sample = custom_mean(Matrix(data[:, sample_cols_unwrapped]))
    global_mean_control = custom_mean(Matrix(data[:, control_cols_unwrapped]))

    for row ∈ axes(data, 1)
        sample_row = collect(data[row, sample_cols_unwrapped])
        control_row = collect(data[row, control_cols_unwrapped])

        row_mean_sample, row_mean_control = custom_mean(sample_row), custom_mean(control_row)
        row_var_sample, row_var_control = custom_var(sample_row), custom_var(control_row)

        row_mean_sample = coalesce(row_mean_sample, global_mean_sample)
        row_mean_control = coalesce(row_mean_control, global_mean_control)

        (ismissing(rowvar_samples) || isnan(rowvar_samples)) && (row_var_sample = 3.0)
        (ismissing(rowvar_controls) || isnan(rowvar_controls)) && (row_var_control = 3.0)

        # impute the missing values
        for column ∈ axes(data, 2)
            ismissing(data[row, column]) == false && continue

            if column ∈ sample_cols_unwrapped
                data[row, column] = rand(Normal(row_mean_sample, row_var_sample))
            elseif column ∈ control_cols_unwrapped
                data[row, column] = rand(Normal(row_mean_control, row_var_control))
            end
        end
    end
    return data
end

"""
    extract_data(
    data::DataFrame, 
    sample_cols::Dict{I, Vector{I}}, 
    control_cols::Dict{I, Vector{I}},
    name_col::I, id_col::I, impute::Bool
    ) where I<:Integer

    Extracts the data from the csv file.

    Args:
    - data (DataFrame): The data matrix.
    - sample_cols (Dict{I, Vector{I}}): A dictionary of column indices for the samples.
    - control_cols (Dict{I, Vector{I}}): A dictionary of column indices for the controls.
    - name_col (I): The column index for the protein names.
    - id_col (I): The column index for the protein IDs.
    - impute (Bool): Whether to impute missing values.

    Returns:
    - samples (Dict{Int, Protocol}): A dictionary of protocols for the samples.
    - controls (Dict{Int, Protocol}): A dictionary of protocols for the controls.
    - protein_ids (Vector{String}): A vector of protein IDs.
    - protein_names (Vector{String}): A vector of protein names.
"""
function extract_data(
    data::DataFrame,
    sample_cols::Dict{I,Vector{I}},
    control_cols::Dict{I,Vector{I}},
    name_col::I, id_col::I, impute::Bool
) where I<:Integer

    # Validate consitency
    length(control_cols) != length(sample_cols) && throw(ArgumentError("The number of experiments for samples and controls must be the same"))

    #Extract identifiers
    protein_ids = string.(data[:, id_col])
    protein_names = string.(data[:, name_col])
    num_experiments = length(sample_cols)

    # Optionally impute
    impute && (data = impute_missing_values!(data, sample_cols, control_cols))

    # Build protocol structure
    samples = create_protocol(data, sample_cols, num_experiments, protein_ids)
    controls = create_protocol(data, control_cols, num_experiments, protein_ids)

    return samples, controls, String.(protein_ids), String.(protein_names)
end

"""
    load_csv(
    file::String, sample_cols::Dict{I, Vector{I}}, control_cols::Dict{I, Vector{I}},
    name_col::I, id_col::I, impute::Bool
    ) where I<:Integer

    Get interaction data from a csv file. 

    Args: 
        - file (String): The path to the csv file.
        - samples (Dict{Int, Protocol}): A dictionary of protocols for the samples.
        - controls (Dict{Int, Protocol}): A dictionary of protocols for the controls.
        - ids (Vector{String}): A vector of protein IDs.
        - name_col (Integer): The column index for the protein names.
        - id_col (Integer): The column index for the protein IDs.
        - impute (Bool): Whether to impute missing values.

    Returns: 
        - samples (Dict{Int, Protocol}): A dictionary of protocols for the samples.
        - controls (Dict{Int, Protocol}): A dictionary of protocols for the controls.
        - ids (Vector{String}): A vector of protein IDs.
        - names (Vector{String}): A vector of protein names.
"""
function load_csv(
    file::String, sample_cols::Dict{I,Vector{I}}, control_cols::Dict{I,Vector{I}},
    name_col::I, id_col::I, impute::Bool
) where I<:Integer

    check_file(file)
    data = read(file, DataFrame)
    return extract_data(data, sample_cols, control_cols, name_col, id_col, impute)
end

"""
    load_xlsx(
    file::String, sample_cols::Dict{I, Vector{I}}, control_cols::Dict{I, Vector{I}},
    name_col::I, id_col::I, impute::Bool; 
    sheet_name::String = "Sheet1"
    ) where I<:Integer

     Get interaction data from an xlsx file. 

    Args: 
        - file (String): The path to the xlsx file.
        - samples (Dict{Int, Protocol}): A dictionary of protocols for the samples.
        - controls (Dict{Int, Protocol}): A dictionary of protocols for the controls.
        - ids (Vector{String}): A vector of protein IDs.
        - name_col (Integer): The column index for the protein names.
        - id_col (Integer): The column index for the protein IDs.
        - impute (Bool): Whether to impute missing values.

    Keyword Args:
        - sheet_name (String): The name of the sheet in the xlsx file.

    Returns: 
        - samples (Dict{Int, Protocol}): A dictionary of protocols for the samples.
        - controls (Dict{Int, Protocol}): A dictionary of protocols for the controls.
        - ids (Vector{String}): A vector of protein IDs.
        - names (Vector{String}): A vector of protein names.
"""
function load_xlsx(
    file::String, sample_cols::Dict{I,Vector{I}}, control_cols::Dict{I,Vector{I}},
    name_col::I, id_col::I, impute::Bool;
    sheet_name::String="Sheet1"
) where {I<:Integer}

    check_file(file)
    data = DataFrame(readtable(file, sheet_name))
    return extract_data(data, sample_cols, control_cols, name_col, id_col, impute)
end


# Normalisation
"""
    compute_protocol_means(
    num_proteins::I, num_protocols::I, 
    samples::Dict, controls::Dict
    ) where I<:Integer

Compute a 3D array of means across all proteins, protocols, and experiments.

# Arguments
- `num_proteins::I`: Number of proteins (rows).
- `num_protocols::I`: Number of protocols.
- `samples::Dict{I, Protocol}`: Dictionary mapping protocol index to sample `Protocol`.
- `controls::Dict{I, Protocol}`: Dictionary mapping protocol index to control `Protocol`.

# Returns
- A 3D array `means[protocol, protein, experiment]` of type `Array{Union{Missing, Float64}, 3}`.

Each entry stores the mean intensity of a protein in a specific experiment and protocol, computed over sample and control values.
"""
function compute_protocol_means(
    num_proteins::I, num_protocols::I,
    samples::Dict, controls::Dict
) where I<:Integer

    # gnerate matrix of means for each protein (row) and protocol (column)
    max_experiments = maximum([samples[i].no_experiments for i in keys(samples)])
    means = zeros(Union{Missing,Float64}, num_protocols, num_proteins, max_experiments)

    for protocol_id ∈ keys(samples)
        proto_means, num_experiments = _compute_means(controls[protocol_id], samples[protocol_id], num_proteins)
        @inbounds means[protocol_id, :, 1:num_experiments] .= proto_means
    end

    return means
end

function _compute_means(control, samples, nproteins)
    nexperiments = samples.no_experiments
    means = zeros(Union{Missing,Float64}, nproteins, nexperiments)
    means .= missing

    # iterate over experiments 
    @inbounds for exp ∈ 1:nexperiments
        control_data = control[exp]
        sample_data = samples[exp]

        for pid ∈ 1:nproteins
            values = vcat(control_data[pid, :], sample_data[pid, :])
            values_no_missing = skipmissing(values)
            n_valid = count(!ismissing, values)
            n_valid > 0 && @inbounds means[pid, exp] = sum(values_no_missing) / n_valid
        end
    end
    return means, nexperiments
end

"""
    compute_protocol_means(data::InteractionData)

    Compute the mean of each protein in each protocol for the samples and controls together.

    Args:
        - data (InteractionData): The interaction data object.

    Returns:
        - protocol_means (Matrix{Union{Missing, Float64}}): A matrix of means for each protein (rows) in each protocol (columns).
"""
function compute_protocol_means(data::InteractionData)
    samples, controls = deepcopy(getSamples(data)), deepcopy(getControls(data))
    nproteins = length(getIDs(data))
    num_protocols = getNoProtocols(data)
    return compute_protocol_means(nproteins, num_protocols, samples, controls)
end

function normalize(data::InteractionData)
    # compute mean
    protocol_means = compute_protocol_means(data)

    # Deepcopy + recreate Protocols with deepcopied matrices
    samples = Dict(
        pid => Protocol(
            p.no_experiments,
            p.protein_ids,
            Dict(exp => copy(p.data[exp]) for exp in keys(p.data))
        ) for (pid, p) in getSamples(data)
    )

    controls = Dict(
        pid => Protocol(
            p.no_experiments,
            p.protein_ids,
            Dict(exp => copy(p.data[exp]) for exp in keys(p.data))
        ) for (pid, p) in getControls(data)
    )

    # perform normalization
    for protocol_id ∈ keys(samples)
        for exp ∈ axes(protocol_means, 3)
            samples[protocol_id][exp] .-= protocol_means[protocol_id, :, exp]
            controls[protocol_id][exp] .-= protocol_means[protocol_id, :, exp]
        end
    end

    return InteractionData(data.protein_IDs, data.protein_names, samples, controls)
end

"""
    load_data(files, sample_cols, control_cols, name_col=1, id_col=1, impute=false; kwargs...)

Load interaction data from multiple files (csv and xlsx).

# Arguments
- `files::Vector{String}`: File paths to load
- `sample_cols::Vector{Dict{I, Vector{I}}}`: Column indices for samples per file
- `control_cols::Vector{Dict{I, Vector{I}}}`: Column indices for controls per file
- `name_col::I=1`: Column index for protein names
- `id_col::I=1`: Column index for protein IDs
- `impute::Bool=false`: Whether to impute missing values

# Keyword Arguments
- `normalise_protocols::Bool=false`: Whether to normalise protocols
- `curate::Bool=true`: Enable protein curation (group splitting, synonym merging)
- `species::Int=9606`: NCBI taxonomy ID for STRING queries
- `curate_interactive::Bool=true`: Prompt user for merge confirmation
- `curate_merge_strategy::Symbol=:max`: Strategy for merging duplicate rows
- `bait_name::Union{Nothing,String}=nothing`: Bait protein name for refID tracking
- `curate_replay::Union{Nothing,String}=nothing`: Path to saved CurationReport for replay
- `curate_remove_contaminants::Bool=true`: Remove CON__/REV__ entries
- `curate_delimiter::String=";"`: Delimiter for protein group splitting
- `curate_auto_approve::Int=0`: Auto-approve merges with shared prefix length

# Returns
- `InteractionData` when `curate=false` or `bait_name` is nothing
- `(InteractionData, bait_index)` when `curate=true` and `bait_name` is set
"""
function load_data(
    files::Vector{String},
    sample_cols::Vector{Dict{I,Vector{I}}},
    control_cols::Vector{Dict{I,Vector{I}}},
    name_col::I=1, id_col::I=1, impute::Bool=false;
    normalise_protocols::Bool=false,
    curate::Bool=true,
    species::Int=9606,
    curate_interactive::Bool=true,
    curate_merge_strategy::Symbol=:max,
    bait_name::Union{Nothing, String}=nothing,
    curate_replay::Union{Nothing, String}=nothing,
    curate_remove_contaminants::Bool=true,
    curate_delimiter::String=";",
    curate_auto_approve::Int=0
) where I<:Integer

    # check that file type is supported
    any(file -> endswith(file, ".csv") == false && endswith(file, ".xlsx") == false, files) &&
        throw(ArgumentError("File type not supported"))

    # initialize variables
    samples, controls = Dict{Int,Protocol{Float64,I}}(), Dict{Int,Protocol{Float64,I}}()
    protein_IDs, protein_names = Vector{String}(), Vector{String}()
    bait_idx = nothing

    # load each file
    for (idx, file) ∈ enumerate(files)
        # Load raw DataFrame
        local raw_df::DataFrame
        if endswith(file, ".csv")
            check_file(file)
            raw_df = read(file, DataFrame)
        elseif endswith(file, ".xlsx")
            check_file(file)
            raw_df = DataFrame(readtable(file, "Sheet1"))
        end

        # ── Curation (before extract_data) ────────────────────────────────
        if curate
            # Set up cache directory next to the first data file
            cache_dir = joinpath(dirname(abspath(file)), ".bayesinteractomics_cache")

            # Load replay report if provided
            replay_report = nothing
            if !isnothing(curate_replay)
                replay_report = load_curation_report(curate_replay)
                if isnothing(replay_report)
                    @warn "Could not load curation replay report from '$curate_replay', running interactively"
                end
            end

            raw_df, report, cur_bait_idx = curate_proteins(
                raw_df, id_col;
                species = species,
                interactive = curate_interactive,
                cache_dir = cache_dir,
                replay_report = replay_report,
                merge_strategy = curate_merge_strategy,
                bait_name = bait_name,
                do_remove_contaminants = curate_remove_contaminants,
                delimiter = curate_delimiter,
                auto_approve_threshold = curate_auto_approve
            )

            # Save curation report next to the data file
            report_base = joinpath(cache_dir, "$(splitext(basename(file))[1])")
            save_curation_report(report, report_base)

            # Track bait index (use from first file that finds it)
            if isnothing(bait_idx) && !isnothing(cur_bait_idx)
                bait_idx = cur_bait_idx
            end
        end

        # Extract data from (possibly curated) DataFrame
        samples[idx], controls[idx], new_ids, new_names = extract_data(
            raw_df, sample_cols[idx], control_cols[idx], name_col, id_col, impute
        )

        # add protein IDs and names to vectors
        append_unique!(protein_IDs, new_ids)
        append_unique!(protein_names, new_names)
    end
    # create InteractionData object
    interaction_data = InteractionData(protein_IDs, protein_names, samples, controls)
    validate(interaction_data) == false && @warn "Protein names of protocols do not match! DO NOT PROCEED with the analysis"

    if normalise_protocols
        interaction_data = normalize(interaction_data)
    end

    # Return with bait index if curation was used with bait tracking
    if curate && !isnothing(bait_name)
        return interaction_data, bait_idx
    else
        return interaction_data
    end
end

############################################################################
# Permute data
############################################################################
"""
    _permute_pair(sample_mat::AbstractMatrix{Union{Missing,F}},
                   control_mat::AbstractMatrix{Union{Missing,F}}, refID::Integer,
                   rng) where {F} -> (new_sample, new_control)

Internal utility that shuffles the column labels of a single
experiment and returns the two permuted matrices as copies.
"""
function _permute_pair(sample_mat::AbstractMatrix{Union{Missing,F}},
    control_mat::AbstractMatrix{Union{Missing,F}}, refID::Integer,
    rng) where {F}

    nS, nC = size(sample_mat, 2), size(control_mat, 2)
    total_cols = nS + nC
    # Combine and shuffle
    combined = hcat(sample_mat, control_mat)
    perm_indices = randperm(rng, total_cols)
    new_sample_mat = combined[:, perm_indices[1:nS]]        # copy
    new_control_mat = combined[:, perm_indices[nS+1:end]]    # copy
    # overwrite the reference‐protein row so it remains identical
    @assert 1 ≤ refID ≤ size(combined, 1) "Reference protein ID out of bounds"
    new_sample_mat[refID, :] .= sample_mat[refID, :]
    new_control_mat[refID, :] .= control_mat[refID, :]
    return new_sample_mat, new_control_mat
end


"""
    permuteLabels(data::InteractionData{F,I}, refID::Integer = 1; rng::AbstractRNG = GLOBAL_RNG) -> InteractionData{F,I}

Return a *new* `InteractionData` object whose _sample_ / _control_ labels have been
randomly reassigned ( **within every experiment** ) while **preserving**  

* the number of samples and controls per experiment,  
* the protein order,  
* the hierarchical structure (protocol → experiment), and  
* the original value multiset (no value is lost or duplicated)
* the reference (i.e., bait) protein.

The permutation is performed **columnwise**:

1.  For every protocol *p* and experiment *e* the two matrices  
   `sample_mat  ∈  ℝ^{n_proteins × nS}` and  
   `control_mat ∈  ℝ^{n_proteins × nC}`  
   are concatenated horizontally.

2.  The columns of this combined matrix are shuffled with
   `randperm(rng, nS+nC)`.

3.  The **first** `nS` shuffled columns become the new sample matrix, the
   **remaining** `nC` columns the new control matrix.

The procedure yields a label-randomised data set that is ideal for
permutation or randomisation tests.

# Arguments
- `data` : Original `InteractionData`.
- `rng`  : (optional) random-number generator to make the permutation
           reproducible (`MersenneTwister` etc.).

# Returns
- A **fresh** `InteractionData` whose contents are the permuted version of
  `data`.  The original object is **never** mutated.

# Examples
```julia
using Random
rng = MersenneTwister(42)
permuted = permuteLabels(original_data; rng)
validate(permuted)               # validate permutation
```
"""
function permuteLabels(data::InteractionData{F,I}, refID::Integer=1; rng::AbstractRNG=GLOBAL_RNG) where {F<:AbstractFloat,I<:Integer}
    # ---------------------------------------------------------------------
    # Create permuted Protocol dictionaries
    # ---------------------------------------------------------------------
    new_samples = Dict{I,Protocol{F,I}}()
    new_controls = Dict{I,Protocol{F,I}}()

    for p in 1:getNoProtocols(data)
        old_s_proto = getSamples(data, p)
        old_c_proto = getControls(data, p)

        perm_s_data = Dict{I,Matrix{Union{Missing,F}}}()
        perm_c_data = Dict{I,Matrix{Union{Missing,F}}}()

        for e in 1:getNoExperiments(data, p)
            sample_mat = @views getExperiment(old_s_proto, e)
            control_mat = @views getExperiment(old_c_proto, e)

            perm_sample, perm_control = _permute_pair(sample_mat, control_mat, refID, rng)
            perm_s_data[e] = perm_sample
            perm_c_data[e] = perm_control
        end

        new_samples[p] = Protocol{F,I}(old_s_proto.no_experiments,
            old_s_proto.protein_ids,
            perm_s_data)
        new_controls[p] = Protocol{F,I}(old_c_proto.no_experiments,
            old_c_proto.protein_ids,
            perm_c_data)
    end

    # ---------------------------------------------------------------------
    # Re-assemble a fresh InteractionData
    # ---------------------------------------------------------------------
    new_data = InteractionData(
        data.protein_IDs, data.protein_names,
        new_samples, new_controls
    )

    validate(new_data) == false && @warn "Protein names of protocols do not match! DO NOT PROCEED with the analysis"
    return new_data

end

###############################################################################
# mergeInteractionData  –  stack proteins from data₂ below those of data₁
###############################################################################
"""
    vcat(data₁, data₂; suffix₂ = "_perm2") → InteractionData

Return a new `InteractionData` whose protein table is the vertical
concatenation of the two inputs.

* **No new experiments / protocols** are created – every protocol × experiment
  keeps its original number of sample and control columns.
* Protein IDs coming from `data₂` are suffixed with `suffix₂`
  (default `"_perm2"`) to guarantee uniqueness even when the two inputs derive
  from the *same* original proteins.

### Requirements
* `data₁` and `data₂` must have the **same number of protocols**.
* For every protocol *p* they must have the **same number of experiments** and
  the **same number of sample / control columns**.

If any of these structural properties disagree the function throws an
`ArgumentError`.

```julia
merged = mergeInteractionData(perm1, perm2; suffix₂ = "_h0copy")
```	
"""
function vcat(data₁::InteractionData{F,I}, data₂::InteractionData{F,I}; suffix₂::AbstractString="_perm2") where {F<:AbstractFloat,I<:Integer}
    # ─────────────── structural sanity checks ────────────────────────────
    nprot = getNoProtocols(data₁)
    nprot == getNoProtocols(data₂) ||
        throw(ArgumentError("Datasets have different numbers of protocols"))

    for p in 1:nprot
        getNoExperiments(data₁, p) == getNoExperiments(data₂, p) ||
            throw(ArgumentError("Protocol $p has different numbers of experiments"))
    end

    # ─────────────── new protein IDs / names ─────────────────────────────
    ids₁ = getIDs(data₁)
    ids₂ = getIDs(data₂)
    ids₂u = string.(ids₂, suffix₂)          # unique IDs for the second block
    names₁ = getNames(data₁)
    names₂ = getNames(data₂)
    names₂u = string.(names₂, suffix₂)

    new_ids = vcat(ids₁, ids₂u)
    new_names = vcat(names₁, names₂u)

    # ─────────────── concatenate matrices per protocol / experiment ─────
    new_samples = Dict{I,Protocol{F,I}}()
    new_controls = Dict{I,Protocol{F,I}}()

    for p in 1:nprot
        nexp = getNoExperiments(data₁, p)

        sample_dict = Dict{I,Matrix{Union{Missing,F}}}()
        control_dict = Dict{I,Matrix{Union{Missing,F}}}()

        for e in 1:nexp
            S₁ = copy(getSamples(data₁, p)[e])
            S₂ = copy(getSamples(data₂, p)[e])
            C₁ = copy(getControls(data₁, p)[e])
            C₂ = copy(getControls(data₂, p)[e])

            size(S₁, 2) == size(S₂, 2) ||
                throw(ArgumentError("Sample column count mismatch in protocol $p, experiment $e"))
            size(C₁, 2) == size(C₂, 2) ||
                throw(ArgumentError("Control column count mismatch in protocol $p, experiment $e"))

            sample_dict[e] = vcat(S₁, S₂)   # stack rows = new proteins
            control_dict[e] = vcat(C₁, C₂)
        end

        new_samples[p] = Protocol{F,I}(nexp, new_ids, sample_dict)
        new_controls[p] = Protocol{F,I}(nexp, new_ids, control_dict)
    end

    return InteractionData(new_ids, new_names, new_samples, new_controls)
end
