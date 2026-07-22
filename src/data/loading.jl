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
    compute_observation_counts(
        protein_IDs::Vector{String},
        samples::Dict{I,Protocol{F,I}},
        controls::Dict{I,Protocol{F,I}}
    ) where {F<:AbstractFloat, I<:Integer} -> (Vector{Int}, Vector{Int})

Count non-missing observations for each protein in samples and controls.

Returns two vectors (n_sample_obs, n_control_obs) where each element is the count
of non-missing values for that protein across all replicates in that group.

# Example
- Protein i with 3 replicates [1.0, missing, 2.0] → n_sample_obs[i] = 2
- Protein i with 3 replicates [missing, missing, missing] → n_control_obs[i] = 0
"""
function compute_observation_counts(
    protein_IDs::Vector{String},
    samples::Dict{I,Protocol{F,I}},
    controls::Dict{I,Protocol{F,I}}
) where {F<:AbstractFloat, I<:Integer}
    n = length(protein_IDs)
    n_sample_obs = zeros(Int, n)
    n_control_obs = zeros(Int, n)

    # Count non-missing values in samples
    for (_, protocol) in samples
        for exp_idx in 1:getNoExperiments(protocol)
            mat = getExperiment(protocol, exp_idx)
            for i in 1:n
                n_sample_obs[i] += count(!ismissing, @view mat[i, :])
            end
        end
    end

    # Count non-missing values in controls
    for (_, protocol) in controls
        for exp_idx in 1:getNoExperiments(protocol)
            mat = getExperiment(protocol, exp_idx)
            for i in 1:n
                n_control_obs[i] += count(!ismissing, @view mat[i, :])
            end
        end
    end

    return n_sample_obs, n_control_obs
end

"""
    filter_insufficient_observations(
        protein_IDs::Vector{String},
        n_sample_obs::Vector{Int},
        n_control_obs::Vector{Int},
        min_sample::Int,
        min_control::Int
    ) -> (Vector{Int}, DataFrame)

Filter proteins based on observation counts in samples and controls.

Returns:
- kept_indices: Vector of indices for proteins meeting both thresholds
- exclusion_report: DataFrame with columns :protein_id, :n_sample_obs, :n_control_obs
  for proteins failing the filter
"""
function filter_insufficient_observations(
    protein_IDs::Vector{String},
    n_sample_obs::Vector{Int},
    n_control_obs::Vector{Int},
    min_sample::Int,
    min_control::Int
)
    kept_indices = Int[]
    excluded_ids = String[]
    excluded_n_sample = Int[]
    excluded_n_control = Int[]

    for i in 1:length(protein_IDs)
        if n_sample_obs[i] >= min_sample && n_control_obs[i] >= min_control
            push!(kept_indices, i)
        else
            push!(excluded_ids, protein_IDs[i])
            push!(excluded_n_sample, n_sample_obs[i])
            push!(excluded_n_control, n_control_obs[i])
        end
    end

    exclusion_report = DataFrame(
        protein_id = excluded_ids,
        n_sample_obs = excluded_n_sample,
        n_control_obs = excluded_n_control
    )

    return kept_indices, exclusion_report
end

"""
    compute_detected_mask(protein_IDs, samples, controls) -> BitVector

Return a `BitVector` where element `i` is `true` if protein `i` is considered detected.

**Single protocol:** protein detected if it has ≥2 sample AND ≥2 control observations
(summed across experiments within that protocol).

**Multiple protocols:** protein detected if ANY single protocol has ≥2 sample AND ≥2 control
observations. This avoids false positives where a protein has 1 observation in each of
two protocols (sum=2 but never properly observed in either).
"""
function compute_detected_mask(
    protein_IDs::Vector{String},
    samples::Dict{I,Protocol{F,I}},
    controls::Dict{I,Protocol{F,I}}
) where {F <: AbstractFloat, I <: Integer}
    n = length(protein_IDs)
    num_protocols = length(samples)

    if num_protocols == 1
        # Single protocol: original behavior — sum across all experiments
        n_sample_obs, n_control_obs = compute_observation_counts(protein_IDs, samples, controls)
        detected = (n_sample_obs .>= 2) .& (n_control_obs .>= 2)
        return detected
    end

    # Multiple protocols: protein detected if ANY protocol qualifies independently
    detected = falses(n)
    for prot_key in keys(samples)
        # Count observations within this single protocol
        prot_sample_obs = zeros(Int, n)
        prot_control_obs = zeros(Int, n)

        protocol_s = samples[prot_key]
        for exp_idx in 1:getNoExperiments(protocol_s)
            mat = getExperiment(protocol_s, exp_idx)
            for i in 1:n
                prot_sample_obs[i] += count(!ismissing, @view mat[i, :])
            end
        end

        protocol_c = controls[prot_key]
        for exp_idx in 1:getNoExperiments(protocol_c)
            mat = getExperiment(protocol_c, exp_idx)
            for i in 1:n
                prot_control_obs[i] += count(!ismissing, @view mat[i, :])
            end
        end

        # OR-reduce: if this protocol qualifies, mark as detected
        detected .|= (prot_sample_obs .>= 2) .& (prot_control_obs .>= 2)
    end

    return detected
end

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

    # compute detection mask: protein detected if it has ≥1 in both samples AND controls
    detected = compute_detected_mask(protein_IDs, samples, controls)

    data = InteractionData(
        protein_IDs, protein_names,
        samples, controls,
        num_protocols, num_experiments_per_protocol,
        num_params_HBM, num_params_regression,
        protocol_positions, experiment_positions, matched_positions,
        detected
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

        (ismissing(row_var_sample) || isnan(row_var_sample)) && (row_var_sample = 3.0)
        (ismissing(row_var_control) || isnan(row_var_control)) && (row_var_control = 3.0)

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
    data = CSV.read(file, DataFrame)
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

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation pipeline
#
# `median_of_ratios` (DESeq size-factor) column-scaling normaliser operating
# directly on an InteractionData, plus the (proteins × MS-runs) matrix round-trip
# helpers it needs. Output stays on
# the log2 scale and preserves missings. `apply_normalisation` is the dispatcher
# the rest of the pipeline calls with an already-RESOLVED concrete method.
# ─────────────────────────────────────────────────────────────────────────────

const _NORM_MF = Union{Missing,Float64}

# missing-aware observed-cell extractor
_norm_obs(v) = Float64[Float64(x) for x in v if !ismissing(x) && isfinite(x)]

# linear ↔ log2 round-trip.
# `<= 0 -> missing` guard prevents NaN/Inf leaking into HBM/regression (T-77.1-02).
_norm_to_linear(X) = _NORM_MF[ismissing(x) ? missing : 2.0^Float64(x) for x in X]
_norm_to_log2(Y)   = _NORM_MF[ismissing(y) ? missing : (Float64(y) <= 0 ? missing : log2(Float64(y))) for y in Y]

"""
    build_run_matrix(data::InteractionData) -> (X, meta, ids)

Flatten an `InteractionData` into a `(n_proteins × n_runs)` `Matrix{Union{Missing,Float64}}`
on the log2 scale. Each column is one MS run.

- `X`: `(n_proteins × n_runs)` log2 matrix, missings preserved.
- `meta`: `Vector{NamedTuple}` of `(protocol, exp, group∈{:sample,:control}, rep)`.
- `ids`: protein ID vector (row order = `getIDs(data)`).

Deterministic column order: `(:sample, getSamples), (:control, getControls)`,
then `sort(keys)` for both protocol and experiment, one column per replicate.
"""
function build_run_matrix(data::InteractionData)
    ids = collect(getIDs(data))
    np = length(ids)
    cols = Vector{Vector{_NORM_MF}}()
    meta = NamedTuple[]
    for (grp, dict) in ((:sample, getSamples(data)), (:control, getControls(data)))
        for pid in sort(collect(keys(dict)))
            proto = dict[pid]
            for exp in sort(collect(keys(proto.data)))
                M = proto.data[exp]               # np × nrep
                for r in 1:size(M, 2)
                    push!(cols, Vector{_NORM_MF}(M[:, r]))
                    push!(meta, (protocol=pid, exp=exp, group=grp, rep=r))
                end
            end
        end
    end
    X = Matrix{_NORM_MF}(undef, np, length(cols))
    for (j, c) in enumerate(cols)
        X[:, j] .= c
    end
    return X, meta, ids
end

"""
    matrix_to_interactiondata(template::InteractionData, X, meta) -> InteractionData

Write a `(proteins × runs)` matrix `X` (with column metadata `meta` from
`build_run_matrix`) back into a fresh `InteractionData` mirroring `template`
exactly (same protocol keytype/valtype, same `no_experiments`, same `protein_ids`).
"""
function matrix_to_interactiondata(template::InteractionData, X, meta)
    function build(grp_sym, tdict)
        out = Dict{keytype(tdict), valtype(tdict)}()   # exact Dict{I, Protocol{F,I}}
        for (pid, proto) in tdict
            newdata = Dict{Int, Matrix{_NORM_MF}}()
            for exp in sort(collect(keys(proto.data)))
                M = proto.data[exp]; nrep = size(M, 2)
                cols = [j for (j, m) in enumerate(meta) if m.group == grp_sym && m.protocol == pid && m.exp == exp]
                @assert length(cols) == nrep "col/rep mismatch grp=$grp_sym pid=$pid exp=$exp ($(length(cols)) vs $nrep)"
                newM = Matrix{_NORM_MF}(undef, size(M, 1), nrep)
                for (r, j) in enumerate(cols); newM[:, r] .= X[:, j]; end
                newdata[exp] = newM
            end
            out[pid] = Protocol(proto.no_experiments, proto.protein_ids, newdata)
        end
        return out
    end
    s = build(:sample, getSamples(template)); c = build(:control, getControls(template))
    return InteractionData(template.protein_IDs, template.protein_names, s, c)
end

"""
    norm_median_of_ratios_id(data::InteractionData) -> InteractionData

DESeq size-factor (`median_of_ratios`) column-scaling normaliser. Round-trips
`build_run_matrix` → linear-scale median-of-ratios → `matrix_to_interactiondata`.

- linear `y = 2^x`; per-protein geometric mean over proteins OBSERVED IN ALL
  columns (`length(o) == nc && all .> 0`); per-column size factor
  `s_j = median_i(y_ij / geomean_i)`; divide column `j` by `s_j`; back to log2
  (`<= 0 -> missing`). Missings preserved.
"""
function norm_median_of_ratios_id(data::InteractionData)
    X, meta, _ = build_run_matrix(data)
    Y = _norm_to_linear(X); nr, nc = size(X)
    geomean = fill(NaN, nr)
    for i in 1:nr
        o = _norm_obs(_NORM_MF[Y[i, j] for j in 1:nc])
        length(o) == nc && all(o .> 0) && (geomean[i] = exp(mean(log.(o))))
    end
    for j in 1:nc
        ratios = Float64[]
        for i in 1:nr
            (isnan(geomean[i]) || ismissing(Y[i, j])) && continue
            push!(ratios, Float64(Y[i, j]) / geomean[i])
        end
        isempty(ratios) && continue
        s = median(ratios)
        for i in 1:nr
            ismissing(Y[i, j]) || (Y[i, j] = Float64(Y[i, j]) / s)
        end
    end
    Xn = _norm_to_log2(Y)   # typed comprehension preserves the (nr × nc) shape
    return matrix_to_interactiondata(data, Xn, meta)
end

"""
    bait_anchor_id(data::InteractionData; bait_row::Int=1) -> InteractionData

Regression-safe per-condition bait-anchor correction.

Equalises the bait's MEAN sample level across conditions (= protocols) by a single
per-condition scalar `δ_c` subtracted from the **SAMPLE cells only** of that protocol.
This shifts each prey's enrichment (sample − control) by the bait-level gap — the
differential-interactomics correction — while leaving CONTROLS (background, bait-free)
untouched. Subtracting `δ_c` from BOTH sample and control would cancel in the
sample−control contrast and leave the differential unchanged, hence sample-only.

`δ_c = mean(bait SAMPLE level in protocol c) − grand mean of the per-protocol bait
sample means`, computed from the bait row (`bait_row`, default 1; the `refID`/bait
position in the protein order) of the data AS PASSED.

REGRESSION-SAFE: each protocol's bait sample cells shift by a per-protocol CONSTANT,
so within-condition run-to-run bait variation (the regression dose axis) is preserved
and the predictor is never zeroed / de-varied.

No-op when fewer than 2 protocols (single-condition data has nothing to anchor).
Missings preserved.

CALLER CONTRACT: `δ_c` is derived from the bait abundance of the data as passed, so
callers must apply `bait_anchor_id` on the data on which RAW bait abundance is
meaningful (i.e. relative to raw bait levels — NOT post-(re)scaling values that would
reposition the high-abundance bait differently per condition). On matched-level
multi-protocol loads (e.g. real HAP40 GST/Strep, ~30 vs ~30.1 log2) `δ_c ≈ 0` and the
anchor is correctly near-inert.

Implemented via the `build_run_matrix` / `matrix_to_interactiondata` round-trip.
"""
function bait_anchor_id(data::InteractionData; bait_row::Int=1)
    getNoProtocols(data) < 2 && return data   # no-op: single condition
    X, meta, _ = build_run_matrix(data)
    protocols = sort(unique(m.protocol for m in meta))
    length(protocols) < 2 && return data
    # per-protocol mean of the bait row's SAMPLE cells (missing-aware)
    bmean = Dict{eltype(protocols), Float64}()
    for p in protocols
        cols = [j for (j, m) in enumerate(meta) if m.protocol == p && m.group == :sample]
        obs = _norm_obs(_NORM_MF[X[bait_row, j] for j in cols])
        bmean[p] = isempty(obs) ? NaN : mean(obs)
    end
    finite = Float64[v for v in values(bmean) if !isnan(v)]
    isempty(finite) && return data            # no observed bait sample level anywhere
    ref = mean(finite)
    Xa = copy(X)
    for p in protocols
        isnan(bmean[p]) && continue
        δ = bmean[p] - ref
        cols = [j for (j, m) in enumerate(meta) if m.protocol == p && m.group == :sample]  # SAMPLE only
        for i in 1:size(Xa, 1), j in cols
            ismissing(Xa[i, j]) || (Xa[i, j] = Float64(Xa[i, j]) - δ)
        end
    end
    return matrix_to_interactiondata(data, Xa, meta)
end

"""
    apply_normalisation(data::InteractionData, method::Symbol) -> InteractionData

Dispatch an already-RESOLVED concrete normalisation `method` onto `data`:

- `:none`             → `data` unchanged (identity; NOT a deepcopy that could reorder).
- `:row_center`       → `normalize(data)` (existing per-protein per-(protocol,exp) row-centering, unchanged).
- `:median_of_ratios` → `norm_median_of_ratios_id(data)` (DESeq size factors).
- `:both`             → `normalize(norm_median_of_ratios_id(data))` (column-scale FIRST then row-center).
- `:auto`             → `ArgumentError` (`:auto` resolution happens upstream; this dispatcher
                         receives a concrete method).

Unknown methods throw `ArgumentError` (T-77.1-01 allowlist).
"""
function apply_normalisation(data::InteractionData, method::Symbol)
    if method === :none
        return data
    elseif method === :row_center
        return normalize(data)
    elseif method === :median_of_ratios
        return norm_median_of_ratios_id(data)
    elseif method === :both
        return normalize(norm_median_of_ratios_id(data))
    elseif method === :auto
        throw(ArgumentError("apply_normalisation received :auto — it expects an already-RESOLVED concrete method (:none, :row_center, :median_of_ratios, :both). :auto resolution is wired upstream."))
    else
        throw(ArgumentError("Unknown normalisation method :$method. Allowed: :none, :row_center, :median_of_ratios, :both (or :auto, resolved upstream)."))
    end
end

const _NORMALISATION_METHODS = (:none, :row_center, :median_of_ratios, :both, :auto)

"""
    _resolve_normalisation_method(normalisation_method::Symbol, normalise_protocols::Bool) -> Symbol

Resolve the effective normalisation method from the new `normalisation_method` selector and the
legacy `normalise_protocols::Bool` (back-compat mapping).

Precedence rule: if `normalisation_method !== :none` the new selector WINS (it is authoritative);
otherwise the legacy bool is mapped (`true -> :row_center`, `false -> :none`). Validates the symbol
against the allowlist `(:none, :row_center, :median_of_ratios, :both, :auto)` (T-77.1-01) and throws
`ArgumentError` on an unknown value.
"""
function _resolve_normalisation_method(normalisation_method::Symbol, normalise_protocols::Bool)
    normalisation_method ∈ _NORMALISATION_METHODS ||
        throw(ArgumentError("normalisation_method must be one of $(_NORMALISATION_METHODS), got :$normalisation_method"))
    if normalisation_method !== :none
        return normalisation_method            # new selector is authoritative
    end
    return normalise_protocols ? :row_center : :none   # legacy bool mapping
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
- `normalise_protocols::Bool=false`: Whether to normalise protocols (legacy back-compat alias for
   `normalisation_method`; true->:row_center, false->:none — superseded when `normalisation_method`
   is set to anything other than `:none`).
- `normalisation_method::Symbol=:none`: Normalisation selector. One of
   `:none | :row_center | :median_of_ratios | :both | :auto`. Default `:none` preserves the exact
   behaviour of callers that pass only `normalise_protocols`. Resolved via
   `_resolve_normalisation_method`.
- `curate::Bool=true`: Enable protein curation (group splitting, synonym merging)
- `species::Int=9606`: NCBI taxonomy ID for STRING queries
- `curate_interactive::Bool=true`: Prompt user for merge confirmation
- `curate_merge_strategy::Symbol=:max`: Strategy for merging duplicate rows
- `bait_name::Union{Nothing,String}=nothing`: Bait protein name for refID tracking
- `curate_replay::Union{Nothing,String}=nothing`: Path to saved CurationReport for replay
- `curate_remove_contaminants::Bool=true`: Remove CON__/REV__ entries
- `curate_delimiter::String=";"`: Delimiter for protein group splitting
- `curate_auto_approve::Int=0`: Auto-approve merges with shared prefix length
- `imputation::Symbol=:mnar`: Imputation-method tag forwarded to cache hashing.
   Allowed values: `:mnar` (default; MNAR-aware imputation), `:mar`
   (deprecated; legacy MICE/MAR — emits @warn, removal in v1.3), `:none` (no
   pre-imputation; downstream HBM/regression treat missings as latent variables).
   Metadata-only — does NOT change file resolution or data transformation.
   Distinct from the legacy positional `impute::Bool` (which triggers
   in-`load_data` imputation, separate concern).

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
    # Normalisation selector. Default :none so callers passing only the
    # legacy normalise_protocols bool keep their EXACT behaviour (see _resolve_normalisation_method).
    normalisation_method::Symbol=:none,
    # refID threaded into the :auto scale-mismatch detector
    # (detect_protocol_scale_mismatch). Default 1 matches the package bait convention.
    refID::Int=1,
    curate::Bool=true,
    species::Int=9606,
    curate_interactive::Bool=true,
    curate_merge_strategy::Symbol=:max,
    bait_name::Union{Nothing, String}=nothing,
    curate_replay::Union{Nothing, String}=nothing,
    curate_remove_contaminants::Bool=true,
    curate_delimiter::String=";",
    curate_auto_approve::Int=0,
    imputation::Symbol=:mnar,
    # When false, SKIP the single-protocol n_obs<2 exclusion.
    # Imputed-analysis workflows MUST load BOTH the raw (imputation=:none) and the imputed
    # (:mnar) data with `filter_insufficient_obs=false`, otherwise raw drops sparse proteins
    # the filled imputed file keeps → the two InteractionData objects become index-misaligned
    # and the multi-imputation `analyse(imputed, raw, …)` (which indexes both by the same i)
    # silently pairs mismatched proteins + drops ~the difference. Default true preserves
    # raw-only (non-imputed) analysis behaviour byte-for-byte.
    filter_insufficient_obs::Bool=true
) where I<:Integer

    # Imputation-method validation + :mar deprecation
    if imputation === :mar
        @warn "imputation = :mar (MICE-imputed input) is deprecated as of v1.2.0 and will be removed in v1.3. Migrate to imputation = :mnar (default). MICE files have been renamed to dataset_imp_mice_*.xlsx; the old dataset_imp_*.xlsx names continue to work in v1.2.x." maxlog=1
    elseif imputation ∉ (:mnar, :mar, :none)
        throw(ArgumentError("imputation must be :mnar, :mar, or :none, got :$imputation"))
    end

    # The `imputation` tag marks the input
    # FILE as already-imputed (:mnar / :mar). Normalisation runs at the END of load_data — i.e.
    # AFTER the imputed file has been read — so requesting an active normalisation here applies it
    # in the WRONG order (normalise-AFTER-impute). load_data cannot guarantee the correct order from
    # the inside because the file is already imputed. Warn the user (maxlog=1) and point them at the
    # correct-order entry point `normalise_then_impute(raw_data, dropout_fit; ...)`. Only fires when
    # an active normalisation is requested (method ∉ :none) — `imputation=:none` raw loads (and
    # imputed loads with no normalisation requested) stay byte-identical to today.
    if imputation ∈ (:mar, :mnar) && normalisation_method !== :none
        @warn "load_data: input file is tagged as already-imputed (imputation=:$imputation) AND a " *
              "normalisation (normalisation_method=:$normalisation_method) was requested. Normalisation " *
              "runs AFTER the imputed file is read, so the normalise-BEFORE-impute order " *
              "cannot be guaranteed from inside load_data. For the correct order, normalise the RAW " *
              "data first and impute in-process via `normalise_then_impute(raw_data, dropout_fit; " *
              "normalisation_method, refID)`." maxlog=1
    end

    # Variante B loud-fail preflight: when the user
    # explicitly requests :mar or :mnar but the BayesInteractomicsImputationExt
    # extension is NOT loaded (i.e. they did not run `using GLM`), surface an
    # ArgumentError BEFORE any file I/O begins. _require_imputation_extension
    # is a no-op for :none and emits a clear `using GLM` instruction for
    # :mar/:mnar. Distinct from the metalearner Variante B SILENT fallback —
    # imputation is an explicit user opt-in, so the failure must be loud.
    if (imputation === :mar || imputation === :mnar) && !_imputation_extension_loaded()
        _require_imputation_extension(imputation)
    end

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
            raw_df = CSV.read(file, DataFrame)
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

        # ── Observation-based filtering ─────────────────────────────────────
        # Only filter per-file for single-protocol setups. Multi-protocol
        # datasets must keep all proteins aligned across protocols; the
        # downstream compute_detected_mask handles per-protocol detection.
        # Gated on `filter_insufficient_obs` so imputed-analysis
        # workflows can keep raw + imputed index-aligned (see kwarg docstring).
        if length(files) == 1 && filter_insufficient_obs
            # Count observations per protein in samples and controls
            n_sample_obs, n_control_obs = compute_observation_counts(
                new_ids, Dict(idx => samples[idx]), Dict(idx => controls[idx])
            )

            # Filter proteins with insufficient observations (min 2+2)
            kept_indices, exclusion_report = filter_insufficient_observations(
                new_ids, n_sample_obs, n_control_obs, 2, 2
            )

            # Save exclusion report to CSV
            if !isempty(exclusion_report)
                # Use .bayesinteractomics_cache directory
                if curate
                    cache_dir = joinpath(dirname(abspath(file)), ".bayesinteractomics_cache")
                else
                    # If no curation, create cache dir
                    cache_dir = joinpath(dirname(abspath(file)), ".bayesinteractomics_cache")
                    mkpath(cache_dir)
                end

                report_path = joinpath(cache_dir, "$(splitext(basename(file))[1])_excluded_proteins.csv")
                CSV.write(report_path, exclusion_report)

                # Print summary
                n_excluded = nrow(exclusion_report)
                total = length(new_ids)
                pct = round(100.0 * n_excluded / total, digits=1)
                println("Excluded $n_excluded out of $total proteins ($(pct)%) with insufficient observations (n_sample_obs < 2 or n_control_obs < 2)")

                # Warn if exclusion > 5%
                if pct > 5.0
                    @warn "High exclusion rate ($(pct)%) — check data quality"
                end
            end

            # Filter protocols and protein lists to keep only sufficient proteins
            new_ids_filtered = new_ids[kept_indices]
            new_names_filtered = new_names[kept_indices]

            # Filter samples and controls for this file
            old_s = samples[idx]
            old_c = controls[idx]

            # Keep only rows corresponding to kept_indices
            new_s_data = Dict(
                exp => old_s.data[exp][kept_indices, :]
                for exp in keys(old_s.data)
            )
            new_c_data = Dict(
                exp => old_c.data[exp][kept_indices, :]
                for exp in keys(old_c.data)
            )

            samples[idx] = Protocol(
                old_s.no_experiments, new_ids_filtered, new_s_data
            )
            controls[idx] = Protocol(
                old_c.no_experiments, new_ids_filtered, new_c_data
            )

            # Update working copies for next file iteration
            new_ids = new_ids_filtered
            new_names = new_names_filtered
        end

        # add protein IDs and names to vectors
        append_unique!(protein_IDs, new_ids)
        append_unique!(protein_names, new_names)
    end
    # create InteractionData object
    interaction_data = InteractionData(protein_IDs, protein_names, samples, controls)
    validate(interaction_data) == false && @warn "Protein names of protocols do not match! DO NOT PROCEED with the analysis"

    # Resolve the selector (new method wins over the legacy bool),
    # then resolve :auto via the multi-protocol scale-mismatch detector, and apply.
    resolved_method = _resolve_normalisation_method(normalisation_method, normalise_protocols)
    if resolved_method === :auto
        # The BREAKING multi-protocol auto-flip: reuse the pooled-residual
        # guard (detect_protocol_scale_mismatch) to decide. On a scale-disparate multi-protocol
        # load → :both (median_of_ratios + row-centering); otherwise → :none.
        # Single-protocol + matched-level multi-protocol both stay :none (no over-correction).
        if detect_protocol_scale_mismatch(interaction_data; refID=refID)
            resolved_method = :both
            @info "normalisation_method=:auto auto-applied :both (median_of_ratios + row-centering) — " *
                  "multi-protocol scale mismatch detected (pooled residual SD > 2.5 log2). " *
                  "This is a deliberate breaking change for scale-disparate " *
                  "multi-protocol loads (set normalisation_method=:none to opt out)."
        else
            resolved_method = :none
        end
    end
    interaction_data = apply_normalisation(interaction_data, resolved_method)

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
