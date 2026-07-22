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

function checkForDuplicates(v)
    if length(v) == length(Set(v))
        return nothing
    end

    # check which are the duplicates
    counts = Dict{String,Int64}()
    [counts[x] = get(counts, x, 0) + 1 for x in v]

    # retrieve duplicates
    duplicates = [k for (k, v) in counts if v > 1]
    @error "Duplicates in column names: $duplicates"
end

"""
    clean_result(data, file::S, resultfile::S) where {S<:AbstractString}

    Clean the results and write the major statistical measures to the new file resultfile

    Args:
      - data<:InteractionData: InteractionData object
      - file<:AbstractString: File path to the results stored as a csv file created by main()
      - resultfile<:AbstractString: File path to the cleaned results (xlsx)
"""
function clean_result(data::InteractionData, file::S, resultfile::S) where {S<:AbstractString}
    if getNoProtocols(data) == 1
        return clean_result_single_protocol(data, file, resultfile)
    end


    protocol_positions = getProtocolPositions(data)
    slope_positions = collect(1:size(protocol_positions, 1))
    df = DataFrame(CSV.File(file, header=1, delim="|"))

    main_summary_names = ["Protein", "bf_log2FC_1_>0.0", "mean_log2FC_1", "sd_log2FC_1", "pd_log2FC_1"]
    append!(main_summary_names, ["bf_log2FC_$(i)_>0.0" for i ∈ protocol_positions])
    append!(main_summary_names, ["mean_log2FC_$(i)" for i ∈ protocol_positions])
    append!(main_summary_names, ["sd_log2FC_$(i)" for i ∈ protocol_positions])
    append!(main_summary_names, ["pd_log2FC_$(i)" for i ∈ protocol_positions])

    push!(main_summary_names, "bf_slope")
    append!(main_summary_names, ["bf_slope_$(i)" for i ∈ slope_positions])

    push!(main_summary_names, "mean_slope")
    append!(main_summary_names, ["mean_slope_$(i)" for i ∈ slope_positions])

    push!(main_summary_names, "sd_slope")
    append!(main_summary_names, ["sd_slope_$(i)" for i ∈ slope_positions])

    push!(main_summary_names, "pd_slope")
    append!(main_summary_names, ["pd_slope_$(i)" for i ∈ slope_positions])

    checkForDuplicates(main_summary_names)

    df_summary = df[:, Symbol.(main_summary_names)]

    # rename columns
    new_names = ["Protein", "BF_log2FC", "mean_log2FC", "sd_log2FC", "PD_log2FC"]
    append!(new_names, names(df_summary)[length(new_names)+1:end])
    rename!(df_summary, new_names)

    # write to file
    writetable(resultfile, "SUMMARY" => df_summary, "Statistics" => df)
    return df_summary
end

function clean_result_single_protocol(data::InteractionData, file::S, resultfile::S) where {S<:AbstractString}
    df = DataFrame(CSV.File(file, header=1, delim="|"))

    main_summary_names = [
        "Protein", "bf_log2FC_1_>0.0", "mean_log2FC_1", "sd_log2FC_1", "pd_log2FC_1",
        "bf_slope", "mean_slope", "sd_slope", "pd_slope"
    ]


    df_summary = df[:, Symbol.(main_summary_names)]

    # rename columns
    new_names = [
        "Protein", "BF_log2FC", "mean_log2FC", "sd_log2FC", "PD_log2FC",
        "bf_slope", "mean_slope", "sd_slope", "pd_slope"
    ]

    rename!(df_summary, new_names)

    # write to file
    writetable(resultfile, "SUMMARY" => df_summary, "Statistics" => df)
    return df_summary
end

@model function HierarchicalBayesianModel(samples, controls, μ, σ, a, b, nparameters, n_protocols, experiment_positions, protocol_positions, matched_positions, n_experiments, parameter_lookup)
    #######################################
    # prior definitions                 ###
    #######################################
    # define priors
    local μ_control #   # Individual means of the control group
    local μ_sample #    # Individual means of the sample group
    local σ_control #   # Individual variance of the control group
    local σ_sample #    # Individual variance deviations of the sample group

    # 1st level priors
    σ_control[1] ~ Gamma(shape=a, scale=b)
    σ_sample[1] ~ Gamma(shape=a, scale=b)
    μ_control[1] ~ Normal(mean=μ, precision=1.0 / σ)
    μ_sample[1] ~ Normal(mean=μ, precision=1.0 / σ)

    # 2nd and 3rd level priors
    experiment_idx = 1

    for idx ∈ 2:nparameters
        σ_control[idx] ~ Gamma(shape=a, scale=b)
        σ_sample[idx] ~ Gamma(shape=a, scale=b)

        # 2nd level priors
        if idx ∈ protocol_positions
            # hyperprior for all experiments of protocol "protocol" and individual experiments
            μ_control[idx] ~ Normal(mean=μ_control[1], precision=σ_control[1])
            μ_sample[idx] ~ Normal(mean=μ_sample[1], precision=σ_sample[1])
        elseif idx ∈ experiment_positions
            # retrieve protocol position
            protocol_position = matched_positions[experiment_idx]
            experiment_idx += 1
            # define low level priors
            μ_control[idx] ~ Normal(mean=μ_control[protocol_position], precision=σ_control[protocol_position])
            μ_sample[idx] ~ Normal(mean=μ_sample[protocol_position], precision=σ_sample[protocol_position])
        else
            throw(BoundsError("Position $idx is not defined"))
        end
    end

    #######################################
    # likelihood definitions
    #######################################
    # Optimized: lookup parameter position outside innermost loop
    for protocol ∈ 1:n_protocols
        for experiment ∈ 1:n_experiments[protocol]
            pos_parameter_vector = parameter_lookup[protocol, experiment]
            pos_parameter_vector == 0 && throw(BoundsError("Position $pos_parameter_vector is not defined"))
            # Use position directly in sample loop
            for idx in 1:size(samples, 3)
                samples[protocol, experiment, idx] ~ Normal(mean=μ_sample[pos_parameter_vector], precision=σ_sample[pos_parameter_vector])
                controls[protocol, experiment, idx] ~ Normal(mean=μ_control[pos_parameter_vector], precision=σ_control[pos_parameter_vector])
            end
        end
    end
end

@model function HierarchicalBayesianModelSingle(samples, controls, μ, σ, a, b)

    #######################################
    # prior definitions                 ###
    #######################################
    # define priors
    local μ_control #   # Individual means of the control group
    local μ_sample #    # Individual means of the sample group
    local σ_control #   # Individual variance of the control group
    local σ_sample #    # Individual variance deviations of the sample group

    σ_control[1] ~ Gamma(shape=a, scale=b)
    σ_sample[1] ~ Gamma(shape=a, scale=b)
    μ_control[1] ~ Normal(mean=μ, precision=1.0 / σ)
    μ_sample[1] ~ Normal(mean=μ, precision=1.0 / σ)

    # One experiment-level parameter per experiment (+ the global level-1 node).
    # The likelihood below indexes μ_/σ_*[experiment+1] over experiments
    # (size(samples,1)), so nparameters MUST track the EXPERIMENT count — not the
    # replicate count (size(samples,2)). Using size(samples,2) only worked when
    # experiments == replicates (e.g. HAP40_Strep 3×3); otherwise it left σ nodes
    # unused (replicates>experiments → RxInfer half-edge) or under-allocated them
    # (experiments>replicates → BoundsError on σ_sample[experiment+1]).
    nparameters = size(samples, 1) + 1

    for idx ∈ 2:nparameters
        σ_control[idx] ~ Gamma(shape=a, scale=b)
        σ_sample[idx] ~ Gamma(shape=a, scale=b)

        if idx != 1
            # define experiment level priors
            μ_control[idx] ~ Normal(mean=μ_control[1], precision=σ_control[1])
            μ_sample[idx] ~ Normal(mean=μ_sample[1], precision=σ_sample[1])
        else
            throw(BoundsError("Position $idx is not defined"))
        end
    end

    #######################################
    # likelihood definitions
    #######################################
    for experiment ∈ 1:size(samples, 1), idx in 1:size(samples, 2)
        samples[experiment, idx] ~ Normal(mean=μ_sample[experiment+1], precision=σ_sample[experiment+1])
        controls[experiment, idx] ~ Normal(mean=μ_control[experiment+1], precision=σ_control[experiment+1])
    end
end



"""
    getParameterLookup(n_protocols, n_experiment)

    Generate parameter lookup table that is used by the HBM model to determine the parameter position in the parameter vector

    Args:   
        n_protocols::Int64:                 the number of protocols
        n_experiment::Dict{Int64, Int64}:   the number of experiments for each protocol

    Returns:
        parameter_lookup::Matrix{Int64}: the parameter lookup table with n_protcols rows and max(n_experiment) columns
"""
function getParameterLookup(n_protocols, n_experiment)
    max_experiments = maximum(values(n_experiment))
    parameter_lookup::Array{Int64,2} = zeros(Int64, n_protocols, max_experiments)

    # initialize counter and values
    value = 3
    protocol::Int64 = 1
    experiment::Int64 = 1

    @inbounds while protocol <= n_protocols
        while experiment <= n_experiment[protocol]
            parameter_lookup[protocol, experiment] = value
            value += 1
            experiment += 1
        end
        value += 1
        protocol += 1
        experiment = 1
    end
    return parameter_lookup
end

# ------------------------------------ Prior Caching -------------------------------#

"""
    precompute_HBM_prior(data::InteractionData; μ_0::F=25.0, σ_0::F=1.0, a_0::F=1.0, b_0::F=1.0) where {F<:AbstractFloat}

Precomputes the HBM prior distribution once for all proteins.
Priors only depend on hyperparameters (μ_0, σ_0, a_0, b_0), not on individual protein data.

# Returns
- `InferenceResult`: The cached prior distribution that can be reused across all proteins.
"""
function precompute_HBM_prior(data::InteractionData; μ_0::F=25.0, σ_0::F=1.0, a_0::F=1.0, b_0::F=1.0) where {F<:AbstractFloat}
    # Get structural parameters from data (these are the same for all proteins)
    protein = getProteinData(data, 1)  # Use first protein to get structure
    interactome_sample = getSampleMatrix(protein)

    n_protocols::Int64 = size(interactome_sample, 1)
    n_experiments::Dict{Int64,Int64} = getNoExperiments(data)
    protocol_positions = getProtocolPositions(data)
    experiment_positions = getExperimentPositions(data)
    matched_positions = getMatchedPositions(data)
    parameter_lookup = getParameterLookup(n_protocols, n_experiments)
    nparameters = data.no_parameters_HBM

    # Constraints and initialization (same as in HBM)
    constraints = @constraints begin
        q(μ_control, σ_control) = q(μ_control)q(σ_control)
        q(μ_sample, σ_sample) = q(μ_sample)q(σ_sample)
    end

    init = @initialization begin
        q(μ_control) = vague(NormalMeanPrecision)
        q(σ_control) = vague(GammaShapeRate)
        q(μ_sample) = vague(NormalMeanPrecision)
        q(σ_sample) = vague(GammaShapeRate)
    end

    # Compute prior with all missing data
    missing_complete = fill(missing, size(interactome_sample))

    prior::InferenceResult = infer(
        model=HierarchicalBayesianModel(
            μ=μ_0, σ=σ_0, a=a_0, b=b_0,
            nparameters=nparameters,
            n_protocols=n_protocols,
            experiment_positions=experiment_positions,
            protocol_positions=protocol_positions,
            matched_positions=matched_positions,
            n_experiments=n_experiments,
            parameter_lookup=parameter_lookup
        ),
        data=(samples=missing_complete, controls=missing_complete),
        initialization=init,
        constraints=constraints,
        iterations=1000,
        returnvars=KeepLast()
    )

    return prior
end

"""
    precompute_HBM_single_protocol_prior(data::InteractionData; μ_0::F=25.0, σ_0::F=1.0, a_0::F=1.0, b_0::F=1.0) where {F<:AbstractFloat}

Precomputes the single-protocol HBM prior distribution once for all proteins.

# Returns
- `InferenceResult`: The cached prior distribution.
"""
function precompute_HBM_single_protocol_prior(data::InteractionData; μ_0::F=25.0, σ_0::F=1.0, a_0::F=1.0, b_0::F=1.0) where {F<:AbstractFloat}
    protein = getProteinData(data, 1)
    sample_data = getSampleMatrix(protein)[1, :, :]

    constraints = @constraints begin
        q(μ_control, σ_control) = q(μ_control)q(σ_control)
        q(μ_sample, σ_sample) = q(μ_sample)q(σ_sample)
    end

    init = @initialization begin
        q(μ_control) = vague(NormalMeanPrecision)
        q(σ_control) = vague(GammaShapeRate)
        q(μ_sample) = vague(NormalMeanPrecision)
        q(σ_sample) = vague(GammaShapeRate)
    end

    missing_complete = fill(missing, size(sample_data))

    prior::InferenceResult = infer(
        model=HierarchicalBayesianModelSingle(μ=μ_0, σ=σ_0, a=a_0, b=b_0),
        data=(samples=missing_complete, controls=missing_complete),
        initialization=init,
        constraints=constraints,
        iterations=1000,
        returnvars=KeepLast()
    )

    return prior
end

"""
    precompute_regression_multi_protocol_prior(data::InteractionData, referenceID::Int64, intercept::Float64, intercept_sigma::Float64)

Precomputes the multi-protocol regression prior distribution once for all proteins.

# Returns
- `InferenceResult`: The cached prior distribution.
"""
function precompute_regression_multi_protocol_prior(data::InteractionData, referenceID::Int64, intercept::Float64, intercept_sigma::Float64)
    # Get structure from first protein
    prepared_data = prepare_regression_data(data, 1, referenceID)
    sample = prepared_data.sample

    constraints_regression = @constraints begin
        q(μ_α, σ_α, μ_β, σ_β, σ) = q(μ_α)q(σ_α)q(μ_β)q(σ_β)q(σ)
        q(μ_α, σ_α, α) = q(μ_α, α)q(σ_α)
        q(μ_β, σ_β, β) = q(μ_β, β)q(σ_β)
        q(predicted_value, σ) = q(predicted_value)q(σ)
    end

    init_regression = @initialization begin
        μ(μ_α) = vague(NormalMeanVariance)
        μ(μ_β) = vague(NormalMeanVariance)
        μ(α) = vague(NormalMeanVariance)
        μ(β) = vague(NormalMeanVariance)
        q(α) = vague(NormalMeanVariance)
        q(β) = vague(NormalMeanVariance)
        q(σ) = Gamma(2.0, 0.1)
        q(σ_α) = Gamma(2.0, 0.1)
        q(σ_β) = Gamma(2.0, 0.1)
    end

    missing_data = fill(missing, size(sample))

    prior = infer(
        model=regression_multi_protocol(
            n_protocols=size(sample, 1),
            max_experiments=size(sample, 2),
            n_samples=size(sample, 3),
            intercept=intercept,
            intercept_sigma=intercept_sigma
        ),
        data=(data=missing_data, reference=missing_data),
        initialization=init_regression,
        constraints=constraints_regression,
        iterations=75,  # Reduced from 100 for performance
        returnvars=KeepLast()
    )

    return prior
end

"""
    precompute_regression_one_protocol_prior(data::InteractionData, referenceID::Int64, intercept::Float64, intercept_sigma::Float64)

Precomputes the single-protocol regression prior distribution once for all proteins.

# Returns
- `InferenceResult`: The cached prior distribution.
"""
function precompute_regression_one_protocol_prior(data::InteractionData, referenceID::Int64, intercept::Float64, intercept_sigma::Float64)
    prepared_data = prepare_regression_data(data, 1, referenceID)
    sample = prepared_data.sample[1, :, :]

    init_regression = @initialization begin
        μ(β) = vague(NormalMeanPrecision)
        q(σ) = vague(GammaShapeRate)
    end

    missing_data = fill(missing, size(sample))

    prior = infer(
        model=regression_one_protocol(
            max_experiments=size(sample, 1),
            n_samples=size(sample, 2),
            intercept=intercept,
            intercept_sigma=intercept_sigma
        ),
        data=(data=missing_data, reference=missing_data),
        initialization=init_regression,
        constraints=MeanField(),
        iterations=75,  # Reduced from 100 for performance
        returnvars=KeepLast()
    )

    return prior
end

"""
    HBM(data::InteractionData, idx::Int64; μ_0::F = 25.0, a_0::F = 1.0, b_0::F = 1.0, cached_prior::Union{Nothing,InferenceResult}=nothing, hbm_iterations::Int=75) where {F<:AbstractFloat}

    Hierarchical Bayesian Model

    Input:
        data::InteractionData
        idx::Int64
        μ_0::F = 25.0: mean of the Normal distribution prior for the protein intensity values
        σ_0::F = 1.0: variance of the Normal distribution prior for the protein intensity values
        a_0::F = 1.0: shape of the Gamma distribution prior for the precision
        b_0::F = 1.0: scale of the Gamma distribution prior for the precision
        cached_prior::Union{Nothing,InferenceResult}=nothing: Optional precomputed prior to reuse across proteins
        hbm_iterations::Int = 75: Number of VMP iterations (the Gaussian-Gamma mean-field converges well before this; 75 is bit-identical to 1000 on single-protocol HAP40)

    Output:
        posterior::InferenceResult
        prior::InferenceResult

    Description:

    This function fits a hierarchical Bayesian model to the protein interaction data using the
    Hierarchical Bayesian Model (HBM). The HBM is a Bayesian model uses a hierarchical
    structure to model the data, where the parameters of the model are estimated from the data.

    The HBM comprises of three levels:
      - 1: top level parameters for the entire dataset
      - 2: second level parameters for the protocols (i.e. different experiments, publications etc.)
      - 3: third level parameters for the individual experiments belonging to a certain protocol
"""
function HBM(data::InteractionData, idx::Int64; μ_0::F=25.0, σ_0::F=1.0, a_0::F=1.0, b_0::F=1.0, cached_prior::Union{Nothing,InferenceResult}=nothing, hbm_iterations::Int=75) where {F<:AbstractFloat}
    # get data
    protein = getProteinData(data, idx)
    interactome_sample::Array{Union{Missing,Float64},3} = getSampleMatrix(protein)
    interactome_control::Array{Union{Missing,Float64},3} = getControlMatrix(protein)
    # get hyperparameters
    n_protocols::Int64 = size(interactome_sample, 1)
    n_experiments::Dict{Int64,Int64} = getNoExperiments(data)
    protocol_positions = getProtocolPositions(data)
    experiment_positions = getExperimentPositions(data)
    matched_positions = getMatchedPositions(data)
    parameter_lookup = getParameterLookup(n_protocols, n_experiments)
    nparameters = data.no_parameters_HBM

    # confirm that protocol_positions is not in experiment_positions
    @assert all(i -> !(i in experiment_positions), protocol_positions) "protocol_positions must not be in experiment_positions"::String
    # assert that protocol_positions are not elements of parameter_lookup
    @assert all(i -> !(i in parameter_lookup), protocol_positions) "protocol_positions must not be in parameter_lookup"::String
    # assert that n_protocols = length(protocol_positions)
    @assert n_protocols == length(protocol_positions) "n_protocols must equal length(protocol_positions)"::String

    # define the constraints
    constraints = @constraints begin
        # μ_control and σ_control are jointly independent
        q(μ_control, σ_control) = q(μ_control)q(σ_control)
        # μ_sample and σ_sample are jointly independent
        q(μ_sample, σ_sample) = q(μ_sample)q(σ_sample)
    end

    # define the initialization
    init = @initialization begin
        q(μ_control) = vague(NormalMeanPrecision)
        q(σ_control) = vague(GammaShapeRate)

        q(μ_sample) = vague(NormalMeanPrecision)
        q(σ_sample) = vague(GammaShapeRate)
    end

    # compute posterior
    posterior::InferenceResult = infer(
        model=HierarchicalBayesianModel(
            μ=μ_0, σ=σ_0, a=a_0, b=b_0,
            nparameters=nparameters,
            n_protocols=n_protocols,
            experiment_positions=experiment_positions,
            protocol_positions=protocol_positions,
            matched_positions=matched_positions,
            n_experiments=n_experiments,
            parameter_lookup=parameter_lookup
        ),
        data=(samples=interactome_sample, controls=interactome_control),
        initialization=init,
        constraints=constraints,
        iterations=hbm_iterations,
        returnvars=KeepLast()
    )

    # Use cached prior if provided, otherwise compute it
    prior::InferenceResult = if !isnothing(cached_prior)
        cached_prior
    else
        missing_complete = fill(missing, size(interactome_sample))
        infer(
            model=HierarchicalBayesianModel(
                μ=μ_0, σ=σ_0, a=a_0, b=b_0,
                nparameters=nparameters,
                n_protocols=n_protocols,
                experiment_positions=experiment_positions,
                protocol_positions=protocol_positions,
                matched_positions=matched_positions,
                n_experiments=n_experiments,
                parameter_lookup=parameter_lookup
            ),
            data=(samples=missing_complete, controls=missing_complete),
            initialization=init,
            constraints=constraints,
            iterations=hbm_iterations,
            returnvars=KeepLast()
        )
    end

    return HBMResultMultipleProtocols(posterior, prior)
end

function HBM_single_protocol(data::InteractionData, idx::Int64; μ_0::F=25.0, σ_0::F=1.0, a_0::F=1.0, b_0::F=1.0, cached_prior::Union{Nothing,InferenceResult}=nothing, hbm_iterations::Int=75) where {F<:AbstractFloat}
    if getNoProtocols(data) != 1
        throw(ArgumentError("Data must only contain one protocol. Use HBM() instead."))
    end

    # get data
    protein = getProteinData(data, idx)
    sample_data = getSampleMatrix(protein)[1, :, :]
    control_data = getControlMatrix(protein)[1, :, :]

    # define the constraints
    constraints = @constraints begin
        # μ_control and σ_control are jointly independent
        q(μ_control, σ_control) = q(μ_control)q(σ_control)
        # μ_sample and σ_sample are jointly independent
        q(μ_sample, σ_sample) = q(μ_sample)q(σ_sample)
    end

    # define the initialization
    init = @initialization begin
        q(μ_control) = vague(NormalMeanPrecision)
        q(σ_control) = vague(GammaShapeRate)

        q(μ_sample) = vague(NormalMeanPrecision)
        q(σ_sample) = vague(GammaShapeRate)
    end

    # compute posterior
    posterior::InferenceResult = infer(
        model=HierarchicalBayesianModelSingle(μ=μ_0, σ=σ_0, a=a_0, b=b_0),
        data=(samples=sample_data, controls=control_data),
        initialization=init,
        constraints=constraints,
        iterations=hbm_iterations,
        returnvars=KeepLast()
    )

    # Use cached prior if provided, otherwise compute it
    prior::InferenceResult = if !isnothing(cached_prior)
        cached_prior
    else
        missing_complete = fill(missing, size(sample_data))
        infer(
            model=HierarchicalBayesianModelSingle(μ=μ_0, σ=σ_0, a=a_0, b=b_0),
            data=(samples=missing_complete, controls=missing_complete),
            initialization=init,
            constraints=constraints,
            iterations=hbm_iterations,
            returnvars=KeepLast()
        )
    end

    return HBMResultSingleProtocol(posterior, prior)
end


# ------------------------------------ Regression -------------------------------#

@model function regression_multi_protocol(data, reference, n_protocols, max_experiments, n_samples, intercept, intercept_sigma)
    # ---------- hyper-means ----------------------------------------------
    μ_α ~ Normal(mean=0.0, variance=(0.3 / 1.96)^2)  # 95 % mass in |α|≤0.5
    μ_β ~ Normal(mean=intercept, variance=intercept_sigma)
    # ---------- hyper-precisions (Gamma on PRECISION = 1/σ²) -------------
    σ_α ~ Gamma(shape=6.303676, scale=7.931880)     # mean 50, 95 % in [20,100]
    σ_β ~ Gamma(shape=10.0, scale=0.3)              # mean 3
    # ---------- per-protocol coefficients --------------------------------
    local α
    local β

    for protocol ∈ 1:n_protocols
        α[protocol] ~ Normal(mean=μ_α, precision=σ_α)
        β[protocol] ~ Normal(mean=μ_β, precision=σ_β)
    end

    σ ~ Gamma(shape=5.0, scale=2.0)

    #local predicted_value

    # likelihood
    for (protocol, experiment, sampleID) ∈ Iterators.product(1:n_protocols, 1:max_experiments, 1:n_samples)
        # compute predicted value using α and β for each protocol
        predicted_value[protocol, experiment, sampleID] ~ β[protocol] + α[protocol] * reference[protocol, experiment, sampleID]
        # likelihood
        data[protocol, experiment, sampleID] ~ Normal(mean=predicted_value[protocol, experiment, sampleID], precision=σ)
    end
    return nothing
end


@model function regression_one_protocol(data, reference, max_experiments, n_samples, intercept, intercept_sigma)
    # --- Priors for a single slope and intercept ---
    α ~ Normal(mean=0.0, variance=(0.3 / 1.96)^2)
    β ~ Normal(mean=intercept, variance=intercept_sigma)

    # Prior for the residual precision
    σ ~ Gamma(shape=5.0, scale=2.0)

    # --- Likelihood ---
    for (experiment, sampleID) in Iterators.product(1:max_experiments, 1:n_samples)
        data[experiment, sampleID] ~ Normal(mean=β + α * reference[experiment, sampleID], precision=σ)
    end

    return nothing
end

# ------------------------------------ Robust Regression (Student-t via scale mixture) ----- #

@model function regression_multi_protocol_robust(data, reference, n_protocols, max_experiments, n_samples, intercept, intercept_sigma, nu_half, τ_base)
    # ---------- hyper-means (same as normal model) ----------------------------
    μ_α ~ Normal(mean=0.0, variance=(0.3 / 1.96)^2)
    μ_β ~ Normal(mean=intercept, variance=intercept_sigma)
    # ---------- hyper-precisions (Gamma on PRECISION = 1/σ²) ------------------
    σ_α ~ Gamma(shape=6.303676, scale=7.931880)
    σ_β ~ Gamma(shape=10.0, scale=0.3)
    # ---------- per-protocol coefficients -------------------------------------
    local α
    local β

    for protocol ∈ 1:n_protocols
        α[protocol] ~ Normal(mean=μ_α, precision=σ_α)
        β[protocol] ~ Normal(mean=μ_β, precision=σ_β)
    end

    # ---------- Per-observation Gamma precision (Empirical Bayes) --------------
    # τ_i ~ Gamma(ν/2, scale = τ_base/(ν/2)) so E[τ_i] = τ_base
    # Marginal: y_i | μ_i ~ Student-t(ν, μ_i, τ_base)
    # Normal-Gamma conjugate pair: fully VMP-compatible
    local τ

    for (protocol, experiment, sampleID) ∈ Iterators.product(1:n_protocols, 1:max_experiments, 1:n_samples)
        τ[protocol, experiment, sampleID] ~ Gamma(shape=nu_half, scale=τ_base / nu_half)
        predicted_value[protocol, experiment, sampleID] ~ β[protocol] + α[protocol] * reference[protocol, experiment, sampleID]
        data[protocol, experiment, sampleID] ~ Normal(
            mean=predicted_value[protocol, experiment, sampleID],
            precision=τ[protocol, experiment, sampleID]
        )
    end
    return nothing
end


@model function regression_one_protocol_robust(data, reference, max_experiments, n_samples, intercept, intercept_sigma, nu_half, τ_base)
    # --- Priors for a single slope and intercept ---
    α ~ Normal(mean=0.0, variance=(0.3 / 1.96)^2)
    β ~ Normal(mean=intercept, variance=intercept_sigma)

    # --- Per-observation Gamma precision (Empirical Bayes) ---
    # τ_i ~ Gamma(ν/2, scale = τ_base/(ν/2)) so E[τ_i] = τ_base
    # Normal-Gamma conjugate pair: fully VMP-compatible
    local τ
    for (experiment, sampleID) in Iterators.product(1:max_experiments, 1:n_samples)
        τ[experiment, sampleID] ~ Gamma(shape=nu_half, scale=τ_base / nu_half)
        data[experiment, sampleID] ~ Normal(
            mean=β + α * reference[experiment, sampleID],
            precision=τ[experiment, sampleID]
        )
    end

    return nothing
end


# ------------------------------------ Robust Regression with JZS Prior ----- #
# JZS (Jeffreys-Zellner-Siow) prior on the slope parameter:
#   τ_g ~ Gamma(1/2, scale = 2/r²)   ← local shrinkage precision
#   α   ~ Normal(0, precision = τ_g)  ← marginal: Cauchy(0, r)
# The Normal-Gamma pair is fully VMP-compatible.
# Reference: Ly et al. (2016), Rouder et al. (2009), JASP default r = √2/4 ≈ 0.354

@model function regression_multi_protocol_robust_jzs(data, reference, n_protocols, max_experiments, n_samples, intercept, intercept_sigma, nu_half, τ_base, jzs_r_scale)
    # ---------- JZS prior on hyper-mean slope (Cauchy via scale mixture) ------
    τ_g ~ Gamma(shape=0.5, scale=2.0 / jzs_r_scale^2)
    μ_α ~ Normal(mean=0.0, precision=τ_g)
    # ---------- hyper-mean intercept ------------------------------------------
    μ_β ~ Normal(mean=intercept, variance=intercept_sigma)
    # ---------- hyper-precisions (Gamma on PRECISION = 1/σ²) ------------------
    σ_α ~ Gamma(shape=6.303676, scale=7.931880)
    σ_β ~ Gamma(shape=10.0, scale=0.3)
    # ---------- per-protocol coefficients -------------------------------------
    local α
    local β

    for protocol ∈ 1:n_protocols
        α[protocol] ~ Normal(mean=μ_α, precision=σ_α)
        β[protocol] ~ Normal(mean=μ_β, precision=σ_β)
    end

    # ---------- Per-observation Gamma precision (Empirical Bayes) --------------
    local τ

    for (protocol, experiment, sampleID) ∈ Iterators.product(1:n_protocols, 1:max_experiments, 1:n_samples)
        τ[protocol, experiment, sampleID] ~ Gamma(shape=nu_half, scale=τ_base / nu_half)
        predicted_value[protocol, experiment, sampleID] ~ β[protocol] + α[protocol] * reference[protocol, experiment, sampleID]
        data[protocol, experiment, sampleID] ~ Normal(
            mean=predicted_value[protocol, experiment, sampleID],
            precision=τ[protocol, experiment, sampleID]
        )
    end
    return nothing
end


# ------------------------------------------------------------------------- #
# Mask-Aware Regression v2b (Path B)
# ------------------------------------------------------------------------- #
# v2b replaces the per-cell `data ~ Normal(..., precision=τ[cell])` observation
# factor with a Path-B latent chain:
#
#     y_bio[cell] ~ Normal(mean=predicted_value[cell], precision=τ[cell])
#     data[cell]  ~ Normal(mean=y_bio[cell],
#                          variance=sigma_sq_imp_mask[cell] + 1e-8)
#
# The per-cell Gamma `τ ~ Gamma(shape=nu_half, scale=τ_base/nu_half)` is
# RETAINED → Student-t marginal robustness preserved.
# Imputed cells receive an ADDITIVE observation variance σ²_imp[cell]; non-
# imputed cells get a `+ 1e-8` floor so the outer Normal is strictly positive.
# The legacy pre-spike function `regression_multi_protocol_robust_jzs` above
# remains untouched — pipeline driver dispatches between the two.

@model function regression_multi_protocol_robust_jzs_v2b(data, reference, sigma_sq_imp_mask, n_protocols, max_experiments, n_samples, intercept, intercept_sigma, nu_half, τ_base, jzs_r_scale)
    # ---------- JZS prior on hyper-mean slope (Cauchy via scale mixture) ------
    τ_g ~ Gamma(shape=0.5, scale=2.0 / jzs_r_scale^2)
    μ_α ~ Normal(mean=0.0, precision=τ_g)
    # ---------- hyper-mean intercept ------------------------------------------
    μ_β ~ Normal(mean=intercept, variance=intercept_sigma)
    # ---------- hyper-precisions (Gamma on PRECISION = 1/σ²) ------------------
    σ_α ~ Gamma(shape=6.303676, scale=7.931880)
    σ_β ~ Gamma(shape=10.0, scale=0.3)
    # ---------- per-protocol coefficients -------------------------------------
    local α
    local β

    for protocol ∈ 1:n_protocols
        α[protocol] ~ Normal(mean=μ_α, precision=σ_α)
        β[protocol] ~ Normal(mean=μ_β, precision=σ_β)
    end

    # ---------- Per-observation Gamma τ (RETAINED — Student-t robustness) ------
    # ---------- + variance-additive observation factor (Path B) ----------------
    local τ
    local y_bio

    for (protocol, experiment, sampleID) ∈ Iterators.product(1:n_protocols, 1:max_experiments, 1:n_samples)
        τ[protocol, experiment, sampleID] ~ Gamma(shape=nu_half, scale=τ_base / nu_half)
        predicted_value[protocol, experiment, sampleID] ~ β[protocol] + α[protocol] * reference[protocol, experiment, sampleID]
        y_bio[protocol, experiment, sampleID] ~ Normal(
            mean=predicted_value[protocol, experiment, sampleID],
            precision=τ[protocol, experiment, sampleID]
        )
        data[protocol, experiment, sampleID] ~ Normal(
            mean=y_bio[protocol, experiment, sampleID],
            variance=sigma_sq_imp_mask[protocol, experiment, sampleID] + 1e-8
        )
    end
    return nothing
end


@model function regression_one_protocol_robust_jzs(data, reference, max_experiments, n_samples, intercept, intercept_sigma, nu_half, τ_base, jzs_r_scale)
    # --- JZS prior on slope (Cauchy via scale mixture) ---
    τ_g ~ Gamma(shape=0.5, scale=2.0 / jzs_r_scale^2)
    α ~ Normal(mean=0.0, precision=τ_g)
    # --- Intercept prior ---
    β ~ Normal(mean=intercept, variance=intercept_sigma)

    # --- Per-observation Gamma precision (Empirical Bayes) ---
    local τ
    for (experiment, sampleID) in Iterators.product(1:max_experiments, 1:n_samples)
        τ[experiment, sampleID] ~ Gamma(shape=nu_half, scale=τ_base / nu_half)
        data[experiment, sampleID] ~ Normal(
            mean=β + α * reference[experiment, sampleID],
            precision=τ[experiment, sampleID]
        )
    end

    return nothing
end


# Mask-Aware Regression v2b — single-protocol path (parity with multi-protocol v2b)
# Same per-cell Gamma τ + Path-B latent y_bio chain as the multi-protocol v2b.
# The only structural difference is the absence of the protocol loop (n_protocols = 1).
@model function regression_one_protocol_robust_jzs_v2b(data, reference, sigma_sq_imp_mask, max_experiments, n_samples, intercept, intercept_sigma, nu_half, τ_base, jzs_r_scale)
    # --- JZS prior on slope (Cauchy via scale mixture) ---
    τ_g ~ Gamma(shape=0.5, scale=2.0 / jzs_r_scale^2)
    α ~ Normal(mean=0.0, precision=τ_g)
    # --- Intercept prior ---
    β ~ Normal(mean=intercept, variance=intercept_sigma)

    # --- Per-observation Gamma τ (RETAINED) + Path B latent y_bio chain ---
    local τ
    local y_bio
    for (experiment, sampleID) in Iterators.product(1:max_experiments, 1:n_samples)
        τ[experiment, sampleID] ~ Gamma(shape=nu_half, scale=τ_base / nu_half)
        predicted_value[experiment, sampleID] ~ β + α * reference[experiment, sampleID]
        y_bio[experiment, sampleID] ~ Normal(
            mean=predicted_value[experiment, sampleID],
            precision=τ[experiment, sampleID]
        )
        data[experiment, sampleID] ~ Normal(
            mean=y_bio[experiment, sampleID],
            variance=sigma_sq_imp_mask[experiment, sampleID] + 1e-8
        )
    end

    return nothing
end


# ─────────────────────────────────────────────────────────────────────────────
# Structured single-protocol regression (production fix).
#
# Root cause of bf_correlation saturation (Spikes 016/017): the mean-field
# q(α)q(β) factorisation collapses the slope-posterior variance (~7× narrower
# than the OLS SE), pinning ~51% of proteins at the 1e6 BF clamp. NO
# observation-factor variant (Path-B / hybrid / fixed-τ) fixes this. The fix is
# to make the (slope, intercept) posterior JOINT — modelled here as a single
# 2-D MvNormal node θ = [slope, intercept] so RxInfer propagates the exact joint
# Gaussian covariance (the loopy scalar-α/β graph stays overconfident even under
# a coupling constraint; only a single joint node works).
#
# Student-t robustness is RETAINED via per-cell Gamma τ (the scale mixture):
# an outlier cell's τ posterior drops (verified ~23× down-weighting at a 3σ
# outlier). The MNAR σ²_imp down-weighting is folded into the per-cell τ PRIOR
# MEAN (imputed cell → Gamma centred at 1/(1/τ_base + σ²_imp)), which keeps the
# model conjugate — avoiding the non-conjugate `1/τ + σ²` Path-A trap.
#
# JZS note: RxInfer cannot place the JZS Cauchy scale-mixture (random τ_g
# precision) on a joint MvNormal component, so the slope POSTERIOR prior is a
# scale-matched Normal(0, jzs_r_scale²). The JZS Cauchy is still the BF's
# analytical null reference (BayesFactorRegression with jzs_r_scale>0 uses
# `_cauchy_sf` for p_prior) — unchanged.
@model function regression_one_protocol_robust_structured(y, Xrow, n_obs, tau_prior_base, nu_half, prior_mean, prior_cov)
    θ ~ MvNormal(mean = prior_mean, covariance = prior_cov)   # θ = [slope, intercept], joint
    local τ
    for k in 1:n_obs
        τ[k] ~ Gamma(shape = nu_half, scale = tau_prior_base[k] / nu_half)
        y[k] ~ Normal(mean = dot(Xrow[k], θ), precision = τ[k])
    end
    return nothing
end

# Structured-VMP factorisation: joint q(θ), per-cell q(τ). The joint q(θ) is the
# whole point — it preserves the slope-intercept posterior covariance that
# mean-field discards.
@constraints function regression_structured_constraints()
    q(θ, τ) = q(θ)q(τ)
end


# Prior-only model for JZS multi-protocol: shared σ instead of per-obs τ.
# Per-observation τ in robust models triggers RuleMethodError in RxInfer when
# all data is missing. Prior marginals on τ_g, μ_α, σ_α, μ_β, σ_β, α, β are
# identical regardless of likelihood precision structure (shared vs per-obs).
@model function _regression_multi_protocol_jzs_prior_model(data, reference, n_protocols, max_experiments, n_samples, intercept, intercept_sigma, jzs_r_scale)
    τ_g ~ Gamma(shape=0.5, scale=2.0 / jzs_r_scale^2)
    μ_α ~ Normal(mean=0.0, precision=τ_g)
    μ_β ~ Normal(mean=intercept, variance=intercept_sigma)
    σ_α ~ Gamma(shape=6.303676, scale=7.931880)
    σ_β ~ Gamma(shape=10.0, scale=0.3)
    local α
    local β

    for protocol ∈ 1:n_protocols
        α[protocol] ~ Normal(mean=μ_α, precision=σ_α)
        β[protocol] ~ Normal(mean=μ_β, precision=σ_β)
    end

    σ ~ Gamma(shape=5.0, scale=2.0)

    for (protocol, experiment, sampleID) ∈ Iterators.product(1:n_protocols, 1:max_experiments, 1:n_samples)
        predicted_value[protocol, experiment, sampleID] ~ β[protocol] + α[protocol] * reference[protocol, experiment, sampleID]
        data[protocol, experiment, sampleID] ~ Normal(
            mean=predicted_value[protocol, experiment, sampleID],
            precision=σ
        )
    end
    return nothing
end


# Prior-only model for v2b mask-aware regression.
# Mirrors the topology of `_regression_multi_protocol_jzs_prior_model`
# (shared σ — per-cell τ in the prior would trigger RuleMethodError with
# missing data — see the rationale comment above) but adds the v2b
# Path-B latent y_bio chain and the variance-additive outer observation.
# Wrapper supplies `sigma_sq_imp_mask = zeros(...)` for the prior fit
# (no imputation noise on `missing` prior data).
@model function _regression_multi_protocol_jzs_prior_model_v2b(data, reference, sigma_sq_imp_mask, n_protocols, max_experiments, n_samples, intercept, intercept_sigma, jzs_r_scale)
    τ_g ~ Gamma(shape=0.5, scale=2.0 / jzs_r_scale^2)
    μ_α ~ Normal(mean=0.0, precision=τ_g)
    μ_β ~ Normal(mean=intercept, variance=intercept_sigma)
    σ_α ~ Gamma(shape=6.303676, scale=7.931880)
    σ_β ~ Gamma(shape=10.0, scale=0.3)
    local α
    local β

    for protocol ∈ 1:n_protocols
        α[protocol] ~ Normal(mean=μ_α, precision=σ_α)
        β[protocol] ~ Normal(mean=μ_β, precision=σ_β)
    end

    σ ~ Gamma(shape=5.0, scale=2.0)
    local y_bio

    for (protocol, experiment, sampleID) ∈ Iterators.product(1:n_protocols, 1:max_experiments, 1:n_samples)
        predicted_value[protocol, experiment, sampleID] ~ β[protocol] + α[protocol] * reference[protocol, experiment, sampleID]
        y_bio[protocol, experiment, sampleID] ~ Normal(
            mean=predicted_value[protocol, experiment, sampleID],
            precision=σ
        )
        data[protocol, experiment, sampleID] ~ Normal(
            mean=y_bio[protocol, experiment, sampleID],
            variance=sigma_sq_imp_mask[protocol, experiment, sampleID] + 1e-8
        )
    end
    return nothing
end


"""
    precompute_regression_multi_protocol_robust_prior(data, referenceID, intercept, intercept_sigma; nu=5.0, τ_base=1.0)

Precomputes the multi-protocol robust regression prior distribution once for all proteins.

# Returns
- `InferenceResult`: The cached prior distribution.
"""
function precompute_regression_multi_protocol_robust_prior(data::InteractionData, referenceID::Int64, intercept::Float64, intercept_sigma::Float64; nu::Float64=5.0, τ_base::Float64=1.0)
    prepared_data = prepare_regression_data(data, 1, referenceID)
    sample = prepared_data.sample
    nu_half = nu / 2.0

    constraints_regression = @constraints begin
        q(μ_α, σ_α, μ_β, σ_β, τ) = q(μ_α)q(σ_α)q(μ_β)q(σ_β)q(τ)
        q(μ_α, σ_α, α) = q(μ_α, α)q(σ_α)
        q(μ_β, σ_β, β) = q(μ_β, β)q(σ_β)
        q(predicted_value, τ) = q(predicted_value)q(τ)
    end

    init_regression = @initialization begin
        μ(μ_α) = vague(NormalMeanVariance)
        μ(μ_β) = vague(NormalMeanVariance)
        μ(α) = vague(NormalMeanVariance)
        μ(β) = vague(NormalMeanVariance)
        q(α) = vague(NormalMeanVariance)
        q(β) = vague(NormalMeanVariance)
        q(τ) = vague(GammaShapeRate)
        q(σ_α) = Gamma(2.0, 0.1)
        q(σ_β) = Gamma(2.0, 0.1)
    end

    missing_data = fill(missing, size(sample))

    prior = infer(
        model=regression_multi_protocol_robust(
            n_protocols=size(sample, 1),
            max_experiments=size(sample, 2),
            n_samples=size(sample, 3),
            intercept=intercept,
            intercept_sigma=intercept_sigma,
            nu_half=nu_half,
            τ_base=τ_base
        ),
        data=(data=missing_data, reference=missing_data),
        initialization=init_regression,
        constraints=constraints_regression,
        iterations=150,
        returnvars=KeepLast()
    )

    return prior
end


"""
    precompute_regression_one_protocol_robust_prior(data, referenceID, intercept, intercept_sigma; nu=5.0, τ_base=1.0)

Precomputes the single-protocol robust regression prior distribution once for all proteins.

# Returns
- `InferenceResult`: The cached prior distribution.
"""
function precompute_regression_one_protocol_robust_prior(data::InteractionData, referenceID::Int64, intercept::Float64, intercept_sigma::Float64; nu::Float64=5.0, τ_base::Float64=1.0)
    prepared_data = prepare_regression_data(data, 1, referenceID)
    sample = prepared_data.sample[1, :, :]
    nu_half = nu / 2.0

    init_regression = @initialization begin
        μ(β) = vague(NormalMeanPrecision)
        q(τ) = vague(GammaShapeRate)
    end

    missing_data = fill(missing, size(sample))

    prior = infer(
        model=regression_one_protocol_robust(
            max_experiments=size(sample, 1),
            n_samples=size(sample, 2),
            intercept=intercept,
            intercept_sigma=intercept_sigma,
            nu_half=nu_half,
            τ_base=τ_base
        ),
        data=(data=missing_data, reference=missing_data),
        initialization=init_regression,
        constraints=MeanField(),
        iterations=150,
        returnvars=KeepLast()
    )

    return prior
end


"""
    precompute_regression_multi_protocol_robust_jzs_prior(data, referenceID, intercept, intercept_sigma; nu=5.0, τ_base=1.0, jzs_r_scale=0.354)

Precomputes the multi-protocol robust regression prior with JZS (Cauchy) prior on μ_α.
"""
function precompute_regression_multi_protocol_robust_jzs_prior(data::InteractionData, referenceID::Int64, intercept::Float64, intercept_sigma::Float64; nu::Float64=5.0, τ_base::Float64=1.0, jzs_r_scale::Float64=0.354)
    prepared_data = prepare_regression_data(data, 1, referenceID)
    sample = prepared_data.sample

    constraints_regression = @constraints begin
        q(τ_g, μ_α, σ_α, μ_β, σ_β, σ) = q(τ_g)q(μ_α)q(σ_α)q(μ_β)q(σ_β)q(σ)
        q(μ_α, σ_α, α) = q(μ_α, α)q(σ_α)
        q(μ_β, σ_β, β) = q(μ_β, β)q(σ_β)
        q(predicted_value, σ) = q(predicted_value)q(σ)
    end

    init_regression = @initialization begin
        μ(μ_α) = vague(NormalMeanVariance)
        μ(μ_β) = vague(NormalMeanVariance)
        μ(α) = vague(NormalMeanVariance)
        μ(β) = vague(NormalMeanVariance)
        q(α) = vague(NormalMeanVariance)
        q(β) = vague(NormalMeanVariance)
        q(σ) = Gamma(2.0, 0.1)
        q(τ_g) = Gamma(2.0, 0.1)
        q(σ_α) = Gamma(2.0, 0.1)
        q(σ_β) = Gamma(2.0, 0.1)
    end

    missing_data = fill(missing, size(sample))

    prior = infer(
        model=_regression_multi_protocol_jzs_prior_model(
            n_protocols=size(sample, 1),
            max_experiments=size(sample, 2),
            n_samples=size(sample, 3),
            intercept=intercept,
            intercept_sigma=intercept_sigma,
            jzs_r_scale=jzs_r_scale
        ),
        data=(data=missing_data, reference=missing_data),
        initialization=init_regression,
        constraints=constraints_regression,
        iterations=150,
        returnvars=KeepLast()
    )

    return prior
end


"""
    precompute_regression_one_protocol_robust_jzs_prior(data, referenceID, intercept, intercept_sigma; nu=5.0, τ_base=1.0, jzs_r_scale=0.354)

Precomputes the single-protocol robust regression prior with JZS (Cauchy) prior on α.
"""
function precompute_regression_one_protocol_robust_jzs_prior(data::InteractionData, referenceID::Int64, intercept::Float64, intercept_sigma::Float64; nu::Float64=5.0, τ_base::Float64=1.0, jzs_r_scale::Float64=0.354)
    prepared_data = prepare_regression_data(data, 1, referenceID)
    sample = prepared_data.sample[1, :, :]
    nu_half = nu / 2.0

    init_regression = @initialization begin
        μ(β) = vague(NormalMeanPrecision)
        q(τ) = vague(GammaShapeRate)
        q(τ_g) = Gamma(2.0, 0.1)
    end

    missing_data = fill(missing, size(sample))

    prior = infer(
        model=regression_one_protocol_robust_jzs(
            max_experiments=size(sample, 1),
            n_samples=size(sample, 2),
            intercept=intercept,
            intercept_sigma=intercept_sigma,
            nu_half=nu_half,
            τ_base=τ_base,
            jzs_r_scale=jzs_r_scale
        ),
        data=(data=missing_data, reference=missing_data),
        initialization=init_regression,
        constraints=MeanField(),
        iterations=150,
        returnvars=KeepLast()
    )

    return prior
end


"""
    RegressionModelRobust(data, idx, referenceID, intercept, intercept_sigma; nu=5.0, τ_base=NaN, cached_prior=nothing, regression_iterations=150)

Computes the robust regression model (Student-t likelihood via Empirical Bayes) for multiple protocols.
If `τ_base` is NaN, it is estimated from the data via `estimate_regression_tau_base`.
"""
function RegressionModelRobust(data::InteractionData, idx::Int64, referenceID::Int64, intercept::Float64, intercept_sigma::Float64; nu::Float64=5.0, τ_base::Float64=NaN, cached_prior::Union{Nothing,InferenceResult}=nothing, regression_iterations::Int=150)
    # Estimate τ_base if not provided
    if isnan(τ_base)
        τ_base = estimate_regression_tau_base(data, referenceID)
    end

    data = prepare_regression_data(data, idx, referenceID)
    sample, reference = data.sample, data.reference
    nu_half = nu / 2.0

    @assert size(sample) == size(reference) "Mismatch in data dimensions"
    @assert ndims(sample) == 3 "Data must be 3-dimensional"

    constraints_regression = @constraints begin
        q(μ_α, σ_α, μ_β, σ_β, τ) = q(μ_α)q(σ_α)q(μ_β)q(σ_β)q(τ)
        q(μ_α, σ_α, α) = q(μ_α, α)q(σ_α)
        q(μ_β, σ_β, β) = q(μ_β, β)q(σ_β)
        q(predicted_value, τ) = q(predicted_value)q(τ)
    end

    init_regression = @initialization begin
        μ(μ_α) = vague(NormalMeanVariance)
        μ(μ_β) = vague(NormalMeanVariance)
        μ(α) = vague(NormalMeanVariance)
        μ(β) = vague(NormalMeanVariance)
        q(α) = vague(NormalMeanVariance)
        q(β) = vague(NormalMeanVariance)
        q(τ) = vague(GammaShapeRate)
        q(σ_α) = Gamma(2.0, 0.1)
        q(σ_β) = Gamma(2.0, 0.1)
    end

    posterior = infer(
        model=regression_multi_protocol_robust(
            n_protocols=size(sample, 1),
            max_experiments=size(sample, 2),
            n_samples=size(sample, 3),
            intercept=intercept,
            intercept_sigma=intercept_sigma,
            nu_half=nu_half,
            τ_base=τ_base
        ),
        data=(data=sample, reference=reference),
        initialization=init_regression,
        constraints=constraints_regression,
        iterations=regression_iterations,
        returnvars=KeepLast()
    )

    prior = if !isnothing(cached_prior)
        cached_prior
    else
        missing_data = fill(missing, size(sample))
        infer(
            model=regression_multi_protocol_robust(
                n_protocols=size(sample, 1),
                max_experiments=size(sample, 2),
                n_samples=size(sample, 3),
                intercept=intercept,
                intercept_sigma=intercept_sigma,
                nu_half=nu_half,
                τ_base=τ_base
            ),
            data=(data=missing_data, reference=missing_data),
            initialization=init_regression,
            constraints=constraints_regression,
            iterations=regression_iterations,
            returnvars=KeepLast()
        )
    end

    return RobustRegressionResultMultipleProtocols(posterior, prior, nu, τ_base)
end


"""
    RegressionModel_one_protocol_robust(data, idx, referenceID, intercept, intercept_sigma; nu=5.0, τ_base=NaN, cached_prior=nothing, regression_iterations=150)

Computes the robust regression model (Student-t likelihood via Empirical Bayes) for a single protocol.
If `τ_base` is NaN, it is estimated from the data via `estimate_regression_tau_base`.
"""
function RegressionModel_one_protocol_robust(data::InteractionData, idx::Int64, referenceID::Int64, intercept::Float64, intercept_sigma::Float64; nu::Float64=5.0, τ_base::Float64=NaN, cached_prior::Union{Nothing,InferenceResult}=nothing, regression_iterations::Int=150)
    @assert getNoProtocols(data) == 1 "Data must only contain one protocol"

    # Estimate τ_base if not provided
    if isnan(τ_base)
        τ_base = estimate_regression_tau_base(data, referenceID)
    end

    data = prepare_regression_data(data, idx, referenceID)
    sample, reference = data.sample, data.reference
    nu_half = nu / 2.0

    sample, reference = sample[1, :, :], reference[1, :, :]

    @assert size(sample) == size(reference) "Mismatch in data dimensions"
    @assert ndims(sample) == 2 "Data must be 2-dimensional"

    init_regression = @initialization begin
        μ(β) = vague(NormalMeanPrecision)
        q(τ) = vague(GammaShapeRate)
    end

    posterior = infer(
        model=regression_one_protocol_robust(
            max_experiments=size(sample, 1),
            n_samples=size(sample, 2),
            intercept=intercept,
            intercept_sigma=intercept_sigma,
            nu_half=nu_half,
            τ_base=τ_base
        ),
        data=(data=sample, reference=reference),
        initialization=init_regression,
        constraints=MeanField(),
        iterations=regression_iterations,
        returnvars=KeepLast()
    )

    prior = if !isnothing(cached_prior)
        cached_prior
    else
        missing_data = fill(missing, size(sample))
        infer(
            model=regression_one_protocol_robust(
                max_experiments=size(sample, 1),
                n_samples=size(sample, 2),
                intercept=intercept,
                intercept_sigma=intercept_sigma,
                nu_half=nu_half,
                τ_base=τ_base
            ),
            data=(data=missing_data, reference=missing_data),
            initialization=init_regression,
            constraints=MeanField(),
            iterations=regression_iterations,
            returnvars=KeepLast()
        )
    end

    return RobustRegressionResultSingleProtocol(posterior, prior, nu, τ_base)
end


"""
    RegressionModelRobustJZS(data, idx, referenceID, intercept, intercept_sigma; nu, τ_base, jzs_r_scale, cached_prior, regression_iterations)

Robust regression with JZS prior (Cauchy via scale mixture) on μ_α for multiple protocols.
"""
function RegressionModelRobustJZS(data::InteractionData, idx::Int64, referenceID::Int64, intercept::Float64, intercept_sigma::Float64; nu::Float64=5.0, τ_base::Float64=NaN, jzs_r_scale::Float64=0.354, cached_prior::Union{Nothing,InferenceResult}=nothing, regression_iterations::Int=150)
    if isnan(τ_base)
        τ_base = estimate_regression_tau_base(data, referenceID)
    end

    data = prepare_regression_data(data, idx, referenceID)
    sample, reference = data.sample, data.reference
    nu_half = nu / 2.0

    @assert size(sample) == size(reference) "Mismatch in data dimensions"
    @assert ndims(sample) == 3 "Data must be 3-dimensional"

    constraints_regression = @constraints begin
        q(τ_g, μ_α, σ_α, μ_β, σ_β, τ) = q(τ_g)q(μ_α)q(σ_α)q(μ_β)q(σ_β)q(τ)
        q(μ_α, σ_α, α) = q(μ_α, α)q(σ_α)
        q(μ_β, σ_β, β) = q(μ_β, β)q(σ_β)
        q(predicted_value, τ) = q(predicted_value)q(τ)
    end

    init_regression = @initialization begin
        μ(μ_α) = vague(NormalMeanVariance)
        μ(μ_β) = vague(NormalMeanVariance)
        μ(α) = vague(NormalMeanVariance)
        μ(β) = vague(NormalMeanVariance)
        q(α) = vague(NormalMeanVariance)
        q(β) = vague(NormalMeanVariance)
        q(τ) = vague(GammaShapeRate)
        q(τ_g) = Gamma(2.0, 0.1)
        q(σ_α) = Gamma(2.0, 0.1)
        q(σ_β) = Gamma(2.0, 0.1)
    end

    posterior = infer(
        model=regression_multi_protocol_robust_jzs(
            n_protocols=size(sample, 1),
            max_experiments=size(sample, 2),
            n_samples=size(sample, 3),
            intercept=intercept,
            intercept_sigma=intercept_sigma,
            nu_half=nu_half,
            τ_base=τ_base,
            jzs_r_scale=jzs_r_scale
        ),
        data=(data=sample, reference=reference),
        initialization=init_regression,
        constraints=constraints_regression,
        iterations=regression_iterations,
        returnvars=KeepLast()
    )

    prior = if !isnothing(cached_prior)
        cached_prior
    else
        prior_constraints = @constraints begin
            q(τ_g, μ_α, σ_α, μ_β, σ_β, σ) = q(τ_g)q(μ_α)q(σ_α)q(μ_β)q(σ_β)q(σ)
            q(μ_α, σ_α, α) = q(μ_α, α)q(σ_α)
            q(μ_β, σ_β, β) = q(μ_β, β)q(σ_β)
            q(predicted_value, σ) = q(predicted_value)q(σ)
        end
        prior_init = @initialization begin
            μ(μ_α) = vague(NormalMeanVariance)
            μ(μ_β) = vague(NormalMeanVariance)
            μ(α) = vague(NormalMeanVariance)
            μ(β) = vague(NormalMeanVariance)
            q(α) = vague(NormalMeanVariance)
            q(β) = vague(NormalMeanVariance)
            q(σ) = Gamma(2.0, 0.1)
            q(τ_g) = Gamma(2.0, 0.1)
            q(σ_α) = Gamma(2.0, 0.1)
            q(σ_β) = Gamma(2.0, 0.1)
        end
        missing_data = fill(missing, size(sample))
        infer(
            model=_regression_multi_protocol_jzs_prior_model(
                n_protocols=size(sample, 1),
                max_experiments=size(sample, 2),
                n_samples=size(sample, 3),
                intercept=intercept,
                intercept_sigma=intercept_sigma,
                jzs_r_scale=jzs_r_scale
            ),
            data=(data=missing_data, reference=missing_data),
            initialization=prior_init,
            constraints=prior_constraints,
            iterations=regression_iterations,
            returnvars=KeepLast()
        )
    end

    return RobustRegressionResultMultipleProtocols(posterior, prior, nu, τ_base)
end


"""
    RegressionModel_multi_protocol_robust_jzs_v2b(data, idx, referenceID, intercept, intercept_sigma;
                                                  is_imputed,
                                                  column_imputation_sigma_sq,
                                                  raw_data = nothing,
                                                  nu, τ_base, jzs_r_scale,
                                                  cached_prior, regression_iterations)

v2b mask-aware robust regression with JZS prior, multi-protocol path
(Path B).

Wraps `regression_multi_protocol_robust_jzs_v2b` (the v2b model body) by
pre-computing a per-cell `sigma_sq_imp_mask::Array{Float64, 3}` from the
caller-supplied `is_imputed` mask and `column_imputation_sigma_sq` lookup:

```
sigma_sq_imp_mask[p,e,s] =
    is_imputed[p,e,s] ? get(column_imputation_sigma_sq, (p,e,s), 0.0) : 0.0
```

For cells flagged non-imputed the additive variance is zero and the outer
observation degenerates to `Normal(y_bio, variance=1e-8)` (a near-Dirac
point-mass — the per-cell Gamma τ on `y_bio` still does the Student-t work).

# Required kwargs
- `is_imputed::Array{Bool, 3}` — mask from `prepare_regression_data(...; raw_data=...)`
- `column_imputation_sigma_sq::Dict{Tuple{Int,Int,Int}, Float64}` — per-cell σ²_imp
- `raw_data::Union{Nothing, InteractionData}` — passed to `prepare_regression_data`

Legacy multi-protocol JZS wrapper `RegressionModelRobustJZS` is preserved
verbatim; dispatches between the two via the `mask_aware_regression`
pipeline kwarg.
"""
function RegressionModel_multi_protocol_robust_jzs_v2b(
    data::InteractionData, idx::Int64, referenceID::Int64,
    intercept::Float64, intercept_sigma::Float64;
    is_imputed::AbstractArray{Bool, 3},
    column_imputation_sigma_sq::Dict{Tuple{Int,Int,Int}, Float64},
    raw_data::Union{Nothing, InteractionData} = nothing,
    nu::Float64=5.0, τ_base::Float64=NaN, jzs_r_scale::Float64=0.354,
    cached_prior::Union{Nothing,InferenceResult}=nothing,
    regression_iterations::Int=150,
)
    if isnan(τ_base)
        τ_base = estimate_regression_tau_base(data, referenceID)
    end

    prepared = prepare_regression_data(data, idx, referenceID; raw_data=raw_data)
    sample, reference = prepared.sample, prepared.reference
    nu_half = nu / 2.0

    @assert size(sample) == size(reference) "Mismatch in data dimensions"
    @assert ndims(sample) == 3 "Data must be 3-dimensional"
    @assert size(is_imputed) == size(sample) "is_imputed mask shape != sample shape"

    # Build per-cell σ²_imp additive-variance array (Path B variance-additive contract)
    sigma_sq_imp_mask = zeros(Float64, size(sample))
    for I in CartesianIndices(size(sample))
        if is_imputed[I]
            p, e, s = Tuple(I)
            sigma_sq_imp_mask[I] = get(column_imputation_sigma_sq, (p, e, s), 0.0)
        end
    end

    # Path-B constraints: production block + ONE new factorisation `q(y_bio, τ) = q(y_bio)q(τ)`
    constraints_regression = @constraints begin
        q(τ_g, μ_α, σ_α, μ_β, σ_β, τ) = q(τ_g)q(μ_α)q(σ_α)q(μ_β)q(σ_β)q(τ)
        q(μ_α, σ_α, α) = q(μ_α, α)q(σ_α)
        q(μ_β, σ_β, β) = q(μ_β, β)q(σ_β)
        q(predicted_value, τ) = q(predicted_value)q(τ)
        q(y_bio, τ) = q(y_bio)q(τ)
    end

    # Path-B init: production block + ONE new line `q(y_bio) = vague(NormalMeanVariance)`
    init_regression = @initialization begin
        μ(μ_α) = vague(NormalMeanVariance)
        μ(μ_β) = vague(NormalMeanVariance)
        μ(α) = vague(NormalMeanVariance)
        μ(β) = vague(NormalMeanVariance)
        q(α) = vague(NormalMeanVariance)
        q(β) = vague(NormalMeanVariance)
        q(τ) = vague(GammaShapeRate)
        q(τ_g) = Gamma(2.0, 0.1)
        q(σ_α) = Gamma(2.0, 0.1)
        q(σ_β) = Gamma(2.0, 0.1)
        q(y_bio) = vague(NormalMeanVariance)
    end

    posterior = infer(
        model=regression_multi_protocol_robust_jzs_v2b(
            n_protocols=size(sample, 1),
            max_experiments=size(sample, 2),
            n_samples=size(sample, 3),
            intercept=intercept,
            intercept_sigma=intercept_sigma,
            nu_half=nu_half,
            τ_base=τ_base,
            jzs_r_scale=jzs_r_scale
        ),
        data=(data=sample, reference=reference, sigma_sq_imp_mask=sigma_sq_imp_mask),
        initialization=init_regression,
        constraints=constraints_regression,
        iterations=regression_iterations,
        returnvars=KeepLast()
    )

    prior = if !isnothing(cached_prior)
        cached_prior
    else
        # Prior fit uses the v2b prior model (shared σ instead of per-cell τ; missing data ⇒
        # no per-cell τ message). σ²_imp is zeroed for the prior fit (no imputation noise).
        prior_constraints = @constraints begin
            q(τ_g, μ_α, σ_α, μ_β, σ_β, σ) = q(τ_g)q(μ_α)q(σ_α)q(μ_β)q(σ_β)q(σ)
            q(μ_α, σ_α, α) = q(μ_α, α)q(σ_α)
            q(μ_β, σ_β, β) = q(μ_β, β)q(σ_β)
            q(predicted_value, σ) = q(predicted_value)q(σ)
            q(y_bio, σ) = q(y_bio)q(σ)
        end
        prior_init = @initialization begin
            μ(μ_α) = vague(NormalMeanVariance)
            μ(μ_β) = vague(NormalMeanVariance)
            μ(α) = vague(NormalMeanVariance)
            μ(β) = vague(NormalMeanVariance)
            q(α) = vague(NormalMeanVariance)
            q(β) = vague(NormalMeanVariance)
            q(σ) = Gamma(2.0, 0.1)
            q(τ_g) = Gamma(2.0, 0.1)
            q(σ_α) = Gamma(2.0, 0.1)
            q(σ_β) = Gamma(2.0, 0.1)
            q(y_bio) = vague(NormalMeanVariance)
        end
        missing_data = fill(missing, size(sample))
        zero_sigma_sq_imp = zeros(Float64, size(sample))
        infer(
            model=_regression_multi_protocol_jzs_prior_model_v2b(
                n_protocols=size(sample, 1),
                max_experiments=size(sample, 2),
                n_samples=size(sample, 3),
                intercept=intercept,
                intercept_sigma=intercept_sigma,
                jzs_r_scale=jzs_r_scale
            ),
            data=(data=missing_data, reference=missing_data, sigma_sq_imp_mask=zero_sigma_sq_imp),
            initialization=prior_init,
            constraints=prior_constraints,
            iterations=regression_iterations,
            returnvars=KeepLast()
        )
    end

    return RobustRegressionResultMultipleProtocols(posterior, prior, nu, τ_base)
end


"""
    RegressionModel_one_protocol_robust_jzs(data, idx, referenceID, intercept, intercept_sigma; nu, τ_base, jzs_r_scale, cached_prior, regression_iterations)

Robust regression with JZS prior (Cauchy via scale mixture) on α for a single protocol.
"""
function RegressionModel_one_protocol_robust_jzs(data::InteractionData, idx::Int64, referenceID::Int64, intercept::Float64, intercept_sigma::Float64; nu::Float64=5.0, τ_base::Float64=NaN, jzs_r_scale::Float64=0.354, cached_prior::Union{Nothing,InferenceResult}=nothing, regression_iterations::Int=150)
    @assert getNoProtocols(data) == 1 "Data must only contain one protocol"

    if isnan(τ_base)
        τ_base = estimate_regression_tau_base(data, referenceID)
    end

    data = prepare_regression_data(data, idx, referenceID)
    sample, reference = data.sample, data.reference
    nu_half = nu / 2.0
    sample, reference = sample[1, :, :], reference[1, :, :]

    @assert size(sample) == size(reference) "Mismatch in data dimensions"
    @assert ndims(sample) == 2 "Data must be 2-dimensional"

    init_regression = @initialization begin
        μ(β) = vague(NormalMeanPrecision)
        q(τ) = vague(GammaShapeRate)
        q(τ_g) = Gamma(2.0, 0.1)
    end

    posterior = infer(
        model=regression_one_protocol_robust_jzs(
            max_experiments=size(sample, 1),
            n_samples=size(sample, 2),
            intercept=intercept,
            intercept_sigma=intercept_sigma,
            nu_half=nu_half,
            τ_base=τ_base,
            jzs_r_scale=jzs_r_scale
        ),
        data=(data=sample, reference=reference),
        initialization=init_regression,
        constraints=MeanField(),
        iterations=regression_iterations,
        returnvars=KeepLast()
    )

    prior = if !isnothing(cached_prior)
        cached_prior
    else
        missing_data = fill(missing, size(sample))
        infer(
            model=regression_one_protocol_robust_jzs(
                max_experiments=size(sample, 1),
                n_samples=size(sample, 2),
                intercept=intercept,
                intercept_sigma=intercept_sigma,
                nu_half=nu_half,
                τ_base=τ_base,
                jzs_r_scale=jzs_r_scale
            ),
            data=(data=missing_data, reference=missing_data),
            initialization=init_regression,
            constraints=MeanField(),
            iterations=regression_iterations,
            returnvars=KeepLast()
        )
    end

    return RobustRegressionResultSingleProtocol(posterior, prior, nu, τ_base)
end


"""
    RegressionModel_one_protocol_robust_jzs_v2b(data, idx, referenceID, intercept, intercept_sigma;
                                                is_imputed, column_imputation_sigma_sq,
                                                raw_data = nothing, nu, τ_base, jzs_r_scale,
                                                cached_prior, regression_iterations)

Single-protocol mask-aware robust regression — **structured-VMP posterior**
(production fix). Default for the single-protocol JZS-robust
mask-aware path since the structured fix.

Models the (slope, intercept) coefficient block as a single joint 2-D MvNormal
node θ so RxInfer propagates the exact joint Gaussian covariance — fixing the
mean-field q(α)q(β) slope-variance collapse that pinned ~51 % of proteins at the
1e6 BF clamp (Spikes 016/017). Per-cell Gamma τ is RETAINED (Student-t
robustness); the MNAR σ²_imp down-weighting is folded into the per-cell τ prior
mean (imputed cell → Gamma centred at `1/(1/τ_base + σ²_imp)`), keeping the model
conjugate. The slope posterior prior is a scale-matched `Normal(0, jzs_r_scale²)`
(RxInfer cannot put the JZS Cauchy scale mixture on a joint MvNormal component);
the JZS Cauchy remains the BF's analytical null reference unchanged.

`cached_prior` is accepted for signature parity but unused (the BF's prior is the
analytical Cauchy, and the structured posterior has no separate prior-inference
pass). The legacy Path-B body is preserved as
`RegressionModel_one_protocol_robust_jzs_v2b_pathb` and reachable only by direct
call; `mask_aware_regression = false` remains the documented opt-out (to the
pre-v2b non-mask-aware path).
"""
function RegressionModel_one_protocol_robust_jzs_v2b(
    data::InteractionData, idx::Int64, referenceID::Int64,
    intercept::Float64, intercept_sigma::Float64;
    is_imputed::AbstractArray{Bool, 3},
    column_imputation_sigma_sq::Dict{Tuple{Int,Int,Int}, Float64},
    raw_data::Union{Nothing, InteractionData} = nothing,
    nu::Float64=5.0, τ_base::Float64=NaN, jzs_r_scale::Float64=0.354,
    cached_prior::Union{Nothing,InferenceResult}=nothing,
    regression_iterations::Int=150,
)
    @assert getNoProtocols(data) == 1 "Data must only contain one protocol"
    if isnan(τ_base)
        τ_base = estimate_regression_tau_base(data, referenceID)
    end
    prepared = prepare_regression_data(data, idx, referenceID; raw_data=raw_data)
    sample_3d, reference_3d = prepared.sample, prepared.reference
    @assert size(is_imputed) == size(sample_3d) "is_imputed mask shape != sample shape"

    nu_half = nu / 2.0
    slope_prior_var = jzs_r_scale^2          # scale-matched Normal slope prior (JZS Cauchy kept in BF)
    s2  = sample_3d[1, :, :]
    r2  = reference_3d[1, :, :]
    im2 = is_imputed[1, :, :]

    # Flatten kept (non-missing) cells → joint-MvNormal design rows + per-cell τ priors.
    y    = Float64[]
    Xrow = Vector{Float64}[]
    tpb  = Float64[]
    for I in CartesianIndices(im2)
        (ismissing(s2[I]) || ismissing(r2[I])) && continue
        push!(y, Float64(s2[I]))
        push!(Xrow, [Float64(r2[I]), 1.0])
        σ²_imp = im2[I] ? get(column_imputation_sigma_sq, (1, Tuple(I)...), 0.0) : 0.0
        push!(tpb, 1.0 / (1.0 / τ_base + σ²_imp))   # σ²_imp folded into the τ prior mean (conjugate)
    end
    n_obs = length(y)
    prior_mean = [0.0, intercept]
    prior_cov  = [slope_prior_var 0.0; 0.0 intercept_sigma]

    slope_posterior, intercept_posterior = if n_obs < 2
        # No identifiable slope (≤1 observed cell) → prior-dominated.
        (Normal(0.0, sqrt(slope_prior_var)), Normal(intercept, sqrt(intercept_sigma)))
    else
        init_structured = @initialization begin
            q(τ) = vague(GammaShapeRate)
            q(θ) = MvNormalMeanCovariance(prior_mean, prior_cov)
        end
        posterior = infer(
            model = regression_one_protocol_robust_structured(
                n_obs = n_obs, tau_prior_base = tpb, nu_half = nu_half,
                prior_mean = prior_mean, prior_cov = prior_cov),
            data = (y = y, Xrow = Xrow),
            constraints = regression_structured_constraints(),
            initialization = init_structured,
            iterations = regression_iterations,
            returnvars = KeepLast(),
        )
        θ_post = posterior.posteriors[:θ]
        m, C = mean(θ_post), cov(θ_post)
        (Normal(m[1], sqrt(C[1, 1])),    # slope marginal from the joint posterior
         Normal(m[2], sqrt(C[2, 2])))    # intercept marginal (read by diagnostics as :β)
    end

    # Downstream reads posterior.posteriors[:α] (BF / stats) and [:β] (predictive checks,
    # residuals). The joint θ posterior supplies both marginals.
    posterior_ir = InferenceResult(
        Dict(:α => slope_posterior, :β => intercept_posterior), nothing, nothing, nothing, nothing)
    prior_ir = InferenceResult(
        Dict(:α => Normal(0.0, sqrt(slope_prior_var)), :β => Normal(intercept, sqrt(intercept_sigma))),
        nothing, nothing, nothing, nothing)
    return RobustRegressionResultSingleProtocol(posterior_ir, prior_ir, nu, τ_base)
end

"""
    RegressionModel_one_protocol_robust_jzs_v2b_pathb(...)

LEGACY Path-B single-protocol v2b body — the per-cell
Gamma τ + latent `y_bio` chain with additive σ²_imp. Spikes 016/017 showed this
topology cannot reliably de-saturate `bf_correlation` on real data (the latent
node + mean-field decouple the slope variance; ~51 % saturated). PRESERVED for
reference / revert only; no longer wired into the default dispatch (superseded by
the structured-VMP `RegressionModel_one_protocol_robust_jzs_v2b` above).
"""
function RegressionModel_one_protocol_robust_jzs_v2b_pathb(
    data::InteractionData, idx::Int64, referenceID::Int64,
    intercept::Float64, intercept_sigma::Float64;
    is_imputed::AbstractArray{Bool, 3},
    column_imputation_sigma_sq::Dict{Tuple{Int,Int,Int}, Float64},
    raw_data::Union{Nothing, InteractionData} = nothing,
    nu::Float64=5.0, τ_base::Float64=NaN, jzs_r_scale::Float64=0.354,
    cached_prior::Union{Nothing,InferenceResult}=nothing,
    regression_iterations::Int=150,
)
    @assert getNoProtocols(data) == 1 "Data must only contain one protocol"

    if isnan(τ_base)
        τ_base = estimate_regression_tau_base(data, referenceID)
    end

    prepared = prepare_regression_data(data, idx, referenceID; raw_data=raw_data)
    sample_3d, reference_3d = prepared.sample, prepared.reference
    nu_half = nu / 2.0

    @assert size(is_imputed) == size(sample_3d) "is_imputed mask shape != sample shape"

    # Build 3-D σ²_imp first (uses (p, e, s) keys), then slice down to 2-D for the
    # single-protocol model body.
    sigma_sq_imp_mask_3d = zeros(Float64, size(sample_3d))
    for I in CartesianIndices(size(sample_3d))
        if is_imputed[I]
            p, e, s = Tuple(I)
            sigma_sq_imp_mask_3d[I] = get(column_imputation_sigma_sq, (p, e, s), 0.0)
        end
    end

    sample             = sample_3d[1, :, :]
    reference          = reference_3d[1, :, :]
    sigma_sq_imp_mask  = sigma_sq_imp_mask_3d[1, :, :]

    @assert size(sample) == size(reference) "Mismatch in data dimensions"
    @assert ndims(sample) == 2 "Data must be 2-dimensional"

    # Path-B init: production single-protocol block + `q(y_bio) = vague(NormalMeanVariance)`
    init_regression = @initialization begin
        μ(β) = vague(NormalMeanPrecision)
        q(τ) = vague(GammaShapeRate)
        q(τ_g) = Gamma(2.0, 0.1)
        q(y_bio) = vague(NormalMeanVariance)
    end

    posterior = infer(
        model=regression_one_protocol_robust_jzs_v2b(
            max_experiments=size(sample, 1),
            n_samples=size(sample, 2),
            intercept=intercept,
            intercept_sigma=intercept_sigma,
            nu_half=nu_half,
            τ_base=τ_base,
            jzs_r_scale=jzs_r_scale
        ),
        data=(data=sample, reference=reference, sigma_sq_imp_mask=sigma_sq_imp_mask),
        initialization=init_regression,
        constraints=MeanField(),
        iterations=regression_iterations,
        returnvars=KeepLast()
    )

    prior = if !isnothing(cached_prior)
        cached_prior
    else
        missing_data = fill(missing, size(sample))
        zero_sigma_sq_imp = zeros(Float64, size(sample))
        infer(
            model=regression_one_protocol_robust_jzs_v2b(
                max_experiments=size(sample, 1),
                n_samples=size(sample, 2),
                intercept=intercept,
                intercept_sigma=intercept_sigma,
                nu_half=nu_half,
                τ_base=τ_base,
                jzs_r_scale=jzs_r_scale
            ),
            data=(data=missing_data, reference=missing_data, sigma_sq_imp_mask=zero_sigma_sq_imp),
            initialization=init_regression,
            constraints=MeanField(),
            iterations=regression_iterations,
            returnvars=KeepLast()
        )
    end

    return RobustRegressionResultSingleProtocol(posterior, prior, nu, τ_base)
end


"""
    prepare_regression_data(data::InteractionData, idx, referenceID;
                            raw_data::Union{Nothing, InteractionData} = nothing)
        -> (sample, reference, is_imputed)

Construct the per-protein regression-input NamedTuple.

`sample` and `reference` are 3-D arrays of shape `(n_protocols, max_experiments, n_samples)`
formed by concatenating sample and control sides of `data`.

`is_imputed` is an `Array{Bool, 3}` of the same shape. When `raw_data === nothing`
(default), all entries are `false` — backward-compatible behaviour. When
`raw_data` is supplied, an entry is `true` iff the corresponding cell was missing
in `raw_data.sample` OR `raw_data.reference` (cell-wise OR over the protein-of-
interest at `idx` and the reference protein at `referenceID`).

This mask is consumed by v2b mask-aware regression to source the
per-cell additive imputation variance σ²_imp[cell]; for non-imputed cells the
v2b observation factor degenerates to the pre-spike `precision=τ[cell]` form.
"""
function prepare_regression_data(
    data::InteractionData,
    idx::I,
    referenceID::I;
    raw_data::Union{Nothing, InteractionData} = nothing,
) where {I<:Integer}
    protein = getProteinData(data, idx)
    interactome_sample = getSampleMatrix(protein)
    interactome_control = getControlMatrix(protein)
    sample = cat(interactome_sample, interactome_control, dims=2)

    RefProtein = getProteinData(data, referenceID)
    reference_sample = getSampleMatrix(RefProtein)
    reference_control = getControlMatrix(RefProtein)
    reference = cat(reference_sample, reference_control, dims=2)

    # is_imputed mask (true iff raw data was missing at that cell)
    if raw_data === nothing
        is_imputed = falses(size(sample))
    else
        raw_protein = getProteinData(raw_data, idx)
        raw_ref     = getProteinData(raw_data, referenceID)
        raw_sample_full = cat(
            getSampleMatrix(raw_protein),
            getControlMatrix(raw_protein),
            dims=2,
        )
        raw_reference_full = cat(
            getSampleMatrix(raw_ref),
            getControlMatrix(raw_ref),
            dims=2,
        )
        @assert size(raw_sample_full) == size(sample) "raw_data shape != imputed shape"
        is_imputed = [
            ismissing(raw_sample_full[i]) || ismissing(raw_reference_full[i])
            for i in eachindex(sample)
        ]
        is_imputed = reshape(is_imputed, size(sample))
    end

    return (sample=sample, reference=reference, is_imputed=is_imputed)
end

"""
    _pooled_regression_residual_sd(data::InteractionData, refID::Int; n_sample::Int=100) -> (sd, v, n_resid)

Shared pooled-OLS-residual machinery. Samples up to `n_sample` proteins,
performs OLS regression (y = β + α*reference) for each, collects all residuals, and
returns a tuple `(sqrt(v), v, n_resid)` where `v = var(pooled_residuals)` and `n_resid`
is the pooled residual count.

This is the SINGLE source of the pooled-residual computation. BOTH the Empirical-Bayes
`estimate_regression_tau_base` (returns `1/v`) AND the `detect_protocol_scale_mismatch`
boolean detector (compares `sqrt(v)` to a threshold) call it — do NOT duplicate the loop.

When fewer than 2 pooled residuals are available, returns `(0.0, 0.0, n_resid)` so callers
can apply their own fallback (`estimate_regression_tau_base` → `1.0`; the detector → `false`).
"""
function _pooled_regression_residual_sd(data::InteractionData, refID::Int; n_sample::Int=100)
    n_proteins = length(getIDs(data))
    sample_indices = n_proteins <= n_sample ? collect(1:n_proteins) : randperm(n_proteins)[1:n_sample]

    pooled_residuals = Float64[]

    for idx in sample_indices
        idx == refID && continue

        reg_data = prepare_regression_data(data, idx, refID)

        # Flatten non-missing paired observations
        y_flat = Float64[]
        x_flat = Float64[]
        for i in eachindex(reg_data.sample)
            y = reg_data.sample[i]
            x = reg_data.reference[i]
            if !ismissing(y) && !ismissing(x)
                push!(y_flat, y)
                push!(x_flat, x)
            end
        end

        length(y_flat) < 3 && continue

        # OLS: y = β + α*x
        n = length(y_flat)
        x_mean = mean(x_flat)
        y_mean = mean(y_flat)

        ss_xx = sum((xi - x_mean)^2 for xi in x_flat)
        ss_xx < 1e-15 && continue

        ss_xy = sum((x_flat[i] - x_mean) * (y_flat[i] - y_mean) for i in 1:n)
        α_ols = ss_xy / ss_xx
        β_ols = y_mean - α_ols * x_mean

        for i in 1:n
            push!(pooled_residuals, y_flat[i] - (β_ols + α_ols * x_flat[i]))
        end
    end

    length(pooled_residuals) < 2 && return (0.0, 0.0, length(pooled_residuals))  # caller-applied fallback
    v = var(pooled_residuals)
    return (sqrt(v), v, length(pooled_residuals))
end

"""
    estimate_regression_tau_base(data::InteractionData, refID::Int; n_sample::Int=100)

Estimate a data-driven residual precision τ_base via pooled OLS residuals.

Samples up to `n_sample` proteins, performs OLS regression (y = β + α*reference)
for each, collects all residuals, and returns `1 / var(pooled_residuals)`.

This is used as the Empirical Bayes constant in the robust regression model,
replacing the latent global precision σ to ensure VMP compatibility.
"""
function estimate_regression_tau_base(data::InteractionData, refID::Int; n_sample::Int=100)
    # Shared pooled-residual machinery (factored into _pooled_regression_residual_sd
    # so detect_protocol_scale_mismatch reuses the SAME computation).
    _sd, v, n_resid = _pooled_regression_residual_sd(data, refID; n_sample=n_sample)

    n_resid < 2 && return 1.0  # fallback

    # Scale-mismatch guard: on multi-protocol data, an implausibly large
    # pooled regression residual SD signals un-normalised cross-protocol / cross-experiment
    # intensity baselines. These inject a spurious bait-correlation that inflates the
    # regression slope MEAN and saturates bf_correlation (NOT the single-protocol
    # mean-field overconfidence — a distinct, data-normalisation axis). Healthy
    # single-protocol residual SD ≈ 1.3 log2; the wtHAP40 GST+Strep mismatch gave ≈ 6.0.
    # Threshold 2.5 log2 fires on the mismatch, not on well-scaled data. `normalise_protocols=true`
    # removes it (verified to leave HBM log2FC invariant + de-saturate the regression).
    if getNoProtocols(data) > 1 && sqrt(v) > 2.5
        @warn "Multi-protocol regression: pooled residual SD ≈ $(round(sqrt(v), digits=2)) " *
              "log2 units is implausibly large (healthy ≈ 1.3). This usually means the " *
              "protocols/experiments are on un-normalised intensity baselines, which inflate " *
              "regression slopes and saturate bf_correlation. Consider reloading " *
              "with `normalise_protocols=true` — it de-saturates the regression and leaves the " *
              "HBM log2FC invariant." maxlog=1
    end

    return v > 0.0 ? 1.0 / v : 1.0
end


"""
    detect_protocol_scale_mismatch(data::InteractionData; refID::Int=1,
                                   threshold::Float64=2.5, n_sample::Int=100) -> Bool

Boolean detector for a multi-protocol cross-protocol intensity scale mismatch.
Reuses the SAME pooled-OLS-residual machinery
(`_pooled_regression_residual_sd`) as the scale-mismatch guard in
`estimate_regression_tau_base` — when un-normalised protocols sit on different
intensity baselines the pooled residual SD inflates well above the healthy
single-protocol level (≈ 1.3 log2; the wtHAP40 GST+Strep mismatch gave ≈ 6.0).

Returns `getNoProtocols(data) > 1 && sqrt(v) > threshold`:
- Single-protocol data (`getNoProtocols(data) == 1`) → always `false` (no
  cross-protocol offset can exist; the structured-VMP fix already de-saturates
  single-protocol `bf_correlation`).
- Matched-level multi-protocol data (e.g. real HAP40 GST/Strep ~30 vs ~30.1
  log2) → `false` (pooled residual SD stays below `threshold`; no over-correction).
- Scale-disparate multi-protocol data (protocols on offset baselines) → `true`.

Consumed by `load_data`'s `:auto` resolution: a `true` result flips the effective
normalisation method to `:both` (median_of_ratios + row-centering = arm F),
a `false` result leaves it `:none`. The `threshold` (2.5 log2) is the
calibrated value — it fires on the mismatch, not on well-scaled data.
"""
function detect_protocol_scale_mismatch(data::InteractionData; refID::Int=1,
                                        threshold::Float64=2.5, n_sample::Int=100)::Bool
    getNoProtocols(data) > 1 || return false   # single-protocol → never a cross-protocol mismatch
    sd, _v, n_resid = _pooled_regression_residual_sd(data, refID; n_sample=n_sample)
    n_resid < 2 && return false                # too few residuals to judge → conservative no-flip
    return sd > threshold
end


"""
    estimate_per_protein_tau_base(data, idx, refID; global_tau_base, min_obs=5)

Estimate a per-protein residual precision τ_base from OLS residuals, with shrinkage
toward the global estimate for proteins with few observations.

For proteins with `n_obs >= min_obs`, uses the protein's own OLS residual variance.
For proteins with fewer observations, shrinks toward `global_tau_base` using a
weighted average: `w * local + (1-w) * global`, where `w = n_obs / min_obs`.

# Returns
- `Float64`: Per-protein τ_base (precision = 1/variance).
"""
function estimate_per_protein_tau_base(data::InteractionData, idx::Int, refID::Int;
                                       global_tau_base::Float64, min_obs::Int=5)
    idx == refID && return global_tau_base

    reg_data = prepare_regression_data(data, idx, refID)

    # Flatten non-missing paired observations
    y_flat = Float64[]
    x_flat = Float64[]
    for i in eachindex(reg_data.sample)
        y = reg_data.sample[i]
        x = reg_data.reference[i]
        if !ismissing(y) && !ismissing(x)
            push!(y_flat, y)
            push!(x_flat, x)
        end
    end

    n_obs = length(y_flat)

    # Too few observations for any local estimate
    n_obs < 3 && return global_tau_base

    # OLS: y = β + α*x
    x_mean = mean(x_flat)
    y_mean = mean(y_flat)
    ss_xx = sum((xi - x_mean)^2 for xi in x_flat)

    # No variance in predictor — can't fit regression
    if ss_xx < 1e-15
        return global_tau_base
    end

    ss_xy = sum((x_flat[i] - x_mean) * (y_flat[i] - y_mean) for i in 1:n_obs)
    α_ols = ss_xy / ss_xx
    β_ols = y_mean - α_ols * x_mean

    residuals = [y_flat[i] - (β_ols + α_ols * x_flat[i]) for i in 1:n_obs]
    v = var(residuals)
    local_tau = v > 0.0 ? 1.0 / v : global_tau_base

    # Shrinkage: full weight to local only when n_obs >= min_obs
    if n_obs >= min_obs
        return local_tau
    else
        w = n_obs / min_obs
        # Shrink on log scale for better behavior with precision values
        return exp(w * log(local_tau) + (1 - w) * log(global_tau_base))
    end
end

function RegressionModel_one_protocol(data::InteractionData, idx::Int64, referenceID::Int64, intercept::Float64, intercept_sigma::Float64; cached_prior::Union{Nothing,InferenceResult}=nothing, regression_iterations::Int=75)
    @assert getNoProtocols(data) == 1 "Data must only contain one protocol"
    # ------------------ data preperation -----------------------------------#    
    # load data
    data = prepare_regression_data(data, idx, referenceID)
    sample, reference = data.sample, data.reference

    # remove protocol dimension
    sample, reference = sample[1, :, :], reference[1, :, :]

    # check conditions
    @assert size(sample) == size(reference) "Mismatch in data dimensions"
    @assert ndims(sample) == 2 "Data must be 2-dimensional where the 1st dimension is the number of experiments, the 2nd dimension is the number of samples"


    # ------------------ fit posterior and prior -----------------------------------#
    init_regression = @initialization begin
        μ(β) = vague(NormalMeanPrecision)
        q(σ) = vague(GammaShapeRate)
    end

    posterior = infer(
        model=regression_one_protocol(
            max_experiments=size(sample, 1),
            n_samples=size(sample, 2),
            intercept=intercept,
            intercept_sigma=intercept_sigma
        ),
        data=(data=sample, reference=reference),
        initialization=init_regression,
        constraints=MeanField(),
        iterations=regression_iterations,
        returnvars=KeepLast()
    )

    # Use cached prior if provided, otherwise compute it
    prior = if !isnothing(cached_prior)
        cached_prior
    else
        missing_arr = fill(missing, size(sample))
        infer(
            model=regression_one_protocol(
                max_experiments=size(sample, 1),
                n_samples=size(sample, 2),
                intercept=intercept,
                intercept_sigma=intercept_sigma
            ),
            data=(data=missing_arr, reference=missing_arr),
            initialization=init_regression,
            constraints=MeanField(),
            iterations=regression_iterations,
            returnvars=KeepLast()
        )
    end

    return RegressionResultSingleProtocol(posterior, prior)
end

"""
    RegressionModel(data::InteractionData, idx::Int64, referenceID::Int64, intercept::Float64, intercept_sigma::Float64; cached_prior::Union{Nothing,InferenceResult}=nothing)

    Computes the regression model for a given protein and reference protein.
    CAVE: This function should be used if the number of protocols is bigger than 1.

    Args:
        - data::InteractionData: The interaction data
        - idx::Int64: The index of the protein to be analyzed
        - referenceID::Int64: The reference ID
        - intercept::Float64: The global intercept
        - intercept_sigma::Float64: The global intercept sigma
        - cached_prior::Union{Nothing,InferenceResult}=nothing: Optional precomputed prior to reuse across proteins

    Returns:
        - posterior: The posterior distribution of the regression model
        - prior: The prior distribution of the regression model

"""
function RegressionModel(data::InteractionData, idx::Int64, referenceID::Int64, intercept::Float64, intercept_sigma::Float64; cached_prior::Union{Nothing,InferenceResult}=nothing, regression_iterations::Int=75)
    data = prepare_regression_data(data, idx, referenceID)
    sample, reference = data.sample, data.reference

    @assert size(sample) == size(reference) "Mismatch in data dimensions"
    @assert ndims(sample) == 3 "Data must be 3-dimensional where the 1st dimension is the number of protocols, the 2nd dimension is the number of experiments and the 3rd dimension is the number of samples"

    constraints_regression = @constraints begin
        # Assume that `μ_α`, `σ_α`, `μ_β`, `σ_β` and `σ` are jointly independent
        q(μ_α, σ_α, μ_β, σ_β, σ) = q(μ_α)q(σ_α)q(μ_β)q(σ_β)q(σ)
        # Assume that `μ_α`, `σ_α`, `α` are jointly independent
        q(μ_α, σ_α, α) = q(μ_α, α)q(σ_α)
        # Assume that `μ_β`, `σ_β`, `β` are jointly independent
        q(μ_β, σ_β, β) = q(μ_β, β)q(σ_β)
        # Assume that 'predicted_value' and 'σ' are jointly independent
        q(predicted_value, σ) = q(predicted_value)q(σ)
    end

    init_regression = @initialization begin
        μ(μ_α) = vague(NormalMeanVariance)
        μ(μ_β) = vague(NormalMeanVariance)
        μ(α) = vague(NormalMeanVariance)
        μ(β) = vague(NormalMeanVariance)
        q(α) = vague(NormalMeanVariance)
        q(β) = vague(NormalMeanVariance)
        q(σ) = Gamma(2.0, 0.1)
        q(σ_α) = Gamma(2.0, 0.1)
        q(σ_β) = Gamma(2.0, 0.1)
    end

    posterior = infer(
        model=regression_multi_protocol(
            n_protocols=size(sample, 1),
            max_experiments=size(sample, 2),
            n_samples=size(sample, 3),
            intercept=intercept,
            intercept_sigma=intercept_sigma
        ),
        data=(data=sample, reference=reference),
        initialization=init_regression,
        constraints=constraints_regression,
        iterations=regression_iterations,
        returnvars=KeepLast()
    )

    # Use cached prior if provided, otherwise compute it
    prior = if !isnothing(cached_prior)
        cached_prior
    else
        missing_arr = fill(missing, size(sample))
        infer(
            model=regression_multi_protocol(
                n_protocols=size(sample, 1),
                max_experiments=size(sample, 2),
                n_samples=size(sample, 3),
                intercept=intercept,
                intercept_sigma=intercept_sigma
            ),
            data=(data=missing_arr, reference=missing_arr),
            initialization=init_regression,
            constraints=constraints_regression,
            iterations=regression_iterations,
            returnvars=KeepLast()
        )
    end

    return RegressionResultMultipleProtocols(posterior, prior)
end

"""
    initiate_folders(base_path::String = "./data")
    Function to create folders for storing result plots. The base path is set to "./data" by default and needs to have no trailing "/". 
"""
function initiate_folders(base_path::String="./data")
    !isdir("$(base_path)/HBM_dists") && mkpath("./data/HBM_dists")
    !isdir("$(base_path)/data/log2FC") && mkpath("./data/log2FC")
    !isdir("$(base_path)/data/regression") && mkpath("./data/regression")
    !isdir("$(base_path)/data/rangeplot") && mkpath("./data/rangeplot")
    return nothing
end

"""
    compute_log2FC(data::InteractionData, idx::Int64)

    Computes the log2FC for a given protein. Returns missing if no data is available

    Args:
        - data::InteractionData: The interaction data
        - idx::Int64: The index of the protein to be analyzed

    Returns:
        - log2FC: The log2FC for the protein at idx 
"""
function compute_log2FC(data::InteractionData, idx::Int64)
    protein = getProteinData(data, idx)
    interactome_sample = getSampleMatrix(protein)
    interactome_control = getControlMatrix(protein)
    mean_interactome_sample = zeros(Float64, size(interactome_sample)[1:2])
    mean_interactome_control = zeros(Float64, size(interactome_control)[1:2])

    for (protocol, experiment) ∈ Iterators.product(axes(interactome_sample, 1), axes(interactome_sample, 2))
        mean_interactome_sample[protocol, experiment] = mean(skipmissing(interactome_sample[protocol, experiment, :]))
        mean_interactome_control[protocol, experiment] = mean(skipmissing(interactome_control[protocol, experiment, :]))
    end

    log2FC = mean_interactome_sample .- mean_interactome_control
    log2FC_complete = Vector{Union{Float64,Missing}}(undef, 1 + size(log2FC, 1) + size(log2FC, 1) * size(log2FC, 2))
    log2FC_complete[1] = mean(log2FC[isnan.(log2FC).==false])
    position = 2

    for protocol ∈ axes(log2FC, 1)
        log2FC_protocol = log2FC[protocol, :]
        log2FC_complete[position] = mean(skipmissing(log2FC_protocol[isnan.(log2FC_protocol).==false]))
        for experiment ∈ axes(log2FC, 2)
            isnan(log2FC[protocol, experiment]) ? log2FC_complete[position] = missing : log2FC_complete[position] = log2FC[protocol, experiment]
            position += 1
        end
    end
    return log2FC_complete
end

"""
    regression(
    data::InteractionData, idx::I, referenceID::I, α::F, intercept::Float64, 
    intercept_sigma::Float64, plotregr::Bool, protein_name::S; 
    verbose::Bool = true
    ) where {I<:Integer, F<:AbstractFloat, S<:String}

    Computes the regression for a given protein and reference protein.

    Args:
        - data::InteractionData: The interaction data
        - idx::Int64: The index of the protein to be analyzed
        - referenceID::Int64: The reference ID
        - α::F: The significance level
        - intercept::Float64: The global intercept
        - intercept_sigma::Float64: The global intercept sigma
        - plotregr::Bool: Whether to plot the regression
        - protein_name::String: The name of the protein

    Keyword Args:
        - verbose::Bool: Whether log messages should be produced (defaults to true) 

    Returns:
        - posterior: The posterior distribution of the regression model
        - prior: The prior distribution of the regression model
        - regression_stats: The regression statistics
        - bfRegression: The Bayes factor of the regression model

"""
# _build_column_imputation_sigma_sq(dropout_fit, intensity_matrices_by_protocol) ->
#     Dict{Tuple{Int,Int,Int}, Float64}
#
# Construct a per-cell σ²_imp lookup keyed by (protocol, experiment, sample) Cartesian
# coordinates of the post-imputation regression input. `intensity_matrices_by_protocol` is a
# Vector{Matrix} whose element `p` is the post-imputation intensity matrix for protocol `p`,
# shape (n_proteins, n_columns_for_protocol). The column index in
# `column_imputation_sigma(fit, col, matrix)` matches the column index in those matrices
# (i.e. the source-matrix column order used by `prepare_regression_data` after concatenating
# sample + control sides).
#
# When `dropout_fit === nothing` or `intensity_matrices_by_protocol === nothing` (e.g.
# `imputation = :none`) returns an empty Dict — the v2b wrapper then sees σ²_imp = 0 at every
# cell and the variance-additive observation factor collapses algebraically to the legacy
# `precision = τ[cell]` form. Cells outside the dropout fit's column range are silently left
# out of the dict (treated as σ²_imp = 0).
#
# Used by `regression()` when `mask_aware_regression = true`.
function _build_column_imputation_sigma_sq(
    dropout_fit::Union{Nothing, DropoutFit},
    intensity_matrices_by_protocol::Union{Nothing, Vector},
)::Dict{Tuple{Int, Int, Int}, Float64}
    lookup = Dict{Tuple{Int, Int, Int}, Float64}()
    if dropout_fit === nothing || intensity_matrices_by_protocol === nothing
        return lookup
    end
    n_protocols = length(intensity_matrices_by_protocol)
    for p in 1:n_protocols
        matrix = intensity_matrices_by_protocol[p]
        matrix === nothing && continue
        n_cols = size(matrix, 2)
        for col in 1:n_cols
            σ = try
                column_imputation_sigma(dropout_fit, col, matrix)
            catch err
                # BoundsError or method-resolution issue → treat as σ²_imp = 0
                # (helper returns 0.0 when col is out of range)
                err isa BoundsError ? 0.0 : rethrow(err)
            end
            σ_sq = σ * σ
            # (col-in-matrix) ↔ (experiment, sample) mapping in `prepare_regression_data`:
            # after `cat(getSampleMatrix, getControlMatrix, dims=2)` the column index encodes
            # (experiment, sample). Without the precise mapping (depends on protocol shape)
            # we fan the same per-column σ²_imp across all (e, s) coordinates for the protocol
            # — appropriate because `column_imputation_sigma` is itself a per-column statistic.
            # leaves the fine-grained (e, s) → source-column mapping to
            # integration work; for the algebraic-collapse opt-out contract this fan-out is
            # immaterial (mask is all-false → σ²_imp is never read).
            # Note: the column index in the regression input matrix corresponds 1:1 to a
            # CartesianIndex over (e, s); the wrapper indexes `column_imputation_sigma_sq`
            # with `(p, e, s)`, so we register every `(p, e, s)` with this σ²_imp value here.
            # Callers that need a finer per-cell lookup pre-build the dict themselves and
            # pass it in via `regression(..., column_imputation_sigma_sq=...)`.
            # This default builder is a "best-available" pre-population; absent a richer
            # column→(e, s) map it cannot be more precise.
            # Subscript pattern: only set if not already present (preserves caller overrides).
            # NOTE: this loop is bounded by `n_cols`, not by experiment / sample dims, so we
            # set the (col index)-th coordinate as a 1-D coordinate; the wrapper reads it via
            # `get(column_imputation_sigma_sq, (p, e, s), 0.0)`. We register a key for every
            # plausible (e, s) by treating `col` as a linear sample index — refines.
            lookup[(p, 1, col)] = σ_sq
            lookup[(p, col, 1)] = σ_sq
        end
    end
    return lookup
end

# _build_column_imputation_sigma_sq_from_data(data::InteractionData) ->
#     Dict{Tuple{Int,Int,Int}, Float64}
#
# integration completion (wiring). Builds the per-replicate-column
# σ²_imp lookup DIRECTLY from a (post-imputation) `InteractionData`, keyed by the exact
# (protocol, experiment, sample) CartesianIndex coordinates that `prepare_regression_data`
# produces. It reuses the IDENTICAL `cat(getSampleMatrix(protein), getControlMatrix(protein),
# dims=2)` construction per protein, so the dict keys align with the v2b wrapper's `Tuple(I)`
# coordinates BY CONSTRUCTION — no fragile (e,s)→source-xlsx-column→dropout-fit-index mapping
# is required (that mapping was the piece deferred-and-never-built by the original
# `_build_column_imputation_sigma_sq` stub above, which is why σ²_imp was empty in production).
#
# σ² for each (p,e,s) is `var(finite values in that replicate column across all proteins)`,
# matching the `column_imputation_sigma` definition (`sqrt(var(finite))`). The `dropout_fit`
# is not needed here — `column_imputation_sigma` only used it for a bounds check; the actual
# statistic is the empirical column variance, which we read straight from the imputed matrices.
#
# Called once per analysis (across all proteins). Returns an empty Dict only if the data has
# no proteins. Used by the multi-imputation `analyse(imputed_data, raw_data, ...)` path; the
# per-protein `is_imputed` mask (from `prepare_regression_data(...; raw_data=...)`) gates which
# cells actually receive the additive variance, so columns with no imputed cells are unaffected.
function _build_column_imputation_sigma_sq_from_data(
    data::InteractionData,
)::Dict{Tuple{Int, Int, Int}, Float64}
    n_proteins = length(getIDs(data))
    acc = Dict{Tuple{Int, Int, Int}, Vector{Float64}}()
    for idx in 1:n_proteins
        protein = getProteinData(data, idx)
        m = cat(getSampleMatrix(protein), getControlMatrix(protein), dims=2)
        for I in CartesianIndices(size(m))
            v = m[I]
            ismissing(v) && continue
            vf = Float64(v)
            isnan(vf) && continue
            push!(get!(() -> Float64[], acc, Tuple(I)), vf)
        end
    end
    lookup = Dict{Tuple{Int, Int, Int}, Float64}()
    for (k, vals) in acc
        lookup[k] = length(vals) >= 2 ? var(vals) : 0.0
    end
    return lookup
end

function regression(
    data::InteractionData, idx::I, referenceID::I, α::F, intercept::Float64,
    intercept_sigma::Float64, plotregr::Bool, protein_name::S;
    verbose::Bool=true, cached_regression_prior::Union{Nothing,InferenceResult}=nothing,
    regression_likelihood::Symbol=:normal, student_t_nu::Float64=5.0,
    robust_tau_base::Float64=NaN,
    regression_iterations::Union{Int,Nothing}=nothing,
    regression_bf_threshold::Float64=0.1,
    jzs_r_scale::Float64=0.0,
    global_tau_base::Float64=NaN,
    regression_min_posterior_var::Float64=0.0,
    # NEW: mask-aware regression dispatch
    mask_aware_regression::Bool=true,
    raw_data::Union{Nothing, InteractionData}=nothing,
    dropout_fit::Union{Nothing, DropoutFit}=nothing,
    intensity_matrices_by_protocol::Union{Nothing, Vector}=nothing,
    column_imputation_sigma_sq::Union{Nothing, Dict{Tuple{Int,Int,Int}, Float64}}=nothing,
) where {I<:Integer,F<:AbstractFloat,S<:String}

    # Per-protein τ_base when global estimate is available
    protein_tau_base = if !isnan(global_tau_base) && regression_likelihood == :robust_t
        estimate_per_protein_tau_base(data, idx, referenceID; global_tau_base=global_tau_base)
    else
        robust_tau_base
    end

    # define regression type based on the number of protocols, likelihood, and prior.
    # The mask-aware v2b dispatch ONLY applies to the JZS-robust path (mask_aware_regression
    # = true AND regression_likelihood = :robust_t AND jzs_r_scale > 0.0). All other
    # likelihood/prior combinations route to their legacy wrappers verbatim. This preserves
    # the byte-identical contract on opt-out.
    use_v2b = mask_aware_regression && regression_likelihood == :robust_t && jzs_r_scale > 0.0
    if regression_likelihood == :robust_t
        if jzs_r_scale > 0.0
            if use_v2b
                regression_fun = getNoProtocols(data) == 1 ?
                    RegressionModel_one_protocol_robust_jzs_v2b :
                    RegressionModel_multi_protocol_robust_jzs_v2b
            else
                regression_fun = getNoProtocols(data) == 1 ? RegressionModel_one_protocol_robust_jzs : RegressionModelRobustJZS
            end
        else
            regression_fun = getNoProtocols(data) == 1 ? RegressionModel_one_protocol_robust : RegressionModelRobust
        end
    else
        regression_fun = getNoProtocols(data) == 1 ? RegressionModel_one_protocol : RegressionModel
    end

    iter_kwarg = isnothing(regression_iterations) ? (;) : (; regression_iterations)
    try
        if regression_likelihood == :robust_t
            jzs_kwarg = jzs_r_scale > 0.0 ? (; jzs_r_scale) : (;)
            if use_v2b
                # Build per-cell σ²_imp lookup ONCE per regression call (caller may supply pre-built).
                σ_sq_lookup = if column_imputation_sigma_sq !== nothing
                    column_imputation_sigma_sq
                else
                    _build_column_imputation_sigma_sq(dropout_fit, intensity_matrices_by_protocol)
                end
                # Construct is_imputed via prepare_regression_data so v2b wrappers receive
                # the right shape; raw_data === nothing → all-false mask → algebraic collapse.
                prepared = prepare_regression_data(data, idx, referenceID; raw_data=raw_data)
                is_imputed_mask = prepared.is_imputed
                v2b_kwarg = (
                    is_imputed = is_imputed_mask,
                    column_imputation_sigma_sq = σ_sq_lookup,
                    raw_data = raw_data,
                )
                regression_result = regression_fun(data, idx, referenceID, intercept, intercept_sigma; nu=student_t_nu, τ_base=protein_tau_base, cached_prior=cached_regression_prior, iter_kwarg..., jzs_kwarg..., v2b_kwarg...)
            else
                regression_result = regression_fun(data, idx, referenceID, intercept, intercept_sigma; nu=student_t_nu, τ_base=protein_tau_base, cached_prior=cached_regression_prior, iter_kwarg..., jzs_kwarg...)
            end
        else
            regression_result = regression_fun(data, idx, referenceID, intercept, intercept_sigma; cached_prior=cached_regression_prior, iter_kwarg...)
        end
        regression_stats = RegressionStatistics(regression_result; α=α)

        protein = getProteinData(data, idx)
        reference_protein = getProteinData(data, referenceID)

        bfRegression, _, _ = BayesFactorRegression(regression_result; threshold=regression_bf_threshold, jzs_r_scale=jzs_r_scale, min_posterior_var=regression_min_posterior_var)

        if plotregr
            y = cat(getSampleMatrix(protein), getControlMatrix(protein), dims=2)
            x = cat(getSampleMatrix(reference_protein), getControlMatrix(reference_protein), dims=2)

            plot_regression(
                regression_result, protein_name, x, y,
                file="data/regression/$(protein_name)_regression.png"
            )
        end

        return regression_result, regression_stats, bfRegression
    catch e
        verbose && @warn "Regression failed for $idx: $protein_name: $e"
        return nothing, nothing, nothing
    end
end

"""
    enrichment(data::InteractionData, idx::Int64; kwargs...)

Wrapper that auto-selects HBM or HBM_single_protocol based on protocol count.
Mirrors the regression() pattern for consistent dispatch.
"""
function enrichment(data::InteractionData, idx::Int64; μ_0::F=25.0, σ_0::F=1.0, a_0::F=1.0, b_0::F=1.0, cached_prior::Union{Nothing,InferenceResult}=nothing, hbm_iterations::Int=75) where {F<:AbstractFloat}
    if getNoProtocols(data) == 1
        return HBM_single_protocol(data, idx; μ_0=μ_0, σ_0=σ_0, a_0=a_0, b_0=b_0, cached_prior=cached_prior, hbm_iterations=hbm_iterations)
    else
        return HBM(data, idx; μ_0=μ_0, σ_0=σ_0, a_0=a_0, b_0=b_0, cached_prior=cached_prior, hbm_iterations=hbm_iterations)
    end
end

"""
    precompute_enrichment_prior(data::InteractionData; kwargs...)

Wrapper that auto-selects HBM or HBM_single_protocol prior precomputation.
"""
function precompute_enrichment_prior(data::InteractionData; μ_0::F=25.0, σ_0::F=1.0, a_0::F=1.0, b_0::F=1.0) where {F<:AbstractFloat}
    if getNoProtocols(data) == 1
        return precompute_HBM_single_protocol_prior(data; μ_0=μ_0, σ_0=σ_0, a_0=a_0, b_0=b_0)
    else
        return precompute_HBM_prior(data; μ_0=μ_0, σ_0=σ_0, a_0=a_0, b_0=b_0)
    end
end


"""
    main(
    data::InteractionData, idx::I, referenceID::I;
    μ_0::Union{F, Nothing} = nothing, σ_0::Union{F, Nothing} = nothing,
    a_0::Union{F, Nothing} = nothing, b_0::Union{F, Nothing} = nothing,
    α::F = 0.95, csv_file = "data/results.csv",
    plotHBMdists::Bool = true, plotlog2fc::Bool = true, plotregr::Bool = true,
    plotbayesrange::Bool = true, writecsv::Bool = true,
    verbose::Bool = true, computeHBM::Bool = true
    ) where {I <:Integer, F<: AbstractFloat}


    Main function to run the analysis

    This function runs the analysis of a single protein (HierarchicalBayesianModel for log2FC and Regression).

    Args:
        - data::InteractionData: The interaction data
        - idx::Int64: The index of the protein to be analyzed
        - referenceID::Int64: The reference ID

    Keyword Args:
        - threshold::Float64: The threshold
        - μ_0::Float64: The mean hyperparameter (if not provided, value will be calculated using μ0)
        - σ_0::Float64: The standard deviation hyperparameter (if not provided, value will be calculated using μ0)
        - a_0::Float64: The shape hyperparameter for the inverse gamma distribution (if not provided, value will be calculated using σ0)
        - b_0::Float64: The rate hyperparameter for the inverse gamma distribution (if not provided, value will be calculated using σ0)
        - α::Float64: The significance level
        - csv_file::String: The name of the csv file
        - plot_HBM_dists::Bool: Whether to plot the HBM distributions
        - plot_log2fc::Bool: Whether to plot the log2FC
        - plot_regr::Bool: Whether to plot the regression
        - plot_bayesrange::Bool: Whether to plot range plots for the Bayes Factor and posterior probability
        - writecsv::Bool: Whether to write the results to a csv file
        - verbose::Bool: Whether log messages should be produced (defaults to true) 

    Returns:
        - posterior plot: The plot of the posterior
        - log2FC plot: The plot of the log2FC
        - regression plot: The plot of the regression
        - csv file with results: The csv file with the results. If "csv_file" already exists the data will be appended.

"""
function main(
    data::InteractionData, idx::I, referenceID::I;
    μ_0::Union{F,Nothing}=nothing,
    σ_0::Union{F,Nothing}=nothing,
    a_0::Union{F,Nothing}=nothing, b_0::Union{F,Nothing}=nothing,
    α::F=0.95, csv_file="data/results.csv",
    plotHBMdists::Bool=true, plotlog2fc::Bool=true, plotregr::Bool=true,
    plotbayesrange::Bool=true, writecsv::Bool=true, verbose::Bool=true,
    computeHBM::Bool=true,
    cached_hbm_prior::Union{Nothing,InferenceResult}=nothing,
    cached_regression_prior::Union{Nothing,InferenceResult}=nothing,
    regression_likelihood::Symbol=:normal,
    student_t_nu::Float64=5.0,
    robust_tau_base::Float64=NaN,
    hbm_iterations::Int=75,
    regression_iterations::Union{Int,Nothing}=nothing,
    h0_mode::Bool=false,
    regression_bf_threshold::Float64=0.1,
    jzs_r_scale::Float64=0.0,
    global_tau_base::Float64=NaN,
    regression_min_posterior_var::Float64=0.0,
    # NEW: forward mask-aware kwargs to regression()
    mask_aware_regression::Bool=true,
    raw_data::Union{Nothing, InteractionData}=nothing,
    dropout_fit::Union{Nothing, DropoutFit}=nothing,
    intensity_matrices_by_protocol::Union{Nothing, Vector}=nothing,
    column_imputation_sigma_sq::Union{Nothing, Dict{Tuple{Int,Int,Int}, Float64}}=nothing,
) where {I<:Integer,F<:AbstractFloat}

    protein_name = getIDs(data)[idx]
    verbose && println("Analysis of Protein $protein_name")
    protocol_positions = getProtocolPositions(data)

    # initiate folders
    if any([plotHBMdists, plotlog2fc, plotregr, plotbayesrange])
        initiate_folders()::Nothing
    end

    if isnothing(μ_0) || isnothing(σ_0)
        μ_0, σ_0 = μ0(data)
    end

    if isnothing(a_0) || isnothing(b_0)
        σ_dist = τ0(data)
        a_0, b_0 = σ_dist.α, σ_dist.θ
    end

    # -------------------------------------------------------------- #
    # HBM of regression
    # -------------------------------------------------------------- #
    regression_result, regression_stats, bfRegression = regression(
        data, idx, referenceID, α, μ_0, σ_0,
        plotregr, protein_name, verbose=verbose,
        cached_regression_prior=cached_regression_prior,
        regression_likelihood=regression_likelihood,
        student_t_nu=student_t_nu,
        robust_tau_base=robust_tau_base,
        regression_iterations=regression_iterations,
        regression_bf_threshold=regression_bf_threshold,
        jzs_r_scale=jzs_r_scale,
        global_tau_base=global_tau_base,
        regression_min_posterior_var=regression_min_posterior_var,
        # NEW: forward mask-aware kwargs
        mask_aware_regression=mask_aware_regression,
        raw_data=raw_data,
        dropout_fit=dropout_fit,
        intensity_matrices_by_protocol=intensity_matrices_by_protocol,
        column_imputation_sigma_sq=column_imputation_sigma_sq,
    )

    ##########################################
    # HBM of log2FC
    ##########################################
    if computeHBM
        hbm_result = enrichment(data, idx; μ_0=μ_0, σ_0=σ_0, a_0=a_0, b_0=b_0, cached_prior=cached_hbm_prior, hbm_iterations=hbm_iterations)
        log2FC = compute_log2FC(data, idx)

        bfHBM = if h0_mode
            # H0 mode: only compute threshold=0 (the only value used via result.bfHBM[1])
            bf_col, _, _ = BayesFactorHBM(hbm_result, threshold=0.0)
            reshape(bf_col, :, 1)
        else
            mat = Matrix{Float64}(undef, length(hbm_result.posterior.posteriors[:μ_sample]), 6)
            for (threshold, col_idx) ∈ zip(0.0:1.0:5.0, 1:6)
                mat[:, col_idx], _, _ = BayesFactorHBM(hbm_result, threshold=threshold)
            end
            mat
        end
        HBM_stats = log2FCStatistics(hbm_result, α=α)

        result = BayesResult(
            bfHBM, bfRegression,
            HBM_stats, regression_stats,
            hbm_result,
            regression_result,
            protein_name
        )
    else
        bfHBM = zeros(Float64, 2, 2)
        HBM_stats = Dict{Symbol,Union{Vector{Vector{Float64}},Vector{Float64},Vector{String}}}(:empty => Float64[])
        result = BayesResult(bfHBM, bfRegression, HBM_stats, regression_stats, nothing, nothing, protein_name)
    end

    ##########################################
    # plotting
    ##########################################

    plotHBMdists && plot_inference_results(result, file="data/HBM_dists/$(protein_name)_dists.png")
    plotlog2fc && plot_log2fc(result, log2FC, file="data/log2FC/$(protein_name)_log2fc.png")

    plotbayesrange && plot_bayesrange(
        result, copy(protocol_positions), protein_name,
        file="data/rangeplot/$(protein_name)_rangeplot.png"
    )


    ##########################################
    # Export to csv
    ##########################################

    if writecsv
        mylock = ReentrantLock()
        @lock mylock write_txt(
            filename=csv_file, protein_name=protein_name,
            HBM_stats=HBM_stats, regression_stats=regression_stats,
            bf=bfHBM, bfR=bfRegression, nprotocols=getNoProtocols(data)
        )
    end

    return result
end

function compute_σ2(interactome_sample::Array{Union{Missing,Float64},3}, interactome_control::Array{Union{Missing,Float64},3})
    τ = Float64[]
    for protocol ∈ axes(interactome_sample, 1), data in (interactome_sample, interactome_control)
        concatenated = Float64[]
        for i in eachindex(data[protocol, :, :])
            ismissing(data[protocol, :, :][i]) ? continue : push!(concatenated, data[protocol, :, :][i])
        end
        length(concatenated) >= 3 && push!(τ, var(concatenated))
    end
    return mean(τ)
end

"""
    τ0(data::InteractionData)

    Compute the precision τ for each protein and fit a Gamma distribution to it to get a prior for the precision.

    Input:
        - data::InteractionData: The InteractionData object.
    Output:
        - fittedΓ::Gamma: The fitted Gamma distribution.
        - no_proteins::Int64: The number of proteins.
        - τ::Vector{Float64}: The precision τ for each protein.
"""
function τ0(data::InteractionData)
    τ_list = Float64[]
    for idx in 1:length(getIDs(data))
        protein = getProteinData(data, idx)
        samples = getSampleMatrix(protein)
        controls = getControlMatrix(protein)
        τ = 1 ./ compute_σ2(samples, controls)
        (isfinite(τ) && τ > 0.0) && push!(τ_list, τ)
    end

    if length(τ_list) < 2
        @warn "τ0: insufficient finite τ samples ($(length(τ_list))); using fallback Gamma(1.0, 1.0)"
        return Gamma(1.0, 1.0)
    end

    try
        return fit(Gamma, τ_list)
    catch e
        # Gamma MLE can fail (NaN α from Newton-Raphson) on near-degenerate
        # τ_list distributions — fall back to a moment-matched Gamma.
        m = mean(τ_list)
        v = var(τ_list)
        if v <= 0 || !isfinite(m) || !isfinite(v)
            @warn "τ0: Gamma fit failed and moment fallback degenerate; using Gamma(1.0, 1.0)" exception=e
            return Gamma(1.0, 1.0)
        end
        α = m^2 / v
        θ = v / m
        @warn "τ0: Gamma MLE failed; using moment-matched Gamma(α=$α, θ=$θ)" exception=e
        return Gamma(α, θ)
    end
end

function compute_μ0(data::InteractionData, idx::Int)
    protein = getProteinData(data, idx)
    sample_vals = skipmissing(getSampleMatrix(protein))
    control_vals = skipmissing(getControlMatrix(protein))
    vals = vcat(collect(sample_vals), collect(control_vals))
    return isempty(vals) ? NaN : mean(vals)
end

function μ0(data::InteractionData)
    means = Float64[]
    for idx in 1:length(getIDs(data))
        μ = compute_μ0(data, idx)
        if !isnan(μ)
            push!(means, μ)
        end
    end

    σ2_list = Float64[]
    for idx in 1:length(getIDs(data))
        protein = getProteinData(data, idx)
        samples = getSampleMatrix(protein)
        controls = getControlMatrix(protein)
        σ2 = compute_σ2(samples, controls)
        (!isnan(σ2) && σ2 > 0.0) && push!(σ2_list, σ2)
    end

    return median(means), maximum(σ2_list)
end
