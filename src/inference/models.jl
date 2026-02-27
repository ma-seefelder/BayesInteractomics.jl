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
    df = DataFrame(File(file, header=1, delim="|"))

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
    df = DataFrame(File(file, header=1, delim="|"))

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

    nparameters = size(samples, 2) + 1

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
    HBM(data::InteractionData, idx::Int64; μ_0::F = 25.0, a_0::F = 1.0, b_0::F = 1.0, cached_prior::Union{Nothing,InferenceResult}=nothing, hbm_iterations::Int=1000) where {F<:AbstractFloat}

    Hierarchical Bayesian Model

    Input:
        data::InteractionData
        idx::Int64
        μ_0::F = 25.0: mean of the Normal distribution prior for the protein intensity values
        σ_0::F = 1.0: variance of the Normal distribution prior for the protein intensity values
        a_0::F = 1.0: shape of the Gamma distribution prior for the precision
        b_0::F = 1.0: scale of the Gamma distribution prior for the precision
        cached_prior::Union{Nothing,InferenceResult}=nothing: Optional precomputed prior to reuse across proteins
        hbm_iterations::Int = 1000: Number of iterations for inference (reduce for faster computation)

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
function HBM(data::InteractionData, idx::Int64; μ_0::F=25.0, σ_0::F=1.0, a_0::F=1.0, b_0::F=1.0, cached_prior::Union{Nothing,InferenceResult}=nothing, hbm_iterations::Int=1000) where {F<:AbstractFloat}
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

function HBM_single_protocol(data::InteractionData, idx::Int64; μ_0::F=25.0, σ_0::F=1.0, a_0::F=1.0, b_0::F=1.0, cached_prior::Union{Nothing,InferenceResult}=nothing, hbm_iterations::Int=1000) where {F<:AbstractFloat}
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


function prepare_regression_data(data::InteractionData, idx::I, referenceID::I) where {I<:Integer}
    protein = getProteinData(data, idx)
    interactome_sample = getSampleMatrix(protein)
    interactome_control = getControlMatrix(protein)
    sample = cat(interactome_sample, interactome_control, dims=2)

    RefProtein = getProteinData(data, referenceID)
    reference_sample = getSampleMatrix(RefProtein)
    reference_control = getControlMatrix(RefProtein)
    reference = cat(reference_sample, reference_control, dims=2)
    return (sample=sample, reference=reference)
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

    length(pooled_residuals) < 2 && return 1.0  # fallback
    v = var(pooled_residuals)
    return v > 0.0 ? 1.0 / v : 1.0
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
function regression(
    data::InteractionData, idx::I, referenceID::I, α::F, intercept::Float64,
    intercept_sigma::Float64, plotregr::Bool, protein_name::S;
    verbose::Bool=true, cached_regression_prior::Union{Nothing,InferenceResult}=nothing,
    regression_likelihood::Symbol=:normal, student_t_nu::Float64=5.0,
    robust_tau_base::Float64=NaN
) where {I<:Integer,F<:AbstractFloat,S<:String}


    # define regression type based on the number of protocols and likelihood
    if regression_likelihood == :robust_t
        regression_fun = getNoProtocols(data) == 1 ? RegressionModel_one_protocol_robust : RegressionModelRobust
    else
        regression_fun = getNoProtocols(data) == 1 ? RegressionModel_one_protocol : RegressionModel
    end

    try
        if regression_likelihood == :robust_t
            regression_result = regression_fun(data, idx, referenceID, intercept, intercept_sigma; nu=student_t_nu, τ_base=robust_tau_base, cached_prior=cached_regression_prior)
        else
            regression_result = regression_fun(data, idx, referenceID, intercept, intercept_sigma; cached_prior=cached_regression_prior)
        end
        regression_stats = RegressionStatistics(regression_result; α=α)

        protein = getProteinData(data, idx)
        reference_protein = getProteinData(data, referenceID)

        bfRegression, _, _ = BayesFactorRegression(regression_result)

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
    μ_0::Union{F,Nothing}=nothing, σ_0::Union{F,Nothing}=nothing,
    a_0::Union{F,Nothing}=nothing, b_0::Union{F,Nothing}=nothing,
    α::F=0.95, csv_file="data/results.csv",
    plotHBMdists::Bool=true, plotlog2fc::Bool=true, plotregr::Bool=true,
    plotbayesrange::Bool=true, writecsv::Bool=true, verbose::Bool=true,
    computeHBM::Bool=true,
    cached_hbm_prior::Union{Nothing,InferenceResult}=nothing,
    cached_regression_prior::Union{Nothing,InferenceResult}=nothing,
    regression_likelihood::Symbol=:normal,
    student_t_nu::Float64=5.0,
    robust_tau_base::Float64=NaN
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
        robust_tau_base=robust_tau_base
    )

    ##########################################
    # HBM of log2FC
    ##########################################
    if computeHBM
        if getNoProtocols(data) == 1
            hbm_result = HBM_single_protocol(data, idx, μ_0=μ_0, σ_0=σ_0, a_0=a_0, b_0=b_0, cached_prior=cached_hbm_prior)
        else
            hbm_result = HBM(data, idx, μ_0=μ_0, σ_0=σ_0, a_0=a_0, b_0=b_0, cached_prior=cached_hbm_prior)
        end
        log2FC = compute_log2FC(data, idx)

        bfHBM::Matrix{Float64} = Matrix(undef, length(hbm_result.posterior.posteriors[:μ_sample]), 6)
        for (i, idx) ∈ zip(collect(0.0:1.0:5.0), 1:6)
            bfHBM[:, idx], _, _ = BayesFactorHBM(hbm_result, threshold=i)
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
        (!isnan(τ) && τ > 0.0) && push!(τ_list, τ)
    end

    return fit(Gamma, τ_list)
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
