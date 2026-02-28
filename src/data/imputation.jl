#=
BayesInteractomics: A Julia package for the analysis of protein interactome data from Affinity-purification mass spectrometry (AP-MS) and proximity labelling experiments
# Version: 0.1.0

Copyright (C) 2025  Dr. rer. nat. Manuel Seefelder
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
    compute_mixture(posterior::Vector{BayesResult})

Compute a mixture model for each parameter across multiple imputed log2FC posterior samples.

# Arguments
- `log2fc_list`: A vector where each element is a vector of `Normal{Float64}` distributions 
  (one per parameter), corresponding to a single imputed posterior.

# Returns
- `mixtures`: A vector of `MixtureModel{Normal{Float64}}` objects, one for each parameter.
"""
function compute_mixture(posterior::Vector{BayesResult})
    ninferences = length(posterior)
    nparameters = length(posterior[1].posteriorHBM.posteriors[:μ_control])
    posterior = [posterior[i].posteriorHBM for i ∈ 1:ninferences]
    # compute log2FC
    log2fc_values = [log2FC.(posterior[i].posteriors[:μ_sample], posterior[i].posteriors[:μ_control]) for i ∈ 1:ninferences]
    # Preallocate result
    mixtures = MixtureModel[]; sizehint!(mixtures, nparameters)

    @inbounds for parameter in 1:nparameters
        push!(mixtures, MixtureModel([log2fc_values[i][parameter] for i ∈ 1:ninferences]))
    end

    return mixtures
end

"""
    compute_statistics(
        mixture::Vector{MixtureModel{Normal{Float64}}}, 
        prior::HBMResult, 
        results::Vector{BayesResult};
        α::Float64 = 0.95,
        thresholds::AbstractVector{<:Real} = 0.0:1.0:5.0
    ) -> NamedTuple

Compute Bayes factors and summary statistics from the log₂FC mixture and prior posteriors.

# Arguments
- `mixture`: A vector of `MixtureModel{Normal{Float64}}` for log₂FC posteriors.
- `prior`: An `HBMResult` containing the prior posteriors.
- `results`: A vector of `BayesResult` structs (for regression info).
- `α`: Credible interval width (default = 0.95).
- `thresholds`: Vector of thresholds for Bayes factor evaluation.

# Returns
- NamedTuple with:
    - `bfHBM::Matrix{Float64}`
    - `HBM_stats`
    - `bfRegression` (or `nothing`)
    - `regression_stats` (or `nothing`)
"""
function compute_statistics(
    mixture::Vector{MixtureModel},
    prior::InferenceResult,
    results::Vector{BayesResult};
    α::Float64 = 0.95,
    thresholds::AbstractVector{<:Real} = 0.0:1.0:5.0
)
    nparams = length(mixture)
    priorlog2FC = log2FC.(prior.posteriors[:μ_sample], prior.posteriors[:μ_control])

    bfHBM = Matrix{Float64}(undef, nparams, length(thresholds))
    for (i, idx) in zip(thresholds, 1:length(thresholds))
        bfHBM[:, idx], _, _ = calculate_bayes_factor(mixture, priorlog2FC, threshold = i)
    end

    HBM_stats = log2FCStatistics(mixture, α = α)

    if !isnothing(results[1].posteriorRegression)
        regression_posterior = miconvertRegression(results)
        regression_stats = RegressionStatistics(regression_posterior)
        bfRegression, _, _ = BayesFactorRegression(regression_posterior, results[1].priorRegression)
    else
        bfRegression = nothing
        regression_stats = nothing
    end

    return (
        bfHBM = bfHBM,
        HBM_stats = HBM_stats,
        bfRegression = bfRegression,
        regression_stats = regression_stats
    )
end

"""
    evaluate_imputed_fc_posteriors(results, protocol_positions; kwargs...)

Pool Bayes factors and summary statistics from multiple imputation runs.

Combines the posterior inference results from several imputed datasets (stored as
a `Vector{BayesResult}`) using Rubin's pooling rules.  For each parameter the
function computes mixture distributions across imputations, derives pooled Bayes
factors for both the HBM and regression models, and generates optional diagnostic
plots.

# Arguments
- `results::Vector{BayesResult}`: One `BayesResult` per imputed dataset for the
  same protein.
- `protocol_positions`: Protocol position indices from `InteractionData`.

# Keywords
- `plotlog2fc::Bool=false`: Generate log₂FC distribution plots.
- `plotbayesrange::Bool=false`: Generate Bayes factor range plots.
- `threshold::Float64=0.0`: Log₂FC threshold for hypothesis testing.
- `csv_file::String="results.csv"`: Path to write pooled CSV results.
- `writecsv::Bool=true`: Whether to write results to CSV.
- `α::Float64=0.95`: Credibility level for intervals.

# Returns
- `NamedTuple` with fields:
  - `bfHBM`: Pooled Bayes factors from the hierarchical model
  - `HBM_stats`: Pooled log₂FC summary statistics
  - `bfRegression`: Pooled Bayes factors from the regression model
  - `regression_stats`: Pooled regression summary statistics
"""
function evaluate_imputed_fc_posteriors(
    results::Vector{BayesResult}, protocol_positions;
    plotlog2fc = false, plotbayesrange = false, threshold = 0.0,
    csv_file = "results.csv", writecsv = true,
    α = 0.95
    )

    first_result = results[1]
    protein_name = first_result.protein_name
    initiate_folders()::Nothing

    # compute prior log2FC
    prior = first_result.priorHBM
    priorlog2FC = log2FC.(prior.posteriors[:μ_sample], prior.posteriors[:μ_control])

    #########################################
    # generate log2FC plot
    #########################################
    # extract posterior of the individual InferenceResult
    nparameters = length(priorlog2FC)
    log2fc_mixt = compute_mixture(results)
    
    # plot
    plotlog2fc && plot_log2fc(
        log2fc_mixt, first_result.priorHBM, 
        threshold = threshold, file = "data/log2FC/$(protein_name)_log2fc.png"
        )

    #########################################
    # Bayes range plot
    #########################################
    plotbayesrange && plot_bayesrange(
        log2fc_mixt, priorlog2FC, protocol_positions, 
        protein_name, file = "data/rangeplot/$(protein_name)_rangeplot.png"
        )

    #########################################
    # Compute statistics
    #########################################
    bfHBM, HBM_stats, bfRegression, regression_stats = compute_statistics(log2fc_mixt, prior, results, α = α)

    ##########################################
    # Export to csv
    ##########################################
    nprotocols = length(regression_stats[:pd_protocol])
    writecsv && write_txt(
        filename = csv_file, protein_name = protein_name, 
        HBM_stats = HBM_stats, regression_stats = regression_stats, 
        bf = bfHBM, bfR = bfRegression, nprotocols = nprotocols
        )

    return log2fc_mixt
end

"""
    miconvertRegression(posterior::Vector{BayesResult})

    Convert posteriors of the regression model from multiple imputed data to MixtureModel.

    # Arguments
    - `posterior::Vector{BayesResult}`: Vector of BayesResult objects

    # Returns
    - `dict::Dict{Symbol, Any}`: Dictionary with posteriors converted to MixtureModels
"""
function miconvertRegression(posterior::Vector{BayesResult})
    # initialize dictionary
    dict = Dict{Symbol, Any}()
    parameter_names = keys(posterior[1].posteriorRegression.posteriors)

    # loop over parameters and populate dictionary
    for parameter in parameter_names
        # skip predicted_value
        parameter == :predicted_value && continue
        if length(posterior[1].posteriorRegression.posteriors[parameter]) == 1
            p = [posterior[i].posteriorRegression.posteriors[parameter] for i in eachindex(posterior)]
            any(isa.(p, Gamma)) ? nothing : (p = to_normal.(p))
            dict[parameter] = MixtureModel(p)
            continue
        end
        
        for idx in eachindex(posterior[1].posteriorRegression.posteriors[parameter])
            p = [posterior[i].posteriorRegression.posteriors[parameter][idx] for i in eachindex(posterior)]
            p = MixtureModel(to_normal.(p))
            idx == 1 ? (dict[parameter] = [p]) : push!(dict[parameter], p)
        end
    end

    return dict
end
