const PosteriorNormalFamily = Union{
    Distribution{Univariate, Continuous},
    NormalMeanPrecision{Float64},
    NormalWeightedMeanPrecision{Float64},
    NormalMeanVariance{Float64}
}

"""
    to_normal(x::InferenceResult, key::Symbol)
    Convert posterior to Normal distribution.

    Args:
        - x<:InferenceResult:     inference result object of type InferenceResult
        - key<:Symbol:            key of the posterior distribution

    Returns:
        - Vector{Normal{Float64}}
"""
function to_normal(x::InferenceResult, key::Symbol)
    posterior::Vector{PosteriorNormalFamily} = x.posteriors[key]
    return [to_normal(dist) for dist in posterior]
end

to_normal(dist::PosteriorNormalFamily) = Normal(mean(dist), std(dist))


function sample(result::HBMResult, ndraws::I) where I<:Integer
    posterior_sample  = to_normal(x, :μ_sample)
    posterior_control = to_normal(x, :μ_control)
    samples::Vector{Vector{Float64}}    = rand.(posterior_sample, ndraws)
    controls::Vector{Vector{Float64}}   = rand.(posterior_control, ndraws)
    return (samples, controls)
end


function sample(x::RegressionResultMultipleProtocols, ndraws::Int64) 
    return Dict(
        :α                  => rand.(x.posteriors[:α], ndraws),
        :β                  => rand.(x.posteriors[:β], ndraws),
        :σ_α                => rand.(x.posteriors[:σ_α], ndraws),
        :σ_β                => rand.(x.posteriors[:σ_β], ndraws),
        :σ                  => rand.(x.posteriors[:σ], ndraws),
        :μ_α                => rand.(x.posteriors[:μ_α], ndraws),
        :μ_β                => rand.(x.posteriors[:μ_β], ndraws)
        )
end

"""
    log2FC(sample::PosteriorNormalFamily, control::PosteriorNormalFamily)

    Compute the log2FC between the sample and control and return a Normal distribution.

    Args:
        - sample<:Union{NormalWeightedMeanPrecision{Float64},NormalMeanPrecision{Float64}}: sample (result of inference)
        - control<:Union{NormalWeightedMeanPrecision{Float64},NormalMeanPrecision{Float64}}: control (result of inference)

    Returns:
        - log2FC<:Normal{Float64}: log2FC
"""
function log2FC(sample::PosteriorNormalFamily, control::PosteriorNormalFamily)::Normal{Float64}
    sample, control                     = map(to_normal, (sample, control))
    sample_mean, control_mean           = map(mean, (sample, control))
    sample_variance, control_variance   = map(var, (sample, control))
    # compute log2FC
    return Normal(sample_mean - control_mean, sqrt(sample_variance + control_variance))
end

"""
    cdf_log2FC(log2FC}; threshold::Float64 = 0.0)

    Compute the cumulative density probability of the log2FC for x <= threshold. 

    Args:
        - log2FC<:Normal{Float64}: log2FC distribution (computed via log2FC)
        - threshold<:Float64: threshold

    Returns:
        - cdf_log2FC<:Float64: cdf_log2FC
"""
cdf_log2FC(log2FC; threshold::Float64 = 0.0) = cdf(log2FC, threshold)

"""
    append_unique!(v1::Vector{T}, v2::Vector{T}) where T

    Appends the elements of `v2` to `v1` if they are not already in `v1` and stores the result in `v1`.
"""
append_unique!(v1::Vector{T}, v2::Vector{T}) where T = append!(v1, filter(x -> !(x ∈ v1), v2))


function check_file(file::String) 
    !isfile(file) && throw(ArgumentError("File $file does not exist"))
    return nothing
end


"""
    q(BF)

    Computes the q-value for a vector of Bayes factors using the method described in:
    - Storey & Tibshirani (2003): Statistical significance for genomewide studies, Proceedings of the National Academy of Sciences, 100(16):9440–9445, 2003
    - John D. Storey, “The positive false discovery rate: A Bayesian interpretation and the q‑value,” The Annals of Statistics, 31(6):2013–2035, 2003.
    - Storey J.D., “QVALUE: The Manual Version 1.0,” 2003. https://genomics.princeton.edu/storeylab/qvalue/manual.pdf
"""

"""
    q(values; isBF::Bool = true)

Computes q-values from a vector of Bayes Factors or posterior probabilities,
correctly and robustly handling missing values.
"""
function q(x; isBF::Bool = true)
    # 1. Convert BFs to posterior probabilities if necessary
    posterior_prob = isBF ? (@. x / (1 + x)) : x
    # 2. Separate the valid (non-missing) probabilities for calculation
    valid_indices = findall(!ismissing, posterior_prob)
    valid_probs   = collect(skipmissing(posterior_prob)) 
    
    # If there are no valid probabilities, there's nothing to calculate.
    if isempty(valid_probs)
        return posterior_prob # Return the original vector (all missings)
    end

    # 3. Perform the entire q-value calculation on the clean `valid_probs` vector
    sorted_idx_valid = sortperm(valid_probs, rev=true)
    probs_sorted = valid_probs[sorted_idx_valid]

    local_fdr_sorted = 1.0 .- probs_sorted

    isfinite_local_fdr_sorted = findall(x -> isfinite(x), local_fdr_sorted)
    cumulative_expected_false_positives = fill(NaN, length(local_fdr_sorted))
    cumulative_expected_false_positives[isfinite_local_fdr_sorted] .= cumsum(local_fdr_sorted[isfinite_local_fdr_sorted])
    
    bfdr = cumulative_expected_false_positives ./ (1:length(cumulative_expected_false_positives))

    # Un-sort the calculated q-values to match the order of `valid_probs`
    qvals_calculated = bfdr[invperm(sorted_idx_valid)]

    # 4. Create a full-length result vector and place the results back
    final_q_values = Vector{Union{Missing, Float64}}(missing, length(posterior_prob))
    final_q_values[valid_indices] = qvals_calculated

    # 5. Set q-value to 1.0 for all proteins with a posterior probability of 0.0
    posterior_prob_is_zero = findall(x -> x == 0.0, posterior_prob[valid_indices])
    final_q_values[valid_indices[posterior_prob_is_zero]] .= 1.0
    
    return final_q_values
end


