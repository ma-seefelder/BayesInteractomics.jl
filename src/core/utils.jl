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
    bfdr(values; isBF::Bool = true)

Computes Bayesian FDR (BFDR) values from a vector of Bayes Factors or posterior
probabilities, correctly and robustly handling missing values.

Uses the method described in:
- Storey & Tibshirani (2003): Statistical significance for genomewide studies, PNAS, 100(16):9440-9445
- John D. Storey, "The positive false discovery rate: A Bayesian interpretation and the q-value," Ann. Stat., 31(6):2013-2035, 2003.
"""
function bfdr(x; isBF::Bool = true)
    # 1. Convert BFs to posterior probabilities if necessary
    posterior_prob = isBF ? (@. x / (1 + x)) : x
    # 2. Separate the valid (non-missing) probabilities for calculation
    valid_indices = findall(!ismissing, posterior_prob)
    valid_probs   = collect(skipmissing(posterior_prob)) 
    
    # If there are no valid probabilities, there's nothing to calculate.
    if isempty(valid_probs)
        return posterior_prob # Return the original vector (all missings)
    end

    # 3. Perform the entire BFDR calculation on the clean `valid_probs` vector
    sorted_idx_valid = sortperm(valid_probs, rev=true)
    probs_sorted = valid_probs[sorted_idx_valid]

    local_fdr_sorted = 1.0 .- probs_sorted

    isfinite_local_fdr_sorted = findall(x -> isfinite(x), local_fdr_sorted)
    cumulative_expected_false_positives = fill(NaN, length(local_fdr_sorted))
    cumulative_expected_false_positives[isfinite_local_fdr_sorted] .= cumsum(local_fdr_sorted[isfinite_local_fdr_sorted])
    
    bfdr_vals = cumulative_expected_false_positives ./ (1:length(cumulative_expected_false_positives))

    # Storey monotone step-down correction: enforce non-increasing BFDR values
    # when sorted by decreasing posterior probability
    bfdr_vals = reverse(accumulate(min, reverse(bfdr_vals)))
    # Defensive clamp against floating-point drift
    bfdr_vals = clamp.(bfdr_vals, 0.0, 1.0)

    # Un-sort the calculated BFDR values to match the order of `valid_probs`
    bfdr_calculated = bfdr_vals[invperm(sorted_idx_valid)]

    # 4. Create a full-length result vector and place the results back
    final_bfdr_values = Vector{Union{Missing, Float64}}(missing, length(posterior_prob))
    final_bfdr_values[valid_indices] = bfdr_calculated

    # 5. Set BFDR to 1.0 for all proteins with a posterior probability of 0.0
    posterior_prob_is_zero = findall(x -> x == 0.0, posterior_prob[valid_indices])
    final_bfdr_values[valid_indices[posterior_prob_is_zero]] .= 1.0
    
    return final_bfdr_values
end

"""
    pep(x)

Compute Posterior Error Probability (PEP = 1 - posterior probability).
Handles missing values.
"""
function pep(x)
    return @. ifelse(ismissing(x), missing, 1.0 - x)
end

"""Deprecated: use `bfdr()` instead."""
function q(x; kwargs...)
    @warn "`q()` is deprecated, use `bfdr()` instead" maxlog=1
    return bfdr(x; kwargs...)
end
