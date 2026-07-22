"""
    BayesFactorHBM(result::HBMResultMultipleProtocols; threshold::Float64 = 0.0)
    BayesFactorHBM(result::HBMResultSingleProtocol; threshold::Float64 = 0.0)

Compute the Bayes factor for each parameter's log₂ fold change between the posterior and prior
from a hierarchical Bayesian model (HBM).

Dispatches on concrete HBMResult subtypes. The field access is identical for both types,
but concrete dispatch ensures type stability and mirrors the BayesFactorRegression pattern.

# Arguments
- `result`: An `HBMResultMultipleProtocols` or `HBMResultSingleProtocol` object.
- `threshold`: The log₂FC threshold for hypothesis testing (default = 0.0).

# Returns
- Tuple of `(BF::Vector{Float64}, postProb::Vector{Float64}, priorProb::Vector{Float64})`,
  representing Bayes factors, posterior probability and prior probability.
"""
function BayesFactorHBM(
    result::HBMResultMultipleProtocols;
    threshold::Float64 = 0.0
)
    posterior = result.posterior
    prior = result.prior
    posterior_control = posterior.posteriors[:μ_control]
    posterior_sample  = posterior.posteriors[:μ_sample]
    prior_control     = prior.posteriors[:μ_control]
    prior_sample      = prior.posteriors[:μ_sample]
    posterior_log2FC = log2FC.(posterior_sample, posterior_control)
    prior_log2FC     = log2FC.(prior_sample, prior_control)
    return calculate_bayes_factor(posterior_log2FC, prior_log2FC; threshold = threshold)
end

function BayesFactorHBM(
    result::HBMResultSingleProtocol;
    threshold::Float64 = 0.0
)
    posterior = result.posterior
    prior = result.prior
    posterior_control = posterior.posteriors[:μ_control]
    posterior_sample  = posterior.posteriors[:μ_sample]
    prior_control     = prior.posteriors[:μ_control]
    prior_sample      = prior.posteriors[:μ_sample]
    posterior_log2FC = log2FC.(posterior_sample, posterior_control)
    prior_log2FC     = log2FC.(prior_sample, prior_control)
    return calculate_bayes_factor(posterior_log2FC, prior_log2FC; threshold = threshold)
end


"""
    BayesFactorRegression(result::RegressionResultMultipleProtocols; threshold=0.0)

    Computes the Bayes factor for a given posterior and prior inference results.
        H1: slope > threshold
        H0: slope <= threshold

    Args:
        - posterior<:InferenceResult:   posterior inference result object of type InferenceResult 
        - prior<:InferenceResult:       prior inference result object of type InferenceResult

    Returns:
        - Bayes factor
        - posterior probability
        - prior probability
"""
function BayesFactorRegression(result::RegressionResultMultipleProtocols; threshold = 0.0, jzs_r_scale::Float64 = 0.0, min_posterior_var::Float64 = 0.0)
    posterior = result.posterior.posteriors
    prior = result.prior.posteriors

    posterior_slopes = posterior[:α]
    prior_slopes     = prior[:α]

    bayes_factor, p_post, p_prior = calculate_bayes_factor(posterior_slopes, prior_slopes; threshold = threshold, min_posterior_var = min_posterior_var)

    # global slope μ_α — compute from the hierarchical hyper-mean, not per-protocol slopes
    # When using JZS prior, use analytical Cauchy CDF for the prior probability on μ_α
    prior_p_override = jzs_r_scale > 0.0 ? _cauchy_sf(threshold, jzs_r_scale) : nothing
    bf_μ_α, p_post_μ_α, p_prior_μ_α = calculate_bayes_factor(
        [posterior[:μ_α]], [prior[:μ_α]]; threshold = threshold, prior_p_override = prior_p_override, min_posterior_var = min_posterior_var)

    # prepend the BF, posterior and prior probability of the global slope
    prepend!(bayes_factor, bf_μ_α[1])
    prepend!(p_post, p_post_μ_α[1])
    prepend!(p_prior, p_prior_μ_α[1])

    return bayes_factor, p_post, p_prior
end

function BayesFactorRegression(result::RegressionResultSingleProtocol; threshold = 0.0, jzs_r_scale::Float64 = 0.0, min_posterior_var::Float64 = 0.0)
    posterior = result.posterior.posteriors
    prior = result.prior.posteriors

    posterior_slope = posterior[:α]
    prior_slope     = prior[:α]

    # When using JZS prior, use analytical Cauchy CDF for the prior probability
    prior_p_override = jzs_r_scale > 0.0 ? _cauchy_sf(threshold, jzs_r_scale) : nothing
    return calculate_bayes_factor(posterior_slope, prior_slope; threshold = threshold, prior_p_override = prior_p_override, min_posterior_var = min_posterior_var)
end

# Robust regression dispatches: delegate to normal equivalents since posterior
# structure for α, β, σ is identical (w is marginalized out in BF computation)
BayesFactorRegression(r::RobustRegressionResultMultipleProtocols; threshold = 0.0, jzs_r_scale::Float64 = 0.0, min_posterior_var::Float64 = 0.0) =
    BayesFactorRegression(RegressionResultMultipleProtocols(r.posterior, r.prior); threshold=threshold, jzs_r_scale=jzs_r_scale, min_posterior_var=min_posterior_var)

BayesFactorRegression(r::RobustRegressionResultSingleProtocol; threshold = 0.0, jzs_r_scale::Float64 = 0.0, min_posterior_var::Float64 = 0.0) =
    BayesFactorRegression(RegressionResultSingleProtocol(r.posterior, r.prior); threshold=threshold, jzs_r_scale=jzs_r_scale, min_posterior_var=min_posterior_var)


"""
    _cauchy_sf(threshold, r)

Survival function P(X > threshold) for a Cauchy(0, r) distribution.
Used for analytical prior probability in JZS Bayes factor computation.
"""
_cauchy_sf(threshold::Real, r::Real) = 0.5 - atan(threshold / r) / π


"""
    calculate_bayes_factor(posterior, prior; threshold = 0.0, prior_p_override = nothing, min_posterior_var = 0.0)

Compute Bayes factors comparing posterior vs. prior probability of exceeding a given threshold.

# Arguments
- `posterior`: A vector of distributions (assumed Normal-like).
- `prior`: A vector of distributions (assumed Normal-like).
- `threshold`: The threshold above which probability is computed (default = 0.0).
- `prior_p_override`: If not `nothing`, use this value as P(x > threshold | prior) for ALL
  parameters instead of computing from the VMP prior. Used for JZS prior where the true
  prior is Cauchy but VMP approximates it as Normal.
- `min_posterior_var`: Minimum variance floor for posterior distributions (default = 0.0,
  i.e., no floor). VMP posteriors can be over-confident (extremely narrow σ), causing
  P(slope>0) to saturate at 0 or 1 and BFs to hit ±max_bf clamp. Setting e.g. 0.01
  ensures min posterior std = 0.1, restoring BF gradation.

# Returns
- `bayes_factor::Vector{Float64}`: The Bayes factor per parameter.
- `p_post::Vector{Float64}`: Posterior probability `P(x > threshold)`.
- `p_prior::Vector{Float64}`: Prior probability `P(x > threshold)`.
"""
function calculate_bayes_factor(posterior, prior; threshold = 0.0, max_bf = 1e6,
                                prior_p_override::Union{Nothing, Float64} = nothing,
                                min_posterior_var::Float64 = 0.0)
    post_normals = to_normal.(posterior)

    # Ensure post_normals is always a vector (single-protocol returns scalar Normal)
    if post_normals isa Normal
        post_normals = [post_normals]
    end

    # Apply posterior variance floor: VMP posteriors can be over-confident
    # (extremely narrow σ), causing P(slope>0) ≈ 0 or 1 and saturating BFs
    # at ±max_bf. The floor ensures minimum uncertainty is preserved.
    if min_posterior_var > 0.0
        min_std = sqrt(min_posterior_var)
        post_normals = [Normal(mean(d), max(std(d), min_std)) for d in post_normals]
    end

    p_post  = 1 .- cdf.(post_normals, threshold)

    # Use analytical prior probability when available (JZS Cauchy prior)
    p_prior = if isnothing(prior_p_override)
        prior_normals = to_normal.(prior)
        if prior_normals isa Normal
            prior_normals = [prior_normals]
        end
        1 .- cdf.(prior_normals, threshold)
    else
        fill(prior_p_override, length(p_post))
    end

    # Clamp probabilities away from 0 and 1 to avoid Inf in log-odds.
    # Using 1e-12 (not eps(Float64) ≈ 2.2e-16) so that the maximum
    # representable odds ≈ 1e12, well within Float64 range.
    ϵ = 1e-12
    p_post_c  = clamp.(p_post,  ϵ, 1 - ϵ)
    p_prior_c = clamp.(p_prior, ϵ, 1 - ϵ)

    # Compute log Bayes factor via log-odds difference (avoids overflow)
    log_bf = @. (log(p_post_c) - log1p(-p_post_c)) - (log(p_prior_c) - log1p(-p_prior_c))

    # Cap: individual BFs beyond max_bf are numerical artifacts, not real evidence
    max_log_bf = log(max_bf)
    log_bf     = clamp.(log_bf, -max_log_bf, max_log_bf)

    bayes_factor = exp.(log_bf)
    return bayes_factor, p_post, p_prior
end

"""
    probability_of_direction(draws::Vector{Float64})
    probability_of_direction(draws::Vector{Vector{Float64}})
    probability_of_direction(posterior::Union{MixtureModel,Normal{Float64}})

Compute the probability of direction (Maximum Probability of Effect - MPE) for a posterior distribution.

# Returns
- `pd::Float64`: Probability of direction (between 0.5 and 1.0).
- `direction::String`: "+" if effect is more likely positive, "-" if negative, or "~" if undecidable.
"""
function probability_of_direction(draws::Vector{Float64})
    p_pos = sum(draws .> 0.0) / length(draws)
    if p_pos > 0.5
        return p_pos, "+"
    elseif p_pos < 0.5
        return 1 - p_pos, "-"
    else
        return 0.5, "~"
    end
end

function probability_of_direction(draws::Vector{Vector{Float64}})
    pd = zeros(Float64, length(draws))
    direction = Vector{String}(undef, length(draws))
    for (idx, x) ∈ pairs(draws)
        pd[idx], direction[idx] = probability_of_direction(x)
    end
    return pd, direction
end

function probability_of_direction(posterior::Union{Normal{Float64}, MixtureModel})
    p_neg = cdf(posterior, 0.0)
    if p_neg < 0.5
        return 1 - p_neg, "+"
    elseif p_neg > 0.5
        return p_neg, "-"
    else
        return 0.5, "~"
    end
end


"""
    pd_to_p_value(pd, two_sided::Bool)
    Converts the probability of direction (pd) to a p-value.

    # Arguments
    - `pd::Float64`: The probability of direction, assumed to be between 0.5 and 1.0.
    - `two_sided::Bool`: Whether to return a two-sided p-value (default = `true`).

    # Returns
    - `p_value::Float64`: The corresponding frequentist-style p-value.
"""
function pd_to_p_value(pd::Float64, two_sided::Bool = true)::Float64
    @assert 0.5 ≤ pd ≤ 1.0 "Probability of direction must be in [0.5, 1.0]"
    return two_sided ? 2 * (1 - pd) : 1 - pd
end

"""
    log2FCStatistics(posterior::InferenceResult; ndraws::Int64 = 1_000_000, α::Float64 = 0.95)

    Computes the mean, median, standard deviation, variance, and probability of direction as well as
    the credibility interval for the log2FC at a given credibility level α

    Args:
        - posterior<:InferenceResult: posterior inference result object of type InferenceResult

    Keyword Args:
        - α<:Float64: credibility level

    Returns:
    
    Dict{Symbol, Union{Vector{Float64}, Vector{String}}} with the following keys:
        - :mean_log2FC:     mean of the log2FC
        - :median_log2FC:   median of the log2FC
        - :sd_log2FC:       standard deviation of the log2FC
        - :variance_log2FC: variance of the log2FC
        - :pd:              probability of direction
        - :pd_direction:    direction of the log2FC (positive or negative)
        - :ci:              credibility interval for the log2FC

"""
function log2FCStatistics(result::HBMResult; α::Float64 = 0.95)
    posterior = result.posterior
    log2fc = log2FC.(posterior.posteriors[:μ_sample], posterior.posteriors[:μ_control])
    return log2FCStatistics(log2fc; α = α)
end

function log2FCStatistics(log2fc; α::Float64 = 0.95)
    mean_log2FC = mean.(log2fc)
    sd_log2FC = std.(log2fc)
    var_log2FC = var.(log2fc)
    median_log2FC = median.(log2fc)

    pd::Vector{Float64} = zeros(Float64, length(log2fc))
    direction::Vector{String} = ["+" for _ in eachindex(log2fc)]
    for (idx, posterior) ∈ enumerate(log2fc)
        pd[idx], direction[idx] = probability_of_direction(posterior)
    end

    ci = [quantile(log2fc[i], [(1-α)/2, 1-(1-α)/2]) for i ∈ eachindex(log2fc)]

    return Dict{Symbol, Union{Vector{Float64}, Vector{Vector{Float64}},Vector{String}}}(
        :mean_log2FC        => mean_log2FC,
        :median_log2FC      => median_log2FC,
        :sd_log2FC          => sd_log2FC,
        :variance_log2FC    => var_log2FC,
        :pd                 => pd,
        :pd_direction       => direction,
        :credible_interval  => ci
    )
end

"""
    RegressionStatistics(result::RegressionResultMultipleProtocols; α::Float64 = 0.95)

    Computes the mean, median, standard deviation, variance, and probability of direction as well as
    the credibility interval for the slope at a given credibility level α.

    Args:
        - posterior<:InferenceResult: posterior inference result object of type InferenceResult

    Keyword Args:
        - α<:Float64: credibility level

    Returns: Dict{Symbol, Union{Vector{Float64}, Vector{Vector{Float64}},Vector{String}}} with the following keys:
        # main slope
        :mean_slope                 => mean_slope,
        :median_slope               => median_slope,
        :sd_slope                   => sd_slope,
        :variance_slope             => var(slope),
        :pd_slope                   => pd_slope,
        :pd_direction_slope         => direction_slope,
        :credible_interval_slope    => ci_slope,
        # protocol slope
        :mean_protocol_slope        => mean_protocol_slope,
        :median_protocol_slope      => median_protocol_slope,
        :sd_protocol_slope          => sd_protocol_slope,
        :variance_protocol_slope    => var.(protocol_slope),
        :pd_protocol                => pd_protocol,
        :pd_direction_protocol      => direction_protocol,
        :credible_interval_protocol => ci_protocol
"""
function RegressionStatistics(result::RegressionResultMultipleProtocols; α::Float64 = 0.95)
    # retrieve posterior
    posterior = result.posterior.posteriors

    slope = to_normal.(posterior[:μ_α])
    protocol_slope = to_normal.(posterior[:α])
    # statistics for slope
    mean_slope = mean(slope)
    sd_slope = std(slope)
    median_slope = median(slope)
    pd_slope, direction_slope = probability_of_direction(slope)
    ci_slope = quantile(slope, [(1-α)/2, 1-(1-α)/2])

    # statistics for protocol slope
    mean_protocol_slope = mean.(protocol_slope)
    sd_protocol_slope = std.(protocol_slope)
    median_protocol_slope = median.(protocol_slope)

    pd_protocol, direction_protocol = Float64[], String[]
    for i ∈ eachindex(protocol_slope)
        tmp_pd, tmp_direction = probability_of_direction(protocol_slope[i])
        push!(pd_protocol, tmp_pd)
        push!(direction_protocol, tmp_direction)
    end

    # credible interval
    ci_protocol = [zeros(Float64,2) for _ in eachindex(protocol_slope)]
    
    for i ∈ eachindex(ci_protocol)
        ci_protocol[i] = quantile(protocol_slope[i], [(1-α)/2, 1-(1-α)/2])
    end

    t = Union{Float64, String, Vector{String}, Vector{Float64}, Vector{Vector{Float64}}}

    return Dict{Symbol, t}(
        # main slope
        :mean_slope                 => mean_slope,
        :median_slope               => median_slope,
        :sd_slope                   => sd_slope,
        :variance_slope             => var(slope),
        :pd_slope                   => pd_slope,
        :pd_direction_slope         => direction_slope,
        :credible_interval_slope    => ci_slope,
        # protocol slope
        :mean_protocol_slope        => mean_protocol_slope,
        :median_protocol_slope      => median_protocol_slope,
        :sd_protocol_slope          => sd_protocol_slope,
        :variance_protocol_slope    => var.(protocol_slope),
        :pd_protocol                => pd_protocol,
        :pd_direction_protocol      => direction_protocol,
        :credible_interval_protocol => ci_protocol
    )
end

"""
    RegressionStatistics(result::RegressionResultSingleProtocol; α::Float64 = 0.95)

Computes summary statistics for the main slope parameter (assumed to be `:α`) from a `RegressionResultSingleProtocol` object.

This function extracts the posterior distribution of the slope, calculates key descriptive statistics such as the mean, median, credible interval, and probability of direction, and returns them in a dictionary.

# Arguments
- `result::RegressionResultSingleProtocol`: The result object from a single-protocol regression analysis. It must contain a `posterior.posteriors` field, which is expected to be a dictionary-like object holding the posterior distribution for the `:α` key.

# Keyword Arguments
- `α::Float64 = 0.95`: The probability mass to be contained within the credible interval. For example, `α = 0.95` corresponds to a 95% credible interval.

# Returns
- `Dict{Symbol, Any}`: A dictionary where keys are statistic names (as `Symbol`s) and values are the calculated statistics. The dictionary includes:
    - `:mean_slope`: Mean of the posterior distribution.
    - `:median_slope`: Median of the posterior distribution.
    - `:sd_slope`: Standard deviation of the posterior distribution.
    - `:variance_slope`: Variance of the posterior distribution.
    - `:pd_slope`: The Probability of Direction (PD), representing the certainty of the effect's direction (e.g., positive or negative).
    - `:pd_direction_slope`: A string (`"+"` or `"-"`) indicating the direction of the effect.
    - `:credible_interval_slope`: A vector with the lower and upper bounds of the `α`% credible interval.
"""
function RegressionStatistics(result::RegressionResultSingleProtocol; α::Float64 = 0.95)
    # retrieve posterior
    posterior = result.posterior.posteriors

    slope = to_normal.(posterior[:α])
    # statistics for slope
    mean_slope = mean(slope)
    sd_slope = std(slope)
    median_slope = median(slope)
    pd_slope, direction_slope = probability_of_direction(slope)
    ci_slope = quantile(slope, [(1-α)/2, 1-(1-α)/2])

    # statistics for protocol slope
    t = Union{Float64, String, Vector{String}, Vector{Float64}, Vector{Vector{Float64}}}

    return Dict{Symbol, t}(
        # main slope
        :mean_slope                 => mean_slope,
        :median_slope               => median_slope,
        :sd_slope                   => sd_slope,
        :variance_slope             => var(slope),
        :pd_slope                   => pd_slope,
        :pd_direction_slope         => direction_slope,
        :credible_interval_slope    => ci_slope,
    )
end


function RegressionStatistics(result::RegressionResult; α::Float64 = 0.95)
    return error("The method RegressionStatistics is not implemented for the abstract type RegressionResult")
end

# Robust regression statistics dispatches
RegressionStatistics(r::RobustRegressionResultMultipleProtocols; α::Float64 = 0.95) =
    RegressionStatistics(RegressionResultMultipleProtocols(r.posterior, r.prior); α=α)

RegressionStatistics(r::RobustRegressionResultSingleProtocol; α::Float64 = 0.95) =
    RegressionStatistics(RegressionResultSingleProtocol(r.posterior, r.prior); α=α)