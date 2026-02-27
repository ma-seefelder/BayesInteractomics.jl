"""
    prob_beta_greater(θx::Beta, θy::Beta)

Computes the analytical probability P(θx > θy) for two Beta distributions using numerical integration.

This replaces the Monte Carlo approach with a more efficient and accurate analytical method.
The probability is computed as:
    P(X > Y) = ∫₀¹ P(X > y) · f_Y(y) dy = ∫₀¹ (1 - F_X(y)) · f_Y(y) dy

where F_X is the CDF of θx and f_Y is the PDF of θy.

# Arguments
- `θx::Beta`: First Beta distribution
- `θy::Beta`: Second Beta distribution

# Returns
- `Float64`: The probability P(θx > θy), computed with relative tolerance 1e-8
"""
function prob_beta_greater(θx::Beta, θy::Beta)
    integrand(y) = (1 - cdf(θx, y)) * pdf(θy, y)
    result, _ = quadgk(integrand, 0, 1, rtol=1e-8)
    return result
end

"""
    betabernoulli(data::InteractionData, idx::I, n_control::I, n_sample::I; create_plot::Bool=false) where {I <: Int}

Performs a Bayesian A/B test on protein detection rates using a Beta-Bernoulli model.

This function compares the detection rate of a specific protein (`idx`) between a sample
group and a control group. It models the detection of a protein as a Bernoulli trial
(detected or not detected). The underlying detection rate parameter, θ, is given a
conjugate Beta prior (hardcoded as `Beta(3, 3)`), which represents a weak belief that
the true rate is centered around 0.5.

The function calculates the analytical posterior distributions for the detection rates in both
the sample group (`θ_sample`) and the control group (`θ_control`). It then uses numerical
integration to compute the posterior probability `p = P(θ_sample > θ_control | data)`.

Finally, it computes the Bayes Factor (BF₁₀) for the one-sided hypothesis:
- H₁: The detection rate in the sample is greater than in the control (θ_sample > θ_control).
- H₀: The detection rate in the sample is not greater than in the control (θ_sample ≤ θ_control).

# Arguments
- `data::InteractionData`: The main data structure containing protein interaction data.
- `idx::Int`: The index of the specific protein to be analyzed from the `data` structure.
- `n_control::Int`: The number of data points in the sample group's data array.
- `n_sample::Int`: The number of data points in the control group's data array.
- `create_plot::Bool=false`: Whether to create visualization plot (expensive, defaults to false).

# Returns
A `Tuple{Float64, Float64, Union{Plots.Plot, Nothing}}` containing:
1.  `BF::Float64`: The Bayes Factor in favor of H₁. A value > 1 indicates evidence for the sample group having a higher detection rate.
2.  `p::Float64`: The posterior probability that the sample detection rate is greater than the control rate.
3.  `plt::Union{Plots.Plot, Nothing}`: A plot object from `StatsPlots.jl` visualizing the prior and the two posterior distributions, or `nothing` if `create_plot=false`.
"""
function betabernoulli(data::InteractionData, idx::I, n_control::I, n_sample::I;
                       create_plot::Bool=false,
                       prior_alpha::Float64=3.0, prior_beta::Float64=3.0) where {I <: Int}
    # extract protein name
    proteinName = getProteinData(data, idx).name

    # get to data for model
    counts = count_detections(data, idx, n_sample, n_control)
    k_sample = counts.k_sample          # successes in sample group
    k_control = counts.k_control        # successes in controll group
    f_samples = counts.f_sample         # failures in sample group
    f_controls = counts.f_control       # failures in sample group

    # return nothing if the number of failures in the sample or control group are blow 0. 
    # This can only happen during the permutations when generating H0
    if f_samples < 0 || f_controls < 0
        return missing, missing, missing
    end

    # Posterior for sample (θx)
    θx = Beta(prior_alpha + k_sample, prior_beta + f_samples)

    # Posterior for control (θy)
    θy = Beta(prior_alpha + k_control, prior_beta + f_controls)

    # Compute P(θ_sample > θ_control) using analytical integration
    p = prob_beta_greater(θx, θy)
    posterior_odds = p / (1 - p)
    prior_odds = 1.0

    BF = posterior_odds / prior_odds

    # Create plot only if requested (expensive operation)
    plt = if create_plot
        StatsPlots.plot(
            [θx, θy, Beta(prior_alpha, prior_beta)],
            label = ["θ_sample" "θ_controls" "Prior Beta($(prior_alpha),$(prior_beta))"],
            fill = true, fillalpha = 0.5,
            lgendfontsize = 9, legend_background_color	= nothing,
            legend_foreground_color = nothing,
            legend = :topleft, title = proteinName,
            xlabel = "θ", xlim = (0.0, 1.0),
            ylabel = "Density"
        )
    else
        nothing
    end

    return BF, p, plt
end

"""
    count_detections(data::InteractionData, idx::I,  n_sample::I, n_control::I = 0) where I<:Integer

Counts the number of detections (successes) and failures for a given protein in sample and control groups.

Detections are defined as non-missing values in the protein's data. Failures are calculated based on the total number of replicates minus the detections, plus any dummy failures provided.

# Arguments
- `data::InteractionData`: The main data structure containing all interaction data.
- `idx::Integer`: The index of the protein of interest within the `data` structure.
- `n_samples::Integer`: Total number of samples in the sample group.
- `n_controls::Integer`: Total number of controls in the sample group.

# Returns
A `NamedTuple` with the following fields:
- `k_sample`: Number of successes (detections) in the sample group.
- `k_control`: Number of successes (detections) in the control group.
- `f_sample`: Number of failures in the sample group.
- `f_control`: Number of failures in the control group.
"""
function count_detections(data::InteractionData, idx::I, n_sample::I, n_control::I) where I<:Integer
     # Extract data from the InteractionData structure
     proteinData = getProteinData(data, idx)  
     interactome_sample::Array{Union{Missing, Float64}, 3}  = getSampleMatrix(proteinData)
     interactome_control::Array{Union{Missing, Float64}, 3} = getControlMatrix(proteinData)

     # 2. Count the number of DETECTIONS (successes), which are the NON-missing values
     k_sample  = count(!ismissing, interactome_sample)
     k_control = count(!ismissing, interactome_control)
     
     # 3. Count failures 
     f_sample  = n_sample - k_sample
     f_control = n_control - k_control
     
     return (
        k_sample = k_sample, 
        k_control = k_control, 
        f_sample = f_sample,
        f_control = f_control
        )
end