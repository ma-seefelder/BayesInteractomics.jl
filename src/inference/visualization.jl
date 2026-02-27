function checkimagefile(file::Union{String, Nothing})::Nothing
    isnothing(file) && return nothing
    if !endswith(file, ".png")
        @error "File $file has an unsupported extension (only .png allowed)"
    elseif isfile(file)
        @error "File $file already exists"
    end
    return nothing
end

computefiguresize(n) = (300 * ceil(Int64, n / 2), 300 * ceil(Int64, n / 2))

function remove_missing(x::AbstractArray{Union{Float64, Missing}}, y::AbstractArray{Union{Float64, Missing}})
    @assert length(x) == length(y) "Vectors x and y must be of equal length"

    # create a boolean mask for complete colnames_slope_bf
    complete_mask = .!ismissing.(x) .& .!ismissing.(y)

    x = convert(Vector{Float64}, x[complete_mask])
    y = convert(Vector{Float64}, y[complete_mask])
    return x,y 
end

"""
    plot_inference_results(result::HBMResult; file::Union{String, Nothing} = nothing)

    Description:
    This function plots the inference results for a given protein. 

    Args:
      - result<:HBMResult: result object of type HBMResult
      - protein_name<:AbstractString: name of the protein

    Keyword Args:
      - file<:AbstractString: name of the file to save the plot. If not provided, the plot is not saved

    Returns:
      - nothing
"""
function plot_inference_results(result::HBMResult; file::Union{String, Nothing} = nothing)
    posterior::InferenceResult, prior::InferenceResult = result.posterior, result.prior
    
    checkimagefile(file)
    # retrieve distributions
    posterior_samples = to_normal(posterior, :μ_sample)
    posterior_controls = to_normal(posterior, :μ_control)
    prior_samples = to_normal(prior, :μ_sample)
    prior_controls = to_normal(prior, :μ_control)

    plt = [
        plot_inference_results(
            posterior_samples[i], posterior_controls[i], prior_samples[i], prior_controls[i], "μ[$i]"
        ) for i in eachindex(posterior_samples)]

    result = StatsPlots.plot(plt..., size = computefiguresize(length(posterior_samples)))

    !isnothing(file) ? StatsPlots.savefig(result, file) : nothing
    return result
end

"""
    plot_inference_results(
    posterior_samples::Normal, posterior_controls::Normal, 
    prior_samples::Normal, prior_controls::Normal, title::S
    ) where {S<:AbstractString}

    Description:
    This function plots the inference results for a given protein. 

    Args:
      - posterior_samples<:Normal: posterior inference result object of type InferenceResult
      - posterior_controls<:Normal: posterior inference result object of type InferenceResult
      - prior_samples<:Normal: prior inference result object of type InferenceResult
      - prior_controls<:Normal: prior inference result object of type InferenceResult
      - title<:AbstractString: title of the plot

    Returns:
      - Plot object
"""
function plot_inference_results(
    posterior_samples::Normal, posterior_controls::Normal, 
    prior_samples::Normal, prior_controls::Normal, title::S
    ) where {S<:AbstractString}

    plt = StatsPlots.plot(
        posterior_samples, label = L"P(μ_{s}|data)", 
        title = title, xlabel = "μ", ylabel = "Density", 
        legend = :topright, legendfontsize = 8, 
        foreground_color_legend = nothing, background_color_legend = nothing,
        fill = true, fillalpha = 0.5, linealpha = 0.5,
        size = (450, 300)
    )

    StatsPlots.plot!(
        plt, posterior_controls, label = L"P(μ_{c}|data)", 
        fill = true, fillalpha = 0.5, linealpha = 0.5
        )

    StatsPlots.plot!(
        plt, prior_samples, label = L"P(μ_{s})",
        fill = true, fillalpha = 0.5, linealpha = 0.5
        )

    StatsPlots.plot!(
        plt, prior_controls, label = L"P(μ_{c})",
        fill = true, fillalpha = 0.5, linealpha = 0.5
        )

    return plt

end

"""
    plot_inference_results(result::BayesResult; file::Union{String, Nothing} = nothing)
    
    Description:
    This function plots the inference results for a given protein. 

    Args:
      - result<:BayesResult: result of Bayesian inference for a given protein

    Keyword Args:
      - file<:AbstractString: name of the file to save the plot. If not provided, the plot is not saved

    Returns:
      - Plot of posterior and prior distributions
"""
function plot_inference_results(result::BayesResult; file::Union{String, Nothing} = nothing) 
    return plot_inference_results(result.hbm_result; file = file)
end

function get_axis_limits(log2fc, observed_log2fc)
    # check for null inputs
    isnothing(log2fc) && throw(ArgumentError("Input cannot be nothing"))
    # concatenate everything
    x = [0.0]
    [append!(x, log2fc[pos]) for pos in eachindex(log2fc)]
    append!(x, skipmissing(observed_log2fc))

    min::Float64 = quantile(x, 0.001)
    max::Float64 = quantile(x, 0.999)
    
    return (min, max, nothing, nothing)
end

############################################################################
# Plot log2FC
############################################################################

"""
    plot_log2fc(
    posterior::Vector{MixtureModel}, prior::InferenceResult; 
    file::Union{S, Nothing} = nothing, threshold::F = 0.0
    ) where {S<:AbstractString, F<:AbstractFloat}

    Description: 
    This function plots the inference results (log2-fold-changes) for a given protein with
    multiple imputation.
    
    Args:
    - posterior:: result of the Bayesian inference for a given protein. A Vector{MixtureModel} where each
                  element corresponds to a parameter.
    - prior::     prior inference result object of type InferenceResult

    Keyword Args:
    - file<:AbstractString: name of the file to save the plot. If not provided, the plot is not saved
    - threshold<:AbstractFloat: threshold value for log2FC
"""
function plot_log2fc(
    posterior::Vector{MixtureModel}, prior::InferenceResult; 
    file::Union{S, Nothing} = nothing, threshold::F = 0.0
    ) where {S<:AbstractString, F<:AbstractFloat}

    # compute prior log2fc
    priorlog2FC     = log2FC.(prior.posteriors[:μ_sample], prior.posteriors[:μ_control])
    nparameters     = length(priorlog2FC)

    # generate plot
    figure_size = computefiguresize(length(priorlog2FC)); checkimagefile(file)  
    plts = [plot_log2fc(posterior[i], priorlog2FC[i], threshold, i) for i in 1:nparameters]
    
    plt = StatsPlots.plot(
        plts..., size = figure_size, figure_title = L"log_{2}FC results for $protein_name" 
    )

    !isnothing(file) && StatsPlots.savefig(plt, file)
    return plt
end

"""
    plot_log2fc(result::BayesResult, reallog2fc::Vector{Union{Missing, F}}; file::Union{S, Nothing} = nothing, threshold::F = 0.0) where {S<:AbstractString, F<:AbstractFloat}

    Description:
    This function plots the inference results (log2-fold-changes) for a given protein.  

    Args:
      - result<:BayesResult:            result of Bayesian inference for a given protein. 
      - reallog2fc<:AbstractFloat:      observed log2-fold-change for the protein.

    Keyword Args:
      - file<:AbstractString:       name of the file to save the plot. If not provided, the plot is not saved.
      - threshold<:AbstractFloat:   threshold above which a fold change is considered significant (i.e., position 
                                    of the red dashed vertical line). Default is 0.0. The same threshold should be 
                                    used for all proteins and should be identical to the one used to compute the 
                                    posterior probability and Bayes Factor.  

    Returns:
      - plot of the log2-fold-change a given protein
"""
function plot_log2fc(result::BayesResult, reallog2fc::Vector{Union{Missing, F}}; file::Union{S, Nothing} = nothing, threshold::F = 0.0) where {S<:AbstractString, F<:AbstractFloat}
    return plot_log2fc(result.hbm_result, reallog2fc; file = file, threshold = threshold)
end
    

"""
    plot_log2fc(result::HBMResult, reallog2fc::Vector{Union{Missing, F}}; file::Union{S, Nothing} = nothing, threshold::F = 0.0) where {S<:AbstractString, F<:AbstractFloat}
    
    Description:
    This function plots the inference results (log2-fold-changes) for a given protein.  

    Args:
      - result<:HBMResult:              result of Bayesian inference for a given protein. 
      - reallog2fc<:AbstractFloat:      observed log2-fold-change for the protein.

    Keyword Args:
      - file<:AbstractString:       name of the file to save the plot. If not provided, the plot is not saved.
      - threshold<:AbstractFloat:   threshold above which a fold change is considered significant (i.e., position 
                                    of the red dashed vertical line). Default is 0.0. The same threshold should be 
                                    used for all proteins and should be identical to the one used to compute the 
                                    posterior probability and Bayes Factor.  

    Returns:
      - plot of the log2-fold-change a given protein
"""
function plot_log2fc(result::HBMResult, reallog2fc::Vector{Union{Missing, F}}; file::Union{S, Nothing} = nothing, threshold::F = 0.0) where {S<:AbstractString, F<:AbstractFloat}  
    figure_size = computefiguresize(length(reallog2fc))    

    checkimagefile(file)  
    posterior, prior = result.posterior, result.prior
    posteriorlog2FC = log2FC.(posterior.posteriors[:μ_sample], posterior.posteriors[:μ_control])
    priorlog2FC     = log2FC.(prior.posteriors[:μ_sample], prior.posteriors[:μ_control])

    plts = [plot_log2fc(posteriorlog2FC[i], priorlog2FC[i], reallog2fc[i], threshold, i) for i in 1:length(posteriorlog2FC)]
    
    plt = StatsPlots.plot(
        plts..., size = figure_size, figure_title = L"log_{2}FC results for $protein_name" 
    )

    if !isnothing(file)
        StatsPlots.savefig(plt, file)
    end

    return plt
end

"""
    plot_log2fc(posterior::Normal{F}, prior::Normal{F},reallog2fc::Union{Missing, F}, threshold::F = 0.0) where {F<:AbstractFloat}
    
    Description:
        This function plots the inference results (log2-fold-changes) for a given protein.  

    Args:
      - posterior:Normal{F}             posterior distribution of the log2-fold-change
      - prior:                          prior distribution of the log2-fold-change
      - reallog2fc<:AbstractFloat:      observed log2-fold-change for the protein.
      - i:                              index of the parameter

    Keyword Args:
      - threshold<:AbstractFloat:   threshold above which a fold change is considered significant (i.e., position 
                                    of the red dashed vertical line). Default is 0.0. The same threshold should be 
                                    used for all proteins and should be identical to the one used to compute the 
                                    posterior probability and Bayes Factor.  

    Returns:
      - plot of the log2-fold-change a given protein
"""
function plot_log2fc(posterior::Normal{F}, prior::Normal{F},reallog2fc::Union{Missing, F}, threshold::F, i) where {F<:AbstractFloat}
    plt = StatsPlots.plot(
        posterior, label = L"P(log_{2}FC|data)", fill = true, fillalpha = 0.5,
        lgendfontsize = 9, legend_background_color	= nothing, legend_foreground_color = nothing,
        legend = :topleft, title = "$i",
        xlabel = "log2FC", ylabel = "Density"
        )

    StatsPlots.plot!(plt, prior, label = L"P(log_{2}FC)", fill = true, fillalpha = 0.5)
    StatsPlots.vline!([threshold], color = :red, label = L"\theta_{log_{2}FC}", linestyle = :dash)
    ismissing(reallog2fc) || StatsPlots.vline!([reallog2fc], color = :black, label = L"true_{log_{2}FC}", linestyle = :dash)
    return plt
end

"""
    plot_log2fc(posterior::MixtureModel, prior::MixtureModel, threshold::F, i; nsamples = 1_000_000) where {F<:AbstractFloat}
    
    Description:
        This function plots the inference results (log2-fold-changes) for a given protein. This method should be used if missing
            values have been imputed by Multiple Imputation   

    Args:
      - posterior::MixtureModel     posterior distribution of the log2-fold-change
      - prior::Normal               prior distribution of the log2-fold-change
      - i:                          index of the parameter
      - threshold<:AbstractFloat:   threshold above which a fold change is considered significant (i.e., position 
                                    of the red dashed vertical line). Default is 0.0. The same threshold should be 
                                    used for all proteins and should be identical to the one used to compute the 
                                    posterior probability and Bayes Factor.  

    Keyword Args:
      - nsamples:                   Number of samples that are drawn from the MixtureModel for plotting

    Returns:
      - plot of the log2-fold-change a given protein
"""
function plot_log2fc(posterior::MixtureModel, prior::Normal, threshold::F, i; nsamples = 1_000_000) where {F<:AbstractFloat}
    samples = rand(posterior, nsamples)
    
    plt = StatsPlots.density(
        samples, label = L"P(log_{2}FC|data)", fill = true, fillalpha = 0.5,
        lgendfontsize = 9, legend_background_color	= nothing, legend_foreground_color = nothing,
        legend = :topleft, title = "$i",
        xlabel = "log2FC", ylabel = "Density"
        )

    StatsPlots.plot!(plt, prior, label = L"P(log_{2}FC)", fill = true, fillalpha = 0.5)
    StatsPlots.vline!([threshold], color = :red, label = L"\theta_{log_{2}FC}", linestyle = :dash)
    return plt
end

###############################################################
## regression plot

"""
    plot_regression(result::BayesResult, x::Array, y::Array; nlines = 100, file::Union{S, Nothing} = nothing) where {S<:AbstractString}

"""
function plot_regression(result::BayesResult, x::Array, y::Array; nlines = 100, file::Union{S, Nothing} = nothing) where {S<:AbstractString} 
    regression_result = result.regression_result
    protein_name = result.protein_name
    
    return plot_regression(regression_result, protein_name, x, y; nlines = nlines, file = file)
end


"""
    plot_regression(result::RegressionResultSingleProtocol, protein_name, x, y; nlines=100, file=nothing)

Plot regression model fit for a single-protocol experiment.

Draws `nlines` posterior predictive regression lines over the observed data,
showing the posterior uncertainty in slope and intercept.

# Arguments
- `result::RegressionResultSingleProtocol`: Fitted regression result.
- `protein_name::String`: Protein name for the plot title.
- `x::Array`: Reference (bait) protein abundance values.
- `y::Array`: Prey protein abundance values.

# Keywords
- `nlines::Int=100`: Number of posterior regression lines to draw.
- `file::Union{String, Nothing}=nothing`: If provided, save the plot to this path.

# Returns
- A StatsPlots plot object.
"""
function plot_regression(
    result::RegressionResultSingleProtocol,
    protein_name::S,
    x::Array, y::Array; nlines = 100,
    file::Union{S, Nothing} = nothing
    ) where {S<:AbstractString}

    posterior, prior = result.posterior, result.prior

    # convert x and y to one dimensional Vectors
    x = flatten_rows(x[1,:,:])
    y = flatten_rows(y[1,:,:])

    missing_value = findall(x -> ismissing(x), x) ∪ findall(y -> ismissing(y), y)

    x = x[setdiff(1:length(x), missing_value)]
    y = y[setdiff(1:length(y), missing_value)]

    # ---- plot 1: scatter plot of intensity with regression line
    minimum_x, maximum_x = findmin(skipmissing(x))[1] - 1, findmax(skipmissing(x))[1] -1 
    plt_left = StatsPlots.plot(
        x, y, seriestype = :scatter, 
        xlabel = "Intensity (Reference)", ylabel = "Intensity ($protein_name)",
        label = nothing,
        lgendfontsize = 9, legend_background_color	= nothing, legend_foreground_color = nothing,
        title = "", xlim = (minimum_x, maximum_x), 
        ylim = (findmin(skipmissing(y))[1] - 1, findmax(skipmissing(y))[1] + 1)
    )
    
    x_regressionlines = collect(minimum_x:0.001:maximum_x)

    posterior_samples = (
        rand(posterior.posteriors[:α], nlines), 
        rand(posterior.posteriors[:β], nlines)
        )
    
    for line ∈ 1:nlines
        y_est = (x_regressionlines .* posterior_samples[1][line]) .+ posterior_samples[2][line]
        StatsPlots.plot!(
            plt_left, x_regressionlines, y_est, alpha = 0.2, label = nothing
            )
    end

    # ---- plot 2: density of slope parameter
    plt_middle = StatsPlots.plot(
        to_normal(posterior.posteriors[:α]), 
        label = L"P(\alpha|data)",
        fill = true, fillalpha = 0.5
    )

    StatsPlots.plot!(
        plt_middle, to_normal(prior.posteriors[:α]), 
        label = L"P(\alpha)",
        fill = true, fillalpha = 0.5
    )

    # ---- plot 3: density of intercept parameter
    plt_right = StatsPlots.plot(
        to_normal(posterior.posteriors[:β]), 
        label = L"P(\beta|data)",
        fill = true, fillalpha = 0.5
    )

    StatsPlots.plot!(
        plt_right, 
        to_normal(prior.posteriors[:α]), 
        label = L"P(\beta)",
        fill = true, fillalpha = 0.5
        )


    # ---- plot 3: compose final plot
    plt_final = StatsPlots.plot(
        plt_left, plt_middle, plt_right,
        layout = (3,1)
        )

    if !isnothing(file) 
        StatsPlots.savefig(plt_final, file)
    end

    return plt_final
end



"""
    plot_regression(result::RegressionResultMultipleProtocols, protein_name, x, y; nlines=100, file=nothing)

Plot regression model fit for a multi-protocol experiment.

Creates a 3-panel figure:
1. Global regression with hyperprior lines
2. Per-protocol regression lines
3. Residual diagnostics

# Arguments
- `result::RegressionResultMultipleProtocols`: Fitted regression result.
- `protein_name::String`: Protein name for the plot title.
- `x::Array`: Reference (bait) protein abundance values (protocols × experiments × replicates).
- `y::Array`: Prey protein abundance values (protocols × experiments × replicates).

# Keywords
- `nlines::Int=100`: Number of posterior regression lines to draw per protocol.
- `file::Union{String, Nothing}=nothing`: If provided, save the plot to this path.

# Returns
- A StatsPlots plot object.
"""
function plot_regression(
    result::RegressionResultMultipleProtocols, protein_name::S,
    x::Array, y::Array; nlines = 100,
    file::Union{S, Nothing} = nothing
    ) where {S<:AbstractString}

    # retrieve distributions
    posterior = result.posterior
    prior     = result.prior

    checkimagefile(file)  
    nprotocols = length(posterior.posteriors[:α])
    # x-y plot with regression lines
    plt_top = plot_regression_top(posterior, protein_name, x, y; nlines = nlines)
    # density of slope hyperparameter
    plt_middle = StatsPlots.plot(
        to_normal(posterior.posteriors[:μ_α]), 
        label = L"\mu_{\alpha}|data",
        fill = true, fillalpha = 0.5,
        lgendfontsize = 9, legend_background_color	= nothing, legend_foreground_color = nothing,
        legend = :topleft, title = "",
        xlabel = "slope α", ylabel = "Density"
        )

    StatsPlots.plot!(
        plt_middle, to_normal(prior.posteriors[:μ_α]), 
        label = L"\mu_{\alpha}", fill = true, fillalpha = 0.5 
        )

    # lower plot
    plt_lower = []
    for i in 1:nprotocols
        pposterior = posterior.posteriors[:α][i] |> to_normal
        pprior = prior.posteriors[:α][i] |> to_normal

        plt_temp = StatsPlots.plot(
            pposterior, label = L"\alpha|data",
            fill = true, fillalpha = 0.5, lgendfontsize = 9, 
            legend_background_color	= nothing, legend_foreground_color = nothing,
            legend = :topleft, title = "Protocol $i",
            xlabel = "α", ylabel = "Density",
            xlim = (-1.0,1.0)
        )

        StatsPlots.plot!(plt_temp, pprior, label = L"\alpha")
        push!(plt_lower, plt_temp)
    end

    figure_size = computefiguresize(length(nprotocols))    
    plt_lower = StatsPlots.plot(plt_lower..., size = figure_size )

    figure_size = (figure_size[1] * 3 , figure_size[2] * 3)
    figure = StatsPlots.plot(plt_top, plt_middle, plt_lower, layout = (3,1), size = figure_size)

    if !isnothing(file) 
        StatsPlots.savefig(figure, file)
    end
    
    return figure
end

"""
    plot_regression_top(posterior, protein_name, x, y; nlines=100)

Internal helper: plot the global (hyperprior) regression panel for multi-protocol data.

Draws posterior predictive regression lines from the global slope and intercept
distributions, overlaid on per-protocol observed data points.
"""
function plot_regression_top(posterior, protein_name, x, y; nlines = 100)
    x1, y1 = vec(x[1,:,:]), vec(y[1,:,:])
    x1, y1 = remove_missing(x1, y1)
    color = StatsPlots.distinguishable_colors(size(x,1))
    minimum_x, maximum_x = findmin(skipmissing(x))[1] - 1, findmax(skipmissing(x))[1] -1 
    x_regressionlines = collect(minimum_x:0.001:maximum_x)
    
    ##################################################
    # plot for hyperprior

    plt = StatsPlots.plot(
        x1, y1, seriestype = :scatter, 
        xlabel = "Intensity (Reference)", ylabel = "Intensity ($protein_name)",
        label = "Protocol 1", color = color[1],
        lgendfontsize = 9, legend_background_color	= nothing, legend_foreground_color = nothing,
        title = "", xlim = (minimum_x, maximum_x), 
        ylim = (findmin(skipmissing(y))[1] - 1, findmax(skipmissing(y))[1] + 1)
        )

    posterior_samples = (
        rand(posterior.posteriors[:α][1], nlines), 
        rand(posterior.posteriors[:β][1], nlines)
        )
    
    for line ∈ 1:nlines
        y_est = (x_regressionlines .* posterior_samples[1][line]) .+ posterior_samples[2][line]
        StatsPlots.plot!(
            plt, x_regressionlines, y_est, alpha = 0.2, 
            color = color[1], label = nothing
            )
    end

    ############################################################################
    # plots for individual protocols

    for protocol ∈ axes(x,1)
        protocol == 1 && continue
        x1, y1 = vec(x[protocol,:,:]), vec(y[protocol,:,:])
        x1, y1 = remove_missing(x1, y1)
        StatsPlots.plot!(plt,
            x1, y1, seriestype = :scatter, 
            xlabel = "Intensity (Reference)", ylabel = "Intensity ($protein_name)",
            label = "Protocol $protocol", color = color[protocol]
        )

        # regression lines
        posterior_samples = (
            rand(posterior.posteriors[:α][protocol], nlines), 
            rand(posterior.posteriors[:β][protocol], nlines)
        )
    
        for line ∈ 1:nlines
            y_est = @. x_regressionlines * posterior_samples[1][line] + posterior_samples[2][line]

            StatsPlots.plot!(
                plt, x_regressionlines, y_est, alpha = 0.2, 
                color = color[protocol], label = nothing
                )
        end
    end

    return plt
end


"""
    plot_bayesrange(posterior::InferenceResult, prior::InferenceResult, protocol_positions::Vector{I}, protein_name::S; file::Union{S, Nothing} = nothing) where {S<:AbstractString, I<:Int64}

    Description:
        Plot the Bayes Factor and posterior probability for a given range of log2FC values, a given protein.

    Args:
      - result::BayesResult:            result object of type BayesResult
      - protocol_positions<:Vector{I}:  vector of protocol positions to plot
      - protein_name<:AbstractString:   name of the protein.

    Keyword Args:
      - file<:Union{S, Nothing}:        path to file where figure should be saved
      - min<:AbstractFloat:             minimum value of log2FC
      - max<:AbstractFloat:             maximum value of log2FC
      - stepsize<:AbstractFloat:        stepsize between min and max

    Returns:
      - fig<:Makie.Figure:         figure object
"""
function plot_bayesrange(
    result::BayesResult, protocol_positions::Vector{I}, protein_name::S;
    file::Union{S, Nothing} = nothing, min_value::F = -10.0, max_value::F = 10.0, stepsize::F = 0.1
    ) where {S<:AbstractString, I<:Int64, F<:AbstractFloat}

    return plot_bayesrange(result.hbm_result, protocol_positions, protein_name; 
        file = file, min_value = min_value, max_value = max_value, stepsize = stepsize)
end


"""
    plot_bayesrange(result::HBMResult, protocol_positions, protein_name; kwargs...)

Plot Bayes factor as a function of the log₂FC threshold for an HBM result.

Computes and plots how the enrichment Bayes factor changes across a range of
threshold values, separately for each protocol and the global estimate.
"""
function plot_bayesrange(
    result::HBMResult, protocol_positions::Vector{I}, protein_name::S;
    file::Union{S, Nothing} = nothing, min_value::F = -10.0, max_value::F = 10.0, stepsize::F = 0.1
    ) where {S<:AbstractString, I<:Int64, F<:AbstractFloat}
    checkimagefile(file)

    # compute Bayes Factors and posterior probabilities for plotting
    bf, posterior_probs = computeBayesFactorPlot(
        result, protocol_positions, 
        min_value = min_value, max_value = max_value, 
        stepsize = stepsize
        )
    return plot_bayesrange(bf, posterior_probs, protocol_positions, protein_name, file = file)
end

"""
    plot_bayesrange(posterior, prior, protocol_positions, protein_name; kwargs...)

Plot Bayes factor range from raw posterior/prior `MixtureModel` and `Normal` vectors.

Low-level method that computes Bayes factors across thresholds from the given
distributions and delegates to the matrix-based `plot_bayesrange` method.
"""
function plot_bayesrange(
    posterior::Vector{MixtureModel}, prior::Vector{Normal{F}},
    protocol_positions::Vector{I}, protein_name::S;
    file::Union{S, Nothing} = nothing, min_value::F = -10.0, max_value::F = 10.0, stepsize::F = 0.1
    ) where {S<:AbstractString, I<:Int64, F<:AbstractFloat}

    checkimagefile(file)
    # compute Bayes Factors and posterior probabilities for plotting
    bf, posterior_probs = computeBayesFactorPlot(
        posterior, prior, protocol_positions, 
        min_value = min_value, max_value = max_value, 
        stepsize = stepsize
        )
    return plot_bayesrange(bf, posterior_probs, protocol_positions, protein_name, file = file)
end

"""
    plot_bayesrange(bf::Matrix, posterior_probs::Matrix, protocol_positions, protein_name; kwargs...)

Plot pre-computed Bayes factor and posterior probability matrices across thresholds.

Creates a two-panel figure: (1) log₁₀ Bayes factor vs. threshold,
(2) posterior probability vs. threshold, with one line per protocol.

# Arguments
- `bf::Matrix{Float64}`: Bayes factors (protocols × thresholds).
- `posterior_probs::Matrix{Float64}`: Posterior probabilities (protocols × thresholds).
- `protocol_positions::Vector{Int}`: Protocol position indices.
- `protein_name::String`: Protein name for the plot title.

# Keywords
- `file::Union{String, Nothing}=nothing`: If provided, save the plot to this path.
- `min_value::Float64=-10.0`: Minimum threshold value.
- `max_value::Float64=10.0`: Maximum threshold value.
- `stepsize::Float64=0.1`: Threshold step size.
"""
function plot_bayesrange(
    bf::Matrix{F}, posterior_probs::Matrix{F}, protocol_positions::Vector{I}, protein_name::S;
    file::Union{S, Nothing} = nothing, min_value::F = -10.0, max_value::F = 10.0, stepsize::F = 0.1
) where {S<:AbstractString, I<:Int64, F<:AbstractFloat}

    x_data = collect(min_value:stepsize:max_value)
    log10_bf = log10.(bf) 

    # plot Bayes Factors
    plt_bf = StatsPlots.plot(
        x_data, log10_bf[1,:], 
        title = "Bayes Factor for $protein_name", 
        label = "Whole dataset", 
        legend = :outertopright,
        ylabel = "log10(Bayes Factor)", xlabel = "log2(Fold Change)",
        legend_background_color = nothing, legend_foreground_color = nothing
    )

    [StatsPlots.plot!(plt_bf, x_data, log10_bf[i,:], label = L"Protocol %$(i-1)") for i ∈ 2:size(log10_bf, 1)]
    StatsPlots.hline!([0], linestyle = :dash, label = L"BF_{log_{2}FC} = 1")
    StatsPlots.hline!([0.5], linestyle = :dash, label = L"BF_{log_{2}FC} = \sqrt{10}")
    StatsPlots.hline!([1], linestyle = :dash, label = L"BF_{log_{2}FC} = 10")
    StatsPlots.hline!([2], linestyle = :dash, label = L"BF_{log_{2}FC} = 100")

    # plot posterior probabilities
    plt_prob = StatsPlots.plot(
        x_data, posterior_probs[1,:], 
        title = "Posterior Probability for $protein_name", 
        label = L"P(log2FC > x | data)", 
        legend = :outertopright,
        ylabel = "Posterior Probability", xlabel = "log2(Fold Change)",
        legend_background_color = nothing, legend_foreground_color = nothing
    )

    for i ∈ axes(posterior_probs, 1)
        i == 1 && continue
        StatsPlots.plot!(
            plt_prob, x_data, posterior_probs[i,:], 
            label = L"P(log2FC_{%$i} > x | data)"
            ) 
    end

    plt = StatsPlots.plot(plt_bf, plt_prob, layout = (2,1), size = (600, 1200))

    if !isnothing(file)
        StatsPlots.savefig(plt, file)
    end

    return plt
end

"""
    computeBayesFactorPlot(
    result::HBMResult, protocol_positions::Vector{I};
    min_value::F = -10.0, max_value::F = 10.0, stepsize::F = 0.1
    ) where {I<:Int64, F<:AbstractFloat}

    Computes the Bayes Factors and posterior probabilities for a given range of log2FC thresholds of
        the whole dataset and the individual protocols.

    Args:
      - result<:HBMResult:              result object of type `HBMResult`
      - protocol_positions<:Vector{I}:  vector of protocol positions to plot

    Keyword Args:
      - min_value<:AbstractFloat:       minimum value of log2FC
      - max_value<:AbstractFloat:       maximum value of log2FC
      - stepsize<:AbstractFloat:        stepsize between min and max

    Returns:
      - bfs<:Matrix{F}:                 matrix of Bayes Factors
      - posterior_probs<:Matrix{F}:     matrix of posterior probabilities
"""
function computeBayesFactorPlot(
    result::HBMResult, protocol_positions::Vector{I};
    min_value::F = -10.0, max_value::F = 10.0, stepsize::F = 0.1
) where {I<:Int64, F<:AbstractFloat}
    posterior, prior = result.posterior, result.prior
    pushfirst!(protocol_positions, 1)
    bf_thresholds = collect(min_value:stepsize:max_value)
    bfs = zeros(Float64, size(protocol_positions, 1), length(bf_thresholds))
    posterior_probs = zeros(Float64, size(protocol_positions, 1), length(bf_thresholds))

    labels = ["Whole dataset"]
    [push!(labels, "Protocol $i") for i ∈ 1:(size(bfs, 1)-1)]
 
    for i ∈ axes(bfs,2)
        bf, posterior_prob, _ = BayesFactorHBM(result, threshold = bf_thresholds[i])
        bfs[:,i], posterior_probs[:,i] = bf[protocol_positions], posterior_prob[protocol_positions]
    end
    return bfs, posterior_probs
end

"""
    computeBayesFactorPlot(posterior, prior, protocol_positions; kwargs...)

Compute Bayes factor and posterior probability matrices from raw `MixtureModel`/`Normal` vectors.

Low-level method used by `plot_bayesrange` when called with raw distributions
instead of an `HBMResult`.
"""
function computeBayesFactorPlot(
    posterior::Vector{MixtureModel}, prior::Vector{Normal{F}},
    protocol_positions::Vector{I};
    min_value::F = -10.0, max_value::F = 10.0, stepsize::F = 0.1
) where {I<:Int64, F<:AbstractFloat}
    pushfirst!(protocol_positions, 1)
    bf_thresholds = collect(min_value:stepsize:max_value)
    bfs = zeros(Float64, size(protocol_positions, 1), length(bf_thresholds))
    posterior_probs = zeros(Float64, size(protocol_positions, 1), length(bf_thresholds))

    labels = ["Whole dataset"]
    [push!(labels, "Protocol $i") for i ∈ 1:(size(bfs, 1)-1)]
 
    for i ∈ axes(bfs,2)
        bf, posterior_prob, _ = calculate_bayes_factor(posterior, prior, threshold = bf_thresholds[i])
        bfs[:,i], posterior_probs[:,i] = bf[protocol_positions], posterior_prob[protocol_positions]
    end
    return bfs, posterior_probs
end


"""
    write_txt(; 
    filename::S, protein_name::S,
    HBM_stats, regression_stats, 
    bf::Matrix{Float64}
    ) where {S<:AbstractString}

"""
function write_txt(; 
    filename::S, protein_name::S,
    HBM_stats, regression_stats, 
    bf, bfR, nprotocols::I
    ) where {S<:AbstractString, I<:Int}


    if nprotocols == 1
        write_txt_singleprotocol(
            filename = filename, 
            protein_name = protein_name,
            HBM_stats = HBM_stats, 
            regression_stats = regression_stats, 
            bf = bf, bfR = bfR
            )
        return nothing
    end

    if !isfile(filename)
        # HBM
        colnames_log2FC_median = join(["median_log2FC_$i" for i in 1:size(HBM_stats[:median_log2FC], 1)], "|")
        colnames_log2FC_mean = join(["mean_log2FC_$i" for i in 1:size(HBM_stats[:mean_log2FC], 1)], "|")
        colnames_log2FC_pd = join(["pd_log2FC_$i" for i in 1:size(HBM_stats[:pd], 1)], "|")
        colnames_log2FC_pd_directions = join(["pd_direction_log2FC_$i" for i in 1:size(HBM_stats[:pd_direction], 1)], "|")
        colnames_log2FC_creds = join(["ci_log2FC_$(i)" for i in 1:size(HBM_stats[:credible_interval], 1)], "|")
        colnames_log2FC_sd = join(["sd_log2FC_$i" for i in 1:size(HBM_stats[:sd_log2FC], 1)], "|")

        colnames_log2FC_bf_g0 = join(["bf_log2FC_$(i)_>0.0" for i in 1:size(bf, 1)], "|")
        
        # regression
        colnames_slope_median   = "median_slope"
        colnames_slope_mean     = "mean_slope"
        colnames_slope_pd       = "pd_slope"
        colnames_slope_pd_dir   = "pd_slope_dir"
        colnames_slope_sd       = "sd_slope" 
        colnames_slope_bf       = "bf_slope"

        colnames_slopes_median      =   join(["median_slope_$(i)" for i in 1:nprotocols], "|")
        colnames_slopes_mean        =   join(["mean_slope_$(i)" for i in 1:nprotocols], "|")
        colnames_slopes_pd          =   join(["pd_slope_$(i)" for i in 1:nprotocols], "|")
        colnames_slopes_pd_dir      =   join(["pd_slope_dir_$(i)" for i in 1:nprotocols], "|")
        colnames_slopes_sd          =   join(["sd_slope_$(i)" for i in 1:nprotocols], "|")
        colnames_slopes_bf          =   join(["bf_slope_$(i)" for i in 1:nprotocols], "|")
    
        open(filename, "w") do f
            println(f, "Protein|$(colnames_log2FC_bf_g0)|$(colnames_log2FC_median)|$(colnames_log2FC_mean)|$(colnames_log2FC_pd)|$(colnames_log2FC_pd_directions)|$(colnames_log2FC_creds)|$(colnames_log2FC_sd)|$(colnames_slope_median)|$(colnames_slopes_median)|$(colnames_slope_mean)|$(colnames_slopes_mean)|$(colnames_slope_pd)|$(colnames_slopes_pd)| $(colnames_slope_pd_dir)|$(colnames_slopes_pd_dir)|$(colnames_slope_sd)|$(colnames_slopes_sd)|$(colnames_slope_bf)|$(colnames_slopes_bf)")
        end
    end

    open(filename, "a") do f
        log2FC_median   = join([HBM_stats[:median_log2FC][i]    for i in axes(HBM_stats[:median_log2FC], 1)], "|")
        log2FC_mean     = join([HBM_stats[:mean_log2FC][i]      for i in axes(HBM_stats[:mean_log2FC], 1)], "|")
        log2FC_pd       = join([HBM_stats[:pd][i]               for i in axes(HBM_stats[:pd], 1)], "|")
        log2FC_pd_dir   = join([HBM_stats[:pd_direction][i]     for i in axes(HBM_stats[:pd_direction], 1)], "|")
        log2FC_sd       = join([HBM_stats[:sd_log2FC][i]        for i in axes(HBM_stats[:sd_log2FC], 1)], "|")
        
        log2FC_creds    = join(["[$(HBM_stats[:credible_interval][i][1]),$(HBM_stats[:credible_interval][i][2])]" for i in axes(HBM_stats[:credible_interval], 1)], "|")
        log2FC_bf_g0    = join([bf[i, 1] for i in axes(bf, 1)], "|")
 
        isnothing(regression_stats) ? (slope_median = "NA")     : (slope_median     = regression_stats[:median_slope])
        isnothing(regression_stats) ? (slope_mean = "NA")       : (slope_mean       = regression_stats[:mean_slope])
        isnothing(regression_stats) ? (slope_pd = "NA")         : (slope_pd         = regression_stats[:pd_slope])
        isnothing(regression_stats) ? (slope_pd_dir = "NA")     : (slope_pd_dir     = regression_stats[:pd_direction_slope])
        isnothing(regression_stats) ? (slope_sd = "NA")         : (slope_sd         = sqrt(regression_stats[:variance_slope]))
        isnothing(bfR)              ? (slope_bf = "NA")         : (slope_bf         = bfR[1])
     
           
        isnothing(regression_stats) ? (slopes_median = join(["NA" for _ in 1:nprotocols],"|"))  : (slopes_median   = join([regression_stats[:median_protocol_slope][i] for i in 1:nprotocols], "|"))
        isnothing(regression_stats) ? (slopes_mean = join(["NA" for _ in 1:nprotocols],"|"))    : (slopes_mean     = join([regression_stats[:mean_protocol_slope][i] for i in 1:nprotocols], "|"))
        isnothing(regression_stats) ? (slopes_pd = join(["NA" for _ in 1:nprotocols],"|"))      : (slopes_pd       = join([regression_stats[:pd_protocol][i] for i in 1:nprotocols], "|"))
        isnothing(regression_stats) ? (slopes_pd_dir = join(["NA" for _ in 1:nprotocols],"|"))  : (slopes_pd_dir   = join([regression_stats[:pd_direction_protocol][i] for i in 1:nprotocols], "|"))
        isnothing(regression_stats) ? (slopes_sd = join(["NA" for _ in 1:nprotocols],"|"))      : (slopes_sd       = join([sqrt.(regression_stats[:variance_protocol_slope][i]) for i in 1:nprotocols], "|"))
        isnothing(regression_stats) ? (slopes_bf = join(["NA" for _ in 1:nprotocols],"|"))      : (slopes_bf       = join([bfR[i+1] for i in 1:nprotocols], "|"))
        
        println(f, "$protein_name|$(log2FC_bf_g0)|$(log2FC_median)|$(log2FC_mean)|$(log2FC_pd)|$(log2FC_pd_dir)|$(log2FC_creds)|$(log2FC_sd)|$slope_median|$slopes_median|$slope_mean|$slopes_mean|$slope_pd|$slopes_pd|$slope_pd_dir|$slopes_pd_dir|$slope_sd|$slopes_sd|$slope_bf|$slopes_bf")  
    end
end


"""
    write_txt_singleprotocol(; filename, protein_name, HBM_stats, regression_stats, bf, bfR)

Write per-protein inference summary to a pipe-delimited text file (single-protocol variant).

Similar to [`write_txt`](@ref) but for datasets with only one experimental protocol,
producing a simpler column layout without per-protocol breakdowns.
"""
function write_txt_singleprotocol(;
    filename::S, protein_name::S,
    HBM_stats, regression_stats,
    bf, bfR,
    ) where {S<:AbstractString}

    if !isfile(filename)
        # HBM
        colnames_log2FC_median = join(["median_log2FC_$i" for i in 1:size(HBM_stats[:median_log2FC], 1)], "|")
        colnames_log2FC_mean = join(["mean_log2FC_$i" for i in 1:size(HBM_stats[:mean_log2FC], 1)], "|")
        colnames_log2FC_pd = join(["pd_log2FC_$i" for i in 1:size(HBM_stats[:pd], 1)], "|")
        colnames_log2FC_pd_directions = join(["pd_direction_log2FC_$i" for i in 1:size(HBM_stats[:pd_direction], 1)], "|")
        colnames_log2FC_creds = join(["ci_log2FC_$(i)" for i in 1:size(HBM_stats[:credible_interval], 1)], "|")
        colnames_log2FC_sd = join(["sd_log2FC_$i" for i in 1:size(HBM_stats[:sd_log2FC], 1)], "|")

        colnames_log2FC_bf_g0 = join(["bf_log2FC_$(i)_>0.0" for i in 1:size(bf, 1)], "|")
        
        # regression
        colnames_slope_median   = "median_slope"
        colnames_slope_mean     = "mean_slope"
        colnames_slope_pd       = "pd_slope"
        colnames_slope_pd_dir   = "pd_slope_dir"
        colnames_slope_sd       = "sd_slope" 
        colnames_slope_bf       = "bf_slope"
    
        open(filename, "w") do f
            println(f, "Protein|$(colnames_log2FC_bf_g0)|$(colnames_log2FC_median)|$(colnames_log2FC_mean)|$(colnames_log2FC_pd)|$(colnames_log2FC_pd_directions)|$(colnames_log2FC_creds)|$(colnames_log2FC_sd)|$(colnames_slope_median)|$(colnames_slope_mean)|$(colnames_slope_pd)| $(colnames_slope_pd_dir)|$(colnames_slope_sd)|$(colnames_slope_bf)|")
        end
    end

    open(filename, "a") do f
        log2FC_median   = join([HBM_stats[:median_log2FC][i]    for i in axes(HBM_stats[:median_log2FC], 1)], "|")
        log2FC_mean     = join([HBM_stats[:mean_log2FC][i]      for i in axes(HBM_stats[:mean_log2FC], 1)], "|")
        log2FC_pd       = join([HBM_stats[:pd][i]               for i in axes(HBM_stats[:pd], 1)], "|")
        log2FC_pd_dir   = join([HBM_stats[:pd_direction][i]     for i in axes(HBM_stats[:pd_direction], 1)], "|")
        log2FC_sd       = join([HBM_stats[:sd_log2FC][i]        for i in axes(HBM_stats[:sd_log2FC], 1)], "|")
        
        log2FC_creds    = join(["[$(HBM_stats[:credible_interval][i][1]),$(HBM_stats[:credible_interval][i][2])]" for i in axes(HBM_stats[:credible_interval], 1)], "|")
        log2FC_bf_g0    = join([bf[i, 1] for i in axes(bf, 1)], "|")
 
        isnothing(regression_stats) ? (slope_median = "NA")     : (slope_median     = regression_stats[:median_slope])
        isnothing(regression_stats) ? (slope_mean = "NA")       : (slope_mean       = regression_stats[:mean_slope])
        isnothing(regression_stats) ? (slope_pd = "NA")         : (slope_pd         = regression_stats[:pd_slope])
        isnothing(regression_stats) ? (slope_pd_dir = "NA")     : (slope_pd_dir     = regression_stats[:pd_direction_slope])
        isnothing(regression_stats) ? (slope_sd = "NA")         : (slope_sd         = sqrt(regression_stats[:variance_slope]))
        isnothing(bfR)              ? (slope_bf = "NA")         : (slope_bf         = bfR[1])
     
        println(f, "$protein_name|$(log2FC_bf_g0)|$(log2FC_median)|$(log2FC_mean)|$(log2FC_pd)|$(log2FC_pd_dir)|$(log2FC_creds)|$(log2FC_sd)|$slope_median|$slope_mean|$slope_pd|$slope_pd_dir|$slope_sd|$slope_bf|")  
    end
end

