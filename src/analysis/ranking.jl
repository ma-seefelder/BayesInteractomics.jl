"""
    top(r::Ranks, n::Integer = 5)

Return the top `n` entities with the lowest mean rank.
"""
function top(r::Ranks, n::Integer = 5)
    idx = sortperm(r.mean_ranks)[1:n]
    return r.names[idx], r.mean_ranks[idx]
end


function Base.show(io::IO, r::Ranks)
    println(io, "Ranks object with $(length(r)) entities and $(size(r.ranks, 2)) posterior samples.")
    println(io, "Top-ranked entities by mean rank:")
    names, means = top(r, min(5, length(r)))
    for (name, rank) in zip(names, means)
        println(io, "  - $name (mean rank: $(round(rank, digits=2)))")
    end
end

function get_ranks(draws::AbstractVector{<:Real})
    rank = zeros(Int, length(draws))
    for (r, idx) in enumerate(sortperm(draws, rev=true))
        rank[idx] = r
    end
    return rank
end

"""
    Ranks(posteriors::Vector, entity_names::Vector{String}; n::Int = 1000) -> Ranks

Construct a `Ranks` object from a collection of posterior distributions.

This function draws `n` samples from each posterior distribution, ranks the values across distributions for each sample (column-wise), and returns a `Ranks` object that contains the full rank matrix along with summary statistics (mean and median ranks).

# Arguments
- `posteriors::Vector`: A vector of univariate distribution objects that support `rand`.
- `entity_names::Vector{String}`: A vector of names (e.g., parameter or protein names), one for each posterior.
- `n::Int=1000`: The number of random samples to draw from each posterior (default: 1000).

# Returns
- A `Ranks` struct with the following fields:
  - `ranks::Matrix{Int}`: Matrix of rank values (rows = parameters, columns = samples).
  - `entity_names::Vector{String}`: Names of each parameter/posterior.
  - `mean_ranks::Vector{Float64}`: Mean rank across samples.
  - `median_ranks::Vector{Float64}`: Median rank across samples.

# Notes
- Higher values are ranked as better (rank 1 = highest value).
- Ranking is performed independently for each sample column.
- Ranks can be used to summarize the relative position of posterior distributions.
"""
function Ranks(posteriors::Vector, entity_names::Vector{String}; n::Int = 1000)
    @assert length(posteriors) == length(entity_names)

    nparams = length(posteriors)
    samples = zeros(Float64, nparams, n)
    for i in 1:nparams
        samples[i, :] = rand(posteriors[i], n)
    end

    ranks = Array{Int}(undef, nparams, n)
    for j in 1:n
        ranks[:, j] = get_ranks(view(samples, :, j))
    end

    return Ranks(ranks, entity_names)
end



"""
    plot(data::Ranks)

    Function to plot the ranks

TBW
"""
function plot(data::Ranks, figure_size = (1600,1200))
    # Compute mean ranks for sorting
    sorted_indices = sortperm(data.mean_ranks)
    maxrank = length(sorted_indices)
    # Sort proteins by their mean rank
    sorted_proteins = collect(1:maxrank)[sorted_indices]
    sorted_medians = data.median_ranks[sorted_indices]
    r = getRanks(ranks)[sorted_indices, :]

    
    yvals = 1:maxrank

    # Initialize plot
    plt = plot(; legend=false,
                xlim = (0, maxrank + 1), xlabel = "Proteins", xticks = (yvals, sorted_proteins), xrotation = 60,
                ylim=(0, maxrank + 1), ylabel = "Median ranks", yticks = (1:maxrank, string.(1:maxrank)),
                legendfont = StatsPlots.font(7), size = figure_size
                )

    StatsPlots.scatter!(plt, 1:maxrank, sorted_medians, marker=:circle, markersize=0.3, color=:blue, label = "Median rank")

    StatsPlots.errorline!(
        plt, 1:maxrank, r, error_style = :ribbon, 
        centertype = :median, errortype = :percentile,
        percentile = (0.025, 0.975), alpha = 0.3, 
        label = "95% credible interval"
    )

    return plt
end

function convertRanksToPlotting(data::Ranks)
    ranks = data.ranks
    plotting_data = zeros(Float64, size(ranks, 1), size(ranks, 1))

    for i in axes(plotting_data, 1)
        plotting_data[i,:] = rankfrequency(ranks[i,:], size(ranks, 1))
    end

    return plotting_data 
end

function rankfrequency(ranks, maxrank)
    frequency = zeros(Int64, maxrank)
    [frequency[ranks[i]] += 1 for i in eachindex(ranks)]
    return frequency ./ sum(frequency)
end

function hpdi(samples, prob::Float64 = 0.95)
    # Sort the samples
    sorted_samples = sort(samples)
    n = length(samples)
    # Number of samples to include in the interval
    interval_size = Int(round(prob * n))
    # Find the narrowest interval
    best_interval = (1, interval_size)
    min_width = sorted_samples[interval_size] - sorted_samples[1]
    for i in 1:(n - interval_size + 1)
		j = i + interval_size - 1
        width = sorted_samples[j] - sorted_samples[i]
        if width < min_width
            min_width = width
            best_interval = (i, j)
        end
    end
    lower = sorted_samples[best_interval[1]]
    upper = sorted_samples[best_interval[2]]
    return (lower, upper)
end

function hpdi(samples::Matrix{R}, prob::Float64 = 0.95) where {R<:Real}
    ncols = size(samples, 2)
    lower = Vector{Float64}(undef, ncols)
    upper = Vector{Float64}(undef, ncols)

    Threads.@threads for col in 1:ncols
        lower[col], upper[col] = hpdi(samples[:, col], prob)
    end

    return lower, upper
end

