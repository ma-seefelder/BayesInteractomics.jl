# Network construction from analysis results

"""
    BayesInteractomics.build_network(ar::BayesInteractomics.AbstractAnalysisResult; kwargs...)

Build an interaction network from Bayesian analysis results.

Accepts both full `AnalysisResult` (from complete pipeline) and `NetworkAnalysisResult`
(lightweight wrapper for custom DataFrames). Automatically detects column name variants
across both types.

# Arguments
- `ar::AbstractAnalysisResult`: Analysis results (AnalysisResult or NetworkAnalysisResult)
- `posterior_threshold::Float64=0.5`: Minimum posterior probability for inclusion
- `bf_threshold::Union{Float64, Nothing}=nothing`: Minimum Bayes factor for inclusion
- `q_threshold::Float64=0.05`: Maximum q-value (FDR) for inclusion
- `log2fc_threshold::Union{Float64, Nothing}=nothing`: Minimum log2 fold change for inclusion
- `include_bait::Bool=true`: Whether to include bait protein as a node
- `weight_by::Symbol=:posterior_prob`: Edge weight source (:posterior_prob, :bayes_factor, :log2fc)

# Returns
- `InteractionNetwork`: Network object with graph and attributes

# Example
```julia
using BayesInteractomics
using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

# With full AnalysisResult
results = analyse(data, "H0.xlsx", refID=1)
ar = AnalysisResult(results, ..., bait_protein="MYC", bait_index=1)
net = build_network(ar, posterior_threshold=0.8, q_threshold=0.01)

# Or with NetworkAnalysisResult
ar = NetworkAnalysisResult(my_df, bait_protein="MYC")
net = build_network(ar, posterior_threshold=0.8)
```
"""
function BayesInteractomics.build_network(
    ar::BayesInteractomics.AbstractAnalysisResult;
    posterior_threshold::Float64 = 0.5,
    bf_threshold::Union{Float64, Nothing} = nothing,
    q_threshold::Float64 = 0.05,
    log2fc_threshold::Union{Float64, Nothing} = nothing,
    include_bait::Bool = true,
    weight_by::Symbol = :posterior_prob
)
    # Validate weight_by parameter
    if !(weight_by in [:posterior_prob, :bayes_factor, :log2fc])
        error("weight_by must be one of :posterior_prob, :bayes_factor, :log2fc")
    end

    # Extract data from results
    df = ar.results

    # Apply filters to identify significant interactions
    mask = trues(nrow(df))

    # Posterior probability filter (support multiple naming conventions)
    # Note: Missing values are treated as not passing the filter
    if hasproperty(df, :PosteriorProbability) || hasproperty(df, :posterior_probability) || hasproperty(df, :posterior_prob)
        post_col = hasproperty(df, :PosteriorProbability) ? :PosteriorProbability :
                   hasproperty(df, :posterior_probability) ? :posterior_probability : :posterior_prob
        mask .&= coalesce.(df[!, post_col] .>= posterior_threshold, false)
    end

    # Bayes factor filter (support multiple naming conventions)
    # Note: Missing values are treated as not passing the filter
    if !isnothing(bf_threshold)
        if hasproperty(df, :BayesFactor) || hasproperty(df, :bayes_factor) || hasproperty(df, :BF)
            bf_col = hasproperty(df, :BayesFactor) ? :BayesFactor :
                     hasproperty(df, :bayes_factor) ? :bayes_factor : :BF
            mask .&= coalesce.(df[!, bf_col] .>= bf_threshold, false)
        end
    end

    # Q-value filter (support multiple naming conventions)
    # Note: Missing values are treated as not passing the filter
    if hasproperty(df, :q_value) || hasproperty(df, :QValue) || hasproperty(df, :q)
        q_col = hasproperty(df, :q_value) ? :q_value :
                hasproperty(df, :QValue) ? :QValue : :q
        mask .&= coalesce.(df[!, q_col] .<= q_threshold, false)
    end

    # Log2FC filter (support multiple naming conventions)
    # Note: Missing values are treated as not passing the filter
    if !isnothing(log2fc_threshold)
        if hasproperty(df, :mean_log2FC) || hasproperty(df, :log2FC)
            fc_col = hasproperty(df, :mean_log2FC) ? :mean_log2FC : :log2FC
            mask .&= coalesce.(abs.(df[!, fc_col]) .>= log2fc_threshold, false)
        end
    end

    # Filter to significant interactions
    significant = df[mask, :]

    if nrow(significant) == 0
        @warn "No interactions passed the filtering criteria. Consider relaxing thresholds."
        # Return empty network
        empty_graph = SimpleWeightedDiGraph(0)
        return InteractionNetwork(
            empty_graph,
            String[],
            DataFrame(),
            DataFrame(),
            ar.bait_protein,
            ar.bait_index,
            (posterior_threshold=posterior_threshold, bf_threshold=bf_threshold,
             q_threshold=q_threshold, log2fc_threshold=log2fc_threshold)
        )
    end

    # Get protein names
    protein_col = hasproperty(significant, :Protein) ? :Protein : :protein
    interactor_names = significant[!, protein_col]

    # Create node list (bait + interactors or just interactors)
    if include_bait && !isnothing(ar.bait_protein)
        all_proteins = vcat([ar.bait_protein], interactor_names)
        bait_idx = 1
    else
        all_proteins = interactor_names
        bait_idx = nothing
    end

    n_nodes = length(all_proteins)

    # Create graph
    graph = SimpleWeightedDiGraph(n_nodes)

    # Determine weight column (support multiple naming conventions)
    weight_col = if weight_by == :posterior_prob
        hasproperty(significant, :PosteriorProbability) ? :PosteriorProbability :
        hasproperty(significant, :posterior_probability) ? :posterior_probability : :posterior_prob
    elseif weight_by == :bayes_factor
        hasproperty(significant, :BayesFactor) ? :BayesFactor :
        hasproperty(significant, :bayes_factor) ? :bayes_factor : :BF
    else  # :log2fc
        hasproperty(significant, :mean_log2FC) ? :mean_log2FC : :log2FC
    end

    # Build node attributes DataFrame
    node_attrs = DataFrame(
        protein = all_proteins,
        node_id = 1:n_nodes
    )

    if include_bait && !isnothing(ar.bait_protein)
        # Add bait row with missing attributes (support multiple naming conventions)
        node_attrs[!, :posterior_prob] = vcat([missing],
            hasproperty(significant, :PosteriorProbability) ? significant.PosteriorProbability :
            hasproperty(significant, :posterior_probability) ? significant.posterior_probability : significant.posterior_prob)
        node_attrs[!, :bayes_factor] = vcat([missing],
            hasproperty(significant, :BayesFactor) ? significant.BayesFactor :
            hasproperty(significant, :bayes_factor) ? significant.bayes_factor : significant.BF)
        node_attrs[!, :q_value] = vcat([missing],
            hasproperty(significant, :q_value) ? significant.q_value :
            hasproperty(significant, :QValue) ? significant.QValue : significant.q)
        if hasproperty(significant, :mean_log2FC) || hasproperty(significant, :log2FC)
            fc_col = hasproperty(significant, :mean_log2FC) ? :mean_log2FC : :log2FC
            node_attrs[!, :mean_log2fc] = vcat([missing], significant[!, fc_col])
        end
        node_attrs[!, :is_bait] = vcat([true], falses(n_nodes - 1))
    else
        # No bait, just interactors (support multiple naming conventions)
        node_attrs[!, :posterior_prob] = hasproperty(significant, :PosteriorProbability) ?
            significant.PosteriorProbability :
            hasproperty(significant, :posterior_probability) ? significant.posterior_probability : significant.posterior_prob
        node_attrs[!, :bayes_factor] = hasproperty(significant, :BayesFactor) ?
            significant.BayesFactor :
            hasproperty(significant, :bayes_factor) ? significant.bayes_factor : significant.BF
        node_attrs[!, :q_value] = hasproperty(significant, :q_value) ?
            significant.q_value :
            hasproperty(significant, :QValue) ? significant.QValue : significant.q
        if hasproperty(significant, :mean_log2FC) || hasproperty(significant, :log2FC)
            fc_col = hasproperty(significant, :mean_log2FC) ? :mean_log2FC : :log2FC
            node_attrs[!, :mean_log2fc] = significant[!, fc_col]
        end
        node_attrs[!, :is_bait] = falses(n_nodes)
    end

    # Add edges (bait -> interactors or interactor -> interactor)
    edge_list = Tuple{Int, Int, Float64}[]

    if include_bait && !isnothing(ar.bait_protein)
        # Edges from bait to all interactors
        for i in 2:n_nodes
            weight = significant[i-1, weight_col]
            # Handle missing or non-finite weights
            if ismissing(weight) || !isfinite(weight)
                weight = 0.0
            end
            add_edge!(graph, 1, i, weight)
            push!(edge_list, (1, i, weight))
        end
    else
        # No bait: create complete graph among interactors (if desired)
        # For now, we'll just create a star-like structure with a virtual center
        # This is a simplification; users can modify as needed
        @warn "No bait protein specified. Creating network with interactors only."
        # Could add interactor-interactor edges based on similarity, but that's beyond scope
    end

    # Build edge attributes DataFrame
    edge_attrs = DataFrame(
        source_node = [e[1] for e in edge_list],
        target_node = [e[2] for e in edge_list],
        source_protein = [all_proteins[e[1]] for e in edge_list],
        target_protein = [all_proteins[e[2]] for e in edge_list],
        weight = [e[3] for e in edge_list]
    )

    # Store thresholds used
    thresholds = (
        posterior_threshold = posterior_threshold,
        bf_threshold = bf_threshold,
        q_threshold = q_threshold,
        log2fc_threshold = log2fc_threshold
    )

    return InteractionNetwork(
        graph,
        all_proteins,
        node_attrs,
        edge_attrs,
        ar.bait_protein,
        bait_idx,
        thresholds
    )
end
