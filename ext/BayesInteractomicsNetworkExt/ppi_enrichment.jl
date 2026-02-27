# Prey-prey network enrichment via STRING database
#
# Implements Bayesian weighting of STRING evidence and the main enrich_network() function.

# ---------------------------------------------------------------------------
# Channel mapping
# ---------------------------------------------------------------------------

const _CHANNEL_TO_SCORE_COL = Dict{Symbol, Symbol}(
    :experimental   => :escore,
    :database       => :dscore,
    :coexpression   => :ascore,
    :textmining     => :tscore,
    :neighborhood   => :nscore,
    :fusion         => :fscore,
    :cooccurrence   => :pscore
)

# ---------------------------------------------------------------------------
# Bayesian weight computation
# ---------------------------------------------------------------------------

"""
    _compute_prey_prey_weight(string_score, co_purification_prior, string_prior) -> Float64

Convert a STRING combined score to a posterior probability of interaction,
using a contextual co-purification prior instead of STRING's genome-wide prior.

The Bayesian update:
1. Extract Bayes factor from STRING's posterior: BF = s(1-π₀) / ((1-s)π₀)
2. Apply co-purification prior: P(H1|data) = BF·π_cop / (BF·π_cop + (1-π_cop))
"""
function _compute_prey_prey_weight(
    string_score::Real,
    co_purification_prior::Float64,
    string_prior::Float64
)::Float64
    # STRING API v12+ returns scores as floats in [0, 1]; clamp to avoid division by zero
    s = clamp(Float64(string_score), 0.001, 0.999)

    # Extract Bayes factor from STRING's calibrated posterior
    bf_string = (s * (1.0 - string_prior)) / ((1.0 - s) * string_prior)

    # Apply co-purification prior
    posterior = (bf_string * co_purification_prior) /
                (bf_string * co_purification_prior + (1.0 - co_purification_prior))

    return clamp(posterior, 0.0, 1.0)
end

"""
    _compute_prey_prey_weight_channels(channel_scores, channels, co_purification_prior, string_prior) -> Float64

Compute posterior probability using per-channel Bayes factor decomposition.
Individual channel Bayes factors are multiplied (assuming conditional independence),
then the combined BF is applied with the co-purification prior.
"""
function _compute_prey_prey_weight_channels(
    channel_scores::Dict{Symbol, Float64},
    channels::Vector{Symbol},
    co_purification_prior::Float64,
    string_prior::Float64
)::Float64
    bf_combined = 1.0
    for ch in channels
        s = clamp(Float64(get(channel_scores, ch, 0.0)), 0.0, 0.999)
        if s > 0.0
            bf_ch = (s * (1.0 - string_prior)) / ((1.0 - s) * string_prior)
            bf_combined *= bf_ch
        end
    end

    posterior = (bf_combined * co_purification_prior) /
                (bf_combined * co_purification_prior + (1.0 - co_purification_prior))

    return clamp(posterior, 0.0, 1.0)
end

"""
    _extract_channel_scores(row, channel_col_map) -> Dict{Symbol, Float64}

Extract per-channel scores from a DataFrame row using the channel-to-column mapping.
"""
function _extract_channel_scores(row, channels::Vector{Symbol})::Dict{Symbol, Float64}
    scores = Dict{Symbol, Float64}()
    for ch in channels
        col = get(_CHANNEL_TO_SCORE_COL, ch, nothing)
        if !isnothing(col) && hasproperty(row, col)
            val = getproperty(row, col)
            if !ismissing(val) && isfinite(val)
                scores[ch] = Float64(val)
            end
        end
    end
    return scores
end

# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------

"""
    BayesInteractomics.enrich_network(net::InteractionNetwork; kwargs...) -> InteractionNetwork

Enrich a bait-prey interaction network with prey-prey edges from the STRING database.

Returns a NEW InteractionNetwork with additional prey-prey edges.
Original bait-prey edges are preserved unchanged. Every edge is annotated with
an `edge_source` attribute (`"experimental"` or `"public_ppi"`).

# Arguments
- `net::InteractionNetwork`: Network from `build_network()`

# Keyword Arguments
- `species::Int=9606`: NCBI taxonomy ID (9606=human, 10090=mouse, 10116=rat)
- `min_string_score::Int=700`: Minimum STRING combined score (0-1000)
- `network_type::Symbol=:physical`: `:physical` or `:functional`
- `co_purification_prior::Float64=0.05`: Prior P(interaction | co-purification)
- `string_prior::Float64=0.002`: STRING's implicit genome-wide prior
- `use_bayesian_weighting::Bool=true`: Use Bayesian framework (false=direct STRING scores)
- `channels::Vector{Symbol}=[:combined]`: Evidence channels to use
- `cache_dir::String=""`: Cache directory (empty=default)
- `offline_file::String=""`: Path to local STRING links file (not yet implemented)
- `force_refresh::Bool=false`: Ignore cache, re-query STRING
- `protein_mapping::Union{Dict{String,String},Nothing}=nothing`: Manual protein→STRING ID overrides
- `caller_identity::String="BayesInteractomics.jl"`: Identifier for STRING API
- `verbose::Bool=true`: Print progress and coverage statistics

# Returns
New `InteractionNetwork` with prey-prey edges added.

# Example
```julia
enriched = enrich_network(net,
    species = 9606,
    min_string_score = 700,
    network_type = :physical,
    co_purification_prior = 0.05,
    verbose = true
)
```
"""
function BayesInteractomics.enrich_network(
    net::InteractionNetwork;
    species::Int = 9606,
    min_string_score::Int = 700,
    network_type::Symbol = :physical,
    co_purification_prior::Float64 = 0.05,
    string_prior::Float64 = 0.002,
    use_bayesian_weighting::Bool = true,
    channels::Vector{Symbol} = [:combined],
    cache_dir::String = "",
    offline_file::String = "",
    force_refresh::Bool = false,
    protein_mapping::Union{Dict{String,String}, Nothing} = nothing,
    caller_identity::String = "BayesInteractomics.jl",
    verbose::Bool = true
)
    n_original = nv(net.graph)
    if n_original == 0
        @warn "Network is empty, nothing to enrich"
        return net
    end

    # Validate parameters
    if !(network_type in (:physical, :functional))
        error("network_type must be :physical or :functional, got :$network_type")
    end
    if min_string_score < 0 || min_string_score > 1000
        error("min_string_score must be between 0 and 1000, got $min_string_score")
    end

    # --- Step 1: Extract prey protein names (exclude bait) ---
    prey_names = if !isnothing(net.bait_protein)
        filter(!=(net.bait_protein), net.protein_names)
    else
        copy(net.protein_names)
    end

    if length(prey_names) < 2
        if verbose
            @info "Fewer than 2 prey proteins — no prey-prey interactions possible"
        end
        return _return_with_edge_source(net)
    end

    # --- Step 2: Advisory warning ---
    if verbose
        @info "enrich_network: Adding prey-prey edges from STRING public database." *
              " These edges are NOT from your experiment — they are from public interaction data." *
              " All edges are annotated with edge_source='experimental' or 'public_ppi'."
    end

    # --- Step 3: Query STRING ---
    if verbose
        nt_str = network_type == :physical ? "physical" : "functional"
        @info "Querying STRING v12.0" n_prey=length(prey_names) species network_type=nt_str
    end

    ppi_result = BayesInteractomics.query_string_ppi(
        prey_names, species;
        network_type, caller_identity, cache_dir, force_refresh, protein_mapping
    )

    # Report coverage
    n_mapped = length(ppi_result.protein_mapping)
    n_unmapped = length(ppi_result.unmapped_proteins)
    if verbose
        @info "Protein mapping" mapped="$n_mapped/$(length(prey_names))" unmapped=n_unmapped
    end

    if nrow(ppi_result.interactions) == 0
        if verbose
            @info "No prey-prey interactions found in STRING. Returning original network."
        end
        return _return_with_edge_source(net)
    end

    # --- Step 4: Filter by min_string_score ---
    interactions = ppi_result.interactions
    score_col = hasproperty(interactions, :score) ? :score :
                hasproperty(interactions, :combined_score) ? :combined_score : nothing

    if isnothing(score_col)
        @warn "STRING response has no score column. Returning original network."
        return _return_with_edge_source(net)
    end

    # STRING API v12+ returns scores as floats in [0, 1]; min_string_score is on [0, 1000] scale
    score_threshold = min_string_score / 1000.0
    filtered = interactions[interactions[!, score_col] .>= score_threshold, :]

    if nrow(filtered) == 0
        if verbose
            @info "No interactions passed min_string_score=$min_string_score. Returning original network."
        end
        return _return_with_edge_source(net)
    end

    # --- Step 5: Map STRING names back to user protein names ---
    # Build reverse mapping: stringId -> user_name
    string_to_user = Dict{String, String}(v => k for (k, v) in ppi_result.protein_mapping)

    # Build lookup for which proteins are in the network
    network_protein_set = Set(net.protein_names)
    name_to_idx = Dict{String, Int}(name => i for (i, name) in enumerate(net.protein_names))

    # Resolve interactions to user protein names
    prey_prey_edges = Tuple{String, String, Float64, Int}[]  # (user_a, user_b, weight, string_score)

    for row in eachrow(filtered)
        # Map stringId back to user name
        str_a = hasproperty(row, :stringId_A) ? string(row.stringId_A) : nothing
        str_b = hasproperty(row, :stringId_B) ? string(row.stringId_B) : nothing

        if isnothing(str_a) || isnothing(str_b)
            continue
        end

        user_a = get(string_to_user, str_a, nothing)
        user_b = get(string_to_user, str_b, nothing)

        if isnothing(user_a) || isnothing(user_b)
            continue
        end

        # Both proteins must be in the network
        if !(user_a in network_protein_set) || !(user_b in network_protein_set)
            continue
        end

        # Skip self-loops
        if user_a == user_b
            continue
        end

        # --- Step 6: Compute weight ---
        raw_score = row[score_col]

        weight = if use_bayesian_weighting
            if channels == [:combined]
                _compute_prey_prey_weight(raw_score, co_purification_prior, string_prior)
            else
                ch_scores = _extract_channel_scores(row, channels)
                _compute_prey_prey_weight_channels(ch_scores, channels, co_purification_prior, string_prior)
            end
        else
            Float64(raw_score)  # Already in [0, 1]
        end

        # Store string_score on conventional 0-1000 scale for display
        score_int = round(Int, Float64(raw_score) * 1000)
        push!(prey_prey_edges, (user_a, user_b, weight, score_int))
    end

    if isempty(prey_prey_edges)
        if verbose
            @info "No prey-prey edges could be mapped to network proteins. Returning original network."
        end
        return _return_with_edge_source(net)
    end

    if verbose
        @info "Found $(length(prey_prey_edges)) prey-prey interactions (STRING score >= $min_string_score)"
        if use_bayesian_weighting
            @info "Applied Bayesian weighting (co-purification prior = $co_purification_prior)"
        end
    end

    # --- Step 7: Build new graph ---
    n = nv(net.graph)
    new_graph = SimpleWeightedDiGraph(n)

    # Re-add all existing edges
    for e in edges(net.graph)
        add_edge!(new_graph, src(e), dst(e), weight(e))
    end

    # Add new prey-prey edges (bidirectional)
    for (user_a, user_b, w, _) in prey_prey_edges
        i = name_to_idx[user_a]
        j = name_to_idx[user_b]
        add_edge!(new_graph, i, j, w)
        add_edge!(new_graph, j, i, w)  # Symmetric
    end

    # --- Step 8: Build new edge_attributes DataFrame ---
    existing_attrs = copy(net.edge_attributes)
    existing_attrs[!, :edge_source] .= "experimental"
    existing_attrs[!, :string_score] = Vector{Union{Int, Missing}}(missing, nrow(existing_attrs))

    new_edge_rows = DataFrame(
        source_node = Int[],
        target_node = Int[],
        source_protein = String[],
        target_protein = String[],
        weight = Float64[],
        edge_source = String[],
        string_score = Union{Int, Missing}[]
    )

    for (user_a, user_b, w, ss) in prey_prey_edges
        i = name_to_idx[user_a]
        j = name_to_idx[user_b]
        # Forward direction
        push!(new_edge_rows, (i, j, user_a, user_b, w, "public_ppi", ss))
        # Reverse direction (symmetric)
        push!(new_edge_rows, (j, i, user_b, user_a, w, "public_ppi", ss))
    end

    combined_attrs = vcat(existing_attrs, new_edge_rows; cols=:union)

    # --- Step 9: Update node_attributes ---
    new_node_attrs = copy(net.node_attributes)

    # Add string_id column
    string_ids = Vector{Union{String, Missing}}(missing, nrow(new_node_attrs))
    for (user_name, string_id) in ppi_result.protein_mapping
        idx = get(name_to_idx, user_name, nothing)
        if !isnothing(idx) && idx <= length(string_ids)
            string_ids[idx] = string_id
        end
    end
    new_node_attrs[!, :string_id] = string_ids

    # Add n_prey_prey_edges column
    prey_prey_counts = zeros(Int, nrow(new_node_attrs))
    for (user_a, user_b, _, _) in prey_prey_edges
        i = name_to_idx[user_a]
        j = name_to_idx[user_b]
        prey_prey_counts[i] += 1
        prey_prey_counts[j] += 1
    end
    new_node_attrs[!, :n_prey_prey_edges] = prey_prey_counts

    # --- Step 10: Extend threshold_used and return ---
    new_thresholds = (;
        net.threshold_used...,
        min_string_score = min_string_score,
        string_network_type = network_type,
        string_species = species,
        co_purification_prior = co_purification_prior,
        use_bayesian_weighting = use_bayesian_weighting
    )

    n_experimental = ne(net.graph)
    n_ppi = ne(new_graph) - n_experimental

    if verbose
        @info "Enriched network" n_nodes=nv(new_graph) n_experimental_edges=n_experimental n_ppi_edges=n_ppi total_edges=ne(new_graph)
    end

    return InteractionNetwork(
        new_graph,
        copy(net.protein_names),
        new_node_attrs,
        combined_attrs,
        net.bait_protein,
        net.bait_index,
        new_thresholds
    )
end

# ---------------------------------------------------------------------------
# Helper: return original network with edge_source column added
# ---------------------------------------------------------------------------

"""
    _return_with_edge_source(net) -> InteractionNetwork

Return a copy of the network with the `edge_source` column set to `"experimental"`
on all edges, for consistency when enrichment yields no new edges.
"""
function _return_with_edge_source(net::InteractionNetwork)
    new_attrs = copy(net.edge_attributes)
    if !hasproperty(new_attrs, :edge_source)
        new_attrs[!, :edge_source] .= "experimental"
    end
    if !hasproperty(new_attrs, :string_score)
        new_attrs[!, :string_score] = Vector{Union{Int, Missing}}(missing, nrow(new_attrs))
    end

    return InteractionNetwork(
        net.graph,
        net.protein_names,
        net.node_attributes,
        new_attrs,
        net.bait_protein,
        net.bait_index,
        net.threshold_used
    )
end
