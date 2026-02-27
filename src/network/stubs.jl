# Network analysis stubs - extended by BayesInteractomicsNetworkExt

"""
    NetworkAnalysisResult <: AbstractAnalysisResult

Lightweight wrapper for network analysis from any DataFrame with bait info.

This is a simpler alternative to the full `AnalysisResult` type for users who
want to perform network analysis on arbitrary DataFrames without running the
complete Bayesian analysis pipeline.

# Fields
- `results::DataFrame`: DataFrame with analysis results (must contain appropriate columns)
- `bait_protein::Union{String, Nothing}`: Name or ID of the bait protein
- `bait_index::Union{Int, Nothing}`: Index of bait protein in the protein list

# Examples
```julia
# Create from any DataFrame with appropriate columns
df = DataFrame(
    Protein = ["A", "B", "C"],
    PosteriorProbability = [0.95, 0.85, 0.75],
    BayesFactor = [100.0, 50.0, 20.0],
    q_value = [0.001, 0.01, 0.02]
)

ar = NetworkAnalysisResult(df, bait_protein="MYC", bait_index=1)
net = build_network(ar)
```
"""
struct NetworkAnalysisResult <: AbstractAnalysisResult
    results::DataFrame
    bait_protein::Union{String, Nothing}
    bait_index::Union{Int, Nothing}

    function NetworkAnalysisResult(results::DataFrame;
                                   bait_protein::Union{String, Nothing}=nothing,
                                   bait_index::Union{Int, Nothing}=nothing)
        new(results, bait_protein, bait_index)
    end
end

# Public accessors for NetworkAnalysisResult
"""
    getProteins(ar::NetworkAnalysisResult) -> Vector{String}

Get protein names from analysis results.
"""
function getProteins(ar::NetworkAnalysisResult)
    return ar.results.Protein
end

"""
    getBayesFactors(ar::NetworkAnalysisResult) -> Vector{Float64}

Get combined Bayes factors from analysis results.
"""
function getBayesFactors(ar::NetworkAnalysisResult)
    return ar.results.BayesFactor
end

"""
    getPosteriorProbs(ar::NetworkAnalysisResult) -> Vector{Float64}

Get posterior probabilities from analysis results.
"""
function getPosteriorProbs(ar::NetworkAnalysisResult)
    return ar.results.PosteriorProbability
end

"""
    getQValues(ar::NetworkAnalysisResult) -> Vector{Float64}

Get q-values (FDR) from analysis results.
"""
function getQValues(ar::NetworkAnalysisResult)
    return ar.results.q_value
end

"""
    getMeanLog2FC(ar::NetworkAnalysisResult) -> Vector{Float64}

Get mean log2 fold changes from analysis results.
"""
function getMeanLog2FC(ar::NetworkAnalysisResult)
    return ar.results.mean_log2FC
end

"""
    getBaitProtein(ar::NetworkAnalysisResult) -> Union{String, Nothing}

Get bait protein name.
"""
function getBaitProtein(ar::NetworkAnalysisResult)
    return ar.bait_protein
end

"""
    AbstractNetworkResult

Abstract type for network analysis results. Concrete implementations provided by extension.
"""
abstract type AbstractNetworkResult end

# Stub function declarations - implementations provided by BayesInteractomicsNetworkExt
# These functions will only work after loading the required packages:
# using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

"""
    build_network(ar::AbstractAnalysisResult; kwargs...)

Build an interaction network from analysis results.

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`

# Arguments
- `ar::AbstractAnalysisResult`: Analysis results wrapper (AnalysisResult or NetworkAnalysisResult)
- `posterior_threshold::Float64=0.5`: Minimum posterior probability for inclusion
- `bf_threshold::Union{Float64, Nothing}=nothing`: Minimum Bayes factor for inclusion
- `q_threshold::Float64=0.05`: Maximum q-value (FDR) for inclusion
- `log2fc_threshold::Union{Float64, Nothing}=nothing`: Minimum log2 fold change for inclusion
- `include_bait::Bool=true`: Whether to include bait protein as a node
- `weight_by::Symbol=:posterior_prob`: Edge weight source (:posterior_prob, :bayes_factor, :log2fc)

# Returns
- `InteractionNetwork`: Network object (from extension)

# Example
```julia
using BayesInteractomics
using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

# With full analysis result
ar = AnalysisResult(...)
ar.bait_protein = "MYC"
net = build_network(ar, posterior_threshold=0.8, q_threshold=0.01)

# Or with lightweight wrapper
ar = NetworkAnalysisResult(my_df, bait_protein="MYC")
net = build_network(ar, posterior_threshold=0.8)
```
"""
function build_network end

"""
    network_statistics(net)

Compute network topology statistics.

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`

# Returns
- `NetworkStatistics`: Statistics object with fields like n_nodes, n_edges, density, etc.
"""
function network_statistics end

"""
    centrality_measures(net)

Compute centrality measures for all nodes.

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`

# Returns
- `CentralityMeasures`: Centrality measures for each protein
"""
function centrality_measures end

"""
    detect_communities(net; algorithm=:louvain)

Detect communities in the network.

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`

# Arguments
- `net`: Network object
- `algorithm::Symbol=:louvain`: Community detection algorithm

# Returns
- `CommunityResult`: Community assignments and statistics
"""
function detect_communities end

"""
    plot_network(net; kwargs...)

Visualize the interaction network.

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`

# Arguments
- `net`: Network object
- `layout::Symbol=:spring`: Layout algorithm (:spring, :circular, :shell, :spectral)
- `node_size::Symbol=:degree`: Node size mapping (:degree, :posterior_prob, :log2fc, :uniform)
- `node_color::Symbol=:posterior_prob`: Node color mapping (:community, :posterior_prob, :log2fc, :uniform)
- `edge_width::Symbol=:weight`: Edge width mapping
- `show_labels::Bool=true`: Whether to show protein labels
- `highlight_bait::Bool=true`: Whether to highlight bait protein
- `figsize::Tuple{Int,Int}=(800,800)`: Figure size in pixels

# Returns
- Compose canvas object
"""
function plot_network end

"""
    save_network_plot(plt, filename::String; figsize=(800,800))

Save a Compose plot (e.g. from `plot_network()`) to file.

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`

# Arguments
- `plt`: Compose context returned by `plot_network()`
- `filename::String`: Output file path (.png, .pdf, or .svg)
- `figsize::Tuple{Int,Int}=(800,800)`: Output size in pixels
"""
function save_network_plot end

"""
    export_graphml(net, filename::String)

Export network to GraphML format (for Cytoscape, Gephi, etc.).

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`
"""
function export_graphml end

"""
    export_edgelist(net, filename::String; include_attributes=true)

Export network edge list to CSV.

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`
"""
function export_edgelist end

"""
    export_node_attributes(net, filename::String)

Export node attributes to CSV.

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`
"""
function export_node_attributes end

"""
    centrality_dataframe(cm) -> DataFrame

Convert centrality measures to a DataFrame for easy inspection and export.

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`
"""
function centrality_dataframe end

"""
    community_dataframe(cr) -> DataFrame

Convert community detection results to a DataFrame for easy inspection and export.

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`
"""
function community_dataframe end

"""
    get_top_hubs(cm; by=:pagerank, n=10) -> DataFrame

Get the top hub proteins by a specified centrality measure.

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`
"""
function get_top_hubs end

"""
    edge_source_summary(net) -> Dict{String, Int}

Count edges by source type (e.g., "experimental" vs "public_ppi").

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`
"""
function edge_source_summary end

# --- Network analysis pipeline types and stubs ---

"""
    NetworkConfig

Configuration for the complete network analysis pipeline (`run_network_analysis`).

# Bait Protein
- `bait_protein::Union{String, Nothing}=nothing`: Bait protein name (overrides value in AnalysisResult)
- `bait_index::Union{Int, Nothing}=nothing`: Bait protein index (overrides value in AnalysisResult)

# Network Construction Thresholds
- `posterior_threshold::Float64=0.5`: Minimum posterior probability for inclusion
- `bf_threshold::Union{Float64, Nothing}=nothing`: Minimum Bayes factor (nothing = no filter)
- `q_threshold::Float64=0.05`: Maximum q-value (FDR) for inclusion
- `log2fc_threshold::Union{Float64, Nothing}=nothing`: Minimum |log2FC| (nothing = no filter)
- `include_bait::Bool=true`: Whether to include bait protein as a node
- `weight_by::Symbol=:posterior_prob`: Edge weight source (`:posterior_prob`, `:bayes_factor`, `:log2fc`)

# PPI Enrichment (STRING)
- `enrich::Bool=false`: Enable prey-prey enrichment from STRING
- `species::Int=9606`: NCBI taxonomy ID (9606=human, 10090=mouse, 10116=rat)
- `min_string_score::Int=700`: Minimum STRING combined score (0–1000)
- `network_type::Symbol=:physical`: `:physical` or `:functional`
- `force_refresh::Bool=false`: Ignore cache, re-query STRING
- `protein_mapping::Union{Dict{String,String}, Nothing}=nothing`: Manual protein→STRING mapping
- `offline_file::String=""`: Path to local STRING links file

# Visualization
- `plot::Bool=true`: Generate network plot
- `layout::Symbol=:spring`: Layout (`:spring`, `:circular`, `:shell`, `:spectral`)
- `node_size::Symbol=:degree`: Node sizing (`:degree`, `:posterior_prob`, `:log2fc`, `:uniform`)
- `node_color::Symbol=:posterior_prob`: Node coloring (`:posterior_prob`, `:log2fc`, `:community`, `:uniform`)
- `plot_format::Symbol=:png`: Output format (`:png`, `:pdf`, `:svg`)
- `figsize::Tuple{Int,Int}=(800,800)`: Figure dimensions in pixels

# Community Detection
- `detect_communities::Bool=true`: Run community detection
- `community_algorithm::Symbol=:louvain`: Algorithm (`:louvain`, `:label_propagation`, `:greedy_modularity`)

# Centrality
- `compute_centrality::Bool=true`: Compute centrality measures
- `top_hubs_n::Int=10`: Number of top hub proteins to report
- `top_hubs_by::Symbol=:pagerank`: Rank hubs by (`:pagerank`, `:degree`, `:betweenness`, `:eigenvector`, `:closeness`, `:weighted_degree`)

# Export
- `export_files::Bool=true`: Write output files
- `output_dir::String="network_results"`: Output directory
- `file_prefix::String="network"`: Prefix for exported files

# Report
- `generate_report::Bool=true`: Generate Markdown report
- `report_filename::String="network_report.md"`: Report file name
- `report_title::String="Network Analysis Report"`: Report title

# Example
```julia
config = NetworkConfig(
    bait_protein = "HAP40",
    bait_index = 1,
    posterior_threshold = 0.8,
    q_threshold = 0.001,
    bf_threshold = 3.0,
    enrich = true,
    min_string_score = 300,
    force_refresh = false,
    output_dir = "hap40_network",
    report_title = "HAP40 Interaction Network"
)
result = run_network_analysis(ar, config)
```
"""
Base.@kwdef mutable struct NetworkConfig
    # ---- Bait protein ----
    bait_protein::Union{String, Nothing}      = nothing
    bait_index::Union{Int, Nothing}           = nothing

    # ---- Network construction thresholds ----
    posterior_threshold::Float64               = 0.5
    bf_threshold::Union{Float64, Nothing}     = nothing
    q_threshold::Float64                      = 0.05
    log2fc_threshold::Union{Float64, Nothing} = nothing
    include_bait::Bool                        = true
    weight_by::Symbol                         = :posterior_prob

    # ---- PPI enrichment (off by default, needs network packages) ----
    enrich::Bool                              = false
    species::Int                              = 9606
    min_string_score::Int                     = 700
    network_type::Symbol                      = :physical
    co_purification_prior::Float64            = 0.05
    string_prior::Float64                     = 0.002
    use_bayesian_weighting::Bool              = true
    channels::Vector{Symbol}                  = [:combined]
    force_refresh::Bool                       = false
    protein_mapping::Union{Dict{String,String}, Nothing} = nothing
    offline_file::String                      = ""
    ppi_verbose::Bool                         = true

    # ---- Visualization ----
    plot::Bool                                = true
    layout::Symbol                            = :spring
    node_size::Symbol                         = :degree
    node_color::Symbol                        = :posterior_prob
    edge_width::Symbol                        = :weight
    edge_style::Symbol                        = :by_source
    show_labels::Bool                         = true
    show_bait::Union{Bool, Nothing}           = nothing
    highlight_bait::Bool                      = true
    show_legend::Bool                         = true
    figsize::Tuple{Int, Int}                  = (800, 800)
    plot_format::Symbol                       = :png

    # ---- Community detection ----
    detect_communities::Bool                  = true
    community_algorithm::Symbol               = :louvain

    # ---- Centrality ----
    compute_centrality::Bool                  = true
    top_hubs_n::Int                           = 10
    top_hubs_by::Symbol                       = :pagerank

    # ---- Export ----
    export_files::Bool                        = true
    output_dir::String                        = "network_results"
    file_prefix::String                       = "network"
    export_graphml::Bool                      = true
    export_edgelist::Bool                     = true
    export_node_attributes::Bool              = true

    # ---- Report ----
    generate_report::Bool                     = true
    report_filename::String                   = "network_report.md"
    report_title::String                      = "Network Analysis Report"

    # ---- Pipeline control ----
    verbose::Bool                             = true
end

"""
    NetworkPipelineResult <: AbstractNetworkResult

Result of the complete network analysis pipeline (`run_network_analysis`).

Stores all intermediate results (network, statistics, centrality, communities, etc.)
along with the generated report and export paths.

# Fields
- `network`: Base InteractionNetwork
- `enriched_network`: Enriched InteractionNetwork or nothing
- `statistics`: NetworkStatistics or nothing
- `centrality`: CentralityMeasures or nothing
- `top_hubs`: Top hub proteins DataFrame or nothing
- `communities`: CommunityResult or nothing
- `edge_sources`: Edge source breakdown Dict or nothing
- `plot_object`: Compose.Context or nothing
- `config`: NetworkConfig used
- `report_path`: Path to generated .md report or nothing
- `report_content`: Markdown string for REPL/notebook display or nothing
- `export_paths`: Dict mapping type => filepath for exported files
- `warnings`: Accumulated pipeline warnings
"""
struct NetworkPipelineResult <: AbstractNetworkResult
    network::Any
    enriched_network::Union{Any, Nothing}
    statistics::Union{Any, Nothing}
    centrality::Union{Any, Nothing}
    top_hubs::Union{DataFrame, Nothing}
    communities::Union{Any, Nothing}
    edge_sources::Union{Dict{String, Int}, Nothing}
    plot_object::Any
    config::NetworkConfig
    report_path::Union{String, Nothing}
    report_content::Union{String, Nothing}
    export_paths::Dict{String, String}
    warnings::Vector{String}
end

"""
    run_network_analysis(ar::AbstractAnalysisResult, config::NetworkConfig=NetworkConfig()) -> NetworkPipelineResult

Run the complete network analysis pipeline with a single call.

Orchestrates: network construction, optional PPI enrichment, topology statistics,
centrality measures, community detection, visualization, file export, and
Markdown report generation.

# Arguments
- `ar::AbstractAnalysisResult`: Analysis results (AnalysisResult or NetworkAnalysisResult)
- `config::NetworkConfig`: Pipeline configuration (defaults to `NetworkConfig()`)

# Returns
- `NetworkPipelineResult`: All pipeline outputs bundled together

# Example
```julia
using BayesInteractomics
using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

result = run_network_analysis(ar, NetworkConfig(
    posterior_threshold = 0.8,
    q_threshold = 0.01,
    enrich = true,
    output_dir = "my_network",
    report_title = "My Network Report"
))

result.statistics     # NetworkStatistics
result.top_hubs       # DataFrame
result.report_path    # "my_network/network_report.md"
```
"""
function run_network_analysis end

"""
    generate_network_report(result::NetworkPipelineResult; filename=nothing, title=nothing) -> (String, String)

Generate a Markdown report from pipeline results.

Returns `(filepath, markdown_string)`. The report is written to disk AND the
Markdown content is returned for display in REPL/notebooks.

# Arguments
- `result::NetworkPipelineResult`: Pipeline result to report on
- `filename::Union{String, Nothing}`: Override output path (default: from config)
- `title::Union{String, Nothing}`: Override report title (default: from config)

# Returns
- `(filepath::String, content::String)`: Path to written file and Markdown string
"""
function generate_network_report end

# --- Prey-prey network enrichment types and stubs ---

"""
    PPIEnrichmentConfig

Configuration for prey-prey network enrichment from public PPI databases (STRING).

# Fields
- `species::Int=9606`: NCBI taxonomy ID (9606=human, 10090=mouse, 10116=rat)
- `min_string_score::Int=700`: Minimum STRING combined score (0-1000)
- `network_type::Symbol=:physical`: `:physical` or `:functional`
- `co_purification_prior::Float64=0.05`: Prior P(interaction | co-purification context)
- `string_prior::Float64=0.002`: STRING's implicit genome-wide prior
- `channels::Vector{Symbol}=[:combined]`: Evidence channels to use
- `use_bayesian_weighting::Bool=true`: Use Bayesian framework (false=direct STRING scores)
- `cache_dir::String=""`: Cache directory (empty=default ~/.bayesinteractomics/ppi_cache)
- `offline_file::String=""`: Path to local STRING links file for offline mode
- `caller_identity::String="BayesInteractomics.jl"`: Identifier sent to STRING API
"""
Base.@kwdef struct PPIEnrichmentConfig
    species::Int = 9606
    min_string_score::Int = 700
    network_type::Symbol = :physical
    co_purification_prior::Float64 = 0.05
    string_prior::Float64 = 0.002
    channels::Vector{Symbol} = [:combined]
    use_bayesian_weighting::Bool = true
    cache_dir::String = ""
    offline_file::String = ""
    caller_identity::String = "BayesInteractomics.jl"
end

"""
    enrich_network(net; kwargs...) -> InteractionNetwork

Enrich a bait-prey interaction network with prey-prey edges from the STRING database.

Returns a NEW InteractionNetwork with additional prey-prey edges.
Original bait-prey edges are preserved unchanged. Every edge is annotated with
an `edge_source` attribute (`"experimental"` or `"public_ppi"`).

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`

# Arguments
- `net`: InteractionNetwork from `build_network()`

# Keyword Arguments
- `species::Int=9606`: NCBI taxonomy ID (9606=human, 10090=mouse, 10116=rat)
- `min_string_score::Int=700`: Minimum STRING combined score (0-1000)
- `network_type::Symbol=:physical`: `:physical` or `:functional`
- `co_purification_prior::Float64=0.05`: Prior P(interaction | co-purification)
- `string_prior::Float64=0.002`: STRING's implicit genome-wide prior
- `use_bayesian_weighting::Bool=true`: Use Bayesian framework (false=direct STRING scores)
- `channels::Vector{Symbol}=[:combined]`: Evidence channels to use
- `force_refresh::Bool=false`: Ignore cache, re-query STRING
- `protein_mapping::Dict=nothing`: Manual protein→STRING ID mapping
- `offline_file::String=""`: Path to local STRING links file
- `verbose::Bool=true`: Print progress and coverage statistics

# Returns
New `InteractionNetwork` with prey-prey edges added. Original bait-prey edges unchanged.

# Example
```julia
enriched = enrich_network(net, species=9606, min_string_score=700)
```
"""
function enrich_network end

"""
    query_string_ppi(proteins, species; kwargs...) -> PPIQueryResult

Query the STRING database for pairwise interactions among a set of proteins.

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`

# Arguments
- `proteins::Vector{String}`: Protein names/identifiers
- `species::Int`: NCBI taxonomy ID

# Returns
- `PPIQueryResult`: Query results with interactions, mappings, and metadata
"""
function query_string_ppi end

"""
    clear_ppi_cache(; cache_dir="")

Delete all cached STRING PPI query results.

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`
"""
function clear_ppi_cache end

"""
    ppi_cache_info(; cache_dir="") -> NamedTuple

Report cache size, number of entries, and age of oldest/newest entries.

Requires: `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose`
"""
function ppi_cache_info end
