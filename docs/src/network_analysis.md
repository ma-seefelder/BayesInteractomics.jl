# Network Analysis

BayesInteractomics provides comprehensive network analysis capabilities through a package extension that enables visualization and topological analysis of protein interaction networks. This extension automatically loads when you import the required graph packages.

## Overview

The network analysis extension transforms your Bayesian analysis results into graph structures that can be:
- Visualized with customizable layouts and styling
- Analyzed for topological properties (density, clustering, path lengths)
- Searched for hub proteins using centrality measures
- Partitioned into communities representing protein complexes or functional modules
- Exported to standard formats for external tools (Cytoscape, Gephi)

## Activation

Network analysis features are provided via package extension and require loading the graph packages:

```julia
using BayesInteractomics
using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

# Network functions are now available
```

Without loading these packages, network functions will throw an error directing you to load the required dependencies.

## Quick Start

```julia
using BayesInteractomics
using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

# Run standard Bayesian analysis
data = load_data(["experiment.xlsx"], sample_cols, control_cols)
results = analyse(data, "H0.xlsx", refID=1, n_controls=3, n_samples=3)

# Wrap results with bait protein metadata
ar = AnalysisResult(results, bait_protein="MYC", bait_index=1)

# Build high-confidence network
net = build_network(ar,
    posterior_threshold = 0.8,
    q_threshold = 0.01
)

# Analyze network properties
stats = network_statistics(net)
println("Network has $(stats.n_nodes) nodes and $(stats.n_edges) edges")
println("Density: $(stats.density)")

# Find hub proteins
cm = centrality_measures(net)
top_hubs = get_top_hubs(cm, by=:pagerank, n=10)

# Detect protein complexes
communities = detect_communities(net, algorithm=:louvain)
println("Found $(communities.n_communities) communities")

# Visualize
plot_network(net, layout=:spring, highlight_bait=true)

# Export for Cytoscape
export_graphml(net, "network.graphml")
```

## Data Types

### AbstractAnalysisResult

Base type for analysis results that can be used for network construction. Two implementations are provided:

#### AnalysisResult

Full analysis result wrapper returned by the `analyse()` pipeline:

```julia
struct AnalysisResult <: AbstractAnalysisResult
    results::DataFrame           # Complete analysis results
    bait_protein::String        # Name of bait protein
    bait_index::Int            # Index in protein list
    # ... additional fields from full pipeline
end
```

#### NetworkAnalysisResult

Lightweight wrapper for custom DataFrames:

```julia
# Create from any DataFrame with appropriate columns
df = DataFrame(
    Protein = ["A", "B", "C"],
    PosteriorProbability = [0.95, 0.85, 0.75],
    BayesFactor = [100.0, 50.0, 20.0],
    q_value = [0.001, 0.01, 0.02],
    mean_log2FC = [3.2, 2.8, 2.1]
)

ar = NetworkAnalysisResult(df, bait_protein="MYC")
net = build_network(ar, posterior_threshold=0.8)
```

**Required DataFrame columns** (supports multiple naming conventions):
- Protein names: `Protein` or `protein`
- Posterior probability: `PosteriorProbability`, `posterior_probability`, or `posterior_prob`
- Bayes factor: `BayesFactor`, `bayes_factor`, or `BF`
- Q-value (FDR): `q_value`, `QValue`, or `q`
- Log2 fold change (optional): `mean_log2FC` or `log2FC`

### InteractionNetwork

Graph representation of protein interactions:

```julia
struct InteractionNetwork
    graph::SimpleWeightedDiGraph    # Directed weighted graph
    protein_names::Vector{String}   # Node labels
    node_attributes::DataFrame      # Node-level data
    edge_attributes::DataFrame      # Edge-level data
    bait_protein::Union{String, Nothing}
    bait_index::Union{Int, Nothing}
    threshold_used::NamedTuple     # Filtering thresholds
end
```

Access graph properties:
```julia
nv(net.graph)  # Number of nodes
ne(net.graph)  # Number of edges
net.protein_names  # Protein names
net.node_attributes  # DataFrame with posterior_prob, bayes_factor, etc.
```

### NetworkStatistics

Topology metrics:

```julia
struct NetworkStatistics
    n_nodes::Int
    n_edges::Int
    density::Float64              # Edges / possible edges
    avg_degree::Float64           # Average connections per node
    avg_weighted_degree::Float64  # Average sum of edge weights
    avg_clustering::Float64       # Clustering coefficient
    n_components::Int            # Connected components
    largest_component_size::Int
    diameter::Union{Int, Nothing}      # Max shortest path (if connected)
    avg_path_length::Union{Float64, Nothing}  # Average shortest path
end
```

### CentralityMeasures

Hub identification scores:

```julia
struct CentralityMeasures
    protein_names::Vector{String}
    degree::Vector{Int}              # Number of connections
    weighted_degree::Vector{Float64} # Sum of edge weights
    betweenness::Vector{Float64}     # Shortest path centrality
    closeness::Vector{Float64}       # Inverse average distance
    eigenvector::Vector{Float64}     # Recursive importance
    pagerank::Vector{Float64}        # Random walk probability
end
```

Convert to DataFrame for easy inspection:
```julia
df = centrality_dataframe(cm)
sort(df, :PageRank, rev=true)  # Rank by PageRank
```

### CommunityResult

Protein complex/module assignments:

```julia
struct CommunityResult
    protein_names::Vector{String}
    membership::Vector{Int}      # Community ID for each protein
    n_communities::Int
    modularity::Float64         # Quality score (higher is better)
    community_sizes::Vector{Int}
end
```

## Network Construction

### build_network

Build interaction network from analysis results with statistical filtering:

```julia
net = build_network(ar::AbstractAnalysisResult;
    posterior_threshold = 0.5,
    bf_threshold = nothing,
    q_threshold = 0.05,
    log2fc_threshold = nothing,
    include_bait = true,
    weight_by = :posterior_prob
)
```

**Arguments:**
- `posterior_threshold`: Minimum posterior probability (default: 0.5)
- `bf_threshold`: Minimum Bayes factor (optional)
- `q_threshold`: Maximum FDR q-value (default: 0.05)
- `log2fc_threshold`: Minimum absolute log2 fold change (optional)
- `include_bait`: Include bait protein as network node (default: true)
- `weight_by`: Edge weight source (`:posterior_prob`, `:bayes_factor`, or `:log2fc`)

**Filtering strategies:**
```julia
# High-confidence network (stringent)
net_high = build_network(ar,
    posterior_threshold = 0.9,
    q_threshold = 0.01,
    bf_threshold = 10.0
)

# Medium-confidence network (balanced)
net_medium = build_network(ar,
    posterior_threshold = 0.8,
    q_threshold = 0.05
)

# Exploratory network (permissive)
net_explore = build_network(ar,
    posterior_threshold = 0.5,
    q_threshold = 0.1
)

# Focus on strong enrichment
net_enriched = build_network(ar,
    posterior_threshold = 0.8,
    log2fc_threshold = 2.0,
    weight_by = :log2fc
)
```

## Network Statistics

### network_statistics

Compute comprehensive topology metrics:

```julia
stats = network_statistics(net)

# Basic properties
stats.n_nodes              # Number of proteins
stats.n_edges              # Number of interactions
stats.density              # 0.0 (sparse) to 1.0 (complete)

# Degree statistics
stats.avg_degree           # Average connections per node
stats.avg_weighted_degree  # Average sum of edge weights

# Clustering
stats.avg_clustering       # Local clustering coefficient

# Connectivity
stats.n_components         # Number of disconnected subgraphs
stats.largest_component_size

# Path properties (if connected)
stats.diameter             # Longest shortest path
stats.avg_path_length      # Average shortest path
```

**Interpretation:**
- **Density**: Higher values indicate more interconnected proteins (typical PPI networks: 0.01-0.1)
- **Clustering**: Higher values suggest modular organization (protein complexes)
- **Components**: Multiple components may indicate distinct functional modules
- **Diameter**: Network "width" - smaller values indicate tighter integration

## Centrality Analysis

### centrality_measures

Identify hub proteins using multiple centrality metrics:

```julia
cm = centrality_measures(net)

# Access centrality vectors
cm.degree          # Simple connection count
cm.weighted_degree # Weighted connection strength
cm.betweenness    # Information flow control
cm.closeness      # Average proximity to all nodes
cm.eigenvector    # Recursive importance (connected to important nodes)
cm.pagerank       # Random walk probability (like Google PageRank)
```

**Centrality measure interpretation:**

| Measure | Identifies | Use Case |
|---------|-----------|----------|
| **Degree** | Highly connected proteins | Core interaction hubs |
| **Weighted Degree** | Strong binders | High-confidence hubs |
| **Betweenness** | Bridge proteins | Inter-complex connectors |
| **Closeness** | Central proteins | Rapid signal propagation |
| **Eigenvector** | Influential proteins | Connected to other hubs |
| **PageRank** | Important proteins | Overall network importance |

### get_top_hubs

Extract top hub proteins by any centrality measure:

```julia
# Top 10 proteins by PageRank
top_hubs = get_top_hubs(cm, by=:pagerank, n=10)

# Other centrality measures
top_by_degree = get_top_hubs(cm, by=:degree, n=10)
top_by_betweenness = get_top_hubs(cm, by=:betweenness, n=10)
top_by_weighted = get_top_hubs(cm, by=:weighted_degree, n=10)
```

**Choosing centrality measures:**
- **General hub identification**: Use PageRank or eigenvector centrality
- **Direct interaction partners**: Use degree centrality
- **Functional importance**: Use betweenness centrality
- **High-confidence hubs**: Use weighted degree with posterior probability weights

### centrality_dataframe

Convert centrality measures to DataFrame for easy analysis:

```julia
df = centrality_dataframe(cm)
# Returns DataFrame with columns: Protein, Degree, WeightedDegree,
# Betweenness, Closeness, Eigenvector, PageRank

# Rank proteins by multiple criteria
sorted = sort(df, [:PageRank, :Degree], rev=true)

# Export to file
CSV.write("hub_proteins.csv", df)
```

## Community Detection

### detect_communities

Identify protein complexes or functional modules:

```julia
communities = detect_communities(net, algorithm=:louvain)

# Results
communities.n_communities     # Number of communities found
communities.modularity       # Quality score (higher is better)
communities.community_sizes  # Size of each community
communities.membership       # Community ID for each protein
```

**Available algorithms:**

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `:louvain` | Modularity optimization | Hierarchical communities |
| `:label_propagation` | Fast local propagation | Large networks |
| `:greedy_modularity` | Greedy merging | Medium networks |

**Modularity score interpretation:**
- **> 0.3**: Strong community structure
- **0.1 - 0.3**: Moderate community structure
- **< 0.1**: Weak or no community structure

### community_dataframe

Convert to DataFrame for analysis:

```julia
df = community_dataframe(communities)
# Returns DataFrame with columns: Protein, Community

# Analyze community composition
using DataFrames
combine(groupby(df, :Community), nrow => :Size)
```

### get_community_proteins

Extract proteins from specific community:

```julia
# Get all proteins in community 1
proteins = get_community_proteins(communities, 1)

# Analyze each community
for i in 1:communities.n_communities
    proteins = get_community_proteins(communities, i)
    println("Community $i has $(length(proteins)) proteins:")
    println(join(proteins, ", "))
end
```

**Biological interpretation:**
- Each community may represent a protein complex or functional module
- Large communities (>20 proteins) may contain multiple sub-complexes
- Small communities (<5 proteins) may be transient interactions
- Compare with known protein complexes (CORUM, ComplexPortal databases)

## Visualization

### plot_network

Create customizable network visualizations:

```julia
plot_network(net;
    layout = :spring,
    node_size = :degree,
    node_color = :posterior_prob,
    edge_width = :weight,
    show_labels = true,
    highlight_bait = true,
    figsize = (800, 800)
)
```

**Layout algorithms:**
- `:spring`: Force-directed layout (good for most networks)
- `:circular`: Nodes arranged in circle (good for small networks)
- `:shell`: Shell layout with multiple levels
- `:spectral`: Spectral decomposition layout

**Node size options:**
- `:degree`: Size by number of connections
- `:posterior_prob`: Size by posterior probability
- `:log2fc`: Size by fold change magnitude
- `:uniform`: All nodes same size

**Node color options:**
- `:posterior_prob`: Blue (low) to red (high) gradient
- `:log2fc`: Diverging blue (down) to red (up) scale
- `:community`: Different color per community (requires running `detect_communities` first)
- `:uniform`: Single color (steel blue)

**Edge width options:**
- `:weight`: Width proportional to edge weight
- `:uniform`: All edges same width

**Examples:**

```julia
# Basic visualization
plot_network(net)

# Highlight bait and show posterior probabilities
plot_network(net,
    layout = :spring,
    node_color = :posterior_prob,
    node_size = :degree,
    highlight_bait = true
)

# Focus on fold change
plot_network(net,
    node_color = :log2fc,
    node_size = :log2fc,
    edge_width = :uniform
)

# Clean circular layout for presentation
plot_network(net,
    layout = :circular,
    node_color = :uniform,
    show_labels = false,
    figsize = (1200, 1200)
)
```

### save_network_plot

Save visualizations to file:

```julia
save_network_plot(net, "network.png")

# Supported formats
save_network_plot(net, "network.png", figsize=(1200, 1200))
save_network_plot(net, "network.pdf")  # Vector format for publications
save_network_plot(net, "network.svg")  # Vector format for web

# With custom styling
save_network_plot(net, "network.png",
    layout = :spring,
    node_color = :posterior_prob,
    highlight_bait = true,
    figsize = (2400, 2400)  # High resolution
)
```

**File format recommendations:**
- **PNG**: Presentations, quick viewing (specify high figsize for print quality)
- **PDF**: Publications, LaTeX documents (vector graphics, scalable)
- **SVG**: Web, interactive applications (vector graphics, editable)

## Export

### export_graphml

Export to GraphML format for Cytoscape, Gephi, and other network tools:

```julia
export_graphml(net, "network.graphml")
```

GraphML preserves all node and edge attributes:
- Node attributes: protein name, posterior probability, Bayes factor, q-value, log2FC, bait status
- Edge attributes: interaction weight

**Using in Cytoscape:**
1. Open Cytoscape
2. File → Import → Network from File
3. Select the .graphml file
4. Attributes are imported as node/edge columns
5. Use Style panel to map attributes to visual properties

**Using in Gephi:**
1. Open Gephi
2. File → Open → Select .graphml file
3. Import as "Directed graph"
4. Attributes available in Data Laboratory

### export_edgelist

Export edge list as CSV:

```julia
export_edgelist(net, "edges.csv")

# Without additional attributes
export_edgelist(net, "edges.csv", include_attributes=false)
```

Creates CSV with columns:
- `source`: Source protein name
- `target`: Target protein name
- `weight`: Edge weight
- Additional edge attributes (if `include_attributes=true`)

**Use cases:**
- Import into R/Python for custom analysis
- Network construction in other tools
- Simple tabular view of interactions

### export_node_attributes

Export node attributes as CSV:

```julia
export_node_attributes(net, "nodes.csv")
```

Creates CSV with columns:
- `protein`: Protein name
- `node_id`: Node index
- `posterior_prob`: Posterior probability
- `bayes_factor`: Bayes factor
- `q_value`: FDR q-value
- `mean_log2fc`: Mean log2 fold change (if available)
- `is_bait`: Boolean indicating bait protein

**Use cases:**
- Statistical analysis in R/Python
- Annotation with external databases
- Custom plotting

### export_network_bundle

Export all formats at once:

```julia
export_network_bundle(net, "results/mynetwork")
# Creates:
#   results/mynetwork.graphml
#   results/mynetwork_edges.csv
#   results/mynetwork_nodes.csv
```

## Complete Workflow Example

Here's a complete workflow from data loading to network export:

```julia
using BayesInteractomics
using Graphs, SimpleWeightedGraphs, GraphPlot, Compose
using DataFrames, CSV

# 1. Load and analyze data
println("Loading data...")
data = load_data(
    ["experiment.xlsx"],
    Dict(1 => [5,6,7]),  # sample columns
    Dict(1 => [2,3,4])   # control columns
)

println("Running Bayesian analysis...")
results = analyse(data, "copula_H0.xlsx",
    refID = 1,
    n_controls = 3,
    n_samples = 3
)

# 2. Wrap results
ar = AnalysisResult(results, bait_protein="MYC", bait_index=1)

# 3. Build high-confidence network
println("Building network...")
net = build_network(ar,
    posterior_threshold = 0.8,
    q_threshold = 0.01,
    weight_by = :posterior_prob
)

# 4. Analyze network topology
println("\n=== Network Statistics ===")
stats = network_statistics(net)
println("Nodes: $(stats.n_nodes)")
println("Edges: $(stats.n_edges)")
println("Density: $(round(stats.density, digits=4))")
println("Avg clustering: $(round(stats.avg_clustering, digits=4))")
println("Components: $(stats.n_components)")

# 5. Identify hub proteins
println("\n=== Top Hub Proteins ===")
cm = centrality_measures(net)
top10 = get_top_hubs(cm, by=:pagerank, n=10)
println(top10)

# Save centrality analysis
CSV.write("hub_analysis.csv", centrality_dataframe(cm))

# 6. Detect protein complexes
println("\n=== Community Detection ===")
communities = detect_communities(net, algorithm=:louvain)
println("Found $(communities.n_communities) communities")
println("Modularity: $(round(communities.modularity, digits=4))")
println("Sizes: $(communities.community_sizes)")

# Save community assignments
CSV.write("communities.csv", community_dataframe(communities))

# Print each community
for i in 1:communities.n_communities
    proteins = get_community_proteins(communities, i)
    println("\nCommunity $i ($(length(proteins)) proteins):")
    println("  ", join(proteins[1:min(5, length(proteins))], ", "),
            length(proteins) > 5 ? "..." : "")
end

# 7. Visualize
println("\n=== Creating Visualizations ===")
save_network_plot(net, "network_overview.png",
    layout = :spring,
    node_color = :posterior_prob,
    node_size = :degree,
    highlight_bait = true,
    figsize = (1200, 1200)
)

save_network_plot(net, "network_publication.pdf",
    layout = :spring,
    show_labels = true,
    figsize = (800, 800)
)

# 8. Export for external tools
println("\n=== Exporting Results ===")
export_network_bundle(net, "results/myc_network")

println("\n✓ Analysis complete!")
println("  Network plots: network_overview.png, network_publication.pdf")
println("  Hub analysis: hub_analysis.csv")
println("  Communities: communities.csv")
println("  Export bundle: results/myc_network.*")
```

## Advanced Topics

### Custom Network Construction

For specialized analyses, you can build networks directly from any DataFrame:

```julia
# Load your own interaction scores
df = CSV.read("my_interactions.csv", DataFrame)

# Must have these columns (or variants):
# - Protein (or protein)
# - PosteriorProbability (or posterior_probability, posterior_prob)
# - BayesFactor (or bayes_factor, BF)
# - q_value (or QValue, q)

# Create analysis result wrapper
ar = NetworkAnalysisResult(df, bait_protein="MYC")

# Build network
net = build_network(ar, posterior_threshold=0.8)
```

### Combining Multiple Networks

Compare networks across conditions:

```julia
# Build networks for different conditions
net_control = build_network(results_control, posterior_threshold=0.8)
net_treatment = build_network(results_treatment, posterior_threshold=0.8)

# Compare statistics
stats_control = network_statistics(net_control)
stats_treatment = network_statistics(net_treatment)

println("Control: $(stats_control.n_edges) edges")
println("Treatment: $(stats_treatment.n_edges) edges")

# Identify condition-specific hubs
cm_control = centrality_measures(net_control)
cm_treatment = centrality_measures(net_treatment)

hubs_control = get_top_hubs(cm_control, n=20)
hubs_treatment = get_top_hubs(cm_treatment, n=20)

# Find condition-specific hubs
control_specific = setdiff(hubs_control.Protein, hubs_treatment.Protein)
treatment_specific = setdiff(hubs_treatment.Protein, hubs_control.Protein)
```

### Subnetwork Extraction

Extract subnetworks for focused analysis:

```julia
# Get proteins in largest community
communities = detect_communities(net)
largest_comm = argmax(communities.community_sizes)
comm_proteins = get_community_proteins(communities, largest_comm)

# Filter results DataFrame
subset_results = filter(row -> row.Protein in comm_proteins, ar.results)

# Build subnetwork
ar_subset = NetworkAnalysisResult(subset_results, bait_protein=ar.bait_protein)
subnet = build_network(ar_subset, posterior_threshold=0.5)

# Analyze subnetwork
plot_network(subnet, layout=:circular)
```

### Integration with External Data

Enrich network with external annotations:

```julia
# Load GO terms, pathways, etc.
annotations = CSV.read("protein_annotations.csv", DataFrame)

# Join with node attributes
enriched_nodes = leftjoin(net.node_attributes, annotations, on=:protein)

# Analyze enrichment
for comm_id in 1:communities.n_communities
    proteins = get_community_proteins(communities, comm_id)
    comm_annotations = filter(row -> row.protein in proteins, enriched_nodes)

    # Count GO terms, pathways, etc.
    println("Community $comm_id enriched terms:")
    println(combine(groupby(comm_annotations, :GO_term), nrow => :count))
end
```

## Best Practices

### Threshold Selection

**Stringent (high confidence):**
```julia
net = build_network(ar,
    posterior_threshold = 0.9,
    q_threshold = 0.01,
    bf_threshold = 10.0
)
```
Use for: Core interactome, publication figures, high-confidence candidates

**Moderate (balanced):**
```julia
net = build_network(ar,
    posterior_threshold = 0.8,
    q_threshold = 0.05
)
```
Use for: General analysis, community detection, hub identification

**Permissive (exploratory):**
```julia
net = build_network(ar,
    posterior_threshold = 0.5,
    q_threshold = 0.1
)
```
Use for: Discovery, pathway analysis, network topology studies

### Visualization Guidelines

1. **For presentations:**
   - Use high-resolution PNG (figsize ≥ 1200×1200)
   - Highlight bait protein
   - Color by posterior probability
   - Size by degree centrality

2. **For publications:**
   - Export as PDF or SVG (vector graphics)
   - Consider circular layout for clarity
   - Use uniform or subtle colors
   - Show labels for key proteins only

3. **For exploration:**
   - Use spring layout for natural clustering
   - Color by communities or functional annotations
   - Size by relevance metric (degree, PageRank)
   - Export to Cytoscape for interactive exploration

### Performance Considerations

- Large networks (>500 nodes): Use `layout=:circular` instead of `:spring` for faster rendering
- Many communities: Use `:louvain` algorithm (faster than `:greedy_modularity`)
- Repeated analysis: Cache `build_network` results to avoid recomputation
- Export large networks to Cytoscape for better performance

## Troubleshooting

### "Network extension not loaded" error

**Problem:** Network functions not available

**Solution:**
```julia
using Graphs, SimpleWeightedGraphs, GraphPlot, Compose
```

### Empty network warning

**Problem:** No interactions pass filtering criteria

**Solution:** Relax thresholds
```julia
net = build_network(ar, posterior_threshold=0.5, q_threshold=0.1)
```

### Disconnected network (diameter = nothing)

**Problem:** Network has multiple components

**Interpretation:** Normal for interaction networks; indicates distinct functional modules

**Analysis:**
```julia
stats = network_statistics(net)
println("Components: $(stats.n_components)")
println("Largest component: $(stats.largest_component_size)")
```

### Community detection yields single community

**Problem:** Low modularity, all proteins in one community

**Causes:**
- Star topology (all proteins connected to bait)
- Too few edges
- Very dense network

**Solutions:**
- Try different algorithm
- Relax filtering thresholds to include more edges
- Consider hierarchical clustering for star networks

## References

For more information on the statistical methods:
- See [Model Evaluation](model_evaluation.md) for Bayesian analysis details
- See [Examples](examples.md) for complete workflows
- See [API Reference](api.md) for function signatures

### Network Analysis Literature

- **Modularity**: Newman, M. E. J. (2006). Modularity and community structure in networks. PNAS, 103(23), 8577-8582.
- **Centrality measures**: Freeman, L. C. (1977). A set of measures of centrality based on betweenness. Sociometry, 40(1), 35-41.
- **PageRank**: Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual web search engine. Computer Networks, 30(1-7), 107-117.
- **Community detection**: Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks. Journal of Statistical Mechanics, 2008(10), P10008.
