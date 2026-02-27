# BayesInteractomics Network Analysis Extension

This package extension provides network analysis capabilities for protein-protein interaction data analyzed with BayesInteractomics.

## Overview

The extension automatically loads when you import the required dependencies:
```julia
using BayesInteractomics
using Graphs, SimpleWeightedGraphs, GraphPlot, Compose
```

## Features

### Network Construction
Build interaction networks from Bayesian analysis results with flexible filtering:
- Posterior probability thresholds
- Bayes factor cutoffs
- FDR (q-value) filtering
- Log2 fold-change filtering
- Customizable edge weights

### Topology Analysis
Compute comprehensive network statistics:
- Basic metrics: nodes, edges, density, average degree
- Structural properties: clustering coefficient, components
- Distance metrics: diameter, average path length

### Centrality Measures
Identify hub proteins using multiple centrality algorithms:
- **Degree centrality**: Number of direct connections
- **Weighted degree**: Sum of edge weights
- **Betweenness**: Frequency on shortest paths between nodes
- **Closeness**: Average distance to all nodes
- **Eigenvector centrality**: Importance based on neighbor importance
- **PageRank**: Random walk probability (Google's algorithm)

### Community Detection
Discover protein complexes and functional modules:
- **Louvain method**: Fast greedy modularity optimization
- **Label propagation**: Efficient community detection
- **Greedy modularity**: Classical modularity maximization

### Visualization
Create publication-quality network plots:
- Multiple layout algorithms (spring, circular, shell, spectral)
- Customizable node sizes (degree, posterior probability, fold-change)
- Flexible node colors (posterior probability, fold-change, community)
- Adjustable edge widths (weight-based or uniform)
- Bait protein highlighting

### Export
Save networks for external analysis tools:
- **GraphML format**: For Cytoscape, Gephi, igraph
- **CSV edge list**: Simple tabular format
- **Node attributes CSV**: For custom analysis
- **PNG/PDF/SVG plots**: Publication-ready figures

## Quick Start

```julia
using BayesInteractomics
using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

# Run Bayesian analysis
data = load_data(["experiment.xlsx"], sample_cols, control_cols)
results = analyse(data, "H0.xlsx", refID=1, n_controls=3, n_samples=3)

# Wrap results with bait protein information
ar = AnalysisResult(results, bait_protein="MYC", bait_index=1)

# Build high-confidence network
net = build_network(ar, posterior_threshold=0.8, q_threshold=0.01)

# Analyze network
stats = network_statistics(net)
println("Network has $(stats.n_nodes) proteins and $(stats.n_edges) interactions")
println("Average clustering: $(round(stats.avg_clustering, digits=3))")

# Find hub proteins
cm = centrality_measures(net)
df = centrality_dataframe(cm)
top_hubs = sort(df, :PageRank, rev=true)[1:10, :]
println("\nTop 10 hub proteins:")
println(top_hubs)

# Detect communities
communities = detect_communities(net, algorithm=:louvain)
println("\nFound $(communities.n_communities) communities")
println("Modularity: $(round(communities.modularity, digits=3))")

# Visualize
plt = plot_network(net,
    node_color=:posterior_prob,
    node_size=:degree,
    highlight_bait=true
)
save_network_plot(net, "network.png")

# Export for Cytoscape
export_graphml(net, "network.graphml")
```

## Detailed Usage

### 1. Network Construction

```julia
net = build_network(ar,
    posterior_threshold = 0.8,      # Minimum posterior probability (0-1)
    bf_threshold = 10.0,            # Minimum Bayes factor (optional)
    q_threshold = 0.01,             # Maximum q-value/FDR (0-1)
    log2fc_threshold = 1.0,         # Minimum absolute log2 fold-change (optional)
    include_bait = true,            # Include bait protein as network node
    weight_by = :posterior_prob     # Edge weight source
)
```

**Edge weight options**:
- `:posterior_prob` - Use posterior probabilities (default, range 0-1)
- `:bayes_factor` - Use Bayes factors (unbounded)
- `:log2fc` - Use log2 fold-changes (can be negative)

### 2. Network Statistics

```julia
stats = network_statistics(net)

# Access fields:
stats.n_nodes              # Total number of proteins
stats.n_edges              # Total number of interactions
stats.density              # Edge density (0-1)
stats.avg_degree           # Average connections per protein
stats.avg_weighted_degree  # Average weighted degree
stats.avg_clustering       # Average clustering coefficient
stats.n_components         # Number of connected components
stats.largest_component_size  # Size of largest component
stats.diameter             # Network diameter (if connected)
stats.avg_path_length      # Average shortest path (if connected)
```

### 3. Centrality Analysis

```julia
cm = centrality_measures(net)

# Convert to DataFrame for analysis
df = centrality_dataframe(cm)

# Find top hubs by different measures
top_degree = sort(df, :Degree, rev=true)[1:10, :]
top_pagerank = sort(df, :PageRank, rev=true)[1:10, :]
top_betweenness = sort(df, :Betweenness, rev=true)[1:10, :]

# Export centrality table
CSV.write("centrality.csv", df)
```

### 4. Community Detection

```julia
# Try different algorithms
comm_louvain = detect_communities(net, algorithm=:louvain)
comm_label = detect_communities(net, algorithm=:label_propagation)
comm_greedy = detect_communities(net, algorithm=:greedy_modularity)

# Compare modularity scores
println("Louvain modularity: ", comm_louvain.modularity)
println("Label prop modularity: ", comm_label.modularity)

# Get proteins in a specific community
community_df = community_dataframe(comm_louvain)
community_1_proteins = filter(row -> row.Community == 1, community_df)

# Or use helper function
proteins_in_comm_1 = BayesInteractomicsNetworkExt.get_community_proteins(comm_louvain, 1)
```

### 5. Visualization

```julia
# Basic plot
plot_network(net)

# Custom styling
plot_network(net,
    layout = :spring,              # Layout algorithm
    node_size = :degree,           # Size by degree centrality
    node_color = :posterior_prob,  # Color by posterior probability
    edge_width = :weight,          # Width by edge weight
    show_labels = true,            # Show protein names
    highlight_bait = true,         # Highlight bait in gold
    figsize = (1000, 1000)         # Size in pixels
)

# Save in different formats
save_network_plot(net, "network.png")
save_network_plot(net, "network.pdf")
save_network_plot(net, "network.svg")

# Advanced: customize plot then save manually
plt = plot_network(net, node_color=:community)
using Cairo  # For high-quality rendering
draw(PNG("network_high_res.png", 2000, 2000), plt)
```

### 6. Export

```julia
# GraphML for Cytoscape/Gephi
export_graphml(net, "network.graphml")
# Open in Cytoscape: File > Import > Network from File

# CSV formats
export_edgelist(net, "edges.csv")
export_node_attributes(net, "nodes.csv")

# Export everything at once
BayesInteractomicsNetworkExt.export_network_bundle(net, "results/network")
# Creates: results/network.graphml, results/network_edges.csv, results/network_nodes.csv
```

## Implementation Details

### Data Structures

**InteractionNetwork**: Core network object containing:
- `graph`: SimpleWeightedDiGraph from SimpleWeightedGraphs.jl
- `protein_names`: Protein identifiers
- `node_attributes`: DataFrame with posterior probabilities, Bayes factors, etc.
- `edge_attributes`: DataFrame with edge weights and source/target info
- `bait_protein`: Bait protein name (if specified)
- `threshold_used`: Record of filtering criteria

**NetworkStatistics**: Topology metrics
**CentralityMeasures**: Hub identification scores
**CommunityResult**: Module/complex assignments

### Extension Architecture

This is a Julia package extension (requires Julia 1.9+):
- **Stub definitions**: `src/network/stubs.jl` (always loaded)
- **Extension code**: `ext/BayesInteractomicsNetworkExt/` (loaded conditionally)
- **Trigger**: Extension activates when user loads Graphs, SimpleWeightedGraphs, GraphPlot, Compose

Benefits:
- No extra dependencies unless you use network features
- Seamless activation when needed
- Type-stable interface (stubs are always available)

### Algorithms

**Community Detection**:
- Louvain: Greedy modularity optimization with refinement
- Label propagation: Fast algorithm using neighbor voting
- Greedy modularity: Classical hierarchical merging

**Centrality**:
- All algorithms from Graphs.jl standard library
- Weighted versions use edge weights when appropriate

**Layouts**:
- Spring: Force-directed layout with attraction/repulsion
- Circular: Proteins arranged in a circle
- Shell: Concentric circles (currently uses circular)
- Spectral: Based on graph Laplacian (currently uses spring)

## Performance Tips

- For large networks (>100 proteins), use relaxed thresholds to reduce size
- Spring layout can be slow for >200 nodes; use circular layout instead
- Community detection scales well up to 1000+ nodes
- Export to GraphML and use Cytoscape for very large networks (>500 nodes)

## Troubleshooting

**Extension not loading**:
```julia
# Make sure all dependencies are loaded
using Graphs, SimpleWeightedGraphs, GraphPlot, Compose
```

**No proteins pass threshold**:
```julia
# Relax filtering criteria
net = build_network(ar, posterior_threshold=0.5, q_threshold=0.1)
```

**Plotting is slow**:
```julia
# Use simpler layout
plot_network(net, layout=:circular)
```

**Need higher resolution plots**:
```julia
using Cairo
plt = plot_network(net)
draw(PNG("network.png", 3000, 3000), plt)
```

## Citation

If you use the network analysis features in your research, please cite:

```
BayesInteractomics: A Bayesian Framework for Protein Interactome Analysis
[Add citation when published]
```

## License

Same as BayesInteractomics main package (GNU AGPL v3.0)
