# BayesInteractomics.jl Documentation

Welcome to the documentation for **BayesInteractomics.jl**, a comprehensive Julia package for Bayesian analysis of protein-protein interactions from mass spectrometry experiments.

## What is BayesInteractomics?

BayesInteractomics provides a rigorous statistical framework for identifying genuine protein-protein interactions from Affinity-Purification Mass Spectrometry (AP-MS) and proximity labeling data. Unlike traditional approaches that rely on single statistical tests or ad-hoc filtering, BayesInteractomics integrates **three complementary lines of evidence** through principled Bayesian inference:

1. **Detection Evidence**: Is the protein consistently detected across sample replicates versus controls? (Beta-Bernoulli model)
2. **Enrichment Evidence**: Is the protein quantitatively enriched in samples? (Hierarchical Bayesian Model)
3. **Correlation Evidence**: Does the protein's abundance correlate with bait protein levels? (Bayesian Regression)

These individual Bayes factors are combined using **copula-based mixture models** to produce joint posterior probabilities that account for complex dependencies between evidence types.

## Why BayesInteractomics?

### Challenges in Interactome Analysis

Protein interaction studies face several statistical challenges:

- **Non-specific binding**: Many detected proteins are contaminants rather than genuine interactors
- **Missing data**: Not all proteins are detected in every replicate
- **Protocol heterogeneity**: Different experimental methods yield different background distributions
- **Small sample sizes**: Limited replicates make traditional frequentist methods unreliable
- **Multiple testing**: Thousands of proteins require careful control of false discovery rates

### The BayesInteractomics Solution

BayesInteractomics addresses these challenges through:

- **Hierarchical modeling**: Shares information across protocols while accounting for protocol-specific variability
- **Principled uncertainty quantification**: Full posterior distributions rather than p-values
- **Multiple evidence integration**: Copulas flexibly model dependencies between different evidence types
- **Missing data handling**: Natural treatment of missing observations within Bayesian framework
- **Scalable inference**: Efficient variational Bayes enables analysis of thousands of proteins

## Package Architecture

### Hierarchical Data Structure

BayesInteractomics organizes data in a natural hierarchy that mirrors experimental design:

```
InteractionData
├── Protocol 1 (e.g., AP-MS)
│   ├── Experiment 1 (biological replicate set 1)
│   │   ├── Control samples
│   │   └── Bait samples
│   └── Experiment 2 (biological replicate set 2)
│       ├── Control samples
│       └── Bait samples
└── Protocol 2 (e.g., proximity labeling)
    └── Experiment 1
        ├── Control samples
        └── Bait samples
```

This structure enables:
- Protocol-level parameters (e.g., baseline detection rates)
- Experiment-level parameters (e.g., batch effects)
- Sample-level observations with missing data

### Analysis Workflow

The typical analysis workflow consists of:

1. **Data Loading**: Import from Excel/CSV files with flexible column specifications
2. **H0 Computation**: Fit null hypothesis copula from negative control proteins
3. **Parallel Analysis**: Compute Bayes factors for all proteins across three models
4. **EM Fitting**: Fit mixture copula to combine evidence (H0 vs. H1)
5. **Results Generation**: Produce ranked interaction lists with posterior probabilities
6. **Visualization**: Generate volcano plots, convergence diagnostics, and evidence distributions

All steps are orchestrated through the `run_analysis()` function with configuration via the `CONFIG` struct.

## Installation

BayesInteractomics requires Julia 1.9 or later.

### From Package Registry

```julia
using Pkg
Pkg.add("BayesInteractomics")
```

### Development Version

```julia
using Pkg
Pkg.add(url="https://github.com/ma-seefelder/BayesInteractomics.jl")
```

### First Steps

After installation, load the package and verify it works:

```julia
using BayesInteractomics

# Check that key functions are available
?CONFIG
?load_data
?run_analysis
```

## Quick Start Example

Here's a minimal working example:

```julia
using BayesInteractomics

# Configure analysis
config = CONFIG(
    datafile = ["data/experiment.xlsx"],
    control_cols = [Dict(1 => [2,3,4])],
    sample_cols = [Dict(1 => [5,6,7])],
    poi = "BAIT_PROTEIN_ID",
    refID = 1,
    n_controls = 3,
    n_samples = 3,
    normalise_protocols = false,
    H0_file = "copula_H0.xlsx",
    results_file = "results.xlsx"
)

# Run complete pipeline
results = run_analysis(config)

# View top interactions
first(sort(results, :Posterior_Probability, rev=true), 10)
```

## Key Features

- **Multiple Protocol Support**: Integrate evidence from different experimental methods (AP-MS, BioID, APEX, etc.)
- **Hierarchical Bayesian Models**: Share information across experiments while modeling heterogeneity
- **Copula-Based Combination**: Flexible dependency modeling for evidence integration
- **Robust Regression**: Student-t likelihood with WAIC-based model comparison
- **Multiple Imputation**: Proper uncertainty propagation for missing data
- **Parallel Computing**: Multi-threaded analysis for efficient large-scale studies
- **Data Curation**: Automated contaminant removal, protein group splitting, and synonym resolution via STRING
- **Diagnostics**: Posterior predictive checks, residual analysis, calibration assessment, and prior sensitivity analysis
- **Differential Analysis**: Compare interactomes between conditions with classification of gained, lost, and unchanged interactions
- **Rich Visualizations**: Volcano plots, convergence diagnostics, evidence distributions, and diagnostic plots
- **Interactive Reports**: Self-contained HTML reports with sortable tables and embedded plots
- **Network Analysis**: Graph construction, topology metrics, centrality analysis, community detection, and export to Cytoscape/Gephi
- **Meta-Learning**: Transfer knowledge across experiments via trained models

## Documentation Navigation

This documentation is organized into several sections:

### [Tutorial](tutorial.md)
**Start here if you're new to BayesInteractomics.** Step-by-step guide covering installation, data preparation, running your first analysis, and interpreting results.

### User Guide
In-depth explanations of each component:
- **[Data Loading](data_loading.md)**: File formats, column specifications, and data structure
- **[Data Curation](data_curation.md)**: Contaminant removal, protein group splitting, synonym resolution, and duplicate merging
- **[Analysis Pipeline](analysis.md)**: Complete workflow, configuration, caching, and parallelization
- **[Model Fitting](model_fitting.md)**: Statistical models, priors, robust regression, and model comparison
- **[Model Evaluation](model_evaluation.md)**: Bayes factors, posterior probabilities, and quality metrics
- **[Diagnostics](diagnostics.md)**: Posterior predictive checks, residual analysis, calibration, and prior sensitivity
- **[Differential Analysis](differential_analysis.md)**: Comparing interactomes between conditions (gained, lost, unchanged)
- **[Visualization](visualization.md)**: Pipeline plots, per-protein plots, and diagnostic visualizations
- **[Reports](reports.md)**: Interactive HTML report generation
- **[Network Analysis](network_analysis.md)**: Network construction, topology analysis, hub identification, community detection, and export

### [Examples](examples.md)
Real-world analysis workflows including:
- HAP40 interactome analysis with multiple protocols
- Meta-analysis combining evidence across experiments
- Multiple imputation workflow
- Custom visualization examples

### [Mathematical Background](mathematical_background.md)
Detailed mathematical exposition of:
- Beta-Bernoulli model for detection probability
- Hierarchical Bayesian Model for enrichment
- Bayesian linear regression for dose-response
- Copula theory and EM algorithm
- Statistical references and derivations

### API Reference
Complete documentation of all exported functions and types. Functions are organized by module for easy navigation.

## Getting Help

If you encounter issues or have questions:

1. **Check the Tutorial**: Many common questions are answered in the step-by-step guide
2. **Browse Examples**: Real workflows may demonstrate what you need
3. **Search the Docs**: Use the search bar to find relevant sections
4. **GitHub Issues**: Report bugs or request features at [github.com/ma-seefelder/BayesInteractomics.jl](https://github.com/ma-seefelder/BayesInteractomics.jl/issues)

## Citation

If you use BayesInteractomics in your research, please cite:

```bibtex
@software{bayesinteractomics2025,
  author = {Seefelder, Manuel},
  title = {BayesInteractomics.jl: Bayesian Analysis of Protein Interactome Data},
  year = {2025},
  url = {https://github.com/ma-seefelder/BayesInteractomics.jl}
}
```

## License

BayesInteractomics.jl is released under the MIT License.
