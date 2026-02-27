# BayesInteractomics.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://ma-seefelder.github.io/BayesInteractomics.jl)
[![Build Status](https://github.com/ma-seefelder/BayesInteractomics.jl/workflows/CI/badge.svg)](https://github.com/ma-seefelder/BayesInteractomics.jl/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**BayesInteractomics.jl** is a Julia package for rigorous Bayesian analysis of protein-protein interactions from Affinity-Purification Mass Spectrometry (AP-MS) and proximity labeling experiments. The package implements a comprehensive statistical framework that combines evidence from multiple sources to identify genuine interacting partners with quantified uncertainty.

## Overview

Identifying true protein-protein interactions from mass spectrometry data is challenging due to:
- Non-specific binding and contaminants
- Missing data across replicates and experiments
- Heterogeneity across experimental protocols
- Complex dose-response relationships

BayesInteractomics addresses these challenges through a **three-model Bayesian framework** that evaluates:

1. **Detection Probability** (Beta-Bernoulli model): Is the protein consistently detected in samples vs. controls?
2. **Enrichment** (Hierarchical Bayesian Model): Is the protein quantitatively enriched in samples?
3. **Dose-Response Correlation** (Bayesian Regression): Does abundance correlate with bait protein levels?

These individual lines of evidence are combined using **copula-based mixture models** to produce joint Bayes factors and posterior probabilities for each candidate interaction. The hierarchical approach naturally handles multiple protocols, missing data, and protocol-level variability.

## Key Features

- **Rigorous Bayesian inference** using RxInfer.jl for principled uncertainty quantification
- **Multiple protocol integration** through hierarchical modeling
- **Copula-based evidence combination** for flexible dependency modeling
- **Multiple imputation support** for handling missing data
- **Parallel computation** for analyzing thousands of proteins efficiently
- **Comprehensive visualization** including volcano plots, evidence distributions, and convergence diagnostics
- **Meta-learning capabilities** for transferring knowledge across experiments

## Installation

BayesInteractomics requires Julia 1.9 or later. Install from the Julia REPL:

```julia
using Pkg
Pkg.add("BayesInteractomics")
```

Or install the development version from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/ma-seefelder/BayesInteractomics.jl")
```

## Quick Start

### Basic Analysis Workflow

```julia
using BayesInteractomics

# Define experimental configuration
config = CONFIG(
    # Input data files
    datafile = ["data/protocol1.xlsx", "data/protocol2.xlsx"],

    # Column mappings for each protocol
    # Format: Dict(experiment_id => column_indices)
    control_cols = [
        Dict(1 => [2,3,4], 2 => [5,6,7]),    # Protocol 1
        Dict(1 => [2,3,4], 2 => [5,6])        # Protocol 2
    ],
    sample_cols = [
        Dict(1 => [8,9,10], 2 => [11,12,13]), # Protocol 1
        Dict(1 => [8,9,10], 2 => [11,12])     # Protocol 2
    ],

    # Analysis parameters
    poi = "UNIPROT_ID_OF_BAIT",  # Protein of interest
    refID = 1,                    # Index of reference protein (bait)
    n_controls = 10,              # Total control samples
    n_samples = 12,               # Total bait samples
    normalise_protocols = true,   # Normalize across protocols

    # Output files
    H0_file = "copula_H0.xlsx",          # Null hypothesis copula parameters
    results_file = "final_results.xlsx",  # Main results table
    volcano_file = "volcano_plot.svg",    # Volcano visualization
    convergence_file = "convergence.svg"  # Convergence diagnostics
)

# Run complete analysis pipeline
results = run_analysis(config)
```

### Understanding the Output

The `results` DataFrame contains:

- **`Protein`**: Protein identifier
- **`BF_enrichment`**, **`BF_correlation`**, **`BF_detection`**: Individual Bayes factors from each model (>1 supports interaction)
- **`Combined_BF`**: Joint Bayes factor from copula combination
- **`Posterior_Probability`**: Probability of genuine interaction (0-1)
- **`log2FC_mean`**, **`log2FC_median`**: Estimated enrichment
- **`pd`**: Probability of direction (directional evidence strength)
- **`rope_percentage`**: Percentage in region of practical equivalence

**Interpretation guideline**:
- **Posterior Probability > 0.95**: Strong evidence for interaction
- **Posterior Probability 0.75-0.95**: Moderate evidence
- **Posterior Probability < 0.25**: Strong evidence against interaction

### Loading Data

```julia
using BayesInteractomics

# Load data from multiple files with specified column mappings
data = load_data(
    ["experiment1.xlsx", "experiment2.csv"],
    sample_cols = [
        Dict(1 => [5,6,7], 2 => [8,9,10]),
        Dict(1 => [5,6,7])
    ],
    control_cols = [
        Dict(1 => [2,3,4]),
        Dict(1 => [2,3,4])
    ],
    normalise_protocols = true
)

# The data structure is hierarchical:
# InteractionData → Protocols → Experiments → Samples
```

### Running Individual Models

For more control, run individual components:

```julia
# Compute Bayes factors for a single protein
protein_data = getProteinData(data, protein_index)

# Beta-Bernoulli model (detection)
bf_detection = betaBernoulli(protein_data, n_controls, n_samples)

# Hierarchical Bayesian Model (enrichment)
hbm_result = HierarchicalBayesianModel(protein_data, data)

# Bayesian regression (correlation)
reg_result = RegressionModel(protein_data, data, refID)

# Access Bayes factors
bf_enrichment = hbm_result.BF
bf_correlation = reg_result.BF
```

## Documentation

For comprehensive documentation, tutorials, and examples, visit:
**[https://ma-seefelder.github.io/BayesInteractomics.jl](https://ma-seefelder.github.io/BayesInteractomics.jl)**

Documentation sections:
- **Tutorial**: Step-by-step guide for beginners
- **Mathematical Background**: Detailed explanation of statistical models
- **User Guide**: In-depth coverage of all features
- **Examples**: Real-world analysis workflows
- **API Reference**: Complete function documentation

## Examples

The `examples/` directory contains complete analysis workflows:

- **`hap40_interactome.jl`**: HAP40 protein interactome analysis with multiple protocols
- **`meta_analysis_workflow.jl`**: Meta-analysis combining evidence across experiments

## Scientific Background

BayesInteractomics implements the statistical framework described in:

> Seefelder et al. (2025). "Bayesian Integration of Multiple Evidence Sources for Protein Interactome Analysis." *(In preparation)*

The three-model approach provides:
- **Complementary evidence**: Different models capture different aspects of interaction
- **Quantified uncertainty**: Full posterior distributions rather than point estimates
- **Flexible integration**: Copulas model complex dependencies between evidence types
- **Robustness**: Hierarchical structure shares information across protocols while accounting for heterogeneity

## Performance

BayesInteractomics is optimized for throughput:
- **Parallel processing**: Automatic multi-threading across proteins
- **Efficient inference**: Variational Bayes via RxInfer.jl for fast convergence
- **Caching**: Intermediate results cached to enable resumption

Typical performance: ~1000 proteins analyzed in 5-15 minutes on an 8-core machine.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests on [GitHub](https://github.com/ma-seefelder/BayesInteractomics.jl).

For development:

```bash
git clone https://github.com/ma-seefelder/BayesInteractomics.jl.git
cd BayesInteractomics.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

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

BayesInteractomics.jl is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This package builds on excellent Julia packages including:
- [RxInfer.jl](https://github.com/reactivebayes/RxInfer.jl) for Bayesian inference
- [Copulas.jl](https://github.com/lrnv/Copulas.jl) for copula modeling
- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) for probability distributions
- [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) for data manipulation

## Contact

For questions, suggestions, or collaboration inquiries:
- **Author**: Manuel Seefelder
- **Email**: manuel.seefelder@uni-ulm.de
- **GitHub**: [https://github.com/ma-seefelder/BayesInteractomics.jl](https://github.com/ma-seefelder/BayesInteractomics.jl)
