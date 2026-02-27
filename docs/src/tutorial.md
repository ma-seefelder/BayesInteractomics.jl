# Tutorial: Getting Started with BayesInteractomics

This tutorial will walk you through your first protein interactome analysis using BayesInteractomics. By the end, you'll understand how to prepare your data, run the analysis pipeline, and interpret the results.

## Prerequisites

- Julia 1.9 or later installed
- Basic familiarity with the Julia REPL
- Mass spectrometry data (AP-MS or proximity labeling)
- Understanding of your experimental design (controls vs. samples, replicates)

## Installation

### Step 1: Install Julia

If you haven't already, download and install Julia from [julialang.org](https://julialang.org/downloads/).

### Step 2: Install BayesInteractomics

Open the Julia REPL and install the package:

```julia
using Pkg
Pkg.add(url="https://github.com/ma-seefelder/BayesInteractomics.jl")
```

This will install BayesInteractomics and all its dependencies. The first installation may take a few minutes.

### Step 3: Verify Installation

Load the package to verify it works:

```julia
using BayesInteractomics
```

If you see no errors, you're ready to proceed!

## Understanding Your Data

### Experimental Design

BayesInteractomics is designed for comparative proteomics experiments:

- **Control samples**: Negative controls (e.g., empty vector, no-bait)
- **Bait samples**: Samples expressing your protein of interest (bait)
- **Replicates**: Multiple biological or technical replicates for each condition
- **Protocols**: Different experimental methods (e.g., different AP-MS protocols, BioID vs. APEX)

### Data Format Requirements

Your data should be organized as:

- **Rows**: Proteins (one protein per row)
- **Columns**: Samples (one sample per column)
- **Values**: Protein intensities/abundances (log2 scale recommended)
- **First column**: Protein identifiers (e.g., UniProt IDs, gene names)
- **Missing data**: Leave cells empty or use NA/missing

Supported file formats:
- Excel (.xlsx) - recommended
- CSV (.csv)

Example structure:

```
| Protein      | Control_1 | Control_2 | Control_3 | Sample_1 | Sample_2 | Sample_3 |
|--------------|-----------|-----------|-----------|----------|----------|----------|
| PROTEIN_A    | 25.3      | 24.8      | 25.1      | 28.5     | 29.2     | 28.8     |
| PROTEIN_B    | 22.1      | NA        | 22.3      | 22.5     | 22.0     | 22.8     |
| BAIT_PROTEIN | 18.2      | 18.5      | 18.0      | 32.1     | 31.8     | 32.5     |
```

## Step-by-Step Analysis

### Step 1: Prepare Your File Paths

First, define where your data files are located:

```julia
using BayesInteractomics

# Define base directory
base_dir = "/path/to/your/data"

# Data files (can be multiple protocols)
data_files = [joinpath(base_dir, "protocol1_data.xlsx")]

# Output directory
output_dir = joinpath(base_dir, "results")
mkpath(output_dir)  # Create directory if it doesn't exist
```

### Step 2: Specify Column Mappings

BayesInteractomics needs to know which columns contain your samples and controls. Use dictionaries to map experiment IDs to column indices:

```julia
# Example: Single protocol with 2 experiments
# Experiment 1: columns 2-4 are controls, 5-7 are samples
# Experiment 2: columns 8-10 are controls, 11-13 are samples

control_columns = [
    Dict(
        1 => [2, 3, 4],      # Experiment 1 controls
        2 => [8, 9, 10]      # Experiment 2 controls
    )
]

sample_columns = [
    Dict(
        1 => [5, 6, 7],      # Experiment 1 samples
        2 => [11, 12, 13]    # Experiment 2 samples
    )
]
```

**Important**: Column indices are 1-based (Julia convention). Column 1 is typically protein IDs.

### Step 3: Configure the Analysis

Create a `CONFIG` struct with all analysis parameters:

```julia
config = CONFIG(
    # Input data
    datafile = data_files,
    control_cols = control_columns,
    sample_cols = sample_columns,

    # Protein identification
    poi = "UNIPROT_ID_OF_BAIT",  # Your bait protein identifier
    refID = 1,                    # Row index of bait protein in data

    # Sample sizes
    n_controls = 6,   # Total number of control samples (3 + 3 in example)
    n_samples = 6,    # Total number of bait samples (3 + 3 in example)

    # Protocol normalization
    normalise_protocols = true,  # Set true if combining multiple protocols

    # Output files
    H0_file = joinpath(output_dir, "copula_H0.xlsx"),
    results_file = joinpath(output_dir, "interaction_results.xlsx"),
    volcano_file = joinpath(output_dir, "volcano_plot.svg"),
    convergence_file = joinpath(output_dir, "convergence_diagnostics.svg"),
    evidence_file = joinpath(output_dir, "evidence_distribution.svg"),

    # Visualization options
    plotHBMdists = false,      # Plot HBM distributions for each protein
    plotlog2fc = false,        # Plot log2 fold change distributions
    plotregr = false,          # Plot regression fits
    plotbayesrange = false,    # Plot Bayes factor ranges

    # Other options
    verbose = true             # Print progress messages
)
```

### Step 4: Run the Analysis

Now execute the complete analysis pipeline:

```julia
# This runs all three models, fits the copula mixture, and generates results
results = run_analysis(config)
```

The analysis will:
1. Load and validate your data
2. Compute null hypothesis copula (H0) from negative proteins
3. Analyze each protein with three statistical models (in parallel)
4. Fit mixture copula to combine evidence
5. Generate results table and visualizations

This typically takes 5-15 minutes for ~1000 proteins on a modern laptop.

### Step 5: Explore the Results

The `results` is a DataFrame containing:

```julia
# View column names
names(results)

# View top 10 interactions by posterior probability
top_hits = first(sort(results, :Posterior_Probability, rev=true), 10)
println(top_hits)

# Filter for high-confidence interactions
high_confidence = filter(row -> row.Posterior_Probability > 0.95, results)
println("Found $(nrow(high_confidence)) high-confidence interactions")
```

## Interpreting Results

### Key Output Columns

1. **`Protein`**: Protein identifier from your input data

2. **Individual Bayes Factors**:
   - `BF_enrichment`: Evidence from quantitative enrichment (HBM)
   - `BF_correlation`: Evidence from dose-response correlation (regression)
   - `BF_detection`: Evidence from detection probability (Beta-Bernoulli)
   - Values > 1 support interaction; > 10 is strong; > 100 is very strong

3. **Combined Evidence**:
   - `Combined_BF`: Joint Bayes factor from copula combination
   - `Posterior_Probability`: Final probability of genuine interaction (0-1 scale)

4. **Enrichment Statistics**:
   - `log2FC_mean`, `log2FC_median`: Estimated log2 fold change
   - `log2FC_sd`: Uncertainty in enrichment
   - `pd`: Probability of direction (% of posterior > 0)
   - `rope_percentage`: % of posterior in "region of practical equivalence" (near zero)

5. **Quality Metrics**:
   - `ess_bulk`, `ess_tail`: Effective sample size (should be > 400 for reliable inference)
   - `rhat`: Convergence diagnostic (should be < 1.01)

### Decision Thresholds

Recommended interpretation:

| Posterior Probability | Interpretation |
|----------------------|----------------|
| > 0.95 | **Strong evidence** for interaction - high confidence hit |
| 0.75 - 0.95 | **Moderate evidence** - promising candidate, validate experimentally |
| 0.25 - 0.75 | **Ambiguous** - insufficient evidence either way |
| < 0.25 | **Strong evidence against** - likely non-specific or contaminant |

**Important**: Always consider biological context and validate key interactions experimentally!

### Visualizations

BayesInteractomics generates several diagnostic plots:

1. **Volcano Plot** (`volcano_plot.svg`):
   - X-axis: log2 fold change (enrichment)
   - Y-axis: -log10(1 - Posterior Probability)
   - Points colored by posterior probability
   - Helps identify enriched and high-probability interactions

2. **Convergence Diagnostics** (`convergence_diagnostics.svg`):
   - Shows model convergence for proteins
   - Check that most proteins have good convergence (low Rhat)

3. **Evidence Distribution** (`evidence_distribution.svg`):
   - Shows distribution of Bayes factors across proteins
   - Helps assess overall data quality

## Advanced Usage

### Multiple Protocols

If you have data from different experimental methods (e.g., AP-MS and BioID):

```julia
config = CONFIG(
    datafile = [
        "data/apms_protocol.xlsx",
        "data/bioid_protocol.xlsx"
    ],
    control_cols = [
        Dict(1 => [2,3,4]),      # AP-MS controls
        Dict(1 => [2,3])         # BioID controls
    ],
    sample_cols = [
        Dict(1 => [5,6,7]),      # AP-MS samples
        Dict(1 => [4,5])         # BioID samples
    ],
    normalise_protocols = true,  # IMPORTANT: set true for multiple protocols
    # ... rest of configuration
)
```

### Multiple Imputation

For datasets with substantial missing data, use multiple imputation:

```julia
# First, impute your data using your preferred method
# (e.g., MissForest, KNN, etc.) to create M imputed datasets

# Load imputed datasets
imputed_data = [
    load_data([imputed_file_1], sample_cols, control_cols),
    load_data([imputed_file_2], sample_cols, control_cols),
    # ... M imputed datasets
]

# Load raw data for detection model
raw_data = load_data(["raw_data.xlsx"], sample_cols, control_cols)

# Run analysis across imputations
results = run_analysis(imputed_data, raw_data, config)
```

The package will pool results across imputations following Rubin's rules.

## Troubleshooting

### Common Issues

**Problem**: "Column index out of bounds"
- **Solution**: Check that your column indices are correct and 1-based. Remember column 1 is typically protein IDs.

**Problem**: "refID protein not found"
- **Solution**: Verify that `refID` corresponds to the correct row number of your bait protein in the data file.

**Problem**: "Not enough control/sample columns"
- **Solution**: Ensure `n_controls` and `n_samples` match the total number of columns specified in `control_cols` and `sample_cols`.

**Problem**: Poor convergence (high Rhat values)
- **Solution**: This can happen for proteins with extreme missing data or very low signal. Filter results by `rhat < 1.01` for reliable proteins only.

**Problem**: All posterior probabilities near 0.5
- **Solution**: Check data quality. This suggests weak signal-to-noise. Consider:
  - Are controls truly negative?
  - Is bait protein detected at sufficient levels?
  - Are there enough replicates?

### Getting Help

If you encounter issues not covered here:

1. Check the [User Guide](analysis.md) for detailed explanations
2. Review [Examples](examples.md) for similar use cases
3. Open an issue on [GitHub](https://github.com/ma-seefelder/BayesInteractomics.jl/issues)

## Next Steps

Now that you've completed your first analysis:

1. **Explore the User Guide**: Learn about individual model components and customization
2. **Read Mathematical Background**: Understand the statistical theory behind the methods
3. **Study Examples**: See real-world analysis workflows
4. **Experiment with Visualization**: Customize plots for your publication needs

### Recommended Reading

- [Data Loading](data_loading.md): Detailed data structure explanation
- [Model Fitting](model_fitting.md): Understanding the three statistical models
- [Model Evaluation](model_evaluation.md): Advanced interpretation of Bayes factors
- [Mathematical Background](mathematical_background.md): Statistical theory and derivations

## Summary

You've learned how to:
- ✅ Install and set up BayesInteractomics
- ✅ Prepare and format your mass spectrometry data
- ✅ Configure and run a complete analysis pipeline
- ✅ Interpret Bayes factors and posterior probabilities
- ✅ Identify high-confidence protein interactions
- ✅ Troubleshoot common issues

Welcome to principled Bayesian interactome analysis!
