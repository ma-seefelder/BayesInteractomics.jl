# Differential Interaction Analysis

## Overview

BayesInteractomics supports **differential interaction analysis** — comparing protein interaction profiles between two experimental conditions (e.g., wild-type vs. mutant, treated vs. untreated). This module identifies interactions that are gained, lost, or unchanged between conditions.

## Quick Start

```julia
using BayesInteractomics

# Option 1: From two CONFIG objects (end-to-end pipeline)
config_wt = CONFIG(datafile=["wt.xlsx"], ...)
config_mut = CONFIG(datafile=["mut.xlsx"], ...)

diff = differential_analysis(config_wt, config_mut,
    condition_A = "WT",
    condition_B = "Mutant",
    config = DifferentialConfig(q_threshold = 0.05)
)

# Option 2: From pre-computed AnalysisResult objects
_, result_wt = run_analysis(config_wt)
_, result_mut = run_analysis(config_mut)

diff = differential_analysis(result_wt, result_mut,
    condition_A = "WT",
    condition_B = "Mutant"
)
```

## Statistical Methodology

For each protein present in both conditions, the module computes:

### 1. Differential Bayes Factor (dBF)

```
dBF = BF_A / BF_B
```

A `log₁₀(dBF) > 0` means stronger evidence for interaction in condition A.

### 2. Per-Evidence Differential

The same ratio is computed separately for enrichment, correlation, and detection Bayes factors, allowing you to diagnose which evidence type drives the differential signal.

### 3. Effect Size

```
Δlog₂FC = mean_log₂FC_A − mean_log₂FC_B
```

### 4. Differential Posterior Probability

```
P(differential | data) = |dBF| / (1 + |dBF|)
```

This is a direction-agnostic measure of evidence for any difference.

### 5. Multiple Testing Correction

Bayesian FDR q-values are computed on the differential posterior probabilities.

### 6. Interaction Classification

Proteins are classified into categories based on the classification method:

| Class | Meaning |
|---|---|
| `GAINED` | Interaction present in A but not in B (or stronger in A) |
| `REDUCED` | Interaction present in B but not in A (or stronger in B) |
| `UNCHANGED` | Similar interaction strength in both conditions |
| `BOTH_NEGATIVE` | Neither condition shows strong interaction evidence |
| `CONDITION_A_SPECIFIC` | Protein only detected in condition A |
| `CONDITION_B_SPECIFIC` | Protein only detected in condition B |

## Configuration

```julia
DifferentialConfig(
    # Significance thresholds
    q_threshold = 0.05,                 # FDR threshold
    dbf_threshold = 1.0,                # |log10(dBF)| threshold
    delta_log2fc_threshold = 1.0,       # |Δlog2FC| threshold
    posterior_threshold = 0.8,          # Min posterior for "interactor"

    # Classification method
    classification_method = :posterior,  # :posterior, :dbf, or :combined

    # Output file paths
    results_file = "differential_results.xlsx",
    volcano_file = "differential_volcano.png",
    evidence_file = "differential_evidence.png",
    scatter_file = "differential_scatter.png",
    classification_file = "differential_classification.png",
    ma_file = "differential_ma.png",

    # HTML report
    generate_report_html = true
)
```

### Classification Methods

- **`:posterior`** (default): Uses per-condition posterior probabilities. A protein is an "interactor" if `posterior > posterior_threshold`. GAINED = interactor in A but not B with Δlog₂FC ≥ 0.
- **`:dbf`**: Uses `|log₁₀(dBF)| > dbf_threshold` directly.
- **`:combined`**: Both posterior and dBF criteria must be satisfied.

## Visualization

Five diagnostic plots are generated automatically:

### Volcano Plot

```julia
plt = differential_volcano_plot(diff)

# Customization
plt = differential_volcano_plot(diff,
    x_axis = :delta_log2fc,     # or :log10_dbf (default)
    y_axis = :differential_q,   # or :differential_posterior
    x_clip = 4.0                # Fixed x-axis range
)
```

### Evidence Plot

Four-panel plot showing per-evidence-type differential Bayes factors:

```julia
plt = differential_evidence_plot(diff)
```

### Scatter Plot

Compare a metric between conditions:

```julia
plt = differential_scatter_plot(diff, metric = :posterior_prob)
# Also: :bf, :log2fc
```

### Classification Plot

Bar chart summarizing the count of proteins in each class:

```julia
plt = differential_classification_plot(diff)
```

### MA Plot

Detect systematic biases where differential enrichment correlates with overall abundance:

```julia
plt = differential_ma_plot(diff)
```

## Accessing Results

```julia
# Filter by classification
gained = gained_interactions(diff)
lost = lost_interactions(diff)
unchanged = unchanged_interactions(diff)

# Significant differential interactions (any direction)
sig = significant_differential(diff, q_threshold = 0.01)

# Export to Excel
export_differential(diff, "results.xlsx")

# Access full results DataFrame
diff.results   # All proteins with statistics
diff.config    # Configuration used
diff.n_gained  # Count summaries
```

## API Reference

```@docs
differential_analysis
DifferentialConfig
DifferentialResult
InteractionClass
gained_interactions
lost_interactions
unchanged_interactions
significant_differential
export_differential
differential_volcano_plot
differential_evidence_plot
differential_scatter_plot
differential_classification_plot
differential_ma_plot
```
