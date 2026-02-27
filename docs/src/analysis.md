# Analysis Pipeline

## Overview

The analysis module orchestrates the complete workflow for identifying protein-protein interactions. It coordinates data loading, statistical modeling, evidence combination, and result generation into a seamless pipeline.

BayesInteractomics provides two levels of entry points:

- **`run_analysis(config)`**: High-level function that handles everything from data loading to final results, with intelligent caching
- **`analyse(data, ...)`**: Lower-level function that runs the core Bayesian pipeline on pre-loaded data

## Pipeline Architecture

### `run_analysis` Workflow

```
CONFIG
  │
  ├─ load_data() ──── Data loading & curation
  │
  ├─ Cache check ──── Skip if config+data unchanged
  │
  ├─ analyse() ────── Core Bayesian pipeline
  │   ├── Step 1: Beta-Bernoulli (detection BFs)
  │   ├── Step 2: HBM + Regression (enrichment & correlation BFs)
  │   └── Step 3: Evidence combination (copula-EM or latent class)
  │
  ├─ Model comparison ── WAIC (Normal vs. robust regression)
  │
  ├─ Diagnostics ──── PPC, residuals, calibration (optional)
  │
  ├─ Sensitivity ──── Prior sensitivity analysis (optional)
  │
  ├─ Visualization ── Volcano, convergence, evidence plots
  │
  └─ Report ────────── Interactive HTML report (optional)
```

### Core Pipeline (`analyse`)

The `analyse()` function implements a multi-stage Bayesian workflow:

#### Step 1: H0 Computation
- Computes or loads null hypothesis Bayes factors from permuted data
- Stores null distribution in `copula_H0.xlsx`
- Required for copula-based evidence combination

#### Step 2: Beta-Bernoulli Detection Model
Parallelized across all proteins:
- Estimates detection probabilities using a Beta-Bernoulli model
- Generates Bayes factors for detection evidence
- Thread-safe computation with progress bar

#### Step 3: Hierarchical Bayesian Model + Regression
Parallelized analysis of enrichment and correlation:
- **Enrichment (HBM)**: Log₂ fold-change across conditions using variational Bayes
- **Correlation (Regression)**: Dose-response relationships via Bayesian linear regression
- Supports both Normal and robust (Student-t) regression likelihoods
- Results cached in thread-specific files under `cache/`

#### Step 4: Evidence Combination
Two methods available:
- **Copula-EM** (`:copula`, default): Fits mixture of copulas with EM algorithm
- **Latent Class** (`:latent_class`): Variational message passing approach

## Configuration

All analysis parameters are centralized in the [`CONFIG`](@ref) struct:

```julia
config = CONFIG(
    # Required: data and column specifications
    datafile = ["experiment.xlsx"],
    sample_cols = [Dict(1 => [5,6,7])],
    control_cols = [Dict(1 => [2,3,4])],
    poi = "BAIT_PROTEIN",

    # Output paths (auto-generated from basedir)
    output = OutputFiles("/path/to/results"),
    # or: output = OutputFiles("/path/to/results", image_ext=".svg")

    # Analysis parameters
    n_controls = 3,
    n_samples = 3,
    refID = 1,
    normalise_protocols = true,

    # Evidence combination
    combination_method = :copula,        # or :latent_class
    em_prior = :default,                 # or :APMS, :BioID, :TurboID, :stringent, :permissive
    em_n_restarts = 20,
    copula_criterion = :BIC,

    # Regression model
    regression_likelihood = :robust_t,   # or :normal
    student_t_nu = 5.0,
    run_model_comparison = true,
    optimize_nu = true,

    # Diagnostics (optional)
    run_diagnostics = false,
    run_sensitivity = true,

    # Data curation
    curate = true,
    species = 9606,
    bait_name = "BAIT_PROTEIN",

    # Report
    generate_report_html = true
)
```

### Output Files

All output paths are managed through the [`OutputFiles`](@ref) struct:

```julia
output = OutputFiles("/path/to/results")
# Auto-generates:
#   copula_H0.xlsx, final_results.xlsx, volcano_plot.png,
#   convergence.png, evidence.png, em_diagnostics.png,
#   interactive_report.html, methods.md, ...

# Override individual paths:
output.results_file = "custom_name.xlsx"

# Use SVG instead of PNG:
output = OutputFiles("/path/to/results", image_ext = ".svg")
```

## Caching

### Result-Level Caching

`run_analysis` uses hash-based caching to skip redundant computation:

```julia
# First run: performs full analysis and caches
final_df, result = run_analysis(config)

# Second run with same config/data: loads from cache
final_df, result = run_analysis(config)

# Force re-computation
final_df, result = run_analysis(config, use_cache = false)
```

Cache files are stored in `.bayesinteractomics_cache/` next to the first data file.

### Intermediate Caching

Individual pipeline steps can be cached separately:

```julia
results = analyse(data, "copula_H0.xlsx",
    use_intermediate_cache = true,
    betabernoulli_cache_file = "bb_cache.jld2",
    hbm_regression_cache_file = "hbm_cache.jld2"
)
```

This is useful when iterating on copula-EM parameters without re-running inference.

## Parallel Processing

The package uses Julia's multi-threading:

```bash
# Launch Julia with multiple threads
julia --threads=auto
```

- **Thread Safety**: Each thread writes to separate cache files
- **Load Balancing**: Proteins distributed evenly across threads
- **Error Handling**: Failures logged per-protein to `log.txt` without blocking others

## Multiple Imputation

For handling missing data with multiple imputation:

```julia
# Pass vector of imputed datasets + raw data
results = analyse(
    imputed_data_vector,  # Vector{InteractionData}
    raw_data,             # InteractionData (for Beta-Bernoulli)
    "copula_H0.xlsx",
    n_controls = 3, n_samples = 3, refID = 1
)
```

Results are automatically pooled:
- HBM and regression are run on each imputed dataset
- Pooled using `evaluate_imputed_fc_posteriors()` (Rubin's rules)
- Beta-Bernoulli uses raw (non-imputed) data

## Evidence Combination Methods

### Copula-EM (default)

```julia
combination_method = :copula
```

- Fits mixture: `π₀ · Copula(H₀) + π₁ · Copula(H₁)`
- Automatic copula selection (Clayton, Frank, Gumbel, Gaussian, Joe)
- Multiple restarts to avoid local optima
- SQUAREM acceleration for faster convergence
- Supports informative priors on π₁

### Latent Class

```julia
combination_method = :latent_class
```

- Variational message passing approach
- Does not require pre-computed H0 file
- Winsorization of extreme Bayes factors
- Dirichlet prior on class proportions

## Output Structure

`run_analysis` returns `(final_df, analysis_result)`:

### `final_df` (DataFrame)

| Column | Description |
|---|---|
| `Protein` | Protein identifier |
| `BF` | Combined Bayes factor |
| `posterior_prob` | Posterior probability of interaction |
| `q` | Bayesian FDR q-value |
| `mean_log2FC` | Mean log₂ fold change |
| `bf_enrichment` | HBM Bayes factor |
| `bf_correlation` | Regression Bayes factor |
| `bf_detected` | Beta-Bernoulli Bayes factor |
| `diagnostic_flag` | Quality flags (when diagnostics enabled) |
| `sensitivity_range` | Prior sensitivity range (when enabled) |

### `analysis_result` (AnalysisResult)

Contains all analysis outputs including copula results, EM model, joint distributions, and caching metadata. Use accessor functions:

```julia
getProteins(result)              # Protein names
getBayesFactors(result)          # Combined BFs
getPosteriorProbabilities(result) # Posterior probabilities
getQValues(result)               # q-values
getMeanLog2FC(result)            # Mean log₂FC
```

## API Reference

```@docs
run_analysis
analyse
CONFIG
OutputFiles
AnalysisResult
save_result
load_result
check_cache
evaluate_imputed_fc_posteriors
```
