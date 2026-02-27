# Diagnostics and Model Validation

## Overview

BayesInteractomics provides a comprehensive diagnostics toolkit to validate the statistical models and assess result reliability. The diagnostics module includes:

- **Posterior Predictive Checks (PPC)**: Simulate data from fitted models and compare to observations
- **Residual Analysis**: Detect model misspecification through standardized residuals
- **Calibration Assessment**: Verify that predicted probabilities match empirical discovery rates
- **Prior Sensitivity Analysis**: Evaluate how robust results are to prior choices

All diagnostics can be run automatically via the `run_analysis` pipeline by setting `run_diagnostics=true` and `run_sensitivity=true` in the [`CONFIG`](@ref) struct.

## Quick Start

```julia
using BayesInteractomics

config = CONFIG(
    datafile = ["data.xlsx"],
    control_cols = [Dict(1 => [2,3,4])],
    sample_cols = [Dict(1 => [5,6,7])],
    poi = "BAIT",
    refID = 1,
    n_controls = 3,
    n_samples = 3,
    # Enable diagnostics
    run_diagnostics = true,
    run_sensitivity = true,
    diagnostics_config = DiagnosticsConfig(
        n_proteins_to_check = 50,
        n_ppc_draws = 500
    )
)

final_df, result = run_analysis(config)
```

When enabled, diagnostic flags and sensitivity metrics are automatically merged into the final results DataFrame.

## Posterior Predictive Checks

PPC validates model fit by re-running inference for a subset of proteins and comparing simulated data to the observations.

### How It Works

1. Select proteins for checking (stratified by posterior probability)
2. Re-run Bayesian inference for each selected protein
3. Draw posterior predictive samples from the fitted model
4. Compare simulated data statistics to observed data

### Configuration

```julia
DiagnosticsConfig(
    n_proteins_to_check = 50,    # Number of proteins to check
    n_ppc_draws = 500,           # Posterior predictive draws per protein
    seed = 42,                   # RNG seed for reproducibility
    protein_selection = :stratified,  # :stratified, :random, or :top
    residual_model = :both,      # :hbm, :regression, or :both
    calibration_bins = 10        # Number of bins for calibration plot
)
```

### PPC Plots

```julia
# After running diagnostics through run_analysis, plots are saved automatically.
# Manual usage:
ppc_density_plot(ppc_result)         # Observed vs. predicted density
ppc_pvalue_histogram(diagnostics)    # Distribution of PPC p-values
bb_ppc_summary_plot(bb_ppcs)         # Beta-Bernoulli PPC summary
```

A well-calibrated model produces a uniform distribution of PPC p-values.

## Residual Analysis

Standardized residuals reveal systematic deviations between model predictions and data.

### Available Diagnostics

```julia
# Q-Q plot: residuals should follow a standard normal distribution
residual_qq_plot(residual_result)

# Scale-location plot: detect heteroscedasticity
scale_location_plot(residual_result)

# Distribution plot: inspect residual shape
residual_distribution_plot(residual_result)
```

### Interpretation

- **Q-Q plot**: Points on the diagonal indicate well-behaved residuals
- **Scale-location plot**: A flat trend indicates constant variance (homoscedasticity)
- **Heavy tails in Q-Q plot**: Consider using robust regression (`regression_likelihood = :robust_t`)

## Calibration Assessment

Calibration plots verify whether predicted posterior probabilities match empirical discovery rates.

```julia
# Standard calibration (all 3 BFs > 1.0 as positive criterion)
calibration_plot(calibration_result)

# Compare strict vs. relaxed calibration criteria
calibration_comparison_plot(diagnostics_result)
```

### Calibration Criteria

Three calibration strategies are computed automatically:

| Strategy | Positive criterion | Use case |
|---|---|---|
| **Strict** | All 3 BFs > 1.0 | Conservative, low false positives |
| **Relaxed** | ≥2 of 3 BFs > 1.0 | Moderate, handles noisy evidence |
| **Enrichment-only** | BF\_enrichment > 3.0 | When detection/correlation are unreliable |

Perfect calibration means the points lie on the diagonal.

## Prior Sensitivity Analysis

Evaluates how robust posterior probabilities are to different prior specifications.

### Configuration

```julia
SensitivityConfig(
    # Beta-Bernoulli prior grids
    bb_alpha_grid = [0.5, 1.0, 2.0, 5.0],
    bb_beta_grid = [0.5, 1.0, 2.0, 5.0],
    # EM prior grids
    em_prior_settings = [
        :default, :permissive, :stringent,
        (α = 10.0, β = 190.0),
        (α = 50.0, β = 100.0)
    ]
)
```

### Sensitivity Plots

```julia
# Tornado plot: which prior has the largest impact?
sensitivity_tornado_plot(sensitivity_result, n_top = 20)

# Heatmap: posterior probability across prior settings
sensitivity_heatmap(sensitivity_result, n_top = 20)

# Rank correlation: do rankings change with different priors?
sensitivity_rank_correlation(sensitivity_result)
```

### Interpretation

- **Low sensitivity (range < 0.1)**: Results are robust; prior choice does not matter
- **Moderate sensitivity (range 0.1–0.3)**: Inspect individual proteins; borderline calls may change
- **High sensitivity (range > 0.3)**: Consider collecting more data or using a more conservative threshold

Per-protein sensitivity metrics (std, min, max, range) are merged into `final_results.xlsx` when both `run_diagnostics` and `run_sensitivity` are enabled.

## Model Comparison (WAIC)

When `run_model_comparison = true` (default), BayesInteractomics fits both Normal and robust (Student-t) regression models and compares them via the Widely Applicable Information Criterion (WAIC).

```julia
config = CONFIG(
    # ...
    run_model_comparison = true,
    regression_likelihood = :robust_t,
    student_t_nu = 5.0,
    optimize_nu = true    # Optimize ν via Brent's method
)
```

### ν Optimization

When `optimize_nu = true`, the degrees-of-freedom parameter ν is optimized over [3, 50] by minimizing WAIC. The result is shown in a diagnostic plot:

```julia
nu_optimization_plot(nu_result)
```

## Diagnostic Flags

The diagnostics pipeline computes per-protein flags that are merged into the results:

| Flag | Meaning |
|---|---|
| `low_data` | Fewer than 3 non-missing observations |
| `residual_outlier` | Standardized residual > 3 in magnitude |
| `ppc_fail_hbm` | HBM PPC p-value < 0.05 |
| `ppc_fail_regression` | Regression PPC p-value < 0.05 |
| `high_sensitivity` | Posterior range across priors > 0.3 |

These flags help identify proteins whose results should be interpreted with caution.

## API Reference

```@docs
model_diagnostics
sensitivity_analysis
DiagnosticsConfig
SensitivityConfig
SensitivityResult
CalibrationResult
ResidualResult
ProteinPPC
BetaBernoulliPPC
```

### Diagnostic Plots

```@docs
ppc_density_plot
ppc_pvalue_histogram
residual_qq_plot
scale_location_plot
residual_distribution_plot
calibration_plot
calibration_comparison_plot
pit_histogram_plot
nu_optimization_plot
bb_ppc_summary_plot
sensitivity_tornado_plot
sensitivity_heatmap
sensitivity_rank_correlation
```
