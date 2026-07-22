# Diagnostics

## Overview

BayesInteractomics provides a comprehensive diagnostics toolkit to validate the statistical models and assess result reliability. The diagnostics module includes:

- **Bayesian FDR (BFDR), local FDR, and PEP** — multiple-testing metrics with Storey monotone step-down
- **Per-protein diagnostic flags** — `:ok` / `:warning` / `:fail` triage from observation counts, residual outliers, PPC results, and prior sensitivity
- **PSIS-LOO Pareto k-hat** — per-protein reliability of LOO predictive density estimates (BMA stacking)
- **Classification stability traffic light** — `robust` / `sensitive` / `fragile` labels from prior sensitivity sweep
- **Posterior Predictive Checks (PPC)** — simulate data from fitted models and compare to observations
- **Residual Analysis** — detect model misspecification through standardized residuals
- **Calibration Assessment** — verify predicted probabilities match empirical discovery rates
- **Prior Sensitivity Analysis** — evaluate how robust results are to prior choices
- **Quality Gates** — automated statistical checks on EM mixture model fits
- **Copula Diagnostic Plots** — visual diagnostics for the copula EM fitting process
- **EM Convergence Diagnostics** — log-likelihood convergence and component assignment visualization

All diagnostics can be run automatically via the `run_analysis` pipeline by setting `run_diagnostics=true` and `run_sensitivity=true` in the [`CONFIG`](@ref) struct.

## BFDR, Local FDR, and PEP

> **BFDR (Bayesian FDR)** is the expected fraction of false positives among rejections at a given threshold.
> **PEP (Posterior Error Probability)** is the per-protein false-positive probability, equal to `1 - posterior_prob`. PEP is sometimes called "local FDR" in the literature.

Both metrics are derived from EM-derived posterior probabilities and ship in every results DataFrame as the `BFDR` and `PEP` columns.

### Storey monotone step-down (BFDR)

`bfdr()` (in `src/core/utils.jl`) implements Storey's monotone step-down correction (Storey, 2002): the BFDR sequence is enforced to be non-increasing when proteins are sorted by decreasing posterior probability. This guarantees a well-behaved FDR curve and avoids the BFDR "wiggles" that can otherwise appear near the decision boundary.

```julia
bfdr_vals = bfdr(combined_BFs)               # from BFs (default)
bfdr_vals = bfdr(posterior_probs; isBF=false) # from posterior probabilities
pep_vals  = pep(posterior_probs)             # PEP = 1 - posterior_prob

# After run_analysis, results DataFrame already has BFDR and PEP columns
high_conf = filter(row -> row.BFDR < 0.05, final_df)
strict    = filter(row -> row.PEP  < 0.01, final_df)
```

`q()` and `getQValues()` are deprecated aliases of `bfdr()` and `getBFDR()` and emit a deprecation warning.

References:
- Storey, J. D. (2002). *A direct approach to false discovery rates*. JRSS B, 64(3), 479-498.
- Storey & Tibshirani (2003). *Statistical significance for genomewide studies*. PNAS, 100(16), 9440-9445.

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
```

### Calibration Criteria

Three calibration strategies are computed automatically:

| Strategy | Positive criterion | Use case |
|---|---|---|
| **Strict** | All 3 BFs > 1.0 | Conservative, low false positives |
| **Relaxed** | >=2 of 3 BFs > 1.0 | Moderate, handles noisy evidence |
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
- **Moderate sensitivity (range 0.1-0.3)**: Inspect individual proteins; borderline calls may change
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
    optimize_nu = true    # Optimize v via Brent's method
)
```

### v Optimization

When `optimize_nu = true`, the degrees-of-freedom parameter v is optimized over [3, 50] by minimizing WAIC. The result is shown in a diagnostic plot:

```julia
nu_optimization_plot(nu_result)
```

## Per-Protein Diagnostic Flags

Each protein receives a triage label in the `diagnostic_flag` column of the results DataFrame:

| Flag | Meaning |
|---|---|
| `:ok` | All checks pass |
| `:warning` | At least one check raised a non-fatal concern |
| `:fail` | At least one check raised a fatal concern; results should be interpreted with caution |

The underlying flag categories (rolled up into the triage label) are:

| Category | Meaning |
|---|---|
| `low_data` | Fewer than 3 non-missing observations |
| `residual_outlier` | Standardized residual > 3 in magnitude |
| `ppc_fail_hbm` | HBM PPC p-value < 0.05 |
| `ppc_fail_regression` | Regression PPC p-value < 0.05 |
| `high_sensitivity` | Posterior range across the prior grid > 0.3 |
| `pareto_k_high` | PSIS-LOO Pareto k-hat > 0.7 (BMA only) |

These flags help identify proteins whose results should be interpreted with caution. The HTML report renders an interactive popover for each flag explaining the failed checks.

## Classification Stability Traffic Light

When prior sensitivity analysis is enabled (`run_sensitivity = true`), each protein receives a stability label in the `classification_stability` column derived from the prior sensitivity sweep:

| Label | Criteria |
|---|---|
| `robust` | Classification (P > 0.5 vs P < 0.5) does not change across any prior grid point |
| `sensitive` | Classification changes for some prior choices but not all |
| `fragile` | Classification varies widely; the result is essentially prior-driven |

The traffic light is displayed in the HTML report's results table and is the recommended primary stability metric when reporting a protein. See [`prior_sensitivity.md`](prior_sensitivity.md) for the underlying prior grid construction (constant-strength simplex, Empirical Bayes Dirichlet, BIC-weighted marginalization).

## PSIS-LOO Pareto k-hat (BMA)

When `combination_method = :bma`, BayesInteractomics uses LOO stacking weights (Yao et al. 2018) to combine the Copula and 3c-EM sub-models. The reliability of each per-protein LOO prediction is summarized by the **Pareto k-hat** statistic from Pareto-smoothed importance sampling (PSIS-LOO).

| k-hat range | Interpretation | Action |
|---|---|---|
| < 0.5 | Reliable | Use stacking weight as-is |
| 0.5 - 0.7 | Borderline | Inspect; potentially flag |
| > 0.7 | Unreliable | Flag (`pareto_k_high`); LOO estimate may be unstable |

```julia
result.bma_result.pareto_k    # per-protein Pareto k-hat (or nothing)
n_high = count(k -> k > 0.7, result.bma_result.pareto_k)
```

Reference: Vehtari, A., Gelman, A., & Gabry, J. (2017). *Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC*. Statistics and Computing, 27(5), 1413-1432.

## Quality Gates

### What It Does

Quality gates are automated statistical checks that validate the fitted EM mixture model before results are used. They detect pathological fits -- component collapse, contamination between classes, or poor separation -- that would corrupt posterior probabilities.

### Why It Matters

The 3-component EM mixture model can fail in subtle ways: the H0 and agnostic components may collapse to identical distributions, the H1 component may contaminate the null, or components may overlap so much that class assignments are meaningless. Quality gates catch these problems automatically.

### How to Use

```julia
# After running the analysis pipeline
final_df, result = run_analysis(config)

# Quality gates run automatically during the pipeline.
# For manual usage on a CombinedBayesResult:
gates = run_quality_gates(combined_result)
```

### Key Metrics Checked

| Gate | What it checks | Pass | Warn | Fail |
|---|---|---|---|---|
| **KS marginal fit** | Do fitted marginals match the data? (Kolmogorov-Smirnov test) | p > 0.05 | p in [0.01, 0.05] | p < 0.01 |
| **KL contamination** | Is there information leakage between H0 and H1? | KL > 1.0 | KL in [0.1, 1.0] | KL < 0.1 |
| **Component separation** | Are components distinguishable? (overlap coefficient) | OVL < 0.3 | OVL in [0.3, 0.6] | OVL > 0.6 |
| **Within-class correlation** | Are components internally consistent? | Low within-class r | Moderate r | High r (heterogeneous component) |

### Cross-References

See [`run_quality_gates`](@ref), [`QualityGateResult`](@ref), [`QualityGateCell`](@ref), [`KLContaminationResult`](@ref), [`ValidationResult`](@ref).

## Copula Diagnostic Plots

Visual diagnostics for the copula/EM fitting process help assess model quality beyond the automated quality gates.

### Bootstrap Confidence Intervals

```julia
copula_bootstrap_ci(combined_result)
copula_bootstrap_plot(combined_result)
```

Computes bootstrap confidence intervals for the mixture weights ($\pi_0, \pi_a, \pi_1$). Wide CIs suggest the mixture proportions are not well-determined by the data. Narrow, non-overlapping CIs indicate stable component estimation.

### Discordant Protein Analysis

```julia
disc = discordant_protein_analysis(combined_result)
discordant_protein_plot(combined_result)
```

Identifies proteins with **conflicting evidence** across the three BF types -- for example, strong enrichment but no detection, or strong detection but negative enrichment. These proteins are informative because they reveal cases where the evidence streams disagree, which is where the copula/latent class combination adds the most value. A large number of discordant proteins may indicate systematic experimental artifacts.

### Agnostic Zone Analysis

```julia
ag = agnostic_zone_analysis(combined_result)
agnostic_zone_plot(combined_result)
```

Visualizes proteins in the **uninformative zone** (BFs near 1 across all evidence types). These are proteins assigned primarily to the agnostic component. A well-behaved model should show a clear cluster of agnostic-zone proteins near the origin (log-BF = 0) with smooth transitions to H0 and H1.

### Within-Class Correlation

```julia
wcc = within_class_correlation(combined_result)
within_class_correlation_plot(combined_result)
```

Measures the **correlation structure within each EM component**. Under the conditional independence assumption (latent class model), within-class correlations should be near zero. High within-class correlations indicate residual dependence not captured by the mixture model, suggesting the copula model may be more appropriate.

### KL Divergence Between H0 and H1

```julia
kl = kl_h1_divergence(combined_result)
kl_divergence_plot(combined_result)
```

Computes the KL divergence between the H0 and H1 component distributions. Low KL divergence means the components are nearly identical and the model cannot distinguish interactors from non-interactors -- a serious problem. High KL divergence (> 2.0) indicates well-separated components.

## EM Convergence Diagnostics

### Convergence Plot

```julia
em_convergence_plot(combined_result)
```

Plots the **log-likelihood versus EM iteration** for all restarts. A well-behaved EM should show:
- Monotonically increasing log-likelihood after burn-in (guaranteed by the step-halving guard)
- Convergence to similar log-likelihood values across restarts (if not, the likelihood surface has multiple modes)
- Fast convergence (typically 20-50 iterations with SQUAREM acceleration)

If the log-likelihood shows non-monotonic behavior or different restarts converge to very different values, this indicates numerical instability or a poorly specified model.

### Component Assignment Plot

```julia
component_assignment_plot(combined_result)
```

Shows the **posterior assignment distribution** across proteins for each component. In a well-separated model, assignments should be **bimodal**: most proteins are confidently assigned to either H0 (near 0) or H1 (near 1), with relatively few proteins at intermediate values. A uniform distribution of assignments indicates poor separation between components.

### Step-Halving Guarantee

The EM implementation includes a step-halving guard that guarantees monotonic log-likelihood after the burn-in period. If the M-step parameter constraints cause a log-likelihood decrease, the update is reverted to the previous parameters. This means:
- The convergence plot should never show LL decreases after the burn-in iterations
- If step-halving reverts are frequent (visible as flat plateaus), the parameter constraints may be too aggressive

## API Reference

```@docs
model_diagnostics
generate_diagnostics_report
DiagnosticsConfig
CalibrationResult
ResidualResult
EnhancedResidualResult
PPCExtendedStatistics
ProteinPPC
BetaBernoulliPPC
ProteinDiagnosticFlag
```

Sensitivity analysis types and the sensitivity sweep entry points are documented on the [Prior Sensitivity](@ref) page.

### Diagnostic Plots

```@docs
ppc_density_plot
ppc_pvalue_histogram
residual_qq_plot
BayesInteractomics.scale_location_plot
residual_distribution_plot
calibration_plot
pit_histogram_plot
nu_optimization_plot
bb_ppc_summary_plot
```

### Quality Gate Types and Functions

```@docs
run_quality_gates
QualityGateResult
QualityGateCell
KLContaminationResult
ValidationResult
compute_kl_contamination
```

### Copula Diagnostic Functions

```@docs
kl_h1_divergence
kl_divergence_plot
within_class_correlation
within_class_correlation_plot
agnostic_zone_analysis
agnostic_zone_plot
copula_bootstrap_ci
copula_bootstrap_plot
discordant_protein_analysis
discordant_protein_plot
```

### EM Diagnostic Plots

```@docs
component_assignment_plot
em_convergence_plot
```
