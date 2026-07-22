# Model Evaluation

## Overview

After running the analysis pipeline, model evaluation tools help assess result quality, validate predictions, and interpret findings. This module provides:

- Bayes factor and posterior interpretation conventions
- The three evidence-combination methods (`combination_method ∈ {:copula, :latent_class, :bma}`) and how to choose between them
- The 3c-EM (3-component latent class) decomposition with Student-t H0, anchored Agnostic, sigmoid-gated H1
- The Copula sub-model (three-component mixture, a single 3-D copula per component)
- **Bayesian Model Averaging** (`:bma`): linear BF pooling across Copula and 3c-EM, LOO stacking weights, sub-model BF columns, disagreement diagnostics, Pareto k-hat
- Simulation-based calibration (Platt scaling) and automated quality gates

The recommended default is `combination_method = :bma`. Use `:copula` or `:latent_class` only when you specifically want to inspect a single sub-model.

## Understanding Bayesian Evidence

### Bayes Factor Interpretation

The Bayes factor (BF) quantifies the ratio of evidence for interaction (H1) versus no interaction (H0):

| Bayes Factor | Evidence Strength | Interpretation |
|---|---|---|
| > 100 | Decisive | Very strong evidence for interaction |
| 10-100 | Strong | Strong evidence for interaction |
| 3-10 | Moderate | Moderate evidence for interaction |
| 1-3 | Weak | Weak evidence for interaction |
| 0.33-1 | Weak | Weak evidence against interaction |
| 0.1-0.33 | Moderate | Moderate evidence against interaction |
| < 0.1 | Strong | Strong evidence against interaction |

### Converting to Posterior Probability

The posterior probability of an interaction given the data is computed from the Bayes factor:

$$P(H_1|data) = \frac{BF \cdot P(H_1)}{BF \cdot P(H_1) + P(H_0)}$$

With uniform priors (P(H0) = P(H1) = 0.5):

| Bayes Factor | Posterior Probability | Confidence |
|---|---|---|
| 99 | 0.99 | 99% |
| 9 | 0.90 | 90% |
| 3 | 0.75 | 75% |
| 1 | 0.50 | 50% (inconclusive) |

## Model Performance Evaluation

### Calibration Assessment

Check if posterior probabilities match actual discovery rates:

1. **Generate predictions** across range of probability thresholds
2. **Validate with independent data** or orthogonal methods
3. **Compare observed vs expected** interaction rates
4. **Adjust threshold** if systematic bias detected

### Confidence by Evidence Source

Evaluate which models contribute most to predictions:

- **Detection-dominated**: Reliable detection but weak enrichment evidence
- **Enrichment-dominated**: Strong fold-change but variable detection
- **Correlation-dominated**: Dose-response but limited to titration experiments
- **Well-balanced**: All three evidence sources agree strongly

Well-balanced results are most trustworthy and reproducible.

## Result Ranking and Filtering

### High-Confidence Interactions

Recommended filtering criteria:

```julia
# Strong candidates for validation
strong_interactions = filter(row -> row.posterior_prob > 0.95, results)

# Moderate confidence (exploratory)
moderate = filter(row -> 0.75 < row.posterior_prob <= 0.95, results)

# Known interactions (positive controls)
known = filter(row -> row.posterior_prob > 0.5, results)
```

### Per-Protein Statistics

Analyze interaction profiles:

- **Number of interactions** per protein (hub vs peripheral)
- **Posterior probability distribution** across interactors
- **Evidence quality** (Bayes factors by model)
- **Reproducibility** (consistency across protocols)

### Protocol Comparison

When multiple experimental protocols are available:

1. **Individual protocol results**: Bayes factors per method
2. **Cross-protocol agreement**: Proteins detected in multiple methods
3. **Discrepancies**: Investigate proteins with conflicting evidence
4. **Meta-analysis**: Combine evidence across protocols

## Interpretation Guidelines

### What High Posterior Probability Means

A posterior probability of 0.95 indicates:

- Given your data and statistical assumptions
- There is 95% probability this protein truly interacts
- 5% probability it's a false positive (background)
- Assumes prior knowledge incorporated in priors

### What Could Go Wrong

**False Positives** (predicted interaction, actually background):
- Systematic contamination in control samples
- Inappropriate copula model for your data
- Prior misspecification

**False Negatives** (no interaction predicted, actually present):
- Proteins with variable detection across replicates
- Low abundance interactors below detection limit
- Transient or weak interactions

**Inconclusive Results** (BF near 1):
- Insufficient power (too few replicates)
- Conflicting evidence across models
- Data quality issues

### Validation Strategies

1. **Orthogonal Methods**
   - Co-immunoprecipitation (co-IP)
   - Yeast two-hybrid (Y2H)
   - Proximity labeling variants

2. **Literature Comparison**
   - Cross-reference with known interaction databases
   - Check PubMed for direct evidence

3. **Biological Validation**
   - Functional assays
   - Localization studies
   - Pathway analysis

## Multiple Testing Correction

When reporting interactions across many proteins:

### Bayesian FDR (BFDR) and PEP

BayesInteractomics handles multiple testing through the Bayesian framework using two complementary metrics:

- **PEP (Posterior Error Probability)** = `1 - posterior_prob`. Per-protein false-positive probability.
- **BFDR (Bayesian FDR)** = expected fraction of false positives among rejections at a given threshold. Implemented via Storey monotone step-down on EM-derived posteriors (`bfdr()` in `src/core/utils.jl`).

The Storey step-down enforces non-increasing BFDR values when sorted by decreasing posterior probability, so the curve is well-behaved for choosing thresholds.

```julia
# After run_analysis, results DataFrame already contains BFDR and PEP columns
selected = filter(row -> row.BFDR < 0.05, final_df)         # FDR control
high_conf = filter(row -> row.PEP < 0.01, final_df)         # Per-protein cap
```

The deprecated alias `q()` / `getQValues()` is preserved for backward compatibility but emits a deprecation warning; new code should use `bfdr()` / `getBFDR()`.

Reference: Storey, J. D. (2002). *A direct approach to false discovery rates*. Journal of the Royal Statistical Society B, 64(3), 479-498.

## 3-Component Latent Class (3c-EM)

The 3c-EM model fits a 3-component mixture directly on the natural-log scale of Bayes factors. Each component captures a qualitatively different protein population, and the EM is constrained to guarantee monotonic log-likelihood.

```julia
config = CONFIG(
    # ...
    combination_method = :latent_class,
    em_n_restarts = 20,
    lc_alpha_prior = :auto,                       # Empirical Bayes (Minka) + BIC grid marginalization
    lc_winsorize = true,
    lc_winsorize_quantiles = (0.01, 0.99),
)
```

### The three components

1. **H0 (Background)** — heavy-tailed null. The enrichment marginal is a **Student-t** distribution (replacing the Normal H0 used in v1.0). The heavy tails absorb extreme negative log-BFs without requiring a separate outlier component. Degrees of freedom are selected by BIC over `{3, 5, 7, 10}`.

2. **Anchored Agnostic** — uninformative middle class. The enrichment mean is **anchored at mu = 0** (not a free parameter). A KL divergence merge check (`KL < 0.1` triggers merge) guards against the H0 / Agnostic redundancy that occurred in earlier 3-component formulations.

3. **H1 (Interactor)** — sigmoid-gated transition around the Jeffreys threshold (`ln(sqrt(10)) ≈ 1.151` nats). The smooth gate replaces the v1.0 hard cutoff and preserves EM monotonicity. The H1 enrichment marginal family is **BIC-selected per dataset** from `{Gamma, LogNormal, Weibull}` at iteration 5.

### Convergence safeguards

- **Step-halving guard** — if a constrained M-step decreases the log-likelihood after burn-in, the update is reverted.
- **Multi-restart initialization** — 20+ restarts (quantile, k-means, random) to avoid local optima.
- **Empirical Bayes Dirichlet** — `lc_alpha_prior = :auto` triggers Minka fixed-point estimation of `alpha`, followed by BIC-weighted marginalization across an EB-centered grid (see [`prior_sensitivity.md`](prior_sensitivity.md)).

See [`LatentClassResult`](@ref), [`BayesInteractomics.combined_BF_latent_class`](@ref).

## Copula

```julia
combination_method = :copula
```

The Copula sub-model fits a **three-component mixture** over the enrichment / correlation / detection log-Bayes-factor triplet: H0 (background), an anchored Agnostic middle class, and H1 (interactor). MAP class assignments are warm-started from the 3c-EM latent-class fit, and the mixing weights use a `Beta(80, 20)` prior on the H1 proportion.

Each component is modelled by a **single 3-D copula** joining its three marginals — not a vine. The H0 and H1 copulas are BIC-selected from four single-parameter families (`Clayton`, `Frank`, `Gumbel`, `Gaussian`); the Agnostic component uses an independence copula. This captures joint dependence per component rather than assuming conditional independence within each class.

Multi-restart EM (MAP-EM, up to 20 restarts) uses quantile, k-means, and random initialization; SQUAREM acceleration provides 2-10x speedup. See [`mathematical_background.md`](mathematical_background.md) for the joint-density derivation.

The user-facing model name in BMA, results, and the HTML report is **"Copula"**.

## Bayesian Model Averaging

`combination_method = :bma` is the **recommended default**. It averages the Copula and 3c-EM sub-models via LOO stacking weights and produces calibrated combined posteriors that are robust to model specification.

### What it does

LOO stacking (Yao, Vehtari, Simpson, Gelman 2018) weights each sub-model by its leave-one-out predictive performance. The two models that are averaged are:

1. **Copula** (three-component mixture, a single 3-D copula per component; models joint dependence)
2. **3c-EM** (3-component latent class; conditional independence within each class, computationally stable)

```julia
config = CONFIG(
    # ...
    combination_method = :bma,    # default
)
```

### Linear BF pooling

BMA combines the two sub-models with **linear BF pooling**: the combined BF is the LOO-stacked weighted average of the sub-model BFs. The combined posterior is then derived from the pooled BF and the shared prior odds (from EM mixing weights). Pooling on the BF scale keeps a proper, prior-free likelihood ratio and preserves per-protein evidence gradation — averaging on the posterior scale instead would collapse strong sub-model evidence toward a prior-dependent constant.

```
BF_BMA = w_copula * BF_copula + w_3cem * BF_3cem
P_BMA  = (BF_BMA * prior_odds) / (1 + BF_BMA * prior_odds)
```

### Sub-model transparency: bf_em and bf_copula

The results DataFrame includes both sub-model BFs alongside the BMA combined BF, so reviewers can see exactly how much each sub-model contributed:

| Column | Meaning |
|---|---|
| `BF` | BMA combined Bayes factor |
| `bf_em` | 3c-EM sub-model Bayes factor |
| `bf_copula` | Copula sub-model Bayes factor |
| `posterior_prob` | BMA posterior probability |

The HTML report displays `bf_em` and `bf_copula` next to the combined column.

### Stacking weights

LOO stacking weights are computed once per dataset by solving:

```math
\hat{w} = \arg\max_{w \in \mathcal{S}_K} \sum_{i=1}^{N} \log \sum_{k=1}^{K} w_k \cdot p_k^{(-i)}(x_i)
```

A **5% weight floor** is applied (`w_k ≥ 0.05`) to prevent winner-take-all degeneracy and ensure that no model is entirely discarded. Weights are logged to the report and accessible on `BMAResult`:

```julia
result.bma_result.copula_weight   # Stacking weight for the Copula sub-model
result.bma_result.em_weight       # Stacking weight for the 3c-EM sub-model
```

### Disagreement diagnostics

When the two sub-models classify a protein differently — e.g., one says high-confidence interactor, the other says background — the protein is flagged on `BMAResult.model_disagreement` (a `BitVector`). Practical heuristic: flag triggers when `bf_em` and `bf_copula` differ by more than two orders of magnitude, or when they yield opposite classifications at `P = 0.5`.

```julia
n_disagree = count(result.bma_result.model_disagreement)
@info "Models disagree on $n_disagree proteins"
```

The HTML report surfaces these proteins in a dedicated panel for manual review.

### Pareto-k diagnostics (PSIS-LOO)

`BMAResult.pareto_k` (when computed) holds the per-protein PSIS-LOO Pareto k-hat statistic, which indicates how reliable the LOO estimate is for that observation. Heuristic interpretation: `k̂ < 0.5` reliable, `0.5-0.7` borderline, `> 0.7` unreliable. High-k̂ proteins should not be used to drive the stacking weights.

Reference: Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). *Using stacking to average Bayesian predictive distributions*. Bayesian Analysis, 13(3), 917-1007.

See [`BMAResult`](@ref), [`bma_weights_plot`](@ref), [`prior_sensitivity.md`](prior_sensitivity.md), [`mathematical_background.md`](mathematical_background.md).

## Pooled Results from Imputation

When using multiple imputation:

### Interpreting Pooled Statistics

- **Point estimates**: Averaged across imputations
- **Uncertainty inflation**: Total variance includes:
  - Within-imputation variance
  - Between-imputation variance (missing data)
- **Always use pooled posterior probabilities** (not individual imputations)

### Checking Imputation Quality

- **Convergence**: Estimates stable across imputations
- **Between-imputation variance**: Should be < 25% of total variance
- **Sensitivity**: Results robust to imputation method choice

## Simulation Engine

The simulation engine generates synthetic AP-MS datasets with known ground truth to evaluate the pipeline's sensitivity and specificity. It provides objective performance metrics without requiring external validation data.

**Why it matters:** In real AP-MS experiments, the true interaction status of most proteins is unknown. The simulation engine creates datasets where the ground truth is known by construction, allowing direct measurement of false discovery rate, sensitivity, and specificity under controlled conditions.

**How it works:**

The engine sweeps a 5x5 grid over two key parameters:
- **`pi_h1` (proportion of true interactors):** Varies from rare (2%) to common (20%) interactors
- **`effect_scale` (effect size multiplier):** Scales the H1 enrichment distribution from weak (0.5x) to strong (1.5x) effects

For each grid point, synthetic log-BF triplets (enrichment, correlation, detection) are sampled from the fitted latent class model parameters, with known labels. The full EM pipeline is then re-run on each synthetic dataset and performance metrics are computed.

**How to use:**

Simulation runs automatically when `run_simulation = true` in CONFIG (the default). To configure it manually:

```julia
config = CONFIG(
    # ...
    run_simulation = true    # Enable simulation (default)
)
```

Or call the simulation function directly:

```julia
sim_result = run_simulation(lc_result;
    n_synthetic = 10_000,                          # Proteins per scenario
    n_replicates = 10,                             # Replicates per scenario
    pi_h1_grid = [0.02, 0.05, 0.10, 0.15, 0.20],  # Interactor proportions
    effect_grid = [0.5, 0.75, 1.0, 1.25, 1.5],    # Effect size multipliers
    n_thresholds = 200                             # Threshold grid for ROC
)
```

**Key outputs:**

| Output | Description |
|---|---|
| `scenarios` | Vector of `ScenarioResult` (per-scenario metrics; see [Simulation & Calibration](@ref)) |
| `fdr_at_p95_range` | (min, max) FDR at posterior > 0.95 across all scenarios |
| `calibration_model` | Fitted `CalibrationModel` for Platt scaling (see [Simulation & Calibration](@ref)) |
| `calibration_cv` | Cross-validation metrics for calibration quality |

**Results in the report:** Simulation results appear in the interactive HTML report's Simulation tab, showing heatmaps of sensitivity and FDR across the parameter grid.

**Caching:** Simulation results are cached in `.bayesinteractomics_cache/` with parameter-based hash validation. Changing the latent class model parameters or simulation grid automatically triggers recomputation.

See also: [`run_simulation`](@ref), [`SimulationResult`](@ref), and [Simulation & Calibration](@ref).

## Platt Calibration

Platt calibration recalibrates posterior probabilities using logistic regression trained on simulation ground truth. Raw posterior probabilities from EM fitting may be systematically over- or under-confident; Platt scaling corrects this bias.

**How it works:**

The calibration applies a logistic transform in logit space:

```
calibrated = logistic(a * logit(raw) + b)
```

where parameters `a` and `b` are fitted by minimizing binary cross-entropy loss against the simulation ground truth labels. This 2-parameter model (Platt, 1999) is preferred over non-parametric alternatives like isotonic regression because it cannot overfit sparse mid-range calibration data.

**ECE safety guard:**

Calibration is only applied if it actually improves the Expected Calibration Error (ECE). The ECE measures the weighted average absolute difference between predicted probabilities and observed frequencies across bins:

```
ECE = sum(w_i * |predicted_i - observed_i|)
```

The safety guard works as follows:
1. Cross-validated ECE is computed using 5-fold stratified CV on simulation data
2. If the cross-validated ECE exceeds 0.10, calibration is **not** applied -- the raw posteriors are kept
3. Calibration quality is reported with a traffic-light badge:
   - **Green** (ECE < 0.02): Excellent calibration
   - **Yellow** (ECE 0.02-0.05): Acceptable calibration
   - **Red** (ECE > 0.05): Poor calibration, investigate model fit

**When calibration is applied:**

- Calibrated posteriors replace raw posteriors in the `posterior_calibrated` column of the results DataFrame
- Calibrated FDR values appear in the `fdr_calibrated` column
- Both raw and calibrated values are available for comparison
- The interactive HTML report's Calibration tab shows reliability diagrams comparing raw vs. calibrated probabilities

**Key types:**

| Type | Description |
|---|---|
| `CalibrationModel` | Platt scaling parameters (a, b) for posterior calibration |
| `FDRCalibrationModel` | Platt scaling parameters for FDR calibration |
| `CalibrationCVMetrics` | Cross-validation ECE metrics and reliability curves |

See also: [`SimulationResult`](@ref) and the [Simulation & Calibration](@ref) page for full descriptions of the calibration types.

## Quality Gates

Quality gates are automated statistical checks on the fitted mixture model that flag potential fitting problems before they propagate to results. They catch pathological EM fits such as component collapse, poor marginal fit, and contamination between components.

**Why it matters:** The 3-component latent class model has many parameters, and EM can converge to degenerate solutions where components overlap excessively or marginal distributions poorly describe the data. Quality gates provide an early warning system for these issues.

**How to use:**

Quality gates run automatically when using `run_analysis`. To invoke them directly:

```julia
quality_gates = run_quality_gates(bf_triplet, lc_result;
    ks_warn = 0.1,     # KS statistic warning threshold
    ks_fail = 0.15     # KS statistic failure threshold
)
```

**The quality gate checks:**

1. **KS test for marginal fit quality:** For each of the 3 marginals (enrichment, correlation, detection) in each of the 3 components (H0, agnostic, H1), a Kolmogorov-Smirnov test compares the weighted empirical distribution to the fitted parametric distribution. KS > 0.10 triggers a warning; KS > 0.15 triggers a failure. When Normal fit fails, automatic remediation with Student-t (LocationScale) is attempted.

2. **KL divergence contamination between components:** Measures how much non-interactors contaminate the H1 component by computing the KL divergence between pure H1 proteins (responsibility > 0.95) and the full H1 distribution. High KL divergence per stream (> 0.5) indicates the H1 component is capturing background proteins.

3. **Component separation metrics:** Checks that the three components are sufficiently separated in log-BF space. Overlapping components indicate the model cannot distinguish background from interactors.

4. **Within-class correlation checks:** Verifies that the conditional independence assumption holds approximately within each component.

**Interpreting results:**

The [`QualityGateResult`](@ref) struct contains a 3x3 matrix of gate outcomes (marginals x components), an overall status (`:pass`, `:warn`, or `:fail`), and remediation details:

```julia
quality_gates.overall_status       # :pass, :warn, or :fail
quality_gates.remediation_details  # Vector of remediation descriptions
quality_gates.cells                # 3x3 matrix of QualityGateCell results
```

Quality gate results are displayed in the interactive HTML report and printed to the console during `run_analysis`.

See also: [`run_quality_gates`](@ref), [`QualityGateResult`](@ref), [`KLContaminationResult`](@ref)

## Reporting Results

### Recommended Statistics to Report

For each high-confidence interaction:

- Posterior probability
- Combined Bayes factor
- Individual Bayes factors (detection, enrichment, correlation)
- Log2 fold-change (with credible interval)
- Number of independent protocols/experiments supporting

### Supplementary Materials

Provide for reproducibility:

- Posterior probability distribution plots
- Calibration curves (predicted vs observed interaction rates)
- Per-protocol results and agreement statistics
- Protocol specifications and column mappings

## API Reference

```@docs
BayesInteractomics.combined_BF_latent_class
LatentClassResult
BMAResult
bma_weights_plot
compute_em_responsibilities
DiscreteEmpirical
```

The simulation engine and Platt calibration internals (`run_simulation`, `SimulationResult`, plus the non-exported `ScenarioResult`, `CalibrationModel`, `FDRCalibrationModel`, `CalibrationCVMetrics` types) are documented on the [Simulation & Calibration](@ref) page. Quality gate types (`run_quality_gates`, `QualityGateResult`, `KLContaminationResult`, …) are documented on the [Diagnostics](@ref) page.
