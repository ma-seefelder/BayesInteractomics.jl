# Model Fitting

## Overview

BayesInteractomics implements a comprehensive Bayesian framework for analyzing protein-protein interactions through three complementary statistical models. Each model extracts a different type of evidence from the experimental data.

## Model Components

### Beta-Bernoulli Detection Model

The Beta-Bernoulli model addresses a fundamental challenge in AP-MS experiments: distinguishing genuine interactions from background contamination. It models protein detection as a Bernoulli process with Beta-distributed probability.

**Key idea**: If a protein is a true interactor, it should be detected more consistently in samples than in controls.

- **Prior**: `Beta(α, β)` on detection probability
- **Likelihood**: Bernoulli observations (detected or not)
- **Posterior**: Updated Beta distribution
- **Bayes factor**: Compares detection rate in samples vs. controls

```julia
# Low-level usage (called automatically by analyse)
bf, posterior_prob, prior_prob = betabernoulli(data, protein_idx, n_controls, n_samples)
```

### Hierarchical Bayesian Model (HBM)

The HBM captures enrichment evidence through log₂ fold-change estimation:

- **Global level**: Overall effect across all protocols
- **Protocol level**: Protocol-specific effects (e.g., AP-MS vs. BioID)
- **Experiment level**: Per-experiment fold-change estimates
- **Inference**: Variational Bayes via RxInfer

**Key advantages**:
- Borrows strength across experiments through hierarchical structure
- Handles missing data naturally within the Bayesian framework
- Produces reliable estimates even with few replicates
- Generates Bayes factors for the enrichment hypothesis

### Bayesian Linear Regression

The regression model assesses dose-response relationships: does prey abundance correlate with bait abundance?

**Standard model**:
```
y = β₀ + β₁·x + ε,    where ε ~ Normal(0, σ²)
```

**Robust model** (Student-t likelihood):
```
y = β₀ + β₁·x + ε,    where ε ~ StudentT(ν, 0, σ²)
```

The robust variant uses a scale-mixture representation of the Student-t distribution, making it resistant to outliers. The degrees-of-freedom parameter ν can be optimized via WAIC.

```julia
# Control regression model in CONFIG:
regression_likelihood = :robust_t   # or :normal
student_t_nu = 5.0                  # degrees of freedom
optimize_nu = true                  # optimize ν via WAIC
```

## Model Comparison (WAIC)

When `run_model_comparison = true`, both Normal and robust regression models are fitted for all proteins and compared using the Widely Applicable Information Criterion (WAIC):

```
WAIC = -2 × (lppd - p_waic)
```

where:
- `lppd` = log pointwise predictive density (model fit)
- `p_waic` = effective number of parameters (model complexity)

Lower WAIC indicates better predictive performance.

```julia
# Results include:
# - WAICResult for each model (waic, lppd, p_waic, se)
# - ModelComparisonResult (delta_waic, preferred model)
# - Per-protein WAIC when n_proteins is small enough
```

### ν Optimization

When `optimize_nu = true`, BayesInteractomics searches for the optimal degrees-of-freedom:

- Search range: ν ∈ [3, 50]
- Method: Brent's method minimizing WAIC
- Result: Automatically sets `student_t_nu` to the optimal value
- Diagnostic: `nu_optimization_plot` shows WAIC vs. ν

## Evidence Combination

### Copula-Based Combination (default)

Individual Bayes factors are combined using copulas:

1. **Convert** BFs to posterior probabilities (assuming uniform prior)
2. **Fit null copula** from permuted data (H₀ distribution)
3. **Fit mixture** via EM: `π₀ · f₀(p) + π₁ · f₁(p)`
4. **Compute** joint Bayes factor from likelihood ratio

**Supported copula families**:
| Family | Tail dependence | Symmetry |
|---|---|---|
| Clayton | Lower tail | Asymmetric |
| Frank | No tail | Symmetric |
| Gumbel | Upper tail | Asymmetric |
| Gaussian | No tail | Symmetric |
| Joe | Upper tail | Asymmetric |

Best copula is selected automatically via BIC (or AIC).

**EM algorithm features**:
- Multiple random restarts (`n_restarts = 20`)
- SQUAREM acceleration for faster convergence
- Weighted H₁ re-fitting in M-step (`h1_refitting = true`)
- Informative Beta prior on π₁ (experiment-type specific)
- Burn-in period before H₁ updates

### Latent Class Combination (alternative)

```julia
combination_method = :latent_class
```

Uses variational message passing (VMP) to fit a 2-class mixture model. Does not require a pre-computed H₀ file, making it useful when permutation-based null distributions are unavailable.

## Priors

### EM Prior on π₁

The mixing proportion π₁ (fraction of true interactors) uses an informative Beta prior:

| Preset | α | β | Expected π₁ |
|---|---|---|---|
| `:APMS` | 20 | 180 | ~10% |
| `:BioID` | 30 | 120 | ~20% |
| `:TurboID` | 40 | 110 | ~25% |
| `:default` | 25 | 175 | ~12.5% |
| `:permissive` | 50 | 100 | ~33% |
| `:stringent` | 10 | 190 | ~5% |
| `:empirical_bayes` | — | — | Data-driven |

Custom priors: `em_prior = (α = 15.0, β = 85.0)`

### HBM Priors

HBM priors are estimated empirically from the data:
- `τ₀()`: Fitted Gamma distribution on precision, estimated from control variability
- `μ₀()`: Returns `(median_of_means, max_variance)`, estimated from overall abundance

## Practical Interpretation

### Bayes Factor Scale

| Bayes Factor | log₁₀(BF) | Interpretation |
|---|---|---|
| > 100 | > 2 | Decisive evidence |
| 30–100 | 1.5–2 | Very strong |
| 10–30 | 1–1.5 | Strong |
| 3–10 | 0.5–1 | Moderate |
| 1–3 | 0–0.5 | Weak |
| < 1 | < 0 | Favors H₀ |

Combined Bayes factors from multiple models are more reliable than any individual model, as they integrate complementary evidence sources.

## API Reference

```@docs
betabernoulli
BayesFactorHBM
BayesFactorRegression
probability_of_direction
pd_to_p_value
compute_waic
compare_regression_models
optimize_nu
combined_BF
fit_copula
compare_copulas
posterior_probability_from_bayes_factor
```
