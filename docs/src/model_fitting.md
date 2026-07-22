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
jzs_r_scale = 0.354                 # JZS Cauchy r-scale (JASP convention sqrt(2)/4); 0.0 = Normal slope prior
```

#### JZS Prior on the Slope

When `jzs_r_scale > 0`, the slope parameter follows a JZS-style Cauchy prior implemented as a Normal-Gamma scale mixture:

```
τ_g ~ Gamma(shape = 1/2, scale = 2 / r²)        # local shrinkage precision
α   ~ Normal(0, precision = τ_g)                # marginal distribution: Cauchy(0, r)
```

For multi-protocol data, the JZS prior sits on the hyper-mean slope `μ_α`; for single-protocol data, it sits on `α` directly. The default `jzs_r_scale = 0.354` follows the JASP convention (`sqrt(2)/4`); `0.0` falls back to a Normal prior. Setting `jzs_r_scale > 0` calls into the `*_jzs` model variants in `src/inference/models.jl`. The Bayes factor uses an analytical Cauchy survival function for the prior probability under H1.

Reference: Rouder, Speckman, Sun, Morey, Iverson (2009). *Bayesian t tests for accepting and rejecting the null hypothesis*. Psychonomic Bulletin & Review.

#### Per-Protein τ_base

The robust regression observation precision is anchored at a per-protein `τ_base` estimate via `estimate_per_protein_tau_base()` (defined in `src/inference/models.jl`). For proteins with `n_obs >= 5`, a protein-specific empirical precision is used; for proteins with fewer observations, the estimate shrinks toward the global precision via log-scale weighting. This avoids over-confident posteriors for sparsely observed proteins while preserving signal for well-observed ones.

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

## BayesResult Output

`BayesResult` is the per-protein container returned by HBM and regression fits. The current fields are:

```julia
struct BayesResult
    bfHBM::Union{Matrix{Float64}, Nothing}
    bfRegression::Union{Vector{Float64}, Nothing, Float64}
    HBM_stats::Dict{Symbol, ...}
    regression_stats::Union{Dict{Symbol, ...}, Nothing}
    hbm_result::Union{Nothing, HBMResult}
    regression_result::Union{Nothing, RegressionResult, RobustRegressionResult}
    protein_name::String
end
```

The result fields are `hbm_result` and `regression_result` (each carrying both `posterior` and `prior`). Access patterns:

```julia
result = ...                          # BayesResult
hbm_post = result.hbm_result.posterior         # InferenceResult (posterior)
hbm_prior = result.hbm_result.prior            # InferenceResult (prior)
reg_post = result.regression_result.posterior  # Posterior for regression
reg_prior = result.regression_result.prior     # Prior for regression

# Drill into specific variables:
log2fc_post = result.hbm_result.posterior.posteriors[:log2fc]
slope_prior = result.regression_result.prior.posteriors[:α]
```

The accessor helpers `getPosterior(result)` and `getPrior(result)` return tuples `(hbm, regression)` of the corresponding `InferenceResult`s.

## Evidence Combination

Three evidence-combination methods are dispatched through the `combination_method` CONFIG field:

- `:copula` -- Copula EM (three-component mixture, a single 3-D copula per component; BIC-selected from four families).
- `:latent_class` -- 3-component latent class (Student-t H0, anchored Agnostic at mu=0, sigmoid-gated H1) with BIC-selected H1 enrichment family.
- `:bma` (recommended default) -- Bayesian Model Averaging across the Copula and 3c-EM sub-models via LOO stacking weights (Yao et al. 2018).

For full details on each method, including diagnostics and disagreement analysis, see [`model_evaluation.md`](model_evaluation.md). The summary below covers HBM/regression-relevant aspects only.

### Copula (default sub-model)

Individual Bayes factors are combined using copulas:

1. **Convert** BFs to posterior probabilities (assuming uniform prior)
2. **Fit null copula** from permuted data (H₀ distribution)
3. **Fit mixture** via EM: a three-component mixture (H0 / anchored Agnostic / H1), each component a single 3-D copula over the enrichment / correlation / detection marginals
4. **Compute** joint Bayes factor from likelihood ratio

**Supported copula families** (per component, BIC-selected):

| Family | Tail dependence | Symmetry |
|---|---|---|
| Clayton | Lower tail | Asymmetric |
| Frank | No tail | Symmetric |
| Gumbel | Upper tail | Asymmetric |
| Gaussian | No tail | Symmetric |

**EM algorithm features**:
- Multiple random restarts (`em_n_restarts = 20`)
- SQUAREM acceleration for faster convergence
- Weighted H₁ re-fitting in M-step (`h1_refitting = true`)
- Informative Beta prior on π₁ (experiment-type specific)
- Burn-in period before H₁ updates

### 3c-EM / Latent Class (alternative sub-model)

```julia
combination_method = :latent_class
```

A 3-component mixture fitted on the log-BF scale: Student-t H0, anchored Agnostic (mu=0), sigmoid-gated H1 with BIC-selected enrichment family (Gamma / LogNormal / Weibull). Step-halving guarantees monotonic log-likelihood after burn-in. Does not require a pre-computed H₀ file.

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
BayesFactorHBM
BayesFactorRegression
HBMResult
HBMResultSingleProtocol
HBMResultMultipleProtocols
RobustRegressionResult
RobustRegressionResultSingleProtocol
RobustRegressionResultMultipleProtocols
NuOptimizationResult
enrichment
precompute_enrichment_prior
log2FC
compute_waic
compare_regression_models
optimize_nu
WAICResult
ModelComparisonResult
```

The Beta-Bernoulli detection model, copula fitting, and Bayes factor computation helpers (`betabernoulli`, `calculate_bayes_factor`, `probability_of_direction`, `pd_to_p_value`, `combined_BF`, `fit_copula`, `compare_copulas`, `posterior_probability_from_bayes_factor`) are internal helpers; access them via `BayesInteractomics.<name>` and consult the source for usage.
