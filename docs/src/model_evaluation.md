# Model Evaluation and Interpretation

## Overview

After running the analysis pipeline, model evaluation tools help assess result quality, validate predictions, and interpret findings. This module provides statistical metrics and ranking systems for identifying high-confidence protein-protein interactions.

## Understanding Bayesian Evidence

### Bayes Factor Interpretation

The Bayes factor (BF) quantifies the ratio of evidence for interaction (H₁) versus no interaction (H₀):

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

With uniform priors (P(H₀) = P(H₁) = 0.5):

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

### False Discovery Rate (FDR)

BayesInteractomics naturally handles multiple testing through the Bayesian framework:
- No correction needed (Bayesian approach)
- Posterior probabilities already account for uncertainty
- However, when making binary decisions, apply FDR control:

```julia
using StatsBase

# Rank by posterior probability
sorted = sort(results, :posterior_prob, rev=true)

# Compute FDR
discovered = cumsum(.!sorted.interaction)  # Count false discoveries
fdr = discovered ./ (1:nrow(sorted))        # Running FDR

# Select threshold where FDR < 0.05
threshold_idx = findfirst(fdr .< 0.05)
```

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

## Reporting Results

### Recommended Statistics to Report

For each high-confidence interaction:

- Posterior probability
- Combined Bayes factor
- Individual Bayes factors (detection, enrichment, correlation)
- Log₂ fold-change (with credible interval)
- Number of independent protocols/experiments supporting

### Supplementary Materials

Provide for reproducibility:

- Posterior probability distribution plots
- Calibration curves (predicted vs observed interaction rates)
- Per-protocol results and agreement statistics
- Protocol specifications and column mappings

## API Reference

```@autodocs
Modules = [BayesInteractomics]
Pages = ["model_evaluation.jl", "ranking.jl"]
```
