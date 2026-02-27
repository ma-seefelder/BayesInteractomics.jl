# Mathematical Background

This document provides a detailed mathematical exposition of the statistical framework implemented in BayesInteractomics. The package integrates three complementary Bayesian models to identify genuine protein-protein interactions from mass spectrometry data.

## Overview of the Bayesian Framework

### The Multiple Evidence Problem

Identifying true protein interactions requires distinguishing genuine interactors from:
- Non-specific binders (proteins that bind regardless of bait)
- Contaminants (proteins present in controls)
- False positives due to experimental noise

BayesInteractomics addresses this by evaluating three independent but complementary questions for each candidate protein:

1. **Detection**: Is the protein consistently detected in samples versus controls?
2. **Enrichment**: Is the protein quantitatively enriched in samples?
3. **Correlation**: Does the protein's abundance correlate with bait levels?

Each question is answered using a Bayesian model that produces a **Bayes factor** quantifying the evidence for interaction. These Bayes factors are then combined using copula-based mixture models.

### Bayes Factors

A Bayes factor compares the evidence for two hypotheses:

```math
BF_{10} = \frac{P(D | H_1)}{P(D | H_0)} = \frac{\text{Evidence for } H_1}{\text{Evidence for } H_0}
```

where:
- $H_1$: Hypothesis of genuine interaction
- $H_0$: Null hypothesis (no interaction)
- $D$: Observed data

**Interpretation**:
- $BF_{10} > 1$: Data favor interaction
- $BF_{10} = 1$: Data equally support both hypotheses
- $BF_{10} < 1$: Data favor null hypothesis
- $BF_{10} > 10$: Strong evidence for interaction
- $BF_{10} > 100$: Very strong evidence for interaction

Bayes factors provide a continuous measure of evidence that naturally accounts for uncertainty and doesn't require arbitrary significance thresholds.

## Model 1: Beta-Bernoulli Model (Detection Probability)

### Biological Motivation

Genuine interactors should be consistently detected in samples but rarely (or never) in negative controls. The Beta-Bernoulli model evaluates whether the **detection rate** (proportion of replicates where protein is detected) is higher in samples than controls.

### Model Specification

For a protein, let:
- $n_s$ = number of sample replicates
- $n_c$ = number of control replicates
- $k_s$ = number of samples where protein is detected
- $k_c$ = number of controls where protein is detected

We model detection as Bernoulli trials:

```math
k_s \sim \text{Binomial}(n_s, \theta_s)
```
```math
k_c \sim \text{Binomial}(n_c, \theta_c)
```

where $\theta_s$ and $\theta_c$ are the true detection rates in samples and controls, respectively.

### Prior Distribution

We use weakly informative Beta priors for both detection rates:

```math
\theta_s \sim \text{Beta}(3, 3)
```
```math
\theta_c \sim \text{Beta}(3, 3)
```

The Beta(3,3) prior is centered at 0.5 with moderate uncertainty, expressing weak prior belief that detection rates are neither very low nor very high.

### Posterior Distribution

Due to conjugacy of the Beta-Binomial model, posteriors are analytical:

```math
\theta_s | D \sim \text{Beta}(3 + k_s, 3 + (n_s - k_s))
```
```math
\theta_c | D \sim \text{Beta}(3 + k_c, 3 + (n_c - k_c))
```

### Bayes Factor Computation

We test the one-sided hypothesis:
- $H_1$: $\theta_s > \theta_c$ (detection rate higher in samples)
- $H_0$: $\theta_s \leq \theta_c$ (detection rate not higher)

The posterior probability is estimated via Monte Carlo:

```math
p = P(\theta_s > \theta_c | D) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\theta_s^{(i)} > \theta_c^{(i)}]
```

where $\theta_s^{(i)}$ and $\theta_c^{(i)}$ are samples from the posterior distributions.

The Bayes factor is computed from posterior and prior odds:

```math
BF_{10} = \frac{p / (1-p)}{0.5 / 0.5} = \frac{p}{1-p}
```

The prior odds are 1:1 (uniform prior on $H_0$ vs $H_1$).

### Implementation Notes

- Monte Carlo estimation uses $N = 10^7$ samples for high precision
- Detection is defined as non-missing observation in the data matrix
- Missing data are naturally handled (not counted as detection)

## Model 2: Hierarchical Bayesian Model (Enrichment)

### Biological Motivation

Genuine interactors should show **quantitative enrichment** in samples compared to controls. The Hierarchical Bayesian Model (HBM) estimates the log2 fold change (log2FC) while accounting for:
- Protocol-level heterogeneity (different experimental methods)
- Experiment-level batch effects
- Missing data across replicates

### Model Specification

Let $y_{pej}$ denote the log-transformed intensity for:
- Protocol $p \in \{1, \ldots, P\}$
- Experiment $e \in \{1, \ldots, E_p\}$ within protocol $p$
- Sample $j$ (either control or bait)

#### Likelihood

```math
y_{pej} | \mu_{pe}, \sigma^2_{pe} \sim \mathcal{N}(\mu_{pe}, \sigma^2_{pe})
```

where $\mu_{pe}$ is the mean intensity and $\sigma^2_{pe}$ is the variance for experiment $e$ in protocol $p$.

#### Hierarchical Structure

**Protocol-level parameters** (shared across experiments within a protocol):

```math
\mu_{pe}^{\text{control}} | \mu_p^0, \tau_p^2 \sim \mathcal{N}(\mu_p^0, \tau_p^2)
```
```math
\mu_{pe}^{\text{sample}} | \mu_p^0, \log_2 FC_p, \tau_p^2 \sim \mathcal{N}(\mu_p^0 + \log_2 FC_p, \tau_p^2)
```

where:
- $\mu_p^0$: Baseline intensity for protocol $p$
- $\log_2 FC_p$: Log2 fold change for protocol $p$ (parameter of interest)
- $\tau_p^2$: Between-experiment variance within protocol $p$

**Experiment-level variance**:

```math
\sigma^2_{pe} \sim \text{InverseGamma}(\alpha_{\sigma}, \beta_{\sigma})
```

#### Priors

**Log2 fold change** (weakly informative):

```math
\log_2 FC_p \sim \mathcal{N}(0, 10)
```

**Baseline intensity**:

```math
\mu_p^0 = \frac{1}{E_p} \sum_{e=1}^{E_p} \bar{y}_{pe}^{\text{control}}
```

where $\bar{y}_{pe}^{\text{control}}$ is the empirical mean of control samples.

**Between-experiment variance**:

```math
\tau_p^2 = \max\left(\sigma_p^2 - \bar{\sigma}^2_{pe}, \epsilon\right)
```

where:
- $\sigma_p^2$ is the empirical variance of experiment means
- $\bar{\sigma}^2_{pe}$ is the average within-experiment variance
- $\epsilon = 10^{-6}$ prevents numerical issues

**Within-experiment variance** (conjugate prior):

```math
\alpha_{\sigma} = 2, \quad \beta_{\sigma} = 0.5
```

### Inference

Posterior inference is performed using **variational Bayes** via RxInfer.jl, which approximates the posterior through optimization rather than sampling. This provides:
- Fast convergence (seconds per protein)
- Automatic convergence diagnostics
- Full posterior distributions for all parameters

### Bayes Factor Computation

The Bayes factor for enrichment tests:
- $H_1$: $\log_2 FC_p > 0$ (enrichment in samples)
- $H_0$: $\log_2 FC_p \leq 0$ (no enrichment)

From the posterior distribution $q(\log_2 FC_p | D)$:

```math
BF_{10} = \frac{P(\log_2 FC_p > 0 | D)}{P(\log_2 FC_p \leq 0 | D)} = \frac{p}{1-p}
```

where $p = \int_0^{\infty} q(\log_2 FC_p | D) \, d(\log_2 FC_p)$.

### Multiple Protocols

For datasets with multiple protocols, we obtain protocol-specific Bayes factors $BF_1, BF_2, \ldots, BF_P$. The overall enrichment Bayes factor is:

```math
BF_{\text{enrichment}} = \prod_{p=1}^{P} BF_p
```

This assumes conditional independence of protocols given the hypothesis.

## Model 3: Bayesian Linear Regression (Dose-Response Correlation)

### Biological Motivation

Genuine interactors often show a **dose-response relationship**: their abundance correlates with the bait protein's abundance. If bait expression varies across samples (e.g., due to transfection efficiency), true interactors should track these variations, while contaminants should not.

### Model Specification

Let:
- $y_i$: Candidate protein intensity in sample $i$
- $x_i$: Bait protein (reference) intensity in sample $i$
- $i \in \{1, \ldots, N\}$: Sample indices

#### Likelihood

```math
y_i | \beta_0, \beta_1, \sigma^2 \sim \mathcal{N}(\beta_0 + \beta_1 x_i, \sigma^2)
```

where:
- $\beta_0$: Intercept
- $\beta_1$: Slope (correlation strength - parameter of interest)
- $\sigma^2$: Residual variance

#### Priors

**Slope** (weakly informative):

```math
\beta_1 \sim \mathcal{N}(0, 10)
```

**Intercept** (weakly informative):

```math
\beta_0 \sim \mathcal{N}(0, 100)
```

**Residual variance** (conjugate):

```math
\sigma^2 \sim \text{InverseGamma}(2, 0.5)
```

### Hierarchical Extension for Multiple Protocols

When multiple protocols are present, we use protocol-specific slopes $\beta_{1p}$ with a hierarchical structure:

```math
\beta_{1p} | \mu_{\beta}, \tau_{\beta}^2 \sim \mathcal{N}(\mu_{\beta}, \tau_{\beta}^2)
```

where:
- $\mu_{\beta}$: Population mean slope (overall correlation)
- $\tau_{\beta}^2$: Between-protocol variance in slopes

**Hyperpriors**:

```math
\mu_{\beta} \sim \mathcal{N}(0, 10)
```
```math
\tau_{\beta}^2 \sim \text{Gamma}(1, 1)
```

### Inference

Posterior inference uses variational Bayes via RxInfer.jl, yielding posterior distributions for all parameters.

### Bayes Factor Computation

The Bayes factor tests:
- $H_1$: $\beta_1 > 0$ (positive correlation with bait)
- $H_0$: $\beta_1 \leq 0$ (no positive correlation)

```math
BF_{10} = \frac{P(\beta_1 > 0 | D)}{P(\beta_1 \leq 0 | D)} = \frac{p}{1-p}
```

where $p$ is computed from the posterior distribution.

For multiple protocols:

```math
BF_{\text{correlation}} = \prod_{p=1}^{P} BF_p
```

## Evidence Combination via Copulas

### The Combination Problem

We now have three Bayes factors for each protein:
- $BF_{\text{detection}}$: Detection evidence
- $BF_{\text{enrichment}}$: Enrichment evidence
- $BF_{\text{correlation}}$: Correlation evidence

These are **not independent**: for example, enriched proteins are more likely to be consistently detected. Simple multiplication (independence assumption) would be incorrect.

**Solution**: Model the joint distribution of Bayes factors using **copulas**, which flexibly capture dependencies while allowing arbitrary marginals.

### Copula Theory

A copula $C$ is a multivariate distribution on $[0,1]^d$ with uniform marginals. By Sklar's theorem, any multivariate distribution $F$ can be decomposed as:

```math
F(x_1, \ldots, x_d) = C(F_1(x_1), \ldots, F_d(x_d))
```

where $F_i$ are the marginal distributions and $C$ is the copula capturing dependence.

### Mixture Copula Model

The distribution of Bayes factors arises from a mixture of two populations:
- **$H_0$ population**: Non-interacting proteins (null hypothesis true)
- **$H_1$ population**: Genuine interactors (alternative hypothesis true)

Let $\mathbf{BF} = (BF_{\text{detection}}, BF_{\text{enrichment}}, BF_{\text{correlation}})$ be the triplet of Bayes factors.

#### Mixture Model

```math
F(\mathbf{BF}) = \pi_0 \cdot F_{H_0}(\mathbf{BF}) + \pi_1 \cdot F_{H_1}(\mathbf{BF})
```

where:
- $\pi_0$: Proportion of non-interactors
- $\pi_1 = 1 - \pi_0$: Proportion of true interactors
- $F_{H_0}$: Joint distribution under $H_0$ (modeled by copula $C_0$)
- $F_{H_1}$: Joint distribution under $H_1$ (modeled by copula $C_1$)

#### Copula Specification

For each component $k \in \{0, 1\}$:

```math
F_{H_k}(\mathbf{BF}) = C_k\left(G_1(BF_{\text{detection}}), G_2(BF_{\text{enrichment}}), G_3(BF_{\text{correlation}})\right)
```

where:
- $C_k$: Copula for component $k$ (e.g., Clayton, Gumbel, Frank, Gaussian)
- $G_i$: Marginal cumulative distribution for evidence type $i$

BayesInteractomics supports multiple copula families:
- **Clayton**: Models lower tail dependence (joint low values)
- **Gumbel**: Models upper tail dependence (joint high values)
- **Frank**: Symmetric dependence
- **Gaussian**: Linear correlation structure
- **Joe**: Asymmetric upper tail dependence

### Expectation-Maximization (EM) Algorithm

The mixture model parameters $\Theta = \{\pi_0, \pi_1, C_0, C_1, G_1, G_2, G_3\}$ are estimated using the EM algorithm.

#### E-Step

Compute posterior probability that protein $i$ belongs to $H_1$:

```math
\gamma_i^{(t)} = \frac{\pi_1^{(t)} \cdot f_{H_1}(\mathbf{BF}_i | \Theta^{(t)})}{\pi_0^{(t)} \cdot f_{H_0}(\mathbf{BF}_i | \Theta^{(t)}) + \pi_1^{(t)} \cdot f_{H_1}(\mathbf{BF}_i | \Theta^{(t)})}
```

where $f_{H_k}$ is the density corresponding to $F_{H_k}$.

#### M-Step

Update parameters to maximize expected complete-data log-likelihood:

**Mixture weights**:

```math
\pi_1^{(t+1)} = \frac{1}{N} \sum_{i=1}^{N} \gamma_i^{(t)}
```

**Copula parameters**: Fit $C_0$ and $C_1$ using weighted data:
- $C_0$ fit to proteins with weights $(1 - \gamma_i^{(t)})$
- $C_1$ fit to proteins with weights $\gamma_i^{(t)}$

**Marginals**: Fit $G_1, G_2, G_3$ using kernel density estimation or empirical CDFs.

#### Initialization

- **$H_0$ initialization**: Use proteins with all Bayes factors < 1 (strong evidence against interaction)
- **$H_1$ initialization**: Use proteins with all Bayes factors > threshold (e.g., > 3)
- **Mixture weight**: $\pi_1^{(0)} = 0.1$ (conservative initial estimate)

#### Convergence

Iterate E-step and M-step until:

```math
\frac{|\pi_1^{(t+1)} - \pi_1^{(t)}|}{|\pi_1^{(t)}|} < \epsilon
```

Typically $\epsilon = 10^{-4}$ and convergence occurs in 10-50 iterations.

### Combined Bayes Factor

After EM convergence, the combined Bayes factor for protein $i$ is:

```math
BF_{\text{combined},i} = \frac{f_{H_1}(\mathbf{BF}_i)}{f_{H_0}(\mathbf{BF}_i)}
```

This is the likelihood ratio using the fitted copula densities.

### Posterior Probability

Assuming uniform prior $P(H_1) = 0.5$, the posterior probability of interaction is:

```math
P(H_1 | \mathbf{BF}_i) = \frac{BF_{\text{combined},i}}{1 + BF_{\text{combined},i}}
```

Alternatively, using the EM-estimated mixture proportion:

```math
P(H_1 | \mathbf{BF}_i) = \gamma_i
```

## Summary Statistics

For each protein, BayesInteractomics reports:

### Bayes Factors
- Individual BFs from each model
- Combined BF from copula mixture
- Log BF for extremely large values

### Posterior Summaries for log2FC
- **Mean**: $\mathbb{E}[\log_2 FC | D]$
- **Median**: $\text{median}(\log_2 FC | D)$
- **SD**: Standard deviation (uncertainty)
- **Credible intervals**: 95% highest density intervals
- **Probability of direction (pd)**: $P(\log_2 FC > 0 | D)$
- **ROPE percentage**: $P(|\log_2 FC| < \epsilon | D)$ where $\epsilon$ is a practical equivalence threshold

### Convergence Diagnostics
- **ESS (Effective Sample Size)**: Measures quality of posterior samples (should be > 400)
- **Rhat**: Gelman-Rubin convergence diagnostic (should be < 1.01)

## Computational Implementation

### Parallelization

BayesInteractomics exploits multi-core parallelism:
- Proteins are analyzed independently in parallel using Julia's multi-threading
- Each thread writes results to a separate cache file to avoid contention
- Results are merged after all proteins complete

### Variational Inference

RxInfer.jl uses **variational message passing** for fast Bayesian inference:
- Factorized approximation: $q(\theta) = \prod_i q_i(\theta_i)$
- Iterative message passing updates until convergence
- Automatically handles missing data through marginalization

### Numerical Stability

- Log-space computation for extreme Bayes factors
- Regularization of variance estimates (lower bound $\epsilon = 10^{-6}$)
- Robust initialization for EM algorithm
- Convergence checks with maximum iteration limits

## References

### Bayesian Inference
- Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed. Chapman & Hall/CRC.
- Kruschke, J. K. (2014). *Doing Bayesian Data Analysis*, 2nd ed. Academic Press.

### Bayes Factors
- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *Journal of the American Statistical Association*, 90(430), 773-795.
- Rouder, J. N., et al. (2009). Bayesian t tests for accepting and rejecting the null hypothesis. *Psychonomic Bulletin & Review*, 16(2), 225-237.

### Hierarchical Models
- Gelman, A., & Hill, J. (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.

### Copula Theory
- Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer.
- Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman & Hall/CRC.

### Variational Inference
- Blei, D. M., et al. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859-877.
- Bagaev, D., & de Vries, B. (2023). RxInfer: A Julia package for reactive message-passing-based Bayesian inference. *Journal of Open Source Software*, 8(84), 5161.

### Proteomics Applications
- Choi, H., et al. (2011). SAINT: Probabilistic scoring of affinity purification-mass spectrometry data. *Nature Methods*, 8(1), 70-73.
- Mellacheruvu, D., et al. (2013). The CRAPome: A contaminant repository for affinity purification-mass spectrometry data. *Nature Methods*, 10(8), 730-736.
