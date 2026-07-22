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
- Detection evidence ($BF_{\text{detection}}$)
- Enrichment evidence ($BF_{\text{enrichment}}$)
- Correlation evidence ($BF_{\text{correlation}}$)

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
- **H0 population** ($H_0$): Non-interacting proteins (null hypothesis true)
- **H1 population** ($H_1$): Genuine interactors (alternative hypothesis true)

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

BayesInteractomics fits exactly four single-parameter copula families:
- **Clayton**: Models lower tail dependence (joint low values)
- **Frank**: Symmetric dependence
- **Gumbel**: Models upper tail dependence (joint high values)
- **Gaussian**: Linear correlation structure

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

- **H0 initialization**: Use proteins with all Bayes factors < 1 (strong evidence against interaction)
- **H1 initialization**: Use proteins with all Bayes factors > threshold (e.g., > 3)
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

## Log-Bayes Factor Scale

### Motivation

Before combining the three Bayes factors, all are transformed to the natural-log scale:

```math
\ell_i = \log(BF_i), \quad i \in \{\text{enrichment}, \text{correlation}, \text{detection}\}
```

This transformation provides three key advantages:

1. **Numerical stability**: Raw Bayes factors span many orders of magnitude ($10^{-6}$ to $10^{6}$). On the log scale, these become bounded values in $[-14, 14]$.
2. **Additive interpretation**: Under independence, $\log(BF_1 \cdot BF_2) = \log(BF_1) + \log(BF_2)$, allowing additive reasoning about evidence.
3. **Better-behaved marginals**: The EM algorithm fits mixture distributions to the log-BF triplets. Log-transformed Bayes factors are approximately symmetric and unimodal under $H_0$, enabling standard parametric families (Normal, Student-t) to capture the null distribution.

### Winsorization

To prevent any single extreme protein from dominating the EM fit, log-BFs are winsorized:

```math
\tilde{\ell}_i = \text{clamp}(\ell_i, -\log(BF_{\max}), \log(BF_{\max}))
```

where $BF_{\max} = 10^6$ by default. The EM operates on winsorized triplets $(\tilde{\ell}_e, \tilde{\ell}_c, \tilde{\ell}_d)$, while the original (non-winsorized) log-BFs are used for final posterior computation.

## Three-Component Mixture Model

The evidence combination uses a three-component mixture model on log-BF triplets:

```math
f(\mathbf{x}) = \pi_0 \cdot f_{H_0}(\mathbf{x}) + \pi_a \cdot f_{\text{agnostic}}(\mathbf{x}) + \pi_1 \cdot f_{H_1}(\mathbf{x})
```

where $\mathbf{x} = (\ell_e, \ell_c, \ell_d)$ is the log-BF triplet and $\pi_0 + \pi_a + \pi_1 = 1$. The three components capture three qualitatively different protein populations.

### $H_0$ Component (Non-Interactors)

Non-interacting proteins should have Bayes factors near 1 (log-BFs near 0), with occasional extreme negative values from random fluctuations. The enrichment marginal uses a **Student-t distribution** to capture these heavy tails:

```math
f_{H_0,e}(x) = \text{LocationScale}(\mu_0, \sigma_0, t_\nu)
```

where $\nu$ is selected from $\{3, 5, 7, 10\}$ by minimizing the Bayesian Information Criterion (BIC) on the enrichment marginal. The heavy tails of the Student-t distribution capture extreme negative log-BFs (e.g., $\ell_e \approx -45$) without requiring a separate outlier component. A BIC margin of 2 is required for selecting Student-t over Normal.

The correlation and detection marginals use Normal distributions:

```math
f_{H_0,c}(x) = \mathcal{N}(\mu_{0c}, \sigma_{0c}^2), \quad f_{H_0,d}(x) = \mathcal{N}(\mu_{0d}, \sigma_{0d}^2)
```

Under conditional independence given component membership, the joint $H_0$ density is:

```math
f_{H_0}(\mathbf{x}) = f_{H_0,e}(\ell_e) \cdot f_{H_0,c}(\ell_c) \cdot f_{H_0,d}(\ell_d)
```

### Agnostic Component (Uninformative Proteins)

The agnostic component captures proteins with Bayes factors near 1 across all evidence types -- proteins for which the data are genuinely uninformative:

```math
f_{\text{ag},e}(x) = \mathcal{N}(0, \sigma_{ae}^2), \quad f_{\text{ag},c}(x) = \mathcal{N}(\mu_{ac}, \sigma_{ac}^2), \quad f_{\text{ag},d}(x) = \mathcal{N}(\mu_{ad}, \sigma_{ad}^2)
```

The enrichment mean is **anchored at zero** ($\mu_{ae} = 0$, not a free parameter), ensuring this component does not drift toward either $H_0$ or $H_1$ during EM fitting. Without this anchor, $H_0$ and agnostic components can become redundant (identical distributions), wasting a mixture component.

### Redundancy Detection

After EM convergence, a KL divergence check detects whether $H_0$ and agnostic components have collapsed to nearly identical distributions:

```math
D_{\text{KL}}(f_{H_0} \| f_{\text{ag}}) < 0.1 \implies \text{merge components}
```

If merging is triggered, the agnostic weight is set to $\pi_a = 0$ and its mass is absorbed by $H_0$ via weighted averaging, preserving the 3-component structure for backward compatibility.

### $H_1$ Component (Interactors)

The $H_1$ enrichment marginal must enforce that genuine interactors have **substantial enrichment evidence**. A smooth sigmoid transition replaces a hard cutoff at the Jeffreys threshold:

```math
w(x) = \frac{1}{1 + e^{-k(x - \text{JEFFREYS\_SHIFT})}}
```

where JEFFREYS\_SHIFT $= \ln(\sqrt{10}) \approx 1.151$ is Jeffreys' "substantial evidence" threshold and $k = 5.0$ is the sigmoid steepness.

The H1 enrichment log-density combines a shifted positive distribution with the sigmoid gate:

```math
\log f_{H_1,e}(x) = \log g(x - \text{JEFFREYS\_SHIFT}) + \log w(x)
```

where $g$ is a positive-support distribution (e.g., Gamma, LogNormal, or Weibull) selected by BIC. At the shift point ($x = \text{JEFFREYS\_SHIFT}$), the sigmoid contributes $\log(0.5) \approx -0.69$ nats, providing a smooth rather than discontinuous penalty.

The correlation and detection marginals for $H_1$ are shifted Normals:

```math
f_{H_1,c}(x) = \mathcal{N}(\mu_{1c}, \sigma_{1c}^2), \quad f_{H_1,d}(x) = \mathcal{N}(\mu_{1d}, \sigma_{1d}^2)
```

with $\mu_{1c} > \mu_{0c}$ and $\mu_{1d} > \mu_{0d}$ enforced by label-ordering constraints.

### BIC-Selected $H_1$ Enrichment Family

The positive-support distribution $g$ for the $H_1$ enrichment marginal is selected at EM iteration 5 from three candidate families:

| Family | Support | Density shape |
|--------|---------|---------------|
| Gamma | $(0, \infty)$ | Flexible skewness |
| LogNormal | $(0, \infty)$ | Heavy right tail |
| Weibull | $(0, \infty)$ | Flexible hazard rate |

The family with the lowest BIC on the shifted enrichment values ($x - \text{JEFFREYS\_SHIFT}$ for $H_1$-assigned proteins) is retained for the remainder of the EM.

### EM Algorithm Details

The EM algorithm includes several convergence guarantees:

**Step-halving guard**: After the M-step applies parameter constraints (sigma floors/caps, mean constraints, label ordering), the log-likelihood is re-evaluated. If the constrained parameters decrease the log-likelihood, the update is reverted:

```math
\text{if } \mathcal{L}(\Theta^{(t+1)}_{\text{constrained}}) < \mathcal{L}(\Theta^{(t)}) - \epsilon \implies \Theta^{(t+1)} \leftarrow \Theta^{(t)}
```

where $\epsilon = 10^{-6}$ is a numerical noise threshold. This guarantees monotonic log-likelihood after the burn-in period.

**Constraint ordering**: The M-step applies constraints in a canonical order to ensure deterministic behavior:
1. Sigma floors and caps (data-dependent IQR-based bounds)
2. Mean constraints (agnostic $\mu_e = 0$, label ordering)
3. Label ordering ($\mu_{H_0,e} < \mu_{\text{ag},e} < \mu_{H_1,e}$)
4. Single log-likelihood check with step-halving revert

**Multiple restarts**: The EM is run from 20+ random initializations (quantile-based, k-means, and random) to avoid local optima. The restart with the highest converged log-likelihood is selected.

**SQUAREM acceleration**: The Squared Iterative Methods algorithm (Varadhan and Roland, 2008) accelerates convergence by using quasi-Newton steps without explicit Hessian computation, achieving 2-10x speedup over standard EM.

**Dirichlet prior**: Mixture weights are regularized with a Dirichlet prior $\text{Dir}(5.0, 2.0, 1.0)$, which biases toward $H_0$ (as expected biologically) and prevents component collapse.

## Single-Copula-per-Component Structure

### Sklar Construction

The Copula sub-model represents each mixture component with a **single 3-D copula** over the enrichment, correlation, and detection dimensions -- it does **not** decompose the joint into a cascade of bivariate pieces. By Sklar's theorem, the joint density of a component's log-BF triplet factors into its marginals and one trivariate copula density:

```math
f_k(\ell_e, \ell_c, \ell_d) = c_k\big(G_{ke}(\ell_e), G_{kc}(\ell_c), G_{kd}(\ell_d)\big) \cdot f_{ke}(\ell_e) \cdot f_{kc}(\ell_c) \cdot f_{kd}(\ell_d)
```

where $c_k$ is the single 3-D copula density for component $k$, the $G_{kj}$ are the component marginal CDFs, and the $f_{kj}$ their densities.

### Three-Component Mixture

The Copula sub-model uses the same **three-component** structure introduced above ($H_0$ / anchored-agnostic / $H_1$), with one single 3-D copula per component:

```math
f(\ell_e, \ell_c, \ell_d) = \pi_0 \, f_0(\ell_e, \ell_c, \ell_d) + \pi_a \, f_a(\ell_e, \ell_c, \ell_d) + \pi_1 \, f_1(\ell_e, \ell_c, \ell_d)
```

Each $f_k$ is the Sklar product above. As with the latent-class model, the anchored-agnostic component keeps its enrichment mean fixed at zero so it cannot drift toward either $H_0$ or $H_1$ during fitting.

### Four-Family BIC Selection

For each component the trivariate copula family is selected by minimizing BIC over exactly four single-parameter families:

| Family | Dependence captured |
|--------|---------------------|
| Clayton | Lower tail (joint low values) |
| Frank | Symmetric |
| Gumbel | Upper tail (joint high values) |
| Gaussian | Linear correlation structure |

Modeling each component with a single 3-D copula -- rather than assuming full conditional independence -- lets the combination retain residual dependence between the three evidence streams while remaining numerically stable and interpretable.

## Latent Class Model

### Formulation

The latent class model provides an alternative to the copula approach by modeling the joint density of log-BF triplets directly as a multivariate mixture:

```math
f(\mathbf{x}) = \pi_0 \cdot \prod_{j=1}^{3} f_{0j}(x_j) + \pi_a \cdot \prod_{j=1}^{3} f_{aj}(x_j) + \pi_1 \cdot \prod_{j=1}^{3} f_{1j}(x_j)
```

where $x_j$ are the three log-BF dimensions (enrichment, correlation, detection) and $f_{kj}$ are the marginal densities for component $k$ and dimension $j$.

### Difference from Copula Approach

The latent class model assumes **conditional independence** of the three evidence streams given the latent class. This is a stronger assumption than the copula approach (which models residual dependence via copula functions), but it is computationally simpler and more numerically stable.

### EM Algorithm for Latent Class

**E-step**: Compute posterior membership probabilities:

```math
\gamma_{ik} = \frac{\pi_k \prod_{j=1}^{3} f_{kj}(x_{ij})}{\sum_{k'} \pi_{k'} \prod_{j=1}^{3} f_{k'j}(x_{ij})}
```

**M-step**: Update parameters using weighted sufficient statistics:

```math
\pi_k = \frac{\sum_i \gamma_{ik} + \alpha_k - 1}{N + \sum_k (\alpha_k - 1)}, \quad \mu_{kj} = \frac{\sum_i \gamma_{ik} x_{ij}}{\sum_i \gamma_{ik}}, \quad \sigma_{kj}^2 = \frac{\sum_i \gamma_{ik} (x_{ij} - \mu_{kj})^2}{\sum_i \gamma_{ik}}
```

where $\alpha_k$ are Dirichlet prior pseudocounts.

### Adaptive Component Count

After convergence, a KL divergence test checks whether the $H_0$ and agnostic components are effectively identical:

```math
D_{\text{KL}}(f_{H_0} \| f_{\text{ag}}) = \sum_j \left[ \log \frac{\sigma_{aj}}{\sigma_{0j}} + \frac{\sigma_{0j}^2 + (\mu_{0j} - \mu_{aj})^2}{2\sigma_{aj}^2} - \frac{1}{2} \right]
```

If $D_{\text{KL}} < 0.1$, the components are merged via weighted averaging.

### Post-Hoc Bayes Factor Constraint

To prevent pathological combined Bayes factors where all individual evidence streams favor $H_0$ but the combined BF favors $H_1$ (due to component assignment artifacts), a post-hoc constraint is applied:

```math
\text{if } BF_e < 1 \text{ and } BF_c < 1 \implies BF_{\text{combined}} \leq \max(BF_e, BF_c, BF_d)
```

This ensures that the combined evidence cannot exceed the strongest individual evidence when both enrichment and correlation disfavor interaction.

## Bayesian Model Averaging

### Motivation

The copula and latent class models make different structural assumptions about evidence dependence. Rather than choosing one, BayesInteractomics combines them via **Bayesian Model Averaging (BMA)** using LOO stacking weights (Yao et al., 2018).

### LOO Stacking Weights

The stacking weights $w_k$ for $K$ models are found by solving:

```math
\hat{w} = \arg\max_{w \in \mathcal{S}_K} \sum_{i=1}^{N} \log \sum_{k=1}^{K} w_k \cdot p_k^{(-i)}(x_i)
```

where:
- $p_k^{(-i)}(x_i)$ is the leave-one-out predictive density of model $k$ at observation $i$
- $\mathcal{S}_K = \{w : w_k \geq 0, \sum_k w_k = 1\}$ is the probability simplex

A 5% weight floor is applied: $w_k \geq 0.05$ for all models, ensuring no model is entirely discarded.

### Models Averaged

Two models are combined:
1. **Copula model**: single 3-D copula per component with BIC-selected family (models residual dependence)
2. **3c-EM model**: Latent class with conditional independence assumption (computationally stable)

### Final Posterior

The BMA posterior probability for protein $i$ is:

```math
P_{\text{BMA}}(H_1 | \mathbf{x}_i) = \sum_{k=1}^{K} w_k \cdot P_k(H_1 | \mathbf{x}_i)
```

where $P_k(H_1 | \mathbf{x}_i)$ is the posterior from model $k$.

## High-Level Summaries of Statistical Components

The subsections below give a one-paragraph mathematical sketch and citation for each statistical component added since v1.0. Full derivations are intentionally kept brief — see the cited papers for proofs.

### Student-t H0 (heavy-tailed null)

The H0 enrichment marginal in the 3-component latent class model uses a Student-t distribution with location `mu_0`, scale `sigma_0`, and degrees of freedom `nu` selected by BIC over `{3, 5, 7, 10}`. The heavy tails absorb extreme negative log-BFs (e.g., `ell_e ≈ -45`) without requiring a separate outlier component. A BIC margin of 2 is required to prefer Student-t over Normal, preventing over-fitting on data sets where the null is genuinely Gaussian.

Reference: Geweke, J. (1993). *Bayesian treatment of the independent Student-t linear model*. Journal of Applied Econometrics, 8, S19-S40.

### Sigmoid-gated H1

The H1 enrichment marginal uses a smooth sigmoid gate `w(x) = 1 / (1 + exp(-k * (x - JEFFREYS_SHIFT)))` with `JEFFREYS_SHIFT = ln(sqrt(10)) ≈ 1.151` (Jeffreys' "substantial evidence" threshold) and steepness `k = 5.0`. Below the threshold, log-density is smoothly suppressed rather than hard-zeroed; this preserves EM monotonicity (no log-likelihood decreases from the gate) and ensures component separation between the Agnostic and H1 components.

Reference: Jeffreys, H. (1961). *Theory of Probability*, 3rd ed. Oxford University Press, Appendix B (Bayes factor scale).

### BIC-Selected H1 Marginal

The positive-support distribution `g` underlying the H1 enrichment density is selected at iteration 5 of the EM from `{Gamma, LogNormal, Weibull}` by minimizing BIC on the shifted enrichment values (`x - JEFFREYS_SHIFT`) for H1-assigned proteins. The selected family is then locked for the remainder of the EM, ensuring the same marginal is used throughout convergence and across all sensitivity-grid restarts.

Reference: Schwarz, G. (1978). *Estimating the dimension of a model*. Annals of Statistics, 6(2), 461-464.

### Storey Monotone Step-Down BFDR

`bfdr()` (`src/core/utils.jl`) computes the Bayesian FDR from posterior probabilities and applies Storey's monotone step-down correction: when proteins are sorted by decreasing posterior probability, the BFDR sequence is enforced to be non-increasing via a backward cumulative-min pass. This eliminates the "wiggles" near the decision boundary that otherwise appear when consecutive proteins have similar PEP values.

Reference: Storey, J. D. (2002). *A direct approach to false discovery rates*. Journal of the Royal Statistical Society B, 64(3), 479-498. See also Storey & Tibshirani (2003), PNAS, 100(16), 9440-9445.

### Empirical Bayes Dirichlet (Minka fixed-point)

When `lc_alpha_prior = :auto`, the Dirichlet concentration vector for the 3c-EM mixing weights is estimated from the data via Minka's fixed-point iteration. The update solves `psi(alpha_k) - psi(sum(alpha)) = mean_log_pi_k` for each component, where `psi` is the digamma function and `mean_log_pi_k` is the empirical mean of `log(pi_k)` across multi-restart EM solutions. Convergence is typically reached in fewer than 30 iterations and replaces the v1.0 fixed `[5, 2, 1]` Dirichlet.

Reference: Minka, T. P. (2000). *Estimating a Dirichlet distribution*. Microsoft Research Technical Report.

### BIC-Weighted Prior Grid Marginalization

After EB Dirichlet estimation, a 9-point constant-strength simplex grid is constructed around the EB centre. The 3c-EM is fit at each grid point, and posterior probabilities are averaged across grid points using BIC weights `w_g ∝ exp(-BIC_g / 2)`. This eliminates residual single-prior sensitivity and produces posteriors that are robust to within-family Dirichlet specification while still being data-driven.

Reference: Hoeting, J. A., Madigan, D., Raftery, A. E., & Volinsky, C. T. (1999). *Bayesian model averaging: A tutorial*. Statistical Science, 14(4), 382-401.

### JZS Prior on the Regression Slope

The JZS prior is a Cauchy prior on the regression slope, implemented as a Normal-Gamma scale mixture: `tau_g ~ Gamma(1/2, 2/r^2)` and `alpha ~ Normal(0, precision = tau_g)`. The marginal distribution of `alpha` is `Cauchy(0, r)`. Default `jzs_r_scale = 0.354` follows the JASP convention `sqrt(2)/4`. For multi-protocol data, the JZS prior sits on the hyper-mean slope `mu_alpha`; for single-protocol data, it sits on `alpha` directly. The Bayes factor uses an analytical Cauchy survival function for the prior probability under H1.

Reference: Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D., & Iverson, G. (2009). *Bayesian t tests for accepting and rejecting the null hypothesis*. Psychonomic Bulletin & Review, 16(2), 225-237.

### Platt Scaling Calibration

Platt scaling is a 2-parameter logistic calibration: `P_calibrated = sigma(a * logit(P_raw) + b)`. Parameters `(a, b)` are fitted by minimising binary cross-entropy on simulation ground-truth labels. An ECE (Expected Calibration Error) safety guard rejects calibration when cross-validated ECE does not improve on the raw posteriors, falling back to the uncalibrated values. This prevents calibration from making things worse on data sets where the raw posteriors are already well-calibrated.

Reference: Platt, J. (1999). *Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods*. Advances in Large Margin Classifiers, 10(3), 61-74.

### LOO Stacking BMA

Bayesian Model Averaging in BayesInteractomics uses LOO stacking weights computed by maximising the leave-one-out predictive log score across the simplex of weight vectors. A 5% weight floor (`w_k >= 0.05`) prevents winner-take-all degeneracy. Pareto-smoothed importance sampling (PSIS-LOO) provides per-protein Pareto-k diagnostics indicating which observations have unstable LOO estimates.

Reference: Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). *Using stacking to average Bayesian predictive distributions*. Bayesian Analysis, 13(3), 917-1007. See also Vehtari, Gelman, & Gabry (2017), *Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC*, Statistics and Computing, 27(5), 1413-1432.

### C2Qscore Docking

C2Qscore is a 4-metric scoring function for AlphaFold 3 docking output, replacing the VoroIF dependency that proved difficult to install reliably across platforms. The four metrics combine ipTM, pDockQ-style features, and the AF3 confidence model. Logistic calibration on the CASP15/CAPRI baseline yields AUC-ROC = 0.929 vs pDockQ2's 0.825, with no clamping required.

Reference: Olechnowicz, J., Aderinwale, T., Joshi, A., Christoffer, C., & Kihara, D. — CASP15/CAPRI baseline (Kihara Lab); see also Bryant, P. & Elofsson, A. (2022). *Improved prediction of protein-protein interactions using AlphaFold2*. Nature Communications, 13, 1265.

### pDockQ — Burke et al. 2023

pDockQ is a logistic calibration of AlphaFold ipTM/pTM and contact-area features that produces a probability-like score for the dockability of a candidate complex. BayesInteractomics uses pDockQ as a Tier 2 docking BF source (logistic calibration; Burke et al. 2023) when full-data JSONs are available; the Tier 1 ipTM step-function is used otherwise.

Reference: Burke, D. F., Bryant, P., Barrio-Hernandez, I., et al. (2023). *Towards a structurally resolved human protein interaction network*. Nature Structural & Molecular Biology, 30, 216-225.

## Discrete Empirical Detection Distribution

### Motivation

The Beta-Bernoulli Bayes factors for detection take on a **finite set of discrete values** determined by the number of possible detection patterns (combinations of sample and control counts). For example, with 3 samples and 3 controls, there are only $(3+1) \times (3+1) = 16$ possible detection count combinations, yielding at most 16 distinct BF values.

### DiscreteEmpirical Distribution

Instead of approximating these discrete BFs with a continuous Normal distribution (which was statistically incorrect), BayesInteractomics uses a `DiscreteEmpirical` distribution:

```math
f_d(x) = \sum_{k=1}^{K} p_k \cdot \delta(x - v_k)
```

where $v_1, \ldots, v_K$ are the unique log-BF values and $p_k$ are their empirical frequencies.

### Jittering for Copula Fitting

Copula fitting requires continuous uniform marginals. Discrete values are converted to pseudo-uniform observations via the randomized CDF (Denuit and Lambert, 2005):

```math
F^*(x) = F(x^-) + U \cdot P(X = x), \quad U \sim \text{Uniform}[0, 1]
```

where $F(x^-)$ is the left limit of the CDF. This preserves the rank structure while enabling copula estimation.

## Platt Scaling

### Calibration Problem

Raw posterior probabilities from the EM may not be perfectly calibrated -- a protein assigned $P = 0.8$ may not have an 80% empirical chance of being a true interactor. **Platt scaling** provides post-hoc calibration.

### Method

A logistic regression is fitted on ground-truth labels from parametric simulation:

```math
P_{\text{calibrated}} = \sigma(a \cdot P_{\text{raw}} + b) = \frac{1}{1 + e^{-(a \cdot P_{\text{raw}} + b)}}
```

where $a$ and $b$ are fitted by maximum likelihood on simulation data with known true labels.

### ECE Safety Guard

Platt scaling is only applied if it improves the **Expected Calibration Error (ECE)**:

```math
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
```

where $B_m$ are calibration bins, $\text{acc}(B_m)$ is the empirical accuracy (fraction of true positives), and $\text{conf}(B_m)$ is the mean predicted probability. If $\text{ECE}_{\text{calibrated}} \geq \text{ECE}_{\text{raw}}$, the raw posteriors are retained.

### Design Choice

Platt scaling (2-parameter logistic) was chosen over isotonic regression (PAVA) because the latter can overfit when mid-range calibration data is sparse -- a common scenario in AP-MS where most proteins are clear non-interactors or strong interactors.

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
- Denuit, M., & Lambert, P. (2005). Constraints on concordance measures in bivariate discrete data. *Journal of Multivariate Analysis*, 93(1), 59-79.

### Variational Inference
- Blei, D. M., et al. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859-877.
- Bagaev, D., & de Vries, B. (2023). RxInfer: A Julia package for reactive message-passing-based Bayesian inference. *Journal of Open Source Software*, 8(84), 5161.

### EM Acceleration
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. *Scandinavian Journal of Statistics*, 35(2), 335-353.

### Model Averaging and Calibration
- Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using stacking to average Bayesian predictive distributions. *Bayesian Analysis*, 13(3), 917-1007.
- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 10(3), 61-74.

### Proteomics Applications
- Choi, H., et al. (2011). SAINT: Probabilistic scoring of affinity purification-mass spectrometry data. *Nature Methods*, 8(1), 70-73.
- Mellacheruvu, D., et al. (2013). The CRAPome: A contaminant repository for affinity purification-mass spectrometry data. *Nature Methods*, 10(8), 730-736.
