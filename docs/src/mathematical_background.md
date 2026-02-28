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

where $H_1$ is the hypothesis of genuine interaction, $H_0$ is the null hypothesis (no interaction), and $D$ is the observed data.

**Interpretation**:

| Condition | Interpretation |
|-----------|---------------|
| $BF_{10} > 1$ | Data favor interaction |
| $BF_{10} = 1$ | Data equally support both hypotheses |
| $BF_{10} < 1$ | Data favor null hypothesis |
| $BF_{10} > 10$ | Strong evidence for interaction |
| $BF_{10} > 100$ | Very strong evidence for interaction |

Bayes factors provide a continuous measure of evidence that naturally accounts for uncertainty and doesn't require arbitrary significance thresholds.

## Model 1: Beta-Bernoulli Model (Detection Probability)

### Biological Motivation

Genuine interactors should be consistently detected in samples but rarely (or never) in negative controls. The Beta-Bernoulli model evaluates whether the **detection rate** (proportion of replicates where protein is detected) is higher in samples than controls.

### Model Specification

For a protein, let $n_s$ be the number of sample replicates, $n_c$ the number of control replicates, $k_s$ the number of samples where the protein is detected, and $k_c$ the number of controls where the protein is detected.

We model detection as Bernoulli trials:

```math
\begin{aligned}
k_s &\sim \text{Binomial}(n_s, \theta_s) \\
k_c &\sim \text{Binomial}(n_c, \theta_c)
\end{aligned}
```

where $\theta_s$ and $\theta_c$ are the true detection rates in samples and controls, respectively.

### Prior Distribution

We use weakly informative Beta priors for both detection rates:

```math
\begin{aligned}
\theta_s &\sim \text{Beta}(3, 3) \\
\theta_c &\sim \text{Beta}(3, 3)
\end{aligned}
```

The Beta(3,3) prior is centered at 0.5 with moderate uncertainty, expressing weak prior belief that detection rates are neither very low nor very high.

### Posterior Distribution

Due to conjugacy of the Beta-Binomial model, posteriors are analytical:

```math
\begin{aligned}
\theta_s | D &\sim \text{Beta}(3 + k_s, 3 + (n_s - k_s)) \\
\theta_c | D &\sim \text{Beta}(3 + k_c, 3 + (n_c - k_c))
\end{aligned}
```

### Bayes Factor Computation

We test the one-sided hypothesis $H_1\colon \theta_s > \theta_c$ (detection rate higher in samples) against $H_0\colon \theta_s \leq \theta_c$ (detection rate not higher).

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

Let $y_{pej}$ denote the log-transformed intensity for protocol $p \in \{1, \ldots, P\}$, experiment $e \in \{1, \ldots, E_p\}$ within protocol $p$, and sample $j$ (either control or bait).

#### Likelihood

```math
y_{pej} | \mu_{pe}, \sigma^2_{pe} \sim \mathcal{N}(\mu_{pe}, \sigma^2_{pe})
```

where $\mu_{pe}$ is the mean intensity and $\sigma^2_{pe}$ is the variance for experiment $e$ in protocol $p$.

#### Hierarchical Structure

**Protocol-level parameters** (shared across experiments within a protocol):

```math
\begin{aligned}
\mu_{pe}^{\text{control}} | \mu_p^0, \tau_p^2 &\sim \mathcal{N}(\mu_p^0, \tau_p^2) \\
\mu_{pe}^{\text{sample}} | \mu_p^0, \log_2 FC_p, \tau_p^2 &\sim \mathcal{N}(\mu_p^0 + \log_2 FC_p, \tau_p^2)
\end{aligned}
```

where $\mu_p^0$ is the baseline intensity for protocol $p$, $\log_2 FC_p$ is the log2 fold change for protocol $p$ (parameter of interest), and $\tau_p^2$ is the between-experiment variance within protocol $p$.

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

where $\sigma_p^2$ is the empirical variance of experiment means, $\bar{\sigma}^2_{pe}$ is the average within-experiment variance, and $\epsilon = 10^{-6}$ prevents numerical issues.

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

The Bayes factor for enrichment tests $H_1\colon \log_2 FC_p > 0$ (enrichment in samples) against $H_0\colon \log_2 FC_p \leq 0$ (no enrichment).

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

### Model Specification (Normal Likelihood)

Let $y_i$ be the candidate protein intensity in sample $i$, $x_i$ the bait protein (reference) intensity in sample $i$, and $i \in \{1, \ldots, N\}$ the sample indices.

#### Likelihood

```math
y_i | \beta_0, \beta_1, \sigma^2 \sim \mathcal{N}(\beta_0 + \beta_1 x_i, \sigma^2)
```

where $\beta_0$ is the intercept, $\beta_1$ is the slope (correlation strength — parameter of interest), and $\sigma^2$ is the residual variance.

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

where $\mu_{\beta}$ is the population mean slope (overall correlation) and $\tau_{\beta}^2$ is the between-protocol variance in slopes.

**Hyperpriors**:

```math
\begin{aligned}
\mu_{\beta} &\sim \mathcal{N}(0, 10) \\
\tau_{\beta}^2 &\sim \text{Gamma}(1, 1)
\end{aligned}
```

### Inference

Posterior inference uses variational Bayes via RxInfer.jl, yielding posterior distributions for all parameters.

### Bayes Factor Computation

The Bayes factor tests $H_1\colon \beta_1 > 0$ (positive correlation with bait) against $H_0\colon \beta_1 \leq 0$ (no positive correlation).

```math
BF_{10} = \frac{P(\beta_1 > 0 | D)}{P(\beta_1 \leq 0 | D)} = \frac{p}{1-p}
```

where $p$ is computed from the posterior distribution.

For multiple protocols:

```math
BF_{\text{correlation}} = \prod_{p=1}^{P} BF_p
```

### Robust Extension: Student-t via Scale Mixture

The standard Normal likelihood assumes homogeneous residual variance, making the regression sensitive to outliers. Proteomics data frequently contain aberrant intensity values caused by misidentifications, interference, or carry-over. A **robust regression** replaces the Normal likelihood with a heavier-tailed Student-t distribution, implemented through a Normal–Gamma scale mixture that is fully compatible with variational message passing.

#### Scale-Mixture Representation

Each observation receives its own precision $\tau_i$, drawn from a Gamma distribution:

```math
\begin{aligned}
\tau_i &\sim \text{Gamma}\!\left(\frac{\nu}{2},\; \text{scale} = \frac{\tau_{\text{base}}}{\nu/2}\right), \quad \mathbb{E}[\tau_i] = \tau_{\text{base}} \\
y_i \mid \mu_i, \tau_i &\sim \mathcal{N}(\mu_i,\; \text{precision} = \tau_i)
\end{aligned}
```

Marginalizing over $\tau_i$ recovers a Student-t distribution:

```math
y_i \mid \mu_i \;\sim\; \text{Student-}t\!\left(\nu,\; \mu_i,\; \tau_{\text{base}}\right)
```

The degrees-of-freedom parameter $\nu$ controls tail heaviness: smaller $\nu$ yields heavier tails and greater outlier robustness; as $\nu \to \infty$ the model reduces to Normal regression.

#### Hierarchical Prior Structure

The prior structure is identical to the Normal model. For the multi-protocol case:

| Parameter | Prior |
|-----------|-------|
| $\mu_\alpha$ (hyper-mean intercept) | $\mathcal{N}(0,\; (0.3/1.96)^2)$ |
| $\mu_\beta$ (hyper-mean slope) | $\mathcal{N}(\hat{\mu}_0,\; \sigma_0^2)$ |
| $\sigma_\alpha$ (hyper-precision intercept) | $\text{Gamma}(6.304,\; \text{scale}=7.932)$ |
| $\sigma_\beta$ (hyper-precision slope) | $\text{Gamma}(10,\; \text{scale}=0.3)$ |
| $\alpha_k$ (per-protocol intercept) | $\mathcal{N}(\mu_\alpha,\; \text{precision}=\sigma_\alpha)$ |
| $\beta_k$ (per-protocol slope) | $\mathcal{N}(\mu_\beta,\; \text{precision}=\sigma_\beta)$ |

The empirical Bayes hyperparameters $\hat{\mu}_0$ and $\sigma_0^2$ are estimated from OLS on the pooled data, identical to the Normal model.

#### Empirical Bayes for $\tau_{\text{base}}$

The baseline precision is set to the inverse residual variance from an OLS fit:

```math
\tau_{\text{base}} = \frac{1}{\text{Var}(\hat{\varepsilon}_{\text{OLS}})}
```

This anchors the Student-t scale to the data, ensuring that $\mathbb{E}[\tau_i] = \tau_{\text{base}}$ matches the observed noise level.

#### Default Configuration

The degrees-of-freedom parameter defaults to $\nu = 5$, which provides moderately heavy tails. When model comparison is enabled (see next section), $\nu$ is optimized over $[3, 50]$ by minimizing WAIC.

## Model Comparison via WAIC

The Widely Applicable Information Criterion (WAIC; Watanabe, 2010) provides a principled method for comparing the Normal and robust (Student-t) regression models. Unlike AIC or BIC, WAIC is fully Bayesian and uses the entire posterior distribution rather than a point estimate.

### WAIC Definition

Given $S$ posterior draws $\theta^{(1)}, \ldots, \theta^{(S)}$ and $n$ observations, WAIC is defined as:

```math
\text{WAIC} = -2\left(\text{lppd} - p_{\text{waic}}\right)
```

where the **log pointwise predictive density** and the **effective number of parameters** are:

```math
\begin{aligned}
\text{lppd} &= \sum_{i=1}^{n} \log\!\left(\frac{1}{S} \sum_{s=1}^{S} p(y_i \mid \theta^{(s)})\right) \\
p_{\text{waic}} &= \sum_{i=1}^{n} \text{Var}_{s}\!\left(\log p(y_i \mid \theta^{(s)})\right)
\end{aligned}
```

Lower WAIC indicates better out-of-sample predictive performance. The implementation uses $S = 1000$ posterior draws from the VMP approximate posteriors.

### Normal vs Robust Regression Comparison

To compare the two regression models, we compute WAIC for each and take the difference:

```math
\Delta\text{WAIC} = \text{WAIC}_{\text{normal}} - \text{WAIC}_{\text{robust}}
```

A positive $\Delta\text{WAIC}$ favors the robust model (lower WAIC). The standard error of the difference is estimated from pointwise WAIC differences:

```math
\text{SE}_\Delta = \sqrt{n \cdot \text{Var}(w_i^{\text{normal}} - w_i^{\text{robust}})}
```

where $w_i$ denotes the pointwise WAIC contribution of observation $i$. When $|\Delta\text{WAIC}| > 2 \cdot \text{SE}_\Delta$, the difference is considered meaningful (Vehtari et al., 2017).

### Degrees-of-Freedom Optimization

When the robust model is selected, the degrees-of-freedom parameter $\nu$ can be optimized to minimize WAIC. BayesInteractomics uses **Brent's method** to search over $\nu \in [3, 50]$ with a tolerance of 0.5 (finer precision is not meaningful given WAIC uncertainty). A fixed random seed ensures a deterministic, smooth objective surface for the optimizer.

The optimization procedure:
1. Compute WAIC for the Normal model once (baseline).
2. For each candidate $\nu$, fit the robust model across all proteins and compute WAIC.
3. Return the $\nu$ that minimizes WAIC, along with the $\Delta\text{WAIC}$ relative to the Normal baseline.

## Evidence Combination

### The Combination Problem

We now have three Bayes factors for each protein: $BF_{\text{detection}}$ (detection evidence), $BF_{\text{enrichment}}$ (enrichment evidence), and $BF_{\text{correlation}}$ (correlation evidence).

These are **not independent**: for example, enriched proteins are more likely to be consistently detected. Simple multiplication (independence assumption) would be incorrect.

**Solution**: Model the joint distribution of Bayes factors using **copulas**, which flexibly capture dependencies while allowing arbitrary marginals. BayesInteractomics also provides a **latent class model** and **Bayesian model averaging** across both combination methods.

### Copula-Based Combination

#### Copula Theory

A copula $C$ is a multivariate distribution on $[0,1]^d$ with uniform marginals. By Sklar's theorem, any multivariate distribution $F$ can be decomposed as:

```math
F(x_1, \ldots, x_d) = C(F_1(x_1), \ldots, F_d(x_d))
```

where $F_i$ are the marginal distributions and $C$ is the copula capturing dependence.

#### Mixture Copula Model

The distribution of Bayes factors arises from a mixture of two populations:
- **$H_0$ population**: Non-interacting proteins (null hypothesis true)
- **$H_1$ population**: Genuine interactors (alternative hypothesis true)

Let $\mathbf{BF} = (BF_{\text{detection}}, BF_{\text{enrichment}}, BF_{\text{correlation}})$ be the triplet of Bayes factors.

**Mixture model**:

```math
F(\mathbf{BF}) = \pi_0 \cdot F_{H_0}(\mathbf{BF}) + \pi_1 \cdot F_{H_1}(\mathbf{BF})
```

where $\pi_0$ is the proportion of non-interactors, $\pi_1 = 1 - \pi_0$ is the proportion of true interactors, $F_{H_0}$ is the joint distribution under $H_0$ (modeled by copula $C_0$), and $F_{H_1}$ is the joint distribution under $H_1$ (modeled by copula $C_1$).

**Copula specification** — for each component $k \in \{0, 1\}$:

```math
F_{H_k}(\mathbf{BF}) = C_k\left(G_1(BF_{\text{detection}}), G_2(BF_{\text{enrichment}}), G_3(BF_{\text{correlation}})\right)
```

where $C_k$ is the copula for component $k$ (e.g., Clayton, Gumbel, Frank, Gaussian) and $G_i$ is the marginal cumulative distribution for evidence type $i$.

BayesInteractomics supports multiple copula families:
- **Clayton**: Models lower tail dependence (joint low values)
- **Gumbel**: Models upper tail dependence (joint high values)
- **Frank**: Symmetric dependence
- **Gaussian**: Linear correlation structure
- **Joe**: Asymmetric upper tail dependence

#### EM Algorithm

The mixture model parameters $\Theta = \{\pi_0, \pi_1, C_0, C_1, G_1, G_2, G_3\}$ are estimated using the EM algorithm.

**E-Step** — compute posterior probability that protein $i$ belongs to $H_1$:

```math
\gamma_i^{(t)} = \frac{\pi_1^{(t)} \cdot f_{H_1}(\mathbf{BF}_i | \Theta^{(t)})}{\pi_0^{(t)} \cdot f_{H_0}(\mathbf{BF}_i | \Theta^{(t)}) + \pi_1^{(t)} \cdot f_{H_1}(\mathbf{BF}_i | \Theta^{(t)})}
```

where $f_{H_k}$ is the density corresponding to $F_{H_k}$.

**M-Step** — update parameters to maximize expected complete-data log-likelihood:

Mixture weights:

```math
\pi_1^{(t+1)} = \frac{1}{N} \sum_{i=1}^{N} \gamma_i^{(t)}
```

Copula parameters: fit $C_0$ and $C_1$ using weighted data — $C_0$ is fit to proteins with weights $(1 - \gamma_i^{(t)})$ and $C_1$ is fit to proteins with weights $\gamma_i^{(t)}$.

Marginals: fit $G_1, G_2, G_3$ using kernel density estimation or empirical CDFs.

**Initialization**:

- The **$H_0$ component** is initialized using proteins with all Bayes factors < 1 (strong evidence against interaction)
- The **$H_1$ component** is initialized using proteins with all Bayes factors > threshold (e.g., > 3)
- The **mixture weight** is set to $\pi_1^{(0)} = 0.1$ (conservative initial estimate)

**Convergence** — iterate E-step and M-step until:

```math
\frac{|\pi_1^{(t+1)} - \pi_1^{(t)}|}{|\pi_1^{(t)}|} < \epsilon
```

Typically $\epsilon = 10^{-4}$ and convergence occurs in 10–50 iterations.

#### Combined Bayes Factor

After EM convergence, the combined Bayes factor for protein $i$ is:

```math
BF_{\text{combined},i} = \frac{f_{H_1}(\mathbf{BF}_i)}{f_{H_0}(\mathbf{BF}_i)}
```

This is the likelihood ratio using the fitted copula densities.

#### Posterior Probability

Assuming uniform prior $P(H_1) = 0.5$, the posterior probability of interaction is:

```math
P(H_1 | \mathbf{BF}_i) = \frac{BF_{\text{combined},i}}{1 + BF_{\text{combined},i}}
```

Alternatively, using the EM-estimated mixture proportion:

```math
P(H_1 | \mathbf{BF}_i) = \gamma_i
```

### Latent Class Model

The latent class model is a simpler alternative to copula-based combination that models the joint distribution of log-Bayes factors directly as a Gaussian mixture. It assumes **conditional independence** of the three evidence arms given the latent interaction status, which trades flexibility in dependence modeling for computational simplicity and robustness.

#### Model Specification

Each protein has a latent class $z_i \in \{0, 1\}$ indicating Background or Interaction status. The three-dimensional score vector $\mathbf{s}_i = (\log BF_{\text{enrich},i},\; \log BF_{\text{corr},i},\; \log BF_{\text{detect},i})$ is modeled as:

```math
\begin{aligned}
z_i &\sim \text{Categorical}(\pi_0, \pi_1), \quad \pi_0 + \pi_1 = 1 \\
p(\mathbf{s}_i \mid z_i = k) &= \prod_{d=1}^{3} \mathcal{N}(s_{id} \mid \mu_{dk},\; \sigma^2_{dk})
\end{aligned}
```

The conditional independence assumption means that each evidence dimension contributes independently to the class assignment, given the latent state. This is a 2-component, 3-dimensional Gaussian mixture with diagonal covariance.

#### Data Preprocessing

Before EM fitting, Bayes factors are transformed and optionally winsorized:

1. **Log-transform**: $s_{id} = \log(BF_{id})$. Log-Bayes factors are approximately Normal by the CLT, with $\log(BF) = 0$ representing no evidence.
2. **Winsorization** (optional, default on): extreme values are clamped to the 1st and 99th percentiles. This protects the EM parameter estimates from extreme outliers. The original (non-winsorized) log-BFs are retained for posterior computation.

#### EM Algorithm

The EM algorithm iterates between responsibility computation and parameter updates:

**E-Step** — for each protein $i$, compute the posterior probability of belonging to the interaction class:

```math
\gamma_i = \frac{\pi_1 \prod_{d} \mathcal{N}(s_{id} \mid \mu_{d1}, \sigma^2_{d1})}{\pi_0 \prod_{d} \mathcal{N}(s_{id} \mid \mu_{d0}, \sigma^2_{d0}) + \pi_1 \prod_{d} \mathcal{N}(s_{id} \mid \mu_{d1}, \sigma^2_{d1})}
```

**M-Step** — update parameters using the responsibilities:

```math
\begin{aligned}
\pi_k^{\text{new}} &= \frac{N_k + \alpha_k - 1}{N + \sum_j \alpha_j - 2}, \quad N_k = \sum_i \gamma_{ik} \\
\mu_{dk}^{\text{new}} &= \frac{\sum_i \gamma_{ik} \, s_{id}}{N_k} \\
\sigma_{dk}^{\text{new}} &= \max\!\left(\sqrt{\frac{\sum_i \gamma_{ik} (s_{id} - \mu_{dk})^2}{N_k}},\; \sigma_{\text{floor}}\right)
\end{aligned}
```

The mixing weights receive a Dirichlet prior with $\boldsymbol{\alpha} = (10, 1)$, encoding a prior expectation that most proteins are non-interactors.

**Label ordering constraint**: after each M-step, if $\mu_{\text{enrich},1} < \mu_{\text{enrich},0}$, all parameters and responsibilities between the two components are swapped. This ensures the interaction class always has a higher mean enrichment score.

**Convergence**: the algorithm terminates when the relative change in log-likelihood falls below $10^{-6}$, or after a maximum of 100 iterations.

#### Posterior Computation and Monotonicity Correction

Final posteriors are computed on the **original (non-winsorized)** log-BF values using the EM-fitted parameters. A monotonicity correction prevents proteins with extremely strong evidence from being penalized:

For each dimension $d$, if $s_{id} > \mu_{d1}$ (the protein's score exceeds the interaction-class mean), the per-dimension log-likelihood ratio is floored at zero:

```math
\text{LLR}_d = \begin{cases}
\log \frac{\mathcal{N}(s_{id} \mid \mu_{d1}, \sigma^2_{d1})}{\mathcal{N}(s_{id} \mid \mu_{d0}, \sigma^2_{d0})} & \text{if } s_{id} \leq \mu_{d1} \\[6pt]
\max\!\left(\log \frac{\mathcal{N}(s_{id} \mid \mu_{d1}, \sigma^2_{d1})}{\mathcal{N}(s_{id} \mid \mu_{d0}, \sigma^2_{d0})},\; 0\right) & \text{if } s_{id} > \mu_{d1}
\end{cases}
```

The posterior probability is then:

```math
P(z_i = 1 \mid \mathbf{s}_i) = \frac{1}{1 + \exp\!\left(-\log\frac{\pi_1}{\pi_0} - \sum_d \text{LLR}_d\right)}
```

### Bayesian Model Averaging (BMA)

When both copula and latent class combination methods are available, Bayesian Model Averaging (BMA) provides a principled way to combine their posterior probabilities, weighting each method by how well it fits the data.

#### BIC Computation and Model Weights

The Bayesian Information Criterion approximates the log marginal likelihood for each combination model $m$:

```math
\text{BIC}_m = -2 \log \hat{L}_m + k_m \log n
```

where $\hat{L}_m$ is the maximized likelihood, $k_m$ is the number of parameters, and $n$ is the number of proteins.

The parameter counts are:

| Model | Parameters | Total $k$ |
|-------|-----------|-----------|
| Copula | copula dependence params + 6 (H0 marginals) + 6 (H1 marginals) + 1 (mixing weight) | $k_{\text{cop}} + 13$ |
| Latent class | 2 classes $\times$ 3 dims $\times$ 2 ($\mu, \sigma$) + 1 (mixing weight) | 13 |

Model weights are computed from BIC differences:

```math
w_m = \frac{\exp(-\tfrac{1}{2}\,\Delta\text{BIC}_m)}{\sum_j \exp(-\tfrac{1}{2}\,\Delta\text{BIC}_j)}, \quad \Delta\text{BIC}_m = \text{BIC}_m - \min_j \text{BIC}_j
```

#### Averaged Posterior

The model-averaged posterior probability for each protein is a weighted combination:

```math
P_{\text{avg}}(H_1 \mid D_i) = w_{\text{copula}} \cdot P_{\text{copula}}(H_1 \mid D_i) + w_{\text{lc}} \cdot P_{\text{lc}}(H_1 \mid D_i)
```

The averaged Bayes factor is derived from the averaged posterior:

```math
BF_{\text{avg},i} = \frac{P_{\text{avg}} / (1 - P_{\text{avg}})}{\pi_1 / \pi_0}
```

where $\pi_1 / \pi_0$ is the prior odds from the copula EM fit.

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

### Probability of Direction (pd)

The probability of direction (also called Maximum Probability of Effect) quantifies the certainty about the sign of an effect. For a posterior distribution of the parameter $\theta$:

**From posterior draws** ($S$ samples):

```math
\text{pd} = \max\!\left(\frac{1}{S}\sum_{s=1}^{S} \mathbb{1}[\theta^{(s)} > 0],\;\; 1 - \frac{1}{S}\sum_{s=1}^{S} \mathbb{1}[\theta^{(s)} > 0]\right)
```

**From an analytical posterior** (e.g., Normal or mixture):

```math
\text{pd} = \max\!\left(\Phi(0),\; 1 - \Phi(0)\right)
```

where $\Phi(0)$ is the CDF evaluated at zero.

The probability of direction is always in $[0.5, 1.0]$. A value of 0.5 indicates complete uncertainty about the direction, while 1.0 indicates certainty. The associated direction label is "+" if the effect is more likely positive, "$-$" if negative, or "~" if exactly 0.5.

**Conversion to p-values** (for compatibility with frequentist frameworks):

| Conversion | Formula |
|------------|---------|
| Two-sided p-value | $p = 2(1 - \text{pd})$ |
| One-sided p-value | $p = 1 - \text{pd}$ |

### Bayesian FDR q-values

To control the false discovery rate in a Bayesian framework, BayesInteractomics computes q-values from the posterior probabilities. The local false discovery rate for protein $i$ is:

```math
\text{lfdr}_i = 1 - P(H_1 \mid D_i)
```

Proteins are sorted by descending posterior probability, and the q-value for the protein at rank $i$ is the cumulative average of local false discovery rates up to that rank:

```math
q_i = \frac{1}{i} \sum_{j=1}^{i} \text{lfdr}_{(j)}
```

where $(j)$ denotes the $j$-th protein in the sorted order. A protein with $q_i < \alpha$ means that among all proteins ranked at least as highly, the expected proportion of false discoveries is at most $\alpha$.

### Convergence Diagnostics
- **ESS (Effective Sample Size)**: Measures quality of posterior samples (should be > 400)
- **Rhat**: Gelman-Rubin convergence diagnostic (should be < 1.01)

## Multiple Imputation

Mass spectrometry data frequently contain missing values (missing not at random or missing at random). BayesInteractomics supports **multiple imputation** to propagate missing-data uncertainty into all downstream inferences.

The procedure follows Rubin's (1987) combining rules:

1. **Generate $M$ imputed datasets**: Each imputed dataset fills in missing values from a plausible imputation model.
2. **Fit the full model on each imputed dataset**: The enrichment model (HBM) and regression model are run independently on each of the $M$ datasets, yielding $M$ posterior distributions per protein.
3. **Pool posteriors as an equal-weight mixture**: For each protein parameter $\theta$, the pooled posterior is a mixture of the $M$ individual posteriors:

```math
q_{\text{pooled}}(\theta) = \frac{1}{M} \sum_{m=1}^{M} q_m(\theta)
```

where $q_m(\theta)$ is the posterior from the $m$-th imputed dataset.

4. **Compute statistics on the mixture**: Bayes factors, log2FC summaries, credible intervals, and all other summary statistics are computed from the pooled mixture distribution. This naturally incorporates both within-imputation uncertainty (each $q_m$ has its own spread) and between-imputation uncertainty (the $q_m$ may have different locations).

## Differential Analysis

BayesInteractomics provides a framework for comparing interaction profiles between two experimental conditions (e.g., wild-type vs. mutant, treated vs. untreated). For each protein present in both conditions, the analysis quantifies whether the interaction evidence differs.

### Differential Bayes Factor

The differential Bayes factor measures relative evidence between condition A and condition B:

```math
\text{dBF}_i = \frac{BF_{i,A}}{BF_{i,B}}
```

computed in log-space as $\log_{10}(\text{dBF}_i) = \log_{10}(BF_{i,A}) - \log_{10}(BF_{i,B})$. A positive $\log_{10}(\text{dBF})$ indicates stronger interaction evidence in condition A. Both the combined Bayes factor and per-evidence Bayes factors (enrichment, correlation, detection) are compared, allowing diagnosis of which evidence arm drives the differential signal.

### Effect Size and Differential Posterior

The effect size is the difference in mean log2 fold changes:

```math
\Delta\text{log2FC}_i = \overline{\text{log2FC}}_{i,A} - \overline{\text{log2FC}}_{i,B}
```

The differential posterior probability quantifies evidence for *any* difference (direction-agnostic):

```math
P(\text{diff} \mid D_i) = \frac{|\text{dBF}_i|}{1 + |\text{dBF}_i|}
```

Multiple testing is controlled by computing Bayesian FDR q-values on the differential posteriors (see [Bayesian FDR q-values](@ref) above).

### Interaction Classification

Each protein is classified into one of five categories based on the differential evidence and configurable thresholds:

| Class | Description |
|-------|-------------|
| `GAINED` | Interaction is stronger or exclusively present in condition A |
| `REDUCED` | Interaction is stronger or exclusively present in condition B |
| `UNCHANGED` | No significant differential evidence |
| `BOTH_NEGATIVE` | Neither condition shows interaction, but differential q-value is significant |
| `CONDITION_SPECIFIC` | Protein detected in only one condition (appended as `CONDITION_A_SPECIFIC` or `CONDITION_B_SPECIFIC`) |

Three classification methods are available:
- **Posterior**: uses per-condition posterior probability thresholds and $\Delta\text{log2FC}$
- **dBF**: uses $|\log_{10}(\text{dBF})|$ exceeding a threshold
- **Combined**: requires both posterior and dBF criteria to hold simultaneously

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

### Robust Regression
- Lange, K. L., Little, R. J. A., & Taylor, J. M. G. (1989). Robust statistical modeling using the t distribution. *Journal of the American Statistical Association*, 84(408), 881-896.

### Model Comparison
- Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and widely applicable information criterion in singular learning theory. *Journal of Machine Learning Research*, 11, 3571-3594.
- Gelman, A., Hwang, J., & Vehtari, A. (2014). Understanding predictive information criteria for Bayesian models. *Statistics and Computing*, 24(6), 997-1016.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413-1432.

### Copula Theory
- Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer.
- Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman & Hall/CRC.

### Mixture Models and BMA
- McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*. Wiley.
- Hoeting, J. A., Madigan, D., Raftery, A. E., & Volinsky, C. T. (1999). Bayesian model averaging: a tutorial. *Statistical Science*, 14(4), 382-417.

### Summary Statistics
- Makowski, D., Ben-Shachar, M. S., Chen, S. H. A., & Lüdecke, D. (2019). Indices of effect existence and significance in the Bayesian framework. *Frontiers in Psychology*, 10, 2767.
- Efron, B., Tibshirani, R., Storey, J. D., & Tusher, V. (2001). Empirical Bayes analysis of a microarray experiment. *Journal of the American Statistical Association*, 96(456), 1151-1160.

### Multiple Imputation
- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.

### Variational Inference
- Blei, D. M., et al. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859-877.
- Bagaev, D., & de Vries, B. (2023). RxInfer: A Julia package for reactive message-passing-based Bayesian inference. *Journal of Open Source Software*, 8(84), 5161.

### Proteomics Applications
- Choi, H., et al. (2011). SAINT: Probabilistic scoring of affinity purification-mass spectrometry data. *Nature Methods*, 8(1), 70-73.
- Mellacheruvu, D., et al. (2013). The CRAPome: A contaminant repository for affinity purification-mass spectrometry data. *Nature Methods*, 10(8), 730-736.
