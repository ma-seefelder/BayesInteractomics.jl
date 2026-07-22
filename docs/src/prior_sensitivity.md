# Prior Sensitivity

Posterior probabilities should not depend strongly on the choice of prior. This page describes how the package detects, quantifies, and corrects for prior sensitivity in the latent class (3c-EM) and copula evidence-combination models.

The audience is **methods-oriented**: equations, fixed-point algorithms, and citations are explicit. For the biologist's view of how sensitivity flags appear in the report, see [Reports](@ref) (Sensitivity tab).

The sensitivity stack consists of four components, applied in order:

1. **Empirical Bayes** estimation of the Dirichlet concentration `α` from the data (Minka 2000).
2. **BIC-weighted prior grid marginalization** — refit the model at multiple Dirichlet centers and average posteriors weighted by BIC.
3. **Sensitivity sweep** — joint Cartesian product over latent-class Dirichlet priors and copula Beta priors, ~27 grid points.
4. **Classification stability traffic light** — per-protein `robust` / `sensitive` / `fragile` labels surfaced in the results DataFrame.

## Empirical Bayes Dirichlet Concentration

### What it does

The 3-component latent class model (`H0` / `Agnostic` / `H1`) places a Dirichlet prior `Dir(α)` on the mixing weights. Choosing `α` by hand is uncomfortable: a too-strong prior can suppress genuine signal, a too-weak prior fails to regularise small classes. **Empirical Bayes** sidesteps the choice by estimating `α` directly from the observed responsibilities — that is, fitting `α` to the data itself.

Given an `N × K` matrix of responsibilities `γ` from a one-shot EM fit (each row sums to ~1), Minka's (2000) fixed-point iteration solves:

```
ψ(α_k) = ψ(Σ_j α_j) + (1/N) Σ_n log γ_{n,k},     k = 1..K
```

where `ψ(·)` is the digamma function. The fixed-point update applies the **inverse digamma** to the right-hand side, yielding a guaranteed-convergent iteration on `α`.

### API

Both functions are exported from `BayesInteractomics`:

- `inv_digamma(y; maxiter=50, tol=1e-12)` — Newton-method solver for the inverse digamma function. Two-regime initialisation per Minka 2000 (`x0 = exp(y) + 0.5` for `y >= -2.22`, otherwise `x0 = -1 / (y - ψ(1))`). Positivity-guarded throughout.
- `estimate_dirichlet_eb(γ; maxiter=1000, tol=1e-8, floor=0.5, sum_bounds=(3.0, 30.0))` — outer fixed-point loop. Returns a `NamedTuple{(:alpha, :converged, :iterations, :was_clamped)}`. Post-convergence clamping enforces individual-component floors (default 0.5) and a total-strength range `[3, 30]` to prevent pathological collapse to a degenerate Dirichlet.

### Citation

Minka, T. P. (2000). *Estimating a Dirichlet distribution*. Technical report, M.I.T. Available at `research.microsoft.com/~minka/papers/dirichlet/`.

## BIC-weighted Prior Grid Marginalization

A single empirical Bayes (EB) point estimate of `α` collapses uncertainty about the prior. To preserve that uncertainty, the package builds a **constant-strength prior grid** centered on the EB estimate and averages posteriors across the grid weighted by per-grid-point BIC.

### Grid construction

`build_prior_grid(α_hat::Vector{Float64})` constructs a 9-point grid on the 2-simplex of mixing-weight proportions, holding the total concentration `S = Σ α_hat` fixed:

| Grid point | Description |
|------------|-------------|
| 1 | EB center (the input `α_hat`) |
| 2 | Uniform `(S/3, S/3, S/3)` |
| 3, 4, 5 | Single-component pushes (~25% mass shift toward H0, Agnostic, or H1) |
| 6, 7, 8 | Two-component pushes (~20% combined shift) |
| 9 | Strong H0 corner (~30% shift) |

Components are floored at 5% of `S` so the simplex remains non-degenerate, and duplicates are removed via `isapprox`. The 9-point structure is laid out as a **ternary plot** (simplex visualisation) in the report's Sensitivity tab.

### BIC averaging

`_marginalize_over_priors(...)` runs a multi-restart EM at each grid point in parallel (`Threads.@spawn`), records the best log-likelihood per grid point, and computes BIC weights:

```
w_g = exp(LL_g - logsumexp(LL_1, ..., LL_G))
```

(All grid points have the same parameter count and data size, so the BIC penalty cancels and the weights reduce to softmax over log-likelihoods.) The marginalised posterior and combined Bayes factor are then:

- **Posterior probability** — arithmetic average on probability scale: `P_avg(i) = Σ_g w_g · P_g(i)`.
- **Combined BF** — geometric average via log scale: `log BF_avg(i) = Σ_g w_g · log BF_g(i)`, clamped to `[-46, 46]` to avoid `1e300` overflows when posteriors saturate.

If a single grid point dominates the BIC weights (`dominant_weight > 0.95`), the function emits a warning — averaging provides limited robustness in that case, and the user should inspect the per-grid log-likelihoods directly.

## Sensitivity Sweep

When `CONFIG.run_sensitivity = true`, the pipeline runs `sensitivity_analysis(ar, data; config = SensitivityConfig(), ...)` after the main BMA fit. The sweep is **joint** over latent-class Dirichlet priors and copula Beta priors — a Cartesian product of grids:

| Prior model | Default grid (in `SensitivityConfig`) | Size |
|-------------|---------------------------------------|------|
| Latent class Dirichlet (`lc_alpha_prior_grid`) | `[10,5,1]`, `[5,5,1]`, `[2,2,1]`, `[1,1,1]` | 4 |
| Copula EM Beta (`em_prior_grid`) | `(α=10, β=190)`, `(α=25, β=175)`, `(α=50, β=100)` | 3 |
| Beta-Bernoulli detection (`bb_priors`) | empty by default — opt-in | 0 |

Sweep size: 4 × 3 + baseline + Beta-Bernoulli (if enabled) ≈ **~27 settings** for a default `:bma` run that exercises both branches. Per-grid-point LOO stacking weights are recomputed when `combination_method = :bma`, so the sweep is **BMA-aware**.

### `SensitivityConfig` and `SensitivityResult`

```@docs
SensitivityConfig
SensitivityResult
PriorSetting
sensitivity_analysis
generate_sensitivity_report
sensitivity_rank_correlation
```

`SensitivityResult` exposes:

- `posterior_matrix::Matrix{Float64}` — `n_proteins × n_settings`.
- `bf_matrix::Matrix{Float64}` — combined BFs across all settings.
- `bfdr_matrix::Matrix{Float64}` — Storey-corrected BFDRs across all settings.
- `prior_settings::Vector{PriorSetting}` — labels and parameter values for each column.
- `baseline_index::Int` — which column corresponds to the actual analysis baseline.
- `summary::DataFrame` — per-protein min/max/mean/std posterior across all settings.
- `classification_stability::DataFrame` — per-protein fraction of settings exceeding each threshold (`P > 0.5`, `P > 0.8`, `P > 0.95`, `BFDR < 0.05`, `BFDR < 0.01`) plus boundary-crossing booleans.

### Outputs

The sensitivity sweep produces six artefacts (paths configured via `OutputFiles`):

1. **Tornado plot** — per-protein posterior range across all priors, sorted by range. Surfaces the proteins most affected by prior choice.
2. **Heatmap** — `n_top_proteins × n_settings` posterior values, colour-coded.
3. **Spearman rank-correlation heatmap** — pairwise rank correlation between every pair of prior settings (`sensitivity_rank_correlation(sr)`). Off-diagonal cells close to 1.0 indicate that ranks are stable; cells near 0 indicate rank reshuffling.
4. **Decision-boundary stability band** — number of proteins crossing the `BFDR < 0.05` boundary at each grid point, with confidence band.
5. **Posterior overlay violins** — for the top-N most-sensitive proteins, violin plots of posterior values across the sweep.
6. **Ternary prior plot** — visualisation of the Dirichlet prior grid on the 2-simplex.

## Classification Stability Traffic Light

For each protein, the sweep produces a single human-readable label appended to the results DataFrame:

| Label | Meaning |
|-------|---------|
| `robust` | Classification (e.g., `BFDR < 0.05` rejection) is unchanged across the entire prior grid. |
| `sensitive` | Classification flips at some grid points but a clear majority is consistent with the baseline. |
| `fragile` | Classification flips frequently across the grid; the protein's status depends on the prior choice. |

The traffic-light label is computed in `predictive_checks.jl` from the `classification_stability` DataFrame using the boundary-crossing booleans. The label appears in the **Sensitivity** tab of the HTML report and in the `classification_stability` column of the merged results DataFrame, alongside the per-protein `frac_BFDR_lt_0_05` and `threshold_crossing_0_95` columns.

## CONFIG `:auto` Dispatch

The `CONFIG.lc_alpha_prior` field controls Dirichlet prior selection:

- `:auto` (**default**) — triggers Empirical Bayes estimation via `estimate_dirichlet_eb`, then the BIC-weighted prior grid marginalization. The estimated `α_hat` becomes the EB center of the 9-point grid.
- `[a, b, c]` — an explicit Dirichlet vector. EB is skipped; the model uses the given vector directly without grid marginalization.
- Any other `Symbol` — reserved for future named-prior extensions.

The cache invalidation logic detects the `:auto` ↔ explicit-alpha switch and recomputes accordingly, so toggling the field does not require manual cache deletion.

## Code Example

```julia
using BayesInteractomics

config = CONFIG(
    datafile = ["my_apms_data.xlsx"],
    control_cols = [Dict(1 => [2, 3, 4])],
    sample_cols  = [Dict(1 => [5, 6, 7])],
    poi = "MYC",
    refID = 1,
    n_controls = 3,
    n_samples = 3,
    combination_method = :bma,
    lc_alpha_prior = :auto,                     # EB + grid marginalization
    run_sensitivity = true,                     # full sweep on top
    sensitivity_config = SensitivityConfig(),   # default grids
)

results = run_analysis(config)
```

For a standalone sensitivity analysis on a previously-completed `AnalysisResult`:

```julia
# Custom grid: tighter than the default
sc = SensitivityConfig(
    lc_alpha_prior_grid = [
        [10.0, 5.0, 1.0],
        [5.0, 5.0, 1.0],
        [2.0, 2.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    em_prior_grid = [
        (α=10.0, β=190.0),
        (α=25.0, β=175.0),
        (α=50.0, β=100.0),
    ],
    n_top_proteins = 30,
)

sr = sensitivity_analysis(ar, data;
    config = sc,
    n_controls = 3, n_samples = 3, refID = 1,
    combination_method = :bma,
    verbose = true,
)

println(sr)                               # SensitivityResult summary
println("Mean posterior range: ", mean(sr.summary.range))

# Generate the markdown report
generate_sensitivity_report(sr;
    filename = "sensitivity_report.md",
    title = "MYC interactome — prior sensitivity",
)
```

The Empirical Bayes step itself can be invoked directly when iterating on the EM:

```julia
# Suppose `gamma` is the N x 3 responsibility matrix from a one-shot 3c-EM fit
eb = estimate_dirichlet_eb(gamma; floor=0.5, sum_bounds=(3.0, 30.0))
println("EB alpha = ", eb.alpha)
println("Converged = ", eb.converged, " in ", eb.iterations, " iterations")

# And build the 9-point grid centered on it
grid = build_prior_grid(eb.alpha)
println("Grid size: ", length(grid), " distinct points on the simplex")
```

## Validation Targets

For HAP40 interactome (the package's reference dataset):

- **Spearman rank correlation across the full prior grid > 0.95** — ranks are essentially insensitive to prior choice.
- **Boundary crossers (`threshold_crossing_0_95 == true`) < 50** — fewer than 50 proteins flip across the `P > 0.95` boundary anywhere in the sweep.

These targets serve as regression gates for changes to the EM and BMA stacks.

## See Also

- [Model Evaluation](@ref) — BMA section. Sensitivity is BMA-aware: per-grid-point LOO stacking weights are recomputed when `combination_method = :bma`.
- [Reports](@ref) — the **Sensitivity** tab in the interactive HTML report visualises tornado plots, ternary prior grid, decision-boundary stability band, posterior violins, and the classification stability traffic light.
- [Simulation & Calibration](@ref) — complementary calibration approach. Sensitivity sweeps prior choice; simulation generates synthetic ground truth. The two are independent and additive.
- [Mathematical Background](@ref) — Minka fixed-point derivation and BIC-weighting math.
- [Diagnostics](@ref) — classification stability traffic light is part of the broader diagnostics suite.
