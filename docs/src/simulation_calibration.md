# Simulation & Calibration

This page documents the parametric simulation engine (`run_simulation`) and the Platt scaling calibration step that uses simulation ground truth to recalibrate posterior probabilities and FDR thresholds.

The audience for this page is **methods-oriented**: it gives the model equations, fitting procedure, and references. For a biologist-friendly walkthrough of how calibrated posteriors appear in the report, see [Reports](@ref) (Calibration tab).

## Why parametric simulation matters

The posteriors that come out of Bayesian Model Averaging (BMA, see [Model Evaluation](@ref)) are *internally* coherent: they reflect the relative weight of evidence under the fitted mixture. Whether they are *externally* calibrated — that is, whether `P(true interactor | posterior_prob = 0.8) ≈ 0.8` — depends on how well the fitted mixture matches the true (unknown) data-generating process.

Without an external gold standard, the only honest way to assess calibration is to generate synthetic data from a reasonable parametric model whose ground truth labels we control, run the full pipeline on that data, and compare predicted posteriors / declared BFDR against the simulation truth. That is the role of the parametric simulation engine.

## Parametric Simulation Engine

### What it does

`run_simulation` draws synthetic Bayes-factor triplets `(BF_enrichment, BF_correlation, BF_detection)` from a fitted [`LatentClassResult`](@ref) (the 3-component EM model: H0 / Agnostic / H1), then sweeps a **5×5 grid** of scenarios over interaction prevalence and effect size. Each scenario is replicated 10 times (default) to produce confidence bands on every calibration metric.

The grid axes:

- **`pi_H1`** — interaction prevalence in the synthetic dataset. Default grid: `[0.02, 0.05, 0.10, 0.15, 0.20]` (5 values).
- **`effect_scale`** — multiplicative scaling of the H1 component's mean BF-vector. Default grid: `[0.5, 0.75, 1.0, 1.25, 1.5]` (5 values).

Total: **25 scenarios × 10 replicates = 250 synthetic datasets** per simulation run.

### `SimulationResult`

`run_simulation` returns a [`SimulationResult`](@ref) struct. Its key fields:

| Field | Type | Description |
|-------|------|-------------|
| `scenarios` | `Vector{ScenarioResult}` | One `ScenarioResult` per (pi_H1, effect_scale) cell. |
| `pi_h1_grid` | `Vector{Float64}` | Prevalence grid actually used. |
| `effect_grid` | `Vector{Float64}` | Effect-scale grid actually used. |
| `n_synthetic` | `Int` | Synthetic proteins per replicate (default 10000). |
| `n_replicates` | `Int` | Replicates per scenario (default 10). |
| `h1_enrichment_family` | `Symbol` | BIC-selected H1 enrichment family (`:gamma`, `:lognormal`, `:weibull`) inherited from the input `LatentClassResult`. |
| `fdr_at_p95_range` | `Tuple{Float64, Float64}` | (min, max) declared BFDR at posterior threshold P > 0.95 across all scenarios. |
| `calibration_model` | `Union{Nothing, CalibrationModel}` | Platt-scaling posterior calibration model (see below). |
| `fdr_calibration_model` | `Union{Nothing, FDRCalibrationModel}` | Platt-scaling FDR threshold calibration model. |
| `calibration_cv` | `Union{Nothing, CalibrationCVMetrics}` | 10-fold stratified cross-validation metrics with ECE per fold. |
| `fdr_curve_empirical` | `Vector{Float64}` | Pre-computed empirical FDR curve from simulation ground truth (100 points). |
| `fdr_curve_declared_bfdr` | `Vector{Float64}` | Pre-computed declared BFDR curve from simulation posteriors (100 points). |

Each `ScenarioResult` stores `n_thresholds`-point curves (default 200 thresholds across `[0, 1]`) for sensitivity, specificity, and FDR, with 2.5th/97.5th-percentile confidence bands across replicates, plus a binned reliability diagram.

### CONFIG flags

Two `CONFIG` fields control simulation:

- `run_simulation::Bool = true` — whether to run the simulation step at all. Default `true` because calibration is generally desirable.
- `sim_n_synthetic::Int = 10_000` — number of synthetic proteins per replicate.

When `run_simulation = true`, the pipeline:

1. Runs the standard analysis end-to-end.
2. Calls `run_simulation(lc_result; n_synthetic = config.sim_n_synthetic, ...)`.
3. Fits the Platt calibration model (next section) using the pooled synthetic posteriors and ground truth labels.
4. Applies calibration to the *real* analysis posteriors only if the ECE safety guard passes.

## Platt Scaling Calibration

### Model

Platt scaling fits a one-parameter logistic regression in logit space (Platt, 1999):

```
calibrated_posterior = logistic(a * logit(raw_posterior) + b)
```

with parameters `a, b` fitted by minimising binary cross-entropy loss against the simulation ground truth (interactor vs non-interactor labels). Optimisation is performed via `Optim.jl`. If fitting fails, the calibration falls back to the identity mapping — calibrated posteriors equal raw posteriors.

A separate `FDRCalibrationModel` is fitted for the threshold-to-FDR mapping:

```
calibrated_fdr = logistic(a_fdr * raw_threshold + b_fdr)
```

This recalibrates declared BFDR cutoffs against the simulated empirical FDR.

### ECE Safety Guard

**Calibration is applied only if it improves Expected Calibration Error (ECE).** This is a hard precondition. The pipeline:

1. Computes `posterior_ece_raw` on held-out cross-validation folds (10-fold stratified).
2. Computes `posterior_ece_calibrated` after applying the fitted Platt model to the same folds.
3. Applies the calibration to the real-data posteriors only if `posterior_ece_calibrated < posterior_ece_raw`.

If raw posteriors are already well calibrated (low ECE), Platt scaling can introduce noise — the ECE guard prevents this regression. When calibration fails the guard, the report flags the situation in the **Calibration** tab with an amber/red badge (`ece_badge_color` field on `CalibrationCVMetrics`).

### Calibration cache

Calibration parameters are stored in a **separate JLD2 file** (`calibration_cache.jld2`) from the rest of the intermediate caches (H0, Beta-Bernoulli, HBM/regression). This independent invalidation matters because:

- Changing `sim_n_synthetic` or the prior grid invalidates `calibration_cache.jld2` only, not the upstream caches.
- Re-running with `run_simulation = false` simply skips the calibration step; the upstream caches remain valid.
- A corrupted `calibration_cache.jld2` does not require recomputing the (expensive) Beta-Bernoulli and HBM caches.

The cache stores `CalibrationModel`, `FDRCalibrationModel`, and `CalibrationCVMetrics` together with a parameter hash. A cache hit reuses the calibration directly.

### Citation

Platt, J. C. (1999). *Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods*. In *Advances in Large Margin Classifiers*, MIT Press, pp. 61–74.

The original paper introduces logistic Platt scaling for SVM outputs; the technique generalises to any binary classifier whose raw scores need to be mapped to calibrated probabilities. Niculescu-Mizil & Caruana (2005) compare Platt scaling against isotonic regression and find Platt to be preferable when training data is limited (which is the case here, since simulation replicates are bounded by compute budget).

## When to Use Simulation

**Use simulation when:**

- You want **calibrated posterior probabilities** for a single dataset before threshold-based selection (e.g., reporting "P > 0.9 ⇒ true positive" in a manuscript).
- You are **benchmarking** the pipeline at a known prevalence to verify expected sensitivity/specificity at given thresholds.
- You need **declared BFDR vs empirical FDR** comparison to set a defensible threshold.
- Your dataset is **non-standard** (very few replicates, very few proteins, very high or very low prevalence) and you want to confirm that the BMA posteriors behave sensibly under those conditions.

**Skipping simulation is reasonable when:**

- You only need a **ranked candidate list** with relative ordering — raw posteriors are monotonic in the underlying evidence, so calibration does not change ranks meaningfully.
- You are running a **screening pipeline** where a fixed BFDR cutoff (e.g., `BFDR < 0.05`) is the only decision you make, *and* you have validated the cutoff on a similar dataset previously.
- Compute is constrained — `run_simulation = false` cuts pipeline runtime substantially because 250 synthetic analyses are skipped.

## Code Example

```julia
using BayesInteractomics

# Standard pipeline — config.run_simulation defaults to true
config = CONFIG(
    datafile = ["my_apms_data.xlsx"],
    control_cols = [Dict(1 => [2, 3, 4])],
    sample_cols  = [Dict(1 => [5, 6, 7])],
    poi = "MYC",
    refID = 1,
    n_controls = 3,
    n_samples = 3,
    combination_method = :bma,
    run_simulation = true,
    sim_n_synthetic = 10_000,
)

results = run_analysis(config)
# results.simulation_result is now a SimulationResult populated with
# 25 scenarios × 10 replicates of synthetic FDR/sensitivity curves.
```

For standalone simulation against a previously-fitted [`LatentClassResult`](@ref) (e.g., when iterating on calibration only):

```julia
# Assume `lc_result` is a LatentClassResult from a prior run_analysis
sim = run_simulation(lc_result;
    n_synthetic = 10_000,
    n_replicates = 10,
    pi_h1_grid  = [0.02, 0.05, 0.10, 0.15, 0.20],
    effect_grid = [0.5, 0.75, 1.0, 1.25, 1.5],
    n_thresholds = 200,
)

# Inspect calibration quality
if sim.calibration_cv !== nothing
    println("Posterior ECE (mean): ", round(sim.calibration_cv.posterior_ece_mean, digits=4))
    println("Passes ECE guard:     ", sim.calibration_cv.passes_ece_threshold)
end

# Range of declared BFDR at P > 0.95 across the 25 scenarios
println("Declared BFDR @ P>0.95 range: ", sim.fdr_at_p95_range)
```

The `calibration_model::CalibrationModel` and `fdr_calibration_model::FDRCalibrationModel` fields hold the fitted Platt parameters `(a, b, n_training, converged)`. These are applied by the pipeline to the real posteriors when the ECE guard passes.

## API Reference

```@docs
run_simulation
SimulationResult
BayesInteractomics.ScenarioResult
BayesInteractomics.CalibrationModel
BayesInteractomics.FDRCalibrationModel
BayesInteractomics.CalibrationCVMetrics
```

## See Also

- [Analysis Pipeline](@ref) — main pipeline and `CONFIG` reference covering `run_simulation`, `sim_n_synthetic`, and how simulation slots into `run_analysis`.
- [Model Evaluation](@ref) — BMA posterior computation and the `LatentClassResult` that feeds into `run_simulation`.
- [Prior Sensitivity](@ref) — complementary sensitivity analysis approach (sweeps priors instead of generating synthetic ground truth). The two are independent and can be combined.
- [Mathematical Background](@ref) — concise math derivation of the Platt scaling logistic + ECE definition.
- [Reports](@ref) — the **Calibration** tab in the interactive HTML report visualises the simulation grid and the ECE-guarded Platt fit.
