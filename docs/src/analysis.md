# Analysis Pipeline

## Overview

The analysis module orchestrates the complete workflow for identifying protein-protein interactions. It coordinates data loading, statistical modeling, evidence combination, simulation-based calibration, automated quality gates, and report generation.

BayesInteractomics provides two levels of entry points:

- **`run_analysis(config)`**: High-level function that handles everything from data loading to final results, with intelligent caching.
- **`analyse(data, ...)`**: Lower-level function that runs the core Bayesian pipeline on pre-loaded data.

## Pipeline Architecture

### `run_analysis` Workflow

```
CONFIG
  |
  +- Input data QC -- Scale, replicates, missingness, distribution shape, PCA (run_input_qc)
  |
  +- load_data() ---- Data loading & curation (curate=true by default)
  |
  +- Cache check ---- Skip if config + data unchanged
  |
  +- analyse() ------ Core Bayesian pipeline
  |   +-- H0 Bayes factors (JLD2 cache; legacy XLSX fallback)
  |   +-- Beta-Bernoulli detection BFs (parallel)
  |   +-- HBM (enrichment) + Regression (correlation) BFs (parallel)
  |   +-- Non-detected protein exclusion
  |   +-- Evidence combination (combination_method = :copula | :latent_class | :bma)
  |
  +- Simulation ----- 5x5 (pi_H1, effect_scale) grid (run_simulation)
  |
  +- Platt calibration ECE-guarded recalibration from simulation ground truth
  |
  +- Quality gates -- KS / KL contamination / separation (run_validation)
  |
  +- Diagnostics ---- PPC, residuals, calibration plots (run_diagnostics)
  |
  +- Sensitivity ---- Prior sensitivity sweep + ternary plot (run_sensitivity)
  |
  +- Docking -------- Optional AlphaFold integration (run_docking)
  |
  +- Visualization -- Volcano (PEP-based), evidence, convergence, etc.
  |
  +- HTML report ---- Interactive Plotly report (generate_report_html)
```

### Core Pipeline (`analyse`)

The `analyse()` function implements a multi-stage Bayesian workflow.

#### Step 1: H0 Computation
- Computes (or loads cached) null-hypothesis Bayes factors for the three evidence streams.
- Cache: JLD2 (`H0Cache`, validated by parameter hash); the legacy XLSX file is read on hit and recomputed otherwise.

#### Step 2: Beta-Bernoulli Detection Model
Parallelized across all proteins:
- Estimates detection probabilities using a Beta-Bernoulli model.
- Generates Bayes factors for detection evidence.
- Thread-safe computation with progress bar.

#### Step 3: Hierarchical Bayesian Model + Regression
Parallelized analysis of enrichment and correlation:
- **Enrichment (HBM)**: Log2 fold-change across conditions using variational Bayes via RxInfer.
- **Correlation (Regression)**: Bayesian linear regression. Supports Normal and robust Student-t likelihoods (`regression_likelihood`); the slope prior follows a JZS-style Cauchy via Normal-Gamma scale mixture (`jzs_r_scale`, default `0.354` = JASP convention sqrt(2)/4).
- Per-thread cache files under `cache/` (deleted on success).

#### Step 4: Non-Detected Protein Exclusion
Proteins with zero sample detections have undefined enrichment / correlation BFs. They are removed from the EM mixture fit, the H0 null estimation, and copula model selection. They still appear in the final results DataFrame with appropriate flags.

#### Step 5: Evidence Combination
Three methods are available -- see [Evidence Combination Methods](#evidence-combination-methods) below:

- **Copula** (`:copula`): three-component mixture (H0 / anchored Agnostic / H1) with a single 3-D copula per component, each BIC-selected from four families (Clayton, Frank, Gumbel, Gaussian).
- **3c-EM / Latent Class** (`:latent_class`): 3-component mixture on log-BF scale (Student-t H0, anchored Agnostic at mu=0, sigmoid-gated H1).
- **BMA** (`:bma`, recommended default): Bayesian Model Averaging across the Copula and 3c-EM sub-models via LOO stacking weights.

See [`model_evaluation.md`](model_evaluation.md) for the full BMA section.

#### Step 6: Simulation (optional, `run_simulation = true`)
Parametric simulation engine generates synthetic AP-MS data with known ground truth. Sweeps a 5x5 grid over `pi_H1` (interactor proportion) and `effect_scale` (effect-size multiplier). Used to compute pipeline sensitivity / specificity and to fit calibration models.

#### Step 7: Platt Calibration (optional, automatic when simulation runs)
Logistic recalibration of posterior probabilities trained on simulation ground truth. ECE (Expected Calibration Error) safety guard: calibration is only applied if it improves cross-validated ECE.

#### Step 8: Quality Gates (optional, `run_validation = true`)
Automated mixture-model validation: KS goodness-of-fit per (component x marginal), KL contamination between H0 and H1, component separation, within-class correlation.

#### Step 9: Output Generation
- Final results DataFrame written to Excel via `output.results_file`.
- Plots: volcano (PEP-based), evidence, convergence, EM diagnostics, prior plots, etc.
- Interactive HTML report (`generate_report_html`, default `true`).

## Configuration

All analysis parameters are centralized in the [`CONFIG`](@ref) struct (`Base.@kwdef mutable struct`). The fields below match the current `src/analysis/pipeline.jl` definition.

```julia
config = CONFIG(
    # Required: data and column specifications
    datafile     = ["experiment.xlsx"],
    sample_cols  = [Dict(1 => [5, 6, 7])],
    control_cols = [Dict(1 => [2, 3, 4])],
    poi          = "BAIT_PROTEIN",

    # Output paths (auto-generated from basedir)
    output = OutputFiles("/path/to/results"),
    # or: output = OutputFiles("/path/to/results", image_ext=".svg")

    # Analysis parameters
    n_controls = 3,
    n_samples = 3,
    refID = 1,
    normalise_protocols = true,

    # Evidence combination
    combination_method = :bma,           # :copula, :latent_class, or :bma (recommended default)
    em_n_restarts = 20,
    copula_criterion = :BIC,

    # Latent class parameters (used by :latent_class and :bma)
    lc_n_iterations = 100,
    lc_alpha_prior = :auto,              # :auto = Empirical Bayes (Minka) + BIC grid marginalization
    lc_winsorize = true,
    lc_winsorize_quantiles = (0.01, 0.99),

    # Regression model
    regression_likelihood = :robust_t,   # :normal or :robust_t
    student_t_nu = 5.0,
    optimize_nu = true,                  # Brent's method on WAIC over [3, 50]
    run_model_comparison = true,
    jzs_r_scale = 0.354,                 # JZS Cauchy r (JASP convention sqrt(2)/4); 0.0 = Normal slope prior
    regression_min_posterior_var = 0.01, # Variance floor against VMP over-confidence

    # Diagnostics
    run_diagnostics = false,
    run_sensitivity = true,
    run_input_qc = true,                 # v1.1.5 input data QC
    run_validation = true,               # v1.1.4 quality gates
    run_copula_diagnostics = true,

    # Simulation + Platt calibration
    run_simulation = true,
    sim_n_synthetic = 10_000,

    # Data curation (default ON since v1.1)
    curate = true,
    species = 9606,
    bait_name = "BAIT_PROTEIN",

    # Docking (optional, off by default)
    run_docking = false,
    docking_config = nothing,
    bait_sequence = "",
    bait_uniprot = "",

    # Interactive HTML report
    generate_report_html = true,
)
```

### OutputFiles

All output paths are managed through the [`OutputFiles`](@ref) struct. Construct with a base directory to auto-generate every path; override individual fields after construction.

```julia
output = OutputFiles("/path/to/results")             # default .png images
output = OutputFiles("/path/to/results", image_ext=".svg")

# Override individual paths after construction
output.results_file = joinpath("/path/to/results", "custom_name.xlsx")
```

`OutputFiles` covers all generated artefacts. Current fields (see `src/analysis/pipeline.jl`):

| Group | Fields |
|---|---|
| Base | `basedir` |
| Cores | `H0_file`, `results_file`, `volcano_file`, `convergence_file`, `evidence_file`, `dnn_file`, `rank_rank_file`, `prior_file` |
| EM diagnostics | `em_diagnostics_file`, `lc_convergence_file`, `bma_weights_file` |
| Sensitivity | `sensitivity_report_file`, `sensitivity_tornado_file`, `sensitivity_heatmap_file`, `sensitivity_rankcorr_file`, `sensitivity_table_file` |
| Diagnostics | `diagnostics_report_file`, `ppc_histogram_file`, `qq_plot_file`, `regression_qq_plot_file`, `calibration_plot_file`, `pit_histogram_file`, `scale_location_hbm_file`, `scale_location_regression_file`, `nu_optimization_file` |
| Marginals | `h0_marginals_file`, `h1_marginals_file` |
| Copula diagnostics | `kl_divergence_file`, `within_class_corr_file`, `agnostic_zone_file`, `copula_bootstrap_file`, `discordant_proteins_file`, `copula_diagnostics_summary_file` |
| Report | `report_file`, `report_methods_file`, `sidecar_file` |
| Caches | `simulation_file` (calibration cache path is derived from `(basedir, imputation_method)` via `calibration_cache_path(config)` — see Caching below) |

## Caching

### Result-Level Caching

`run_analysis` uses hash-based caching to skip redundant computation:

```julia
# First run: performs full analysis and caches
final_df, result = run_analysis(config)

# Second run with same config/data: loads from cache
final_df, result = run_analysis(config)

# Force re-computation
final_df, result = run_analysis(config, use_cache = false)
```

Cache files are stored in `.bayesinteractomics_cache/` next to the first data file.

### Intermediate Caching

Individual pipeline stages have their own JLD2 caches with parameter validation:

- **`H0Cache`** (replaces legacy `copula_H0.xlsx`)
- **`BetaBernoulliCache`** (per-protein detection BFs)
- **`HBMRegressionCache`** (HBM and regression posteriors)
- **`calibration_cache.jld2`** (Platt scaling, separate invalidation from simulation)

These can also be supplied directly when calling `analyse`:

```julia
results = analyse(data, "copula_H0.xlsx";
    use_intermediate_cache = true,
    betabernoulli_cache_file = "bb_cache.jld2",
    hbm_regression_cache_file = "hbm_cache.jld2"
)
```

## Parallel Processing

The package uses Julia multi-threading:

```bash
# Launch Julia with multiple threads (avoid --threads=auto on Windows; pin to N)
julia --threads=8
```

- **Thread Safety**: Each thread writes to a separate per-thread cache file.
- **Load Balancing**: Proteins are distributed evenly across threads.
- **Error Handling**: Per-protein failures are logged to `log.txt` without aborting the run.

## Multiple Imputation

For multiple imputation, pass a vector of imputed datasets together with the raw data:

```julia
results = analyse(
    imputed_data_vector,  # Vector{InteractionData}
    raw_data,             # InteractionData (used for Beta-Bernoulli)
    "copula_H0.xlsx",
    n_controls = 3, n_samples = 3, refID = 1,
)
```

- HBM and regression are run on each imputed dataset.
- Pooled via `evaluate_imputed_fc_posteriors()` (Rubin's rules).
- Beta-Bernoulli uses raw (non-imputed) data for detection counts.

## Evidence Combination Methods

### Copula (`:copula`)

```julia
combination_method = :copula
```

- Fits a **three-component mixture** over the enrichment / correlation / detection log-Bayes-factor triplet: H0 (background), an anchored Agnostic middle class, and H1 (interactor).
- Each component is modelled by a **single 3-D copula** joining its three marginals — not a vine. The H0 and H1 copulas are BIC-selected from four single-parameter families (Clayton, Frank, Gumbel, Gaussian); the Agnostic component uses an independence copula. See [`mathematical_background.md`](mathematical_background.md) for the joint-density derivation. The user-facing model name is "Copula".
- Multi-restart EM with quantile, k-means, and random initialization strategies.
- SQUAREM acceleration for faster convergence.
- Supports informative Beta priors on `pi_1` (`em_prior` named tuple or preset symbol).

### Latent Class / 3c-EM (`:latent_class`)

```julia
combination_method = :latent_class
em_n_restarts = 20
lc_alpha_prior = :auto       # Empirical Bayes (Minka) + BIC grid marginalization
```

A 3-component mixture fitted on the log-BF scale with monotonicity guarantees:

1. **H0 (Background)**: Student-t enrichment marginal (heavy-tailed null), Normal correlation/detection. Degrees of freedom selected from `{3, 5, 7, 10}` by BIC.
2. **Agnostic (Uninformative)**: Enrichment mean anchored at `mu = 0`, preventing redundancy with H0 during EM.
3. **H1 (Interactor)**: Sigmoid-gated transition around the Jeffreys threshold (`ln(sqrt(10)) approx 1.151` nats), with a positive-support marginal (Gamma / LogNormal / Weibull) selected by BIC at iteration 5.

Step-halving guarantees monotonic log-likelihood after burn-in. Winsorization (`lc_winsorize_quantiles`) is applied before fitting. See [`LatentClassResult`](@ref) and [`BayesInteractomics.combined_BF_latent_class`](@ref).

### Bayesian Model Averaging (`:bma`, recommended default)

```julia
combination_method = :bma
```

BMA averages the Copula and 3c-EM models via LOO stacking weights (Yao et al. 2018) with a 5% weight floor. Posteriors are merged using linear BF pooling. Sub-model BFs (`bf_em`, `bf_copula`) are exported alongside the BMA combined `bf` for transparency, and disagreement diagnostics flag proteins where the two sub-models classify differently.

```julia
result.bma_result.em_weight           # Stacking weight for 3c-EM
result.bma_result.copula_weight       # Stacking weight for Copula
result.bma_result.model_disagreement  # BitVector: true where sub-models disagree
result.bma_result.pareto_k            # PSIS-LOO Pareto k-hat (or nothing)
```

See [`BMAResult`](@ref) and the BMA section in [`model_evaluation.md`](model_evaluation.md) for full details.

### Non-Detected Protein Exclusion

Proteins not detected in any sample replicate (zero detections across all conditions) are automatically excluded from EM fitting and from the H0 null computation. They still appear in the final results DataFrame with appropriate flags. This applies uniformly across all combination methods (`:copula`, `:latent_class`, and `:bma`).

## Output Structure

`run_analysis` returns `(final_df, analysis_result)`:

### `final_df` (DataFrame)

| Column | Description |
|---|---|
| `Protein` | Protein identifier |
| `BF` | Combined Bayes factor (BMA / Copula / 3c-EM depending on `combination_method`) |
| `posterior_prob` | Posterior probability of interaction |
| `BFDR` | Bayesian FDR (Storey monotone step-down) |
| `PEP` | Posterior Error Probability (`1 - posterior_prob`) |
| `bf_em` | 3c-EM sub-model BF (BMA only, transparency column) |
| `bf_copula` | Copula sub-model BF (BMA only, transparency column) |
| `mean_log2FC` | Mean log2 fold change |
| `bf_enrichment` | HBM Bayes factor |
| `bf_correlation` | Regression Bayes factor |
| `bf_detected` | Beta-Bernoulli Bayes factor |
| `diagnostic_flag` | Per-protein quality flag (`:ok` / `:warning` / `:fail`, when diagnostics enabled) |
| `sensitivity_range` | Posterior range across the prior grid (when sensitivity enabled) |
| `classification_stability` | `robust` / `sensitive` / `fragile` traffic-light label (when sensitivity enabled) |

### `analysis_result` (AnalysisResult)

Contains all analysis outputs including the chosen combination result, joint distributions, and caching metadata. Use accessor functions:

```julia
getProteins(result)               # Protein names
getBayesFactors(result)           # Combined BFs
getPosteriorProbabilities(result) # Posterior probabilities
getBFDR(result)                   # BFDR values (Storey monotone step-down)
getMeanLog2FC(result)             # Mean log2FC
```

`getQValues(result)` is a deprecated alias for `getBFDR(result)` and emits a deprecation warning.

## Cross-references

- Evidence combination details (BMA, 3c-EM, Copula sub-models, disagreement diagnostics): [`model_evaluation.md`](model_evaluation.md)
- Simulation engine + Platt calibration: [`simulation_calibration.md`](simulation_calibration.md)
- Input data QC (v1.1.5: scale, replicates, missingness, distribution shape, PCA): [`data_quality_control.md`](data_quality_control.md)
- Empirical Bayes Dirichlet + ternary prior grid + classification stability: [`prior_sensitivity.md`](prior_sensitivity.md)
- AlphaFold docking integration (C2Qscore, ipTM, pDockQ): [`docking.md`](docking.md)
- Mathematical foundations and citations: [`mathematical_background.md`](mathematical_background.md)

## API Reference

```@docs
run_analysis
CONFIG
OutputFiles
AnalysisResult
save_result
load_result
evaluate_imputed_fc_posteriors
BetaBernoulliCache
HBMRegressionCache
H0Cache
IntermediateCacheStatus
save_betabernoulli_cache
load_betabernoulli_cache
save_hbm_regression_cache
load_hbm_regression_cache
save_h0_cache
load_h0_cache
CacheStatus
BayesInteractomics.check_cache
BayesInteractomics.check_betabernoulli_cache
BayesInteractomics.check_hbm_regression_cache
BayesInteractomics.compute_config_hash
BayesInteractomics.compute_data_hash
```

The lower-level `analyse` and `check_cache` helpers are internal to the pipeline; access them via `BayesInteractomics.analyse` / `BayesInteractomics.check_cache` and consult the source for their full call signatures. The `CombinedBayesResult` struct is exposed via the BMA / 3c-EM combination flow documented on the [Model Evaluation](@ref) page.
