# BayesInteractomics.jl Documentation

Welcome to the documentation for **BayesInteractomics.jl**, a comprehensive Julia package for Bayesian analysis of protein–protein interactions from mass-spectrometry experiments.

## What is BayesInteractomics?

BayesInteractomics provides a rigorous statistical framework for identifying genuine protein–protein interactions from Affinity-Purification Mass Spectrometry (AP-MS) and proximity-labeling data. Unlike traditional approaches that rely on single statistical tests or ad-hoc filtering, it couples **two engines** — a deep-learning **prior engine** and a multi-evidence Bayesian **evidence engine** — whose outputs flow into a single calibrated posterior probability per interaction, then into Bayesian false-discovery-rate control and differential comparison across conditions.

### The prior engine

A deep neural network (DNN) predicts direct physical interaction from protein sequence and STRING network data. Its score is combined with orthogonal knowledge channels by a self-contained metalearner stack to form a **data-driven prior** on each candidate interaction. See [Prior Engine (Metalearner)](metalearner.md).

### The evidence engine

The experimental data are then weighed through **three complementary lines of evidence**, each yielding a Bayes factor:

1. **Detection evidence** — Is the protein consistently detected across sample replicates versus controls? (Beta-Bernoulli model)
2. **Enrichment evidence** — Is the protein quantitatively enriched in samples? (Hierarchical Bayesian Model)
3. **Correlation evidence** — Does the protein's abundance correlate with bait protein levels? (Bayesian regression)

These Bayes factors are combined using **Bayesian Model Averaging (BMA)** over two sub-models: a **Copula** mixture and a **3c-EM** 3-component latent class model. The Copula sub-model is itself a three-component mixture — a null (H0), an uninformative agnostic component, and a genuine-interactor (H1) component — with a single three-dimensional copula per component, selected from four bivariate families (Clayton, Frank, Gumbel, Gaussian). LOO stacking weights combine the two sub-models by linear Bayes-factor pooling, chosen for robustness to model misspecification. Posteriors are then **calibrated** against parametric simulation ground truth using Platt scaling with an Expected Calibration Error safety guard.

## Why BayesInteractomics?

### Challenges in interactome analysis

Protein interaction studies face several statistical challenges:

- **Non-specific binding**: many detected proteins are contaminants rather than genuine interactors.
- **Missing data**: not all proteins are detected in every replicate.
- **Protocol heterogeneity**: different experimental methods yield different background distributions.
- **Small sample sizes**: limited replicates make traditional frequentist methods unreliable.
- **Multiple testing**: thousands of proteins require careful control of false-discovery rates.

### Key Features

BayesInteractomics addresses these through a stack of complementary capabilities:

- **Deep-learning prior engine** — a DNN predicts direct protein–protein interaction from sequence and STRING network data; a self-contained metalearner stack (14-feature `:tr_ddi` schema) blends the DNN score with orthogonal knowledge channels into a calibrated per-interaction prior. See [Prior Engine (Metalearner)](metalearner.md).
- **Bayesian Model Averaging** — LOO stacking weights with a 5% floor over a Copula mixture and a 3-component latent class (3c-EM) model. Sub-model `bf_em` and `bf_copula` columns are exposed for transparency. Linear Bayes-factor pooling on the BF scale preserves strong individual evidence.
- **Input Data Quality Control** — five automated checks (scale detection, replicate correlation, missingness asymmetry, intensity-distribution shape, PCA separation) with `:ok` / `:warning` / `:fail` triage flags surfaced before the full analysis runs. See [Input Data Quality Control](data_quality_control.md).
- **Parametric simulation engine + Platt scaling** — `run_simulation` synthesises 25 scenarios × 10 replicates of AP-MS data with known ground-truth interactors; Platt's logistic recalibrates posteriors and FDR thresholds, applied only when ECE improves on held-out folds. See [Simulation & Calibration](simulation_calibration.md).
- **Empirical Bayes Dirichlet + BIC grid marginalization** — Minka's fixed-point estimates the latent-class concentration `α` from the data; a 9-point simplex grid is then BIC-averaged for prior-robust posteriors, with a per-protein `robust` / `sensitive` / `fragile` traffic light. See [Prior Sensitivity](prior_sensitivity.md).
- **AlphaFold structural docking validation** — three scoring tiers: ipTM step function (Tier 1), pDockQ logistic (Burke et al. 2023, Tier 2), and C2Qscore (Tier 2, preferred for AF3, AUC-ROC 0.93 vs pDockQ 0.83). Combined via two-stage Bayesian update `P_combined = odds_ms * BF_dock / (1 + odds_ms * BF_dock)`. See [Docking Integration](docking.md).
- **Storey monotone step-down BFDR** + per-protein PEP and local FDR — strict global FDR control with per-protein interpretability.
- **Robust regression** — Student-t likelihood (configurable ν, optimised via Brent's method on WAIC) and JZS Cauchy slope prior (`jzs_r_scale = 0.354`, JASP convention).
- **Hierarchical Bayesian modeling** — shares information across protocols/experiments while modeling protocol-level heterogeneity.
- **Differential analysis** — paired comparison between two conditions with `GAINED` / `REDUCED` / `UNCHANGED` classification.
- **Network analysis** (extension) — graph construction, centrality, community detection, Cytoscape/Gephi export.
- **Multiple imputation support** — Rubin's rules pooling for high-missingness datasets.
- **Interactive HTML report** — single-file output with nine tabs (Results, Evidence, Calibration, Sensitivity, Mixture Model, Structural Evidence, Differential, Data Quality, Methods).
- **Resumable JLD2 caches** — H0, Beta-Bernoulli, HBM/regression, simulation, and calibration caches with independent invalidation on parameter change.
- **Parallel execution** — multi-threaded across thousands of proteins.

## Package Architecture

### Hierarchical data structure

BayesInteractomics organises data in a hierarchy that mirrors experimental design:

```
InteractionData
├── Protocol 1 (e.g., AP-MS)
│   ├── Experiment 1 (biological replicate set 1)
│   │   ├── Control samples
│   │   └── Bait samples
│   └── Experiment 2 (biological replicate set 2)
│       ├── Control samples
│       └── Bait samples
└── Protocol 2 (e.g., proximity labeling)
    └── Experiment 1
        ├── Control samples
        └── Bait samples
```

This structure enables:

- Protocol-level parameters (e.g., baseline detection rates),
- Experiment-level parameters (e.g., batch effects),
- Sample-level observations with missing data.

### Analysis Workflow

The `run_analysis(config)` function orchestrates ten stages:

1. **Data loading + curation** — Excel/CSV import via `load_data`, with optional STRING-API protein curation (synonym resolution, contaminant removal, group splitting). See [Data Loading](data_loading.md), [Data Curation](data_curation.md).
2. **Input QC** (optional, on by default) — five-check QC system flags scale, replicate correlation, missingness, distribution shape, and PCA separation. See [Input Data Quality Control](data_quality_control.md).
3. **H0 computation** — null Bayes factors fit from negative-control proteins, cached as JLD2 with parameter-hash invalidation (legacy XLSX fallback supported).
4. **Per-protein parallel analysis** — Beta-Bernoulli (detection), HBM (enrichment), Bayesian regression (correlation), all parallelised across proteins via `Threads.@threads`. See [Model Fitting](model_fitting.md).
5. **Non-detected protein exclusion** — proteins absent from all sample replicates are dropped from downstream inference.
6. **Evidence combination via BMA** — Copula mixture and 3-component latent class models fit independently; LOO stacking weights with 5% floor produce the BMA posterior. See [Model Evaluation](model_evaluation.md).
7. **Simulation + Platt calibration** (optional, on by default) — 5×5 grid × 10 replicates synthesises ground truth; Platt logistic recalibrates posteriors with ECE safety guard. See [Simulation & Calibration](simulation_calibration.md).
8. **Quality gates** (optional, on by default) — KS tests on marginals, KL contamination between H0 and H1 components, component separation. See [Diagnostics](diagnostics.md).
9. **Docking integration** (optional, off by default) — generate AlphaFold Server request JSONs from high-confidence MS hits; after manual upload + parse, two-stage Bayesian update produces `posterior_prob_combined`. See [Docking Integration](docking.md).
10. **Report generation** — interactive single-file HTML at `config.output.report_file` plus auto-generated methods text. See [Reports](reports.md).

The complete configuration sits in the `CONFIG` struct; outputs are routed through the `OutputFiles` struct.

## Installation

BayesInteractomics requires Julia 1.9 or later.

### From the package registry

```julia
using Pkg
Pkg.add("BayesInteractomics")
```

### Development version

```julia
using Pkg
Pkg.add(url="https://github.com/ma-seefelder/BayesInteractomics.jl")
```

### First steps

After installation, load the package and verify it works:

```julia
using BayesInteractomics

# Check that key functions are available
?CONFIG
?load_data
?run_analysis
```

## Quick Start Example

A complete v1.2.1 analysis:

```julia
using BayesInteractomics

config = CONFIG(
    datafile     = ["data/experiment.xlsx"],
    control_cols = [Dict(1 => [2,3,4])],
    sample_cols  = [Dict(1 => [5,6,7])],
    poi          = "BAIT_UNIPROT_ID",
    refID        = 1,
    n_controls   = 3,
    n_samples    = 3,
    normalise_protocols = false,
    output       = OutputFiles("results"),
    combination_method = :bma,           # BMA over Copula + 3c-EM (recommended)
    run_input_qc       = true,
    run_simulation     = true,           # parametric simulation + Platt calibration
    run_validation     = true,           # automated quality gates
)

# Run complete pipeline
results, ar = run_analysis(config)

# View top interactions
using DataFrames
first(sort(results, :posterior_prob, rev = true), 10)
```

For a use-case-branched walkthrough, see the [Tutorial](tutorial.md).

## Documentation Navigation

This documentation is organised into several sections:

### [Tutorial](tutorial.md)

**Start here if you're new to BayesInteractomics.** The tutorial branches by use case: single dataset, multiple protocols, differential interactome, or AlphaFold docking. Each branch is self-contained.

### User Guide

In-depth explanations of each component:

- **[Data Loading](data_loading.md)** — file formats, column specifications, hierarchical data structure.
- **[Data Curation](data_curation.md)** — STRING-based contaminant removal, protein-group splitting, synonym resolution, duplicate merging.
- **[Prior Engine (Metalearner)](metalearner.md)** — the DNN structural-contact prior, the self-contained MLJ.Stack over the 14-feature `:tr_ddi` schema, the post-hoc isotonic calibrator, and the graceful fallback when the extension is not loaded.
- **[Input Data Quality Control](data_quality_control.md)** — five-check QC system, what each warning/fail means, what to do.
- **[Analysis Pipeline](analysis.md)** — complete workflow, configuration, caching, parallelism.
- **[Model Fitting](model_fitting.md)** — Beta-Bernoulli, HBM, Bayesian regression internals; JZS prior; per-protein τ_base.
- **[Model Evaluation](model_evaluation.md)** — Bayes factors, BMA section (linear pooling, LOO stacking, sub-model BFs), latent class details.
- **[Diagnostics](diagnostics.md)** — Storey BFDR, local FDR, Pareto-k, classification stability traffic light, posterior predictive checks.
- **[Simulation & Calibration](simulation_calibration.md)** — parametric simulation engine, Platt scaling, ECE safety guard.
- **[Prior Sensitivity](prior_sensitivity.md)** — empirical Bayes Dirichlet (Minka), BIC grid marginalization, sensitivity sweep, traffic light.
- **[Differential Analysis](differential_analysis.md)** — comparing interactomes between conditions; gained/reduced/unchanged classification.
- **[Visualization](visualization.md)** — pipeline plots, per-protein plots, sensitivity bands, within-class correlation.
- **[Reports](reports.md)** — interactive HTML report, nine-tab inventory, auto-generated methods text.
- **[Docking Integration](docking.md)** — AlphaFold Server workflow, three scoring tiers (ipTM, pDockQ, C2Qscore), two-stage Bayesian update.
- **[Network Analysis](network_analysis.md)** — graph construction, topology analysis, hub identification, community detection, Cytoscape/Gephi export.
- **[Optional Features and Extensions](optional_features.md)** — Metalearner + Imputation extension triggers (`using Flux, MLJ, MLJScikitLearnInterface, HDF5` and `using GLM`), Variante B graceful fallback, explicit-error path, TTFX numbers (cold −41.52 %, warm −60.30 %).

### [Examples](examples.md)

Real-world analysis workflows including:

- HAP40 single-dataset analysis with multi-protocol options,
- HAP40 wild-type vs. mutant differential interactome,
- HTT meta-analysis combining six AP-MS datasets across labs,
- Multiple imputation workflow,
- AlphaFold docking validation.

### [Mathematical Background](mathematical_background.md)

Detailed mathematical exposition of:

- Beta-Bernoulli model for detection probability,
- Hierarchical Bayesian Model for enrichment,
- Bayesian linear regression for dose response,
- Copula theory and EM algorithm,
- Student-t H0, sigmoid-gated H1, BIC-selected H1 marginal (3c-EM),
- Storey monotone step-down BFDR,
- Empirical Bayes Dirichlet (Minka 2000),
- BIC-weighted prior grid marginalization,
- JZS Cauchy regression prior (Rouder et al. 2009),
- Platt scaling sigmoid calibration,
- LOO stacking BMA (Yao et al. 2018),
- C2Qscore docking scoring + pDockQ (Burke et al. 2023).

### API Reference

Complete documentation of all exported functions and types. Functions are organised by module for easy navigation.

## Getting Help

If you encounter issues or have questions:

1. **Check the Tutorial** — many common questions are answered in the use-case-branched guide.
2. **Browse Examples** — real workflows may demonstrate what you need.
3. **Search the Docs** — use the search bar to find relevant sections.
4. **GitHub Issues** — report bugs or request features at [github.com/ma-seefelder/BayesInteractomics.jl](https://github.com/ma-seefelder/BayesInteractomics.jl/issues).

## Citation

If you use BayesInteractomics in your research, please cite:

```bibtex
@software{bayesinteractomics2025,
  author = {Seefelder, Manuel},
  title = {BayesInteractomics.jl: Bayesian Analysis of Protein Interactome Data},
  year = {2025},
  url = {https://github.com/ma-seefelder/BayesInteractomics.jl}
}
```

## License

BayesInteractomics.jl is released under the MIT License.
