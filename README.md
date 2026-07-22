# BayesInteractomics.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://ma-seefelder.github.io/BayesInteractomics.jl)
[![CI](https://github.com/ma-seefelder/BayesInteractomics.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ma-seefelder/BayesInteractomics.jl/actions/workflows/CI.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**BayesInteractomics.jl** is a Julia package for rigorous Bayesian analysis of protein–protein interactions from Affinity-Purification Mass Spectrometry (AP-MS) and proximity-labeling experiments. It couples a **deep-learning structural-interaction prior** with a **multi-evidence Bayesian evidence engine**, calibrates the resulting posterior probabilities against simulation ground truth, and ships an interactive single-file HTML report.

## Overview

Identifying genuine protein–protein interactions from mass-spectrometry data is challenging because of non-specific binding and contaminants, missing data across replicates and protocols, heterogeneity between experimental methods, and complex dose–response relationships. Traditional single-statistic filters (a t-test on fold changes, or one scoring scheme) discard most of the multi-dimensional structure of the data and leave prior biological knowledge unused.

BayesInteractomics answers this with **two engines that meet at a posterior**:

1. **Prior engine.** A deep neural network (DNN) predicts the probability that two proteins are direct interactors from sequence and network features. That DNN score becomes one of the inputs to a **metalearner** — a self-contained `MLJ.Stack` over a 14-feature schema (`:tr_ddi`) with a bundled isotonic calibrator — which turns orthogonal structural and database evidence into a calibrated **prior** for each candidate.
2. **Evidence engine.** For every candidate the package computes three log-Bayes factors — detection (Beta-Bernoulli), enrichment (Hierarchical Bayesian Model), and dose-response correlation (Bayesian regression) — and combines them with **Bayesian Model Averaging (BMA)** over a **Copula** mixture and a **3-component latent class model (3c-EM)**, yielding a combined Bayes factor `BF_BMA`.

The two engines meet as **posterior odds = prior odds × BF_BMA**, giving a calibrated `posterior_prob` per protein, controlled globally at a Bayesian false discovery rate (**BFDR**) with per-protein Posterior Error Probability (**PEP**) and `local_fdr`.

## Key Features

- **Deep-learning prior via the metalearner.** A DNN structural-contact predictor feeds a self-contained `MLJ.Stack` metalearner over the 14-feature `:tr_ddi` schema (STRING channels + the `DNN` score + Pfam domain–domain interaction counts) with a post-hoc isotonic calibrator. The 14-feature schema is the production default; a 15-feature MC-Dropout variant (`:tr_ddi_mc`) is available as an opt-in via `metalearner_use_mc_dropout = true`. When the metalearner extension is not loaded, `posterior_prob` falls back to the evidence-engine Bayes factor.
- **Bayesian Model Averaging (BMA)** over the **Copula** and **3c-EM** models with LOO stacking weights and a 5% weight floor — robust posteriors even when a single model is misspecified. Sub-model `bf_em` and `bf_copula` columns are exposed for transparency.
- **MNAR dropout-aware imputation.** A per-column logistic dropout curve drives a tilted-Gaussian missing-not-at-random sampler (`imputation_method = :mnar`), with an optional multi-imputation Rubin's-rules pooling path.
- **Differential interactome analysis** (`differential_analysis`): 2-group differential Bayes factor (dBF), k-group all-pairs / one-vs-reference contrasts with a closed-form Gaussian omnibus test, and **Bayesian decision risk** (`optimal_call` + `decision_risk` under a user-overrideable loss matrix) for ranking validation candidates.
- **Multi-protocol normalisation** (`normalisation_method`: `:none` / `:row_center` / `:median_of_ratios` / `:both` / `:auto`) with automatic scale-mismatch detection and normalise-before-impute ordering.
- **Protein and sample embeddings** (`EmbeddingsConfig`): DNN-derived per-protein UMAP/t-SNE coordinates, sample-level embeddings, and pairwise condition similarity feeding the multi-condition report tab.
- **Input data quality control**: five automated checks — scale detection, replicate correlation, missingness asymmetry, intensity-distribution shape, PCA separation — surfaced as `:ok` / `:warning` / `:fail` flags before the full analysis.
- **Parametric simulation engine + Platt calibration**: a 5×5 grid (interaction prevalence × effect scale) × replicates synthesises ground truth; Platt logistic recalibrates predicted posteriors with an Expected Calibration Error (ECE) safety guard.
- **AlphaFold-based structural docking validation**: three scoring tiers — ipTM step function, pDockQ logistic (Burke et al. 2023), and C2Qscore (preferred for AF3) — combined via a two-stage Bayesian update.
- **Robust regression** with Student-t likelihood (configurable ν) and a JZS Cauchy slope prior (`jzs_r_scale = 0.354`, JASP convention).
- **Network analysis** (extension): graph construction, centrality, community detection, Cytoscape/Gephi export.
- **Interactive HTML report** with tabs for Results, Evidence, Calibration, Sensitivity, Mixture Model, Structural Evidence, Differential, Data Quality, and Methods — single-file output, no server.
- **Parallel execution** across thousands of proteins; resumable JLD2 caches for H0, Beta-Bernoulli, HBM/regression, simulation, and calibration.

## Installation

BayesInteractomics requires Julia 1.9 or later:

```julia
using Pkg
Pkg.add(url="https://github.com/ma-seefelder/BayesInteractomics.jl")
```

## Quickstart

A complete v1.2.1 analysis fits in a dozen lines. Replace the file path and column dictionaries with your own data — see the [Tutorial](docs/src/tutorial.md) for column-mapping details.

```julia
using BayesInteractomics

config = CONFIG(
    datafile     = ["data/my_apms_data.xlsx"],            # replace with your file
    control_cols = [Dict(1 => [2,3,4], 2 => [5,6,7])],
    sample_cols  = [Dict(1 => [8,9,10], 2 => [11,12,13])],
    poi          = "BAIT_UNIPROT_ID",
    n_controls   = 6,
    n_samples    = 6,
    refID        = 1,
    output       = OutputFiles("results", image_ext=".svg"),
    combination_method = :bma,           # BMA over Copula + 3c-EM (recommended)
    run_input_qc       = true,           # five-check input QC
    run_simulation     = true,           # parametric simulation + Platt calibration
    run_validation     = true,           # automated quality gates
)
results, ar = run_analysis(config)
run(`open $(config.output.report_file)`)  # open the interactive HTML report
```

To activate the deep-learning prior, load the metalearner extension trigger packages before `using BayesInteractomics` (see [Optional Features](docs/src/optional_features.md)):

```julia
using Flux, MLJ, MLJScikitLearnInterface, HDF5    # metalearner (14-feat :tr_ddi prior)
using GLM                                          # MNAR imputation
using BayesInteractomics
```

The pipeline produces a results DataFrame (with `posterior_prob`, `PEP`, `BFDR`, `local_fdr`, `bf_em`, `bf_copula`, `Combined_BF`, `log2FC_mean`, `diagnostic_flag`, `classification_stability`, etc.) and a single-file HTML report at `results/interactive_report.html`.

## Documentation

The full documentation lives at **[https://ma-seefelder.github.io/BayesInteractomics.jl](https://ma-seefelder.github.io/BayesInteractomics.jl)**.

A good place to start is the use-case-branched **[Tutorial](docs/src/tutorial.md)**, which walks through:

- Single dataset (one protocol),
- Multiple protocols / meta-analysis,
- Differential interactome (two or more conditions),
- AlphaFold docking validation.

Other dedicated pages: [Optional Features](docs/src/optional_features.md) (metalearner prior), [Imputation](docs/src/imputation.md), [Differential Analysis](docs/src/differential_analysis.md), [Decision Risk](docs/src/decision_risk.md), [Data Quality Control](docs/src/data_quality_control.md), [Simulation & Calibration](docs/src/simulation_calibration.md), [Prior Sensitivity](docs/src/prior_sensitivity.md), [Model Evaluation](docs/src/model_evaluation.md) (BMA section), [Diagnostics](docs/src/diagnostics.md), [Docking Integration](docs/src/docking.md), [Reports](docs/src/reports.md).

## Examples

The `examples/` directory contains runnable scripts:

- **`hap40_interactome.jl`** — single-dataset analysis + structural docking on the HAP40 bait.
- **`hap40_differential_interactome.jl`** — wild-type vs. mutant differential interactome.
- **`meta_analysis_workflow.jl`** — multi-protocol meta-analysis combining AP-MS datasets across labs and conditions.

## Scientific Background

BayesInteractomics implements a hybrid framework combining a deep-learning–informed prior with a multi-evidence Bayesian mixture model, described in an accompanying manuscript (**in preparation** — see [`CITATION.cff`](CITATION.cff)).

Key statistical references underpinning the components: Yao et al. 2018 (BMA stacking), Storey 2002 (FDR), Burke et al. 2023 (pDockQ), Minka 2000 (Empirical Bayes Dirichlet fixed-point), Rouder et al. 2009 (JZS prior, JASP r=0.354), Platt 1999 (sigmoid calibration), Geweke 1993 (Student-t likelihood). Full citations in [Mathematical Background](docs/src/mathematical_background.md).

## Performance

- **Parallel processing**: automatic multi-threading across proteins via `Threads.@threads`.
- **Variational inference** via RxInfer.jl for fast convergence (no MCMC chain mixing concerns).
- **Resumable caches**: JLD2 caches for H0, Beta-Bernoulli, HBM/regression, simulation, calibration — independently invalidated on parameter change.

Typical performance: ~1000 proteins in 5–15 minutes on an 8-core laptop with `combination_method = :bma`, `run_simulation = true`, `run_validation = true`.

## Contributing

Contributions are welcome — please submit issues, feature requests, or PRs on [GitHub](https://github.com/ma-seefelder/BayesInteractomics.jl).

For development:

```bash
git clone https://github.com/ma-seefelder/BayesInteractomics.jl.git
cd BayesInteractomics.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Citation

A peer-reviewed publication describing BayesInteractomics is **in preparation**. Until it is posted, please cite the software using the metadata in [`CITATION.cff`](CITATION.cff) at the repository root (a DOI will be added once the preprint is available):

> Seefelder, M. *BayesInteractomics: a Bayesian framework for protein interactome analysis.* Manuscript in preparation.

## License

BayesInteractomics.jl is released under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

This package builds on excellent Julia packages including:

- [RxInfer.jl](https://github.com/reactivebayes/RxInfer.jl) for Bayesian inference,
- [Copulas.jl](https://github.com/lrnv/Copulas.jl) for copula modeling,
- [Flux.jl](https://github.com/FluxML/Flux.jl) and [MLJ.jl](https://github.com/JuliaAI/MLJ.jl) for the deep-learning prior and metalearner,
- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) for probability distributions,
- [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) for data manipulation.

## Contact

For questions, suggestions, or collaboration inquiries:

- **Author**: Manuel Seefelder
- **Email**: manuel.seefelder@uni-ulm.de
- **GitHub**: [https://github.com/ma-seefelder/BayesInteractomics.jl](https://github.com/ma-seefelder/BayesInteractomics.jl)
