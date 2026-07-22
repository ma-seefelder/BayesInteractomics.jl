# Examples

This page enumerates the four example scripts shipped under `examples/` in the
package repository. Each one targets a distinct use case and pairs with one of
the four self-contained branches of the [Tutorial](tutorial.md). Read the
minimal example first; the others build on the same pipeline shape.

| # | Script                                       | Use case                                       | Tutorial cross-ref     |
|---|----------------------------------------------|------------------------------------------------|------------------------|
| 1 | `examples/minimal_workflow.jl`               | "If you read one example, read this"           | [Branch A](tutorial.md#branch-a-single-dataset-one-protocol) |
| 2 | `examples/hap40_interactome.jl`              | Single-bait, two-protocol with optional docking | [Branch A](tutorial.md#branch-a-single-dataset-one-protocol) and [Branch D](tutorial.md#branch-d-with-alphafold-docking-validation) |
| 3 | `examples/hap40_differential_interactome.jl` | Two-condition differential interactome         | [Branch C](tutorial.md#branch-c-differential-interactome-two-conditions) |
| 4 | `examples/meta_analysis_workflow.jl`         | Multi-protocol meta-analysis with multiple imputation | [Branch B](tutorial.md#branch-b-multiple-protocols-meta-analysis) |

All four scripts activate the `examples/` Julia environment via
`Pkg.activate(@__DIR__)` so they pick up the same package versions as the rest
of the project. Replace the placeholder paths and column dictionaries with your
own data layout before running.

---

## 1. `examples/minimal_workflow.jl`

**Use case:** smallest end-to-end pipeline run that still exercises every v1.1.x
capability — STRING ID curation, input data QC, BMA evidence combination,
parametric simulation + Platt calibration, mixture-model validation, and an
interactive HTML report. About 40 executable lines once placeholders are
filled in. This is the recommended starting point for new users; once it runs
on your data, branch out to one of the more specialized scripts below.

Cross-reference: [Tutorial Branch A](tutorial.md#branch-a-single-dataset-one-protocol),
[Data Quality Control](data_quality_control.md),
[Simulation & Calibration](simulation_calibration.md),
[Model Evaluation](model_evaluation.md).

```julia
using BayesInteractomics

data, bait_idx = load_data(
    [data_file], sample_cols, control_cols;
    normalise_protocols = true,
    curate              = true,
    bait_name           = "MYBAIT",
)

config = CONFIG(
    datafile              = [data_file],
    sample_cols           = sample_cols,
    control_cols          = control_cols,
    poi                   = "MYBAIT",
    refID                 = bait_idx,
    n_controls            = 3,
    n_samples             = 3,
    combination_method    = :bma,
    run_input_qc          = true,
    run_validation        = true,
    run_simulation        = true,
    output                = OutputFiles("./results", image_ext = ".svg"),
)

results, ar = run_analysis(config)
```

---

## 2. `examples/hap40_interactome.jl`

**Use case:** a single bait (HAP40) profiled with two affinity-purification
protocols (GST-tagged and Strep-tagged), merged in one analysis with
`normalise_protocols = false` (the per-protocol scales are kept). The script
demonstrates the optional three-phase AlphaFold Server docking workflow at the
end: generate JSON requests, manually upload to alphafoldserver.com, parse
result ZIPs and update posteriors via the two-stage Bayesian update
`P_final = odds_ms · BF_dock / (1 + odds_ms · BF_dock)`.

Cross-reference: [Tutorial Branch A](tutorial.md#branch-a-single-dataset-one-protocol),
[Tutorial Branch D](tutorial.md#branch-d-with-alphafold-docking-validation),
[Docking Integration](docking.md), [Reports](reports.md).

```julia
wtHAP40_config = CONFIG(
    datafile = ["data/GST_HAP40.xlsx", "data/HAP40_Strep.xlsx"],
    control_cols = [
        Dict(1 => [2,3,4], 2 => [5,6,7], 3 => [8,9,10]),    # GST
        Dict(1 => [2,3,4], 2 => [5],     3 => [6,7]),       # Strep
    ],
    sample_cols = [
        Dict(1 => [11,12,13], 2 => [14,15,16], 3 => [17,18,19]),
        Dict(1 => [8,9,10],   2 => [11,12,13], 3 => [14,15]),
    ],
    poi                = "9606.ENSP00000479624",
    refID              = 1,
    n_controls         = 15,
    n_samples          = 17,
    combination_method = :bma,
    output             = OutputFiles("results/wtHAP40", image_ext = ".svg"),
)
results, ar = run_analysis(wtHAP40_config)
```

The docking phase (excerpted):

```julia
docking_config = DockingConfig(
    posterior_threshold = 0.8,
    pep_threshold       = 0.01,
    max_pairs           = 200,
    parse_full_data     = true,
)
batch = generate_docking_requests(results, HAP40_SEQUENCE;
    bait_name = "HAP40", config = docking_config)
# ... user uploads batch.guide_path JSONs to alphafoldserver.com ...
docking         = import_docking_results(docking_results_dir, results; config = docking_config)
updated_results = apply_docking_update(results, docking)
```

---

## 3. `examples/hap40_differential_interactome.jl`

**Use case:** head-to-head comparison of two single-protocol HAP40
interactomes (HAP40-Strep vs GST-HAP40) using `differential_analysis`. Each
condition is fitted independently with `combination_method = :bma`; the
differential pass produces per-protein `BFDR_A`, `BFDR_B`, `differential_BFDR`,
`PEP_A`, `PEP_B`, `diff_PEP`, `delta_log2fc`, and a `classification` column
(GAINED / REDUCED / UNCHANGED / BOTH_NEGATIVE / CONDITION_A_SPECIFIC /
CONDITION_B_SPECIFIC).

Cross-reference: [Tutorial Branch C](tutorial.md#branch-c-differential-interactome-two-conditions),
[Differential Analysis](differential_analysis.md), [Reports](reports.md).

```julia
diff_result = differential_analysis(
    HAP40_Strep_config,
    HAP40_GST_config,
    condition_A = "HAP40-Strep",
    condition_B = "GST-HAP40",
    config = DifferentialConfig(
        posterior_threshold    = 0.8,
        bfdr_threshold         = 0.05,
        delta_log2fc_threshold = 1.0,
        classification_method  = :posterior,
        results_file = "results/differential_results.xlsx",
        volcano_file = "results/differential_volcano.svg",
    ),
    scatter_metric = :posterior_prob,
)

# Inspect gained interactors
gained_df = gained_interactions(diff_result)
```

---

## 4. `examples/meta_analysis_workflow.jl`

**Use case:** six-protocol meta-analysis of HTT (Huntingtin) AP-MS data from
four published studies, with substantial missing data (~40%) handled via
multiple imputation. The script demonstrates the multi-protocol `load_data`
API with dummy-column padding, curation report replay
(`.bayesinteractomics_cache/dataset_curation_report.jld2`), and the
imputed-data overload `run_analysis(config, imputed_data, raw_data)` that pools
Bayes factors across imputations. It then runs `differential_analysis` between
wild-type and mutant HTT.

Cross-reference: [Tutorial Branch B](tutorial.md#branch-b-multiple-protocols-meta-analysis),
[Data Loading](data_loading.md), [Data Curation](data_curation.md),
[Model Evaluation](model_evaluation.md).

```julia
# Replay the wt curation decisions for mut + every imputed dataset
replay_path = joinpath(BASEPATH, ".bayesinteractomics_cache",
                       "dataset_curation_report.jld2")

wt_data  = InteractionData[]
mut_data = InteractionData[]
for i in 1:5
    files = [joinpath(BASEPATH, "imputed_data/dataset_imp_$i.xlsx") for _ in 1:6]
    push!(wt_data,  load_data(files, wt_sample_cols,  control_cols, 1, 1, false;
                              normalise_protocols = true, curate_replay = replay_path))
    push!(mut_data, load_data(files, mut_sample_cols, control_cols, 1, 1, false;
                              normalise_protocols = true, curate_replay = replay_path))
end

# Multiple-imputation overload: pools Bayes factors across imputed datasets
(; diff, result_A, result_B) = differential_analysis(
    wtHTT_config, mHTT_config,
    wt_data, wt_raw_data,
    mut_data, mut_raw_data,
    condition_A = "wtHTT", condition_B = "mHTT",
    config = DifferentialConfig(bfdr_threshold = 0.05))
```

---

## Reading the Results

Every example produces the same set of output artifacts under
`output.basedir`:

| Artifact                       | What it is                                                 |
|--------------------------------|------------------------------------------------------------|
| `final_results.xlsx`           | Per-protein DataFrame: `posterior_prob`, `PEP`, `BFDR`, `log2FC_mean`, `bf_em`, `bf_copula`, plus diagnostic flags + sensitivity range when enabled |
| `interactive_report.html`      | Standalone HTML report with 9 tabs (Results, Evidence, Calibration, Sensitivity, Mixture Model, Structural Evidence, Differential, Data Quality, Methods) |
| `volcano_plot.{png,svg}`       | Volcano with PEP < 0.001 / 0.01 / 0.05 threshold lines     |
| `bma_weights.{png,svg}`        | LOO stacking weights for Copula vs 3c-EM                   |
| `simulation_cache.jld2`        | Cached simulation grid output (when `run_simulation = true`) |
| `calibration_cache.jld2`       | Cached Platt scaling parameters (independent invalidation) |

Refer to the [Tutorial — Reading Your Results](tutorial.md#reading-your-results)
section for the full column inventory and decision threshold table.

---

## Where to go next

- [Tutorial](tutorial.md) — the four use-case branches with full prose context.
- [Data Quality Control](data_quality_control.md) — the v1.1.5 input QC gates
  triggered by `run_input_qc = true`.
- [Simulation & Calibration](simulation_calibration.md) — what
  `run_simulation = true` does and how to read the calibration plot.
- [Prior Sensitivity](prior_sensitivity.md) — the empirical Bayes Dirichlet +
  BIC grid behaviour controlled by `run_sensitivity = true`.
- [Model Evaluation](model_evaluation.md) — the BMA section explaining the
  Copula + 3c-EM stacking and the `bf_em` / `bf_copula` columns.
- [Docking Integration](docking.md) — full BF tier formulas, C2Qscore
  calibration, and the three-phase user-mediated workflow used in
  `examples/hap40_interactome.jl`.
- [Reports](reports.md) — the 9-tab interactive HTML report walkthrough.
