# Tutorial: Getting Started with BayesInteractomics

This tutorial walks you through your first protein-interactome analysis with BayesInteractomics. It is **branched by use case**: pick the section that matches your data and follow it end-to-end. You do **not** need to read the branches in order — each one is self-contained.

## Prerequisites

- **Julia 1.9 or later** ([install](https://julialang.org/downloads/)).
- Mass spectrometry data — AP-MS or proximity labeling (BioID, APEX), in Excel (`.xlsx`) or CSV format.
- Familiarity with the Julia REPL.
- Understanding of your experimental design: which columns are controls, which are samples, how many replicates, and which protein is the bait.

The package supports any number of protocols and experiments; missing data is handled natively.

## Installation

From the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/ma-seefelder/BayesInteractomics.jl")
```

Verify:

```julia
using BayesInteractomics
```

If that runs without errors you are ready to proceed.

!!! note "Loading optional features"
    For the full metalearner-adjusted posterior pipeline and dropout-aware MNAR imputation, also run:
    ```julia
    using Flux, MLJ, MLJScikitLearnInterface, HDF5   # activates the Metalearner extension
    using GLM                                         # activates the Imputation extension
    ```
    **before** `using BayesInteractomics`. Without the metalearner trigger packages, `posterior_prob` falls back to `bf / (1 + bf)` (the interactive report renders a status banner). Without `GLM`, `imputation_method ∈ (:mar, :mnar)` raises an `ArgumentError`. See [Optional Features and Extensions](optional_features.md) for the full fallback semantics.

## Quick Start: Your First Run

If you just want to see the pipeline run end-to-end, here is the shortest path. It uses the current v1.2.1 defaults: the metalearner (a self-contained `MLJ.Stack` over the 14-feature `:tr_ddi` schema) is the default prior engine that produces `posterior_prob`, MNAR-aware imputation is the default missing-data model, and the three lines of evidence are combined by Bayesian Model Averaging over the Copula and 3c-EM sub-models.

**1. Activate the project environment and load the full stack.** From the Julia REPL in your analysis directory:

```julia
using Pkg
Pkg.activate(".")                                 # your analysis environment

using Flux, MLJ, MLJScikitLearnInterface, HDF5    # metalearner prior engine
using GLM                                          # MNAR imputation
using BayesInteractomics
```

**2. Describe your experiment with a minimal `CONFIG`.** Point `datafile` at your file, list the control and sample columns (1-based indices), and name the bait:

```julia
config = CONFIG(
    datafile     = ["data/my_apms_data.xlsx"],
    control_cols = [Dict(1 => [2, 3, 4])],        # experiment 1: control replicate columns
    sample_cols  = [Dict(1 => [5, 6, 7])],        # experiment 1: sample replicate columns
    poi          = "BAIT_UNIPROT_ID",             # bait STRING / UniProt accession
    n_controls   = 3,
    n_samples    = 3,
    refID        = 1,                             # row index of the bait
    output       = OutputFiles("results"),        # output directory
)
```

Every other setting keeps its default: BMA combination, MNAR imputation, input QC, simulation + Platt calibration, quality gates, and prior sensitivity all run automatically.

**3. Run the analysis.**

```julia
results, ar = run_analysis(config)
```

**4. Read the four columns that matter most.** `results` is a `DataFrame` with one row per protein:

- `posterior_prob` — calibrated probability of a genuine interaction (0–1).
- `PEP` — Posterior Error Probability, `1 - posterior_prob`, per protein.
- `BFDR` — Bayesian False Discovery Rate (global, Storey monotone step-down).
- `log2FC_mean` — posterior mean enrichment (sample vs. control).

```julia
using DataFrames

hits = filter(r -> r.posterior_prob > 0.95, results)
show(sort(hits, :posterior_prob, rev = true)[!, [:Protein, :posterior_prob, :PEP, :BFDR, :log2FC_mean]])
```

The pipeline also writes an interactive single-file HTML report to `config.output.report_file` — open it in any browser (no server needed). That is the whole loop. The rest of this tutorial expands each step and branches by use case.

## Data Format

Your input file must have:

- **Rows** = proteins. **Columns** = samples (one sample per column).
- **First column** = protein identifier (UniProt accession, STRING ID, or gene symbol).
- **Values** = log2-transformed intensities. Missing values left empty or `NA`.

If your data is in linear intensity (typical MaxQuant raw output, range `1e5`–`1e9`), the [Input Data Quality Control](data_quality_control.md) check will flag this with a `:warning` and you should `log2`-transform before running the analysis.

Example layout (column indices are 1-based; column 1 is the protein name):

```
| Protein   | C_1   | C_2   | C_3  | S_1  | S_2  | S_3  |
|-----------|-------|-------|------|------|------|------|
| BAIT      | 18.2  | 18.5  | 18.0 | 32.1 | 31.8 | 32.5 |
| PARTNER_A | 25.3  | 24.8  | 25.1 | 28.5 | 29.2 | 28.8 |
| PARTNER_B | 22.1  | NA    | 22.3 | 22.5 | 22.0 | 22.8 |
```

## Choose Your Path

Pick one branch based on your dataset:

| Branch | Use case |
|---|---|
| **[Branch A — Single dataset (one protocol)](#branch-a-single-dataset-one-protocol)** | One AP-MS or proximity-labeling dataset with controls and samples. The most common starting point. |
| **[Branch B — Multiple protocols / meta-analysis](#branch-b-multiple-protocols-meta-analysis)** | Two or more datasets to combine — e.g., AP-MS + BioID for the same bait, or replicate experiments across labs. |
| **[Branch C — Differential interactome (two conditions)](#branch-c-differential-interactome-two-conditions)** | Compare two interactomes — e.g., wild-type vs mutant bait, ligand-bound vs unbound. Identifies *gained* / *lost* / *unchanged* partners. |
| **[Branch D — With AlphaFold docking validation](#branch-d-with-alphafold-docking-validation)** | Add structural validation on top of any of the above. Distinguishes direct from indirect interactors using AlphaFold Server predictions. |

All branches share the same [Reading Your Results](#reading-your-results) section at the end.

---

## Branch A — Single dataset (one protocol)

Use this branch if you have one AP-MS or proximity-labeling dataset (a single Excel file with controls and samples). This is the most common starting point.

The walkthrough mirrors `examples/hap40_interactome.jl`.

### A.1 Set up the configuration

```julia
using BayesInteractomics

basepath = "/path/to/your/results"   # output directory (created by the pipeline)

config = CONFIG(
    datafile     = ["data/my_apms_data.xlsx"],
    control_cols = [Dict(1 => [2, 3, 4], 2 => [5, 6, 7], 3 => [8, 9, 10])],
    sample_cols  = [Dict(1 => [11, 12, 13], 2 => [14, 15, 16], 3 => [17, 18, 19])],
    poi          = "9606.ENSP00000479624",       # bait STRING/UniProt ID
    n_controls   = 9,
    n_samples    = 9,
    refID        = 1,                            # row index of bait
    normalise_protocols = false,
    output       = OutputFiles(basepath, image_ext = ".svg"),

    # Recommended v1.2.1 defaults (all on by default):
    combination_method = :bma,                   # Bayesian Model Averaging over Copula + 3c-EM
    run_input_qc       = true,                   # input data QC
    run_simulation     = true,                   # parametric simulation + Platt calibration
    run_validation     = true,                   # automated quality gates
    run_diagnostics    = true,
    optimize_nu        = true,                   # Brent's method on Student-t df
    curate_interactive = false,                  # set true for first run to confirm protein merges
)
```

`OutputFiles(basepath)` auto-generates ~30 paths (results, volcano, evidence, calibration, sensitivity, report, etc.). Override individual paths after construction if needed.

The column dictionaries are `Dict(experiment_id => [column_indices...])`. Indices are **1-based**.

### A.2 Run the analysis

```julia
results, ar = run_analysis(config)
```

The pipeline executes ten stages:

1. **Data loading + curation** (STRING API resolves protein synonyms, removes contaminants).
2. **Input QC** — five checks (scale, replicate correlation, missingness, distribution shape, PCA separation).
3. **H0 computation** — null Bayes factors from negative-control proteins (cached as JLD2 for resumption).
4. **Per-protein parallel inference** — Beta-Bernoulli (detection), HBM (enrichment), regression (correlation).
5. **Non-detected protein exclusion** — proteins absent from all sample replicates are dropped.
6. **Evidence combination via BMA** — LOO stacking weights between Copula and 3c-EM models.
7. **Simulation + Platt calibration** — 25 scenarios × 10 replicates synthesises ground truth for posterior recalibration.
8. **Quality gates** — KS tests, KL contamination check, component separation.
9. **Prior sensitivity sweep** — classification stability traffic light per protein.
10. **Report generation** — interactive single-file HTML report.

When the metalearner trigger packages are loaded, the metalearner prior engine adjusts `posterior_prob` after evidence combination (stage 6). If they are not loaded, `posterior_prob` falls back to `bf / (1 + bf)` and the report flags the fallback — see [Optional Features and Extensions](optional_features.md).

Typical wall-clock: **5–15 minutes for ~1000 proteins** on an 8-core laptop.

### A.3 Open the report

```julia
report_path = config.output.report_file        # /path/to/your/results/interactive_report.html
run(`open $report_path`)                        # macOS; use `xdg-open` on Linux, `start` on Windows
```

The report has nine tabs (Results, Evidence, Calibration, Sensitivity, Mixture Model, Structural Evidence, Differential, Data Quality, Methods). See [Reports](reports.md) for a full tour.

### A.4 Pull a high-confidence list programmatically

```julia
using DataFrames

high_confidence = filter(r -> r.posterior_prob > 0.95 && r.PEP < 0.05, results)
println("Found $(nrow(high_confidence)) high-confidence interactions.")

show(sort(high_confidence, :posterior_prob, rev = true)[!, [:Protein, :posterior_prob, :PEP, :BFDR, :log2FC_mean]], allrows = false)
```

`posterior_prob`, `PEP`, `BFDR`, and `log2FC_mean` are columns added to the results DataFrame.

Continue to [Reading Your Results](#reading-your-results).

---

## Branch B — Multiple protocols / meta-analysis

Use this branch if you have **two or more datasets** to combine — for example, AP-MS + BioID for the same bait, or replicate AP-MS experiments from different publications.

The walkthrough mirrors `examples/meta_analysis_workflow.jl`.

### B.1 Specify per-protocol column maps

Each protocol has its own dictionary; the outer vector has one entry per file.

```julia
using BayesInteractomics

# Protocol 1: AP-MS, three experiments with three controls + three samples each.
ap_controls = Dict(1 => [2, 3, 4], 2 => [5, 6, 7], 3 => [8, 9, 10])
ap_samples  = Dict(1 => [11, 12, 13], 2 => [14, 15, 16], 3 => [17, 18, 19])

# Protocol 2: BioID, three experiments with mixed replicate counts.
bioid_controls = Dict(1 => [2, 3, 4], 2 => [5], 3 => [6, 7])
bioid_samples  = Dict(1 => [8, 9, 10], 2 => [11, 12, 13], 3 => [14, 15])

config = CONFIG(
    datafile     = ["data/apms_protocol.xlsx", "data/bioid_protocol.xlsx"],
    control_cols = [ap_controls, bioid_controls],
    sample_cols  = [ap_samples,  bioid_samples],
    poi          = "9606.ENSP00000479624",
    normalise_protocols = true,                  # IMPORTANT: rescale across protocols
    n_controls   = 15,                           # 9 (AP) + 6 (BioID)
    n_samples    = 17,                           # 9 (AP) + 8 (BioID)
    refID        = 1,
    output       = OutputFiles("/path/to/meta_results", image_ext = ".svg"),
    combination_method = :bma,
    run_simulation     = true,
    run_validation     = true,
    curate_interactive = false,
)

results, ar = run_analysis(config)
```

Set `normalise_protocols = true` whenever you combine multiple protocols. The Hierarchical Bayesian Model uses per-protocol parameters that share information across experiments while accounting for protocol-level variability.

### B.2 Multiple imputation (optional)

Missing values are handled natively by the default MNAR-aware imputation — most workflows never need this branch. If you nevertheless want to pool over several externally-imputed replicates, produce `M` imputed Excel files and pass them alongside the raw data:

```julia
# Load the imputed replicates and the raw file with a shared column layout.
# Load raw with imputation = :none and the imputed files with imputation = :mnar,
# and pass filter_insufficient_obs = false so both stay index-aligned.
imputed_data = [
    load_data([f], sample_cols, control_cols;
              normalise_protocols = true, imputation = :mnar, filter_insufficient_obs = false)
    for f in ["data/imputed_1.xlsx", "data/imputed_2.xlsx", "data/imputed_3.xlsx"]
]

raw_data = load_data(["data/raw.xlsx"], sample_cols, control_cols;
                     normalise_protocols = true, imputation = :none, filter_insufficient_obs = false)

results, ar = run_analysis(config, imputed_data, raw_data)
```

The detection (Beta-Bernoulli) model uses `raw_data`; the enrichment and correlation models pool across imputations following Rubin's rules in [`evaluate_imputed_fc_posteriors`](@ref).

Continue to [Reading Your Results](#reading-your-results).

---

## Branch C — Differential interactome (two conditions)

Use this branch when you want to **compare two interactomes** — e.g., wild-type vs mutant bait, ligand-bound vs unbound, or healthy vs disease state. The pipeline classifies each protein as `GAINED`, `REDUCED`, `UNCHANGED`, `BOTH_NEGATIVE`, `CONDITION_A_SPECIFIC`, or `CONDITION_B_SPECIFIC`.

The walkthrough mirrors `examples/hap40_differential_interactome.jl`.

### C.1 Build two CONFIGs (one per condition)

```julia
using BayesInteractomics

condA_config = CONFIG(
    datafile     = ["data/wt_apms.xlsx"],
    control_cols = [Dict(1 => [2, 3, 4], 2 => [5, 6, 7], 3 => [8, 9, 10])],
    sample_cols  = [Dict(1 => [11, 12, 13], 2 => [14, 15, 16], 3 => [17, 18, 19])],
    poi          = "9606.ENSP00000479624",
    normalise_protocols = false,
    n_controls   = 9,
    n_samples    = 9,
    refID        = 1,
    output       = OutputFiles("/path/to/results/wt", image_ext = ".svg"),
    combination_method = :bma,
    run_simulation     = true,
    run_validation     = true,
    curate_interactive = false,
)

condB_config = CONFIG(
    datafile     = ["data/mut_apms.xlsx"],
    control_cols = [Dict(1 => [2, 3, 4], 2 => [5, 6, 7], 3 => [8, 9, 10])],
    sample_cols  = [Dict(1 => [11, 12, 13], 2 => [14, 15, 16], 3 => [17, 18, 19])],
    poi          = "9606.ENSP00000479624",
    normalise_protocols = false,
    n_controls   = 9,
    n_samples    = 9,
    refID        = 1,
    output       = OutputFiles("/path/to/results/mut", image_ext = ".svg"),
    combination_method = :bma,
    run_simulation     = true,
    run_validation     = true,
    curate_interactive = false,
)
```

### C.2 Run the differential analysis

```julia
diff_base = "/path/to/results/wt_vs_mut"

(; diff, result_A, result_B) = differential_analysis(
    condA_config, condB_config,
    condition_A = "Wild-type",
    condition_B = "Mutant",
    config = DifferentialConfig(
        results_file        = joinpath(diff_base, "differential_results.xlsx"),
        volcano_file        = joinpath(diff_base, "differential_volcano.svg"),
        evidence_file       = joinpath(diff_base, "differential_evidence.svg"),
        scatter_file        = joinpath(diff_base, "differential_scatter.svg"),
        classification_file = joinpath(diff_base, "differential_classification.svg"),
        ma_file             = joinpath(diff_base, "differential_ma.svg"),
    ),
    scatter_metric = :posterior_prob,
)
```

`differential_analysis` runs a full per-condition pipeline for each side, then performs a paired comparison. The destructured return:

- `diff::DifferentialResult` — the merged per-protein classification + Bayes factors.
- `result_A::AnalysisResult`, `result_B::AnalysisResult` — full per-condition outputs (including the standalone HTML reports).

### C.3 Pull gained/lost lists

```julia
using DataFrames

gained = gained_interactions(diff)
lost   = lost_interactions(diff)
println("$(nrow(gained)) gained, $(nrow(lost)) lost (BFDR ≤ 0.05).")

show(first(sort(gained, :delta_log2FC, rev = true), 10)[!, [:Protein, :delta_log2FC, :posterior_prob_A, :posterior_prob_B, :class]])
```

Open `differential_volcano.svg`, `differential_classification.svg`, or the `interactive_report.html` for visual exploration. The differential report has its own dedicated layout — see [Differential Analysis](differential_analysis.md).

Continue to [Reading Your Results](#reading-your-results).

---

## Branch D — With AlphaFold docking validation

Use this branch on top of any other branch when you want **structural validation** of your hits to distinguish direct (physically docking) from indirect (in-complex but not contact) interactors.

This is a three-step workflow: generate request JSONs (automated), upload to alphafoldserver.com (manual — there is no public API), then parse and update posteriors (automated).

The walkthrough mirrors `examples/hap40_interactome.jl` (post-analysis section).

### D.1 Run a standard analysis first

Follow Branch A (or B / C) to obtain a `results` DataFrame. Then load the bait sequence:

```julia
HAP40_SEQUENCE = "MAAAAAGLGGGGAGPGPEAGDFLARYRLVSNKLKKRFLRKPNVAEAGEQFGQLGRELRAQE" *
                 "CLPYAAWCQLAVARCQQALFHGPGEALALTEAARLFLRQERDARQRLVCPAAYGEPLQAAA" *
                 "..."   # truncated — full FASTA from UniProt

docking_output = "/path/to/docking_requests"

docking_config = DockingConfig(
    posterior_threshold = 0.8,        # only dock high-confidence MS hits
    pep_threshold       = 0.01,       # max per-protein posterior error probability
    max_pairs           = 200,        # cap on AF Server jobs (~4 days at the daily limit)
    max_tokens_per_job  = 5000,       # AF Server token cap (bait + prey residues)
    max_jobs_per_batch  = 30,         # AF Server daily limit
    parse_full_data     = true,       # parse pDockQ from full_data JSONs (Tier 2)
    request_output_dir  = docking_output,
    verbose             = true,
)
```

`DockingConfig.pep_threshold` is per-protein; the differential pipeline's `DifferentialConfig.bfdr_threshold` is the global FDR analogue.

### D.2 Step 1 — generate request JSONs

```julia
batch = generate_docking_requests(
    results, HAP40_SEQUENCE;
    bait_name  = "HAP40",
    output_dir = docking_output,
    fasta_file = "",                    # leave empty to auto-fetch from UniProt
    config     = docking_config,
)

println("Generated $(batch.n_requests) requests in $(batch.n_batches) batches.")
println("Upload guide: $(batch.guide_path)")
```

### D.3 Step 2 — upload (manual)

1. Open <https://alphafoldserver.com>.
2. Upload each `.json` file from `docking_output/batch_1/`, `batch_2/`, …
3. Download all result ZIPs into a single directory, e.g. `docking_results/`.

There is no public API — this step is unavoidably manual. Plan ~30 jobs/day per Google account.

### D.4 Step 3 — parse + update posteriors

```julia
docking = import_docking_results("/path/to/docking_results", results; config = docking_config)

println("Docked: $(docking.n_docked) / $(docking.n_total)")
println("Pending: $(docking.n_pending)")
println("Disordered (BF=1): $(docking.n_disordered)")

# Two-stage Bayesian update: P_combined = odds_ms * BF_dock / (1 + odds_ms * BF_dock)
updated_results = apply_docking_update(results, docking)
```

Each scored pair receives a `BF_dock` from one of three tiers — ipTM step function (Tier 1), pDockQ logistic (Tier 2), or C2Qscore (Tier 2, preferred for AF3). Quality gates (high pAE, low pLDDT, high `iptm_std`, fraction disordered > 0.5) clamp or neutralize the BF before the update. See [Docking Integration](docking.md) for the full tier specification.

The output adds 18 new columns including `posterior_prob_combined`, `bf_docking`, `c2qscore`, `iptm_best`, `pdockq`, `BFDR_combined`, `calibration_tier`, and `docking_status`.

### D.5 Regenerate the report with docking data

```julia
sidecar = BayesInteractomics._sidecar_path(config.output.report_file)
generate_report(updated_results, config;
                docking_result = docking,
                sidecar_path   = sidecar)
```

This re-renders the HTML report with the **Structural Evidence** tab populated. The sidecar JSON merge preserves calibration, simulation, diagnostics, and sensitivity data from the original run.

Continue to [Reading Your Results](#reading-your-results).

---

## Reading Your Results

Independent of which branch you took, the results DataFrame columns and report tabs are the same.

### Key columns

| Column | Meaning |
|---|---|
| `Protein` | Protein identifier (curated name after STRING-based cleanup). |
| `BF_enrichment` | Bayes factor from the Hierarchical Bayesian Model (log2 fold change > 0). |
| `BF_correlation` | Bayes factor from the Bayesian regression (slope > threshold). |
| `BF_detection` | Bayes factor from the Beta-Bernoulli detection model. |
| `bf_em`, `bf_copula` | Sub-model BFs after BMA — transparency on which model contributed. |
| `Combined_BF` | Linearly pooled BF: `bf = w_em * bf_em + w_cop * bf_copula`. |
| `posterior_prob` | Posterior probability of genuine interaction (0–1). |
| `PEP` | Posterior Error Probability — `1 - posterior_prob`, per-protein. |
| `BFDR` | Bayesian False Discovery Rate (Storey monotone step-down) — global. |
| `local_fdr` | Local FDR at this protein's posterior. |
| `log2FC_mean`, `log2FC_median`, `log2FC_sd` | Posterior summary of enrichment. |
| `pd` | Probability of direction (% of posterior > 0). |
| `rope_percentage` | % of posterior in the region of practical equivalence (near zero). |
| `diagnostic_flag` | `:ok` / `:warning` / `:fail` from input QC + posterior predictive checks. |
| `sensitivity_range` | Posterior probability range across the prior sensitivity sweep. |
| `classification_stability` | `robust` / `sensitive` / `fragile`. |

### Decision thresholds

| Posterior probability | PEP | Interpretation |
|---|---|---|
| `> 0.95` | `< 0.05` | **Strong evidence** — high-confidence interaction. |
| `0.80 – 0.95` | `0.05 – 0.20` | **Moderate evidence** — promising candidate, validate experimentally. |
| `0.25 – 0.80` | `0.20 – 0.75` | **Ambiguous** — insufficient evidence either way. |
| `< 0.25` | `> 0.75` | **Strong evidence against** — likely non-specific or contaminant. |

These are guidelines, not bright lines. Always consider biological context and orthogonal validation (co-IP, co-localization, mutational analysis) for downstream claims.

### Report tabs

The HTML report at `config.output.report_file` opens in any browser — single file, no server needed. Nine tabs in order:

1. **Results** — sortable table of all proteins.
2. **Evidence** — volcano plot with PEP-band coloring (`grey > 0.05`, `amber ≤ 0.05`, `dark blue ≤ 0.01`, `red ≤ 0.001`).
3. **Calibration** — Platt scaling fit, ECE before/after, reliability diagram.
4. **Sensitivity** — tornado, heatmap, ternary prior, classification stability traffic light.
5. **Mixture Model** — fitted H0 / Agnostic / H1 components, per-protein responsibilities.
6. **Structural Evidence** — docking BF distribution, ipTM scatter, two-stage update plot (only populated when Branch D was run).
7. **Differential** — gained/lost/unchanged classification (only populated when `differential_analysis` was run).
8. **Data Quality** — five input QC checks with thresholds and per-experiment results.
9. **Methods** — auto-generated publication-ready methods text + reproducibility block (Project.toml versions, git SHA, RNG seed).

Full tab walkthrough: [Reports](reports.md).

## Where to Go Next

Each capability has its own dedicated docs page:

- **[Data Loading](data_loading.md)** — file formats, column specifications, hierarchical data structure.
- **[Data Curation](data_curation.md)** — STRING-based contaminant removal, group splitting, synonym resolution.
- **[Input Data Quality Control](data_quality_control.md)** — five-check QC system, what each warning/fail means.
- **[Analysis Pipeline](analysis.md)** — end-to-end orchestration, caching, parallelism.
- **[Model Fitting](model_fitting.md)** — Beta-Bernoulli / HBM / regression internals, JZS prior, robust regression.
- **[Model Evaluation](model_evaluation.md)** — Bayes factors, BMA section (linear BF pooling, LOO stacking), latent class details.
- **[Diagnostics](diagnostics.md)** — Storey BFDR, local FDR, classification stability traffic light, Pareto-k.
- **[Simulation & Calibration](simulation_calibration.md)** — parametric simulation engine, Platt scaling, ECE safety guard.
- **[Prior Sensitivity](prior_sensitivity.md)** — empirical Bayes Dirichlet, BIC grid marginalization, sensitivity sweep.
- **[Differential Analysis](differential_analysis.md)** — paired comparison, gained/lost classification.
- **[Visualization](visualization.md)** — volcano, evidence, ternary, sensitivity bands, within-class correlation.
- **[Reports](reports.md)** — full report tab tour, methods text, sidecar JSON.
- **[Docking Integration](docking.md)** — three scoring tiers, two-stage update, AlphaFold Server workflow.
- **[Network Analysis](network_analysis.md)** — graph construction, centrality, communities, Cytoscape export.
- **[Examples](examples.md)** — runnable scripts in `examples/`.

For statistical theory + citations: [Mathematical Background](mathematical_background.md).
