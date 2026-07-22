# Data Quality Control

Most analyses fail because of input data problems, not algorithm problems. The QC system catches the most common ones — wrong intensity scale, an outlier replicate, asymmetric missingness, distribution-shape anomalies, no separation between samples and controls — *before* you spend hours running an analysis on a bad dataset.

This page is written for **biologists at the bench**: it explains what each check looks at, what a `:warning` or `:fail` means in plain language, and what to do if you see one.

## What Gets Checked (and Why)

Five checks run automatically when `CONFIG.run_input_qc = true` (the default). Each returns its own diagnostic flag, and the worst flag across all five becomes the overall QC verdict.

### 1. Scale detection

**What it is.** A simple check on the maximum intensity value seen anywhere in your data.

**What it catches.** AP-MS analyses in this package expect **log2-transformed intensities**. If you accidentally feed in raw linear-scale intensities (typical MaxQuant output range: `1e5` to `1e9`), every downstream model will misbehave — fold changes will be enormous, posteriors will saturate, the BMA mixture will not converge.

**How it flags.** `:warning` if the max value exceeds 1000 (a heuristic for "this looks linear"). There is no `:fail` for scale detection — the check is a hint, not a hard stop.

### 2. Replicate correlation

**What it is.** Pairwise Spearman correlation between replicates within each experiment group (samples or controls), computed on shared non-missing proteins.

**What it catches.** A low pairwise correlation almost always signals a sample swap (one replicate is from a different condition), a technical failure (clogged column, dropped digestion), or a labelling mistake. Genuine biological replicates of an AP-MS experiment typically show pairwise Spearman correlations above 0.85.

**How it flags.** Based on the *minimum* pairwise correlation within a group:

| Minimum pairwise correlation | Flag |
|------------------------------|------|
| `>= 0.80` | `:ok` |
| `0.60 <= corr < 0.80` | `:warning` |
| `< 0.60` | `:fail` |

### 3. Missingness asymmetry

**What it is.** For each replicate, the fraction of proteins with missing intensity. The check reports the **maximum ratio** of any single replicate's missing fraction to the median missing fraction within the group.

**What it catches.** When one replicate has substantially more missing values than the others — for example, three controls with ~30% missingness and one with 80% missingness — that asymmetric replicate biases enrichment estimation. The HBM model assumes replicate-level missingness is random; large asymmetries break the assumption.

**How it flags.** Based on the per-replicate-vs-median ratio:

| Max ratio (worst replicate / median) | Flag |
|--------------------------------------|------|
| `<= 2x` | `:ok` |
| `2x < ratio <= 3x` | `:warning` |
| `> 3x` | `:fail` |

### 4. Intensity distribution shape

**What it is.** For each replicate, three shape statistics on the intensity distribution: excess kurtosis (heavy/light tails), skewness, and a "spike fraction" (fraction of values clustered at zero or the minimum).

**What it catches.**

- **Bimodality** (`excess_kurtosis < -1.2`) — two populations of intensities, e.g., a true protein population plus a contaminant population.
- **Spikes at zero/min** — substantial fraction of identical low values, typically a poorly-handled missing-value imputation step.
- **Heavy tails** — extreme outlier intensities that can dominate the regression posterior.

**How it flags.** Each replicate gets three sub-flags (`bimodality_flag`, `spike_flag`, `tail_flag`) and an overall flag that is the worst of the three.

### 5. PCA separation

**What it is.** Principal Component Analysis on a complete-case-cascading subset of proteins, scoring how well PC1 (and PC2) separates samples from controls using **Fisher's discriminant ratio**.

**What it catches.** If PC1 does not separate samples vs controls, your bait pulldown is essentially indistinguishable from your control pulldown — either the bait was not enriched, the control protocol was wrong, or the experimental conditions are too similar.

**Complete-case fallback ladder.** Because AP-MS data is often missing-rich, the PCA falls back through three thresholds: complete-case → 80% complete → 50% complete. The `fallback_level` field on `PCASeparationResult` records which threshold was actually used.

**How it flags.** `:ok` when Fisher's ratio on PC1 (and optionally PC2) suggests genuine separation; `:warning` otherwise. PCA never returns `:fail` directly; it is a soft hint.

## Diagnostic Flags

All checks share a three-level vocabulary:

| Flag | Meaning |
|------|---------|
| `:ok` | Pass — no concerns. |
| `:warning` | Review recommended — analysis can proceed but you should look at the underlying numbers. |
| `:fail` | Analysis may be unreliable — fix or document the issue before drawing conclusions. |

The `worst_flag(flags...)` helper aggregates a collection of flags using the order `:fail > :warning > :ok`. The overall QC verdict on `InputQCResult.overall_flag` is `worst_flag` over all sub-checks that ran successfully.

## API

```@docs
run_input_qc
worst_flag
InputQCResult
```

The most common interaction:

```julia
using BayesInteractomics

# Load your data first
data = load_data(["my_apms_data.xlsx"], sample_cols, control_cols;
                 normalise_protocols = true)

# Run QC standalone before launching the full analysis
qc = run_input_qc(data)

println(qc)                    # pretty summary line
println(qc.overall_flag)       # :ok, :warning, or :fail
println(qc.scale.flag)         # scale check flag
println(qc.replicate_correlation.flag)  # replicate correlation flag
println(qc.missingness.flag)   # missingness asymmetry flag
println(qc.intensity_shape.flag)        # distribution shape flag
println(qc.pca_separation.flag)         # PCA separation flag

# Drill into the worst replicate correlation pair
for chk in qc.replicate_correlation.checks
    if chk.flag != :ok
        println("Protocol $(chk.protocol_index), exp $(chk.experiment_index), ",
                "$(chk.group): min pairwise corr = $(chk.min_correlation)")
    end
end
```

Each `InputQCResult` field is `Union{Nothing, T}` — if a check fails internally (e.g., raises an exception), the corresponding field is `nothing` and `overall_flag` defaults to `:warning` rather than crashing the analysis. This is by design: QC failure should not prevent the analysis from running.

## When QC Runs

Two ways:

1. **Automatically** when `CONFIG.run_input_qc = true` (the default). The pipeline runs `run_input_qc(data)` after `load_data` and *before* the Beta-Bernoulli / HBM / regression models. The result is stored on the analysis output and surfaces in the **Data Quality** tab of the interactive HTML report (see [Reports](@ref)).

2. **Standalone** by calling `run_input_qc(data)` yourself before invoking `analyse` or `run_analysis`. This is the recommended pattern when you are vetting a new dataset for the first time, before committing to a full analysis run.

If `run_input_qc = false`, no checks are run, no flags are produced, and the Data Quality tab in the report is hidden.

## What to Do When a Check Fails

Practical guidance per check:

| Check | `:warning` / `:fail` | Action |
|-------|----------------------|--------|
| **Scale detection** (`:warning`) | Linear-scale intensities detected | Apply `log2(x + 1)` (or comparable) to your intensity columns and reload. The pipeline does *not* auto-transform; the responsibility is yours. |
| **Replicate correlation** (`:warning` / `:fail`) | Low pairwise Spearman correlation | Inspect the `correlation_matrix` field for the offending group. Identify the outlier replicate. Consider dropping it (re-run `load_data` with that column excluded), or document the issue if it must be kept. |
| **Missingness asymmetry** (`:warning` / `:fail`) | One replicate has much more missingness than the others | Either drop the offending replicate or apply multiple imputation upstream of `analyse` (the package supports `Vector{InteractionData}` for imputed datasets — see [Analysis Pipeline](@ref)). |
| **Intensity distribution shape** (`:warning`) | Bimodality, spikes, or heavy tails | Bimodality is often a contamination problem (CON__ entries leaking through curation); enable `curate_remove_contaminants = true`. Spikes near zero are usually leftover imputation artefacts. Heavy tails may be genuine — check the top-intensity proteins. |
| **PCA separation** (`:warning`) | PC1 does not separate samples from controls | Verify your sample/control column assignments are correct in the `sample_cols` / `control_cols` dictionaries. Check that the bait actually enriches in your samples. Consider adding more replicates if Fisher's ratio is borderline. |

A `:warning` is a hint — analysis will proceed and produce results. A `:fail` is a strong recommendation to fix the issue first; the resulting posteriors may be unreliable.

## See Also

- [Data Loading](@ref) — getting data into the package via `load_data`. QC runs *after* loading.
- [Data Curation](@ref) — cleaning protein IDs, removing contaminants, resolving synonyms via the STRING API. Curation runs *after* loading and *before* QC.
- [Reports](@ref) — the **Data Quality** tab in the interactive HTML report visualises every check's output (replicate correlation heatmap, missingness bar chart, intensity histograms, PCA scatter).
- [Analysis Pipeline](@ref) — `run_analysis` invokes `run_input_qc` automatically when `CONFIG.run_input_qc = true`.
