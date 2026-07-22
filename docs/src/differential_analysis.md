# Differential Interaction Analysis

## Overview

BayesInteractomics supports **differential interaction analysis** — comparing protein interaction profiles between two experimental conditions (e.g., wild-type vs. mutant, treated vs. untreated). This module identifies interactions that are gained, lost, or unchanged between conditions.

The differential API lives in the `BayesInteractomics.Differential` organisational submodule; all its public symbols (`differential_analysis`, `DifferentialConfig`, `DifferentialResult`, and the classification enums) are re-exported at the top-level `BayesInteractomics` namespace, so `using BayesInteractomics` makes them available unqualified.

## Quick Start

```julia
using BayesInteractomics

# Option 1: From two CONFIG objects (end-to-end pipeline)
config_wt = CONFIG(datafile=["wt.xlsx"], ...)
config_mut = CONFIG(datafile=["mut.xlsx"], ...)

diff = differential_analysis(config_wt, config_mut,
    condition_A = "WT",
    condition_B = "Mutant",
    config = DifferentialConfig(bfdr_threshold = 0.05)
)

# Option 2: From pre-computed AnalysisResult objects
_, result_wt = run_analysis(config_wt)
_, result_mut = run_analysis(config_mut)

diff = differential_analysis(result_wt, result_mut,
    condition_A = "WT",
    condition_B = "Mutant"
)
```

## Statistical Methodology

For each protein present in both conditions, the module computes:

### 1. Differential Bayes Factor (dBF)

```
dBF = BF_A / BF_B
```

A `log₁₀(dBF) > 0` means stronger evidence for interaction in condition A.

### 2. Per-Evidence Differential

The same ratio is computed separately for enrichment, correlation, and detection Bayes factors, allowing you to diagnose which evidence type drives the differential signal.

### 3. Effect Size

```
Δlog₂FC = mean_log₂FC_A − mean_log₂FC_B
```

### 4. Differential Posterior Probability

```
dBF_two-sided = 10^|log₁₀ dBF|
P(differential | data) = dBF_two-sided / (1 + dBF_two-sided)
```

This is a direction-agnostic measure of evidence for any difference. The **two-sided**
fold (`10^|log₁₀ dBF|`) is essential: a protein reduced in A has `dBF < 1`, so using the
raw `dBF` directly would collapse its posterior below 0.5 and it would never reach
significance — a direction bias that suppressed all `REDUCED` calls. The *direction*
(gained vs reduced) is resolved separately during classification from the sign of
`log₁₀ dBF` and `Δlog₂FC`.

### 5. Multiple Testing Correction

Bayesian FDR (BFDR) values are computed on the differential posterior probabilities. Per-protein posterior error probability (`PEP = 1 - posterior_prob`) and `local_fdr` are also reported.

### 6. Interaction Classification

Proteins are classified into categories based on the classification method:

| Class | Meaning |
|---|---|
| `GAINED` | Interaction present in A but not in B (or stronger in A) |
| `REDUCED` | Interaction present in B but not in A (or stronger in B) |
| `UNCHANGED` | Similar interaction strength in both conditions |
| `BOTH_NEGATIVE` | Neither condition shows strong interaction evidence |
| `CONDITION_A_SPECIFIC` | Protein only detected in condition A |
| `CONDITION_B_SPECIFIC` | Protein only detected in condition B |

## PEP — Posterior Error Probability

PEP is the per-protein complement of the posterior probability — the probability that a call is wrong **for this specific protein** (in contrast to `BFDR`, which is the expected global false-discovery rate across a set of calls). BayesInteractomics exposes PEP in two forms on a differential result, the direction-agnostic **α-PEP** and the class-conditional **γ-PEP** family. Both follow the `PEP` / `BFDR` / `local_fdr` terminology convention.

### α-PEP (direct complement)

`differential_pep = 1 - posterior_prob` is the **direct complement** of the differential posterior probability. It lives on `DifferentialResult.results` and answers the question "for THIS protein, what is the probability that the differential call is wrong (in any direction)?" This is the column the volcano plot's saturation channel reads (see [§Volcano colour contract](#volcano-colour-contract-7a) below).

When Platt calibration is available (the default when `run_simulation = true`), `differential_pep` is computed off the **calibrated** posterior. The `is_calibrated_A::Bool` and `is_calibrated_B::Bool` flags on `DifferentialResult` indicate per-condition calibration status — if either is `false`, PEP for that side is read off the un-calibrated posterior. (See also [Simulation & Calibration](simulation_calibration.md).)

### γ-PEP (class-conditional)

Four additional columns surface per-class PEP on `DifferentialResult.results`:

- `pep_gained` — `1 - P(class = GAINED | data)`
- `pep_reduced` — `1 - P(class = REDUCED | data)`
- `pep_unchanged` — `1 - P(class = UNCHANGED | data)`
- `pep_both_negative` — `1 - P(class = BOTH_NEGATIVE | data)`

These are **not** the same as α-PEP: they are class-specific, and the four `P(class | data)` values across the four classes sum to 1, so the four PEP columns themselves sum to `4 − 1 = 3`. A small value of `pep_gained` means strong evidence that the protein belongs to the `GAINED` class specifically (not that the differential call is right in any direction).

**Conditional-independence caveat.** The γ-PEP columns assume the four class posteriors are obtained from the **same** Bayesian fit. Reading them as marginal posteriors **across** BMA sub-models (e.g. mixing `pep_gained` from the Copula sub-model with `pep_reduced` from the 3c-EM sub-model) without re-pooling would over-state class separation. The package always reports γ-PEP off the **BMA-pooled** posterior, so users do not need to do this re-pooling themselves — but downstream code that splits sub-model BFs (`bf_em`, `bf_copula`) MUST NOT mix γ-PEPs across them.

### When to use α vs γ

| Question | Column to filter on | Example |
|---|---|---|
| "Is the differential call right?" (direction-agnostic) | `differential_pep` (α) | Ranking validation candidates by direction-agnostic error |
| "Is the protein **specifically** in this class?" (direction-aware) | `pep_<class>` (γ) | Filtering for proteins confidently in `GAINED` ONLY, with low `pep_gained` |

α-PEP is the right answer when downstream validation does not care about direction (e.g. "we will validate any protein that differs between conditions"). γ-PEP is the right answer when you want a confident assignment to a specific class (e.g. "we only validate confidently GAINED proteins to follow up on a known gain-of-function mechanism").

### Calibration coupling and the `is_calibrated` flag

PEP is computed off the **Platt-calibrated** posterior when calibration is active (the recommendation §7.3 default — see [Simulation & Calibration](simulation_calibration.md) for the ECE safety guard mechanics). `AnalysisResult.is_calibrated::Bool` flows through to `DifferentialResult.is_calibrated_A` and `DifferentialResult.is_calibrated_B`.

When `is_calibrated = false`, PEP is read off the **un-calibrated** posterior. This is NOT an error — it means the ECE safety guard declined to apply the calibration overlay (the un-calibrated posterior had a lower ECE than the Platt-fitted one, so applying it would have hurt calibration). The PEP column is still well-defined and usable; the flag exists for transparency, not as an error gate. The report HTML surfaces the `is_calibrated` status in the Methods tab so users can audit the per-condition calibration decision.

### Accessor functions

```julia
getPEP(ar)                                       # α-PEP for a single AnalysisResult
getDifferentialPEP(diff)                         # α-PEP (default; class = :alpha)
getDifferentialPEP(diff; class = :gained)        # γ-PEP for the GAINED class
getDifferentialPEP(diff; class = :reduced)       # γ-PEP for REDUCED
getDifferentialPEP(diff; class = :unchanged)     # γ-PEP for UNCHANGED
getDifferentialPEP(diff; class = :both_negative) # γ-PEP for BOTH_NEGATIVE
isCalibrated(ar)                                 # Bool — per-condition calibration flag
isCalibrated(diff)                               # (is_calibrated_A, is_calibrated_B)
getDifferentialBFDR(diff)                        # Recommended BFDR accessor
```

The legacy `getDifferentialQValues(diff)` accessor still exists for one-version backward compatibility but emits `@warn "getDifferentialQValues is deprecated; use getDifferentialBFDR"` on every call. Migrate downstream code to `getDifferentialBFDR`.

## Configuration

```julia
DifferentialConfig(
    # Significance thresholds
    bfdr_threshold = 0.05,              # Bayesian FDR threshold
    dbf_threshold = 1.0,                # |log10(dBF)| threshold
    delta_log2fc_threshold = 1.0,       # |Δlog2FC| threshold
    posterior_threshold = 0.8,          # Min posterior for "interactor"

    # Classification method
    classification_method = :posterior,  # :posterior, :dbf, or :combined

    # Output file paths
    results_file = "differential_results.xlsx",
    volcano_file = "differential_volcano.png",
    evidence_file = "differential_evidence.png",
    scatter_file = "differential_scatter.png",
    classification_file = "differential_classification.png",
    ma_file = "differential_ma.png",

    # HTML report
    generate_report_html = true
)
```

### Classification Methods

- **`:posterior`** (default): Uses per-condition posterior probabilities. A protein is an "interactor" if `posterior > posterior_threshold`. GAINED = interactor in A but not B with Δlog₂FC ≥ 0.
- **`:dbf`**: Uses `|log₁₀(dBF)| > dbf_threshold` directly.
- **`:combined`**: Both posterior and dBF criteria must be satisfied.

## Visualization

Five diagnostic plots are generated automatically:

### Volcano Plot

```julia
plt = differential_volcano_plot(diff)

# Customization
plt = differential_volcano_plot(diff,
    x_axis = :delta_log2fc,     # or :log10_dbf (default)
    y_axis = :differential_BFDR, # or :differential_posterior
    x_clip = 4.0                # Fixed x-axis range
)
```

### Evidence Plot

Four-panel plot showing per-evidence-type differential Bayes factors:

```julia
plt = differential_evidence_plot(diff)
```

### Scatter Plot

Compare a metric between conditions:

```julia
plt = differential_scatter_plot(diff, metric = :posterior_prob)
# Also: :bf, :log2fc
```

### Classification Plot

Bar chart summarizing the count of proteins in each class:

```julia
plt = differential_classification_plot(diff)
```

### MA Plot

Detect systematic biases where differential enrichment correlates with overall abundance:

```julia
plt = differential_ma_plot(diff)
```

## Accessing Results

```julia
# Filter by classification
gained = gained_interactions(diff)
lost = lost_interactions(diff)
unchanged = unchanged_interactions(diff)

# Significant differential interactions (any direction)
sig = significant_differential(diff, bfdr_threshold = 0.01)

# Export to Excel
export_differential(diff, "results.xlsx")

# Access full results DataFrame
diff.results   # All proteins with statistics
diff.config    # Configuration used
diff.n_gained  # Count summaries
```

## k-Group Analysis

`differential_analysis` generalises to arbitrary `k ≥ 2` conditions via a
keyword-only overload. The legacy 2-group positional call is unchanged.

### Basic usage

```julia
ar_wt   = run_analysis(config_wt).analysis_result
ar_mut1 = run_analysis(config_mut1).analysis_result
ar_mut2 = run_analysis(config_mut2).analysis_result

diff = differential_analysis(
    conditions = (wt = ar_wt, mut1 = ar_mut1, mut2 = ar_mut2),
)

# Iterate over the (k choose 2) = 3 pairwise contrasts.
for (pair, df) in diff.pairwise_results
    n_gained = count(==(GAINED), df.classification)
    println("Contrast $pair: n_gained = $n_gained")
end
```

### Selecting which contrasts to run

The `contrasts` keyword accepts three forms:

- `:all_pairs` (default) — all `(k choose 2)` pairs in NamedTuple key order
- `:vs_reference => :sym` — `(k - 1)` pairs `:sym => :other` (one-vs-reference)
- `Vector{Pair{Symbol, Symbol}}` — explicit user-specified list

```julia
# Default: all 3 pairs for k=3
differential_analysis(conditions = (wt = ar_wt, mut1 = ar_mut1, mut2 = ar_mut2))

# One-vs-reference: 2 pairs (wt vs mut1, wt vs mut2)
differential_analysis(conditions = (wt = ar_wt, mut1 = ar_mut1, mut2 = ar_mut2),
                      contrasts = :vs_reference => :wt)

# Explicit: only these 2 pairs
differential_analysis(conditions = (wt = ar_wt, mut1 = ar_mut1, mut2 = ar_mut2),
                      contrasts = [:wt => :mut1, :mut1 => :mut2])
```

### Cross-pair multi-test correction

The `multi_test_method` keyword applies a correction over the flattened
`(n_proteins × n_contrasts)` family of `differential_BFDR` values. The corrected
values land in `differential_BFDR_pairwise_BH` (or `_Bonferroni` / `_Holm` /
`_None`) on BOTH the wide `results::DataFrame` and each
`pairwise_results[pair]::DataFrame`.

| Method | Use when |
|--------|----------|
| `:bh` (default) | Standard genomics / proteomics correction; well-behaved across all k |
| `:bonferroni` | FWER control; conservative; suitable for confirmatory hypotheses |
| `:holm` | FWER step-down; less conservative than Bonferroni |
| `:none` | Already corrected externally; column populated with verbatim `differential_BFDR` |

For k=2 (the most common case), the BH correction degenerates to identity over a
single contrast — preserving byte-equality with the legacy 2-group call.

### Parallelism

The `parallel_pairs` keyword controls how the `(k choose 2)` pairwise calls are dispatched:

- `:auto` (default) — `Threads.@threads` when `k ≥ 3` AND `Threads.nthreads() ≥ 4`
- `true` — force parallel (with `@warn` when `nthreads < 2`)
- `false` — force serial (useful for deterministic test runs)

Per-pair failures are isolated via try/catch with `@warn maxlog=10`; failed pairs are
omitted from `pairwise_results` but the overall call continues.

### Per-pair sub-result access

`DifferentialResult` carries two additive fields populated by k-group calls:

- `contrasts::Vector{Pair{Symbol, Symbol}}` — the contrast list (empty for legacy 2-group)
- `pairwise_results::Union{Nothing, Dict{Pair{Symbol, Symbol}, DataFrame}}` — per-pair DataFrames in the existing 2-group shape (`nothing` for legacy 2-group)

```julia
# Access a specific pair's DataFrame
df_wt_vs_mut1 = diff.pairwise_results[:wt => :mut1]

# k-aware labels
condition_labels(diff)   # → ["wt", "mut1", "mut2"]
```

### Backward compatibility

The legacy 2-group call is bit-equivalent to a k=2 NamedTuple call:

```julia
d_legacy = differential_analysis(ar_wt, ar_mut1; condition_A = "wt", condition_B = "mut1")
d_kgroup = differential_analysis(conditions = (wt = ar_wt, mut1 = ar_mut1))

# d_kgroup.results equals d_legacy.results modulo the new
# differential_BFDR_pairwise_BH column (which for k=2 is identity over differential_BFDR).
```

See also the Laplace omnibus + multi-condition small-multiples report (next
section), which build on top of `pairwise_results` without further struct changes.
Per-pair BMA inherits the "Copula" + "3c-EM" terminology; FDR terminology stays
`BFDR` / `PEP` / `local_fdr`.

## k-Group omnibus

A **Laplace omnibus Bayes factor** layers on top of the pairwise infrastructure. The omnibus answers a different question from the pairwise BH family: "is the protein different across **all** k conditions?" rather than "is the protein different in any **specified pair**?" The two families are independent and are NOT cross-corrected against each other — see [Dual FDR families](#dual-fdr-families-when-to-use-which) below for the user-facing decision table.

No new struct fields are needed for the omnibus; all outputs land as additive columns on the wide `results::DataFrame` plus the multi-condition report tab.

### Omnibus Bayes factor

The omnibus is a closed-form Gaussian Bayes factor on the per-condition `(mean_log2FC, sd_log2FC)` posteriors, comparing two hypotheses:

- **M0** — all k conditions share one common mean log2FC
- **M1** — each condition has its own mean log2FC

The simplified heterogeneity-LR form (the form actually used in the current implementation) is:

```
log_BF_omnibus = 0.5 · Σ_c (μ_c − μ̂_shared)² / σ_c²
```

where `μ_c` and `σ_c` are the per-condition posterior mean and standard deviation of log2FC, and `μ̂_shared` is the precision-weighted shared mean. Two sanity contracts are locked:

- BF = 1 exactly when all per-condition means are identical (`μ_c ≡ μ̂_shared`), so log_BF = 0.
- BF → ∞ when the per-condition means are separated by ≥ 4σ on at least one pair.

### Empirical-Bayes pooled prior

A pooled prior `(μ_pool, τ²_pool)` is computed once per `differential_analysis(; conditions, ...)` call:

```julia
μ_pool   = median(finite μ across all proteins)
τ²_pool  = max((1.4826 · MAD)², 0.01)
```

This prior is exposed on the omnibus helper signature for Methods documentation, but it does **not** currently enter the test statistic — the simplified heterogeneity-LR formula above is used directly. The prior is reserved for a future v1.3.0 extension where the omnibus moves to a fully Bayes-factor-with-prior formulation.

### Output columns

Five omnibus columns appear on `DifferentialResult.results`:

| Column | Type | Description |
|---|---|---|
| `bf_omnibus` | `Float64` | k-way Bayes factor (M1 / M0) |
| `log10_bf_omnibus` | `Float64` | `log10(bf_omnibus)`, NaN-safe (retains pre-clamp value when `log_bf` is clamped to ±700) |
| `posterior_omnibus` | `Float64` | `bf / (1 + bf)` under a 0.5 / 0.5 model prior |
| `differential_BFDR_omnibus` | `Float64` | Storey monotone step-down BFDR within the omnibus family (independent of the pairwise BH family) |
| `differential_pep_omnibus` | `Float64` | `1 - posterior_omnibus` — α-PEP form for the omnibus call |

### Dual FDR families — when to use which

The cross-pair `differential_BFDR_pairwise_BH` and the `differential_BFDR_omnibus` answer different scientific questions and are NOT cross-corrected:

| Family | Question | Correction scope |
|---|---|---|
| `differential_BFDR_pairwise_BH` | "Is the protein different in **any** of the specified pairs?" | BH across `n_proteins × n_contrasts` |
| `differential_BFDR_omnibus` | "Is the protein different across **all k** conditions?" | Storey within `n_proteins` |

Filter on whichever family answers your downstream question. Concretely: use `_pairwise_BH` when you have a specific pair of interest (or a small set of pairs) and want to know which proteins differ in that pair; use `_omnibus` when you want a broad scan for any heterogeneity across all conditions (e.g. screening for proteins that respond differently to any of several treatments). Decision Risk reads `_omnibus` for its "Validation Candidates" pre-filter — see [decision_risk.md](decision_risk.md).

### k-aware classification columns

Three additional columns generalise the legacy 6-class `InteractionClass` enum to k ≥ 3 conditions:

| Column | Type | Description |
|---|---|---|
| `enriched_in` | `Vector{Symbol}` | conditions where `posterior_prob ≥ posterior_threshold` (default 0.8) |
| `depleted_in` | `Vector{Symbol}` | conditions where `posterior_prob < 1 − posterior_threshold` (default 0.2) |
| `kgroup_class` | `Symbol` | coarse 5-class enum (see below) |

The 5-class `kgroup_class` enum is:

| `kgroup_class` | Meaning |
|---|---|
| `:omnibus_null` | omnibus BFDR > threshold; insufficient evidence for any heterogeneity. Example: a housekeeping protein that is detected with high posterior in all conditions but at indistinguishable levels. |
| `:none_enriched` | omnibus significant but no individual condition crosses `posterior_threshold`. Example: a protein that is consistently borderline (`posterior_prob ≈ 0.5`) in all conditions, with significant heterogeneity in log2FC but no confident "interactor" call. |
| `:condition_specific` | `enriched_in` is a strict non-empty proper subset of conditions — the classical "specific to subset S" pattern. Example: a protein enriched in `{wt}` only, depleted or absent in `{mut1, mut2}` — the cleanest "loss-of-interaction in mutants" signal. |
| `:all_enriched` | every condition is enriched. Example: a stable scaffold protein that interacts with the bait in all variants. |
| `:fully_resolved` | some conditions enriched, some depleted; no abstentions. Example: a protein that is gained in `wt` and reduced in `mut1` — the strongest "differential" signal. |

The existing 6-class `classification` column (`GAINED / REDUCED / UNCHANGED / BOTH_NEGATIVE / CONDITION_A_SPECIFIC / CONDITION_B_SPECIFIC`) is **preserved verbatim** — `kgroup_class` is an **additional**, coarser column, not a replacement. The string-typed `classification_summary` column also remains alongside the new structured columns for backward compatibility.

### Multi-Condition report tab

The Multi-Condition tab is gated by **two** conditions:

```julia
length(diff.contrasts) >= 2 AND length(condition_labels(diff)) >= 3
```

That is, both the contrast count and the condition count must satisfy the threshold. A matrix-view dropdown widget appears when `length(diff.contrasts) >= 6` (k ≥ 4 with `contrasts = :all_pairs`); for k = 3 the tab inlines a small-multiples volcano grid directly. Per-pair volcano sub-plots inherit the [§7a colour contract](#volcano-colour-contract-7a) verbatim.

See also [reports.md](reports.md) for the full multi-condition report layout.

## Embeddings and similarity

Three layers of similarity / embedding analysis sit on top of the differential pipeline. All three are computed at the wide-DataFrame stage (post-BMA) and surface in the report HTML's Data Quality + Multi-Condition tabs.

### Sample-level PCA + UMAP + optional t-SNE

PCA on the log-intensity matrix (post-curation, post-imputation) is the **always-on** linear projection, coloured by condition / replicate / experiment / protocol. UMAP via UMAP.jl is the **default** non-linear embedding (`config.embeddings_config.method = :umap`, the `EmbeddingsConfig` default). t-SNE via TSne.jl is selected via `config.embeddings_config.method = :tsne` and shares the same report panel — there is no separate report tab.

**t-SNE caveat.** t-SNE preserves **local** neighbourhood structure but distorts inter-cluster distances; do NOT interpret t-SNE inter-cluster distances quantitatively. UMAP preserves local structure better and also gives a more interpretable global geometry; the default is UMAP for this reason.

### Protein-level UMAP

A per-protein UMAP is computed on the posterior-feature vector:

```julia
[log_BF_enrichment, log_BF_correlation, log_BF_detection, posterior_prob, log2FC]
```

The protein-level UMAP is coloured by:

- For differential analysis: `classification` (gained / reduced / unchanged / both_negative).
- For single-condition analysis: H0 / Agnostic / H1 (the 3c-EM latent class assignment).

**Supervised mode.** Pass `y = classification` (or any per-protein label vector) to UMAP.jl to obtain a **supervised** embedding that pulls same-class proteins closer together. The default is unsupervised.

### Condition-level similarity matrix

For k ≥ 2 conditions, a `(k × k)` condition similarity matrix is computed:

- **Primary metric.** Spearman correlation on `log₁₀(BF)` per protein across the k conditions. Spearman is rank-based and robust to outliers in the BF distribution.
- **Togglable views.** Pearson on log2FC and Pearson on `posterior_prob` are available as alternative views in the report HTML.
- **Secondary metric.** Jaccard@Top-50 — the Jaccard overlap of the top-50 proteins (ranked by BF) between two conditions.

A hierarchical-clustering dendrogram on `1 − ρ` (one-minus-Spearman as distance) is rendered alongside the similarity heatmap. The diagonal sanity-check (ρ ≡ 1 along the diagonal) is rendered explicitly to make any swapped-label bug visible at a glance.

### Where embeddings render in the report

| Embedding | Report tab |
|---|---|
| Sample-level PCA + UMAP / t-SNE | Data Quality (always) |
| Protein-level UMAP | Data Quality (always) |
| Condition-level similarity matrix + dendrogram | Data Quality (always) AND Multi-Condition (centerpiece, for k ≥ 3) |

See also [Data Quality Control](data_quality_control.md) for the surrounding QC checks.

The embedding method lives on `EmbeddingsConfig` (field `method::Symbol`, default `:umap`, one of `:umap`, `:tsne`, `:none`), which is carried by `CONFIG.embeddings_config::EmbeddingsConfig`. The runtime accessor is `config.embeddings_config.method`. To opt into t-SNE, construct an `EmbeddingsConfig` and pass it through `CONFIG`:

```julia
config = CONFIG(
    # ... usual fields ...
    embeddings_config = EmbeddingsConfig(method = :tsne),
)
```

Set `method = :none` to skip the non-linear embedding entirely (PCA remains always-on). See the [Configuration](configuration.md) page for the full `embeddings_config` entry.

## Volcano colour contract (§7a)

The volcano plot in the report HTML is the primary visual entry point to differential results. The §7a contract codifies its hue + saturation channels so that downstream tooling can read consistent colours across versions, plot variants, and the multi-condition small-multiples grid.

### Hue = classification

The `CLS_COLOR` palette (§7a):

| Class | Hue |
|---|---|
| `GAINED` | red |
| `REDUCED` | blue |
| `UNCHANGED` | mid-grey |
| `BOTH_NEGATIVE` | light-grey |

`CONDITION_A_SPECIFIC` and `CONDITION_B_SPECIFIC` are rendered with the same hues as `GAINED` and `REDUCED` respectively, with a marker-shape difference (the "_specific" classes get a triangle marker; the 4-class block gets a filled circle).

### Saturation = 1 − differential_pep

Saturation encodes confidence:

```
saturation = clamp(1.0 − differential_pep, 0.25, 1.0)
```

The mapping is **linear** and **clamped to [0.25, 1.0]**, so:

- `differential_pep = 0` → fully saturated (vivid colour, high-confidence call).
- `differential_pep = 0.75` (or anywhere above) → saturation floor of 0.25 (washed-out, low-confidence call).

The floor of 0.25 ensures that even low-confidence points remain visible against the white plot background — going to 0 would make them disappear entirely, hiding genuine outliers in the un-confident region.

### α-PEP vs γ-PEP toggle

A UI toggle in the report HTML swaps the saturation source:

- **α default** — saturation = `1 - differential_pep` (the direction-agnostic α-PEP).
- **γ override** — saturation = `1 - pep_<class>` of the protein's MAP class (the class-conditional γ-PEP for the class to which the protein is classified).

The hue channel is unchanged across the toggle — only the saturation source changes. This lets users see at a glance which proteins are confidently in their assigned class vs which are confidently differential-but-class-ambiguous.

### Marginal KDEs

The volcano includes top + right marginal KDEs (kernel density estimates) coloured by `classification`. Each class gets a continuous KDE curve on each margin (per-class, not stacked-by-PEP-bin — the stacked-by-PEP-bin variant was relaxed to continuous KDE per the CHANGELOG v1.2.0 deferred-to-v1.3 note). Both marginals surface the same hue × saturation contract as the main scatter.

### When to use the volcano vs the Validation Candidates pane

The volcano shows the **full landscape** — every protein, with its class, log10(dBF), Δlog2FC, and PEP all visible at a glance. It is the right view for hypothesis generation, QC, and understanding the structure of the differential signal.

For **downstream experimental prioritisation**, the Validation Candidates pane ranks proteins by expected loss under a 4 × 4 loss matrix — this answers "which protein should I validate first?" rather than "which proteins differ?" See [decision_risk.md](decision_risk.md) for the full Validation Candidates ranking workflow.

## Beta-Bernoulli ↔ MNAR diagnostic flag

The Beta-Bernoulli (BB) detection model and the post-MNAR HBM enrichment model can become **co-driven** by the same missingness pattern — both BFs fire on the same protein not because of two independent lines of evidence, but because heavy MNAR-style missingness in one condition both depresses the BB detection probability AND inflates the HBM enrichment estimate after MNAR imputation fills in low values. A diagnostic flag surfaces this co-driving so downstream consumers can audit it.

### Definition

```julia
bb_codriven = (BB_BF > cfg.bb_bf_threshold) ∧
              (post-MNAR HBM_BF > cfg.hbm_bf_threshold) ∧
              (missing_fraction > cfg.missing_fraction_threshold)
```

with defaults `10.0 / 10.0 / 0.5` via the `BBMnarCodrivenConfig` struct. All three conditions must fire — both BFs must be > 10, AND the missingness must exceed 50% — for the flag to fire.

### Per-side columns on differential output

The differential pipeline runs the diagnostic **per side** (one BB-vs-MNAR check for condition A, one for condition B):

- `bb_codriven_A::Bool` on `DifferentialResult.results`
- `bb_codriven_B::Bool` on `DifferentialResult.results`

Per-bait single-condition analyses (the non-differential `run_analysis` path) get a single `bb_mnar_codriven::Bool` column on `BayesResult`.

### Rendering

A warning icon ⚠ appears in the result tables next to flagged rows (bait + per-side on differential tables). The Methods tab of both `report.html` and `differential_report.html` carries an in-report explanation of the flag and its interpretation. The dataTables tooltip on the flag column reads:

> "BB and HBM both fire on this protein, AND missingness > 50% — the BMA posterior may be over-weighted by the same MNAR-driven signal. See Methods for context."

### Interpretation

When `bb_codriven_<side>` is `true`, the BB-derived detection probability and the HBM-derived enrichment estimate for that side are likely **co-driven by the same MNAR missingness pattern**:

1. Heavy missingness depresses the BB detection probability for the controls → `BB_BF` fires.
2. MNAR imputation fills in the missing control values with low-but-non-zero log-intensities (post-imputation σ²_imp accounts for the imputation uncertainty under the v2b mask-aware regression — see [imputation.md](imputation.md)).
3. The HBM enrichment compares samples vs the (now imputed-low) controls → `HBM_BF` fires.

The two BFs are not independent — they are both downstream of the same missingness pattern. The BMA posterior at the bait may therefore be **over-weighted** because it integrates "two" pieces of evidence that are actually one.

This is a flag, not an exclusion: the protein is **not** removed from the analysis. The flag is informational, intended to drive the user's audit of the high-confidence calls — for any flagged protein, the user should check the raw missingness pattern and decide whether the call is supported by independent evidence (e.g. detection in non-zero replicates, or correlation in dose-response). The Methods explanation in the report HTML walks through the v2 unified-dropout deferral context: a future v1.3.0 may replace the current per-side BB + post-MNAR HBM with a single unified dropout-aware model that resolves the co-driving structurally rather than flagging it after the fact.

### Configuration override

Users who want stricter thresholds (e.g. only flag proteins where both BFs are > 20 AND missingness > 60%) can override the defaults:

```julia
cfg = CONFIG(
    ...,
    bb_mnar_codriven_config = BBMnarCodrivenConfig(
        bb_bf_threshold          = 20.0,
        hbm_bf_threshold         = 20.0,
        missing_fraction_threshold = 0.6,
    ),
)
```

Setting any threshold to `Inf` effectively disables that arm of the conjunction (e.g. `missing_fraction_threshold = Inf` makes the flag never fire regardless of missingness). Setting all three to `Inf` disables the diagnostic entirely.

## API Reference

```@docs
differential_analysis
DifferentialConfig
DifferentialResult
InteractionClass
gained_interactions
lost_interactions
unchanged_interactions
significant_differential
export_differential
differential_volcano_plot
differential_evidence_plot
differential_scatter_plot
differential_classification_plot
differential_ma_plot
```
