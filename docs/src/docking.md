# Docking Integration

## Overview

BayesInteractomics integrates with the [AlphaFold Server](https://alphafoldserver.com) to refine protein-protein interaction posterior probabilities using structural predictions. This provides an independent, orthogonal source of evidence that complements the mass-spectrometry-derived Bayes factors. The docking update happens *after* the BMA evidence combination, so the input prior is the BMA posterior — see the BMA section in [Model Evaluation](@ref) for how `bf_em` and `bf_copula` are stacked into the MS posterior that feeds this stage.

The integration follows a **two-stage Bayesian update** framework: the posterior probability from the standard AP-MS analysis (enrichment, correlation, detection) serves as the prior for a second update using a docking-derived Bayes factor. This yields a combined posterior that incorporates both biochemical and structural evidence:

```math
P_{\text{final}} = \frac{\text{odds}_{\text{MS}} \times BF_{\text{dock}}}{1 + \text{odds}_{\text{MS}} \times BF_{\text{dock}}}
```

In code, the same identity reads `P_final = odds_ms * BF_dock / (1 + odds_ms * BF_dock)`, where `odds_ms = P_MS / (1 - P_MS)` is the prior odds derived from the MS posterior. The update is performed in log-odds space for numerical stability (`_bayesian_update_log_odds` in `src/docking/stubs.jl`). Because `log(odds)` diverges as the MS posterior approaches 0 or 1, BayesInteractomics clamps `posterior_prob` to `[ε, 1 - ε]` before taking odds. The epsilon is **data-derived**: it is set to one tenth of the smallest non-zero `BFDR` value in the results table (a fixed `1e-10` is too aggressive for highly significant baits where the lowest BFDR can reach `1e-300`). When no positive `BFDR` exists, the fallback is `1e-10`.

After the update, BayesInteractomics recomputes the global Bayesian FDR over the new combined posteriors and stores it in a `BFDR_combined` column.

Because the AlphaFold Server requires manual upload of prediction jobs, the workflow is **user-mediated** and proceeds in three distinct steps: request generation, server submission, and result parsing.

The docking API lives in the `BayesInteractomics.Docking` organisational submodule; its public symbols (`DockingConfig`, `DockingResult`, `apply_docking_update`, and the rest of the two-stage update API) are re-exported at the top-level `BayesInteractomics` namespace, so they are available unqualified after `using BayesInteractomics`.

## Quick Start

```julia
using BayesInteractomics

config = CONFIG(
    # ... standard analysis parameters ...
    run_docking = true,
    docking_config = DockingConfig(
        posterior_threshold = 0.8,
        pep_threshold = 0.01,        # max Posterior Error Probability (= 1 - posterior_prob)
        max_pairs = 50
    ),
    bait_sequence = "MSEQ...",   # Amino acid sequence of the bait protein
    bait_uniprot = "P12345"      # UniProt accession for sequence retrieval
)

# Run standard analysis, then docking is integrated automatically
final_df, result = run_analysis(config)
```

## Three-Step Workflow

### Step 1: Generate Docking Requests

The first phase creates JSON request files formatted for upload to the AlphaFold Server. Each file specifies one bait-prey protein pair for structure prediction.

```julia
# After running analysis, generate docking requests
batch = generate_docking_requests(
    results_df,
    bait_sequence;
    bait_name = "MYC",
    output_dir = "docking_requests",
    config = DockingConfig(
        posterior_threshold = 0.8,   # Only dock high-confidence candidates
        pep_threshold = 0.01,        # Max PEP (= 1 - posterior_prob)
        max_pairs = 50,              # Limit number of pairs
        max_tokens_per_job = 5000,   # AlphaFold Server token limit
        max_jobs_per_batch = 30      # Daily batch limit
    )
)
```

The function:

- Filters candidates by posterior probability and PEP thresholds (note: `DockingConfig` uses `pep_threshold` directly — the equivalent `bfdr_threshold` is the field used by `DifferentialConfig`; in docking we filter per-protein on PEP rather than on the global BFDR)
- Retrieves prey protein sequences from UniProt (or a local FASTA file)
- Skips pairs already present in the local cache
- Skips pairs that exceed the AlphaFold Server token limit (5000 residues)
- Organizes requests into daily batches of up to 30 jobs each
- Writes an `upload_guide.txt` with step-by-step submission instructions

The output `DockingRequestBatch` contains paths to the batch directories, a manifest file listing all pairs, and counts of submitted, cached, and skipped pairs.

### Step 2: User Uploads to AlphaFold Server

This is a **manual step**. The user:

1. Visits [alphafoldserver.com](https://alphafoldserver.com)
2. Uploads each JSON request file from the batch directories
3. Waits for predictions to complete (typical turnaround: minutes to hours depending on protein size)
4. Downloads the result ZIP files into a local directory

The `upload_guide.txt` generated in Step 1 provides detailed instructions for this process.

### Step 3: Parse Results and Update Posteriors

Once the AlphaFold Server results are downloaded, parse them and compute docking Bayes factors:

```julia
docking_result = import_docking_results(
    "path/to/downloaded/results/",   # Directory containing ZIP files
    results_df;
    config = DockingConfig(parse_full_data = true)
)
```

The parser:

- Scans the results directory for ZIP files and extracted directories
- Extracts `summary_confidences` JSON for ipTM and ranking scores
- Parses `full_data` JSON files for C2Qscore metrics (ipLDDT, iPAE, pTM, ipTM), pDockQ, and PAE
- Applies quality gates and computes the docking Bayes factor for each pair
- Caches per-pair results for future reuse

To apply the docking update to the results DataFrame:

```julia
updated_results = apply_docking_update(results_df, docking_result)
```

This adds columns `posterior_prob_ms` (original MS posterior), `bf_docking`, `posterior_prob_combined` (the two-stage updated posterior), and a recomputed `BFDR_combined` column. The full set of structural columns appended by `apply_docking_update` includes `iptm_best`, `iptm_std`, `pdockq`, `c2qscore`, `iplddt_interface`, `ipae`, `ranking_score_dock`, `fraction_disordered`, `chain_pair_iptm`, `chain_pair_pae_min`, `mean_plddt_a`, `mean_plddt_b`, `n_interface_contacts`, `calibration_tier`, and `docking_status`.

## Scoring Architecture

BayesInteractomics converts docking confidence scores into Bayes factors using a tiered scoring system. `compute_bf_dock` selects the preferred tier automatically based on which inputs are present:

1. C2Qscore (Tier 2, **preferred when `full_data` is parsed**)
2. pDockQ logistic (Tier 2, fallback for legacy / non-AF3 cached pairs)
3. ipTM step-function (Tier 1, fallback when `full_data` is unavailable)

Tier selection is recorded in the `calibration_tier` column of `DockingPairResult`. The implementations live in `src/docking/stubs.jl` (`compute_bf_from_c2qscore`, `compute_bf_from_pdockq`, `compute_bf_from_iptm`).

### Tier 1: ipTM step-function (fallback)

When `full_data` JSONs are unavailable (only `summary_confidences` parsed), the package falls back to a conservative step-function based on ipTM. The cutoffs and Bayes factors implemented in `compute_bf_from_iptm` are:

| ipTM Range   | Bayes Factor | Interpretation                    |
| ------------ | ------------ | --------------------------------- |
| < 0.20       | 1.0          | Below noise floor (no update)     |
| 0.20 -- 0.40 | 0.7          | Weak evidence against interaction |
| 0.40 -- 0.60 | 1.5          | Ambiguous                         |
| 0.60 -- 0.80 | 5.0          | Moderate confidence               |
| > 0.80       | 12.0         | High confidence                   |

Calibration tier label: `tier1_iptm`.

### Tier 2: pDockQ logistic (legacy)

When ipTM-only summaries are unavailable but pDockQ has been parsed (legacy results, or pre-AF3 model output), `compute_bf_from_pdockq` converts pDockQ to a Bayes factor with a logistic calibrated to the empirical Burke et al. 2023 anchor points (`pDockQ = 0.0 → P ≈ 0.003`, `pDockQ = 0.23 → P ≈ 0.70`, `pDockQ = 0.50 → P ≈ 0.80`):

```
P(correct | pDockQ) = 1 / (1 + exp(-(-4.56 + 12.8 * pDockQ)))
BF = P(correct)/(1 - P(correct)) / (base_rate / (1 - base_rate))
```

The base rate defaults to `0.15`. The resulting BF is clamped to `[0.1, 20.0]`.

pDockQ itself follows the published Bryant et al. 2022 sigmoid:

```
pDockQ = 0.707 / (1 + exp(-0.03148 * (avg_iface_pLDDT * log(n_iface_contacts) - 388.06))) + 0.03138
```

Calibration tier label: `tier2_pdockq`.

### Tier 2: C2Qscore (preferred, AF3)

When `full_data` JSONs are available from the AlphaFold Server (set `parse_full_data = true` in `DockingConfig`), the package computes C2Qscore (Olechnowicz et al. / Genz et al. 2025), a linear regression combining four structural quality metrics with AF3-specific coefficients (`compute_c2qscore` in `src/docking/stubs.jl`):

```
C2Qscore = -0.331 + (-0.036 * ipLDDT_norm) + (0.169 * iPAE_norm)
                  + (0.335 * pTM) + (0.683 * ipTM)
```

Where:

- **ipLDDT**: interface pLDDT (average pLDDT of interface residues), normalized as `ipLDDT_norm = ipLDDT / 100`
- **iPAE**: interface PAE (mean predicted aligned error over all cross-chain residue pairs), normalized as `iPAE_norm = 1 - raw_iPAE / 31.75`
- **pTM**: predicted TM-score
- **ipTM**: interface predicted TM-score (dominant contributor, weight 0.683)

C2Qscore is converted to a Bayes factor via logistic calibration fitted on the Genz et al. benchmark (1265 AF3 models, DockQ ≥ 0.23 threshold for "correct"):

```
P(correct) = 1 / (1 + exp(-(-2.3721 + 8.7486 * C2Qscore)))
BF = P(correct)/(1 - P(correct)) / (base_rate / (1 - base_rate))
```

**No clamping is applied** to the C2Qscore Bayes factor — the logistic calibration is sufficiently well-behaved on the AF3 benchmark that the standard `[0.1, 20.0]` clamp is redundant and would suppress genuine high-confidence calls. All other tiers retain the clamp.

Calibration tier label: `tier2_c2qscore`.

!!! note "iPAE computation"
    BayesInteractomics computes iPAE as the mean PAE over all cross-chain residue pairs from the `full_data` JSON `pae` matrix. This differs from the C2Qscore reference implementation which uses interface-only residue pairs (within 5 Å). The logistic calibration compensates for this difference.

### VoroIF integration (optional)

VoroIF (Voronoi tessellation Interface scoring; Olechnowicz et al.) was the fifth metric in the original C2Qscore paper. BayesInteractomics evaluated a 5-metric variant (`C2Qscore + VoroIF`) on the same Genz et al. AF3 benchmark and found that VoroIF *reduces* AUC-ROC from 0.9289 to 0.9185. The 4-metric variant is therefore the production scorer for AF3 predictions, and VoroIF integration is **not required** — the package ships without a VoroIF dependency. See the Benchmark Results table below.

## Benchmark Results

Evaluated on the Genz et al. 2025 AF3 benchmark dataset (1265 models, DockQ >= 0.23 threshold for correct predictions):

| Metric                          | AUC-ROC    | AUC-PR     |
| ------------------------------- | ---------- | ---------- |
| **C2Qscore 4-metric**           | **0.9289** | **0.9782** |
| C2Qscore 5-metric (with VoroIF) | 0.9185     | 0.9643     |
| ipTM alone                      | 0.9240     | 0.9743     |
| pDockQ2                         | 0.8270     | 0.9525     |

The 4-metric C2Qscore (without VoroIF) achieves the best discrimination. VoroIF provides no improvement and is not included.

Benchmark data available at:

- Data: `c2qscore/models_scores/benchmark_set/af3/`

## Quality Gates

Before the final Bayes factor is assigned, several quality gates are applied to ensure that low-quality predictions do not produce misleading updates. Gates are applied in order inside `compute_bf_dock`:

| Gate                | Condition                       | Action                  | Applies to                    |
| ------------------- | ------------------------------- | ----------------------- | ----------------------------- |
| **Disorder filter** | `fraction_disordered > 0.5`     | BF = 1.0 (no update)    | All tiers                     |
| **pLDDT gate**      | `mean_plddt_min < 50`           | Clamp BF to `[0.7, 1.5]`| **Tier 1 only**               |
| **PAE gate**        | `chain_pair_pae_min > 25`       | Clamp BF to `[0.5, 2.0]`| All tiers                     |
| **Model agreement** | `iptm_std > 0.15`               | Damp BF toward 1.0      | All tiers                     |
| **Final clamp**     | always                          | Clamp BF to `[0.1, 20.0]` | **All tiers except Tier 2 C2Qscore** |

**Disorder filter**: Proteins with more than 50% predicted disordered residues are unlikely to form stable interfaces. The docking BF is set to 1.0 (neutral), preserving the original MS-based posterior. The disorder threshold is configurable via `DockingConfig.disorder_threshold`.

**pLDDT gate (Tier 1 only)**: Applies only to the ipTM step-function fallback, where the raw BF has no structural quality awareness. When `mean_plddt_min < 50` across chain atoms, the BF is clamped to `[0.7, 1.5]`. This gate is **not** applied to Tier 2 scores — C2Qscore and pDockQ both incorporate interface pLDDT in their published logistic calibration, and an additional whole-chain pLDDT clamp would double-penalize large, partially disordered proteins that nevertheless have a confident docked interface (e.g., scaffold proteins with ordered interaction domains embedded in long IDRs).

**PAE gate**: High predicted aligned error (`chain_pair_pae_min > 25` Å) between chains suggests uncertain relative positioning. The BF is constrained to `[0.5, 2.0]`.

**Model agreement**: When multiple AlphaFold models disagree substantially (`iptm_std > 0.15`), the BF is damped toward 1.0 proportionally to the disagreement: `BF' = 1 + (BF - 1) * max(0, 1 - (iptm_std - 0.10) / 0.15)`.

**Final clamp**: After all conditional gates, the BF is clamped to `[0.1, 20.0]` for Tier 1 (ipTM) and Tier 2 pDockQ scoring. Tier 2 C2Qscore **skips this clamp** because the logistic calibration is well-calibrated on the AF3 benchmark and the clamp would suppress genuine high-confidence calls. The clamp range is configurable via `DockingCalibration.clamp_range`.

## Configuration

### DockingConfig Fields

| Field                    | Type      | Default              | Description                                           |
| ------------------------ | --------- | -------------------- | ----------------------------------------------------- |
| `posterior_threshold`    | `Float64` | `0.8`                | Minimum posterior probability for candidate selection |
| `pep_threshold`          | `Float64` | `0.01`               | Maximum Posterior Error Probability (`= 1 - posterior_prob`) for candidate selection |
| `max_pairs`              | `Int`     | `100`                | Maximum number of pairs to dock                       |
| `max_tokens_per_job`     | `Int`     | `5000`               | AlphaFold Server token limit per job                  |
| `max_jobs_per_batch`     | `Int`     | `30`                 | Maximum jobs per daily batch                          |
| `iptm_missing_threshold` | `Float64` | `0.20`               | Deprecated; ipTM below this is treated as no interaction |
| `disorder_threshold`     | `Float64` | `0.50`               | Fraction-disordered threshold for the disorder gate   |
| `parse_full_data`        | `Bool`    | `true`               | Parse `full_data` JSONs for C2Qscore (Tier 2) scoring |
| `cache_dir`              | `String`  | `""`                 | Custom cache directory (empty = default location)     |
| `save_raw_zips`          | `Bool`    | `false`              | Retain raw ZIP files after parsing                    |
| `request_output_dir`     | `String`  | `"docking_requests"` | Output directory for request files                    |
| `verbose`                | `Bool`    | `true`               | Enable informational logging                          |
| `species`                | `Int`     | `9606`               | NCBI taxonomy ID for UniProt sequence resolution      |
| `dockability_weight`     | `Float64` | `0.3`                | Weight for AlphaFold-DB structural dockability in the composite candidate score (0 = posterior only, 1 = dockability only) |

!!! note "Threshold field naming across configs"
    `DockingConfig` filters per-protein on `pep_threshold` (Posterior Error Probability) because it is selecting *individual* high-confidence candidates for structural validation. `DifferentialConfig` uses `bfdr_threshold` because it operates on *the global FDR* across the differential comparison.

### CONFIG Fields for Docking

| Field            | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `run_docking`    | Enable/disable docking integration in `run_analysis`            |
| `docking_config` | `DockingConfig` instance with docking parameters                |
| `bait_sequence`  | Amino acid sequence of the bait protein                         |
| `bait_uniprot`   | UniProt accession ID for the bait (used for sequence retrieval) |

## Dependencies

### C2Qscore

C2Qscore computation is built into BayesInteractomics -- no external installation is needed. The AF3 coefficients from Genz et al. 2025 are embedded in the package (`src/docking/c2qscore_calibration.json`).

Reference implementation: [C2Qscore GitLab](https://gitlab.com/topf-lab/c2qscore)

Paper: Genz et al. 2025, Protein Science, doi:10.1002/pro.70327

### VoroIF (evaluated, not required)

VoroIF (Voronoi tessellation Interface scoring; Olechnowicz et al.) was the fifth metric in the original C2Qscore paper. BayesInteractomics evaluated a 5-metric variant on the same Genz et al. AF3 benchmark and found that VoroIF *reduces* AUC-ROC from 0.9289 (4-metric) to 0.9185 (5-metric with VoroIF) — see Benchmark Results above. VoroIF is therefore not used by BayesInteractomics; the package ships **without** any VoroIF runtime dependency. AF3-quality predictions already saturate the structural-confidence signal at the four AF3-native metrics. Should a future analysis require VoroIF (for example, on non-AF3 docking sources where pTM/ipTM behave differently), it would need to be wired in as a separate optional integration.

## Caching

Docking results are cached per-pair as JLD2 files at:

```
.bayesinteractomics_cache/docking/{cache_key}.jld2
```

The cache key is derived from the canonical (order-independent) pair of protein identifiers using `docking_cache_key(id_a, id_b)`. This means that re-running analysis with the same protein pairs will skip the parsing step and load cached scores directly.

Cached entries store the full `DockingPairResult` including all quality metrics (C2Qscore, iPAE, ipLDDT, calibration tier), so quality gates can be re-evaluated with different thresholds without re-parsing the AlphaFold output. Old cache entries (before C2Qscore) are backward-compatible and load with NaN defaults for the new fields.

## Report Integration

When docking results are available, the interactive HTML report includes a dedicated **Docking** tab with:

- **ipTM vs Posterior scatter plot**: Shows the relationship between structural prediction confidence and the MS-derived posterior probability
- **Before/after update plot**: Compares the original MS posterior (`posterior_prob_ms`) with the docking-updated posterior (`posterior_prob_combined`) for each docked protein
- **Score distribution plots**: C2Qscore histogram (or pDockQ for legacy data), BF distribution, ipTM distribution, and posterior violin
- **Docking results table**: Sortable table with protein pairs, ipTM scores, C2Qscore, calibration tier, BF_dock, and quality gate status
- **CSV export**: Download docking results including C2Qscore and calibration tier columns
- **3D structure viewer**: Mol* viewer for inspecting predicted structures (requires .cif files)

## API Reference

### Scoring Functions

```@docs
compute_c2qscore
compute_bf_from_c2qscore
compute_bf_dock
compute_pdockq
compute_bf_from_pdockq
compute_bf_from_iptm
```

`compute_c2qscore(iplddt_norm, ipae_norm, ptm, iptm)` computes the 4-metric C2Qscore from normalized structural quality metrics using the Genz et al. 2025 AF3 coefficients.

`compute_bf_from_c2qscore(c2qscore; base_rate=0.15)` converts a C2Qscore value to a docking Bayes factor via the pre-fitted logistic calibration. The `base_rate` parameter sets the prior odds for the odds-ratio conversion. **No clamp** is applied.

`compute_bf_from_pdockq(pdockq; base_rate=0.15)` converts a pDockQ value to a docking Bayes factor via the Burke et al. 2023 logistic calibration. The result is clamped to `[0.1, 20.0]`.

`compute_bf_from_iptm(iptm)` is the Tier 1 fallback; returns the conservative step-function Bayes factor for an ipTM value (no quality gates applied at this layer).

`compute_bf_dock(iptm, fraction_disordered; pdockq=NaN, c2qscore=NaN, iptm_std=0.0, chain_pair_pae_min=NaN, mean_plddt_min=NaN, calibration=default_calibration())` is the main entry point that selects the appropriate scoring tier (C2Qscore > pDockQ > ipTM) based on which inputs are non-NaN and applies the quality gates documented above. Returns `(bf_dock, calibration_tier)`.

### Types

```@docs
DockingConfig
DockingCalibration
DockingPairResult
DockingResult
DockingRequestBatch
```

### Pipeline Functions

```@docs
apply_docking_update
default_calibration
docking_cache_key
generate_docking_requests
import_docking_results
```

`generate_docking_requests(results_df, bait_sequence; bait_name, output_dir, config)` filters candidates from a results DataFrame by `posterior_threshold` and `pep_threshold`, resolves prey sequences (UniProt or local FASTA), and writes batched AlphaFold Server JSON requests plus an `upload_guide.txt` for the user-mediated submission step.

`import_docking_results(results_dir, results_df; config)` scans a directory of downloaded AlphaFold Server result ZIPs / extracted folders, parses `summary_confidences` and (when `parse_full_data = true`) `full_data` JSONs, applies quality gates, computes per-pair Bayes factors, and persists each pair to the per-pair JLD2 cache. Returns a `DockingResult` ready to feed into `apply_docking_update`.

## See Also

- [Model Evaluation](@ref) — explains the BMA evidence combination that produces the MS posterior used as the prior for the two-stage docking update.
- [Reports](@ref) — the **Structural Evidence** tab visualises the docking outputs (ipTM-vs-posterior scatter, before/after update plot, C2Qscore histogram, pair table, Mol\* viewer).
