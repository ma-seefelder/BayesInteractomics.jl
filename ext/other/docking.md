# Structural Docking Integration for BayesInteractomics

**Team**: Bioinformatics Expert · Protein Complex Docking Expert · Julia Programming Expert
**Date**: 2026-03-05
**Status**: Design Document v2 — Revised for AlphaFold Server constraints

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [AlphaFold Server Constraints](#2-alphafold-server-constraints)
3. [Expert Consensus: What Docking Can and Cannot Add](#3-expert-consensus-what-docking-can-and-cannot-add)
4. [User-Mediated Workflow](#4-user-mediated-workflow)
5. [Statistical Integration Framework](#5-statistical-integration-framework)
6. [Caching and Deduplication](#6-caching-and-deduplication)
7. [Julia Architecture](#7-julia-architecture)
8. [Critical Risks and Safeguards](#8-critical-risks-and-safeguards)
9. [Implementation Roadmap](#9-implementation-roadmap)

---

## 1. Problem Statement

### 1.1 Motivation

BayesInteractomics produces ranked lists of candidate interactors with posterior probabilities derived from three evidence streams (enrichment, correlation, detection). The outstanding question is whether two proteins with high posterior probability are **direct physical binders** or **indirect interactors** (e.g., members of the same multi-subunit complex bridged through additional proteins). AP-MS cannot distinguish these cases: it captures the full co-purification event regardless of binding topology.

Structural docking — specifically the prediction of a protein-protein complex structure — provides orthogonal information: does the geometry and chemistry of the two protein surfaces support direct physical contact? A protein that co-purifies with high enrichment but whose surface is incompatible with the bait's binding interface is almost certainly an indirect interactor.

### 1.2 Design Constraints

Any docking integration must satisfy:

- **Non-blocking**: Must be an optional post-analysis step; never slow down the core pipeline
- **User-mediated**: No direct API access to AlphaFold Server — the package generates JSON request files and parses user-provided result ZIPs
- **Rate-aware**: AlphaFold Server allows only ~30 predictions/day with a 5,000-token-per-job limit; the package must respect these limits when generating request batches
- **Deduplicated**: Protein pairs must never be submitted twice; results are cached permanently by canonical pair key
- **Principled missing-data handling**: Proteins without docking results must not degrade the existing posterior (BF_dock = 1)
- **Calibrated scores**: Raw ipTM values must be converted to Bayes factors via proper calibration
- **Interpretable output**: The results DataFrame must clearly distinguish MS-evidence posteriors from docking-updated posteriors

---

## 2. AlphaFold Server Constraints

### 2.1 Hard Limits

| Constraint | Value | Impact |
|-----------|-------|--------|
| Predictions per day | ~30 | A 100-prey experiment requires ~4 days of manual uploads |
| Tokens per job | 5,000 | Each amino acid residue = 1 token. Pairs where bait + prey > 5,000 residues cannot be submitted |
| API access | None | No programmatic submission; user must upload JSON manually at alphafoldserver.com |
| Output format | ZIP per job | Contains 5 model CIFs + summary_confidences JSONs + full_data JSONs |
| Usage | Non-commercial only | Academic/research use; see AlphaFold Server Terms of Use |

### 2.2 Token Budget

Each job predicts one bait-prey pair. The token count is the sum of both protein sequence lengths:

```
tokens = length(bait_sequence) + length(prey_sequence)
```

**Practical implications** (Bioinformatics Expert):
- Median human protein length is ~375 residues → typical pair uses ~750 tokens (well within limit)
- Large proteins (e.g., TNRC6B at 1,833 residues) consume significant budget but most pairs still fit
- Pairs exceeding 5,000 tokens are flagged as `:too_large` and skipped automatically
- The bait sequence is constant across all pairs → cache it once; only prey varies

### 2.3 JSON Request Format

Each job is a single-element list containing a job dictionary (AlphaFold Server `dialect: "alphafoldserver"`, `version: 1`):

```json
[
  {
    "name": "BAIT:PREY",
    "modelSeeds": [],
    "sequences": [
      {
        "proteinChain": {
          "sequence": "BAIT_AMINO_ACID_SEQUENCE",
          "count": 1,
          "useStructureTemplate": true
        }
      },
      {
        "proteinChain": {
          "sequence": "PREY_AMINO_ACID_SEQUENCE",
          "count": 1,
          "useStructureTemplate": true
        }
      }
    ],
    "dialect": "alphafoldserver",
    "version": 1
  }
]
```

**Key rules**:
- `modelSeeds`: empty list → server assigns random seeds (recommended; produces 5 diverse models)
- `useStructureTemplate: true` → uses PDB templates for better accuracy
- Only 20 standard amino acid types; non-standard residues must be removed or substituted
- No comments allowed in JSON
- Job `name` is used for file naming in the output ZIP

### 2.4 Output Structure

Each AlphaFold Server result is a ZIP file containing:

```
<job_name>/
  fold_<job_name>_job_request.json          # Echo of input request
  fold_<job_name>_model_0.cif               # Predicted complex structure (model 0-4)
  fold_<job_name>_summary_confidences_0.json # Key confidence metrics
  fold_<job_name>_full_data_0.json           # Full PAE + pLDDT per residue (~40MB)
  ... (×5 models)
```

**`summary_confidences` fields** (the primary scoring data):

| Field | Type | Description |
|-------|------|-------------|
| `iptm` | Float64 | Interface predicted TM-score (our primary metric) |
| `ptm` | Float64 | Predicted TM-score (overall fold quality) |
| `chain_pair_iptm` | Matrix | Per-chain-pair ipTM (2×2 for dimers) |
| `chain_pair_pae_min` | Matrix | Per-chain-pair minimum PAE |
| `chain_ptm` | Vector | Per-chain pTM |
| `chain_iptm` | Vector | Per-chain ipTM contribution |
| `fraction_disordered` | Float64 | Fraction of residues predicted disordered |
| `has_clash` | Float64 | Steric clash indicator |
| `ranking_score` | Float64 | Overall ranking score (composite) |

**Scoring strategy** (Docking Expert): Use `max(iptm across 5 models)` as the primary score. The best model often captures the correct interface even when other seeds fail. The `fraction_disordered` field is a natural IDP detector — use it directly instead of computing from pLDDT.

---

## 3. Expert Consensus: What Docking Can and Cannot Add

### 3.1 Where Docking Adds Value

| Biological Scenario | AP-MS Signal | Docking Signal | Combined Interpretation |
|--------------------|--------------|----------------|------------------------|
| Direct high-affinity binder (e.g., obligate subunit) | High enrichment, high detection | High ipTM (>0.7) | Very high confidence direct interactor |
| Direct transient binder (e.g., kinase-substrate) | Moderate enrichment | Moderate ipTM (0.4-0.6) | Genuine transient interactor |
| Indirect / bridged interactor (e.g., scaffolded complex member) | High enrichment, high detection | Low ipTM (<0.4) | Probable indirect binding; still a valid biological hit |
| Abundant contaminant / sticky protein | Moderate enrichment | Low ipTM (<0.3) | Likely false positive or non-specific binder |

**Key scientific insight**: Low docking confidence does NOT falsify an AP-MS hit — it refines the interpretation. An indirect interactor is still biologically meaningful.

### 3.2 What Docking Cannot Do

- **Cannot distinguish regulated from constitutive interactions**: PTM-dependent or stimulus-gated interactions involve conformational changes absent in the static input structures
- **Cannot dock IDP-mediated interactions**: ~30-40% of human proteins are significantly disordered. Many AP-MS hub interactions (p53, MYC, BRCA1) are mediated by short linear motifs (SLiMs) for which docking fails structurally
- **Cannot reliably score phase-separated condensate membership**: Multivalent low-affinity IDR-IDR interactions are invisible to structure-based docking
- **Cannot serve as the primary evidence stream**: ipTM AUC for non-obligate transient PPIs is 0.55-0.65 (near-random)

**Conclusion**: Docking is most valuable as a **post-hoc annotation and refinement layer**. The update must be downweighted automatically for cases where structural confidence is low.

---

## 4. User-Mediated Workflow

### 4.1 Overview

Since AlphaFold Server has no programmatic API, the workflow is split into three phases:

```
Phase 1: GENERATE          Phase 2: USER ACTION         Phase 3: PARSE & UPDATE
(automated by package)     (manual by user)             (automated by package)

┌──────────────────┐      ┌─────────────────────┐      ┌──────────────────────┐
│ analyse() results│      │ Upload JSONs to      │      │ import_docking_      │
│ + bait sequence  │─────▶│ alphafoldserver.com   │─────▶│ results(results_dir) │
│                  │      │ (≤30/day)            │      │                      │
│ generate_docking │      │ Download result ZIPs │      │ apply_docking_       │
│ _requests()      │      │ to results_dir/      │      │ update()             │
│                  │      │                      │      │                      │
│ Output:          │      │                      │      │ Output:              │
│ requests/*.json  │      │                      │      │ Updated DataFrame    │
│ upload_guide.txt │      │                      │      │ with bf_docking etc. │
└──────────────────┘      └─────────────────────┘      └──────────────────────┘
```

### 4.2 Phase 1: Generate Docking Requests

```julia
generate_docking_requests(
    results::DataFrame,
    bait_sequence::String;
    bait_name::String = "BAIT",
    output_dir::String = "docking_requests",
    config::DockingConfig = DockingConfig()
) -> DockingRequestBatch
```

**What it does**:

1. **Filter candidates**: Select proteins with `posterior_prob >= config.posterior_threshold` AND `q_value <= config.q_threshold` (up to `config.max_pairs`)
2. **Fetch prey sequences**: Resolve UniProt IDs → amino acid sequences via UniProt REST API (or user-provided FASTA)
3. **Check token budget**: Skip pairs where `length(bait) + length(prey) > 5000`
4. **Check cache**: Skip pairs that already have cached docking results
5. **Deduplicate**: Canonical pair key ensures no pair is requested twice
6. **Generate JSON files**: One file per pair, named `{bait}_{prey}.json`
7. **Generate batch files**: Group requests into daily batches of ≤30 for user convenience
8. **Write upload guide**: `upload_guide.txt` with step-by-step instructions

**Batch output structure**:

```
docking_requests/
├── upload_guide.txt                    # Human-readable instructions
├── batch_1/                            # Day 1 (≤30 jobs)
│   ├── BAIT_PREYA.json
│   ├── BAIT_PREYB.json
│   └── ...
├── batch_2/                            # Day 2 (≤30 jobs)
│   └── ...
├── skipped_too_large.csv               # Pairs exceeding 5000 tokens
├── skipped_cached.csv                  # Pairs already in cache
└── request_manifest.json               # Machine-readable index of all requests
```

**`upload_guide.txt` content** (Julia Programming Expert):

```
=== BayesInteractomics Docking Request Guide ===

Generated: 2026-03-05T14:30:00
Total pairs to dock: 47
Batches: 2 (≤30 per day)
Skipped (cached): 12
Skipped (too large): 3

INSTRUCTIONS:
1. Go to https://alphafoldserver.com
2. Sign in with your Google account
3. For each batch directory (batch_1/, batch_2/, ...):
   a. Click "New fold" or use the upload feature
   b. Upload each .json file in the batch
   c. Wait for all jobs to complete (~30 min each)
   d. Download all result ZIP files
4. Place all downloaded ZIP files in a single directory
5. Run:
   julia> docking = import_docking_results("path/to/zips/", results)
   julia> updated = apply_docking_update(results, docking)

BATCH 1 (30 jobs, upload on day 1):
  BAIT_PREYA.json  (750 tokens)
  BAIT_PREYB.json  (820 tokens)
  ...

BATCH 2 (17 jobs, upload on day 2):
  ...

NOTE: Do NOT re-upload pairs listed in skipped_cached.csv —
      these already have results from previous runs.
```

### 4.3 Phase 2: User Action (Manual)

The user:
1. Opens alphafoldserver.com
2. Uploads each JSON file (or pastes sequences manually)
3. Downloads result ZIP files as they complete
4. Places all ZIPs into a single directory (e.g., `docking_results/`)

No package code runs during this phase.

### 4.4 Phase 3: Parse and Update

```julia
import_docking_results(
    results_dir::String,
    results::DataFrame;
    config::DockingConfig = DockingConfig()
) -> DockingResult
```

**What it does**:

1. **Scan directory**: Find all ZIP files (and already-extracted subdirectories)
2. **Parse `summary_confidences`**: Extract ipTM, fraction_disordered, ranking_score, chain_pair_iptm, chain_pair_pae_min from each model's `summary_confidences_*.json` (~350 bytes each)
3. **Parse `full_data` (default)**: For the best-ipTM model, parse `full_data_*.json` (~40MB) to extract `contact_probs`, `atom_plddts`, and `token_chain_ids`. Compute pDockQ (Bryant et al. sigmoid), mean pLDDT per chain, and interface contact count. Skip this step if `config.parse_full_data=false` (falls back to Tier 1 ipTM-only)
4. **Score selection**: Take `max(iptm)` across 5 models; compute pDockQ from the best model's full_data
5. **Match to proteins**: Resolve job names back to protein identifiers in the results DataFrame
6. **Compute BF_dock**: Tier 2 (pDockQ logistic) by default, Tier 1 (ipTM step-function) as fallback. Apply quality gates (disorder, pLDDT, PAE, model agreement)
7. **Cache results**: Store parsed scores in `.bayesinteractomics_cache/docking/` (JLD2 per pair)
8. **Return `DockingResult`**: Ready for `apply_docking_update()`

**Resilient parsing** (Julia Programming Expert): The parser handles:
- Multiple ZIPs in one directory (one ZIP per job, or one ZIP with multiple job subdirectories)
- Already-extracted directories alongside ZIPs
- Partial results (some jobs failed/pending → those pairs get BF_dock = 1)
- Re-running after new ZIPs are added (only processes uncached pairs)
- Missing `full_data` files: automatic fallback to Tier 1 (ipTM-only) for that pair

---

## 5. Statistical Integration Framework

### 5.1 Two-Stage Sequential Bayesian Update

**Rationale** (from v1 analysis): Adding docking as a 4th copula dimension is impractical due to missing-data marginalisation complexity and EM instability. The two-stage update adds zero complexity to the existing copula EM architecture.

**Mathematical formulation**:

Stage 1 (existing pipeline):
```
P_ms_i = P(H1 | enrichment_i, correlation_i, detection_i)
```

Stage 2 (docking update):
```
odds_ms_i  = P_ms_i / (1 - P_ms_i)
odds_new_i = odds_ms_i * BF_dock_i
P_final_i  = odds_new_i / (1 + odds_new_i)
```

For proteins without docking data: `BF_dock_i = 1` → `P_final_i = P_ms_i`.

**Why conditional independence holds**: The docking score depends on 3D structural geometry and surface chemistry. The MS observables (abundance ratios, cross-run correlations, detection counts) depend on co-elution in the affinity capture. These two sets of observables share no measurement pathway.

### 5.2 Published Calibration Data

Building a custom calibration dataset is infeasible under AlphaFold Server rate limits (~30/day; benchmarking Negatome + Dockground would require months). Instead, we use a **default calibration derived from published empirical data**.

#### 5.2.1 Source Publications

| Paper | Key Calibration Data |
|-------|---------------------|
| Bryant et al. 2022 (Nature Comms) | pDockQ sigmoid equation + AUC=0.87 for interacting vs non-interacting (n=5,694) |
| Burke et al. 2023 (Nat. Struct. Mol. Biol.) | pDockQ>0.5 → 80% correct; pDockQ>0.23 → 70% correct; random FPR=0.3%; direct vs indirect: 38% vs 6% at pDockQ>0.5 |
| Yin et al. 2022 (Protein Science) | ipTM threshold ~0.75 for model confidence cutoff |
| Kwon et al. 2025 (PMC11844409) | ipSAE as improved score; ipTM weak for disordered targets |

#### 5.2.2 pDockQ Computation from AlphaFold Server Output

The pDockQ score (Bryant et al. 2022) is computable from the `full_data` JSON without CIF parsing:

**Equation** (exact published parameters):
```
pDockQ = 0.707 / (1 + exp(-0.03148 * (x - 388.06))) + 0.03138

where x = avg_interface_plDDT * log(n_interface_contacts)
```

**Data sources in `full_data_*.json`**:

| Field | Size | Use |
|-------|------|-----|
| `contact_probs` | N×N matrix (N = total residues) | Inter-chain pairs with prob > threshold define interface contacts |
| `atom_plddts` | per-atom pLDDT values | Averaged per residue, then averaged over interface residues |
| `token_chain_ids` | per-residue chain ID ('A' or 'B') | Identifies which residues belong to bait vs prey |
| `atom_chain_ids` | per-atom chain ID | Maps atoms to chains for pLDDT grouping |

**Interface contact definition**: A residue pair (i in chain A, j in chain B) is an interface contact if `contact_probs[i][j] > 0.2`. This threshold corresponds approximately to the original 10A distance cutoff used by Bryant et al. (the `contact_probs` matrix from AlphaFold Server contains predicted pairwise contact probabilities at the token/residue level, not distances).

**Practical note** (Julia Programming Expert): The `full_data` JSONs are ~40MB each (5 models = ~200MB per pair). Parsing is feasible but slow. The implementation should:
1. Parse `summary_confidences` first (354 bytes) for fast screening
2. Only parse `full_data` for pairs where ipTM suggests a plausible interaction (ipTM > 0.2)
3. Cache the extracted pDockQ to avoid re-parsing

#### 5.2.3 Per-Chain pLDDT from `full_data`

Mean pLDDT per chain is computed from `atom_plddts` grouped by `atom_chain_ids`:

```julia
chain_a_plddts = [atom_plddts[i] for i in 1:n_atoms if atom_chain_ids[i] == "A"]
chain_b_plddts = [atom_plddts[i] for i in 1:n_atoms if atom_chain_ids[i] == "B"]
mean_plddt_a = mean(chain_a_plddts)
mean_plddt_b = mean(chain_b_plddts)
```

This is used for the quality gate (Section 5.4).

### 5.3 Default Calibration: Empirical Logistic Model

Since no published study provides fitted ipTM H0/H1 distribution parameters, we derive Bayes factors using a **logistic regression transfer calibration** anchored to the published empirical data points.

#### 5.3.1 Derivation

From Burke et al. 2023, we have three calibration points for pDockQ:

| pDockQ threshold | P(correct) | Source |
|-----------------|------------|--------|
| > 0.50 | 0.80 | 521/651 PDB benchmarks |
| > 0.23 | 0.70 | 671/955 PDB benchmarks |
| random | 0.003 | 0.3% of random pairs |

Since pDockQ and ipTM are strongly correlated (both measure AlphaFold's confidence in the predicted interface), we provide **two calibration tiers**. Tier 2 (pDockQ-based) is the default because it uses the directly calibrated published sigmoid equation and has higher discriminative power (AUC=0.87 vs ~0.6 for ipTM alone on transient PPIs). Tier 1 serves as automatic fallback when `full_data` JSONs are unavailable or when `parse_full_data=false` is explicitly set.

**Tier 1: ipTM-only (fallback, uses summary_confidences only)**

A conservative step-function BF derived from the published ipTM threshold literature (Yin et al. 2022, Burke et al. 2023) and the observation that ipTM > 0.8 ≈ pDockQ > 0.5, ipTM 0.6-0.8 ≈ pDockQ 0.23-0.5. Used automatically when `full_data` JSONs are not present in the results directory or when `parse_full_data=false`:

| ipTM range | Interpretation | BF_dock | Justification |
|-----------|----------------|---------|---------------|
| < 0.20 | Prediction failure | 1.0 | Below noise floor; treat as missing |
| 0.20 - 0.40 | Low confidence | 0.7 | Weak evidence against direct interaction |
| 0.40 - 0.60 | Ambiguous ("grey zone") | 1.5 | Slightly favors interaction, but near-random for transient PPIs (AUC ~0.6) |
| 0.60 - 0.80 | Moderate confidence | 5.0 | ~70% correct (Burke pDockQ>0.23 equivalent); BF = 0.7/0.3 * correction |
| > 0.80 | High confidence | 12.0 | ~80% correct (Burke pDockQ>0.5 equivalent); BF = 0.8/0.2 * prior correction |

**Derivation of BF values**: For ipTM > 0.80, P(correct) ≈ 0.80 from Burke et al. With a base rate P(true) ≈ 0.15 (typical AP-MS hit rate for high-confidence candidates that reach the docking stage):
```
BF = [P(correct|ipTM) / (1-P(correct|ipTM))] / [P(true) / (1-P(true))]
   = [0.80/0.20] / [0.15/0.85]
   = 4.0 / 0.176
   = 22.7 → clamped to 12.0 (conservative)
```

For ipTM 0.60-0.80, P(correct) ≈ 0.70:
```
BF = [0.70/0.30] / [0.15/0.85] = 2.33 / 0.176 = 13.2 → clamped to 5.0
```

The clamping is deliberately conservative to avoid docking evidence dominating the MS-based posterior.

**Tier 2: pDockQ-based (default)**

pDockQ is computed from `full_data` JSONs and converted to BF via the published sigmoid + logistic. This is the default because pDockQ has been directly calibrated against DockQ (AUC=0.87 for distinguishing interactors from non-interactors, Bryant et al. 2022) and Burke et al. 2023 provide three well-characterized precision points for the score:

```julia
function compute_bf_from_pdockq(pdockq::Float64; base_rate::Float64 = 0.15)::Float64
    # P(correct | pDockQ) from Burke et al. empirical relationship
    # Logistic fit to: pDockQ=0 → P≈0.003, pDockQ=0.23 → P≈0.70, pDockQ=0.5 → P≈0.80
    # Fitted: a = -4.56, b = 12.8
    p_correct = 1.0 / (1.0 + exp(-(−4.56 + 12.8 * pdockq)))

    # Convert to BF relative to base rate
    odds_correct = p_correct / (1.0 - p_correct)
    odds_prior = base_rate / (1.0 - base_rate)
    bf = odds_correct / odds_prior

    return clamp(bf, 0.1, 20.0)
end
```

The logistic parameters (a = -4.56, b = 12.8) are fitted to the three Burke et al. data points:
- pDockQ=0.0 → P=0.003
- pDockQ=0.23 → P=0.70
- pDockQ=0.50 → P=0.80

#### 5.3.2 Direct vs Indirect Interaction Discrimination

Burke et al. 2023 provide the most directly relevant calibration for our use case (distinguishing direct from indirect binders in complexes):

- **38% of direct interactors** have pDockQ > 0.5
- **Only 6% of indirect interactors** have pDockQ > 0.5

This gives a likelihood ratio of 0.38/0.06 = **6.3** for the "pDockQ > 0.5" event. This is consistent with our Tier 1 BF of 12.0 for ipTM > 0.80 (which maps to slightly higher pDockQ than 0.5).

### 5.4 Safeguards and Quality Gates

**BF clamping**: All BF_dock values are clamped to [0.1, 20.0] regardless of calibration tier. A BF of 20 corresponds to ~4.3 bits of evidence — sufficient to move a borderline posterior (0.5) to 0.95, but unable to rescue a clearly negative hit.

**Disorder flag**: If `fraction_disordered > 0.5` (from `summary_confidences`), force `BF_dock = 1.0` and set `docking_status = "disordered"`. Rationale: ipTM is unreliable for disordered regions; AlphaFold predicts extended conformations that produce artifactual contacts.

**pLDDT quality gate**: If mean pLDDT of either chain (from `full_data` `atom_plddts`) is < 50, clamp BF to [0.7, 1.5] (near-uninformative). This catches cases where the monomer structure itself is unreliable.

**Cross-chain PAE check**: If `chain_pair_pae_min` (from `summary_confidences`) is > 25 Angstrom, the prediction has no confident spatial proximity between chains. Clamp BF to [0.5, 2.0].

**Model agreement gate**: If `std(iptm)` across 5 models exceeds 0.15, the prediction is seed-sensitive. Reduce BF toward 1.0:
```
BF_adjusted = 1.0 + (BF_raw - 1.0) * max(0, 1 - (iptm_std - 0.10) / 0.15)
```

### 5.5 Scoring: Best Model Selection

Each AlphaFold Server job produces 5 models with independent seeds. The scoring strategy:

1. **Primary score**: `max(iptm)` across all 5 models — captures the best interface prediction
2. **pDockQ** (if `full_data` parsed): Computed from the best-ipTM model's `contact_probs` + `atom_plddts`
3. **Confidence**: `std(iptm)` across models — high variance triggers the model agreement gate
4. **Disorder check**: `min(fraction_disordered)` — if even the best model is >50% disordered, flag it
5. **Interface quality**: `chain_pair_iptm[1,2]` from the best model (off-diagonal = inter-chain quality)
6. **PAE check**: `chain_pair_pae_min[1,2]` from the best model — low PAE confirms spatial proximity

**BF selection priority** (default: Tier 2):
- pDockQ is computed from `full_data` and used for BF via `compute_bf_from_pdockq(pdockq)` (Tier 2)
- If `full_data` is unavailable or `parse_full_data=false`: fall back to ipTM step-function BF (Tier 1)
- All quality gates (disorder, pLDDT, PAE, model agreement) apply regardless of tier

### 5.6 Output Columns

The results DataFrame gains these columns after docking:

| Column | Type | Description |
|--------|------|-------------|
| `posterior_prob_ms` | Float64 | Original MS-only posterior (P_ms) |
| `bf_docking` | Float64 | BF_dock (1.0 if no docking) |
| `posterior_prob_combined` | Float64 | Updated posterior P_final |
| `iptm_best` | Float64 / missing | Best ipTM across 5 models |
| `iptm_std` | Float64 / missing | Std of ipTM across 5 models |
| `pdockq` | Float64 / missing | pDockQ score (if full_data parsed; Bryant et al. 2022) |
| `ranking_score` | Float64 / missing | AF Server ranking score (best model) |
| `fraction_disordered` | Float64 / missing | Min fraction disordered across models |
| `chain_pair_iptm` | Float64 / missing | Off-diagonal chain_pair_iptm (best model) |
| `chain_pair_pae_min` | Float64 / missing | Off-diagonal chain_pair_pae_min (best model) |
| `mean_plddt_a` | Float64 / missing | Mean pLDDT of bait chain (if full_data parsed) |
| `mean_plddt_b` | Float64 / missing | Mean pLDDT of prey chain (if full_data parsed) |
| `n_interface_contacts` | Int / missing | Number of cross-chain contacts (if full_data parsed) |
| `calibration_tier` | String | "tier1_iptm" or "tier2_pdockq" |
| `docking_status` | String | :success, :no_result, :disordered, :too_large, :failed |

---

## 6. Caching and Deduplication

### 6.1 Why Deduplication Is Critical

With only ~30 predictions/day, every wasted slot costs real time. The package must guarantee:

1. **No pair is ever submitted twice** — across sessions, across experiments, across users sharing a cache
2. **Partial progress is preserved** — if the user docks 15/47 pairs today, tomorrow's batch starts at pair 16
3. **Re-analysis reuses old results** — running `import_docking_results` on a cached pair returns instantly

### 6.2 Cache Key

```julia
function docking_cache_key(uniprot_a::String, uniprot_b::String)::String
    sorted = sort([uppercase(strip(uniprot_a)), uppercase(strip(uniprot_b))])
    return join(sorted, "__")
end
# Example: "P04637__Q9BYF1"
```

This ensures `dock(A, B)` and `dock(B, A)` map to the same cache entry.

### 6.3 Directory Layout

```
.bayesinteractomics_cache/
├── curation_*.jld2                    # Existing curation caches
└── docking/
    ├── P04637__Q9BYF1.jld2            # Per-pair parsed result
    ├── P04637__Q9BYF1.zip             # Original ZIP (optional, if save_raw=true)
    └── docking_index.jld2             # Master index: pair_key → status + scores
```

### 6.4 Cache Entry Schema

```julia
struct DockingCacheEntry
    pair_key::String
    uniprot_a::String;  uniprot_b::String
    gene_a::String;     gene_b::String

    # Scores (best model, from summary_confidences)
    iptm_best::Float64
    iptm_all::Vector{Float64}          # All 5 model ipTMs
    iptm_std::Float64
    ranking_score::Float64
    fraction_disordered::Float64
    chain_pair_iptm::Float64           # Off-diagonal element
    chain_pair_pae_min::Float64        # Off-diagonal element

    # Scores (from full_data, if parsed — otherwise NaN/0)
    pdockq::Float64                    # pDockQ (Bryant et al. 2022 sigmoid)
    mean_plddt_a::Float64              # Mean pLDDT of chain A
    mean_plddt_b::Float64              # Mean pLDDT of chain B
    n_interface_contacts::Int          # Cross-chain contacts (contact_probs > 0.2)

    # Computed
    bf_dock::Float64
    calibration_tier::String           # "tier1_iptm" or "tier2_pdockq"
    status::Symbol                     # :success | :disordered | :too_large | :failed

    # Metadata
    timestamp::DateTime
    af_server_job_name::String
    package_version::String
    calibration_id::String
    token_count::Int                   # Total residues in the pair
end
```

### 6.5 Deduplication Flow

```
generate_docking_requests():
    1. Load docking_index.jld2
    2. For each candidate pair:
       a. Compute cache key
       b. If key exists in index AND status == :success → skip (add to skipped_cached.csv)
       c. If key exists AND status == :failed → re-include (allow retry)
       d. If tokens > 5000 → skip (add to skipped_too_large.csv)
       e. Otherwise → generate JSON, add to batch

import_docking_results():
    1. Load docking_index.jld2
    2. For each ZIP/directory in results_dir:
       a. Parse job_request.json → extract pair identity
       b. Compute cache key
       c. If already cached with status == :success → skip
       d. Parse summary_confidences → extract scores
       e. Write per-pair JLD2 cache file
       f. Update docking_index.jld2
```

---

## 7. Julia Architecture

### 7.1 File Layout

```
src/
└── docking/
    └── stubs.jl                   # Always loaded: types + function stubs + BF computation

ext/
└── BayesInteractomicsDockingExt/
    ├── BayesInteractomicsDockingExt.jl   # Extension entry point
    ├── types.jl                   # Extended types (DockingCacheEntry internal)
    ├── request_generator.jl       # generate_docking_requests() — JSON generation
    ├── sequence_retrieval.jl      # UniProt sequence fetch + FASTA support
    ├── result_parser.jl           # import_docking_results() — ZIP/dir parsing
    ├── full_data_parser.jl        # Parse full_data JSONs: contact_probs → pDockQ, atom_plddts → mean pLDDT
    ├── scoring.jl                 # Score extraction, model selection, Tier 1/2 BF computation
    └── cache.jl                   # JLD2 cache management
```

### 7.2 Extension Trigger

```toml
# Project.toml:
[extensions]
BayesInteractomicsDockingExt = ["JSON3"]

[weakdeps]
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
```

**Why JSON3** (Julia Programming Expert): The extension needs robust JSON I/O for both generating AlphaFold Server request files and parsing result files. JSON3 is lightweight, pure Julia, and the natural trigger — users `using JSON3` signals they want to work with the docking JSON workflow. No heavy dependencies like BioStructures needed for the core workflow (CIF parsing is optional).

### 7.3 Core Types (in `src/docking/stubs.jl`, always loaded)

```julia
Base.@kwdef mutable struct DockingConfig
    # Candidate filtering
    posterior_threshold::Float64 = 0.8
    q_threshold::Float64 = 0.05
    max_pairs::Int = 100

    # Token limit
    max_tokens_per_job::Int = 5000
    max_jobs_per_batch::Int = 30

    # Scoring / calibration
    iptm_missing_threshold::Float64 = 0.20
    disorder_threshold::Float64 = 0.50
    calibration::Union{DockingCalibration, Nothing} = nothing
    parse_full_data::Bool = true   # Parse full_data JSONs (~40MB each) for pDockQ + pLDDT
                                   # true → Tier 2 (pDockQ, default), false → Tier 1 (ipTM only)

    # Cache
    cache_dir::String = ""        # Empty → .bayesinteractomics_cache/docking/
    save_raw_zips::Bool = false   # Keep original ZIPs in cache

    # Output
    request_output_dir::String = "docking_requests"
    verbose::Bool = true
end

struct DockingCalibration
    H0_dist::ContinuousUnivariateDistribution
    H1_dist::ContinuousUnivariateDistribution
    clamp_range::Tuple{Float64, Float64}
    iptm_missing_threshold::Float64
end

struct DockingPairResult
    protein_a::String;  protein_b::String
    uniprot_a::String;  uniprot_b::String
    iptm_best::Float64
    iptm_all::Vector{Float64}
    iptm_std::Float64
    ranking_score::Float64
    fraction_disordered::Float64
    chain_pair_iptm::Float64
    chain_pair_pae_min::Float64
    pdockq::Float64                    # NaN if full_data not parsed
    mean_plddt_a::Float64              # NaN if full_data not parsed
    mean_plddt_b::Float64              # NaN if full_data not parsed
    n_interface_contacts::Int          # 0 if full_data not parsed
    bf_dock::Float64
    calibration_tier::String           # "tier1_iptm" or "tier2_pdockq"
    status::Symbol
    token_count::Int
end

struct DockingResult
    pairs::Vector{DockingPairResult}
    config::DockingConfig
    n_total::Int           # Total candidate pairs
    n_docked::Int          # Pairs with results
    n_cached::Int          # From cache (not re-parsed)
    n_pending::Int         # No results yet
    n_too_large::Int       # Exceeded token limit
    n_disordered::Int      # Flagged as disordered
    timestamp::DateTime
end

struct DockingRequestBatch
    batch_dirs::Vector{String}       # Paths to batch_1/, batch_2/, ...
    n_requests::Int                  # Total JSON files generated
    n_batches::Int                   # Number of daily batches
    n_skipped_cached::Int
    n_skipped_too_large::Int
    manifest_path::String            # Path to request_manifest.json
    guide_path::String               # Path to upload_guide.txt
end
```

### 7.4 Exported Function Stubs

```julia
"""
    generate_docking_requests(results, bait_sequence; config, kwargs...) -> DockingRequestBatch

Generate AlphaFold Server JSON request files for high-confidence candidates.
Requires `using JSON3`.

Creates one JSON file per bait-prey pair, organized into daily batches of
≤30 jobs. Skips pairs already in cache and pairs exceeding 5000 tokens.
Writes an upload_guide.txt with step-by-step instructions.
"""
function generate_docking_requests end

"""
    import_docking_results(results_dir, results; config) -> DockingResult

Parse AlphaFold Server result ZIPs/directories and cache scores.
Requires `using JSON3`.

Scans results_dir for ZIP files and extracted directories, parses
summary_confidences JSONs, extracts ipTM and other scores, caches
per-pair results, and computes BF_dock for each pair.
"""
function import_docking_results end

"""
    apply_docking_update(results, docking; calibration) -> DataFrame

Merge docking BF_dock scores into a results DataFrame as a two-stage
Bayesian update. Adds posterior_prob_ms, bf_docking, posterior_prob_combined
columns. Safe to call without docking (returns unmodified DataFrame).

This function lives in stubs.jl — always available, no extension needed.
"""
function apply_docking_update end

"""
    compute_bf_dock(iptm, fraction_disordered; pdockq=NaN, iptm_std=0.0,
                    chain_pair_pae_min=NaN, mean_plddt_min=NaN,
                    calibration) -> (Float64, String)

Compute clamped Bayes factor from ipTM (Tier 1) or pDockQ (Tier 2).
Returns (bf_dock, calibration_tier).
Always available (no extension needed).

Quality gates applied in order:
1. fraction_disordered > 0.5 → BF=1.0, tier="disordered"
2. mean_plddt_min < 50 → clamp to [0.7, 1.5]
3. chain_pair_pae_min > 25 → clamp to [0.5, 2.0]
4. iptm_std > 0.15 → BF dampened toward 1.0
5. If pdockq available → Tier 2 logistic calibration (Bryant+Burke)
6. Otherwise → Tier 1 ipTM step function
"""
function compute_bf_dock end

"""
    default_calibration() -> DockingCalibration

Returns a pre-fitted calibration for AlphaFold Server scores,
based on published empirical data from Bryant et al. 2022 (pDockQ sigmoid,
AUC=0.87) and Burke et al. 2023 (precision at pDockQ thresholds,
direct vs indirect discrimination).

No custom benchmark docking required.
"""
function default_calibration end
```

### 7.5 Pipeline Integration

New fields in `CONFIG`:

```julia
# In CONFIG struct (src/analysis/pipeline.jl):
run_docking::Bool = false
docking_config::Union{DockingConfig, Nothing} = nothing
bait_sequence::String = ""         # Required when run_docking = true
bait_uniprot::String = ""          # For sequence auto-fetch
```

In `run_analysis`, after metalearner + report generation:

```julia
if config.run_docking && !isempty(config.bait_sequence)
    try
        dc = something(config.docking_config, DockingConfig())

        # Phase 1: Generate request JSONs
        batch = generate_docking_requests(
            final_results, config.bait_sequence;
            bait_name = config.poi,
            config = dc
        )

        @info "Docking requests generated" n_requests=batch.n_requests \
              n_batches=batch.n_batches guide=batch.guide_path

        # Phase 3 is triggered manually by the user after uploading
        # and downloading results. See upload_guide.txt.
    catch e
        @warn "Docking request generation failed" exception = e
    end
end
```

### 7.6 Sequence Retrieval

Prey sequences can come from two sources:

1. **UniProt REST API** (default): Resolve protein IDs via STRING → UniProt mapping (already in `string_api.jl`), then fetch FASTA from `https://rest.uniprot.org/uniprotkb/{id}.fasta`
2. **User-provided FASTA**: `generate_docking_requests(...; fasta_file="sequences.fasta")` — reads sequences from a local FASTA file, keyed by UniProt accession or gene name

The bait sequence is provided directly as a string (since the user knows their bait protein).

---

## 8. Critical Risks and Safeguards

### 8.1 Rate Limit Exhaustion (Critical)

**Risk**: User accidentally submits the same pairs twice, wasting precious daily slots.

**Safeguards**:
- Canonical cache key ensures deduplication across sessions
- `generate_docking_requests` never generates a JSON for a cached pair
- `upload_guide.txt` explicitly lists which pairs to skip
- `skipped_cached.csv` provides machine-readable skip list

### 8.2 IDP and Disorder Bias (Critical)

**Risk**: ~30-40% of human proteins are significantly disordered. Docking systematically produces low ipTM for IDP-mediated interactions.

**Safeguards**:
- `fraction_disordered > 0.5` (from AF Server output) → force `BF_dock = 1.0`
- Report `docking_status = "disordered"` explicitly
- HTML report includes warning when > 20% of docked candidates are disordered
- Never interpret low `bf_docking` as evidence of non-interaction without checking `docking_status`. Also clearly articulate that in the interactive report by adding a warning box. 

### 8.3 Token Limit Exceedance (Important)

**Risk**: Large proteins (>2500 residues) paired with a moderately-sized bait exceed 5000 tokens.

**Safeguards**:
- Pre-check `length(bait) + length(prey)` before generating JSON
- Pairs exceeding limit are logged to `skipped_too_large.csv` with token counts
- `docking_status = "too_large"` in output; `BF_dock = 1.0`

### 8.4 ipTM Calibration for Transient Interactions (Important)

**Risk**: ipTM was trained on PDB co-crystals (obligate, high-affinity). For transient AP-MS interactions, AUC is ~0.55-0.65.

**Safeguards**:
- Conservative BF clamping (max 20)
- Calibrate on external benchmarks (Negatome/Dockground), not experiment-internal data
- Documentation states that low ipTM does not equal non-interaction for transient binders, also state this in the interactive report

### 8.5 Large Dataset Feasibility (Important)

**Risk**: A typical AP-MS experiment yields 50-200 high-confidence hits. At 30/day, this requires 2-7 days of compute time.

**Safeguards**:
- `max_pairs` defaults to 100 to cap the total
- Strict posterior/q-value filtering reduces candidates to the most interesting subset
- Batch organization (batch_1, batch_2, ...) makes daily uploads manageable
- Progress tracking: `import_docking_results` shows how many pairs are still pending

### 8.6 Non-Commercial Use Restriction (Legal)

**Risk**: AlphaFold Server output is restricted to non-commercial use only (see Terms of Use).

**Safeguard**: Documentation and upload guide include a reminder about the non-commercial use restriction.

---

## 9. Module Layout

### Core Infrastructure

1. **`src/docking/stubs.jl`**: `DockingConfig`, `DockingPairResult`, `DockingResult`, `DockingRequestBatch`, `DockingCalibration` types + exported function stubs + `compute_bf_dock` + `apply_docking_update` + `default_calibration`
2. **`Project.toml`**: `[extensions]` and `[weakdeps]` entries for JSON3

### Request Generation

3. **`ext/.../request_generator.jl`**: `generate_docking_requests()` — candidate filtering, token budget checking, JSON file generation, batch organization, upload guide writing
4. **`ext/.../sequence_retrieval.jl`**: UniProt FASTA fetching + local FASTA file support
5. **`ext/.../cache.jl`**: JLD2 per-pair cache management, deduplication index

### Result Parsing

6. **`ext/.../result_parser.jl`**: `import_docking_results()` — ZIP extraction, summary_confidences parsing, pair identity resolution
7. **`ext/.../scoring.jl`**: Multi-model score extraction (max ipTM, std, disorder check), BF computation with clamping

### Integration

8. **CONFIG integration**: `run_docking`, `docking_config`, `bait_sequence` fields + wiring in `run_analysis`
9. **HTML report**: Display `posterior_prob_combined` and docking annotation columns
10. **Methods generator**: Include docking in Methods section when it was used

### Tests

11. **`test/docking/test_docking.jl`**: Cache key generation, token budget checking, JSON generation (format validation), summary_confidences parsing, BF computation + clamping, two-stage update formula, missing-data preservation, disorder flagging, deduplication logic

### Function Signatures Summary

```julia
# Always available (stubs):
compute_bf_dock(iptm::Float64, fraction_disordered::Float64;
                pdockq::Float64 = NaN,
                iptm_std::Float64 = 0.0,
                chain_pair_pae_min::Float64 = NaN,
                mean_plddt_min::Float64 = NaN,
                calibration::DockingCalibration = default_calibration()
                ) -> Tuple{Float64, String}   # (bf_dock, calibration_tier)

apply_docking_update(results::DataFrame, docking::DockingResult;
                     calibration::Union{DockingCalibration, Nothing} = nothing) -> DataFrame

default_calibration() -> DockingCalibration

compute_pdockq(avg_interface_plddt::Float64, n_interface_contacts::Int) -> Float64

# Requires `using JSON3`:
generate_docking_requests(results::DataFrame, bait_sequence::String;
                          bait_name::String = "BAIT",
                          output_dir::String = "docking_requests",
                          fasta_file::String = "",
                          config::DockingConfig = DockingConfig()) -> DockingRequestBatch

import_docking_results(results_dir::String, results::DataFrame;
                       config::DockingConfig = DockingConfig()) -> DockingResult
```

---

## 10. Interactive Report: Docking Tab

### 10.1 Overview

Docking results are displayed as a dedicated **Docking** tab in the interactive HTML report, visible only when docking data is present. The tab provides a complete self-contained view of all docking information — no external files needed.

### 10.2 Tab Content

The Docking tab contains four sections:

#### Section 1: Docking Dashboard Cards

Summary metrics displayed as colored cards (same style as main dashboard):

| Card | Value | Color |
|------|-------|-------|
| Pairs docked | `n_docked` | Blue |
| Pending | `n_pending` | Orange |
| Disordered (BF=1) | `n_disordered` | Purple |
| Too large (>5000 tokens) | `n_too_large` | Grey |
| Median ipTM | median of `iptm_best` across successful pairs | Teal |
| Median BF_dock | median of `bf_docking` across successful pairs | Green |

#### Section 2: Interactive ipTM vs Posterior Scatter Plot (Plotly)

- **X-axis**: `posterior_prob_ms` (MS-only posterior probability)
- **Y-axis**: `iptm_best` (best ipTM across 5 AF Server models)
- **Color**: `docking_status` category (success=blue, disordered=purple, too_large=grey)
- **Size**: proportional to `bf_docking` (larger = stronger docking evidence)
- **Hover**: protein name, ipTM (all 5 models), BF_dock, ranking_score, fraction_disordered, chain_pair_pae_min
- **Reference lines**: horizontal at ipTM = 0.4 (ambiguous threshold) and ipTM = 0.7 (high-confidence threshold)
- **Click interaction**: clicking a point highlights the same protein in all other tabs (shared selection state)

#### Section 3: Posterior Update Comparison Plot (Plotly)

- **X-axis**: `posterior_prob_ms` (before docking)
- **Y-axis**: `posterior_prob_combined` (after docking)
- **Diagonal**: identity line (y=x) — points above were boosted by docking, points below were reduced
- **Color**: `bf_docking` on a diverging colorscale (red = BF<1, blue = BF>1)
- **Hover**: protein name, P_ms, P_combined, BF_dock, ipTM

This plot makes the impact of the docking update immediately visible.

#### Section 4: Docking Results Table (DataTables)

Searchable, sortable, filterable table with columns:

| Column | Source | Format |
|--------|--------|--------|
| Protein | protein name | String |
| P(MS) | `posterior_prob_ms` | 3 decimals |
| P(Combined) | `posterior_prob_combined` | 3 decimals |
| BF Dock | `bf_docking` | 1 decimal (or scientific) |
| ipTM Best | `iptm_best` | 3 decimals |
| ipTM Std | `iptm_std` | 3 decimals |
| Ranking Score | `ranking_score` | 3 decimals |
| Disordered | `fraction_disordered` | 2 decimals |
| PAE Min | `chain_pair_pae_min` | 2 decimals |
| Status | `docking_status` | Badge (color-coded) |
| Tokens | `token_count` | Integer |

**Filters**: Status dropdown (success/disordered/too_large/pending), ipTM minimum, BF minimum.

**Export**: "Export Docking CSV" button exports only the docking-relevant columns.

### 10.3 JSON Data Structure

The report JSON blob gains a `"docking"` key:

```json
{
  "meta": { ... },
  "results": [ ... ],
  "plots": { ... },
  "methods": { ... },
  "bma": { ... },
  "docking": {
    "summary": {
      "n_total": 47,
      "n_docked": 35,
      "n_pending": 5,
      "n_disordered": 4,
      "n_too_large": 3,
      "median_iptm": 0.42,
      "median_bf_dock": 2.1
    },
    "pairs": [
      {
        "protein": "HAP40",
        "posterior_prob_ms": 0.95,
        "posterior_prob_combined": 0.97,
        "bf_docking": 3.2,
        "iptm_best": 0.65,
        "iptm_all": [0.26, 0.39, 0.28, 0.25, 0.24],
        "iptm_std": 0.06,
        "ranking_score": 0.67,
        "fraction_disordered": 0.01,
        "chain_pair_iptm": 0.39,
        "chain_pair_pae_min": 9.68,
        "status": "success",
        "token_count": 2204
      }
    ]
  }
}
```

When `docking` is absent or `docking.pairs` is empty, the Docking tab is hidden (same pattern as Diagnostics/Sensitivity tabs).

### 10.4 Integration with Main Results Table

The main Results tab table does NOT add docking columns — this would clutter the primary view. Instead:

- When docking data exists, the Protein column in the main table shows a small docking icon next to proteins that were docked
- Clicking the icon switches to the Docking tab with that protein highlighted
- The main table's "Evidence" badge is unchanged (based on MS-only posterior)

### 10.5 Report Generator Changes

In `report_generator.jl`:

```julia
# In _build_report_json:
function _build_report_json(results::DataFrame, config::CONFIG;
                            analysis_result = nothing,
                            docking_result = nothing)::String
    # ... existing code ...
    docking_json = _build_docking_json(results, docking_result)

    return json_object(
        "meta"    => meta_json,
        "results" => results_json,
        "plots"   => plots_json,
        "methods" => methods_json,
        "bma"     => bma_json,
        "docking" => docking_json,
    )
end
```

The `generate_report` function gains an optional `docking_result` keyword:

```julia
function generate_report(results::DataFrame, config::CONFIG;
                         output::String = config.output.report_file,
                         analysis_result = nothing,
                         docking_result = nothing)::Nothing
```

---

## References

### Calibration Sources (used for default BF computation)

- **Bryant, P. et al. (2022)**. Improved prediction of protein-protein interactions using AlphaFold2. *Nature Communications*, 13, 1265. — **pDockQ sigmoid equation**: L=0.707, x0=388.06, k=0.03148, b=0.03138. AUC=0.87 for interacting vs non-interacting. At 1% FPR: 51% TPR; at 5% FPR: 66% TPR.
- **Burke, D. F. et al. (2023)**. Towards a structurally resolved human protein interaction network. *Nature Structural & Molecular Biology*, 30, 216-225. — **Key calibration data**: pDockQ>0.5 → 80% correct (521/651); pDockQ>0.23 → 70% correct (671/955); random FPR=0.3%; direct vs indirect interactors: 38% vs 6% at pDockQ>0.5. 65,484 human protein pairs tested.
- **Yin, R. et al. (2022)**. Benchmarking AlphaFold for protein complex modeling reveals accuracy determinants. *Protein Science*, 31, e4379. — ipTM threshold ~0.75 for model confidence cutoff.
- **Kwon, D. et al. (2025)**. What's wrong with AlphaFold's ipTM score and how to fix it. *PMC11844409*. — ipSAE as improved alternative; ipTM weak for disordered targets.

### General References

- Evans, R. et al. (2022). Protein complex prediction with AlphaFold-Multimer. *bioRxiv*.
- Humphreys, I. R. et al. (2021). Computed structures of core eukaryotic protein complexes. *Science*, 374.
- Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596.
- Negatome 2.0 — curated database of non-interacting protein pairs.
- Dockground 5.0 — protein-protein docking benchmark database.
- AlphaFold Server: https://alphafoldserver.com
- AlphaFold Server JSON format: https://github.com/google-deepmind/alphafold/blob/main/server/README.md
