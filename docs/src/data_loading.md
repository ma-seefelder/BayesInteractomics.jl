# Data Loading

This page explains how to load and structure your mass spectrometry data for analysis with BayesInteractomics.

## Overview

BayesInteractomics uses a **hierarchical data structure** that naturally reflects experimental design:

```
InteractionData
├── Protocol 1 (e.g., GST-tagged AP-MS)
│   ├── Experiment 1 (biological replicate set 1)
│   │   ├── Control samples: [c1, c2, c3]
│   │   └── Bait samples: [s1, s2, s3]
│   ├── Experiment 2 (biological replicate set 2)
│   │   ├── Control samples: [c4, c5]
│   │   └── Bait samples: [s4, s5]
│   └── ...
└── Protocol 2 (e.g., Strep-tagged AP-MS)
    ├── Experiment 1
    │   ├── Control samples: [c1, c2, c3, c4]
    │   └── Bait samples: [s1, s2, s3]
    └── ...
```

This structure enables:
- **Protocol-level modeling**: Account for systematic differences between experimental methods
- **Experiment-level variation**: Handle batch effects within protocols
- **Sample-level observations**: Work with individual replicate measurements
- **Missing data handling**: Naturally accommodate proteins not detected in all replicates

## Data Format Requirements

### File Formats

BayesInteractomics supports:
- **Excel files** (.xlsx) - Recommended for most users
- **CSV files** (.csv) - Alternative for programmatic workflows

### Data Matrix Structure

Your data files should be organized as:

| Column 1 | Column 2 | Column 3 | ... | Column N |
|----------|----------|----------|-----|----------|
| **Protein ID** | **Sample/Control 1** | **Sample/Control 2** | ... | **Sample/Control N** |
| PROTEIN_A | 25.3 | 24.8 | ... | 28.5 |
| PROTEIN_B | 22.1 | NA | ... | 22.0 |
| BAIT | 18.2 | 18.5 | ... | 32.1 |

**Requirements**:
- **First column**: Protein identifiers (e.g., UniProt IDs, gene names, Ensembl IDs)
- **Subsequent columns**: Intensity values for each sample
- **Values**: Preferably log2-transformed intensities (though not strictly required)
- **Missing data**: Leave cells empty, use NA, or use Julia `missing`
- **Consistent protein order**: All files should have proteins in the same order

### Intensity Normalization

It's recommended to:
1. **Log-transform** intensities (log2 or ln) before loading
2. **Within-sample normalization** (e.g., median normalization, TMM) if appropriate
3. Let BayesInteractomics handle **protocol normalization** via `normalise_protocols=true`

## Loading Data: Column Specification

The key to loading data correctly is specifying **which columns contain which samples**.

### Column Mapping Format

Column mappings use nested dictionaries:

```julia
sample_cols = [
    # Protocol 1 column mapping
    Dict(
        1 => [5, 6, 7],      # Experiment 1: columns 5-7
        2 => [8, 9]          # Experiment 2: columns 8-9
    ),

    # Protocol 2 column mapping
    Dict(
        1 => [11, 12, 13, 14]  # Experiment 1: columns 11-14
    )
]
```

**Structure**:
- **Outer array**: One element per protocol
- **Inner Dict**: Maps experiment IDs to column indices
- **Column indices**: 1-based (Julia convention), column 1 is protein IDs

### Example: Single Protocol

For a simple experiment with one protocol and two experiments:

```julia
using BayesInteractomics

# Data file structure:
# Column 1: Protein IDs
# Columns 2-4: Control replicates from experiment 1
# Columns 5-7: Control replicates from experiment 2
# Columns 8-10: Sample replicates from experiment 1
# Columns 11-13: Sample replicates from experiment 2

control_cols = [
    Dict(
        1 => [2, 3, 4],      # Experiment 1 controls
        2 => [5, 6, 7]       # Experiment 2 controls
    )
]

sample_cols = [
    Dict(
        1 => [8, 9, 10],     # Experiment 1 samples
        2 => [11, 12, 13]    # Experiment 2 samples
    )
]

# Load data
data = load_data(
    ["experiment_data.xlsx"],
    sample_cols,
    control_cols,
    normalise_protocols = false  # Single protocol, no normalization needed
)
```

### Example: Multiple Protocols

For combining data from different experimental methods:

```julia
# Protocol 1: AP-MS with GST tag (file: apms_gst.xlsx)
# Protocol 2: AP-MS with Strep tag (file: apms_strep.xlsx)

# Protocol 1 has 2 experiments, Protocol 2 has 1 experiment

control_cols = [
    # Protocol 1
    Dict(1 => [2, 3, 4], 2 => [5, 6]),

    # Protocol 2
    Dict(1 => [2, 3, 4, 5])
]

sample_cols = [
    # Protocol 1
    Dict(1 => [7, 8, 9], 2 => [10, 11, 12]),

    # Protocol 2
    Dict(1 => [6, 7, 8])
]

data = load_data(
    ["apms_gst.xlsx", "apms_strep.xlsx"],
    sample_cols,
    control_cols,
    normalise_protocols = true  # IMPORTANT: Normalize across protocols
)
```

## Normalisation

BayesInteractomics applies sample / protocol normalisation at the END of [`load_data`](@ref) (BEFORE imputation). Normalisation operates on the log2 scale and preserves `missing` cells. The pipeline supports four concrete normalisation methods plus an `:auto` selector that auto-detects multi-protocol scale mismatch and applies the appropriate combination.

### The `normalisation_method` selector

Set via the `CONFIG.normalisation_method::Symbol` field (or the `normalisation_method=` kwarg to [`load_data`](@ref)). Allowed values:

| Value                | Behaviour                                                                                                                       | Backing function                                                       |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `:none`              | Identity (no normalisation). Byte-identical to the pre-v1.2.0 `normalise_protocols=false` behaviour.                            | passthrough                                                            |
| `:row_center`        | Per-protein per-(protocol, exp) row-centering. Byte-identical to the pre-v1.2.0 `normalise_protocols=true` behaviour.           | `normalize(data)`                                                      |
| `:median_of_ratios`  | DESeq size-factor column-scaling normaliser. Stays on log2; missing-aware; per-protein geometric mean over fully-observed rows. | [`norm_median_of_ratios_id`](@ref)                                     |
| `:both`              | Apply `:median_of_ratios` FIRST, then `:row_center` (column-scale then row-center).                                            | `normalize(norm_median_of_ratios_id(data))`                            |
| `:auto` (default)    | Auto-detect multi-protocol scale mismatch via `detect_protocol_scale_mismatch`; apply `:both` if mismatch detected, else `:none`. Resolution happens at `load_data` apply time. | `_resolve_normalisation_method` |

The dispatcher [`apply_normalisation`](@ref) routes an already-RESOLVED concrete method onto an `InteractionData`. `:auto` is resolved BEFORE dispatch; passing `:auto` to `apply_normalisation` directly throws `ArgumentError`.

### Legacy `normalise_protocols::Bool` back-compat

The pre-v1.2.0 `CONFIG.normalise_protocols::Bool` kwarg still works. Precedence rule (from `_resolve_normalisation_method` in `src/data/loading.jl`): if `normalisation_method !== :none` the new selector is AUTHORITATIVE; otherwise the legacy bool is mapped (`true → :row_center`, `false → :none`). Existing scripts that pass only `normalise_protocols=true` continue to row-center exactly as before.

### Method details

**`:median_of_ratios` — DESeq size-factor column scaling.** For each MS-run column `j`, compute the size factor `s_j = median_i(y_ij / geomean_i)` where `geomean_i` is the per-protein geometric mean over proteins observed in ALL columns; divide column `j` by `s_j`; round-trip back to log2. The `<= 0 → missing` guard prevents NaN/Inf leaking into HBM/regression downstream. Operates via the `build_run_matrix` / `matrix_to_interactiondata` round-trip — flatten `InteractionData` into a `(n_proteins × n_runs)` log2 matrix, normalise, write back. Composition-bias + missing robust; beats row-centering on replicate noise and MA flatness in benchmarks.

**`:row_center` — per-protein cross-protocol row-centering.** The pre-v1.2.0 `normalize()` behaviour: subtract the per-protein per-(protocol, exp) row mean from each cell. Required to fix multi-protocol `bf_correlation` saturation (median_of_ratios alone leaves ~38% saturated; row-centering brings it to 0%; the two compose). Row-centering subtracts a CONSTANT from sample AND control, which cancels in the sample−control contrast — HBM-safe, log2FC-invariant.

**`:both` — column-scale then row-center.** Column-scale FIRST then row-center. Orthogonal axes (column-scaling fixes sample loading, row-centering fixes the per-protein cross-protocol baseline offset). Used by `:auto` when a multi-protocol scale mismatch is detected.

### Ordering: normalise BEFORE impute

Normalisation runs BEFORE MNAR imputation. The MNAR dropout curve `σ(ρ_c + ζ_c · ȳ_i)` is intensity-scale-sensitive, so size factors must be computed on the pre-imputation observed data. Imputing first contaminates the size factors — imputation accuracy is higher with normalise→impute ordering for all normalisers.

For the **user-pre-imputes workflow** (where you produce an imputed XLSX file then re-load it), use [`normalise_then_impute`](@ref) to guarantee the correct order. The two-step file path (`impute_mnar_from_paths` writes `dataset_mnar.xlsx`, then `load_data(...; normalisation_method=...)` reads it) normalises AFTER imputation — the WRONG order. `load_data` emits a warning if it reads an already-imputed file with a non-`:none` normalisation requested.

**Signature:**

```julia
normalise_then_impute(
    raw_data::InteractionData,
    dropout_fit::DropoutFit;
    normalisation_method::Symbol = :auto,
    refID::Int = 1,
) -> InteractionData
```

Steps: (1) resolve `normalisation_method` (`:auto` → `detect_protocol_scale_mismatch` → `:both` or `:none`, identical to the `load_data` apply site); (2) `apply_normalisation(raw_data, resolved)` on the OBSERVED data; (3) extract the post-normalisation intensity matrix, MNAR-impute once via the imputation extension's `impute_mnar`, and round-trip back into a complete `InteractionData` mirroring `raw_data`'s schema. Requires `using GLM` (loads `BayesInteractomicsImputationExt`) — errors loudly via `_require_imputation_extension(:mnar)` when the extension is absent. See [Imputation](imputation.md) for the imputation half of the workflow.

### Worked example

```julia
using BayesInteractomics, GLM

# (1) Default — auto-detect multi-protocol scale mismatch
config = CONFIG(
    datafile = ["data.xlsx"],
    # normalisation_method defaults to :auto
)

# (2) Force the full :both recipe unconditionally
config = CONFIG(
    datafile = ["data.xlsx"],
    normalisation_method = :both,
)

# (3) User-pre-imputes workflow — normalise FIRST, impute SECOND
raw_data = load_data(["data.xlsx"], sample_cols, control_cols;
                      normalisation_method = :none, impute = false)
fit = fit_dropout_curves(raw_data)
imputed = normalise_then_impute(raw_data, fit; normalisation_method = :auto)
```

See [Configuration](configuration.md) for the CONFIG-field reference and [Imputation](imputation.md) for the imputation half of the user-pre-imputes workflow.

## Bait Anchoring

Bait anchoring is a regression-safe per-condition normalisation step that equalises the bait protein's MEAN sample level across protocols (= conditions). It is used by [`differential_analysis`](@ref) when multiple conditions (e.g. WT vs mutant) are compared and the bait's expression level differs systematically between them. Anchoring removes the per-condition bait-level gap from each prey's enrichment (sample − control) while preserving the dose axis used by the regression model.

### The `bait_anchor_id` helper

**Signature:**

```julia
bait_anchor_id(data::InteractionData; bait_row::Int=1) -> InteractionData
```

Computes a per-protocol scalar `δ_c` and subtracts it from the SAMPLE cells (controls untouched) of that protocol's bait row:

```
δ_c = mean(bait SAMPLE level in protocol c) − grand mean of per-protocol bait sample means
```

where the grand mean averages over protocols with at least one observed bait sample. `bait_row` defaults to `1` (the `refID` / bait position in the protein order); pass an explicit row index if the bait is not the first protein.

**No-op when fewer than 2 protocols** — single-condition data has nothing to anchor. Missings preserved.

**SAMPLE cells only.** Subtracting `δ_c` from BOTH sample and control would cancel in the sample−control contrast and leave the differential unchanged. The whole point of bait-anchoring is to shift the differential by the bait-level gap — hence sample-only.

### Regression-safety contract

Each protocol's bait sample cells shift by a per-protocol CONSTANT. Within-condition run-to-run bait variation (the regression dose axis) is preserved and the predictor is never zeroed or de-varied. This is the property that makes bait-anchoring safe to apply BEFORE the dose-response Bayesian regression — the alternative `quantile` / `cyclic_loess` approaches destroy the dose axis and are forbidden for differential AP-MS.

### Caller contract: apply on RAW bait abundance

`δ_c` is derived from the bait abundance of the data AS PASSED, so callers MUST apply `bait_anchor_id` on the data on which RAW bait abundance is meaningful — relative to RAW bait levels, NOT post-(re)scaling values that would reposition the high-abundance bait differently per condition. On matched-level multi-protocol loads (e.g. real HAP40 GST/Strep, ~30 vs ~30.1 log2) `δ_c ≈ 0` and the anchor is correctly near-inert.

### Differential analysis bait-anchor pathway

The in-pipeline bait-anchor helper `_apply_bait_anchor_diff!` (in `src/differential/analysis.jl`) is the entry point used by [`differential_analysis`](@ref). It calls `bait_anchor_id` on each condition's `InteractionData` before the dBF computation begins, ensuring the per-condition bait-level gap is removed from the enrichment statistic that feeds into the BMA (Copula + 3c-EM) sub-models. See [Differential Analysis](differential_analysis.md) for the dBF flow that consumes the bait-anchored data.

### `CONFIG.bait_name` + `refID` interaction with curation

The bait position in the protein order can be RELOCATED by the data curation step (`curate=true` default). `load_data` accepts a `bait_name::Union{Nothing, String}=nothing` kwarg; when set, the curation pipeline tracks the bait through synonym resolution / contaminant removal / protein-group splitting and returns the bait's NEW row index after curation. `run_analysis` then updates `CONFIG.refID` to point at the relocated bait, so downstream consumers of the bait row (including `bait_anchor_id(data; bait_row=config.refID)`) target the correct row.

Pass `bait_name` whenever curation may reorder proteins:

```julia
data, bait_idx = load_data(
    files, sample_cols, control_cols;
    bait_name = "HAP40",      # canonical name; curation resolves synonyms
    curate = true,             # default
)
config = CONFIG(
    datafile = files,
    sample_cols = sample_cols,
    control_cols = control_cols,
    poi = "HAP40",
    refID = bait_idx,          # post-curation position
    # ...
)
```

See [Data Curation](data_curation.md) for the curation pipeline; `bait_anchor_id` is a downstream consumer of the curation-stable `refID`.

## The `load_data` Function

### Basic Usage

```julia
data = load_data(
    files::Vector{String},
    sample_cols::Vector{Dict{Int, Vector{Int}}},
    control_cols::Vector{Dict{Int, Vector{Int}}};
    normalise_protocols::Bool = false
)
```

### Parameters

- **`files`**: Vector of file paths (one per protocol)
- **`sample_cols`**: Column mappings for bait/treatment samples
- **`control_cols`**: Column mappings for negative control samples
- **`normalise_protocols`**: If `true`, normalizes data across protocols using z-score transformation

### Return Value

Returns an `InteractionData` object containing:
- All protocols with their experiments
- Sample and control data for each experiment
- Protein identifiers
- Metadata about data structure (positions, dimensions)

## Data Structure Reference

### InteractionData Type

The `InteractionData` type is the central data container:

```julia
# Access protocols
for protocol in data
    println("Protocol: $(protocol.name)")

    # Access experiments within protocol
    for (exp_id, experiment_data) in protocol.experiments
        println("  Experiment $exp_id")
        println("    Size: $(size(experiment_data))")
    end
end

# Get number of proteins
n_proteins = getNoProteins(data)

# Get number of protocols
n_protocols = getNoProtocols(data)

# Extract data for specific protein
protein_data = getProteinData(data, protein_index)
```

### Protocol Type

Each protocol contains:
- **`name`**: Protocol identifier
- **`proteinNames`**: Vector of protein IDs
- **`experiments`**: Dictionary mapping experiment IDs to data matrices

### Protein Type

Extracting protein-specific data:

```julia
protein = getProteinData(data, 42)  # Get protein at index 42

# Access samples and controls for each protocol
for (protocol_idx, (samples, controls)) in enumerate(protein)
    println("Protocol $protocol_idx:")
    println("  Samples: $samples")
    println("  Controls: $controls")
end
```

## Worked Example

Let's walk through loading a complete dataset:

### Step 1: Prepare Your Files

Suppose you have:
- `protocol1.xlsx`: AP-MS data with 2 experiments
- `protocol2.xlsx`: BioID data with 1 experiment

Each file has the same proteins in the same order.

### Step 2: Identify Column Indices

Open your files and note which columns contain which data:

**protocol1.xlsx**:
- Column 1: Protein IDs
- Columns 2-4: Experiment 1 controls (3 replicates)
- Columns 5-7: Experiment 1 samples (3 replicates)
- Columns 8-10: Experiment 2 controls (3 replicates)
- Columns 11-13: Experiment 2 samples (3 replicates)

**protocol2.xlsx**:
- Column 1: Protein IDs
- Columns 2-5: Experiment 1 controls (4 replicates)
- Columns 6-8: Experiment 1 samples (3 replicates)

### Step 3: Create Column Mappings

```julia
control_cols = [
    # Protocol 1 (2 experiments)
    Dict(1 => [2, 3, 4], 2 => [8, 9, 10]),

    # Protocol 2 (1 experiment)
    Dict(1 => [2, 3, 4, 5])
]

sample_cols = [
    # Protocol 1 (2 experiments)
    Dict(1 => [5, 6, 7], 2 => [11, 12, 13]),

    # Protocol 2 (1 experiment)
    Dict(1 => [6, 7, 8])
]
```

### Step 4: Load the Data

```julia
using BayesInteractomics

data = load_data(
    ["protocol1.xlsx", "protocol2.xlsx"],
    sample_cols,
    control_cols,
    normalise_protocols = true
)

@info "Loaded data successfully"
@info "Proteins: $(getNoProteins(data))"
@info "Protocols: $(getNoProtocols(data))"
```

### Step 5: Verify the Data

```julia
# Check structure
println("Protocol 1:")
for (exp_id, exp_data) in data.protocols[1].experiments
    println("  Experiment $exp_id: $(size(exp_data))")
end

println("Protocol 2:")
for (exp_id, exp_data) in data.protocols[2].experiments
    println("  Experiment $exp_id: $(size(exp_data))")
end

# Check a specific protein
protein_42 = getProteinData(data, 42)
println("\nProtein at index 42:")
println("  Name: $(protein_42.name)")
println("  Protocol 1 samples: $(protein_42[1][1])")
println("  Protocol 1 controls: $(protein_42[1][2])")
```

## Protocol Normalization

When `normalise_protocols = true`, BayesInteractomics applies z-score normalization to each protocol:

```math
x_{\text{norm}} = \frac{x - \mu}{\sigma}
```

where $\mu$ and $\sigma$ are computed from all samples within that protocol.

**When to use**:
- ✅ Multiple protocols with potentially different intensity scales
- ✅ Combining data from different labs or instruments
- ✅ Different experimental methods (e.g., AP-MS vs BioID)

**When not to use**:
- ❌ Single protocol analysis
- ❌ Data already normalized to common scale
- ❌ Relative quantification methods (e.g., SILAC ratios)

## Troubleshooting

### Common Errors and Solutions

**Error: "Column index out of bounds"**

**Cause**: Column indices don't match file structure

**Solution**:
- Check that column indices are 1-based
- Verify file has the expected number of columns
- Remember column 1 is protein IDs

---

**Error: "Protein names don't match across files"**

**Cause**: Different files have different protein sets or different order

**Solution**:
- Ensure all files contain the same proteins
- Proteins must be in the same row order across files
- Use the same protein ID format in all files

---

**Error: "All values missing for protein X"**

**Cause**: Protein has no detected values in any sample

**Solution**:
- This is a warning, not an error
- Proteins with all missing values will have undefined Bayes factors
- Consider pre-filtering proteins detected in at least some samples

---

**Error: "refID protein not found"**

**Cause**: The bait protein specified by `refID` doesn't exist at that row index

**Solution**:
- Verify the row number of your bait protein (1-indexed)
- Check that bait protein is present in all data files
- Use `findfirst(==("BAIT_ID"), protein_ids)` to find the correct index

## Best Practices

1. **Consistent formatting**: Use the same protein ID format across all files
2. **Quality control**: Pre-filter proteins detected in very few samples
3. **Missing data**: Don't artificially impute zeros - leave as missing
4. **Log transformation**: Apply log2 transformation before loading if using raw intensities
5. **Documentation**: Keep notes on column mappings for future reference
6. **Validation**: Always inspect loaded data structure before running analysis

## API Reference

Functions and types for loading and processing AP-MS and proximity labeling data.

```@docs
load_data
InteractionData
Protocol
Protein
getProteinData
```
