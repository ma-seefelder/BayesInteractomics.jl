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

Functions for loading and processing AP-MS and proximity labeling data.

```@autodocs
Modules = [BayesInteractomics]
Pages = ["data_loading.jl", "utils.jl"]
```
