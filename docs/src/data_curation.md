# Data Curation and Quality Control

## Overview

Before running the Bayesian analysis, input data often requires curation to ensure protein identifiers are clean, unambiguous, and consistent. BayesInteractomics includes an automated curation pipeline that handles:

- **Contaminant removal**: Filter out common MS contaminants (CON\_\_, REV\_\_)
- **Protein group splitting**: Expand semicolon-delimited protein groups into individual entries
- **Synonym resolution**: Query the STRING database to map protein names to canonical identifiers
- **Duplicate merging**: Merge rows that map to the same canonical protein
- **Bait tracking**: Track the bait protein index through all curation steps

## Quick Start

Curation is enabled by default in `load_data` and `run_analysis`:

```julia
# Curation is automatic when using load_data
data = load_data(
    ["experiment.xlsx"],
    sample_cols, control_cols,
    curate = true,          # enabled by default
    species = 9606,         # NCBI taxonomy ID (9606 = human)
    bait_name = "HTT"       # track bait through curation
)

# Returns (InteractionData, bait_index) when bait_name is set
data, bait_idx = load_data(["experiment.xlsx"], sample_cols, control_cols,
    bait_name = "HTT")
```

## Curation Pipeline

### Step 1: Contaminant Removal

Removes rows whose protein ID starts with common contaminant prefixes:

```julia
# Controlled via:
curate_remove_contaminants = true  # default
```

Recognized prefixes: `CON__`, `REV__`, and other standard MaxQuant contaminant markers.

### Step 2: Protein Group Splitting

Mass spectrometry software often reports protein groups (multiple protein IDs separated by delimiters). The curation pipeline splits these into individual entries:

```julia
# Control the delimiter:
curate_delimiter = ";"  # default
```

Each protein in a group receives a copy of the original quantification data.

### Step 3: Synonym Resolution via STRING

Protein names are resolved to canonical identifiers by querying the STRING protein-protein interaction database:

```julia
curate = true
species = 9606          # Human (NCBI taxonomy ID)
# Other common species: 10090 (mouse), 10116 (rat), 7227 (fly), 6239 (worm)
```

Results are cached locally in `.bayesinteractomics_cache/` to avoid repeated API calls.

### Step 4: Duplicate Merging

After synonym resolution, multiple rows may map to the same canonical protein. These are merged using a configurable strategy:

```julia
curate_merge_strategy = :max   # Keep maximum value per cell (default)
# Alternative: :mean           # Average values per cell
```

When `curate_interactive = true` (default), the user is prompted to confirm each merge. Use `curate_auto_approve` to auto-approve merges where proteins share a common prefix:

```julia
curate_auto_approve = 3  # Auto-approve if first 3 characters match
```

## Curation Reports

Every curation run produces a `CurationReport` that logs all actions taken:

```julia
# Reports are saved automatically next to data files
# Location: .bayesinteractomics_cache/<filename>_curation_report.jld2
```

### Replay Mode

Re-run a previous curation without interactive prompts:

```julia
data = load_data(["experiment.xlsx"], sample_cols, control_cols,
    curate_replay = "path/to/curation_report.jld2"
)
```

This is useful for reproducible analyses and batch processing.

## Configuration via CONFIG

All curation parameters can be set in the `CONFIG` struct:

```julia
config = CONFIG(
    # ... other fields ...
    curate = true,
    species = 9606,
    curate_interactive = true,
    curate_merge_strategy = :max,
    bait_name = "HTT",
    curate_replay = nothing,
    curate_remove_contaminants = true,
    curate_delimiter = ";",
    curate_auto_approve = 0
)
```

## Curation Types

```@docs
curate_proteins
CurationReport
CurationEntry
CurationActionType
MergeCandidate
```

## Utility Functions

```@docs
remove_contaminants
parse_protein_id
split_protein_groups
```
