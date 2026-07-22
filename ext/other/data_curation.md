
# Data Curation Strategy for BayesInteractomics

**Authors**: Bioinformatics Expert, Julia Implementation Expert, Critical Reviewer (Devil's Advocate)
**Date**: 2026-02-21
**Status**: Design Document

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Bioinformatics Analysis](#2-bioinformatics-analysis)
3. [Julia Implementation Design](#3-julia-implementation-design)
4. [Critical Review & Risk Mitigations](#4-critical-review--risk-mitigations)
5. [Final Recommended Strategy (Consensus)](#5-final-recommended-strategy-consensus)
6. [Implementation Roadmap](#6-implementation-roadmap)

---

## 1. Problem Statement

### 1.1 Context

BayesInteractomics analyzes protein interactome data from Affinity-Purification Mass Spectrometry (AP-MS) experiments. The data loading pipeline (`load_data()`) reads Excel/CSV files where each row is a protein with intensity values across sample and control columns. Three issues arise in real-world proteomics datasets:

### 1.2 Protein Groups

Mass spectrometry search engines (MaxQuant, Proteome Discoverer, MSFragger) report **protein groups** when shared peptides cannot unambiguously assign spectra to a single protein. These appear as semicolon-delimited entries:

```
RBFOX3;RBFOX2;RBFOX1
CALM1;CALM2;CALM3
```

For BayesInteractomics, each group must be split into individual protein entries because the Bayesian analysis operates per-protein. However, this splitting introduces statistical complications (see Section 4).

### 1.3 Protein Synonyms

The same protein may appear under different names across experiments, especially when datasets come from different laboratories or use different FASTA databases:

| Name in Experiment A | Name in Experiment B | Canonical Gene | STRING ID            |
| -------------------- | -------------------- | -------------- | -------------------- |
| HAP40                | F8A1                 | F8A1           | 9606.ENSP00000479624 |
| RAB5A                | RAB5A/RAB5B          | RAB5A          | 9606.ENSP00000418169 |

These synonyms result in fragmented evidence across rows that should represent a single protein.

### 1.4 Requirements

1. **Split** protein groups into individual entries with duplicated quantitative data
2. **Resolve** protein names to canonical identifiers (via STRING DB or UniProt)
3. **Merge** rows that map to the same canonical protein, with **interactive user confirmation** for non-identical names
4. **Cache** curated datasets and mapping decisions for reproducibility
5. **Document** all curation decisions in a human-readable report

---

## 2. Bioinformatics Analysis

### 2.1 Protein Group Splitting Strategy

**Standard practice**: The first protein in a semicolon-delimited group is the "lead" or "master" protein — the one with the strongest peptide evidence (most unique peptides, highest sequence coverage, highest score). This convention is consistent across MaxQuant ("Majority protein IDs"), Proteome Discoverer ("Master Protein"), and Scaffold.

**Recommended splitting approach**:

1. Split on the semicolon delimiter (`;`), strip whitespace from each element
2. **Duplicate the entire quantitative row** for each member protein — shared peptides contribute to the quantification of all group members equally, and there is no principled way to deconvolve the shared signal
3. Add metadata columns:
   - `protein_group_id::String` — the original unsplit group identifier
   - `is_lead_protein::Bool` — `true` for the first protein in the group
   - `group_size::Int` — number of proteins in the original group

**Slash-separated ambiguous identifiers** (e.g., `"RAB5A/RAB5B"`, `"CALM1/CALM2/CALM3"`) should be treated identically: split on `/`, resolve each, and if they map to different canonical IDs, treat as a protein group.

**Delimiter handling**: Support `;` as primary delimiter and `/` as secondary. Always `strip()` whitespace from each element.

### 2.2 Synonym Resolution via STRING DB

**Primary API endpoint**:

```
POST https://string-db.org/api/json/get_string_ids
Parameters:
  identifiers: gene names separated by %0d (carriage return)
  species: NCBI taxonomy ID (e.g., 9606 for human)
  limit: 1 (single best match)
  echo_query: 1 (include original name in response)
  caller_identity: "BayesInteractomics"
```

**Response fields**: `queryItem`, `stringId`, `preferredName`, `annotation`, `ncbiTaxonId`

**Batch strategy**:

- Batch size: 200–500 identifiers per POST request (avoids URL length limits)
- Rate limit: ~1 request/second
- For a typical 2000–5000 protein dataset: 4–25 API calls, ~5 minutes total

**Handling ambiguous mappings**:

- Use `limit=1` for the single best match
- If two different query names resolve to the same `stringId`, they are confirmed synonyms → merge candidates
- If a name has no STRING mapping, retain it with `mapped_to_string = false`

**Handling identifier formats**: Many AP-MS datasets use UniProt accession format (`sp|P12345|MYC_HUMAN`). Parse on `|` and submit the accession (second field). Regex patterns for auto-detection:

| Format              | Pattern                                                                         | Example               |
| ------------------- | ------------------------------------------------------------------------------- | --------------------- |
| UniProt SP/TrEMBL   | `^[OPQ][0-9][A-Z0-9]{3}[0-9]$` or `^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$` | P04637                |
| Ensembl Protein     | `^ENSP\d+$`                                                                     | ENSP00000269305       |
| RefSeq              | `^[NXY]P_\d+$`                                                                  | NP_000537.3           |
| UniProt pipe format | `^(sp\|tr)\|[A-Z0-9]+\|.*$`                                                     | sp\|P04637\|P53_HUMAN |

**Isoform handling**: Strip isoform suffixes (`P12345-2` → `P12345`) for synonym resolution. Preserve original isoform annotation in metadata. STRING maps all isoforms to the same canonical Ensembl protein ID.

**Alternative: UniProt ID Mapping** (`POST https://rest.uniprot.org/idmapping/run`):

- More authoritative for protein identity, accessions are permanent
- Native gene name synonym support
- Asynchronous API (poll for results)
- Recommended as **fallback** when STRING fails or for datasets using UniProt accessions directly

### 2.3 Data Merging Rules

When two rows with different names map to the same canonical protein, merging must respect quantification semantics.

**Recommended two-step merge rule** (for LFQ intensities, the primary use case):

1. For each replicate column: if only one row has a non-missing value, **use that value** (fill gaps)
2. If both rows have non-missing values in the same column, take the **maximum** value (preserves signal; see Section 4.3 for rationale)

For spectral counts: **sum** (counts are additive).
For MS1/iBAQ/raw intensities: **maximum** (robust to partial identifications).

**Missing value handling example**:

```
Protein A: [10.5, missing, 8.2,  missing]
Protein B: [missing, 7.3,  9.1,  missing]
Merged:    [10.5,   7.3,  max(8.2, 9.1) = 9.1, missing]
```

**Metadata preservation**: Concatenate all original names into a `synonyms` field. Use STRING `preferredName` as the canonical display name. Record the `stringId` as the stable identifier.

### 2.4 Contaminant and Decoy Removal

Contaminant/decoy entries should be removed during curation (opt-in, default on):

- MaxQuant: `CON__` (contaminants), `REV__` (decoy)
- Case-insensitive matching: also handle `con_`, `rev_`, `CONTAMINANT_`
- Common contaminants: keratins, trypsin, BSA, serum albumin

**Filter rule**: `startswith(lowercase(id), "con_") || startswith(lowercase(id), "rev_")`

### 2.5 Recommended Processing Order

```
Raw DataFrame
  │
  ├─ 1. Remove contaminants/decoys (CON__, REV__)
  ├─ 2. Parse identifiers (handle UniProt pipe format, strip isoform suffixes)
  ├─ 3. Split protein groups (semicolon-delimited → duplicate rows)
  ├─ 4. Split slash-separated ambiguous identifiers
  ├─ 5. Resolve all protein names via STRING get_string_ids (with local cache)
  ├─ 6. Identify synonyms (group rows by canonical stringId)
  ├─ 7. Interactive merge confirmation (user approves/rejects each merge)
  ├─ 8. Merge confirmed synonym rows (fill-then-max strategy)
  ├─ 9. Flag remaining protein group members for downstream deduplication
  └─ 10. Output: curated DataFrame + CurationReport
```

---

## 3. Julia Implementation Design

### 3.1 Module Location and Structure

The curation module should live in the core source tree as **optional preprocessing**, not as a package extension. Unlike the network extension (which requires heavy graph dependencies), curation uses only stdlib + existing deps (Downloads, JLD2, SHA, CSV, DataFrames).

**No new dependencies required**: The STRING API `get_string_ids` endpoint supports TSV output, which can be parsed with the existing `CSV` dependency — no need to add `JSON3`. This follows the same pattern as `ext/BayesInteractomicsNetworkExt/ppi_query.jl`, which already queries STRING via TSV.

**File structure**:

```
src/
  data/
    loading.jl          # Existing — add `curate` keyword
    curation.jl         # NEW — main curation logic + orchestrator
    curation_types.jl   # NEW — CurationReport, MergeCandidate, etc.
    string_api.jl       # NEW — STRING DB API client + cache
```

**Include order in `BayesInteractomics.jl`**: Add after `include("data/loading.jl")`:

```julia
include("data/curation_types.jl")
include("data/string_api.jl")
include("data/curation.jl")
```

### 3.2 Type Definitions (`src/data/curation_types.jl`)

```julia
"""Enum for curation action types — cleaner than raw Symbols."""
@enum CurationActionType begin
    CURATE_KEEP       # No change needed
    CURATE_SPLIT      # Protein group was split into individual rows
    CURATE_MERGE      # Multiple rows merged to single canonical ID
    CURATE_RENAME     # Name resolved to canonical STRING preferred name
    CURATE_REMOVE     # Contaminant/decoy removed
    CURATE_UNMAPPED   # Could not resolve in STRING
end

"""Record of a single curation action — immutable for audit trail safety."""
struct CurationEntry
    original_name::String
    canonical_name::String
    canonical_id::String           # STRING ID or "" if unmapped
    action::CurationActionType
    reason::String                 # Human-readable explanation
    user_approved::Bool            # true if user confirmed interactively
    source_row_indices::Vector{Int}  # original DataFrame row indices involved
    group_id::Union{String, Nothing} # protein group origin, if split
    is_lead::Bool                  # true if lead protein in group
    timestamp::DateTime
end

"""A candidate merge for user confirmation."""
struct MergeCandidate
    names::Vector{String}       # All names mapping to same canonical ID
    canonical_id::String        # STRING ID
    preferred_name::String      # STRING preferred name
    annotation::String          # STRING protein description
    row_indices::Dict{String, Vector{Int}}  # name => row indices in DataFrame
end

"""User's decision on a merge candidate."""
struct MergeDecision
    candidate::MergeCandidate
    approved::Bool
    chosen_name::String         # which name to keep as canonical
    timestamp::DateTime
end

"""Complete curation report — the key reproducibility artifact."""
struct CurationReport
    entries::Vector{CurationEntry}
    merge_decisions::Vector{MergeDecision}
    species::Int
    string_api_version::String
    data_hash::UInt64           # hash of original input for replay validation
    timestamp::DateTime
    package_version::String
    n_proteins_before::Int
    n_proteins_after::Int
    summary::Dict{Symbol, Int}  # :splits, :merges, :removals, :unmapped, :kept
end

"""Local cache for STRING ID mappings."""
struct CurationCache
    mapping::Dict{String, String}            # original_name => STRING_ID
    preferred_names::Dict{String, String}    # STRING_ID => preferred_name
    species::Int
    string_version::String
    query_timestamp::DateTime
end
```

### 3.3 Core Function Signatures (`src/data/curation.jl`)

```julia
"""
    curate_proteins(data::DataFrame, id_col::Integer;
        species::Int = 9606,
        delimiter::String = ";",
        remove_contaminants::Bool = true,
        cache_dir::String = ".bayesinteractomics_cache",
        interactive::Bool = true,
        merge_strategy::Symbol = :max,
        replay_report::Union{Nothing, String} = nothing,
        bait_name::Union{Nothing, String} = nothing
    ) -> (DataFrame, CurationReport)

Main entry point. Orchestrates the full curation pipeline:
1. Remove contaminants
2. Split protein groups
3. Resolve names via STRING API
4. Identify and confirm merges
5. Merge confirmed duplicates
6. Return curated DataFrame + report
"""
function curate_proteins end

"""
    split_protein_groups(data::DataFrame, id_col::Integer;
        delimiter::String = ";",
        report::CurationReport
    ) -> DataFrame

Split semicolon-delimited protein groups into individual rows.
Adds columns: :protein_group_id, :is_lead_protein, :group_size
"""
function split_protein_groups end

"""
    resolve_to_string_ids(protein_names::Vector{String};
        species::Int = 9606,
        cache_dir::String = ".bayesinteractomics_cache",
        batch_size::Int = 200
    ) -> Dict{String, NamedTuple{(:string_id, :preferred_name, :score), ...}}

Query STRING API with local caching. Returns mapping from
input name to (string_id, preferred_name, confidence_score).
Unmappable proteins get string_id = "UNMAPPED".
"""
function resolve_to_string_ids end

"""
    find_merge_candidates(
        name_to_id::Dict{String, NamedTuple},
        data::DataFrame, id_col::Integer
    ) -> Vector{MergeCandidate}

Group rows by canonical STRING ID to find merge candidates
(different names mapping to the same protein).
"""
function find_merge_candidates end

"""
    confirm_merges_interactive(
        candidates::Vector{MergeCandidate};
        auto_approve_threshold::Float64 = 900.0
    ) -> Vector{MergeDecision}

Present merge candidates to user for confirmation.
Auto-approves high-confidence mappings (score > threshold).
Supports: [y]es, [n]o, [a]pprove all, [s]kip all
"""
function confirm_merges_interactive end

"""
    replay_merges(
        candidates::Vector{MergeCandidate},
        report::CurationReport
    ) -> Vector{MergeDecision}

Replay merge decisions from a previous CurationReport.
Deterministic — no user interaction needed.
"""
function replay_merges end

"""
    merge_protein_rows!(data::DataFrame, decisions::Vector{MergeDecision},
        id_col::Integer;
        strategy::Symbol = :max,
        report::CurationReport
    ) -> DataFrame

Merge approved duplicate rows. Strategy:
  :max   — take maximum non-missing value per column (default, preserves signal)
  :mean  — average non-missing values per column
  :first — keep the row with most non-missing values
"""
function merge_protein_rows! end
```

### 3.4 STRING API Client (`src/data/string_api.jl`)

The implementation reuses patterns from the existing `ext/BayesInteractomicsNetworkExt/ppi_query.jl`, which already queries STRING via TSV endpoints. Key design: use the **TSV endpoint** (not JSON) to avoid adding a JSON parsing dependency — the response is parsed with the existing `CSV` dependency.

```julia
# Version-pinned URL (same pattern as ppi_query.jl)
const CURATION_STRING_API_BASE = "https://version-12-0.string-db.org/api"
const CURATION_BATCH_SIZE = 500   # STRING handles up to 2000 per POST
const CURATION_RATE_LIMIT_MS = 1100  # slightly over 1 second

"""
    query_string_ids(names::Vector{String}; species::Int=9606)
    -> DataFrame

Send a batch query to STRING get_string_ids TSV endpoint.
Returns DataFrame with columns: queryItem, stringId, preferredName, annotation.
"""
function query_string_ids(names::Vector{String}; species::Int=9606)
    url = "$CURATION_STRING_API_BASE/tsv/get_string_ids"
    identifiers = join(names, "\r")  # carriage-return separated

    body = "identifiers=$(Downloads.URIs.escapeuri(identifiers))" *
           "&species=$species&limit=1&echo_query=1" *
           "&caller_identity=BayesInteractomics"

    io = IOBuffer()
    Downloads.request(url; method="POST", input=IOBuffer(body), output=io,
                      headers=["Content-Type" => "application/x-www-form-urlencoded"])
    response_str = String(take!(io))

    # Parse TSV with existing CSV dependency (same as ppi_query.jl)
    return CSV.read(IOBuffer(response_str), DataFrame; delim='\t')
end

"""
    resolve_to_string_ids(protein_names::Vector{String};
        species::Int=9606, cache_dir::String="", batch_size::Int=500)
    -> (name_to_id::Dict, id_to_preferred::Dict, unmapped::Vector)

Orchestrator: check cache, query STRING for uncached names, merge results.
"""
function resolve_to_string_ids end

"""Cache key: SHA256 of sorted names + species (same pattern as intermediate_cache.jl)."""
function _curation_cache_key(names::Vector{String}, species::Int)::String
    sorted = sort(names)
    input_str = join(sorted, ",") * "|curation|" * string(species)
    return bytes2hex(sha256(input_str))
end

"""Load/save CurationCache from JLD2 (same pattern as intermediate_cache.jl)."""
function load_string_cache(cache_dir::String, key::String)::Union{CurationCache, Nothing}
    filepath = joinpath(cache_dir, "curation_$(key).jld2")
    isfile(filepath) || return nothing
    return JLD2.load(filepath, "cache")
end

function save_string_cache(cache::CurationCache, cache_dir::String, key::String)
    mkpath(cache_dir)
    filepath = joinpath(cache_dir, "curation_$(key).jld2")
    JLD2.jldsave(filepath; cache=cache)
end
```

**Error handling for network failures**: If STRING is unreachable (timeout, DNS failure, HTTP error), the function warns and returns empty mappings. Curation continues in "offline mode" — splitting still works, but synonym resolution is skipped. This matches the graceful degradation pattern from `ppi_query.jl`.

### 3.5 Integration with `load_data()`

The integration point is in `load_data()` in `src/data/loading.jl`. Curation operates on each raw DataFrame **before** `extract_data()` is called:

```julia
function load_data(
    files::Vector{String},
    sample_cols::Vector{Dict{I,Vector{I}}},
    control_cols::Vector{Dict{I,Vector{I}}},
    name_col::I=1, id_col::I=1, impute::Bool=false;
    normalise_protocols::Bool=false,
    # NEW keyword arguments:
    curate::Bool=false,
    species::Int=9606,
    curate_interactive::Bool=true,
    curate_cache_dir::String=".bayesinteractomics_cache",
    curate_replay::Union{Nothing, String}=nothing,
    bait_name::Union{Nothing, String}=nothing
) where I<:Integer

    # ... existing file loading ...

    for (idx, file) in enumerate(files)
        # Load raw DataFrame (existing logic)
        raw_df = endswith(file, ".csv") ? CSV.read(file, DataFrame) :
                 DataFrame(readtable(file, "Sheet1"))

        # NEW: Optional curation step
        if curate
            raw_df, report = curate_proteins(raw_df, id_col;
                species=species,
                interactive=curate_interactive,
                cache_dir=curate_cache_dir,
                replay_report=curate_replay,
                bait_name=bait_name
            )
            # Save curation report alongside data file
            save_curation_report(report, replace(file, r"\.(csv|xlsx)$" => "_curation_report"))
        end

        # Continue with existing extract_data() logic
        samples[idx], controls[idx], new_ids, new_names =
            extract_data(raw_df, sample_cols[idx], control_cols[idx], name_col, id_col, impute)

        # ... rest unchanged ...
    end
    # ... rest unchanged ...
end
```

**Data flow**: Curation must happen on the raw DataFrame BEFORE `extract_data()` calls `create_protocol()`, because `create_protocol()` uses `Matrix(data[:, cols[i]])` to build data matrices and `protein_ids = string.(data[:, id_col])` to set the ID vector. After curation, the DataFrame has the correct rows and IDs, so all downstream code is unaffected.

**Multi-file consistency**: When `load_data()` processes multiple files (one per protocol), curation happens per-file. Because names are resolved to canonical STRING IDs, the existing `append_unique!()` call correctly merges across files — "HAP40" in file 1 and "F8A1" in file 2 are recognized as the same protein after curation.

### 3.6 Curation Report Persistence

```julia
"""Save curation report as both JLD2 (machine-readable) and CSV (human-readable)."""
function save_curation_report(report::CurationReport, base_path::String)
    # JLD2 for programmatic replay
    JLD2.jldsave(base_path * ".jld2"; report=report)

    # CSV for human inspection
    df = DataFrame(
        original_name = [a.original_name for a in report.actions],
        canonical_name = [a.canonical_name for a in report.actions],
        canonical_id = [a.canonical_id for a in report.actions],
        action = [string(a.action) for a in report.actions],
        reason = [a.reason for a in report.actions],
        user_approved = [a.user_approved for a in report.actions],
        group_id = [something(a.group_id, "") for a in report.actions],
        is_lead = [a.is_lead for a in report.actions]
    )
    CSV.write(base_path * ".csv", df)
end

"""Load a previous CurationReport for replay."""
function load_curation_report(path::String)::CurationReport
    return JLD2.load(path, "report")
end
```

### 3.7 Interactive Terminal UI

```julia
function confirm_merges_interactive(
    candidates::Vector{MergeCandidate};
    auto_approve_threshold::Float64=900.0
)::Vector{MergeDecision}

    decisions = MergeDecision[]
    approve_all = false
    skip_all = false

    for (i, c) in enumerate(candidates)
        # Auto-approve high-confidence single mappings
        if c.string_score >= auto_approve_threshold
            push!(decisions, MergeDecision(c, true, now()))
            continue
        end

        if approve_all
            push!(decisions, MergeDecision(c, true, now()))
            continue
        end
        if skip_all
            push!(decisions, MergeDecision(c, false, now()))
            continue
        end

        # Interactive prompt
        println("\n┌─ Merge Candidate ($i/$(length(candidates))) ──────────")
        println("│ Names found:    $(join(c.names, ", "))")
        println("│ STRING ID:      $(c.canonical_id)")
        println("│ Preferred name: $(c.preferred_name)")
        println("│ Confidence:     $(c.string_score)")
        println("└──────────────────────────────────────────")
        print("  Merge these entries? [y]es / [n]o / [a]pprove all / [s]kip all: ")

        response = lowercase(strip(readline()))
        if response == "y" || response == "yes"
            push!(decisions, MergeDecision(c, true, now()))
        elseif response == "a" || response == "all"
            approve_all = true
            push!(decisions, MergeDecision(c, true, now()))
        elseif response == "s" || response == "skip"
            skip_all = true
            push!(decisions, MergeDecision(c, false, now()))
        else
            push!(decisions, MergeDecision(c, false, now()))
        end
    end

    n_approved = count(d -> d.approved, decisions)
    n_rejected = count(d -> !d.approved, decisions)
    println("\n Summary: $n_approved merges approved, $n_rejected rejected.")
    return decisions
end
```

### 3.8 Dependency Considerations

**No new dependencies required** — this is a key design advantage:

| Dependency   | Status                  | Usage                                      |
| ------------ | ----------------------- | ------------------------------------------ |
| `Downloads`  | stdlib                  | STRING API HTTP requests                   |
| `JLD2`       | already in Project.toml | Cache persistence                          |
| `SHA`        | already in Project.toml | Cache key hashing                          |
| `CSV`        | already in Project.toml | Parse STRING TSV responses + report output |
| `DataFrames` | already in Project.toml | DataFrame manipulation                     |
| `Dates`      | stdlib                  | Timestamps in reports                      |

By using the STRING **TSV endpoint** (not JSON), we parse responses with the existing `CSV` dependency — the same pattern used in `ppi_query.jl`. No need for `JSON3` or any new package.

### 3.9 Exports

Add to `BayesInteractomics.jl`:

```julia
# Data curation (public API)
export curate_proteins, CurationReport, CurationActionType
export split_protein_groups, resolve_to_string_ids
export merge_protein_rows, confirm_merges_interactive
export save_curation_report, load_curation_report
```

### 3.10 Thread Safety

- STRING API calls are sequential (rate-limited) — no concurrency concerns
- DataFrame operations in splitting/merging are single-threaded — they operate on the raw DataFrame **before** any parallel processing begins
- The existing `Threads.@threads` in `analyse()` happens AFTER `load_data()` returns, so curation is always complete before parallelism starts
- JLD2 cache writes use `jldsave()` which is atomic (writes to temp file, then renames)

### 3.11 Test Plan (`test/data/test_curation.jl`)

```julia
@testitem "split_protein_groups: semicolon-separated IDs" begin
    # "A;B;C" → 3 rows with duplicated data, group_id and is_lead tags
end

@testitem "split_protein_groups: no groups present" begin
    # Verify no-op behavior — DataFrame unchanged
end

@testitem "split_protein_groups: slash-separated IDs" begin
    # "RAB5A/RAB5B" treated as protein group
end

@testitem "merge_protein_rows: max strategy" begin
    # Row A = [10, missing, 8], Row B = [missing, 7, 9]
    # Merged = [10, 7, 9] (max per column)
end

@testitem "merge_protein_rows: mean strategy" begin
    # Verify numerical averaging with missing value handling
end

@testitem "merge_protein_rows: protein order preserved" begin
    # CRITICAL: verify row order is deterministic after merge
end

@testitem "merge_protein_rows: bait protein tracked" begin
    # Verify bait protein index is correct after row changes
end

@testitem "resolve_to_string_ids: cache roundtrip" begin
    # Save cache, reload, verify mapping integrity
end

@testitem "curate_proteins: end-to-end non-interactive" begin
    # Full pipeline with small test data, interactive=false
end

@testitem "curation_report: save and reload for replay" begin
    # JLD2 serialization roundtrip
end

@testitem "curate_proteins: contaminant removal" begin
    # CON__, REV__ entries removed
end

@testitem "curate_proteins: UniProt pipe format parsing" begin
    # "sp|P04637|P53_HUMAN" → P04637
end

@testitem "curate_proteins: isoform stripping" begin
    # "P12345-2" → "P12345" for resolution
end
```

Network-dependent tests (actual STRING API calls) should be tagged and skippable:

```julia
@testitem "resolve_to_string_ids: live API" tags=[:network] begin
    # Only runs when explicitly enabled
end
```

---

## 4. Critical Review & Risk Mitigations

### 4.1 CRITICAL: Statistical Inflation from Protein Group Splitting

**Risk**: Splitting `"RBFOX3;RBFOX2;RBFOX1"` into 3 rows creates 3 proteins with **identical data matrices**. This affects all three Bayesian models:

- **Beta-Bernoulli**: Three identical proteins produce three identical `(k_sample, k_control)` tuples → three identical Bayes factors. Not inflation within the model, but creates **three identical votes** in the downstream copula.

- **HBM/Regression**: Similarly produce identical Bayes factors for identical data.

- **Copula combination** (`combined_BF()`): The EM algorithm fits mixture proportions `pi_0`/`pi_1` using ALL proteins. Three identical high-signal entries inflate the H1 population estimate. The `find_H1_initialization_set()` uses quantiles of mean evidence strength — duplicates bias initialization. The EM weights for these three proteins are identical, effectively **triple-counting** one protein's evidence when computing `sum_weights` for the `pi_1` MAP update.

**Net effect**: The `pi_1` estimate (proportion of true interactors) is inflated, which inflates prior odds for ALL proteins in the dataset. This is a **systematic bias**, not just a local error.

**Mandatory mitigation — Group-Aware Deduplication**:

1. After splitting, flag all members with a shared `group_id`
2. Run each member through the Bayesian models normally
3. Before passing to `combined_BF()`, **keep only the lead protein** (first in group) in the `BayesFactorTriplet`
4. Report other group members as "alternative identifications" in a separate column
5. The lead protein carries the group's evidence — others inherit results but don't participate in mixture fitting

### 4.2 CRITICAL: Reference Protein (refID) Corruption

**Risk**: The bait protein is identified by integer index `refID` throughout the pipeline:

- `analyse()` takes `refID::Int=1`
- `combined_BF()` excludes the bait from H0 by index
- `getProteinData(data, refID)` extracts reference protein data for regression
- `computeH0_BayesFactors()` skips the bait during permutation by index

If splitting inserts rows before the bait, or merging removes/shifts rows, the integer index becomes invalid. **The bait protein's data would silently be applied to a different protein.**

**Mandatory mitigation**:

1. `curate_proteins()` requires `bait_name::String` parameter
2. After all row manipulations, recompute `refID = findfirst(==(bait_name), protein_ids)`
3. Validate: `@assert protein_ids[new_refID] == bait_name "Bait protein lost during curation!"`
4. Return the new `refID` alongside the curated DataFrame
5. If the bait protein was part of a protein group, warn the user explicitly

### 4.3 HIGH: Merge Signal Dilution

**Risk**: If protein A (true interactor, high intensity) and protein B (same canonical ID, low-intensity noise) are merged by averaging, the signal is diluted.

**Example**: Row A = `[1000, 800, 1200]`, Row B = `[0, missing, 50]`. Average = `[500, 400, 625]` — dramatically lower log2FC, reducing the HBM Bayes factor.

**Mitigation**: Use **max-pooling** (`:max`) as the default merge strategy instead of averaging:

- For each column: if both rows have values, take the **maximum**
- If only one row has a value, use it (fill gaps)
- Rationale: the higher value reflects the more complete peptide identification
- Preserves signal while still filling gaps in detection

Alternative `:first` strategy: keep the row with the most non-missing values (highest detection count). This is the safest option — it avoids combining potentially inconsistent measurements.

### 4.4 HIGH: Copula Independence Assumption Violation

**Risk**: The copula combination assumes independence across proteins. Split protein groups are perfectly correlated observations (pseudoreplication). If 5% of proteins are in groups of average size 3, ~15% of "proteins" are copies, reducing effective sample size to ~87%.

**Mitigation**: Handled by the Group-Aware Deduplication (Section 4.1). Only the lead protein participates in copula fitting. Other group members inherit the lead's combined Bayes factor and posterior probability.

### 4.5 HIGH: Reproducibility

**Risk**: Interactive decisions are inherently non-reproducible. Different users on the same data produce different curated datasets.

**Mandatory mitigation — Report as both output AND input**:

1. Every curation run produces a `CurationReport` saved as JLD2 + CSV
2. The report records: input file hash, STRING API version, every decision with timestamp
3. `curate_proteins(..., replay_report="path/to/report.jld2")` replays all decisions deterministically
4. Replay mode validates input data hash matches the report's hash; warns if not
5. **No API calls needed** in replay mode — fully offline

### 4.6 MEDIUM: STRING API Reliability

**Risks**: Rate limits, downtime, version changes (~5% of IDs change between major versions), gene name ambiguity.

**Mitigations**:

1. **Mandatory local cache** with configurable TTL (30 days default) in `.bayesinteractomics_cache/string_mappings.jld2`
2. **Graceful degradation**: if STRING is unreachable, use cache-only mode with a warning
3. **Version pinning**: record STRING version in the cache metadata; warn if version differs on re-query
4. **Species validation**: require explicit `species::Int` parameter (no auto-detection)
5. **Rate limiting**: enforce 1.1s sleep between API calls

### 4.7 MEDIUM: Interactive Confirmation Scalability

**Risk**: 50–250 merge prompts cause user fatigue and inconsistent decisions.

**Mitigations**:

1. **Confidence-based auto-approval**: STRING score > 900 → auto-approve (no prompt)
2. **Batch controls**: `[a]pprove all` and `[s]kip all` options in the interactive UI
3. **Non-interactive mode**: `curate_proteins(..., interactive=false)` auto-approves high-confidence, skips low-confidence, generates report for manual review
4. **Pre-computed mapping files**: users can supply a mapping CSV that replays decisions without interaction

### 4.8 MEDIUM: Alternative to STRING — UniProt ID Mapping

| Criterion          | STRING                   | UniProt                                      |
| ------------------ | ------------------------ | -------------------------------------------- |
| Authority          | Genomics consortium      | Protein database gold standard               |
| REST API           | Documented but sparse    | Well-documented, versioned                   |
| ID stability       | Changes between versions | Accessions are permanent                     |
| Gene name handling | Aliases via Ensembl      | Native synonym support                       |
| MS integration     | Indirect                 | Direct (UniProt accessions used by MaxQuant) |

**Recommendation**: Use **STRING as primary** (since the package already uses STRING IDs for the network extension, `poi` parameter, and the user's HAP40 example uses STRING IDs like `9606.ENSP00000479624`). UniProt is a **strong alternative** and should be considered as a future fallback option, but adding both would increase complexity.

### 4.9 Package Scope — Extension vs. Core

**Decision: Include in core `src/data/` directory**, not as a package extension.

**Rationale**:

- No new heavy dependencies required (uses only stdlib + existing deps)
- Curation is tightly coupled with `load_data()` and must handle `refID` tracking
- The package extension pattern (`ext/`) is designed for optional heavy dependencies (Graphs, Cairo, etc.)
- Curation is **opt-in** via `curate=false` default — zero impact if not used

---

## 5. Final Recommended Strategy (Consensus)

After synthesizing all three expert analyses, the team recommends the following approach:

### 5.1 Architecture Decision

**Opt-in preprocessing in `src/data/`** with three new files:

- `curation_types.jl` — type definitions
- `curation.jl` — orchestration and merge logic
- `string_api.jl` — STRING DB client and cache

### 5.2 Processing Pipeline

```
                     ┌─────────────────────┐
                     │   Raw DataFrame(s)   │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │ Remove contaminants  │  CON__, REV__
                     │ (opt-in, default on) │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │ Parse identifiers    │  UniProt pipe format,
                     │                      │  isoform suffixes
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │ Split protein groups │  "A;B;C" → 3 rows
                     │ + slash-separated    │  Tag: group_id, is_lead
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │  STRING API resolve  │  name → string_id
                     │  (cached, batched)   │  preferredName, score
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │ Find merge candidates│  Same string_id,
                     │                      │  different names
                     └──────────┬──────────┘
                                │
                  ┌─────────────┼─────────────┐
                  │             │              │
           ┌──────▼─────┐ ┌────▼────┐ ┌───────▼──────┐
           │ Interactive │ │  Auto-  │ │   Replay     │
           │   confirm   │ │ approve │ │   from       │
           │  (terminal) │ │  (batch)│ │   report     │
           └──────┬──────┘ └────┬────┘ └───────┬──────┘
                  └─────────────┼──────────────┘
                                │
                     ┌──────────▼──────────┐
                     │ Merge approved rows  │  Strategy: :max (default)
                     │ (fill-then-max)      │  :mean, :first available
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │ Recompute refID      │  Track bait by name
                     │ Validate bait intact │
                     └──────────┬──────────┘
                                │
                  ┌─────────────┼─────────────┐
                  │                           │
           ┌──────▼──────┐          ┌─────────▼─────────┐
           │ Curated      │          │ CurationReport     │
           │ DataFrame    │          │ (.jld2 + .csv)     │
           │ → extract_   │          │ Reproducible       │
           │   data()     │          │ replay artifact    │
           └──────────────┘          └───────────────────┘
```

### 5.3 Key Design Decisions

| Decision               | Choice                            | Rationale                                                                 |
| ---------------------- | --------------------------------- | ------------------------------------------------------------------------- |
| Default merge strategy | `:max` (not `:mean`)              | Preserves signal; avoids dilution from noise rows                         |
| Protein group handling | Split + deduplicate before copula | Prevents pi_1 inflation; lead protein carries evidence                    |
| API choice             | STRING (primary)                  | Consistent with existing package usage (STRING IDs in `poi`, network ext) |
| Interactive mode       | Auto-approve above threshold      | Balances thoroughness with usability                                      |
| Reproducibility        | Report as input for replay        | Deterministic re-runs, no API needed                                      |
| Module location        | `src/data/` (core, not extension) | No new heavy deps; tightly coupled with `load_data()`                     |
| Default behavior       | `curate=true`                     | Zero impact on existing users                                             |

### 5.4 User-Facing API

```julia
# Basic usage — interactive curation
data = load_data(
    ["experiment.xlsx"],
    sample_cols, control_cols;
    curate = true,           # Enable curation
    species = 9606,          # Human
    bait_name = "HAP40"      # Track bait through curation
)

# Replay previous curation decisions (reproducible)
data = load_data(
    ["experiment.xlsx"],
    sample_cols, control_cols;
    curate = true,
    curate_replay = "experiment_curation_report.jld2"
)

# Non-interactive batch mode
data = load_data(
    ["experiment.xlsx"],
    sample_cols, control_cols;
    curate = true,
    curate_interactive = false  # Auto-approve high confidence only
)

# Standalone curation (outside load_data)
using DataFrames, CSV
df = CSV.read("experiment.csv", DataFrame)
curated_df, report = curate_proteins(df, 1;  # id_col = 1
    species = 9606,
    bait_name = "HAP40"
)
```

### 5.5 Downstream Analysis Integration

After curation, the `InteractionData` object is constructed normally. However, split protein groups require special handling in `combined_BF()`:

1. Add an optional `protein_groups::Union{Nothing, Dict{String, Vector{String}}}` field to track which proteins came from the same group
2. In `combined_BF()`, if `protein_groups` is provided, include only lead proteins in the `BayesFactorTriplet` for EM fitting
3. After EM fitting, assign combined BF and posterior probabilities to non-lead group members from their lead protein
4. Flag group members in the results DataFrame with a `protein_group` column

---

## Appendix A: STRING API Response Format

Example response from `get_string_ids`:

```json
[
  {
    "queryItem": "HAP40",
    "queryIndex": 0,
    "stringId": "9606.ENSP00000479624",
    "ncbiTaxonId": 9606,
    "taxonName": "Homo sapiens",
    "preferredName": "F8A1",
    "annotation": "Coagulation factor VIII-associated protein 1"
  },
  {
    "queryItem": "TP53",
    "queryIndex": 1,
    "stringId": "9606.ENSP00000269305",
    "ncbiTaxonId": 9606,
    "taxonName": "Homo sapiens",
    "preferredName": "TP53",
    "annotation": "Cellular tumor antigen p53"
  }
]
```

## Appendix B: Curation Report CSV Format

| original_name        | canonical_name | canonical_id         | action | reason                          | user_approved | group_id             | is_lead |
| -------------------- | -------------- | -------------------- | ------ | ------------------------------- | ------------- | -------------------- | ------- |
| RBFOX3;RBFOX2;RBFOX1 | RBFOX3         | 9606.ENSP00000360622 | split  | Protein group split (lead)      | true          | RBFOX3;RBFOX2;RBFOX1 | true    |
| RBFOX3;RBFOX2;RBFOX1 | RBFOX2         | 9606.ENSP00000318890 | split  | Protein group split             | true          | RBFOX3;RBFOX2;RBFOX1 | false   |
| RBFOX3;RBFOX2;RBFOX1 | RBFOX1         | 9606.ENSP00000362447 | split  | Protein group split             | true          | RBFOX3;RBFOX2;RBFOX1 | false   |
| HAP40                | F8A1           | 9606.ENSP00000479624 | merge  | Synonym → STRING preferred name | true          |                      | true    |
| F8A1                 | F8A1           | 9606.ENSP00000479624 | merge  | Synonym → STRING preferred name | true          |                      | true    |
| CON__K2C1_HUMAN      |                |                      | remove | Contaminant (CON__ prefix)      | true          |                      | false   |
| HYPOTHETICAL1        | HYPOTHETICAL1  | UNMAPPED             | keep   | No STRING mapping found         | true          |                      | true    |

## Appendix C: Edge Case Decision Matrix

| Scenario                               | Action                               | Rationale                       |
| -------------------------------------- | ------------------------------------ | ------------------------------- |
| `"RBFOX3;RBFOX2;RBFOX1"`               | Split → 3 rows                       | Standard protein group handling |
| `"RAB5A/RAB5B"`                        | Split → 2 rows                       | Slash-separated ambiguous ID    |
| `"HAP40"` + `"F8A1"` in different rows | Merge (user confirms)                | Same STRING ID                  |
| `"CON__K2C1_HUMAN"`                    | Remove                               | Contaminant                     |
| `"REV__P53_HUMAN"`                     | Remove                               | Decoy sequence                  |
| `"sp\|P04637\|P53_HUMAN"`              | Parse → `P04637` → resolve           | UniProt pipe format             |
| `"P12345-2"`                           | Strip to `P12345` for resolution     | Isoform                         |
| `"LOC12345"`                           | Resolve via STRING; keep if unmapped | Uncharacterized protein         |
| `"Hypothetical protein"`               | Keep, flag as unmapped               | Valid protein, no gene name     |
| Bait protein in group                  | Warn user explicitly                 | refID at risk                   |
| STRING returns no result               | Keep with original name, flag        | Graceful degradation            |
| STRING returns multiple hits           | Use best match (limit=1), flag       | Ambiguity handled by API        |
