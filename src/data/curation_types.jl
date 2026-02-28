#=
Data Curation Types for BayesInteractomics

Type definitions for protein data curation: splitting protein groups,
resolving synonyms via STRING DB, and merging duplicate proteins.
=#

import Dates: DateTime, now

# ─────────────────────────────────────────────────────────────────────────────
# Action Type Enum
# ─────────────────────────────────────────────────────────────────────────────

"""
    CurationActionType

Enum describing the curation action taken for a protein entry.

Values:
- `CURATE_KEEP`:     No change needed — protein retained as-is
- `CURATE_SPLIT`:    Protein group was split into individual rows
- `CURATE_MERGE`:    Multiple rows merged to single canonical ID
- `CURATE_RENAME`:   Name resolved to STRING preferred name
- `CURATE_REMOVE`:   Contaminant or decoy entry removed
- `CURATE_UNMAPPED`: Could not resolve in STRING DB
"""
@enum CurationActionType begin
    CURATE_KEEP
    CURATE_SPLIT
    CURATE_MERGE
    CURATE_RENAME
    CURATE_REMOVE
    CURATE_UNMAPPED
end

# ─────────────────────────────────────────────────────────────────────────────
# Curation Entry — one per protein decision (immutable for audit safety)
# ─────────────────────────────────────────────────────────────────────────────

"""
    CurationEntry

Record of a single curation decision. Immutable for audit trail safety.

# Fields
- `original_name::String`:          Original protein name/ID from the input data
- `canonical_name::String`:         Canonical name after curation (STRING preferred name or original if unmapped)
- `canonical_id::String`:           STRING ID (e.g., "9606.ENSP00000479624") or "" if unmapped
- `action::CurationActionType`:     Type of curation action performed
- `reason::String`:                 Human-readable explanation of the action
- `user_approved::Bool`:            Whether user explicitly confirmed (interactive mode)
- `source_row_indices::Vector{Int}`: Original DataFrame row indices involved
- `group_id::Union{String,Nothing}`: Original protein group identifier, if this entry came from splitting
- `is_lead::Bool`:                  Whether this is the lead protein in its group (first in group)
"""
struct CurationEntry
    original_name::String
    canonical_name::String
    canonical_id::String
    action::CurationActionType
    reason::String
    user_approved::Bool
    source_row_indices::Vector{Int}
    group_id::Union{String, Nothing}
    is_lead::Bool
end

# ─────────────────────────────────────────────────────────────────────────────
# Merge Candidate — presented to user for confirmation
# ─────────────────────────────────────────────────────────────────────────────

"""
    MergeCandidate

A set of rows in the DataFrame that map to the same canonical STRING ID
but have different original names. Presented to the user for merge confirmation.

# Fields
- `names::Vector{String}`:                All distinct protein names mapping to the same canonical ID
- `string_id::String`:                    Canonical STRING identifier
- `preferred_name::String`:               STRING's preferred (official) gene symbol
- `annotation::String`:                   STRING protein function description
- `row_indices::Dict{String,Vector{Int}}`: Maps each name to its row indices in the DataFrame
"""
struct MergeCandidate
    names::Vector{String}
    string_id::String
    preferred_name::String
    annotation::String
    row_indices::Dict{String, Vector{Int}}
end

# ─────────────────────────────────────────────────────────────────────────────
# Merge Decision — user's response to a MergeCandidate
# ─────────────────────────────────────────────────────────────────────────────

"""
    MergeDecision

The user's decision on a [`MergeCandidate`](@ref).

# Fields
- `candidate::MergeCandidate`: The merge candidate that was presented
- `approved::Bool`:            Whether the user approved the merge
- `chosen_name::String`:       Which name to keep as the canonical identifier
"""
struct MergeDecision
    candidate::MergeCandidate
    approved::Bool
    chosen_name::String
end

# ─────────────────────────────────────────────────────────────────────────────
# Curation Report — the key reproducibility artifact
# ─────────────────────────────────────────────────────────────────────────────

"""
    CurationReport

Complete record of a curation run. Saved as JLD2 (machine-readable) and CSV
(human-readable). Can be loaded and replayed for deterministic re-runs.

# Fields
- `entries::Vector{CurationEntry}`:       All curation actions taken
- `merge_decisions::Vector{MergeDecision}`: All merge decisions (approved and rejected)
- `species::Int`:                         NCBI taxonomy ID used for STRING queries
- `string_api_version::String`:           STRING API URL version (e.g., "version-12-0")
- `data_hash::UInt64`:                    Hash of the original input data for replay validation
- `timestamp::DateTime`:                  When this curation was performed
- `package_version::String`:              BayesInteractomics version
- `n_proteins_before::Int`:               Number of protein rows before curation
- `n_proteins_after::Int`:                Number of protein rows after curation
- `summary::Dict{Symbol,Int}`:            Counts — :splits, :merges, :removals, :unmapped, :kept, :renames
"""
struct CurationReport
    entries::Vector{CurationEntry}
    merge_decisions::Vector{MergeDecision}
    species::Int
    string_api_version::String
    data_hash::UInt64
    timestamp::DateTime
    package_version::String
    n_proteins_before::Int
    n_proteins_after::Int
    summary::Dict{Symbol, Int}
end

function Base.show(io::IO, r::CurationReport)
    println(io, "CurationReport ($(r.timestamp))")
    println(io, "────────────────────────────────────")
    println(io, "  Species:           $(r.species)")
    println(io, "  Proteins before:   $(r.n_proteins_before)")
    println(io, "  Proteins after:    $(r.n_proteins_after)")
    println(io, "  Groups split:      $(get(r.summary, :splits, 0))")
    println(io, "  Merges performed:  $(get(r.summary, :merges, 0))")
    println(io, "  Contaminants removed: $(get(r.summary, :removals, 0))")
    println(io, "  Unmapped proteins: $(get(r.summary, :unmapped, 0))")
    println(io, "  Renamed:           $(get(r.summary, :renames, 0))")
    print(io,   "  Kept unchanged:    $(get(r.summary, :kept, 0))")
end

# ─────────────────────────────────────────────────────────────────────────────
# STRING API Cache
# ─────────────────────────────────────────────────────────────────────────────

"""
    CurationCache

Local cache for STRING ID mappings. Stored as JLD2 in the project's
`.bayesinteractomics_cache/` directory. Avoids redundant API calls.

# Fields
- `mapping::Dict{String,String}`:          original_name → STRING_ID
- `preferred_names::Dict{String,String}`:  STRING_ID → preferred_name
- `annotations::Dict{String,String}`:      STRING_ID → annotation text
- `species::Int`:                          NCBI taxonomy ID
- `string_version::String`:                STRING API version string
- `query_timestamp::DateTime`:             When the cache was populated
"""
struct CurationCache
    mapping::Dict{String, String}
    preferred_names::Dict{String, String}
    annotations::Dict{String, String}
    species::Int
    string_version::String
    query_timestamp::DateTime
end
