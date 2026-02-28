#=
Protein Data Curation Pipeline for BayesInteractomics

Handles splitting protein groups, resolving synonyms via STRING DB,
merging duplicate proteins, and removing contaminants. All decisions
are recorded in a CurationReport for reproducibility.
=#

import CSV as CurCSV
import DataFrames: DataFrame, nrow, ncol, names as df_names, select!, insertcols!
import Dates: DateTime, now

# ─────────────────────────────────────────────────────────────────────────────
# Contaminant removal
# ─────────────────────────────────────────────────────────────────────────────

"""
    remove_contaminants(df::DataFrame, id_col::Int) -> (DataFrame, Vector{CurationEntry})

Remove contaminant and decoy entries (CON__, REV__ prefixes) from the DataFrame.
Returns the filtered DataFrame and curation log entries for removed rows.
"""
function remove_contaminants(df::DataFrame, id_col::Int)
    entries = CurationEntry[]
    keep_mask = trues(nrow(df))

    for i in 1:nrow(df)
        name = string(df[i, id_col])
        if startswith(uppercase(name), "CON__") || startswith(uppercase(name), "REV__")
            keep_mask[i] = false
            push!(entries, CurationEntry(
                name, "", "", CURATE_REMOVE,
                "Contaminant/decoy entry removed (prefix: $(first(name, 5)))",
                false, [i], nothing, false
            ))
        end
    end

    return df[keep_mask, :], entries
end

# ─────────────────────────────────────────────────────────────────────────────
# Protein ID parsing
# ─────────────────────────────────────────────────────────────────────────────

"""
    parse_protein_id(id::AbstractString) -> String

Parse a protein identifier, handling common formats:
- UniProt pipe format: `sp|P04637|P53_HUMAN` → `P53_HUMAN`
- UniProt accession with isoform: `P12345-2` → `P12345`
- Plain gene symbol: returned as-is
"""
function parse_protein_id(id::AbstractString)::String
    s = strip(string(id))

    # UniProt pipe format: sp|ACCESSION|ENTRY_NAME or tr|ACCESSION|ENTRY_NAME
    if occursin('|', s)
        parts = split(s, '|')
        if length(parts) >= 3 && parts[1] in ("sp", "tr")
            # Prefer entry name (3rd part), fall back to accession (2nd part)
            entry_name = strip(String(parts[3]))
            if !isempty(entry_name)
                return entry_name
            end
            accession = strip(String(parts[2]))
            if !isempty(accession)
                return accession
            end
        end
        # Fallback: return last non-empty part
        for p in reverse(parts)
            !isempty(strip(p)) && return String(strip(p))
        end
    end

    # Strip isoform suffix: P12345-2 → P12345
    # Only for strict UniProt accession format:
    #   [OPQ][0-9][A-Z0-9]{3}[0-9] (6 chars) or [A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9] (6 chars)
    #   Extended: [A-Z][0-9][A-Z0-9]{3}[0-9][A-Z0-9]{0,4} (6-10 chars)
    # Requires at least one digit after the dash
    m = match(r"^([OPQ][0-9][A-Z0-9]{3}[0-9])-\d+$", s)
    if m !== nothing
        return String(m.captures[1])
    end
    m = match(r"^([A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])-\d+$", s)
    if m !== nothing
        return String(m.captures[1])
    end

    return String(s)
end

# ─────────────────────────────────────────────────────────────────────────────
# Protein group splitting
# ─────────────────────────────────────────────────────────────────────────────

"""
    split_protein_groups(df::DataFrame, id_col::Int; delimiter=";") -> (DataFrame, Vector{CurationEntry})

Split protein groups (e.g., `"RBFOX3;RBFOX2;RBFOX1"`) into individual rows,
duplicating all data columns. The first protein in each group is marked as lead.

Returns the expanded DataFrame and curation log entries.
"""
function split_protein_groups(df::DataFrame, id_col::Int; delimiter::String=";")
    entries = CurationEntry[]
    col_names = df_names(df)
    id_colname = col_names[id_col]

    # Build result as column vectors for efficiency
    result_cols = Dict{String, Vector{Any}}(cn => Any[] for cn in col_names)

    for i in 1:nrow(df)
        raw_name = string(df[i, id_col])
        parts = strip.(split(raw_name, delimiter))
        parts = filter(!isempty, parts)

        if length(parts) <= 1
            # No group — keep row as-is
            for cn in col_names
                push!(result_cols[cn], df[i, cn])
            end
            push!(entries, CurationEntry(
                raw_name, parse_protein_id(raw_name), "",
                CURATE_KEEP, "Single protein, no splitting needed",
                false, [i], nothing, true
            ))
        else
            # Protein group — split into individual rows
            group_id = raw_name
            for (j, part) in enumerate(parts)
                parsed = parse_protein_id(part)
                for cn in col_names
                    if cn == id_colname
                        push!(result_cols[cn], parsed)
                    else
                        push!(result_cols[cn], df[i, cn])
                    end
                end
                push!(entries, CurationEntry(
                    raw_name, parsed, "",
                    CURATE_SPLIT,
                    "Split from group '$(raw_name)' (member $j/$(length(parts)))",
                    false, [i], group_id, j == 1
                ))
            end
        end
    end

    # Build DataFrame preserving original column order
    result_df = DataFrame([cn => result_cols[cn] for cn in col_names])
    return result_df, entries
end

# ─────────────────────────────────────────────────────────────────────────────
# STRING ID resolution (orchestrator with caching)
# ─────────────────────────────────────────────────────────────────────────────

"""
    resolve_to_string_ids(names::Vector{String}; species=9606, cache_dir="") -> NamedTuple

Resolve protein names to STRING IDs, using a local cache when available.

Returns a NamedTuple with:
- `name_to_id`, `id_to_preferred`, `id_to_annotation`, `unmapped`
- `cache_used::Bool`
"""
function resolve_to_string_ids(
    names::Vector{String};
    species::Int = 9606,
    cache_dir::String = ""
)
    unique_names = unique(names)
    key = _curation_cache_key(unique_names, species)

    # Try cache first
    if !isempty(cache_dir)
        cached = load_curation_cache(cache_dir, key)
        if !isnothing(cached) && cached.species == species
            @info "Using cached STRING ID mappings" n_cached=length(cached.mapping) age_days=div(
                Dates.value(Dates.Millisecond(now() - cached.query_timestamp)), 86_400_000
            )
            # Determine unmapped from cache
            unmapped = [n for n in unique_names if !haskey(cached.mapping, n)]
            return (
                name_to_id = cached.mapping,
                id_to_preferred = cached.preferred_names,
                id_to_annotation = cached.annotations,
                unmapped = unmapped,
                cache_used = true
            )
        end
    end

    # Query STRING API
    result = _resolve_names_via_string(unique_names, species)

    # Save to cache
    if !isempty(cache_dir)
        cache = CurationCache(
            result.name_to_id,
            result.id_to_preferred,
            result.id_to_annotation,
            species,
            "version-12-0",
            now()
        )
        save_curation_cache(cache, cache_dir, key)
    end

    return (
        name_to_id = result.name_to_id,
        id_to_preferred = result.id_to_preferred,
        id_to_annotation = result.id_to_annotation,
        unmapped = result.unmapped,
        cache_used = false
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Merge candidate identification
# ─────────────────────────────────────────────────────────────────────────────

"""
    find_merge_candidates(name_to_id, id_to_preferred, id_to_annotation, df, id_col) -> Vector{MergeCandidate}

Identify groups of rows that map to the same STRING ID but have different names.
These are candidates for merging (they represent the same protein under different synonyms).
"""
function find_merge_candidates(
    name_to_id::Dict{String,String},
    id_to_preferred::Dict{String,String},
    id_to_annotation::Dict{String,String},
    df::DataFrame,
    id_col::Int
)::Vector{MergeCandidate}
    # Group rows by STRING ID
    id_to_rows = Dict{String, Dict{String, Vector{Int}}}()

    for i in 1:nrow(df)
        name = string(df[i, id_col])
        string_id = get(name_to_id, name, nothing)
        isnothing(string_id) && continue

        if !haskey(id_to_rows, string_id)
            id_to_rows[string_id] = Dict{String, Vector{Int}}()
        end
        if !haskey(id_to_rows[string_id], name)
            id_to_rows[string_id][name] = Int[]
        end
        push!(id_to_rows[string_id][name], i)
    end

    # Only keep groups with multiple distinct names
    candidates = MergeCandidate[]
    for (string_id, name_rows) in id_to_rows
        length(name_rows) <= 1 && continue

        names_list = sort(collect(keys(name_rows)))
        preferred = get(id_to_preferred, string_id, first(names_list))
        annotation = get(id_to_annotation, string_id, "")

        push!(candidates, MergeCandidate(
            names_list,
            string_id,
            preferred,
            annotation,
            name_rows
        ))
    end

    # Sort by number of names (most complex first) for user review
    sort!(candidates, by=c -> length(c.names), rev=true)
    return candidates
end

# ─────────────────────────────────────────────────────────────────────────────
# Interactive merge confirmation (terminal UI)
# ─────────────────────────────────────────────────────────────────────────────

"""
    confirm_merges_interactive(candidates; auto_approve_threshold=0) -> Vector{MergeDecision}

Present merge candidates to the user for interactive confirmation.
User can respond with:
- `y` / `Enter` — approve merge using STRING preferred name
- `n` — reject merge (keep rows separate)
- `a` — approve all remaining candidates
- `s` — skip all remaining (reject all remaining)
- A number (1-N) — approve merge but use the Nth name as canonical

`auto_approve_threshold`: if > 0, auto-approve candidates where all names share
the same first N characters (0 = always ask).
"""
function confirm_merges_interactive(
    candidates::Vector{MergeCandidate};
    auto_approve_threshold::Int = 0
)::Vector{MergeDecision}
    decisions = MergeDecision[]
    approve_all = false
    skip_all = false

    for (idx, candidate) in enumerate(candidates)
        if skip_all
            push!(decisions, MergeDecision(candidate, false, ""))
            continue
        end

        if approve_all
            push!(decisions, MergeDecision(candidate, true, candidate.preferred_name))
            continue
        end

        # Auto-approve if names share a long common prefix
        if auto_approve_threshold > 0
            prefixes = [first(n, auto_approve_threshold) for n in candidate.names]
            if length(unique(prefixes)) == 1
                push!(decisions, MergeDecision(candidate, true, candidate.preferred_name))
                @info "Auto-approved merge" names=candidate.names preferred=candidate.preferred_name
                continue
            end
        end

        # Print merge candidate info
        println("\n┌─────────────────────────────────────────────────────")
        println("│ Merge candidate $(idx)/$(length(candidates))")
        println("│ STRING ID:       $(candidate.string_id)")
        println("│ Preferred name:  $(candidate.preferred_name)")
        if !isempty(candidate.annotation)
            println("│ Annotation:      $(candidate.annotation)")
        end
        println("│")
        println("│ The following names map to the same protein:")
        for (j, name) in enumerate(candidate.names)
            n_rows = length(candidate.row_indices[name])
            println("│   $(j). $(name) ($(n_rows) row$(n_rows > 1 ? "s" : ""))")
        end
        println("│")
        println("│ Merge these into '$(candidate.preferred_name)'?")
        println("│ [y/Enter] approve  [n] reject  [a] approve all  [s] skip all  [1-$(length(candidate.names))] use Nth name")
        println("└─────────────────────────────────────────────────────")
        print("  > ")

        response = strip(lowercase(readline()))

        if response == "" || response == "y"
            push!(decisions, MergeDecision(candidate, true, candidate.preferred_name))
        elseif response == "n"
            push!(decisions, MergeDecision(candidate, false, ""))
        elseif response == "a"
            approve_all = true
            push!(decisions, MergeDecision(candidate, true, candidate.preferred_name))
        elseif response == "s"
            skip_all = true
            push!(decisions, MergeDecision(candidate, false, ""))
        else
            # Try to parse as number
            num = tryparse(Int, response)
            if !isnothing(num) && 1 <= num <= length(candidate.names)
                chosen = candidate.names[num]
                push!(decisions, MergeDecision(candidate, true, chosen))
            else
                @warn "Invalid input '$(response)', defaulting to approve with preferred name"
                push!(decisions, MergeDecision(candidate, true, candidate.preferred_name))
            end
        end
    end

    return decisions
end

# ─────────────────────────────────────────────────────────────────────────────
# Replay merges from saved report
# ─────────────────────────────────────────────────────────────────────────────

"""
    replay_merges(candidates::Vector{MergeCandidate}, report::CurationReport) -> Vector{MergeDecision}

Replay merge decisions from a previously saved curation report.
Matches candidates to saved decisions by STRING ID.
"""
function replay_merges(
    candidates::Vector{MergeCandidate},
    report::CurationReport
)::Vector{MergeDecision}
    # Build lookup: string_id → MergeDecision from saved report
    saved = Dict{String, MergeDecision}()
    for d in report.merge_decisions
        saved[d.candidate.string_id] = d
    end

    decisions = MergeDecision[]
    for candidate in candidates
        if haskey(saved, candidate.string_id)
            old = saved[candidate.string_id]
            push!(decisions, MergeDecision(candidate, old.approved, old.chosen_name))
        else
            @warn "No saved decision for merge candidate $(candidate.string_id), defaulting to reject"
            push!(decisions, MergeDecision(candidate, false, ""))
        end
    end

    return decisions
end

# ─────────────────────────────────────────────────────────────────────────────
# Row merging
# ─────────────────────────────────────────────────────────────────────────────

"""
    merge_protein_rows(df::DataFrame, decisions::Vector{MergeDecision}, id_col::Int;
                       strategy::Symbol=:max) -> (DataFrame, Vector{CurationEntry})

Merge rows for approved merge decisions. Data columns are combined using the
specified strategy (`:max` or `:mean`).

- `:max` — take the maximum non-missing value (preserves signal)
- `:mean` — take the mean of non-missing values
"""
function merge_protein_rows(
    df::DataFrame,
    decisions::Vector{MergeDecision},
    id_col::Int;
    strategy::Symbol = :max
)
    strategy in (:max, :mean) || throw(ArgumentError("strategy must be :max or :mean, got :$strategy"))

    entries = CurationEntry[]
    rows_to_remove = Set{Int}()
    col_names = df_names(df)
    id_colname = col_names[id_col]

    # Determine which columns are numeric data
    data_cols = Int[]
    for c in 1:ncol(df)
        c == id_col && continue
        eltype(df[!, c]) <: Union{Missing, Number} && push!(data_cols, c)
    end

    for decision in decisions
        if !decision.approved
            # Keep all rows as-is, mark as CURATE_KEEP
            for (name, indices) in decision.candidate.row_indices
                for idx in indices
                    push!(entries, CurationEntry(
                        name, name, decision.candidate.string_id,
                        CURATE_KEEP, "Merge rejected by user",
                        true, [idx], nothing, false
                    ))
                end
            end
            continue
        end

        # Collect all row indices for this merge group
        all_indices = Int[]
        for (_, indices) in decision.candidate.row_indices
            append!(all_indices, indices)
        end
        sort!(all_indices)

        if length(all_indices) <= 1
            # Nothing to merge
            if !isempty(all_indices)
                idx = all_indices[1]
                old_name = string(df[idx, id_col])
                push!(entries, CurationEntry(
                    old_name, decision.chosen_name, decision.candidate.string_id,
                    old_name == decision.chosen_name ? CURATE_KEEP : CURATE_RENAME,
                    "Single row, renamed to '$(decision.chosen_name)'",
                    true, [idx], nothing, true
                ))
                df[idx, id_colname] = decision.chosen_name
            end
            continue
        end

        # Keep the first row, merge data from subsequent rows into it
        keep_idx = all_indices[1]
        remove_indices = all_indices[2:end]

        # Merge data columns
        for c in data_cols
            values = [df[idx, c] for idx in all_indices]
            non_missing = filter(!ismissing, values)

            if isempty(non_missing)
                df[keep_idx, c] = missing
            elseif strategy == :max
                df[keep_idx, c] = maximum(non_missing)
            else  # :mean
                df[keep_idx, c] = mean(non_missing)
            end
        end

        # Rename kept row
        old_name = string(df[keep_idx, id_col])
        df[keep_idx, id_colname] = decision.chosen_name

        # Log entries
        for (name, indices) in decision.candidate.row_indices
            for idx in indices
                push!(entries, CurationEntry(
                    name, decision.chosen_name, decision.candidate.string_id,
                    CURATE_MERGE,
                    "Merged to '$(decision.chosen_name)' using :$(strategy) strategy",
                    true, [idx], nothing, idx == keep_idx
                ))
            end
        end

        # Mark rows for removal (all except the first)
        union!(rows_to_remove, remove_indices)
    end

    # Remove merged rows
    keep_mask = trues(nrow(df))
    for idx in rows_to_remove
        keep_mask[idx] = false
    end

    return df[keep_mask, :], entries
end

# ─────────────────────────────────────────────────────────────────────────────
# Rename unmapped / renamed proteins
# ─────────────────────────────────────────────────────────────────────────────

"""
    apply_renames!(df, id_col, name_to_id, id_to_preferred) -> Vector{CurationEntry}

For proteins that mapped to STRING but weren't part of a merge group,
rename to the STRING preferred name and log the action.
"""
function apply_renames!(
    df::DataFrame,
    id_col::Int,
    name_to_id::Dict{String,String},
    id_to_preferred::Dict{String,String},
    merged_names::Set{String}
)::Vector{CurationEntry}
    entries = CurationEntry[]
    col_names = df_names(df)
    id_colname = col_names[id_col]

    for i in 1:nrow(df)
        name = string(df[i, id_col])
        name in merged_names && continue

        string_id = get(name_to_id, name, nothing)

        if isnothing(string_id)
            push!(entries, CurationEntry(
                name, name, "",
                CURATE_UNMAPPED,
                "Could not resolve in STRING DB",
                false, [i], nothing, false
            ))
        else
            preferred = get(id_to_preferred, string_id, name)
            if preferred != name
                df[i, id_colname] = preferred
                push!(entries, CurationEntry(
                    name, preferred, string_id,
                    CURATE_RENAME,
                    "Renamed from '$(name)' to STRING preferred name '$(preferred)'",
                    false, [i], nothing, true
                ))
            else
                push!(entries, CurationEntry(
                    name, name, string_id,
                    CURATE_KEEP,
                    "Already using STRING preferred name",
                    false, [i], nothing, true
                ))
            end
        end
    end

    return entries
end

# ─────────────────────────────────────────────────────────────────────────────
# Bait protein tracking
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Post-curation deduplication of identical protein names
# ─────────────────────────────────────────────────────────────────────────────

"""
    _deduplicate_same_name_rows(df, id_col; strategy=:max) -> (DataFrame, Vector{CurationEntry})

After splitting protein groups and renaming to canonical names, multiple rows may
share the same protein name. This function merges those rows using the specified
strategy (`:max` or `:mean`) for numeric columns.

This is distinct from the STRING-based synonym merge (Phase 5-6) which handles
proteins with *different* names mapping to the same STRING ID. This handles
proteins that already have the *same* name after all renaming is done.
"""
function _deduplicate_same_name_rows(
    df::DataFrame,
    id_col::Int;
    strategy::Symbol = :max
)
    entries = CurationEntry[]
    col_names = df_names(df)
    id_colname = col_names[id_col]

    # Find which columns are numeric data
    data_cols = Int[]
    for c in 1:ncol(df)
        c == id_col && continue
        eltype(df[!, c]) <: Union{Missing, Number} && push!(data_cols, c)
    end

    # Group rows by protein name
    name_to_rows = Dict{String, Vector{Int}}()
    for i in 1:nrow(df)
        name = string(df[i, id_col])
        if !haskey(name_to_rows, name)
            name_to_rows[name] = Int[]
        end
        push!(name_to_rows[name], i)
    end

    # Check if any deduplication is needed
    max_count = maximum(length(v) for v in values(name_to_rows))
    if max_count <= 1
        return df, entries  # Nothing to deduplicate
    end

    rows_to_remove = Set{Int}()

    for (name, indices) in name_to_rows
        length(indices) <= 1 && continue

        # Keep first row, merge data from duplicates into it
        keep_idx = indices[1]
        remove_indices = indices[2:end]

        for c in data_cols
            values_vec = [df[idx, c] for idx in indices]
            non_missing = filter(!ismissing, values_vec)

            if isempty(non_missing)
                df[keep_idx, c] = missing
            elseif strategy == :max
                df[keep_idx, c] = maximum(non_missing)
            else  # :mean
                df[keep_idx, c] = mean(non_missing)
            end
        end

        # Log the deduplication
        for idx in indices
            push!(entries, CurationEntry(
                name, name, "",
                CURATE_MERGE,
                "Deduplicated: $(length(indices)) identical rows merged to 1 using :$(strategy)",
                false, [idx], nothing, idx == keep_idx
            ))
        end

        union!(rows_to_remove, remove_indices)
    end

    # Remove duplicate rows
    keep_mask = trues(nrow(df))
    for idx in rows_to_remove
        keep_mask[idx] = false
    end

    return df[keep_mask, :], entries
end

# ─────────────────────────────────────────────────────────────────────────────
# Bait protein tracking
# ─────────────────────────────────────────────────────────────────────────────

"""
    find_bait_index(df::DataFrame, id_col::Int, bait_name::String) -> Union{Int, Nothing}

Find the row index of the bait protein after curation.
Searches both the ID column and checks for STRING ID matches.
"""
function find_bait_index(df::DataFrame, id_col::Int, bait_name::String)::Union{Int, Nothing}
    for i in 1:nrow(df)
        name = string(df[i, id_col])
        if name == bait_name || lowercase(name) == lowercase(bait_name)
            return i
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ─────────────────────────────────────────────────────────────────────────────

"""
    curate_proteins(df::DataFrame, id_col::Int; kwargs...) -> (DataFrame, CurationReport, Union{Int, Nothing})

Main curation pipeline. Processes a raw proteomics DataFrame through:
1. Contaminant removal (CON__, REV__)
2. Protein group splitting (semicolons)
3. STRING ID resolution (API + cache)
4. Merge candidate identification
5. Interactive merge confirmation (or replay from saved report)
6. Row merging
7. Protein renaming to canonical names

# Keyword Arguments
- `species::Int=9606`: NCBI taxonomy ID
- `interactive::Bool=true`: Whether to prompt user for merge confirmation
- `cache_dir::String=""`: Directory for STRING API cache
- `replay_report::Union{Nothing,CurationReport}=nothing`: Replay decisions from saved report
- `merge_strategy::Symbol=:max`: How to combine data when merging (`:max` or `:mean`)
- `bait_name::Union{Nothing,String}=nothing`: Bait protein name for refID tracking
- `do_remove_contaminants::Bool=true`: Whether to remove CON__/REV__ entries
- `delimiter::String=";"`: Delimiter for protein groups
- `auto_approve_threshold::Int=0`: Auto-approve merges with shared prefix length

# Returns
- `(curated_df, report, bait_index)` where `bait_index` is the new position of the bait
"""
function curate_proteins(
    df::DataFrame,
    id_col::Int;
    species::Int = 9606,
    interactive::Bool = true,
    cache_dir::String = "",
    replay_report::Union{Nothing, CurationReport} = nothing,
    merge_strategy::Symbol = :max,
    bait_name::Union{Nothing, String} = nothing,
    do_remove_contaminants::Bool = true,
    delimiter::String = ";",
    auto_approve_threshold::Int = 0
)
    n_before = nrow(df)
    all_entries = CurationEntry[]
    data_hash = hash(df)
    df = copy(df)  # Don't mutate the input

    @info "Starting protein curation" n_proteins=n_before species=species

    # ── Phase 1: Remove contaminants ──────────────────────────────────────
    if do_remove_contaminants
        df, contam_entries = remove_contaminants(df, id_col)
        append!(all_entries, contam_entries)
        n_removed = length(contam_entries)
        n_removed > 0 && @info "Removed $n_removed contaminant/decoy entries"
    end

    # ── Phase 2: Split protein groups ─────────────────────────────────────
    df, split_entries = split_protein_groups(df, id_col; delimiter=delimiter)
    # Only keep CURATE_SPLIT entries (CURATE_KEEP from splitting is preliminary)
    split_only = filter(e -> e.action == CURATE_SPLIT, split_entries)
    append!(all_entries, split_only)
    n_split = count(e -> e.action == CURATE_SPLIT && e.is_lead, split_entries)
    n_split > 0 && @info "Split $n_split protein groups into individual entries ($(nrow(df)) rows)"

    # ── Phase 3: Resolve via STRING ───────────────────────────────────────
    protein_names = String.(unique(string.(df[:, id_col])))
    resolution = try
        resolve_to_string_ids(protein_names; species=species, cache_dir=cache_dir)
    catch e
        @warn "STRING ID resolution failed, continuing without synonym resolution" exception=e
        (
            name_to_id = Dict{String,String}(),
            id_to_preferred = Dict{String,String}(),
            id_to_annotation = Dict{String,String}(),
            unmapped = protein_names,
            cache_used = false
        )
    end
    n_mapped = length(resolution.name_to_id)
    n_unmapped = length(resolution.unmapped)
    @info "STRING ID resolution" mapped=n_mapped unmapped=n_unmapped cache_used=resolution.cache_used

    # ── Phase 4: Find merge candidates ────────────────────────────────────
    candidates = find_merge_candidates(
        resolution.name_to_id,
        resolution.id_to_preferred,
        resolution.id_to_annotation,
        df, id_col
    )

    # ── Phase 5: Confirm or replay merges ─────────────────────────────────
    merge_decisions = MergeDecision[]
    if !isempty(candidates)
        if !isnothing(replay_report)
            merge_decisions = replay_merges(candidates, replay_report)
            @info "Replayed $(length(merge_decisions)) merge decisions from saved report"
        elseif interactive
            @info "Found $(length(candidates)) merge candidate(s) requiring confirmation"
            merge_decisions = confirm_merges_interactive(
                candidates; auto_approve_threshold=auto_approve_threshold
            )
        else
            # Non-interactive: auto-approve all
            merge_decisions = [MergeDecision(c, true, c.preferred_name) for c in candidates]
            @info "Auto-approved $(length(merge_decisions)) merge(s) (non-interactive mode)"
        end
    end

    # ── Phase 6: Apply merges ─────────────────────────────────────────────
    merged_names = Set{String}()
    if !isempty(merge_decisions)
        df, merge_entries = merge_protein_rows(df, merge_decisions, id_col; strategy=merge_strategy)
        append!(all_entries, merge_entries)
        n_merged = count(d -> d.approved, merge_decisions)
        n_merged > 0 && @info "Merged $n_merged protein group(s)"

        # Track which names were handled by merge decisions
        for d in merge_decisions
            for name in keys(d.candidate.row_indices)
                push!(merged_names, name)
            end
        end
    end

    # ── Phase 7: Rename remaining proteins ────────────────────────────────
    rename_entries = apply_renames!(
        df, id_col,
        resolution.name_to_id,
        resolution.id_to_preferred,
        merged_names
    )
    append!(all_entries, rename_entries)
    n_renamed = count(e -> e.action == CURATE_RENAME, rename_entries)
    n_renamed > 0 && @info "Renamed $n_renamed protein(s) to STRING preferred names"

    # ── Phase 8: Deduplicate same-name rows ───────────────────────────────
    # After splitting groups and renaming to canonical names, multiple rows
    # may now share the same protein name. Merge them using the configured strategy.
    n_before_dedup = nrow(df)
    df, dedup_entries = _deduplicate_same_name_rows(df, id_col; strategy=merge_strategy)
    append!(all_entries, dedup_entries)
    n_deduped = n_before_dedup - nrow(df)
    n_deduped > 0 && @info "Deduplicated $n_deduped rows with identical protein names ($(nrow(df)) unique proteins remain)"

    # ── Build summary ─────────────────────────────────────────────────────
    summary = Dict{Symbol, Int}(
        :splits => count(e -> e.action == CURATE_SPLIT, all_entries),
        :merges => count(e -> e.action == CURATE_MERGE, all_entries),
        :removals => count(e -> e.action == CURATE_REMOVE, all_entries),
        :unmapped => count(e -> e.action == CURATE_UNMAPPED, all_entries),
        :renames => count(e -> e.action == CURATE_RENAME, all_entries),
        :kept => count(e -> e.action == CURATE_KEEP, all_entries),
    )

    report = CurationReport(
        all_entries,
        merge_decisions,
        species,
        "version-12-0",
        data_hash,
        now(),
        string(pkgversion(@__MODULE__)),
        n_before,
        nrow(df),
        summary
    )

    # ── Find bait protein ─────────────────────────────────────────────────
    bait_idx = nothing
    if !isnothing(bait_name)
        bait_idx = find_bait_index(df, id_col, bait_name)
        if isnothing(bait_idx)
            # Try with preferred name from STRING
            bait_string_id = get(resolution.name_to_id, bait_name, nothing)
            if !isnothing(bait_string_id)
                preferred = get(resolution.id_to_preferred, bait_string_id, nothing)
                if !isnothing(preferred)
                    bait_idx = find_bait_index(df, id_col, preferred)
                end
            end
            # Also try the STRING ID directly
            if isnothing(bait_idx) && !isnothing(bait_string_id)
                bait_idx = find_bait_index(df, id_col, bait_string_id)
            end
        end
        if isnothing(bait_idx)
            @warn "Bait protein '$(bait_name)' not found after curation! Check your bait_name parameter."
        else
            @info "Bait protein '$(bait_name)' found at row $bait_idx after curation"
        end
    end

    @info "Curation complete" proteins_before=n_before proteins_after=nrow(df) report

    return df, report, bait_idx
end

# ─────────────────────────────────────────────────────────────────────────────
# Report persistence
# ─────────────────────────────────────────────────────────────────────────────

"""
    save_curation_report(report::CurationReport, base_path::String)

Save curation report as both JLD2 (machine-readable, for replay) and CSV (human-readable).

Files created:
- `{base_path}_curation_report.jld2`
- `{base_path}_curation_log.csv`
"""
function save_curation_report(report::CurationReport, base_path::String)
    # ── JLD2 (full report for replay) ─────────────────────────────────────
    jld2_path = base_path * "_curation_report.jld2"
    try
        jldsave(jld2_path; compress=true,
            entries = report.entries,
            merge_decisions = report.merge_decisions,
            species = report.species,
            string_api_version = report.string_api_version,
            data_hash = report.data_hash,
            timestamp = string(report.timestamp),
            package_version = report.package_version,
            n_proteins_before = report.n_proteins_before,
            n_proteins_after = report.n_proteins_after,
            summary = report.summary
        )
        @info "Saved curation report (JLD2)" path=jld2_path
    catch e
        @warn "Failed to save JLD2 curation report" path=jld2_path exception=e
    end

    # ── CSV (human-readable log) ──────────────────────────────────────────
    csv_path = base_path * "_curation_log.csv"
    try
        log_df = DataFrame(
            original_name = [e.original_name for e in report.entries],
            canonical_name = [e.canonical_name for e in report.entries],
            canonical_id = [e.canonical_id for e in report.entries],
            action = [string(e.action) for e in report.entries],
            reason = [e.reason for e in report.entries],
            user_approved = [e.user_approved for e in report.entries],
            group_id = [something(e.group_id, "") for e in report.entries],
            is_lead = [e.is_lead for e in report.entries],
        )
        CurCSV.write(csv_path, log_df)
        @info "Saved curation log (CSV)" path=csv_path
    catch e
        @warn "Failed to save CSV curation log" path=csv_path exception=e
    end
end

"""
    load_curation_report(path::String) -> Union{CurationReport, Nothing}

Load a curation report from JLD2 file for replay.
"""
function load_curation_report(path::String)::Union{CurationReport, Nothing}
    !isfile(path) && return nothing

    try
        data = JLD2.load(path)

        timestamp = data["timestamp"]
        if timestamp isa String
            timestamp = DateTime(timestamp)
        end

        return CurationReport(
            data["entries"],
            data["merge_decisions"],
            data["species"],
            data["string_api_version"],
            data["data_hash"],
            timestamp,
            data["package_version"],
            data["n_proteins_before"],
            data["n_proteins_after"],
            data["summary"]
        )
    catch e
        @warn "Failed to load curation report" path exception=e
        return nothing
    end
end
