# STRING-ID ↔ UniProt-accession mapping for the TR/DDI feature builders.
#
# Reads `9606.protein.aliases.v12.0.txt` (STRING ships this with three columns:
# #string_protein_id, alias, source). The "source" column carries the
# provider of the alias — we filter to rows whose source contains "UniProt"
# (covers UniProt_AC, BLAST_UniProt_AC, Ensembl_UniProt_AC). When a single
# STRING-ID maps to multiple UniProt accessions (typical for isoforms /
# multiple xrefs), we deterministically pick the alphabetically-first
# accession to keep the mapping reproducible across runs.

module ProteinIDMapping

using DataFrames

export load_string_to_uniprot, ALIAS_SOURCES_PREFERRED

# Source-substrings that indicate a UniProt-style alias, in priority order
# (lower index = preferred). The most common label in STRING v12 aliases is
# `Ensembl_UniProt` (no `_AC` suffix); the original 3-entry list missed it
# and yielded only 65% mapping coverage on Spike-009's 3000-protein set.
const ALIAS_SOURCES_PREFERRED = [
    "UniProt_AC",          # reviewed SwissProt primary accession
    "BLAST_UniProt_AC",
    "Ensembl_UniProt",     # any UniProt (reviewed or TrEMBL) — most common
]

"""
    load_string_to_uniprot(alias_tsv::String;
                           restrict_to::Union{Nothing, AbstractSet{<:AbstractString}}=nothing)
        -> (string_to_up::Dict{String, String},
            up_to_string::Dict{String, String},
            coverage::NamedTuple)

Build the STRING→UniProt mapping. If `restrict_to` is supplied (a set of
STRING-IDs we care about), only mappings for those IDs are returned and
coverage is computed against that set.
"""
function load_string_to_uniprot(alias_tsv::String;
                                 restrict_to::Union{Nothing, AbstractSet{<:AbstractString}}=nothing)
    # Multi-priority collector: for each STRING-ID keep the best UniProt
    # source seen so far. Lower priority index = better source.
    best_priority = Dict{String, Int}()
    string_to_up = Dict{String, String}()

    n_total = 0
    open(alias_tsv, "r") do io
        for line in eachline(io)
            startswith(line, "#") && continue
            isempty(line) && continue
            fields = split(line, '\t')
            length(fields) < 3 && continue
            string_id = String(fields[1])
            alias = String(fields[2])
            source = String(fields[3])
            if restrict_to !== nothing && !(string_id in restrict_to)
                continue
            end
            # Find priority
            prio = 0
            for (i, key) in enumerate(ALIAS_SOURCES_PREFERRED)
                if occursin(key, source)
                    prio = i; break
                end
            end
            prio == 0 && continue
            n_total += 1
            current = get(best_priority, string_id, typemax(Int))
            if prio < current
                best_priority[string_id] = prio
                string_to_up[string_id] = alias
            elseif prio == current
                # Same source priority — prefer SHORTER accession (Swiss-Prot
                # canonical 6-char like P12345 over TrEMBL 10-char like
                # A0A024R9N3), tie-break alphabetically. Swiss-Prot accessions
                # tend to have richer SL/Pfam annotation than TrEMBL.
                current_best = string_to_up[string_id]
                if length(alias) < length(current_best) ||
                   (length(alias) == length(current_best) && alias < current_best)
                    string_to_up[string_id] = alias
                end
            end
        end
    end

    up_to_string = Dict{String, String}()
    for (s, u) in string_to_up
        # If two STRING-IDs map to the same UniProt, keep the
        # alphabetically-first STRING-ID (deterministic reverse map).
        if !haskey(up_to_string, u) || s < up_to_string[u]
            up_to_string[u] = s
        end
    end

    coverage = if restrict_to !== nothing
        n_req = length(restrict_to)
        n_mapped = count(k -> haskey(string_to_up, k), restrict_to)
        (requested=n_req, mapped=n_mapped,
         fraction=n_mapped / max(n_req, 1),
         n_alias_rows_scanned=n_total)
    else
        (requested=length(string_to_up), mapped=length(string_to_up),
         fraction=1.0, n_alias_rows_scanned=n_total)
    end

    return (string_to_up=string_to_up,
            up_to_string=up_to_string,
            coverage=coverage)
end

end # module ProteinIDMapping
