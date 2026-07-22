# STRING transferred-channels feature builder (Tier 1.2 feature set).
#
# Reads `9606.protein.links.detailed.v12.0.txt` (15 columns per row), keeps
# only the 4 transferred channels we want:
#   neighborhood_transferred, experiments_transferred,
#   database_transferred, textmining_transferred
#
# For pairs not present in the STRING detailed file (low-confidence edges)
# the 4 features default to 0.0 — matching STRING's own convention.

module FeaturesStringTransferred

using DataFrames

export load_string_detailed, build_string_transferred_features,
       STRING_TRANSFERRED_COLS

# STRING v12 "full" columns (space-separated, with header row). The v12
# detailed file dropped the *_transferred channels — they live in "full".
const STRING_DETAILED_COLS = [
    "protein1", "protein2",
    "neighborhood", "neighborhood_transferred",
    "fusion", "cooccurence", "homology",
    "coexpression", "coexpression_transferred",
    "experiments", "experiments_transferred",
    "database", "database_transferred",
    "textmining", "textmining_transferred",
    "combined_score",
]
const STRING_DETAILED_N_COLS = length(STRING_DETAILED_COLS)  # 16

const STRING_TRANSFERRED_COLS = [
    "neighborhood_transferred",
    "experiments_transferred",
    "database_transferred",
    "textmining_transferred",
]

const TR_COL_INDICES = [findfirst(==(c), STRING_DETAILED_COLS) for c in STRING_TRANSFERRED_COLS]
const PROTEIN1_IDX = 1
const PROTEIN2_IDX = 2

"""
    load_string_detailed(path; restrict_to=nothing)
        -> Dict{Tuple{String, String}, NTuple{4, Float64}}

Streams the STRING detailed file, returns a dict keyed by ordered
(protein1, protein2) tuple → 4-tuple of (n_tr, exp_tr, db_tr, tm_tr).

If `restrict_to::AbstractSet{<:AbstractString}` is supplied, only rows
where BOTH proteins are in that set are loaded — keeps memory bounded
to the train+test scope.

STRING stores each undirected edge in BOTH directions (A→B and B→A).
We store the row as-given; the lookup helper handles both orderings.
"""
function load_string_detailed(path::String;
                               restrict_to::Union{Nothing, AbstractSet{<:AbstractString}}=nothing)
    out = Dict{Tuple{String, String}, NTuple{4, Float64}}()
    n_rows_total = 0
    n_kept = 0
    open(path, "r") do io
        # First line is the header — STRING uses spaces, not tabs
        header_line = readline(io)
        header = split(header_line)
        @assert header == STRING_DETAILED_COLS "Unexpected STRING header: $header"
        for line in eachline(io)
            n_rows_total += 1
            fields = split(line)
            length(fields) < STRING_DETAILED_N_COLS && continue
            p1 = String(fields[PROTEIN1_IDX])
            p2 = String(fields[PROTEIN2_IDX])
            if restrict_to !== nothing
                !(p1 in restrict_to) && continue
                !(p2 in restrict_to) && continue
            end
            vals = (parse(Float64, fields[TR_COL_INDICES[1]]),
                    parse(Float64, fields[TR_COL_INDICES[2]]),
                    parse(Float64, fields[TR_COL_INDICES[3]]),
                    parse(Float64, fields[TR_COL_INDICES[4]]))
            out[(p1, p2)] = vals
            n_kept += 1
        end
    end
    @info "[STRING-detailed loaded]" total_rows=n_rows_total kept=n_kept
    return out
end

"""
    build_string_transferred_features(pairs, edge_to_tr)
        -> NamedTuple(neighborhood_tr, experiments_tr, database_tr, textmining_tr)

Per-pair lookup. Tries (A, B) first, then (B, A). Missing edges default
to 0.0 across all 4 channels.
"""
function build_string_transferred_features(
        pairs::AbstractMatrix{<:AbstractString},
        edge_to_tr::Dict{Tuple{String, String}, NTuple{4, Float64}})
    n = size(pairs, 1)
    nbh = zeros(Float64, n)
    exp_ = zeros(Float64, n)
    db   = zeros(Float64, n)
    tm   = zeros(Float64, n)
    for i in 1:n
        a, b = pairs[i, 1], pairs[i, 2]
        rec = get(edge_to_tr, (a, b), nothing)
        if rec === nothing
            rec = get(edge_to_tr, (b, a), nothing)
        end
        if rec !== nothing
            nbh[i], exp_[i], db[i], tm[i] = rec
        end
    end
    return (neighborhood_tr=nbh, experiments_tr=exp_,
            database_tr=db, textmining_tr=tm)
end

end # module FeaturesStringTransferred
