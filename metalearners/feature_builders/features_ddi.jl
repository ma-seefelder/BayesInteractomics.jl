# Pfam Domain-Domain-Interaction feature builder (Tier 1.3 feature set).
#
# Inputs:
#   - `uniprot_human_sl_pfam.tsv` (already loaded by FeaturesSubcellular)
#   - `3did_flat.txt`              (3did v2024 catalogue)
#
# Outputs (per pair):
#   - ddi_n_known    — number of Pfam-pair combinations in domains(A) × domains(B)
#                       that appear in the 3did catalogue
#   - ddi_has_known  — 1.0 if ddi_n_known > 0 else 0.0
#
# Pfam versioning: 3did stores accession-with-version (e.g. PF00244.25);
# UniProt's `xref_pfam` strips versions (e.g. PF00244). We canonicalise
# to bare `PFxxxxx` for the lookup.

module FeaturesDDI

using DataFrames

export load_3did_catalogue, parse_pfam_column, build_protein_pfam_map,
       build_ddi_features

const PFAM_PAIR_RE = r"\((PF\d+)\.\d+@Pfam\s+(PF\d+)\.\d+@Pfam\)"

"""
    load_3did_catalogue(path::String) -> Set{Tuple{String, String}}

Returns a Set of unordered Pfam-pair tuples (lexicographically ordered
so (A, B) and (B, A) are stored only once).
"""
function load_3did_catalogue(path::String)
    pairs = Set{Tuple{String, String}}()
    n_lines = 0
    open(path, "r") do io
        for line in eachline(io)
            startswith(line, "#=ID") || continue
            m = match(PFAM_PAIR_RE, line)
            m === nothing && continue
            a, b = String(m.captures[1]), String(m.captures[2])
            push!(pairs, a <= b ? (a, b) : (b, a))
            n_lines += 1
        end
    end
    @info "[3did loaded]" n_ddi_pairs=length(pairs) n_id_rows=n_lines
    return pairs
end

"""
    parse_pfam_column(s::AbstractString) -> Vector{String}

UniProt's `xref_pfam` column is semicolon-separated `PFxxxxx;PFyyyyy;`.
Returns the bare list.
"""
function parse_pfam_column(s::AbstractString)
    isempty(strip(s)) && return String[]
    parts = split(s, ';')
    out = String[]
    for p in parts
        ps = strip(p)
        isempty(ps) && continue
        push!(out, String(ps))
    end
    return out
end

"""
    build_protein_pfam_map(uniprot_df, string_to_up)
        -> Dict{String, Vector{String}}  (keyed by STRING-ID)
"""
function build_protein_pfam_map(uniprot_df::DataFrame,
                                 string_to_up::Dict{String, String})
    up_to_pfam = Dict{String, Vector{String}}()
    for row in eachrow(uniprot_df)
        up_to_pfam[row.accession] = parse_pfam_column(row.xref_pfam)
    end
    string_to_pfam = Dict{String, Vector{String}}()
    for (s, u) in string_to_up
        string_to_pfam[s] = get(up_to_pfam, u, String[])
    end
    return string_to_pfam
end

"""
    build_ddi_features(pairs, string_to_pfam, ddi_set)
        -> NamedTuple(n_known, has_known)
"""
function build_ddi_features(pairs::AbstractMatrix{<:AbstractString},
                             string_to_pfam::Dict{String, Vector{String}},
                             ddi_set::Set{Tuple{String, String}})
    n = size(pairs, 1)
    nk = zeros(Float64, n)
    hk = zeros(Float64, n)
    for i in 1:n
        a = get(string_to_pfam, pairs[i, 1], String[])
        b = get(string_to_pfam, pairs[i, 2], String[])
        (isempty(a) || isempty(b)) && continue
        count = 0
        for da in a, db in b
            key = da <= db ? (da, db) : (db, da)
            if key in ddi_set
                count += 1
            end
        end
        nk[i] = Float64(count)
        hk[i] = count > 0 ? 1.0 : 0.0
    end
    return (n_known=nk, has_known=hk)
end

end # module FeaturesDDI
