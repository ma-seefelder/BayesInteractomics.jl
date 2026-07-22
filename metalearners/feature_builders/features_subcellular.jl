# Subcellular-Co-Localization feature builder (Tier 1.1 feature set).
#
# Reads `uniprot_human_sl_pfam.tsv` (UniProt bulk-stream), parses the
# `cc_subcellular_location` column into a controlled vocabulary of 12
# compartment buckets, builds per-protein Set{Symbol}, and emits
# per-pair (jaccard, n_shared, has_shared) features.

module FeaturesSubcellular

using DataFrames

export load_uniprot_sl_pfam_tsv, parse_sl_text, build_sl_features,
       SL_BUCKETS, SL_KEYWORD_MAP

# 12 compartment buckets — priority-ordered match list. Earlier entries
# win over later ones (because "Mitochondrion outer membrane" should
# match Mitochondrion, not Plasma_membrane via "membrane").
const SL_BUCKETS = [:Nucleus, :Cytoplasm, :ER, :Golgi, :Mitochondrion,
                    :Lysosome, :Peroxisome, :Endosome, :Plasma_membrane,
                    :Cytoskeleton, :Extracellular, :Other]

const SL_KEYWORD_MAP = [
    # (regex pattern → bucket Symbol), priority-ordered
    (r"mitochondr"i,           :Mitochondrion),
    (r"endoplasmic reticulum"i, :ER),
    (r"\bgolgi"i,              :Golgi),
    (r"lysosom"i,              :Lysosome),
    (r"peroxisom"i,            :Peroxisome),
    (r"endosom"i,              :Endosome),
    (r"nucleus|nucleolus|nuclear"i, :Nucleus),
    (r"cytoskelet|microtubule|\bactin\b|intermediate filament"i, :Cytoskeleton),
    (r"cell membrane|plasma membrane|cell surface"i, :Plasma_membrane),
    (r"cytoplasm|cytosol"i,    :Cytoplasm),
    (r"extracellular|secreted"i, :Extracellular),
]

"""
    parse_sl_text(text) -> Set{Symbol}

Convert a raw UniProt `cc_subcellular_location` field into a Set of
compartment buckets. Strategy: strip "SUBCELLULAR LOCATION:" prefixes,
split on punctuation (".", ";"), match each fragment against
`SL_KEYWORD_MAP` priority-ordered; collect all matched buckets.

Note=... blocks are deliberately retained — they often contain
additional compartment references in free text.
"""
function parse_sl_text(text::AbstractString)
    result = Set{Symbol}()
    isempty(strip(text)) && return result
    # Normalise: strip "SUBCELLULAR LOCATION:" markers, then split on
    # period/semicolon to get individual location fragments.
    cleaned = replace(text, r"SUBCELLULAR LOCATION[ ]?:" => " ")
    fragments = split(cleaned, r"[\.\;]+")
    for frag in fragments
        s = strip(frag)
        isempty(s) && continue
        for (pat, bucket) in SL_KEYWORD_MAP
            if occursin(pat, s)
                push!(result, bucket)
            end
        end
    end
    isempty(result) && !isempty(strip(text)) && push!(result, :Other)
    return result
end

"""
    load_uniprot_sl_pfam_tsv(path) -> DataFrame

Lightweight TSV reader that handles UniProt's bulk-stream format. Each
row carries `accession`, `id`, `reviewed`, `cc_subcellular_location`,
`xref_pfam`. Fields may contain embedded tabs in `cc_subcellular_location`
if UniProt escapes them — we use rsplit-like logic since the header is
known to be 5 columns.
"""
function load_uniprot_sl_pfam_tsv(path::String)
    lines = readlines(path)
    @assert length(lines) >= 2 "UniProt TSV is empty"
    header = split(lines[1], '\t')
    @assert length(header) == 5 "Expected 5 columns, got $(length(header)): $header"
    accs = String[]
    ids  = String[]
    revs = String[]
    sls  = String[]
    pfs  = String[]
    sizehint!(accs, length(lines))
    for line in lines[2:end]
        fields = split(line, '\t')
        length(fields) < 5 && continue
        push!(accs, String(fields[1]))
        push!(ids,  String(fields[2]))
        push!(revs, String(fields[3]))
        push!(sls,  String(fields[4]))
        push!(pfs,  String(fields[5]))
    end
    return DataFrame(
        accession = accs,
        id = ids,
        reviewed = revs,
        cc_subcellular_location = sls,
        xref_pfam = pfs,
    )
end

"""
    build_protein_sl_map(uniprot_df, string_to_up) -> Dict{String, Set{Symbol}}

For every STRING-ID, look up its UniProt accession and parse the SL
field. Unmapped STRING-IDs get an empty Set. Returns a dict keyed by
STRING-ID.
"""
function build_protein_sl_map(uniprot_df::DataFrame,
                              string_to_up::Dict{String, String})
    up_to_sl = Dict{String, Set{Symbol}}()
    for row in eachrow(uniprot_df)
        up_to_sl[row.accession] = parse_sl_text(row.cc_subcellular_location)
    end
    string_to_sl = Dict{String, Set{Symbol}}()
    for (s, u) in string_to_up
        string_to_sl[s] = get(up_to_sl, u, Set{Symbol}())
    end
    return string_to_sl
end

"""
    build_sl_features(pairs::AbstractMatrix{<:AbstractString},
                       string_to_sl::Dict{String, Set{Symbol}})
        -> NamedTuple(jaccard, n_shared, has_shared)

`pairs` is an (N × 2) matrix of STRING-IDs (one row per protein pair).
Returns 3 length-N feature vectors.
"""
function build_sl_features(pairs::AbstractMatrix{<:AbstractString},
                            string_to_sl::Dict{String, Set{Symbol}})
    n = size(pairs, 1)
    jacc = zeros(Float64, n)
    nshr = zeros(Float64, n)
    hshr = zeros(Float64, n)
    for i in 1:n
        a = get(string_to_sl, pairs[i, 1], Set{Symbol}())
        b = get(string_to_sl, pairs[i, 2], Set{Symbol}())
        if isempty(a) || isempty(b)
            continue  # leave 0.0 — no annotation = no signal
        end
        inter = length(intersect(a, b))
        uni   = length(union(a, b))
        jacc[i] = inter / uni
        nshr[i] = Float64(inter)
        hshr[i] = inter > 0 ? 1.0 : 0.0
    end
    return (jaccard=jacc, n_shared=nshr, has_shared=hshr)
end

end # module FeaturesSubcellular
