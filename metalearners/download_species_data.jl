# Per-species data-acquisition script.
#
# Species-agnostic generalisation of the reference human-only
# `download_external_data.jl`. Pulls, per species `<sp>` in the multi-species
# train/val/test split:
#   1. <sp>.protein.links.full.v12.0.txt      — STRING TR (transferred) channels  [TRAINING]
#   2. <sp>.protein.links.detailed.v12.0.txt  — STRING combined-evidence channels → onlyAB [INFERENCE]
#   3. <sp>.protein.aliases.v12.0.txt         — STRING-ID ↔ UniProt mapping
#   4. <sp>.protein.info.v12.0.txt            — STRING protein info
#   5. uniprot_<sp>_sl_pfam.tsv               — UniProt SL + Pfam bulk stream
# plus, ONCE (species-agnostic):
#   6. 3did_flat.txt                          — 3did domain-domain catalogue
#
# Pitfall 2 (load-bearing): links.full (TR/training) and links.detailed→onlyAB
# (baseline/inference) are NOT interchangeable — both are downloaded.
#
# The file is `include`-able WITHOUT triggering any download: the pure helpers
# (build_string_url / build_uniprot_url / enumerate_split_species / make_onlyAB
# and the test-contract aliases string_links_full_url / string_aliases_url /
# string_detailed_onlyab_target / uniprot_sl_pfam_url / enumerate_species) live
# at top level; the download driver is guarded behind
# `if abspath(PROGRAM_FILE) == @__FILE__`.
#
# For sp == "9606" every URL + the onlyAB target filename reproduce the
# human-path literals byte-for-byte (matching the reference feature-source
# download script and the onlyAB transform note).
#
# SECURITY: `sp` is validated as a non-empty all-digits NCBI taxon before ANY
# URL/path interpolation — blocks path/URL injection.

# ---- species-token validation (security gate) -------------------------------

"""
    _validate_species(sp) -> String

Return `sp` as a `String` after asserting it is a non-empty, all-digits NCBI
taxon id. Throws `ArgumentError` otherwise. This is the single trust gate that
every URL/path builder funnels through before interpolating the species token,
preventing path-traversal / URL-injection via a crafted species string.
"""
function _validate_species(sp)
    s = String(sp)
    (!isempty(s) && all(isdigit, s)) ||
        throw(ArgumentError("species token must be a non-empty all-digits NCBI taxon id; got $(repr(sp))"))
    return s
end

# ---- pure URL / filename builders (testable cores) --------------------------

"""
    build_string_url(sp, kind) -> String

Build the STRING v12 download URL for species `sp` and `kind` ∈
(`:links_full`, `:links_detailed`, `:aliases`, `:info`). `sp` must be a numeric
NCBI taxon id (validated). For `sp == "9606"` the URLs are byte-identical to the
reference human path.
"""
function build_string_url(sp, kind::Symbol)
    s = _validate_species(sp)
    if kind === :links_full
        return "https://stringdb-downloads.org/download/protein.links.full.v12.0/$(s).protein.links.full.v12.0.txt.gz"
    elseif kind === :links_detailed
        return "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/$(s).protein.links.detailed.v12.0.txt.gz"
    elseif kind === :aliases
        return "https://stringdb-downloads.org/download/protein.aliases.v12.0/$(s).protein.aliases.v12.0.txt.gz"
    elseif kind === :info
        return "https://stringdb-downloads.org/download/protein.info.v12.0/$(s).protein.info.v12.0.txt.gz"
    else
        throw(ArgumentError("unknown STRING url kind $(repr(kind)); expected one of :links_full, :links_detailed, :aliases, :info"))
    end
end

"""
    build_uniprot_url(sp) -> String

Build the UniProtKB stream URL (SL + Pfam fields) for organism `sp`. Byte-identical
to the reference human form for `sp == "9606"`.
"""
function build_uniprot_url(sp)
    s = _validate_species(sp)
    return "https://rest.uniprot.org/uniprotkb/stream?query=organism_id%3A$(s)&format=tsv&fields=accession%2Cid%2Creviewed%2Ccc_subcellular_location%2Cxref_pfam"
end

# The 3did catalogue is species-agnostic — downloaded once.
const DDI_3DID_URL = "https://3did.irbbarcelona.org/download/current/3did_flat.gz"

# ---- test-contract aliases -------------------------------------------------
# The in-process test imports these exact names; they are thin wrappers over the
# parameterised builders above so there is ONE source of truth per URL template.

string_links_full_url(sp)      = build_string_url(sp, :links_full)
string_links_detailed_url(sp)  = build_string_url(sp, :links_detailed)
string_aliases_url(sp)         = build_string_url(sp, :aliases)
string_info_url(sp)            = build_string_url(sp, :info)
uniprot_sl_pfam_url(sp)        = build_uniprot_url(sp)

"""
    string_detailed_onlyab_target(sp) -> String

The project-local derived `onlyAB` inference-baseline filename (basename only) for
species `sp`. For `sp == "9606"` this equals the existing human artefact
`9606.protein.links.detailed.v12.0.onlyAB.txt`.
"""
function string_detailed_onlyab_target(sp)
    s = _validate_species(sp)
    return "$(s).protein.links.detailed.v12.0.onlyAB.txt"
end

# ---- species enumeration from a split proteins matrix -----------------------

"""
    enumerate_split_species(proteins_matrix) -> Set{String}

Return the set of NCBI taxon ids present in an n×2 (or any-shape) matrix of
STRING-prefixed protein ids, via `split(id, ".")[1]` over every entry. Mirrors
`scripts/dnn_training/generate_dataset.jl::get_species_ids` (the prefix rule).
"""
function enumerate_split_species(proteins_matrix)
    species = Set{String}()
    for id in proteins_matrix
        push!(species, String(split(String(id), ".")[1]))
    end
    return species
end

# Test-contract alias (the in-process test imports `enumerate_species`).
enumerate_species(proteins_matrix) = enumerate_split_species(proteins_matrix)

"""
    sorted_split_species(proteins_matrix) -> Vector{String}

Sorted `Vector{String}` of the unique taxon ids (deterministic driver iteration order).
"""
sorted_split_species(proteins_matrix) = sort!(collect(enumerate_split_species(proteins_matrix)))

# ---- onlyAB column transform ------------------------------------------------

"""
    make_onlyAB(detailed_path::AbstractString, out_path::AbstractString) -> String

Produce the per-species `onlyAB` inference-baseline file from a raw STRING
`links.detailed` table.

Contract (byte-validated against the human reference):

  * The consumer (`prediction_data` in metalearner.jl) reads channel columns
    STRICTLY BY NAME, so the **raw STRING `links.detailed` column ORDER is
    preserved verbatim** — we do NOT hardcode a positional reorder. Required
    header tokens (exact spellings, incl. STRING's `cooccurence` single-r and
    singular `experimental`): protein1 protein2 neighborhood fusion cooccurence
    coexpression experimental database textmining.
  * `combined_score` is KEPT (the consumer is indifferent; keeping it makes the
    non-human files structurally identical to the human reference).
  * STRING `links.detailed` is already an undirected score per pair; the file is
    copied through (single space-delimited table) — no symmetric-row synthesis.

For the human `9606` detailed table this reproduces the existing
`encodings/9606.protein.links.detailed.v12.0.onlyAB.txt` header + schema.
"""
function make_onlyAB(detailed_path::AbstractString, out_path::AbstractString)
    isfile(detailed_path) || throw(ArgumentError("raw STRING detailed file not found: $(detailed_path)"))

    # Required consumer tokens (exact STRING spellings). Order is checked as a
    # SUBSET (present-and-spelled), not as a positional sequence — we preserve
    # the raw file's physical order.
    required = ["protein1", "protein2", "neighborhood", "fusion", "cooccurence",
                "coexpression", "experimental", "database", "textmining"]

    open(detailed_path, "r") do io
        header_line = readline(io)
        header = split(strip(header_line))
        missing_tokens = setdiff(required, header)
        isempty(missing_tokens) ||
            error("STRING detailed header at $(detailed_path) is missing required onlyAB token(s) " *
                  "$(missing_tokens); got header $(header). " *
                  "Refusing to write a schema-incompatible onlyAB file (download-integrity guard).")

        # Raw STRING links.detailed is already the onlyAB column order + content
        # (combined_score kept). Stream the file through verbatim, preserving the
        # space-delimited layout the consumer's CSV.read autodetect expects.
        open(out_path, "w") do out
            println(out, header_line)
            buf = Vector{UInt8}(undef, 1 << 20)  # 1 MiB chunks
            while !eof(io)
                n = readbytes!(io, buf)
                n > 0 && write(out, view(buf, 1:n))
            end
        end
    end
    return String(out_path)
end

# ---- idempotent download helpers --------------------------------------------

using Downloads
using CodecZlib   # pure-Julia cross-platform gunzip (NEVER shell out — Windows-unsafe)

const FORCE = get(ENV, "SPIKE_FORCE_REDOWNLOAD", "0") == "1"

"""
    ensure(target, url; min_bytes=10_000) -> String

Idempotent download with a sanity-size threshold. Re-downloads only when the
target is missing/undersized (or `FORCE`). Errors below `min_bytes`
(truncated/tampered-download guard).
"""
function ensure(target::String, url::String; min_bytes::Int=10_000)
    if !FORCE && isfile(target) && filesize(target) >= min_bytes
        @info "[cache hit] $(basename(target))" size_mb=round(filesize(target)/1e6; digits=1)
        return target
    end
    @info "[download] $(basename(target))" url
    Downloads.download(url, target; timeout=600.0)
    sz = filesize(target)
    @info "[done]     $(basename(target))" size_mb=round(sz/1e6; digits=1)
    sz < min_bytes && error("Downloaded $(target) only $(sz) bytes — under sanity threshold $(min_bytes)")
    return target
end

"""
    gunzip_if_needed(gz_path, target) -> String

Pure-Julia CodecZlib gunzip (cross-platform — no `gunzip` shell-out).
Idempotent: skips when an already-unzipped target ≥ the gz size exists.
"""
function gunzip_if_needed(gz_path::String, target::String)
    if !FORCE && isfile(target) && filesize(target) >= filesize(gz_path)
        @info "[cache hit] $(basename(target)) (unzipped already)"
        return target
    end
    @info "[gunzip]    $(basename(gz_path)) → $(basename(target))"
    open(target, "w") do out
        open(gz_path, "r") do gz
            stream = GzipDecompressorStream(gz)
            buf = Vector{UInt8}(undef, 1 << 20)   # 1 MiB chunks
            while !eof(stream)
                n = readbytes!(stream, buf)
                n > 0 && write(out, view(buf, 1:n))
            end
            close(stream)
        end
    end
    @info "[done]      $(basename(target))" size_mb=round(filesize(target)/1e6; digits=1)
    return target
end

# ---- per-species staging driver (guarded — NO network on `include`) ---------

const ENCODINGS_DIR = "encodings"

# Sanity-size thresholds. Sparse (small-proteome) species ship far smaller
# STRING/UniProt tables than human — we lower min_bytes so they are FEATURISED,
# not skipped. These are deliberately permissive floors that still catch an empty
# / HTML-error response, NOT a per-species exact size.
const MIN_BYTES_LINKS   = 50_000    # sparse-species links table can be tiny (featurise, don't skip)
const MIN_BYTES_ALIASES = 10_000
const MIN_BYTES_INFO    = 10_000
const MIN_BYTES_UNIPROT = 10_000    # sparse proteomes → small UniProt stream
const MIN_BYTES_3DID    = 100_000   # species-agnostic; full catalogue is large

"""
    split_species_ids(; encodings_dir = ENCODINGS_DIR) -> Vector{String}

Enumerate the union of taxon ids across `encodings/{train,val,test}_data.h5` by
reading each file's `"proteins"` n×2 String matrix and applying the prefix split.
Requires HDF5 (loaded by the driver only, not at `include` time).
"""
function split_species_ids(; encodings_dir::AbstractString = ENCODINGS_DIR)
    @eval Main begin
        import HDF5 as _HDF5
    end
    species = Set{String}()
    for split_name in ("train", "val", "test")
        path = joinpath(encodings_dir, "$(split_name)_data.h5")
        isfile(path) || (@warn "[split] missing $(path) — skipping"; continue)
        # Julia 1.12 world-age: HDF5 was just `@eval`-imported above, so its
        # methods are too new to call directly from this (older-world) function.
        # Dispatch through invokelatest so the call resolves in the current world.
        proteins = Base.invokelatest() do
            Main._HDF5.h5open(path, "r") do f
                Main._HDF5.read(f, "proteins")
            end
        end
        union!(species, enumerate_split_species(proteins))
    end
    return sort!(collect(species))
end

"""
    stage_species!(sp; encodings_dir = ENCODINGS_DIR) -> NamedTuple

Download + stage every per-species artefact for taxon `sp`:
links.full (TR/training) AND links.detailed→onlyAB (baseline/inference) [Pitfall 2],
aliases, info, and the UniProt SL+Pfam tsv. Returns a per-file cache-hit/download
summary NamedTuple.
"""
function stage_species!(sp; encodings_dir::AbstractString = ENCODINGS_DIR)
    s = _validate_species(sp)
    mkpath(encodings_dir)

    # 1. STRING links.full (TR channels — TRAINING). Pitfall 2: NOT detailed.
    full_gz = joinpath(encodings_dir, "$(s).protein.links.full.v12.0.txt.gz")
    full_tx = joinpath(encodings_dir, "$(s).protein.links.full.v12.0.txt")
    ensure(full_gz, build_string_url(s, :links_full); min_bytes=MIN_BYTES_LINKS)
    gunzip_if_needed(full_gz, full_tx)

    # 2. STRING links.detailed (combined-evidence — INFERENCE). Pitfall 2: NOT full.
    det_gz = joinpath(encodings_dir, "$(s).protein.links.detailed.v12.0.txt.gz")
    det_tx = joinpath(encodings_dir, "$(s).protein.links.detailed.v12.0.txt")
    ensure(det_gz, build_string_url(s, :links_detailed); min_bytes=MIN_BYTES_LINKS)
    gunzip_if_needed(det_gz, det_tx)
    # onlyAB inference-baseline transform (one row per pair, raw column order).
    onlyab = joinpath(encodings_dir, string_detailed_onlyab_target(s))
    if !FORCE && isfile(onlyab) && filesize(onlyab) >= filesize(det_tx)
        @info "[cache hit] $(basename(onlyab)) (onlyAB already built)"
    else
        make_onlyAB(det_tx, onlyab)
        @info "[onlyAB]    $(basename(onlyab))" size_mb=round(filesize(onlyab)/1e6; digits=1)
    end

    # 3. STRING aliases.
    al_gz = joinpath(encodings_dir, "$(s).protein.aliases.v12.0.txt.gz")
    al_tx = joinpath(encodings_dir, "$(s).protein.aliases.v12.0.txt")
    ensure(al_gz, build_string_url(s, :aliases); min_bytes=MIN_BYTES_ALIASES)
    gunzip_if_needed(al_gz, al_tx)

    # 4. STRING info.
    in_gz = joinpath(encodings_dir, "$(s).protein.info.v12.0.txt.gz")
    in_tx = joinpath(encodings_dir, "$(s).protein.info.v12.0.txt")
    ensure(in_gz, build_string_url(s, :info); min_bytes=MIN_BYTES_INFO)
    gunzip_if_needed(in_gz, in_tx)

    # 5. UniProt SL + Pfam stream (per-organism tsv).
    uni_tsv = joinpath(encodings_dir, "uniprot_$(s)_sl_pfam.tsv")
    ensure(uni_tsv, build_uniprot_url(s); min_bytes=MIN_BYTES_UNIPROT)

    return (species=s, links_full=full_tx, links_detailed=det_tx, onlyAB=onlyab,
            aliases=al_tx, info=in_tx, uniprot=uni_tsv)
end

"""
    ensure_3did!(; encodings_dir = ENCODINGS_DIR) -> String

Download the species-agnostic 3did flat catalogue ONCE.
"""
function ensure_3did!(; encodings_dir::AbstractString = ENCODINGS_DIR)
    mkpath(encodings_dir)
    gz  = joinpath(encodings_dir, "3did_flat.gz")
    txt = joinpath(encodings_dir, "3did_flat.txt")
    try
        ensure(gz, DDI_3DID_URL; min_bytes=MIN_BYTES_3DID)
        gunzip_if_needed(gz, txt)
    catch err
        @warn "3did download failed; please fetch manually from https://3did.irbbarcelona.org/" exception=err
    end
    return txt
end

"""
    run_download_driver(; encodings_dir = ENCODINGS_DIR)

Enumerate every species in the split and stage all per-species artefacts, plus the
one-time 3did catalogue. Prints a per-species cache-hit/download summary. Invoked
ONLY when the file is run as a program (guarded), never on `include`.
"""
function run_download_driver(; encodings_dir::AbstractString = ENCODINGS_DIR)
    species_ids = split_species_ids(; encodings_dir)
    @info "[driver] species in split" n=length(species_ids) species=species_ids
    for (i, sp) in enumerate(species_ids)
        @info "[driver] ($(i)/$(length(species_ids))) staging species $(sp)"
        stage_species!(sp; encodings_dir)
    end
    ensure_3did!(; encodings_dir)
    @info "[driver] all species staged" encodings=encodings_dir
    return species_ids
end

# Download driver runs ONLY when executed as a program. `include`-ing the file
# (e.g. from the unit test) loads the pure helpers with NO network access.
if abspath(PROGRAM_FILE) == @__FILE__
    run_download_driver()
end
