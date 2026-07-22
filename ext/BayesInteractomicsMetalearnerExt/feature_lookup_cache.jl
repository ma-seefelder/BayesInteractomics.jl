# Feature lookup cache for the TR+DDI metalearner schema.
#
# This file provides the `Dict{Tuple{String,String}, NamedTuple}` lookup that
# both the training path and the inference path consume to
# obtain the 6 TR+DDI features per protein pair:
#
#     (neighborhood_tr, experiments_tr, database_tr, textmining_tr,
#      ddi_n_known, ddi_has_known)
#
# The lookup is a single global JLD2 dict keyed by a canonicalised
# (protein_a, protein_b) tuple (lexicographic min/max ordering, matching the
# 3did module's convention in features_ddi.jl).
#
# INFERENCE loads a self-contained artefact shipped inside the repo at
# `metalearners/feature_lookup.jld2` (resolved package-relative). This trusted
# artefact is loaded DIRECTLY — it is NOT re-validated against the ~1.9 GB
# STRING/3did/UniProt source files (those are dev-only inputs, gitignored and
# never shipped). The legacy
# `.bayesinteractomics_cache/metalearner_features/lookup.jld2` + SHA-rebuild
# path is retained ONLY as the developer/refit fallback, reached when the
# shipped artefact is absent (a bare source checkout). This keeps a public
# metalearner run fully self-contained — no dependency on any developer
# working-tree state.
#
# Cache invalidation: the SHA256 hashes of the 4 spike-009 source files
# (`9606.protein.aliases.v12.0.txt`, `9606.protein.links.full.v12.0.txt`,
# `uniprot_human_sl_pfam.tsv`, `3did_flat.txt`) are persisted inside the JLD2
# under the sibling key `source_sha256`. Any change to any source file forces
# a rebuild on the next call.
#
# The spike-009 DDI module emits the
# internal field names `n_known` / `has_known`; these are renamed to
# `ddi_n_known` / `ddi_has_known` at THIS construction site so all downstream
# consumers (training script + inference path) see the production-schema
# column names. The 4 TR column names already match production schema verbatim.

using JLD2
using SHA
using Dates: now

# ---- Constants ---------------------------------------------------------------

"""
    METALEARNER_FEATURE_LOOKUP_CACHE_TYPE

Discriminator string stored inside the JLD2 cache file so a corrupted or
mis-typed JLD2 can be rejected at load time. Mirrors the precedent in
`src/core/intermediate_cache.jl::save_h0_cache` which uses `cache_type =
"h0_logbf"`.
"""
const METALEARNER_FEATURE_LOOKUP_CACHE_TYPE = "metalearner_feature_lookup_v1"

"""
    METALEARNER_FEATURE_LOOKUP_SOURCE_FILES

The legacy fixed 4-human-file set whose SHA256 hashes were persisted in the
cache before the multi-species generalisation. Retained verbatim for the
human-only fast path and as a default when no species set is supplied. The
multi-species generalisation expands this to a
**per-species** set via `metalearner_feature_lookup_source_files(species)` —
the SHA256 manifest is computed over the full per-species file list so any new
or changed species file forces a rebuild (no version constant; the
SHA manifest is the project norm).

Order matters only for stable presentation in error messages — the SHA256
manifest is a `Dict{String,String}` keyed by basename so order is irrelevant
for validation.
"""
const METALEARNER_FEATURE_LOOKUP_SOURCE_FILES = [
    "9606.protein.aliases.v12.0.txt",
    "9606.protein.links.full.v12.0.txt",
    "uniprot_human_sl_pfam.tsv",
    "3did_flat.txt",
]

"""
    METALEARNER_FEATURE_LOOKUP_SHARED_FILES

Species-agnostic source files that appear ONCE in the per-species manifest no
matter how many species are in the split. The 3did domain-domain catalogue is
shared across all taxa — Pfam domain pairs are not species-specific.
"""
const METALEARNER_FEATURE_LOOKUP_SHARED_FILES = [
    "3did_flat.txt",
]

"""
    metalearner_feature_lookup_source_files(species) -> Vector{String}

Expand a species set (vector of NCBI taxonomy-ID strings, e.g. `["9606",
"10090"]`) into the full per-species feature-lookup source-file list:

  - `<sp>.protein.aliases.v12.0.txt`   (per species — STRING-ID ↔ UniProt)
  - `<sp>.protein.links.full.v12.0.txt` (per species — STRING TR channels)
  - `uniprot_<sp>_sl_pfam.tsv`          (per species — UniProt SL + Pfam)

plus the single species-agnostic `3did_flat.txt`. The returned list is
the basis for the SHA256 manifest: adding a species adds 3 entries so
the manifest changes and the cache rebuilds.

`species` may be a `Vector`, `Set`, or any iterable of `AbstractString`; the
returned list is de-duplicated and the shared 3did file is appended once.
"""
function metalearner_feature_lookup_source_files(species)
    files = String[]
    seen = Set{String}()
    for sp_raw in species
        sp = String(sp_raw)
        for fname in (
                "$(sp).protein.aliases.v12.0.txt",
                "$(sp).protein.links.full.v12.0.txt",
                "uniprot_$(sp)_sl_pfam.tsv",
            )
            if !(fname in seen)
                push!(files, fname); push!(seen, fname)
            end
        end
    end
    for fname in METALEARNER_FEATURE_LOOKUP_SHARED_FILES
        if !(fname in seen)
            push!(files, fname); push!(seen, fname)
        end
    end
    return files
end

"""
    metalearner_species_filenames(sp::AbstractString) -> NamedTuple

Resolve the per-species source filenames for taxon `sp`. Production filenames
follow the per-species scheme produced by `metalearners/download_species_data.jl`
(`<sp>.protein.aliases.v12.0.txt`, `<sp>.protein.links.full.v12.0.txt`,
`uniprot_<sp>_sl_pfam.tsv`); the shared 3did catalogue is `3did_flat.txt`.

Returns `(aliases, links, uniprot, ddi)` basenames.
"""
function metalearner_species_filenames(sp::AbstractString)
    return (
        aliases = "$(sp).protein.aliases.v12.0.txt",
        links   = "$(sp).protein.links.full.v12.0.txt",
        uniprot = "uniprot_$(sp)_sl_pfam.tsv",
        ddi     = "3did_flat.txt",
    )
end

"""
    _resolve_source_file(source_dir, sp, kind) -> String

Resolve the on-disk path for a per-species source file of `kind`
(`:aliases`, `:links`, `:uniprot`, `:ddi`) under `source_dir`.

Probes the production filename first, then the committed test-fixture
`_mini` spellings (so the same builder serves both
`test/fixtures/metalearner_multispecies/` and a real staged-data run). The
first existing candidate wins; if none exist the production path is returned
(so the downstream `isfile` guard surfaces a clear missing-input error).
"""
function _resolve_source_file(source_dir::AbstractString, sp::AbstractString, kind::Symbol)
    names = metalearner_species_filenames(sp)
    prod = getproperty(names, kind)
    # Fixture (`_mini`) fallbacks (see test/fixtures/metalearner_multispecies/EXPECTED.md).
    mini = if kind === :aliases
        "$(sp)_mini.aliases.txt"
    elseif kind === :links
        "$(sp)_mini.links.full.txt"
    elseif kind === :uniprot
        "uniprot_mini_sl_pfam.tsv"
    elseif kind === :ddi
        "3did_mini_flat.txt"
    else
        prod
    end
    candidates = (prod, mini)
    for cand in candidates
        p = joinpath(source_dir, cand)
        isfile(p) && return p
    end
    return joinpath(source_dir, prod)
end

"""
    METALEARNER_FEATURE_LOOKUP_COLUMNS

Canonical column order of the production schema NamedTuple values. The 4 TR
columns precede the 2 DDI columns (the C6_TR_DDI schema). The DDI names
are the production-schema names `ddi_n_known` / `ddi_has_known` (NOT the
spike-internal `n_known` / `has_known`).
"""
const METALEARNER_FEATURE_LOOKUP_COLUMNS = (
    :neighborhood_tr,
    :experiments_tr,
    :database_tr,
    :textmining_tr,
    :ddi_n_known,
    :ddi_has_known,
)

# ---- Canonical key ordering --------------------------------------------------

"""
    _canonical_pair_key(a::AbstractString, b::AbstractString) -> Tuple{String,String}

Return the protein pair in lexicographic (min, max) order so that
`(a, b)` and `(b, a)` map to the same dict entry. Mirrors the canonicalisation
in `spike-009/src/features_ddi.jl::load_3did_catalogue` where Pfam pairs are
stored as `a <= b ? (a, b) : (b, a)`.
"""
function _canonical_pair_key(a::AbstractString, b::AbstractString)
    sa = String(a)
    sb = String(b)
    return sa <= sb ? (sa, sb) : (sb, sa)
end

# ---- SHA256 manifest ---------------------------------------------------------

"""
    _compute_source_sha256(spike_dir::String;
                           source_files = METALEARNER_FEATURE_LOOKUP_SOURCE_FILES,
                           cache_subdir = "cache") -> Dict{String,String}

Compute SHA256 of each source file in `source_files` under
`joinpath(spike_dir, cache_subdir)`. Files that are absent map to the literal
string `"absent"` so the manifest remains comparable (a previously cached
manifest built with the file present will mismatch and force a rebuild).

`source_files` is the **per-species** list when building a
multi-species lookup (`metalearner_feature_lookup_source_files(species)`), so
adding or changing any species file invalidates the cache.
"""
function _compute_source_sha256(spike_dir::String;
        source_files = METALEARNER_FEATURE_LOOKUP_SOURCE_FILES,
        cache_subdir::AbstractString = "cache")::Dict{String,String}
    cache_dir = isempty(cache_subdir) ? spike_dir : joinpath(spike_dir, cache_subdir)
    sha_map = Dict{String,String}()
    for fname in source_files
        fpath = joinpath(cache_dir, fname)
        if isfile(fpath)
            sha_map[fname] = bytes2hex(SHA.sha256(read(fpath)))
        else
            sha_map[fname] = "absent"
        end
    end
    return sha_map
end

# ---- Per-species featuriser core --------------------------------------------

"""
    _included_spike009_modules(spike_src_dir::String) -> NamedTuple

Runtime-`include` the four species-neutral spike-009 builder modules
(`protein_id_mapping.jl`, `features_string_transferred.jl`,
`features_subcellular.jl`, `features_ddi.jl`) and return their module handles.

The builders take filenames + a `restrict_to` set + a pairs matrix and contain
no species hardcoding (RESEARCH §"Don't Build"), so they are reused verbatim
across taxa. Included via `Base.include` into THIS extension module so the
`module ProteinIDMapping … end` etc. bind here.

Throws `ArgumentError` if any builder source file is missing.
"""
function _included_spike009_modules(spike_src_dir::String)
    src_idmap = joinpath(spike_src_dir, "protein_id_mapping.jl")
    src_tr    = joinpath(spike_src_dir, "features_string_transferred.jl")
    src_sub   = joinpath(spike_src_dir, "features_subcellular.jl")
    src_ddi   = joinpath(spike_src_dir, "features_ddi.jl")

    missing_src = String[]
    for p in (src_idmap, src_tr, src_sub, src_ddi)
        isfile(p) || push!(missing_src, p)
    end
    if !isempty(missing_src)
        throw(ArgumentError(
            "On-the-fly featuriser requires the spike-009 builder " *
            "modules. Missing:\n" *
            join(("  - " * p for p in missing_src), "\n")
        ))
    end

    IDMap = Base.include(@__MODULE__, src_idmap)
    TR    = Base.include(@__MODULE__, src_tr)
    Sub   = Base.include(@__MODULE__, src_sub)
    DDI   = Base.include(@__MODULE__, src_ddi)
    return (idmap = IDMap, tr = TR, sub = Sub, ddi = DDI)
end

# Package root: two levels up from ext/BayesInteractomicsMetalearnerExt/.
# Mirrors the `dirname(dirname(@__DIR__))` idiom already used for artefact
# resolution in metalearner.jl.
_pkg_root() = dirname(dirname(@__DIR__))

"""
    _default_spike_src_dir() -> String

Default location of the four self-contained feature-builder modules. Prefers
the repo-core copy shipped at `metalearners/feature_builders/` (resolved
package-relative), so a public checkout featurises non-human baits without any
developer-tree dependency. Falls back to the `BAYESINT_FEATURE_SRC_DIR`
environment override only in a developer tree where the repo-core copy is absent.
"""
function _default_spike_src_dir()
    repo_core = joinpath(_pkg_root(), "metalearners", "feature_builders")
    isdir(repo_core) && return repo_core
    return get(ENV, "BAYESINT_FEATURE_SRC_DIR", repo_core)
end

"""
    featurise_pairs_onthefly(pairs, species;
        source_dir,
        spike_src_dir = _default_spike_src_dir(),
        restrict_to = nothing,
        mods = nothing)
        -> Dict{Tuple{String,String}, NamedTuple}

Derive the 6-column production-schema TR/DDI NamedTuple for every row of
`pairs` (an `n × 2` matrix of STRING IDs) for a SINGLE `species`, reading the
species-correct source files under `source_dir` and reusing the spike-009
builders verbatim.

This is the single species-general route used for BOTH training (pairs = the
split proteins matrix) and live inference (pairs = `(poi, every prey)`); see
RESEARCH §"On-the-fly Featuriser" (Open Q2 — one path reduces drift).

`restrict_to` (a `Set` of STRING IDs) bounds the STRING-detailed load to the
pairs of interest (memory-bounded); when `nothing` it is derived from `pairs`.
The Pitfall 3 rename (`n_known → ddi_n_known`, `has_known → ddi_has_known`) is
applied here.
"""
function featurise_pairs_onthefly(pairs::AbstractMatrix, species::AbstractString;
        source_dir::AbstractString,
        spike_src_dir::AbstractString = _default_spike_src_dir(),
        restrict_to::Union{Nothing, AbstractSet} = nothing,
        mods = nothing)

    sp = String(species)
    pairs_s = String.(pairs)

    # restrict_to: bound the STRING-detailed + alias load to the pairs of interest.
    rset = if restrict_to === nothing
        s = Set{String}()
        for i in 1:size(pairs_s, 1)
            push!(s, pairs_s[i, 1]); push!(s, pairs_s[i, 2])
        end
        s
    else
        Set(String(x) for x in restrict_to)
    end

    M = mods === nothing ? _included_spike009_modules(spike_src_dir) : mods

    alias_path   = _resolve_source_file(source_dir, sp, :aliases)
    links_path   = _resolve_source_file(source_dir, sp, :links)
    uniprot_path = _resolve_source_file(source_dir, sp, :uniprot)
    ddi_path     = _resolve_source_file(source_dir, sp, :ddi)

    missing_files = String[]
    isfile(alias_path)   || push!(missing_files, alias_path)
    isfile(links_path)   || push!(missing_files, links_path)
    isfile(uniprot_path) || push!(missing_files, uniprot_path)
    isfile(ddi_path)     || push!(missing_files, ddi_path)
    if !isempty(missing_files)
        throw(ArgumentError(
            "Featuriser: missing source files for species $(sp) " *
            "under $(source_dir):\n" *
            join(("  - " * p for p in missing_files), "\n")
        ))
    end

    # The spike-009 builder modules are `Base.include`d at runtime, so both the
    # binding *and* the method live in a newer world than this function. Resolve
    # the binding AND dispatch via invokelatest (Julia 1.12 strict world-age
    # semantics for runtime-defined bindings — a bare `M.mod.fn` access in the
    # old world warns and is slated to error in a future Julia).
    _fn(mod, sym) = Base.invokelatest(getproperty, mod, sym)

    # STRING → UniProt mapping (restricted to the pairs' STRING IDs).
    map_res = Base.invokelatest(_fn(M.idmap, :load_string_to_uniprot), alias_path; restrict_to = rset)
    string_to_up = map_res.string_to_up

    # TR channels (restrict_to keeps memory bounded).
    edge_to_tr = Base.invokelatest(_fn(M.tr, :load_string_detailed), links_path; restrict_to = rset)
    tr_feat = Base.invokelatest(_fn(M.tr, :build_string_transferred_features), pairs_s, edge_to_tr)

    # DDI: UniProt SL+Pfam → per-protein Pfam map → 3did catalogue.
    uniprot_df = Base.invokelatest(_fn(M.sub, :load_uniprot_sl_pfam_tsv), uniprot_path)
    string_to_pfam = Base.invokelatest(_fn(M.ddi, :build_protein_pfam_map), uniprot_df, string_to_up)
    ddi_set = Base.invokelatest(_fn(M.ddi, :load_3did_catalogue), ddi_path)
    ddi_feat = Base.invokelatest(_fn(M.ddi, :build_ddi_features), pairs_s, string_to_pfam, ddi_set)

    lookup = Dict{Tuple{String,String}, NamedTuple}()
    n = size(pairs_s, 1)
    for i in 1:n
        key = _canonical_pair_key(pairs_s[i, 1], pairs_s[i, 2])
        lookup[key] = (
            neighborhood_tr = Float64(tr_feat.neighborhood_tr[i]),
            experiments_tr  = Float64(tr_feat.experiments_tr[i]),
            database_tr     = Float64(tr_feat.database_tr[i]),
            textmining_tr   = Float64(tr_feat.textmining_tr[i]),
            # Pitfall 3 rename applied at THIS construction site.
            ddi_n_known     = Float64(ddi_feat.n_known[i]),
            ddi_has_known   = Float64(ddi_feat.has_known[i]),
        )
    end
    return lookup
end

# ---- Build: fast path + slow path -------------------------------------------

"""
    build_feature_lookup_from_spike_009(spike_dir::String)
        -> Dict{Tuple{String,String}, NamedTuple}

Build the production-schema feature lookup from the spike-009 assets.

Two paths:

1. **Fast path** — if `joinpath(spike_dir, "cache", "feature_matrices.jld2")`
   exists, load its `train_extras` and `test_extras` NamedTuples and the
   matching `train_pairs` / `test_pairs` matrices (from
   `encodings/train_data.h5` + `encodings/test_data.h5` under the project
   root). Project each row into the 6-column production-schema NamedTuple,
   applying the Pitfall 3 rename `n_known → ddi_n_known`,
   `has_known → ddi_has_known`. The dict key is the canonical
   (lexicographic-min, lexicographic-max) ordering of the pair.

2. **Slow path** — if the cache is absent OR the matching pair matrices are
   not available, include the spike-009 source modules
   (`features_string_transferred.jl` + `features_ddi.jl`) and re-derive the
   features from the 4 source files in `joinpath(spike_dir, "cache")`. This
   requires the 4 source files to be present.

Throws `ArgumentError` if neither path can succeed (typically: spike-009 has
never been downloaded). The error message names every file that was
attempted so the operator can diagnose the missing input.
"""
function build_feature_lookup_from_spike_009(spike_dir::String)
    cache_dir = joinpath(spike_dir, "cache")
    features_cache_path = joinpath(cache_dir, "feature_matrices.jld2")

    # ---- Fast path: spike-009 already has feature_matrices.jld2 -------------

    if isfile(features_cache_path)
        try
            return _build_lookup_from_feature_matrices(features_cache_path, spike_dir)
        catch e
            @warn "Fast-path build from feature_matrices.jld2 failed; falling through to slow path." exception=(e, catch_backtrace())
        end
    end

    # ---- Slow path: rebuild from the 4 source files -------------------------

    src_dir = joinpath(spike_dir, "src")
    src_tr  = joinpath(src_dir, "features_string_transferred.jl")
    src_ddi = joinpath(src_dir, "features_ddi.jl")

    alias_path        = joinpath(cache_dir, "9606.protein.aliases.v12.0.txt")
    string_links_path = joinpath(cache_dir, "9606.protein.links.full.v12.0.txt")
    uniprot_path      = joinpath(cache_dir, "uniprot_human_sl_pfam.tsv")
    ddi_path          = joinpath(cache_dir, "3did_flat.txt")

    missing_files = String[]
    isfile(src_tr)            || push!(missing_files, src_tr)
    isfile(src_ddi)           || push!(missing_files, src_ddi)
    isfile(alias_path)        || push!(missing_files, alias_path)
    isfile(string_links_path) || push!(missing_files, string_links_path)
    isfile(uniprot_path)      || push!(missing_files, uniprot_path)
    isfile(ddi_path)          || push!(missing_files, ddi_path)

    if !isempty(missing_files)
        throw(ArgumentError(
            "Feature lookup cache cannot be built: neither " *
            "$(features_cache_path) nor the 4 spike-009 source files (with " *
            "their builder modules) were found. Missing inputs:\n" *
            join(("  - " * p for p in missing_files), "\n") *
            "\nRun the feature-source download script first (set BAYESINT_FEATURE_SRC_DIR\n" *
            "to the builder-module directory if it is not the repo-core default)."
        ))
    end

    # The slow path is a real on-the-fly
    # featuriser. The protein-pair list is enumerated directly from the
    # links.full file (no feature_matrices.jld2 / encodings/*.h5 dependency),
    # and the 6 TR/DDI columns are derived via the species-neutral spike-009
    # builders. The 4 source files belong to the human (9606) cache here.
    pairs = _pairs_from_links_full(string_links_path)
    return featurise_pairs_onthefly(pairs, "9606";
        source_dir = cache_dir, spike_src_dir = src_dir)
end

"""
    _build_lookup_from_feature_matrices(features_cache_path::String, spike_dir::String)
        -> Dict{Tuple{String,String}, NamedTuple}

Internal helper. Reads `feature_matrices.jld2` (which carries `train_extras`
and `test_extras` NamedTuples) AND the matching protein-pair matrices from
`encodings/train_data.h5` + `encodings/test_data.h5` under the project root,
and assembles the production-schema lookup dict.

Applies the Pitfall 3 rename (`n_known → ddi_n_known`,
`has_known → ddi_has_known`) at THIS construction site.

The project root is computed as the parent of `dirname(spike_dir)`'s parent,
which works for the canonical dev layout `<root>/<feature-src>/009-…/cache/`.
"""
function _build_lookup_from_feature_matrices(features_cache_path::String, spike_dir::String)
    # spike_dir is .../<project_root>/<feature-src>/009-extra-features-tier1
    # so project_root = dirname(dirname(dirname(spike_dir)))
    project_root = dirname(dirname(dirname(spike_dir)))
    train_h5 = joinpath(project_root, "encodings", "train_data.h5")
    test_h5  = joinpath(project_root, "encodings", "test_data.h5")

    if !isfile(train_h5) || !isfile(test_h5)
        throw(ArgumentError(
            "feature_matrices.jld2 is present but the matching protein-pair " *
            "matrices are not. Missing one or both of:\n  - $(train_h5)\n  - $(test_h5)"
        ))
    end

    data = JLD2.load(features_cache_path)
    train_extras = data["train_extras"]
    test_extras  = data["test_extras"]

    train_pairs = HDF5.h5open(train_h5, "r") do f
        HDF5.read(f, "proteins")
    end
    test_pairs = HDF5.h5open(test_h5, "r") do f
        HDF5.read(f, "proteins")
    end

    lookup = Dict{Tuple{String,String}, NamedTuple}()
    _ingest_extras!(lookup, train_pairs, train_extras)
    _ingest_extras!(lookup, test_pairs,  test_extras)
    return lookup
end

"""
    _ingest_extras!(lookup, pairs, extras) -> nothing

Append rows from a single (pairs, extras) slice into the lookup dict.

`pairs` is an `n × 2` matrix of STRING protein IDs; `extras` is the
spike-009 NamedTuple with fields including the 4 TR columns
(`neighborhood_tr, experiments_tr, database_tr, textmining_tr`) plus the
DDI columns under their spike-internal names (`n_known, has_known`).

We apply the Pitfall 3 rename here so the dict values use the
production-schema names `ddi_n_known` / `ddi_has_known`.
"""
function _ingest_extras!(
        lookup::Dict{Tuple{String,String}, NamedTuple},
        pairs::AbstractMatrix,
        extras)
    n = size(pairs, 1)
    for i in 1:n
        a = String(pairs[i, 1])
        b = String(pairs[i, 2])
        key = _canonical_pair_key(a, b)
        nt = (
            neighborhood_tr = Float64(extras.neighborhood_tr[i]),
            experiments_tr  = Float64(extras.experiments_tr[i]),
            database_tr     = Float64(extras.database_tr[i]),
            textmining_tr   = Float64(extras.textmining_tr[i]),
            ddi_n_known     = Float64(extras.ddi_n_known[i]),
            ddi_has_known   = Float64(extras.ddi_has_known[i]),
        )
        lookup[key] = nt
    end
    return nothing
end

# ---- Save / load -------------------------------------------------------------

"""
    save_feature_lookup_cache(lookup, filepath; source_sha256) -> String

Persist the lookup dict to disk under `filepath`. The parent directory is
created with `mkpath`. The JLD2 carries 4 sibling keys:

  - `cache_type::String`           — discriminator (`"metalearner_feature_lookup_v1"`)
  - `lookup::Dict{Tuple{String,String}, NamedTuple}` — the payload
  - `source_sha256::Dict{String,String}` — SHA256 manifest of the 4 source files
  - `timestamp::String`            — ISO-ish timestamp for debugging

Mirrors the `save_h0_cache` precedent in `src/core/intermediate_cache.jl`
(`jldsave` with `compress=true` + a top-level `cache_type` discriminator).
"""
function save_feature_lookup_cache(
        lookup::Dict{Tuple{String,String}, NamedTuple},
        filepath::String;
        source_sha256::Dict{String,String})
    mkpath(dirname(filepath))
    JLD2.jldsave(filepath; compress=true,
        cache_type = METALEARNER_FEATURE_LOOKUP_CACHE_TYPE,
        lookup = lookup,
        source_sha256 = source_sha256,
        timestamp = string(now()),
    )
    return filepath
end

"""
    load_feature_lookup_cache(filepath; current_source_sha256)
        -> Union{Dict{Tuple{String,String}, NamedTuple}, Nothing}

Load the lookup dict from `filepath`, validating the cache discriminator and
the SHA256 manifest against the current state of the spike-009 source files.

Returns:
  - `nothing` if the file doesn't exist
  - `nothing` if the JLD2 lacks the discriminator or it doesn't match
  - `nothing` if the persisted `source_sha256` doesn't match
    `current_source_sha256` (any source file changed → rebuild)
  - `nothing` (with `@warn`) if JLD2 loading throws any exception
  - the persisted `lookup::Dict` otherwise

Mirrors the `load_h0_cache` precedent in `src/core/intermediate_cache.jl`
(graceful `nothing` on any mismatch / corruption).
"""
function load_feature_lookup_cache(
        filepath::String;
        current_source_sha256::Dict{String,String})::Union{Dict{Tuple{String,String}, NamedTuple}, Nothing}
    !isfile(filepath) && return nothing
    try
        data = JLD2.load(filepath)
        get(data, "cache_type", "") == METALEARNER_FEATURE_LOOKUP_CACHE_TYPE || return nothing
        stored_sha = get(data, "source_sha256", nothing)
        stored_sha === nothing && return nothing
        stored_sha == current_source_sha256 || return nothing
        return data["lookup"]::Dict{Tuple{String,String}, NamedTuple}
    catch e
        @warn "Failed to load metalearner feature lookup cache: $e"
        return nothing
    end
end

# ---- Shipped (repo-core) artefact --------------------------------------------

"""
    _shipped_feature_lookup_path() -> String

Package-relative path to the feature-lookup artefact shipped inside the repo at
`metalearners/feature_lookup.jld2`. This is the self-contained inference
artefact — see `load_shipped_feature_lookup`.
"""
_shipped_feature_lookup_path() =
    _shipped_model_path(joinpath("metalearners", "feature_lookup.jld2"))

"""
    load_shipped_feature_lookup(filepath) -> Union{Dict, Nothing}

Load the shipped feature-lookup dict, validating ONLY the `cache_type`
discriminator — NOT the source-SHA manifest. The shipped artefact is a trusted,
version-controlled build; the ~1.9 GB STRING/3did/UniProt sources it was built
from are not shipped, so the SHA gate that `load_feature_lookup_cache` applies
cannot (and must not) run here. Returns `nothing` when absent or unreadable so
the caller can fall through to the developer rebuild path.
"""
function load_shipped_feature_lookup(
        filepath::String)::Union{Dict{Tuple{String,String}, NamedTuple}, Nothing}
    isfile(filepath) || return nothing
    try
        data = JLD2.load(filepath)
        get(data, "cache_type", "") == METALEARNER_FEATURE_LOOKUP_CACHE_TYPE || return nothing
        return data["lookup"]::Dict{Tuple{String,String}, NamedTuple}
    catch e
        @warn "Failed to load shipped metalearner feature lookup ($filepath); \
               falling back to developer rebuild path." exception=e
        return nothing
    end
end

# ---- Public entry point ------------------------------------------------------

"""
    get_or_build_feature_lookup(;
        shipped_path::String = _shipped_feature_lookup_path(),
        spike_dir::String = get(ENV, "BAYESINT_FEATURE_SRC_ROOT", ""),
        cache_path::String = ".bayesinteractomics_cache/metalearner_features/lookup.jld2",
    ) -> Dict{Tuple{String,String}, NamedTuple}

Return the production TR+DDI feature lookup.

Resolution order:
 0. **Shipped artefact (production/inference).** `metalearners/feature_lookup.jld2`
    inside the repo, loaded directly (trusted; no source-SHA validation). This
    is the self-contained path — a public run needs no developer-tree state and
    no multi-GB STRING/3did/UniProt sources.
 1. **SHA-validated dev cache.** `.bayesinteractomics_cache/…/lookup.jld2`,
    accepted only if its stored SHA256 manifest matches the current source files.
 2. **Rebuild (dev/refit).** Build from the spike-009 sources and persist.

Steps 1–2 are reached only when the shipped artefact is absent (a bare source
checkout without `metalearners/feature_lookup.jld2`). They still `throw`
`ArgumentError` when the feature-source inputs are missing (see
`build_feature_lookup_from_spike_009`) — but that path is never hit by a normal
shipped run. Developers refitting from raw sources point `spike_dir` at their
feature-source root via the `BAYESINT_FEATURE_SRC_ROOT` environment override.
"""
function get_or_build_feature_lookup(;
        shipped_path::String = _shipped_feature_lookup_path(),
        spike_dir::String = get(ENV, "BAYESINT_FEATURE_SRC_ROOT", ""),
        cache_path::String = ".bayesinteractomics_cache/metalearner_features/lookup.jld2",
    )::Dict{Tuple{String,String}, NamedTuple}

    # 0. Preferred: the self-contained lookup shipped inside the repo.
    shipped = load_shipped_feature_lookup(shipped_path)
    if shipped !== nothing
        return shipped
    end

    # 1. Developer fallback: SHA-validated on-disk cache.
    current_sha = _compute_source_sha256(spike_dir)
    cached = load_feature_lookup_cache(cache_path; current_source_sha256=current_sha)
    if cached !== nothing
        return cached
    end

    # 2. Developer fallback: rebuild from the spike-009 sources, then persist.
    lookup = build_feature_lookup_from_spike_009(spike_dir)
    save_feature_lookup_cache(lookup, cache_path; source_sha256=current_sha)
    return lookup
end

# ---- Public entry points -----------------------------------------------------

"""
    _pairs_from_links_full(links_path::String) -> Matrix{String}

Extract the unique within-file protein pairs from a STRING `links.full` file
(space-separated, 16-col header `STRING_DETAILED_COLS`). Each undirected edge
is stored in both directions; we canonicalise to `(min, max)` and de-duplicate.
Returns an `m × 2` `Matrix{String}`.
"""
function _pairs_from_links_full(links_path::String)
    seen = Set{Tuple{String,String}}()
    open(links_path, "r") do io
        readline(io)  # header
        for line in eachline(io)
            isempty(line) && continue
            fields = split(line)
            length(fields) < 2 && continue
            a = String(fields[1]); b = String(fields[2])
            push!(seen, _canonical_pair_key(a, b))
        end
    end
    m = length(seen)
    pairs = Matrix{String}(undef, m, 2)
    for (i, (a, b)) in enumerate(seen)
        pairs[i, 1] = a
        pairs[i, 2] = b
    end
    return pairs
end

"""
    build_feature_lookup_multispecies(source_dir;
        species,
        spike_src_dir = _default_spike_src_dir())
        -> Dict{Tuple{String,String}, NamedTuple}

Build the production-schema TR/DDI feature lookup over **all** `species`
(a vector/iterable of NCBI taxonomy-ID strings) present in the split.
Reads the species-correct source files under `source_dir`,
reuses the species-neutral spike-009 builders verbatim via
`featurise_pairs_onthefly`, and merges every species' pairs into a single
`Dict` keyed by the canonical `(min, max)` STRING-ID pair (the taxon prefix is
part of the STRING ID → no separate species field).

Per-species pairs are enumerated from each species' `links.full` file. **No
coverage threshold** — sparse species are featurised, not zero-filled,
so a mouse (10090) pair returns its non-zero TR/DDI NamedTuple.

The spike-009 builder modules are `include`d once and shared across all species
(they take filenames + `restrict_to` + a pairs matrix; no species hardcoding).
"""
function build_feature_lookup_multispecies(source_dir::AbstractString;
        species,
        spike_src_dir::AbstractString = _default_spike_src_dir())

    mods = _included_spike009_modules(spike_src_dir)
    lookup = Dict{Tuple{String,String}, NamedTuple}()

    for sp_raw in species
        sp = String(sp_raw)
        links_path = _resolve_source_file(source_dir, sp, :links)
        if !isfile(links_path)
            throw(ArgumentError(
                "build_feature_lookup_multispecies: missing links.full for " *
                "species $(sp) under $(source_dir) (looked for $(links_path))."
            ))
        end
        pairs = _pairs_from_links_full(links_path)
        isempty(pairs) && continue  # no skip-by-coverage, only skip empty file
        sp_lookup = featurise_pairs_onthefly(pairs, sp;
            source_dir = source_dir, spike_src_dir = spike_src_dir, mods = mods)
        merge!(lookup, sp_lookup)
    end
    return lookup
end

"""
    featurise_pair_onthefly(poi, prey;
        source_dir,
        species,
        restrict_to = nothing,
        spike_src_dir = _default_spike_src_dir())
        -> NamedTuple

On-the-fly slow path: derive the 6-column production-schema
TR/DDI NamedTuple for a SINGLE arbitrary novel `(poi, prey)` pair, live, from
the species-correct source files under `source_dir`. This is the re-enabled
slow path — an arbitrary non-human pair now gets the 6 TR/DDI columns instead
of a hard `throw` or zero-fill.

`restrict_to` (a `Set` of STRING IDs) bounds the STRING-detailed/alias load;
when `nothing` it defaults to `Set([poi, prey])`. Reuses
`featurise_pairs_onthefly` over the single-row pairs matrix.
"""
function featurise_pair_onthefly(poi::AbstractString, prey::AbstractString;
        source_dir::AbstractString,
        species::AbstractString,
        restrict_to::Union{Nothing, AbstractSet} = nothing,
        spike_src_dir::AbstractString = _default_spike_src_dir())

    pairs = reshape(String[String(poi), String(prey)], 1, 2)
    rset = restrict_to === nothing ? Set([String(poi), String(prey)]) : restrict_to
    lookup = featurise_pairs_onthefly(pairs, String(species);
        source_dir = source_dir, spike_src_dir = spike_src_dir, restrict_to = rset)
    key = _canonical_pair_key(String(poi), String(prey))
    return lookup[key]
end
