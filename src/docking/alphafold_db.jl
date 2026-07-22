# ═══════════════════════════════════════════════════════════════════════════════
# AlphaFold DB API: monomer pLDDT retrieval and structural dockability scoring
# ═══════════════════════════════════════════════════════════════════════════════

import Dates: DateTime, now, Millisecond, Day, value

# ─── Constants ────────────────────────────────────────────────────────────────

const ALPHAFOLD_DB_API_BASE = "https://alphafold.ebi.ac.uk/api/prediction"
const ALPHAFOLD_RATE_LIMIT_MS = 500
const ALPHAFOLD_TIMEOUT_S = 30
const ALPHAFOLD_CACHE_DEFAULT_TTL_DAYS = 7
const _alphafold_last_query_time = Ref(DateTime(0))

# ─── Rate limiter ─────────────────────────────────────────────────────────────

function _alphafold_rate_limit_wait()
    elapsed_ms = value(Millisecond(now() - _alphafold_last_query_time[]))
    if elapsed_ms < ALPHAFOLD_RATE_LIMIT_MS
        sleep((ALPHAFOLD_RATE_LIMIT_MS - elapsed_ms) / 1000.0)
    end
    _alphafold_last_query_time[] = now()
end

# ─── pLDDT cache ──────────────────────────────────────────────────────────────

function _plddt_cache_dir(base::String="")::String
    root = isempty(base) ? ".bayesinteractomics_cache" : base
    dir = joinpath(root, "alphafold_db")
    mkpath(dir)
    return dir
end

function _load_plddt_cache(uniprot_id::String; cache_base::String="")::Union{NamedTuple, Nothing}
    cache_path = joinpath(_plddt_cache_dir(cache_base), "$(uppercase(uniprot_id)).jld2")
    !isfile(cache_path) && return nothing
    try
        data = JLD2.load(cache_path)
        cached_at = haskey(data, "cached_at") ? DateTime(String(data["cached_at"])) : DateTime(0)
        af_version = haskey(data, "af_version") ? String(data["af_version"]) : ""
        return (mean_plddt=Float64(data["mean_plddt"]),
                frac_disordered=Float64(data["frac_disordered"]),
                n_residues=Int(data["n_residues"]),
                cached_at=cached_at,
                af_version=af_version)
    catch e
        @warn "Failed to load pLDDT cache" uniprot_id exception=e
        return nothing
    end
end

function _save_plddt_cache(uniprot_id::String, mean_plddt::Float64, frac_disordered::Float64,
                           n_residues::Int, af_version::String; cache_base::String="")
    cache_path = joinpath(_plddt_cache_dir(cache_base), "$(uppercase(uniprot_id)).jld2")
    JLD2.jldsave(cache_path;
        mean_plddt=mean_plddt,
        frac_disordered=frac_disordered,
        n_residues=n_residues,
        af_version=af_version,
        cached_at=string(now()))
end

# ─── Cache freshness checks ───────────────────────────────────────────────────

"""
Cache entry is considered fresh if its `cached_at` is within `max_age_days` of now.
A fresh entry is returned immediately without contacting the API.
"""
function _is_cache_fresh(cached_at::DateTime, max_age_days::Int)::Bool
    max_age_days <= 0 && return false       # 0 or negative disables TTL fast-path
    age = now() - cached_at
    return value(Millisecond(age)) < max_age_days * 86_400_000
end

"""
Lightweight version check against AlphaFold DB. Returns:
- `:same`        → API responded and `entryId` matches the cached version (cheap refresh)
- `:newer`       → API responded with a different version; cache must be refetched
- `:unreachable` → API call failed (network/timeout); caller should fall back to stale data
"""
function _check_af_version(uniprot_id::String, cached_version::String;
                           timeout::Int=ALPHAFOLD_TIMEOUT_S)::Symbol
    try
        _alphafold_rate_limit_wait()
        meta_url = "$ALPHAFOLD_DB_API_BASE/$uniprot_id"
        buf = IOBuffer()
        Downloads.download(meta_url, buf; timeout=timeout)
        meta = JSON3.read(String(take!(buf)))
        isempty(meta) && return :unreachable
        latest = String(meta[1].entryId)
        return latest == cached_version ? :same : :newer
    catch e
        @debug "AlphaFold DB version check failed" uniprot_id exception=e
        return :unreachable
    end
end

"""
Touch `cached_at` to extend the TTL window without refetching scores. Used after a
successful version check confirms the cached entry is still current.
"""
function _refresh_cache_timestamp(uniprot_id::String; cache_base::String="")
    cache_path = joinpath(_plddt_cache_dir(cache_base), "$(uppercase(uniprot_id)).jld2")
    !isfile(cache_path) && return
    try
        data = JLD2.load(cache_path)
        JLD2.jldsave(cache_path;
            mean_plddt   = data["mean_plddt"],
            frac_disordered = data["frac_disordered"],
            n_residues   = data["n_residues"],
            af_version   = get(data, "af_version", ""),
            cached_at    = string(now()))
    catch e
        @warn "Failed to refresh pLDDT cache timestamp" uniprot_id exception=e
    end
end

# ─── Core fetch ───────────────────────────────────────────────────────────────

"""
    fetch_alphafold_plddt(uniprot_id; timeout=30, use_cache=true,
                          cache_max_age_days=7, force_refresh=false) -> NamedTuple

Fetch monomer pLDDT data from AlphaFold DB REST API. Returns a NamedTuple with
fields `(mean_plddt, frac_disordered, n_residues, cached_at, af_version)`.

Two-step API: metadata endpoint for confidence URL, then per-residue JSON.
Results are cached in `.bayesinteractomics_cache/alphafold_db/{UNIPROT_ID}.jld2`.

Cache invalidation strategy (hybrid TTL + version check):

- **Fresh cache** (cached < `cache_max_age_days` ago): returned immediately.
- **Stale cache**: lightweight metadata-only call to AlphaFold DB. If `entryId`
  matches the cached `af_version`, the timestamp is refreshed and cached scores
  are returned (no per-residue refetch needed). If `entryId` is newer, a full
  refetch runs. If the API is unreachable, the stale cache is returned.
- **Cache miss** or `force_refresh=true`: full two-step fetch.

Set `cache_max_age_days = 0` to disable the TTL fast-path (always version-check).
Returns `nothing` on full-fetch failure (caller should use neutral dockability = 0.5).
"""
function fetch_alphafold_plddt(uniprot_id::String;
                               timeout::Int=ALPHAFOLD_TIMEOUT_S,
                               use_cache::Bool=true,
                               cache_base::String="",
                               cache_max_age_days::Int=ALPHAFOLD_CACHE_DEFAULT_TTL_DAYS,
                               force_refresh::Bool=false)
    uid = uppercase(strip(uniprot_id))
    isempty(uid) && return nothing

    # Cache lookup with hybrid TTL + version-check
    if use_cache && !force_refresh
        cached = _load_plddt_cache(uid; cache_base=cache_base)
        if cached !== nothing
            if _is_cache_fresh(cached.cached_at, cache_max_age_days)
                return cached
            end
            # Stale → cheap version check
            if !isempty(cached.af_version)
                status = _check_af_version(uid, cached.af_version; timeout=timeout)
                if status === :same
                    _refresh_cache_timestamp(uid; cache_base=cache_base)
                    return cached
                elseif status === :unreachable
                    # API down — fall back to stale cache rather than failing the run
                    @debug "AF DB unreachable; serving stale cache for $uid"
                    return cached
                end
                # status === :newer → fall through to full refetch
            end
        end
    end

    try
        # Step 1: Fetch prediction metadata
        _alphafold_rate_limit_wait()
        meta_url = "$ALPHAFOLD_DB_API_BASE/$uid"
        buf = IOBuffer()
        Downloads.download(meta_url, buf; timeout=timeout)
        meta = JSON3.read(String(take!(buf)))

        # API returns array -- use first element (fragment F1, canonical)
        isempty(meta) && error("Empty response for $uid")
        plddt_url = meta[1].plddtDocUrl
        af_version = haskey(meta[1], :entryId) ? String(meta[1].entryId) : ""

        # Step 2: Fetch per-residue confidence JSON
        _alphafold_rate_limit_wait()
        buf2 = IOBuffer()
        Downloads.download(String(plddt_url), buf2; timeout=timeout)
        confidence = JSON3.read(String(take!(buf2)))
        scores = Float64.(confidence.confidenceScore)

        length(scores) == 0 && error("No residue scores for $uid")

        # Compute mean pLDDT and disorder fraction
        mean_plddt = sum(scores) / length(scores)
        frac_disordered = count(s -> s < 50.0, scores) / length(scores)
        n_residues = length(scores)

        # Cache result
        if use_cache
            _save_plddt_cache(uid, mean_plddt, frac_disordered, n_residues, af_version; cache_base=cache_base)
        end

        return (mean_plddt=mean_plddt, frac_disordered=frac_disordered,
                n_residues=n_residues, cached_at=now(), af_version=af_version)
    catch e
        @warn "AlphaFold DB lookup failed for $uid" exception=e
        return nothing
    end
end

# ─── UniProt ID resolution ───────────────────────────────────────────────────

function _uniprot_mapping_cache_dir(base::String="")::String
    root = isempty(base) ? ".bayesinteractomics_cache" : base
    return joinpath(root, "uniprot_mapping")
end

function _load_uniprot_mapping_cache(; cache_base::String="")::Dict{String, String}
    cache_path = joinpath(_uniprot_mapping_cache_dir(cache_base), "mapping.jld2")
    isfile(cache_path) || return Dict{String, String}()
    try
        JLD2.load(cache_path, "mapping")
    catch
        Dict{String, String}()
    end
end

function _save_uniprot_mapping_cache(mapping::Dict{String, String}; cache_base::String="")
    dir = _uniprot_mapping_cache_dir(cache_base)
    mkpath(dir)
    JLD2.save(joinpath(dir, "mapping.jld2"), "mapping", mapping)
end

"""
    _resolve_uniprot_ids(gene_names; species=9606, timeout=30, resolve_uniprot=true) -> Dict{String, String}

Batch-resolve gene names to UniProt primary accessions via the UniProt REST search API.
Results are cached in `.bayesinteractomics_cache/uniprot_mapping/mapping.jld2`.

Set `resolve_uniprot=false` to skip live API calls (returns only cached hits).
"""
function _resolve_uniprot_ids(gene_names::Vector{String}; species::Int=9606, timeout::Int=30, resolve_uniprot::Bool=true, cache_base::String="")::Dict{String, String}
    mapping = _load_uniprot_mapping_cache(; cache_base=cache_base)

    # If resolution is disabled (unit tests), return only cached hits
    if !resolve_uniprot
        result = Dict{String, String}()
        for name in gene_names
            key = "$(species)_$(name)"
            uid = get(mapping, key, "")
            if !isempty(uid)
                result[name] = uid
            end
        end
        return result
    end

    to_resolve = String[]
    for name in gene_names
        key = "$(species)_$(name)"
        haskey(mapping, key) || push!(to_resolve, name)
    end

    if isempty(to_resolve)
        result = Dict{String, String}()
        for name in gene_names
            uid = get(mapping, "$(species)_$(name)", "")
            if !isempty(uid)
                result[name] = uid
            end
        end
        return result
    end

    prog = Progress(length(to_resolve); desc="Resolving UniProt IDs ", enabled=length(to_resolve) > 1)
    for name in to_resolve
        key = "$(species)_$(name)"
        try
            _alphafold_rate_limit_wait()
            url = "https://rest.uniprot.org/uniprotkb/search?query=(gene_exact:$(name))+AND+(organism_id:$(species))&fields=accession&format=tsv&size=1"
            buf = IOBuffer()
            Downloads.download(url, buf; timeout=timeout)
            text = String(take!(buf))
            lines = split(strip(text), '\n')
            if length(lines) >= 2
                accession = strip(String(lines[2]))  # First data row
                if !isempty(accession) && !startswith(accession, "Entry")
                    mapping[key] = accession
                else
                    mapping[key] = ""
                end
            else
                mapping[key] = ""
            end
        catch e
            @debug "UniProt mapping failed for $name" exception=e
            mapping[key] = ""
        end
        ProgressMeter.next!(prog)
    end
    ProgressMeter.finish!(prog)

    _save_uniprot_mapping_cache(mapping; cache_base=cache_base)

    result = Dict{String, String}()
    for name in gene_names
        uid = get(mapping, "$(species)_$(name)", "")
        if !isempty(uid)
            result[name] = uid
        end
    end
    return result
end

# ─── Dockability scoring ─────────────────────────────────────────────────────

"""
    compute_dockability(mean_plddt, frac_disordered) -> Float64

Structural dockability score in [0, 1]. High pLDDT and low disorder = high dockability.
Formula: `mean_pLDDT / 100 * (1 - frac_disordered)`.
"""
function compute_dockability(mean_plddt::Float64, frac_disordered::Float64)::Float64
    return (mean_plddt / 100.0) * (1.0 - frac_disordered)
end

# ─── Batch fetch ──────────────────────────────────────────────────────────────

"""
    fetch_dockability_scores(candidates::DataFrame; species=9606, resolve_uniprot=true) -> Tuple{Dict{String, Float64}, Dict{String, String}}

Fetch pLDDT data for all candidates and compute dockability scores.
First resolves gene names to UniProt accessions via UniProt REST API (cached).

Returns a tuple of:
- `scores::Dict{String, Float64}`: Protein name -> dockability score (0.5 = neutral fallback)
- `uniprot_map::Dict{String, String}`: Protein name -> UniProt accession (only successful mappings)

Set `resolve_uniprot=false` to skip live API calls (tests, offline mode).
"""
function fetch_dockability_scores(candidates::DataFrame; species::Int=9606,
                                  resolve_uniprot::Bool=true, cache_base::String="",
                                  cache_max_age_days::Int=ALPHAFOLD_CACHE_DEFAULT_TTL_DAYS,
                                  force_refresh::Bool=false)::Tuple{Dict{String, Float64}, Dict{String, String}}
    scores = Dict{String, Float64}()
    n_fetched = 0
    n_failed = 0

    # Batch-resolve gene names to UniProt IDs
    gene_names = String[row.Protein for row in eachrow(candidates)]
    uniprot_map = _resolve_uniprot_ids(gene_names; species=species, resolve_uniprot=resolve_uniprot, cache_base=cache_base)

    prog = Progress(nrow(candidates); desc="Fetching pLDDT scores  ", enabled=nrow(candidates) > 1)
    for row in eachrow(candidates)
        protein = row.Protein

        # Try resolved mapping first, then fall back to DataFrame columns
        uniprot_id = get(uniprot_map, protein, "")
        if isempty(uniprot_id)
            uniprot_id = _get_uniprot(row)
        end

        if isempty(uniprot_id)
            scores[protein] = 0.5
            n_failed += 1
            ProgressMeter.next!(prog)
            continue
        end

        result = fetch_alphafold_plddt(uniprot_id;
                                        cache_base=cache_base,
                                        cache_max_age_days=cache_max_age_days,
                                        force_refresh=force_refresh)
        if result === nothing
            scores[protein] = 0.5
            n_failed += 1
        else
            scores[protein] = compute_dockability(result.mean_plddt, result.frac_disordered)
            n_fetched += 1
        end
        ProgressMeter.next!(prog)
    end
    ProgressMeter.finish!(prog)

    @info "AlphaFold DB pLDDT fetch complete" n_fetched n_failed total=nrow(candidates)
    return (scores, uniprot_map)
end
