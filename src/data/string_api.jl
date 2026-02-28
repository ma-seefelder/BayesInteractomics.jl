#=
STRING API Client for Protein Data Curation

Self-contained STRING DB client for resolving protein names to canonical IDs.
Replicates patterns from ext/BayesInteractomicsNetworkExt/ppi_query.jl but
independent of the extension, using only stdlib (Downloads, Dates, SHA) and CSV.
=#

import Downloads
import Dates: DateTime, now, Millisecond, value
import SHA: sha256
import CSV: read as csv_read
import DataFrames: DataFrame, nrow, eachrow

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

const CURATION_STRING_API_BASE = "https://version-12-0.string-db.org/api"
const CURATION_RATE_LIMIT_MS = 1000
const CURATION_CHUNK_SIZE = 500
const CURATION_MAX_RETRIES = 3
const CURATION_RETRY_DELAY_S = 5
const CURATION_TIMEOUT_S = 30
const CURATION_CACHE_VERSION = 1

# Module-level mutable state for rate limiting
const _curation_last_query_time = Ref(DateTime(0))

# ─────────────────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────────────────

"""
    CurationAPIError <: Exception

Error from STRING database API during protein curation.
"""
struct CurationAPIError <: Exception
    msg::String
    status_code::Union{Int, Nothing}
    suggestion::String
end

function Base.showerror(io::IO, e::CurationAPIError)
    println(io, "CurationAPIError: ", e.msg)
    if !isnothing(e.status_code)
        println(io, "  HTTP status: ", e.status_code)
    end
    print(io, "  Suggestion: ", e.suggestion)
end

# ─────────────────────────────────────────────────────────────────────────────
# Rate limiting
# ─────────────────────────────────────────────────────────────────────────────

function _curation_rate_limit_wait()
    elapsed_ms = value(Millisecond(now() - _curation_last_query_time[]))
    if elapsed_ms < CURATION_RATE_LIMIT_MS
        sleep((CURATION_RATE_LIMIT_MS - elapsed_ms) / 1000.0)
    end
    _curation_last_query_time[] = now()
end

# ─────────────────────────────────────────────────────────────────────────────
# Form encoding
# ─────────────────────────────────────────────────────────────────────────────

function _curation_percent_encode(s::String)::String
    io = IOBuffer()
    for c in s
        if c in 'A':'Z' || c in 'a':'z' || c in '0':'9' || c in ('-', '_', '.', '~')
            write(io, c)
        elseif c == ' '
            write(io, '+')
        elseif c == '%'
            write(io, c)
        else
            write(io, string('%', uppercase(string(UInt8(c), base=16, pad=2))))
        end
    end
    return String(take!(io))
end

function _curation_form_encode(params::Dict{String,String})::String
    parts = String[]
    for (k, v) in params
        push!(parts, string(_curation_percent_encode(k), "=", _curation_percent_encode(v)))
    end
    return join(parts, "&")
end

# ─────────────────────────────────────────────────────────────────────────────
# HTTP request with retries
# ─────────────────────────────────────────────────────────────────────────────

"""
    _curation_string_request(endpoint, params; timeout_s) -> String

Execute a POST request to the STRING API with rate limiting and retry logic.
Returns the response body as a String.
"""
function _curation_string_request(
    endpoint::String,
    params::Dict{String,String};
    timeout_s::Int = CURATION_TIMEOUT_S
)::String
    _curation_rate_limit_wait()

    url = CURATION_STRING_API_BASE * "/" * endpoint
    body_str = _curation_form_encode(params)
    headers = ["Content-Type" => "application/x-www-form-urlencoded"]

    last_error = nothing

    for attempt in 1:CURATION_MAX_RETRIES
        try
            body_io = IOBuffer(body_str)
            response_io = IOBuffer()

            resp = Downloads.request(
                url;
                input = body_io,
                output = response_io,
                method = "POST",
                headers = headers,
                timeout = timeout_s,
                throw = false
            )

            status = resp.status

            if status == 200
                return String(take!(response_io))
            elseif status == 429
                @warn "STRING API rate limited (429), waiting 10s before retry $attempt/$CURATION_MAX_RETRIES"
                sleep(10)
                _curation_last_query_time[] = now()
                continue
            elseif status >= 500
                @warn "STRING API server error ($status), retrying in $(CURATION_RETRY_DELAY_S)s (attempt $attempt/$CURATION_MAX_RETRIES)"
                sleep(CURATION_RETRY_DELAY_S)
                _curation_last_query_time[] = now()
                continue
            else
                response_body = String(take!(response_io))
                throw(CurationAPIError(
                    "HTTP $status from STRING API endpoint '$endpoint'",
                    status,
                    "Check parameters and species ID. Response: $(first(response_body, 200))"
                ))
            end
        catch e
            if e isa CurationAPIError
                rethrow(e)
            end
            last_error = e
            if attempt < CURATION_MAX_RETRIES
                @warn "STRING API request failed, retrying in $(CURATION_RETRY_DELAY_S)s (attempt $attempt/$CURATION_MAX_RETRIES)" exception=e
                sleep(CURATION_RETRY_DELAY_S)
                _curation_last_query_time[] = now()
            end
        end
    end

    throw(CurationAPIError(
        "STRING API request failed after $CURATION_MAX_RETRIES attempts: $(something(last_error, "unknown error"))",
        nothing,
        "Check your internet connection or provide a pre-computed curation report for replay."
    ))
end

# ─────────────────────────────────────────────────────────────────────────────
# ID Resolution — batch query with preferred names and annotations
# ─────────────────────────────────────────────────────────────────────────────

"""
    _resolve_names_via_string(names, species; caller_identity) -> NamedTuple

Resolve protein names to STRING identifiers, preferred names, and annotations.
Chunks large queries into batches of `$CURATION_CHUNK_SIZE`.

# Returns
A NamedTuple with fields:
- `name_to_id::Dict{String,String}`:       original_name → STRING_ID
- `id_to_preferred::Dict{String,String}`:  STRING_ID → preferred gene name
- `id_to_annotation::Dict{String,String}`: STRING_ID → functional annotation
- `unmapped::Vector{String}`:              names not found in STRING
"""
function _resolve_names_via_string(
    names::Vector{String},
    species::Int;
    caller_identity::String = "BayesInteractomics.jl"
)
    name_to_id = Dict{String, String}()
    id_to_preferred = Dict{String, String}()
    id_to_annotation = Dict{String, String}()
    unmapped = String[]

    unique_names = unique(names)

    for chunk_start in 1:CURATION_CHUNK_SIZE:length(unique_names)
        chunk_end = min(chunk_start + CURATION_CHUNK_SIZE - 1, length(unique_names))
        chunk = unique_names[chunk_start:chunk_end]

        # Use %0d (carriage return) as separator — STRING API convention
        identifiers_str = join(chunk, "%0d")

        params = Dict{String,String}(
            "identifiers" => identifiers_str,
            "species" => string(species),
            "echo_query" => "1",
            "caller_identity" => caller_identity
        )

        response_text = _curation_string_request("tsv/get_string_ids", params)

        if isempty(strip(response_text))
            append!(unmapped, chunk)
            continue
        end

        df = csv_read(IOBuffer(response_text), DataFrame; delim='\t', silencewarnings=true)

        if nrow(df) == 0
            append!(unmapped, chunk)
            continue
        end

        # Take best match per query item (STRING ranks by relevance)
        seen_queries = Set{String}()
        for row in eachrow(df)
            query_item = string(row.queryItem)
            if query_item in seen_queries
                continue  # Skip secondary/ambiguous matches
            end
            push!(seen_queries, query_item)

            string_id = string(row.stringId)
            name_to_id[query_item] = string_id
            id_to_preferred[string_id] = string(row.preferredName)

            # Annotation column may not always be present
            if hasproperty(row, :annotation) && !ismissing(row.annotation)
                id_to_annotation[string_id] = string(row.annotation)
            else
                id_to_annotation[string_id] = ""
            end
        end

        # Find unmapped proteins in this chunk
        for name in chunk
            if !haskey(name_to_id, name)
                push!(unmapped, name)
            end
        end
    end

    return (
        name_to_id = name_to_id,
        id_to_preferred = id_to_preferred,
        id_to_annotation = id_to_annotation,
        unmapped = unmapped
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Cache key computation
# ─────────────────────────────────────────────────────────────────────────────

"""
    _curation_cache_key(names, species) -> String

Compute a SHA256-based cache key for a set of protein names and species.
"""
function _curation_cache_key(names::Vector{String}, species::Int)::String
    sorted = sort(unique(names))
    input_str = join(sorted, ",") * "|" * string(species)
    return bytes2hex(sha256(input_str))
end

# ─────────────────────────────────────────────────────────────────────────────
# JLD2 cache persistence
# ─────────────────────────────────────────────────────────────────────────────

"""
    save_curation_cache(cache::CurationCache, cache_dir::String, key::String)

Save STRING ID mapping cache to JLD2 file.
"""
function save_curation_cache(cache::CurationCache, cache_dir::String, key::String)
    mkpath(cache_dir)
    filepath = joinpath(cache_dir, "curation_$(key).jld2")
    try
        jldsave(filepath; compress=true,
            cache_version = CURATION_CACHE_VERSION,
            mapping = cache.mapping,
            preferred_names = cache.preferred_names,
            annotations = cache.annotations,
            species = cache.species,
            string_version = cache.string_version,
            query_timestamp = string(cache.query_timestamp)
        )
    catch e
        @warn "Failed to save curation cache" filepath exception=e
    end
end

"""
    load_curation_cache(cache_dir::String, key::String) -> Union{CurationCache, Nothing}

Load STRING ID mapping cache from JLD2 file.
Returns `nothing` if cache is missing, incompatible, or corrupted.
"""
function load_curation_cache(cache_dir::String, key::String)::Union{CurationCache, Nothing}
    filepath = joinpath(cache_dir, "curation_$(key).jld2")
    !isfile(filepath) && return nothing

    try
        data = JLD2.load(filepath)

        # Version check
        get(data, "cache_version", 0) != CURATION_CACHE_VERSION && return nothing

        # Reconstruct DateTime (JLD2 can have issues with DateTime)
        timestamp = data["query_timestamp"]
        if timestamp isa String
            timestamp = DateTime(timestamp)
        end

        cache = CurationCache(
            data["mapping"],
            data["preferred_names"],
            data["annotations"],
            data["species"],
            data["string_version"],
            timestamp
        )

        # Warn if cache is old (>90 days)
        age_ms = value(Millisecond(now() - cache.query_timestamp))
        age_days = div(age_ms, 86_400_000)
        if age_days > 90
            @warn "Curation cache is $(age_days) days old. Consider clearing it to get updated STRING mappings."
        end

        return cache
    catch e
        @warn "Failed to load curation cache, will re-query STRING" filepath exception=e
        return nothing
    end
end
