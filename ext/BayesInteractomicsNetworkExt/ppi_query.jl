# STRING database API client for prey-prey interaction queries
#
# Uses Downloads.jl (stdlib) for HTTP requests and CSV for TSV parsing.
# Implements ID resolution, network querying, rate limiting, caching, and error handling.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
const STRING_API_BASE = "https://version-12-0.string-db.org/api"
const STRING_RATE_LIMIT_MS = 1000
const STRING_ID_CHUNK_SIZE = 500
const STRING_NETWORK_CHUNK_SIZE = 2000
const STRING_MAX_RETRIES = 3
const STRING_RETRY_DELAY_S = 5
const STRING_TIMEOUT_S = 30

# Module-level mutable state for rate limiting
const _last_query_time = Ref(DateTime(0))

# ---------------------------------------------------------------------------
# Custom exception types
# ---------------------------------------------------------------------------

"""
    StringAPIError <: Exception

Error from STRING database API communication.
"""
struct StringAPIError <: Exception
    msg::String
    status_code::Union{Int, Nothing}
    suggestion::String
end

function Base.showerror(io::IO, e::StringAPIError)
    println(io, "StringAPIError: ", e.msg)
    if !isnothing(e.status_code)
        println(io, "  HTTP status: ", e.status_code)
    end
    print(io, "  Suggestion: ", e.suggestion)
end

# ---------------------------------------------------------------------------
# PPIQueryResult type
# ---------------------------------------------------------------------------

"""
    PPIQueryResult

Result of a STRING database query for pairwise protein interactions.

# Fields
- `interactions::DataFrame`: Interaction data with columns protein_a, protein_b, combined_score, and per-channel scores
- `protein_mapping::Dict{String,String}`: user_name => STRING_id mapping
- `unmapped_proteins::Vector{String}`: Proteins not found in STRING
- `species::Int`: NCBI taxonomy ID used
- `string_version::String`: STRING version (e.g., "12.0")
- `query_timestamp::DateTime`: When the query was made
- `network_type::String`: "physical" or "functional"
- `n_query_proteins::Int`: Number of proteins queried
- `n_pairs_with_data::Int`: Number of pairs with STRING score > 0
"""
struct PPIQueryResult
    interactions::DataFrame
    protein_mapping::Dict{String, String}
    unmapped_proteins::Vector{String}
    species::Int
    string_version::String
    query_timestamp::DateTime
    network_type::String
    n_query_proteins::Int
    n_pairs_with_data::Int
end

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

function _rate_limit_wait()
    elapsed_ms = Dates.value(Dates.Millisecond(now() - _last_query_time[]))
    if elapsed_ms < STRING_RATE_LIMIT_MS
        sleep((STRING_RATE_LIMIT_MS - elapsed_ms) / 1000.0)
    end
    _last_query_time[] = now()
end

# ---------------------------------------------------------------------------
# Form encoding (manual, since protein names are alphanumeric)
# ---------------------------------------------------------------------------

"""
    _form_encode(params::Dict{String,String}) -> String

Manually form-encode parameters for HTTP POST. Handles basic percent-encoding
for spaces and special characters commonly found in protein identifiers.
"""
function _form_encode(params::Dict{String,String})::String
    parts = String[]
    for (k, v) in params
        encoded_k = _percent_encode(k)
        encoded_v = _percent_encode(v)
        push!(parts, string(encoded_k, "=", encoded_v))
    end
    return join(parts, "&")
end

function _percent_encode(s::String)::String
    io = IOBuffer()
    for c in s
        if c in 'A':'Z' || c in 'a':'z' || c in '0':'9' || c in ('-', '_', '.', '~')
            write(io, c)
        elseif c == ' '
            write(io, '+')
        elseif c == '%'
            # Already encoded sequences like %0d pass through
            write(io, c)
        else
            # Percent-encode
            write(io, string('%', uppercase(string(UInt8(c), base=16, pad=2))))
        end
    end
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# STRING API request
# ---------------------------------------------------------------------------

"""
    _string_api_request(endpoint, params; timeout_s) -> String

Execute a POST request to the STRING API with rate limiting and retry logic.
Returns the response body as a String.
"""
function _string_api_request(
    endpoint::String,
    params::Dict{String,String};
    timeout_s::Int = STRING_TIMEOUT_S
)::String
    _rate_limit_wait()

    url = STRING_API_BASE * "/" * endpoint
    body_str = _form_encode(params)
    headers = ["Content-Type" => "application/x-www-form-urlencoded"]

    last_error = nothing

    for attempt in 1:STRING_MAX_RETRIES
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
                # Rate limited — wait longer and retry
                @warn "STRING API rate limited (429), waiting 10s before retry $attempt/$STRING_MAX_RETRIES"
                sleep(10)
                _last_query_time[] = now()
                continue
            elseif status >= 500
                # Server error — retry after delay
                @warn "STRING API server error ($status), retrying in $(STRING_RETRY_DELAY_S)s (attempt $attempt/$STRING_MAX_RETRIES)"
                sleep(STRING_RETRY_DELAY_S)
                _last_query_time[] = now()
                continue
            else
                response_body = String(take!(response_io))
                throw(StringAPIError(
                    "HTTP $status from STRING API endpoint '$endpoint'",
                    status,
                    "Check your parameters and species ID. Response: $(first(response_body, 200))"
                ))
            end
        catch e
            if e isa StringAPIError
                rethrow(e)
            end
            last_error = e
            if attempt < STRING_MAX_RETRIES
                @warn "STRING API request failed, retrying in $(STRING_RETRY_DELAY_S)s (attempt $attempt/$STRING_MAX_RETRIES)" exception=e
                sleep(STRING_RETRY_DELAY_S)
                _last_query_time[] = now()
            end
        end
    end

    throw(StringAPIError(
        "STRING API request failed after $STRING_MAX_RETRIES attempts: $(something(last_error, "unknown error"))",
        nothing,
        "Check your internet connection or use offline mode with a local STRING file."
    ))
end

# ---------------------------------------------------------------------------
# ID Resolution
# ---------------------------------------------------------------------------

"""
    _resolve_string_ids(protein_names, species; caller_identity) -> (mapping, unmapped)

Resolve protein names to STRING identifiers using the get_string_ids endpoint.
Chunks large queries into batches of $STRING_ID_CHUNK_SIZE.
Returns a tuple of (Dict{String,String} mapping, Vector{String} unmapped).
"""
function _resolve_string_ids(
    protein_names::Vector{String},
    species::Int;
    caller_identity::String = "BayesInteractomics.jl"
)
    mapping = Dict{String, String}()
    unmapped = String[]

    # Process in chunks
    for chunk_start in 1:STRING_ID_CHUNK_SIZE:length(protein_names)
        chunk_end = min(chunk_start + STRING_ID_CHUNK_SIZE - 1, length(protein_names))
        chunk = protein_names[chunk_start:chunk_end]

        identifiers_str = join(chunk, "%0d")

        params = Dict{String,String}(
            "identifiers" => identifiers_str,
            "species" => string(species),
            "echo_query" => "1",
            "caller_identity" => caller_identity
        )

        response_text = _string_api_request("tsv/get_string_ids", params)

        if isempty(strip(response_text))
            append!(unmapped, chunk)
            continue
        end

        # Parse TSV response
        df = CSV.read(IOBuffer(response_text), DataFrame; delim='\t', silencewarnings=true)

        if nrow(df) == 0
            append!(unmapped, chunk)
            continue
        end

        # Take best match per query item (STRING ranks by relevance; first hit is best)
        seen_queries = Set{String}()
        for row in eachrow(df)
            query_item = string(row.queryItem)
            if query_item in seen_queries
                continue  # Skip ambiguous secondary matches
            end
            push!(seen_queries, query_item)
            mapping[query_item] = string(row.stringId)
        end

        # Find unmapped proteins in this chunk
        for name in chunk
            if !haskey(mapping, name)
                push!(unmapped, name)
            end
        end
    end

    return mapping, unmapped
end

# ---------------------------------------------------------------------------
# Network Query
# ---------------------------------------------------------------------------

"""
    _query_string_network(string_ids, species; network_type, caller_identity) -> DataFrame

Query STRING /network endpoint for all pairwise interactions among the given STRING IDs.
Chunks large queries. Returns raw interaction DataFrame.
"""
function _query_string_network(
    string_ids::Vector{String},
    species::Int;
    network_type::Symbol = :physical,
    caller_identity::String = "BayesInteractomics.jl"
)
    all_interactions = DataFrame[]

    nt_str = network_type == :physical ? "physical" : "functional"

    for chunk_start in 1:STRING_NETWORK_CHUNK_SIZE:length(string_ids)
        chunk_end = min(chunk_start + STRING_NETWORK_CHUNK_SIZE - 1, length(string_ids))
        chunk = string_ids[chunk_start:chunk_end]

        identifiers_str = join(chunk, "%0d")

        params = Dict{String,String}(
            "identifiers" => identifiers_str,
            "species" => string(species),
            "required_score" => "0",
            "network_type" => nt_str,
            "caller_identity" => caller_identity
        )

        response_text = _string_api_request("tsv/network", params)

        if isempty(strip(response_text))
            continue
        end

        df = CSV.read(IOBuffer(response_text), DataFrame; delim='\t', silencewarnings=true)

        if nrow(df) > 0
            push!(all_interactions, df)
        end
    end

    if isempty(all_interactions)
        return DataFrame(
            stringId_A = String[],
            stringId_B = String[],
            preferredName_A = String[],
            preferredName_B = String[],
            score = Int[]
        )
    end

    result = vcat(all_interactions...; cols=:union)

    # Deduplicate: normalize pair ordering and keep first occurrence
    if nrow(result) > 0
        # Create canonical pair key for dedup
        pair_keys = Set{Tuple{String,String}}()
        keep_mask = trues(nrow(result))
        for i in 1:nrow(result)
            a = result.stringId_A[i]
            b = result.stringId_B[i]
            pair = a < b ? (a, b) : (b, a)
            if pair in pair_keys
                keep_mask[i] = false
            else
                push!(pair_keys, pair)
            end
        end
        result = result[keep_mask, :]
    end

    return result
end

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

function _default_cache_dir()::String
    return joinpath(homedir(), ".bayesinteractomics", "ppi_cache")
end

function _resolve_cache_dir(cache_dir::String)::String
    dir = isempty(cache_dir) ? _default_cache_dir() : cache_dir
    return dir
end

function _cache_key(proteins::Vector{String}, species::Int, network_type::Symbol)::String
    sorted = sort(proteins)
    input_str = join(sorted, ",") * "|" * string(species) * "|" * string(network_type)
    return bytes2hex(sha256(input_str))
end

function _cache_filepath(key::String, cache_dir::String)::String
    return joinpath(_resolve_cache_dir(cache_dir), key * ".jld2")
end

function _load_cache(key::String, cache_dir::String)::Union{PPIQueryResult, Nothing}
    filepath = _cache_filepath(key, cache_dir)
    if !isfile(filepath)
        return nothing
    end

    try
        data = JLD2.load(filepath)

        # Reconstruct DateTime from string (JLD2 can have issues with DateTime)
        timestamp = data["query_timestamp"]
        if timestamp isa String
            timestamp = DateTime(timestamp)
        end

        result = PPIQueryResult(
            data["interactions"],
            data["protein_mapping"],
            data["unmapped_proteins"],
            data["species"],
            data["string_version"],
            timestamp,
            data["network_type"],
            data["n_query_proteins"],
            data["n_pairs_with_data"]
        )

        # Warn if cache is old (>90 days)
        age_ms = Dates.value(Dates.Millisecond(now() - result.query_timestamp))
        age_days = div(age_ms, 86_400_000)
        if age_days > 90
            @warn "Cached STRING results are $(age_days) days old. Consider using force_refresh=true."
        end

        return result
    catch e
        @warn "Failed to load cached PPI data, will re-query" filepath exception=e
        return nothing
    end
end

function _save_cache(key::String, result::PPIQueryResult, cache_dir::String)
    dir = _resolve_cache_dir(cache_dir)
    mkpath(dir)
    filepath = joinpath(dir, key * ".jld2")

    try
        JLD2.jldsave(filepath;
            interactions = result.interactions,
            protein_mapping = result.protein_mapping,
            unmapped_proteins = result.unmapped_proteins,
            species = result.species,
            string_version = result.string_version,
            query_timestamp = string(result.query_timestamp),  # Store as String for JLD2 compat
            network_type = result.network_type,
            n_query_proteins = result.n_query_proteins,
            n_pairs_with_data = result.n_pairs_with_data
        )
    catch e
        @warn "Failed to cache PPI results" filepath exception=e
    end
end

# ---------------------------------------------------------------------------
# Public API: query_string_ppi
# ---------------------------------------------------------------------------

"""
    BayesInteractomics.query_string_ppi(proteins, species; kwargs...) -> PPIQueryResult

Query STRING database for all pairwise interactions among the given proteins.

# Arguments
- `proteins::Vector{String}`: Protein names (gene symbols, UniProt IDs, etc.)
- `species::Int`: NCBI taxonomy ID (9606=human, 10090=mouse, 10116=rat)

# Keyword Arguments
- `network_type::Symbol=:physical`: `:physical` or `:functional`
- `caller_identity::String="BayesInteractomics.jl"`: Identifier for STRING API
- `cache_dir::String=""`: Cache directory (empty=default)
- `force_refresh::Bool=false`: Ignore cache and re-query
- `protein_mapping::Union{Dict{String,String},Nothing}=nothing`: Manual protein→STRING ID overrides
"""
function BayesInteractomics.query_string_ppi(
    proteins::Vector{String},
    species::Int;
    network_type::Symbol = :physical,
    caller_identity::String = "BayesInteractomics.jl",
    cache_dir::String = "",
    force_refresh::Bool = false,
    protein_mapping::Union{Dict{String,String}, Nothing} = nothing
)
    if isempty(proteins)
        return PPIQueryResult(
            DataFrame(), Dict{String,String}(), String[],
            species, "12.0", now(), string(network_type), 0, 0
        )
    end

    # Check cache first
    key = _cache_key(proteins, species, network_type)
    if !force_refresh
        cached = _load_cache(key, cache_dir)
        if !isnothing(cached)
            @info "Using cached STRING results" n_proteins=cached.n_query_proteins n_interactions=nrow(cached.interactions)
            return cached
        end
    end

    # Resolve protein IDs
    if !isnothing(protein_mapping)
        # Use provided mapping for specified proteins, resolve rest
        pre_mapped = Dict{String,String}()
        to_resolve = String[]
        for p in proteins
            if haskey(protein_mapping, p)
                pre_mapped[p] = protein_mapping[p]
            else
                push!(to_resolve, p)
            end
        end

        if !isempty(to_resolve)
            resolved, unmapped = _resolve_string_ids(to_resolve, species; caller_identity)
            merge!(pre_mapped, resolved)
        else
            unmapped = String[]
        end
        final_mapping = pre_mapped
    else
        final_mapping, unmapped = _resolve_string_ids(proteins, species; caller_identity)
    end

    n_mapped = length(final_mapping)
    n_unmapped = length(unmapped)

    if !isempty(unmapped)
        @warn "Some proteins could not be mapped to STRING" n_unmapped unmapped=unmapped
    end

    if n_mapped == 0
        @warn "No proteins could be mapped to STRING IDs"
        return PPIQueryResult(
            DataFrame(), final_mapping, unmapped,
            species, "12.0", now(), string(network_type),
            length(proteins), 0
        )
    end

    if n_mapped < 2
        @info "Only 1 protein mapped to STRING — no pairwise interactions possible"
        return PPIQueryResult(
            DataFrame(), final_mapping, unmapped,
            species, "12.0", now(), string(network_type),
            length(proteins), 0
        )
    end

    # Query network
    string_ids = collect(values(final_mapping))
    interactions_df = _query_string_network(string_ids, species; network_type, caller_identity)

    n_pairs = nrow(interactions_df)

    result = PPIQueryResult(
        interactions_df,
        final_mapping,
        unmapped,
        species,
        "12.0",
        now(),
        string(network_type),
        length(proteins),
        n_pairs
    )

    # Cache the result
    _save_cache(key, result, cache_dir)

    return result
end

# ---------------------------------------------------------------------------
# Public API: clear_ppi_cache
# ---------------------------------------------------------------------------

"""
    BayesInteractomics.clear_ppi_cache(; cache_dir="")

Delete all cached STRING PPI query results.
"""
function BayesInteractomics.clear_ppi_cache(; cache_dir::String = "")
    dir = _resolve_cache_dir(cache_dir)
    if !isdir(dir)
        @info "No PPI cache directory found at $dir"
        return
    end

    files = filter(f -> endswith(f, ".jld2"), readdir(dir; join=true))
    n = length(files)
    for f in files
        rm(f)
    end
    @info "Cleared PPI cache" n_entries=n cache_dir=dir
end

# ---------------------------------------------------------------------------
# Public API: ppi_cache_info
# ---------------------------------------------------------------------------

"""
    BayesInteractomics.ppi_cache_info(; cache_dir="") -> NamedTuple

Report cache size, number of entries, and age of oldest/newest entries.
"""
function BayesInteractomics.ppi_cache_info(; cache_dir::String = "")
    dir = _resolve_cache_dir(cache_dir)

    if !isdir(dir)
        return (n_entries=0, total_size_bytes=0, oldest=nothing, newest=nothing, cache_dir=dir)
    end

    files = filter(f -> endswith(f, ".jld2"), readdir(dir; join=true))
    n = length(files)

    if n == 0
        return (n_entries=0, total_size_bytes=0, oldest=nothing, newest=nothing, cache_dir=dir)
    end

    total_size = sum(filesize.(files))
    mtimes = [Dates.unix2datetime(mtime(f)) for f in files]
    oldest = minimum(mtimes)
    newest = maximum(mtimes)

    return (n_entries=n, total_size_bytes=total_size, oldest=oldest, newest=newest, cache_dir=dir)
end
