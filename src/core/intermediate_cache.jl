# Intermediate Results Caching System
# Provides separate caching for Beta-Bernoulli and HBM+Regression results
# to enable partial cache hits when only some parameters change

using JLD2
using Dates
import DataFrames: DataFrame

const INTERMEDIATE_CACHE_VERSION = 2

"""
    IntermediateCacheStatus

Enum representing intermediate cache validation status.

# Values
- `INTERMEDIATE_CACHE_HIT`: Cache is valid and can be used
- `INTERMEDIATE_CACHE_MISS_NO_FILE`: Cache file doesn't exist
- `INTERMEDIATE_CACHE_MISS_PARAMS`: Parameters changed (n_controls, n_samples, or refID)
- `INTERMEDIATE_CACHE_MISS_DATA`: Data hash mismatch (input data changed)

See also: [`check_betabernoulli_cache`](@ref), [`check_hbm_regression_cache`](@ref)
"""
@enum IntermediateCacheStatus begin
    INTERMEDIATE_CACHE_HIT
    INTERMEDIATE_CACHE_MISS_NO_FILE
    INTERMEDIATE_CACHE_MISS_PARAMS
    INTERMEDIATE_CACHE_MISS_DATA
end

"""
    BetaBernoulliCache

Cache container for Beta-Bernoulli model results.

Beta-Bernoulli results depend on:
- Input data (samples and controls)
- `n_controls`: Number of control replicates
- `n_samples`: Number of sample replicates

Beta-Bernoulli does NOT depend on `refID` (reference protein).

# Fields
- `bf_detected::Vector{Float64}`: Bayes factors for detection (one per protein)
- `protein_ids::Vector{String}`: Protein identifiers for validation
- `n_controls::Int`: Number of controls used
- `n_samples::Int`: Number of samples used
- `data_hash::UInt64`: Hash of input data for validation
- `timestamp::DateTime`: When cache was created
- `package_version::String`: Package version for compatibility checking

See also: [`save_betabernoulli_cache`](@ref), [`load_betabernoulli_cache`](@ref)
"""
struct BetaBernoulliCache
    bf_detected::Vector{Float64}
    protein_ids::Vector{String}
    n_controls::Int
    n_samples::Int
    data_hash::UInt64
    timestamp::DateTime
    package_version::String
end

"""
    HBMRegressionCache

Cache container for HBM (Hierarchical Bayesian Model) and Regression results.

HBM+Regression results depend on:
- Input data (samples and controls)
- `refID`: Reference protein index (bait protein)
- `regression_likelihood`: Likelihood type (`:normal` or `:robust_t`)
- `student_t_nu`: Degrees of freedom for Student-t (only relevant when `:robust_t`)

HBM+Regression does NOT depend on `n_controls` or `n_samples` (those only affect Beta-Bernoulli).

# Fields
- `df_hierarchical::DataFrame`: Detailed hierarchical model results
- `bf_enrichment::Vector{Float64}`: Bayes factors for enrichment (log2FC)
- `bf_correlation::Vector{Float64}`: Bayes factors for dose-response correlation
- `protein_ids::Vector{String}`: Protein identifiers for validation
- `refID::Int`: Reference protein index used
- `regression_likelihood::Symbol`: Regression likelihood (`:normal` or `:robust_t`)
- `student_t_nu::Float64`: Student-t degrees of freedom used
- `data_hash::UInt64`: Hash of input data for validation
- `timestamp::DateTime`: When cache was created
- `package_version::String`: Package version for compatibility checking

See also: [`save_hbm_regression_cache`](@ref), [`load_hbm_regression_cache`](@ref)
"""
struct HBMRegressionCache
    df_hierarchical::DataFrame
    bf_enrichment::Vector{Float64}
    bf_correlation::Vector{Float64}
    protein_ids::Vector{String}
    refID::Int
    regression_likelihood::Symbol
    student_t_nu::Float64
    data_hash::UInt64
    timestamp::DateTime
    package_version::String
end

# ----------------------- Hash Functions ----------------------- #

"""
    compute_betabernoulli_hash(data, n_controls::Int, n_samples::Int)::UInt64

Compute hash for Beta-Bernoulli cache validation.

Combines data hash with n_controls and n_samples parameters.

# Arguments
- `data`: InteractionData or (Vector{InteractionData}, InteractionData) for multiple imputation
- `n_controls::Int`: Number of control replicates
- `n_samples::Int`: Number of sample replicates

# Returns
- `UInt64`: Hash value for cache validation

# Examples
```julia
h = compute_betabernoulli_hash(data, 3, 3)
```
"""
function compute_betabernoulli_hash(data, n_controls::Int, n_samples::Int)::UInt64
    data_h = compute_data_hash(data)
    return hash((data_h, n_controls, n_samples))
end

"""
    compute_hbm_regression_hash(data, refID::Int, regression_likelihood::Symbol, student_t_nu::Float64)::UInt64

Compute hash for HBM+Regression cache validation.

Combines data hash with refID, regression_likelihood, and student_t_nu parameters.

# Arguments
- `data`: InteractionData or (Vector{InteractionData}, InteractionData) for multiple imputation
- `refID::Int`: Reference protein index
- `regression_likelihood::Symbol`: Likelihood type (`:normal` or `:robust_t`)
- `student_t_nu::Float64`: Degrees of freedom for Student-t

# Returns
- `UInt64`: Hash value for cache validation

# Examples
```julia
h = compute_hbm_regression_hash(data, 1, :robust_t, 5.0)
```
"""
function compute_hbm_regression_hash(data, refID::Int, regression_likelihood::Symbol, student_t_nu::Float64)::UInt64
    data_h = compute_data_hash(data)
    return hash((data_h, refID, regression_likelihood, student_t_nu))
end

# ----------------------- Serialization ----------------------- #

"""
    save_betabernoulli_cache(cache::BetaBernoulliCache, filepath::String)

Save Beta-Bernoulli cache to disk using JLD2 format with compression.

# Arguments
- `cache::BetaBernoulliCache`: Cache structure to save
- `filepath::String`: Output file path (typically .jld2 extension)

# Examples
```julia
cache = BetaBernoulliCache(bf_detected, protein_ids, 3, 3, hash_val, now(), version)
save_betabernoulli_cache(cache, ".bayesinteractomics_cache/betabernoulli_abc123.jld2")
```

See also: [`load_betabernoulli_cache`](@ref)
"""
function save_betabernoulli_cache(cache::BetaBernoulliCache, filepath::String)
    jldsave(filepath; compress=true,
        cache_version = INTERMEDIATE_CACHE_VERSION,
        bf_detected = cache.bf_detected,
        protein_ids = cache.protein_ids,
        n_controls = cache.n_controls,
        n_samples = cache.n_samples,
        data_hash = cache.data_hash,
        timestamp = cache.timestamp,
        package_version = cache.package_version
    )
end

"""
    load_betabernoulli_cache(filepath::String)::Union{BetaBernoulliCache, Nothing}

Load Beta-Bernoulli cache from disk, or return nothing if invalid/missing.

Returns `nothing` if:
- File doesn't exist
- Cache version mismatch
- Loading fails due to corruption

# Arguments
- `filepath::String`: Path to cached result file

# Returns
- `BetaBernoulliCache`: Loaded cache structure
- `nothing`: If file is missing, incompatible, or corrupted

# Examples
```julia
cached = load_betabernoulli_cache("betabernoulli_abc123.jld2")
if !isnothing(cached)
    bf_detected = cached.bf_detected
else
    # Compute from scratch
end
```

See also: [`save_betabernoulli_cache`](@ref)
"""
function load_betabernoulli_cache(filepath::String)::Union{BetaBernoulliCache, Nothing}
    !isfile(filepath) && return nothing
    try
        data = load(filepath)
        get(data, "cache_version", 0) != INTERMEDIATE_CACHE_VERSION && return nothing
        return BetaBernoulliCache(
            data["bf_detected"],
            data["protein_ids"],
            data["n_controls"],
            data["n_samples"],
            data["data_hash"],
            data["timestamp"],
            data["package_version"]
        )
    catch e
        @warn "Failed to load Beta-Bernoulli cache: $e"
        return nothing
    end
end

"""
    save_hbm_regression_cache(cache::HBMRegressionCache, filepath::String)

Save HBM+Regression cache to disk using JLD2 format with compression.

# Arguments
- `cache::HBMRegressionCache`: Cache structure to save
- `filepath::String`: Output file path (typically .jld2 extension)

# Examples
```julia
cache = HBMRegressionCache(df, bf_enrich, bf_corr, protein_ids, 1, :robust_t, 5.0, hash_val, now(), version)
save_hbm_regression_cache(cache, ".bayesinteractomics_cache/hbm_regression_abc123_ref1_robust_t_nu5.0.jld2")
```

See also: [`load_hbm_regression_cache`](@ref)
"""
function save_hbm_regression_cache(cache::HBMRegressionCache, filepath::String)
    jldsave(filepath; compress=true,
        cache_version = INTERMEDIATE_CACHE_VERSION,
        df_hierarchical = cache.df_hierarchical,
        bf_enrichment = cache.bf_enrichment,
        bf_correlation = cache.bf_correlation,
        protein_ids = cache.protein_ids,
        refID = cache.refID,
        regression_likelihood = cache.regression_likelihood,
        student_t_nu = cache.student_t_nu,
        data_hash = cache.data_hash,
        timestamp = cache.timestamp,
        package_version = cache.package_version
    )
end

"""
    load_hbm_regression_cache(filepath::String)::Union{HBMRegressionCache, Nothing}

Load HBM+Regression cache from disk, or return nothing if invalid/missing.

Returns `nothing` if:
- File doesn't exist
- Cache version mismatch
- Loading fails due to corruption

# Arguments
- `filepath::String`: Path to cached result file

# Returns
- `HBMRegressionCache`: Loaded cache structure
- `nothing`: If file is missing, incompatible, or corrupted

# Examples
```julia
cached = load_hbm_regression_cache("hbm_regression_abc123_ref1.jld2")
if !isnothing(cached)
    df_hierarchical = cached.df_hierarchical
    bf_enrichment = cached.bf_enrichment
else
    # Compute from scratch
end
```

See also: [`save_hbm_regression_cache`](@ref)
"""
function load_hbm_regression_cache(filepath::String)::Union{HBMRegressionCache, Nothing}
    !isfile(filepath) && return nothing
    try
        data = load(filepath)
        get(data, "cache_version", 0) != INTERMEDIATE_CACHE_VERSION && return nothing
        return HBMRegressionCache(
            data["df_hierarchical"],
            data["bf_enrichment"],
            data["bf_correlation"],
            data["protein_ids"],
            data["refID"],
            data["regression_likelihood"],
            data["student_t_nu"],
            data["data_hash"],
            data["timestamp"],
            data["package_version"]
        )
    catch e
        @warn "Failed to load HBM+Regression cache: $e"
        return nothing
    end
end

# ----------------------- Cache Validation ----------------------- #

"""
    check_betabernoulli_cache(filepath::String, data, n_controls::Int, n_samples::Int)

Validate cached Beta-Bernoulli results against current data and parameters.

# Arguments
- `filepath::String`: Path to cache file
- `data`: Current data (InteractionData or tuple for multiple imputation)
- `n_controls::Int`: Current n_controls parameter
- `n_samples::Int`: Current n_samples parameter

# Returns
- `Tuple{IntermediateCacheStatus, Union{BetaBernoulliCache, Nothing}}`: Status and cache (if valid)

# Examples
```julia
status, cached = check_betabernoulli_cache(cache_path, data, 3, 3)

if status == INTERMEDIATE_CACHE_HIT
    bf_detected = cached.bf_detected
    @info "Using cached Beta-Bernoulli results from \$(cached.timestamp)"
elseif status == INTERMEDIATE_CACHE_MISS_PARAMS
    @info "Parameters changed, recomputing Beta-Bernoulli..."
elseif status == INTERMEDIATE_CACHE_MISS_DATA
    @info "Data changed, recomputing Beta-Bernoulli..."
else
    @info "No cache found, computing Beta-Bernoulli..."
end
```

See also: [`IntermediateCacheStatus`](@ref), [`load_betabernoulli_cache`](@ref)
"""
function check_betabernoulli_cache(filepath::String, data, n_controls::Int, n_samples::Int)
    cached = load_betabernoulli_cache(filepath)
    isnothing(cached) && return (INTERMEDIATE_CACHE_MISS_NO_FILE, nothing)

    # Check parameters
    if cached.n_controls != n_controls || cached.n_samples != n_samples
        return (INTERMEDIATE_CACHE_MISS_PARAMS, nothing)
    end

    # Check data hash
    if compute_data_hash(data) != cached.data_hash
        return (INTERMEDIATE_CACHE_MISS_DATA, nothing)
    end

    # Validate protein IDs match
    current_ids = getIDs(data isa Tuple ? data[2] : data)
    if cached.protein_ids != current_ids
        @warn "Protein IDs mismatch in cache, treating as data change"
        return (INTERMEDIATE_CACHE_MISS_DATA, nothing)
    end

    @info "Beta-Bernoulli cache hit! Using results from $(cached.timestamp)"
    return (INTERMEDIATE_CACHE_HIT, cached)
end

"""
    check_hbm_regression_cache(filepath::String, data, refID::Int, regression_likelihood::Symbol, student_t_nu::Float64)

Validate cached HBM+Regression results against current data and parameters.

# Arguments
- `filepath::String`: Path to cache file
- `data`: Current data (InteractionData or tuple for multiple imputation)
- `refID::Int`: Current reference protein index
- `regression_likelihood::Symbol`: Current regression likelihood (`:normal` or `:robust_t`)
- `student_t_nu::Float64`: Current Student-t degrees of freedom

# Returns
- `Tuple{IntermediateCacheStatus, Union{HBMRegressionCache, Nothing}}`: Status and cache (if valid)

# Examples
```julia
status, cached = check_hbm_regression_cache(cache_path, data, 1, :robust_t, 5.0)

if status == INTERMEDIATE_CACHE_HIT
    df_hierarchical = cached.df_hierarchical
    bf_enrichment = cached.bf_enrichment
    bf_correlation = cached.bf_correlation
    @info "Using cached HBM+Regression results from \$(cached.timestamp)"
elseif status == INTERMEDIATE_CACHE_MISS_PARAMS
    @info "Parameters changed, recomputing HBM+Regression..."
elseif status == INTERMEDIATE_CACHE_MISS_DATA
    @info "Data changed, recomputing HBM+Regression..."
else
    @info "No cache found, computing HBM+Regression..."
end
```

See also: [`IntermediateCacheStatus`](@ref), [`load_hbm_regression_cache`](@ref)
"""
function check_hbm_regression_cache(filepath::String, data, refID::Int, regression_likelihood::Symbol, student_t_nu::Float64)
    cached = load_hbm_regression_cache(filepath)
    isnothing(cached) && return (INTERMEDIATE_CACHE_MISS_NO_FILE, nothing)

    # Check refID, regression_likelihood, and student_t_nu
    if cached.refID != refID || cached.regression_likelihood != regression_likelihood || cached.student_t_nu != student_t_nu
        return (INTERMEDIATE_CACHE_MISS_PARAMS, nothing)
    end

    # Check data hash
    if compute_data_hash(data) != cached.data_hash
        return (INTERMEDIATE_CACHE_MISS_DATA, nothing)
    end

    # Validate protein IDs match
    current_ids = getIDs(data isa Tuple ? data[2] : data)
    if cached.protein_ids != current_ids
        @warn "Protein IDs mismatch in cache, treating as data change"
        return (INTERMEDIATE_CACHE_MISS_DATA, nothing)
    end

    @info "HBM+Regression cache hit! Using results from $(cached.timestamp)"
    return (INTERMEDIATE_CACHE_HIT, cached)
end

# ----------------------- Cache File Paths ----------------------- #

"""
    get_betabernoulli_cache_filepath(config::CONFIG)::String

Get cache file path for Beta-Bernoulli results based on config.

Creates `.bayesinteractomics_cache/` directory next to the first input data file
and returns path with hash-based naming.

# Arguments
- `config::CONFIG`: Configuration struct

# Returns
- `String`: Full path to Beta-Bernoulli cache file

# Examples
```julia
cache_path = get_betabernoulli_cache_filepath(config)
# Returns: "/path/to/data/.bayesinteractomics_cache/betabernoulli_<hash>.jld2"
```

See also: [`get_hbm_regression_cache_filepath`](@ref)
"""
function get_betabernoulli_cache_filepath(config::CONFIG)::String
    cache_dir = joinpath(dirname(config.datafile[1]), ".bayesinteractomics_cache")
    mkpath(cache_dir)
    datafile_hash = hash(config.datafile)
    return joinpath(cache_dir, "betabernoulli_$(string(datafile_hash, base=16)).jld2")
end

"""
    get_hbm_regression_cache_filepath(config::CONFIG)::String

Get cache file path for HBM+Regression results based on config.

Creates `.bayesinteractomics_cache/` directory next to the first input data file
and returns path with hash-based naming including refID and regression model type.

# Arguments
- `config::CONFIG`: Configuration struct

# Returns
- `String`: Full path to HBM+Regression cache file

# Examples
```julia
cache_path = get_hbm_regression_cache_filepath(config)
# Returns: "/path/to/data/.bayesinteractomics_cache/hbm_regression_<hash>_ref1_robust_t_nu5.0.jld2"
```

See also: [`get_betabernoulli_cache_filepath`](@ref)
"""
function get_hbm_regression_cache_filepath(config::CONFIG)::String
    cache_dir = joinpath(dirname(config.datafile[1]), ".bayesinteractomics_cache")
    mkpath(cache_dir)
    datafile_hash = hash(config.datafile)
    likelihood_str = string(config.regression_likelihood)
    nu_str = config.regression_likelihood == :robust_t ? "_nu$(config.student_t_nu)" : ""
    return joinpath(cache_dir, "hbm_regression_$(string(datafile_hash, base=16))_ref$(config.refID)_$(likelihood_str)$(nu_str).jld2")
end
