# Intermediate Results Caching System
# Provides separate caching for Beta-Bernoulli and HBM+Regression results
# to enable partial cache hits when only some parameters change

using JLD2
using Dates
import DataFrames: DataFrame

# Per-cache version constants. Split out of INTERMEDIATE_CACHE_VERSION
# in so that the v2b regression-cache invalidation
# in does NOT churn the BetaBernoulli, H0, or Platt-calibration
# caches. Each save_*_cache / load_*_cache function references EXACTLY
# its own constant. To invalidate a specific cache type after a model
# change, bump only the corresponding constant.
#
# HBM_REGRESSION_CACHE_VERSION bumped from 16 to 17
# because the regression observation factor changed to the variance-additive
# form (v2b). Stale cache files are explicitly rejected with a "regression
# model changed; recompute" warning. Other caches are unaffected.
#
# bumped 17 → 18 — the single-protocol mask-aware regression
# switched to the structured-VMP joint-MvNormal posterior (fixes the mean-field
# slope-variance collapse / bf_correlation saturation). Stale v2b-Path-B caches
# are rejected and recomputed. BB / H0 / calibration caches are unaffected.
#
# the normalisation flip (median_of_ratios / per-protein
# row-centering / :auto -> :both for scale-disparate multi-protocol loads) changes
# the numeric intensity scale fed to the regression, H0, and calibration stages.
# Caches computed on the old (un-normalised) scale MUST be rejected, not silently
# reused — otherwise the BREAKING normalisation change is masked by cache hits.
# Bumped: HBM_REGRESSION 18 -> 19, H0 16 -> 17, CALIBRATION 16 -> 17. BB is NOT
# bumped — Beta-Bernoulli detection is presence/absence (count_detections counts
# non-missing cells); row-centering subtracts a constant and column-scaling
# rescales, neither flips a cell between missing/observed, so detection counts are
# invariant. Over-bumping BB would force an unnecessary expensive recompute,
# contradicting the per-cache split rationale.
const BB_CACHE_VERSION             = 16  # H1 enrichment hard clamps, gate removal (unaffected by normalisation — detection is presence/absence)
const HBM_REGRESSION_CACHE_VERSION = 19  # normalisation flip changes the intensity scale fed to the regression; stale old-scale caches rejected
const H0_CACHE_VERSION             = 17  # H0 null Bayes factors depend on the normalised data scale; stale old-scale caches rejected
const CALIBRATION_CACHE_VERSION    = 17  # simulation/Platt calibration is fit from the rescaled data; stale old-scale caches rejected

# Deprecated alias — retained so external callers that read
# `INTERMEDIATE_CACHE_VERSION` (e.g. tests, scripts) keep compiling.
# New code should reference the per-cache constant explicitly.
const INTERMEDIATE_CACHE_VERSION   = BB_CACHE_VERSION

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
- `imputation_method::Symbol`: Imputation method tag (`:mnar`, `:mar`, or `:none`) — cache discriminator

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
    imputation_method::Symbol
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
- `imputation_method::Symbol`: Imputation method tag (`:mnar`, `:mar`, or `:none`) — cache discriminator

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
    regression_bf_threshold::Float64
    data_hash::UInt64
    timestamp::DateTime
    package_version::String
    imputation_method::Symbol
end

"""
    H0Cache

Cache container for null hypothesis (H0) data on log-BF scale.

Stores log-BF vectors for H0 proteins, fitted marginal parameters,
KS diagnostics, and metadata. Used by `combined_BF` to avoid redundant
H0 computation.

# Fields
- `log_bf_enrichment::Vector{Float64}`: Log-BF enrichment values for H0 proteins
- `log_bf_correlation::Vector{Float64}`: Log-BF correlation values for H0 proteins
- `log_bf_detection::Vector{Float64}`: Log-BF detection values for H0 proteins
- Marginal parameters (mu, sigma, nu per dimension; nu=0.0 means Normal, >0 means TDist)
- KS statistics per marginal
- `n_h0_proteins::Int`: Number of H0 proteins
- `data_hash::UInt64`: Hash of input data for validation
- `timestamp::DateTime`: When cache was created
- `package_version::String`: Package version for compatibility checking
- `imputation_method::Symbol`: Imputation method tag (`:mnar`, `:mar`, or `:none`) — cache discriminator

See also: [`save_h0_cache`](@ref), [`load_h0_cache`](@ref)
"""
struct H0Cache
    # Log-BF vectors for H0 proteins
    log_bf_enrichment::Vector{Float64}
    log_bf_correlation::Vector{Float64}
    log_bf_detection::Vector{Float64}
    # Fitted marginal parameters (serializable)
    marginal_enrichment_mu::Float64
    marginal_enrichment_sigma::Float64
    marginal_enrichment_nu::Float64  # 0.0 = Normal, >0 = TDist nu
    marginal_correlation_mu::Float64
    marginal_correlation_sigma::Float64
    marginal_correlation_nu::Float64
    marginal_detection_mu::Float64
    marginal_detection_sigma::Float64
    marginal_detection_nu::Float64
    # KS results
    ks_enrichment::Float64
    ks_correlation::Float64
    ks_detection::Float64
    # Metadata
    n_h0_proteins::Int
    data_hash::UInt64
    timestamp::DateTime
    package_version::String
    imputation_method::Symbol
end

# ----------------------- Hash Functions -----------------------

"""
    compute_betabernoulli_hash(data, n_controls::Int, n_samples::Int, imputation_method::Symbol = :mnar)::UInt64

Compute hash for Beta-Bernoulli cache validation.

Combines data hash with n_controls, n_samples, and imputation_method parameters.

# Arguments
- `data`: InteractionData or (Vector{InteractionData}, InteractionData) for multiple imputation
- `n_controls::Int`: Number of control replicates
- `n_samples::Int`: Number of sample replicates
- `imputation_method::Symbol = :mnar`: cache discriminator (`:mnar`, `:mar`, or `:none`)

# Returns
- `UInt64`: Hash value for cache validation

# Examples
```julia
h = compute_betabernoulli_hash(data, 3, 3, :mnar)
```
"""
function compute_betabernoulli_hash(data, n_controls::Int, n_samples::Int, imputation_method::Symbol = :mnar)::UInt64
    data_h = compute_data_hash(data)
    return hash((data_h, n_controls, n_samples, imputation_method))
end

"""
    compute_hbm_regression_hash(data, refID::Int, regression_likelihood::Symbol, student_t_nu::Float64, imputation_method::Symbol = :mnar)::UInt64

Compute hash for HBM+Regression cache validation.

Combines data hash with refID, regression_likelihood, student_t_nu, and imputation_method parameters.

# Arguments
- `data`: InteractionData or (Vector{InteractionData}, InteractionData) for multiple imputation
- `refID::Int`: Reference protein index
- `regression_likelihood::Symbol`: Likelihood type (`:normal` or `:robust_t`)
- `student_t_nu::Float64`: Degrees of freedom for Student-t
- `imputation_method::Symbol = :mnar`: cache discriminator (`:mnar`, `:mar`, or `:none`)

# Returns
- `UInt64`: Hash value for cache validation

# Examples
```julia
h = compute_hbm_regression_hash(data, 1, :robust_t, 5.0, :mnar)
```
"""
function compute_hbm_regression_hash(data, refID::Int, regression_likelihood::Symbol, student_t_nu::Float64, imputation_method::Symbol = :mnar)::UInt64
    data_h = compute_data_hash(data)
    return hash((data_h, refID, regression_likelihood, student_t_nu, imputation_method))
end

# ----------------------- Serialization -----------------------

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
        cache_version = BB_CACHE_VERSION,
        bf_detected = cache.bf_detected,
        protein_ids = cache.protein_ids,
        n_controls = cache.n_controls,
        n_samples = cache.n_samples,
        data_hash = cache.data_hash,
        timestamp = cache.timestamp,
        package_version = cache.package_version,
        imputation_method = cache.imputation_method
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
        get(data, "cache_version", 0) != BB_CACHE_VERSION && return nothing
        return BetaBernoulliCache(
            data["bf_detected"],
            data["protein_ids"],
            data["n_controls"],
            data["n_samples"],
            data["data_hash"],
            data["timestamp"],
            data["package_version"],
            get(data, "imputation_method", :mar)
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
cache = HBMRegressionCache(df, bf_enrich, bf_corr, protein_ids, 1, :robust_t, 5.0, 0.0, hash_val, now(), version)
save_hbm_regression_cache(cache, ".bayesinteractomics_cache/hbm_regression_abc123_ref1_robust_t_nu5.0.jld2")
```

See also: [`load_hbm_regression_cache`](@ref)
"""
function save_hbm_regression_cache(cache::HBMRegressionCache, filepath::String)
    jldsave(filepath; compress=true,
        cache_version = HBM_REGRESSION_CACHE_VERSION,
        df_hierarchical = cache.df_hierarchical,
        bf_enrichment = cache.bf_enrichment,
        bf_correlation = cache.bf_correlation,
        protein_ids = cache.protein_ids,
        refID = cache.refID,
        regression_likelihood = cache.regression_likelihood,
        student_t_nu = cache.student_t_nu,
        regression_bf_threshold = cache.regression_bf_threshold,
        data_hash = cache.data_hash,
        timestamp = cache.timestamp,
        package_version = cache.package_version,
        imputation_method = cache.imputation_method
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
        cached_ver = get(data, "cache_version", 0)
        if cached_ver != HBM_REGRESSION_CACHE_VERSION
            @warn "regression model changed; recompute" cached_version=cached_ver expected=HBM_REGRESSION_CACHE_VERSION maxlog=3
            return nothing
        end
        return HBMRegressionCache(
            data["df_hierarchical"],
            data["bf_enrichment"],
            data["bf_correlation"],
            data["protein_ids"],
            data["refID"],
            data["regression_likelihood"],
            data["student_t_nu"],
            get(data, "regression_bf_threshold", 0.0),
            data["data_hash"],
            data["timestamp"],
            data["package_version"],
            get(data, "imputation_method", :mar)
        )
    catch e
        @warn "Failed to load HBM+Regression cache: $e"
        return nothing
    end
end

# ----------------------- Cache Validation -----------------------

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
function check_betabernoulli_cache(filepath::String, data, n_controls::Int, n_samples::Int, imputation_method::Symbol = :mnar)
    cached = load_betabernoulli_cache(filepath)
    isnothing(cached) && return (INTERMEDIATE_CACHE_MISS_NO_FILE, nothing)

    # Check parameters (imputation_method discriminates MICE/MNAR caches)
    if cached.n_controls != n_controls || cached.n_samples != n_samples || cached.imputation_method != imputation_method
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
    check_hbm_regression_cache(filepath::String, data, refID::Int, regression_likelihood::Symbol, student_t_nu::Float64, regression_bf_threshold::Float64=0.1)

Validate cached HBM+Regression results against current data and parameters.

# Arguments
- `filepath::String`: Path to cache file
- `data`: Current data (InteractionData or tuple for multiple imputation)
- `refID::Int`: Current reference protein index
- `regression_likelihood::Symbol`: Current regression likelihood (`:normal` or `:robust_t`)
- `student_t_nu::Float64`: Current Student-t degrees of freedom
- `regression_bf_threshold::Float64`: Regression BF threshold (default = 0.1)

# Returns
- `Tuple{IntermediateCacheStatus, Union{HBMRegressionCache, Nothing}}`: Status and cache (if valid)

# Examples
```julia
status, cached = check_hbm_regression_cache(cache_path, data, 1, :robust_t, 5.0, 0.0)

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
function check_hbm_regression_cache(filepath::String, data, refID::Int, regression_likelihood::Symbol, student_t_nu::Float64, regression_bf_threshold::Float64=0.1, imputation_method::Symbol = :mnar)
    cached = load_hbm_regression_cache(filepath)
    isnothing(cached) && return (INTERMEDIATE_CACHE_MISS_NO_FILE, nothing)

    # Check refID, regression_likelihood, student_t_nu, regression_bf_threshold, and imputation_method
    if cached.refID != refID || cached.regression_likelihood != regression_likelihood || cached.student_t_nu != student_t_nu || cached.regression_bf_threshold != regression_bf_threshold || cached.imputation_method != imputation_method
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

# ----------------------- Cache File Paths -----------------------

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
    return joinpath(cache_dir, "betabernoulli_$(string(datafile_hash, base=16))_$(string(config.imputation_method)).jld2")
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
    return joinpath(cache_dir, "hbm_regression_$(string(datafile_hash, base=16))_ref$(config.refID)_$(likelihood_str)$(nu_str)_$(string(config.imputation_method)).jld2")
end

# ----------------------- H0 Cache -----------------------

"""
    save_h0_cache(cache::H0Cache, filepath::String)

Save H0 cache to disk using JLD2 format with compression.

# Arguments
- `cache::H0Cache`: Cache structure to save
- `filepath::String`: Output file path (typically .jld2 extension)

See also: [`load_h0_cache`](@ref)
"""
function save_h0_cache(cache::H0Cache, filepath::String)
    jldsave(filepath; compress=true,
        cache_version = H0_CACHE_VERSION,
        cache_type = "h0_logbf",
        log_bf_enrichment = cache.log_bf_enrichment,
        log_bf_correlation = cache.log_bf_correlation,
        log_bf_detection = cache.log_bf_detection,
        marginal_enrichment_mu = cache.marginal_enrichment_mu,
        marginal_enrichment_sigma = cache.marginal_enrichment_sigma,
        marginal_enrichment_nu = cache.marginal_enrichment_nu,
        marginal_correlation_mu = cache.marginal_correlation_mu,
        marginal_correlation_sigma = cache.marginal_correlation_sigma,
        marginal_correlation_nu = cache.marginal_correlation_nu,
        marginal_detection_mu = cache.marginal_detection_mu,
        marginal_detection_sigma = cache.marginal_detection_sigma,
        marginal_detection_nu = cache.marginal_detection_nu,
        ks_enrichment = cache.ks_enrichment,
        ks_correlation = cache.ks_correlation,
        ks_detection = cache.ks_detection,
        n_h0_proteins = cache.n_h0_proteins,
        data_hash = cache.data_hash,
        timestamp = string(cache.timestamp),
        package_version = cache.package_version,
        imputation_method = cache.imputation_method
    )
end

"""
    load_h0_cache(filepath::String)::Union{H0Cache, Nothing}

Load H0 cache from disk, or return nothing if invalid/missing.

Returns `nothing` if:
- File doesn't exist
- Cache version mismatch
- Cache type mismatch
- Loading fails due to corruption

See also: [`save_h0_cache`](@ref)
"""
function load_h0_cache(filepath::String)::Union{H0Cache, Nothing}
    !isfile(filepath) && return nothing
    try
        data = load(filepath)
        get(data, "cache_version", 0) != H0_CACHE_VERSION && return nothing
        get(data, "cache_type", "") != "h0_logbf" && return nothing
        return H0Cache(
            data["log_bf_enrichment"],
            data["log_bf_correlation"],
            data["log_bf_detection"],
            data["marginal_enrichment_mu"],
            data["marginal_enrichment_sigma"],
            data["marginal_enrichment_nu"],
            data["marginal_correlation_mu"],
            data["marginal_correlation_sigma"],
            data["marginal_correlation_nu"],
            data["marginal_detection_mu"],
            data["marginal_detection_sigma"],
            data["marginal_detection_nu"],
            data["ks_enrichment"],
            data["ks_correlation"],
            data["ks_detection"],
            data["n_h0_proteins"],
            data["data_hash"],
            DateTime(data["timestamp"]),
            data["package_version"],
            get(data, "imputation_method", :mar)
        )
    catch e
        @warn "Failed to load H0 cache: $e"
        return nothing
    end
end

"""
    check_h0_cache(filepath::String, data_hash::UInt64)

Validate cached H0 results against current data hash.

# Returns
- `Tuple{IntermediateCacheStatus, Union{H0Cache, Nothing}}`: Status and cache (if valid)

See also: [`IntermediateCacheStatus`](@ref), [`load_h0_cache`](@ref)
"""
function check_h0_cache(filepath::String, data_hash::UInt64, imputation_method::Symbol = :mnar)
    cached = load_h0_cache(filepath)
    isnothing(cached) && return (INTERMEDIATE_CACHE_MISS_NO_FILE, nothing)

    # Check imputation_method — discriminates MICE/MNAR caches
    if cached.imputation_method != imputation_method
        return (INTERMEDIATE_CACHE_MISS_PARAMS, nothing)
    end

    # Check data hash
    if cached.data_hash != data_hash
        return (INTERMEDIATE_CACHE_MISS_DATA, nothing)
    end

    @info "H0 cache hit! Using null distribution from $(cached.timestamp)"
    return (INTERMEDIATE_CACHE_HIT, cached)
end

"""
    get_h0_cache_filepath(config::CONFIG)::String

Get cache file path for H0 results based on config.

Creates `.bayesinteractomics_cache/` directory next to the first input data file
and returns path with hash-based naming.

See also: [`get_betabernoulli_cache_filepath`](@ref), [`get_hbm_regression_cache_filepath`](@ref)
"""
function get_h0_cache_filepath(config::CONFIG)::String
    cache_dir = joinpath(dirname(config.datafile[1]), ".bayesinteractomics_cache")
    mkpath(cache_dir)
    datafile_hash = hash(config.datafile)
    likelihood_str = string(config.regression_likelihood)
    nu_str = config.regression_likelihood == :robust_t ? "_nu$(config.student_t_nu)" : ""
    return joinpath(cache_dir, "h0_$(string(datafile_hash, base=16))_ref$(config.refID)_$(likelihood_str)$(nu_str)_$(string(config.imputation_method)).jld2")
end

"""
    get_nu_cache_filepath(config::CONFIG) -> String

Path for the per-condition Student-t ν optimization cache. Lives alongside the
ν optimization plot (`config.output.nu_optimization_file`) so it is per-output-dir
(per-condition) rather than per-data-source.

The cache is keyed by `(data_hash, regression_likelihood, imputation_method)`
and stores the optimal ν found by Brent's method. Re-runs with matching keys
skip the multi-evaluation Brent search (~30-60 min on HD data) and reuse the
cached value.
"""
function get_nu_cache_filepath(config::CONFIG)::String
    nu_plot = config.output.nu_optimization_file
    return joinpath(dirname(nu_plot), "nu_cache.jld2")
end

"""
    save_nu_cache(path, data_hash, optimal_nu, optimal_waic, normal_waic,
                  n_evaluations, likelihood, imputation_method)

Persist the result of one Student-t ν optimization run. Schema is intentionally
flat (not a struct) to keep the cache append-friendly across versions.
"""
function save_nu_cache(path::String;
                       data_hash::UInt64,
                       optimal_nu::Float64,
                       optimal_waic::Float64,
                       normal_waic::Float64,
                       n_evaluations::Int,
                       likelihood::Symbol,
                       imputation_method::Symbol)
    mkpath(dirname(path))
    jldsave(path;
        cache_version       = HBM_REGRESSION_CACHE_VERSION,
        data_hash           = data_hash,
        optimal_nu          = optimal_nu,
        optimal_waic        = optimal_waic,
        normal_waic         = normal_waic,
        n_evaluations       = n_evaluations,
        likelihood          = string(likelihood),
        imputation_method   = string(imputation_method),
        timestamp           = string(now()),
        package_version     = string(pkgversion(@__MODULE__)),
    )
    return path
end

"""
    load_nu_cache(path; data_hash, likelihood, imputation_method)
        -> Union{NamedTuple, Nothing}

Load and validate a ν cache. Returns `nothing` (cache miss) on any of:
- file does not exist
- cache_version mismatch
- data_hash mismatch
- likelihood / imputation_method mismatch
- I/O or schema error

On success returns `(; optimal_nu, optimal_waic, normal_waic, n_evaluations)`.
"""
function load_nu_cache(path::String;
                       data_hash::UInt64,
                       likelihood::Symbol,
                       imputation_method::Symbol)
    isfile(path) || return nothing
    try
        d = load(path)
        get(d, "cache_version", 0) == HBM_REGRESSION_CACHE_VERSION || return nothing
        get(d, "data_hash", UInt64(0)) == data_hash               || return nothing
        get(d, "likelihood", "")        == string(likelihood)        || return nothing
        get(d, "imputation_method", "") == string(imputation_method) || return nothing
        return (
            optimal_nu     = Float64(d["optimal_nu"]),
            optimal_waic   = Float64(d["optimal_waic"]),
            normal_waic    = Float64(get(d, "normal_waic", NaN)),
            n_evaluations  = Int(get(d, "n_evaluations", 0)),
        )
    catch e
        @warn "Failed to read ν cache at $path: $e"
        return nothing
    end
end
