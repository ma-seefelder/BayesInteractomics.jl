# Analysis Results Caching System
# Provides AnalysisResult type with hash-based caching to avoid recomputation

using JLD2
using Dates
import DataFrames: nrow

const CACHE_VERSION = 26  # TR+DDI metalearner schema (14 features default; +mc_std opt-in). Legacy AnalysisResult caches with the old 8-feature metalearner posterior are silently discarded by the existing per-cache version-discard idiom (per-cache constants in src/core/intermediate_cache.jl are unchanged). Non-MC :tr_ddi is the default (metalearner_use_mc_dropout=false); stale v25 caches discarded.

"""
    AbstractAnalysisResult

Abstract type for analysis results. Subtypes must implement:
- `copula_results` field or `results` field for the results DataFrame
- `bait_protein` field returning Union{String, Nothing}
- `bait_index` field returning Union{Int, Nothing}
"""
abstract type AbstractAnalysisResult end

"""
    AnalysisResult <: AbstractAnalysisResult

Complete results container from a BayesInteractomics analysis run with caching support.

This structure stores all outputs from `run_analysis()` along with hash-based validation
to enable intelligent caching and avoid redundant computation.

# Fields
- `copula_results::DataFrame`: Final results with combined Bayes factors, posterior probabilities, PEP, and BFDR values
- `df_hierarchical::DataFrame`: Detailed hierarchical model results including individual protein statistics
- `em::Union{EMResult, Nothing}`: EM algorithm results from copula mixture fitting (copula mode only)
- `joint_H0::Union{SklarDist, Nothing}`: Fitted copula distribution under null hypothesis (copula mode only)
- `joint_H1::Union{SklarDist, Nothing}`: Fitted copula distribution under alternative hypothesis (copula mode only)
- `latent_class_result::Union{LatentClassResult, Nothing}`: VMP latent class results (latent class mode only)
- `bma_result::Union{BMAResult, Nothing}`: BMA results (BMA mode only)
- `combination_method::Symbol`: Evidence combination method used (`:copula`, `:latent_class`, or `:bma`)
- `em_diagnostics::Union{DataFrame, Nothing}`: EM algorithm restart diagnostics
- `em_diagnostics_summary::Union{NamedTuple, Nothing}`: Summary of EM restart diagnostics
- `config_hash::UInt64`: Hash of configuration parameters affecting computation (for cache validation)
- `data_hash::UInt64`: Hash of input data (for cache validation)
- `timestamp::DateTime`: Analysis completion time
- `package_version::String`: Package version used for analysis
- `bait_protein::Union{String, Nothing}`: Name or ID of the bait protein
- `bait_index::Union{Int, Nothing}`: Index of bait protein in the protein list
- `sensitivity::Union{SensitivityResult, Nothing}`: Prior sensitivity analysis results (if `run_sensitivity=true`)
- `diagnostics::Union{DiagnosticsResult, Nothing}`: Posterior predictive check and model diagnostics results (if `run_diagnostics=true`)
- `metalearner_status::Symbol`: Indicates how `posterior_prob` was computed. `:loaded` = metalearner-adjusted, `:extension_not_loaded` = BF-derived (Variante B fallback), `:prediction_failed` = metalearner extension loaded but call failed.

# Iterator Interface
Iterates over `(protein_name, row_data)` tuples from copula_results.

```julia
for (protein, row) in result
    println("\$protein: BF = \$(row.BF), BFDR = \$(row.BFDR)")
end
```

# Indexing
- `result[i]`: Get row i from copula_results (Integer indexing)
- `result[protein]`: Get row for specific protein (String indexing)

# Examples
```julia
# After analysis
final_results, analysis_result = run_analysis(config)

# Access results
println("Analysis completed at: ", analysis_result.timestamp)
println("Number of proteins: ", length(analysis_result))

# Get specific protein
row = analysis_result["ProteinA"]
println("BF: ", row.BF, ", BFDR: ", row.BFDR)

# Iterate over all proteins
for (protein, data) in analysis_result
    if data.posterior_prob > 0.95
        println("\$protein is likely an interactor")
    end
end

# Access sensitivity analysis (if available)
if !isnothing(analysis_result.sensitivity)
    sr = analysis_result.sensitivity
    println("Tested ", length(sr.prior_settings), " prior settings")
    println("Max posterior range: ", maximum(sr.summary.range))
end

# Save for later
save_result(analysis_result, "analysis_cache.jld2")
```

See also: [`save_result`](@ref), [`load_result`](@ref), [`check_cache`](@ref), [`sensitivity_analysis`](@ref)
"""
mutable struct AnalysisResult <: AbstractAnalysisResult
    copula_results::DataFrame
    df_hierarchical::DataFrame
    em::Union{EMResult, Nothing}
    joint_H0::Union{SklarDist, Nothing}
    joint_H1::Union{SklarDist, Nothing}
    latent_class_result::Union{LatentClassResult, Nothing}
    bma_result::Union{BMAResult, Nothing}
    combination_method::Symbol
    em_diagnostics::Union{DataFrame, Nothing}
    em_diagnostics_summary::Union{NamedTuple, Nothing}
    config_hash::UInt64
    data_hash::UInt64
    timestamp::DateTime
    package_version::String
    bait_protein::Union{String, Nothing}
    bait_index::Union{Int, Nothing}
    sensitivity::Union{SensitivityResult, Nothing}
    diagnostics::Union{DiagnosticsResult, Nothing}
    input_qc::Union{Nothing, InputQCResult}
    metalearner_status::Symbol  # :extension_not_loaded | :loaded | :prediction_failed
    # (Pitfall 2): per-bait sim result, populated in run_analysis(::CONFIG);
    # nothing for path-based / cached load. Required by the differential report's
    # per-condition Calibration tab which iterates diff.analyses[i].simulation_result.
    # NOT JLD2-cached in BetaBernoulliCache or HBMRegressionCache (verified by grep
    # over src/core/intermediate_cache.jl); the only JLD2 path is save_result/load_result
    # in this file, both updated to round-trip the new fields. CACHE_VERSION bumped 21 → 22.
    simulation_result::Union{Nothing, SimulationResult}
    # (Pitfall 4): per-bait CONFIG, populated in run_analysis(::CONFIG);
    # nothing for path-based / cached load. Required by the differential report's
    # per-condition Methods tab which calls _build_methods_json(ar.config, ...).
    config::Union{Nothing, CONFIG}
    # dataset-level ECE-gate provenance flag. true when Platt
    # calibration was applied and passed the ECE safety check; false when calibration
    # was skipped, failed, or run on a legacy/cached path that pre-dates this flag.
    # Settable post-construction (mutable struct) so pipeline.jl can assign after
    # the calibration decision is made. Round-tripped through save_result/load_result
    # under CACHE_VERSION = 23; legacy caches (≤ 22) load as nothing via the existing
    # cache_version mismatch path and never reach the is_calibrated field.
    is_calibrated::Bool
    # Embeddings result for sample/protein UMAP/t-SNE and condition
    # similarity (single-bait case). `nothing` for legacy paths, path-based loads,
    # or when CONFIG.embeddings_config.run_embeddings == false. Round-tripped via
    # save_result/load_result under CACHE_VERSION = 24. Partial-invalidation gate
    # `_should_recompute_embeddings(ar, cfg)` compares the embedded
    # `config_snapshot` against the current EmbeddingsConfig snapshot.
    embeddings::Union{Nothing, EmbeddingsResult}
end

# backward-compat constructor: pre-Phase-69 callers (20 positional args)
# keep working with simulation_result=nothing and config=nothing defaults.
# extended to also default is_calibrated=false.
# extended to also default embeddings=nothing.
AnalysisResult(copula_results, df_hierarchical, em, joint_H0, joint_H1,
               latent_class_result, bma_result, combination_method,
               em_diagnostics, em_diagnostics_summary, config_hash, data_hash,
               timestamp, package_version, bait_protein, bait_index,
               sensitivity, diagnostics, input_qc, metalearner_status) =
    AnalysisResult(copula_results, df_hierarchical, em, joint_H0, joint_H1,
                   latent_class_result, bma_result, combination_method,
                   em_diagnostics, em_diagnostics_summary, config_hash, data_hash,
                   timestamp, package_version, bait_protein, bait_index,
                   sensitivity, diagnostics, input_qc, metalearner_status,
                   nothing, nothing, false, nothing)

# backward-compat constructor: pre-Phase-70 callers (22 positional args,
# i.e. the canonical signature) keep working with is_calibrated=false default.
# extended to also default embeddings=nothing.
AnalysisResult(copula_results, df_hierarchical, em, joint_H0, joint_H1,
               latent_class_result, bma_result, combination_method,
               em_diagnostics, em_diagnostics_summary, config_hash, data_hash,
               timestamp, package_version, bait_protein, bait_index,
               sensitivity, diagnostics, input_qc, metalearner_status,
               simulation_result, config) =
    AnalysisResult(copula_results, df_hierarchical, em, joint_H0, joint_H1,
                   latent_class_result, bma_result, combination_method,
                   em_diagnostics, em_diagnostics_summary, config_hash, data_hash,
                   timestamp, package_version, bait_protein, bait_index,
                   sensitivity, diagnostics, input_qc, metalearner_status,
                   simulation_result, config, false, nothing)

# backward-compat constructor: pre-Phase-71 callers (23 positional args,
# i.e. the canonical signature including is_calibrated) keep working with
# embeddings=nothing default.
AnalysisResult(copula_results, df_hierarchical, em, joint_H0, joint_H1,
               latent_class_result, bma_result, combination_method,
               em_diagnostics, em_diagnostics_summary, config_hash, data_hash,
               timestamp, package_version, bait_protein, bait_index,
               sensitivity, diagnostics, input_qc, metalearner_status,
               simulation_result, config, is_calibrated) =
    AnalysisResult(copula_results, df_hierarchical, em, joint_H0, joint_H1,
                   latent_class_result, bma_result, combination_method,
                   em_diagnostics, em_diagnostics_summary, config_hash, data_hash,
                   timestamp, package_version, bait_protein, bait_index,
                   sensitivity, diagnostics, input_qc, metalearner_status,
                   simulation_result, config, is_calibrated, nothing)

# Property accessor for backward compatibility with network extension
"""
    Base.getproperty(r::AnalysisResult, s::Symbol)

Property accessor that provides backward compatibility.
Accessing `ar.results` returns `ar.copula_results`.
"""
function Base.getproperty(r::AnalysisResult, s::Symbol)
    if s === :results
        return getfield(r, :copula_results)
    else
        return getfield(r, s)
    end
end

"""
    Base.propertynames(r::AnalysisResult, private::Bool=false)

Return property names including the `results` alias.
"""
function Base.propertynames(r::AnalysisResult, private::Bool=false)
    return (:results, fieldnames(AnalysisResult)...)
end

# ----------------------- Hash Functions -----------------------

"""
    compute_config_hash(config::CONFIG)::UInt64

Compute hash of configuration parameters that affect computational results.

Only includes fields that influence analysis outcomes. Excludes plotting flags,
output file paths, and other parameters that don't affect numerical results.
The `poi` field is also excluded since the meta-learner always re-runs.

# Arguments
- `config::CONFIG`: Configuration struct

# Returns
- `UInt64`: Hash value for cache validation

# Notes
- Two configs with different output paths but same analysis parameters will have the same hash
- Plotting flags (plotHBMdists, plotlog2fc, etc.) are excluded
- Meta-learner protein of interest (poi) is excluded

# Examples
```julia
config1 = CONFIG(datafile=["data.xlsx"], control_cols=[...], ...)
config2 = deepcopy(config1)
config2.results_file = "different_output.xlsx"  # Won't affect hash

compute_config_hash(config1) == compute_config_hash(config2)  # true
```

See also: [`compute_data_hash`](@ref), [`check_cache`](@ref)
"""
function compute_config_hash(config::CONFIG)::UInt64
    # Only fields that affect computation (exclude plotting flags, output paths, poi)
    hashable = (
        config.datafile,
        config.control_cols,
        config.sample_cols,
        config.normalise_protocols,
        config.output.H0_file,
        config.n_controls,
        config.n_samples,
        config.refID,
        config.combination_method,
        config.regression_likelihood,
        config.student_t_nu,
        config.regression_bf_threshold,
        string(config.lc_alpha_prior)
    )
    return hash(hashable)
end

"""
    compute_data_hash(data::InteractionData)::UInt64

Compute hash of InteractionData for cache validation.

Hashes protein IDs and all sample/control matrices across protocols and experiments.
Missing values are coalesced to NaN for consistent hashing.

# Arguments
- `data::InteractionData`: Input data structure

# Returns
- `UInt64`: Hash value for cache validation

# Examples
```julia
data1 = load_data(["file.xlsx"], sample_cols, control_cols)
data2 = load_data(["file.xlsx"], sample_cols, control_cols)

compute_data_hash(data1) == compute_data_hash(data2)  # true
```

See also: [`compute_data_hash(::Vector{InteractionData}, ::InteractionData)`](@ref)
"""
function compute_data_hash(data::InteractionData)::UInt64
    h = hash(data.protein_IDs)
    for p in 1:data.no_protocols
        for e in 1:data.no_experiments[p]
            sample_mat = coalesce.(data.samples[p].data[e], NaN)
            control_mat = coalesce.(data.controls[p].data[e], NaN)
            h = hash(sample_mat, h)
            h = hash(control_mat, h)
        end
    end
    return h
end

"""
    compute_data_hash(data::Vector{InteractionData}, raw_data::InteractionData)::UInt64

Compute hash for multiple imputation datasets.

Combines hash of raw data (used for Beta-Bernoulli) with hashes of all imputed datasets.

# Arguments
- `data::Vector{InteractionData}`: Vector of imputed datasets
- `raw_data::InteractionData`: Original non-imputed data

# Returns
- `UInt64`: Combined hash value for cache validation

# Examples
```julia
imputed_data = [impute_data(raw_data, method=:ppca) for _ in 1:10]
h = compute_data_hash(imputed_data, raw_data)
```

See also: [`compute_data_hash(::InteractionData)`](@ref)
"""
function compute_data_hash(data::Vector{InteractionData}, raw_data::InteractionData)::UInt64
    h = compute_data_hash(raw_data)  # Raw data first
    for d in data
        h = hash(compute_data_hash(d), h)
    end
    return h
end

# Tuple form: pipeline.jl's check_cache passes (imputed_data, raw_data) as a
# single Tuple argument. Forward to the (Vector, InteractionData) method.
function compute_data_hash(pair::Tuple{Vector{InteractionData}, InteractionData})::UInt64
    return compute_data_hash(pair[1], pair[2])
end

# ----------------------- Iterator Interface -----------------------

"""
    Base.length(r::AnalysisResult)

Get number of proteins in the analysis results.

# Examples
```julia
println("Analyzed \$(length(result)) proteins")
```
"""
Base.length(r::AnalysisResult) = nrow(r.copula_results)

"""
    Base.iterate(r::AnalysisResult[, state])

Iterate over (protein_name, row_data) tuples from copula_results.

# Examples
```julia
for (protein, row) in result
    println("\$protein: BF=\$(row.BF)")
end
```
"""
Base.iterate(r::AnalysisResult, state=1) = state > length(r) ? nothing :
    ((r.copula_results.Protein[state], r.copula_results[state, :]), state + 1)

"""
    Base.getindex(r::AnalysisResult, protein::String)

Get results for a specific protein by name.

# Examples
```julia
protein_row = result["MyProtein"]
println("BF: ", protein_row.BF)
```
"""
Base.getindex(r::AnalysisResult, protein::String) =
    r.copula_results[findfirst(==(protein), r.copula_results.Protein), :]

"""
    Base.getindex(r::AnalysisResult, i::Integer)

Get results for protein at index i.

# Examples
```julia
first_protein = result[1]
```
"""
Base.getindex(r::AnalysisResult, i::Integer) = r.copula_results[i, :]

# ----------------------- Serialization -----------------------

"""
    save_result(result::AnalysisResult, filepath::String)

Save AnalysisResult to disk using JLD2 format with compression.

# Arguments
- `result::AnalysisResult`: Result structure to save
- `filepath::String`: Output file path (typically .jld2 extension)

# Examples
```julia
save_result(analysis_result, ".bayesinteractomics_cache/analysis_cache.jld2")
```

See also: [`load_result`](@ref), [`AnalysisResult`](@ref)
"""
function save_result(result::AnalysisResult, filepath::String)
    jldsave(filepath; compress=true,
        cache_version = CACHE_VERSION,
        copula_results = result.copula_results,
        df_hierarchical = result.df_hierarchical,
        em = result.em,
        joint_H0 = result.joint_H0,
        joint_H1 = result.joint_H1,
        latent_class_result = result.latent_class_result,
        bma_result = result.bma_result,
        combination_method = result.combination_method,
        em_diagnostics = result.em_diagnostics,
        em_diagnostics_summary = result.em_diagnostics_summary,
        config_hash = result.config_hash,
        data_hash = result.data_hash,
        timestamp = result.timestamp,
        package_version = result.package_version,
        bait_protein = result.bait_protein,
        bait_index = result.bait_index,
        sensitivity = result.sensitivity,
        diagnostics = result.diagnostics,
        input_qc = result.input_qc,
        metalearner_status = result.metalearner_status,
        # persist simulation_result + config so reloading from cache
        # preserves AR-overload differential_analysis ability to populate analyses.
        # Both default to nothing on legacy caches via get(...) fallbacks below.
        simulation_result = result.simulation_result,
        config = result.config,
        # persist dataset-level ECE-gate provenance flag.
        # Defaults to false on legacy caches via get(...) fallback in load_result.
        is_calibrated = result.is_calibrated,
        # persist EmbeddingsResult (or nothing) for partial cache
        # invalidation. Defaults to nothing on legacy caches via get(...) fallback.
        embeddings = result.embeddings
    )
end

"""
    load_result(filepath::String)::Union{AnalysisResult, Nothing}

Load AnalysisResult from disk, or return nothing if invalid/missing.

Returns `nothing` if:
- File doesn't exist
- Cache version mismatch (indicates incompatible format)
- Loading fails due to corruption or other errors

# Arguments
- `filepath::String`: Path to cached result file

# Returns
- `AnalysisResult`: Loaded result structure
- `nothing`: If file is missing, incompatible, or corrupted

# Examples
```julia
cached = load_result("analysis_cache.jld2")
if !isnothing(cached)
    println("Loaded cached results from ", cached.timestamp)
else
    println("No valid cache found, running analysis...")
end
```

See also: [`save_result`](@ref), [`check_cache`](@ref)
"""
function load_result(filepath::String)::Union{AnalysisResult, Nothing}
    !isfile(filepath) && return nothing
    try
        data = load(filepath)
        get(data, "cache_version", 0) != CACHE_VERSION && return nothing
        return AnalysisResult(
            data["copula_results"],
            data["df_hierarchical"],
            data["em"],
            data["joint_H0"],
            data["joint_H1"],
            get(data, "latent_class_result", nothing),
            get(data, "bma_result", nothing),
            get(data, "combination_method", :copula),
            get(data, "em_diagnostics", nothing),
            get(data, "em_diagnostics_summary", nothing),
            data["config_hash"],
            data["data_hash"],
            data["timestamp"],
            data["package_version"],
            get(data, "bait_protein", nothing),
            get(data, "bait_index", nothing),
            get(data, "sensitivity", nothing),
            get(data, "diagnostics", nothing),
            get(data, "input_qc", nothing),
            get(data, "metalearner_status", :extension_not_loaded),
            get(data, "simulation_result", nothing),
            get(data, "config", nothing),
            get(data, "is_calibrated", false),
            get(data, "embeddings", nothing)
        )
    catch e
        @warn "Failed to load cache: $e"
        return nothing
    end
end

# ----------------------- Cache Validation -----------------------

"""
    CacheStatus

Enum representing cache validation status.

# Values
- `CACHE_HIT`: Cache is valid and can be used
- `CACHE_MISS_NO_FILE`: Cache file doesn't exist
- `CACHE_MISS_CONFIG`: Config hash mismatch (analysis parameters changed)
- `CACHE_MISS_DATA`: Data hash mismatch (input data changed)

See also: [`check_cache`](@ref)
"""
@enum CacheStatus CACHE_HIT CACHE_MISS_NO_FILE CACHE_MISS_CONFIG CACHE_MISS_DATA

"""
    check_cache(cache_file::String, config::CONFIG, data)

Validate cached results against current configuration and data.

Checks if cached analysis results are still valid for the current config and data.
Returns the cache status and loaded result (if valid).

# Arguments
- `cache_file::String`: Path to cache file
- `config::CONFIG`: Current configuration
- `data`: Current data (InteractionData or Tuple{Vector{InteractionData}, InteractionData})

# Returns
- `Tuple{CacheStatus, Union{AnalysisResult, Nothing}}`: Status and cached result (if hit)

# Cache Hit
When status is `CACHE_HIT`, the second element contains the valid cached result.

# Cache Miss
When status is any MISS variant, the second element is `nothing`.

# Examples
```julia
status, cached = check_cache("cache.jld2", config, data)

if status == CACHE_HIT
    @info "Using cached results from \$(cached.timestamp)"
    # Use cached.copula_results, cached.joint_H0, etc.
elseif status == CACHE_MISS_CONFIG
    @info "Config changed, recomputing..."
elseif status == CACHE_MISS_DATA
    @info "Data changed, recomputing..."
else  # CACHE_MISS_NO_FILE
    @info "No cache found, running analysis..."
end
```

See also: [`CacheStatus`](@ref), [`compute_config_hash`](@ref), [`compute_data_hash`](@ref)
"""
function check_cache(cache_file::String, config::CONFIG, data)
    cached = load_result(cache_file)
    isnothing(cached) && return (CACHE_MISS_NO_FILE, nothing)

    compute_config_hash(config) != cached.config_hash && return (CACHE_MISS_CONFIG, nothing)
    compute_data_hash(data) != cached.data_hash && return (CACHE_MISS_DATA, nothing)

    @info "Cache hit! Using results from $(cached.timestamp)"
    return (CACHE_HIT, cached)
end

"""
    get_cache_filepath(config::CONFIG)::String

Get default cache file path based on config.

Creates `.bayesinteractomics_cache/` directory next to the first input data file
and returns path to cache file with hash-based naming.

# Arguments
- `config::CONFIG`: Configuration struct

# Returns
- `String`: Full path to cache file

# Notes
- Cache directory is created if it doesn't exist
- Cache filename includes hash of datafile list for uniqueness

# Examples
```julia
cache_path = get_cache_filepath(config)
# Returns: "/path/to/data/.bayesinteractomics_cache/analysis_cache_<hash>.jld2"
```

See also: [`check_cache`](@ref), [`save_result`](@ref)
"""
function get_cache_filepath(config::CONFIG)::String
    cache_dir = joinpath(dirname(config.datafile[1]), ".bayesinteractomics_cache")
    mkpath(cache_dir)
    return joinpath(cache_dir, "analysis_cache_$(hash(config.datafile)).jld2")
end

# ----------------------- Convenience Functions -----------------------

"""
    set_bait_info!(result::AnalysisResult; bait_protein=nothing, bait_index=nothing)

Set bait protein information on an existing AnalysisResult.

This is useful after `run_analysis()` which creates AnalysisResult with bait fields set to nothing.

# Arguments
- `result::AnalysisResult`: Result object to modify
- `bait_protein::Union{String, Nothing}`: Name or ID of bait protein
- `bait_index::Union{Int, Nothing}`: Index of bait protein in protein list

# Returns
- `AnalysisResult`: The modified result object (for method chaining)

# Examples
```julia
final_results, analysis_result = run_analysis(config)
set_bait_info!(analysis_result, bait_protein="MYC", bait_index=1)

# Or with method chaining
net = build_network(
    set_bait_info!(analysis_result, bait_protein="MYC", bait_index=1),
    posterior_threshold=0.8
)
```
"""
function set_bait_info!(result::AnalysisResult; bait_protein=nothing, bait_index=nothing)
    if !isnothing(bait_protein)
        result.bait_protein = bait_protein
    end
    if !isnothing(bait_index)
        result.bait_index = bait_index
    end
    return result
end

# ----------------------- Convenience Accessors -----------------------

"""
    getProteins(r::AnalysisResult)

Get vector of protein names from results.

# Examples
```julia
proteins = getProteins(result)
println("First 5 proteins: ", proteins[1:5])
```
"""
getProteins(r::AnalysisResult) = r.copula_results.Protein

"""
    getBayesFactors(r::AnalysisResult)

Get vector of combined Bayes factors from results.

# Examples
```julia
bfs = getBayesFactors(result)
high_evidence = bfs .> 10
```
"""
getBayesFactors(r::AnalysisResult) = r.copula_results.BF

"""
    getPosteriorProbabilities(r::AnalysisResult)

Get vector of posterior probabilities from results.

# Examples
```julia
probs = getPosteriorProbabilities(result)
likely_interactors = sum(probs .> 0.95)
```
"""
getPosteriorProbabilities(r::AnalysisResult) = r.copula_results.posterior_prob

"""
    getBFDR(r::AnalysisResult)

Get vector of BFDR (Bayesian FDR) values from results.

# Examples
```julia
bfdr_vals = getBFDR(result)
significant = sum(bfdr_vals .< 0.05)
```
"""
getBFDR(r::AnalysisResult) = r.copula_results.BFDR

"""Deprecated: use `getBFDR()` instead."""
function getQValues(r::AnalysisResult)
    @warn "`getQValues()` is deprecated, use `getBFDR()` instead" maxlog=1
    return getBFDR(r)
end

"""
    getPEP(r::AnalysisResult) -> Vector{Union{Missing, Float64}}

Return the canonical posterior error probability column from the bait result
DataFrame. The column reflects the dataset-level calibration gate:

- When `r.is_calibrated == true` (ECE gate `cal_ece < 0.10` passed), `pep_i = 1 − posterior_calibrated_i`.
- Otherwise, `pep_i = 1 − posterior_prob_i`.

Use `isCalibrated(r)` to query provenance.
"""
getPEP(r::AnalysisResult) = r.copula_results.pep

"""
    isCalibrated(r::AnalysisResult) -> Bool

Whether `r.copula_results.pep` reflects Platt-calibrated posteriors. Calibration
is a dataset-level decision gated by the ECE safety guard (`cal_ece < 0.10`).
"""
isCalibrated(r::AnalysisResult) = r.is_calibrated

"""
    getMeanLog2FC(r::AnalysisResult)

Get vector of mean log2 fold changes from results.

# Examples
```julia
log2fc = getMeanLog2FC(result)
upregulated = sum(log2fc .> 1.0)
```
"""
getMeanLog2FC(r::AnalysisResult) = r.copula_results.mean_log2FC

"""
    getBaitProtein(r::AnalysisResult)

Get bait protein name from results.

# Examples
```julia
bait = getBaitProtein(result)
println("Bait protein: ", bait)
```
"""
getBaitProtein(r::AnalysisResult) = r.bait_protein

"""
    getPosteriorProbs(r::AnalysisResult)

Alias for getPosteriorProbabilities. Get vector of posterior probabilities from results.

# Examples
```julia
probs = getPosteriorProbs(result)
likely_interactors = sum(probs .> 0.95)
```
"""
getPosteriorProbs(r::AnalysisResult) = r.copula_results.posterior_prob

# ----------------------- Display -----------------------

function Base.show(io::IO, r::AnalysisResult)
    println(io, "🧬 AnalysisResult")
    println(io, "─────────────────────────────")
    println(io, " • Proteins analyzed      : $(length(r))")
    println(io, " • Timestamp              : $(r.timestamp)")
    println(io, " • Package version        : $(r.package_version)")
    println(io, " • Combination method     : $(r.combination_method)")

    if r.combination_method == :copula && !isnothing(r.em)
        println(io, " • EM converged           : $(r.em.has_converged)")
    elseif r.combination_method == :latent_class && !isnothing(r.latent_class_result)
        println(io, " • VMP converged          : $(r.latent_class_result.converged)")
    elseif r.combination_method == :bma && !isnothing(r.bma_result)
        println(io, " • EM weight              : $(round(r.bma_result.em_weight, digits=4))")
        println(io, " • Copula weight          : $(round(r.bma_result.copula_weight, digits=4))")
        n_disagree = count(r.bma_result.model_disagreement)
        println(io, " • Model disagreement     : $(n_disagree)/$(length(r.bma_result.bf)) proteins")
    end

    println(io, " • Config hash            : $(string(r.config_hash, base=16))")
    println(io, " • Data hash              : $(string(r.data_hash, base=16))")
    println(io)

    # Summary statistics
    sig_05 = sum(r.copula_results.BFDR .< 0.05)
    sig_01 = sum(r.copula_results.BFDR .< 0.01)
    high_prob = sum(r.copula_results.posterior_prob .> 0.95)

    println(io, "Significant hits:")
    println(io, " • BFDR < 0.05            : $sig_05 ($(round(100*sig_05/length(r), digits=1))%)")
    println(io, " • BFDR < 0.01            : $sig_01 ($(round(100*sig_01/length(r), digits=1))%)")
    println(io, " • P(interaction) > 0.95  : $high_prob ($(round(100*high_prob/length(r), digits=1))%)")

    if !isnothing(r.sensitivity)
        sr = r.sensitivity
        n_settings = length(sr.prior_settings)
        mean_range = mean(sr.summary.range)
        max_range = maximum(sr.summary.range)
        println(io)
        println(io, "Sensitivity analysis:")
        println(io, " • Prior settings tested  : $n_settings")
        println(io, " • Mean posterior range   : $(round(mean_range, digits=4))")
        println(io, " • Max posterior range    : $(round(max_range, digits=4))")
    end

    if !isnothing(r.input_qc)
        println(io)
        println(io, "Input QC: $(r.input_qc)")
    end

    if !isnothing(r.diagnostics)
        dr = r.diagnostics
        n_checked = length(dr.protein_ppcs) + length(dr.bb_ppcs)
        ppc_pvals = [p.pvalue_mean for p in dr.protein_ppcs]
        mean_pval = isempty(ppc_pvals) ? NaN : mean(ppc_pvals)
        println(io)
        println(io, "Model diagnostics:")
        println(io, " • Proteins checked       : $n_checked")
        println(io, " • Mean PPC p-value (mean): $(round(mean_pval, digits=4))")
        if !isnothing(dr.calibration)
            println(io, " • ECE                    : $(round(dr.calibration.ece, digits=4))")
        end
    end
end
