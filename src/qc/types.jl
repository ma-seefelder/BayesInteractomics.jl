"""
    Input data quality control types.

Type hierarchy for QC checks that run after `load_data()` and before `analyse()`.
Each sub-result stores per-protocol results as a Vector, matching InteractionData's
multi-protocol structure. All flags use `:ok`/`:warning`/`:fail` vocabulary (QC-08).
"""

# ============================================================================
# Flag aggregation helper
# ============================================================================

"""
    worst_flag(flags::Symbol...) -> Symbol

Return the worst (most severe) flag from a collection.
Severity order: `:fail` > `:warning` > `:ok`.
Returns `:ok` if no flags are provided.
"""
function worst_flag(flags::Symbol...)
    :fail in flags && return :fail
    :warning in flags && return :warning
    return :ok
end

# ============================================================================
# Scale check types
# ============================================================================

"""
    ProtocolScaleCheck

Scale check result for a single protocol.

# Fields
- `protocol_index::Int`: Index of the protocol in InteractionData.
- `max_value::Float64`: Maximum observed intensity value across all experiments/replicates.
- `flag::Symbol`: `:ok` if max <= 1000, `:warning` if max > 1000.
"""
struct ProtocolScaleCheck
    protocol_index::Int
    max_value::Float64
    flag::Symbol
end

"""
    ScaleCheckResult

Aggregated scale check result across all protocols.

# Fields
- `protocols::Vector{ProtocolScaleCheck}`: Per-protocol scale check results.
- `flag::Symbol`: Worst flag across all protocols.
"""
struct ScaleCheckResult
    protocols::Vector{ProtocolScaleCheck}
    flag::Symbol
end

# ============================================================================
# Replicate correlation types
# ============================================================================

"""
    ReplicateCorrelation

Replicate correlation result for a single experiment group (sample or control).

# Fields
- `protocol_index::Int`: Index of the protocol.
- `experiment_index::Int`: Index of the experiment within the protocol.
- `group::Symbol`: `:sample` or `:control`.
- `correlation_matrix::Matrix{Float64}`: Pairwise Spearman correlation matrix.
- `shared_counts::Matrix{Int}`: Number of shared non-missing proteins per pair.
- `n_replicates::Int`: Number of replicates in the group.
- `min_correlation::Float64`: Minimum pairwise correlation value.
- `flag::Symbol`: `:ok` if min >= 0.80, `:warning` if 0.60 <= min < 0.80, `:fail` if min < 0.60.
"""
struct ReplicateCorrelation
    protocol_index::Int
    experiment_index::Int
    group::Symbol
    correlation_matrix::Matrix{Float64}
    shared_counts::Matrix{Int}
    n_replicates::Int
    min_correlation::Float64
    flag::Symbol
end

"""
    ReplicateCorrelationResult

Aggregated replicate correlation result across all protocols and experiments.

# Fields
- `checks::Vector{ReplicateCorrelation}`: Per-group correlation results.
- `flag::Symbol`: Worst flag across all checks.
"""
struct ReplicateCorrelationResult
    checks::Vector{ReplicateCorrelation}
    flag::Symbol
end

# ============================================================================
# Missingness types
# ============================================================================

"""
    ReplicateMissingness

Missingness asymmetry result for a single experiment group.

# Fields
- `protocol_index::Int`: Index of the protocol.
- `experiment_index::Int`: Index of the experiment within the protocol.
- `group::Symbol`: `:sample` or `:control`.
- `missing_fractions::Vector{Float64}`: Per-replicate missing fraction.
- `median_fraction::Float64`: Median missing fraction across replicates.
- `max_ratio::Float64`: Maximum ratio of replicate missing fraction to median.
- `flag::Symbol`: `:ok` if max_ratio <= 2, `:warning` if 2 < max_ratio <= 3, `:fail` if max_ratio > 3.
"""
struct ReplicateMissingness
    protocol_index::Int
    experiment_index::Int
    group::Symbol
    missing_fractions::Vector{Float64}
    median_fraction::Float64
    max_ratio::Float64
    flag::Symbol
end

"""
    MissingnessResult

Aggregated missingness asymmetry result across all protocols and experiments.

# Fields
- `checks::Vector{ReplicateMissingness}`: Per-group missingness results.
- `flag::Symbol`: Worst flag across all checks.
"""
struct MissingnessResult
    checks::Vector{ReplicateMissingness}
    flag::Symbol
end

# ============================================================================
# Intensity shape types
# ============================================================================

"""
    IntensityShapeCheck

Intensity distribution shape check for a single replicate.

# Fields
- `protocol_index::Int`: Index of the protocol.
- `experiment_index::Int`: Index of the experiment within the protocol.
- `group::Symbol`: `:sample` or `:control`.
- `replicate_index::Int`: Index of the replicate within the group.
- `n_values::Int`: Number of non-missing intensity values.
- `excess_kurtosis::Float64`: Excess kurtosis of the intensity distribution.
- `skewness_val::Float64`: Skewness of the intensity distribution.
- `spike_fraction::Float64`: Fraction of values at zero or minimum.
- `bimodality_flag::Symbol`: `:ok` or `:warning` based on kurtosis < -1.2.
- `spike_flag::Symbol`: `:ok` or `:warning` based on spike fraction.
- `tail_flag::Symbol`: `:ok` or `:warning` based on excess kurtosis.
- `flag::Symbol`: Worst of bimodality_flag, spike_flag, tail_flag.
"""
struct IntensityShapeCheck
    protocol_index::Int
    experiment_index::Int
    group::Symbol
    replicate_index::Int
    n_values::Int
    excess_kurtosis::Float64
    skewness_val::Float64
    spike_fraction::Float64
    bimodality_flag::Symbol
    spike_flag::Symbol
    tail_flag::Symbol
    flag::Symbol
end

"""
    IntensityShapeResult

Aggregated intensity shape check result across all protocols, experiments, and replicates.

# Fields
- `checks::Vector{IntensityShapeCheck}`: Per-replicate intensity shape results.
- `flag::Symbol`: Worst flag across all checks.
"""
struct IntensityShapeResult
    checks::Vector{IntensityShapeCheck}
    flag::Symbol
end

# ============================================================================
# PCA separation types
# ============================================================================

"""
    PCASeparationResult

PCA-based separation analysis for input data quality control.
Stores PC scores, sample/control labels, variance explained, and Fisher's discriminant
ratio for assessing whether experimental condition drives the primary variance.

# Fields
- `pc_scores::Matrix{Float64}`: PC scores matrix (n_samples x 2, columns = PC1, PC2)
- `condition_labels::Vector{String}`: "sample" or "control" per sample point
- `protocol_labels::Vector{Int}`: Protocol index per sample point (for color-coding)
- `variance_explained::Vector{Float64}`: Variance explained percentages [PC1%, PC2%, ...]
- `fisher_ratio_pc1::Float64`: Fisher's discriminant ratio on PC1 scores
- `fisher_ratio_pc2::Float64`: Fisher's discriminant ratio on PC2 scores
- `n_proteins_used::Int`: Number of proteins passing complete-case filter
- `n_proteins_total::Int`: Total number of proteins before filtering
- `fallback_level::Symbol`: :complete_case, :threshold_80, :threshold_50, or :skipped
- `flag::Symbol`: :ok or :warning
- `message::String`: Diagnostic message explaining the flag
- `per_protocol::Union{Nothing, Vector{PCASeparationResult}}`: Per-protocol results for multi-protocol datasets. Nothing for single-protocol or when this IS a per-protocol result.
"""
struct PCASeparationResult
    pc_scores::Matrix{Float64}
    condition_labels::Vector{String}
    protocol_labels::Vector{Int}
    variance_explained::Vector{Float64}
    fisher_ratio_pc1::Float64
    fisher_ratio_pc2::Float64
    n_proteins_used::Int
    n_proteins_total::Int
    fallback_level::Symbol
    flag::Symbol
    message::String
    per_protocol::Union{Nothing, Vector{PCASeparationResult}}
end

"""
    PCASeparationResult(; n_proteins_total::Int, message::String="PCA skipped: insufficient proteins")

Convenience constructor for the skipped case where PCA cannot be performed.
"""
function PCASeparationResult(; n_proteins_total::Int, message::String="PCA skipped: insufficient proteins")
    PCASeparationResult(
        Matrix{Float64}(undef, 0, 2),  # empty scores
        String[], Int[],                 # empty labels
        Float64[],                       # empty variance explained
        0.0, 0.0,                        # Fisher ratios
        0, n_proteins_total,
        :skipped, :warning, message, nothing
    )
end

function Base.show(io::IO, r::PCASeparationResult)
    flag_str = r.flag == :ok ? "ok" : "warning"
    println(io, "PCASeparationResult(:$flag_str)")
    println(io, "  Proteins used      : $(r.n_proteins_used) / $(r.n_proteins_total)")
    println(io, "  Fallback level     : :$(r.fallback_level)")
    if r.fallback_level != :skipped
        println(io, "  PC1 var explained  : $(round(r.variance_explained[1], digits=1))%")
        println(io, "  PC2 var explained  : $(round(r.variance_explained[2], digits=1))%")
        println(io, "  Fisher ratio PC1   : $(round(r.fisher_ratio_pc1, digits=3))")
        println(io, "  Fisher ratio PC2   : $(round(r.fisher_ratio_pc2, digits=3))")
    end
    if r.per_protocol !== nothing
        print(io, "  Per-protocol       : $(length(r.per_protocol)) results")
    end
end

# ============================================================================
# Top-level QC result
# ============================================================================

"""
    InputQCResult

Top-level container for all input data quality check results.
Follows the `DiagnosticsResult` pattern with `Union{Nothing, T}` fields.

# Fields
- `scale::Union{Nothing, ScaleCheckResult}`: Scale detection results.
- `replicate_correlation::Union{Nothing, ReplicateCorrelationResult}`: Replicate correlation results.
- `missingness::Union{Nothing, MissingnessResult}`: Missingness asymmetry results.
- `intensity_shape::Union{Nothing, IntensityShapeResult}`: Intensity distribution shape results.
- `pca_separation::Union{Nothing, PCASeparationResult}`: PCA separation analysis results.
- `overall_flag::Symbol`: Worst flag across all non-nothing sub-checks.
"""
struct InputQCResult
    scale::Union{Nothing, ScaleCheckResult}
    replicate_correlation::Union{Nothing, ReplicateCorrelationResult}
    missingness::Union{Nothing, MissingnessResult}
    intensity_shape::Union{Nothing, IntensityShapeResult}
    pca_separation::Union{Nothing, PCASeparationResult}
    overall_flag::Symbol
end

function Base.show(io::IO, r::InputQCResult)
    flags = String[]
    r.scale !== nothing && push!(flags, "scale=$(r.scale.flag)")
    r.replicate_correlation !== nothing && push!(flags, "correlation=$(r.replicate_correlation.flag)")
    r.missingness !== nothing && push!(flags, "missingness=$(r.missingness.flag)")
    r.intensity_shape !== nothing && push!(flags, "shape=$(r.intensity_shape.flag)")
    pca_str = isnothing(r.pca_separation) ? "not run" : ":$(r.pca_separation.flag)"
    push!(flags, "pca=$pca_str")
    print(io, "InputQCResult($(r.overall_flag); $(join(flags, ", ")))")
end
