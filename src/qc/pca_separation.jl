"""
    PCA separation analysis for input data quality control.

Complete-case PCA with cascading fallback, Fisher's discriminant ratio scoring,
and multi-protocol handling. Verifies that experimental condition (sample vs control)
drives the primary variance in AP-MS data.
"""

import LinearAlgebra: svd
import Statistics: mean, var

# ============================================================================
# Complete-case filtering with cascading fallback
# ============================================================================

"""
    filter_complete_case(data_matrix::Matrix{Union{Missing, Float64}}; min_proteins::Int=20)

Filter proteins (columns) by non-missing fraction with cascading fallback.
Tries 100% -> 80% -> 50% thresholds, returns (mask, level).

Cascading thresholds with a minimum of 20 proteins.
"""
function filter_complete_case(data_matrix::Matrix{Union{Missing, Float64}}; min_proteins::Int=20)
    n_samples, n_proteins = size(data_matrix)
    non_missing_frac = [count(!ismissing, @view(data_matrix[:, j])) / n_samples for j in 1:n_proteins]

    for (threshold, level) in [(1.0, :complete_case), (0.80, :threshold_80), (0.50, :threshold_50)]
        mask = non_missing_frac .>= threshold
        n_passing = count(mask)
        if n_passing >= min_proteins
            if level != :complete_case
                @warn "[QC] Complete-case PCA: using $(Int(threshold*100))% non-missing threshold ($n_passing / $n_proteins proteins)"
            end
            return mask, level
        end
        @warn "[QC] Complete-case PCA: only $n_passing proteins at $(Int(threshold*100))% threshold (need $min_proteins)"
    end

    @warn "[QC] Complete-case PCA: skipped -- fewer than $min_proteins proteins even at 50% threshold"
    return falses(n_proteins), :skipped
end

# ============================================================================
# Fisher's discriminant ratio
# ============================================================================

"""
    fishers_ratio(scores::AbstractVector{Float64}, labels::AbstractVector{String})

Compute Fisher's discriminant ratio: (mu_s - mu_c)^2 / (var_s + var_c).
Returns 0.0 when either group has fewer than 2 elements (small-group guard).
"""
function fishers_ratio(scores::AbstractVector{Float64}, labels::AbstractVector{String})
    sample_idx = labels .== "sample"
    control_idx = labels .== "control"

    n_s = count(sample_idx)
    n_c = count(control_idx)

    # Guard: need at least 2 per group for meaningful variance
    (n_s < 2 || n_c < 2) && return 0.0

    sample_scores = scores[sample_idx]
    control_scores = scores[control_idx]

    mu_s, mu_c = mean(sample_scores), mean(control_scores)
    var_s, var_c = var(sample_scores), var(control_scores)

    denom = var_s + var_c
    denom < eps(Float64) && return 0.0

    return (mu_s - mu_c)^2 / denom
end

# ============================================================================
# Flag assignment
# ============================================================================

"""
    assign_pca_flag(fisher_pc1::Float64, fisher_pc2::Float64, variance_explained::Vector{Float64})

Assign PCA quality flag based on Fisher's ratio and variance explained.
- :ok if Fisher >= 1.0 on PC1 OR PC2, AND PC1 variance >= 25%
- :warning otherwise
"""
function assign_pca_flag(fisher_pc1::Float64, fisher_pc2::Float64, variance_explained::Vector{Float64})
    messages = String[]
    flag = :ok

    # Check Fisher's ratio on PC1 and PC2
    if fisher_pc1 < 1.0 && fisher_pc2 < 1.0
        flag = :warning
        push!(messages, "Low Fisher separation: PC1=$(round(fisher_pc1, digits=2)), PC2=$(round(fisher_pc2, digits=2)) (both < 1.0)")
    end

    # Check PC1 variance explained (threshold 25%)
    if !isempty(variance_explained) && variance_explained[1] < 25.0
        flag = :warning
        push!(messages, "PC1 explains only $(round(variance_explained[1], digits=1))% variance (< 25%)")
    end

    if isempty(messages)
        return :ok, "Good condition separation on PCA"
    else
        return :warning, join(messages, "; ")
    end
end

# ============================================================================
# Data matrix construction
# ============================================================================

"""
    build_data_matrix(data::InteractionData, protocol_indices::AbstractVector{Int})

Extract a combined data matrix (samples-as-rows, proteins-as-columns) with
condition and protocol labels from InteractionData.

Per Pitfall 1 (data orientation) and Pitfall 4 (protocol extraction).
Protocol matrices are (n_proteins x n_replicates); we transpose to (n_samples x n_proteins).
"""
function build_data_matrix(data::InteractionData, protocol_indices::AbstractVector{Int})
    n_proteins = length(getIDs(data))
    all_columns = Vector{Vector{Union{Missing, Float64}}}()
    condition_labels = String[]
    protocol_labels = Int[]

    for p in protocol_indices
        samples_proto = getSamples(data, p)
        controls_proto = getControls(data, p)

        # Collect sample replicates
        for exp_idx in 1:getNoExperiments(samples_proto)
            mat = getExperiment(samples_proto, exp_idx)  # n_proteins x n_replicates
            for col in 1:size(mat, 2)
                push!(all_columns, mat[:, col])
                push!(condition_labels, "sample")
                push!(protocol_labels, p)
            end
        end

        # Collect control replicates
        for exp_idx in 1:getNoExperiments(controls_proto)
            mat = getExperiment(controls_proto, exp_idx)
            for col in 1:size(mat, 2)
                push!(all_columns, mat[:, col])
                push!(condition_labels, "control")
                push!(protocol_labels, p)
            end
        end
    end

    # Stack: n_samples x n_proteins
    data_matrix = Matrix{Union{Missing, Float64}}(reduce(hcat, all_columns)')
    return data_matrix, condition_labels, protocol_labels
end

# ============================================================================
# SVD-based PCA
# ============================================================================

"""
    compute_pca_scores(X::Matrix{Float64})

Compute PCA scores via SVD on centered data matrix X (n_samples x n_proteins).
Returns (scores[:, 1:2], variance_explained[1:2]).

Per Research Pattern 2.
"""
function compute_pca_scores(X::Matrix{Float64})
    col_means = mean(X, dims=1)
    X_centered = X .- col_means
    F = svd(X_centered)
    scores = F.U .* F.S'  # n_samples x n_components
    total_var = sum(F.S .^ 2)
    var_explained = (F.S .^ 2) ./ total_var .* 100.0
    # Return only first 2 PCs
    n_pcs = min(2, size(scores, 2))
    return scores[:, 1:n_pcs], var_explained[1:min(length(var_explained), 2)]
end

# ============================================================================
# Internal PCA pipeline for a single data matrix
# ============================================================================

"""
    _pca_for_matrix(data_matrix, condition_labels, protocol_labels, n_proteins_total)

Internal function: complete-case filtering + PCA + Fisher scoring on a single data matrix.
Returns PCASeparationResult.
"""
function _pca_for_matrix(data_matrix::Matrix{Union{Missing, Float64}},
                          condition_labels::Vector{String},
                          protocol_labels::Vector{Int},
                          n_proteins_total::Int)
    mask, level = filter_complete_case(data_matrix)

    if level == :skipped
        return PCASeparationResult(n_proteins_total=n_proteins_total,
                                    message="PCA skipped: fewer than 20 proteins at 50% non-missing threshold")
    end

    n_proteins_used = count(mask)
    X_filtered = data_matrix[:, mask]

    # Convert to Float64, imputing remaining missing with column mean (per Pitfall 2)
    X_float = Matrix{Float64}(undef, size(X_filtered))
    for j in 1:size(X_filtered, 2)
        col = X_filtered[:, j]
        non_missing = collect(skipmissing(col))
        col_mean = isempty(non_missing) ? 0.0 : mean(non_missing)
        for i in 1:size(X_filtered, 1)
            X_float[i, j] = ismissing(X_filtered[i, j]) ? col_mean : Float64(X_filtered[i, j])
        end
    end

    scores, var_explained = compute_pca_scores(X_float)

    # Pad to 2 PCs if needed (edge case: only 1 sample)
    if size(scores, 2) < 2
        scores = hcat(scores, zeros(size(scores, 1)))
        push!(var_explained, 0.0)
    end

    fr_pc1 = fishers_ratio(scores[:, 1], condition_labels)
    fr_pc2 = fishers_ratio(scores[:, 2], condition_labels)

    flag, message = assign_pca_flag(fr_pc1, fr_pc2, var_explained)

    return PCASeparationResult(
        scores[:, 1:2], condition_labels, protocol_labels,
        var_explained, fr_pc1, fr_pc2,
        n_proteins_used, n_proteins_total,
        level, flag, message, nothing
    )
end

# ============================================================================
# Main entry point
# ============================================================================

"""
    run_pca_separation(data::InteractionData)::Union{Nothing, PCASeparationResult}

Run PCA separation analysis on InteractionData.

For single-protocol datasets: returns combined PCA result.
For multi-protocol datasets: returns combined PCA with per-protocol results
nested in the `per_protocol` field.
"""
function run_pca_separation(data::InteractionData)::Union{Nothing, PCASeparationResult}
    n_protocols = getNoProtocols(data)
    n_proteins_total = length(getIDs(data))

    # Combined PCA (all protocols merged)
    all_protocols = collect(1:n_protocols)
    combined_matrix, combined_labels, combined_proto_labels = build_data_matrix(data, all_protocols)
    combined_result = _pca_for_matrix(combined_matrix, combined_labels, combined_proto_labels, n_proteins_total)

    if n_protocols == 1
        # Single protocol: combined IS the only result
        return combined_result
    end

    # Multi-protocol: run per-protocol PCA
    per_protocol_results = PCASeparationResult[]
    for p in 1:n_protocols
        proto_matrix, proto_labels, proto_plabels = build_data_matrix(data, [p])
        proto_result = _pca_for_matrix(proto_matrix, proto_labels, proto_plabels, n_proteins_total)
        push!(per_protocol_results, proto_result)
    end

    # If combined PCA shows poor separation but per-protocol shows good,
    # emit :warning (not :fail)
    combined_poor = combined_result.flag == :warning
    any_good = any(r -> r.flag == :ok, per_protocol_results)

    if combined_poor && any_good
        message = "Combined PCA shows poor condition separation (possible protocol batch effects), but per-protocol PCA shows good separation"
        @warn "[QC] $message"
        combined_flag = :warning
        combined_message = message
    else
        combined_flag = combined_result.flag
        combined_message = combined_result.message
    end

    # Return combined result with per-protocol nested
    return PCASeparationResult(
        combined_result.pc_scores,
        combined_result.condition_labels,
        combined_result.protocol_labels,
        combined_result.variance_explained,
        combined_result.fisher_ratio_pc1,
        combined_result.fisher_ratio_pc2,
        combined_result.n_proteins_used,
        combined_result.n_proteins_total,
        combined_result.fallback_level,
        combined_flag,
        combined_message,
        per_protocol_results
    )
end
