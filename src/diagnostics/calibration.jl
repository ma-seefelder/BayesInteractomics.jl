# Calibration Assessment
# Compares predicted posterior probabilities to observed agreement rates

# ============================================================================ #
# Shared binning logic
# ============================================================================ #

"""
    _bin_calibration(posterior_probs, empirical_positive; n_bins=10) -> CalibrationResult

Shared binning logic for all calibration proxies. Bins proteins by predicted
posterior probability and compares to observed agreement rates.

# Arguments
- `posterior_probs::Vector{Float64}`: Predicted posterior probabilities
- `empirical_positive::BitVector`: Boolean mask of "empirically positive" proteins
- `n_bins::Int`: Number of equal-width bins across [0, 1]

# Returns
- `CalibrationResult` with bin midpoints, predicted/observed rates, ECE, and MCE
"""
function _bin_calibration(
    posterior_probs::Vector{Float64},
    empirical_positive::BitVector;
    n_bins::Int = 10
)
    bin_edges = range(0.0, 1.0, length=n_bins + 1)

    bin_midpoints = Float64[]
    predicted_rate = Float64[]
    observed_rate = Float64[]
    bin_counts = Int[]

    for i in 1:n_bins
        lo = bin_edges[i]
        hi = bin_edges[i + 1]

        # Include right endpoint for last bin
        if i == n_bins
            mask = (posterior_probs .>= lo) .& (posterior_probs .<= hi)
        else
            mask = (posterior_probs .>= lo) .& (posterior_probs .< hi)
        end

        count_in_bin = sum(mask)
        push!(bin_counts, count_in_bin)

        midpoint = (lo + hi) / 2.0
        push!(bin_midpoints, midpoint)

        if count_in_bin > 0
            push!(predicted_rate, mean(posterior_probs[mask]))
            push!(observed_rate, mean(empirical_positive[mask]))
        else
            push!(predicted_rate, midpoint)
            push!(observed_rate, 0.0)
        end
    end

    # Expected Calibration Error (ECE)
    total = sum(bin_counts)
    ece = 0.0
    mce = 0.0
    for i in eachindex(bin_midpoints)
        gap = abs(predicted_rate[i] - observed_rate[i])
        weight = bin_counts[i] / max(total, 1)
        ece += weight * gap
        mce = max(mce, gap)
    end

    return CalibrationResult(bin_midpoints, predicted_rate, observed_rate, bin_counts, ece, mce)
end

# ============================================================================ #
# Calibration proxies
# ============================================================================ #

"""
    _compute_calibration(ar; n_bins=10) -> CalibrationResult

Strict calibration proxy: a protein is "empirically positive" if ALL three
individual Bayes factors (enrichment, correlation, detection) exceed 1.0.

**Important:** Without a gold-standard dataset, this is a self-consistency check,
not true calibration. This strict proxy systematically miscounts constitutive
interactors that lack dose-response correlation.

# Arguments
- `ar`: Completed `AnalysisResult`
- `n_bins::Int`: Number of equal-width bins across [0, 1]
"""
function _compute_calibration(ar; n_bins::Int = 10)
    cr = ar.copula_results

    posterior_probs = Vector{Float64}(cr.posterior_prob)
    bf_enrichment = Vector{Float64}(cr.bf_enrichment)
    bf_correlation = Vector{Float64}(cr.bf_correlation)
    bf_detected = Vector{Float64}(cr.bf_detected)

    # Strict: all 3 BFs > 1.0
    empirical_positive = (bf_enrichment .> 1.0) .& (bf_correlation .> 1.0) .& (bf_detected .> 1.0)

    return _bin_calibration(posterior_probs, empirical_positive; n_bins=n_bins)
end

"""
    _compute_calibration_relaxed(ar; n_bins=10, min_bf_count=2) -> CalibrationResult

Relaxed calibration proxy: a protein is "empirically positive" if at least
`min_bf_count` of the 3 individual Bayes factors exceed 1.0.

This accommodates constitutive interactors (strong enrichment + detection,
weak correlation) and dose-dependent interactors alike. In the HAP40 interactome,
many genuine complex members (HTT, dynein/dynactin subunits, Rab5 effectors)
bind at saturating stoichiometry regardless of bait concentration, producing
BF_correlation ≈ 1.0 despite being true interactors.

# Arguments
- `ar`: Completed `AnalysisResult`
- `n_bins::Int`: Number of equal-width bins across [0, 1]
- `min_bf_count::Int`: Minimum number of BFs that must exceed 1.0 (default: 2)
"""
function _compute_calibration_relaxed(ar; n_bins::Int = 10, min_bf_count::Int = 2)
    cr = ar.copula_results

    posterior_probs = Vector{Float64}(cr.posterior_prob)
    bf_enrichment = Vector{Float64}(cr.bf_enrichment)
    bf_correlation = Vector{Float64}(cr.bf_correlation)
    bf_detected = Vector{Float64}(cr.bf_detected)

    # Count how many BFs exceed 1.0 per protein
    n_positive_bfs = Int.((bf_enrichment .> 1.0)) .+
                     Int.((bf_correlation .> 1.0)) .+
                     Int.((bf_detected .> 1.0))
    empirical_positive = BitVector(n_positive_bfs .>= min_bf_count)

    return _bin_calibration(posterior_probs, empirical_positive; n_bins=n_bins)
end

"""
    _compute_calibration_enrichment_only(ar; n_bins=10, bf_threshold=3.0) -> CalibrationResult

Enrichment-only calibration proxy: a protein is "empirically positive" if
BF_enrichment exceeds `bf_threshold`.

This is the least circular proxy available without external data, since enrichment
(log2 fold-change between sample and control) is the most direct AP-MS evidence
of interaction and is computed independently from the copula combination step.

# Arguments
- `ar`: Completed `AnalysisResult`
- `n_bins::Int`: Number of equal-width bins across [0, 1]
- `bf_threshold::Float64`: BF_enrichment threshold for "positive" (default: 3.0)
"""
function _compute_calibration_enrichment_only(ar; n_bins::Int = 10, bf_threshold::Float64 = 3.0)
    cr = ar.copula_results

    posterior_probs = Vector{Float64}(cr.posterior_prob)
    bf_enrichment = Vector{Float64}(cr.bf_enrichment)

    empirical_positive = BitVector(bf_enrichment .> bf_threshold)

    return _bin_calibration(posterior_probs, empirical_positive; n_bins=n_bins)
end
