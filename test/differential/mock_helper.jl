# Shared helper for creating mock AnalysisResult in differential tests.
# Include this file at the top of each @testitem that needs mock results.

using BayesInteractomics
using BayesInteractomics: EMResult
using DataFrames, Dates, Distributions, Copulas, Random

function _make_mock_result(;
    proteins = ["P$i" for i in 1:10],
    bfs = nothing,
    posteriors = nothing,
    q_vals = nothing,
    log2fcs = nothing,
    bf_enrichments = nothing,
    bf_correlations = nothing,
    bf_detecteds = nothing,
    seed = 42
)
    Random.seed!(seed)
    n = length(proteins)

    bfs = isnothing(bfs) ? rand(n) .* 100 : bfs
    posteriors = isnothing(posteriors) ? rand(n) : posteriors
    q_vals = isnothing(q_vals) ? rand(n) : q_vals
    log2fcs = isnothing(log2fcs) ? randn(n) : log2fcs
    bf_enrichments = isnothing(bf_enrichments) ? rand(n) .* 10 : bf_enrichments
    bf_correlations = isnothing(bf_correlations) ? rand(n) .* 10 : bf_correlations
    bf_detecteds = isnothing(bf_detecteds) ? rand(n) .* 10 : bf_detecteds

    copula_results = DataFrame(
        Protein = proteins,
        BF = Float64.(bfs),
        posterior_prob = Float64.(posteriors),
        q = q_vals,
        mean_log2FC = Float64.(log2fcs),
        bf_enrichment = Float64.(bf_enrichments),
        bf_correlation = Float64.(bf_correlations),
        bf_detected = Float64.(bf_detecteds)
    )
    df_hierarchical = DataFrame(
        Protein = proteins,
        BF_log2FC = Float64.(bf_enrichments),
        bf_slope = Float64.(bf_correlations),
        mean_log2FC = Float64.(log2fcs)
    )

    logs_df = DataFrame(
        iter = 1:5,
        loglikelihood = cumsum(randn(5)),
        pi0 = fill(0.7, 5),
        pi1 = fill(0.3, 5)
    )
    marginals = (Normal(0, 1), Normal(0, 1), Normal(0, 1))
    cop = GaussianCopula([1.0 0.5 0.3; 0.5 1.0 0.4; 0.3 0.4 1.0])
    joint = SklarDist(cop, marginals)
    em = EMResult(0.7, 0.3, joint, logs_df, true)

    return AnalysisResult(
        copula_results, df_hierarchical, em, joint, joint,
        nothing, :copula_em, nothing, nothing,
        UInt64(0), UInt64(0), now(), "0.1.0", nothing, nothing,
        nothing, nothing
    )
end
