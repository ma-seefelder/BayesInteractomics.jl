# ──────────────────────────────────────────────────────────────────────────────
# Copula Diagnostic Tests
# Five diagnostic analyses + five plot functions for validating copula/LC models
# ──────────────────────────────────────────────────────────────────────────────

using Random
import StatsBase
using Distributions: Chisq, ccdf, cdf

# ─── 0. Quality Gate Engine ───────────────────────────────────────────────────

"""
    _merge_small_bins(observed::Vector{Int}, expected::Vector{Float64}; min_expected::Float64=5.0)

Merge adjacent bins until each has expected count >= min_expected.
Merges from the ends inward.
"""
function _merge_small_bins(observed::Vector{Int}, expected::Vector{Float64}; min_expected::Float64=5.0)
    obs = copy(observed)
    exp = copy(expected)
    # Merge from left
    while length(exp) > 1 && exp[1] < min_expected
        exp[2] += exp[1]; obs[2] += obs[1]
        deleteat!(exp, 1); deleteat!(obs, 1)
    end
    # Merge from right
    while length(exp) > 1 && exp[end] < min_expected
        exp[end-1] += exp[end]; obs[end-1] += obs[end]
        deleteat!(exp, length(exp)); deleteat!(obs, length(obs))
    end
    # Merge any remaining interior bins with expected < min_expected
    i = 1
    while i <= length(exp)
        if exp[i] < min_expected && length(exp) > 1
            if i < length(exp)
                exp[i+1] += exp[i]; obs[i+1] += obs[i]
                deleteat!(exp, i); deleteat!(obs, i)
            else
                exp[i-1] += exp[i]; obs[i-1] += obs[i]
                deleteat!(exp, i); deleteat!(obs, i)
            end
        else
            i += 1
        end
    end
    return obs, exp
end

"""
    _chisq_gof_discrete(observed::Vector{Int}, expected::Vector{Float64}; min_expected::Float64=5.0)

Chi-squared goodness-of-fit test with automatic bin merging.
Returns (chi2_statistic, p_value, df). If fewer than 3 bins remain after merging,
returns (0.0, 1.0, 0) indicating test is uninformative.
"""
function _chisq_gof_discrete(observed::Vector{Int}, expected::Vector{Float64}; min_expected::Float64=5.0)
    merged_obs, merged_exp = _merge_small_bins(observed, expected; min_expected=min_expected)
    k = length(merged_obs)
    if k < 3
        return (0.0, 1.0, 0)  # Too few bins for meaningful test
    end
    chi2 = sum((merged_obs[i] - merged_exp[i])^2 / merged_exp[i] for i in 1:k)
    df = k - 1
    p_value = ccdf(Chisq(df), chi2)
    return (chi2, p_value, df)
end

"""
    _fit_with_remediation(data::Vector{Float64}; ks_warn=0.1, ks_fail=0.15)

Fit Normal to data, compute KS. If KS > ks_fail, attempt TLocationScale remediation.
Returns (fitted_dist, ks_statistic, status, remediation_applied).
"""
function _fit_with_remediation(data::Vector{Float64}; ks_warn::Float64=0.1, ks_fail::Float64=0.15)
    # Fit Normal first
    fitted_normal = Distributions.fit(Normal, data)
    ks_normal = _ks_statistic(data, fitted_normal)

    if ks_normal <= ks_fail
        status = ks_normal < ks_warn ? :pass : :warn
        return (fitted_normal, ks_normal, status, false)
    end

    # Try TLocationScale remediation
    try
        fitted_t = Distributions.fit(LocationScale, data)
        ks_t = _ks_statistic(data, fitted_t)
        status = ks_t < ks_warn ? :pass : ks_t < ks_fail ? :warn : :fail
        return (fitted_t, ks_t, status, true)
    catch
        # TLocationScale failed, fall back to Normal
        status = ks_normal < ks_warn ? :pass : ks_normal < ks_fail ? :warn : :fail
        return (fitted_normal, ks_normal, status, false)
    end
end

"""
    _compute_hist_and_pdf(data::Vector{Float64}, fitted_dist; n_bins=30, n_pdf_points=200)

Compute normalized histogram (density) and fitted PDF curve for overlay plots.
Returns (bin_edges, counts, pdf_x, pdf_y).
"""
function _compute_hist_and_pdf(data::Vector{Float64}, fitted_dist; n_bins::Int=30, n_pdf_points::Int=200)
    if isempty(data)
        return (Float64[], Float64[], Float64[], Float64[])
    end

    h = StatsBase.fit(StatsBase.Histogram, data; nbins=n_bins)
    bin_edges = collect(Float64, h.edges[1])
    # Normalize to density: count / (n * bin_width)
    n_data = length(data)
    counts = Float64[]
    for i in 1:length(h.weights)
        bin_width = bin_edges[i+1] - bin_edges[i]
        push!(counts, bin_width > 0 ? h.weights[i] / (n_data * bin_width) : 0.0)
    end

    # PDF curve over extended range
    lo = minimum(data) - 0.5 * (maximum(data) - minimum(data) + 1e-6)
    hi = maximum(data) + 0.5 * (maximum(data) - minimum(data) + 1e-6)
    pdf_x = collect(range(lo, hi; length=n_pdf_points))
    pdf_y = Float64[pdf(fitted_dist, x) for x in pdf_x]

    return (bin_edges, counts, pdf_x, pdf_y)
end

"""
    run_quality_gates(bf::BayesFactorTriplet, lc_result::LatentClassResult;
                      ks_warn=0.1, ks_fail=0.15, min_effective_n=5)

Produce a 9-cell QualityGateResult (3 marginals x 3 components) with KS tests
and optional auto-remediation (Normal -> TLocationScale when KS > ks_fail).
Each cell includes histogram and fitted PDF data for diagnostic overlay plots.

Components with fewer than `min_effective_n` effective proteins get :pass by default.
"""
function run_quality_gates(bf::BayesFactorTriplet, lc_result::LatentClassResult;
                           ks_warn::Float64=0.1, ks_fail::Float64=0.15,
                           min_effective_n::Int=5)
    if lc_result.responsibilities === nothing
        error("LatentClassResult must have non-nothing responsibilities for quality gates")
    end

    n_resp = size(lc_result.responsibilities, 1)
    n_bf = length(bf.enrichment)
    if n_resp != n_bf
        error("Size mismatch: responsibilities has $n_resp rows but BayesFactorTriplet has $n_bf entries. " *
              "Ensure bf_triplet is filtered to detected proteins only.")
    end

    log_bf = [
        log.(max.(bf.enrichment, 1e-300)),
        log.(max.(bf.correlation, 1e-300)),
        log.(max.(bf.detection, 1e-300))
    ]
    marginal_labels = [:enrichment, :correlation, :detection]
    component_labels = [:H0, :agnostic, :H1]

    cells = Matrix{QualityGateCell}(undef, 3, 3)
    remediation_details = String[]

    # Use argmax assignment as PRIMARY method to produce non-overlapping
    # subsets per component. The previous 0.1-threshold approach caused all proteins
    # to appear in all component subsets (since most have responsibility > 0.1 for
    # all 3 components), leading to uniform ~0.49 KS values across all 9 cells.
    # Argmax ensures each protein is assigned to exactly one component, so the KS
    # test evaluates genuinely component-specific data.
    argmax_comp = [argmax(lc_result.responsibilities[i, :]) for i in 1:size(lc_result.responsibilities, 1)]

    for k in 1:3  # components
        w_k = lc_result.responsibilities[:, k]
        high_resp_idx = findall(argmax_comp .== k)
        n_effective = sum(w_k)

        for m in 1:3  # marginals
            if n_effective < min_effective_n || length(high_resp_idx) < min_effective_n
                @warn "Component $(component_labels[k]) has n_assigned=$(length(high_resp_idx)), n_effective=$(round(n_effective, digits=1)) (min=$min_effective_n) for $(marginal_labels[m]); skipping KS test"
                cells[m, k] = QualityGateCell(
                    marginal_labels[m], component_labels[k],
                    0.0, :pass, Normal(0.0, 1.0), n_effective, false,
                    Float64[], Float64[], Float64[], Float64[]
                )
                continue
            end

            data_subset = log_bf[m][high_resp_idx]

            # Detection marginal (m==3): chi-squared GOF against the fitted DiscreteEmpirical
            # Detection BFs are inherently discrete (Beta-Bernoulli counts), so the EM fits
            # per-component DiscreteEmpirical distributions — test against those, not a Normal.
            if m == 3
                # Retrieve the fitted DiscreteEmpirical for this component
                disc_dists = (lc_result.disc_detection_H0, lc_result.disc_detection_ag, lc_result.disc_detection_H1)
                fitted_disc = disc_dists[k]

                if fitted_disc === nothing
                    # No discrete distribution available — skip test
                    cells[m, k] = QualityGateCell(
                        marginal_labels[m], component_labels[k],
                        1.0, :pass, Normal(0.0, 1.0), n_effective, false,
                        Float64[], Float64[], Float64[], Float64[]
                    )
                    continue
                end

                n_data = length(data_subset)

                # Build observed counts per unique value in the data
                unique_vals = sort(unique(data_subset))
                if length(unique_vals) <= 1
                    # All values identical — perfect fit
                    cells[m, k] = QualityGateCell(
                        marginal_labels[m], component_labels[k],
                        1.0, :pass, fitted_disc, n_effective, false,
                        Float64[], Float64[], Float64[], Float64[]
                    )
                    continue
                end

                observed = Int[count(==(v), data_subset) for v in unique_vals]
                expected = Float64[n_data * pdf(fitted_disc, v) for v in unique_vals]

                chi2, p_value, df = _chisq_gof_discrete(observed, expected)

                # Map p-value to status (low p-value = poor fit)
                status = if df == 0
                    :pass  # Test uninformative (too few bins after merging)
                elseif p_value < 0.01
                    :fail
                elseif p_value < 0.05
                    :warn
                else
                    :pass
                end

                # Build histogram data for overlay plot using discrete support points
                bin_edges_out = Float64[]
                hist_counts_out = Float64[]
                pdf_x_out = Float64[v for v in unique_vals]
                pdf_y_out = Float64[pdf(fitted_disc, v) for v in unique_vals]

                cells[m, k] = QualityGateCell(
                    marginal_labels[m], component_labels[k],
                    p_value, status, fitted_disc, n_effective, false,
                    bin_edges_out, hist_counts_out, pdf_x_out, pdf_y_out
                )
                continue
            end

            # Enrichment and correlation marginals (m==1,2): use KS test
            fitted, ks, status, remediated = _fit_with_remediation(data_subset; ks_warn=ks_warn, ks_fail=ks_fail)
            bin_edges, hist_counts, pdf_x, pdf_y = _compute_hist_and_pdf(data_subset, fitted)

            cells[m, k] = QualityGateCell(
                marginal_labels[m], component_labels[k],
                ks, status, fitted, n_effective, remediated,
                bin_edges, hist_counts, pdf_x, pdf_y
            )

            if remediated
                push!(remediation_details,
                    "$(marginal_labels[m]) x $(component_labels[k]): Normal -> LocationScale (KS improved)")
            end
        end
    end

    # Overall status = worst cell
    status_order = Dict(:pass => 0, :warn => 1, :fail => 2)
    worst = :pass
    for cell in cells
        if get(status_order, cell.status, 0) > get(status_order, worst, 0)
            worst = cell.status
        end
    end

    return QualityGateResult(cells, worst, remediation_details)
end

"""
    compute_kl_contamination(bf::BayesFactorTriplet, lc_result::LatentClassResult;
                              pure_threshold=0.95, n_samples=10_000, seed=42)

Compute KL divergence between pure H1 (responsibility > pure_threshold) and
full H1 (responsibility > 0.5) distributions per evidence stream.
"""
function compute_kl_contamination(bf::BayesFactorTriplet, lc_result::LatentClassResult;
                                   pure_threshold::Float64=0.95,
                                   n_samples::Int=10_000,
                                   seed::Int=42)
    if lc_result.responsibilities === nothing
        error("LatentClassResult must have non-nothing responsibilities for KL contamination")
    end

    rng = Random.MersenneTwister(seed)
    h1_resp = lc_result.responsibilities[:, 3]
    pure_idx = findall(h1_resp .> pure_threshold)
    full_idx = findall(h1_resp .> 0.5)
    pure_count = length(pure_idx)

    if pure_count < 5
        @warn "Only $pure_count proteins have P(H1|x) > $pure_threshold; returning KL = 0"
        return KLContaminationResult(0.0, 0.0, 0.0, 0.0, pure_count, true)
    end

    log_bf = [
        log.(max.(bf.enrichment, 1e-300)),
        log.(max.(bf.correlation, 1e-300)),
        log.(max.(bf.detection, 1e-300))
    ]

    kl_vals = Float64[]
    for m in 1:3
        full_data = log_bf[m][full_idx]
        pure_data = log_bf[m][pure_idx]

        full_dist = Distributions.fit(Normal, full_data)
        pure_dist = Distributions.fit(Normal, pure_data)

        samples = rand(rng, pure_dist, n_samples)
        log_pure = logpdf.(pure_dist, samples)
        log_full = logpdf.(full_dist, samples)

        valid = isfinite.(log_pure) .& isfinite.(log_full)
        if sum(valid) > 0
            kl = mean((log_pure .- log_full)[valid])
            push!(kl_vals, max(0.0, kl))
        else
            push!(kl_vals, 0.0)
        end
    end

    kl_joint = sum(kl_vals)
    per_stream_pass = all(kl_vals .< 0.5)

    return KLContaminationResult(kl_vals[1], kl_vals[2], kl_vals[3], kl_joint, pure_count, per_stream_pass)
end


# ─── 1. KL(H1_full ‖ H1_pure) — Garbage Collection Severity ─────────────────

"""
    kl_h1_divergence(bf::BayesFactorTriplet, lc_result::LatentClassResult;
                     pure_threshold=0.95, n_samples=10_000, seed=42)

Measure KL divergence between pure H1 (responsibility > pure_threshold) and full H1
(responsibility > 0.5) distributions on the log-BF scale. High KL (> 2 nats) indicates
the H1 component has absorbed many non-interactors.

Returns a NamedTuple with:
- `kl_enrichment`, `kl_correlation`, `kl_detection`: per-dimension KL divergences
- `kl_total`: sum of per-dimension KLs
- `pure_marginals`: tuple of Normal distributions fitted to pure H1 (log-BF scale)
- `full_marginals`: tuple of Normal distributions fitted to full H1 (log-BF scale)
- `top_indices`: indices of pure H1 proteins
"""
function kl_h1_divergence(bf::BayesFactorTriplet, lc_result::LatentClassResult;
                          pure_threshold::Float64 = 0.95,
                          n_samples::Int = 10_000,
                          seed::Int = 42)
    if lc_result.responsibilities === nothing
        error("LatentClassResult must have non-nothing responsibilities")
    end

    rng = Random.MersenneTwister(seed)
    h1_resp = lc_result.responsibilities[:, 3]
    pure_idx = findall(h1_resp .> pure_threshold)
    full_idx = findall(h1_resp .> 0.5)

    log_bf = [
        log.(max.(bf.enrichment, 1e-300)),
        log.(max.(bf.correlation, 1e-300)),
        log.(max.(bf.detection, 1e-300))
    ]

    if length(pure_idx) < 5
        @warn "Only $(length(pure_idx)) proteins with P(H1|x) > $pure_threshold"
        zero_margs = Tuple(Normal(0.0, 1.0) for _ in 1:3)
        return (kl_enrichment=0.0, kl_correlation=0.0, kl_detection=0.0,
                kl_total=0.0, pure_marginals=zero_margs, full_marginals=zero_margs,
                top_indices=pure_idx)
    end

    pure_margs = []
    full_margs = []
    kl_vals = Float64[]

    for m in 1:3
        full_data = log_bf[m][full_idx]
        pure_data = log_bf[m][pure_idx]

        full_dist = Distributions.fit(Normal, full_data)
        pure_dist = Distributions.fit(Normal, pure_data)
        push!(full_margs, full_dist)
        push!(pure_margs, pure_dist)

        samples = rand(rng, pure_dist, n_samples)
        log_pure = logpdf.(pure_dist, samples)
        log_full = logpdf.(full_dist, samples)

        valid = isfinite.(log_pure) .& isfinite.(log_full)
        if sum(valid) > 0
            kl = mean((log_pure .- log_full)[valid])
            push!(kl_vals, max(0.0, kl))
        else
            push!(kl_vals, 0.0)
        end
    end

    return (
        kl_enrichment  = kl_vals[1],
        kl_correlation = kl_vals[2],
        kl_detection   = kl_vals[3],
        kl_total       = sum(kl_vals),
        pure_marginals = Tuple(pure_margs),
        full_marginals = Tuple(full_margs),
        top_indices    = pure_idx,
    )
end

# Deprecated: old signature wraps to old implementation
"""
    kl_h1_divergence(combined_result::CombinedBayesResult, p::PosteriorProbabilityTriplet; ...)

Deprecated: use `kl_h1_divergence(bf, lc_result)` for log-BF scale diagnostics.
"""
function kl_h1_divergence(combined_result::CombinedBayesResult,
                          p::PosteriorProbabilityTriplet;
                          top_n::Int = 100,
                          n_samples::Int = 10_000,
                          seed::Int = 42)
    @warn "Deprecated: use kl_h1_divergence(bf, lc_result) for log-BF scale diagnostics" maxlog=1
    rng = Random.MersenneTwister(seed)

    # Get H1 marginals from EM result
    full_marginals = extract_marginals(combined_result.em_result.joint_H1)

    # Identify top-N proteins by combined posterior probability
    pp = combined_result.posterior_prob
    n = length(pp)
    top_n_actual = min(top_n, n)
    top_indices = partialsortperm(pp, 1:top_n_actual, rev = true)

    # Fit "pure" H1 marginals on top-N proteins only
    dims = (:enrichment, :correlation, :detection)
    p_vecs = (p.enrichment, p.correlation, p.detection)

    pure_margs = []
    kl_vals = Float64[]

    for (i, dim) in enumerate(dims)
        top_data = p_vecs[i][top_indices]
        # Squeeze to (ϵ, 1-ϵ) for Beta fitting
        ϵ = 1e-6
        top_sq = clamp.(top_data, ϵ, 1 - ϵ)

        pure_marg = fit_beta_mixture(top_sq; max_K = 3, verbose = false)
        push!(pure_margs, pure_marg)

        # Monte Carlo KL: E_pure[log(pure(x)) - log(full(x))]
        samples = rand(rng, pure_marg, n_samples)
        samples = clamp.(samples, ϵ, 1 - ϵ)

        log_pure = logpdf.(pure_marg, samples)
        log_full = logpdf.(full_marginals[i], samples)

        # Filter out -Inf or NaN
        valid = isfinite.(log_pure) .& isfinite.(log_full)
        if sum(valid) > 0
            kl = mean((log_pure .- log_full)[valid])
            push!(kl_vals, max(0.0, kl))  # KL >= 0 by definition
        else
            push!(kl_vals, 0.0)
        end
    end

    return (
        kl_enrichment  = kl_vals[1],
        kl_correlation = kl_vals[2],
        kl_detection   = kl_vals[3],
        kl_total       = sum(kl_vals),
        pure_marginals = Tuple(pure_margs),
        full_marginals = Tuple(full_marginals),
        top_indices    = top_indices,
    )
end

"""
    kl_divergence_plot(kl_result, bf::BayesFactorTriplet; file="")

3×1 layout: overlay H1_full (blue) vs H1_pure (red) density for each dimension
on the log-BF scale. KL value annotated per panel. Flags KL > 2.0 with a warning marker.
"""
function kl_divergence_plot(kl_result, bf::BayesFactorTriplet; file::String = "")
    dims = ("Enrichment", "Correlation", "Detection")
    fields = (:enrichment, :correlation, :detection)
    kls = (kl_result.kl_enrichment, kl_result.kl_correlation, kl_result.kl_detection)

    plt = StatsPlots.plot(layout = (3, 1), size = (700, 900),
        plot_title = "KL Divergence: H₁ Full vs Pure",
        left_margin = 10 * StatsPlots.Plots.mm,
        bottom_margin = 5 * StatsPlots.Plots.mm)

    for (i, (dim, field)) in enumerate(zip(dims, fields))
        log_data = log.(max.(getproperty(bf, field), 1e-300))
        xs = range(minimum(log_data) - 1.0, maximum(log_data) + 1.0, length = 200)

        full_d = kl_result.full_marginals[i]
        pure_d = kl_result.pure_marginals[i]

        y_full = pdf.(full_d, xs)
        y_pure = pdf.(pure_d, xs)

        flag = kls[i] > 2.0 ? " ⚠" : ""
        StatsPlots.plot!(plt, xs, y_full,
            label = "H₁ full", color = :steelblue, lw = 2, subplot = i)
        StatsPlots.plot!(plt, xs, y_pure,
            label = "H₁ pure (top-$(length(kl_result.top_indices)))", color = :red, lw = 2, subplot = i,
            title = "$dim — KL = $(round(kls[i], digits=3))$flag",
            xlabel = "log(BF)", ylabel = "Density")
    end

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end


# ─── 2. Within-Class Correlation — Conditional Independence ──────────────────

"""
    within_class_correlation(bf::BayesFactorTriplet, lc_result::LatentClassResult; threshold=0.5)

Test conditional independence assumption of the 3-component latent class model.
Computes pairwise Spearman correlations of log-BFs within each of the 3 classes
(H0, agnostic, H1) using the responsibilities matrix.

Returns a NamedTuple with `h0_corr`, `agnostic_corr`, `h1_corr` (3x3 matrices),
`h0_n`, `agnostic_n`, `h1_n` (counts).
"""
function within_class_correlation(bf::BayesFactorTriplet,
                                  lc_result::LatentClassResult;
                                  threshold::Float64 = 0.5)
    log_e = log.(max.(bf.enrichment, 1e-300))
    log_c = log.(max.(bf.correlation, 1e-300))
    log_d = log.(max.(bf.detection, 1e-300))

    function _spearman_matrix(idx)
        if length(idx) < 5
            return fill(NaN, 3, 3)
        end
        mat = hcat(log_e[idx], log_c[idx], log_d[idx])
        corr = zeros(3, 3)
        for i in 1:3, j in 1:3
            if i == j
                corr[i, j] = 1.0
            else
                corr[i, j] = StatsBase.corspearman(mat[:, i], mat[:, j])
            end
        end
        return corr
    end

    # Use responsibilities for 3-component assignment
    if lc_result.responsibilities !== nothing
        h0_idx = findall(lc_result.responsibilities[:, 1] .> threshold)
        ag_idx = findall(lc_result.responsibilities[:, 2] .> threshold)
        h1_idx = findall(lc_result.responsibilities[:, 3] .> threshold)
    else
        # Fallback to posterior_prob for 2-class backward compat
        pp = lc_result.posterior_prob
        h0_idx = findall(pp .< threshold)
        ag_idx = Int[]
        h1_idx = findall(pp .>= threshold)
    end

    # Precompute correlation matrices to avoid redundant calls
    h0_corr_mat = _spearman_matrix(h0_idx)
    ag_corr_mat = _spearman_matrix(ag_idx)
    h1_corr_mat = _spearman_matrix(h1_idx)

    return (
        h0_corr       = h0_corr_mat,
        agnostic_corr = ag_corr_mat,
        h1_corr       = h1_corr_mat,
        h0_n          = length(h0_idx),
        agnostic_n    = length(ag_idx),
        h1_n          = length(h1_idx),
        # Class membership indices
        h0_idx        = h0_idx,
        ag_idx        = ag_idx,
        h1_idx        = h1_idx,
        # Backward-compat aliases
        bg_corr       = h0_corr_mat,
        int_corr      = h1_corr_mat,
        bg_count      = length(h0_idx),
        int_count     = length(h1_idx),
        max_abs_bg    = let m = h0_corr_mat
            vals = [abs(m[i,j]) for i in 1:3 for j in 1:3 if i!=j]
            finite_vals = filter(isfinite, vals)
            isempty(finite_vals) ? NaN : maximum(finite_vals)
        end,
        max_abs_int   = let m = h1_corr_mat
            vals = [abs(m[i,j]) for i in 1:3 for j in 1:3 if i!=j]
            finite_vals = filter(isfinite, vals)
            isempty(finite_vals) ? NaN : maximum(finite_vals)
        end,
    )
end

"""
    within_class_correlation_plot(wc_result, bf::BayesFactorTriplet; file="")

3×3 layout: rows=(H0, Agnostic, H1), cols=(e-c, e-d, c-d scatter).
Each subplot shows only proteins belonging to that class. Flags |ρ| > 0.3 with red color.
"""
function within_class_correlation_plot(wc_result, bf::BayesFactorTriplet; file::String = "")
    pp_labels = ["Enrichment", "Correlation", "Detection"]
    pairs = [(1, 2), (1, 3), (2, 3)]
    log_vecs = [log.(max.(bf.enrichment, 1e-300)),
                log.(max.(bf.correlation, 1e-300)),
                log.(max.(bf.detection, 1e-300))]

    plt = StatsPlots.plot(layout = (3, 3), size = (1200, 1050),
        plot_title = "Within-Class Correlation (Conditional Independence Check)",
        left_margin = 10 * StatsPlots.Plots.mm,
        bottom_margin = 8 * StatsPlots.Plots.mm,
        top_margin = 8 * StatsPlots.Plots.mm)

    corr_mats = [wc_result.h0_corr, wc_result.agnostic_corr, wc_result.h1_corr]
    class_indices = [wc_result.h0_idx, wc_result.ag_idx, wc_result.h1_idx]
    class_names = ["H0 (n=$(wc_result.h0_n))", "Agnostic (n=$(wc_result.agnostic_n))", "H1 (n=$(wc_result.h1_n))"]

    for (row, (corr_mat, cls_name, idx)) in enumerate(zip(corr_mats, class_names, class_indices))
        for (col, (i, j)) in enumerate(pairs)
            sp = (row - 1) * 3 + col

            if isempty(idx)
                StatsPlots.plot!(plt, subplot = sp,
                    xlabel = "log(BF_$(pp_labels[i]))", ylabel = "log(BF_$(pp_labels[j]))",
                    title = "$cls_name — ρ=N/A", titlefontsize = 9)
                StatsPlots.annotate!(plt, [(0.5, 0.5, StatsPlots.text("No proteins", 10, :gray))], subplot = sp)
                continue
            end

            rho = corr_mat[i, j]
            flag_color = abs(rho) > 0.3 ? :red : :steelblue
            rho_str = isfinite(rho) ? "ρ=$(round(rho, digits=3))" : "ρ=N/A"

            # Filter scatter data to class-specific proteins only
            StatsPlots.scatter!(plt, log_vecs[i][idx], log_vecs[j][idx],
                alpha = 0.15, ms = 2, color = flag_color,
                xlabel = "log(BF_$(pp_labels[i]))", ylabel = "log(BF_$(pp_labels[j]))",
                title = "$cls_name — $rho_str",
                titlefontsize = 9,
                label = "", subplot = sp)

            # Identity line from filtered data only
            all_vals = vcat(log_vecs[i][idx], log_vecs[j][idx])
            lims = (minimum(all_vals), maximum(all_vals))
            StatsPlots.plot!(plt, [lims[1], lims[2]], [lims[1], lims[2]],
                color = :gray, ls = :dash, lw = 1, label = "", subplot = sp)
        end
    end

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end


# ─── 3. Agnostic-Zone Exclusion — 0.5-Spike Influence ────────────────────────

"""
    agnostic_zone_analysis(bf::BayesFactorTriplet, lc_result::LatentClassResult; threshold=0.5)

Identify proteins assigned to the agnostic component (responsibilities[:, 2] > threshold).
These carry minimal evidence and may dominate EM convergence.

Returns a NamedTuple with zone_indices, zone_fraction, n_zone, n_total, mean log-BF stats.
"""
function agnostic_zone_analysis(bf::BayesFactorTriplet,
                                lc_result::LatentClassResult;
                                threshold::Float64 = 0.5)
    n = length(bf.enrichment)
    if lc_result.responsibilities !== nothing
        zone_indices = findall(lc_result.responsibilities[:, 2] .> threshold)
    else
        zone_indices = Int[]
    end
    rest_indices = setdiff(1:n, zone_indices)

    combined_bf = lc_result.bf
    abs_log_bf = abs.(log.(max.(combined_bf, 1e-300)))

    mean_zone = isempty(zone_indices) ? 0.0 : mean(abs_log_bf[zone_indices])
    mean_rest = isempty(rest_indices) ? 0.0 : mean(abs_log_bf[rest_indices])

    return (
        zone_indices       = zone_indices,
        zone_fraction      = length(zone_indices) / n,
        n_zone             = length(zone_indices),
        n_total            = n,
        mean_abs_log_bf_zone = mean_zone,
        mean_abs_log_bf_rest = mean_rest,
    )
end

# Deprecated: old signature
function agnostic_zone_analysis(bf::BayesFactorTriplet,
                                p::PosteriorProbabilityTriplet,
                                combined_result::CombinedBayesResult;
                                zone_lo::Float64 = 0.4,
                                zone_hi::Float64 = 0.6)
    n = length(p.enrichment)
    zone_mask = (p.enrichment .>= zone_lo) .& (p.enrichment .<= zone_hi) .&
                (p.correlation .>= zone_lo) .& (p.correlation .<= zone_hi) .&
                (p.detection .>= zone_lo) .& (p.detection .<= zone_hi)
    zone_indices = findall(zone_mask)
    rest_indices = findall(.!zone_mask)

    combined_bf = combined_result.bf
    abs_log_bf = abs.(log.(max.(combined_bf, 1e-300)))

    mean_zone = isempty(zone_indices) ? 0.0 : mean(abs_log_bf[zone_indices])
    mean_rest = isempty(rest_indices) ? 0.0 : mean(abs_log_bf[rest_indices])

    return (
        zone_indices       = zone_indices,
        zone_fraction      = length(zone_indices) / n,
        n_zone             = length(zone_indices),
        n_total            = n,
        mean_abs_log_bf_zone = mean_zone,
        mean_abs_log_bf_rest = mean_rest,
    )
end

"""
    agnostic_zone_plot(az_result, p::PosteriorProbabilityTriplet,
                       combined_result::CombinedBayesResult; file="")

2-panel: (1) enrichment vs correlation scatter colored by zone membership,
(2) overlaid log(combined BF) histograms for zone vs non-zone.
"""
function agnostic_zone_plot(az_result, p::PosteriorProbabilityTriplet,
                            combined_result::CombinedBayesResult; file::String = "")
    plt = StatsPlots.plot(layout = (1, 2), size = (1000, 450),
        plot_title = "Agnostic Zone Analysis ($(az_result.n_zone)/$(az_result.n_total) = $(round(100*az_result.zone_fraction, digits=1))%)",
        left_margin = 10 * StatsPlots.Plots.mm,
        bottom_margin = 8 * StatsPlots.Plots.mm)

    zone = az_result.zone_indices
    all_idx = 1:az_result.n_total
    rest = setdiff(all_idx, zone)

    # Panel 1: enrichment vs correlation scatter
    StatsPlots.scatter!(plt, p.enrichment[rest], p.correlation[rest],
        alpha = 0.2, ms = 2, color = :steelblue, label = "Non-zone",
        xlabel = "P(H₁|enrichment)", ylabel = "P(H₁|correlation)",
        title = "Posterior Space", subplot = 1)
    if !isempty(zone)
        StatsPlots.scatter!(plt, p.enrichment[zone], p.correlation[zone],
            alpha = 0.5, ms = 3, color = :orange, label = "Agnostic zone", subplot = 1)
    end

    # Panel 2: log(BF) histograms
    log_bf = log.(max.(combined_result.bf, 1e-300))
    if !isempty(rest)
        StatsPlots.histogram!(plt, log_bf[rest],
            bins = 50, alpha = 0.5, color = :steelblue, label = "Non-zone",
            xlabel = "log(Combined BF)", ylabel = "Count",
            title = "Combined BF Distribution", subplot = 2)
    end
    if !isempty(zone)
        StatsPlots.histogram!(plt, log_bf[zone],
            bins = 30, alpha = 0.6, color = :orange, label = "Agnostic zone", subplot = 2)
    end

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end


# ─── 4. Copula Bootstrap CI — H1 Reliability ─────────────────────────────────

"""
    copula_bootstrap_ci(bf::BayesFactorTriplet, lc_result::LatentClassResult;
                        n_bootstrap=10, seed=42)

Bootstrap stability of log-BF combination by resampling proteins and re-fitting
Normal marginals. Returns KS and Kendall tau CIs.
"""
function copula_bootstrap_ci(bf::BayesFactorTriplet,
                             lc_result::LatentClassResult;
                             n_bootstrap::Int = 10,
                             seed::Int = 42)
    rng = Random.MersenneTwister(seed)
    n = length(bf.enrichment)

    log_e = log.(max.(bf.enrichment, 1e-300))
    log_c = log.(max.(bf.correlation, 1e-300))
    log_d = log.(max.(bf.detection, 1e-300))

    ks_stats = Float64[]
    tau_stats = Float64[]
    completed = 0

    for b in 1:n_bootstrap
        idx = rand(rng, 1:n, n)
        try
            boot_log = hcat(log_e[idx], log_c[idx], log_d[idx])
            # Fit Normal marginals on bootstrap sample
            boot_margs = [Distributions.fit(Normal, boot_log[:, i]) for i in 1:3]
            ks_max = maximum(_ks_statistic(boot_log[:, i], boot_margs[i]) for i in 1:3)
            push!(ks_stats, ks_max)

            tau_mat = StatsBase.corkendall(boot_log)
            tau_vals = [tau_mat[i, j] for i in 1:3 for j in (i+1):3]
            push!(tau_stats, mean(tau_vals))
            completed += 1
        catch
            continue
        end
    end

    ks_ci = isempty(ks_stats) ? (NaN, NaN) : (quantile(ks_stats, 0.025), quantile(ks_stats, 0.975))
    tau_ci = isempty(tau_stats) ? (NaN, NaN) : (quantile(tau_stats, 0.025), quantile(tau_stats, 0.975))

    return (
        ks_stats = ks_stats,
        ks_ci = ks_ci,
        tau_stats = tau_stats,
        tau_ci = tau_ci,
        n_bootstrap_completed = completed,
    )
end

"""
    copula_bootstrap_ci(p::PosteriorProbabilityTriplet,
                        combined_result::CombinedBayesResult;
                        n_bootstrap=30, seed=42)

Assess stability of H1 copula family selection by bootstrap resampling.
Resamples proteins, refits copula + marginals, collects family name and KS stats.

Uses only the 4 parametric Archimedean + Gaussian families (skips EmpiricalCopula
which is slow and not used in BMA).

Returns a NamedTuple with:
- `family_counts`: Dict of family name => count
- `family_fractions`: Dict of family name => fraction
- `dominant_family`: most frequently selected family
- `dominant_fraction`: fraction of bootstraps selecting dominant family
- `ks_stats`: vector of per-bootstrap KS statistics (from best family)
- `ks_ci`: (lo, hi) 95% CI for KS
- `tau_stats`: vector of mean pairwise Kendall τ per bootstrap
- `tau_ci`: (lo, hi) 95% CI for τ
"""
function copula_bootstrap_ci(p::PosteriorProbabilityTriplet,
                             combined_result::CombinedBayesResult;
                             n_bootstrap::Int = 30,
                             seed::Int = 42)
    rng = Random.MersenneTwister(seed)
    n = length(p.enrichment)

    # Fast copula families only (skip EmpiricalCopula — slow and not used in BMA)
    boot_families = Dict(
        "ClaytonCopula"  => ClaytonCopula,
        "FrankCopula"    => FrankCopula,
        "GumbelCopula"   => GumbelCopula,
        "GaussianCopula" => GaussianCopula,
        "JoeCopula"      => JoeCopula,
    )
    error_only_logger = MinLevelLogger(current_logger(), Logging.Error)

    family_counts = Dict{String, Int}()
    ks_stats = Float64[]
    tau_stats = Float64[]

    @info "Copula bootstrap: running $n_bootstrap iterations (n=$n proteins)..."
    t0 = time()

    for b in 1:n_bootstrap
        idx = rand(rng, 1:n, n)
        p_boot = PosteriorProbabilityTriplet(
            p.enrichment[idx],
            p.correlation[idx],
            p.detection[idx]
        )
        p_sq = squeeze(p_boot, ϵ = 1e-6)

        try
            # Fast inline copula comparison (avoids compare_copulas overhead + EmpiricalCopula)
            u = hcat(p_sq.enrichment, p_sq.correlation, p_sq.detection)'
            best_family = ""
            best_bic = Inf
            for (copula_name, fam) in boot_families
                try
                    with_logger(error_only_logger) do
                        cop = fit(fam, u)
                        ll = loglikelihood(cop, u)
                        k = copula_nparams(fam)
                        bic = -2 * ll + k * log(n)
                        if bic < best_bic
                            best_bic = bic
                            best_family = copula_name
                        end
                    end
                catch; end
            end
            isempty(best_family) && continue
            family_counts[best_family] = get(family_counts, best_family, 0) + 1

            # Fit Beta marginals per dimension and compute KS
            dims_data = [p_sq.enrichment, p_sq.correlation, p_sq.detection]
            boot_margs = [fit_beta_mixture(d; max_K = 2, verbose = false) for d in dims_data]
            ks_max = maximum(_ks_statistic(dims_data[i], boot_margs[i]) for i in 1:3)
            push!(ks_stats, ks_max)

            # Mean pairwise Kendall τ
            mat = hcat(p_sq.enrichment, p_sq.correlation, p_sq.detection)
            tau_mat = StatsBase.corkendall(mat)
            # Average off-diagonal elements
            tau_vals = [tau_mat[i, j] for i in 1:3 for j in (i+1):3]
            push!(tau_stats, mean(tau_vals))
        catch
            # Skip failed bootstraps
            continue
        end

        # Progress every 10 iterations
        if b % 10 == 0
            elapsed = round(time() - t0, digits=1)
            @info "Copula bootstrap: $b/$n_bootstrap done ($(elapsed)s elapsed)"
        end
    end

    elapsed_total = round(time() - t0, digits=1)
    @info "Copula bootstrap: completed in $(elapsed_total)s"

    # Compute summary stats
    total = sum(values(family_counts))
    family_fracs = Dict(k => v / max(total, 1) for (k, v) in family_counts)
    dominant = isempty(family_counts) ? "N/A" : argmax(family_counts)
    dom_frac = isempty(family_fracs) ? 0.0 : get(family_fracs, dominant, 0.0)

    ks_ci = isempty(ks_stats) ? (NaN, NaN) : (quantile(ks_stats, 0.025), quantile(ks_stats, 0.975))
    tau_ci = isempty(tau_stats) ? (NaN, NaN) : (quantile(tau_stats, 0.025), quantile(tau_stats, 0.975))

    return (
        family_counts    = family_counts,
        family_fractions = family_fracs,
        dominant_family  = dominant,
        dominant_fraction = dom_frac,
        ks_stats         = ks_stats,
        ks_ci            = ks_ci,
        tau_stats        = tau_stats,
        tau_ci           = tau_ci,
    )
end

"""
    copula_bootstrap_plot(boot_result; file="")

3-panel: (1) family selection bar chart, (2) KS statistic boxplot, (3) τ histogram with CI.
"""
function copula_bootstrap_plot(boot_result; file::String = "")
    plt = StatsPlots.plot(layout = (1, 3), size = (1200, 400),
        plot_title = "Copula Bootstrap Confidence (n=$(sum(values(boot_result.family_counts))))",
        left_margin = 10 * StatsPlots.Plots.mm,
        bottom_margin = 10 * StatsPlots.Plots.mm)

    # Panel 1: Family bar chart
    families = collect(keys(boot_result.family_fractions))
    fracs = [boot_result.family_fractions[f] for f in families]
    if !isempty(families)
        perm = sortperm(fracs, rev = true)
        families = families[perm]
        fracs = fracs[perm]
        colors = [f == boot_result.dominant_family ? :steelblue : :lightgray for f in families]
        StatsPlots.bar!(plt, 1:length(families), fracs,
            xticks = (1:length(families), families), xrotation = 30,
            color = colors, label = "",
            ylabel = "Selection frequency",
            title = "Copula Family Stability", subplot = 1)
        StatsPlots.hline!(plt, [0.7], ls = :dash, color = :red, label = "70% threshold", subplot = 1)
    end

    # Panel 2: KS statistic boxplot
    if !isempty(boot_result.ks_stats)
        StatsPlots.boxplot!(plt, ["KS"], boot_result.ks_stats,
            color = :steelblue, label = "",
            ylabel = "KS statistic",
            title = "Marginal Fit (KS)", subplot = 2)
        lo, hi = boot_result.ks_ci
        StatsPlots.annotate!(plt,
            [(1, hi + 0.005, StatsPlots.Plots.text("95% CI: [$(round(lo,digits=3)), $(round(hi,digits=3))]", 8, :center))],
            subplot = 2)
    end

    # Panel 3: τ histogram with CI
    if !isempty(boot_result.tau_stats)
        StatsPlots.histogram!(plt, boot_result.tau_stats,
            bins = 30, color = :steelblue, alpha = 0.7, label = "",
            xlabel = "Mean pairwise τ", ylabel = "Count",
            title = "Dependence Strength (τ)", subplot = 3)
        lo, hi = boot_result.tau_ci
        StatsPlots.vline!(plt, [lo, hi], ls = :dash, color = :red, label = "95% CI", subplot = 3)
    end

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end


# ─── 5. Discordant Proteins — Copula vs Marginal Decomposition ───────────────

"""
    discordant_protein_analysis(bf::BayesFactorTriplet, lc_result::LatentClassResult; top_n=50)

Decompose each protein's combined log(BF) into marginal and model contributions.
"Discordant" = marginal contribution < 0 but combined BF > 1.
Uses LatentClassResult for combined BFs on the log-BF scale.
"""
function discordant_protein_analysis(bf::BayesFactorTriplet,
                                     lc_result::LatentClassResult;
                                     top_n::Int = 50)
    n = length(bf.enrichment)
    combined_log_bf = log.(max.(lc_result.bf, 1e-300))

    log_bf_e = log.(max.(bf.enrichment, 1e-300))
    log_bf_c = log.(max.(bf.correlation, 1e-300))
    log_bf_d = log.(max.(bf.detection, 1e-300))
    marginal_contrib = log_bf_e .+ log_bf_c .+ log_bf_d

    copula_contrib = combined_log_bf .- marginal_contrib

    discordant = (marginal_contrib .< 0) .& (combined_log_bf .> 0)
    disc_idx = findall(discordant)

    return (
        marginal_contrib   = marginal_contrib,
        copula_contrib     = copula_contrib,
        combined_log_bf    = combined_log_bf,
        discordant_indices = disc_idx,
        discordant_fraction = length(disc_idx) / n,
        n_discordant       = length(disc_idx),
    )
end

# Deprecated: old signature
function discordant_protein_analysis(bf::BayesFactorTriplet,
                                     combined_result::CombinedBayesResult,
                                     p::PosteriorProbabilityTriplet)
    n = length(bf.enrichment)
    combined_log_bf = log.(max.(combined_result.bf, 1e-300))

    log_bf_e = log.(max.(bf.enrichment, 1e-300))
    log_bf_c = log.(max.(bf.correlation, 1e-300))
    log_bf_d = log.(max.(bf.detection, 1e-300))
    marginal_contrib = log_bf_e .+ log_bf_c .+ log_bf_d

    copula_contrib = combined_log_bf .- marginal_contrib

    discordant = (marginal_contrib .< 0) .& (combined_log_bf .> 0)
    disc_idx = findall(discordant)

    return (
        marginal_contrib   = marginal_contrib,
        copula_contrib     = copula_contrib,
        combined_log_bf    = combined_log_bf,
        discordant_indices = disc_idx,
        discordant_fraction = length(disc_idx) / n,
        n_discordant       = length(disc_idx),
    )
end

"""
    discordant_protein_plot(disc_result; file="")

2-panel: (1) marginal vs copula contribution scatter (discordant=red),
(2) copula contribution histogram for discordant proteins.
"""
function discordant_protein_plot(disc_result; file::String = "")
    plt = StatsPlots.plot(layout = (1, 2), size = (1000, 450),
        plot_title = "Discordant Protein Decomposition ($(disc_result.n_discordant) discordant, $(round(100*disc_result.discordant_fraction, digits=1))%)",
        left_margin = 10 * StatsPlots.Plots.mm,
        bottom_margin = 8 * StatsPlots.Plots.mm)

    disc = disc_result.discordant_indices
    n = length(disc_result.marginal_contrib)
    rest = setdiff(1:n, disc)

    # Panel 1: marginal vs copula scatter
    StatsPlots.scatter!(plt, disc_result.marginal_contrib[rest], disc_result.copula_contrib[rest],
        alpha = 0.2, ms = 2, color = :steelblue, label = "Concordant",
        xlabel = "Marginal contribution (Σlog BF_i)",
        ylabel = "Copula contribution",
        title = "Marginal vs Copula Decomposition", subplot = 1)
    if !isempty(disc)
        StatsPlots.scatter!(plt, disc_result.marginal_contrib[disc], disc_result.copula_contrib[disc],
            alpha = 0.6, ms = 3, color = :red, label = "Discordant", subplot = 1)
    end
    StatsPlots.hline!(plt, [0], ls = :dash, color = :gray, label = "", subplot = 1)
    StatsPlots.vline!(plt, [0], ls = :dash, color = :gray, label = "", subplot = 1)

    # Panel 2: copula contribution histogram for discordant
    if !isempty(disc)
        StatsPlots.histogram!(plt, disc_result.copula_contrib[disc],
            bins = min(30, length(disc)),
            color = :red, alpha = 0.7, label = "",
            xlabel = "Copula contribution", ylabel = "Count",
            title = "Copula Contribution (Discordant)", subplot = 2)
    else
        StatsPlots.annotate!(plt,
            [(0.5, 0.5, StatsPlots.Plots.text("No discordant proteins", 12, :center))],
            subplot = 2)
    end

    !isempty(file) && StatsPlots.savefig(plt, file)
    return plt
end
