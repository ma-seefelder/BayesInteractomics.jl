"""
    test_sensitivity.jl

Tests for the prior sensitivity analysis framework.
"""

# ============================================================================ #
# Unit tests — Type construction
# ============================================================================ #

@testitem "SensitivityConfig defaults" begin
    using BayesInteractomics

    cfg = SensitivityConfig()
    @test isempty(cfg.bb_priors)  # BB sweep disabled by default (pipeline uses fixed prior)
    @test length(cfg.em_prior_grid) == 3
    # 4 three-element Dirichlet vectors for 3-component model
    @test length(cfg.lc_alpha_prior_grid) == 4
    @test all(length(v) == 3 for v in cfg.lc_alpha_prior_grid)
    @test cfg.n_top_proteins == 20
end

@testitem "SensitivityConfig custom" begin
    using BayesInteractomics

    cfg = SensitivityConfig(
        bb_priors = [(1.0, 1.0), (5.0, 5.0)],
        n_top_proteins = 10
    )
    @test length(cfg.bb_priors) == 2
    @test cfg.n_top_proteins == 10
end

@testitem "PriorSetting construction" begin
    using BayesInteractomics

    ps = PriorSetting(:betabernoulli, "BB(3.0,3.0)", (α=3.0, β=3.0))
    @test ps.model == :betabernoulli
    @test ps.label == "BB(3.0,3.0)"
    @test ps.params.α == 3.0
end

@testitem "SensitivityResult construction" begin
    using BayesInteractomics
    using DataFrames
    using Dates
    using Statistics

    n_proteins = 5
    n_settings = 3
    protein_names = ["P$i" for i in 1:n_proteins]
    posterior_matrix = rand(n_proteins, n_settings)
    bf_matrix = rand(n_proteins, n_settings) .* 10
    bfdr_matrix = rand(n_proteins, n_settings)

    prior_settings = [
        PriorSetting(:betabernoulli, "BB(1,1)", (α=1.0, β=1.0)),
        PriorSetting(:betabernoulli, "BB(3,3)", (α=3.0, β=3.0)),
        PriorSetting(:betabernoulli, "BB(5,5)", (α=5.0, β=5.0))
    ]

    summary_df = DataFrame(
        Protein = protein_names,
        baseline_posterior = posterior_matrix[:, 2],
        mean_posterior = vec(mean(posterior_matrix, dims=2)),
        std_posterior = vec(std(posterior_matrix, dims=2)),
        min_posterior = vec(minimum(posterior_matrix, dims=2)),
        max_posterior = vec(maximum(posterior_matrix, dims=2)),
        range = vec(maximum(posterior_matrix, dims=2) .- minimum(posterior_matrix, dims=2))
    )

    stability_df = DataFrame(
        Protein = protein_names,
        frac_P_gt_0_5 = fill(1.0, n_proteins),
        frac_P_gt_0_8 = fill(0.5, n_proteins),
        frac_P_gt_0_95 = fill(0.0, n_proteins),
        frac_BFDR_lt_0_05 = fill(0.5, n_proteins),
        frac_BFDR_lt_0_01 = fill(0.0, n_proteins)
    )

    sr = SensitivityResult(
        SensitivityConfig(),
        prior_settings,
        posterior_matrix,
        bf_matrix,
        bfdr_matrix,
        protein_names,
        2,  # baseline_index
        summary_df,
        stability_df,
        now()
    )

    @test sr.baseline_index == 2
    @test size(sr.posterior_matrix) == (n_proteins, n_settings)
    @test length(sr.protein_names) == n_proteins
    @test nrow(sr.summary) == n_proteins
    @test nrow(sr.classification_stability) == n_proteins
end

# ============================================================================ #
# Unit tests — betabernoulli with custom priors
# ============================================================================ #

@testitem "betabernoulli with custom prior parameters" begin
    using BayesInteractomics
    using BayesInteractomics: betabernoulli, Protocol, InteractionData

    # Create data where protein is detected in samples but not controls
    m_sample::Matrix{Union{Missing, Float64}} = reshape(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, missing], 1, 9
    )
    m_control::Matrix{Union{Missing, Float64}} = reshape(
        [missing, missing, missing, missing, 5.0, 6.0], 1, 6
    )

    p_sample = Protocol(1, ["P1"], Dict(1 => m_sample))
    p_control = Protocol(1, ["P1"], Dict(1 => m_control))

    data = InteractionData(
        ["P1"], ["Protein1"],
        Dict(1 => p_sample),
        Dict(1 => p_control),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1],
        trues(1)
    )

    # Default prior (3,3)
    bf_default, p_default, _ = betabernoulli(data, 1, 6, 9)

    # Flat prior (1,1)
    bf_flat, p_flat, _ = betabernoulli(data, 1, 6, 9; prior_alpha=1.0, prior_beta=1.0)

    # Informative prior (10,10)
    bf_info, p_info, _ = betabernoulli(data, 1, 6, 9; prior_alpha=10.0, prior_beta=10.0)

    # All should indicate enrichment (BF > 1) but with different magnitudes
    @test bf_default > 1.0
    @test bf_flat > 1.0
    @test bf_info > 1.0

    # Flat prior should give more extreme BF than informative prior
    # (informative prior pulls toward 0.5, reducing difference)
    @test bf_flat > bf_info

    # All posteriors should be > 0.5
    @test p_default > 0.5
    @test p_flat > 0.5
    @test p_info > 0.5
end

# ============================================================================ #
# Unit tests — Internal helpers
# ============================================================================ #

@testitem "_compute_sensitivity_summary" begin
    using BayesInteractomics
    using BayesInteractomics: _compute_sensitivity_summary
    using DataFrames
    using Statistics

    # 3 proteins, 4 settings
    posterior_matrix = [
        0.9 0.85 0.95 0.88;
        0.1 0.15 0.12 0.08;
        0.5 0.6  0.55 0.45
    ]
    protein_names = ["P1", "P2", "P3"]
    baseline_idx = 1

    summary = _compute_sensitivity_summary(posterior_matrix, protein_names, baseline_idx)

    @test nrow(summary) == 3
    @test summary.Protein == protein_names
    @test summary.baseline_posterior == [0.9, 0.1, 0.5]

    # Check range computation
    @test summary.range[1] ≈ 0.95 - 0.85  # max - min for P1
    @test summary.range[2] ≈ 0.15 - 0.08  # max - min for P2
    @test summary.range[3] ≈ 0.6 - 0.45   # max - min for P3

    # Check mean
    @test summary.mean_posterior[1] ≈ mean([0.9, 0.85, 0.95, 0.88])
end

@testitem "_compute_classification_stability" begin
    using BayesInteractomics
    using BayesInteractomics: _compute_classification_stability
    using DataFrames

    # 3 proteins, 4 settings
    posterior_matrix = [
        0.99 0.98 0.97 0.96;   # Always high
        0.1  0.2  0.3  0.4;    # Always low
        0.6  0.85 0.45 0.92    # Mixed
    ]
    bfdr_matrix = [
        0.001 0.002 0.003 0.005;
        0.9   0.8   0.7   0.6;
        0.04  0.02  0.1   0.008
    ]
    protein_names = ["High", "Low", "Mixed"]

    stability = _compute_classification_stability(posterior_matrix, bfdr_matrix, protein_names)

    @test nrow(stability) == 3
    @test stability.Protein == protein_names

    # "High" protein: all > 0.95, all > 0.8, all > 0.5
    @test stability.frac_P_gt_0_95[1] == 1.0
    @test stability.frac_P_gt_0_8[1] == 1.0
    @test stability.frac_P_gt_0_5[1] == 1.0

    # "Low" protein: none above any threshold
    @test stability.frac_P_gt_0_5[2] == 0.0
    @test stability.frac_P_gt_0_8[2] == 0.0
    @test stability.frac_P_gt_0_95[2] == 0.0

    # "Mixed" protein: some above 0.5, some above 0.8
    @test 0.0 < stability.frac_P_gt_0_5[3] < 1.0
    @test 0.0 < stability.frac_P_gt_0_8[3] < 1.0

    # BFDR checks for "High" protein
    @test stability.frac_BFDR_lt_0_05[1] == 1.0
    @test stability.frac_BFDR_lt_0_01[1] == 1.0  # all 4 below 0.01 (0.001, 0.002, 0.003, 0.005)
end

# ============================================================================ #
# Unit tests — Report generation
# ============================================================================ #

@testitem "generate_sensitivity_report produces valid Markdown" begin
    using BayesInteractomics
    using DataFrames
    using Dates
    using Statistics

    n_proteins = 10
    n_settings = 5
    protein_names = ["Protein$i" for i in 1:n_proteins]
    posterior_matrix = rand(n_proteins, n_settings)
    bf_matrix = rand(n_proteins, n_settings) .* 10
    bfdr_matrix = rand(n_proteins, n_settings)

    prior_settings = [
        PriorSetting(:betabernoulli, "BB(1,1)", (α=1.0, β=1.0)),
        PriorSetting(:betabernoulli, "BB(2,2)", (α=2.0, β=2.0)),
        PriorSetting(:betabernoulli, "BB(3,3)", (α=3.0, β=3.0)),
        PriorSetting(:betabernoulli, "BB(5,5)", (α=5.0, β=5.0)),
        PriorSetting(:betabernoulli, "BB(10,10)", (α=10.0, β=10.0))
    ]

    summary_df = DataFrame(
        Protein = protein_names,
        baseline_posterior = posterior_matrix[:, 3],
        mean_posterior = vec(mean(posterior_matrix, dims=2)),
        std_posterior = vec(std(posterior_matrix, dims=2)),
        min_posterior = vec(minimum(posterior_matrix, dims=2)),
        max_posterior = vec(maximum(posterior_matrix, dims=2)),
        range = vec(maximum(posterior_matrix, dims=2) .- minimum(posterior_matrix, dims=2))
    )

    stability_df = DataFrame(
        Protein = protein_names,
        frac_P_gt_0_5 = rand(n_proteins),
        frac_P_gt_0_8 = rand(n_proteins),
        frac_P_gt_0_95 = rand(n_proteins),
        frac_BFDR_lt_0_05 = rand(n_proteins),
        frac_BFDR_lt_0_01 = rand(n_proteins)
    )

    sr = SensitivityResult(
        SensitivityConfig(n_top_proteins=5),
        prior_settings,
        posterior_matrix,
        bf_matrix,
        bfdr_matrix,
        protein_names,
        3,  # baseline_index (BB(3,3))
        summary_df,
        stability_df,
        now()
    )

    mktempdir() do tmpdir
        filepath = joinpath(tmpdir, "test_report.md")
        result_path, content = generate_sensitivity_report(sr; filename=filepath)

        @test isfile(result_path)
        @test !isempty(content)

        # Check expected sections
        @test occursin("# Prior Sensitivity Analysis Report", content)
        @test occursin("## Summary", content)
        @test occursin("## Global Robustness", content)
        @test occursin("## Classification Stability", content)
        @test occursin("## Most Sensitive Proteins", content)
        @test occursin("## Prior Settings Used", content)
        @test occursin("### Beta-Bernoulli Priors", content)

        # Check table content
        @test occursin("Proteins analyzed", content)
        @test occursin("BB(3,3)", content)

        # Verify it's valid Markdown (has proper table separators)
        @test occursin("|--------|", content)
    end
end

@testitem "generate_sensitivity_report with mixed model types" begin
    using BayesInteractomics
    using DataFrames
    using Dates
    using Statistics

    n_proteins = 5
    n_settings = 3
    protein_names = ["P$i" for i in 1:n_proteins]
    posterior_matrix = rand(n_proteins, n_settings)
    bf_matrix = rand(n_proteins, n_settings) .* 10
    bfdr_matrix = rand(n_proteins, n_settings)

    prior_settings = [
        PriorSetting(:betabernoulli, "BB(3,3)", (α=3.0, β=3.0)),
        PriorSetting(:copula_em, "EM(α=25,β=175)", (α=25.0, β=175.0)),
        PriorSetting(:latent_class, "LC(α=[10,1])", (alpha_prior=[10.0, 1.0],))
    ]

    summary_df = DataFrame(
        Protein = protein_names,
        baseline_posterior = posterior_matrix[:, 1],
        mean_posterior = vec(mean(posterior_matrix, dims=2)),
        std_posterior = vec(std(posterior_matrix, dims=2)),
        min_posterior = vec(minimum(posterior_matrix, dims=2)),
        max_posterior = vec(maximum(posterior_matrix, dims=2)),
        range = vec(maximum(posterior_matrix, dims=2) .- minimum(posterior_matrix, dims=2))
    )

    stability_df = DataFrame(
        Protein = protein_names,
        frac_P_gt_0_5 = rand(n_proteins),
        frac_P_gt_0_8 = rand(n_proteins),
        frac_P_gt_0_95 = rand(n_proteins),
        frac_BFDR_lt_0_05 = rand(n_proteins),
        frac_BFDR_lt_0_01 = rand(n_proteins)
    )

    sr = SensitivityResult(
        SensitivityConfig(n_top_proteins=3),
        prior_settings,
        posterior_matrix,
        bf_matrix,
        bfdr_matrix,
        protein_names,
        1,
        summary_df,
        stability_df,
        now()
    )

    mktempdir() do tmpdir
        filepath = joinpath(tmpdir, "mixed_report.md")
        _, content = generate_sensitivity_report(sr; filename=filepath)

        # Should have sections for all model types
        @test occursin("### Beta-Bernoulli Priors", content)
        @test occursin("### Copula-EM Priors", content)
        @test occursin("### Latent Class Priors", content)
    end
end

# ============================================================================ #
# Unit tests — _recompute_bb_bf
# ============================================================================ #

@testitem "_recompute_bb_bf produces valid BFs" begin
    using BayesInteractomics
    using BayesInteractomics: _recompute_bb_bf, Protocol, InteractionData

    # Create data with 3 proteins
    m_sample::Matrix{Union{Missing, Float64}} = [
        1.0 2.0 3.0;
        missing missing 6.0;
        7.0 8.0 9.0
    ]
    m_control::Matrix{Union{Missing, Float64}} = [
        missing missing;
        4.0 5.0;
        missing missing
    ]

    p_sample = Protocol(1, ["P1", "P2", "P3"], Dict(1 => m_sample))
    p_control = Protocol(1, ["P1", "P2", "P3"], Dict(1 => m_control))

    data = InteractionData(
        ["P1", "P2", "P3"], ["Protein1", "Protein2", "Protein3"],
        Dict(1 => p_sample),
        Dict(1 => p_control),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1],
        trues(3)
    )

    bf_default = _recompute_bb_bf(data, 2, 3; prior_alpha=3.0, prior_beta=3.0)
    bf_flat = _recompute_bb_bf(data, 2, 3; prior_alpha=1.0, prior_beta=1.0)

    @test length(bf_default) == 3
    @test length(bf_flat) == 3
    @test all(bf_default .>= 0.0)
    @test all(bf_flat .>= 0.0)

    # Different priors should give different BFs (at least for some proteins)
    @test bf_default != bf_flat
end

# ============================================================================ #
# Unit tests — Sensitivity plots
# ============================================================================ #

@testitem "sensitivity_rank_correlation returns plot and saves file" begin
    using BayesInteractomics
    using DataFrames
    using Dates
    using Statistics
    using StatsPlots

    n_proteins = 10
    n_settings = 4
    protein_names = ["P$i" for i in 1:n_proteins]
    posterior_matrix = rand(n_proteins, n_settings)
    bf_matrix = rand(n_proteins, n_settings) .* 10
    bfdr_matrix = rand(n_proteins, n_settings)

    prior_settings = [
        PriorSetting(:betabernoulli, "BB(1,1)", (α=1.0, β=1.0)),
        PriorSetting(:betabernoulli, "BB(3,3)", (α=3.0, β=3.0)),
        PriorSetting(:betabernoulli, "BB(5,5)", (α=5.0, β=5.0)),
        PriorSetting(:betabernoulli, "BB(10,10)", (α=10.0, β=10.0))
    ]

    summary_df = DataFrame(
        Protein = protein_names,
        baseline_posterior = posterior_matrix[:, 2],
        mean_posterior = vec(mean(posterior_matrix, dims=2)),
        std_posterior = vec(std(posterior_matrix, dims=2)),
        min_posterior = vec(minimum(posterior_matrix, dims=2)),
        max_posterior = vec(maximum(posterior_matrix, dims=2)),
        range = vec(maximum(posterior_matrix, dims=2) .- minimum(posterior_matrix, dims=2))
    )

    stability_df = DataFrame(
        Protein = protein_names,
        frac_P_gt_0_5 = rand(n_proteins),
        frac_P_gt_0_8 = rand(n_proteins),
        frac_P_gt_0_95 = rand(n_proteins),
        frac_BFDR_lt_0_05 = rand(n_proteins),
        frac_BFDR_lt_0_01 = rand(n_proteins)
    )

    sr = SensitivityResult(
        SensitivityConfig(n_top_proteins=5),
        prior_settings,
        posterior_matrix,
        bf_matrix,
        bfdr_matrix,
        protein_names,
        2,
        summary_df,
        stability_df,
        now()
    )

    plt = sensitivity_rank_correlation(sr)
    @test plt isa StatsPlots.Plots.Plot

    mktempdir() do tmpdir
        filepath = joinpath(tmpdir, "rankcorr.png")
        plt2 = sensitivity_rank_correlation(sr; file=filepath)
        @test isfile(filepath)
        @test plt2 isa StatsPlots.Plots.Plot
    end
end

@testitem "generate_sensitivity_report without file kwargs has no images" begin
    using BayesInteractomics
    using DataFrames
    using Dates
    using Statistics

    n_proteins = 5
    n_settings = 3
    protein_names = ["P$i" for i in 1:n_proteins]
    posterior_matrix = rand(n_proteins, n_settings)
    bf_matrix = rand(n_proteins, n_settings) .* 10
    bfdr_matrix = rand(n_proteins, n_settings)

    prior_settings = [
        PriorSetting(:betabernoulli, "BB(1,1)", (α=1.0, β=1.0)),
        PriorSetting(:betabernoulli, "BB(3,3)", (α=3.0, β=3.0)),
        PriorSetting(:betabernoulli, "BB(5,5)", (α=5.0, β=5.0))
    ]

    summary_df = DataFrame(
        Protein = protein_names,
        baseline_posterior = posterior_matrix[:, 2],
        mean_posterior = vec(mean(posterior_matrix, dims=2)),
        std_posterior = vec(std(posterior_matrix, dims=2)),
        min_posterior = vec(minimum(posterior_matrix, dims=2)),
        max_posterior = vec(maximum(posterior_matrix, dims=2)),
        range = vec(maximum(posterior_matrix, dims=2) .- minimum(posterior_matrix, dims=2))
    )

    stability_df = DataFrame(
        Protein = protein_names,
        frac_P_gt_0_5 = rand(n_proteins),
        frac_P_gt_0_8 = rand(n_proteins),
        frac_P_gt_0_95 = rand(n_proteins),
        frac_BFDR_lt_0_05 = rand(n_proteins),
        frac_BFDR_lt_0_01 = rand(n_proteins)
    )

    sr = SensitivityResult(
        SensitivityConfig(n_top_proteins=3),
        prior_settings,
        posterior_matrix,
        bf_matrix,
        bfdr_matrix,
        protein_names,
        2,
        summary_df,
        stability_df,
        now()
    )

    mktempdir() do tmpdir
        report_file = joinpath(tmpdir, "report.md")
        _, content = generate_sensitivity_report(sr; filename=report_file)

        # Should NOT contain image references
        @test !occursin("![", content)
        @test !occursin("## Posterior Divergence Across Prior Settings", content)

        # Should still contain text sections
        @test occursin("## Summary", content)
        @test occursin("## Global Robustness", content)
    end
end

# ============================================================================ #
# Unit tests — Classification stability threshold crossings
# ============================================================================ #

@testitem "Classification stability threshold crossings" begin
    using BayesInteractomics
    using BayesInteractomics: _compute_classification_stability

    # 4 proteins, 3 settings
    posterior_matrix = [
        0.99  0.97  0.96;    # Always above 0.95 -> no crossing at 0.95
        0.94  0.96  0.93;    # Crosses 0.95 boundary (above & below)
        0.60  0.40  0.55;    # Crosses 0.5 boundary (above & below)
        0.30  0.20  0.10     # Always below 0.5 -> no crossing at 0.5
    ]
    bfdr_matrix = ones(4, 3) .* 0.5  # BFDR values don't matter for this test

    protein_names = ["StableHigh", "Crossing95", "Crossing50", "StableLow"]

    stability = _compute_classification_stability(posterior_matrix, bfdr_matrix, protein_names)

    # threshold_crossing_0_95: true if protein crosses P=0.95 boundary
    @test hasproperty(stability, :threshold_crossing_0_95)
    @test stability.threshold_crossing_0_95[1] == false   # StableHigh: always >= 0.95
    @test stability.threshold_crossing_0_95[2] == true    # Crossing95: above & below 0.95
    @test stability.threshold_crossing_0_95[3] == false   # Crossing50: all < 0.95
    @test stability.threshold_crossing_0_95[4] == false   # StableLow: all < 0.95

    # threshold_crossing_0_5: true if protein crosses P=0.5 boundary
    @test hasproperty(stability, :threshold_crossing_0_5)
    @test stability.threshold_crossing_0_5[1] == false    # StableHigh: always > 0.5
    @test stability.threshold_crossing_0_5[2] == false    # Crossing95: always > 0.5
    @test stability.threshold_crossing_0_5[3] == true     # Crossing50: above & below 0.5
    @test stability.threshold_crossing_0_5[4] == false    # StableLow: always < 0.5
end

# ============================================================================ #
# Unit tests — ValidationResult construction
# ============================================================================ #

@testitem "sensitivity _recombine_evidence respects n_restarts kwarg" begin
    using BayesInteractomics
    using BayesInteractomics: _recombine_evidence
    using Random

    # Create synthetic BF vectors (small, fast to test)
    n = 20
    rng = MersenneTwister(42)
    bf_e = exp.(randn(rng, n) .* 0.5)
    bf_c = exp.(randn(rng, n) .* 0.3)
    bf_d = exp.(randn(rng, n) .* 0.4)
    refID = 1

    # Verify _recombine_evidence works with explicit n_restarts for latent_class mode
    bf, posterior, bfdr_vals = _recombine_evidence(
        bf_e, bf_c, bf_d, refID;
        combination_method = :latent_class,
        n_restarts = 3,
        lc_n_iterations = 50,
        verbose = false
    )

    @test length(bf) == n
    @test length(posterior) == n
    @test length(bfdr_vals) == n
    @test all(isfinite, posterior)
    @test all(p -> 0.0 <= p <= 1.0, posterior)
end

@testitem "ValidationResult construction" begin
    using BayesInteractomics
    using Dates
    using DataFrames

    # Test with all fields populated
    qg_cells = Matrix{QualityGateCell}(undef, 3, 3)
    for i in 1:3, j in 1:3
        marginals = [:enrichment, :correlation, :detection]
        components = [:H0, :agnostic, :H1]
        qg_cells[i, j] = QualityGateCell(
            marginals[i], components[j],
            0.05, :pass, nothing, 50.0, false
        )
    end
    qg = QualityGateResult(qg_cells, :pass, String[])

    kl = KLContaminationResult(0.1, 0.2, 0.15, 0.45, 10, true)

    crossings = DataFrame(Protein=["P1"], threshold_crossing_0_95=[true])

    consistency = Dict{String, Bool}(
        "all_ks_pass" => true,
        "kl_pass" => true,
        "h1_lt_200" => true,
        "F8A1_P1" => true,
    )

    vr = ValidationResult(qg, kl, crossings, consistency, true, now())

    @test vr.quality_gates === qg
    @test vr.kl_contamination === kl
    @test vr.sensitivity_crossings !== nothing
    @test vr.overall_pass == true
    @test vr.consistency_checks["all_ks_pass"] == true

    # Test with nothing fields
    vr2 = ValidationResult(nothing, nothing, nothing, Dict{String,Bool}(), true, now())
    @test vr2.quality_gates === nothing
    @test vr2.kl_contamination === nothing
    @test vr2.overall_pass == true

    # Test show method
    io = IOBuffer()
    show(io, vr)
    output = String(take!(io))
    @test occursin("ValidationResult", output)
    @test occursin("PASS", output)
end

# ============================================================================ #
# Unit tests — BMA sweep
# ============================================================================ #

@testitem "BMA _recombine_evidence produces genuine BMA posteriors" begin
    using BayesInteractomics
    using BayesInteractomics: _recombine_evidence
    using Random

    # Create synthetic BF vectors (small n for speed)
    n = 30
    rng = MersenneTwister(42)
    bf_e = exp.(randn(rng, n) .* 1.5)  # enrichment BFs
    bf_c = exp.(randn(rng, n) .* 0.8)  # correlation BFs
    bf_d = exp.(randn(rng, n) .* 0.5 .+ 0.5)  # detection BFs
    bf_e[1] = 100.0  # bait: strong enrichment

    # Run BMA recombine
    bf_bma, post_bma, bfdr_bma = _recombine_evidence(
        bf_e, bf_c, bf_d, 1;
        combination_method = :bma,
        verbose = false
    )

    # Verify outputs are valid
    @test length(bf_bma) == n
    @test length(post_bma) == n
    @test length(bfdr_bma) == n
    @test all(isfinite, post_bma)
    @test all(p -> 0.0 <= p <= 1.0, post_bma)
    @test all(isfinite, bf_bma)

    # Run copula-only for comparison -- BMA posteriors should differ
    bf_cop, post_cop, bfdr_cop = _recombine_evidence(
        bf_e, bf_c, bf_d, 1;
        combination_method = :copula,
        verbose = false
    )
    # BMA and copula posteriors should not be identical (BMA is weighted average)
    @test post_bma != post_cop
end

@testitem "BMA _recombine_evidence forwards em_prior to copula" begin
    using BayesInteractomics
    using BayesInteractomics: _recombine_evidence
    using Random

    n = 30
    rng = MersenneTwister(99)
    bf_e = exp.(randn(rng, n) .* 2.0)
    bf_c = exp.(randn(rng, n) .* 1.0)
    bf_d = exp.(randn(rng, n) .* 0.3 .+ 0.3)
    bf_e[1] = 200.0

    # Run BMA with default em_prior (should use [5,2,1] Dirichlet)
    bf_default, post_default, _ = _recombine_evidence(
        bf_e, bf_c, bf_d, 1;
        combination_method = :bma,
        verbose = false
    )

    # Run BMA with very different em_prior (strongly favoring H1)
    bf_h1, post_h1, _ = _recombine_evidence(
        bf_e, bf_c, bf_d, 1;
        combination_method = :bma,
        em_prior = (α=100.0, β=100.0),  # E[pi1]=0.5, much higher than default
        verbose = false
    )

    # Posteriors should differ because the copula Dirichlet prior changed
    @test all(isfinite, post_default)
    @test all(isfinite, post_h1)
    @test all(p -> 0.0 <= p <= 1.0, post_default)
    @test all(p -> 0.0 <= p <= 1.0, post_h1)
    # With such a strong H1 prior, at least some posteriors should shift
    @test post_default != post_h1
end

@testitem "BMA sensitivity grid is Cartesian product" begin
    using BayesInteractomics: SensitivityConfig

    config = SensitivityConfig()
    n_em = length(config.em_prior_grid)
    @test n_em == 3  # default: 3 copula EM settings

    # With 4-point fallback LC grid, total should be 4 * 3 = 12
    n_lc_fallback = length(config.lc_alpha_prior_grid)
    @test n_lc_fallback * n_em == 12  # 4 * 3 = 12 with fallback grid

    # Verify EM prior grid has expected structure
    for ep in config.em_prior_grid
        @test hasproperty(ep, :α)  # Greek letter field name
        @test hasproperty(ep, :β)
        @test ep.α > 0  # alpha positive
        @test ep.β > 0  # beta positive
    end
end

@testitem "BMA PriorSetting stores stacking weights" begin
    using BayesInteractomics: PriorSetting

    # Simulate what the BMA sweep creates
    ps = PriorSetting(:bma, "EB center | E[π₁]=0.05", (
        alpha_prior = [5.0, 2.0, 1.0],
        em_alpha = 10.0,
        em_beta = 190.0,
        w_em = 0.65,
        w_cop = 0.35,
    ))

    @test ps.model == :bma
    @test contains(ps.label, "EB center")
    @test contains(ps.label, "E[π₁]=")
    @test ps.params.w_em == 0.65
    @test ps.params.w_cop == 0.35
    @test ps.params.em_alpha == 10.0
    @test ps.params.em_beta == 190.0
    @test ps.params.alpha_prior == [5.0, 2.0, 1.0]
end
