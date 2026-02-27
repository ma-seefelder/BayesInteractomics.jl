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
    @test length(cfg.bb_priors) == 5
    @test (3.0, 3.0) in cfg.bb_priors
    @test length(cfg.em_prior_grid) == 3
    @test length(cfg.lc_alpha_prior_grid) == 3
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

    n_proteins = 5
    n_settings = 3
    protein_names = ["P$i" for i in 1:n_proteins]
    posterior_matrix = rand(n_proteins, n_settings)
    bf_matrix = rand(n_proteins, n_settings) .* 10
    q_matrix = rand(n_proteins, n_settings)

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
        frac_q_lt_0_05 = fill(0.5, n_proteins),
        frac_q_lt_0_01 = fill(0.0, n_proteins)
    )

    sr = SensitivityResult(
        SensitivityConfig(),
        prior_settings,
        posterior_matrix,
        bf_matrix,
        q_matrix,
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
        [2], [3], [1]
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

    # 3 proteins, 4 settings
    posterior_matrix = [
        0.99 0.98 0.97 0.96;   # Always high
        0.1  0.2  0.3  0.4;    # Always low
        0.6  0.85 0.45 0.92    # Mixed
    ]
    q_matrix = [
        0.001 0.002 0.003 0.005;
        0.9   0.8   0.7   0.6;
        0.04  0.02  0.1   0.008
    ]
    protein_names = ["High", "Low", "Mixed"]

    stability = _compute_classification_stability(posterior_matrix, q_matrix, protein_names)

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

    # q-value checks for "High" protein
    @test stability.frac_q_lt_0_05[1] == 1.0
    @test stability.frac_q_lt_0_01[1] == 0.75  # 3 of 4 below 0.01
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
    q_matrix = rand(n_proteins, n_settings)

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
        frac_q_lt_0_05 = rand(n_proteins),
        frac_q_lt_0_01 = rand(n_proteins)
    )

    sr = SensitivityResult(
        SensitivityConfig(n_top_proteins=5),
        prior_settings,
        posterior_matrix,
        bf_matrix,
        q_matrix,
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
    q_matrix = rand(n_proteins, n_settings)

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
        frac_q_lt_0_05 = rand(n_proteins),
        frac_q_lt_0_01 = rand(n_proteins)
    )

    sr = SensitivityResult(
        SensitivityConfig(n_top_proteins=3),
        prior_settings,
        posterior_matrix,
        bf_matrix,
        q_matrix,
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
        [2], [3], [1]
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

@testitem "sensitivity_tornado_plot returns plot and saves file" begin
    using BayesInteractomics
    using DataFrames
    using Dates
    using Statistics
    using StatsPlots

    n_proteins = 10
    n_settings = 5
    protein_names = ["Protein$i" for i in 1:n_proteins]
    posterior_matrix = rand(n_proteins, n_settings)
    bf_matrix = rand(n_proteins, n_settings) .* 10
    q_matrix = rand(n_proteins, n_settings)

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
        frac_q_lt_0_05 = rand(n_proteins),
        frac_q_lt_0_01 = rand(n_proteins)
    )

    sr = SensitivityResult(
        SensitivityConfig(n_top_proteins=5),
        prior_settings,
        posterior_matrix,
        bf_matrix,
        q_matrix,
        protein_names,
        3,
        summary_df,
        stability_df,
        now()
    )

    # Test returns a plot
    plt = sensitivity_tornado_plot(sr; n_top=5)
    @test plt isa StatsPlots.Plots.Plot

    # Test saves to file
    mktempdir() do tmpdir
        filepath = joinpath(tmpdir, "tornado.png")
        plt2 = sensitivity_tornado_plot(sr; n_top=5, file=filepath)
        @test isfile(filepath)
        @test plt2 isa StatsPlots.Plots.Plot
    end
end

@testitem "sensitivity_heatmap returns plot and saves file" begin
    using BayesInteractomics
    using DataFrames
    using Dates
    using Statistics
    using StatsPlots

    n_proteins = 8
    n_settings = 4
    protein_names = ["P$i" for i in 1:n_proteins]
    posterior_matrix = rand(n_proteins, n_settings)
    bf_matrix = rand(n_proteins, n_settings) .* 10
    q_matrix = rand(n_proteins, n_settings)

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
        frac_q_lt_0_05 = rand(n_proteins),
        frac_q_lt_0_01 = rand(n_proteins)
    )

    sr = SensitivityResult(
        SensitivityConfig(n_top_proteins=5),
        prior_settings,
        posterior_matrix,
        bf_matrix,
        q_matrix,
        protein_names,
        2,
        summary_df,
        stability_df,
        now()
    )

    plt = sensitivity_heatmap(sr; n_top=5)
    @test plt isa StatsPlots.Plots.Plot

    mktempdir() do tmpdir
        filepath = joinpath(tmpdir, "heatmap.png")
        plt2 = sensitivity_heatmap(sr; file=filepath)
        @test isfile(filepath)
        @test plt2 isa StatsPlots.Plots.Plot
    end
end

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
    q_matrix = rand(n_proteins, n_settings)

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
        frac_q_lt_0_05 = rand(n_proteins),
        frac_q_lt_0_01 = rand(n_proteins)
    )

    sr = SensitivityResult(
        SensitivityConfig(n_top_proteins=5),
        prior_settings,
        posterior_matrix,
        bf_matrix,
        q_matrix,
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

@testitem "generate_sensitivity_report embeds images when file paths provided" begin
    using BayesInteractomics
    using DataFrames
    using Dates
    using Statistics

    n_proteins = 5
    n_settings = 3
    protein_names = ["P$i" for i in 1:n_proteins]
    posterior_matrix = rand(n_proteins, n_settings)
    bf_matrix = rand(n_proteins, n_settings) .* 10
    q_matrix = rand(n_proteins, n_settings)

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
        frac_q_lt_0_05 = rand(n_proteins),
        frac_q_lt_0_01 = rand(n_proteins)
    )

    sr = SensitivityResult(
        SensitivityConfig(n_top_proteins=3),
        prior_settings,
        posterior_matrix,
        bf_matrix,
        q_matrix,
        protein_names,
        2,
        summary_df,
        stability_df,
        now()
    )

    mktempdir() do tmpdir
        # Create dummy image files
        tornado_file = joinpath(tmpdir, "tornado.png")
        heatmap_file = joinpath(tmpdir, "heatmap.png")
        rankcorr_file = joinpath(tmpdir, "rankcorr.png")
        write(tornado_file, "dummy")
        write(heatmap_file, "dummy")
        write(rankcorr_file, "dummy")

        report_file = joinpath(tmpdir, "report.md")
        _, content = generate_sensitivity_report(sr;
            filename = report_file,
            tornado_file = tornado_file,
            heatmap_file = heatmap_file,
            rankcorr_file = rankcorr_file
        )

        # Should contain image references
        @test occursin("![", content)
        @test occursin("tornado.png", content)
        @test occursin("heatmap.png", content)
        @test occursin("rankcorr.png", content)
        @test occursin("## Posterior Divergence Across Prior Settings", content)
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
    q_matrix = rand(n_proteins, n_settings)

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
        frac_q_lt_0_05 = rand(n_proteins),
        frac_q_lt_0_01 = rand(n_proteins)
    )

    sr = SensitivityResult(
        SensitivityConfig(n_top_proteins=3),
        prior_settings,
        posterior_matrix,
        bf_matrix,
        q_matrix,
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
