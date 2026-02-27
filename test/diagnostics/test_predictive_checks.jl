"""
    test_predictive_checks.jl

Tests for the posterior predictive checks and model diagnostics framework.
"""

# ============================================================================ #
# Unit tests — Type construction
# ============================================================================ #

@testitem "DiagnosticsConfig defaults" begin
    using BayesInteractomics

    cfg = DiagnosticsConfig()
    @test cfg.n_ppc_draws == 1000
    @test cfg.n_proteins_to_check == 50
    @test cfg.ppc_protein_selection == :top_and_random
    @test cfg.residual_model == :both
    @test cfg.calibration_bins == 10
    @test cfg.n_top_display == 20
    @test cfg.seed == 42
    @test cfg.enhanced_residuals == true
end

@testitem "DiagnosticsConfig custom" begin
    using BayesInteractomics

    cfg = DiagnosticsConfig(
        n_ppc_draws = 500,
        n_proteins_to_check = 25,
        ppc_protein_selection = :random,
        calibration_bins = 5,
        seed = 123,
        enhanced_residuals = false
    )
    @test cfg.n_ppc_draws == 500
    @test cfg.n_proteins_to_check == 25
    @test cfg.ppc_protein_selection == :random
    @test cfg.calibration_bins == 5
    @test cfg.seed == 123
    @test cfg.enhanced_residuals == false
end

@testitem "ProteinPPC construction" begin
    using BayesInteractomics

    observed = randn(10)
    simulated = randn(10, 100)

    ppc = ProteinPPC("TestProtein", :hbm, observed, simulated, 0.45, 0.52, 0.48)
    @test ppc.protein_name == "TestProtein"
    @test ppc.model == :hbm
    @test length(ppc.observed) == 10
    @test size(ppc.simulated) == (10, 100)
    @test 0.0 <= ppc.pvalue_mean <= 1.0
    @test 0.0 <= ppc.pvalue_sd <= 1.0
    @test 0.0 <= ppc.pvalue_log2fc <= 1.0
end

@testitem "ProteinPPC regression" begin
    using BayesInteractomics

    observed = randn(15)
    simulated = randn(15, 200)

    ppc = ProteinPPC("RegProtein", :regression, observed, simulated, 0.5, 0.6, NaN)
    @test ppc.model == :regression
    @test isnan(ppc.pvalue_log2fc)
end

@testitem "BetaBernoulliPPC construction" begin
    using BayesInteractomics

    sim_s = rand(0:10, 500)
    sim_c = rand(0:8, 500)

    bb = BetaBernoulliPPC("BBProtein", 8, 3, sim_s, sim_c, 0.35)
    @test bb.protein_name == "BBProtein"
    @test bb.observed_k_sample == 8
    @test bb.observed_k_control == 3
    @test length(bb.simulated_k_sample) == 500
    @test length(bb.simulated_k_control) == 500
    @test 0.0 <= bb.pvalue_detection_diff <= 1.0
end

@testitem "EnhancedResidualResult construction" begin
    using BayesInteractomics

    base = ResidualResult(
        :hbm,
        ["P1", "P2"],
        [[0.1, -0.2], [0.3, 0.4]],
        [0.05, 0.35],
        [0.1, -0.2, 0.3, 0.4],
        [1.0, 1.0, 2.0, 2.0],
        0.1,
        -0.3,
        String[]
    )

    qresids = [[0.05, -0.1], [0.2, 0.3]]
    pit_vals = [0.45, 0.52, 0.61, 0.38]

    enh = EnhancedResidualResult(base, qresids, pit_vals)
    @test enh.base.model == :hbm
    @test length(enh.quantile_residuals) == 2
    @test length(enh.pit_values) == 4
    @test all(0.0 .<= enh.pit_values .<= 1.0)
end

@testitem "PPCExtendedStatistics construction" begin
    using BayesInteractomics

    ext = PPCExtendedStatistics("P1", :hbm, 0.45, 0.62, 0.33)
    @test ext.protein_name == "P1"
    @test ext.model == :hbm
    @test 0.0 <= ext.pvalue_skewness <= 1.0
    @test 0.0 <= ext.pvalue_kurtosis <= 1.0
    @test 0.0 <= ext.pvalue_iqr_ratio <= 1.0
end

@testitem "ProteinDiagnosticFlag construction" begin
    using BayesInteractomics

    flag = ProteinDiagnosticFlag("P1", 10, 0.5, 2.3, false, false, :ok)
    @test flag.protein_name == "P1"
    @test flag.n_observations == 10
    @test flag.overall_flag == :ok

    flag_warn = ProteinDiagnosticFlag("P2", 3, 1.5, 3.1, false, true, :warning)
    @test flag_warn.is_low_data == true
    @test flag_warn.overall_flag == :warning

    flag_fail = ProteinDiagnosticFlag("P3", 2, 2.5, 4.0, true, true, :fail)
    @test flag_fail.overall_flag == :fail
end

@testitem "DiagnosticsResult construction" begin
    using BayesInteractomics
    using DataFrames
    using Dates

    cfg = DiagnosticsConfig()

    # Build minimal protein PPCs
    ppc1 = ProteinPPC("P1", :hbm, randn(5), randn(5, 10), 0.4, 0.5, 0.6)
    ppc2 = ProteinPPC("P2", :regression, randn(5), randn(5, 10), 0.55, 0.45, NaN)

    bb1 = BetaBernoulliPPC("P1", 5, 2, rand(0:7, 100), rand(0:5, 100), 0.3)

    summary_df = DataFrame(
        Protein = ["P1", "P2", "P1"],
        model = [:hbm, :regression, :betabernoulli],
        pvalue_mean = [0.4, 0.55, 0.3],
        pvalue_sd = [0.5, 0.45, NaN],
        pvalue_log2fc = [0.6, NaN, NaN]
    )

    dr = DiagnosticsResult(
        cfg,
        [ppc1, ppc2],
        [bb1],
        nothing,       # hbm_residuals
        nothing,       # regression_residuals
        nothing,       # calibration
        nothing,       # calibration_relaxed
        nothing,       # calibration_enrichment_only
        nothing,       # enhanced_hbm_residuals
        nothing,       # enhanced_regression_residuals
        nothing,       # ppc_extended
        nothing,       # protein_flags
        nothing,       # model_comparison
        nothing,       # nu_optimization
        summary_df,
        now()
    )

    @test length(dr.protein_ppcs) == 2
    @test length(dr.bb_ppcs) == 1
    @test isnothing(dr.hbm_residuals)
    @test isnothing(dr.calibration)
    @test isnothing(dr.enhanced_hbm_residuals)
    @test isnothing(dr.ppc_extended)
    @test isnothing(dr.protein_flags)
    @test nrow(dr.summary) == 3
end

@testitem "Protein selection top_and_random" begin
    using BayesInteractomics
    using DataFrames
    using Random

    n = 100
    cr = DataFrame(
        Protein = ["P$i" for i in 1:n],
        BF = rand(n) .* 100,
        posterior_prob = rand(n),
        q = rand(n),
        mean_log2FC = randn(n),
        bf_enrichment = rand(n) .* 10,
        bf_correlation = rand(n) .* 5,
        bf_detected = rand(n) .* 20
    )

    cfg = DiagnosticsConfig(n_proteins_to_check=20)
    rng = MersenneTwister(42)

    selected = BayesInteractomics._select_proteins_for_ppc(cr, cfg, rng)
    @test length(selected) == 20
    @test length(unique(selected)) == 20  # No duplicates

    # Top 10 by BF should be included
    bf_order = sortperm(cr.BF, rev=true)
    top10 = bf_order[1:10]
    @test all(idx -> idx in selected, top10)
end

@testitem "Protein selection stratified" begin
    using BayesInteractomics
    using DataFrames
    using Random

    n = 100
    cr = DataFrame(
        Protein = ["P$i" for i in 1:n],
        BF = rand(n) .* 100,
        posterior_prob = collect(range(0.0, 1.0, length=n)),  # Evenly spaced for clear quartiles
        q = rand(n),
        mean_log2FC = randn(n),
        bf_enrichment = rand(n) .* 10,
        bf_correlation = rand(n) .* 5,
        bf_detected = rand(n) .* 20
    )

    cfg = DiagnosticsConfig(n_proteins_to_check=20, ppc_protein_selection=:stratified)
    rng = MersenneTwister(42)

    selected = BayesInteractomics._select_proteins_for_ppc(cr, cfg, rng)
    @test length(selected) >= 16  # At least 4 per quartile (might be slightly less due to unique)
    @test length(selected) <= 20
    @test length(unique(selected)) == length(selected)  # No duplicates

    # Check proteins come from different quartiles
    pp = cr.posterior_prob[selected]
    @test any(pp .< 0.25)  # Some from Q1
    @test any(pp .> 0.75)  # Some from Q4
end

@testitem "Protein selection respects data size" begin
    using BayesInteractomics
    using DataFrames
    using Random

    # Small dataset (fewer proteins than requested)
    n = 5
    cr = DataFrame(
        Protein = ["P$i" for i in 1:n],
        BF = rand(n) .* 100,
        posterior_prob = rand(n),
        q = rand(n),
        mean_log2FC = randn(n),
        bf_enrichment = rand(n),
        bf_correlation = rand(n),
        bf_detected = rand(n)
    )

    cfg = DiagnosticsConfig(n_proteins_to_check=50)
    rng = MersenneTwister(42)

    selected = BayesInteractomics._select_proteins_for_ppc(cr, cfg, rng)
    @test length(selected) == 5
end

@testitem "Diagnostics summary builder" begin
    using BayesInteractomics
    using DataFrames

    ppcs = [
        ProteinPPC("P1", :hbm, Float64[], zeros(0, 0), 0.3, 0.4, 0.5),
        ProteinPPC("P2", :regression, Float64[], zeros(0, 0), 0.6, 0.7, NaN)
    ]
    bbs = [
        BetaBernoulliPPC("P1", 5, 2, Int[], Int[], 0.25)
    ]

    df = BayesInteractomics._build_diagnostics_summary(ppcs, bbs)
    @test nrow(df) == 3
    @test "Protein" in names(df)
    @test "model" in names(df)
    @test "pvalue_mean" in names(df)
end

@testitem "DiagnosticsResult in AnalysisResult" begin
    using BayesInteractomics

    # Verify AnalysisResult has diagnostics field
    @test :diagnostics in fieldnames(AnalysisResult)
end

@testitem "_compute_protein_flags with data" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, getPositions, getIDs, getSampleMatrix, getProteinData

    # Construct InteractionData directly with controlled missing values
    n_proteins = 6
    protein_ids = ["P1", "P2", "P3", "P4", "P5", "P6"]
    protein_names_list = protein_ids

    # Single protocol, 1 experiment, 3 sample replicates, 3 control replicates
    # P1: 3 non-missing samples
    # P2: 1 non-missing sample (2 missing)
    # P3: 2 non-missing samples (1 missing)
    # P4: 3 non-missing samples
    # P5: 3 non-missing samples
    # P6: 5 non-missing samples (across 2 experiments)
    sample_mat1 = Matrix{Union{Missing, Float64}}([
        10.0  10.1  10.2;    # P1: 3 obs
        20.0  missing missing;# P2: 1 obs
        30.0  30.1  missing;  # P3: 2 obs
        40.0  40.1  40.2;    # P4: 3 obs
        50.0  50.1  50.2;    # P5: 3 obs
        60.0  60.1  60.2     # P6: 3 obs in exp1
    ])
    sample_mat2 = Matrix{Union{Missing, Float64}}([
        missing missing missing;  # P1: 0 extra
        missing missing missing;  # P2: 0 extra
        missing missing missing;  # P3: 0 extra
        missing missing missing;  # P4: 0 extra
        missing missing missing;  # P5: 0 extra
        61.0    61.1   missing    # P6: 2 extra obs in exp2
    ])

    control_mat = Matrix{Union{Missing, Float64}}(ones(n_proteins, 3) .+ 1.0)

    sample_proto = Protocol(2, protein_ids, Dict(1 => sample_mat1, 2 => sample_mat2))
    control_proto = Protocol(2, protein_ids, Dict(1 => control_mat, 2 => copy(control_mat)))

    no_experiments = Dict(1 => 2)
    no_parameters_HBM = 1 + 1 + 2  # intercept + 1 protocol + 2 experiments
    no_parameters_Regression = 1 + 1
    protocol_positions, experiment_positions, matched_positions =
        getPositions(no_experiments, no_parameters_HBM)

    data = InteractionData(
        protein_ids, protein_names_list,
        Dict(1 => sample_proto), Dict(1 => control_proto),
        1, no_experiments,
        no_parameters_HBM, no_parameters_Regression,
        protocol_positions, experiment_positions, matched_positions
    )

    all_ids = getIDs(data)
    name_to_idx = Dict(all_ids[i] => i for i in eachindex(all_ids))

    # Build a mock ResidualResult for PPC subset (only P1 and P4)
    ppc_res = ResidualResult(
        :hbm,
        ["P1", "P4"],
        [
            [0.1, -0.2, 0.3],   # P1: normal residuals
            [2.5, 3.0, 2.8, 2.1, 2.9]  # P4: outlier residuals, mean ~2.66
        ],
        [0.067, 2.66],
        [0.1, -0.2, 0.3, 2.5, 3.0, 2.8, 2.1, 2.9],
        [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        0.1, -0.3, ["P4"]
    )

    flags = BayesInteractomics._compute_protein_flags(
        data, all_ids, name_to_idx;
        ppc_residuals = ppc_res
    )

    @test length(flags) == length(all_ids)

    # Build lookup for convenience
    flag_map = Dict(f.protein_name => f for f in flags)

    # P1: 3 sample obs, residuals available, not outlier → 3 < 4 so is_low_data
    @test flag_map["P1"].n_observations == 3
    @test flag_map["P1"].is_residual_outlier == false
    @test flag_map["P1"].is_low_data == true
    @test flag_map["P1"].overall_flag == :warning

    # P2: 1 non-missing sample obs, no residuals → is_low_data, NaN residual
    @test flag_map["P2"].n_observations == 1
    @test flag_map["P2"].is_low_data == true
    @test isnan(flag_map["P2"].mean_residual)
    @test flag_map["P2"].is_residual_outlier == false
    @test flag_map["P2"].overall_flag == :warning

    # P3: 2 non-missing samples, no residuals → is_low_data
    @test flag_map["P3"].n_observations == 2
    @test flag_map["P3"].is_low_data == true

    # P4: 3 obs, residuals with |mean| > 2 → outlier + low_data = fail
    @test flag_map["P4"].is_residual_outlier == true
    @test flag_map["P4"].is_low_data == true
    @test flag_map["P4"].overall_flag == :fail

    # P5: 3 obs, no residuals → low_data warning
    @test flag_map["P5"].n_observations == 3
    @test isnan(flag_map["P5"].mean_residual)
    @test flag_map["P5"].overall_flag == :warning

    # P6: 5 non-missing obs (3 from exp1 + 2 from exp2) → not low_data
    @test flag_map["P6"].n_observations == 5
    @test flag_map["P6"].is_low_data == false
    @test flag_map["P6"].overall_flag == :ok

    # Test without PPC residuals — all proteins get NaN residuals
    flags_no_ppc = BayesInteractomics._compute_protein_flags(
        data, all_ids, name_to_idx
    )
    @test length(flags_no_ppc) == length(all_ids)
    @test all(f -> isnan(f.mean_residual), flags_no_ppc)
    @test all(f -> f.is_residual_outlier == false, flags_no_ppc)
end

@testitem "_merge_diagnostics_to_results" begin
    using BayesInteractomics
    using DataFrames
    using Dates

    # Build a mock results DataFrame
    results_df = DataFrame(
        Protein = ["P1", "P2", "P3", "P4"],
        BF = [100.0, 50.0, 10.0, 1.0],
        posterior_prob = [0.95, 0.80, 0.50, 0.10],
        q = [0.01, 0.05, 0.10, 0.50],
        mean_log2FC = [3.0, 2.0, 1.0, 0.5]
    )

    # Build mock protein flags (for ALL proteins)
    flags = [
        ProteinDiagnosticFlag("P1", 10, 0.067, 0.3, false, false, :ok),
        ProteinDiagnosticFlag("P2", 3, NaN, NaN, false, true, :warning),
        ProteinDiagnosticFlag("P3", 8, 0.5, 1.2, false, false, :ok),
        ProteinDiagnosticFlag("P4", 2, 2.5, 3.0, true, true, :fail)
    ]

    # Build mock extended PPC stats (only P1 and P3 have PPC)
    ext_stats = [
        PPCExtendedStatistics("P1", :hbm, 0.45, 0.62, 0.33),
        PPCExtendedStatistics("P3", :regression, 0.80, 0.40, 0.55)
    ]

    cfg = DiagnosticsConfig()
    summary_df = DataFrame(
        Protein = String[], model = Symbol[],
        pvalue_mean = Float64[], pvalue_sd = Float64[], pvalue_log2fc = Float64[]
    )

    dr = DiagnosticsResult(
        cfg, ProteinPPC[], BetaBernoulliPPC[],
        nothing, nothing, nothing, nothing, nothing,
        nothing, nothing,
        ext_stats, flags, nothing, nothing, summary_df, now()
    )

    merged = BayesInteractomics._merge_diagnostics_to_results(results_df, dr)

    # Check original columns preserved
    @test "Protein" in names(merged)
    @test "BF" in names(merged)

    # Check flag columns present
    @test "n_observations" in names(merged)
    @test "diagnostic_flag" in names(merged)
    @test "is_low_data" in names(merged)
    @test "is_residual_outlier" in names(merged)
    @test "mean_residual" in names(merged)
    @test "max_abs_residual" in names(merged)

    # Check extended PPC columns present
    @test "ppc_pvalue_skewness" in names(merged)
    @test "ppc_pvalue_kurtosis" in names(merged)
    @test "ppc_pvalue_iqr_ratio" in names(merged)

    # Check values for specific proteins
    p1_row = filter(row -> row.Protein == "P1", merged)
    @test nrow(p1_row) == 1
    @test p1_row.n_observations[1] == 10
    @test p1_row.diagnostic_flag[1] == "ok"
    @test p1_row.ppc_pvalue_skewness[1] ≈ 0.45

    p4_row = filter(row -> row.Protein == "P4", merged)
    @test nrow(p4_row) == 1
    @test p4_row.diagnostic_flag[1] == "fail"
    @test p4_row.is_low_data[1] == true
    # P4 has no extended PPC stats → should be missing
    @test ismissing(p4_row.ppc_pvalue_skewness[1])

    # P2 has no residuals → NaN mean_residual from flags
    p2_row = filter(row -> row.Protein == "P2", merged)
    @test isnan(p2_row.mean_residual[1])

    # Original df should not be modified
    @test !("n_observations" in names(results_df))
end

@testitem "_merge_diagnostics_to_results with sensitivity" begin
    using BayesInteractomics
    using DataFrames
    using Dates
    using Statistics

    # Build a mock results DataFrame
    results_df = DataFrame(
        Protein = ["P1", "P2", "P3", "P4"],
        BF = [100.0, 50.0, 10.0, 1.0],
        posterior_prob = [0.95, 0.80, 0.50, 0.10],
        q = [0.01, 0.05, 0.10, 0.50],
        mean_log2FC = [3.0, 2.0, 1.0, 0.5]
    )

    # Build mock SensitivityResult
    n_proteins = 4
    n_settings = 3
    posterior_matrix = [0.9 0.92 0.95;
                        0.8 0.75 0.85;
                        0.5 0.45 0.55;
                        0.1 0.08 0.12]
    bf_matrix = ones(n_proteins, n_settings)
    q_matrix = [0.01 0.02 0.005;
                0.05 0.06 0.04;
                0.10 0.12 0.08;
                0.50 0.55 0.45]

    summary_df = DataFrame(
        Protein = ["P1", "P2", "P3", "P4"],
        baseline_posterior = posterior_matrix[:, 1],
        mean_posterior = vec(Statistics.mean(posterior_matrix, dims=2)),
        std_posterior = vec(Statistics.std(posterior_matrix, dims=2)),
        min_posterior = vec(minimum(posterior_matrix, dims=2)),
        max_posterior = vec(maximum(posterior_matrix, dims=2)),
        range = vec(maximum(posterior_matrix, dims=2) .- minimum(posterior_matrix, dims=2))
    )

    n_set_f = Float64(n_settings)
    stab_df = DataFrame(
        Protein = ["P1", "P2", "P3", "P4"],
        frac_P_gt_0_5 = [3/n_set_f, 3/n_set_f, 1/n_set_f, 0.0],
        frac_P_gt_0_8 = [3/n_set_f, 1/n_set_f, 0.0, 0.0],
        frac_P_gt_0_95 = [1/n_set_f, 0.0, 0.0, 0.0],
        frac_q_lt_0_05 = [3/n_set_f, 1/n_set_f, 0.0, 0.0],
        frac_q_lt_0_01 = [1/n_set_f, 0.0, 0.0, 0.0]
    )

    sr = SensitivityResult(
        SensitivityConfig(),
        PriorSetting[],
        posterior_matrix, bf_matrix, q_matrix,
        ["P1", "P2", "P3", "P4"],
        1,
        summary_df, stab_df,
        now()
    )

    # Merge with sensitivity only (no diagnostics)
    merged = BayesInteractomics._merge_diagnostics_to_results(
        results_df, nothing; sensitivity = sr
    )

    # Check sensitivity summary columns
    @test "sensitivity_std_posterior" in names(merged)
    @test "sensitivity_range" in names(merged)
    @test "sensitivity_min_posterior" in names(merged)
    @test "sensitivity_max_posterior" in names(merged)

    # Check stability columns
    @test "frac_P_gt_0_8" in names(merged)
    @test "frac_q_lt_0_05" in names(merged)

    # Check values
    p1_row = filter(row -> row.Protein == "P1", merged)
    @test nrow(p1_row) == 1
    @test p1_row.sensitivity_range[1] ≈ 0.05  atol=1e-10
    @test p1_row.frac_P_gt_0_8[1] ≈ 1.0

    p4_row = filter(row -> row.Protein == "P4", merged)
    @test p4_row.frac_P_gt_0_8[1] ≈ 0.0

    # Original df should not be modified
    @test !("sensitivity_range" in names(results_df))
end

@testitem "_ks_test_uniform" begin
    using BayesInteractomics

    # Perfect uniform: sorted values [0.1, 0.2, ..., 1.0] vs ECDF [0.1, 0.2, ..., 1.0]
    pvals_perfect = collect(range(0.1, 1.0, length=10))
    ks = BayesInteractomics._ks_test_uniform(pvals_perfect)
    @test ks < 0.15  # Should be very small for approximately uniform data

    # All extreme: all 0.0 should give KS ≈ 1.0
    pvals_extreme = zeros(20)
    ks_extreme = BayesInteractomics._ks_test_uniform(pvals_extreme)
    @test ks_extreme > 0.9

    # Empty input
    ks_empty = BayesInteractomics._ks_test_uniform(Float64[])
    @test isnan(ks_empty)

    # NaN handling
    pvals_nan = [NaN, NaN, 0.5]
    ks_nan = BayesInteractomics._ks_test_uniform(pvals_nan)
    @test !isnan(ks_nan)
end

@testitem "Report generation with minimal data" begin
    using BayesInteractomics
    using DataFrames
    using Dates

    cfg = DiagnosticsConfig()
    ppc1 = ProteinPPC("P1", :hbm, randn(5), randn(5, 10), 0.4, 0.5, 0.6)
    bb1 = BetaBernoulliPPC("P1", 5, 2, rand(0:7, 100), rand(0:5, 100), 0.3)

    cal = CalibrationResult(
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.05, 0.25, 0.5, 0.75, 0.95],
        [10, 20, 30, 20, 10],
        0.03,
        0.05
    )

    summary_df = DataFrame(
        Protein = ["P1", "P1"],
        model = [:hbm, :betabernoulli],
        pvalue_mean = [0.4, 0.3],
        pvalue_sd = [0.5, NaN],
        pvalue_log2fc = [0.6, NaN]
    )

    dr = DiagnosticsResult(cfg, [ppc1], [bb1], nothing, nothing, cal, nothing, nothing,
                           nothing, nothing, nothing, nothing, nothing, nothing, summary_df, now())

    # Generate report to temp file
    tmpfile = tempname() * ".md"
    content = generate_diagnostics_report(dr; filename=tmpfile)

    @test occursin("Model Diagnostics Report", content)
    @test occursin("Posterior Predictive P-values", content)
    @test occursin("Calibration Assessment", content)
    @test occursin("ECE", content)
    @test occursin("Why calibration may deviate from the diagonal", content)
    @test occursin("Regression slope threshold", content)
    @test occursin("Constitutive interactors", content)
    @test isfile(tmpfile)
    rm(tmpfile, force=true)
end

@testitem "Report generation with enhanced sections" begin
    using BayesInteractomics
    using DataFrames
    using Dates

    cfg = DiagnosticsConfig()
    ppc1 = ProteinPPC("P1", :hbm, randn(10), randn(10, 50), 0.4, 0.5, 0.6)
    bb1 = BetaBernoulliPPC("P1", 5, 2, rand(0:7, 100), rand(0:5, 100), 0.3)

    # Build mock enhanced residuals
    base_res = ResidualResult(
        :hbm, ["P1", "P2"],
        [[0.1, -0.2, 0.3], [0.5, -0.1]],
        [0.067, 0.2], [0.1, -0.2, 0.3, 0.5, -0.1],
        [1.0, 1.0, 1.0, 2.0, 2.0],
        0.1, -0.3, String[]
    )
    enh_hbm = EnhancedResidualResult(
        base_res,
        [[0.05, -0.1, 0.2], [0.3, -0.05]],
        [0.45, 0.52, 0.61, 0.38, 0.72]
    )

    # Build mock extended PPC stats
    ext_stats = [
        PPCExtendedStatistics("P1", :hbm, 0.45, 0.62, 0.33)
    ]

    # Build mock protein flags
    flags = [
        ProteinDiagnosticFlag("P1", 5, 0.067, 0.3, false, false, :ok),
        ProteinDiagnosticFlag("P2", 2, 2.5, 3.0, true, true, :fail)
    ]

    summary_df = DataFrame(
        Protein = ["P1", "P1"],
        model = [:hbm, :betabernoulli],
        pvalue_mean = [0.4, 0.3],
        pvalue_sd = [0.5, NaN],
        pvalue_log2fc = [0.6, NaN]
    )

    dr = DiagnosticsResult(cfg, [ppc1], [bb1], base_res, nothing, nothing, nothing, nothing,
                           enh_hbm, nothing, ext_stats, flags, nothing, nothing, summary_df, now())

    # Generate report to temp file
    tmpfile = tempname() * ".md"
    content = generate_diagnostics_report(dr; filename=tmpfile)

    # Check new sections appear
    @test occursin("PIT (Probability Integral Transform) Histogram", content)
    @test occursin("KS statistic", content)
    @test occursin("Extended PPC Statistics", content)
    @test occursin("p(skewness)", content)
    @test occursin("p(kurtosis)", content)
    @test occursin("p(IQR/SD)", content)
    @test occursin("Per-Protein Diagnostic Flags", content)
    @test occursin("P2", content)  # P2 is flagged as :fail
    @test isfile(tmpfile)
    rm(tmpfile, force=true)
end
