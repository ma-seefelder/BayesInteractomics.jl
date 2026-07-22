# ═══════════════════════════════════════════════════════════════════════════════
# Tests for docking integration (stubs + extension)
# ═══════════════════════════════════════════════════════════════════════════════

@testitem "DockingConfig defaults" begin
    using BayesInteractomics

    config = DockingConfig()
    @test config.posterior_threshold == 0.8
    @test config.pep_threshold == 0.01
    @test config.max_pairs == 100
    @test config.max_tokens_per_job == 5000
    @test config.max_jobs_per_batch == 30
    @test config.parse_full_data == true
    @test config.verbose == true
    @test config.dockability_weight == 0.3
end

@testitem "DockingConfig dockability_weight customization" begin
    using BayesInteractomics

    # Posterior-only mode
    config = DockingConfig(dockability_weight=0.0)
    @test config.dockability_weight == 0.0

    # Dockability-dominant mode
    config2 = DockingConfig(dockability_weight=0.8)
    @test config2.dockability_weight == 0.8
end

@testitem "docking_cache_key" begin
    using BayesInteractomics

    # Order independent
    @test docking_cache_key("P04637", "Q9BYF1") == "P04637__Q9BYF1"
    @test docking_cache_key("Q9BYF1", "P04637") == "P04637__Q9BYF1"

    # Case insensitive
    @test docking_cache_key("p04637", "q9byf1") == "P04637__Q9BYF1"

    # Strips whitespace
    @test docking_cache_key("  P04637 ", " Q9BYF1  ") == "P04637__Q9BYF1"
end

@testitem "compute_pdockq" begin
    using BayesInteractomics

    # Zero contacts → 0
    @test compute_pdockq(80.0, 0) == 0.0

    # Published sigmoid: pDockQ = 0.707 / (1 + exp(-0.03148*(x-388.06))) + 0.03138
    # where x = avg_plddt * log(n_contacts)
    # For avg_plddt=80, n_contacts=100: x = 80 * log(100) = 80 * 4.605 ≈ 368.4
    pdockq = compute_pdockq(80.0, 100)
    @test 0.0 < pdockq < 1.0
    @test pdockq ≈ 0.707 / (1.0 + exp(-0.03148 * (80.0 * log(100) - 388.06))) + 0.03138

    # Higher pLDDT and more contacts → higher pDockQ
    @test compute_pdockq(90.0, 200) > compute_pdockq(60.0, 50)
end

@testitem "compute_bf_from_iptm step function" begin
    using BayesInteractomics

    # Below noise floor
    @test compute_bf_from_iptm(0.10) == 1.0
    @test compute_bf_from_iptm(0.19) == 1.0

    # Low confidence
    @test compute_bf_from_iptm(0.20) == 0.7
    @test compute_bf_from_iptm(0.35) == 0.7

    # Ambiguous
    @test compute_bf_from_iptm(0.40) == 1.5
    @test compute_bf_from_iptm(0.55) == 1.5

    # Moderate confidence
    @test compute_bf_from_iptm(0.60) == 5.0
    @test compute_bf_from_iptm(0.75) == 5.0

    # High confidence
    @test compute_bf_from_iptm(0.80) == 12.0
    @test compute_bf_from_iptm(0.95) == 12.0
end

@testitem "compute_bf_from_pdockq logistic" begin
    using BayesInteractomics

    # Low pDockQ → low BF (but clamped above 0.1)
    bf_low = compute_bf_from_pdockq(0.0)
    @test bf_low >= 0.1
    @test bf_low < 1.0

    # Medium pDockQ → moderate BF
    bf_mid = compute_bf_from_pdockq(0.23)
    @test bf_mid > 1.0

    # High pDockQ → high BF (but clamped at 20)
    bf_high = compute_bf_from_pdockq(0.50)
    @test bf_high > bf_mid
    @test bf_high <= 20.0

    # Very high pDockQ → clamped at 20
    bf_max = compute_bf_from_pdockq(0.90)
    @test bf_max <= 20.0

    # Monotonicity
    @test compute_bf_from_pdockq(0.1) < compute_bf_from_pdockq(0.3) < compute_bf_from_pdockq(0.6)
end

@testitem "compute_bf_dock quality gates" begin
    using BayesInteractomics

    # Gate 1: Disorder → BF=1.0
    bf, tier = compute_bf_dock(0.85, 0.60)  # High ipTM but disordered
    @test bf == 1.0
    @test tier == "disordered"

    # Gate 2: Low pLDDT → clamp to [0.7, 1.5] (Tier 1 only)
    bf, tier = compute_bf_dock(0.85, 0.0; mean_plddt_min=30.0)
    @test 0.7 <= bf <= 1.5
    @test tier == "tier1_iptm"

    # Gate 2 does NOT fire for Tier 2 (C2Qscore) — logistic calibration already
    # incorporates interface pLDDT, so whole-chain pLDDT clamp is skipped.
    bf_c2q_gate, tier_c2q_gate = compute_bf_dock(0.74, 0.01;
        c2qscore=0.6, mean_plddt_min=30.0)
    @test tier_c2q_gate == "tier2_c2qscore"
    @test bf_c2q_gate > 1.5  # not clamped to 1.5

    # Gate 2 does NOT fire for Tier 2 (pDockQ) either
    bf_pdq_gate, tier_pdq_gate = compute_bf_dock(0.74, 0.01;
        pdockq=0.5, mean_plddt_min=30.0)
    @test tier_pdq_gate == "tier2_pdockq"
    @test bf_pdq_gate > 1.5  # not clamped to 1.5

    # Gate 3: High PAE → clamp to [0.5, 2.0]
    bf, tier = compute_bf_dock(0.85, 0.0; chain_pair_pae_min=30.0)
    @test 0.5 <= bf <= 2.0

    # Gate 4: High model variance → damping toward 1.0
    bf_nodamp, _ = compute_bf_dock(0.85, 0.0; iptm_std=0.0)
    bf_damp, _ = compute_bf_dock(0.85, 0.0; iptm_std=0.20)
    @test abs(bf_damp - 1.0) < abs(bf_nodamp - 1.0)  # Dampened is closer to 1

    # Normal case (no gates triggered, Tier 1)
    bf_normal, tier_normal = compute_bf_dock(0.75, 0.01)
    @test bf_normal == 5.0
    @test tier_normal == "tier1_iptm"

    # pDockQ triggers Tier 2
    bf_t2, tier_t2 = compute_bf_dock(0.75, 0.01; pdockq=0.45)
    @test tier_t2 == "tier2_pdockq"
    @test bf_t2 > 1.0

    # BF clamping: always within [0.1, 20.0]
    bf_clamped, _ = compute_bf_dock(0.99, 0.0; pdockq=0.99)
    @test bf_clamped <= 20.0
    @test bf_clamped >= 0.1
end

@testitem "default_calibration" begin
    using BayesInteractomics

    cal = default_calibration()
    @test cal.clamp_range == (0.1, 20.0)
    @test cal.iptm_missing_threshold == 0.20
end

@testitem "apply_docking_update" begin
    using BayesInteractomics
    using DataFrames
    using Dates

    # Create mock results
    results = DataFrame(
        Protein = ["ProtA", "ProtB", "ProtC"],
        posterior_prob = [0.95, 0.50, 0.10],
        BFDR = [0.01, 0.10, 0.50],
        BF = [19.0, 1.0, 0.11],
    )

    # Create mock docking result (ProtA has high docking, ProtC not docked)
    pairs = [
        DockingPairResult(
            "BAIT", "ProtA", "", "", 0.85, [0.80, 0.82, 0.85, 0.81, 0.83], 0.02,
            0.9, 0.01, 0.85, 5.0, 0.6, 80.0, 75.0, 150, 12.0, "tier1_iptm", :success, 800,
            NaN, NaN, NaN,
        ),
        DockingPairResult(
            "BAIT", "ProtB", "", "", 0.30, [0.28, 0.30, 0.25, 0.27, 0.31], 0.02,
            0.3, 0.10, 0.30, 20.0, NaN, NaN, NaN, 0, 0.7, "tier1_iptm", :success, 750,
            NaN, NaN, NaN,
        ),
    ]

    config = DockingConfig()
    docking = DockingResult(pairs, config, 3, 2, 0, 1, 0, 0, now())

    updated = apply_docking_update(results, docking)

    # Check new columns exist
    @test hasproperty(updated, :posterior_prob_ms)
    @test hasproperty(updated, :bf_docking)
    @test hasproperty(updated, :posterior_prob_combined)
    @test hasproperty(updated, :docking_status)

    # ProtA: boosted by docking (BF=12)
    @test updated.bf_docking[1] == 12.0
    @test updated.posterior_prob_combined[1] > updated.posterior_prob_ms[1]
    @test updated.docking_status[1] == "success"

    # ProtB: reduced by docking (BF=0.7)
    @test updated.bf_docking[2] == 0.7
    @test updated.posterior_prob_combined[2] < updated.posterior_prob_ms[2]

    # ProtC: no docking data → BF=1, unchanged
    @test updated.bf_docking[3] == 1.0
    @test updated.posterior_prob_combined[3] ≈ updated.posterior_prob_ms[3]
    @test updated.docking_status[3] == "no_result"

    # Two-stage update formula verification
    pp_ms = 0.95
    bf = 12.0
    odds = pp_ms / (1.0 - pp_ms)
    odds_new = odds * bf
    expected = odds_new / (1.0 + odds_new)
    @test updated.posterior_prob_combined[1] ≈ expected
end

@testitem "CONFIG docking fields" begin
    using BayesInteractomics

    config = CONFIG(
        datafile = ["test.xlsx"],
        control_cols = [Dict(1 => [1, 2])],
        sample_cols = [Dict(1 => [3, 4])],
        poi = "TEST",
        run_docking = true,
        bait_sequence = "MAFKLPTG",
    )

    @test config.run_docking == true
    @test config.bait_sequence == "MAFKLPTG"
    @test config.docking_config === nothing
    @test config.bait_uniprot == ""
end

@testitem "generate_docking_requests" begin
    using BayesInteractomics
    using DataFrames

    results = DataFrame(
        Protein = ["ProtA", "ProtB", "ProtC"],
        posterior_prob = [0.995, 0.985, 0.30],
        PEP = [0.005, 0.015, 0.70],
        BFDR = [0.005, 0.010, 0.50],
    )

    bait_seq = "MAFKLPTGQRST"  # 12 residues

    mktempdir() do tmpdir
        config = DockingConfig(
            posterior_threshold = 0.8,
            pep_threshold = 0.01,
            max_pairs = 10,
            request_output_dir = joinpath(tmpdir, "requests"),
            verbose = false,
        )

        # No UniProt IDs → no sequences fetched → 0 requests, but structure is correct
        batch = generate_docking_requests(
            results, bait_seq;
            bait_name = "BAIT",
            output_dir = joinpath(tmpdir, "requests"),
            config = config,
        )

        @test batch isa DockingRequestBatch
        @test isfile(batch.guide_path)
        @test isfile(batch.manifest_path)
    end
end

@testitem "import_docking_results with example ZIP" begin
    using BayesInteractomics
    using DataFrames

    example_zip = joinpath(@__DIR__, "..", "..", "ext", "BayesInteractomicsDocking",
                           "example_data", "folds_2026_03_05_10_03.zip")

    if isfile(example_zip)
        results = DataFrame(
            Protein = ["HAP40", "TNRC6B"],
            posterior_prob = [0.95, 0.85],
            BFDR = [0.01, 0.03],
        )

        mktempdir() do tmpdir
            cp(example_zip, joinpath(tmpdir, "folds.zip"))

            config = DockingConfig(
                parse_full_data = false,  # Skip full_data parsing (too large for test)
                cache_dir = joinpath(tmpdir, "cache"),
                verbose = false,
            )

            docking = import_docking_results(tmpdir, results; config=config)

            @test docking isa DockingResult
            @test length(docking.pairs) >= 1

            pair = docking.pairs[1]
            @test pair.iptm_best ≈ 0.39 atol=0.01
            @test length(pair.iptm_all) == 5
            @test pair.status == :success || pair.calibration_tier == "disordered"
            @test pair.token_count == 2204

            # Verify BF is computed
            @test isfinite(pair.bf_dock)
            @test pair.bf_dock >= 0.1
            @test pair.bf_dock <= 20.0

            # Check caching works
            docking2 = import_docking_results(tmpdir, results; config=config)
            @test length(docking2.pairs) == length(docking.pairs)
            @test docking2.n_cached == length(docking.pairs)
        end
    end
end

@testitem "Two-stage update preserves missing data" begin
    using BayesInteractomics
    using DataFrames
    using Dates

    # All proteins without docking → posteriors unchanged
    results = DataFrame(
        Protein = ["A", "B", "C"],
        posterior_prob = [0.9, 0.5, 0.1],
    )

    empty_docking = DockingResult(
        DockingPairResult[], DockingConfig(), 0, 0, 0, 3, 0, 0, now(),
    )

    updated = apply_docking_update(results, empty_docking)

    @test updated.posterior_prob_combined ≈ results.posterior_prob
    @test all(updated.bf_docking .== 1.0)
end

@testitem "P=0/1 epsilon clamp in docking update" begin
    using BayesInteractomics: _bayesian_update_log_odds, _derive_epsilon

    # P=1.0 case: should NOT silently skip
    result = _bayesian_update_log_odds(1.0, 0.5, 1e-10)
    @test isfinite(result)
    @test result < 1.0  # BF < 1 should reduce posterior

    # P=0.0 case: should NOT silently skip
    result = _bayesian_update_log_odds(0.0, 10.0, 1e-10)
    @test isfinite(result)
    @test result > 0.0  # BF > 1 should increase posterior

    # Normal case: P=0.8, BF=5.0
    result = _bayesian_update_log_odds(0.8, 5.0, 1e-10)
    @test isfinite(result)
    @test result > 0.8  # BF > 1 increases posterior

    # Epsilon derivation
    @test _derive_epsilon([0.01, 0.05, 0.1]) == 0.001
    @test _derive_epsilon([0.0, 0.0, 0.0]) == 1e-10  # all zeros fallback
    @test _derive_epsilon([missing, missing]) == 1e-10  # all missing fallback
end

@testitem "q_combined recomputed after docking" begin
    using BayesInteractomics
    using DataFrames
    using Dates

    # Create mock results with known posteriors
    results = DataFrame(
        Protein = ["BAIT", "PREY1", "PREY2", "PREY3"],
        posterior_prob = [1.0, 0.95, 0.5, 0.01],
        q = [0.0, 0.01, 0.3, 0.99],
        log2FC = [5.0, 3.0, 1.0, -0.5],
    )

    # Create minimal DockingResult
    pairs = [
        BayesInteractomics.DockingPairResult(
            "BAIT", "PREY1", "", "",
            0.7, [0.7], 0.0, 0.5, 0.1,
            0.7, 5.0, 0.5, 80.0, 75.0,
            20, 5.0, "tier1", :success, 500,
            NaN, NaN, NaN,
        ),
    ]
    docking = BayesInteractomics.DockingResult(
        pairs, DockingConfig(), 1, 1, 0, 0, 0, 0, now(),
    )

    updated = apply_docking_update(results, docking)

    # BFDR_combined should exist (q_value → BFDR rename)
    @test hasproperty(updated, :BFDR_combined)
    @test length(updated.BFDR_combined) == 4

    # PREY1 posterior should be updated (BF=5 increases it)
    @test updated.posterior_prob_combined[2] > 0.95

    # P=1.0 protein (BAIT) should NOT be skipped -- bf_docking is 1.0 (no docking pair)
    # but the update code should still handle it without error
    @test isfinite(updated.posterior_prob_combined[1])
end

# ═══════════════════════════════════════════════════════════════════════════════
# C2Qscore tests (Genz et al. 2025)
# ═══════════════════════════════════════════════════════════════════════════════

@testitem "C2Qscore computation" begin
    using BayesInteractomics

    # Bias-only case: all inputs zero → returns bias
    @test compute_c2qscore(0.0, 0.0, 0.0, 0.0) ≈ -0.331

    # Known input: compute_c2qscore(0.8, 0.5, 0.7, 0.6)
    # = -0.331 + (-0.036*0.8) + (0.169*0.5) + (0.335*0.7) + (0.683*0.6)
    # = -0.331 + (-0.0288) + 0.0845 + 0.2345 + 0.4098
    # = 0.369
    expected = -0.331 + (-0.036 * 0.8) + (0.169 * 0.5) + (0.335 * 0.7) + (0.683 * 0.6)
    @test compute_c2qscore(0.8, 0.5, 0.7, 0.6) ≈ expected

    # All ones: -0.331 + (-0.036) + 0.169 + 0.335 + 0.683 = 0.82
    expected_ones = -0.331 + (-0.036) + 0.169 + 0.335 + 0.683
    @test compute_c2qscore(1.0, 1.0, 1.0, 1.0) ≈ expected_ones

    # Monotonicity: higher iptm → higher score (iptm has largest weight)
    @test compute_c2qscore(0.5, 0.5, 0.5, 0.9) > compute_c2qscore(0.5, 0.5, 0.5, 0.3)

    # Constants exist
    @test BayesInteractomics.C2QSCORE_AF3_WEIGHTS == [-0.036, 0.169, 0.335, 0.683]
    @test BayesInteractomics.C2QSCORE_AF3_BIAS == -0.331
end

@testitem "C2Qscore BF conversion" begin
    using BayesInteractomics

    # c2q=0.0: P = sigmoid(-2.3721) ≈ 0.0854; odds = 0.0933; BF = 0.0933/0.1765 ≈ 0.529
    bf_zero = compute_bf_from_c2qscore(0.0)
    @test bf_zero < 1.0
    @test bf_zero ≈ 0.529 atol=0.01

    # c2q=0.5: P = sigmoid(-2.3721 + 8.7486*0.5) = sigmoid(2.0022) ≈ 0.881
    # odds = 0.881/0.119 ≈ 7.403; BF = 7.403/0.1765 ≈ 41.96
    bf_mid = compute_bf_from_c2qscore(0.5)
    @test bf_mid ≈ 41.96 atol=0.5

    # c2q=0.7: P = sigmoid(-2.3721 + 8.7486*0.7) = sigmoid(3.752) ≈ 0.977
    # odds = 0.977/0.023 ≈ 42.5; BF = 42.5/0.1765 ≈ 240.8
    bf_high = compute_bf_from_c2qscore(0.7)
    @test bf_high > 200.0  # No clamp

    # No clamping: logistic calibration is well-calibrated
    bf_extreme = compute_bf_from_c2qscore(1.0)
    @test bf_extreme > 20.0  # Would be clamped if clamp applied

    # Monotonicity
    @test compute_bf_from_c2qscore(0.0) < compute_bf_from_c2qscore(0.3) < compute_bf_from_c2qscore(0.6)
end

@testitem "C2Qscore tier selection" begin
    using BayesInteractomics

    # C2Qscore preferred over pDockQ when available
    bf_c2q, tier_c2q = compute_bf_dock(0.7, 0.01; pdockq=0.5, c2qscore=0.4)
    @test tier_c2q == "tier2_c2qscore"

    # Falls back to pDockQ when c2qscore is NaN
    bf_pdq, tier_pdq = compute_bf_dock(0.7, 0.01; pdockq=0.5, c2qscore=NaN)
    @test tier_pdq == "tier2_pdockq"

    # Falls back to ipTM when both are NaN
    bf_iptm, tier_iptm = compute_bf_dock(0.7, 0.01; pdockq=NaN, c2qscore=NaN)
    @test tier_iptm == "tier1_iptm"

    # C2Qscore tier does NOT get final clamped
    bf_unclamped, tier = compute_bf_dock(0.9, 0.0; c2qscore=0.7)
    @test tier == "tier2_c2qscore"
    @test bf_unclamped > 20.0  # Would be clamped to 20 if clamp applied

    # pDockQ tier IS still clamped
    bf_clamped, tier2 = compute_bf_dock(0.99, 0.0; pdockq=0.99)
    @test tier2 == "tier2_pdockq"
    @test bf_clamped <= 20.0

    # Disorder gate still applies to C2Qscore
    bf_dis, tier_dis = compute_bf_dock(0.9, 0.6; c2qscore=0.7)
    @test tier_dis == "disordered"
    @test bf_dis == 1.0
end

@testitem "iPAE extraction from full_data JSON" begin
    using BayesInteractomics
    using JSON3

    # Create mock full_data JSON with known PAE matrix
    # 4 residues: 2 in chain A, 2 in chain B
    pae_matrix = [
        [1.0, 2.0, 5.0, 6.0],   # A1 → [A1, A2, B1, B2]
        [2.0, 1.0, 7.0, 8.0],   # A2 → [A1, A2, B1, B2]
        [9.0, 10.0, 1.0, 2.0],  # B1 → [A1, A2, B1, B2]
        [11.0, 12.0, 2.0, 1.0], # B2 → [A1, A2, B1, B2]
    ]
    # Cross-chain pairs (A→B): 5, 6, 7, 8  (4 pairs)
    # Cross-chain pairs (B→A): 9, 10, 11, 12  (4 pairs)
    # Mean = (5+6+7+8+9+10+11+12) / 8 = 68/8 = 8.5

    mock_data = Dict(
        :token_chain_ids => ["A", "A", "B", "B"],
        :pae => pae_matrix,
    )

    ipae = BayesInteractomics._compute_ipae(mock_data)
    @test ipae ≈ 8.5

    # Test with :predicted_aligned_error key instead of :pae
    mock_data2 = Dict(
        :token_chain_ids => ["A", "A", "B", "B"],
        :predicted_aligned_error => pae_matrix,
    )
    ipae2 = BayesInteractomics._compute_ipae(mock_data2)
    @test ipae2 ≈ 8.5

    # Single chain → NaN
    mock_single = Dict(
        :token_chain_ids => ["A", "A", "A"],
        :pae => [[1.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 1.0]],
    )
    @test isnan(BayesInteractomics._compute_ipae(mock_single))

    # No PAE key → NaN
    mock_nopae = Dict(:token_chain_ids => ["A", "B"])
    @test isnan(BayesInteractomics._compute_ipae(mock_nopae))

    # iPAE normalization: 1 - (raw/31.75)
    raw_ipae = 8.5
    normalized = 1.0 - (raw_ipae / 31.75)
    @test normalized ≈ 0.7323 atol=0.001
end

@testitem "DockingPairResult new fields" begin
    using BayesInteractomics

    pair = DockingPairResult(
        "A", "B", "P1", "P2",
        0.8, [0.8], 0.0, 0.9, 0.1,
        0.8, 5.0, 0.5, 80.0, 75.0,
        20, 5.0, "tier2_c2qscore", :success, 500,
        0.42, 8.5, 0.75,
    )

    @test pair.c2qscore == 0.42
    @test pair.ipae == 8.5
    @test pair.iplddt_interface == 0.75

    # NaN defaults work
    pair_nan = DockingPairResult(
        "A", "B", "P1", "P2",
        0.8, [0.8], 0.0, 0.9, 0.1,
        0.8, 5.0, 0.5, 80.0, 75.0,
        20, 5.0, "tier1_iptm", :success, 500,
        NaN, NaN, NaN,
    )
    @test isnan(pair_nan.c2qscore)
    @test isnan(pair_nan.ipae)
    @test isnan(pair_nan.iplddt_interface)
end

@testitem "Cache backward compatibility with new fields" begin
    using BayesInteractomics
    using JLD2
    using Dates

    mktempdir() do tmpdir
        pair_key = "TEST_A__TEST_B"
        cache_path = joinpath(tmpdir, "$(pair_key).jld2")

        # Create an old-format cache entry (missing c2qscore, ipae, iplddt_interface)
        old_dict = Dict{String, Any}(
            "protein_a"           => "TEST_A",
            "protein_b"           => "TEST_B",
            "uniprot_a"           => "P1",
            "uniprot_b"           => "P2",
            "iptm_best"           => 0.8,
            "iptm_all"            => [0.8, 0.75],
            "iptm_std"            => 0.035,
            "ranking_score"       => 0.9,
            "fraction_disordered" => 0.05,
            "chain_pair_iptm"     => 0.8,
            "chain_pair_pae_min"  => 5.0,
            "pdockq"              => 0.5,
            "mean_plddt_a"        => 80.0,
            "mean_plddt_b"        => 75.0,
            "n_interface_contacts"=> 20,
            "bf_dock"             => 5.0,
            "calibration_tier"    => "tier2_pdockq",
            "status"              => "success",
            "token_count"         => 500,
            "timestamp"           => "2026-01-01T00:00:00",
        )

        JLD2.jldsave(cache_path; pair=old_dict)

        # Load with new code — should get NaN defaults for missing fields
        loaded = BayesInteractomics._load_cached_pair(tmpdir, pair_key)
        @test loaded !== nothing
        @test loaded.protein_a == "TEST_A"
        @test loaded.bf_dock == 5.0
        @test isnan(loaded.c2qscore)
        @test isnan(loaded.ipae)
        @test isnan(loaded.iplddt_interface)

        # Save with new code, then reload — round-trips correctly
        new_pair = DockingPairResult(
            "NEW_A", "NEW_B", "P3", "P4",
            0.9, [0.9], 0.0, 0.95, 0.02,
            0.9, 3.0, 0.6, 85.0, 80.0,
            30, 10.0, "tier2_c2qscore", :success, 600,
            0.55, 7.2, 0.82,
        )
        BayesInteractomics._save_cached_pair(tmpdir, "NEW_A__NEW_B", new_pair)
        reloaded = BayesInteractomics._load_cached_pair(tmpdir, "NEW_A__NEW_B")
        @test reloaded !== nothing
        @test reloaded.c2qscore == 0.55
        @test reloaded.ipae == 7.2
        @test reloaded.iplddt_interface == 0.82
    end
end

@testitem "FullDataScores has ipae field" begin
    using BayesInteractomics

    scores = BayesInteractomics.FullDataScores(0.5, 80.0, 75.0, 20, 70.0, 8.5)
    @test scores.ipae == 8.5

    # NaN for no iPAE
    scores_nan = BayesInteractomics.FullDataScores(0.5, 80.0, 75.0, 20, 70.0, NaN)
    @test isnan(scores_nan.ipae)
end
