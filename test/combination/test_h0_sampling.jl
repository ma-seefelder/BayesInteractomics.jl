"""
    test_h0_sampling.jl

Tests for the H0 null-distribution sampling pipeline:
  - permuteLabels: structural integrity and bait-row preservation
  - vcat (InteractionData): protein count, ID uniqueness
  - computeH0_BayesFactors accumulation loop: linear (not exponential) growth
  - computeH0_BayesFactors smoke test: produces a valid, non-empty H0 DataFrame
"""

# ---------------------------------------------------------------------------
# permuteLabels — structural preservation
# ---------------------------------------------------------------------------
@testitem "permuteLabels preserves protein count and protocol structure" setup=[DataStructureFixtures] begin
    using BayesInteractomics
    using BayesInteractomics: permuteLabels, getIDs, getNoProtocols, getNoExperiments

    data = DataStructureFixtures.create_mock_interaction_data(10, 1; n_experiments_per_protocol=2)
    perm = permuteLabels(data, 1)

    @test length(getIDs(perm)) == length(getIDs(data))
    @test getNoProtocols(perm) == getNoProtocols(data)
    @test getNoExperiments(perm, 1) == getNoExperiments(data, 1)
end

@testitem "permuteLabels preserves bait row values" setup=[DataStructureFixtures] begin
    using BayesInteractomics
    using BayesInteractomics: permuteLabels, getSamples, getControls, getExperiment

    data  = DataStructureFixtures.create_mock_interaction_data(10, 1; n_experiments_per_protocol=2)
    refID = 1
    perm  = permuteLabels(data, refID)

    for p in 1:BayesInteractomics.getNoProtocols(data)
        for e in 1:BayesInteractomics.getNoExperiments(data, p)
            orig_sample  = getExperiment(getSamples(data,  p), e)
            perm_sample  = getExperiment(getSamples(perm,  p), e)
            orig_control = getExperiment(getControls(data, p), e)
            perm_control = getExperiment(getControls(perm, p), e)

            # Bait row must be bit-identical after permutation
            @test perm_sample[refID, :]  == orig_sample[refID, :]
            @test perm_control[refID, :] == orig_control[refID, :]
        end
    end
end

@testitem "permuteLabels preserves value multiset per experiment" setup=[DataStructureFixtures] begin
    using BayesInteractomics
    using BayesInteractomics: permuteLabels, getSamples, getControls, getExperiment

    data = DataStructureFixtures.create_mock_interaction_data(10, 1; n_experiments_per_protocol=2)
    perm = permuteLabels(data, 1)

    for p in 1:BayesInteractomics.getNoProtocols(data)
        for e in 1:BayesInteractomics.getNoExperiments(data, p)
            orig_s = getExperiment(getSamples(data,  p), e)
            orig_c = getExperiment(getControls(data, p), e)
            perm_s = getExperiment(getSamples(perm,  p), e)
            perm_c = getExperiment(getControls(perm, p), e)

            # All original values (combined) must appear in permuted matrices
            orig_vals = sort(skipmissing(vcat(vec(orig_s), vec(orig_c))) |> collect)
            perm_vals = sort(skipmissing(vcat(vec(perm_s), vec(perm_c))) |> collect)
            @test orig_vals ≈ perm_vals
        end
    end
end

# ---------------------------------------------------------------------------
# vcat — protein count and ID uniqueness
# ---------------------------------------------------------------------------
@testitem "vcat(InteractionData) doubles the protein count" setup=[DataStructureFixtures] begin
    using BayesInteractomics
    using BayesInteractomics: permuteLabels, getIDs

    n = 10
    data  = DataStructureFixtures.create_mock_interaction_data(n, 1; n_experiments_per_protocol=2)
    perm  = permuteLabels(data, 1)
    merged = BayesInteractomics.vcat(data, perm)

    @test length(getIDs(merged)) == 2 * n
end

@testitem "vcat(InteractionData) produces unique protein IDs" setup=[DataStructureFixtures] begin
    using BayesInteractomics
    using BayesInteractomics: permuteLabels, getIDs

    data   = DataStructureFixtures.create_mock_interaction_data(10, 1; n_experiments_per_protocol=2)
    perm   = permuteLabels(data, 1)
    merged = BayesInteractomics.vcat(data, perm)
    ids    = getIDs(merged)

    @test length(ids) == length(unique(ids))
end

# ---------------------------------------------------------------------------
# CRITICAL: accumulation loop grows linearly, not exponentially
# ---------------------------------------------------------------------------
@testitem "H0 dataset accumulation is linear (not exponential)" setup=[DataStructureFixtures] begin
    using BayesInteractomics
    using BayesInteractomics: permuteLabels, getIDs

    n_proteins   = 5
    n_datasets   = 4

    data = DataStructureFixtures.create_mock_interaction_data(n_proteins, 1; n_experiments_per_protocol=2)

    # ── Fixed loop (uses original data) ──────────────────────────────────
    fixed = permuteLabels(data, 1)
    for _ in 2:n_datasets
        fixed = BayesInteractomics.vcat(fixed, permuteLabels(data))
    end
    @test length(getIDs(fixed)) == n_datasets * n_proteins

    # ── Buggy loop (would use growing permuted_data) ─────────────────────
    # Demonstrates what the bug produced: 2^(n_datasets-1) × n_proteins
    buggy = permuteLabels(data, 1)
    for _ in 2:n_datasets
        buggy = BayesInteractomics.vcat(buggy, permuteLabels(buggy))
    end
    @test length(getIDs(buggy)) == 2^(n_datasets - 1) * n_proteins
    @test length(getIDs(fixed)) < length(getIDs(buggy))  # fixed is smaller (correct)
end

# ---------------------------------------------------------------------------
# computeH0_BayesFactors smoke test
# ---------------------------------------------------------------------------
@testitem "computeH0_BayesFactors produces valid H0 DataFrame" setup=[DataStructureFixtures] begin
    using BayesInteractomics
    using DataFrames

    # Minimal dataset: 5 proteins, 1 protocol, 2 experiments × 3 replicates
    # n=5 with n_proteins=5 → n_datasets=1 (loop does not execute), total 5 permuted proteins
    # n_controls = 1 protocol × 2 experiments × 3 replicates = 6; n_samples = same
    n_proteins = 5
    data       = DataStructureFixtures.create_mock_interaction_data(n_proteins, 1; n_experiments_per_protocol=2)
    savefile   = tempname() * ".xlsx"

    try
        # computeH0_BayesFactors returns the H0 DataFrame directly
        H0 = BayesInteractomics.computeH0_BayesFactors(
            data;
            savefile   = savefile,
            n_controls = 6,
            n_samples  = 6,
            refID      = 1,
            n          = n_proteins
        )

        @test isfile(savefile)
        @test H0 isa DataFrame
        @test nrow(H0) > 0
        @test hasproperty(H0, :bf_enrichment)
        @test hasproperty(H0, :bf_correlation)
        @test hasproperty(H0, :bf_detected)

        # BFs from permuted (null) data must not all be zero or infinite
        valid_bf = filter(x -> isfinite(x) && x > 0, H0.bf_correlation)
        @test length(valid_bf) > 0

    finally
        isfile(savefile) && rm(savefile)
    end
end
