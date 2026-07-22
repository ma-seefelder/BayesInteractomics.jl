"""
    test_pca_separation.jl

Tests for PCA separation analysis quality control.
Each @testitem runs in an isolated environment.
"""

# ============================================================================
# Type construction tests (run immediately with types only)
# ============================================================================

@testitem "PCASeparationResult construction" begin
    using BayesInteractomics
    # Test full constructor
    result = PCASeparationResult(
        [1.0 0.5; 2.0 0.3; -1.0 -0.2; -2.0 -0.4],  # 4 samples x 2 PCs
        ["sample", "sample", "control", "control"],
        [1, 1, 1, 1],
        [45.0, 25.0],  # variance explained
        3.5, 0.8,       # Fisher ratios
        50, 100,         # proteins used/total
        :complete_case, :ok, "Good separation", nothing
    )
    @test result.n_proteins_used == 50
    @test result.n_proteins_total == 100
    @test result.flag == :ok
    @test result.fallback_level == :complete_case
    @test size(result.pc_scores) == (4, 2)
    @test length(result.condition_labels) == 4
    @test length(result.protocol_labels) == 4
    @test length(result.variance_explained) == 2
    @test result.fisher_ratio_pc1 == 3.5
    @test result.fisher_ratio_pc2 == 0.8
    @test result.message == "Good separation"
    @test result.per_protocol === nothing
end

@testitem "PCASeparationResult skipped constructor" begin
    using BayesInteractomics
    result = PCASeparationResult(n_proteins_total=100)
    @test result.fallback_level == :skipped
    @test result.flag == :warning
    @test result.n_proteins_used == 0
    @test result.n_proteins_total == 100
    @test size(result.pc_scores, 1) == 0
    @test size(result.pc_scores, 2) == 2
    @test isempty(result.condition_labels)
    @test isempty(result.protocol_labels)
    @test isempty(result.variance_explained)
    @test result.fisher_ratio_pc1 == 0.0
    @test result.fisher_ratio_pc2 == 0.0
    @test result.per_protocol === nothing
    @test occursin("insufficient", result.message)

    # Custom message
    result2 = PCASeparationResult(n_proteins_total=5, message="Custom skip reason")
    @test result2.message == "Custom skip reason"
    @test result2.n_proteins_total == 5
end

@testitem "PCASeparationResult show method" begin
    using BayesInteractomics
    # Normal result
    result = PCASeparationResult(
        [1.0 0.5; -1.0 -0.5], ["sample", "control"], [1, 1],
        [55.0, 20.0], 4.0, 0.5, 80, 100,
        :complete_case, :ok, "OK", nothing
    )
    s = sprint(show, result)
    @test occursin("PCASeparationResult(:ok)", s)
    @test occursin("80 / 100", s)
    @test occursin(":complete_case", s)
    @test occursin("55.0%", s)
    @test occursin("Fisher ratio PC1", s)

    # Skipped result
    skipped = PCASeparationResult(n_proteins_total=10)
    s2 = sprint(show, skipped)
    @test occursin("PCASeparationResult(:warning)", s2)
    @test occursin(":skipped", s2)
    # Should NOT print variance/Fisher for skipped
    @test !occursin("var explained", s2)

    # With per-protocol results
    per = PCASeparationResult(
        [1.0 0.5; -1.0 -0.5], ["sample", "control"], [1, 1],
        [50.0, 25.0], 3.0, 1.0, 40, 50,
        :complete_case, :ok, "OK",
        [PCASeparationResult(n_proteins_total=50)]
    )
    s3 = sprint(show, per)
    @test occursin("Per-protocol", s3)
    @test occursin("1 results", s3)
end

@testitem "PCASeparationResult per_protocol field" begin
    using BayesInteractomics
    # Create per-protocol results (these should have per_protocol === nothing)
    proto1 = PCASeparationResult(
        [1.0 0.5; -1.0 -0.5], ["sample", "control"], [1, 1],
        [60.0, 20.0], 5.0, 0.3, 30, 50,
        :complete_case, :ok, "Protocol 1 OK", nothing
    )
    proto2 = PCASeparationResult(
        [2.0 0.1; -2.0 -0.1], ["sample", "control"], [2, 2],
        [70.0, 15.0], 8.0, 0.1, 25, 50,
        :complete_case, :ok, "Protocol 2 OK", nothing
    )
    # Create combined result with per-protocol
    combined = PCASeparationResult(
        [1.0 0.5; 2.0 0.1; -1.0 -0.5; -2.0 -0.1],
        ["sample", "sample", "control", "control"],
        [1, 2, 1, 2],
        [45.0, 30.0], 2.0, 0.5, 55, 100,
        :complete_case, :ok, "Combined OK",
        [proto1, proto2]
    )
    @test combined.per_protocol !== nothing
    @test length(combined.per_protocol) == 2
    @test combined.per_protocol[1].per_protocol === nothing
    @test combined.per_protocol[2].per_protocol === nothing
    @test combined.per_protocol[1].fisher_ratio_pc1 == 5.0
    @test combined.per_protocol[2].fisher_ratio_pc1 == 8.0
end

# ============================================================================
# Complete-case filtering tests
# DEPENDS ON: implementation of filter_complete_case
# ============================================================================

@testitem "PCA complete-case filtering - strict" begin
    using BayesInteractomics
    # 6 samples x 10 proteins; proteins 9,10 have missing values
    data = Matrix{Union{Missing, Float64}}(rand(6, 10))
    data[1, 9] = missing
    data[3, 10] = missing
    mask, level = BayesInteractomics.filter_complete_case(data; min_proteins=5)
    @test level == :complete_case
    @test count(mask) == 8
    @test !mask[9]
    @test !mask[10]
end

@testitem "PCA complete-case filtering - fallback to 80%" begin
    using BayesInteractomics
    using Random; Random.seed!(42)
    # 5 samples x 30 proteins. Only 15 complete, but 25 have >= 80% present
    data = Matrix{Union{Missing, Float64}}(rand(5, 30))
    for j in 16:30; data[rand(1:5), j] = missing; end  # 1 missing each = 80% present
    # Force < 20 complete by adding missing to some
    for j in 10:19; data[1, j] = missing; data[2, j] = missing; end  # now only 9 complete
    mask, level = BayesInteractomics.filter_complete_case(data)
    @test level == :threshold_80
    @test count(mask) >= 20
    # DEPENDS ON: implementation of filter_complete_case
end

@testitem "PCA complete-case filtering - skip when too few" begin
    using BayesInteractomics
    # 4 samples x 10 proteins, all heavily missing
    data = Matrix{Union{Missing, Float64}}(undef, 4, 10)
    fill!(data, missing)
    data[1, 1] = 1.0  # only 1 value per protein = 25% present
    data[1, 2] = 2.0
    mask, level = BayesInteractomics.filter_complete_case(data; min_proteins=20)
    @test level == :skipped
    @test count(mask) == 0
    # DEPENDS ON: implementation of filter_complete_case
end

# ============================================================================
# Fisher discriminant ratio tests
# DEPENDS ON: implementation of fishers_ratio
# ============================================================================

@testitem "Fisher discriminant ratio - well separated" begin
    using BayesInteractomics
    scores = [5.0, 5.5, 4.8, -3.0, -3.5, -2.8]
    labels = ["sample", "sample", "sample", "control", "control", "control"]
    ratio = BayesInteractomics.fishers_ratio(scores, labels)
    # (5.1 - (-3.1))^2 / (var_s + var_c) >> 1.0
    @test ratio > 10.0
    # DEPENDS ON: implementation of fishers_ratio
end

@testitem "Fisher discriminant ratio - overlapping groups" begin
    using BayesInteractomics
    using Random; Random.seed!(42)
    scores = [randn() for _ in 1:20]  # all from same distribution
    labels = vcat(fill("sample", 10), fill("control", 10))
    ratio = BayesInteractomics.fishers_ratio(scores, labels)
    @test ratio < 1.0  # Should be low -- no real separation
    # DEPENDS ON: implementation of fishers_ratio
end

@testitem "Fisher discriminant ratio - single replicate edge case" begin
    using BayesInteractomics
    scores = [5.0, -3.0]
    labels = ["sample", "control"]
    ratio = BayesInteractomics.fishers_ratio(scores, labels)
    # var of single element is NaN or 0; function should return 0.0
    @test ratio == 0.0
    # DEPENDS ON: implementation of fishers_ratio
end

# ============================================================================
# Flag assignment tests
# DEPENDS ON: implementation of assign_pca_flag
# ============================================================================

@testitem "PCA flag assignment - ok when Fisher >= 1.0" begin
    using BayesInteractomics
    # Good: PC1 Fisher >= 1.0
    flag1, msg1 = BayesInteractomics.assign_pca_flag(2.0, 0.3, [40.0, 20.0])
    @test flag1 == :ok

    # Good: PC2 Fisher >= 1.0 (PC1 low)
    flag2, msg2 = BayesInteractomics.assign_pca_flag(0.5, 1.5, [30.0, 25.0])
    @test flag2 == :ok

    # Warning: both Fisher < 1.0
    flag3, msg3 = BayesInteractomics.assign_pca_flag(0.3, 0.2, [35.0, 20.0])
    @test flag3 == :warning
    @test occursin("Fisher", msg3) || occursin("separation", msg3)

    # Warning: PC1 variance < 25%
    flag4, msg4 = BayesInteractomics.assign_pca_flag(2.0, 0.5, [20.0, 15.0])
    @test flag4 == :warning
    @test occursin("variance", msg4) || occursin("PC1", msg4)
    # DEPENDS ON: implementation of assign_pca_flag
end

# ============================================================================
# Multi-protocol and batch effect tests
# DEPENDS ON: implementation of run_pca_separation
# ============================================================================

@testitem "PCA multi-protocol - combined and per-protocol" begin
    using BayesInteractomics
    using Random; Random.seed!(42)
    # Construct synthetic 2-protocol InteractionData
    # (implementation uses Protocol constructor and InteractionData constructor)
    # DEPENDS ON: implementation of run_pca_separation
    # This test verifies:
    # 1. Combined result exists
    # 2. per_protocol is a Vector of length 2
    # 3. Each per-protocol result has per_protocol === nothing

    # For now, test the type contract only
    proto1 = PCASeparationResult(
        rand(4, 2), ["sample", "sample", "control", "control"], [1, 1, 1, 1],
        [50.0, 25.0], 3.0, 0.5, 40, 100,
        :complete_case, :ok, "OK", nothing
    )
    proto2 = PCASeparationResult(
        rand(4, 2), ["sample", "sample", "control", "control"], [2, 2, 2, 2],
        [55.0, 20.0], 4.0, 0.3, 35, 100,
        :complete_case, :ok, "OK", nothing
    )
    combined = PCASeparationResult(
        rand(8, 2),
        vcat(fill("sample", 4), fill("control", 4)),
        [1, 1, 2, 2, 1, 1, 2, 2],
        [40.0, 30.0], 2.0, 0.8, 75, 200,
        :complete_case, :ok, "OK",
        [proto1, proto2]
    )
    @test combined.per_protocol !== nothing
    @test length(combined.per_protocol) == 2
    @test combined.per_protocol[1].per_protocol === nothing
    @test combined.per_protocol[2].per_protocol === nothing
end

@testitem "PCA protocol batch effect detection" begin
    using BayesInteractomics
    # When protocols have different baselines (batch effect) causing poor condition
    # separation in combined PCA, the flag should be :warning (not :fail).
    # DEPENDS ON: implementation of run_pca_separation

    # For now, test the type contract for a warning result
    batch_result = PCASeparationResult(
        [1.0 0.5; 2.0 0.3; -1.0 -0.2; -2.0 -0.4],
        ["sample", "sample", "control", "control"],
        [1, 2, 1, 2],
        [35.0, 25.0], 0.3, 0.2, 50, 100,
        :complete_case, :warning,
        "Condition does not separate on PC1 or PC2; possible batch effect",
        nothing
    )
    @test batch_result.flag == :warning
    @test batch_result.fisher_ratio_pc1 < 1.0
    @test batch_result.fisher_ratio_pc2 < 1.0
    @test occursin("batch", batch_result.message) || occursin("separate", batch_result.message)
end

# ============================================================================
# InputQCResult integration test
# ============================================================================

@testitem "InputQCResult with PCA field" begin
    using BayesInteractomics
    using BayesInteractomics: InputQCResult

    pca = PCASeparationResult(
        [1.0 0.5; -1.0 -0.5], ["sample", "control"], [1, 1],
        [50.0, 30.0], 5.0, 1.2, 40, 100, :complete_case, :ok, "OK", nothing
    )
    result = InputQCResult(nothing, nothing, nothing, nothing, pca, :ok)
    @test result.pca_separation !== nothing
    @test result.pca_separation.flag == :ok
    @test result.overall_flag == :ok

    # Verify show includes PCA
    s = sprint(show, result)
    @test occursin("pca=:ok", s)

    # With nothing PCA
    result2 = InputQCResult(nothing, nothing, nothing, nothing, nothing, :ok)
    @test result2.pca_separation === nothing
    s2 = sprint(show, result2)
    @test occursin("pca=not run", s2)
end
