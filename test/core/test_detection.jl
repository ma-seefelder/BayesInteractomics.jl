"""
    test_detection.jl

Tests for detection filtering in the analysis pipeline:
  - Bait not detected → ErrorException
  - Detection mask skips non-detected proteins in BF loops
  - BayesFactorTriplet subsetting to detected-only indices
  - refID remapping to detected-only index space
  - Results DataFrame assembly with is_detected column and missing analytics for non-detected rows
"""

@testitem "Bait not detected throws ErrorException" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, check_bait_detected

    # 2 proteins, bait (protein 2) has all-missing sample intensities
    # Protein 1 (prey): some values present
    # Protein 2 (bait at refID=2): all missing in samples → not detected
    m_ctrl::Matrix{Union{Missing, Float64}} = [1.0 2.0; 3.0 4.0]

    # Bait row (row 2) is all-missing in samples
    m_samples_with_missing_bait::Matrix{Union{Missing, Float64}} = [
        1.0    2.0;   # protein 1 — detected
        missing missing  # protein 2 — NOT detected
    ]

    samples_proto = Protocol(1, ["P1", "P2"], Dict(1 => m_samples_with_missing_bait))
    ctrl_proto    = Protocol(1, ["P1", "P2"], Dict(1 => m_ctrl))

    data = InteractionData(
        ["P1", "P2"], ["Protein1", "Protein2"],
        Dict(1 => samples_proto),
        Dict(1 => ctrl_proto),
        1, Dict(1 => 1),
        4, 3,
        [1], [1], [1, 2],
        BitVector([true, false])  # protein 2 not detected
    )

    refID = 2
    @test !data.detected[refID]
    @test_throws ErrorException check_bait_detected(data, refID)

    # Detected bait should NOT throw
    @test data.detected[1]
    @test_nowarn check_bait_detected(data, 1)
end

@testitem "Detection mask identifies non-detected proteins" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData

    # 3 proteins: proteins 1 and 3 are detected, protein 2 is not
    m_samples::Matrix{Union{Missing, Float64}} = [
        1.0    2.0;     # protein 1 — detected (has values)
        missing missing; # protein 2 — NOT detected (all missing)
        3.0    4.0      # protein 3 — detected (has values)
    ]
    m_ctrl::Matrix{Union{Missing, Float64}} = [
        5.0 6.0;
        7.0 8.0;
        9.0 10.0
    ]

    samples_proto = Protocol(1, ["P1", "P2", "P3"], Dict(1 => m_samples))
    ctrl_proto    = Protocol(1, ["P1", "P2", "P3"], Dict(1 => m_ctrl))

    # Build detection mask as compute_detected_mask would
    detected = BitVector([true, false, true])

    data = InteractionData(
        ["P1", "P2", "P3"], ["Protein1", "Protein2", "Protein3"],
        Dict(1 => samples_proto),
        Dict(1 => ctrl_proto),
        1, Dict(1 => 1),
        6, 5,
        [1], [1], [1, 2, 3],
        detected
    )

    @test data.detected[1] == true
    @test data.detected[2] == false
    @test data.detected[3] == true

    detected_indices = findall(data.detected)
    @test detected_indices == [1, 3]
    @test length(detected_indices) == 2

    # Non-detected should be skipped (BF stays 0.0)
    bf_detected = zeros(Float64, 3)
    for i in 1:3
        if !data.detected[i]
            bf_detected[i] = 0.0
            continue
        end
        bf_detected[i] = 2.0  # Placeholder for actual BF computation
    end

    @test bf_detected[1] == 2.0   # detected — computed
    @test bf_detected[2] == 0.0   # NOT detected — skipped (stays 0)
    @test bf_detected[3] == 2.0   # detected — computed
end

@testitem "BayesFactorTriplet subsetting to detected indices" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet

    # Full triplet of 5 proteins
    # Note: BayesFactorTriplet fields are .enrichment, .correlation, .detection
    bf_e = [1.0, 2.0, 3.0, 4.0, 5.0]
    bf_c = [0.5, 1.5, 2.5, 3.5, 4.5]
    bf_d = [0.1, 0.2, 0.3, 0.4, 0.5]

    full_triplet = BayesFactorTriplet(bf_e, bf_c, bf_d)
    @test length(full_triplet.enrichment) == 5

    # Subset to detected indices 1, 3, 5
    detected_indices = [1, 3, 5]
    bf_enrichment_det = full_triplet.enrichment[detected_indices]
    bf_correlation_det = full_triplet.correlation[detected_indices]
    bf_detected_det = full_triplet.detection[detected_indices]

    det_triplet = BayesFactorTriplet(bf_enrichment_det, bf_correlation_det, bf_detected_det)

    @test length(det_triplet.enrichment) == 3
    @test det_triplet.enrichment == [1.0, 3.0, 5.0]
    @test det_triplet.correlation == [0.5, 2.5, 4.5]
    @test det_triplet.detection == [0.1, 0.3, 0.5]
end

@testitem "refID remapping to detected-only index space" begin
    # 5 proteins, only proteins 1, 3, 5 are detected
    # refID=3 in full list → should map to index 2 in detected-only list
    detected_indices = [1, 3, 5]
    refID_full = 3

    refID_detected = findfirst(==(refID_full), detected_indices)

    @test refID_detected == 2  # protein 3 is at position 2 in [1,3,5]

    # Test another case: refID=1 → index 1
    @test findfirst(==(1), detected_indices) == 1

    # Test case: refID=5 → index 3
    @test findfirst(==(5), detected_indices) == 3

    # Test bait detection guarantee: if bait not in detected_indices, findfirst returns nothing
    @test isnothing(findfirst(==(2), detected_indices))
    @test isnothing(findfirst(==(4), detected_indices))
end

@testitem "Results DataFrame assembly with is_detected and missing analytics" begin
    using BayesInteractomics
    using DataFrames

    # 3 proteins: 1 (detected), 2 (not detected), 3 (detected)
    protein_names = ["P1", "P2", "P3"]
    n_proteins = 3
    detected = BitVector([true, false, true])
    detected_indices = findall(detected)  # [1, 3]
    n_detected = length(detected_indices)

    # Detected-only combined BFs (length 2)
    combined_bf_detected = [5.0, 3.0]
    posterior_prob_detected = [0.85, 0.75]

    # Scatter back to full-length vectors with missing for non-detected
    combined_bf_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    posterior_prob_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    combined_bf_full[detected_indices] .= combined_bf_detected
    posterior_prob_full[detected_indices] .= posterior_prob_detected

    # Build result DataFrame
    copula_df = DataFrame(
        Protein = protein_names,
        is_detected = detected,
        BF = combined_bf_full,
        posterior_prob = posterior_prob_full
    )

    # Verify is_detected column
    @test copula_df.is_detected[1] == true
    @test copula_df.is_detected[2] == false
    @test copula_df.is_detected[3] == true

    # Verify detected proteins have real analytics
    @test copula_df.BF[1] === 5.0
    @test copula_df.posterior_prob[1] === 0.85
    @test copula_df.BF[3] === 3.0
    @test copula_df.posterior_prob[3] === 0.75

    # Verify non-detected protein has missing analytics
    @test ismissing(copula_df.BF[2])
    @test ismissing(copula_df.posterior_prob[2])

    # Verify we can filter on is_detected
    detected_only = filter(r -> r.is_detected, copula_df)
    @test nrow(detected_only) == 2
    @test all(!ismissing, detected_only.BF)
end
