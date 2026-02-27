"""
    test_betabernoulli.jl

Tests for the Beta-Bernoulli detection model used to calculate Bayes factors
for protein detection rate differences between samples and controls.
"""

@testitem "count_detections with all detected" begin
    using BayesInteractomics
    using BayesInteractomics: count_detections, Protocol, InteractionData

    # Create a simple interaction data with 2 proteins, 1 protocol, 1 experiment
    m_sample::Matrix{Union{Missing, Float64}} = [1.0 2.0 3.0; 4.0 5.0 6.0]  # All values present
    m_control::Matrix{Union{Missing, Float64}} = [7.0 8.0; 9.0 10.0]        # 2 replicates

    p_sample = Protocol(1, ["P1", "P2"], Dict(1 => m_sample))
    p_control = Protocol(1, ["P1", "P2"], Dict(1 => m_control))

    data = InteractionData(
        ["P1", "P2"], ["Protein1", "Protein2"],
        Dict(1 => p_sample),
        Dict(1 => p_control),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1]
    )

    # Count detections for first protein
    counts = count_detections(data, 1, 3, 2)

    @test counts.k_sample == 3  # All 3 samples detected
    @test counts.k_control == 2  # All 2 controls detected
    @test counts.f_sample == 0   # No failures
    @test counts.f_control == 0  # No failures
end

@testitem "count_detections with partial detections" begin
    using BayesInteractomics
    using BayesInteractomics: count_detections, Protocol, InteractionData

    # Data with some missing values
    # count_detections counts across ALL protocols and experiments
    # So we need to ensure n_sample and n_control match the total replicates
    m_sample::Matrix{Union{Missing, Float64}} = [1.0 missing 3.0; 4.0 5.0 6.0]
    m_control::Matrix{Union{Missing, Float64}} = [7.0 8.0; missing 10.0]

    p_sample = Protocol(1, ["P1", "P2"], Dict(1 => m_sample))
    p_control = Protocol(1, ["P1", "P2"], Dict(1 => m_control))

    data = InteractionData(
        ["P1", "P2"], ["Protein1", "Protein2"],
        Dict(1 => p_sample),
        Dict(1 => p_control),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1]
    )

    counts = count_detections(data, 1, 3, 2)

    # First protein row: [1.0, missing, 3.0] in samples -> 2 detections, 1 missing
    @test counts.k_sample == 2
    @test counts.f_sample == 1

    # First protein row: [7.0, 8.0] in controls -> 2 detections, 0 missing
    # BUT n_control is 2, so f_control = 2 - 2 = 0
    @test counts.k_control == 2
    @test counts.f_control == 0
end

@testitem "count_detections with no detections" begin
    using BayesInteractomics
    using BayesInteractomics: count_detections, Protocol, InteractionData

    # All missing values for first protein
    m_sample::Matrix{Union{Missing, Float64}} = [missing missing missing; 4.0 5.0 6.0]
    m_control::Matrix{Union{Missing, Float64}} = [missing missing; missing missing]

    p_sample = Protocol(1, ["P1", "P2"], Dict(1 => m_sample))
    p_control = Protocol(1, ["P1", "P2"], Dict(1 => m_control))

    data = InteractionData(
        ["P1", "P2"], ["Protein1", "Protein2"],
        Dict(1 => p_sample),
        Dict(1 => p_control),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1]
    )

    counts = count_detections(data, 1, 3, 2)

    @test counts.k_sample == 0    # No detections
    @test counts.f_sample == 3    # All failures
    @test counts.k_control == 0   # No detections
    @test counts.f_control == 2   # All failures
end

@testitem "betabernoulli with enriched detection in samples" begin
    using BayesInteractomics
    using BayesInteractomics: betabernoulli, Protocol, InteractionData

    # Create data where first protein is heavily detected in samples but not in controls
    # Sample: 8 out of 9 detected
    # Control: 2 out of 6 detected
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

    bf, p, plt = betabernoulli(data, 1, 6, 9)  # n_control=6, n_sample=9

    # With 8/9 detections in sample vs 2/6 in control, we expect:
    # - BF > 1 (evidence for H1: sample detection rate > control detection rate)
    # - Posterior probability p > 0.5
    @test bf > 1.0
    @test p > 0.5
    @test !ismissing(bf)
    @test !ismissing(p)
end

@testitem "betabernoulli with reduced detection in samples (H0)" begin
    using BayesInteractomics
    using BayesInteractomics: betabernoulli, Protocol, InteractionData

    # Create data where first protein is less detected in samples
    # Sample: less detected (3 out of 9)
    # Control: heavily detected (5 out of 6)
    m_sample::Matrix{Union{Missing, Float64}} = reshape([missing missing missing missing missing missing 7.0 8.0 9.0], 1, 9)
    m_control::Matrix{Union{Missing, Float64}} = reshape([1.0 2.0 3.0 4.0 5.0 missing], 1, 6)

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

    bf, p, plt = betabernoulli(data, 1, 6, 9)  # n_control=6, n_sample=9

    # BF should be < 1 (evidence for H0: sample detection < control detection)
    @test bf < 1.0
    # Posterior probability should be < 0.5
    @test p < 0.5
end

@testitem "betabernoulli with negative failures returns missing" begin
    using BayesInteractomics
    using BayesInteractomics: betabernoulli, Protocol, InteractionData

    # Create data with all detections
    m_sample::Matrix{Union{Missing, Float64}} = reshape([1.0 2.0 3.0], 1, 3)
    m_control::Matrix{Union{Missing, Float64}} = reshape([7.0 8.0], 1, 2)

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

    # Call with n_sample < actual detections (creates negative failures)
    # This should return missing values
    bf, p, plt = betabernoulli(data, 1, 2, 1)

    # When failures are negative, function returns missing
    @test ismissing(bf)
    @test ismissing(p)
end

@testitem "betabernoulli posterior probability bounds" begin
    using BayesInteractomics
    using BayesInteractomics: betabernoulli, Protocol, InteractionData

    m_sample::Matrix{Union{Missing, Float64}} = reshape([1.0, 2.0, 3.0], 1, 3)
    m_control::Matrix{Union{Missing, Float64}} = reshape([7.0, 8.0], 1, 2)

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

    # Note: n_sample and n_control must match total counts
    # Sample has 3 values, control has 2 values
    bf, p, plt = betabernoulli(data, 1, 2, 3)  # n_control=2, n_sample=3

    # Posterior probability must be between 0 and 1
    @test !ismissing(p)
    @test 0.0 <= p <= 1.0
    @test !ismissing(bf)
    @test bf > 0  # Bayes factor must be positive
end

@testitem "betabernoulli returns plot object" begin
    using BayesInteractomics
    using BayesInteractomics: betabernoulli, Protocol, InteractionData
    # StatsPlots is used by the package for plotting, not Plots directly
    using StatsPlots

    m_sample::Matrix{Union{Missing, Float64}} = reshape([1.0, 2.0, 3.0], 1, 3)
    m_control::Matrix{Union{Missing, Float64}} = reshape([7.0, 8.0], 1, 2)

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

    # Test with create_plot=true (explicit request for plot)
    bf, p, plt = betabernoulli(data, 1, 2, 3, create_plot=true)  # n_control=2, n_sample=3

    # Check that a plot object is returned (StatsPlots.Plot is a Plots.Plot)
    @test !ismissing(plt)
    @test plt isa Plots.Plot

    # Test with create_plot=false (default, should return nothing)
    bf2, p2, plt2 = betabernoulli(data, 1, 2, 3)
    @test isnothing(plt2)
    # But BF and p should still be computed
    @test bf2 == bf
    @test p2 == p
end
