"""
    test_volcano.jl

Tests for dynamic PEP floor and PEP=0 marker handling in volcano plots.
"""

@testitem "Dynamic PEP-floor: smallest non-zero / 10" begin
    # Given PEP values with zeros and non-zeros, floor = min(nonzero) / 10
    pep_vals = [0.0, 0.001, 0.01, 0.5]
    pep_nonzero = filter(x -> x > 0, pep_vals)
    pep_floor = isempty(pep_nonzero) ? 1e-20 : minimum(pep_nonzero) / 10

    @test pep_floor ≈ 0.0001
    @test pep_floor == 0.001 / 10

    # Clamped values should use dynamic floor
    pep_clamped = clamp.(pep_vals, pep_floor, 1.0)
    @test pep_clamped[1] == pep_floor  # was 0.0, clamped to floor
    @test pep_clamped[2] == 0.001      # unchanged
    @test pep_clamped[3] == 0.01       # unchanged
    @test pep_clamped[4] == 0.5        # unchanged
end

@testitem "Dynamic PEP-floor: all-zero fallback" begin
    # When all PEP values are zero, fallback to 1e-20
    pep_vals = [0.0, 0.0, 0.0]
    pep_nonzero = filter(x -> x > 0, pep_vals)
    pep_floor = isempty(pep_nonzero) ? 1e-20 : minimum(pep_nonzero) / 10

    @test pep_floor == 1e-20
end

@testitem "PEP=0 proteins separated into distinct index set" begin
    pep_vals = [0.0, 0.001, 0.0, 0.5, 0.01]

    # Compute dynamic floor
    pep_nonzero = filter(x -> x > 0, pep_vals)
    pep_floor = isempty(pep_nonzero) ? 1e-20 : minimum(pep_nonzero) / 10

    # Separate PEP=0 indices
    idx_pep_zero = findall(x -> x == 0.0, pep_vals)
    @test idx_pep_zero == [1, 3]
    @test length(idx_pep_zero) == 2

    # Non-zero PEP values clamped with dynamic floor, not 1e-300
    pep_clamped = clamp.(pep_vals, pep_floor, 1.0)
    neg_log_pep = -log10.(pep_clamped)

    # No value should be as extreme as -log10(1e-300) = 300
    @test all(neg_log_pep .< 300)
    # The floor-clamped values should give -log10(0.0001) = 4.0
    @test neg_log_pep[1] ≈ 4.0
    @test neg_log_pep[3] ≈ 4.0
end

@testitem "Dynamic PEP-floor with single non-zero value" begin
    pep_vals = [0.0, 0.0, 0.05]
    pep_nonzero = filter(x -> x > 0, pep_vals)
    pep_floor = isempty(pep_nonzero) ? 1e-20 : minimum(pep_nonzero) / 10

    @test pep_floor ≈ 0.005
    @test length(findall(x -> x == 0.0, pep_vals)) == 2
end
