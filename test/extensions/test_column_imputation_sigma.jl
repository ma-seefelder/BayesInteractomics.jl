# Unit test for column_imputation_sigma helper.
#
# Locks the empirical-σ semantics for the v2b mask-aware regression observation
# factor. The helper lives in the BayesInteractomicsImputationExt
# extension; activation requires `using GLM`. This test file is fully self-
# contained (no test/test_fixtures.jl dependency) so it runs in isolation under
# TestItemRunner.

@testitem "column_imputation_sigma helper" tags=[:extension, :mask_aware] begin
    using GLM
    using BayesInteractomics
    using Statistics: var
    using Test

    # Build a synthetic DropoutFit with 3 columns. Field layout is
    # 8 fields, all positional in the inner ctor.
    fit = BayesInteractomics.DropoutFit(
        [0.1, 0.2, 0.3],                 # rho
        [1.0, 1.0, 1.0],                 # zeta
        ["c1", "c2", "c3"],              # column_names
        10,                              # n_proteins
        [10, 1, 0],                      # n_detections_per_column
        "2026-05-21T00:00:00Z",          # fit_timestamp
        "1.2.0-test",                    # software_version
        "sha256:" * "0"^64,              # dataset_hash
    )

    # 10×3 intensity matrix:
    #   col 1 — all 10 finite values (happy path)
    #   col 2 — 1 finite value + 9 missings (degenerate n=1 path)
    #   col 3 — all missing (degenerate n=0 path)
    intensity_matrix = Matrix{Union{Missing, Float64}}(missing, 10, 3)
    col1_vals = Float64[20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
    intensity_matrix[:, 1] .= col1_vals
    intensity_matrix[3, 2] = 18.5
    # col 3 already all missing

    # --- Behaviour 1: happy path (n=10 finite) → sqrt(var(col_vals)) ---
    σ1 = BayesInteractomics.column_imputation_sigma(fit, 1, intensity_matrix)
    expected_σ1 = sqrt(var(col1_vals))
    @test σ1 ≈ expected_σ1 atol=1e-12

    # Also assert against the docstring-equivalent skipmissing form
    @test σ1 ≈ sqrt(var(collect(skipmissing(intensity_matrix[:, 1])))) atol=1e-12

    # --- Behaviour 2: degenerate n=1 → 0.0 ---
    σ2 = BayesInteractomics.column_imputation_sigma(fit, 2, intensity_matrix)
    @test σ2 == 0.0

    # --- Behaviour 3: degenerate n=0 → 0.0 ---
    σ3 = BayesInteractomics.column_imputation_sigma(fit, 3, intensity_matrix)
    @test σ3 == 0.0

    # --- Behaviour 4: out-of-bounds col index throws BoundsError ---
    @test_throws BoundsError BayesInteractomics.column_imputation_sigma(fit, 0, intensity_matrix)
    @test_throws BoundsError BayesInteractomics.column_imputation_sigma(fit, 4, intensity_matrix)

    # --- Robustness: NaN values in an otherwise-finite column are filtered out ---
    intensity_nan = Matrix{Union{Missing, Float64}}(missing, 10, 3)
    intensity_nan[:, 1] .= [20.0, NaN, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
    σ_nan = BayesInteractomics.column_imputation_sigma(fit, 1, intensity_nan)
    expected_σ_nan = sqrt(var(Float64[20.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]))
    @test σ_nan ≈ expected_σ_nan atol=1e-12
end
