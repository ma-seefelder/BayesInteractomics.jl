# test/differential/test_differential_config_loss_matrix.jl
#
# DifferentialConfig.loss_matrix + validation_candidates_top_n
# constructor validation testitems.
#
# Quick run:
#   julia --project=. -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("test_differential_config_loss_matrix", ti.filename)'

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 1 — defaults populated correctly when no kwargs supplied
# ─────────────────────────────────────────────────────────────────────────────

@testitem "DifferentialConfig defaults loss_matrix to DEFAULT_DIFFERENTIAL_LOSS and validation_candidates_top_n to 20" begin
    using BayesInteractomics
    cfg = DifferentialConfig()
    @test cfg.loss_matrix == DEFAULT_DIFFERENTIAL_LOSS
    @test size(cfg.loss_matrix) == (4, 4)
    @test eltype(cfg.loss_matrix) == Float64
    @test cfg.validation_candidates_top_n == 20
    @test cfg.validation_candidates_top_n isa Int
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 2 — kwarg overrides land and produce non-default matrix
# ─────────────────────────────────────────────────────────────────────────────

@testitem "DifferentialConfig kwarg overrides land for loss_matrix and validation_candidates_top_n" begin
    using BayesInteractomics
    custom = Float64[0 1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0]
    cfg = DifferentialConfig(; loss_matrix = custom, validation_candidates_top_n = 50)
    @test cfg.loss_matrix == custom
    @test cfg.loss_matrix != DEFAULT_DIFFERENTIAL_LOSS
    @test cfg.validation_candidates_top_n == 50

    # Asymmetric custom matrix
    custom2 = Float64[0 5 2 2; 5 0 2 2; 3 3 0 1; 3 3 1 0]
    cfg2 = DifferentialConfig(; loss_matrix = custom2, validation_candidates_top_n = 100)
    @test cfg2.loss_matrix[1, 2] == 5.0
    @test cfg2.loss_matrix[3, 1] == 3.0
    @test cfg2.validation_candidates_top_n == 100
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 3 — loss_matrix validation (5 invalid-input paths)
# ─────────────────────────────────────────────────────────────────────────────

@testitem "DifferentialConfig loss_matrix validation throws ArgumentError on malformed input" begin
    using BayesInteractomics
    # Wrong shape — 3×3
    @test_throws ArgumentError DifferentialConfig(; loss_matrix = ones(3, 3))
    # Wrong shape — 2×2
    @test_throws ArgumentError DifferentialConfig(; loss_matrix = Float64[0 1; 1 0])
    # Wrong shape — 5×5
    @test_throws ArgumentError DifferentialConfig(; loss_matrix = zeros(5, 5))
    # Nonzero diagonal
    bad_diag = Float64[1 1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0]
    @test_throws ArgumentError DifferentialConfig(; loss_matrix = bad_diag)
    # Negative entry
    bad_neg = Float64[0 -1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0]
    @test_throws ArgumentError DifferentialConfig(; loss_matrix = bad_neg)
    # Non-finite (Inf) entry
    bad_inf = Float64[0 Inf 1 1; Inf 0 1 1; 1 1 0 1; 1 1 1 0]
    @test_throws ArgumentError DifferentialConfig(; loss_matrix = bad_inf)
    # Non-finite (NaN) entry
    bad_nan = Float64[0 NaN 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0]
    @test_throws ArgumentError DifferentialConfig(; loss_matrix = bad_nan)
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 4 — validation_candidates_top_n must be > 0
# ─────────────────────────────────────────────────────────────────────────────

@testitem "DifferentialConfig validation_candidates_top_n must be > 0" begin
    using BayesInteractomics
    @test_throws ArgumentError DifferentialConfig(; validation_candidates_top_n = 0)
    @test_throws ArgumentError DifferentialConfig(; validation_candidates_top_n = -1)
    @test_throws ArgumentError DifferentialConfig(; validation_candidates_top_n = -100)
    # Positive values work, including 1 (minimum valid)
    cfg1 = DifferentialConfig(; validation_candidates_top_n = 1)
    @test cfg1.validation_candidates_top_n == 1
    cfg2 = DifferentialConfig(; validation_candidates_top_n = 1000)
    @test cfg2.validation_candidates_top_n == 1000
end
