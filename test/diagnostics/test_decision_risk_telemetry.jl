# test/diagnostics/test_decision_risk_telemetry.jl
#
# Regression tests for the decision_risk end-of-call summary telemetry.
#
# Asserts the end-of-call summary `@info "[decision_risk summary]"`
# emitted by `compute_decision_risk` (and inherited by `compute_decision_risk!`)
# fires exactly once per call, carries the expected fields (n, degenerate,
# frac_degen, Z_min, Z_median, Z_p95, Z_max), and that the `degenerate=N` count
# matches a hand-constructed degenerate-row fixture.
#
# The telemetry is INFORMATIONAL ONLY; no algorithm change. The summary below
# is the canonical signature that future production datasets will reuse if the
# degenerate-row rate crosses 50%.
#
# Unchanged contracts:
#   - γ-PEP renormalisation 1e-12 threshold
#   - DEFAULT_DIFFERENTIAL_LOSS
#   - DECISION_RISK_ACTIONS sentinel
#   - per-row ENV-gated @info at decision_risk.jl (independent diagnostic channel)
#
# Filter command:
#   julia --project=. --threads=4 -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("test_decision_risk_telemetry", ti.filename)'

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 1 — summary @info fires exactly once per compute_decision_risk! call
# ─────────────────────────────────────────────────────────────────────────────

@testitem "summary @info emitted once per compute_decision_risk! call" begin
    using BayesInteractomics
    using BayesInteractomics: GAINED, REDUCED, UNCHANGED, BOTH_NEGATIVE
    using DataFrames
    using Logging
    using Test

    # 10-row non-degenerate fixture: each row has a clear winner in one of the
    # four γ-PEPs so the renormalised posterior is well-conditioned (Z >> 1e-12).
    df = DataFrame(
        pep_gained        = [0.05, 0.10, 0.80, 0.90, 0.20, 0.30, 0.40, 0.60, 0.05, 0.15],
        pep_reduced       = [0.80, 0.85, 0.05, 0.10, 0.40, 0.50, 0.70, 0.20, 0.95, 0.85],
        pep_unchanged     = [0.90, 0.95, 0.85, 0.30, 0.05, 0.10, 0.20, 0.50, 0.50, 0.45],
        pep_both_negative = [0.95, 0.80, 0.90, 0.85, 0.85, 0.90, 0.10, 0.80, 0.85, 0.60],
        classification    = [GAINED, GAINED, REDUCED, REDUCED, UNCHANGED,
                             UNCHANGED, BOTH_NEGATIVE, GAINED, REDUCED, REDUCED],
    )

    logger = TestLogger(min_level = Logging.Info)
    Logging.with_logger(logger) do
        compute_decision_risk!(df)
    end

    summary_records = filter(r -> occursin("decision_risk summary", r.message),
                             logger.logs)
    @test length(summary_records) == 1
    # All 10 rows participated in the 4-action loss matrix → no
    # CONDITION_A/B_SPECIFIC short-circuit.
    @test occursin("n=10", summary_records[1].message)
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 2 — summary log carries all required fields
# ─────────────────────────────────────────────────────────────────────────────

@testitem "summary log carries all required fields" begin
    using BayesInteractomics
    using BayesInteractomics: GAINED, REDUCED, UNCHANGED, BOTH_NEGATIVE
    using DataFrames
    using Logging
    using Test

    df = DataFrame(
        pep_gained        = [0.05, 0.10, 0.80, 0.90, 0.20],
        pep_reduced       = [0.80, 0.85, 0.05, 0.10, 0.40],
        pep_unchanged     = [0.90, 0.95, 0.85, 0.30, 0.05],
        pep_both_negative = [0.95, 0.80, 0.90, 0.85, 0.85],
        classification    = [GAINED, GAINED, REDUCED, REDUCED, UNCHANGED],
    )

    logger = TestLogger(min_level = Logging.Info)
    Logging.with_logger(logger) do
        compute_decision_risk!(df)
    end

    summary_records = filter(r -> occursin("decision_risk summary", r.message),
                             logger.logs)
    @test length(summary_records) == 1
    msg = summary_records[1].message

    @test occursin("n=",          msg)
    @test occursin("degenerate=", msg)
    @test occursin("frac_degen=", msg)
    @test occursin("Z_min=",      msg)
    @test occursin("Z_median=",   msg)
    @test occursin("Z_p95=",      msg)
    @test occursin("Z_max=",      msg)
end

# ─────────────────────────────────────────────────────────────────────────────
# Testitem 3 — degenerate-row count matches the z_history accumulator
# ─────────────────────────────────────────────────────────────────────────────

@testitem "degenerate-row count matches z_history" begin
    using BayesInteractomics
    using BayesInteractomics: GAINED, REDUCED, UNCHANGED, BOTH_NEGATIVE
    using DataFrames
    using Logging
    using Test

    # 10-row fixture: rows 1, 4, 7 force Σ(1 − γ-PEP) ≈ 0 (each PEP ≈ 1.0 →
    # degenerate uniform-fallback). The remaining 7 rows have clear posterior
    # mass. Expected `degenerate=3` in the summary.
    df = DataFrame(
        pep_gained        = [1.0,  0.05, 0.10, 1.0,  0.30, 0.40, 1.0,  0.60, 0.05, 0.15],
        pep_reduced       = [1.0,  0.80, 0.85, 1.0,  0.50, 0.70, 1.0,  0.20, 0.95, 0.85],
        pep_unchanged     = [1.0,  0.90, 0.95, 1.0,  0.10, 0.20, 1.0,  0.50, 0.50, 0.45],
        pep_both_negative = [1.0,  0.95, 0.80, 1.0,  0.90, 0.10, 1.0,  0.80, 0.85, 0.60],
        classification    = [GAINED, GAINED, REDUCED, REDUCED, UNCHANGED,
                             UNCHANGED, BOTH_NEGATIVE, GAINED, REDUCED, REDUCED],
    )

    logger = TestLogger(min_level = Logging.Info)
    Logging.with_logger(logger) do
        compute_decision_risk!(df)
    end

    summary_records = filter(r -> occursin("decision_risk summary", r.message),
                             logger.logs)
    @test length(summary_records) == 1
    msg = summary_records[1].message

    # Parse `degenerate=N` value out of the formatted message.
    m = match(r"degenerate=(\d+)", msg)
    @test m !== nothing
    n_degen = parse(Int, m.captures[1])
    @test n_degen == 3

    # Sanity: `n=10` (no CONDITION_A/B_SPECIFIC rows in this fixture).
    @test occursin("n=10", msg)

    # Sanity: `frac_degen` = 3/10 = 0.3 (4-digit rounding → "0.3").
    fmd = match(r"frac_degen=([0-9.]+)", msg)
    @test fmd !== nothing
    @test parse(Float64, fmd.captures[1]) ≈ 0.3 atol = 1e-4

    # Sanity: degenerate rows fall back to uniform → decision_risk == 2.75
    # exactly (uniform-fallback signature).
    @test df.decision_risk[1] ≈ 2.75 atol = 1e-12
    @test df.decision_risk[4] ≈ 2.75 atol = 1e-12
    @test df.decision_risk[7] ≈ 2.75 atol = 1e-12
end
