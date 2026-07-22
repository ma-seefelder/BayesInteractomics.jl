# test/differential/test_decision_risk_diagnosis.jl
#
# Diagnosis tests for the `decision_risk == 0` symptom in the differential report.
#
# Two @testitem assertions covering:
#   A. compute_decision_risk Σ(1 − γ-PEP) telemetry hook
#      — exercised via ENV["GSD_DECISION_RISK_TELEMETRY"] = "1" + @test_logs
#   B. count(>(0), decision_risk_min) > 0 on k=3 fixture
#      — with the outer-join + suffixed JS keys in place, the wide-table
#        `:decision_risk_min` column carries finite positive per-pair minima
#        for all 6 fixture rows; the all-zero symptom was a consequence of a
#        JS key mismatch, NOT a genuine degenerate-uniform branch firing in
#        `compute_decision_risk`.
#
# The γ-PEP renormalisation 1e-12 threshold is untouched (telemetry is
# read-only / observational); loss_matrix_default stays as a column on results,
# not a struct field. BMA = Copula + 3c-EM; FDR = BFDR / PEP / local_fdr.
#
# Filter command:
#   julia --project=. --threads=16 -e 'using TestItemRunner;
#     @run_package_tests filter=ti->occursin("test_decision_risk_diagnosis", ti.filename)'

using TestItemRunner

# ─────────────────────────────────────────────────────────────────────────────
# Stub A — Σ(1 − γ-PEP) telemetry hook fires when ENV var is set
# ─────────────────────────────────────────────────────────────────────────────
@testitem "compute_decision_risk Σ(1 − γ-PEP) telemetry hook" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames
    using Logging
    using Test

    fx = DifferentialFixtures.create_three_condition_result()

    # Capture @info logs from the telemetry block. The hook fires per-row
    # (maxlog=Inf inside the ENV-gated branch), so on a 6-row k=3 fixture with
    # 3 pairs we expect at least 6 invocations of the telemetry message.
    #
    # We use Logging.with_logger + TestLogger (the canonical TestItemRunner
    # idiom for capturing @info records) rather than @test_logs, because
    # @test_logs matches a strict log sequence while TestLogger collects all
    # records into a vector we can count + inspect.
    logger = TestLogger(min_level = Logging.Info)
    withenv("GSD_DECISION_RISK_TELEMETRY" => "1") do
        Logging.with_logger(logger) do
            differential_analysis(
                conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
                contrasts  = :all_pairs,
            )
        end
    end

    telemetry_records = filter(r -> occursin("decision_risk telemetry", r.message), logger.logs)
    @test !isempty(telemetry_records)
    @test length(telemetry_records) >= 6   # at least one per fixture row × pair (6 rows × 3 pairs = 18 expected)

    # Each telemetry record carries Σ, degenerate flag, four pep_<class> values.
    first_rec = telemetry_records[1]
    @test haskey(first_rec.kwargs, :Σ)
    @test haskey(first_rec.kwargs, :degenerate)
    @test haskey(first_rec.kwargs, :pep_gained)
    @test haskey(first_rec.kwargs, :pep_reduced)
    @test haskey(first_rec.kwargs, :pep_unchanged)
    @test haskey(first_rec.kwargs, :pep_both_negative)
    @test isa(first_rec.kwargs[:Σ], Float64)
    @test isa(first_rec.kwargs[:degenerate], Bool)

    # Production-side: when ENV var is UNSET, the telemetry block should NOT
    # fire (no allocation overhead beyond a single get(ENV, ...) call).
    logger_off = TestLogger(min_level = Logging.Info)
    withenv("GSD_DECISION_RISK_TELEMETRY" => nothing) do
        Logging.with_logger(logger_off) do
            differential_analysis(
                conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
                contrasts  = :all_pairs,
            )
        end
    end
    @test isempty(filter(r -> occursin("decision_risk telemetry", r.message), logger_off.logs))
end

# ─────────────────────────────────────────────────────────────────────────────
# Stub B — count(>(0), decision_risk_min) > 0 on k=3 fixture
# ─────────────────────────────────────────────────────────────────────────────
@testitem "decision_risk_min variation > 0 on k=3 fixture" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    fx   = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        contrasts  = :all_pairs,
    )
    @test :decision_risk_min in propertynames(diff.results)

    # With the outer-join + suffixed JS keys in place, every fixture row has
    # a finite positive decision_risk_min (smallest expected-loss across pairs).
    # The all-zero symptom was a JS key mismatch — `compute_decision_risk`
    # itself was always producing correct non-zero values per-pair. This
    # assertion confirms that.
    drm = diff.results.decision_risk_min
    @test count(>(0), skipmissing(drm)) > 0
    @test count(>(0), skipmissing(drm)) == nrow(diff.results)  # all 6 rows positive
    @test !any(ismissing, drm)
    @test !any(isnan, skipmissing(drm))
end
