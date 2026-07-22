# test/reports/test_methods_mask_aware.jl
#
# Tests for the Methods-tab `<h4>Mask-aware regression (v2b)</h4>` subsection +
# the per-condition `pct_imputed_cells` chip rendering in BOTH `report.html` and
# `differential_report.html`.
#
# Contract: the `AnalysisResult` struct must NOT be modified — `pct_imputed_cells`
# is computed locally inside the report generator (or threaded via a kwarg).
# Test 4 below asserts this invariant.

@testitem "v2b Methods H4 helper renders when mask_aware_regression=true" tags=[:reports, :mask_aware] begin
    using BayesInteractomics

    cfg = CONFIG(
        datafile     = [""],
        control_cols = [Dict(1 => [2])],
        sample_cols  = [Dict(1 => [3])],
        poi          = "X",
        n_controls   = 1,
        n_samples    = 1,
        refID        = 1,
        output       = OutputFiles(mktempdir()),
        mask_aware_regression = true,
    )
    html = BayesInteractomics._methods_mask_aware_regression_block(cfg)

    @test !isempty(html)
    @test occursin("<h4", html)
    @test occursin("Mask-aware regression (v2b)", html)
    @test occursin("var(data[cell])", html)
    # The rendered Methods block documents current behaviour only — it must NOT
    # embed internal evidence-record paths in the user-facing report DOM.
    @test !occursin(".planning", html)
    @test occursin("mask_aware_regression = false", html)
end

@testitem "v2b Methods H4 helper is empty when mask_aware_regression=false" tags=[:reports, :mask_aware] begin
    using BayesInteractomics

    cfg = CONFIG(
        datafile     = [""],
        control_cols = [Dict(1 => [2])],
        sample_cols  = [Dict(1 => [3])],
        poi          = "X",
        n_controls   = 1,
        n_samples    = 1,
        refID        = 1,
        output       = OutputFiles(mktempdir()),
        mask_aware_regression = false,
    )
    html = BayesInteractomics._methods_mask_aware_regression_block(cfg)
    @test html == ""
end

@testitem "v2b mask-aware chip helper renders scalar + Dict + nothing paths" tags=[:reports, :mask_aware] begin
    using BayesInteractomics

    # Scalar path (single-condition report).
    chip_scalar = BayesInteractomics._mask_aware_chip_html(12.5)
    @test occursin("imputed</span>", chip_scalar)
    @test occursin("12.5", chip_scalar)
    @test occursin("badge", chip_scalar)
    @test occursin("text-bg-info", chip_scalar)

    # 0% path — chip MUST render explicitly (not suppressed). Julia's
    # `string(round(0.0; digits=2))` is "0.0" (trailing zeros stripped), so
    # the chip body reads "0.0% imputed" — the explicit-zero contract is met.
    chip_zero = BayesInteractomics._mask_aware_chip_html(0.0)
    @test occursin("0.0% imputed", chip_zero)
    @test occursin("badge", chip_zero)

    # Dict path (differential report — per-condition chips).
    chip_dict = BayesInteractomics._mask_aware_chip_html(Dict(:wt => 5.2, :mut => 15.7))
    @test occursin("wt", chip_dict)
    @test occursin("mut", chip_dict)
    @test count(c -> c == 'b', chip_dict) >= 2  # at least 2 `badge` chips

    # nothing path — empty string (no chip when source is unknown).
    chip_nothing = BayesInteractomics._mask_aware_chip_html(nothing)
    @test chip_nothing == ""
end

@testitem "AnalysisResult struct is NOT modified by mask-aware report rendering" tags=[:reports, :mask_aware] begin
    using BayesInteractomics

    # Contract: pct_imputed_cells is computed locally inside
    # the report generator, NOT added as a struct field. This assertion locks
    # the contract: any future field addition with these names would break the
    # AnalysisResult ctor arity + CACHE_VERSION discipline.
    @test !(:pct_imputed_cells in fieldnames(AnalysisResult))
    @test !(:imputed_mask in fieldnames(AnalysisResult))
    @test !(:is_imputed in fieldnames(AnalysisResult))
end

@testitem "v2b report JSON carries methods.mask_aware block_html + qc_data chip" tags=[:reports, :mask_aware] begin
    using BayesInteractomics
    using DataFrames

    cfg_on = CONFIG(
        datafile     = [""],
        control_cols = [Dict(1 => [2])],
        sample_cols  = [Dict(1 => [3])],
        poi          = "X",
        n_controls   = 1,
        n_samples    = 1,
        refID        = 1,
        output       = OutputFiles(mktempdir()),
        mask_aware_regression = true,
    )

    # Minimal results DataFrame — only columns required by builders we hit here.
    results = DataFrame(
        Protein = ["P1"],
        BF = [1.0],
        posterior_prob = [0.5],
        BFDR = [0.1],
        PEP = [0.5],
        mean_log2FC = [0.0],
        bf_enrichment = [1.0],
        bf_correlation = [1.0],
        bf_detected = [1.0],
        bf_em = [1.0],
        bf_copula = [1.0],
        log2FC_mean = [0.0],
        is_detected = [true],
    )

    # JSON blob must contain the v2b Methods block when mask_aware_regression=true.
    blob_on = BayesInteractomics._build_report_json(results, cfg_on; pct_imputed_cells = 12.34)
    @test occursin("Mask-aware regression (v2b)", blob_on)
    @test occursin("12.34% imputed", blob_on)

    # Same fixture, opt-out — methods block hidden but chip MUST still render
    # (the chip is an observational fact about the data, not a model setting).
    cfg_off = CONFIG(
        datafile     = [""],
        control_cols = [Dict(1 => [2])],
        sample_cols  = [Dict(1 => [3])],
        poi          = "X",
        n_controls   = 1,
        n_samples    = 1,
        refID        = 1,
        output       = OutputFiles(mktempdir()),
        mask_aware_regression = false,
    )
    blob_off = BayesInteractomics._build_report_json(results, cfg_off; pct_imputed_cells = 12.34)
    @test !occursin("Mask-aware regression (v2b)", blob_off)
    @test occursin("12.34% imputed", blob_off)
end
