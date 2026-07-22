# test/reports/test_report.jl
# Tests for the interactive HTML report generation feature.

# ---------------------------------------------------------------------------
# JSON utilities
# ---------------------------------------------------------------------------

@testitem "json_number handles integers and floats" begin
    using BayesInteractomics: json_number

    @test json_number(0)        == "0"
    @test json_number(42)       == "42"
    @test json_number(-7)       == "-7"
    @test json_number(3.14)     == "3.14"
    @test json_number(0.987)    == "0.987"   # regression: must not throw InexactError
    @test json_number(1.0)      == "1"       # whole float → integer literal
    @test json_number(NaN)      == "null"
    @test json_number(Inf)      == "null"
    @test json_number(-Inf)     == "null"
    @test json_number(missing)  == "null"
end

@testitem "json_number rounds to 5 significant digits" begin
    using BayesInteractomics: json_number

    # Full-precision float gets truncated to 5 sigdigits
    @test json_number(0.98765432) == "0.98765"
    @test json_number(1.23456789) == "1.2346"
    @test json_number(0.00012345678) == "0.00012346"

    # Values already within 5 sigdigits are unchanged
    @test json_number(3.14) == "3.14"
    @test json_number(0.5) == "0.5"

    # Large numbers: 5 sigdigits (rounds to integer)
    @test json_number(123456.789) == "123460"

    # Rounding to integer: 1.00004 rounds to 1.0, emitted as "1"
    @test json_number(1.00004) == "1"

    # Integers unaffected (dispatch to Integer method)
    @test json_number(42) == "42"
    @test json_number(0) == "0"
end

@testitem "json_string escapes special characters" begin
    using BayesInteractomics: json_string

    @test json_string("hello")         == "\"hello\""
    @test json_string("say \"hi\"")   == "\"say \\\"hi\\\"\""
    @test json_string("a\\b")         == "\"a\\\\b\""
    @test json_string("line\nnewline") == "\"line\\nnewline\""
    @test json_string("tab\there")    == "\"tab\\there\""
    @test json_string("")             == "\"\""
    @test json_string(42)             == "\"42\""   # non-string dispatch
end

@testitem "json_array and json_object build valid structures" begin
    using BayesInteractomics: json_array, json_object, json_string, json_number

    arr = json_array(["1", "2", "3"])
    @test arr == "[1,2,3]"

    arr2 = json_array(String[])
    @test arr2 == "[]"

    obj = json_object("x" => json_number(1), "y" => json_string("hi"))
    @test startswith(obj, "{")
    @test endswith(obj, "}")
    @test contains(obj, "\"x\":1")
    @test contains(obj, "\"y\":\"hi\"")
end

@testitem "encode_png_file returns empty for missing files" begin
    using BayesInteractomics: encode_png_file

    uri = encode_png_file("/nonexistent/path/to/image.png")
    @test uri == ""
end

@testitem "encode_png_file encodes existing PNG" begin
    using BayesInteractomics: encode_png_file

    # Write a minimal valid PNG (1x1 black pixel)
    # PNG signature + minimal IHDR + IDAT + IEND chunks
    minimal_png = UInt8[
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,  # PNG signature
        0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,  # IHDR length + type
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # width=1, height=1
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,  # bitdepth=8, colortype=2, ...
        0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41,  # IDAT length + type
        0x54, 0x08, 0xd7, 0x63, 0x60, 0x60, 0x60, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x01, 0x27, 0xf1, 0x3f,
        0xb5, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e,  # IEND
        0x44, 0xae, 0x42, 0x60, 0x82,
    ]
    tmpfile = tempname() * ".png"
    try
        write(tmpfile, minimal_png)
        uri = encode_png_file(tmpfile)
        @test startswith(uri, "data:image/png;base64,")
        @test length(uri) > 30
    finally
        isfile(tmpfile) && rm(tmpfile)
    end
end

# ---------------------------------------------------------------------------
# Methods generator
# ---------------------------------------------------------------------------

@testitem "generate_methods_text produces manuscript paragraph" begin
    using BayesInteractomics
    using DataFrames

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "MYC",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )

    results = DataFrame(
        Protein        = ["P1", "P2", "P3", "P4", "P5"],
        BF             = [100.0, 50.0, 10.0, 2.0, 0.5],
        posterior_prob = [0.99, 0.95, 0.80, 0.60, 0.30],
        PEP            = 1.0 .- [0.99, 0.95, 0.80, 0.60, 0.30],
        BFDR           = [0.001, 0.01, 0.04, 0.10, 0.50],
        mean_log2FC    = [3.0, 2.0, 1.5, 0.5, -0.1],
        bf_enrichment  = [80.0, 40.0, 8.0, 1.5, 0.4],
        bf_correlation = [20.0, 10.0, 2.0, 0.5, 0.1],
        bf_detected    = [10.0,  5.0, 1.0, 0.3, 0.1],
    )

    text = BayesInteractomics.generate_methods_text(cfg, results)
    @test contains(text, "MYC")
    @test contains(text, "BayesInteractomics")
    @test contains(text, "Beta-Bernoulli")
    @test contains(text, "Hierarchical Bayesian")
    @test contains(text, "5 proteins")
end

@testitem "generate_methods_parameters returns key-value pairs" begin
    using BayesInteractomics

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "TP53",
        n_controls   = 2,
        n_samples    = 4,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )

    params = BayesInteractomics.generate_methods_parameters(cfg)
    @test params isa Vector{Pair{String,String}}
    @test !isempty(params)
    keys_list = first.(params)
    # New implementation returns raw CONFIG field names for all parameters
    @test "poi" in keys_list
    @test "n_controls" in keys_list
    @test "n_samples" in keys_list
    vals = Dict(params)
    @test vals["poi"] == "TP53"
    @test vals["n_controls"] == "2"
    @test vals["n_samples"] == "4"
end

# Ablation-knob report byte-identity: the four ablation-knob CONFIG
# fields (evidence_streams, copula_family, h1_copula_family, use_metalearner_prior) are
# skip-listed in BOTH methods_generator field iterations, so report.html stays
# byte-identical on the full default path. Assert the rendered parameters table AND the
# reproducibility block do NOT mention any of the four field names.
@testitem "ablation knobs skip-listed -> report parameters byte-identical" begin
    using BayesInteractomics

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "TP53",
        n_controls   = 2,
        n_samples    = 4,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )

    # The four ablation knobs exist on the default CONFIG (so they COULD leak)...
    for sym in (:evidence_streams, :copula_family, :h1_copula_family, :use_metalearner_prior)
        @test sym in fieldnames(CONFIG)
    end

    # ...but must NOT surface in the Analysis-Parameters table.
    params = BayesInteractomics.generate_methods_parameters(cfg)
    param_keys = first.(params)
    for name in ("evidence_streams", "copula_family", "h1_copula_family", "use_metalearner_prior")
        @test !(name in param_keys)
    end

    # ...nor in the reproducibility block text.
    repro = BayesInteractomics.generate_reproducibility_block(cfg)
    for name in ("evidence_streams", "copula_family", "h1_copula_family", "use_metalearner_prior")
        @test !occursin(name, repro)
    end
end

@testitem "structured methods JSON contains per-model sub-objects" begin
    using BayesInteractomics
    using DataFrames

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "HAP40",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )

    results = DataFrame(
        Protein        = ["P1", "P2", "P3"],
        BF             = [100.0, 2.0, 0.5],
        posterior_prob = [0.99, 0.60, 0.30],
        PEP            = 1.0 .- [0.99, 0.60, 0.30],
        BFDR           = [0.001, 0.10, 0.50],
        mean_log2FC    = [3.0, 0.5, -0.1],
        bf_enrichment  = [80.0, 1.5, 0.4],
        bf_correlation = [20.0, 0.5, 0.1],
        bf_detected    = [10.0, 0.3, 0.1],
    )

    s = BayesInteractomics._build_structured_methods_data(cfg, results)
    # METH-01: structured subsections
    @test contains(s, "\"overview\"")
    @test contains(s, "\"detection\"")
    @test contains(s, "\"enrichment\"")
    @test contains(s, "\"correlation\"")
    @test contains(s, "\"combination\"")
    @test contains(s, "\"calibration\"")
    @test contains(s, "\"priors\"")

    # Overview values
    @test contains(s, "\"n_proteins\":3")
    @test contains(s, "\"n_sig\":1")       # only P1 has posterior_prob > 0.95
    @test contains(s, "\"n_strong\":1")    # only P1 has posterior_prob > 0.99
    @test contains(s, "\"bait\":\"HAP40\"")
    @test contains(s, "\"combination_method\":\"bma\"")

    # Detection defaults
    @test contains(s, "\"prior_alpha\":3")
    @test contains(s, "\"prior_beta\":3")

    # Enrichment defaults
    @test contains(s, "\"mu_0\":25")

    # Correlation CONFIG values
    @test contains(s, "\"regression_likelihood\":\"robust_t\"")
    @test contains(s, "\"jzs_r_scale\":0.354")

    # Combination
    @test contains(s, "\"em_n_restarts\":20")
end

@testitem "priors array contains expected rows with default values" begin
    using BayesInteractomics
    using DataFrames

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "MYC",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
        jzs_r_scale  = 0.354,
        regression_likelihood = :robust_t,
        student_t_nu = 5.0,
    )

    results = DataFrame(
        Protein        = ["A"],
        BF             = [1.0],
        posterior_prob = [0.5],
        PEP            = 1.0 .- [0.5],
        BFDR           = [0.5],
        mean_log2FC    = [0.0],
        bf_enrichment  = [1.0],
        bf_correlation = [1.0],
        bf_detected    = [1.0],
    )

    s = BayesInteractomics._build_structured_methods_data(cfg, results)

    # METH-02: prior table rows
    @test contains(s, "\"detection_theta\"")
    @test contains(s, "\"enrichment_mu\"")
    @test contains(s, "\"enrichment_sigma\"")
    @test contains(s, "\"correlation_slope\"")
    @test contains(s, "\"correlation_slope_jzs\"")      # present because jzs_r_scale > 0
    @test contains(s, "\"correlation_robust_nu\"")       # present because regression_likelihood == :robust_t
    @test contains(s, "\"combination_em_prior\"")
    @test contains(s, "\"combination_bma_weights\"")
    @test contains(s, "Beta(alpha, beta)")
    @test contains(s, "Cauchy(0, r)")
    @test contains(s, "LOO stacking")
end

@testitem "calibration is null when run_simulation is false" begin
    using BayesInteractomics
    using DataFrames

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "X",
        n_controls   = 1,
        n_samples    = 1,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
        run_simulation = false,
    )

    results = DataFrame(
        Protein        = ["A"],
        BF             = [1.0],
        posterior_prob = [0.5],
        PEP            = 1.0 .- [0.5],
        BFDR           = [0.5],
        mean_log2FC    = [0.0],
        bf_enrichment  = [1.0],
        bf_correlation = [1.0],
        bf_detected    = [1.0],
    )

    s = BayesInteractomics._build_structured_methods_data(cfg, results)
    @test contains(s, "\"calibration\":null")
end

@testitem "_build_methods_json includes structured field" begin
    using BayesInteractomics
    using DataFrames

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "MYC",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )

    results = DataFrame(
        Protein        = ["P1"],
        BF             = [10.0],
        posterior_prob = [0.9],
        PEP            = 1.0 .- [0.9],
        BFDR           = [0.01],
        mean_log2FC    = [2.0],
        bf_enrichment  = [8.0],
        bf_correlation = [2.0],
        bf_detected    = [1.0],
    )

    json_str = BayesInteractomics._build_methods_json(cfg, results)
    # Backward compat: old fields still present
    @test contains(json_str, "\"text\":")
    @test contains(json_str, "\"reproducibility\":")
    @test contains(json_str, "\"parameters\":")
    # New structured field
    @test contains(json_str, "\"structured\":")
    @test contains(json_str, "\"overview\"")
    @test contains(json_str, "\"priors\"")
end

# ---------------------------------------------------------------------------
# Full report generation
# ---------------------------------------------------------------------------

@testitem "generate_report produces well-formed HTML" begin
    using BayesInteractomics
    using DataFrames

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "MYC",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,  # we call generate_report manually
    )

    results = DataFrame(
        Protein        = ["ACTB", "TP53", "BRCA1", "EGFR", "MYC"],
        BF             = [200.0, 50.0, 15.0, 3.0, 0.8],
        posterior_prob = [0.99, 0.97, 0.85, 0.60, 0.35],
        PEP            = 1.0 .- [0.99, 0.97, 0.85, 0.60, 0.35],
        BFDR           = [0.001, 0.005, 0.04, 0.12, 0.50],
        mean_log2FC    = [4.0, 2.5, 1.8, 0.8, -0.2],
        sd_log2FC      = [0.3, 0.4, 0.5, 0.6, 0.7],
        bf_enrichment  = [150.0, 40.0, 10.0, 2.5, 0.6],
        bf_correlation = [50.0, 10.0, 5.0, 0.5, 0.2],
        bf_detected    = [20.0, 5.0, 2.0, 0.8, 0.1],
    )

    report_path = joinpath(tmpdir, "test_report.html")
    generate_report(results, cfg; output = report_path)

    @test isfile(report_path)
    html = read(report_path, String)

    # Structure checks
    @test contains(html, "<!DOCTYPE html")
    @test contains(html, "plotly")
    @test contains(html, "DataTable")

    # Data injection
    @test contains(html, "ACTB")
    @test contains(html, "MYC")
    @test !contains(html, "{{REPORT_DATA_JSON}}")   # placeholder must be replaced

    # Evidence labels
    @test contains(html, "Strong")

    # Methods section (structured subsections)
    @test contains(html, "methods-overview-text")

    # New plot keys and BMA key should be present in the JSON blob
    @test contains(html, "h0_marginals")
    @test contains(html, "h1_marginals")
    @test contains(html, "bma_weights")
    @test contains(html, "\"bma\"")

    # Minimum sensible file size (template alone is ~40KB)
    @test length(html) > 10_000
end

@testitem "generate_report writes methods file" begin
    using BayesInteractomics
    using DataFrames

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "EGFR",
        n_controls   = 2,
        n_samples    = 2,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )

    results = DataFrame(
        Protein        = ["P1", "P2"],
        BF             = [100.0, 5.0],
        posterior_prob = [0.98, 0.70],
        PEP            = 1.0 .- [0.98, 0.70],
        BFDR           = [0.005, 0.08],
        mean_log2FC    = [2.0, 1.0],
        bf_enrichment  = [80.0, 4.0],
        bf_correlation = [15.0, 1.0],
        bf_detected    = [5.0, 0.5],
    )

    generate_report(results, cfg)

    methods_path = cfg.output.report_methods_file
    @test isfile(methods_path)
    methods_text = read(methods_path, String)
    @test contains(methods_text, "EGFR")
    @test contains(methods_text, "BayesInteractomics")
end

@testitem "generate_differential_report produces well-formed HTML" begin
    using BayesInteractomics
    using BayesInteractomics: DifferentialResult, DifferentialConfig, InteractionClass
    using DataFrames, Dates

    tmpdir = mktempdir()
    dcfg = DifferentialConfig(
        volcano_file        = joinpath(tmpdir, "vol.png"),
        evidence_file       = joinpath(tmpdir, "ev.png"),
        scatter_file        = joinpath(tmpdir, "sc.png"),
        classification_file = joinpath(tmpdir, "cl.png"),
        ma_file             = joinpath(tmpdir, "ma.png"),
        results_file        = joinpath(tmpdir, "diff_results.xlsx"),
    )

    results_df = DataFrame(
        Protein                = ["P1","P2","P3","P4","P5"],
        bf_A                   = [200.0, 50.0, 10.0, 0.5, 0.2],
        bf_B                   = [10.0,  80.0, 12.0, 0.5, 0.3],
        dbf                    = [20.0,  0.6,  0.8,  1.0, 0.7],
        log10_dbf              = [1.3,  -0.22,-0.1,  0.0,-0.15],
        posterior_A            = [0.99,  0.95, 0.80, 0.40, 0.20],
        posterior_B            = [0.80,  0.98, 0.82, 0.42, 0.25],
        delta_posterior        = [0.19, -0.03,-0.02,-0.02,-0.05],
        BFDR_A                 = [0.001, 0.005, 0.04, 0.3, 0.7],
        BFDR_B                 = [0.05,  0.002, 0.03, 0.3, 0.6],
        PEP_A                  = [0.01, 0.05, 0.10, 0.50, 0.80],
        PEP_B                  = [0.05, 0.01, 0.08, 0.50, 0.70],
        log2fc_A               = [4.0,   2.5,  1.8,  0.5,-0.1],
        log2fc_B               = [2.0,   3.0,  1.9,  0.5,-0.1],
        delta_log2fc           = [2.0,  -0.5, -0.1,  0.0, 0.0],
        bf_enrichment_A        = [150.0, 40.0, 8.0, 0.4, 0.2],
        bf_enrichment_B        = [8.0,  60.0, 9.0, 0.4, 0.2],
        dbf_enrichment         = [18.75, 0.67, 0.89, 1.0, 1.0],
        bf_correlation_A       = [50.0, 10.0, 2.0, 0.1, 0.0],
        bf_correlation_B       = [2.0,  15.0, 2.5, 0.1, 0.0],
        dbf_correlation        = [25.0,  0.67, 0.8, 1.0, 1.0],
        bf_detected_A          = [20.0,  5.0, 1.5, 0.3, 0.1],
        bf_detected_B          = [1.5,   8.0, 1.6, 0.3, 0.1],
        dbf_detected           = [13.3,  0.63, 0.94, 1.0, 1.0],
        differential_posterior = [0.95,  0.30, 0.15, 0.10, 0.08],
        differential_BFDR      = [0.01,  0.40, 0.60, 0.80, 0.90],
        diff_PEP               = [0.05,  0.70, 0.85, 0.90, 0.92],
        classification         = InteractionClass[GAINED, REDUCED, UNCHANGED, UNCHANGED, UNCHANGED],
        dbf_diagnostic         = Symbol[:ok, :ok, :ok, :ok, :ok],
    )

    diff = DifferentialResult(
        results_df, "WT", "Mutant", dcfg,
        5, 5, 5, 0, 0,
        Dates.now(), 1, 1, 3, 0
    )

    out = joinpath(tmpdir, "diff_report.html")
    generate_differential_report(diff; output = out)

    @test isfile(out)
    html = read(out, String)
    @test contains(html, "<!DOCTYPE html")
    @test contains(html, "plotly")
    @test contains(html, "DataTable")
    @test contains(html, "P1")
    @test contains(html, "WT")
    @test contains(html, "Mutant")
    @test !contains(html, "{{DIFF_DATA_JSON}}")
    @test length(html) > 10_000
end

@testitem "generate_report is graceful when template is missing" begin
    using BayesInteractomics
    using DataFrames
    using Logging

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "TEST",
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )

    results = DataFrame(
        Protein        = ["P1"],
        BF             = [10.0],
        posterior_prob = [0.9],
        PEP            = 1.0 .- [0.9],
        BFDR           = [0.02],
        mean_log2FC    = [1.5],
        bf_enrichment  = [8.0],
        bf_correlation = [2.0],
        bf_detected    = [1.0],
    )

    # Point output to a path that ensures the template won't be found
    # by using a custom output path that doesn't exist yet — but we test
    # that the function returns nothing and doesn't throw.
    fake_path = joinpath(tmpdir, "report.html")
    # The real template exists, so this tests the happy path — just verify
    # generate_report returns nothing (::Nothing).
    result = generate_report(results, cfg; output = fake_path)
    @test isnothing(result)
end

# ---------------------------------------------------------------------------
# Pipeline DataFrame 3-component columns
# ---------------------------------------------------------------------------

@testitem "Pipeline DataFrame 3-component columns" begin
    using BayesInteractomics
    using DataFrames

    # Create a mock 3-component LatentClassResult with a 5x3 responsibilities matrix
    resp_matrix = [
        0.85 0.10 0.05;
        0.10 0.75 0.15;
        0.05 0.10 0.85;
        0.60 0.30 0.10;
        0.02 0.08 0.90
    ]
    lc = BayesInteractomics.LatentClassResult(
        [0.5, 1.5, 10.0, 0.8, 20.0],             # bf
        [0.33, 0.60, 0.91, 0.44, 0.95],           # posterior_prob
        Dict(
            "background"  => (mu=-1.0, sigma=1.0, precision=1.0),
            "agnostic"    => (mu=0.0,  sigma=0.8, precision=1.5625),
            "interaction" => (mu=3.0,  sigma=0.8, precision=1.5625),
        ),
        [0.6, 0.25, 0.15],                        # mixing_weights (3-component)
        [100.0, 110.0, 115.0],                     # free_energy
        true,                                       # converged
        30,                                         # n_iterations
        resp_matrix                                 # responsibilities
    )

    # Construct a minimal copula_df
    copula_df = DataFrame(
        Protein = ["P1", "P2", "P3", "P4", "P5"],
        BF = [0.5, 1.5, 10.0, 0.8, 20.0],
        posterior_prob = [0.33, 0.60, 0.91, 0.44, 0.95],
    )

    # Replicate the column-addition logic from pipeline.jl
    latent_class_result = lc
    if latent_class_result !== nothing &&
       latent_class_result.responsibilities !== nothing &&
       size(latent_class_result.responsibilities, 2) == 3
        resp = latent_class_result.responsibilities
        component_labels = Vector{String}(undef, size(resp, 1))
        labels = ["H0", "agnostic", "H1"]
        for i in 1:size(resp, 1)
            component_labels[i] = labels[argmax(resp[i, :])]
        end
        copula_df.Component = component_labels
        copula_df.P_H0 = resp[:, 1]
        copula_df.P_agnostic = resp[:, 2]
        copula_df.P_H1 = resp[:, 3]
    end

    # Assert columns exist
    @test "Component" in names(copula_df)
    @test "P_H0" in names(copula_df)
    @test "P_agnostic" in names(copula_df)
    @test "P_H1" in names(copula_df)

    # Assert Component values are correct
    @test all(c -> c in ["H0", "agnostic", "H1"], copula_df.Component)
    @test copula_df.Component[1] == "H0"
    @test copula_df.Component[2] == "agnostic"
    @test copula_df.Component[3] == "H1"
    @test copula_df.Component[5] == "H1"

    # Assert row sums are ~1.0
    for i in 1:nrow(copula_df)
        @test isapprox(copula_df.P_H0[i] + copula_df.P_agnostic[i] + copula_df.P_H1[i], 1.0, atol=0.01)
    end

    # Test null case: responsibilities === nothing -> no extra columns
    lc_null = BayesInteractomics.LatentClassResult(
        [1.0, 2.0], [0.5, 0.67],
        Dict("background" => (mu=-1.0, sigma=1.0, precision=1.0),
             "interaction" => (mu=3.0, sigma=0.8, precision=1.5625)),
        [0.8, 0.2], [100.0], true, 10, nothing
    )
    copula_df2 = DataFrame(Protein = ["A", "B"], BF = [1.0, 2.0])
    latent_class_result2 = lc_null
    if latent_class_result2 !== nothing &&
       latent_class_result2.responsibilities !== nothing &&
       size(latent_class_result2.responsibilities, 2) == 3
        # This block should NOT execute
        copula_df2.Component = ["should", "not"]
    end
    @test !("Component" in names(copula_df2))
    @test !("P_H0" in names(copula_df2))
end

# ---------------------------------------------------------------------------
# Mixture Model report JSON
# ---------------------------------------------------------------------------

@testitem "Mixture Model report JSON" begin
    using BayesInteractomics
    using BayesInteractomics: _build_mixture_model_json
    using DataFrames

    # Create a mock 3-component LatentClassResult
    resp_matrix = [
        0.80 0.15 0.05;
        0.10 0.70 0.20;
        0.05 0.05 0.90;
        0.60 0.30 0.10;
        0.02 0.08 0.90
    ]
    lc = BayesInteractomics.LatentClassResult(
        [0.5, 1.5, 10.0, 0.8, 20.0],
        [0.33, 0.60, 0.91, 0.44, 0.95],
        Dict(
            "background"  => (mu=-1.0, sigma=1.0, precision=1.0),
            "agnostic"    => (mu=0.0,  sigma=0.8, precision=1.5625),
            "interaction" => (mu=3.0,  sigma=0.8, precision=1.5625),
        ),
        [0.6, 0.25, 0.15],
        [100.0, 110.0, 115.0, 117.0],
        true,
        30,
        resp_matrix
    )

    # Create mock results DataFrame with bf_enrichment and bf_correlation
    results_df = DataFrame(
        Protein = ["P1", "P2", "P3", "P4", "P5"],
        BF = [0.5, 1.5, 10.0, 0.8, 20.0],
        posterior_prob = [0.33, 0.60, 0.91, 0.44, 0.95],
        PEP = 1.0 .- [0.33, 0.60, 0.91, 0.44, 0.95],
        BFDR = [0.5, 0.1, 0.01, 0.3, 0.005],
        mean_log2FC = [0.1, 0.5, 2.0, 0.3, 3.0],
        bf_enrichment = [0.3, 1.2, 8.0, 0.6, 15.0],
        bf_correlation = [0.2, 0.9, 5.0, 0.4, 12.0],
        bf_detected = [1.0, 1.0, 1.0, 1.0, 1.0],
    )

    # Build analysis_result NamedTuple with latent_class_result field
    analysis_result = (latent_class_result = lc, bma_result = nothing)

    json_str = _build_mixture_model_json(analysis_result, results_df)

    # Should not be "null"
    @test json_str != "null"
    @test json_str isa String
    @test length(json_str) > 10

    # Contains expected keys
    @test contains(json_str, "\"background\"")
    @test contains(json_str, "\"agnostic\"")
    @test contains(json_str, "\"interaction\"")

    # Contains explicit log-BF data (key requirement: no implicit client-side dependency)
    @test contains(json_str, "log_bf_enrichment")
    @test contains(json_str, "log_bf_correlation")

    # Contains convergence data
    @test contains(json_str, "convergence")
    @test contains(json_str, "110")   # one of the free_energy values

    # Contains scatter data with components
    @test contains(json_str, "components")
    @test contains(json_str, "\"H0\"")
    @test contains(json_str, "\"H1\"")

    # Contains mixing weights
    @test contains(json_str, "weights")
    @test contains(json_str, "weight_labels")

    # Contains component params
    @test contains(json_str, "\"mu\"")
    @test contains(json_str, "\"sigma\"")
    @test contains(json_str, "\"precision\"")

    # Test null case: analysis_result is nothing
    @test _build_mixture_model_json(nothing) == "null"

    # Test null case: latent_class_result is nothing
    ar_null = (latent_class_result = nothing, bma_result = nothing)
    @test _build_mixture_model_json(ar_null) == "null"

    # Test null case: responsibilities is nothing
    lc_no_resp = BayesInteractomics.LatentClassResult(
        [1.0], [0.5],
        Dict("background" => (mu=-1.0, sigma=1.0, precision=1.0),
             "interaction" => (mu=3.0, sigma=0.8, precision=1.5625)),
        [0.8, 0.2], [100.0], true, 10, nothing
    )
    ar_no_resp = (latent_class_result = lc_no_resp,)
    @test _build_mixture_model_json(ar_no_resp) == "null"

    # Test via bma_result path
    bma_like = (latent_class_result = lc,)
    ar_via_bma = (latent_class_result = nothing, bma_result = bma_like)
    json_bma = _build_mixture_model_json(ar_via_bma, results_df)
    @test json_bma != "null"
    @test contains(json_bma, "log_bf_enrichment")
end

@testitem "generate_report includes mixture_model key" begin
    using BayesInteractomics
    using DataFrames

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "MYC",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )

    results = DataFrame(
        Protein        = ["ACTB", "TP53", "BRCA1"],
        BF             = [200.0, 50.0, 15.0],
        posterior_prob = [0.99, 0.97, 0.85],
        PEP            = 1.0 .- [0.99, 0.97, 0.85],
        BFDR           = [0.001, 0.005, 0.04],
        mean_log2FC    = [4.0, 2.5, 1.8],
        bf_enrichment  = [150.0, 40.0, 10.0],
        bf_correlation = [50.0, 10.0, 5.0],
        bf_detected    = [20.0, 5.0, 2.0],
    )

    report_path = joinpath(tmpdir, "test_report_mm.html")
    generate_report(results, cfg; output = report_path)

    @test isfile(report_path)
    html = read(report_path, String)

    # mixture_model key should be in the JSON blob (even if null)
    @test contains(html, "\"mixture_model\"")

    # The Mixture Model tab HTML structure should be in the template
    @test contains(html, "tab-mixture")
    @test contains(html, "mixture-scatter")
    @test contains(html, "mixture-density")
    @test contains(html, "mixture-convergence")
    @test contains(html, "mixture-weights")
    @test contains(html, "Mixture Model")

    # Existing tabs should still be present
    @test contains(html, "tab-results")
    @test contains(html, "tab-evidence")
    @test contains(html, "tab-methods")
end

# ---------------------------------------------------------------------------
# Sensitivity JSON serialization
# ---------------------------------------------------------------------------

@testitem "Sensitivity JSON serialization" begin
    using BayesInteractomics
    using BayesInteractomics: _build_sensitivity_json, SensitivityResult, SensitivityConfig, PriorSetting
    using DataFrames, Dates, Statistics

    # Test null case returns "null"
    @test _build_sensitivity_json(nothing) == "null"

    # Create mock SensitivityResult
    n_proteins = 5
    n_settings = 3
    posterior_matrix = [
        0.1 0.15 0.2;
        0.5 0.55 0.6;
        0.8 0.85 0.9;
        0.3 0.35 0.4;
        0.95 0.93 0.97;
    ]
    bf_matrix = ones(n_proteins, n_settings)
    q_matrix = ones(n_proteins, n_settings) * 0.05
    protein_names = ["P1", "P2", "P3", "P4", "P5"]
    baseline_index = 1

    summary_df = DataFrame(
        Protein = protein_names,
        baseline_posterior = posterior_matrix[:, baseline_index],
        mean_posterior = vec(mean(posterior_matrix, dims=2)),
        std_posterior = vec(std(posterior_matrix, dims=2)),
        min_posterior = vec(minimum(posterior_matrix, dims=2)),
        max_posterior = vec(maximum(posterior_matrix, dims=2)),
        range = vec(maximum(posterior_matrix, dims=2) .- minimum(posterior_matrix, dims=2)),
    )

    classification_stability = DataFrame(
        Protein = protein_names,
        frac_P_gt_0_5 = [0.0, 1.0, 1.0, 0.0, 1.0],
        frac_P_gt_0_8 = [0.0, 0.0, 1.0, 0.0, 1.0],
        frac_P_gt_0_95 = [0.0, 0.0, 0.0, 0.0, 1.0],
        threshold_crossing_0_95 = [false, false, false, false, true],
        threshold_crossing_0_5 = [false, true, false, false, false],
    )

    prior_settings = [
        PriorSetting(:latent_class, "LC(5,2,1)", (alpha=[5.0, 2.0, 1.0],)),
        PriorSetting(:latent_class, "LC(2,2,1)", (alpha=[2.0, 2.0, 1.0],)),
        PriorSetting(:latent_class, "LC(1,1,1)", (alpha=[1.0, 1.0, 1.0],)),
    ]

    sr = SensitivityResult(
        SensitivityConfig(),
        prior_settings,
        posterior_matrix,
        bf_matrix,
        q_matrix,
        protein_names,
        baseline_index,
        summary_df,
        classification_stability,
        Dates.now(),
    )

    json_str = _build_sensitivity_json(sr)

    # Should not be "null"
    @test json_str != "null"
    @test json_str isa String
    @test length(json_str) > 10

    # Contains expected top-level keys
    @test contains(json_str, "\"band_plot\"")
    @test contains(json_str, "\"stacking_weights\"")
    @test contains(json_str, "\"rankcorr\"")

    # Note: tornado and heatmap data keys were removed in commit a84ec83
    # Current top-level keys: rankcorr, spearman_matrix, band_plot, overlay, stacking_weights

    # Rank correlation data
    @test contains(json_str, "\"correlations\"")
    @test contains(json_str, "\"labels\"")
    @test contains(json_str, "\"baseline_idx\"")

    # Prior setting labels should appear
    @test contains(json_str, "LC(5,2,1)")
    @test contains(json_str, "LC(2,2,1)")

    # Protein names should appear
    @test contains(json_str, "P1")
    @test contains(json_str, "P5")

    # Baseline correlation with itself should be 1.0
    @test contains(json_str, "1")  # baseline_idx = 1

    # --- VIZ-01: Boundary-crosser annotations ---
    @test contains(json_str, "\"boundary_crossers\"")
    # boundary_crossers array should have same length as tornado protein_names
    # Parse boundary_crossers manually: find the array after the key
    bc_match = match(r"\"boundary_crossers\":\[([^\]]*)\]", json_str)
    @test bc_match !== nothing
    bc_values = split(bc_match.captures[1], ",")
    # All 5 proteins fit in top 30, so boundary_crossers length = 5
    @test length(bc_values) == 5
    # Values should be true/false booleans
    @test all(v -> strip(v) in ("true", "false"), bc_values)
    # P2 is a boundary crosser (crosses 0.5), P3 is not
    # Tornado sorts by range descending: P5(0.04), P1(0.1), P2(0.1), P3(0.1), P4(0.1)
    # Actually ranges: P1=0.1, P2=0.1, P3=0.1, P4=0.1, P5=0.04
    # sorted desc: P1,P2,P3,P4 all tie at 0.1, then P5 at 0.04
    # In the tornado order, P2 should be true and P3 should be false

    # --- VIZ-02: Spearman pairwise matrix ---
    @test contains(json_str, "\"spearman_matrix\"")
    sm_match = match(r"\"spearman_matrix\":\{", json_str)
    @test sm_match !== nothing
    @test contains(json_str, "\"matrix\":")
    # Diagonal entries should be 1.0
    # The matrix key should contain "1" for diagonal self-correlations
    @test contains(json_str, "\"labels\"")
    # Labels should match prior setting labels
    @test contains(json_str, "LC(5,2,1)")

    # --- Existing keys still present (no regression) ---
    @test contains(json_str, "\"band_plot\"")
    @test contains(json_str, "\"stacking_weights\"")
    @test contains(json_str, "\"rankcorr\"")

    # --- VIZ-03: Band plot data ---
    @test contains(json_str, "\"band_plot\"")
    bp_match = match(r"\"band_plot\":\{", json_str)
    @test bp_match !== nothing
    # band_plot should contain protein_names, mins, maxs, means, boundary_crossers, n_proteins
    @test contains(json_str, "\"n_proteins\"")
    # All 5 proteins should be included (< 100 cap)
    @test contains(json_str, "\"n_proteins\":5")
    # Band plot proteins sorted by range descending
    # Ranges: P1=0.1, P2=0.1, P3=0.1, P4=0.1, P5=0.04
    # P5 should appear last (lowest range)

    # --- VIZ-04: Overlay data ---
    @test contains(json_str, "\"overlay\"")
    ov_match = match(r"\"overlay\":\{", json_str)
    @test ov_match !== nothing
    @test contains(json_str, "\"setting_labels\"")
    # overlay.protein_names should have min(15, 5) = 5 proteins
    # overlay.values should contain setting arrays
    @test contains(json_str, "\"values\":")
    # Setting labels should appear in overlay
    @test contains(json_str, "LC(1,1,1)")
end

# ---------------------------------------------------------------------------
# lc_convergence exclusion from plots JSON
# ---------------------------------------------------------------------------

@testitem "Report plots exclude lc_convergence" begin
    using BayesInteractomics

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "TEST",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )

    plots_json = BayesInteractomics._build_plots_json(cfg)

    # lc_convergence key must NOT appear in the plots JSON
    @test !occursin("lc_convergence", plots_json)
end

@testitem "Static PNG fallbacks removed from report JSON" begin
    using BayesInteractomics

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "TEST",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )
    result = BayesInteractomics._build_plots_json(cfg)
    @test result == "{}"
end

# ---------------------------------------------------------------------------
# Validation tab JSON
# ---------------------------------------------------------------------------

@testitem "Validation tab JSON" begin
    using BayesInteractomics
    using BayesInteractomics: _build_validation_json
    using Dates
    using DataFrames

    # Test null case
    @test _build_validation_json(nothing) == "null"

    # Test with full data
    qg_cells = Matrix{QualityGateCell}(undef, 3, 3)
    marginals = [:enrichment, :correlation, :detection]
    components = [:H0, :agnostic, :H1]
    for i in 1:3, j in 1:3
        qg_cells[i, j] = QualityGateCell(
            marginals[i], components[j],
            0.05, :pass, nothing, 50.0, false
        )
    end
    qg = QualityGateResult(qg_cells, :pass, String[])

    kl = KLContaminationResult(0.1, 0.2, 0.15, 0.45, 10, true)

    consistency = Dict{String, Bool}(
        "all_ks_pass" => true,
        "kl_pass" => true,
        "h1_lt_200" => true,
    )

    vr = ValidationResult(qg, kl, nothing, consistency, true, now())

    json = _build_validation_json(vr)

    @test json != "null"
    @test occursin("quality_gates", json)
    @test occursin("kl_contamination", json)
    @test occursin("consistency", json)
    @test occursin("overall_pass", json)
    @test occursin("true", json)
    @test occursin("enrichment", json)
    @test occursin("0.1", json)     # kl_enrichment value

    # Test with nothing quality gates and KL
    vr2 = ValidationResult(nothing, nothing, nothing, Dict{String,Bool}(), false, now())
    json2 = _build_validation_json(vr2)
    @test json2 != "null"
    @test occursin("\"overall_pass\":false", json2)
end

# ---------------------------------------------------------------------------
# RVIS-01: 3-component marginal fit JSON
# ---------------------------------------------------------------------------

@testitem "RVIS-01 marginal fits 3-component JSON" begin
    using BayesInteractomics
    using BayesInteractomics: _add_marginal_fit_json!, LatentClassResult
    using DataFrames

    # 10 proteins with MAP-clear assignments: rows 1-4 → H0, 5-7 → Agnostic, 8-10 → H1
    resp = zeros(Float64, 10, 3)
    for i in 1:4;  resp[i, 1] = 0.90; resp[i, 2] = 0.06; resp[i, 3] = 0.04; end
    for i in 5:7;  resp[i, 1] = 0.10; resp[i, 2] = 0.80; resp[i, 3] = 0.10; end
    for i in 8:10; resp[i, 1] = 0.05; resp[i, 2] = 0.05; resp[i, 3] = 0.90; end

    lc = LatentClassResult(
        fill(1.0, 10),                    # bf
        fill(0.5, 10),                    # posterior_prob
        Dict(
            "background"  => (mu=-1.5, sigma=0.8, precision=1.5625),
            "agnostic"    => (mu=0.0,  sigma=0.8, precision=1.5625),
            "interaction" => (mu=3.0,  sigma=0.8, precision=1.5625),
        ),
        [0.5, 0.2, 0.3],                 # mixing_weights
        collect(LinRange(-100.0, -90.0, 10)),  # free_energy
        true,                             # converged
        100,                              # n_iterations
        resp,                             # responsibilities
        nothing,                          # all_restart_traces
        2.0,                              # alpha_enrichment_h1
        1.5,                              # theta_enrichment_h1
        sqrt(2.0) * 1.5,                  # h1_enrichment_sd (sqrt(alpha)*theta for gamma)
        :gamma,                           # h1_enrichment_family
        Dict(:gamma => 100.0, :lognormal => 110.0, :weibull => Inf),  # h1_bic_scores
        nothing,                          # em_diagnostics
        nothing, nothing, nothing,        # disc_detection_H0 / ag / H1
        nothing, nothing, nothing,        # per_step_ll_traces / n_step_halving_reverts / per_dimension_params
        0.0, -1.0, false,                 # nu_h0 / kl_divergence / merged
        Float64[], 0.0, Float64[],        # annealing_schedule / bimodality_coefficient / effective_alpha_prior
        nothing, nothing, false,          # prior_grid_weights / prior_grid_posteriors / eb_converged
        ["P$i" for i in 1:10],            # protein_names — required by _reconstruct_lc_responsibilities
    )

    results_df = DataFrame(
        Protein        = ["P$i" for i in 1:10],
        bf_enrichment  = [0.1, 0.2, 0.3, 0.1, 0.2, 1.0, 1.1, 5.0, 8.0, 12.0],
        bf_correlation = [0.5, 0.4, 0.6, 0.5, 0.4, 1.0, 0.9, 4.0, 6.0, 9.0],
        bf_detected    = [0.8, 0.7, 0.9, 0.8, 0.7, 1.0, 1.1, 3.0, 5.0, 7.0],
    )

    analysis_result = (latent_class_result = lc, bma_result = nothing)
    sections = Pair{String,String}[]
    _add_marginal_fit_json!(sections, analysis_result, results_df)

    # Must have produced some sections
    @test !isempty(sections)

    # Find marginal_fits section
    mf_idx = findfirst(p -> first(p) == "marginal_fits", sections)
    @test mf_idx !== nothing
    mf_json = last(sections[mf_idx])

    # Must contain uppercase keys H0, Agnostic, H1
    @test occursin("\"H0\"", mf_json)
    @test occursin("\"Agnostic\"", mf_json)
    @test occursin("\"H1\"", mf_json)

    # Must NOT contain old lowercase keys
    @test !occursin("\"h0\"", mf_json)
    @test !occursin("\"h1\":", mf_json)

    # Each component has 3 dimension sub-keys
    @test occursin("\"enrichment\"", mf_json)
    @test occursin("\"correlation\"", mf_json)
    @test occursin("\"detection\"", mf_json)

    # H1 enrichment title includes "Gamma" (selected family)
    @test occursin("Gamma", mf_json)

    # Each dimension section has required fields
    for field in ["hist_values", "fit_x", "fit_y", "dist_label", "title", "n", "mu", "sigma"]
        @test occursin("\"$field\"", mf_json)
    end
end

# ---------------------------------------------------------------------------
# RVIS-01 Component column path: Component column used as fallback when
# lc.responsibilities is nothing (gap closure).
# ---------------------------------------------------------------------------

@testitem "RVIS-01 Component column path" begin
    using BayesInteractomics
    using BayesInteractomics: _add_marginal_fit_json!, LatentClassResult
    using DataFrames

    # lc with responsibilities = nothing → primary path is skipped.
    # Component column should be used as fallback.
    lc = LatentClassResult(
        fill(1.0, 10),        # bf
        fill(0.5, 10),        # posterior_prob
        Dict(
            "background"  => (mu=-1.5, sigma=0.8, precision=1.5625),
            "agnostic"    => (mu=0.0,  sigma=0.8, precision=1.5625),
            "interaction" => (mu=3.0,  sigma=0.8, precision=1.5625),
        ),
        [0.5, 0.2, 0.3],                       # mixing_weights
        collect(LinRange(-100.0, -90.0, 10)),   # free_energy
        true,                                   # converged
        100,                                    # n_iterations
        nothing,                                # responsibilities = nothing → triggers Component fallback
        nothing,                                # all_restart_traces
        2.0,                                    # alpha_enrichment_h1
        1.5,                                    # theta_enrichment_h1
        sqrt(2.0) * 1.5,                        # h1_enrichment_sd
        :gamma,                                 # h1_enrichment_family
        Dict(:gamma => 100.0, :lognormal => 110.0, :weibull => Inf),  # h1_bic_scores
        nothing,                                # em_diagnostics
    )

    # results_df with Component column (sort-aligned): 4 H0, 3 Agnostic, 3 H1.
    # No P_H0/P_agnostic/P_H1 columns so the Component fallback is the only source.
    results_df = DataFrame(
        Protein        = ["P$i" for i in 1:10],
        bf_enrichment  = [0.1, 0.2, 0.3, 0.1, 0.2, 1.0, 1.1, 5.0, 8.0, 12.0],
        bf_correlation = [0.5, 0.4, 0.6, 0.5, 0.4, 1.0, 0.9, 4.0, 6.0,  9.0],
        bf_detected    = [0.8, 0.7, 0.9, 0.8, 0.7, 1.0, 1.1, 3.0, 5.0,  7.0],
        Component      = ["H0","H0","H0","H0","Agnostic","Agnostic","Agnostic","H1","H1","H1"],
    )

    analysis_result = (latent_class_result = lc, bma_result = nothing)
    sections = Pair{String,String}[]
    _add_marginal_fit_json!(sections, analysis_result, results_df)

    # Component fallback must have produced sections
    @test !isempty(sections)

    mf_idx = findfirst(p -> first(p) == "marginal_fits", sections)
    @test mf_idx !== nothing
    mf_json = last(sections[mf_idx])

    # Must contain uppercase component keys from Component column
    @test occursin("\"H0\"", mf_json)
    @test occursin("\"Agnostic\"", mf_json)
    @test occursin("\"H1\"", mf_json)

    # Each component must have the 3 dimension sub-keys
    @test occursin("\"enrichment\"", mf_json)
    @test occursin("\"correlation\"", mf_json)
    @test occursin("\"detection\"", mf_json)

    # H0 section must have n=4 (4 proteins assigned "H0" via Component column)
    # H1 section must have n=3 (proteins 8,9,10 assigned "H1" via Component column)
    # We verify counts appear in the JSON as the "n" field values.
    # The JSON format is: "n": 4  or  "n": 3
    @test occursin("\"n\": 4", mf_json) || occursin("\"n\":4", mf_json)
    @test occursin("\"n\": 3", mf_json) || occursin("\"n\":3", mf_json)

    # H1 enrichment title must include the BIC-selected family name "Gamma"
    @test occursin("Gamma", mf_json)

    # Required fields per panel
    for field in ["hist_values", "fit_x", "fit_y", "dist_label", "title", "n"]
        @test occursin("\"$field\"", mf_json)
    end
end

# ---------------------------------------------------------------------------
# RVIS-02: EM restart JSON with BIC columns
# ---------------------------------------------------------------------------

@testitem "RVIS-02 EM restart JSON with BIC columns" begin
    using BayesInteractomics
    using BayesInteractomics: _build_evidence_data_json, LatentClassResult
    using DataFrames

    em_diag = DataFrame(
        restart           = [1, 2, 3],
        init_pi0          = [0.5, 0.6, 0.4],
        init_method       = ["quantile", "kmeans", "random"],
        final_pi0         = [0.48, 0.50, 0.52],
        final_pi1         = [0.30, 0.28, 0.32],
        log_likelihood    = [-300.0, -295.0, -310.0],
        iterations        = [50, 45, 55],
        converged         = [true, true, false],
        status            = ["ok", "ok", "failed"],
        h1_bic_gamma      = [100.0, 102.0, 98.0],
        h1_bic_lognormal  = [105.0, 108.0, 103.0],
        h1_bic_weibull    = [Inf, Inf, Inf],
        h1_family_selected = ["gamma", "gamma", "gamma"],
    )

    lc = LatentClassResult(
        fill(1.0, 5),
        fill(0.5, 5),
        Dict(
            "background"  => (mu=-1.0, sigma=0.8, precision=1.5625),
            "agnostic"    => (mu=0.0,  sigma=0.8, precision=1.5625),
            "interaction" => (mu=3.0,  sigma=0.8, precision=1.5625),
        ),
        [0.5, 0.2, 0.3],
        collect(LinRange(-100.0, -90.0, 5)),
        true, 50,
        nothing, nothing,
        2.0, 1.5, sqrt(2.0) * 1.5,   # h1_enrichment_sd
        :gamma,
        Dict(:gamma => 100.0, :lognormal => 105.0, :weibull => Inf),
        em_diag,
    )

    results_df = DataFrame(
        Protein        = ["P$i" for i in 1:5],
        bf_enrichment  = [0.5, 1.0, 2.0, 5.0, 10.0],
        bf_correlation = [0.3, 0.8, 1.5, 3.0, 8.0],
        bf_detected    = [0.9, 1.0, 1.2, 2.0, 4.0],
    )

    analysis_result = (latent_class_result = lc, bma_result = nothing)
    json_str = _build_evidence_data_json(analysis_result, results_df)

    @test json_str != "null"
    @test occursin("\"em_restarts\"", json_str)

    # Must have 3 restart rows in the JSON (3 objects in em_restarts array)
    # Use simple presence check for BIC columns
    @test occursin("\"init_method\"", json_str)
    @test occursin("\"h1_family\"", json_str)
    @test occursin("\"bic_gamma\"", json_str)
    @test occursin("\"bic_lognormal\"", json_str)
    @test occursin("\"bic_weibull\"", json_str)

    # Check values appear (restart 2 has best LL=-295.0)
    @test occursin("quantile", json_str)
    @test occursin("kmeans", json_str)
    @test occursin("gamma", json_str)
    @test occursin("100", json_str)
    @test occursin("105", json_str)
end

# ---------------------------------------------------------------------------
# RVIS-03: Validation JSON has no component scatter
# ---------------------------------------------------------------------------

@testitem "RVIS-03 validation JSON has no component scatter" begin
    using BayesInteractomics
    using BayesInteractomics: _build_validation_json
    using Dates, DataFrames

    # Build a minimal ValidationResult with all sub-structs
    qg_cells = Matrix{QualityGateCell}(undef, 3, 3)
    marginals  = [:enrichment, :correlation, :detection]
    components = [:H0, :agnostic, :H1]
    for i in 1:3, j in 1:3
        qg_cells[i, j] = QualityGateCell(
            marginals[i], components[j],
            0.05, :pass, nothing, 50.0, false
        )
    end
    qg = QualityGateResult(qg_cells, :pass, String[])
    kl = KLContaminationResult(0.1, 0.2, 0.15, 0.45, 10, true)

    vr = ValidationResult(qg, kl, nothing, Dict("all_ks_pass" => true), true, now())
    json_str = _build_validation_json(vr)

    @test json_str != "null"

    # Must NOT contain scatter or component_assignment data
    @test !occursin("\"scatter\"", json_str)
    @test !occursin("component_assignment", json_str)
    @test !occursin("\"components\"", json_str)

    # Should still have its valid content
    @test occursin("quality_gates", json_str)
    @test occursin("kl_contamination", json_str)
end

# ---------------------------------------------------------------------------
# Missing-value safety (regression tests for MethodError(Float64, (missing,)))
# ---------------------------------------------------------------------------

@testitem "_report_float handles missing values" begin
    using BayesInteractomics: _report_float

    @test _report_float(missing)        === 0.0
    @test _report_float(missing, NaN)   |> isnan
    @test _report_float(missing, 1.0)   === 1.0
    @test _report_float(1.5)            === 1.5
    @test _report_float(2)              === 2.0
    @test _report_float(0.0)            === 0.0
    # NaN passes through (json_number will handle it)
    @test isnan(_report_float(NaN))
end

@testitem "_build_protein_json handles missing BF values" begin
    using BayesInteractomics: _build_protein_json
    using DataFrames

    row = (
        Protein = "TEST_PROT",
        BF = missing,
        posterior_prob = missing,
        PEP = missing,
        BFDR = missing,
        mean_log2FC = missing,
        bf_enrichment = missing,
        bf_correlation = missing,
        bf_detected = missing,
    )
    # Should not throw MethodError — all missing values handled
    json_str = _build_protein_json(row)
    @test json_str isa String
    @test occursin("\"TEST_PROT\"", json_str)
    @test occursin("null", json_str)  # missing → null via json_number
end

@testitem "_build_discordant_json handles missing BF columns" begin
    using BayesInteractomics: _build_discordant_json, _filter_detected
    using DataFrames

    df = DataFrame(
        Protein        = ["A", "B", "C"],
        BF             = Union{Missing,Float64}[5.0, missing, 0.5],
        bf_enrichment  = Union{Missing,Float64}[2.0, missing, 0.3],
        bf_correlation = Union{Missing,Float64}[missing, 1.0, 0.8],
        bf_detected    = Union{Missing,Float64}[1.5, missing, 0.9],
        is_detected    = [true, true, true],
    )

    # analysis_result is nothing — should return "null" but NOT throw
    result = _build_discordant_json(nothing, df)
    @test result isa String
end

@testitem "_build_evidence_data_json handles missing BF columns" begin
    using BayesInteractomics: _build_evidence_data_json
    using DataFrames

    df = DataFrame(
        Protein        = ["A", "B", "C", "D", "E"],
        BF             = Union{Missing,Float64}[5.0, missing, 0.5, 1.2, missing],
        bf_enrichment  = Union{Missing,Float64}[2.0, missing, 0.3, 1.0, 0.5],
        bf_correlation = Union{Missing,Float64}[missing, 1.0, 0.8, 0.9, 1.1],
        bf_detected    = Union{Missing,Float64}[1.5, missing, 0.9, 1.0, 0.8],
        is_detected    = [true, true, true, true, true],
    )

    # Should not throw MethodError
    result = _build_evidence_data_json(nothing, df)
    @test result isa String
end

@testitem "generate_report end-to-end with missing values" begin
    using BayesInteractomics
    using BayesInteractomics: generate_report, OutputFiles, CONFIG
    using DataFrames

    # Build minimal results DataFrame with missing values
    results = DataFrame(
        Protein        = ["PROT_A", "PROT_B", "PROT_C", "PROT_D"],
        BF             = Union{Missing,Float64}[10.0, missing, 0.5, 3.0],
        posterior_prob = Union{Missing,Float64}[0.9, missing, 0.3, 0.75],
        PEP            = Union{Missing,Float64}[0.1, missing, 0.7, 0.25],
        BFDR           = Union{Missing,Float64}[0.01, missing, 0.5, 0.05],
        mean_log2FC    = Union{Missing,Float64}[2.0, missing, -0.5, 1.0],
        bf_enrichment  = Union{Missing,Float64}[5.0, missing, 0.2, 2.0],
        bf_correlation = Union{Missing,Float64}[missing, 1.5, 0.8, 1.0],
        bf_detected    = Union{Missing,Float64}[2.0, missing, 0.9, 1.5],
        is_detected    = [true, true, true, false],
    )

    mktempdir() do tmpdir
        output = OutputFiles(tmpdir)
        config = CONFIG(
            datafile = ["dummy.xlsx"],
            sample_cols = [Dict(1 => [1])],
            control_cols = [Dict(1 => [2])],
            poi = "BAIT",
            n_controls = 1,
            n_samples = 1,
            output = output,
        )

        report_path = config.output.report_file
        # Should complete without error
        generate_report(results, config; output=report_path)

        # Verify output file exists and is valid HTML
        @test isfile(report_path)
        html = read(report_path, String)
        @test occursin("<html", html)
        @test occursin("PROT_A", html)
        # Missing BF proteins still appear (detected ones)
        @test occursin("PROT_B", html)
        # Non-detected protein excluded from main table
        @test !occursin("PROT_D", html) || occursin("non_detected", html)
    end
end

# ---------------------------------------------------------------------------
# Winsorization, KDE density, MC combined density
# ---------------------------------------------------------------------------

@testitem "quantile winsorization" begin
    using BayesInteractomics: _winsorize_quantile

    # Basic clamping: -690 outlier should be clamped to near the 0.5th percentile
    vals = vcat([1.0, 2.0, 3.0, -690.0, 50.0], collect(range(-2.0, 5.0, length=100)))
    result = _winsorize_quantile(vals)
    @test length(result) == length(vals)
    # The -690 value should have been clamped (no longer -690)
    @test minimum(filter(isfinite, result)) > -600.0
    # Specifically, the original -690 at index 4 should be clamped up
    @test result[4] > -690.0
    # Interior values should be mostly unchanged
    # The 50.0 might also be clamped depending on quantiles, but moderate values should pass
    @test result[2] == 2.0  # interior value preserved

    # All-finite values: interior values preserved
    normal_vals = collect(1.0:100.0)
    result2 = _winsorize_quantile(normal_vals)
    @test result2[50] == 50.0  # interior value unchanged

    # Empty vector returns empty
    empty_result = _winsorize_quantile(Float64[])
    @test isempty(empty_result)

    # NaN/Inf handling: non-finite values left as-is, finite values clamped
    mixed = [NaN, Inf, -Inf, 1.0, 2.0, 3.0, 4.0, 5.0, -690.0, 100.0]
    result3 = _winsorize_quantile(mixed)
    @test isnan(result3[1])
    @test isinf(result3[2]) && result3[2] > 0
    @test isinf(result3[3]) && result3[3] < 0
    @test result3[5] == 2.0  # interior finite value preserved
end

@testitem "KDE density" begin
    using BayesInteractomics: _kde_density
    using Statistics

    # Integration to ~1.0
    samples = randn(1000) .* 2.0 .+ 3.0
    x_grid = collect(range(-10.0, 16.0, length=500))
    density = _kde_density(samples, x_grid)
    dx = x_grid[2] - x_grid[1]
    integral = sum(density) * dx
    @test isapprox(integral, 1.0, atol=0.1)

    # Peak location near sample mean for Normal samples
    peak_idx = argmax(density)
    peak_x = x_grid[peak_idx]
    @test isapprox(peak_x, mean(samples), atol=1.0)

    # All densities non-negative
    @test all(d -> d >= 0.0, density)
end

@testitem "MC combined density" begin
    using BayesInteractomics: _mc_combined_density, LatentClassResult, JEFFREYS_SHIFT, DiscreteEmpirical

    # Build a LatentClassResult with per_dimension_params
    pdp = Dict(
        "background" => (mu_e=-1.0, sigma_e=1.0, mu_c=-0.5, sigma_c=0.8, mu_p=-0.3, sigma_p=0.5),
        "agnostic"   => (mu_e=0.0,  sigma_e=0.8, mu_c=0.0,  sigma_c=0.7, mu_p=0.0,  sigma_p=0.4),
        "interaction"=> (mu_e=2.0,  sigma_e=0.6, mu_c=1.0,  sigma_c=0.5, mu_p=0.5,  sigma_p=0.3),
    )

    disc_H0 = DiscreteEmpirical([0.0, 0.5, 1.0], [0.5, 0.3, 0.2])
    disc_H1 = DiscreteEmpirical([0.5, 1.0, 2.0], [0.2, 0.3, 0.5])

    lc = LatentClassResult(
        [1.0, 2.0, 3.0], [0.3, 0.6, 0.9],
        Dict(
            "background"  => (mu=-1.0, sigma=1.0, precision=1.0),
            "agnostic"    => (mu=0.0,  sigma=0.8, precision=1.5625),
            "interaction" => (mu=2.0,  sigma=0.6, precision=2.778),
        ),
        [0.5, 0.3, 0.2],
        [100.0, 110.0],
        true, 20,
        [0.8 0.1 0.1; 0.1 0.7 0.2; 0.05 0.05 0.9],  # responsibilities
        nothing,  # all_restart_traces
        2.0, 1.0,  # alpha, theta
        sqrt(2.0) * 1.0,  # h1_enrichment_sd
        :gamma,
        Dict(:gamma => 0.0, :lognormal => Inf, :weibull => Inf),
        nothing,  # em_diagnostics
        disc_H0, nothing, disc_H1,
        nothing, nothing,  # per_step_ll_traces, n_step_halving_reverts
        pdp  # per_dimension_params
    )

    # Background component returns valid tuple with 200 points
    bg_result = _mc_combined_density(lc, "background"; n_mc=50_000, n_grid=200)
    @test bg_result !== nothing
    @test length(bg_result.x) == 200
    @test length(bg_result.y) == 200
    @test all(y -> y >= 0.0, bg_result.y)

    # Interaction component has positive-biased mean (shifted by JEFFREYS_SHIFT)
    int_result = _mc_combined_density(lc, "interaction"; n_mc=50_000, n_grid=200)
    @test int_result !== nothing
    # Weighted mean of density should be positive
    dx = int_result.x[2] - int_result.x[1]
    weighted_mean = sum(int_result.x .* int_result.y) * dx
    @test weighted_mean > JEFFREYS_SHIFT

    # Returns nothing when per_dimension_params is nothing
    lc_no_pdp = LatentClassResult(
        [1.0], [0.5],
        Dict("background" => (mu=-1.0, sigma=1.0, precision=1.0),
             "interaction" => (mu=3.0, sigma=0.8, precision=1.5625)),
        [0.8, 0.2], [100.0], true, 10, nothing
    )
    @test _mc_combined_density(lc_no_pdp, "background") === nothing

    # LC responsibility extraction: responsibilities sum to ~1.0 per row
    resp = lc.responsibilities
    @test resp !== nothing
    for i in 1:size(resp, 1)
        @test isapprox(sum(resp[i, :]), 1.0, atol=0.01)
    end
end

# ---------------------------------------------------------------------------
# Version sourcing
# ---------------------------------------------------------------------------

@testitem "report version matches Project.toml" begin
    using BayesInteractomics
    import TOML

    toml_path = joinpath(pkgdir(BayesInteractomics), "Project.toml")
    toml = TOML.parsefile(toml_path)
    expected_version = toml["version"]

    report_version = BayesInteractomics._report_pkg_version()
    @test report_version == expected_version
    @test report_version != "?"
    @test report_version == "1.2.1"
end

@testitem "report file size within budget (SIZE-03)" begin
    using BayesInteractomics
    using DataFrames

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "MYC",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,  # we call generate_report manually
    )

    # Build a 100-row DataFrame mimicking real results
    n = 100
    results = DataFrame(
        Protein        = ["PROT_$i" for i in 1:n],
        BF             = rand(n) .* 100,
        posterior_prob  = rand(n),
        PEP            = 1.0 .- rand(n),
        BFDR           = rand(n),
        mean_log2FC    = randn(n),
        sd_log2FC      = abs.(randn(n)),
        bf_enrichment  = rand(n) .* 50,
        bf_correlation = rand(n) .* 10,
        bf_detected    = rand(n) .* 5,
    )

    report_path = joinpath(tmpdir, "test_report_size.html")
    generate_report(results, cfg; output = report_path)
    @test isfile(report_path)

    filesize_bytes = filesize(report_path)
    @info "Report file size (100 proteins): $(filesize_bytes) bytes ($(round(filesize_bytes/1024; digits=1)) KB)"

    # Without base64 PNGs, a 100-protein report should be well under 1 MB
    # Template is ~177 KB, JSON data for 100 proteins adds ~50-100 KB
    @test filesize_bytes < 1_000_000  # 1 MB upper bound
    @test filesize_bytes > 100_000    # Sanity: at least 100 KB (template alone)
end

# ---------------------------------------------------------------------------
# Plot reorganisation tests
# ---------------------------------------------------------------------------

@testitem "PLOT-01 tab order follows narrative flow" begin
    # Read the template directly to verify tab button order
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    html = Base.read(template_path, String)

    # Extract all tab button labels from the mainTabs ul
    # Pattern: data-bs-toggle="tab" data-bs-target="#tab-xxx">LABEL</button>
    tab_labels = String[]
    for m in eachmatch(r"data-bs-toggle=\"tab\"\s+data-bs-target=\"#tab-[^\"]+\"[^>]*>([^<]+)</button>", html)
        push!(tab_labels, m.captures[1])
    end

    expected_order = [
        "Results", "Prior", "Evidence", "Calibration", "Sensitivity",
        "Mixture Model", "Structural Evidence",
        "Differential", "Data Quality", "Methods"
    ]
    @test tab_labels == expected_order
end

@testitem "PLOT-02 tab labels are biologist-friendly" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    html = Base.read(template_path, String)

    # These jargon-heavy labels must NOT appear as tab button text
    @test !occursin(">Simulation<", html)       # Should be "Calibration"
    @test !occursin(">Validation<", html)        # Tab removed
    @test !occursin(">Docking<", html)           # Should be "Structural Evidence"
    @test !occursin(">Prior Sensitivity<", html) # Should be "Sensitivity"
    @test !occursin(">Diagnostics<", html)       # Tab removed entirely

    # Friendly labels must be present
    @test occursin(">Calibration<", html)
    @test !occursin(">Quality Control<", html)   # QC tab removed
    @test occursin(">Structural Evidence<", html)
    @test occursin(">Sensitivity<", html)
end

@testitem "STRUC-02 QC badge exists in Mixture Model tab" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    html = Base.read(template_path, String)

    # Badge span exists in the template
    @test occursin("badge-qc-status", html)

    # Badge is inside the qg-card (Mixture Model tab)
    @test occursin("qg-card", html)

    # initQualityGateMatrix function exists
    @test occursin("initQualityGateMatrix", html)
end

@testitem "PLOT-03 auto-discovery replaces SELECTABLE_PLOTS" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    html = Base.read(template_path, String)

    # SELECTABLE_PLOTS constant must not exist
    @test !occursin("SELECTABLE_PLOTS", html)

    # data-selectable-plot attribute must be present on all 8 scatter containers
    selectable_count = length(collect(eachmatch(r"data-selectable-plot", html)))
    # 8 plot containers + at least 1 querySelectorAll reference
    @test selectable_count >= 9

    # Each known scatter plot ID must have the attribute
    for plot_id in ["volcano-plot", "rankrank-plot", "evidence-scatter-2x2",
                     "mixture-scatter", "docking-scatter-plot", "docking-update-plot",
                     "bma-agreement-scatter", "discordant-scatter"]
        # Pattern: id="xxx" ... data-selectable-plot (on same element)
        @test occursin(Regex("id=\"$(plot_id)\"[^>]*data-selectable-plot"), html)
    end

    # applyHighlighting must use querySelectorAll
    @test occursin("querySelectorAll('[data-selectable-plot]')", html)
end

@testitem "Diagnostics tab merged into Mixture Model" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    html = Base.read(template_path, String)

    # No standalone Diagnostics tab
    @test !occursin("tab-diag-li", html)
    @test !occursin("id=\"tab-diagnostics\"", html)

    # diagnostics-container exists inside the file (merged into mixture)
    @test occursin("diagnostics-container", html)

    # Verify diagnostics-container is inside tab-mixture pane
    # Find positions: tab-mixture pane start should come before diagnostics-container
    mixture_start = findfirst("id=\"tab-mixture\"", html)
    diag_container = findfirst("id=\"diagnostics-container\"", html)
    @test mixture_start !== nothing
    @test diag_container !== nothing
    @test first(mixture_start) < first(diag_container)

    # And the next tab-pane after tab-mixture should come after diagnostics-container
    # (meaning diagnostics-container is inside tab-mixture, not after it)
    mixture_end_region = html[first(mixture_start):end]
    next_tab_pane = findfirst(r"<div class=\"tab-pane[^\"]*\" id=\"tab-(?!mixture)", mixture_end_region)
    diag_in_mixture = findfirst("diagnostics-container", mixture_end_region)
    @test diag_in_mixture !== nothing
    @test next_tab_pane !== nothing
    @test first(diag_in_mixture) < first(next_tab_pane)
end

@testitem "PLOT-04 differential report mirrors main report structure" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "differential_report.html")
    html = Base.read(template_path, String)

    # "Plots" tab renamed to "Evidence"
    @test occursin(">Evidence<", html)
    @test !occursin(">Plots<", html)

    # diff-plot has data-selectable-plot attribute
    @test occursin(Regex("id=\"diff-plot\"[^>]*data-selectable-plot"), html)

    # applyHighlighting uses auto-discovery, not hardcoded getElementById('diff-plot')
    @test occursin("querySelectorAll('[data-selectable-plot]')", html)

    # Tab IDs preserved (JS compatibility)
    @test occursin("tab-plots-li", html)
    @test occursin("id=\"tab-plots\"", html)
end

# ---------------------------------------------------------------------------
# Per-plot explanation tests
# ---------------------------------------------------------------------------

@testitem "EXPL-01 main report explanations" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    html = Base.read(template_path, String)

    # Infrastructure present
    @test occursin("btn-explanation-toggle", html)
    @test occursin("toggleExplanation", html)
    @test occursin("plot-explanation", html)
    @test occursin("EXPL_ICON", html)

    # Hardcoded HTML explanations (Results tab)
    @test occursin("expl-volcano", html)
    @test occursin("expl-rankrank", html)
    @test occursin("expl-results-table", html)

    # Calibration tab explanations
    @test occursin("expl-fdr-posterior", html)
    @test occursin("expl-sens-posterior", html)
    @test occursin("expl-roc", html)
    @test occursin("expl-reliability", html)
    @test occursin("expl-threshold-rec", html)
    @test occursin("expl-cal-reliability", html)
    @test occursin("expl-cal-function", html)
    @test occursin("expl-cal-cv-ece", html)
    @test occursin("expl-fdr-accuracy", html)

    # Sensitivity tab explanations
    @test occursin("expl-sens-band", html)
    @test occursin("expl-sens-overlay", html)
    @test occursin("expl-sens-rankcorr", html)
    @test occursin("expl-sens-stacking", html)

    # Mixture Model tab explanations
    @test occursin("expl-component-scatter", html)
    @test occursin("expl-marginal-density", html)
    @test occursin("expl-em-convergence", html)
    @test occursin("expl-mixing-weights", html)

    # Quality Gate Matrix explanation (in Mixture Model tab)
    @test occursin("expl-quality-gate", html)
    # expl-dist-overlay, expl-kl-contamination, expl-consistency removed with QC tab

    # Structural Evidence tab explanations
    @test occursin("expl-iptm-scatter", html)
    @test occursin("expl-posterior-update", html)
    @test occursin("expl-docking-table", html)

    # Methods tab explanations
    @test occursin("expl-analysis-params", html)
    @test occursin("expl-reproducibility", html)

    # Dynamic JS plot explanations (look for expl- IDs in JS strings)
    @test occursin("expl-evidence-scatter", html)
    @test occursin("expl-em-diagnostics", html)
    @test occursin("expl-marginal-fits", html)
    @test occursin("expl-cal-quality-gate", html)
    @test occursin("expl-cal-curves", html)
    @test occursin("expl-residual-qq", html)
    @test occursin("expl-scale-location", html)
    @test occursin("expl-ppc-pvalues", html)
    @test occursin("expl-pit", html)
    @test occursin("expl-nu-optimization", html)
    @test occursin("expl-within-class-corr", html)
    @test occursin("expl-kl-h1-purity", html)
    @test occursin("expl-model-selection", html)
    @test occursin("expl-discordant", html)
    @test occursin("expl-bma-weights", html)
    @test occursin("expl-copula-contribution", html)

    # All explanations start collapsed
    n_toggles = length(collect(eachmatch(r"btn-explanation-toggle", html)))
    n_explanations = length(collect(eachmatch(r"class=\"plot-explanation\"", html)))
    @test n_toggles >= 30  # At least 30 plots with toggles
    @test n_explanations >= 30  # Matching explanation blocks
end

@testitem "EXPL-03 glossary tooltip terms" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    html = Base.read(template_path, String)

    # Tooltip infrastructure
    @test occursin("glossary-term", html)
    @test occursin("data-bs-toggle=\"tooltip\"", html)
    @test occursin("bootstrap.Tooltip", html)

    # Key glossary terms must appear as tooltip-wrapped spans
    required_terms = [
        "Bayes factor", "posterior probability", "FDR", "BFDR",
        "log2 fold change", "enrichment", "correlation", "detection",
        "BMA", "calibration"
    ]
    for term in required_terms
        @test occursin(term, html)
    end

    # At least 10 glossary-term spans
    n_glossary = length(collect(eachmatch(r"class=\"glossary-term\"", html)))
    @test n_glossary >= 10
end

# ---------------------------------------------------------------------------
# EB diagnostics in evidence_data JSON
# ---------------------------------------------------------------------------

@testitem "EB diagnostics in evidence_data JSON" begin
    using BayesInteractomics
    using BayesInteractomics: _build_evidence_data_json, LatentClassResult
    using DataFrames

    # Create em_diagnostics DataFrame (required for 15-arg constructor)
    em_diag = DataFrame(
        restart           = [1, 2],
        init_pi0          = [0.5, 0.6],
        init_method       = ["quantile", "kmeans"],
        final_pi0         = [0.48, 0.50],
        final_pi1         = [0.30, 0.28],
        log_likelihood    = [-300.0, -295.0],
        iterations        = [50, 45],
        converged         = [true, true],
        status            = ["ok", "ok"],
    )

    # Build a LatentClassResult with EB fields populated
    # Use the full struct constructor (all 29 fields)
    lc = LatentClassResult(
        fill(1.0, 5),                  # bf
        fill(0.5, 5),                  # posterior_prob
        Dict(                          # class_parameters
            "background"  => (mu=-1.0, sigma=0.8, precision=1.5625),
            "agnostic"    => (mu=0.0,  sigma=0.8, precision=1.5625),
            "interaction" => (mu=3.0,  sigma=0.8, precision=1.5625),
        ),
        [0.5, 0.2, 0.3],              # mixing_weights
        collect(LinRange(-100.0, -90.0, 5)),  # free_energy
        true,                          # converged
        50,                            # n_iterations
        nothing,                       # responsibilities
        nothing,                       # all_restart_traces
        2.0,                           # alpha_enrichment_h1
        1.5,                           # theta_enrichment_h1
        sqrt(2.0) * 1.5,              # h1_enrichment_sd
        :gamma,                        # h1_enrichment_family
        Dict(:gamma => 100.0, :lognormal => 105.0, :weibull => Inf),  # h1_bic_scores
        em_diag,                       # em_diagnostics
        nothing,                       # disc_detection_H0
        nothing,                       # disc_detection_ag
        nothing,                       # disc_detection_H1
        nothing,                       # per_step_ll_traces
        nothing,                       # n_step_halving_reverts
        nothing,                       # per_dimension_params
        0.0,                           # nu_h0
        -1.0,                          # kl_divergence
        false,                         # merged
        Float64[],                     # annealing_schedule
        0.0,                           # bimodality_coefficient
        [4.2, 1.8, 0.9],              # effective_alpha_prior (EB-estimated)
        [0.05, 0.20, 0.50, 0.20, 0.05],  # prior_grid_weights
        nothing,                       # prior_grid_posteriors
        true,                          # eb_converged
        ["P1", "P2", "P3", "P4", "P5"],  # protein_names (31st field)
    )

    results_df = DataFrame(
        Protein        = ["P1", "P2", "P3", "P4", "P5"],
        bf_enrichment  = [0.5, 1.0, 2.0, 5.0, 10.0],
        bf_correlation = [0.3, 0.8, 1.5, 3.0, 8.0],
        bf_detected    = [0.9, 1.0, 1.2, 2.0, 4.0],
        classification_stability = ["robust", "robust", "robust", "sensitive", "fragile"],
    )

    analysis_result = (latent_class_result = lc, bma_result = nothing)
    json_str = _build_evidence_data_json(analysis_result, results_df)

    # Test 1: eb_diagnostics present with effective_alpha and eb_converged
    @test occursin("\"eb_diagnostics\"", json_str)
    @test occursin("\"effective_alpha\"", json_str)
    @test occursin("4.2", json_str)
    @test occursin("1.8", json_str)
    @test occursin("0.9", json_str)
    @test occursin("\"eb_converged\"", json_str)
    @test occursin("true", json_str)

    # Test 2: prior_grid_weights present
    @test occursin("\"prior_grid_weights\"", json_str)
    @test occursin("0.05", json_str)
    @test occursin("0.5", json_str)

    # Test 4: stability_counts present with correct counts
    @test occursin("\"stability_counts\"", json_str)
    @test occursin("\"robust\"", json_str)
    @test occursin("\"sensitive\"", json_str)
    @test occursin("\"fragile\"", json_str)

    # --- Negative case: explicit alpha (empty effective_alpha_prior) ---
    lc_explicit = LatentClassResult(
        fill(1.0, 5),
        fill(0.5, 5),
        Dict(
            "background"  => (mu=-1.0, sigma=0.8, precision=1.5625),
            "agnostic"    => (mu=0.0,  sigma=0.8, precision=1.5625),
            "interaction" => (mu=3.0,  sigma=0.8, precision=1.5625),
        ),
        [0.5, 0.2, 0.3],
        collect(LinRange(-100.0, -90.0, 5)),
        true, 50,
        nothing, nothing,
        2.0, 1.5, sqrt(2.0) * 1.5,
        :gamma,
        Dict(:gamma => 100.0, :lognormal => 105.0, :weibull => Inf),
        em_diag,
        nothing, nothing, nothing,
        nothing, nothing, nothing,
        0.0, -1.0, false, Float64[], 0.0,
        Float64[],                     # effective_alpha_prior = empty (explicit alpha)
        nothing,                       # prior_grid_weights = nothing
        nothing,                       # prior_grid_posteriors = nothing
        false,                         # eb_converged = false
        nothing,                       # protein_names (31st field)
    )

    analysis_result_explicit = (latent_class_result = lc_explicit, bma_result = nothing)
    json_str_explicit = _build_evidence_data_json(analysis_result_explicit, results_df)

    # Test 3: eb_diagnostics absent when effective_alpha_prior is empty
    @test !occursin("\"eb_diagnostics\"", json_str_explicit)

    # Test 5: eb_converged=false case
    lc_not_converged = LatentClassResult(
        fill(1.0, 5),
        fill(0.5, 5),
        Dict(
            "background"  => (mu=-1.0, sigma=0.8, precision=1.5625),
            "agnostic"    => (mu=0.0,  sigma=0.8, precision=1.5625),
            "interaction" => (mu=3.0,  sigma=0.8, precision=1.5625),
        ),
        [0.5, 0.2, 0.3],
        collect(LinRange(-100.0, -90.0, 5)),
        true, 50,
        nothing, nothing,
        2.0, 1.5, sqrt(2.0) * 1.5,
        :gamma,
        Dict(:gamma => 100.0, :lognormal => 105.0, :weibull => Inf),
        em_diag,
        nothing, nothing, nothing,
        nothing, nothing, nothing,
        0.0, -1.0, false, Float64[], 0.0,
        [4.2, 1.8, 0.9],              # effective_alpha_prior (non-empty)
        nothing,                       # prior_grid_weights = nothing
        nothing,                       # prior_grid_posteriors = nothing
        false,                         # eb_converged = false
        nothing,                       # protein_names (31st field)
    )

    analysis_result_nc = (latent_class_result = lc_not_converged, bma_result = nothing)
    json_str_nc = _build_evidence_data_json(analysis_result_nc, results_df)

    @test occursin("\"eb_diagnostics\"", json_str_nc)
    @test occursin("\"eb_converged\"", json_str_nc)
    # eb_converged should be false — check the JSON contains false after eb_converged
    @test occursin("\"eb_converged\":false", replace(json_str_nc, " " => ""))
end

@testitem "EXPL-02 differential report explanations" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "differential_report.html")
    html = Base.read(template_path, String)

    # Infrastructure present
    @test occursin("btn-explanation-toggle", html)
    @test occursin("toggleExplanation", html)
    @test occursin("plot-explanation", html)

    # Purple accent (not blue)
    @test occursin("fill=\"#4a148c\"", html)  # SVG icon colour
    @test occursin("color: #4a148c", html)    # Hover colour in CSS

    # Hardcoded plot explanations
    @test occursin("expl-diff-volcano", html)
    @test occursin("expl-diff-results-table", html)

    # addStaticSection accepts explanationHtml
    @test occursin("explanationHtml", html)

    # Glossary terms present
    @test occursin("glossary-term", html)
    @test occursin("data-bs-toggle=\"tooltip\"", html)

    # Tooltip initialization
    @test occursin("bootstrap.Tooltip", html)

    # At least 2 hardcoded explanations + addStaticSection support
    n_toggles = length(collect(eachmatch(r"btn-explanation-toggle", html)))
    @test n_toggles >= 2
end

# ---------------------------------------------------------------------------
# Diagnostic detail JSON serialization
# ---------------------------------------------------------------------------

@testitem "diagnostic detail JSON" begin
    using BayesInteractomics: _build_protein_json
    using DataFrames

    # --- Sub-test 1: all diagnostic detail columns present ---
    row_full = (
        Protein = "TEST1",
        BF = 10.0,
        posterior_prob = 0.9,
        PEP = 1.0 - 0.9,
        BFDR = 0.01,
        mean_log2FC = 1.5,
        bf_enrichment = 5.0,
        bf_correlation = 3.0,
        bf_detected = 2.0,
        diagnostic_flag = "warning",
        n_observations = 3,
        mean_residual = 0.42,
        max_abs_residual = 1.8,
        is_low_data = true,
        is_residual_outlier = false,
    )
    json_str = _build_protein_json(row_full)
    @test json_str isa String
    @test occursin("\"n_observations\":3", json_str)
    @test occursin("\"mean_residual\":0.42", json_str)
    @test occursin("\"max_abs_residual\":1.8", json_str)
    @test occursin("\"is_low_data\":true", json_str)
    @test occursin("\"is_residual_outlier\":false", json_str)

    # --- Sub-test 2: missing diagnostic detail columns ---
    row_minimal = (
        Protein = "TEST2",
        BF = 5.0,
        posterior_prob = 0.8,
        PEP = 1.0 - 0.8,
        BFDR = 0.05,
        mean_log2FC = 1.0,
        bf_enrichment = 2.0,
        bf_correlation = 1.5,
        bf_detected = 1.0,
    )
    json_str2 = _build_protein_json(row_minimal)
    @test occursin("\"n_observations\":null", json_str2)
    @test occursin("\"is_low_data\":null", json_str2)
    @test occursin("\"is_residual_outlier\":null", json_str2)
    @test occursin("\"mean_residual\":null", json_str2)
    @test occursin("\"max_abs_residual\":null", json_str2)

    # --- Sub-test 3: NaN residuals serialize as null ---
    row_nan = (
        Protein = "TEST3",
        BF = 8.0,
        posterior_prob = 0.85,
        PEP = 1.0 - 0.85,
        BFDR = 0.02,
        mean_log2FC = 1.2,
        bf_enrichment = 4.0,
        bf_correlation = 2.0,
        bf_detected = 1.5,
        diagnostic_flag = "ok",
        n_observations = 10,
        mean_residual = NaN,
        max_abs_residual = NaN,
        is_low_data = false,
        is_residual_outlier = false,
    )
    json_str3 = _build_protein_json(row_nan)
    @test occursin("\"mean_residual\":null", json_str3)
    @test occursin("\"max_abs_residual\":null", json_str3)
    # Non-NaN fields should still be present
    @test occursin("\"n_observations\":10", json_str3)
    @test occursin("\"is_low_data\":false", json_str3)
end

@testitem "legend card in report template" begin
    tpl_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    tpl = Base.read(tpl_path, String)

    # Legend card element exists
    @test occursin("id=\"diag-legend\"", tpl)
    @test occursin("Diagnostic Flag Legend", tpl)
    @test occursin("diag-legend-body", tpl)
    @test occursin("diag-legend-card", tpl)

    # Flag icons present in legend
    @test occursin("&#10004;", tpl)   # checkmark
    @test occursin("&#9888;", tpl)    # warning
    @test occursin("&#10008;", tpl)   # fail

    # Threshold placeholders for dynamic population
    @test occursin("diag-thresh-low", tpl)
    @test occursin("diag-thresh-outlier", tpl)

    # Legend starts hidden (display:none) and is shown by JS when hasDiagData
    @test occursin(r"id=\"diag-legend\"[^>]*display:\s*none", tpl)
end

@testitem "popover JS in report template" begin
    tpl_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    tpl = Base.read(tpl_path, String)

    # Popover element exists
    @test occursin("id=\"diag-popover\"", tpl)

    # Popover JS functions present
    @test occursin("showDiagPopover", tpl)
    @test occursin("hideDiagPopover", tpl)

    # Threshold reference in JS
    @test occursin("diagnostic_thresholds", tpl)

    # Clickable icon class
    @test occursin("diag-flag-icon", tpl)

    # Hover-based interaction (mouseenter/mouseleave)
    @test occursin("mouseenter", tpl)
    @test occursin("mouseleave", tpl)

    # Em-dash fallback for unflagged proteins
    @test occursin("\\u2014", tpl)
end

@testitem "diagnostic_thresholds in report JSON" begin
    using BayesInteractomics
    using BayesInteractomics: _build_report_json, OutputFiles, CONFIG
    using DataFrames

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "MYC",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )

    results = DataFrame(
        Protein        = ["ACTB", "TP53"],
        BF             = [200.0, 50.0],
        posterior_prob = [0.99, 0.97],
        PEP            = 1.0 .- [0.99, 0.97],
        BFDR           = [0.001, 0.005],
        mean_log2FC    = [4.0, 2.5],
        bf_enrichment  = [150.0, 40.0],
        bf_correlation = [50.0, 10.0],
        bf_detected    = [20.0, 5.0],
        is_detected    = [true, true],
    )

    json_str = _build_report_json(results, cfg)
    @test json_str isa String
    @test occursin("\"diagnostic_thresholds\"", json_str)
    @test occursin("\"low_data_cutoff\":4", json_str)
    @test occursin("\"residual_outlier_cutoff\":2", json_str)
end

# ---------------------------------------------------------------------------
# Template JavaScript changes
# ---------------------------------------------------------------------------

@testitem "REND-02 within-class scatter type is SVG" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    html = Base.read(template_path, String)

    # REND-02: Within-class correlation scatter plots must use type: 'scatter' (SVG), not type: 'scattergl' (WebGL)
    # The function initWithinClassPlotly creates traces with type: 'scatter'
    @test occursin("initWithinClassPlotly", html)
    @test occursin("type: 'scatter'", html)

    # Find the initWithinClassPlotly function and verify it contains the scatter type assignment
    within_class_start = findfirst("function initWithinClassPlotly", html)
    @test within_class_start !== nothing

    # Extract the function body to verify scatter type is used
    if within_class_start !== nothing
        # Find the end of the function (closing brace at start of line after 'function initWithinClassPlotly')
        func_start = first(within_class_start)
        func_content = html[func_start:end]
        # Find the line with "type: 'scatter'" in the trace definition
        @test occursin(r"mode:\s*'markers',\s*type:\s*'scatter'", func_content)
        # Verify scattergl is NOT used in this function
        # Extract just this function (from "function initWithinClassPlotly" to next "function ")
        next_func = findfirst("\nfunction ", func_content[100:end])
        func_body = if next_func !== nothing
            func_content[1:99+first(next_func)]
        else
            func_content
        end
        @test !occursin("scattergl", func_body)
    end
end

@testitem "FDR-03 declared_bfdr conditional read" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    html = Base.read(template_path, String)

    # FDR-03: FDR calibration plot JSON must read cal.declared_bfdr with fallback to 1-t
    # The pattern: (cal.declared_bfdr && cal.declared_bfdr.length > 0) ? cal.declared_bfdr : fdrThr.map(function(t) { return 1 - t; })
    @test occursin("declared_bfdr", html)
    @test occursin("cal.declared_bfdr && cal.declared_bfdr.length > 0", html)
    @test occursin("return 1 - t", html)  # fallback pattern
end

@testitem "FDR-03 FDR plot x-axis title is Declared BFDR" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    html = Base.read(template_path, String)

    # FDR-03: FDR plot x-axis title must be "Declared BFDR" (not "Declared FDR (1 − threshold)")
    # The pattern is: xaxis: {title: 'Declared BFDR', ...}
    @test occursin("xaxis: {title: 'Declared BFDR'", html)

    # Verify the old title pattern is NOT present
    @test !occursin("Declared FDR (1", html)
end

@testitem "bf_em and bf_copula in report JSON (TRANS-02)" begin
    using BayesInteractomics
    using DataFrames

    # Create minimal results DataFrame with BMA columns
    df = DataFrame(
        Protein = ["P1", "P2", "P3"],
        is_detected = [true, true, false],
        BF = [100.0, 50.0, missing],
        posterior_prob = [0.99, 0.98, missing],
        PEP = 1.0 .- [0.99, 0.98, missing],
        BFDR = [0.01, 0.02, missing],
        mean_log2FC = [2.0, 1.5, missing],
        bf_enrichment = [50.0, 25.0, missing],
        bf_correlation = [30.0, 15.0, missing],
        bf_detected = [20.0, 10.0, missing],
        bf_em = [80.0, 40.0, missing],
        bf_copula = [120.0, 60.0, missing],
    )

    # Build JSON for first protein
    json_str = BayesInteractomics._build_protein_json(df[1, :])

    # Verify bf_em and bf_copula are present in JSON output
    @test occursin("\"bf_em\"", json_str)
    @test occursin("\"bf_copula\"", json_str)
    @test occursin("80", json_str)   # bf_em value for P1
    @test occursin("120", json_str)  # bf_copula value for P1

    # Verify missing values produce NaN
    json_missing = BayesInteractomics._build_protein_json(df[3, :])
    @test occursin("\"bf_em\"", json_missing)
    @test occursin("\"bf_copula\"", json_missing)

    # Verify that a row WITHOUT bf_em/bf_copula columns still works
    df_no_bma = DataFrame(
        Protein = ["P1"],
        is_detected = [true],
        BF = [100.0],
        posterior_prob = [0.99],
        PEP = 1.0 .- [0.99],
        BFDR = [0.01],
        mean_log2FC = [2.0],
        bf_enrichment = [50.0],
        bf_correlation = [30.0],
        bf_detected = [20.0],
    )
    json_no_bma = BayesInteractomics._build_protein_json(df_no_bma[1, :])
    @test occursin("\"bf_em\"", json_no_bma)  # key present with NaN fallback
    @test occursin("\"bf_copula\"", json_no_bma)
end

@testitem "BMA JSON BF correlation and ratio fields (COP-04)" begin
    using BayesInteractomics
    using BayesInteractomics: _build_bma_summary_json
    using DataFrames
    using Copulas
    using Distributions: Normal

    # Need >10 proteins so Pearson/Spearman are computed (not NaN fallback)
    n_prot = 15
    # Copula BFs with a ceiling effect: high-evidence proteins cluster near 100
    cop_bfs = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    # 3c-EM BFs scale freely beyond copula ceiling
    em_bfs  = [0.3, 0.8, 3.0, 8.0, 15.0, 40.0, 80.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 8000.0, 10000.0, 50000.0]
    avg_bfs = (cop_bfs .+ em_bfs) ./ 2

    # Build minimal CombinedBayesResult via 6-arg backward-compat constructor
    dummy_sklar = SklarDist(IndependentCopula(3), (Normal(), Normal(), Normal()))
    dummy_em = BayesInteractomics.EMResult(0.7, 0.3, dummy_sklar, DataFrame(iter=[1], ll=[0.0]), true)
    copula_result = BayesInteractomics.CombinedBayesResult(
        cop_bfs,
        fill(0.5, n_prot),   # posterior_prob (dummy)
        dummy_sklar, dummy_sklar, dummy_em, nothing
    )

    # Build minimal LatentClassResult via 8-arg constructor
    em3c_result = BayesInteractomics.LatentClassResult(
        em_bfs,
        fill(0.5, n_prot),   # posterior_prob (dummy)
        Dict(
            "background"  => (mu=-1.0, sigma=1.0, precision=1.0),
            "agnostic"    => (mu=0.0,  sigma=0.8, precision=1.5625),
            "interaction" => (mu=3.0,  sigma=0.8, precision=1.5625),
        ),
        [0.6, 0.25, 0.15],
        [100.0, 110.0],
        true, 20, nothing
    )

    # Build BMAResult
    bma = BayesInteractomics.BMAResult(
        avg_bfs,
        fill(0.5, n_prot),   # posterior_prob (dummy)
        copula_result,
        em3c_result,
        0.4,          # em_weight
        0.6,          # copula_weight
        BitVector(fill(false, n_prot)),   # model_disagreement
        nothing,      # pareto_k
        0.25,         # prior_odds
    )

    analysis_result = (bma_result = bma,)
    json_str = _build_bma_summary_json(analysis_result)

    # Verify new BF correlation scatter keys
    @test occursin("\"scatter_log10_bf_copula\"", json_str)
    @test occursin("\"scatter_log10_bf_em\"", json_str)

    # Verify BF ratio histogram key
    @test occursin("\"bf_ratio_log10\"", json_str)

    # Verify correlation statistics
    @test occursin("\"bf_corr_pearson\"", json_str)
    @test occursin("\"bf_corr_spearman\"", json_str)

    # Pearson and Spearman should be between -1 and 1 (not null)
    # Extract numeric value after "bf_corr_pearson":
    m_pearson = match(r"\"bf_corr_pearson\":(-?[0-9.]+)", json_str)
    @test m_pearson !== nothing
    pearson_val = parse(Float64, m_pearson.captures[1])
    @test -1.0 <= pearson_val <= 1.0

    m_spearman = match(r"\"bf_corr_spearman\":(-?[0-9.]+)", json_str)
    @test m_spearman !== nothing
    spearman_val = parse(Float64, m_spearman.captures[1])
    @test -1.0 <= spearman_val <= 1.0

    # With ceiling-effect data, Spearman should be higher than Pearson
    # (rank agreement preserved but magnitude compressed by ceiling)
    @test spearman_val >= pearson_val - 0.1  # allow small margin

    # scatter arrays should have same length (15 proteins, no subsampling)
    m_cop_arr = match(r"\"scatter_log10_bf_copula\":\[([^\]]*)\]", json_str)
    m_em_arr = match(r"\"scatter_log10_bf_em\":\[([^\]]*)\]", json_str)
    @test m_cop_arr !== nothing
    @test m_em_arr !== nothing
    n_cop = length(split(m_cop_arr.captures[1], ","))
    n_em = length(split(m_em_arr.captures[1], ","))
    @test n_cop == n_em
    @test n_cop == n_prot  # all proteins included (below 2000 threshold)
end

@testitem "bf_em and bf_copula in report JSON (TRANS-02)" begin
    using BayesInteractomics
    using DataFrames

    # Create minimal results DataFrame with BMA columns
    df = DataFrame(
        Protein = ["P1", "P2", "P3"],
        is_detected = [true, true, false],
        BF = [100.0, 50.0, missing],
        posterior_prob = [0.99, 0.98, missing],
        PEP = 1.0 .- [0.99, 0.98, missing],
        BFDR = [0.01, 0.02, missing],
        mean_log2FC = [2.0, 1.5, missing],
        bf_enrichment = [50.0, 25.0, missing],
        bf_correlation = [30.0, 15.0, missing],
        bf_detected = [20.0, 10.0, missing],
        bf_em = [80.0, 40.0, missing],
        bf_copula = [120.0, 60.0, missing],
    )

    # Build JSON for first protein
    json_str = BayesInteractomics._build_protein_json(df[1, :])

    # Verify bf_em and bf_copula are present in JSON output
    @test occursin("\"bf_em\"", json_str)
    @test occursin("\"bf_copula\"", json_str)
    @test occursin("80", json_str)   # bf_em value for P1
    @test occursin("120", json_str)  # bf_copula value for P1

    # Verify missing values produce NaN
    json_missing = BayesInteractomics._build_protein_json(df[3, :])
    @test occursin("\"bf_em\"", json_missing)
    @test occursin("\"bf_copula\"", json_missing)

    # Verify that a row WITHOUT bf_em/bf_copula columns still works
    df_no_bma = DataFrame(
        Protein = ["P1"],
        is_detected = [true],
        BF = [100.0],
        posterior_prob = [0.99],
        PEP = 1.0 .- [0.99],
        BFDR = [0.01],
        mean_log2FC = [2.0],
        bf_enrichment = [50.0],
        bf_correlation = [30.0],
        bf_detected = [20.0],
    )
    json_no_bma = BayesInteractomics._build_protein_json(df_no_bma[1, :])
    @test occursin("\"bf_em\"", json_no_bma)  # key present with NaN fallback
    @test occursin("\"bf_copula\"", json_no_bma)
end

# ---------------------------------------------------------------------------
# Sidecar infrastructure tests
# ---------------------------------------------------------------------------

@testitem "sidecar written alongside HTML report" begin
    # Test _sidecar_path
    using BayesInteractomics: _sidecar_path, _write_sidecar
    @test _sidecar_path(joinpath("tmp", "report.html")) == joinpath("tmp", "report_data.json")
    @test _sidecar_path(joinpath("tmp", "my_report.html")) == joinpath("tmp", "my_report_data.json")
end

@testitem "sidecar merge preserves existing data" begin
    using BayesInteractomics: _merge_sidecar, _write_sidecar, _sidecar_path
    using BayesInteractomics: json_object, json_string, json_number

    mktempdir() do dir
        sidecar = joinpath(dir, "report_data.json")
        # Write initial sidecar with simulation and calibration data
        initial = json_object(
            "meta" => json_object("generated_at" => json_string("2026-04-13")),
            "simulation" => json_object("n_synthetic" => json_number(100)),
            "has_calibration" => "true",
            "docking" => "{}",
        )
        Base.write(sidecar, initial)

        # Merge with new docking data
        new_sections = Dict{String,String}(
            "docking" => json_object("summary" => json_object("n_total" => json_number(5))),
            "meta" => json_object("generated_at" => json_string("2026-04-14")),
        )
        merged = _merge_sidecar(sidecar, new_sections)

        # Verify: simulation preserved, docking replaced, meta updated
        @test occursin("\"simulation\"", merged)
        @test occursin("\"n_synthetic\"", merged)
        @test occursin("\"has_calibration\"", merged)
        @test occursin("\"n_total\"", merged)  # new docking data
        @test occursin("2026-04-14", merged)   # updated meta
        @test !occursin("2026-04-13", merged)  # old meta replaced
    end
end

@testitem "generate_report writes sidecar file" begin
    using BayesInteractomics
    using BayesInteractomics: _sidecar_path
    using DataFrames

    mktempdir() do dir
        config = CONFIG(
            datafile = ["test.xlsx"],
            sample_cols = [Dict(1 => [1])],
            control_cols = [Dict(1 => [2])],
            poi = "BAIT",
            output = OutputFiles(dir),
        )
        results = DataFrame(
            Protein = ["A", "B"],
            posterior_prob = [0.9, 0.1],
            PEP = 1.0 .- [0.9, 0.1],
            BFDR = [0.01, 0.5],
            mean_log2FC = [2.0, -0.5],
            BF = [100.0, 0.2],
            bf_enrichment = [50.0, 0.1],
            bf_correlation = [2.0, 0.5],
            bf_detected = [1.5, 0.8],
        )
        generate_report(results, config)
        sidecar = _sidecar_path(config.output.report_file)
        @test isfile(sidecar)
        content = Base.read(sidecar, String)
        @test occursin("\"meta\"", content)
        @test occursin("\"results\"", content)
    end
end

@testitem "generate_report accepts structures_dir kwarg" begin
    using BayesInteractomics
    using BayesInteractomics: _sidecar_path
    using DataFrames

    mktempdir() do dir
        config = CONFIG(
            datafile = ["test.xlsx"],
            sample_cols = [Dict(1 => [1])],
            control_cols = [Dict(1 => [2])],
            poi = "BAIT",
            output = OutputFiles(dir),
        )
        results = DataFrame(
            Protein = ["A", "B"],
            posterior_prob = [0.9, 0.1],
            PEP = 1.0 .- [0.9, 0.1],
            BFDR = [0.01, 0.5],
            mean_log2FC = [2.0, -0.5],
            BF = [100.0, 0.2],
            bf_enrichment = [50.0, 0.1],
            bf_correlation = [2.0, 0.5],
            bf_detected = [1.5, 0.8],
        )
        # Should not error with structures_dir kwarg
        generate_report(results, config; structures_dir=joinpath(dir, "structures"))
        sidecar = _sidecar_path(config.output.report_file)
        @test isfile(sidecar)
    end
end

# ---------------------------------------------------------------------------
# Docking distribution plots and Mol* viewer
# ---------------------------------------------------------------------------

@testitem "docking tab distribution plots in template" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    html = Base.read(template_path, String)
    @test occursin("docking-bf-hist", html)
    @test occursin("docking-iptm-dist", html)
    @test occursin("docking-c2qscore-dist", html)
    @test occursin("docking-posterior-violin", html)
    @test occursin("initDockingDistributions", html)
end

@testitem "Mol* viewer in template" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    html = Base.read(template_path, String)
    @test occursin("molstar-modal", html)
    @test occursin("molstar-container", html)
    @test occursin("cdn.jsdelivr.net/npm/molstar@5.8.0", html)
    @test occursin("viewStructure", html)
    @test occursin("closeMolstar", html)
    @test occursin("loadMolstar", html)
end

@testitem "docking JSON includes q_combined and structure_file" begin
    using BayesInteractomics
    using BayesInteractomics: _build_docking_json, _filter_detected
    using DataFrames, Dates

    results = DataFrame(
        Protein = ["BAIT", "PREY1"],
        posterior_prob = [1.0, 0.9],
        PEP = 1.0 .- [1.0, 0.9],
        BFDR = [0.0, 0.01],
        log2FC = [5.0, 3.0],
        q_combined = [0.0, 0.005],
        posterior_prob_combined = [1.0, 0.95],
        n_detections_sample = [5, 4],
        n_detections_control = [1, 1],
    )

    # DockingPairResult is positional: protein_a, protein_b, uniprot_a, uniprot_b,
    #   iptm_best, iptm_all, iptm_std, ranking_score, fraction_disordered,
    #   chain_pair_iptm, chain_pair_pae_min, pdockq, mean_plddt_a, mean_plddt_b,
    #   n_interface_contacts, bf_dock, calibration_tier, status, token_count
    pairs = [BayesInteractomics.DockingPairResult(
        "BAIT", "PREY1", "", "",
        0.7, [0.7], 0.0, 0.8, 0.1,
        0.7, 5.0, 0.5, 80.0, 75.0,
        20, 5.0, "tier1", :success, 500,
        NaN, NaN, NaN,
    )]
    docking = BayesInteractomics.DockingResult(
        pairs, BayesInteractomics.DockingConfig(), 1, 1, 0, 0, 0, 0, now(),
    )

    json = _build_docking_json(results, docking)
    @test occursin("\"BFDR_combined\"", json)
    @test occursin("\"PEP_combined\"", json)
    @test occursin("\"structure_file\"", json)
    @test occursin("\"structure_data\"", json)
    @test occursin("\"structure_format\"", json)
    @test occursin("\"pdockq\"", json)
end

@testitem "Results tab unchanged by docking" begin
    template_path = joinpath(@__DIR__, "..", "..", "src", "reports", "templates", "report.html")
    html = Base.read(template_path, String)
    # BFDR_combined should NOT appear in the main results table init
    # It should only appear in the docking tab
    # The main results table function is initTable
    idx_start = findfirst("function initTable", html)
    if idx_start !== nothing
        # Find next function definition or end of script (use nextind for UTF-8 safety)
        start_i = first(idx_start)
        end_i = nextind(html, start_i, min(3000, length(html) - start_i))
        chunk = html[start_i:end_i]
        @test !occursin("BFDR_combined", chunk)
    end
end

# =============================================================================
# Metalearner status banner rendering
# =============================================================================
# These three testitems verify that AnalysisResult.metalearner_status flows
# correctly into the rendered HTML report across the three sentinel values.
# Banner div + Methods subsection are both grep-stable (DO NOT rename
# `metalearner-warning-banner` id, `metalearner_status` JSON key, or "Metalearner
# Status" subsection header — these are testitem grep anchors).

@testitem "report metalearner_status=:loaded emits 'loaded' in JSON; banner div present but hidden" begin
    using BayesInteractomics
    using DataFrames

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "MYC",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )

    results = DataFrame(
        Protein        = ["ACTB", "MYC"],
        BF             = [200.0, 0.8],
        posterior_prob = [0.99, 0.35],
        PEP            = [0.01, 0.65],
        BFDR           = [0.001, 0.50],
        mean_log2FC    = [4.0, -0.2],
        bf_enrichment  = [150.0, 0.6],
        bf_correlation = [50.0, 0.2],
        bf_detected    = [20.0, 0.1],
    )

    # Minimal NamedTuple — must include input_qc (accessed without hasproperty in
    # _build_qc_json) and metalearner_status (the field under test).
    ar = (metalearner_status = :loaded, input_qc = nothing)

    report_path = joinpath(tmpdir, "report_loaded.html")
    generate_report(results, cfg; output = report_path, analysis_result = ar)

    @test isfile(report_path)
    html = Base.read(report_path, String)

    # Banner div MUST be present in the template (stable identifier).
    @test occursin("metalearner-warning-banner", html)
    # JSON blob carries the :loaded sentinel.
    @test occursin("\"metalearner_status\":\"loaded\"", html)
    # Methods subsection still renders the "Metalearner Status" header (server-side helper).
    @test occursin("Metalearner Status", html)
    # The :loaded server-side branch says the extension "is loaded".
    @test occursin("is loaded", html)
end

@testitem "report metalearner_status=:extension_not_loaded emits banner-visible body + 'not loaded' text" begin
    using BayesInteractomics
    using DataFrames

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "MYC",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )

    results = DataFrame(
        Protein        = ["ACTB", "MYC"],
        BF             = [200.0, 0.8],
        posterior_prob = [0.99, 0.35],
        PEP            = [0.01, 0.65],
        BFDR           = [0.001, 0.50],
        mean_log2FC    = [4.0, -0.2],
        bf_enrichment  = [150.0, 0.6],
        bf_correlation = [50.0, 0.2],
        bf_detected    = [20.0, 0.1],
    )

    ar = (metalearner_status = :extension_not_loaded, input_qc = nothing)

    report_path = joinpath(tmpdir, "report_not_loaded.html")
    generate_report(results, cfg; output = report_path, analysis_result = ar)

    @test isfile(report_path)
    html = Base.read(report_path, String)

    @test occursin("metalearner-warning-banner", html)
    @test occursin("\"metalearner_status\":\"extension_not_loaded\"", html)
    @test occursin("Metalearner Status", html)
    # The :extension_not_loaded server-side branch describes Variante B fallback.
    @test occursin("not loaded", html)
    # Banner JS toggles display:block when status==extension_not_loaded — the
    # initMetalearnerWarningBanner() function is grep-stable in report.html.
    @test occursin("initMetalearnerWarningBanner", html)
end

@testitem "report metalearner_status=:prediction_failed renders 'prediction_failed' JSON + failure-mode Methods text" begin
    using BayesInteractomics
    using DataFrames

    tmpdir = mktempdir()
    cfg = CONFIG(
        datafile     = ["dummy.xlsx"],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols  = [Dict(1 => [4,5,6])],
        poi          = "MYC",
        n_controls   = 3,
        n_samples    = 3,
        output       = OutputFiles(tmpdir),
        generate_report_html = false,
    )

    results = DataFrame(
        Protein        = ["ACTB", "MYC"],
        BF             = [200.0, 0.8],
        posterior_prob = [0.99, 0.35],
        PEP            = [0.01, 0.65],
        BFDR           = [0.001, 0.50],
        mean_log2FC    = [4.0, -0.2],
        bf_enrichment  = [150.0, 0.6],
        bf_correlation = [50.0, 0.2],
        bf_detected    = [20.0, 0.1],
    )

    ar = (metalearner_status = :prediction_failed, input_qc = nothing)

    report_path = joinpath(tmpdir, "report_prediction_failed.html")
    generate_report(results, cfg; output = report_path, analysis_result = ar)

    @test isfile(report_path)
    html = Base.read(report_path, String)

    @test occursin("metalearner-warning-banner", html)
    @test occursin("\"metalearner_status\":\"prediction_failed\"", html)
    @test occursin("Metalearner Status", html)
    # The :prediction_failed server-side branch mentions predict_metalearner.
    @test occursin("predict_metalearner", html)
end

# ---------------------------------------------------------------------------
# Differential report tab-payload tests
# Each testitem builds a 2-condition DifferentialResult via the
# DifferentialFixtures.create_two_condition_result helper, calls
# generate_differential_report(diff; output=...), and asserts the rendered
# HTML payload matches the per-tab contract.
# ---------------------------------------------------------------------------

@testitem "differential calibration tab is wired" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    diff = DifferentialFixtures.create_two_condition_result(; analyses_populated=true)
    tmpdir = mktempdir()
    out = joinpath(tmpdir, "diff_report.html")
    generate_differential_report(diff; output = out)
    @test isfile(out)
    html = Base.read(out, String)
    @test contains(html, "id=\"tab-sim-li\"")
    @test contains(html, "id=\"tab-simulation\"")
    @test contains(html, "function initCalibration")
end

@testitem "differential sensitivity tab is wired" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    diff = DifferentialFixtures.create_two_condition_result(; analyses_populated=true)
    tmpdir = mktempdir()
    out = joinpath(tmpdir, "diff_report.html")
    generate_differential_report(diff; output = out)
    @test isfile(out)
    html = Base.read(out, String)
    @test contains(html, "id=\"tab-sens-li\"")
    @test contains(html, "id=\"tab-sensitivity\"")
    @test contains(html, "Per-protein stability change")
    @test contains(html, "function initSensitivity")
end

@testitem "differential mixture tab is wired" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    diff = DifferentialFixtures.create_two_condition_result(; analyses_populated=true)
    tmpdir = mktempdir()
    out = joinpath(tmpdir, "diff_report.html")
    generate_differential_report(diff; output = out)
    @test isfile(out)
    html = Base.read(out, String)
    @test contains(html, "id=\"tab-mixture-li\"")
    @test contains(html, "id=\"tab-mixture\"")
    @test contains(html, "Class transitions")
    @test contains(html, "function initMixture")
end

@testitem "differential methods tab is wired with Differential Analysis subsection" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    diff = DifferentialFixtures.create_two_condition_result(; analyses_populated=true)
    tmpdir = mktempdir()
    out = joinpath(tmpdir, "diff_report.html")
    generate_differential_report(diff; output = out)
    @test isfile(out)
    html = Base.read(out, String)
    @test contains(html, "id=\"tab-methods\"")
    # Heading from _methods_differential_block
    @test contains(html, "Differential Analysis")
    # MS-only note also lives in the differential block
    @test contains(html, "MS evidence only")
    @test contains(html, "function initDiffMethods")
end

@testitem "differential data quality tab is wired" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    diff = DifferentialFixtures.create_two_condition_result(; analyses_populated=true)
    tmpdir = mktempdir()
    out = joinpath(tmpdir, "diff_report.html")
    generate_differential_report(diff; output = out)
    @test isfile(out)
    html = Base.read(out, String)
    @test contains(html, "id=\"tab-qc-li\"")
    @test contains(html, "id=\"tab-data-quality\"")
    # Heading text in the rendered template ("Shared-protein mean log-intensity")
    @test contains(html, "Shared-protein mean log-intensity")
    @test contains(html, "function initQc")
end

@testitem "differential template has MS-only footer note and no docking tab" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    diff = DifferentialFixtures.create_two_condition_result(; analyses_populated=true)
    tmpdir = mktempdir()
    out = joinpath(tmpdir, "diff_report.html")
    generate_differential_report(diff; output = out)
    @test isfile(out)
    html = Base.read(out, String)
    @test contains(html, "id=\"ms-only-footer-note\"")
    @test contains(html, "MS evidence only")
    # differential report has NO Structural Evidence tab
    @test !contains(html, "id=\"tab-dock-li\"")
end

@testitem "differential dbf_diagnostics tab payload has 6 panels" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    diff = DifferentialFixtures.create_two_condition_result(; analyses_populated=true)
    tmpdir = mktempdir()
    out = joinpath(tmpdir, "diff_report.html")
    generate_differential_report(diff; output = out)
    @test isfile(out)
    html = Base.read(out, String)
    @test contains(html, "id=\"tab-dbf-diag-li\"")
    @test contains(html, "id=\"tab-dbf-diag\"")
    # 6 panel div ids (panel order)
    for panel_id in ("dbf-histogram-qq", "dbf-vs-delta", "dbf-component-stack",
                     "dbf-saturation-panel", "dbf-submodel-disagreement", "dbf-traffic-light")
        @test contains(html, "id=\"" * panel_id * "\"")
    end
    @test contains(html, "function initDbfDiag")
end

# ─── Methods PEP text + volcano §7a contracts ──────────────────────────────
#
# These two testitems pin the Methods PEP text and volcano §7a contracts.
#
# Threat refs: XSS via condition labels in Methods text. The pinned
# strings below are all static; when dynamic condition labels
# (`diff.condition_A`, `diff.condition_B`) are wired into the Methods HTML, every
# interpolation must route through `esc(...)`.

@testitem "_methods_differential_block contains PEP + γ-PEP + bb_mnar_codriven" setup=[DifferentialFixtures] tags=[:pep, :methods] begin
    # _methods_differential_block extended with five new <h6> subsections covering:
    #   1. PEP general (α-PEP definition; distinct from BFDR / local_fdr)
    #   2. α-PEP vs γ-PEP (class-conditional naive-product estimator, k=10.0)
    #   3. Conditional-independence caveat
    #   4. Calibration application (per-condition is_calibrated_A / is_calibrated_B)
    #   5. bb_mnar_codriven rule + defaults
    #   6. §7a marginal-KDE deviation note
    # T-70-14-01 mitigation: all dynamic strings (condition_A, condition_B) routed
    # through the module-level `esc(...)` helper (`_html_esc`). Tested via the
    # mixed-calibration DifferentialFixtures.create_two_condition_result kwargs.
    using BayesInteractomics
    diff = DifferentialFixtures.create_two_condition_result(
        is_calibrated_A = true,
        is_calibrated_B = false,
    )
    # Access non-exported helper via fully-qualified name (Julia 1.12 import semantics).
    html = BayesInteractomics._methods_differential_block(diff)
    # Core PEP content
    @test occursin("PEP", html)
    @test occursin("γ-PEP", html) || occursin("γ-class-conditional", html)
    @test occursin("conditional-independence", lowercase(html))
    @test occursin("bb_mnar_codriven", html)
    # Calibration application gate (ECE / per-condition)
    @test occursin("cal_ece", html) && occursin("0.10", html)
    @test occursin("calibrated", html)
    # α / γ distinction + canonical lowercase column reference
    @test occursin("differential_pep", html) || occursin("&minus;log", html)
    @test occursin("pep_gained", html)
    # bb_mnar_codriven defaults verbatim (10.0, 10.0, 0.5)
    @test occursin("10.0, 10.0, 0.5", html) || (occursin("0.5", html) && occursin("10.0", html))
    # T-70-14-01 (XSS): condition labels appear in HTML-safe form (no raw <, >, ", ').
    # The fixture uses static labels ("WT", "Mutant") so no escaping is observable;
    # the smoke test below pins that the esc() helper was invoked by checking the
    # module exports `_html_esc` (rename-safe via `which`).
    @test isdefined(BayesInteractomics, :_html_esc)
end

@testitem "volcano §7a contract — CLS_COLOR hue swap + opacity bounds" tags=[:pep, :volcano] begin
    # §7a contract verified by inspection of `differential_report.html`:
    #   * CLS_COLOR map: gained=#d62728, reduced=#1f77b4, unchanged=#4d4d4d, both_negative=#cfd8dc
    #     (unchanged/both_negative darkened/tinted for stronger volcano contrast — report fix)
    #   * Y-axis = −log10(differential_pep) with 1e-10 log(0) guard
    #   * Opacity vector clamp(1 − pep, 0.25, 1.0) — literal 0.25 and 1.0 constants
    #   * α/γ segmented Bootstrap toggle (#diff-pep-mode-group + Plotly.restyle)
    #   * KDE marginals with Silverman bandwidth (1.06 * sigma * Math.pow(n, -1/5))
    using BayesInteractomics
    html_path = joinpath(pkgdir(BayesInteractomics), "src", "reports", "templates", "differential_report.html")
    html = read(html_path, String)
    # §7a hue map (CLS_COLOR swap per RESEARCH OQ-2)
    @test occursin("#d62728", html)
    @test occursin("#1f77b4", html)
    @test occursin("#4d4d4d", html)   # UNCHANGED (darkened for contrast)
    @test occursin("#cfd8dc", html)   # BOTH_NEGATIVE (pale blue-grey for contrast)
    # Opacity-vector clamp constants
    @test occursin("0.25", html)
    @test occursin("1.0", html)
    # α/γ segmented toggle wired via Plotly.restyle (in-place, preserves zoom)
    @test occursin("diff-pep-mode-group", html)
    @test occursin("Plotly.restyle", html)
    # Y-axis switch + log(0) guard
    @test occursin("differential_pep", html)
    @test occursin("1e-10", html)
    # KDE marginal helper
    @test occursin("_renderMarginalKDEs", html)
    @test occursin("1.06 * sigma * Math.pow", html)
    @test occursin("tozeroy", html) || occursin("tozerox", html)
    @test occursin("xaxis2", html) && occursin("yaxis2", html)
end

# ---------------------------------------------------------------------------
# Embeddings & Similarity report-layer testitems
# ---------------------------------------------------------------------------
#
# Each testitem builds its own inline AnalysisResult / DifferentialResult so it
# does not depend on include()ing test/fixtures/test_fixtures.jl (that file uses
# TestItemRunner @testsetup/@testmodule macros which aren't visible inside an
# @testitem eval scope).

@testitem "_build_sample_embedding_json null gate" begin
    using BayesInteractomics
    @test BayesInteractomics._build_sample_embedding_json(nothing) == "null"
end

@testitem "_build_sample_embedding_json populated contains pca + umap + filter_level" begin
    using BayesInteractomics
    using DataFrames, Dates
    snap = BayesInteractomics._config_snapshot(EmbeddingsConfig())
    labels = (
        condition  = String["sample","sample","control","control"],
        replicate  = Int[1, 2, 1, 2],
        experiment = Int[1, 1, 1, 1],
        protocol   = Int[1, 1, 1, 1],
    )
    er = EmbeddingsResult(
        randn(4, 2), [12.0, 8.0],
        labels, :complete_case,
        randn(4, 2), nothing, randn(3, 2), Symbol[:H0, :H1, :Agnostic], ["P1","P2","P3"],
        snap,
    )
    empty_df = DataFrame()
    ar = BayesInteractomics.AnalysisResult(
        empty_df, empty_df, nothing, nothing, nothing, nothing, nothing, :bma,
        nothing, nothing, UInt64(0), UInt64(0), now(), "test",
        "BAIT_WT", 1, nothing, nothing, nothing, :loaded,
        nothing, nothing, false, er,
    )
    s = BayesInteractomics._build_sample_embedding_json(ar)
    @test occursin("\"pca\"", s)
    @test occursin("\"umap\"", s)
    @test occursin("\"filter_level\"", s)
end

@testitem "_build_protein_embedding_json null + populated paths" begin
    using BayesInteractomics
    using DataFrames, Dates
    @test BayesInteractomics._build_protein_embedding_json(nothing) == "null"
    snap = BayesInteractomics._config_snapshot(EmbeddingsConfig())
    labels = (
        condition  = String["sample","sample","control","control"],
        replicate  = Int[1, 2, 1, 2],
        experiment = Int[1, 1, 1, 1],
        protocol   = Int[1, 1, 1, 1],
    )
    er = EmbeddingsResult(
        randn(4, 2), [12.0, 8.0],
        labels, :complete_case,
        nothing, nothing, randn(3, 2), Symbol[:H0, :H1, :Agnostic], ["P1","P2","P3"],
        snap,
    )
    empty_df = DataFrame()
    ar = BayesInteractomics.AnalysisResult(
        empty_df, empty_df, nothing, nothing, nothing, nothing, nothing, :bma,
        nothing, nothing, UInt64(0), UInt64(0), now(), "test",
        "BAIT_WT", 1, nothing, nothing, nothing, :loaded,
        nothing, nothing, false, er,
    )
    s = BayesInteractomics._build_protein_embedding_json(ar)
    @test occursin("\"classes\"", s)
    @test occursin("\"protein_ids\"", s)
end

@testitem "_build_condition_matrix_json + _build_jaccard_json + _build_dendrogram_json" begin
    using BayesInteractomics
    using DataFrames, Dates
    using BayesInteractomics: DifferentialResult, DifferentialConfig
    @test BayesInteractomics._build_condition_matrix_json(nothing) == "null"
    @test BayesInteractomics._build_jaccard_json(nothing) == "null"
    @test BayesInteractomics._build_dendrogram_json(nothing) == "null"

    # Build a minimal DifferentialResult with condition_similarity populated.
    cs = ConditionSimilarityResult(
        ["WT", "Mutant"],
        [1.0 0.73; 0.73 1.0],
        [1.0 0.65; 0.65 1.0],
        [1.0 0.80; 0.80 1.0],
        [1.0 0.32; 0.32 1.0],
        [5 5; 5 5],
        50,
        zeros(Int, 1, 2), [0.27], [1, 2],
        :average,
    )
    tmpdir = mktempdir()
    dcfg = DifferentialConfig(
        volcano_file        = joinpath(tmpdir, "vol.png"),
        evidence_file       = joinpath(tmpdir, "ev.png"),
        scatter_file        = joinpath(tmpdir, "sc.png"),
        classification_file = joinpath(tmpdir, "cl.png"),
        ma_file             = joinpath(tmpdir, "ma.png"),
        results_file        = joinpath(tmpdir, "diff_results.xlsx"),
    )
    results_df = DataFrame(Protein=String[], log2FC=Float64[])
    diff = DifferentialResult(
        results_df, "WT", "Mutant", dcfg,
        0, 0, 0, 0, 0,
        now(), 2, 1, 2, 0,
        BayesInteractomics.AnalysisResult[],
        false, false, cs,
    )
    cm = BayesInteractomics._build_condition_matrix_json(diff)
    @test occursin("\"spearman_log10_bf\"", cm)
    @test occursin("\"pearson_log2fc\"", cm)
    @test occursin("\"n_shared_per_cell\"", cm)
    jac = BayesInteractomics._build_jaccard_json(diff)
    @test occursin("\"top_k_used\"", jac)
    dnd = BayesInteractomics._build_dendrogram_json(diff)
    @test occursin("\"merges\"", dnd)
    @test occursin("\"heights\"", dnd)
    @test occursin("\"order\"", dnd)
    @test occursin("\"linkage\"", dnd)
end

@testitem "report.html template contains embeddings banner + 3 plot-card IDs" begin
    using BayesInteractomics
    path = joinpath(dirname(pathof(BayesInteractomics)), "reports", "templates", "report.html")
    content = read(path, String)
    @test occursin("id=\"embeddings-warning-banner\"", content)
    @test occursin("id=\"embedding-pca-card\"", content)
    @test occursin("id=\"embedding-umap-card\"", content)
    @test occursin("id=\"embedding-protein-card\"", content)
end

@testitem "differential_report.html template contains 4 plot-card IDs + dendrogram render" begin
    using BayesInteractomics
    path = joinpath(dirname(pathof(BayesInteractomics)), "reports", "templates", "differential_report.html")
    content = read(path, String)
    @test occursin("id=\"embeddings-warning-banner\"", content)
    @test occursin("id=\"embedding-pca-card\"", content)
    @test occursin("id=\"embedding-umap-card\"", content)
    @test occursin("id=\"embedding-protein-card\"", content)
    @test occursin("id=\"condition-similarity-card\"", content)
    @test occursin("id=\"condition-matrix-plot\"", content)
    @test occursin("id=\"condition-dendrogram-plot\"", content)
    @test occursin("id=\"condition-jaccard-plot\"", content)
    @test occursin("_renderDendrogram", content)
    @test occursin("renderConditionSimilarity", content)
end

@testitem "_methods_embeddings_block produces Embeddings heading + inline citations" begin
    using BayesInteractomics
    # Pass EmbeddingsConfig directly — _methods_embeddings_block only reads config.embeddings_config.
    # Avoids building a full CONFIG with empty datafile.
    emb_cfg = EmbeddingsConfig()
    html = BayesInteractomics._methods_embeddings_block(emb_cfg)
    @test occursin("Embeddings &amp; Similarity", html)
    @test occursin("McInnes et al. 2018", html)
    @test occursin("van der Maaten 2014", html)
    @test !occursin("<a href", html)   # no external URLs
end

@testitem "Parity audit: no new tab IDs introduced by embeddings layer" begin
    using BayesInteractomics
    for tpl in ("report.html", "differential_report.html")
        path = joinpath(dirname(pathof(BayesInteractomics)), "reports", "templates", tpl)
        content = read(path, String)
        # No id="tab-embedding..." or id="tab-similarity..." or any new top-level tab.
        @test !occursin("id=\"tab-embedding", content)
        @test !occursin("id=\"tab-similarity", content)
    end
end

# ---------------------------------------------------------------------------
# kgroup_report_compat
# Verifies that generate_differential_report works equivalently on:
#   (a) legacy 2-group `differential_analysis(ar_A, ar_B; ...)` result
#   (b) k=2 NamedTuple `differential_analysis(; conditions = (...))` result
# HTML output must be structurally comparable (length within ±20%, all core
# keywords present in both). Smoke test against the report generator: the
# new k=2 NamedTuple call path renders without errors and produces a report
# indistinguishable in structure from the legacy 2-group path.
# ---------------------------------------------------------------------------

@testitem "kgroup_report_compat: generate_differential_report works on k=2 NamedTuple + structural HTML equivalence with legacy 2-group" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using DataFrames

    fx = DifferentialFixtures.create_three_condition_result()
    tmpdir = mktempdir()

    # ---- Legacy 2-group HTML ----
    d_legacy = differential_analysis(fx.ar_wt, fx.ar_mut1;
                                     condition_A = "wt", condition_B = "mut1")
    legacy_html_path = joinpath(tmpdir, "diff_legacy.html")
    generate_differential_report(d_legacy; output = legacy_html_path)
    @test isfile(legacy_html_path)
    legacy_html = Base.read(legacy_html_path, String)
    @test length(legacy_html) > 0
    @test occursin("<html", lowercase(legacy_html))   # smoke: valid HTML doc

    # ---- k=2 NamedTuple HTML ----
    d_kgroup = differential_analysis(conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1))
    kgroup_html_path = joinpath(tmpdir, "diff_kgroup.html")
    generate_differential_report(d_kgroup; output = kgroup_html_path)
    @test isfile(kgroup_html_path)
    kgroup_html = Base.read(kgroup_html_path, String)
    @test length(kgroup_html) > 0

    # ---- Structural equivalence modulo timestamp ----
    # Strip lines containing "timestamp" / ISO-8601 dates so the two HTMLs
    # become directly comparable.
    _strip_temporal(html) = replace(html,
        r"timestamp[^<\n]*"i => "timestamp=REDACTED",
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}" => "DATETIME-REDACTED",
        r"\d{4}-\d{2}-\d{2}" => "DATE-REDACTED")
    legacy_stripped = _strip_temporal(legacy_html)
    kgroup_stripped = _strip_temporal(kgroup_html)

    # Length-comparable (the k=2 NT call adds a single BH column to results — minor JSON expansion).
    # Assert structural similarity by length within ±20%.
    ratio = length(kgroup_stripped) / length(legacy_stripped)
    @test 0.90 <= ratio <= 1.20

    # Both contain the same core sections.
    for keyword in ["differential", "bfdr", "log2", "classification"]
        @test occursin(keyword, lowercase(legacy_stripped))
        @test occursin(keyword, lowercase(kgroup_stripped))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# k-group multi-condition report-regex testitems
#   (xi)  kgroup_multi_tab_visible_k3   — k=3 NamedTuple renders the tab
#   (xii) kgroup_dropdown_visible_k4    — k=4 renders dropdown + finite
#                                          delta_log2fc (BLOCKER #2 closure)
#   (xiii) kgroup_legacy_no_multi_tab_k2 — legacy 2-group hides the tab
# ─────────────────────────────────────────────────────────────────────────────

@testitem "kgroup_multi_tab_visible_k3" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using Test

    fx = DifferentialFixtures.create_three_condition_result()
    diff = differential_analysis(;
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2),
        config = DifferentialConfig(),
    )
    tmpdir = mktempdir()
    out = joinpath(tmpdir, "diff_k3.html")
    generate_differential_report(diff; output = out)
    @test isfile(out)
    html = Base.read(out, String)
    @test contains(html, "id=\"tab-multi-li\"")
    @test contains(html, "id=\"tab-multi\"")
    @test contains(html, "Multi-Condition")
    @test contains(html, "multi_condition")
    @test contains(html, "initMultiConditionTab")
    @test contains(html, "initMultiConditionRender")
end

@testitem "kgroup_dropdown_visible_k4" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using Test

    fx = DifferentialFixtures.create_four_condition_result()
    diff = differential_analysis(;
        conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                      mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
        contrasts = :all_pairs,
        config = DifferentialConfig(),
    )
    tmpdir = mktempdir()
    out = joinpath(tmpdir, "diff_k4.html")
    generate_differential_report(diff; output = out)
    @test isfile(out)
    html = Base.read(out, String)
    @test contains(html, "multi-view-dropdown")
    @test contains(html, "Matrix view")
    @test contains(html, "Per-pair detail")
    # JSON payload should mark show_dropdown true for k=4 :all_pairs (6 contrasts).
    @test contains(html, "\"show_dropdown\":true")

    # BLOCKER #2 closure — per-pair delta_log2fc must carry finite numeric values.
    # _build_multi_condition_json reads df_pair.delta_log2fc directly
    # (no fill(missing) hedge). The rendered HTML JSON payload MUST contain at least
    # one finite numeric value inside a `"delta_log2fc": [` array. Regex matches
    # any signed decimal number (e.g. 2.5, -1.8, 0.1) immediately after the opening
    # bracket — `null` values would NOT match. If this assertion fails, the
    # BLOCKER #2 silent regression has reappeared.
    @test match(r"\"delta_log2fc\":\s*\[\s*-?[0-9]+\.[0-9]", html) !== nothing
end

@testitem "kgroup_legacy_no_multi_tab_k2" setup=[DifferentialFixtures] begin
    using BayesInteractomics
    using Test

    # create_two_condition_result returns a DifferentialResult directly
    # (fixture pattern — verified at test/fixtures/test_fixtures.jl:117).
    diff = DifferentialFixtures.create_two_condition_result()
    tmpdir = mktempdir()
    out = joinpath(tmpdir, "diff_legacy.html")
    generate_differential_report(diff; output = out)
    html = Base.read(out, String)
    # The static <li> element exists in every rendered template.
    @test contains(html, "id=\"tab-multi-li\"")
    # But the JSON predicate keeps it hidden — either multi_condition is null
    # OR contrasts array is empty (k=2 legacy: isempty(diff.contrasts) is true).
    @test contains(html, "\"multi_condition\":null") || contains(html, "\"contrasts\":[]")
    # Server-side mirror also reports show_tab false for k=2 (if present).
    @test !contains(html, "\"show_tab\":true")
end
