# src/reports/report_generator.jl
# Main interactive HTML report generator.

"""
    generate_report(results::DataFrame, config::CONFIG; output=config.output.report_file)

Generate a self-contained interactive HTML report from analysis results.

The report includes:
- Interactive volcano plot and rank-rank plot (Plotly.js, CDN)
- Searchable/filterable results table (DataTables.js, CDN)
- Static PNG embeds for evidence, diagnostic, and sensitivity plots
- Auto-generated methods section text
- Reproducibility metadata

The generated HTML file can be opened in any modern browser without
any additional software installation.

# Arguments
- `results::DataFrame`: Final results DataFrame from `run_analysis()`.
- `config::CONFIG`: Analysis configuration (used for parameters and file paths).

# Keywords
- `output::String`: Path for the generated HTML file. Defaults to `config.output.report_file`.

# Example
```julia
results, ar = run_analysis(config)
generate_report(results, config)
```
"""
function generate_report(results::DataFrame, config::CONFIG;
                         output::String = config.output.report_file)::Nothing
    @info "Generating interactive HTML report..."

    # Build the full JSON data blob
    json_blob = _build_report_json(results, config)

    # Load HTML template
    template_path = joinpath(@__DIR__, "templates", "report.html")
    if !isfile(template_path)
        @warn "Report template not found at $template_path; skipping report generation."
        return nothing
    end
    template = Base.read(template_path, String)

    # Inject data
    html = replace(template, "{{REPORT_DATA_JSON}}" => json_blob)

    # Write output
    mkpath(dirname(output))
    Base.write(output, html)
    @info "Interactive report saved to: $output"

    # Also write standalone methods text
    methods_path = config.output.report_methods_file
    try
        methods_text = generate_methods_text(config, results)
        mkpath(dirname(methods_path))
        Base.write(methods_path, methods_text)
        @info "Methods text saved to: $methods_path"
    catch e
        @warn "Failed to write methods file: $e"
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""Build the complete JSON string to be embedded in the HTML template."""
function _build_report_json(results::DataFrame, config::CONFIG)::String
    meta_json   = _build_meta_json(results, config)
    results_json = _build_results_json(results)
    plots_json  = _build_plots_json(config)
    methods_json = _build_methods_json(config, results)

    return json_object(
        "meta"    => meta_json,
        "results" => results_json,
        "plots"   => plots_json,
        "methods" => methods_json,
    )
end

"""Build meta/dashboard JSON object."""
function _build_meta_json(results::DataFrame, config::CONFIG)::String
    n_proteins   = nrow(results)
    n_sig        = sum(ismissing.(results.q) .== false .&& results.q .≤ 0.05)
    n_strong     = sum(ismissing.(results.q) .== false .&& results.q .≤ 0.01)

    return json_object(
        "bait"            => json_string(config.poi),
        "n_proteins"      => json_number(n_proteins),
        "n_significant"   => json_number(n_sig),
        "n_strong"        => json_number(n_strong),
        "generated_at"    => json_string(Dates.format(now(), "yyyy-mm-dd HH:MM")),
        "package_version" => json_string(_report_pkg_version()),
        "julia_version"   => json_string(string(VERSION)),
        "n_controls"      => json_number(config.n_controls),
        "n_samples"       => json_number(config.n_samples),
        "combination_method" => json_string(string(config.combination_method)),
    )
end

"""Build results array JSON from the DataFrame."""
function _build_results_json(results::DataFrame)::String
    rows = String[]
    for row in eachrow(results)
        push!(rows, _build_protein_json(row))
    end
    return json_array(rows)
end

"""Serialize a single protein result row to a JSON object."""
function _build_protein_json(row)::String
    protein   = string(row.Protein)
    bf        = _safe_float(row.BF)
    pp        = _safe_float(row.posterior_prob)
    q         = _safe_float(row.q)
    lfc       = _safe_float(row.mean_log2FC)
    bfe       = _safe_float(row.bf_enrichment)
    bfc       = _safe_float(row.bf_correlation)
    bfd       = _safe_float(row.bf_detected)

    # Fallback: if metalearner posterior is missing, approximate from BF using flat prior.
    # This can happen when protein names after curation don't match the metalearner output.
    if pp === nothing && bf !== nothing && bf ≥ 0.0
        pp = bf / (1.0 + bf)
    end

    # Derived values
    fc        = lfc === nothing ? nothing : 2.0^lfc
    sd_lfc    = hasproperty(row, :sd_log2FC) ? _safe_float(row.sd_log2FC) : nothing
    fc_lo     = (sd_lfc !== nothing && lfc !== nothing) ? 2.0^(lfc - 1.96*sd_lfc) : nothing
    fc_hi     = (sd_lfc !== nothing && lfc !== nothing) ? 2.0^(lfc + 1.96*sd_lfc) : nothing
    ev_label  = (q !== nothing && pp !== nothing) ? _evidence_label(pp, q) : ""

    # Optional columns: present only when diagnostics / sensitivity analysis ran
    diag_flag  = hasproperty(row, :diagnostic_flag) ?
                     (ismissing(row.diagnostic_flag) ? "" : string(row.diagnostic_flag)) : ""
    sens_range = hasproperty(row, :sensitivity_range) ? _safe_float(row.sensitivity_range) : nothing

    return json_object(
        "protein"           => json_string(protein),
        "bf"                => json_number(bf === nothing ? NaN : bf),
        "posterior_prob"    => json_number(pp === nothing ? NaN : pp),
        "q"                 => json_number(q  === nothing ? NaN : q),
        "mean_log2fc"       => json_number(lfc === nothing ? NaN : lfc),
        "sd_log2fc"         => json_number(sd_lfc === nothing ? NaN : sd_lfc),
        "fold_change"       => json_number(fc  === nothing ? NaN : fc),
        "fold_change_lo"    => json_number(fc_lo === nothing ? NaN : fc_lo),
        "fold_change_hi"    => json_number(fc_hi === nothing ? NaN : fc_hi),
        "bf_enrichment"     => json_number(bfe === nothing ? NaN : bfe),
        "bf_correlation"    => json_number(bfc === nothing ? NaN : bfc),
        "bf_detected"       => json_number(bfd === nothing ? NaN : bfd),
        "evidence_label"    => json_string(ev_label),
        "diagnostic_flag"   => json_string(diag_flag),
        "sensitivity_range" => json_number(sens_range === nothing ? NaN : sens_range),
    )
end

"""Base64-encode all existing plot files into a JSON object."""
function _build_plots_json(config::CONFIG)::String
    out = config.output
    plot_map = [
        # Evidence tab
        "evidence"                 => out.evidence_file,
        "convergence"              => out.convergence_file,
        "em_diagnostics"           => out.em_diagnostics_file,
        "lc_convergence"           => out.lc_convergence_file,
        # Diagnostics tab
        "qq_plot"                  => out.qq_plot_file,
        "regression_qq_plot"       => out.regression_qq_plot_file,
        "scale_location_hbm"       => out.scale_location_hbm_file,
        "scale_location_regression"=> out.scale_location_regression_file,
        "calibration"              => out.calibration_plot_file,
        "calibration_comparison"   => out.calibration_comparison_file,
        "ppc_histogram"            => out.ppc_histogram_file,
        "pit_histogram"            => out.pit_histogram_file,
        "nu_optimization"          => out.nu_optimization_file,
        # Sensitivity tab
        "sensitivity_tornado"      => out.sensitivity_tornado_file,
        "sensitivity_heatmap"      => out.sensitivity_heatmap_file,
        "sensitivity_rankcorr"     => out.sensitivity_rankcorr_file,
    ]

    pairs = Pair{String,String}[]
    for (key, filepath) in plot_map
        uri = encode_png_file(filepath)
        isempty(uri) && continue
        push!(pairs, key => json_string(uri))
    end

    isempty(pairs) && return "{}"
    parts = [json_string(k) * ":" * v for (k, v) in pairs]
    return "{" * join(parts, ",") * "}"
end

"""Build methods section JSON."""
function _build_methods_json(config::CONFIG, results::DataFrame)::String
    methods_text  = try generate_methods_text(config, results)  catch; "" end
    repro_block   = try generate_reproducibility_block(config)  catch; "" end
    params        = try generate_methods_parameters(config)     catch; Pair{String,String}[] end

    param_pairs = [json_string(k) * ":" * json_string(v) for (k, v) in params]
    params_json = "{" * join(param_pairs, ",") * "}"

    return json_object(
        "text"           => json_string(methods_text),
        "reproducibility"=> json_string(repro_block),
        "parameters"     => params_json,
    )
end

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

"""Return the Float64 value of x, or nothing if missing/NaN/Inf."""
function _safe_float(x)
    ismissing(x) && return nothing
    v = Float64(x)
    isfinite(v) ? v : nothing
end

# ---------------------------------------------------------------------------
# Differential analysis report
# ---------------------------------------------------------------------------

"""
    generate_differential_report(diff::DifferentialResult; output)

Generate a self-contained interactive HTML report for a differential interaction analysis.

The report includes:
- Dashboard summary (N gained / lost / unchanged / condition-specific)
- Interactive Plotly differential volcano plot (Δlog₂FC vs log₁₀(dBF), coloured by classification)
- Searchable/filterable DataTables results table
- Static PNG / SVG embeds for all existing differential plots
- Analysis metadata

# Arguments
- `diff::DifferentialResult`: Output from `differential_analysis()`.

# Keywords
- `output::String`: Path for the HTML file. Defaults to replacing the results file
  extension with `_report.html`.
"""
function generate_differential_report(
        diff::DifferentialResult;
        output::String = _diff_report_default_path(diff))::Nothing
    @info "Generating differential interaction report..."

    json_blob = _build_diff_json(diff)

    template_path = joinpath(@__DIR__, "templates", "differential_report.html")
    if !isfile(template_path)
        @warn "Differential report template not found at $template_path; skipping."
        return nothing
    end
    template = Base.read(template_path, String)

    html = replace(template, "{{DIFF_DATA_JSON}}" => json_blob)

    mkpath(dirname(abspath(output)))
    Base.write(output, html)
    @info "Differential report saved to: $output"
    return nothing
end

"""Default output path: replace results file extension with _report.html."""
function _diff_report_default_path(diff::DifferentialResult)::String
    base = diff.config.results_file
    stem = replace(base, r"\.(xlsx|csv|tsv)$"i => "")
    return stem * "_report.html"
end

"""Build the full JSON blob for the differential report template."""
function _build_diff_json(diff::DifferentialResult)::String
    return json_object(
        "meta"    => _build_diff_meta_json(diff),
        "results" => _build_diff_results_json(diff),
        "plots"   => _build_diff_plots_json(diff),
    )
end

"""Build dashboard / metadata JSON for the differential report."""
function _build_diff_meta_json(diff::DifferentialResult)::String
    n_total = nrow(diff.results)
    return json_object(
        "condition_A"       => json_string(diff.condition_A),
        "condition_B"       => json_string(diff.condition_B),
        "n_total"           => json_number(n_total),
        "n_gained"          => json_number(diff.n_gained),
        "n_lost"            => json_number(diff.n_reduced),
        "n_unchanged"       => json_number(diff.n_unchanged),
        "n_both_negative"   => json_number(diff.n_both_negative),
        "n_A_specific"      => json_number(diff.n_condition_A_specific),
        "n_B_specific"      => json_number(diff.n_condition_B_specific),
        "posterior_threshold" => json_number(diff.config.posterior_threshold),
        "q_threshold"       => json_number(diff.config.q_threshold),
        "generated_at"      => json_string(Dates.format(now(), "yyyy-mm-dd HH:MM")),
        "package_version"   => json_string(_report_pkg_version()),
    )
end

"""Serialize each row of diff.results to JSON."""
function _build_diff_results_json(diff::DifferentialResult)::String
    rows = String[]
    for row in eachrow(diff.results)
        push!(rows, _build_diff_protein_json(row))
    end
    return json_array(rows)
end

"""Serialize a single differential result row."""
function _build_diff_protein_json(row)::String
    _sf = _safe_float
    cls = hasproperty(row, :classification) ?
              (ismissing(row.classification) ? "" : string(row.classification)) : ""

    # Optional diagnostic / sensitivity columns
    diag_A = hasproperty(row, :diagnostic_flag_A) ?
                 coalesce(string(row.diagnostic_flag_A), "") : ""
    diag_B = hasproperty(row, :diagnostic_flag_B) ?
                 coalesce(string(row.diagnostic_flag_B), "") : ""
    sens_A = hasproperty(row, :sensitivity_range_A) ? (_sf(row.sensitivity_range_A) === nothing ? NaN : _sf(row.sensitivity_range_A)) : NaN
    sens_B = hasproperty(row, :sensitivity_range_B) ? (_sf(row.sensitivity_range_B) === nothing ? NaN : _sf(row.sensitivity_range_B)) : NaN

    return json_object(
        "protein"               => json_string(string(row.Protein)),
        "dbf"                   => json_number(_sf(row.dbf) === nothing ? NaN : _sf(row.dbf)),
        "log10_dbf"             => json_number(_sf(row.log10_dbf) === nothing ? NaN : _sf(row.log10_dbf)),
        "delta_log2fc"          => json_number(_sf(row.delta_log2fc) === nothing ? NaN : _sf(row.delta_log2fc)),
        "posterior_A"           => json_number(_sf(row.posterior_A) === nothing ? NaN : _sf(row.posterior_A)),
        "posterior_B"           => json_number(_sf(row.posterior_B) === nothing ? NaN : _sf(row.posterior_B)),
        "delta_posterior"       => json_number(_sf(row.delta_posterior) === nothing ? NaN : _sf(row.delta_posterior)),
        "q_A"                   => json_number(_sf(row.q_A) === nothing ? NaN : _sf(row.q_A)),
        "q_B"                   => json_number(_sf(row.q_B) === nothing ? NaN : _sf(row.q_B)),
        "log2fc_A"              => json_number(_sf(row.log2fc_A) === nothing ? NaN : _sf(row.log2fc_A)),
        "log2fc_B"              => json_number(_sf(row.log2fc_B) === nothing ? NaN : _sf(row.log2fc_B)),
        "differential_posterior"=> json_number(_sf(row.differential_posterior) === nothing ? NaN : _sf(row.differential_posterior)),
        "differential_q"        => json_number(_sf(row.differential_q) === nothing ? NaN : _sf(row.differential_q)),
        "classification"        => json_string(cls),
        "diagnostic_flag_A"     => json_string(diag_A),
        "diagnostic_flag_B"     => json_string(diag_B),
        "sensitivity_range_A"   => json_number(sens_A),
        "sensitivity_range_B"   => json_number(sens_B),
    )
end

"""Base64-encode all existing differential plot files into a JSON object."""
function _build_diff_plots_json(diff::DifferentialResult)::String
    cfg = diff.config
    plot_map = [
        "volcano"        => cfg.volcano_file,
        "evidence"       => cfg.evidence_file,
        "scatter"        => cfg.scatter_file,
        "classification" => cfg.classification_file,
        "ma"             => cfg.ma_file,
    ]
    pairs = Pair{String,String}[]
    for (key, filepath) in plot_map
        uri = encode_png_file(filepath)
        isempty(uri) && continue
        push!(pairs, key => json_string(uri))
    end
    isempty(pairs) && return "{}"
    parts = [json_string(k) * ":" * v for (k, v) in pairs]
    return "{" * join(parts, ",") * "}"
end

# ---------------------------------------------------------------------------
# Evidence labels and other helpers
# ---------------------------------------------------------------------------

"""
Plain-language evidence badge based on posterior probability and q-value.
"""
function _evidence_label(pp::Float64, q::Float64)::String
    pp ≥ 0.95 && q ≤ 0.01 && return "Strong \u2605\u2605\u2605"
    pp ≥ 0.80 && q ≤ 0.05 && return "Moderate \u2605\u2605"
    pp ≥ 0.50             && return "Weak \u2605"
    return ""
end
