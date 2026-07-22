# src/reports/report_generator.jl
# Main interactive HTML report generator.

import JSON3
import Downloads
using CodecZlib: GzipCompressor, transcode
using Base64

# ---------------------------------------------------------------------------
# Mol* vendoring (lazy, on-demand)
# ---------------------------------------------------------------------------

const MOLSTAR_VERSION = "5.8.0"
const MOLSTAR_JS_URL  = "https://cdn.jsdelivr.net/npm/molstar@$(MOLSTAR_VERSION)/build/viewer/molstar.js"
const MOLSTAR_CSS_URL = "https://cdn.jsdelivr.net/npm/molstar@$(MOLSTAR_VERSION)/build/viewer/molstar.css"

"""
    _molstar_vendor_dir(cache_base="") -> String

Resolve the on-disk vendor directory for Mol* assets. Defaults to
`.bayesinteractomics_cache/vendor/molstar/`. Created on demand.
"""
function _molstar_vendor_dir(cache_base::String="")::String
    root = isempty(cache_base) ? ".bayesinteractomics_cache" : cache_base
    dir  = joinpath(root, "vendor", "molstar")
    mkpath(dir)
    return dir
end

"""
    _ensure_molstar_vendored(cache_base="") -> NamedTuple{(:js, :css), Tuple{String, String}} | Nothing

Download Mol* JS+CSS into the vendor directory if missing. Returns paths to the
local files, or `nothing` if the download failed (caller should fall back to the
CDN-loading code path baked into the report template).
"""
function _ensure_molstar_vendored(cache_base::String="")::Union{NamedTuple, Nothing}
    dir = _molstar_vendor_dir(cache_base)
    js_path  = joinpath(dir, "molstar-$(MOLSTAR_VERSION).js")
    css_path = joinpath(dir, "molstar-$(MOLSTAR_VERSION).css")
    if isfile(js_path) && isfile(css_path)
        return (js=js_path, css=css_path)
    end
    try
        @info "Vendoring Mol* $(MOLSTAR_VERSION) for self-contained reports (one-time download, ~3 MB)" js=js_path
        Downloads.download(MOLSTAR_JS_URL, js_path)
        Downloads.download(MOLSTAR_CSS_URL, css_path)
        return (js=js_path, css=css_path)
    catch e
        @warn "Failed to vendor Mol*; report will fall back to CDN load at view time" exception=e
        # Clean up partial files
        isfile(js_path)  && rm(js_path; force=true)
        isfile(css_path) && rm(css_path; force=true)
        return nothing
    end
end

"""
    _inline_molstar_bundle(html, cache_base="") -> String

Replace the `<!-- {{MOLSTAR_INLINE_BUNDLE}} -->` placeholder in `html` with
inline `<style>` and `<script>` blocks containing the vendored Mol* assets.
On any failure (download error, missing files), the placeholder is replaced
with a comment and the report falls back to the CDN loader baked into the
template's `loadMolstar()` JS.
"""
function _inline_molstar_bundle(html::String, cache_base::String="")::String
    placeholder = "<!-- {{MOLSTAR_INLINE_BUNDLE}} -->"
    !occursin(placeholder, html) && return html

    paths = _ensure_molstar_vendored(cache_base)
    if paths === nothing
        return replace(html, placeholder =>
            "<!-- Mol* vendoring unavailable; runtime will load from CDN -->")
    end

    css = read(paths.css, String)
    js  = read(paths.js,  String)
    inlined = string(
        "<style>\n", css, "\n</style>\n",
        "<script>\n", js,  "\nwindow.__MOLSTAR_INLINED__ = true;\n</script>")
    return replace(html, placeholder => inlined)
end

# ---------------------------------------------------------------------------
# Sidecar helpers
# ---------------------------------------------------------------------------

"""Return the sidecar JSON file path derived from a report HTML path."""
function _sidecar_path(report_path::String)::String
    dir = dirname(report_path)
    base = first(splitext(basename(report_path)))
    return joinpath(dir, base * "_data.json")
end

"""Write the complete JSON blob to a sidecar file alongside the HTML report."""
function _write_sidecar(json_blob::String, report_path::String)::Nothing
    sidecar = _sidecar_path(report_path)
    mkpath(dirname(sidecar))
    Base.write(sidecar, json_blob)
    @info "Report sidecar written to: $sidecar"
    return nothing
end

"""
    _merge_sidecar(existing_sidecar_path::String, new_sections::Dict{String,String})::String

Read an existing sidecar JSON, merge new sections into it (replacing keys that exist
in `new_sections`), and return the merged JSON string.

Uses a full-Dict approach: parse existing JSON into a Dict via JSON3, replace keys
with pre-serialized new sections wrapped in `JSON3.read`, then serialize the whole
dict back with JSON3.write. This avoids mixing hand-rolled json_string() with
JSON3.write() which can produce malformed JSON (RESEARCH.md Pitfall 1).
"""
function _merge_sidecar(existing_sidecar_path::String, new_sections::Dict{String,String})::String
    existing_str = Base.read(existing_sidecar_path, String)
    # Parse into a mutable Dict
    existing = Dict{String,Any}(String(k) => v for (k, v) in pairs(JSON3.read(existing_str)))

    # Replace keys present in new_sections (values are pre-serialized JSON strings,
    # so parse them back into objects before re-serializing the whole dict)
    for (k, v) in new_sections
        existing[k] = JSON3.read(v)
    end

    return JSON3.write(existing)
end

# ---------------------------------------------------------------------------

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
                         output::String = config.output.report_file,
                         analysis_result = nothing,
                         docking_result = nothing,
                         validation_result = nothing,
                         sensitivity_result = nothing,
                         simulation_result = nothing,
                         sidecar_path::String = "",
                         structures_dir::String = "",
                         pct_imputed_cells::Union{Nothing, Real} = nothing)::Nothing
    @info "Generating interactive HTML report..."

    local json_blob::String

    if !isempty(sidecar_path) && isfile(sidecar_path)
        # Merge path: read existing sidecar, replace only new sections
        @info "Merging with existing sidecar: $sidecar_path"
        new_sections = Dict{String,String}()
        # Only rebuild sections where we have new data
        if docking_result !== nothing
            new_sections["docking"] = _build_docking_json(results, docking_result;
                                                          structures_dir=structures_dir)
        end
        # Update meta timestamp
        new_sections["meta"] = _build_meta_json(results, config)
        # Refresh results (may have docking columns now)
        new_sections["results"] = _build_results_json(results)
        new_sections["non_detected"] = _build_non_detected_json(results)

        json_blob = _merge_sidecar(sidecar_path, new_sections)
    else
        # Fresh build path (original behavior)
        json_blob = _build_report_json(results, config; analysis_result=analysis_result,
                                       docking_result=docking_result,
                                       validation_result=validation_result,
                                       sensitivity_result=sensitivity_result,
                                       simulation_result=simulation_result,
                                       structures_dir=structures_dir,
                                       pct_imputed_cells=pct_imputed_cells)
    end

    # Load HTML template
    template_path = joinpath(@__DIR__, "templates", "report.html")
    if !isfile(template_path)
        @warn "Report template not found at $template_path; skipping report generation."
        return nothing
    end
    template = Base.read(template_path, String)

    # Inject data
    html = replace(template, "{{REPORT_DATA_JSON}}" => json_blob)

    # Inline Mol* assets into the report when docking content is present, so the
    # 3D structure viewer works offline / behind firewalls. Falls back to the
    # CDN loader baked into loadMolstar() if vendoring fails.
    if docking_result !== nothing
        cache_base = ""
        if config.docking_config !== nothing && !isempty(config.docking_config.cache_dir)
            cache_base = config.docking_config.cache_dir
        end
        html = _inline_molstar_bundle(html, cache_base)
    end

    # Write output
    mkpath(dirname(output))
    Base.write(output, html)
    @info "Interactive report saved to: $output"

    # Always write sidecar
    _write_sidecar(json_blob, output)

    # Also write standalone methods text
    methods_path = config.output.report_methods_file
    try
        methods_text = generate_methods_text(config, results)
        mkpath(dirname(methods_path))
        Base.write(methods_path, methods_text)
        @info "Methods text saved to: $methods_path"
    catch e
        @warn "Failed to write methods file" exception=(e, catch_backtrace())
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
Filter DataFrame to detected proteins only.

Backward-compatible: if `is_detected` column is absent (pre-Phase-16 results),
returns the full DataFrame unchanged (all proteins treated as detected).
"""
function _filter_detected(results::DataFrame)::DataFrame
    if !hasproperty(results, :is_detected)
        return results  # backward compat: no is_detected column means all detected
    end
    return filter(r -> coalesce(r.is_detected, false), results)
end

"""
    _report_float(v, default::Float64 = 0.0) -> Float64

Safely convert a possibly-missing value to Float64 for report generation.
Returns `default` when `v` is `missing`.
"""
_report_float(v, default::Float64 = 0.0)::Float64 = ismissing(v) ? default : Float64(v)

# ---------------------------------------------------------------------------
# Winsorization, KDE density, MC combined density utilities
# ---------------------------------------------------------------------------

"""
    _winsorize_quantile(x; lo_q=0.005, hi_q=0.995) -> Vector{Float64}

Clamp finite values in `x` to the [`lo_q`, `hi_q`] quantile range.
Non-finite values (NaN, Inf, -Inf) are left unchanged.
"""
function _winsorize_quantile(x::AbstractVector{<:Real}; lo_q::Float64=0.005, hi_q::Float64=0.995)::Vector{Float64}
    isempty(x) && return Float64[]
    finite_x = filter(isfinite, x)
    isempty(finite_x) && return float.(x)
    lo_val = quantile(finite_x, lo_q)
    hi_val = quantile(finite_x, hi_q)
    return [isfinite(v) ? clamp(Float64(v), lo_val, hi_val) : Float64(v) for v in x]
end

"""
    _kde_density(samples, x_grid; bandwidth=0.0) -> Vector{Float64}

Gaussian kernel density estimate evaluated on `x_grid`.
Uses Silverman's rule-of-thumb bandwidth when `bandwidth <= 0`.
"""
function _kde_density(samples::Vector{Float64}, x_grid::AbstractVector{Float64}; bandwidth::Float64=0.0)::Vector{Float64}
    n = length(samples)
    n == 0 && return zeros(length(x_grid))
    s = std(samples)
    s = max(s, 1e-10)
    h = bandwidth > 0.0 ? bandwidth : 1.06 * s * n^(-0.2)
    inv_norm = 1.0 / (n * h * sqrt(2 * pi))
    density = zeros(length(x_grid))
    cutoff = 4.0 * h
    for (j, x) in enumerate(x_grid)
        acc = 0.0
        for s_val in samples
            d = abs(x - s_val)
            d > cutoff && continue
            z = (x - s_val) / h
            acc += exp(-0.5 * z * z)
        end
        density[j] = acc * inv_norm
    end
    return density
end

"""
    _mc_combined_density(lc::LatentClassResult, component; n_mc=100_000, n_grid=200)

Monte Carlo convolution: sample enrichment + correlation + detection from per-dimension
marginals, sum them, then KDE the result to produce the combined log-BF density.

Returns `(x=Vector, y=Vector)` or `nothing` if per_dimension_params is missing.
"""
function _mc_combined_density(lc::LatentClassResult, component::String; n_mc::Int=100_000, n_grid::Int=200)
    lc.per_dimension_params === nothing && return nothing
    pdp = lc.per_dimension_params
    !haskey(pdp, component) && return nothing
    params = pdp[component]

    # Sample enrichment
    if component == "interaction"
        # Use the BIC-selected family + JEFFREYS_SHIFT
        if lc.h1_enrichment_family == :lognormal
            e_samples = rand(LogNormal(lc.alpha_enrichment_h1, lc.theta_enrichment_h1), n_mc) .+ JEFFREYS_SHIFT
        elseif lc.h1_enrichment_family == :weibull
            e_samples = rand(Weibull(lc.alpha_enrichment_h1, lc.theta_enrichment_h1), n_mc) .+ JEFFREYS_SHIFT
        else  # :gamma
            e_samples = rand(Gamma(lc.alpha_enrichment_h1, lc.theta_enrichment_h1), n_mc) .+ JEFFREYS_SHIFT
        end
    else
        e_samples = rand(Normal(params.mu_e, max(params.sigma_e, 0.01)), n_mc)
    end

    # Sample correlation
    c_samples = rand(Normal(params.mu_c, max(params.sigma_c, 0.01)), n_mc)

    # Sample detection
    disc = if component == "background"
        lc.disc_detection_H0
    elseif component == "agnostic"
        lc.disc_detection_ag
    else
        lc.disc_detection_H1
    end

    if disc !== nothing && !isempty(disc.values)
        p_samples = Float64[rand(disc) for _ in 1:n_mc]
    else
        p_samples = rand(Normal(params.mu_p, max(params.sigma_p, 0.01)), n_mc)
    end

    # Combine via summation (log-BF scale)
    combined = e_samples .+ c_samples .+ p_samples

    # KDE on the combined distribution
    h = 1.06 * std(combined) * n_mc^(-0.2)
    lo = minimum(combined) - 3 * h
    hi = maximum(combined) + 3 * h
    x_grid = collect(range(lo, hi, length=n_grid))
    density = _kde_density(combined, x_grid; bandwidth=h)
    return (x=x_grid, y=density)
end

"""
Build JSON array of non-detected proteins for the collapsed section.

Returns `"[]"` when no `is_detected` column is present or when all proteins
are detected.
"""
function _build_non_detected_json(results::DataFrame)::String
    if !hasproperty(results, :is_detected)
        return "[]"
    end
    non_det_df = filter(r -> !coalesce(r.is_detected, true), results)
    rows = String[]
    for r in eachrow(non_det_df)
        push!(rows, json_object(
            "protein" => json_string(string(r.Protein)),
            "name"    => json_string(hasproperty(r, :protein_name) ?
                             string(coalesce(r.protein_name, "")) : "")
        ))
    end
    return json_array(rows)
end

"""Build the complete JSON string to be embedded in the HTML template.

accepts an optional `pct_imputed_cells` kwarg
threaded from the pipeline driver. When non-nothing it drives the
Data-Quality chip HTML; when nothing the chip is suppressed (genuinely
unknown imputation status — distinct from the `0.00%` explicit-zero path).
The `AnalysisResult` struct is NOT modified; `pct_imputed_cells`
is computed locally inside the pipeline driver from the `is_imputed` mask
that `prepare_regression_data` exposes.
"""
function _build_report_json(results::DataFrame, config::CONFIG;
                            analysis_result = nothing,
                            docking_result = nothing,
                            validation_result = nothing,
                            sensitivity_result = nothing,
                            simulation_result = nothing,
                            structures_dir::String = "",
                            pct_imputed_cells::Union{Nothing, Real} = nothing)::String
    # Resolve metalearner_status with backward compatibility.
    # The `AnalysisResult.metalearner_status` field was added at CACHE_VERSION 21.
    # Cached AnalysisResults that predate it (or `analysis_result=nothing` callers)
    # default to `:loaded` so the HTML banner stays hidden — preserving the earlier
    # visual behaviour. The JS consumer uses `D.metalearner_status ?? 'loaded'`
    # as a second-layer fallback for old sidecar JSONs that lack the key.
    metalearner_status_sym = if analysis_result !== nothing &&
                                hasproperty(analysis_result, :metalearner_status)
        getfield(analysis_result, :metalearner_status)
    else
        :loaded
    end
    metalearner_status_json = json_string(string(metalearner_status_sym))

    meta_json    = _build_meta_json(results, config)
    results_json = _build_results_json(results)
    non_detected_json = _build_non_detected_json(results)
    # DNN Prior + MC-Dropout JSON sidecar
    dnn_prior_json = _build_dnn_prior_json(results, config)
    plots_json   = _build_plots_json(config)
    methods_json = _build_methods_json(config, results, metalearner_status_sym)
    bma_json     = _build_bma_summary_json(analysis_result, results)
    docking_json = _build_docking_json(results, docking_result; structures_dir=structures_dir)
    mixture_json = _build_mixture_model_json(analysis_result, results)
    validation_json = _build_validation_json(validation_result)
    sensitivity_json = _build_sensitivity_json(sensitivity_result)
    diagnostics_data_json = _build_diagnostics_data_json(analysis_result, results)
    evidence_data_json = _build_evidence_data_json(analysis_result, results)
    simulation_json = _build_simulation_json(simulation_result)
    input_qc_json = _build_qc_json(analysis_result)

    # has_calibration and calibration_warning flags
    has_cal = !isnothing(simulation_result) &&
              hasproperty(simulation_result, :calibration_model) &&
              !isnothing(simulation_result.calibration_model)
    has_calibration_json = has_cal ? "true" : "false"
    calibration_warning_json = isnothing(simulation_result) ? "true" : "false"

    # diagnostic thresholds for JS popover (single source of truth)
    diag_thresholds_json = json_object(
        "low_data_cutoff"         => json_number(4),
        "residual_outlier_cutoff" => json_number(2),
    )

    # Variance Recovery report payload.
    # `block_html` is the rendered HTML fragment (empty when mode == :off, so the
    # Methods-tab card stays hidden). The other keys carry structured data for
    # forward-compat JS rendering or downstream tooling. Seed contract:
    # seeds = [base_seed * 1_000_003 + i for i ∈ 1..m] (matches the production
    # _generate_multi_impute_data formula in src/analysis/pipeline.jl).
    variance_recovery_json = json_object(
        "mode"               => json_string(string(config.mnar_variance_recovery)),
        "m"                  => config.mnar_variance_recovery === :multi_impute ?
                                  json_number(config.mnar_m) : "null",
        "seeds"              => config.mnar_variance_recovery === :multi_impute ?
                                  json_array([json_number(Int(config.mnar_base_seed) * 1_000_003 + i)
                                              for i in 1:config.mnar_m]) : "null",
        "inflation_max"      => config.mnar_variance_recovery === :inflation ?
                                  json_number(config.mnar_inflation_max) : "null",
        "inflation_override" => (config.mnar_variance_recovery === :inflation &&
                                 config.mnar_inflation_factor !== nothing) ?
                                  json_number(config.mnar_inflation_factor) : "null",
        "block_html"         => json_string(_methods_variance_recovery_block(config)),
    )

    # Mask-aware regression v2b payload.
    # `block_html` is the rendered Methods-tab <h4> subsection (empty when
    # config.mask_aware_regression == false → card stays hidden).
    # `chip_html` is the Data-Quality Bootstrap chip rendered from the
    # locally-computed pct_imputed_cells scalar (kwarg). When the kwarg is
    # nothing the chip is suppressed (mask genuinely unknown — distinct from
    # the explicit 0.00% chip). The AnalysisResult struct is NOT modified.
    # the Normalisation Methods-tab subsection renders
    # UNCONDITIONALLY (every analysis applies a — possibly identity —
    # normalisation). It is injected into the SAME Methods-tab card as the
    # mask-aware block (template-free wiring): the combined block_html carries
    # the always-present normalisation <h4> first, followed by the (possibly
    # empty) mask-aware v2b <h4>. The card therefore always renders.
    mask_aware_block = _methods_normalisation_block(config) *
                       _methods_mask_aware_regression_block(config)
    mask_aware_json = json_object(
        "enabled"    => config.mask_aware_regression ? "true" : "false",
        "block_html" => json_string(mask_aware_block),
        "chip_html"  => json_string(_mask_aware_chip_html(pct_imputed_cells)),
        "pct_imputed_cells" => pct_imputed_cells === nothing ?
                               "null" : json_number(Float64(pct_imputed_cells)),
    )

    return json_object(
        "meta"                => meta_json,
        "results"             => results_json,
        "non_detected"        => non_detected_json,
        # DNN Prior + MC-Dropout uncertainty sidecar
        "dnn_prior"           => dnn_prior_json,
        "plots"               => plots_json,
        "methods"             => methods_json,
        "bma"                 => bma_json,
        "docking"             => docking_json,
        "mixture_model"       => mixture_json,
        "validation"          => validation_json,
        "sensitivity"         => sensitivity_json,
        "diagnostics_data"    => diagnostics_data_json,
        "evidence_data"       => evidence_data_json,
        "simulation"          => simulation_json,
        "has_calibration"     => has_calibration_json,
        "calibration_warning" => calibration_warning_json,
        "diagnostic_thresholds" => diag_thresholds_json,
        "input_qc"              => input_qc_json,
        # metalearner_status surfaces Variante B fallback
        # state to the report banner + Methods tab + posterior_prob tooltip.
        "metalearner_status"    => metalearner_status_json,
        # variance_recovery surfaces the MNAR variance-recovery mode + parameters
        # + rendered HTML block to the Methods-tab card; empty block_html for :off.
        "variance_recovery"     => variance_recovery_json,
        # mask-aware regression v2b — surfaces
        # the Methods H4 subsection (block_html) + Data-Quality chip (chip_html)
        # to the templates. Empty strings → cards/chips stay hidden.
        "mask_aware"            => mask_aware_json,
        # sample-level + protein-level embeddings JSON.
        # Both keys gate on `ar.embeddings === nothing` (return "null") so the
        # consumer JS uniformly checks `if (!data) ...` to short-circuit.
        "embeddings_sample"     => _build_sample_embedding_json(analysis_result),
        "embeddings_protein"    => _build_protein_embedding_json(analysis_result),
    )
end

"""Build meta/dashboard JSON object."""
function _build_meta_json(results::DataFrame, config::CONFIG)::String
    detected_results = _filter_detected(results)
    n_proteins   = nrow(detected_results)
    n_excluded   = nrow(results) - n_proteins
    n_sig        = sum(ismissing.(detected_results.posterior_prob) .== false .&& detected_results.posterior_prob .> 0.95)
    n_strong     = sum(ismissing.(detected_results.posterior_prob) .== false .&& detected_results.posterior_prob .> 0.99)

    return json_object(
        "bait"            => json_string(config.poi),
        "n_proteins"      => json_number(n_proteins),
        "n_excluded"      => json_number(n_excluded),
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

"""Build results array JSON from the DataFrame (detected proteins only)."""
function _build_results_json(results::DataFrame)::String
    detected_results = _filter_detected(results)
    rows = String[]
    for row in eachrow(detected_results)
        push!(rows, _build_protein_json(row))
    end
    return json_array(rows)
end

"""Serialize a single protein result row to a JSON object."""
function _build_protein_json(row)::String
    protein   = string(row.Protein)
    bf        = _safe_float(row.BF)
    pp        = _safe_float(row.posterior_prob)
    bfdr_val  = _safe_float(row.BFDR)
    pep_val   = _safe_float(row.PEP)
    lfc       = _safe_float(row.mean_log2FC)
    bfe       = _safe_float(row.bf_enrichment)
    bfc       = _safe_float(row.bf_correlation)
    bfd       = _safe_float(row.bf_detected)

    # Fallback: if metalearner posterior is missing, approximate from BF using flat prior.
    # This can happen when protein names after curation don't match the metalearner output.
    if pp === nothing && bf !== nothing && bf ≥ 0.0
        pp = bf / (1.0 + bf)
    end

    # use calibrated posterior when available
    pp_cal = hasproperty(row, :posterior_calibrated) ?
             _safe_float(row.posterior_calibrated) : nothing
    pp_for_plot = pp_cal !== nothing ? pp_cal : pp

    # Derived values
    fc        = lfc === nothing ? nothing : 2.0^lfc
    sd_lfc    = hasproperty(row, :sd_log2FC) ? _safe_float(row.sd_log2FC) : nothing
    fc_lo     = (sd_lfc !== nothing && lfc !== nothing) ? 2.0^(lfc - 1.96*sd_lfc) : nothing
    fc_hi     = (sd_lfc !== nothing && lfc !== nothing) ? 2.0^(lfc + 1.96*sd_lfc) : nothing
    ev_label  = (bfdr_val !== nothing && pp_for_plot !== nothing) ? _evidence_label(pp_for_plot, bfdr_val) : ""

    # Optional columns: present only when diagnostics / sensitivity analysis ran
    diag_flag  = hasproperty(row, :diagnostic_flag) ?
                     (ismissing(row.diagnostic_flag) ? "" : string(row.diagnostic_flag)) : ""
    sens_range = hasproperty(row, :sensitivity_range) ? _safe_float(row.sensitivity_range) : nothing
    fdr_cal    = hasproperty(row, :fdr_calibrated) ? _safe_float(row.fdr_calibrated) : nothing

    # diagnostic detail fields for popover
    n_obs      = hasproperty(row, :n_observations) ?
                     (ismissing(row.n_observations) ? nothing : row.n_observations) : nothing
    mean_res   = hasproperty(row, :mean_residual) ?
                     (ismissing(row.mean_residual) ? nothing : Float64(row.mean_residual)) : nothing
    max_abs    = hasproperty(row, :max_abs_residual) ?
                     (ismissing(row.max_abs_residual) ? nothing : Float64(row.max_abs_residual)) : nothing
    is_low     = hasproperty(row, :is_low_data) ?
                     (ismissing(row.is_low_data) ? nothing : Bool(row.is_low_data)) : nothing
    is_outlier = hasproperty(row, :is_residual_outlier) ?
                     (ismissing(row.is_residual_outlier) ? nothing : Bool(row.is_residual_outlier)) : nothing

    # Per-protein Pareto k-hat (PSIS-LOO diagnostic)
    pk_val     = hasproperty(row, :pareto_k) ? _safe_float(row.pareto_k) : nothing

    # Sub-model BF columns (TRANS-02, BMA-only)
    bf_em_val  = hasproperty(row, :bf_em) ? _safe_float(row.bf_em) : nothing
    bf_cop_val = hasproperty(row, :bf_copula) ? _safe_float(row.bf_copula) : nothing

    # bb_mnar_codriven flag + per-row evidence (warning tooltip)
    bb_codriven   = hasproperty(row, :bb_mnar_codriven) ?
                        (ismissing(row.bb_mnar_codriven) ? false : Bool(row.bb_mnar_codriven)) : false
    missing_frac  = hasproperty(row, :missing_fraction) ?
                        _safe_float(row.missing_fraction) : nothing

    # DNN Prior + MC-Dropout uncertainty columns
    prior_mc_mean      = hasproperty(row, :prior_mc_mean)      ? _safe_float(row.prior_mc_mean)      : nothing
    prior_mc_std       = hasproperty(row, :prior_mc_std)       ? _safe_float(row.prior_mc_std)       : nothing
    prior_mc_ci_low    = hasproperty(row, :prior_mc_ci_low)    ? _safe_float(row.prior_mc_ci_low)    : nothing
    prior_mc_ci_high   = hasproperty(row, :prior_mc_ci_high)   ? _safe_float(row.prior_mc_ci_high)   : nothing
    prior_contribution = hasproperty(row, :prior_contribution) ? _safe_float(row.prior_contribution) : nothing

    return json_object(
        "protein"               => json_string(protein),
        "bf"                    => json_number(bf === nothing ? NaN : bf),
        "posterior_prob"        => json_number(pp_for_plot === nothing ? NaN : pp_for_plot),
        "raw_posterior_prob"    => json_number(pp === nothing ? NaN : pp),
        "posterior_calibrated"  => json_number(pp_cal === nothing ? NaN : pp_cal),
        "fdr_calibrated"        => json_number(fdr_cal === nothing ? NaN : fdr_cal),
        # 5 MC-Dropout prior columns
        "prior_mc_mean"         => json_number(prior_mc_mean      === nothing ? NaN : prior_mc_mean),
        "prior_mc_std"          => json_number(prior_mc_std       === nothing ? NaN : prior_mc_std),
        "prior_mc_ci_low"       => json_number(prior_mc_ci_low    === nothing ? NaN : prior_mc_ci_low),
        "prior_mc_ci_high"      => json_number(prior_mc_ci_high   === nothing ? NaN : prior_mc_ci_high),
        "prior_contribution"    => json_number(prior_contribution === nothing ? NaN : prior_contribution),
        "BFDR"                  => json_number(bfdr_val === nothing ? NaN : bfdr_val),
        "PEP"                   => json_number(pep_val === nothing ? NaN : pep_val),
        "mean_log2fc"           => json_number(lfc === nothing ? NaN : lfc),
        "sd_log2fc"             => json_number(sd_lfc === nothing ? NaN : sd_lfc),
        "fold_change"           => json_number(fc  === nothing ? NaN : fc),
        "fold_change_lo"        => json_number(fc_lo === nothing ? NaN : fc_lo),
        "fold_change_hi"        => json_number(fc_hi === nothing ? NaN : fc_hi),
        "bf_enrichment"         => json_number(bfe === nothing ? NaN : bfe),
        "bf_correlation"        => json_number(bfc === nothing ? NaN : bfc),
        "bf_detected"           => json_number(bfd === nothing ? NaN : bfd),
        "evidence_label"        => json_string(ev_label),
        "diagnostic_flag"       => json_string(diag_flag),
        "sensitivity_range"     => json_number(sens_range === nothing ? NaN : sens_range),
        "model_disagreement" => hasproperty(row, :model_disagreement) ?
            (ismissing(row.model_disagreement) ? "false" : (Bool(row.model_disagreement) ? "true" : "false")) : "false",
        "component" => hasproperty(row, :Component) ?
            json_string(ismissing(row.Component) ? "" : string(row.Component)) : json_string(""),
        "p_h0" => hasproperty(row, :P_H0) ? json_number(_safe_float(row.P_H0) === nothing ? NaN : _safe_float(row.P_H0)) : json_number(NaN),
        "p_agnostic" => hasproperty(row, :P_agnostic) ? json_number(_safe_float(row.P_agnostic) === nothing ? NaN : _safe_float(row.P_agnostic)) : json_number(NaN),
        "p_h1" => hasproperty(row, :P_H1) ? json_number(_safe_float(row.P_H1) === nothing ? NaN : _safe_float(row.P_H1)) : json_number(NaN),
        "classification_stability" => json_string(
            hasproperty(row, :classification_stability) ?
                (ismissing(row.classification_stability) ? "" : string(row.classification_stability)) : ""
        ),
        "n_observations"      => n_obs === nothing ? "null" : json_number(n_obs),
        "mean_residual"       => json_number(mean_res === nothing ? NaN : mean_res),
        "max_abs_residual"    => json_number(max_abs === nothing ? NaN : max_abs),
        "is_low_data"         => is_low === nothing ? "null" : json_bool(is_low),
        "is_residual_outlier" => is_outlier === nothing ? "null" : json_bool(is_outlier),
        "bf_em"               => json_number(bf_em_val === nothing ? NaN : bf_em_val),
        "bf_copula"           => json_number(bf_cop_val === nothing ? NaN : bf_cop_val),
        "pareto_k"            => json_number(pk_val === nothing ? NaN : pk_val),
        # bait-report BB×MNAR warning icon payload
        "bb_mnar_codriven"    => json_bool(bb_codriven),
        "missing_fraction"    => json_number(missing_frac === nothing ? NaN : missing_frac),
    )
end

"""
    _build_dnn_prior_json(results::DataFrame, config) -> String

Build the top-level `dnn_prior` JSON block consumed
by the new DNN Prior tab's JS init.

Returns an empty-state object `{"empty": true, "reason": "..."}` when the 5
MC-Dropout prior columns are absent OR all-NaN (extension not loaded OR
`run_dnn_prior_mc_dropout = false`). Otherwise returns `{"empty": false,
"rows": [...]}` where each row carries 8 keys: `protein`, `prior_mc_mean`,
`prior_mc_std`, `prior_mc_ci_low`, `prior_mc_ci_high`, `posterior_prob`,
`prior_contribution`, `BFDR`.

audit: this helper emits only the 4 derived MC stats + posterior +
contribution + BFDR per row. The K-sample matrix lives only as a stack-local
inside `compute_mc_prior!` and is never persisted to JSON or JLD2.

The `config` parameter is accepted for API symmetry with other `_build_*_json`
helpers; the current body does not consume it (the empty-state path infers
from the column presence/NaN-ness on the DataFrame).
"""
function _build_dnn_prior_json(results::DataFrame, config)::String
    if !hasproperty(results, :prior_mc_mean) ||
       all(x -> ismissing(x) || !isfinite(x), results.prior_mc_mean)
        return json_object(
            "empty"  => json_bool(true),
            "reason" => json_string("MC-Dropout columns NaN — extension not loaded or run_dnn_prior_mc_dropout=false"),
        )
    end
    rows = String[]
    for r in eachrow(results)
        protein_str       = string(r.Protein)
        pmm = hasproperty(r, :prior_mc_mean)      ? _safe_float(r.prior_mc_mean)      : nothing
        pms = hasproperty(r, :prior_mc_std)       ? _safe_float(r.prior_mc_std)       : nothing
        pcl = hasproperty(r, :prior_mc_ci_low)    ? _safe_float(r.prior_mc_ci_low)    : nothing
        pch = hasproperty(r, :prior_mc_ci_high)   ? _safe_float(r.prior_mc_ci_high)   : nothing
        pp  = hasproperty(r, :posterior_prob)     ? _safe_float(r.posterior_prob)     : nothing
        pc  = hasproperty(r, :prior_contribution) ? _safe_float(r.prior_contribution) : nothing
        bfdr_val = hasproperty(r, :BFDR)          ? _safe_float(r.BFDR)               : nothing
        # Metalearner Stack output (the FINAL prior fed into the Bayesian update),
        # distinct from the DNN-only prior_mc_mean. Absent → NaN (plot hides).
        mlp = hasproperty(r, :MetaClassifier)     ? _safe_float(r.MetaClassifier)     : nothing
        push!(rows, json_object(
            "protein"            => json_string(protein_str),
            "prior_mc_mean"      => json_number(pmm === nothing ? NaN : pmm),
            "prior_mc_std"       => json_number(pms === nothing ? NaN : pms),
            "prior_mc_ci_low"    => json_number(pcl === nothing ? NaN : pcl),
            "prior_mc_ci_high"   => json_number(pch === nothing ? NaN : pch),
            "posterior_prob"     => json_number(pp  === nothing ? NaN : pp),
            "prior_contribution" => json_number(pc  === nothing ? NaN : pc),
            "BFDR"               => json_number(bfdr_val === nothing ? NaN : bfdr_val),
            "metalearner_prior"  => json_number(mlp === nothing ? NaN : mlp),
        ))
    end
    return json_object("empty" => json_bool(false), "rows" => "[" * join(rows, ",") * "]")
end

"""Base64-encode all existing plot files into a JSON object."""
function _build_plots_json(config::CONFIG)::String
    # All main-report plots now have interactive Plotly.js equivalents.
    # Static PNG fallbacks are no longer embedded — saves ~5-8 MB.
    # Differential-report plots are handled in differential_report.html.
    return "{}"
end

"""Build methods section JSON.

The optional `metalearner_status` symbol is passed through
to the structured-data builder so the Methods tab can render a "Metalearner
Status" subsection describing Variante B fallback semantics. Defaults to
`:loaded` (no fallback notice) for backward compatibility with old callers.
"""
function _build_methods_json(config::CONFIG, results::DataFrame,
                             metalearner_status::Symbol = :loaded)::String
    methods_text  = try generate_methods_text(config, results)  catch; "" end
    repro_block   = try generate_reproducibility_block(config)  catch; "" end
    params        = try generate_methods_parameters(config)     catch; Pair{String,String}[] end
    structured    = try _build_structured_methods_data(config, results, metalearner_status) catch; "{}" end
    # Embeddings & Similarity Methods block. Guarded with isdefined
    # so this file can land independently of methods_generator.jl ordering.
    embeddings_block = (isdefined(@__MODULE__, :_methods_embeddings_block) ?
                        (try _methods_embeddings_block(config) catch; "" end) : "")

    param_pairs = [json_string(k) * ":" * json_string(v) for (k, v) in params]
    params_json = "{" * join(param_pairs, ",") * "}"

    return json_object(
        "text"           => json_string(methods_text),
        "reproducibility"=> json_string(repro_block),
        "parameters"     => params_json,
        "structured"     => structured,
        # also surface at methods top level for convenience.
        "metalearner_status" => json_string(string(metalearner_status)),
        # rendered Methods HTML for Embeddings & Similarity. Empty
        # string when `cfg.embeddings_config.run_embeddings = false` so the Methods
        # tab consumer can keep the card hidden.
        "embeddings_block"   => json_string(embeddings_block),
    )
end

"""Build BMA summary JSON from analysis_result. Returns `"{}"` when not applicable."""
function _build_bma_summary_json(analysis_result, results_df=nothing)::String
    analysis_result === nothing && return "{}"

    # Filter to detected proteins only (non-detected have missing BFs)
    if results_df !== nothing && results_df isa DataFrame
        results_df = _filter_detected(results_df)
    end

    # Check if BMA result is available
    bma = nothing
    if hasproperty(analysis_result, :bma_result)
        bma = analysis_result.bma_result
    end
    (bma === nothing || !isa(bma, BMAResult)) && return "{}"

    n = length(bma.bf)
    log10_cop = log10.(max.(bma.copula_result.bf, 1e-300))
    log10_em  = log10.(max.(bma.em3c_result.bf, 1e-300))
    log10_avg = log10.(max.(bma.bf, 1e-300))

    n_disagree   = count(bma.model_disagreement)
    n_extreme_10 = count(x -> abs(x) > 10, log10_avg)
    n_extreme_20 = count(x -> abs(x) > 20, log10_avg)

    pairs = Pair{String,String}[
        "em_weight"               => json_number(bma.em_weight),
        "copula_weight"           => json_number(bma.copula_weight),
        "prior_odds"              => json_number(bma.prior_odds),
        "median_log10_bf_copula"  => json_number(median(log10_cop)),
        "median_log10_bf_em"      => json_number(median(log10_em)),
        "median_log10_bf_avg"     => json_number(median(log10_avg)),
        "n_disagree"              => json_number(n_disagree),
        "n_extreme_10"            => json_number(n_extreme_10),
        "n_extreme_20"            => json_number(n_extreme_20),
        "n_proteins"              => json_number(n),
        "weighting_method"        => json_string("LOO stacking (Yao et al. 2018)"),
    ]

    # Per-protein disagreement scatter data (P_EM vs P_copula)
    P_EM = bma.em3c_result.posterior_prob
    # Clamp BFs to prevent 0/1 extremes in posterior
    clamped_bf = clamp.(bma.copula_result.bf, 1e-10, 1e10)
    P_copula_vec = (clamped_bf .* bma.prior_odds) ./ (1.0 .+ clamped_bf .* bma.prior_odds)
    P_copula_vec = clamp.(P_copula_vec, 0.0, 1.0)

    # Subsample for performance if > 2000 proteins
    scatter_indices = if n > 2000
        sort(collect(1:n)[sortperm(rand(n))[1:2000]])
    else
        1:n
    end

    scatter_em = json_array([json_number(P_EM[i]) for i in scatter_indices])
    scatter_cop = json_array([json_number(P_copula_vec[i]) for i in scatter_indices])
    scatter_disagree = json_array([Bool(bma.model_disagreement[i]) ? "true" : "false" for i in scatter_indices])

    push!(pairs, "scatter_p_em" => scatter_em)
    push!(pairs, "scatter_p_copula" => scatter_cop)
    push!(pairs, "scatter_disagree" => scatter_disagree)

    # Add protein names for cross-tab selection in BMA scatter
    # Guard: scatter_indices are into bma.bf (all proteins), results_df only has detected proteins.
    # Only index into results_df when every scatter index is within bounds.
    n_results = (results_df !== nothing && results_df isa DataFrame) ? nrow(results_df) : 0
    scatter_in_results = n_results > 0 && maximum(scatter_indices) <= n_results
    if scatter_in_results && hasproperty(results_df, :Protein)
        scatter_names = json_array([json_string(string(results_df.Protein[i])) for i in scatter_indices])
        push!(pairs, "scatter_names" => scatter_names)
    end

    # BF correlation scatter data — log10 BFs for subsampled proteins
    scatter_log10_cop = json_array([json_number(log10_cop[i]) for i in scatter_indices])
    scatter_log10_em_arr = json_array([json_number(log10_em[i]) for i in scatter_indices])
    push!(pairs, "scatter_log10_bf_copula" => scatter_log10_cop)
    push!(pairs, "scatter_log10_bf_em" => scatter_log10_em_arr)

    # BF ratio histogram data — log10(bf_em / bf_copula) = log10_em - log10_cop
    log10_ratio = log10_em .- log10_cop
    ratio_arr = json_array([json_number(log10_ratio[i]) for i in scatter_indices])
    push!(pairs, "bf_ratio_log10" => ratio_arr)

    # Correlation statistics — cor is imported at module level via Statistics
    valid_mask = isfinite.(log10_cop) .& isfinite.(log10_em)
    n_valid = sum(valid_mask)
    if n_valid > 10
        pearson_r = cor(log10_cop[valid_mask], log10_em[valid_mask])
        push!(pairs, "bf_corr_pearson" => json_number(pearson_r))
        # Spearman: rank correlation via double sortperm
        ranked_cop = sortperm(sortperm(log10_cop[valid_mask]))
        ranked_em = sortperm(sortperm(log10_em[valid_mask]))
        spearman_r = cor(Float64.(ranked_cop), Float64.(ranked_em))
        push!(pairs, "bf_corr_spearman" => json_number(spearman_r))
    else
        push!(pairs, "bf_corr_pearson" => json_number(NaN))
        push!(pairs, "bf_corr_spearman" => json_number(NaN))
    end

    # Copula contribution: log10(combined_BF) - log10(product of individual BFs)
    # This shows how much the copula dependence structure changes the ranking
    if scatter_in_results &&
       hasproperty(results_df, :BF) && hasproperty(results_df, :bf_enrichment) &&
       hasproperty(results_df, :bf_correlation) && hasproperty(results_df, :bf_detected)
        combined_log_bf = Float64[]
        copula_contribution = Float64[]
        for i in scatter_indices
            bf_comb = results_df.BF[i]
            bf_e = results_df.bf_enrichment[i]
            bf_c = results_df.bf_correlation[i]
            bf_d = results_df.bf_detected[i]
            log_comb = log10(max(_report_float(bf_comb), 1e-300))
            log_indep = log10(max(_report_float(bf_e), 1e-300)) + log10(max(_report_float(bf_c), 1e-300)) + log10(max(_report_float(bf_d), 1e-300))
            push!(combined_log_bf, log_comb)
            push!(copula_contribution, log_comb - log_indep)
        end
        push!(pairs, "scatter_combined_log_bf" => json_array([json_number(v) for v in combined_log_bf]))
        push!(pairs, "scatter_copula_contribution" => json_array([json_number(v) for v in copula_contribution]))
    end

    # Pareto k-hat summary if available
    if bma.pareto_k !== nothing
        k = bma.pareto_k
        push!(pairs, "pareto_k_median" => json_number(median(k)))
        push!(pairs, "pareto_k_max" => json_number(maximum(k)))
        push!(pairs, "pareto_k_n_problematic" => json_number(count(x -> x > 0.7, k)))
    end

    return json_object(pairs...)
end

# ---------------------------------------------------------------------------
# Mixture model tab data
# ---------------------------------------------------------------------------

"""Serialize a LatentClassResult class parameter NamedTuple to JSON."""
function _lc_params_json(params)::String
    return json_object(
        "mu"        => json_number(params.mu),
        "sigma"     => json_number(params.sigma),
        "precision" => json_number(params.precision),
    )
end

"""
Build mixture model JSON for the report Mixture Model tab.

Returns `"null"` when no 3-component latent class result is available.
If `results_df` contains `bf_enrichment` and `bf_correlation` columns,
their log values are embedded directly in the scatter data so the
client-side plot has no implicit dependency on other data sections.
"""
function _build_mixture_model_json(analysis_result, results_df=nothing)::String
    analysis_result === nothing && return "null"

    # Filter to detected proteins only (non-detected have missing BFs; exclude from plots)
    if results_df !== nothing && results_df isa DataFrame
        results_df = _filter_detected(results_df)
    end

    # Extract latent_class_result from analysis_result
    lc = nothing
    if hasproperty(analysis_result, :latent_class_result)
        lc = analysis_result.latent_class_result
    end
    if lc === nothing && hasproperty(analysis_result, :bma_result) && analysis_result.bma_result !== nothing
        bma = analysis_result.bma_result
        if hasproperty(bma, :em3c_result)
            lc = bma.em3c_result
        elseif hasproperty(bma, :latent_class_result)
            lc = bma.latent_class_result
        end
    end
    (lc === nothing || !isa(lc, LatentClassResult)) && return "null"
    lc.responsibilities === nothing && return "null"

    # a. Component parameters (for density overlays)
    params = json_object(
        "background"  => _lc_params_json(lc.class_parameters["background"]),
        "agnostic"    => haskey(lc.class_parameters, "agnostic") ?
            _lc_params_json(lc.class_parameters["agnostic"]) : "null",
        "interaction" => _lc_params_json(lc.class_parameters["interaction"]),
    )

    # MC convolution for combined BF density (replaces sum-of-Gaussians)
    if lc.per_dimension_params !== nothing
        component_keys = ["background", "agnostic", "interaction"]
        mc_densities = Dict{String, Any}()
        for ck in component_keys
            result = _mc_combined_density(lc, ck; n_mc=100_000, n_grid=200)
            if result !== nothing
                mc_densities[ck] = result
            end
        end
        # Rebuild params with density_x/density_y appended
        if !isempty(mc_densities)
            params_with_density_pairs = Pair{String,String}[]
            for ck in component_keys
                cp = get(lc.class_parameters, ck, nothing)
                cp === nothing && continue
                base_pairs = Pair{String,String}[
                    "mu" => json_number(cp.mu),
                    "sigma" => json_number(cp.sigma),
                    "precision" => json_number(cp.precision),
                ]
                if haskey(mc_densities, ck)
                    d = mc_densities[ck]
                    push!(base_pairs, "density_x" => json_array([json_number(v) for v in d.x]))
                    push!(base_pairs, "density_y" => json_array([json_number(v) for v in d.y]))
                end
                push!(params_with_density_pairs, ck => json_object(base_pairs...))
            end
            params = json_object(params_with_density_pairs...)
        end
    end

    # b. Mixing weights (for weight bar chart)
    weights = json_array([json_number(w) for w in lc.mixing_weights])
    weight_labels = length(lc.mixing_weights) == 3 ?
        json_array([json_string("H0"), json_string("Agnostic"), json_string("H1")]) :
        json_array([json_string("H0"), json_string("H1")])

    # c. Convergence trace (for EM convergence plot)
    trace = json_array([json_number(ll) for ll in lc.free_energy])

    # d. Scatter data with embedded log-BF coordinates (explicit data dependency)
    # IMPORTANT: lc.responsibilities is in EM-order (after innerjoin reordering in pipeline),
    # but results_df is in copula_df order (getIDs order). These differ when innerjoin reorders
    # proteins. Use the pre-mapped Component/P_H0/P_agnostic/P_H1 columns from results_df
    # (set in pipeline.jl:611-638 via detected_full_indices) which are latent class results
    # correctly aligned to the results_df protein order.
    scatter = "null"
    has_mapped_lc = results_df !== nothing && results_df isa DataFrame &&
                    hasproperty(results_df, :Component) &&
                    hasproperty(results_df, :P_H0)
    has_raw_lc = lc.responsibilities !== nothing

    if has_mapped_lc || has_raw_lc
        n = has_mapped_lc ? nrow(results_df) : size(lc.responsibilities, 1)

        # Build component assignments — prefer pre-mapped columns from results_df
        if has_mapped_lc
            components = [ismissing(c) ? "Uncertain" : string(c) for c in results_df.Component]
        else
            components = Vector{String}(undef, n)
            labels = length(lc.mixing_weights) == 3 ? ["H0", "agnostic", "H1"] : ["H0", "H1"]
            threshold = 0.7
            for i in 1:n
                max_idx = argmax(lc.responsibilities[i, :])
                components[i] = lc.responsibilities[i, max_idx] > threshold ? labels[max_idx] : "Uncertain"
            end
        end

        scatter_pairs = Pair{String, String}[
            "components" => json_array([json_string(c) for c in components]),
            "n_proteins" => json_number(n),
        ]

        # If results_df is provided with bf_enrichment/bf_correlation, embed log-BFs
        if results_df !== nothing &&
           results_df isa DataFrame &&
           hasproperty(results_df, :bf_enrichment) &&
           hasproperty(results_df, :bf_correlation)
            log_bf_e_raw = [log(max(_report_float(v), 1e-300)) for v in results_df.bf_enrichment]
            log_bf_c_raw = [log(max(_report_float(v), 1e-300)) for v in results_df.bf_correlation]
            log_bf_e_win = _winsorize_quantile(log_bf_e_raw)
            log_bf_c_win = _winsorize_quantile(log_bf_c_raw)
            log_bf_e = [json_number(v) for v in log_bf_e_win]
            log_bf_c = [json_number(v) for v in log_bf_c_win]
            push!(scatter_pairs, "log_bf_enrichment" => json_array(log_bf_e))
            push!(scatter_pairs, "log_bf_correlation" => json_array(log_bf_c))
        end

        # Protein names for rich tooltips
        if results_df !== nothing && results_df isa DataFrame && hasproperty(results_df, :Protein)
            push!(scatter_pairs, "protein_names" => json_array([json_string(string(p)) for p in results_df.Protein]))
        end

        # Per-protein responsibilities — prefer pre-mapped columns from results_df
        if has_mapped_lc && hasproperty(results_df, :P_agnostic)
            push!(scatter_pairs, "p_h0" => json_array([json_number(_report_float(v)) for v in results_df.P_H0]))
            push!(scatter_pairs, "p_agnostic" => json_array([json_number(_report_float(v)) for v in results_df.P_agnostic]))
            push!(scatter_pairs, "p_h1" => json_array([json_number(_report_float(v)) for v in results_df.P_H1]))
        elseif has_mapped_lc && hasproperty(results_df, :P_H1)
            push!(scatter_pairs, "p_h0" => json_array([json_number(_report_float(v)) for v in results_df.P_H0]))
            push!(scatter_pairs, "p_h1" => json_array([json_number(_report_float(v)) for v in results_df.P_H1]))
        elseif has_raw_lc
            # Fallback: use raw responsibilities (only correct if results_df is absent)
            if size(lc.responsibilities, 2) >= 3
                push!(scatter_pairs, "p_h0" => json_array([json_number(lc.responsibilities[i, 1]) for i in 1:n]))
                push!(scatter_pairs, "p_agnostic" => json_array([json_number(lc.responsibilities[i, 2]) for i in 1:n]))
                push!(scatter_pairs, "p_h1" => json_array([json_number(lc.responsibilities[i, 3]) for i in 1:n]))
            elseif size(lc.responsibilities, 2) == 2
                push!(scatter_pairs, "p_h0" => json_array([json_number(lc.responsibilities[i, 1]) for i in 1:n]))
                push!(scatter_pairs, "p_h1" => json_array([json_number(lc.responsibilities[i, 2]) for i in 1:n]))
            end
        end

        # Log-BF detection values
        if results_df !== nothing && results_df isa DataFrame && hasproperty(results_df, :bf_detected)
            log_bf_d = [json_number(log(max(_report_float(v), 1e-300))) for v in results_df.bf_detected]
            push!(scatter_pairs, "log_bf_detection" => json_array(log_bf_d))
        end

        scatter = json_object(scatter_pairs...)
    end

    # e. Full JSON
    return json_object(
        "params"       => params,
        "weights"      => weights,
        "weight_labels" => weight_labels,
        "convergence"  => trace,
        "scatter"      => scatter,
    )
end

# ---------------------------------------------------------------------------
# Validation tab data
# ---------------------------------------------------------------------------

"""Build validation JSON object for the report. Returns `"null"` when no validation data."""
function _build_validation_json(validation_result)::String
    validation_result === nothing && return "null"

    # Quality gate matrix (9 cells) with histogram/PDF overlay data
    qg_json = "null"
    if validation_result.quality_gates !== nothing
        qg = validation_result.quality_gates
        cells_json = String[]
        for j in 1:size(qg.cells, 2)      # components
            for i in 1:size(qg.cells, 1)   # marginals
                cell = qg.cells[i, j]
                # Serialize histogram and PDF arrays as JSON number arrays
                _float_arr(v) = "[" * join([json_number(x) for x in v], ",") * "]"
                # Detection uses chi-squared GOF p-value, enrichment/correlation use KS statistic
                test_type = cell.marginal == :detection ? "chisq" : "ks"
                push!(cells_json, json_object(
                    "marginal" => json_string(String(cell.marginal)),
                    "component" => json_string(String(cell.component)),
                    "ks" => json_number(cell.ks_statistic),
                    "test_type" => json_string(test_type),
                    "status" => json_string(String(cell.status)),
                    "n_effective" => json_number(cell.n_effective),
                    "remediation" => cell.remediation_applied ? "true" : "false",
                    "hist_bin_edges" => _float_arr(cell.hist_bin_edges),
                    "hist_counts" => _float_arr(cell.hist_counts),
                    "fitted_pdf_x" => _float_arr(cell.fitted_pdf_x),
                    "fitted_pdf_y" => _float_arr(cell.fitted_pdf_y),
                ))
            end
        end
        qg_json = json_object(
            "cells" => json_array(cells_json),
            "overall_status" => json_string(String(qg.overall_status)),
            "remediation_details" => json_array([json_string(d) for d in qg.remediation_details]),
        )
    end

    # KL contamination
    kl_json = "null"
    if validation_result.kl_contamination !== nothing
        kl = validation_result.kl_contamination
        kl_json = json_object(
            "enrichment" => json_number(kl.kl_enrichment),
            "correlation" => json_number(kl.kl_correlation),
            "detection" => json_number(kl.kl_detection),
            "joint" => json_number(kl.kl_joint),
            "pure_h1_count" => json_number(kl.pure_h1_count),
            "per_stream_pass" => kl.per_stream_pass ? "true" : "false",
        )
    end

    # Consistency checks
    check_pairs = Pair{String,String}[]
    for (k, v) in validation_result.consistency_checks
        push!(check_pairs, k => (v ? "true" : "false"))
    end
    checks_json = isempty(check_pairs) ? "{}" : json_object(check_pairs...)

    return json_object(
        "quality_gates" => qg_json,
        "kl_contamination" => kl_json,
        "consistency" => checks_json,
        "overall_pass" => validation_result.overall_pass ? "true" : "false",
    )
end

# ---------------------------------------------------------------------------
# Sensitivity tab data
# ---------------------------------------------------------------------------

"""
Build sensitivity JSON for the report. Returns `"null"` when no sensitivity data.

Serializes rank-correlation, Spearman matrix, band plot, overlay, and stacking weight
data from a `SensitivityResult` for rendering as interactive Plotly plots in the HTML report.
"""
function _build_sensitivity_json(sensitivity_result)::String
    sensitivity_result === nothing && return "null"

    sr = sensitivity_result
    n_proteins = size(sr.posterior_matrix, 1)
    n_settings = size(sr.posterior_matrix, 2)

    # Format prior labels from PriorSetting
    prior_labels = [json_string(ps.label) for ps in sr.prior_settings]

    # --- Rank correlation data: Spearman rho per setting vs baseline ---
    baseline_col = sr.posterior_matrix[:, sr.baseline_index]
    correlations = Float64[]
    labels = String[]
    for i in 1:n_settings
        rho = corspearman(sr.posterior_matrix[:, i], baseline_col)
        push!(correlations, isnan(rho) ? 1.0 : rho)
        push!(labels, sr.prior_settings[i].label)
    end

    rankcorr_json = json_object(
        "labels"       => json_array([json_string(l) for l in labels]),
        "correlations" => json_array([json_number(c) for c in correlations]),
        "baseline_idx" => json_number(sr.baseline_index),
    )

    cs = sr.classification_stability

    # --- Pairwise Spearman correlation matrix (VIZ-02) ---
    spearman_mat = zeros(n_settings, n_settings)
    for i in 1:n_settings, j in 1:n_settings
        rho = corspearman(sr.posterior_matrix[:, i], sr.posterior_matrix[:, j])
        spearman_mat[i, j] = isnan(rho) ? 1.0 : rho
    end
    spearman_rows = String[]
    for i in 1:n_settings
        push!(spearman_rows, json_array([json_number(spearman_mat[i, j]) for j in 1:n_settings]))
    end
    spearman_json = json_object(
        "matrix" => "[" * join(spearman_rows, ",") * "]",
        "labels" => json_array([json_string(ps.label) for ps in sr.prior_settings]),
    )

    # --- BMA stacking weight trace (optional, only for BMA sweeps) ---
    bma_settings = filter(ps -> ps.model == :bma, sr.prior_settings)
    stacking_json = if !isempty(bma_settings)
        w_em_vals = [json_number(hasproperty(ps.params, :w_em) ? ps.params.w_em : NaN) for ps in bma_settings]
        w_cop_vals = [json_number(hasproperty(ps.params, :w_cop) ? ps.params.w_cop : NaN) for ps in bma_settings]
        bma_labels = [json_string(ps.label) for ps in bma_settings]
        json_object(
            "labels" => json_array(bma_labels),
            "w_em"   => json_array(w_em_vals),
            "w_cop"  => json_array(w_cop_vals),
        )
    else
        "null"
    end

    # --- Band plot data (VIZ-03): all proteins sorted by range, capped ---
    sorted_all = sort(sr.summary, :range, rev=true)

    # Cap to top 100 + all boundary crossers
    crosser_names = Set(cs.Protein[cs.threshold_crossing_0_5])
    n_cap = 100
    band_proteins = if nrow(sorted_all) <= n_cap
        sorted_all
    else
        top_band = first(sorted_all, n_cap)
        top_names = Set(top_band.Protein)
        # Add any crossers not already in top
        extra_crossers = filter(row -> row.Protein in crosser_names && !(row.Protein in top_names), sorted_all)
        vcat(top_band, extra_crossers)
    end

    # Look up boundary-crosser status for band proteins
    band_bc = Bool[]
    for p in band_proteins.Protein
        idx = findfirst(==(p), cs.Protein)
        push!(band_bc, idx !== nothing && cs.threshold_crossing_0_5[idx])
    end

    band_json = json_object(
        "protein_names"     => json_array([json_string(string(p)) for p in band_proteins.Protein]),
        "mins"              => json_array([json_number(v) for v in band_proteins.min_posterior]),
        "maxs"              => json_array([json_number(v) for v in band_proteins.max_posterior]),
        "means"             => json_array([json_number(v) for v in band_proteins.mean_posterior]),
        "boundary_crossers" => json_array([json_bool(b) for b in band_bc]),
        "n_proteins"        => json_number(nrow(band_proteins)),
    )

    # --- Overlay data (VIZ-04): top 15 most sensitive proteins ---
    n_overlay = min(15, nrow(sorted_all))
    top_overlay = first(sorted_all, n_overlay)
    top_overlay_idx = [findfirst(==(p), sr.protein_names) for p in top_overlay.Protein]

    # Per-setting arrays: overlay_values[setting_idx] = [posterior for each top protein]
    overlay_setting_arrays = String[]
    for si in 1:n_settings
        vals = [json_number(sr.posterior_matrix[pi, si]) for pi in top_overlay_idx]
        push!(overlay_setting_arrays, json_array(vals))
    end

    overlay_json = json_object(
        "protein_names"  => json_array([json_string(string(p)) for p in top_overlay.Protein]),
        "setting_labels" => json_array([json_string(ps.label) for ps in sr.prior_settings]),
        "values"         => "[" * join(overlay_setting_arrays, ",") * "]",
    )

    return json_object(
        "rankcorr"          => rankcorr_json,
        "spearman_matrix"   => spearman_json,
        "band_plot"         => band_json,
        "overlay"           => overlay_json,
        "stacking_weights"  => stacking_json,
    )
end

# ---------------------------------------------------------------------------
# Input QC JSON builders
# ---------------------------------------------------------------------------

"""Build input QC JSON for the Data Quality tab."""
function _build_qc_json(analysis_result)::String
    analysis_result === nothing && return "null"
    qc = analysis_result.input_qc
    qc === nothing && return "null"
    return json_object(
        "overall_flag"          => json_string(String(qc.overall_flag)),
        "scale"                 => _build_qc_scale_json(qc.scale),
        "replicate_correlation" => _build_qc_correlation_json(qc.replicate_correlation),
        "missingness"           => _build_qc_missingness_json(qc.missingness),
        "intensity_shape"       => _build_qc_shape_json(qc.intensity_shape),
        "pca_separation"        => _build_qc_pca_json(qc.pca_separation),
    )
end

"""Serialize a ScaleCheckResult to JSON."""
function _build_qc_scale_json(scale)::String
    scale === nothing && return "null"
    protocol_jsons = String[]
    for p in scale.protocols
        push!(protocol_jsons, json_object(
            "index"     => json_number(p.protocol_index),
            "max_value" => json_number(p.max_value),
            "flag"      => json_string(String(p.flag)),
        ))
    end
    return json_object(
        "flag"      => json_string(String(scale.flag)),
        "protocols" => json_array(protocol_jsons),
    )
end

"""Serialize a ReplicateCorrelationResult to JSON."""
function _build_qc_correlation_json(corr)::String
    corr === nothing && return "null"
    check_jsons = String[]
    for c in corr.checks
        m = c.correlation_matrix
        n = size(m, 1)
        rows = String[]
        for i in 1:n
            row = [json_number(m[i, j]) for j in 1:n]
            push!(rows, json_array(row))
        end
        sc = c.shared_counts
        sc_rows = String[]
        for i in 1:size(sc, 1)
            row = [json_number(sc[i, j]) for j in 1:size(sc, 2)]
            push!(sc_rows, json_array(row))
        end
        push!(check_jsons, json_object(
            "protocol_index"     => json_number(c.protocol_index),
            "experiment_index"   => json_number(c.experiment_index),
            "group"              => json_string(String(c.group)),
            "correlation_matrix" => json_array(rows),
            "shared_counts"      => json_array(sc_rows),
            "n_replicates"       => json_number(c.n_replicates),
            "min_correlation"    => json_number(c.min_correlation),
            "flag"               => json_string(String(c.flag)),
        ))
    end
    return json_object(
        "flag"   => json_string(String(corr.flag)),
        "checks" => json_array(check_jsons),
    )
end

"""Serialize a MissingnessResult to JSON."""
function _build_qc_missingness_json(miss)::String
    miss === nothing && return "null"
    check_jsons = String[]
    for m in miss.checks
        fracs = [json_number(f) for f in m.missing_fractions]
        push!(check_jsons, json_object(
            "protocol_index"   => json_number(m.protocol_index),
            "experiment_index" => json_number(m.experiment_index),
            "group"            => json_string(String(m.group)),
            "missing_fractions" => json_array(fracs),
            "median_fraction"  => json_number(m.median_fraction),
            "max_ratio"        => json_number(m.max_ratio),
            "flag"             => json_string(String(m.flag)),
        ))
    end
    return json_object(
        "flag"   => json_string(String(miss.flag)),
        "checks" => json_array(check_jsons),
    )
end

"""Serialize an IntensityShapeResult to JSON."""
function _build_qc_shape_json(shape)::String
    shape === nothing && return "null"
    check_jsons = String[]
    for s in shape.checks
        push!(check_jsons, json_object(
            "protocol_index"   => json_number(s.protocol_index),
            "experiment_index" => json_number(s.experiment_index),
            "group"            => json_string(String(s.group)),
            "replicate_index"  => json_number(s.replicate_index),
            "n_values"         => json_number(s.n_values),
            "excess_kurtosis"  => json_number(s.excess_kurtosis),
            "skewness_val"     => json_number(s.skewness_val),
            "spike_fraction"   => json_number(s.spike_fraction),
            "flag"             => json_string(String(s.flag)),
        ))
    end
    return json_object(
        "flag"   => json_string(String(shape.flag)),
        "checks" => json_array(check_jsons),
    )
end

"""Serialize a PCASeparationResult to JSON."""
function _build_qc_pca_json(pca)::String
    pca === nothing && return "null"
    if pca.fallback_level == :skipped
        return json_object(
            "flag"              => json_string(String(pca.flag)),
            "fallback_level"    => json_string("skipped"),
            "message"           => json_string(pca.message),
            "n_proteins_used"   => json_number(0),
            "n_proteins_total"  => json_number(pca.n_proteins_total),
            "pc_scores"         => json_array(String[]),
            "condition_labels"  => json_array(String[]),
            "protocol_labels"   => json_array(String[]),
            "variance_explained" => json_array(String[]),
            "fisher_ratio_pc1"  => json_number(0.0),
            "fisher_ratio_pc2"  => json_number(0.0),
            "per_protocol"      => "null",
        )
    end
    score_pairs = String[]
    for i in 1:size(pca.pc_scores, 1)
        push!(score_pairs, json_array([json_number(pca.pc_scores[i, 1]),
                                        json_number(pca.pc_scores[i, 2])]))
    end
    cond_labels = [json_string(l) for l in pca.condition_labels]
    proto_labels = [json_number(p) for p in pca.protocol_labels]
    var_explained = [json_number(v) for v in pca.variance_explained]
    per_proto = if pca.per_protocol !== nothing
        json_array([_build_qc_pca_json(pp) for pp in pca.per_protocol])
    else
        "null"
    end
    return json_object(
        "flag"              => json_string(String(pca.flag)),
        "fallback_level"    => json_string(String(pca.fallback_level)),
        "message"           => json_string(pca.message),
        "n_proteins_used"   => json_number(pca.n_proteins_used),
        "n_proteins_total"  => json_number(pca.n_proteins_total),
        "pc_scores"         => json_array(score_pairs),
        "condition_labels"  => json_array(cond_labels),
        "protocol_labels"   => json_array(proto_labels),
        "variance_explained" => json_array(var_explained),
        "fisher_ratio_pc1"  => json_number(pca.fisher_ratio_pc1),
        "fisher_ratio_pc2"  => json_number(pca.fisher_ratio_pc2),
        "per_protocol"      => per_proto,
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
        output::String = _diff_report_default_path(diff),
        pct_imputed_cells::Union{Nothing, AbstractDict, Real} = nothing,
        mask_aware_regression::Union{Nothing, Bool} = nothing)::Nothing
    @info "Generating differential interaction report..."

    json_blob = _build_diff_json(diff;
                                 pct_imputed_cells = pct_imputed_cells,
                                 mask_aware_regression = mask_aware_regression)

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

"""Build the full JSON blob for the differential report template.

`metalearner_status` is emitted at the top level so the
banner JS in `differential_report.html` can render the Variante B fallback
notice. Since the differential pipeline strips metalearner posteriors and
recomputes from copula-only evidence (see the `copula-note` in the template:
"metalearner stripped for differential comparison"), the differential report's
`metalearner_status` is independent of how the two underlying single-bait
runs computed their posteriors. We therefore emit `:loaded` (banner hidden):
the differential output never reflects metalearner adjustments by design, so
showing a Variante B warning would be misleading. The field is present for
schema parity with the single-bait sidecar and for forward-compat with future
plans that might wire per-condition status into the differential view.
"""
function _build_diff_json(diff::DifferentialResult;
                          pct_imputed_cells::Union{Nothing, AbstractDict, Real} = nothing,
                          mask_aware_regression::Union{Nothing, Bool} = nothing)::String
    # variance_recovery block. DifferentialResult does not carry
    # a CONFIG (only a DifferentialConfig with no mnar_* fields), so the
    # differential report cannot know which variance-recovery mode the upstream
    # per-condition runs used. Emit an empty payload (mode = "unknown", empty
    # block_html) so the card stays hidden. Schema parity with the single-bait
    # report JSON is preserved; future plans may thread per-condition CONFIGs
    # through DifferentialResult to populate this properly.
    variance_recovery_json = json_object(
        "mode"               => json_string("unknown"),
        "m"                  => "null",
        "seeds"              => "null",
        "inflation_max"      => "null",
        "inflation_override" => "null",
        "block_html"         => json_string(""),
    )

    # mask-aware regression v2b payload.
    # The differential report has no top-level CONFIG, but per-condition
    # AnalysisResult instances each carry their own `.config`.
    # We resolve `mask_aware_regression` in priority order:
    #   1. explicit kwarg (caller knows definitively),
    #   2. first AR's `config.mask_aware_regression` field (when available),
    #   3. fallback to `true` (the v2b default).
    # `block_html` renders only when v2b is enabled. `chip_html` renders the
    # Dict / scalar / nothing payload regardless — it is an observational fact
    # about the data, not a model setting.
    resolved_mask_aware = if mask_aware_regression !== nothing
        mask_aware_regression
    elseif diff.analyses !== nothing && !isempty(diff.analyses)
        first_ar = first(diff.analyses)
        if hasproperty(first_ar, :config) && first_ar.config !== nothing &&
           hasproperty(first_ar.config, :mask_aware_regression)
            first_ar.config.mask_aware_regression
        else
            true
        end
    else
        true
    end

    # resolve the first AR's CONFIG so the Normalisation
    # Methods-tab subsection (rendered UNCONDITIONALLY) can document the active
    # `normalisation_method`. The block is injected into the SAME Methods-tab
    # card as the mask-aware v2b block (template-free wiring): the always-present
    # normalisation <h4> first, followed by the (possibly empty) mask-aware <h4>.
    first_ar_config = (diff.analyses !== nothing && !isempty(diff.analyses)) ? begin
        far = first(diff.analyses)
        (hasproperty(far, :config) && far.config !== nothing) ? far.config : nothing
    end : nothing

    normalisation_block_html = first_ar_config === nothing ? "" :
                               _methods_normalisation_block(first_ar_config)

    mask_aware_block_html = if resolved_mask_aware &&
                              diff.analyses !== nothing && !isempty(diff.analyses)
        first_ar = first(diff.analyses)
        if hasproperty(first_ar, :config) && first_ar.config !== nothing
            _methods_mask_aware_regression_block(first_ar.config)
        else
            ""
        end
    else
        ""
    end

    mask_aware_json = json_object(
        "enabled"    => resolved_mask_aware ? "true" : "false",
        "block_html" => json_string(normalisation_block_html * mask_aware_block_html),
        "chip_html"  => json_string(_mask_aware_chip_html(pct_imputed_cells)),
    )

    # the DNN Prior + MC-Dropout Methods
    # subsection lives at `D.methods.dnn_prior_block_html` — added directly
    # inside `_build_diff_methods_json` below to avoid naming conflict with
    # the top-level `D.dnn_prior` per-row payload.

    return json_object(
        "meta"    => _build_diff_meta_json(diff),
        "results" => _build_diff_results_json(diff),
        "plots"   => _build_diff_plots_json(diff),
        # see docstring above for the `:loaded` choice.
        "metalearner_status" => json_string("loaded"),
        # variance_recovery payload (always empty in differential).
        "variance_recovery"  => variance_recovery_json,
        # mask-aware regression v2b — Methods
        # H4 + Data-Quality chip payload. Resolved from first AR's CONFIG when
        # available, else falls back to true (v2b default).
        "mask_aware"         => mask_aware_json,
        # six new differential per-tab payloads.
        # Five per-condition keys gate on `diff.analyses !== nothing` (return "null"
        # via Pattern A); `dbf_diagnostics` is unconditional (pure DataFrame compute
        # from `diff.results.dbf_diagnostic`).
        "calibration"        => _build_diff_calibration_json(diff),
        "sensitivity"        => _build_diff_sensitivity_json(diff),
        "qc"                 => _build_diff_qc_json(diff),
        "mixture"            => _build_diff_mixture_json(diff),
        "methods"            => _build_diff_methods_json(diff),
        "dbf_diagnostics"    => _build_diff_dbf_diagnostics_json(diff),
        # embeddings + condition similarity payloads.
        # The two embeddings_* keys reuse the single-bait builders against the
        # first AR in `diff.analyses` (pooled-samples convention).
        # The three condition_* keys reflect the k×k matrices + dendrogram for k≥2.
        "embeddings_sample"     => _build_sample_embedding_json(
            (diff.analyses === nothing || isempty(diff.analyses)) ? nothing : first(diff.analyses)),
        # B-08: switch to the diff-aware variant so the
        # `classes` array reflects `wide_df.kgroup_class` (k≥3) /
        # `wide_df.classification` (k=2) — NOT the single-AR
        # `LatentClassResult.protein_classes` labels (which were ~90% H0 →
        # single-colour visualisation).
        "embeddings_protein"    => _build_protein_embedding_json_for_diff(diff),
        "condition_matrix"      => _build_condition_matrix_json(diff),
        "condition_jaccard"     => _build_jaccard_json(diff),
        "condition_dendrogram"  => _build_dendrogram_json(diff),
        # Multi-Condition tab payload — per-pair
        # volcano data + k×k posterior median matrix. Returns "null" for k<3.
        "multi_condition"       => _build_multi_condition_json(diff),
        # top-level Validation Candidates ranked list (Results-tab pill)
        "validation_candidates" => _build_validation_candidates_block(diff),
        # per-condition payloads keyed by
        # condition label. Drives the per-tab dropdowns in the Calibration /
        # Sensitivity / Mixture Model / Data Quality tabs (B-04 / B-05 / B-06).
        # For k=2 the dict still carries both entries — the JS dropdown UI is
        # suppressed downstream, but the data path stays uniform.
        "per_condition"         => _build_per_condition_json(diff),
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
        "posterior_threshold"    => json_number(diff.config.posterior_threshold),
        "bfdr_threshold"        => json_number(diff.config.bfdr_threshold),
        "delta_log2fc_threshold"=> json_number(diff.config.delta_log2fc_threshold),
        "dbf_threshold"         => json_number(diff.config.dbf_threshold),
        "classification_method" => json_string(string(diff.config.classification_method)),
        "standardize_log2fc"    => diff.config.standardize_log2fc ? "true" : "false",
        "generated_at"          => json_string(Dates.format(now(), "yyyy-mm-dd HH:MM")),
        "package_version"       => json_string(_report_pkg_version()),
        # per-condition calibration status (Methods/banner consumers)
        "is_calibrated_A"       => diff.is_calibrated_A ? "true" : "false",
        "is_calibrated_B"       => diff.is_calibrated_B ? "true" : "false",
        # canonical D.meta.* access path for the
        # Multi-Condition tab visibility predicate.
        # condition_labels: unique condition labels (length(diff.contrasts)+1 for k-group,
        # [diff.condition_A, diff.condition_B] for legacy 2-group).
        # contrasts: nested [first, last] string pairs; empty array for legacy 2-group.
        "condition_labels" => json_array([json_string(l) for l in condition_labels(diff)]),
        "contrasts"        => json_array([
            json_array([json_string(String(first(p))), json_string(String(last(p)))])
            for p in diff.contrasts
        ]),
        # Sensitivity stability strip column count (top-N).
        # JS reads `D.meta.top_n_stability` to size the heatmap columns.
        "top_n_stability"  => json_number(hasproperty(diff.config, :top_n_stability) ?
                                              Int(diff.config.top_n_stability) : 20),
    )
end

"""Serialize each row of diff.results to JSON.

precompute `pair_suffixes` once per call from
`diff.contrasts` so `_build_diff_protein_json` can emit per-pair suffixed
keys for k≥3. For k=2 (legacy 2-group call) `diff.contrasts` is empty and
`pair_suffixes == String[]`, so no suffixed keys are emitted (2-group
byte-equality preserved).
"""
function _build_diff_results_json(diff::DifferentialResult)::String
    # precompute pair suffixes from declaration order.
    # For legacy 2-group, diff.contrasts is empty → pair_suffixes == String[].
    # For k≥3 (and k=2 NamedTuple-style with single contrast — `_aggregate_pairwise_results`
    # short-circuits and returns the per-pair DF verbatim with unsuffixed columns —
    # so an empty pair_suffixes is the right call for length(contrasts) <= 1 too).
    pair_suffixes = length(diff.contrasts) >= 2 ?
        ["_$(String(first(p)))_vs_$(String(last(p)))" for p in diff.contrasts] :
        String[]
    rows = String[]
    for row in eachrow(diff.results)
        push!(rows, _build_diff_protein_json(row, pair_suffixes))
    end
    return json_array(rows)
end

"""Serialize a single differential result row.

Accepts a `pair_suffixes::Vector{String}` second
arg. For k≥3 the caller passes the precomputed `_<a>_vs_<b>` suffixes in
declaration order; this function then emits per-pair suffixed numerical
keys (e.g. `bf_A_wt_vs_mut1`, `dbf_wt_vs_mut1`, `posterior_B_wt_vs_mut1`,
`delta_log2fc_wt_vs_mut1`, `classification_wt_vs_mut1`, `diff_PEP_wt_vs_mut1`,
`decision_risk_wt_vs_mut1`, …) so the JS pair-selector dropdown can drive
the result-table row builder.

For k=2 (legacy 2-group: `pair_suffixes == String[]`) the per-pair loop is
skipped entirely and the legacy `bf_A`/`bf_B`/`dbf`/`posterior_A`/…/`classification`
keys are emitted verbatim — 2-group byte-equality preserved.

The legacy fixed-pair keys ARE still emitted for k≥3 (via the existing
`_getf(:bf_A)` etc. fallbacks, which return `NaN`/`""` when the unsuffixed
column is absent from the wide DF). This keeps the JSON shape backward
compatible for any downstream consumer that does not yet read the suffixed
keys.

identifier columns (`uniprot_id`, `gene_name`,
`enriched_in`, `depleted_in`, `n_pairs_with_data`) land UNSUFFIXED on the
wide DF; we emit them unconditionally
under their bare names. `kgroup_class` was already emitted upstream by the
omnibus block.

`diff.contrasts` is the canonical source of pair labels
(already emitted to `D.meta.contrasts` by `_build_diff_meta_json`). The JS
pair-selector reads `D.meta.contrasts` and builds the same `_<a>_vs_<b>`
strings to drive the lookup into `r['bf_A_' + activePair]` etc.
"""
function _build_diff_protein_json(row, pair_suffixes::Vector{String} = String[])::String
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

    # canonical lowercase columns + bb_codriven flags
    diff_pep_lc = hasproperty(row, :differential_pep) ?
                      (_sf(row.differential_pep) === nothing ? NaN : _sf(row.differential_pep)) :
                      NaN
    pep_g = hasproperty(row, :pep_gained)        ? (_sf(row.pep_gained)        === nothing ? NaN : _sf(row.pep_gained))        : NaN
    pep_r = hasproperty(row, :pep_reduced)       ? (_sf(row.pep_reduced)       === nothing ? NaN : _sf(row.pep_reduced))       : NaN
    pep_u = hasproperty(row, :pep_unchanged)     ? (_sf(row.pep_unchanged)     === nothing ? NaN : _sf(row.pep_unchanged))     : NaN
    pep_bn= hasproperty(row, :pep_both_negative) ? (_sf(row.pep_both_negative) === nothing ? NaN : _sf(row.pep_both_negative)) : NaN
    bb_A = hasproperty(row, :bb_codriven_A) ?
               (ismissing(row.bb_codriven_A) ? false : Bool(row.bb_codriven_A)) : false
    bb_B = hasproperty(row, :bb_codriven_B) ?
               (ismissing(row.bb_codriven_B) ? false : Bool(row.bb_codriven_B)) : false

    # k-group orchestrator wiring: the wide_df
    # from `_aggregate_pairwise_results` carries per-pair columns suffixed
    # `_<a>_vs_<b>` (e.g. `bf_A_wt_vs_mut1`, `dbf_wt_vs_mut1`). For k≥3 we emit
    # the suffixed keys alongside the legacy fixed-pair keys (`bf_A`/`bf_B`/`dbf`/
    # `posterior_A`/`posterior_B`/`delta_log2fc`/`classification`); the legacy keys
    # remain NaN/"" for k≥3 via the `_getf` / hasproperty guards and are consumed
    # only by the k=2 JS path. The JS pair-selector dropdown
    # reads the suffixed keys via `r['bf_A_' + activePair]`, etc.
    #
    # For k=2 the suffixed keys are absent (`pair_suffixes::Vector{String}` empty
    # — `_aggregate_pairwise_results` short-circuits when `n_contrasts == 1` and
    # returns the per-pair DF verbatim with unsuffixed columns; byte-equality
    # preserved).
    _getf(sym) = hasproperty(row, sym) ?
                 (_sf(getproperty(row, sym)) === nothing ? NaN : _sf(getproperty(row, sym))) :
                 NaN

    # tiny helper for unsuffixed identifier columns. Treats
    # `missing` and absent properties as the default sentinel.
    _get_sym(sym, default) = hasproperty(row, sym) ?
        (ismissing(getproperty(row, sym)) ? default : getproperty(row, sym)) :
        default

    # serialise a `Vector{Symbol}` / `Vector{String}` /
    # `Missing` to a JSON string-array (or `null` for missing/empty).
    _json_str_array(v) = (v === nothing || ismissing(v)) ?
        "null" :
        json_array([json_string(string(s)) for s in v])

    # Build the JSON pair list incrementally so we can append k≥3-only per-pair
    # suffixed keys + identifier columns without re-flattening
    # the existing 48-key legacy block.
    kv = Pair{String, String}[
        "protein"               => json_string(string(row.Protein)),
        "bf_A"                  => json_number(_getf(:bf_A)),
        "bf_B"                  => json_number(_getf(:bf_B)),
        "dbf"                   => json_number(_getf(:dbf)),
        "log10_dbf"             => json_number(_getf(:log10_dbf)),
        "delta_log2fc"          => json_number(_getf(:delta_log2fc)),
        "posterior_A"           => json_number(_getf(:posterior_A)),
        "posterior_B"           => json_number(_getf(:posterior_B)),
        "delta_posterior"       => json_number(_getf(:delta_posterior)),
        "BFDR_A"                => json_number(_getf(:BFDR_A)),
        "BFDR_B"                => json_number(_getf(:BFDR_B)),
        "PEP_A"                 => json_number(_getf(:PEP_A)),
        "PEP_B"                 => json_number(_getf(:PEP_B)),
        "log2fc_A"              => json_number(_getf(:log2fc_A)),
        "log2fc_B"              => json_number(_getf(:log2fc_B)),
        "differential_posterior"=> json_number(_getf(:differential_posterior)),
        "differential_BFDR"     => json_number(_getf(:differential_BFDR)),
        # canonical lowercase `differential_pep`. `diff_PEP` mirror is
        # retained for back-compat consumers (volcano JS uses the new column).
        "differential_pep"      => json_number(diff_pep_lc),
        "diff_PEP"              => json_number(_getf(:diff_PEP)),
        # four γ class-conditional PEPs (consumed by α/γ volcano toggle)
        "pep_gained"            => json_number(pep_g),
        "pep_reduced"           => json_number(pep_r),
        "pep_unchanged"         => json_number(pep_u),
        "pep_both_negative"     => json_number(pep_bn),
        "classification"        => json_string(cls),
        "diagnostic_flag_A"     => json_string(diag_A),
        "diagnostic_flag_B"     => json_string(diag_B),
        "sensitivity_range_A"   => json_number(sens_A),
        "sensitivity_range_B"   => json_number(sens_B),
        # per-side BB×MNAR codriven flags (warning icons)
        "bb_codriven_A"         => json_bool(bb_A),
        "bb_codriven_B"         => json_bool(bb_B),
        # omnibus columns (k-group; absent for non-k-group rows)
        "posterior_omnibus"        => json_number(_getf(:posterior_omnibus)),
        "differential_BFDR_omnibus" => json_number(_getf(:differential_BFDR_omnibus)),
        "kgroup_class"             => json_string(hasproperty(row, :kgroup_class) && !ismissing(row.kgroup_class) ? string(row.kgroup_class) : ""),
        # Decision Risk columns (always populated by the pipeline wiring)
        "optimal_call"        => json_symbol_or_string(hasproperty(row, :optimal_call) && !ismissing(row.optimal_call) ? row.optimal_call : :unchanged),
        "decision_risk"       => json_number_nan_safe(hasproperty(row, :decision_risk) ? row.decision_risk : NaN),
        "risk_gained"         => json_number_nan_safe(hasproperty(row, :risk_gained) ? row.risk_gained : NaN),
        "risk_reduced"        => json_number_nan_safe(hasproperty(row, :risk_reduced) ? row.risk_reduced : NaN),
        "risk_unchanged"      => json_number_nan_safe(hasproperty(row, :risk_unchanged) ? row.risk_unchanged : NaN),
        "risk_both_negative"  => json_number_nan_safe(hasproperty(row, :risk_both_negative) ? row.risk_both_negative : NaN),
        "loss_matrix_default" => json_bool(hasproperty(row, :loss_matrix_default) ? Bool(row.loss_matrix_default) : true),
        # wide-table k-group aggregates (only present for k>=3)
        "decision_risk_min"   => hasproperty(row, :decision_risk_min) ? json_number_nan_safe(row.decision_risk_min) : "null",
        "optimal_call_min"    => hasproperty(row, :optimal_call_min) && !ismissing(row.optimal_call_min) ? json_symbol_or_string(row.optimal_call_min) : "null",
    ]

    # ─────────────────────────────────────────────────────────────────────
    # Unsuffixed identifier columns (k≥3 only on the wide
    # DF; absent on legacy 2-group output. hasproperty-guards make this safe
    # for all k).
    # ─────────────────────────────────────────────────────────────────────
    push!(kv, "uniprot_id"        => json_string(string(_get_sym(:uniprot_id, ""))))
    push!(kv, "gene_name"         => json_string(string(_get_sym(:gene_name, ""))))
    push!(kv, "enriched_in"       => _json_str_array(_get_sym(:enriched_in, nothing)))
    push!(kv, "depleted_in"       => _json_str_array(_get_sym(:depleted_in, nothing)))
    push!(kv, "n_pairs_with_data" => json_number(_getf(:n_pairs_with_data)))

    # ─────────────────────────────────────────────────────────────────────
    # Per-pair suffixed numerical keys (k≥3 only).
    # The JS pair-selector dropdown reads these via `r['bf_A_' + activePair]`,
    # `r['dbf_' + activePair]`, etc. (where `activePair` is the `_vs_`-joined
    # condition pair from `D.meta.contrasts`).
    #
    # The per-pair DF columns (from legacy 2-group `differential_analysis` recursion)
    # are: bf_A, bf_B, dbf, log10_dbf, posterior_A, posterior_B, delta_posterior,
    # delta_log2fc, log2fc_A, log2fc_B, differential_BFDR, differential_pep, diff_PEP,
    # classification, decision_risk, optimal_call, risk_{gained,reduced,unchanged,
    # both_negative}, loss_matrix_default. `_aggregate_pairwise_results`
    # suffixes ALL of these (non-ID, non-Protein) by `_<a>_vs_<b>`.
    # ─────────────────────────────────────────────────────────────────────
    for suffix in pair_suffixes
        push!(kv, "bf_A$(suffix)"                  => json_number(_getf(Symbol("bf_A$(suffix)"))))
        push!(kv, "bf_B$(suffix)"                  => json_number(_getf(Symbol("bf_B$(suffix)"))))
        push!(kv, "dbf$(suffix)"                   => json_number(_getf(Symbol("dbf$(suffix)"))))
        push!(kv, "log10_dbf$(suffix)"             => json_number(_getf(Symbol("log10_dbf$(suffix)"))))
        push!(kv, "delta_log2fc$(suffix)"          => json_number(_getf(Symbol("delta_log2fc$(suffix)"))))
        push!(kv, "log2fc_A$(suffix)"              => json_number(_getf(Symbol("log2fc_A$(suffix)"))))
        push!(kv, "log2fc_B$(suffix)"              => json_number(_getf(Symbol("log2fc_B$(suffix)"))))
        push!(kv, "posterior_A$(suffix)"           => json_number(_getf(Symbol("posterior_A$(suffix)"))))
        push!(kv, "posterior_B$(suffix)"           => json_number(_getf(Symbol("posterior_B$(suffix)"))))
        # PEP_A/B per-pair keys (paired with posterior_A/B above).
        push!(kv, "PEP_A$(suffix)"                 => json_number(_getf(Symbol("PEP_A$(suffix)"))))
        push!(kv, "PEP_B$(suffix)"                 => json_number(_getf(Symbol("PEP_B$(suffix)"))))
        # report fix — per-condition within-pair BFDR_A/B. These were
        # never emitted per-pair (only the unsuffixed k=2 keys + differential_BFDR),
        # so for k≥3 the BFDR(A)/(B) table columns rendered "—" AND classifyProtein
        # (which reads BFDR_A/B for per-condition interactor status) collapsed every
        # protein to BOTH_NEGATIVE + greyed the volcano. The source columns exist in
        # the wide table (BFDR_A_<a>_vs_<b>); they just weren't surfaced to the JSON.
        push!(kv, "BFDR_A$(suffix)"                => json_number(_getf(Symbol("BFDR_A$(suffix)"))))
        push!(kv, "BFDR_B$(suffix)"                => json_number(_getf(Symbol("BFDR_B$(suffix)"))))
        push!(kv, "delta_posterior$(suffix)"       => json_number(_getf(Symbol("delta_posterior$(suffix)"))))
        push!(kv, "differential_BFDR$(suffix)"     => json_number(_getf(Symbol("differential_BFDR$(suffix)"))))
        push!(kv, "differential_pep$(suffix)"      => json_number(_getf(Symbol("differential_pep$(suffix)"))))
        push!(kv, "diff_PEP$(suffix)"              => json_number(_getf(Symbol("diff_PEP$(suffix)"))))
        # classification + optimal_call → Symbol/String; Missing → "".
        cls_sym  = Symbol("classification$(suffix)")
        opt_sym  = Symbol("optimal_call$(suffix)")
        push!(kv, "classification$(suffix)"        => json_string(
            (hasproperty(row, cls_sym) && !ismissing(getproperty(row, cls_sym))) ?
                string(getproperty(row, cls_sym)) : ""))
        push!(kv, "optimal_call$(suffix)"          => json_string(
            (hasproperty(row, opt_sym) && !ismissing(getproperty(row, opt_sym))) ?
                string(getproperty(row, opt_sym)) : ""))
        # Decision Risk columns per pair.
        push!(kv, "decision_risk$(suffix)"         => json_number(_getf(Symbol("decision_risk$(suffix)"))))
        push!(kv, "risk_gained$(suffix)"           => json_number(_getf(Symbol("risk_gained$(suffix)"))))
        push!(kv, "risk_reduced$(suffix)"          => json_number(_getf(Symbol("risk_reduced$(suffix)"))))
        push!(kv, "risk_unchanged$(suffix)"        => json_number(_getf(Symbol("risk_unchanged$(suffix)"))))
        push!(kv, "risk_both_negative$(suffix)"    => json_number(_getf(Symbol("risk_both_negative$(suffix)"))))
        # loss_matrix_default may be Missing for cells absent from the source pair-DF
        # (outerjoin widens Bool → Union{Missing, Bool}); default to true.
        lmd_sym = Symbol("loss_matrix_default$(suffix)")
        push!(kv, "loss_matrix_default$(suffix)"   => json_bool(
            (hasproperty(row, lmd_sym) && !ismissing(getproperty(row, lmd_sym))) ?
                Bool(getproperty(row, lmd_sym)) : true))
        # report fix — per-pair sensitivity / diagnostic / BB-codriven.
        # _aggregate_pairwise_results suffixes EVERY non-ID column, so these exist on
        # the wide DF (sensitivity_range_A_<a>_vs_<b>, diagnostic_flag_A_<...>,
        # bb_codriven_A_<...>) but were never surfaced to the JSON — so for k≥3 the
        # Sens/Diag table columns + BB icons rendered blank. The JS row-builder now
        # reads these pair-aware via _pairGet.
        push!(kv, "sensitivity_range_A$(suffix)"   => json_number(_getf(Symbol("sensitivity_range_A$(suffix)"))))
        push!(kv, "sensitivity_range_B$(suffix)"   => json_number(_getf(Symbol("sensitivity_range_B$(suffix)"))))
        dfa_sym = Symbol("diagnostic_flag_A$(suffix)"); dfb_sym = Symbol("diagnostic_flag_B$(suffix)")
        push!(kv, "diagnostic_flag_A$(suffix)"     => json_string(
            (hasproperty(row, dfa_sym) && !ismissing(getproperty(row, dfa_sym))) ?
                string(getproperty(row, dfa_sym)) : ""))
        push!(kv, "diagnostic_flag_B$(suffix)"     => json_string(
            (hasproperty(row, dfb_sym) && !ismissing(getproperty(row, dfb_sym))) ?
                string(getproperty(row, dfb_sym)) : ""))
        bba_sym = Symbol("bb_codriven_A$(suffix)"); bbb_sym = Symbol("bb_codriven_B$(suffix)")
        push!(kv, "bb_codriven_A$(suffix)"         => json_bool(
            (hasproperty(row, bba_sym) && !ismissing(getproperty(row, bba_sym))) ?
                Bool(getproperty(row, bba_sym)) : false))
        push!(kv, "bb_codriven_B$(suffix)"         => json_bool(
            (hasproperty(row, bbb_sym) && !ismissing(getproperty(row, bbb_sym))) ?
                Bool(getproperty(row, bbb_sym)) : false))
        # report fix — per-pair γ class-conditional PEPs so the volcano
        # α/γ toggle differs for k≥3 (bare pep_* are NaN on the wide DF → γ collapsed
        # to the 0.25 opacity floor, making α and γ visually identical).
        push!(kv, "pep_gained$(suffix)"            => json_number(_getf(Symbol("pep_gained$(suffix)"))))
        push!(kv, "pep_reduced$(suffix)"           => json_number(_getf(Symbol("pep_reduced$(suffix)"))))
        push!(kv, "pep_unchanged$(suffix)"         => json_number(_getf(Symbol("pep_unchanged$(suffix)"))))
        push!(kv, "pep_both_negative$(suffix)"     => json_number(_getf(Symbol("pep_both_negative$(suffix)"))))
    end

    parts = [json_string(k) * ":" * v for (k, v) in kv]
    return "{" * join(parts, ",") * "}"
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
# differential per-condition tab JSON helpers
# ---------------------------------------------------------------------------
#
# Six builders feed differential_report.html's tabs. Five iterate over
# `diff.analyses` (the per-condition reuse pattern); the
# sixth (`_build_diff_dbf_diagnostics_json`) is pure DataFrame computation
# from `diff.results.dbf_diagnostic`.
#
# Pitfall guards:
# - Pitfall 1: _build_diff_calibration_json calls _build_simulation_json
#   (the SimulationResult flavour at simulation.jl:851), NOT the diag-flavor
#   _build_calibration_json(diag) at report_generator.jl:1729.
# - Pitfall 3: _build_diff_mixture_json passes _extract_copula_df(ar) (stripped
#   posteriors) to _build_mixture_model_json, NOT ar.copula_results directly.
# - Pitfall 4: the prior
#   `ar.config === nothing && continue` skip is replaced by a 4-level cascade
#   (in-memory CONFIG -> sidecar JSON -> methods.md -> placeholder card).
#   Path-based ARs from older caches still degrade gracefully via the
#   sidecar / markdown / placeholder branches instead of being silently
#   dropped from the per-condition payload. See `_build_diff_methods_json`
#   below + helpers in `methods_generator.jl`.
#
# Defence-in-depth: every builder that reaches into `diff.results` for the
# `dbf_diagnostic` column ALSO `hasproperty`-guards the column so that hand-built
# test fixtures (e.g. test_report.jl:458, which lacks the column) do not regress.

"""
    _build_diff_calibration_json(diff::DifferentialResult) -> String

k Platt panels (one per condition) + Δ-overlay (k=2 only).
Pitfall 1: per-panel data is built from `_build_simulation_json` (SimulationResult flavour)
NOT `_build_calibration_json(diag)` (Diagnostics flavour).

Returns `"null"` when `diff.analyses === nothing`.
"""
function _build_diff_calibration_json(diff::DifferentialResult)::String
    diff.analyses === nothing && return "null"
    labels = condition_labels(diff)
    panels = String[]
    for (i, ar) in enumerate(diff.analyses)
        sim = ar.simulation_result  # wired this onto AR
        sim === nothing && continue
        per_panel = _build_simulation_json(sim)
        per_panel == "null" && continue
        push!(panels, json_object(
            "label" => json_string(labels[i]),
            "data"  => per_panel,
        ))
    end
    isempty(panels) && return "null"
    overlay = (length(diff.analyses) == 2) ? _build_diff_calibration_overlay(diff, labels) : "null"
    return json_object(
        "panels"  => json_array(panels),
        "overlay" => overlay,
    )
end

"""
    _build_diff_calibration_overlay(diff, labels) -> String

emit the per-condition Platt curve points so the differential template
can synthesise the three-trace Δ-band (A curve, B curve, fill='tonexty' shading).
Returns `"null"` when either condition's calibration model is absent.
"""
function _build_diff_calibration_overlay(diff::DifferentialResult, labels::Vector{String})::String
    diff.analyses === nothing && return "null"
    length(diff.analyses) == 2 || return "null"
    sim_A = diff.analyses[1].simulation_result
    sim_B = diff.analyses[2].simulation_result
    (sim_A === nothing || sim_B === nothing) && return "null"
    curve_A = _extract_diff_platt_curve(sim_A)
    curve_B = _extract_diff_platt_curve(sim_B)
    (curve_A === nothing || curve_B === nothing) && return "null"
    return json_object(
        "labels" => json_array([json_string(labels[1]), json_string(labels[2])]),
        "curves" => json_array([curve_A, curve_B]),
    )
end

"""
    _extract_diff_platt_curve(sim) -> Union{String, Nothing}

Sample the Platt mapping `calibrated = logistic(a * logit(raw) + b)`
on the open interval (0.01, 0.99) (avoiding logit(0) / logit(1) singularities).
Verified against `CalibrationModel` schema (src/simulation/simulation.jl:114-119):
fields `a::Float64`, `b::Float64`, `n_training::Int`, `converged::Bool`.

Returns the JSON object string `{"x":[...],"y":[...]}` or `nothing` when the
calibration model is absent.
"""
function _extract_diff_platt_curve(sim)::Union{String, Nothing}
    cal = hasproperty(sim, :calibration_model) ? sim.calibration_model : nothing
    cal === nothing && return nothing
    (hasproperty(cal, :a) && hasproperty(cal, :b)) || return nothing
    _logit(p)   = log(p / (1 - p))
    _sigmoid(z) = 1 / (1 + exp(-z))
    xs = collect(0.01:0.01:0.99)
    ys = [_sigmoid(cal.a * _logit(x) + cal.b) for x in xs]
    return json_object(
        "x" => json_array([json_number(x) for x in xs]),
        "y" => json_array([json_number(y) for y in ys]),
    )
end

"""
    _build_diff_sensitivity_json(diff::DifferentialResult) -> String

k traffic-light bars (per condition) + top-N stability-change
strip ranked by per-protein A-vs-B classification swing.
"""
function _build_diff_sensitivity_json(diff::DifferentialResult)::String
    diff.analyses === nothing && return "null"
    labels = condition_labels(diff)
    panels = String[]
    for (i, ar) in enumerate(diff.analyses)
        sens = hasproperty(ar, :sensitivity) ? ar.sensitivity : nothing
        per_panel = _build_sensitivity_json(sens)
        per_panel == "null" && continue
        push!(panels, json_object(
            "label" => json_string(labels[i]),
            "data"  => per_panel,
        ))
    end
    # read top_n_stability from DifferentialConfig (default 20).
    top_n = hasproperty(diff.config, :top_n_stability) ?
                Int(diff.config.top_n_stability) : 20
    strip = _build_diff_stability_change_strip(diff, top_n)
    return json_object(
        "panels"          => json_array(panels),
        "stability_strip" => strip,
    )
end

"""
    _build_diff_stability_change_strip(diff, N) -> String

Per-protein stability category swing across conditions. Generalised to k≥2 in
for k conditions the swing is true iff at least one pairwise
mismatch exists across the k categories. Ranked by Boolean swing then by summed
`frac_P_gt_0_5` distance from 0/1 (most in-flux proteins first); top-N kept.

The traffic-light category for a protein in condition `i` is derived inline from
the condition's `SensitivityResult.classification_stability` DataFrame:
- "fragile" if `threshold_crossing_0_5` is true
- "robust"  if `frac_P_gt_0_5 ∈ {0.0, 1.0}`
- "sensitive" otherwise
This mirrors `predictive_checks.jl:843-854`.

Returns a JSON array of `{protein, categories: [cat_1, cat_2, …, cat_k]}` rows
in `condition_labels(diff)` declaration order. For k=2 the emitted shape is
byte-equal to the legacy 2-group output (length-2 `categories` array)
modulo column ordering — the protein name set and per-row category labels are
preserved.

Returns `"[]"` when fewer than 2 conditions are available OR when ANY condition's
sensitivity result is missing.
"""
function _build_diff_stability_change_strip(diff::DifferentialResult, N::Int)::String
    diff.analyses === nothing && return "[]"
    k = length(diff.analyses)
    k >= 2 || return "[]"

    # derive a per-protein category for each of the k conditions.
    # Returns "[]" if ANY condition is missing sensitivity data — the strip is
    # only meaningful when every condition contributes.
    cats_per_cond = Vector{Dict{String, NamedTuple{(:category, :distance), Tuple{String, Float64}}}}(undef, k)
    for i in 1:k
        sens_i = hasproperty(diff.analyses[i], :sensitivity) ? diff.analyses[i].sensitivity : nothing
        sens_i === nothing && return "[]"
        ci = _stability_categories_per_protein(sens_i)
        isempty(ci) && return "[]"
        cats_per_cond[i] = ci
    end

    # Inner-join on protein name across ALL k conditions.
    shared       = String[]
    cats_by_row  = Vector{Vector{String}}()   # row x condition
    dist_by_row  = Vector{Vector{Float64}}()  # row x condition
    for (prot, _) in cats_per_cond[1]
        all(haskey(cats_per_cond[i], prot) for i in 2:k) || continue
        push!(shared, prot)
        push!(cats_by_row, [cats_per_cond[i][prot].category for i in 1:k])
        push!(dist_by_row, [cats_per_cond[i][prot].distance for i in 1:k])
    end
    isempty(shared) && return "[]"

    # Swing per protein: true iff at least one pair of conditions disagrees.
    swing = [length(unique(cats_by_row[r])) > 1 for r in eachindex(shared)]
    # Combined distance summed across k conditions.
    combined_dist = [sum(dist_by_row[r]) for r in eachindex(shared)]
    order = sortperm(collect(zip(swing, combined_dist)); rev=true,
                     by = t -> (t[1] ? 1 : 0, t[2]))
    keep = order[1:min(N, length(order))]

    rows = String[]
    for r in keep
        push!(rows, json_object(
            "protein"    => json_string(shared[r]),
            "categories" => json_array([json_string(c) for c in cats_by_row[r]]),
            "swing"      => swing[r] ? "true" : "false",
        ))
    end
    return json_array(rows)
end

"""
    _stability_categories_per_protein(sens::SensitivityResult)
        -> Dict{String, NamedTuple{(:category, :distance), Tuple{String, Float64}}}

Internal helper: per-protein traffic-light category derived from
`sens.classification_stability` columns `frac_P_gt_0_5` + `threshold_crossing_0_5`.
`distance` = `min(frac_P_gt_0_5, 1 - frac_P_gt_0_5)` so 0.0 = at extreme, 0.5 = centred.
"""
function _stability_categories_per_protein(sens)
    out = Dict{String, NamedTuple{(:category, :distance), Tuple{String, Float64}}}()
    cs = sens.classification_stability
    (hasproperty(cs, :Protein) && hasproperty(cs, :frac_P_gt_0_5)) || return out
    has_cross = hasproperty(cs, :threshold_crossing_0_5)
    n = nrow(cs)
    for i in 1:n
        prot = string(cs.Protein[i])
        f = cs.frac_P_gt_0_5[i]
        ismissing(f) && continue
        crossed = has_cross ? cs.threshold_crossing_0_5[i] : false
        cat = if crossed === true
            "fragile"
        elseif f == 1.0 || f == 0.0
            "robust"
        else
            "sensitive"
        end
        dist = min(f, 1.0 - f)
        out[prot] = (category = cat, distance = dist)
    end
    return out
end

"""
    _build_diff_qc_json(diff::DifferentialResult) -> String

k Data-Quality panels + shared-proteins log10 mean-intensity
scatter (k=2 only).
"""
function _build_diff_qc_json(diff::DifferentialResult)::String
    diff.analyses === nothing && return "null"
    labels = condition_labels(diff)
    panels = String[]
    for (i, ar) in enumerate(diff.analyses)
        per_panel = _build_qc_json(ar)
        per_panel == "null" && continue
        push!(panels, json_object(
            "label" => json_string(labels[i]),
            "data"  => per_panel,
        ))
    end
    scatter = (length(diff.analyses) == 2) ? _build_diff_shared_intensity_scatter(diff) : "null"
    return json_object(
        "panels"                   => json_array(panels),
        "shared_intensity_scatter" => scatter,
    )
end

"""
    _build_diff_shared_intensity_scatter(diff) -> String

§ Specific Ideas. Scatter `log10(mean_intensity_A + 1)` vs
`log10(mean_intensity_B + 1)` for shared proteins. When `mean_intensity` columns are
absent (current schema), fall back to `mean_log2FC` as the per-protein log-intensity
proxy (a standard column on copula_results).

Returns `{x, y, proteins, axis_max}` for client-side Plotly scatter with a `y=x`
diagonal reference. Returns `"null"` when neither column is present on either condition.
"""
function _build_diff_shared_intensity_scatter(diff::DifferentialResult)::String
    diff.analyses === nothing && return "null"
    length(diff.analyses) == 2 || return "null"
    df_A = diff.analyses[1].copula_results
    df_B = diff.analyses[2].copula_results
    col = if hasproperty(df_A, :mean_intensity) && hasproperty(df_B, :mean_intensity)
        :mean_intensity
    elseif hasproperty(df_A, :mean_log2FC) && hasproperty(df_B, :mean_log2FC)
        :mean_log2FC
    else
        return "null"
    end
    (hasproperty(df_A, :Protein) && hasproperty(df_B, :Protein)) || return "null"

    # Inner-join on Protein
    map_B = Dict{String, Float64}()
    for i in 1:nrow(df_B)
        v = df_B[i, col]
        ismissing(v) && continue
        map_B[string(df_B.Protein[i])] = Float64(v)
    end

    xs = Float64[]
    ys = Float64[]
    prots = String[]
    for i in 1:nrow(df_A)
        prot = string(df_A.Protein[i])
        haskey(map_B, prot) || continue
        v = df_A[i, col]
        ismissing(v) && continue
        # log10(x + 1) requires positive values; mean_log2FC can be negative — apply
        # only for `mean_intensity`, otherwise pass through untransformed.
        x_val = col === :mean_intensity ? log10(Float64(v) + 1.0) : Float64(v)
        y_val = col === :mean_intensity ? log10(map_B[prot] + 1.0) : map_B[prot]
        push!(xs, x_val)
        push!(ys, y_val)
        push!(prots, prot)
    end
    isempty(xs) && return "null"

    axis_max = maximum(vcat(xs, ys))
    axis_min = minimum(vcat(xs, ys))
    return json_object(
        "x"        => json_array([json_number(v) for v in xs]),
        "y"        => json_array([json_number(v) for v in ys]),
        "proteins" => json_array([json_string(p) for p in prots]),
        "axis_max" => json_number(axis_max),
        "axis_min" => json_number(axis_min),
        "column"   => json_string(string(col)),
    )
end

"""
    _build_diff_mixture_json(diff::DifferentialResult) -> String

k mixture scatters + 3×3 transition matrix (k=2 only).

Pitfall 3: per-condition mixture scatters use `_extract_copula_df(ar)` (stripped
posteriors) for consistency with the differential transition matrix (also computed
from stripped posteriors).
"""
function _build_diff_mixture_json(diff::DifferentialResult)::String
    diff.analyses === nothing && return "null"
    labels = condition_labels(diff)
    panels = String[]
    for (i, ar) in enumerate(diff.analyses)
        stripped = try
            _extract_copula_df(ar)
        catch
            nothing
        end
        stripped === nothing && continue
        per_panel = _build_mixture_model_json(ar, stripped)
        per_panel == "null" && continue
        push!(panels, json_object(
            "label" => json_string(labels[i]),
            "data"  => per_panel,
        ))
    end
    transition = (length(diff.analyses) == 2) ? _build_diff_transition_matrix(diff) : "null"
    return json_object(
        "panels"            => json_array(panels),
        "transition_matrix" => transition,
    )
end

"""
    _build_diff_transition_matrix(diff) -> String

3×3 cell counts of (cat_A, cat_B) latent class pairs.
- Rows (top-down): `[H1, Agnostic, H0]` (interaction first)
- Cols (L-to-R):   `[H0, Agnostic, H1]`
Component label per protein = argmax over the corresponding `LatentClassResult.responsibilities`
row (responsibilities are N×3 in column order [background, agnostic, interaction];
verified in src/diagnostics/copula_diagnostics.jl:168 and src/reports/report_generator.jl:2425).

Hover content per cell: top-5 protein names by `|log10_dbf|` from `diff.results`.
Returns `"null"` when either condition's `latent_class_result` is missing or its
`responsibilities` matrix is `nothing`.
"""
function _build_diff_transition_matrix(diff::DifferentialResult)::String
    diff.analyses === nothing && return "null"
    length(diff.analyses) == 2 || return "null"

    lcr_A = hasproperty(diff.analyses[1], :latent_class_result) ?
                diff.analyses[1].latent_class_result : nothing
    lcr_B = hasproperty(diff.analyses[2], :latent_class_result) ?
                diff.analyses[2].latent_class_result : nothing
    (lcr_A === nothing || lcr_B === nothing) && return "null"
    (lcr_A.responsibilities === nothing || lcr_B.responsibilities === nothing) && return "null"

    # Per-protein component index (1=background/H0, 2=agnostic, 3=interaction/H1).
    # responsibilities is N×3 — rows = proteins, cols = components.
    cat_idx_A = _per_protein_components(diff.analyses[1])
    cat_idx_B = _per_protein_components(diff.analyses[2])
    (cat_idx_A === nothing || cat_idx_B === nothing) && return "null"

    # Build per-protein lookup keyed on diff.results.Protein.
    # diff.results.Protein is the canonical row order; need to map each protein
    # to its component index in each condition.
    # cat_idx_A is a Dict{String, Int} keyed on protein name.
    df = diff.results
    hasproperty(df, :Protein) || return "null"

    # 3×3 count grid. We use a working layout with rows = A class (H0=1, Agn=2, H1=3),
    # cols = B class (H0=1, Agn=2, H1=3) — then permute rows for the user-facing
    # display order [H1, Agnostic, H0] when emitting.
    counts = zeros(Int, 3, 3)
    # Per-cell top-5 protein lists by |log10_dbf|
    cell_proteins = [String[] for _ in 1:3, _ in 1:3]
    cell_scores   = [Float64[] for _ in 1:3, _ in 1:3]

    has_l10 = hasproperty(df, :log10_dbf)
    for i in 1:nrow(df)
        prot = string(df.Protein[i])
        ia = get(cat_idx_A, prot, 0)
        ib = get(cat_idx_B, prot, 0)
        (ia == 0 || ib == 0) && continue
        counts[ia, ib] += 1
        score = if has_l10
            v = df.log10_dbf[i]
            ismissing(v) ? 0.0 : abs(Float64(v))
        else
            0.0
        end
        push!(cell_proteins[ia, ib], prot)
        push!(cell_scores[ia, ib], score)
    end

    # User-facing row order: [H1, Agnostic, H0]; col order [H0, Agnostic, H1]
    row_perm = [3, 2, 1]   # source idx for output rows top→bottom
    col_perm = [1, 2, 3]   # cols already H0→Agn→H1
    z = [counts[row_perm[r], col_perm[c]] for r in 1:3, c in 1:3]

    # Per-cell top-5 hover lists, applying the same permutation
    hover_jsons = String[]
    for r in 1:3, c in 1:3
        ra = row_perm[r]; ca = col_perm[c]
        prots = cell_proteins[ra, ca]
        scores = cell_scores[ra, ca]
        if isempty(prots)
            push!(hover_jsons, json_array(String[]))
            continue
        end
        order = sortperm(scores; rev=true)
        keep = order[1:min(5, length(order))]
        push!(hover_jsons, json_array([json_string(prots[k]) for k in keep]))
    end

    # Flatten z 3×3 to row-major nested arrays for JSON
    z_rows = String[]
    for r in 1:3
        push!(z_rows, json_array([json_number(z[r, c]) for c in 1:3]))
    end
    hover_rows = String[]
    for r in 1:3
        row_cells = String[]
        for c in 1:3
            push!(row_cells, hover_jsons[(r - 1) * 3 + c])
        end
        push!(hover_rows, json_array(row_cells))
    end

    return json_object(
        "row_labels"      => json_array([json_string("H1"), json_string("Agnostic"), json_string("H0")]),
        "col_labels"      => json_array([json_string("H0"), json_string("Agnostic"), json_string("H1")]),
        "z"               => json_array(z_rows),
        "hover_proteins"  => json_array(hover_rows),
    )
end

"""
    _build_per_condition_json(diff::DifferentialResult) -> String

Build the `per_condition` top-level
JSON dict for the differential report sidecar. Keyed by
`String(condition_label)` in `condition_labels(diff)` declaration order.

Each value carries the four tab payloads consumed by the per-condition
dropdowns (Calibration / Sensitivity + Mixture / Data
Quality): a minimum of four
keys, NOT the full per-condition `report.html` shape — keeps sidecar
size bounded for k=4 (~4× minimal payload, not 4× full report).

For k=2 the dict still carries both entries (only the JS
dropdown UI is suppressed; the data path stays uniform). Returns `"null"`
when `diff.analyses === nothing` (matches sibling diff builders).
"""
function _build_per_condition_json(diff::DifferentialResult)::String
    diff.analyses === nothing && return "null"
    isempty(diff.analyses) && return "null"
    labels = condition_labels(diff)
    pairs = String[]
    for (i, ar) in enumerate(diff.analyses)
        # Calibration: mirror _build_diff_calibration_json per-panel source
        sim = hasproperty(ar, :simulation_result) ? ar.simulation_result : nothing
        calib_json = sim === nothing ? "null" : _build_simulation_json(sim)

        # Sensitivity: mirror _build_diff_sensitivity_json per-panel source
        sens = hasproperty(ar, :sensitivity) ? ar.sensitivity : nothing
        sens_json = _build_sensitivity_json(sens)

        # Mixture: mirror _build_diff_mixture_json per-panel source (Pitfall 3 —
        # pass stripped copula_df so the scatter is consistent with diff transition).
        mixture_json = try
            stripped = _extract_copula_df(ar)
            stripped === nothing ? "null" : _build_mixture_model_json(ar, stripped)
        catch
            "null"
        end

        # Data Quality: mirror _build_diff_qc_json per-panel source
        qc_json = _build_qc_json(ar)

        # DNN Prior payload per condition (data is per-condition only).
        # Defensive try/catch: the per-condition wrap must NEVER crash even
        # if one sub-builder errors. The fallback "null" is the JSON-literal null string and the
        # differential template's `initDiffDnnPriorTab()` handles the null/empty case gracefully.
        # _config_from_ar does not exist as a helper — we pass `ar.config` when present (mirrors
        # the Methods cascade Level-1 pattern at line ~2523); the helper accepts `nothing` because
        # its body never consumes config (see `_build_dnn_prior_json` docstring).
        dnn_prior_json = try
            stripped = _extract_copula_df(ar)
            if stripped === nothing
                "null"
            else
                ar_config = (hasproperty(ar, :config) && ar.config !== nothing) ? ar.config : nothing
                _build_dnn_prior_json(stripped, ar_config)
            end
        catch
            "null"
        end

        inner = json_object(
            "calibration" => calib_json,
            "sensitivity" => sens_json,
            "mixture"     => mixture_json,
            "qc"          => qc_json,
            "dnn_prior"   => dnn_prior_json,
        )
        push!(pairs, json_string(String(labels[i])) * ":" * inner)
    end
    return "{" * join(pairs, ",") * "}"
end

"""
    _per_protein_components(ar) -> Union{Dict{String, Int}, Nothing}

Internal helper: derive per-protein component index (1=H0/background, 2=Agnostic,
3=H1/interaction) from an `AnalysisResult`'s `latent_class_result.responsibilities`
matrix. Keyed on the protein name from `ar.copula_results.Protein` (responsibilities
row order matches copula_results row order — verified in src/analysis/pipeline.jl).

Returns `nothing` when `latent_class_result` or `responsibilities` is missing.
"""
function _per_protein_components(ar)::Union{Dict{String, Int}, Nothing}
    lcr = hasproperty(ar, :latent_class_result) ? ar.latent_class_result : nothing
    lcr === nothing && return nothing
    lcr.responsibilities === nothing && return nothing
    df = ar.copula_results
    hasproperty(df, :Protein) || return nothing
    R = lcr.responsibilities
    n_rows = size(R, 1)
    out = Dict{String, Int}()
    # responsibilities is N×3 — argmax over cols gives the component index
    for i in 1:min(n_rows, nrow(df))
        prot = string(df.Protein[i])
        out[prot] = argmax(R[i, :])
    end
    return out
end

"""
    _build_diff_methods_json(diff::DifferentialResult) -> String

k per-condition Methods JSON + "Differential Analysis"
HTML subsection. The `differential_block_html` field is filled by the
`_methods_differential_block(diff)` helper — guarded with `isdefined` so this
builder can load even when that helper is absent.

supersedes the prior Pitfall 4 skip:
per-condition ARs without an in-memory `CONFIG` fall through a 4-level
cascade (sidecar JSON `methods.text` -> raw `methods.md` -> placeholder card)
so the Methods tab ALWAYS emits a non-empty entry per condition. The
placeholder branch is the only level that logs (`@warn ... maxlog=1`).
"""
function _build_diff_methods_json(diff::DifferentialResult)::String
    diff.analyses === nothing && return "null"
    labels = condition_labels(diff)
    per_condition = String[]
    for (i, ar) in enumerate(diff.analyses)
        # 4-level Methods cascade.
        # Replaces the prior `ar.config === nothing && continue` skip so that
        # every per-condition AR emits a non-empty Methods card. Helpers live
        # in methods_generator.jl; see _methods_placeholder_card for the
        # level-4 placeholder contract.
        status = hasproperty(ar, :metalearner_status) ? ar.metalearner_status : :loaded
        methods_card_html = if hasproperty(ar, :config) && ar.config !== nothing
            # Level 1 (existing path) — in-memory CONFIG.
            _build_methods_json(ar.config, ar.copula_results, status)
        elseif (sp = _try_locate_sidecar(ar); sp !== nothing) &&
               (parsed = _read_methods_from_sidecar_json(sp); parsed !== nothing)
            # Level 2 — sidecar JSON `methods.text` payload (prose only).
            json_object("html" => json_string(parsed))
        elseif (mp = _try_locate_methods_md(ar)) !== nothing
            # Level 3 — raw markdown rendered verbatim inside markdown-body.
            md_body = Base.read(mp, String)
            json_object("html" => json_string("<div class=\"card-body markdown-body\">" * md_body * "</div>"))
        else
            # Level 4 — placeholder card + single @warn per condition (maxlog=1).
            @warn "[Methods] no methods source for $(labels[i])" maxlog=1
            json_object("html" => json_string(_methods_placeholder_card(labels[i])))
        end
        push!(per_condition, json_object(
            "label" => json_string(labels[i]),
            "data"  => methods_card_html,
        ))
    end
    # supplies _methods_differential_block in methods_generator.jl. Guard
    # with isdefined so this builder loads even when that helper is absent.
    diff_block_html = isdefined(@__MODULE__, :_methods_differential_block) ?
                          _methods_differential_block(diff) : ""
    # Embeddings & Similarity Methods block — sourced from the first
    # AR's CONFIG (the per-condition runs share a CONFIG.embeddings_config snapshot).
    # Falls back to "" when no AR is available or `run_embeddings=false`.
    emb_block_html = if isdefined(@__MODULE__, :_methods_embeddings_block) &&
                        diff.analyses !== nothing && !isempty(diff.analyses) &&
                        hasproperty(first(diff.analyses), :config) &&
                        first(diff.analyses).config !== nothing
        try _methods_embeddings_block(first(diff.analyses).config) catch; "" end
    else
        ""
    end
    # DNN Prior + MC-Dropout Methods block
    # — sourced from the first AR's CONFIG. The helper always returns a
    # non-empty string in the loaded path (opt-out branch renders the explicit
    # "disabled" message + the five new column names). Empty string only when
    # `diff.analyses` is nothing/empty or the first AR carries no CONFIG (e.g.
    # rehydrated JLD2 fixture). Lives at `D.methods.dnn_prior_block_html` —
    # nested under `methods` to avoid naming conflict with the top-level
    # `D.dnn_prior` per-row payload.
    dnn_prior_block_html = if isdefined(@__MODULE__, :_methods_dnn_prior_block) &&
                              diff.analyses !== nothing && !isempty(diff.analyses) &&
                              hasproperty(first(diff.analyses), :config) &&
                              first(diff.analyses).config !== nothing
        try _methods_dnn_prior_block(first(diff.analyses).config) catch; "" end
    else
        ""
    end
    return json_object(
        "per_condition"           => json_array(per_condition),
        "differential_block_html" => json_string(diff_block_html),
        "embeddings_block_html"   => json_string(emb_block_html),
        "dnn_prior_block_html"    => json_string(dnn_prior_block_html),
    )
end

"""
    _build_diff_dbf_diagnostics_json(diff::DifferentialResult) -> String

Six panels (panel order):
  (1) histogram of `log10_dbf` + null Q-Q from UNCHANGED proteins
  (2) `log10_dbf` vs `delta_log2fc` scatter
  (3) per-component stacked bar (top-N=20 by `|log10_dbf|`)
  (4) saturation panel (counts + protein list where `dbf_diagnostic == :saturated`)
  (5) sub-model dBF disagreement scatter (`log10(dbf_em)` vs `log10(dbf_copula)`)
  (6) per-protein traffic-light summary (counts of each `dbf_diagnostic` Symbol)

Hard guard: hand-built test fixtures (e.g. test_report.jl:458) may
lack the `dbf_diagnostic` column. Return `"null"` early to keep that regression test
working with the orchestrator. The production pipeline always
populates the column.
"""
function _build_diff_dbf_diagnostics_json(diff::DifferentialResult)::String
    df = diff.results
    hasproperty(df, :dbf_diagnostic) || return "null"
    return json_object(
        "histogram_qq"          => _build_dbf_histogram_qq(df),
        "dbf_vs_delta"          => _build_dbf_vs_delta(df),
        "component_stack"       => _build_dbf_component_stack(df, 20),
        "saturation_panel"      => _build_dbf_saturation_panel(df),
        "submodel_disagreement" => _build_dbf_submodel_disagreement(df),
        "traffic_light"         => _build_dbf_traffic_light_summary(df),
    )
end

# ---------------------------------------------------------------------------
# Embeddings + Condition Similarity JSON builders
# ---------------------------------------------------------------------------
#
# Five helpers feed the new plot-cards in both report.html and differential_report.html
# All five gate on `=== nothing` and return the literal string `"null"` so
# the consumer JS reliably checks `if (!data) ...` short-circuit. Matrix serialisation
# uses row-major Vector{Vector{Float64}} (idiomatic for Plotly.js heatmap `z` input).
#
# Wiring:
#   _build_report_json  → embeddings_sample + embeddings_protein
#   _build_diff_json    → embeddings_sample + embeddings_protein + condition_matrix
#                        + condition_jaccard + condition_dendrogram

"""
    _build_sample_embedding_json(ar) -> String

JSON payload for sample-level embeddings. Returns `"null"` when `ar.embeddings` is
`nothing`. Keys:
  - pca: {x, y, var_explained, labels: {condition, replicate, experiment, protocol}}
  - umap: null | {x, y}
  - tsne: null | {x, y}
  - filter_level: String (Symbol stringified)
  - config: snapshot NamedTuple flattened to 6 keys
"""
function _build_sample_embedding_json(ar)::String
    (ar === nothing || !hasproperty(ar, :embeddings) || ar.embeddings === nothing) && return "null"
    e = ar.embeddings
    n_pca = size(e.sample_pca_scores, 1)
    pca_obj = json_object(
        "x"             => json_array([json_number(e.sample_pca_scores[i, 1]) for i in 1:n_pca]),
        "y"             => json_array([json_number(e.sample_pca_scores[i, 2]) for i in 1:n_pca]),
        "var_explained" => json_array([json_number(v) for v in e.sample_pca_var_explained]),
        "labels"        => json_object(
            "condition"  => json_array([json_string(string(s)) for s in e.sample_labels.condition]),
            "replicate"  => json_array([json_number(x) for x in e.sample_labels.replicate]),
            "experiment" => json_array([json_number(x) for x in e.sample_labels.experiment]),
            "protocol"   => json_array([json_number(x) for x in e.sample_labels.protocol]),
        ),
    )
    umap_obj = if e.sample_umap_coords === nothing
        "null"
    else
        n_u = size(e.sample_umap_coords, 1)
        json_object(
            "x" => json_array([json_number(e.sample_umap_coords[i, 1]) for i in 1:n_u]),
            "y" => json_array([json_number(e.sample_umap_coords[i, 2]) for i in 1:n_u]),
        )
    end
    tsne_obj = if e.sample_tsne_coords === nothing
        "null"
    else
        n_t = size(e.sample_tsne_coords, 1)
        json_object(
            "x" => json_array([json_number(e.sample_tsne_coords[i, 1]) for i in 1:n_t]),
            "y" => json_array([json_number(e.sample_tsne_coords[i, 2]) for i in 1:n_t]),
        )
    end
    cfg = e.config_snapshot
    config_obj = json_object(
        "method"        => json_string(string(cfg.method)),
        "seed"          => json_number(cfg.seed),
        "n_neighbors"   => json_number(cfg.n_neighbors),
        "min_dist"      => json_number(cfg.min_dist),
        "supervised"    => json_bool(cfg.supervised),
        "top_k_jaccard" => json_number(cfg.top_k_jaccard),
    )
    return json_object(
        "pca"          => pca_obj,
        "umap"         => umap_obj,
        "tsne"         => tsne_obj,
        "filter_level" => json_string(string(e.sample_filter_level)),
        "config"       => config_obj,
    )
end

"""
    _build_protein_embedding_json(ar) -> String

JSON for protein-level UMAP. Returns `"null"` when `ar.embeddings` is `nothing` OR
`protein_umap_coords` is `nothing`. When non-null, keys: `{x, y, classes, protein_ids}`.

Legacy single-AR / per-condition-report variant. The differential report path
should use `_build_protein_embedding_json_for_diff(diff)` instead
so the `classes` array reflects the *differential* class
labels (`kgroup_class` for k≥3, `classification` for k=2) rather than the
single-AR 3-class `H0/Agnostic/H1` labels (which collapse to a single colour
under typical settings: ~90% H0).
"""
function _build_protein_embedding_json(ar)::String
    (ar === nothing || !hasproperty(ar, :embeddings) || ar.embeddings === nothing) && return "null"
    e = ar.embeddings
    e.protein_umap_coords === nothing && return "null"
    n = size(e.protein_umap_coords, 1)
    return json_object(
        "x"           => json_array([json_number(e.protein_umap_coords[i, 1]) for i in 1:n]),
        "y"           => json_array([json_number(e.protein_umap_coords[i, 2]) for i in 1:n]),
        "classes"     => json_array([json_string(string(c)) for c in e.protein_classes]),
        "protein_ids" => json_array([json_string(p) for p in e.protein_ids]),
    )
end

"""
    _classes_for_protein_embedding(diff::DifferentialResult, protein_ids::AbstractVector) -> Vector{String}

Build the `classes` array for the differential
protein-embedding payload, indexed by `protein_ids` (matched against the
`:Protein` column of `diff.results`).

Source-column selection:
- **k≥3** (`length(diff.contrasts) >= 2` AND `:kgroup_class` present): reads
  `kgroup_class` (Symbol enum — 5 lowercase values). Strings emitted as the
  enum names (`"condition_specific"`, etc.).
- **k=2** (legacy or `length(diff.contrasts) <= 1`): reads `classification`
  (`InteractionClass` enum — 6 uppercase values: `"GAINED"`, `"REDUCED"`,
  `"UNCHANGED"`, `"BOTH_NEGATIVE"`, `"CONDITION_A_SPECIFIC"`,
  `"CONDITION_B_SPECIFIC"`). Strings match `CLS_COLOR` palette keys verbatim.
- **Fallback**: when neither column is present, returns all `"unknown"`.

Missing matches (a protein in `protein_ids` but absent from `diff.results`)
emit `"unknown"` — the JS palette has an explicit grey entry for this key.
"""
function _classes_for_protein_embedding(diff, protein_ids::AbstractVector)::Vector{String}
    res = diff.results
    cols = propertynames(res)

    # Pick the source column based on k.
    src_col = if length(diff.contrasts) >= 2 && :kgroup_class in cols
        :kgroup_class
    elseif :classification in cols
        :classification
    else
        nothing
    end
    src_col === nothing && return fill("unknown", length(protein_ids))

    # Build name → class lookup. Protein column may be `:Protein` (canonical) or
    # `:protein` (defensive — some legacy paths lowercase).
    name_col = :Protein in cols ? :Protein :
               (:protein in cols ? :protein : nothing)
    name_col === nothing && return fill("unknown", length(protein_ids))

    lookup = Dict{String, String}()
    for row in eachrow(res)
        name = string(row[name_col])
        cls  = row[src_col]
        if !ismissing(cls)
            lookup[name] = string(cls)
        end
    end

    return [get(lookup, string(pid), "unknown") for pid in protein_ids]
end

"""
    _build_protein_embedding_json_for_diff(diff) -> String

The differential-report Protein Embedding card
uses `wide_df.kgroup_class` (k≥3) or `wide_df.classification` (k=2) as the
`classes` array — NOT `first(diff.analyses).embeddings.protein_classes`,
which carried 3-class `H0/Agnostic/H1` labels from a single AR's
`LatentClassResult` (90%+ H0 → single-colour visualisation).

Spatial coordinates still come from the first AR's embedding (it's a joint
embedding across conditions anyway). Returns `"null"`
when no embedding payload is available (matches sibling diff builders).

Locks honored:
- The `CLS_COLOR` palette (palette extension is JS-side, additive).
- k=2 byte-equality — classification column already
  heterogeneous; same data emitted as before.
- kgroup_class enum — 5 values verbatim.
"""
function _build_protein_embedding_json_for_diff(diff)::String
    (diff === nothing || !hasproperty(diff, :analyses) ||
        diff.analyses === nothing || isempty(diff.analyses)) && return "null"
    ar = first(diff.analyses)
    (ar === nothing || !hasproperty(ar, :embeddings) || ar.embeddings === nothing) && return "null"
    e = ar.embeddings
    e.protein_umap_coords === nothing && return "null"
    n = size(e.protein_umap_coords, 1)

    # fix: derive classes from wide_df (kgroup_class for k≥3,
    # classification for k=2) instead of the single-AR LatentClassResult labels.
    classes = _classes_for_protein_embedding(diff, e.protein_ids)

    return json_object(
        "x"           => json_array([json_number(e.protein_umap_coords[i, 1]) for i in 1:n]),
        "y"           => json_array([json_number(e.protein_umap_coords[i, 2]) for i in 1:n]),
        "classes"     => json_array([json_string(c) for c in classes]),
        "protein_ids" => json_array([json_string(p) for p in e.protein_ids]),
    )
end

"""
    _build_condition_matrix_json(diff) -> String

JSON payload for the k×k similarity matrices. Returns `"null"` when
`diff.condition_similarity` is `nothing`. Single-bait reports always render `"null"`
since k=1 short-circuits in `_compute_condition_similarity`.
"""
function _build_condition_matrix_json(diff)::String
    (diff === nothing || !hasproperty(diff, :condition_similarity) ||
        diff.condition_similarity === nothing) && return "null"
    cs = diff.condition_similarity
    _mat = (m::AbstractMatrix{<:Real}) -> json_array([
        json_array([json_number(m[i, j]) for j in 1:size(m, 2)]) for i in 1:size(m, 1)
    ])
    return json_object(
        "labels"            => json_array([json_string(l) for l in cs.condition_labels]),
        "spearman_log10_bf" => _mat(cs.spearman_log10_bf),
        "pearson_log2fc"    => _mat(cs.pearson_log2fc),
        "pearson_posterior" => _mat(cs.pearson_posterior),
        "n_shared_per_cell" => _mat(cs.n_shared_per_cell),
    )
end

"""
    _build_jaccard_json(diff) -> String

JSON for the Jaccard@Top-K matrix. Returns `"null"` gate same as condition_matrix.
"""
function _build_jaccard_json(diff)::String
    (diff === nothing || !hasproperty(diff, :condition_similarity) ||
        diff.condition_similarity === nothing) && return "null"
    cs = diff.condition_similarity
    _mat = (m::AbstractMatrix{<:Real}) -> json_array([
        json_array([json_number(m[i, j]) for j in 1:size(m, 2)]) for i in 1:size(m, 1)
    ])
    return json_object(
        "matrix"     => _mat(cs.jaccard_top_k),
        "top_k_used" => json_number(cs.top_k_used),
        "labels"     => json_array([json_string(l) for l in cs.condition_labels]),
    )
end

"""
    _build_dendrogram_json(diff) -> String

JSON for the hclust dendrogram payload. Returns `"null"` gate same as condition_matrix.
"""
function _build_dendrogram_json(diff)::String
    (diff === nothing || !hasproperty(diff, :condition_similarity) ||
        diff.condition_similarity === nothing) && return "null"
    cs = diff.condition_similarity
    merges_arr = json_array([
        json_array([json_number(cs.dendrogram_merges[i, j]) for j in 1:size(cs.dendrogram_merges, 2)])
        for i in 1:size(cs.dendrogram_merges, 1)
    ])
    return json_object(
        "merges"  => merges_arr,
        "heights" => json_array([json_number(h) for h in cs.dendrogram_heights]),
        "order"   => json_array([json_number(o) for o in cs.dendrogram_order]),
        "linkage" => json_string(string(cs.linkage)),
        "labels"  => json_array([json_string(l) for l in cs.condition_labels]),
    )
end

"""
    _build_decision_risk_heatmap_block(diff::DifferentialResult) -> String

Decision Risk heatmap data block for the Multi-Condition tab.
Returns "null" for k<3 (legacy 2-group + 2-condition NamedTuple calls);
returns a populated JSON object (top-N proteins x pairs) for k>=3.

When no proteins pass the omnibus BFDR pre-filter, returns a JSON object with
`empty_state: true` and an `alert_html` string for the JS renderer.
"""
function _build_decision_risk_heatmap_block(diff::DifferentialResult)::String
    n_contrasts = length(diff.contrasts)
    n_labels    = length(condition_labels(diff))
    # k<3 suppression
    if n_contrasts < 2 || n_labels < 3
        return "null"
    end
    # Pre-filter: drop rows where omnibus BFDR exceeds threshold
    bfdr_threshold = diff.config.bfdr_threshold
    res = diff.results
    bfdr_col = hasproperty(res, :differential_BFDR_omnibus) ?
                   res.differential_BFDR_omnibus :
                   (hasproperty(res, :differential_BFDR) ?
                        res.differential_BFDR :
                        fill(missing, nrow(res)))
    keep_mask = Bool[(!ismissing(b) && isfinite(Float64(b)) && Float64(b) <= bfdr_threshold)
                     for b in bfdr_col]
    surviving = res[keep_mask, :]
    if nrow(surviving) < 1
        return json_object(
            "empty_state" => json_bool(true),
            "alert_html"  => json_string("<div class=\"alert alert-info\">No proteins pass the omnibus BFDR pre-filter -- Decision Risk heatmap suppressed.</div>"),
        )
    end
    # Sort ascending by decision_risk_min (k>=3 wide table always has it)
    sort_col = hasproperty(surviving, :decision_risk_min) ? :decision_risk_min : :decision_risk
    top_n_cfg = hasproperty(diff.config, :validation_candidates_top_n) ?
                    Int(diff.config.validation_candidates_top_n) : 20
    n_keep = min(top_n_cfg, nrow(surviving))
    if hasproperty(surviving, sort_col)
        vals = [(ismissing(x) || (x isa AbstractFloat && isnan(x))) ? Inf : Float64(x)
                for x in getproperty(surviving, sort_col)]
        order = sortperm(vals)
        top_idxs = order[1:n_keep]
    else
        top_idxs = collect(1:n_keep)
    end
    top_proteins = String[String(surviving.Protein[i]) for i in top_idxs]
    pair_labels  = String["$(String(first(p))) vs $(String(last(p)))" for p in diff.contrasts]
    risk_matrix_rows    = String[]
    optimal_matrix_rows = String[]
    map_matrix_rows     = String[]
    for name in top_proteins
        risk_row    = String[]
        optimal_row = String[]
        map_row     = String[]
        for pair in diff.contrasts
            sub = diff.pairwise_results === nothing ? nothing :
                  get(diff.pairwise_results, pair, nothing)
            if sub === nothing
                push!(risk_row, "null")
                push!(optimal_row, json_string("n/a"))
                push!(map_row, json_string("n/a"))
                continue
            end
            row_idx = findfirst(==(name), String.(sub.Protein))
            if row_idx === nothing
                push!(risk_row, "null")
                push!(optimal_row, json_string("n/a"))
                push!(map_row, json_string("n/a"))
            else
                v = hasproperty(sub, :decision_risk) ? sub.decision_risk[row_idx] : NaN
                push!(risk_row, json_number_nan_safe(v))
                oc = hasproperty(sub, :optimal_call) && !ismissing(sub.optimal_call[row_idx]) ?
                         string(sub.optimal_call[row_idx]) : "n/a"
                push!(optimal_row, json_string(oc))
                mc = hasproperty(sub, :classification) && !ismissing(sub.classification[row_idx]) ?
                         lowercase(string(sub.classification[row_idx])) : "n/a"
                push!(map_row, json_string(mc))
            end
        end
        push!(risk_matrix_rows,    json_array(risk_row))
        push!(optimal_matrix_rows, json_array(optimal_row))
        push!(map_matrix_rows,     json_array(map_row))
    end
    return json_object(
        "empty_state"   => json_bool(false),
        "proteins"      => json_array([json_string(p) for p in top_proteins]),
        "pairs"         => json_array([json_string(s) for s in pair_labels]),
        "risk_matrix"   => json_array(risk_matrix_rows),
        "optimal_calls" => json_array(optimal_matrix_rows),
        "map_classes"   => json_array(map_matrix_rows),
    )
end

"""
    _build_validation_candidates_block(diff::DifferentialResult) -> String

top-level JSON block driving the
Validation Candidates pill in the Results tab.

layer:
- Pre-filters on omnibus BFDR (falls back to differential_BFDR for legacy
  2-group), sorts ascending by decision_risk_min (k>=3) or decision_risk
  (k=2), and emits up to N candidates (config.validation_candidates_top_n;
  default 20).

polish layer:
- **BOTH_NEGATIVE exclusion** (`candidates` top-N list): rows where
  `classification == BOTH_NEGATIVE` (k=2) OR every per-pair
  `classification_<a>_vs_<b>` is `BOTH_NEGATIVE` (k>=3) are dropped AFTER
  the omnibus-BFDR pre-filter (final filter step).
- **`map_class` population for k>=3**: reads `row.kgroup_class` (the
  5-value enum) instead of `row.classification` (which is absent on
  the k>=3 wide DF — only suffixed `classification_<a>_vs_<b>` columns
  exist there).
- **show-all panel emission**: a sibling `all_proteins` array carries
  the full ranked list INCLUDING BOTH_NEGATIVE rows AND rows that failed
  the omnibus pre-filter — for QC inspection. The JS lazy-renders this
  list into a `<details>`
  collapsible panel below the top-N grid.
"""
function _build_validation_candidates_block(diff::DifferentialResult)::String
    # no user-visible phase-number
    # literals exist in this block — all such
    # hits in this function are Julia `#` comments or `"""..."""` docstrings,
    # never runtime strings emitted into the JSON payload. Regression test
    # `test/reports/test_report_phase_number_strip.jl` guards against future
    # reintroduction of phase-number literals in tooltips / headings / card
    # subtitles (Methods subsection citations remain allowed verbatim).
    top_n = hasproperty(diff.config, :validation_candidates_top_n) ?
                Int(diff.config.validation_candidates_top_n) : 20
    bfdr_threshold = diff.config.bfdr_threshold
    res = diff.results
    is_kgroup = length(diff.contrasts) >= 2

    bfdr_col = hasproperty(res, :differential_BFDR_omnibus) ?
                   res.differential_BFDR_omnibus :
                   (hasproperty(res, :differential_BFDR) ?
                        res.differential_BFDR :
                        fill(missing, nrow(res)))
    keep_mask = Bool[(!ismissing(b) && isfinite(Float64(b)) && Float64(b) <= bfdr_threshold)
                     for b in bfdr_col]
    surviving = res[keep_mask, :]

    # Helper: derive map_class for a single row.
    # k>=3 reads kgroup_class; k=2 reads classification (InteractionClass enum).
    function _map_class_for(row)::String
        if is_kgroup && :kgroup_class in propertynames(row)
            v = row.kgroup_class
            return ismissing(v) ? "" : string(v)
        elseif :classification in propertynames(row)
            v = row.classification
            return ismissing(v) ? "" : string(v)
        end
        return ""
    end

    # ── Validation candidates rank/label by the cheapest ACTIONABLE pair ──────
    # A validation candidate is a protein worth experimentally validating: it must
    # have an ACTIONABLE optimal call (anything other than `both_negative`) in at
    # least one contrast. We rank by — and label with — the cheapest such pair.
    #
    # The previous logic dropped only all-BOTH_NEGATIVE proteins and ranked by
    # `decision_risk_min` (min over ALL pairs). The min-risk pair is almost always
    # a `both_negative` call (expected loss ≈ 0), so proteins that are negative in
    # their cheapest pair surfaced at the very TOP labelled `both_negative`. Ranking
    # over actionable pairs only guarantees no `both_negative` row appears in the
    # candidate list AND that the displayed risk/call/pair are the validation target.
    _ACTIONABLE_EXCLUDE = ("both_negative", "")
    function _best_actionable(row)  # → (risk::Float64, call::String, pair_label::String) | nothing
        best = nothing
        if is_kgroup
            for p in diff.contrasts
                a, b = String(first(p)), String(last(p))
                oc_col = Symbol("optimal_call_$(a)_vs_$(b)")
                dr_col = Symbol("decision_risk_$(a)_vs_$(b)")
                (oc_col in propertynames(row) && dr_col in propertynames(row)) || continue
                oc = row[oc_col]; dr = row[dr_col]
                (ismissing(oc) || lowercase(string(oc)) in _ACTIONABLE_EXCLUDE) && continue
                (ismissing(dr) || (dr isa AbstractFloat && isnan(dr))) && continue
                drf = Float64(dr)
                if best === nothing || drf < best[1]
                    best = (drf, string(oc), "$(a) vs $(b)")
                end
            end
        else
            oc = hasproperty(row, :optimal_call) ? row.optimal_call : missing
            dr = hasproperty(row, :decision_risk) ? row.decision_risk : missing
            if !ismissing(oc) && !(lowercase(string(oc)) in _ACTIONABLE_EXCLUDE) &&
               !ismissing(dr) && !(dr isa AbstractFloat && isnan(dr))
                best = (Float64(dr), string(oc),
                        string(diff.condition_A) * " vs " * string(diff.condition_B))
            end
        end
        return best
    end

    best_list = Tuple{Float64, String, String}[]
    keep_idx  = Int[]
    for (i, row) in enumerate(eachrow(surviving))
        b = _best_actionable(row)
        b === nothing && continue
        push!(best_list, b); push!(keep_idx, i)
    end
    surviving = surviving[keep_idx, :]
    if !isempty(best_list)
        order     = sortperm([b[1] for b in best_list])
        surviving = surviving[order, :]
        best_list = best_list[order]
    end

    n_keep = min(top_n, nrow(surviving))
    cands = String[]
    for i in 1:n_keep
        row = surviving[i, :]
        risk, call, pair_label = best_list[i]
        push!(cands, json_object(
            "protein"       => json_string(String(row.Protein)),
            "decision_risk" => json_number_nan_safe(risk),
            "optimal_call"  => json_string(call),
            "pair_label"    => json_string(pair_label),
            "map_class"     => json_string(_map_class_for(row)),
        ))
    end

    # build `all_proteins` — full ranked list INCLUDING BOTH_NEGATIVE rows
    # AND rows that failed the omnibus pre-filter (broader QC inspection view per
    # RESEARCH.md Open Q 2). Sorted ascending by the same decision_risk{_min}
    # column; missing/NaN rows sort to the bottom.
    sort_col_all = hasproperty(res, :decision_risk_min) ? :decision_risk_min :
                   (hasproperty(res, :decision_risk) ? :decision_risk : nothing)
    all_rows_df = res
    if sort_col_all !== nothing && nrow(all_rows_df) > 0
        vals_all = [(ismissing(x) || (x isa AbstractFloat && isnan(x))) ? Inf : Float64(x)
                    for x in getproperty(all_rows_df, sort_col_all)]
        order_all = sortperm(vals_all)
        all_rows_df = all_rows_df[order_all, :]
    end
    all_proteins = String[]
    for row in eachrow(all_rows_df)
        dr_val_all = sort_col_all === nothing ? NaN :
                     (hasproperty(row, sort_col_all) ? row[sort_col_all] : NaN)
        oc_sym_all = hasproperty(row, :optimal_call_min) && !ismissing(row.optimal_call_min) ?
                         row.optimal_call_min :
                         (hasproperty(row, :optimal_call) && !ismissing(row.optimal_call) ?
                              row.optimal_call : :unchanged)
        ba = _best_actionable(row)
        pair_lbl_all = ba === nothing ? "—" : ba[3]
        push!(all_proteins, json_object(
            "protein"       => json_string(String(row.Protein)),
            "decision_risk" => json_number_nan_safe(dr_val_all),
            "optimal_call"  => json_string(string(oc_sym_all)),
            "pair_label"    => json_string(pair_lbl_all),
            "map_class"     => json_string(_map_class_for(row)),
        ))
    end

    return json_object(
        "top_n"              => json_number(top_n),
        "sort_column"        => json_string("decision_risk"),
        "bfdr_threshold"     => json_number(bfdr_threshold),
        "n_surviving_filter" => json_number(nrow(surviving)),
        "n_total"            => json_number(nrow(res)),
        "is_kgroup"          => json_bool(is_kgroup),
        "candidates"         => json_array(cands),
        "all_proteins"       => json_array(all_proteins),
    )
end

"""
    _build_multi_condition_json(diff::DifferentialResult) -> String

JSON payload driving the Multi-Condition tab +
matrix-view dropdown in `differential_report.html`. Returns `"null"` for
k<3 (legacy 2-group + 2-condition NamedTuple calls); returns a populated
JSON object for k>=3.

Schema:
- contrasts             : Array of [first_str, last_str] pairs from `diff.contrasts`
- condition_labels      : Array of unique condition labels from `condition_labels(diff)`
- per_pair_volcano_data : Array of per-pair objects (consumed for the
                          small-multiples grid). Each carries `delta_log2fc`
                          read DIRECTLY from `diff.pairwise_results[p].delta_log2fc`
                          (always present per `_compute_differential_statistics`).
- matrix_view_data      : { labels, posterior_median_matrix } (consumed
                          for the k>=4 heatmap)
- show_tab              : JSON Boolean — server-side mirror of the JS predicate
- show_dropdown         : JSON Boolean
"""
function _build_multi_condition_json(diff::DifferentialResult)::String
    # Early-return guards
    n_contrasts = length(diff.contrasts)
    cond_labels = condition_labels(diff)
    n_labels    = length(cond_labels)

    # k<3 path: return null. The HTML element exists in the template but
    # `initMultiConditionTab` keeps tab-multi-li hidden when the predicate
    # `n_contrasts >= 2 && n_labels >= 3` is not satisfied.
    if n_contrasts < 2 || n_labels < 3
        return "null"
    end

    # contrasts as nested [first, last] string pairs
    contrasts_json = json_array([
        json_array([json_string(String(first(p))), json_string(String(last(p)))])
        for p in diff.contrasts
    ])

    # condition_labels as string array (XSS-defanged via json_string)
    labels_json = json_array([json_string(l) for l in cond_labels])

    # per_pair_volcano_data: one object per pair carrying minimal columns for
    # small-multiples. BLOCKER #2 CLOSURE: delta_log2fc read DIRECTLY — the
    # per-pair DF is the output of `_compute_differential_statistics`, which
    # ALWAYS populates :delta_log2fc. Defensive outer-level skip remains for
    # empty/missing pair DFs (no shared proteins).
    per_pair_data = if diff.pairwise_results === nothing
        json_array(String[])
    else
        pair_objects = String[]
        for p in diff.contrasts
            df_pair = get(diff.pairwise_results, p, nothing)
            df_pair === nothing && continue
            nrow(df_pair) == 0 && continue
            hasproperty(df_pair, :delta_log2fc) || error(
                "_build_multi_condition_json: per-pair DF for $p missing :delta_log2fc " *
                "(check _compute_differential_statistics — should always emit this column)")

            # Optional columns: differential_BFDR_pairwise_BH →
            # differential_BFDR (Storey, always present on a Phase-72+ DF) →
            # all-missing fallback.
            bfdr_col = hasproperty(df_pair, :differential_BFDR_pairwise_BH) ?
                       df_pair.differential_BFDR_pairwise_BH :
                       (hasproperty(df_pair, :differential_BFDR) ?
                            df_pair.differential_BFDR :
                            fill(missing, nrow(df_pair)))
            pep_col = hasproperty(df_pair, :differential_pep) ?
                      df_pair.differential_pep :
                      fill(missing, nrow(df_pair))
            cls_col = hasproperty(df_pair, :classification) ?
                      df_pair.classification :
                      fill(:UNCHANGED, nrow(df_pair))

            pair_obj = json_object(
                "pair"        => json_array([json_string(String(first(p))),
                                             json_string(String(last(p)))]),
                "n_proteins"  => json_number(nrow(df_pair)),
                "proteins"    => json_array([json_string(String(x)) for x in df_pair.Protein]),
                "delta_log2fc" => json_array([
                    ismissing(x) ? "null" : json_number(Float64(x))
                    for x in df_pair.delta_log2fc
                ]),
                "differential_BFDR_pairwise_BH" => json_array([
                    ismissing(x) ? "null" : json_number(Float64(x))
                    for x in bfdr_col
                ]),
                "differential_pep" => json_array([
                    ismissing(x) ? "null" : json_number(Float64(x))
                    for x in pep_col
                ]),
                "classification" => json_array([
                    json_string(string(x)) for x in cls_col
                ]),
            )
            push!(pair_objects, pair_obj)
        end
        json_array(pair_objects)
    end

    # matrix_view_data: k×k posterior median matrix.
    # Off-diagonal cells: median of `posterior_omnibus` (if
    # absent, fall back to differential_posterior) for that pair.
    # Diagonal cells: emitted as JSON null (the renderer blanks them).
    k = n_labels
    matrix = fill("null", k, k)
    if diff.pairwise_results !== nothing
        label_idx = Dict(cond_labels[i] => i for i in 1:k)
        for p in diff.contrasts
            df_pair = get(diff.pairwise_results, p, nothing)
            df_pair === nothing && continue
            pp_col = hasproperty(df_pair, :posterior_omnibus) ?
                     df_pair.posterior_omnibus :
                     (hasproperty(df_pair, :differential_posterior) ?
                          df_pair.differential_posterior :
                          nothing)
            pp_col === nothing && continue
            valid = collect(skipmissing(pp_col))
            isempty(valid) && continue
            med = median(Float64.(valid))
            i = get(label_idx, String(first(p)), nothing)
            j = get(label_idx, String(last(p)), nothing)
            (i === nothing || j === nothing) && continue
            matrix[i, j] = json_number(med)
            matrix[j, i] = json_number(med)  # symmetric fill
        end
    end
    matrix_rows = [json_array([matrix[i, j] for j in 1:k]) for i in 1:k]
    matrix_view = json_object(
        "labels"                  => labels_json,
        "posterior_median_matrix" => json_array(matrix_rows),
    )

    # show_tab + show_dropdown — server-side mirror of the JS predicate
    show_tab      = json_bool(n_contrasts >= 2 && n_labels >= 3)
    show_dropdown = json_bool(n_contrasts >= 6)

    return json_object(
        "contrasts"             => contrasts_json,
        "condition_labels"      => labels_json,
        "per_pair_volcano_data" => per_pair_data,
        "matrix_view_data"      => matrix_view,
        "show_tab"              => show_tab,
        "show_dropdown"         => show_dropdown,
        # Decision Risk heatmap data block (k>=3 only)
        "decision_risk_heatmap" => _build_decision_risk_heatmap_block(diff),
    )
end

"""
    _build_dbf_histogram_qq(df) -> String

Panel 1: histogram of `log10_dbf` (all proteins) + reference Q-Q against the
empirical null built from rows classified `UNCHANGED`. Returns `"null"` if the
required columns are missing.
"""
function _build_dbf_histogram_qq(df::DataFrame)::String
    hasproperty(df, :log10_dbf) || return "null"
    vals = Float64[]
    for v in df.log10_dbf
        (ismissing(v) || !isfinite(Float64(v))) && continue
        push!(vals, Float64(v))
    end
    isempty(vals) && return "null"

    # Null sample — proteins classified UNCHANGED (defensive: also accept missing
    # classification by treating as the broader pool).
    null_vals = if hasproperty(df, :classification)
        out = Float64[]
        for i in 1:nrow(df)
            cls = df.classification[i]
            ismissing(cls) && continue
            if string(cls) == "UNCHANGED"
                v = df.log10_dbf[i]
                (ismissing(v) || !isfinite(Float64(v))) && continue
                push!(out, Float64(v))
            end
        end
        out
    else
        Float64[]
    end

    return json_object(
        "values"     => json_array([json_number(v) for v in vals]),
        "null_values"=> json_array([json_number(v) for v in null_vals]),
    )
end

"""
    _build_dbf_vs_delta(df) -> String

Panel 2: scatter of `log10_dbf` vs `delta_log2fc`, with classification color
encoding (the JS layer maps each classification to a Plotly color).
"""
function _build_dbf_vs_delta(df::DataFrame)::String
    (hasproperty(df, :log10_dbf) && hasproperty(df, :delta_log2fc)) || return "null"
    xs = Float64[]
    ys = Float64[]
    cls = String[]
    prots = String[]
    has_cls  = hasproperty(df, :classification)
    has_prot = hasproperty(df, :Protein)
    for i in 1:nrow(df)
        x = df.delta_log2fc[i]
        y = df.log10_dbf[i]
        (ismissing(x) || ismissing(y)) && continue
        xv = Float64(x); yv = Float64(y)
        (isfinite(xv) && isfinite(yv)) || continue
        push!(xs, xv); push!(ys, yv)
        push!(cls,  has_cls  ? (ismissing(df.classification[i]) ? "" : string(df.classification[i])) : "")
        push!(prots, has_prot ? string(df.Protein[i]) : "")
    end
    return json_object(
        "x"              => json_array([json_number(v) for v in xs]),
        "y"              => json_array([json_number(v) for v in ys]),
        "classification" => json_array([json_string(c) for c in cls]),
        "proteins"       => json_array([json_string(p) for p in prots]),
    )
end

"""
    _build_dbf_component_stack(df, N) -> String

Panel 3: per-component contribution stacked bar for the top-N proteins by `|log10_dbf|`.
Decomposition uses the existing `dbf_enrichment` / `dbf_correlation` / `dbf_detected`
columns (always present on `DifferentialResult.results`).
"""
function _build_dbf_component_stack(df::DataFrame, N::Int)::String
    needed = (:log10_dbf, :dbf_enrichment, :dbf_correlation, :dbf_detected, :Protein)
    all(c -> hasproperty(df, c), needed) || return "null"

    n = nrow(df)
    abs_l10 = Float64[]
    for v in df.log10_dbf
        push!(abs_l10, ismissing(v) ? -Inf : abs(Float64(v)))
    end
    order = sortperm(abs_l10; rev=true)
    keep = order[1:min(N, length(order))]

    prots = String[]
    enr   = Float64[]
    corr  = Float64[]
    det   = Float64[]
    for i in keep
        push!(prots, string(df.Protein[i]))
        push!(enr,  ismissing(df.dbf_enrichment[i])  ? 0.0 : log10(max(Float64(df.dbf_enrichment[i]),  1e-300)))
        push!(corr, ismissing(df.dbf_correlation[i]) ? 0.0 : log10(max(Float64(df.dbf_correlation[i]), 1e-300)))
        push!(det,  ismissing(df.dbf_detected[i])    ? 0.0 : log10(max(Float64(df.dbf_detected[i]),    1e-300)))
    end
    return json_object(
        "proteins"        => json_array([json_string(p) for p in prots]),
        "log10_dbf_e"     => json_array([json_number(v) for v in enr]),
        "log10_dbf_c"     => json_array([json_number(v) for v in corr]),
        "log10_dbf_d"     => json_array([json_number(v) for v in det]),
    )
end

"""
    _build_dbf_saturation_panel(df) -> String

Panel 4: count of proteins with `dbf_diagnostic == :saturated` plus the protein
list. Defence-in-depth: also `hasproperty`-guards `dbf_diagnostic` so direct calls
outside the orchestrator do not regress.
"""
function _build_dbf_saturation_panel(df::DataFrame)::String
    hasproperty(df, :dbf_diagnostic) || return "null"
    has_prot = hasproperty(df, :Protein)
    sat_proteins = String[]
    for i in 1:nrow(df)
        d = df.dbf_diagnostic[i]
        ismissing(d) && continue
        if d === :saturated
            push!(sat_proteins, has_prot ? string(df.Protein[i]) : string(i))
        end
    end
    return json_object(
        "count"    => json_number(length(sat_proteins)),
        "proteins" => json_array([json_string(p) for p in sat_proteins]),
    )
end

"""
    _build_dbf_submodel_disagreement(df) -> String

Panel 5: scatter of `log10(dbf_em)` vs `log10(dbf_copula)` per protein. The dBF
sub-model columns (`dbf_em` / `dbf_copula`) may be absent when either condition
ran without `:bma` — guard with `hasproperty` and return `"null"` in that case.
"""
function _build_dbf_submodel_disagreement(df::DataFrame)::String
    em_col = hasproperty(df, :dbf_em)     ? :dbf_em     : nothing
    co_col = hasproperty(df, :dbf_copula) ? :dbf_copula : nothing
    (em_col === nothing || co_col === nothing) && return "null"
    has_prot = hasproperty(df, :Protein)
    xs = Float64[]
    ys = Float64[]
    prots = String[]
    for i in 1:nrow(df)
        ve = df[i, em_col]; vc = df[i, co_col]
        (ismissing(ve) || ismissing(vc)) && continue
        x = log10(max(Float64(vc), 1e-300))
        y = log10(max(Float64(ve), 1e-300))
        (isfinite(x) && isfinite(y)) || continue
        push!(xs, x); push!(ys, y)
        push!(prots, has_prot ? string(df.Protein[i]) : string(i))
    end
    return json_object(
        "x"         => json_array([json_number(v) for v in xs]),
        "y"         => json_array([json_number(v) for v in ys]),
        "proteins"  => json_array([json_string(p) for p in prots]),
        "x_label"   => json_string("log10(dbf_copula)"),
        "y_label"   => json_string("log10(dbf_em)"),
    )
end

"""
    _build_dbf_traffic_light_summary(df) -> String

Panel 6: count of each `dbf_diagnostic` Symbol value across all proteins.
Defence-in-depth: also `hasproperty`-guards `dbf_diagnostic` so direct calls
outside the orchestrator do not regress.
"""
function _build_dbf_traffic_light_summary(df::DataFrame)::String
    hasproperty(df, :dbf_diagnostic) || return "null"
    counts = Dict{String, Int}()
    for d in df.dbf_diagnostic
        ismissing(d) && continue
        k = string(d)
        counts[k] = get(counts, k, 0) + 1
    end
    keys_sorted = sort(collect(keys(counts)))
    pairs = Pair{String, String}[]
    for k in keys_sorted
        push!(pairs, k => json_number(counts[k]))
    end
    return json_object(pairs...)
end

# ---------------------------------------------------------------------------
# Docking tab data
# ---------------------------------------------------------------------------

"""Build docking JSON object for the report. Returns `"{}"` when no docking data."""
function _build_docking_json(results::DataFrame, docking_result;
                             structures_dir::String="")::String
    docking_result === nothing && return "{}"

    # Auto-derive structures_dir from docking config cache if not explicitly provided
    if isempty(structures_dir)
        cache_dir = docking_result.config.cache_dir
        if isempty(cache_dir)
            cache_dir = joinpath(".bayesinteractomics_cache", "docking")
        end
        candidate = joinpath(cache_dir, "structures")
        if isdir(candidate)
            structures_dir = candidate
        end
    end

    # Filter to detected proteins only for posterior lookup
    results = _filter_detected(results)

    # Extract pairs from DockingResult
    pairs = docking_result.pairs
    isempty(pairs) && return "{}"

    # Summary statistics
    successful = filter(p -> p.status == :success, pairs)
    iptm_vals  = [p.iptm_best for p in successful if isfinite(p.iptm_best)]
    bf_vals    = [p.bf_dock for p in successful if isfinite(p.bf_dock)]

    summary_json = json_object(
        "n_total"       => json_number(docking_result.n_total),
        "n_docked"      => json_number(docking_result.n_docked),
        "n_pending"     => json_number(docking_result.n_pending),
        "n_disordered"  => json_number(docking_result.n_disordered),
        "n_too_large"   => json_number(docking_result.n_too_large),
        "median_iptm"   => json_number(isempty(iptm_vals) ? NaN : median(iptm_vals)),
        "median_bf_dock"=> json_number(isempty(bf_vals) ? NaN : median(bf_vals)),
    )

    # Serialize each pair
    pair_jsons = String[]
    for p in pairs
        iptm_all_json = json_array([json_number(v) for v in p.iptm_all])

        # Look up MS-only posterior from results DataFrame
        pp_ms = NaN
        pp_combined = NaN
        bfdr_combined_val = NaN
        idx = findfirst(==(p.protein_b), results.Protein)
        if idx === nothing
            idx = findfirst(==(p.protein_a), results.Protein)
        end
        if idx !== nothing
            pp_ms = _safe_float(results.posterior_prob[idx])
            pp_ms = pp_ms === nothing ? NaN : pp_ms
            if hasproperty(results, :posterior_prob_combined)
                pp_combined = _safe_float(results.posterior_prob_combined[idx])
                pp_combined = pp_combined === nothing ? NaN : pp_combined
            else
                # Compute from BF update
                if isfinite(pp_ms) && isfinite(p.bf_dock)
                    odds = pp_ms / (1.0 - pp_ms)
                    odds_new = odds * p.bf_dock
                    pp_combined = odds_new / (1.0 + odds_new)
                end
            end
            if hasproperty(results, :BFDR_combined)
                bc = results.BFDR_combined[idx]
                bfdr_combined_val = ismissing(bc) ? NaN : Float64(bc)
            end
        end

        # Embed structure data directly for portable reports (no broken relative paths)
        structure_data = ""
        structure_format = ""
        if !isempty(structures_dir)
            pair_key = "$(p.protein_a)_$(p.protein_b)"
            for (ext, fmt) in [(".cif", "mmcif"), (".pdb", "pdb")]
                candidate = joinpath(structures_dir, pair_key * ext)
                if isfile(candidate)
                    try
                        raw = Base.read(candidate)
                        structure_data = Base64.base64encode(transcode(GzipCompressor, raw))
                        structure_format = fmt
                    catch e
                        @warn "Failed to read structure file $candidate" exception=e
                    end
                    break
                end
            end
        end

        push!(pair_jsons, json_object(
            "protein"               => json_string(p.protein_b),
            "protein_a"             => json_string(p.protein_a),
            "protein_b"             => json_string(p.protein_b),
            "posterior_prob_ms"      => json_number(pp_ms),
            "posterior_prob_combined"=> json_number(pp_combined),
            "bf_docking"            => json_number(p.bf_dock),
            "iptm_best"             => json_number(p.iptm_best),
            "iptm_all"              => iptm_all_json,
            "iptm_std"              => json_number(p.iptm_std),
            "ranking_score"         => json_number(p.ranking_score),
            "fraction_disordered"   => json_number(p.fraction_disordered),
            "chain_pair_iptm"       => json_number(p.chain_pair_iptm),
            "chain_pair_pae_min"    => json_number(p.chain_pair_pae_min),
            "status"                => json_string(string(p.status)),
            "token_count"           => json_number(p.token_count),
            "structure_file"        => json_string(""),
            "structure_data"        => json_string(structure_data),
            "structure_format"      => json_string(structure_format),
            "pdockq"                => json_number(p.pdockq),
            "c2qscore"              => json_number(p.c2qscore),
            "ipae"                  => json_number(p.ipae),
            "iplddt_interface"      => json_number(p.iplddt_interface),
            "calibration_tier"      => json_string(p.calibration_tier),
            "BFDR_combined"         => json_number(bfdr_combined_val),
            "PEP_combined"          => json_number(isfinite(pp_combined) ? 1.0 - pp_combined : NaN),
        ))
    end

    return json_object(
        "summary" => summary_json,
        "pairs"   => json_array(pair_jsons),
    )
end

# ---------------------------------------------------------------------------
# Diagnostics data (interactive Plotly)
# ---------------------------------------------------------------------------

"""Build diagnostics data JSON for interactive Plotly plots in the report."""
function _build_diagnostics_data_json(analysis_result, results_df=nothing)::String
    analysis_result === nothing && return "null"

    # Filter to detected proteins only (non-detected have missing BFs; exclude from plots)
    if results_df !== nothing && results_df isa DataFrame
        results_df = _filter_detected(results_df)
    end

    sections = Pair{String,String}[]

    diag = nothing
    if hasproperty(analysis_result, :diagnostics)
        diag = analysis_result.diagnostics
    end

    if diag !== nothing
        # 1. Calibration data (combine all three into one plot)
        cal_json = _build_calibration_json(diag)
        cal_json != "null" && push!(sections, "calibration" => cal_json)

        # 2. Residual Q-Q data
        qq_json = _build_qq_json(diag)
        qq_json != "null" && push!(sections, "residuals" => qq_json)

        # 3. PPC p-value histogram
        ppc_json = _build_ppc_json(diag)
        ppc_json != "null" && push!(sections, "ppc" => ppc_json)

        # 4. PIT histogram
        pit_json = _build_pit_json(diag)
        pit_json != "null" && push!(sections, "pit" => pit_json)

        # 5. Nu optimization
        nu_json = _build_nu_json(diag)
        nu_json != "null" && push!(sections, "nu_optimization" => nu_json)
    end

    # 6. Within-class correlation (needs analysis_result, not diag)
    wc_json = _build_within_class_json(analysis_result)
    wc_json != "null" && push!(sections, "within_class" => wc_json)

    # 7. KL divergence: H1 component purity (reconstructed from LC result + BFs)
    kl_json = _build_kl_divergence_json(analysis_result, results_df)
    kl_json != "null" && push!(sections, "kl_divergence" => kl_json)

    # 8. Discordant protein decomposition (from LC result + individual BFs)
    disc_json = _build_discordant_json(analysis_result, results_df)
    disc_json != "null" && push!(sections, "discordant" => disc_json)

    # 9. Copula bootstrap confidence (placeholder — data not stored in analysis_result)
    boot_json = _build_copula_bootstrap_json(analysis_result, results_df)
    boot_json != "null" && push!(sections, "copula_bootstrap" => boot_json)

    isempty(sections) && return "null"
    return json_object(sections...)
end

function _build_calibration_json(diag)::String
    cals = Pair{String,String}[]
    for (key, field) in [("strict", :calibration), ("relaxed", :calibration_relaxed), ("enrichment_only", :calibration_enrichment_only)]
        cal = getfield(diag, field)
        cal === nothing && continue
        push!(cals, key => json_object(
            "bin_midpoints" => json_array([json_number(v) for v in cal.bin_midpoints]),
            "predicted_rate" => json_array([json_number(v) for v in cal.predicted_rate]),
            "observed_rate" => json_array([json_number(v) for v in cal.observed_rate]),
            "bin_counts" => json_array([json_number(v) for v in cal.bin_counts]),
            "ece" => json_number(cal.ece),
            "mce" => json_number(cal.mce),
        ))
    end
    isempty(cals) && return "null"
    return json_object(cals...)
end

function _build_qq_json(diag)::String
    parts = Pair{String,String}[]
    for (key, field) in [("hbm", :hbm_residuals), ("regression", :regression_residuals)]
        res = getfield(diag, field)
        res === nothing && continue
        # Sort pooled residuals for Q-Q plot
        sorted = sort(res.pooled_residuals)
        n = length(sorted)
        n == 0 && continue
        # Theoretical quantiles (standard normal)
        theoretical = [_qnorm((i - 0.5) / n) for i in 1:n]
        push!(parts, key => json_object(
            "sample_quantiles" => json_array([json_number(v) for v in sorted]),
            "theoretical_quantiles" => json_array([json_number(v) for v in theoretical]),
            "n" => json_number(n),
            "skewness" => json_number(res.skewness),
            "kurtosis" => json_number(res.kurtosis),
        ))
    end
    # Scale-location data
    for (key, field) in [("hbm_scale_loc", :hbm_residuals), ("regression_scale_loc", :regression_residuals)]
        res = getfield(diag, field)
        res === nothing && continue
        length(res.pooled_fitted) == 0 && continue
        push!(parts, key => json_object(
            "fitted" => json_array([json_number(v) for v in res.pooled_fitted]),
            "sqrt_abs_residuals" => json_array([json_number(sqrt(abs(v))) for v in res.pooled_residuals]),
        ))
    end
    isempty(parts) && return "null"
    return json_object(parts...)
end

function _build_ppc_json(diag)::String
    ppcs = diag.protein_ppcs
    (ppcs === nothing || isempty(ppcs)) && return "null"
    # Separate HBM and regression p-values
    hbm_pvals = Float64[]
    reg_pvals = Float64[]
    for ppc in ppcs
        if ppc.model == :hbm
            push!(hbm_pvals, ppc.pvalue_mean)
        else
            push!(reg_pvals, ppc.pvalue_mean)
        end
    end
    parts = Pair{String,String}[]
    !isempty(hbm_pvals) && push!(parts, "hbm_pvalues" => json_array([json_number(v) for v in hbm_pvals]))
    !isempty(reg_pvals) && push!(parts, "regression_pvalues" => json_array([json_number(v) for v in reg_pvals]))
    isempty(parts) && return "null"
    return json_object(parts...)
end

function _build_pit_json(diag)::String
    parts = Pair{String,String}[]
    if diag.enhanced_hbm_residuals !== nothing
        vals = diag.enhanced_hbm_residuals.pit_values
        !isempty(vals) && push!(parts, "hbm" => json_array([json_number(v) for v in vals]))
    end
    if diag.enhanced_regression_residuals !== nothing
        vals = diag.enhanced_regression_residuals.pit_values
        !isempty(vals) && push!(parts, "regression" => json_array([json_number(v) for v in vals]))
    end
    isempty(parts) && return "null"
    return json_object(parts...)
end

function _build_nu_json(diag)::String
    nu = diag.nu_optimization
    nu === nothing && return "null"
    return json_object(
        "nu_trace" => json_array([json_number(v) for v in nu.nu_trace]),
        "waic_trace" => json_array([json_number(v) for v in nu.waic_trace]),
        "optimal_nu" => json_number(nu.optimal_nu),
        "search_bounds" => json_array([json_number(nu.search_bounds[1]), json_number(nu.search_bounds[2])]),
    )
end

function _build_within_class_json(analysis_result)::String
    # Get responsibilities and BF triplet
    lc = nothing
    if hasproperty(analysis_result, :bma_result) && analysis_result.bma_result !== nothing
        lc = analysis_result.bma_result.em3c_result
    elseif hasproperty(analysis_result, :latent_class_result)
        lc = analysis_result.latent_class_result
    end
    (lc === nothing || !isa(lc, LatentClassResult) || lc.responsibilities === nothing) && return "null"

    # Legacy LatentClassResult without protein_names hides within-class section entirely
    (lc.protein_names === nothing || isempty(lc.protein_names)) && return "null"

    # Get BFs from results
    results = nothing
    if hasproperty(analysis_result, :copula_results)
        results = analysis_result.copula_results
    end
    (results === nothing || !hasproperty(results, :bf_enrichment)) && return "null"

    # Filter to detected proteins only (non-detected have missing BFs that become log(1e-300) garbage)
    results = _filter_detected(results)

    resp = lc.responsibilities

    # Name-based join: align responsibilities rows with results rows by protein name
    name_to_resp_idx = Dict{String,Int}(lc.protein_names[i] => i for i in eachindex(lc.protein_names))

    matched_results_idx = Int[]
    matched_resp_idx = Int[]
    for i in 1:nrow(results)
        pname = string(results.Protein[i])
        if haskey(name_to_resp_idx, pname)
            push!(matched_results_idx, i)
            push!(matched_resp_idx, name_to_resp_idx[pname])
        end
    end

    # warn if >10% unmatched
    n_total = nrow(results)
    n_matched = length(matched_results_idx)
    if n_matched < n_total && (n_total - n_matched) / n_total > 0.10
        @warn "Within-class correlation: $(n_total - n_matched)/$(n_total) proteins unmatched by name"
    end

    isempty(matched_results_idx) && return "null"

    log_bf_e = _winsorize_quantile(log.(max.(_report_float.(results.bf_enrichment[matched_results_idx]), 1e-300)))
    log_bf_c = _winsorize_quantile(log.(max.(_report_float.(results.bf_correlation[matched_results_idx]), 1e-300)))
    log_bf_d = _winsorize_quantile(log.(max.(_report_float.(results.bf_detected[matched_results_idx]), 1e-300)))
    resp_matched = resp[matched_resp_idx, :]

    n_comp = size(resp_matched, 2)

    labels_3c = ["H0", "agnostic", "H1"]
    labels_2c = ["H0", "H1"]
    labels = n_comp == 3 ? labels_3c : labels_2c
    bf_names = ["enrichment", "correlation", "detection"]

    class_data = Pair{String,String}[]
    for c in 1:n_comp
        idx = findall(i -> argmax(resp_matched[i, :]) == c && resp_matched[i, c] > 0.7, 1:size(resp_matched, 1))
        length(idx) < 3 && continue  # skip small components

        # Compute pairwise Spearman correlations
        data_mat = hcat(log_bf_e[idx], log_bf_c[idx], log_bf_d[idx])
        corr_pairs = Pair{String,String}[]
        for i in 1:3
            for j in (i+1):3
                rho = _spearman_corr(data_mat[:, i], data_mat[:, j])
                push!(corr_pairs, "$(bf_names[i])_$(bf_names[j])" => json_number(isnan(rho) ? 0.0 : rho))
            end
        end

        # Scatter data for each pair
        scatter_pairs = Pair{String,String}[]
        push!(scatter_pairs, "log_bf_enrichment" => json_array([json_number(v) for v in log_bf_e[idx]]))
        push!(scatter_pairs, "log_bf_correlation" => json_array([json_number(v) for v in log_bf_c[idx]]))
        push!(scatter_pairs, "log_bf_detection" => json_array([json_number(v) for v in log_bf_d[idx]]))
        push!(scatter_pairs, "protein_names" => json_array([json_string(string(results.Protein[matched_results_idx[i]])) for i in idx]))

        push!(class_data, labels[c] => json_object(
            "n" => json_number(length(idx)),
            "correlations" => json_object(corr_pairs...),
            "scatter" => json_object(scatter_pairs...),
        ))
    end

    isempty(class_data) && return "null"
    return json_object(class_data...)
end

"""Simple Spearman rank correlation."""
function _spearman_corr(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n < 3 && return NaN
    rx = _rank(x)
    ry = _rank(y)
    d = rx .- ry
    return 1.0 - 6.0 * sum(d .^ 2) / (n * (n^2 - 1))
end

function _rank(x::AbstractVector)
    n = length(x)
    idx = sortperm(x)
    r = Vector{Float64}(undef, n)
    for (rank, i) in enumerate(idx)
        r[i] = Float64(rank)
    end
    return r
end

"""Approximate inverse normal CDF for Q-Q plots."""
function _qnorm(p::Float64)
    # Rational approximation (Abramowitz & Stegun 26.2.23)
    p <= 0.0 && return -Inf
    p >= 1.0 && return Inf
    p < 0.5 && return -_qnorm(1.0 - p)
    t = sqrt(-2.0 * log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1*t + c2*t^2) / (1.0 + d1*t + d2*t^2 + d3*t^3)
end

# ---------------------------------------------------------------------------
# KL Divergence: H1 Component Purity (interactive Plotly)
# ---------------------------------------------------------------------------

"""
Build KL divergence JSON for the diagnostics tab.

Reconstructs the KL(pure H1 || full H1) per dimension from the LatentClassResult
responsibilities and the per-protein log-BFs in results_df.
"""
function _build_kl_divergence_json(analysis_result, results_df)::String
    (results_df === nothing || !isa(results_df, DataFrame)) && return "null"

    # Filter to detected proteins only (non-detected have missing BFs)
    results_df = _filter_detected(results_df)

    has_e = hasproperty(results_df, :bf_enrichment)
    has_c = hasproperty(results_df, :bf_correlation)
    has_d = hasproperty(results_df, :bf_detected)
    (!has_e || !has_c || !has_d) && return "null"

    # Try to get H1 responsibilities from LatentClassResult
    h1_resp = nothing
    lc = _extract_lc_result(analysis_result)
    if lc !== nothing && isa(lc, LatentClassResult) && lc.responsibilities !== nothing
        n_comp = size(lc.responsibilities, 2)
        if n_comp >= 3
            h1_resp = lc.responsibilities[:, n_comp]  # last column = H1
        end
    end

    # Fallback: reconstruct approximate responsibilities from Component column
    if h1_resp === nothing && hasproperty(results_df, :Component)
        @debug "_build_kl_divergence_json: using Component column fallback"
        n = nrow(results_df)
        h1_resp = zeros(Float64, n)
        for i in 1:n
            comp = results_df.Component[i]
            if !ismissing(comp) && string(comp) == "H1"
                h1_resp[i] = 1.0
            end
        end
    end

    h1_resp === nothing && return "null"

    pure_threshold = 0.95
    pure_idx = findall(h1_resp .> pure_threshold)
    full_idx = findall(h1_resp .> 0.5)

    length(pure_idx) < 5 && return "null"
    length(full_idx) < 5 && return "null"

    log_bf_e = [log(max(_report_float(v), 1e-300)) for v in results_df.bf_enrichment]
    log_bf_c = [log(max(_report_float(v), 1e-300)) for v in results_df.bf_correlation]
    log_bf_d = [log(max(_report_float(v), 1e-300)) for v in results_df.bf_detected]

    dim_names = ["enrichment", "correlation", "detection"]
    log_bfs = [log_bf_e, log_bf_c, log_bf_d]

    kl_vals = Float64[]
    full_params = Pair{String,String}[]
    pure_params = Pair{String,String}[]

    rng = Random.MersenneTwister(42)
    for (i, dim) in enumerate(dim_names)
        full_data = log_bfs[i][full_idx]
        pure_data = log_bfs[i][pure_idx]

        full_mu, full_sigma = mean(full_data), std(full_data)
        pure_mu, pure_sigma = mean(pure_data), std(pure_data)
        full_median = median(full_data)
        pure_median = median(pure_data)
        full_sigma = max(full_sigma, 1e-6)
        pure_sigma = max(pure_sigma, 1e-6)

        # KL(pure || full) via Monte Carlo
        samples = pure_mu .+ pure_sigma .* randn(rng, 10_000)
        log_pure = -0.5 .* ((samples .- pure_mu) ./ pure_sigma).^2 .- log(pure_sigma)
        log_full = -0.5 .* ((samples .- full_mu) ./ full_sigma).^2 .- log(full_sigma)
        kl = max(0.0, mean(log_pure .- log_full))
        push!(kl_vals, kl)

        push!(full_params, dim => json_object(
            "mu" => json_number(full_mu),
            "sigma" => json_number(full_sigma),
            "median" => json_number(full_median),
        ))
        push!(pure_params, dim => json_object(
            "mu" => json_number(pure_mu),
            "sigma" => json_number(pure_sigma),
            "median" => json_number(pure_median),
        ))
    end

    return json_object(
        "dimensions" => json_array([json_string(d) for d in dim_names]),
        "kl_values" => json_array([json_number(v) for v in kl_vals]),
        "kl_total" => json_number(sum(kl_vals)),
        "full_h1_params" => json_object(full_params...),
        "pure_h1_params" => json_object(pure_params...),
        "n_pure" => json_number(length(pure_idx)),
        "n_full" => json_number(length(full_idx)),
        "threshold" => json_number(pure_threshold),
    )
end

# ---------------------------------------------------------------------------
# Copula Bootstrap Confidence (interactive Plotly)
# ---------------------------------------------------------------------------

"""
Build copula bootstrap JSON for the diagnostics tab.

Bootstrap CI data is not stored in analysis_result, so this only provides
BMA model weight information when available. The Plotly function renders
a summary of model selection confidence from the BMA weights.
"""
function _build_copula_bootstrap_json(analysis_result, results_df=nothing)::String
    analysis_result === nothing && return "null"

    # Filter to detected proteins only (matching pareto_k_values length)
    if results_df !== nothing && results_df isa DataFrame
        results_df = _filter_detected(results_df)
    end

    bma = nothing
    if hasproperty(analysis_result, :bma_result) && analysis_result.bma_result !== nothing
        bma = analysis_result.bma_result
    end
    bma === nothing && return "null"
    !isa(bma, BMAResult) && return "null"

    pairs = Pair{String,String}[
        "copula_weight" => json_number(bma.copula_weight),
        "em_weight" => json_number(bma.em_weight),
    ]

    # Pareto k-hat for diagnostic quality
    if bma.pareto_k !== nothing
        k = bma.pareto_k
        push!(pairs, "pareto_k_median" => json_number(median(k)))
        push!(pairs, "pareto_k_max" => json_number(maximum(k)))
        push!(pairs, "pareto_k_n_problematic" => json_number(count(x -> x > 0.7, k)))
        push!(pairs, "pareto_k_values" => json_array([json_number(v) for v in k]))
    end

    # Component assignments for strip plot coloring
    # Derive from responsibilities matrix (LatentClassResult has NO .assignments field)
    if bma.em3c_result !== nothing && bma.em3c_result.responsibilities !== nothing
        resp = bma.em3c_result.responsibilities
        n_resp = size(resp, 1)
        assignments = Vector{Int}(undef, n_resp)
        for j in 1:n_resp
            assignments[j] = argmax(resp[j, :])  # 1=H0, 2=Agnostic, 3=H1
        end
        push!(pairs, "component_assignments" => json_array([string(a) for a in assignments]))
    end

    # Protein names for hover tooltips (filtered to detected-only to match pareto_k_values length)
    if results_df !== nothing && results_df isa DataFrame && hasproperty(results_df, :Protein)
        n_results = nrow(results_df)
        # Only serialize if lengths align with pareto_k_values
        k_len = bma.pareto_k !== nothing ? length(bma.pareto_k) : 0
        if n_results == k_len || k_len == 0
            detected_names = [string(results_df.Protein[i]) for i in 1:n_results]
            push!(pairs, "protein_names" => json_array([json_string(n) for n in detected_names]))
        end
    end

    return json_object(pairs...)
end

# ---------------------------------------------------------------------------
# Discordant Protein Decomposition (interactive Plotly)
# ---------------------------------------------------------------------------

"""
Build discordant protein JSON for the diagnostics tab.

Decomposes combined BF into marginal (sum of individual log-BFs) and dependence
(copula/model) contributions. Discordant proteins have negative marginal evidence
but positive combined BF.
"""
function _build_discordant_json(analysis_result, results_df)::String
    lc = _extract_lc_result(analysis_result)
    (lc === nothing || !isa(lc, LatentClassResult)) && return "null"
    (results_df === nothing || !isa(results_df, DataFrame)) && return "null"

    # Filter to detected proteins only (non-detected have missing BFs)
    results_df = _filter_detected(results_df)

    has_e = hasproperty(results_df, :bf_enrichment)
    has_c = hasproperty(results_df, :bf_correlation)
    has_d = hasproperty(results_df, :bf_detected)
    has_bf = hasproperty(results_df, :BF)
    (!has_e || !has_c || !has_d || !has_bf) && return "null"

    n = nrow(results_df)
    log_bf_e = [log(max(_report_float(v), 1e-300)) for v in results_df.bf_enrichment]
    log_bf_c = [log(max(_report_float(v), 1e-300)) for v in results_df.bf_correlation]
    log_bf_d = [log(max(_report_float(v), 1e-300)) for v in results_df.bf_detected]
    combined_log_bf = [log(max(_report_float(v), 1e-300)) for v in results_df.BF]

    marginal_contrib = log_bf_e .+ log_bf_c .+ log_bf_d
    copula_contrib = combined_log_bf .- marginal_contrib

    discordant = (marginal_contrib .< 0) .& (combined_log_bf .> 0)
    disc_idx = findall(discordant)
    n_discordant = length(disc_idx)

    # Subsample for performance if > 2000 proteins
    scatter_idx = if n > 2000
        sort(collect(1:n)[sortperm(rand(n))[1:2000]])
    else
        1:n
    end

    protein_names = hasproperty(results_df, :Protein) ?
        json_array([json_string(string(results_df.Protein[i])) for i in scatter_idx]) : "[]"

    return json_object(
        "marginal_contrib" => json_array([json_number(marginal_contrib[i]) for i in scatter_idx]),
        "copula_contrib" => json_array([json_number(copula_contrib[i]) for i in scatter_idx]),
        "combined_log_bf" => json_array([json_number(combined_log_bf[i]) for i in scatter_idx]),
        "is_discordant" => json_array([(discordant[i] ? "true" : "false") for i in scatter_idx]),
        "protein_names" => protein_names,
        "n_discordant" => json_number(n_discordant),
        "discordant_fraction" => json_number(n > 0 ? n_discordant / n : 0.0),
        "n_total" => json_number(n),
    )
end

# ---------------------------------------------------------------------------
# Evidence data (interactive Plotly)
# ---------------------------------------------------------------------------

"""Build evidence data JSON for interactive Plotly plots."""
function _build_evidence_data_json(analysis_result, results_df)::String
    analysis_result === nothing && return "null"

    # Filter to detected proteins only (non-detected have missing BFs; exclude from plots)
    if results_df !== nothing && results_df isa DataFrame
        results_df = _filter_detected(results_df)
    end

    sections = Pair{String,String}[]

    # 1. EM convergence trace
    em = nothing
    if hasproperty(analysis_result, :em) && analysis_result.em !== nothing
        em = analysis_result.em
    end
    if em !== nothing && hasproperty(em, :logs) && em.logs !== nothing && nrow(em.logs) > 0
        logs = em.logs
        push!(sections, "em_convergence" => json_object(
            "iterations" => json_array([json_number(i) for i in 1:nrow(logs)]),
            "log_likelihood" => json_array([json_number(_report_float(logs.ll[i], NaN)) for i in 1:nrow(logs)]),
            "pi0" => hasproperty(logs, :pi0) ? json_array([json_number(_report_float(logs.pi0[i], NaN)) for i in 1:nrow(logs)]) : "[]",
            "pi1" => hasproperty(logs, :pi1) ? json_array([json_number(_report_float(logs.pi1[i], NaN)) for i in 1:nrow(logs)]) : "[]",
        ))
    end

    # 2. EM restart diagnostics — sourced from LatentClassResult.em_diagnostics
    em_diag = nothing
    lc_for_diag = _extract_lc_result(analysis_result)
    if lc_for_diag !== nothing && isa(lc_for_diag, LatentClassResult) &&
       lc_for_diag.em_diagnostics !== nothing
        em_diag = lc_for_diag.em_diagnostics
    end
    # Fallback: top-level em_diagnostics field on analysis_result (legacy path)
    if em_diag === nothing && hasproperty(analysis_result, :em_diagnostics) &&
       analysis_result.em_diagnostics !== nothing
        em_diag = analysis_result.em_diagnostics
    end
    if em_diag !== nothing && em_diag isa DataFrame && nrow(em_diag) > 0
        restart_rows = String[]
        for row in eachrow(em_diag)
            push!(restart_rows, json_object(
                "restart" => json_number(hasproperty(row, :restart) ? Int(row.restart) : 0),
                "init_pi0" => json_number(hasproperty(row, Symbol("init_π0")) ? _report_float(row.init_π0, NaN) : (hasproperty(row, :init_pi0) ? _report_float(row.init_pi0, NaN) : NaN)),
                "final_pi0" => json_number(hasproperty(row, Symbol("final_π0")) ? _report_float(row.final_π0, NaN) : (hasproperty(row, :final_pi0) ? _report_float(row.final_pi0, NaN) : NaN)),
                "final_pi1" => json_number(hasproperty(row, Symbol("final_π1")) ? _report_float(row.final_π1, NaN) : (hasproperty(row, :final_pi1) ? _report_float(row.final_pi1, NaN) : NaN)),
                "log_likelihood" => json_number(hasproperty(row, :log_likelihood) ? _report_float(row.log_likelihood, NaN) : NaN),
                "iterations" => json_number(hasproperty(row, :iterations) ? Int(row.iterations) : 0),
                "converged" => (hasproperty(row, :converged) && row.converged) ? "true" : "false",
                "init_method" => json_string(hasproperty(row, :init_method) && !ismissing(row.init_method) ? string(row.init_method) : "unknown"),
                "h1_family" => json_string(hasproperty(row, :h1_family_selected) && !ismissing(row.h1_family_selected) ? string(row.h1_family_selected) : "unknown"),
                "bic_gamma" => json_number(hasproperty(row, :h1_bic_gamma) ? _report_float(row.h1_bic_gamma, NaN) : NaN),
                "bic_lognormal" => json_number(hasproperty(row, :h1_bic_lognormal) ? _report_float(row.h1_bic_lognormal, NaN) : NaN),
                "bic_weibull" => json_number(hasproperty(row, :h1_bic_weibull) ? _report_float(row.h1_bic_weibull, NaN) : NaN),
                "n_reverts"      => json_number(hasproperty(row, :n_step_halving_reverts) && !ismissing(row.n_step_halving_reverts) ? Int(row.n_step_halving_reverts) : 0),
                "h0_nu"          => json_number(hasproperty(row, :h0_nu) && !ismissing(row.h0_nu) ? _report_float(row.h0_nu, NaN) : NaN),
                "anneal_T_final" => json_number(hasproperty(row, :anneal_T_final) && !ismissing(row.anneal_T_final) ? _report_float(row.anneal_T_final, NaN) : NaN),
                "bimodality_bc"  => json_number(hasproperty(row, :bimodality_bc) && !ismissing(row.bimodality_bc) ? _report_float(row.bimodality_bc, NaN) : NaN),
            ))
        end
        push!(sections, "em_restarts" => json_array(restart_rows))
    end

    # 2b. Merge diagnostic note
    lc_merge = _extract_lc_result(analysis_result)
    if lc_merge !== nothing && isa(lc_merge, LatentClassResult) &&
       hasproperty(lc_merge, :merged) && lc_merge.merged
        kl_str = hasproperty(lc_merge, :kl_divergence) ? string(round(lc_merge.kl_divergence, digits=4)) : "N/A"
        push!(sections, "merge_note" => json_string(
            "H0-Agnostic merge applied (KL = $kl_str). " *
            "The Agnostic component was absorbed into H0. P(agn|data) = 0 for all proteins."
        ))
    end

    # 3. Marginal fit data (H0 and H1 component fits per dimension)
    _add_marginal_fit_json!(sections, analysis_result, results_df)

    # 4. All-restart convergence traces (from LatentClassResult)
    lc = _extract_lc_result(analysis_result)
    if lc !== nothing && lc.all_restart_traces !== nothing && !isempty(lc.all_restart_traces)
        traces = lc.all_restart_traces
        final_lls = [isempty(t) ? -Inf : t[end] for t in traces]
        best_idx = argmax(final_lls)

        # include per-step LL traces if available
        has_per_step = lc.per_step_ll_traces !== nothing && length(lc.per_step_ll_traces) == length(traces)

        restart_jsons = String[]
        for (i, trace) in enumerate(traces)
            fields = Pair{String,String}[
                "trace" => json_array([json_number(v) for v in trace]),
                "is_best" => (i == best_idx ? "true" : "false"),
            ]
            if has_per_step
                pst = lc.per_step_ll_traces[i]
                push!(fields, "trace_e" => json_array([json_number(v) for v in pst.ll_after_e]))
                push!(fields, "trace_m" => json_array([json_number(v) for v in pst.ll_after_m]))
            end
            # Add family_switch_iter if available from em_diagnostics
            if lc.em_diagnostics !== nothing && hasproperty(lc.em_diagnostics, :family_switch_iter) && i <= nrow(lc.em_diagnostics)
                fsi = lc.em_diagnostics[i, :family_switch_iter]
                push!(fields, "family_switch" => (ismissing(fsi) ? "null" : json_number(fsi)))
            end
            push!(restart_jsons, json_object(fields...))
        end
        push!(sections, "em_convergence_all" => json_array(restart_jsons))
    end

    # 5. EB diagnostics — only when EB was used (effective_alpha_prior non-empty)
    if lc !== nothing
        _eap = try; lc.effective_alpha_prior; catch; Float64[]; end
        if !isempty(_eap)
            eb_pairs = Pair{String,String}[
                "effective_alpha" => json_array([json_number(a) for a in _eap]),
                "eb_converged" => (try; lc.eb_converged; catch; false; end ? "true" : "false"),
            ]
            _pgw = try; lc.prior_grid_weights; catch; nothing; end
            if _pgw !== nothing
                push!(eb_pairs, "prior_grid_weights" => json_array([json_number(w) for w in _pgw]))
            end
            _pgp = try; lc.prior_grid_posteriors; catch; nothing; end
            if _pgp !== nothing
                push!(eb_pairs, "n_grid_points" => json_number(length(_pgp)))
            end
            # Stability counts from results_df
            if results_df !== nothing && results_df isa DataFrame && hasproperty(results_df, :classification_stability)
                stab_col = skipmissing(results_df.classification_stability)
                n_robust = count(==("robust"), stab_col)
                n_sensitive = count(==("sensitive"), stab_col)
                n_fragile = count(==("fragile"), stab_col)
                push!(eb_pairs, "stability_counts" => json_object(
                    "robust" => json_number(n_robust),
                    "sensitive" => json_number(n_sensitive),
                    "fragile" => json_number(n_fragile),
                ))
            end
            # Ternary plot data: grid proportions and per-grid log-likelihoods
            _grid = try; build_prior_grid(_eap); catch; nothing; end
            if _grid !== nothing && length(_grid) > 0
                # Proportions: normalize each grid point to sum=1
                _grid_props = [json_array([json_number(g[i] / sum(g)) for i in 1:length(g)]) for g in _grid]
                push!(eb_pairs, "prior_grid_proportions" => json_array(_grid_props))

                # Labels for tooltip display
                _labels = ["EB center", "Uniform", "Push H0", "Push Agnostic", "Push H1",
                            "H0+Agnostic", "H0+H1", "Agnostic+H1", "Strong H0"]
                _n = min(length(_grid), length(_labels))
                push!(eb_pairs, "prior_grid_labels" => json_array([json_string(_labels[i]) for i in 1:_n]))
            end

            if _pgw !== nothing && length(_pgw) > 0
                # Use log(weights) as proxy for relative log-likelihood (color axis)
                _ll_proxy = [json_number(w > 0 ? log(w) : -30.0) for w in _pgw]
                push!(eb_pairs, "prior_grid_ll" => json_array(_ll_proxy))
            end

            push!(sections, "eb_diagnostics" => json_object(eb_pairs...))
        end
    end

    isempty(sections) && return "null"
    return json_object(sections...)
end

# defensive instrumentation for the responsibility-matrix
# row-count contract enforced by `_reconstruct_lc_responsibilities`.
# An earlier path emitted a `@warn "responsibility matrix row count (X) != results_df rows (Y); falling back to P_H0/P_agnostic/P_H1"`
# on every condition of the HTT/HAP40 run; the gap pattern
# (off-by-one for wtHTT/mHTT, +1083 for HAP40_Strep, +210 for GST_HAP40) did NOT correlate
# with the BetaBernoulli `n_sample_obs < 2 || n_control_obs < 2` exclusion count, ruling out
# a "suppress on expected branch" mechanism. The current path
# structurally resolves the warning by reconstructing the matrix row-wise to
# `nrow(results_df)`; the @warn was removed.
#
# This helper exists for forward-compat: if a future change re-introduces a row-count gap
# (e.g. someone bypasses `_reconstruct_lc_responsibilities`), the per-condition `@debug`
# block inside `_add_marginal_fit_json!` (below) logs the gap together with the
# BetaBernoulli-exclusion count returned here so post-mortems can distinguish the
# expected vs unexpected branch without re-running the pipeline.
#
# Returns:
# - the exclusion count as `Int >= 0` when the analysis_result (or the results_df itself)
#   exposes `n_sample_obs` + `n_control_obs` columns; this is rare in normal use because
#   BB-excluded proteins are filtered out at `load_data` time and never enter results_df.
# - `-1` (sentinel "unknown") otherwise. The instrumentation @debug branch treats
#   `n_excluded < 0` as "no expected-gap reference available — report raw counts only".
function _count_betabernoulli_excluded(analysis_result, results_df)
    # Try column names that may appear on results_df or on an attached exclusion report.
    candidate_pairs = ((:n_sample_obs, :n_control_obs),
                       (:n_sample, :n_control),
                       (:sample_count, :control_count))
    # WR-02: defensive shielding. The previous body used `Int(s) < 2`
    # which would throw `InexactError` for any non-integer numeric (e.g., Float64
    # column with fractional values). The probed column names are not
    # contract-locked, so a future contributor adding a `:sample_count` with
    # Float64 eltype would crash this instrumentation helper instead of falling
    # back to the `-1` sentinel. Wrap in try/catch and use `floor(Int, s)` with
    # explicit `s isa Real && isfinite(s)` checks.
    for (s_col, c_col) in candidate_pairs
        if hasproperty(results_df, s_col) && hasproperty(results_df, c_col)
            try
                return count(zip(getproperty(results_df, s_col), getproperty(results_df, c_col))) do (s, c)
                    (ismissing(s) || ismissing(c)) && return false
                    (!(s isa Real) || !(c isa Real)) && return false
                    (!isfinite(s) || !isfinite(c)) && return false
                    floor(Int, s) < 2 || floor(Int, c) < 2
                end
            catch
                # Any conversion / arithmetic failure → fall back to sentinel.
                return -1
            end
        end
    end
    # Probe the analysis_result for an `exclusion_report` field carrying excluded proteins.
    if analysis_result !== nothing
        for fname in (:exclusion_report, :excluded_proteins, :bb_exclusion_report)
            if hasproperty(analysis_result, fname)
                rep = getproperty(analysis_result, fname)
                rep === nothing && continue
                # rep is typically a DataFrame from `filter_insufficient_observations`.
                if applicable(size, rep) && applicable(nrow, rep)
                    # WR-02: guard against non-Int return from nrow
                    # (e.g., a future custom DataFrame-like with BigInt nrow).
                    try
                        return Int(nrow(rep))
                    catch
                        return -1
                    end
                end
            end
        end
    end
    return -1
end

"""
    _reconstruct_lc_responsibilities(results_df, lc) -> Matrix{Float64}

row-wise reconstruction of the 3-component responsibility matrix
used by `_add_marginal_fit_json!`. Replaces the prior all-or-nothing row-count guard
that silently dropped the canonical LC-direct path whenever LC was
fitted on a filtered subset of proteins.

Priority chain (applied row-by-row, by `results_df.Protein[i]`):

1. **LC hit** — when the protein name appears in `lc.protein_names`, the row is
   copied from `lc.responsibilities` (the LC model's own view; preferred per
   Bug 2).
2. **P_* fallback** — when the protein is NOT in `lc.protein_names` but the
   results DataFrame carries `P_H0`/`P_agnostic`/`P_H1` columns (always
   populated post-BMA per `pipeline.jl:903-905, 1534-1536`), the row reads
   `[P_H0, P_agnostic, P_H1]` with `missing` substituted as `0.0`.
3. **`:Component` one-hot fallback** — when neither LC
   nor any non-missing P_* values are available, but `results_df` carries a
   `:Component` column (legitimate post-BMA path), emit a one-hot row keyed
   by the component label ("H0" / "Agnostic" / "H1"). Matches the legacy
   third-tier behaviour in `_add_marginal_fit_json!` that the prior row-wise
   reconstruction had inadvertently dropped — a regression that would
   silently mis-classify into the H0 marginal panel because
   `argmax([0, 0, 0]) == 1`.
4. **Zero-row terminal** — when none of the above produce a row, the row
   stays at `[0.0, 0.0, 0.0]` (the array is zero-initialised).

# Preconditions
- `lc.responsibilities !== nothing` (caller's responsibility — `_add_marginal_fit_json!`
  guards this before invoking the helper).
- `results_df` carries a `:Protein` column convertible via `string()`.
- `lc.protein_names` is iterable as a vector of names convertible via `string()`.

# Returns
`Matrix{Float64}` of size `(nrow(results_df), size(lc.responsibilities, 2))`.

The helper accepts any object exposing `.protein_names` and `.responsibilities`
(duck-typed) — a real `LatentClassResult` AND any test stand-in `NamedTuple` both work.

The `@debug` line records `n_lc_hits / n_p_fallback / n_zero` counts; set
`JULIA_DEBUG=BayesInteractomics` to surface it.
"""
function _reconstruct_lc_responsibilities(results_df, lc)
    n = nrow(results_df)
    n_components = size(lc.responsibilities, 2)
    responsibilities = zeros(Float64, n, n_components)
    # Build name → row-in-LC map ONCE (avoids O(n × n_lc) scan)
    name_to_lc_idx = Dict{String, Int}(string(lc.protein_names[i]) => i for i in eachindex(lc.protein_names))
    has_p_cols = hasproperty(results_df, :P_H0) && hasproperty(results_df, :P_agnostic) && hasproperty(results_df, :P_H1)
    # WR-06: legacy third-tier `:Component` one-hot fallback.
    has_component_col = hasproperty(results_df, :Component)
    n_lc_hits = 0
    n_p_fallback = 0
    n_one_hot = 0
    n_zero = 0
    for i in 1:n
        pname = string(results_df.Protein[i])
        if haskey(name_to_lc_idx, pname)
            responsibilities[i, :] .= lc.responsibilities[name_to_lc_idx[pname], :]
            n_lc_hits += 1
        elseif has_p_cols && n_components == 3 &&
               !(ismissing(results_df.P_H0[i]) && ismissing(results_df.P_agnostic[i]) && ismissing(results_df.P_H1[i]))
            # P_* fallback: at least one P_* value is non-missing.
            responsibilities[i, 1] = ismissing(results_df.P_H0[i]) ? 0.0 : Float64(results_df.P_H0[i])
            responsibilities[i, 2] = ismissing(results_df.P_agnostic[i]) ? 0.0 : Float64(results_df.P_agnostic[i])
            responsibilities[i, 3] = ismissing(results_df.P_H1[i]) ? 0.0 : Float64(results_df.P_H1[i])
            n_p_fallback += 1
        elseif has_component_col && !ismissing(results_df.Component[i])
            # legacy third-tier one-hot from `:Component`.
            # Restores the legacy fallback that was inadvertently dropped during
            # the row-wise rewrite. Without this tier a row with a known
            # Component label but no P_* values would land at [0,0,0] and
            # silently mis-classify into the H0 marginal panel because
            # argmax([0, 0, 0]) == 1.
            comp_str = string(results_df.Component[i])
            if comp_str == "H0"
                responsibilities[i, 1] = 1.0
                n_one_hot += 1
            elseif comp_str == "Agnostic" && n_components >= 3
                responsibilities[i, 2] = 1.0
                n_one_hot += 1
            elseif comp_str == "H1"
                # H1 lives in the last column (col 3 for 3-component, col 2 for 2-component).
                responsibilities[i, n_components] = 1.0
                n_one_hot += 1
            else
                # Unrecognised component string — fall through to zero-row.
                n_zero += 1
            end
        else
            n_zero += 1
            # row stays [0, 0, 0]
        end
    end
    @debug "_reconstruct_lc_responsibilities: $n_lc_hits LC-direct / $n_p_fallback BMA-fallback / $n_one_hot Component-one-hot / $n_zero zero-row (n=$n, n_components=$n_components)"
    return responsibilities
end

"""Build marginal fit JSON for 3-component (H0/Agnostic/H1) density overlays per dimension.

Emits a `marginal_fits` section with uppercase component keys "H0", "Agnostic", "H1".
Each component has "enrichment", "correlation", "detection" sub-keys with histogram
data, fitted density curves, and panel metadata (title, n).

H1 enrichment uses the BIC-selected LocationShifted{T} density (Gamma/LogNormal/Weibull).
All other 8 panels use Normal(mu_w, sigma_w) density from weighted EM parameters.
"""
function _add_marginal_fit_json!(sections::Vector{Pair{String,String}}, analysis_result, results_df)
    results_df === nothing && return

    # Extract log-BF values per dimension
    has_e = hasproperty(results_df, :bf_enrichment)
    has_c = hasproperty(results_df, :bf_correlation)
    has_d = hasproperty(results_df, :bf_detected)
    (!has_e || !has_c || !has_d) && return

    log_bf_e = _winsorize_quantile([log(max(_report_float(v), 1e-300)) for v in results_df.bf_enrichment])
    log_bf_c = _winsorize_quantile([log(max(_report_float(v), 1e-300)) for v in results_df.bf_correlation])
    log_bf_d = _winsorize_quantile([log(max(_report_float(v), 1e-300)) for v in results_df.bf_detected])

    n = nrow(results_df)

    # PRIMARY source is LC model's own responsibilities (Bug 2 fix).
    # P_H0/P_agnostic/P_H1 columns are BMA-weighted and may not reflect the LC
    # model's own class view. Use lc.responsibilities first when available.
    responsibilities = nothing
    n_components = 0
    lc = _extract_lc_result(analysis_result)

    if lc !== nothing && isa(lc, LatentClassResult) && lc.responsibilities !== nothing
        # PRIMARY: row-wise reconstruction.
        # Preserves the Bug 2 LC-direct path for the rows where LC fitted the protein;
        # falls back to BMA-weighted P_H0/P_agnostic/P_H1 row-wise for non-fitted rows.
        @debug "_add_marginal_fit_json!: row-wise reconstruction"
        responsibilities = _reconstruct_lc_responsibilities(results_df, lc)
        n_components = size(responsibilities, 2)

        # defensive row-count invariant + BB-exclusion-aware
        # telemetry. _reconstruct_lc_responsibilities is contracted to return a matrix
        # with `nrow(results_df)` rows; this block surfaces gaps if a future change
        # re-introduces them and classifies them as expected (BB-exclusion-driven) vs
        # structural (everything else). The original observed gaps were
        # structural, NOT BB-exclusion — so the structural branch
        # is the one operators should investigate when it fires.
        n_results = size(results_df, 1)
        n_resp = size(responsibilities, 1)
        if n_resp != n_results
            n_excluded = _count_betabernoulli_excluded(analysis_result, results_df)
            actual_gap = n_results - n_resp
            if n_excluded >= 0 && actual_gap == n_excluded
                # Expected BetaBernoulli-exclusion branch — P_H0/P_agnostic/P_H1 fallback
                # is correct behaviour; downgrade noise to @debug.
                @debug "_add_marginal_fit_json!: responsibility matrix row count ($(n_resp)) != results_df rows ($(n_results)); gap=$(actual_gap) == BetaBernoulli-exclusion count ($(n_excluded)) — expected fallback to P_H0/P_agnostic/P_H1 columns"
            else
                @warn "_add_marginal_fit_json!: responsibility matrix row count ($(n_resp)) != results_df rows ($(n_results)); gap=$(actual_gap), BetaBernoulli-excluded=$(n_excluded) — UNEXPECTED gap mismatch (gap != exclusion count or count unknown); falling back to P_H0/P_agnostic/P_H1 columns" maxlog=1
            end
        end
    end
    if responsibilities === nothing && hasproperty(results_df, :P_H0) && hasproperty(results_df, :P_agnostic) && hasproperty(results_df, :P_H1)
        # FALLBACK: P_H0/P_agnostic/P_H1 columns (BMA-weighted, but better than nothing)
        @debug "_add_marginal_fit_json!: falling back to P_H0/P_agnostic/P_H1 columns"
        n_components = 3
        responsibilities = zeros(Float64, n, n_components)
        for i in 1:n
            responsibilities[i, 1] = ismissing(results_df.P_H0[i]) ? 0.0 : Float64(results_df.P_H0[i])
            responsibilities[i, 2] = ismissing(results_df.P_agnostic[i]) ? 0.0 : Float64(results_df.P_agnostic[i])
            responsibilities[i, 3] = ismissing(results_df.P_H1[i]) ? 0.0 : Float64(results_df.P_H1[i])
        end
    elseif responsibilities === nothing && hasproperty(results_df, :Component)
        # FALLBACK: one-hot from Component column (loses "Uncertain" proteins)
        @debug "_add_marginal_fit_json!: using Component column (one-hot, sort-aligned)"
        n_components = 3
        responsibilities = zeros(Float64, n, n_components)
        for i in 1:n
            comp = results_df.Component[i]
            ismissing(comp) && continue
            comp_str = string(comp)
            if comp_str == "H0"
                responsibilities[i, 1] = 1.0
            elseif comp_str == "Agnostic"
                responsibilities[i, 2] = 1.0
            elseif comp_str == "H1"
                responsibilities[i, 3] = 1.0
            end
        end
    end

    (responsibilities === nothing || n_components < 2) && return

    # Assign proteins to components by MAP (every protein in exactly one panel)
    comp_idx = [argmax(responsibilities[i, :]) for i in 1:n]

    # 3-component configuration: (label, component_index, is_h1)
    # For 2-component fallback, Agnostic maps to comp_idx=2=H1 so comp 2 becomes H1
    comp_configs = if n_components >= 3
        [("H0", 1, false), ("Agnostic", 2, false), ("H1", n_components, true)]
    else
        [("H0", 1, false), ("H1", n_components, true)]
    end

    # Read H1 enrichment family info from LatentClassResult
    alpha_h1_global::Float64 = 2.0
    theta_h1_global::Float64 = 2.0
    selected_h1_family_global::Symbol = :gamma
    if lc !== nothing && isa(lc, LatentClassResult)
        alpha_h1_global = lc.alpha_enrichment_h1
        theta_h1_global = lc.theta_enrichment_h1
        selected_h1_family_global = lc.h1_enrichment_family
    end

    # Extract DiscreteEmpirical distributions for detection panels from LatentClassResult.
    # These were fitted during EM and stored in the result.
    # Indexed by component index (1=H0, 2=Agnostic, 3=H1).
    disc_dists = Dict{Int,Union{Nothing,DiscreteEmpirical}}(
        1 => (lc !== nothing && isa(lc, LatentClassResult) ? lc.disc_detection_H0 : nothing),
        2 => (lc !== nothing && isa(lc, LatentClassResult) ? lc.disc_detection_ag : nothing),
        3 => (lc !== nothing && isa(lc, LatentClassResult) ? lc.disc_detection_H1 : nothing),
    )

    # Dimension config: (name, log_bf_vector, display_label)
    dims = [
        ("enrichment",  log_bf_e, "Enrichment"),
        ("correlation", log_bf_c, "Correlation"),
        ("detection",   log_bf_d, "Detection"),
    ]

    # Build per-component JSON objects
    comp_json_pairs = Pair{String,String}[]

    for (comp_label, comp_i, is_h1_comp) in comp_configs
        mask = comp_idx .== comp_i
        dim_sections = Pair{String,String}[]

        for (dim_name, log_bf_vals, dim_display) in dims
            comp_data = log_bf_vals[mask]
            n_comp = length(comp_data)
            @debug "Marginal fit $comp_label/$dim_name: $n_comp MAP-assigned proteins"

            # Weighted statistics using responsibilities (soft weights from ALL proteins,
            # not just MAP-assigned — valid even if few proteins are MAP-assigned to this component)
            w = responsibilities[:, comp_i]
            w_sum = sum(w)
            if w_sum < 1e-10
                @debug "Marginal fit $comp_label/$dim_name: skipping (w_sum < 1e-10)"
                continue
            end
            mu_w = sum(w .* log_bf_vals) / w_sum
            sigma_w = sqrt(sum(w .* (log_bf_vals .- mu_w).^2) / w_sum)
            sigma_w = max(sigma_w, 0.01)  # floor

            # --- DISCRETE DETECTION PANEL PATH ---
            # Detection panels use DiscreteEmpirical fitted during EM (if available).
            if dim_name == "detection"
                disc = get(disc_dists, comp_i, nothing)
                panel_title_disc = "$comp_label $dim_display (Discrete, N=$n_comp)"
                if disc !== nothing && !isempty(disc.values)
                    push!(dim_sections, dim_name => json_object(
                        "hist_values"  => json_array([json_number(v) for v in comp_data]),
                        "disc_values"  => json_array([json_number(v) for v in disc.values]),
                        "disc_probs"   => json_array([json_number(p) for p in disc.probs]),
                        "is_discrete"  => "true",
                        "title"        => json_string(panel_title_disc),
                        "n"            => json_number(n_comp),
                    ))
                else
                    # Fallback: old result without DiscreteEmpirical — use Normal density
                    panel_title_fallback = "$comp_label $dim_display (N=$n_comp)"
                    fb_min = n_comp >= 1 ? minimum(comp_data) - 1.0 : mu_w - 4 * sigma_w
                    fb_max = n_comp >= 1 ? maximum(comp_data) + 1.0 : mu_w + 4 * sigma_w
                    fit_x_fb = range(fb_min, fb_max, length=200)
                    fit_y_fb = [exp(-0.5 * ((x - mu_w) / sigma_w)^2) / (sigma_w * sqrt(2*pi)) for x in fit_x_fb]
                    push!(dim_sections, dim_name => json_object(
                        "hist_values" => json_array([json_number(v) for v in comp_data]),
                        "fit_x"       => json_array([json_number(v) for v in fit_x_fb]),
                        "fit_y"       => json_array([json_number(v) for v in fit_y_fb]),
                        "dist_label"  => json_string("Normal($(round(mu_w, digits=3)), $(round(sigma_w, digits=3)))"),
                        "title"       => json_string(panel_title_fallback),
                        "n"           => json_number(n_comp),
                        "mu"          => json_number(mu_w),
                        "sigma"       => json_number(sigma_w),
                    ))
                end
                continue
            end

            # --- CONTINUOUS PANEL PATH (enrichment, correlation) ---

            # Determine if this is the H1 enrichment panel (uses LocationShifted{T})
            is_shifted_h1 = is_h1_comp && dim_name == "enrichment"

            # H1 enrichment parameters (only set for the shifted panel)
            local alpha_h1::Float64 = alpha_h1_global
            local theta_h1::Float64 = theta_h1_global
            local selected_h1_family::Symbol = selected_h1_family_global

            if is_shifted_h1 && lc === nothing
                # Fallback: fit Gamma from positive-shifted data
                shifted_data = log_bf_vals[log_bf_vals .> JEFFREYS_SHIFT] .- JEFFREYS_SHIFT
                if length(shifted_data) >= 5
                    gfit = Distributions.fit(Gamma, shifted_data)
                    alpha_h1 = clamp(shape(gfit), 0.5, 50.0)
                    theta_h1 = clamp(scale(gfit), 0.05, 20.0)
                end
                selected_h1_family = :gamma
            end

            # Generate density curve (200 points)
            # Use MAP-assigned data range if available, otherwise fall back to weighted mu ± 4σ
            data_min = n_comp >= 1 ? minimum(comp_data) - 1.0 : mu_w - 4 * sigma_w
            data_max = n_comp >= 1 ? maximum(comp_data) + 1.0 : mu_w + 4 * sigma_w
            n_pts = 200
            fit_x = range(data_min, data_max, length=n_pts)
            fit_y = Float64[]

            for x in fit_x
                if is_shifted_h1
                    # LocationShifted{T}: zero density below JEFFREYS_SHIFT, family-dispatched above
                    if x >= JEFFREYS_SHIFT
                        xs = x - JEFFREYS_SHIFT
                        density = if selected_h1_family == :lognormal
                            pdf(LogNormal(alpha_h1, theta_h1), xs)
                        elseif selected_h1_family == :weibull
                            pdf(Weibull(alpha_h1, theta_h1), xs)
                        else  # :gamma (default)
                            pdf(Gamma(alpha_h1, theta_h1), xs)
                        end
                        push!(fit_y, density)
                    else
                        push!(fit_y, 0.0)
                    end
                else
                    # Normal(mu_w, sigma_w) density for all non-H1-enrichment panels
                    z = (x - mu_w) / sigma_w
                    push!(fit_y, exp(-0.5 * z^2) / (sigma_w * sqrt(2*pi)))
                end
            end

            # Scale H1 density by fraction of data above JEFFREYS_SHIFT
            # The histogram normalizes ALL comp_data to integrate to 1 (probability density),
            # but the shifted PDF only has support on [JEFFREYS_SHIFT, inf).
            # Scale PDF by the fraction of data in its support region.
            # Threat T-58-06: guard against division by zero
            if is_shifted_h1 && !isempty(comp_data)
                n_above = count(x -> x >= JEFFREYS_SHIFT, comp_data)
                frac_above = n_above / length(comp_data)
                if frac_above > 0 && frac_above < 1
                    fit_y .*= frac_above
                end
            end

            # Build distribution label and panel title
            dist_label = if is_shifted_h1
                fam_str = uppercasefirst(string(selected_h1_family))
                "H1 Enrichment Marginal ($fam_str): $(fam_str)($(round(alpha_h1, digits=2)), $(round(theta_h1, digits=2))) + $(round(JEFFREYS_SHIFT, digits=3))"
            else
                "Normal($(round(mu_w, digits=3)), $(round(sigma_w, digits=3)))"
            end

            panel_title = if is_shifted_h1
                fam_str = uppercasefirst(string(selected_h1_family))
                "H1 $dim_display ($fam_str, N=$n_comp)"
            else
                "$comp_label $dim_display (N=$n_comp)"
            end

            push!(dim_sections, dim_name => json_object(
                "hist_values" => json_array([json_number(v) for v in comp_data]),
                "fit_x" => json_array([json_number(v) for v in fit_x]),
                "fit_y" => json_array([json_number(v) for v in fit_y]),
                "dist_label" => json_string(dist_label),
                "title" => json_string(panel_title),
                "n" => json_number(n_comp),
                "mu" => json_number(mu_w),
                "sigma" => json_number(sigma_w),
            ))
        end

        if !isempty(dim_sections)
            push!(comp_json_pairs, comp_label => json_object(dim_sections...))
        end
    end

    if !isempty(comp_json_pairs)
        push!(sections, "marginal_fits" => json_object(comp_json_pairs...))
    end

    # Add LC-specific responsibilities to JSON for JS-side class coloring
    if lc !== nothing && isa(lc, LatentClassResult) && lc.responsibilities !== nothing
        resp_mat = lc.responsibilities
        if size(resp_mat, 2) >= 3 && size(resp_mat, 1) == n
            push!(sections, "lc_p_h0" => json_array([json_number(resp_mat[i, 1]) for i in 1:n]))
            push!(sections, "lc_p_agnostic" => json_array([json_number(resp_mat[i, 2]) for i in 1:n]))
            push!(sections, "lc_p_h1" => json_array([json_number(resp_mat[i, 3]) for i in 1:n]))
        end
    end

    # BIC selection summary (if available from LatentClassResult)
    if lc !== nothing && isa(lc, LatentClassResult)
        bic_scores = lc.h1_bic_scores
        selected_fam = lc.h1_enrichment_family
        if !all(isinf, values(bic_scores))
            bic_entries = [
                json_object(
                    "family" => json_string(string(k)),
                    "bic" => json_number(isfinite(v) ? v : 1e308),
                    "selected" => (k == selected_fam ? "true" : "false")
                )
                for (k, v) in sort(collect(bic_scores), by=x -> get(bic_scores, x[1], Inf))
            ]
            push!(sections, "h1_bic_selection" => json_object(
                "selected_family" => json_string(string(selected_fam)),
                "scores" => json_array(bic_entries)
            ))
        end
    end
end

"""Extract LatentClassResult from analysis_result (BMA or direct).

Handles both live objects and JLD2-deserialized structs where custom `getproperty`
may not dispatch correctly. Falls back to `getfield` when property access fails.
"""
function _extract_lc_result(analysis_result)
    analysis_result === nothing && return nothing

    # Helper: safely get a field, trying getproperty then getfield
    _safe_get(obj, field) = try
        hasproperty(obj, field) ? getproperty(obj, field) : nothing
    catch
        try; getfield(obj, field); catch; nothing; end
    end

    # Try direct latent_class_result field
    lc = _safe_get(analysis_result, :latent_class_result)
    @debug "_extract_lc_result: direct latent_class_result = $(lc !== nothing ? typeof(lc) : nothing)"
    lc !== nothing && isa(lc, LatentClassResult) && return lc

    # Try BMA sub-result
    bma = _safe_get(analysis_result, :bma_result)
    @debug "_extract_lc_result: bma_result = $(bma !== nothing ? typeof(bma) : nothing)"
    bma === nothing && return nothing

    # Check em3c_result first (BMA default path), then latent_class_result
    for field in (:em3c_result, :latent_class_result)
        sub = _safe_get(bma, field)
        @debug "_extract_lc_result: bma.$field = $(sub !== nothing ? typeof(sub) : nothing)"
        sub !== nothing && isa(sub, LatentClassResult) && return sub
    end
    return nothing
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
