#!/usr/bin/env julia
#=
Per-column dropout-curve fit CLI wrapper.

Reads a raw `dataset.xlsx` (rows = proteins, columns = MS runs), fits one
logistic dropout curve `p_detect = σ(ρ_c + ζ_c · ȳ_i)` per column via GLM.jl,
and writes:
  <output_dir>/dropout_curves.json                        — curve schema
  <output_dir>/dropout_diagnostics/col_NN_<name>.png × N  — per-column plot
  <output_dir>/dropout_diagnostics/all_sigmoids.png       — overlay plot
  <output_dir>/dropout_diagnostics/zeta_distribution.png  — zeta distribution plot
  <output_dir>/dropout_diagnostics/SANITY.md              — sanity report

Usage:
    julia --project=. --threads=4 scripts/fit_dropout.jl <dataset.xlsx> [output_dir]
        [--log-transform=auto|true|false] [--allow-imperfect-fit]

`output_dir` defaults to `<dirname(dataset.xlsx)>/imputed_data/`.

`--log-transform=auto` (default) inspects the value distribution:
  - max(values) < 100  → data is already log-transformed → log_transform=false
  - otherwise          → apply log2 → log_transform=true
This guards against double-log when input is already an Abundance / log2-LFQ matrix.

`--allow-imperfect-fit` downgrades the <50% well-fit catastrophic SANITY error to a
warning. Use when fitting heterogeneous AP-MS datasets where some columns reflect
already-normalized abundances (ζ̂ ≈ 0) that the dropout-curve model intentionally
does not characterize. Curves are still written to disk in either mode.

Note (Julia 1.12 / Windows): use explicit `--threads=4`, never the auto thread mode.
=#

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BayesInteractomics
using Statistics: maximum

# ---------------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------------

function _parse_args(args::Vector{String})
    positional = String[]
    log_transform_mode = "auto"  # auto | true | false
    allow_imperfect_fit = false
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--log-transform=")
            log_transform_mode = String(a[length("--log-transform=")+1:end])
            (log_transform_mode in ("auto", "true", "false")) ||
                error("--log-transform must be one of: auto, true, false")
            i += 1
        elseif a == "--allow-imperfect-fit"
            allow_imperfect_fit = true
            i += 1
        else
            push!(positional, a)
            i += 1
        end
    end
    return positional, log_transform_mode, allow_imperfect_fit
end

positional, log_transform_mode, allow_imperfect_fit = _parse_args(ARGS)

if length(positional) < 1
    error("usage: julia --project=. scripts/fit_dropout.jl <dataset.xlsx> [output_dir] " *
          "[--log-transform=auto|true|false] [--allow-imperfect-fit]\n\n" *
          "  <dataset.xlsx>          Path to the raw intensity matrix (sheet 'Sheet1', col 1 = protein IDs).\n" *
          "  [output_dir]            Optional. Defaults to <dirname(dataset.xlsx)>/imputed_data/\n" *
          "  --log-transform=MODE    auto (default) | true | false. auto detects if input is already log2.\n" *
          "  --allow-imperfect-fit   Downgrade SANITY <50% well-fit error to warning.")
end

xlsx_path  = positional[1]
output_dir = length(positional) >= 2 ? positional[2] :
             joinpath(dirname(abspath(xlsx_path)), "imputed_data")

isfile(xlsx_path) || error("dataset.xlsx not found at: $xlsx_path")
mkpath(output_dir)

@info "dropout fit" xlsx_path output_dir

# ---------------------------------------------------------------------------
# Load raw intensity matrix
# ---------------------------------------------------------------------------

# _load_intensity_matrix is non-exported; access via the qualified path.
intensity_matrix, column_names =
    BayesInteractomics._load_intensity_matrix(xlsx_path)

n_proteins, n_cols = size(intensity_matrix)
@info "Loaded matrix" n_proteins n_cols

# ---------------------------------------------------------------------------
# Build per-column protocol assignment from column-name conventions
# ---------------------------------------------------------------------------

# Heuristic: protocol prefix = second underscore-separated token of column name.
# E.g.  "wt_grecco_1_exp1_rep1" → "grecco" (second token if first is condition).
# We use a simple "second token" rule for HD-style headers; falls back to
# protocol = 1 for everyone if names don't follow the convention.

function _build_protocol_assignment(names::Vector{String})::Vector{Int}
    # Identify protocol by the second underscore-separated token, if present.
    keys = String[]
    for name in names
        parts = split(name, '_')
        if length(parts) >= 2
            push!(keys, String(parts[2]))
        else
            push!(keys, "default")
        end
    end
    unique_keys = unique(keys)
    idx_lookup = Dict(k => i for (i, k) in enumerate(unique_keys))
    return Int[idx_lookup[k] for k in keys]
end

protocol_assignment = _build_protocol_assignment(column_names)
@info "Protocol assignment" n_protocols = maximum(protocol_assignment)

# ---------------------------------------------------------------------------
# Resolve log_transform mode
# ---------------------------------------------------------------------------
# AUTO heuristic: if the maximum non-missing value in the matrix is < 100,
# the input is almost certainly already log-transformed (typical log2(intensity)
# range is 7..40 for proteomics; raw intensities are 10^7..10^11 = >> 100).
# This guards against the double-log bug seen on Abundance / log2-LFQ inputs.

local log_transform_resolved::Bool
if log_transform_mode == "true"
    log_transform_resolved = true
elseif log_transform_mode == "false"
    log_transform_resolved = false
else  # "auto"
    max_val = maximum(skipmissing(intensity_matrix))
    log_transform_resolved = max_val >= 100.0
    @info "Log-transform auto-detect" max_value = max_val resolved = log_transform_resolved decision = (log_transform_resolved ? "raw → applying log2" : "already log-transformed → skipping log2")
end

# ---------------------------------------------------------------------------
# Fit + persist + diagnostics
# ---------------------------------------------------------------------------

output_path     = joinpath(output_dir, "dropout_curves.json")
diagnostics_dir = joinpath(output_dir, "dropout_diagnostics")

fit = fit_dropout_curves(
    intensity_matrix;
    column_names              = column_names,
    log_transform             = log_transform_resolved,
    min_detections_per_column = 5,
    output_path               = output_path,
    diagnostics_dir           = diagnostics_dir,
    protocol_assignment       = protocol_assignment,
)

# ---------------------------------------------------------------------------
# Soft sanity check
# ---------------------------------------------------------------------------
# >70% well-fit is the expectation, but failing it is a *finding* (the
# condition_groups decision), not a bug. Hard-fail only on catastrophic
# <50% well-fit, unless --allow-imperfect-fit was passed.

n_excl = count(isnan, fit.rho)
n_fit_cols = length(fit.rho) - n_excl
z_well = count(z -> 0.5 ≤ z ≤ 3.0, filter(!isnan, fit.zeta))
pct_well = n_fit_cols == 0 ? 0.0 : 100.0 * z_well / n_fit_cols

if pct_well < 50.0 && n_fit_cols > 0
    if allow_imperfect_fit
        @warn "Imperfect dropout-curve fit ($(round(pct_well; digits=1))% well-fit) " *
              "downgraded to warning by --allow-imperfect-fit. Inspect $diagnostics_dir/SANITY.md " *
              "before relying on these curves for downstream MNAR imputation."
    else
        @error "Catastrophic dropout-curve fit: only $(round(pct_well; digits=1))% of columns " *
               "have ζ̂_c ∈ [0.5, 3]. Inspect $diagnostics_dir/SANITY.md. " *
               "Pass --allow-imperfect-fit to override (curves still written)."
        exit(1)
    end
elseif pct_well < 70.0
    @warn "Soft warning: only $(round(pct_well; digits=1))% of columns have " *
          "ζ̂_c ∈ [0.5, 3] (target > 70%). See $diagnostics_dir/SANITY.md — " *
          "may indicate per-protocol heterogeneity."
else
    @info "Sanity check passed" pct_well_fit = round(pct_well; digits=1)
end

@info "dropout fit done" output_path diagnostics_dir
