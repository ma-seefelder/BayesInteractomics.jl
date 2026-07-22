#!/usr/bin/env julia
#=
MNAR-aware single-imputation CLI wrapper.

Reads a raw `dataset.xlsx` (rows = proteins, columns = MS runs) and the
per-column dropout-curve JSON, then samples each missing entry
once from a tilted Gaussian
    p̃(y | μ̂_i, σ̂_i², ρ̂_c, ζ̂_c) ∝ φ(y; μ̂_i, σ̂_i²) · (1 − σ(ρ̂_c + ζ̂_c · y))
via inverse-CDF interpolation on a 50-point grid. Writes:

  <output_dir>/dataset_mnar.xlsx              — single MNAR imputation, raw layout
  <output_dir>/dataset_mnar_manifest.json     — reproducibility sidecar

Reads:
  <output_dir>/dropout_curves.json            — produced by scripts/fit_dropout.jl

Usage:
    julia --project=. --threads=4 scripts/impute_mnar.jl <dataset.xlsx> [output_dir] [--seed N]

`output_dir` defaults to `<dirname(dataset.xlsx)>/imputed_data/`.
`--seed N` overrides the default seed (42); written into the manifest.

Note (Julia 1.12 / Windows): always use explicit `--threads=N` — never `--threads=auto`
(auto can segfault intermittently).
=#

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BayesInteractomics

# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------

if length(ARGS) < 1
    @error """usage: julia --project=. --threads=4 scripts/impute_mnar.jl <dataset.xlsx> [output_dir] [--seed N]

  <dataset.xlsx>   Path to the raw intensity matrix (sheet 'Sheet1', col 1 = protein IDs).
  [output_dir]     Optional. Defaults to <dirname(dataset.xlsx)>/imputed_data/.
                   Must already contain dropout_curves.json (from scripts/fit_dropout.jl).
  --seed N         Optional. Random seed for reproducibility. Default 42; written to the manifest."""
    exit(1)
end

# Strip flag-like args from positional candidates so a user passing
# `julia ... script.jl data.xlsx --seed 7` doesn't capture "--seed" as output_dir.
function _parse_args(args::Vector{String})
    positional = String[]
    seed = 42
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--seed" && i < length(args)
            seed = parse(Int, args[i+1])
            i += 2
        else
            push!(positional, a)
            i += 1
        end
    end
    return positional, seed
end

positional, seed = _parse_args(ARGS)

if length(positional) < 1
    @error "missing <dataset.xlsx> positional argument. See usage above."
    exit(1)
end

xlsx_path = positional[1]
isfile(xlsx_path) || (@error "dataset.xlsx not found at: $xlsx_path"; exit(1))

output_dir = length(positional) >= 2 ? positional[2] :
             joinpath(dirname(abspath(xlsx_path)), "imputed_data")
mkpath(output_dir)

curves_path = joinpath(output_dir, "dropout_curves.json")
if !isfile(curves_path)
    @error """dropout_curves.json not found at: $curves_path

Run the dropout fit first:
    julia --project=. --threads=4 scripts/fit_dropout.jl $xlsx_path $output_dir
"""
    exit(1)
end

@info "MNAR imputation" xlsx_path output_dir curves_path seed

# ----------------------------------------------------------------------------
# Run imputation via the library entry point
# ----------------------------------------------------------------------------

output_path   = joinpath(output_dir, "dataset_mnar.xlsx")
manifest_path = joinpath(output_dir, "dataset_mnar_manifest.json")

result = BayesInteractomics.impute_mnar_from_paths(
    xlsx_path, curves_path;
    output_path   = output_path,
    manifest_path = manifest_path,
    seed          = seed,
)

# ----------------------------------------------------------------------------
# Summary log
# ----------------------------------------------------------------------------

n_imputed  = result.manifest["n_missing_entries_imputed"]
n_proteins = result.manifest["n_proteins"]
n_columns  = result.manifest["n_columns_imputed"]

@info "MNAR imputation done" output_path = result.output_path manifest_path = result.manifest_path n_proteins n_columns n_missing_entries_imputed = n_imputed seed
