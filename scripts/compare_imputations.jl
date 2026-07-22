#!/usr/bin/env julia
#=
MNAR-vs-MICE 4-panel diagnostic comparison.

Generates `<imputed_dir>/mnar_diagnostics/mnar_vs_mice.png`, a 4-panel
800x800 PNG comparing the new MNAR single imputation against the
existing MICE 5-imputation set.

Panels:
  1. HTT     — imputed-value distribution: MNAR (single draw) vs MICE (5 draws pooled)
  2. HAP40   — same overlay
  3. F8A1    — same overlay (v1.1 P=1.0 anchor)
  4. Aggregate left-censored shift — histogram of
     (mean(MNAR_imputed) - mean(MICE_imputed)) over all proteins with
     missing_fraction > 0.5 AND mean(observed) in the bottom quartile.

Inputs:
  <dataset_xlsx>                       — raw dataset.xlsx (drives missing-mask + protein IDs)
  <imputed_dir>/dataset_mnar.xlsx      — MNAR output
  <imputed_dir>/dataset_imp_1.xlsx     — existing MICE imputation #1
  ...
  <imputed_dir>/dataset_imp_5.xlsx     — existing MICE imputation #5

Output:
  <imputed_dir>/mnar_diagnostics/mnar_vs_mice.png

Usage:
    julia --project=. --threads=4 scripts/compare_imputations.jl <dataset.xlsx> [imputed_dir] \
          [--htt-id ID] [--hap40-id ID] [--f8a1-id ID]

`imputed_dir` defaults to `<dirname(dataset.xlsx)>/imputed_data/`.

Anchor protein IDs MUST match `dataset.xlsx` column 1 strings exactly. Defaults
are placeholders (HTT_PLACEHOLDER / HAP40_PLACEHOLDER / F8A1_PLACEHOLDER); if a
flag is omitted, the corresponding panel renders an empty "NOT FOUND" placeholder
plot and emits a warning. HD validation supplies the real IDs.
=#

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BayesInteractomics
using Statistics
import StatsPlots

# ----------------------------------------------------------------------------
# Argument parsing (flag-aware)
# ----------------------------------------------------------------------------

if length(ARGS) < 1
    @error """usage: julia --project=. --threads=4 scripts/compare_imputations.jl <dataset.xlsx> [imputed_dir] [--htt-id ID --hap40-id ID --f8a1-id ID]

  <dataset.xlsx>   Raw intensity matrix (sheet 'Sheet1', col 1 = protein IDs).
  [imputed_dir]    Optional. Defaults to <dirname(dataset.xlsx)>/imputed_data/.
                   Must contain dataset_mnar.xlsx + dataset_imp_1.xlsx..dataset_imp_5.xlsx.
  --htt-id ID      Protein ID for HTT (default placeholder; required for anchor panel 1).
  --hap40-id ID    Protein ID for HAP40 (default placeholder; required for anchor panel 2).
  --f8a1-id ID     Protein ID for F8A1 (default placeholder; required for anchor panel 3)."""
    exit(1)
end

function _parse_args(args::Vector{String})
    positional = String[]
    htt_id   = "HTT_PLACEHOLDER"
    hap40_id = "HAP40_PLACEHOLDER"
    f8a1_id  = "F8A1_PLACEHOLDER"
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--htt-id" && i < length(args)
            htt_id = args[i+1]; i += 2
        elseif a == "--hap40-id" && i < length(args)
            hap40_id = args[i+1]; i += 2
        elseif a == "--f8a1-id" && i < length(args)
            f8a1_id = args[i+1]; i += 2
        else
            push!(positional, a); i += 1
        end
    end
    return positional, htt_id, hap40_id, f8a1_id
end

positional, htt_id, hap40_id, f8a1_id = _parse_args(ARGS)

if length(positional) < 1
    @error "missing <dataset.xlsx> positional argument. See usage above."
    exit(1)
end

xlsx_path = positional[1]
isfile(xlsx_path) || (@error "dataset.xlsx not found at: $xlsx_path"; exit(1))

imputed_dir = length(positional) >= 2 ? positional[2] :
              joinpath(dirname(abspath(xlsx_path)), "imputed_data")

mnar_path = joinpath(imputed_dir, "dataset_mnar.xlsx")
isfile(mnar_path) || (@error "dataset_mnar.xlsx not found at: $mnar_path. Run scripts/impute_mnar.jl first."; exit(1))

mice_paths = [joinpath(imputed_dir, "dataset_imp_$i.xlsx") for i in 1:5]
for p in mice_paths
    isfile(p) || (@error "MICE imputation not found: $p"; exit(1))
end

@info "MNAR-vs-MICE comparison" xlsx_path imputed_dir htt_id hap40_id f8a1_id

# ----------------------------------------------------------------------------
# Load matrices via the library's non-exported helper (returns 4-tuple:
#   (intensity_matrix, column_names, protein_ids, id_col_name))
# ----------------------------------------------------------------------------

raw_matrix, _, raw_ids, _ = BayesInteractomics._load_intensity_matrix_with_ids(xlsx_path)
mnar_matrix, _, _, _      = BayesInteractomics._load_intensity_matrix_with_ids(mnar_path)
mice_matrices = [BayesInteractomics._load_intensity_matrix_with_ids(p)[1] for p in mice_paths]

# Sanity — all matrices share row count + column count of raw
@assert size(mnar_matrix) == size(raw_matrix)  "MNAR matrix shape mismatch"
for (k, m) in enumerate(mice_matrices)
    @assert size(m) == size(raw_matrix) "MICE imputation $k shape mismatch"
end

# ----------------------------------------------------------------------------
# Panel builders
# ----------------------------------------------------------------------------

function _build_anchor_panel(protein_id::String, label::String,
                              raw, mnar, mice_matrices, ids::Vector{String})
    idx = findfirst(==(protein_id), ids)
    if idx === nothing
        @warn "Anchor protein not found in dataset.xlsx column 1; rendering placeholder panel" protein_id label
        return StatsPlots.plot(; title = "$label ($protein_id) — NOT FOUND",
                                  legend = false, framestyle = :box)
    end
    mnar_imp = Float64[]
    mice_imp = Float64[]
    for c in 1:size(raw, 2)
        if ismissing(raw[idx, c])
            push!(mnar_imp, Float64(mnar[idx, c]))
            for m in mice_matrices
                push!(mice_imp, Float64(m[idx, c]))
            end
        end
    end
    if isempty(mnar_imp)
        return StatsPlots.plot(; title = "$label ($protein_id) — no missings", legend = false)
    end
    plt = StatsPlots.histogram(mnar_imp;
        bins = 20, alpha = 0.6, label = "MNAR (n=$(length(mnar_imp)))",
        title = "$label ($protein_id)", xlabel = "imputed intensity", ylabel = "count")
    StatsPlots.histogram!(plt, mice_imp;
        bins = 20, alpha = 0.4, label = "MICE (n=$(length(mice_imp)))")
    return plt
end

function _build_aggregate_panel(raw, mnar, mice_matrices)
    n_p = size(raw, 1)
    obs_means = [let v = collect(skipmissing(raw[i, :])); isempty(v) ? NaN : mean(v) end for i in 1:n_p]
    miss_frac = [mean(ismissing, raw[i, :]) for i in 1:n_p]
    valid_obs = filter(!isnan, obs_means)
    obs_q1 = isempty(valid_obs) ? NaN : quantile(valid_obs, 0.25)

    shifts = Float64[]
    if !isnan(obs_q1)
        for i in 1:n_p
            if miss_frac[i] > 0.5 && !isnan(obs_means[i]) && obs_means[i] <= obs_q1
                mnar_vals = Float64[]
                mice_vals = Float64[]
                for c in 1:size(raw, 2)
                    if ismissing(raw[i, c])
                        push!(mnar_vals, Float64(mnar[i, c]))
                        for m in mice_matrices
                            push!(mice_vals, Float64(m[i, c]))
                        end
                    end
                end
                if !isempty(mnar_vals) && !isempty(mice_vals)
                    push!(shifts, mean(mnar_vals) - mean(mice_vals))
                end
            end
        end
    end

    plt = StatsPlots.histogram(shifts;
        bins = 30, label = nothing,
        title = "Aggregate left-censored shift (n=$(length(shifts)))",
        xlabel = "mean(MNAR) − mean(MICE)", ylabel = "count")
    StatsPlots.vline!(plt, [0.0]; label = nothing, color = :black, linestyle = :dash, linewidth = 0.8)
    return plt
end

# ----------------------------------------------------------------------------
# Build 4-panel and save
# ----------------------------------------------------------------------------

p_htt   = _build_anchor_panel(htt_id,   "HTT",   raw_matrix, mnar_matrix, mice_matrices, raw_ids)
p_hap40 = _build_anchor_panel(hap40_id, "HAP40", raw_matrix, mnar_matrix, mice_matrices, raw_ids)
p_f8a1  = _build_anchor_panel(f8a1_id,  "F8A1",  raw_matrix, mnar_matrix, mice_matrices, raw_ids)
p_aggr  = _build_aggregate_panel(raw_matrix, mnar_matrix, mice_matrices)

plt = StatsPlots.plot(p_htt, p_hap40, p_f8a1, p_aggr;
                       layout = (2, 2), size = (800, 800))

out_path = joinpath(imputed_dir, "mnar_diagnostics", "mnar_vs_mice.png")
mkpath(dirname(abspath(out_path)))
StatsPlots.savefig(plt, out_path)

@info "MNAR-vs-MICE comparison done" out_path
