# =============================================================================
# compute_hap40_mutants.jl — compute ONLY the two HAP40 mutant AnalysisResults
# (Δcenter, 4K) in a fresh, low-allocation process, and save their AR caches.
#
# Why standalone: the full capstone loads + curates + imputes all 6 conditions
# and deserialises 4 cached baits in one process — that cumulative allocation
# (~2.4 B) reliably trips a Julia-1.12/Windows GC EXCEPTION_ACCESS_VIOLATION the
# moment a FRESH mutant run_analysis allocates. Computing the mutants alone (only
# the two small single-protocol files) stays well under that threshold.
#
# Then run examples/diff_6condition_from_cache.jl to build the 6-way differential
# from all six cached ARs (also low-allocation — no data loading).
#
# Env: BASEPATH (source+output, default HTT_meta), OUTBASE (output, default
# BASEPATH), NORM_METHOD (default auto), FAST_REGEN (1 = skip PPC).
# =============================================================================
import Pkg
Pkg.activate(@__DIR__)

using Flux, MLJ, MLJScikitLearnInterface, HDF5    # metalearner ext
using GLM                                          # imputation ext
using BayesInteractomics
using XLSX, DataFrames

const BASEPATH = get(ENV, "BASEPATH", "C:/Users/Manuel/Desktop/HTT_meta/")
const OUTBASE  = get(ENV, "OUTBASE", BASEPATH)
const NORM     = Symbol(get(ENV, "NORM_METHOD", "auto"))
@info "compute_hap40_mutants" BASEPATH OUTBASE NORM threads=Threads.nthreads()

# Dropout-fit + MNAR imputation (copied from the capstone, Task 1).
function impute_hap40_dataset(source_xlsx::String, basename::String)
    imp_dir = joinpath(BASEPATH, "imputed_data"); mkpath(imp_dir)
    curves_path   = joinpath(imp_dir, "$(basename)_dropout_curves.json")
    imputed_path  = joinpath(imp_dir, "$(basename)_mnar.xlsx")
    manifest_path = joinpath(imp_dir, "$(basename)_mnar_manifest.json")
    if isfile(imputed_path) && isfile(manifest_path) && isfile(curves_path) && mtime(imputed_path) > mtime(source_xlsx)
        @info "$basename: re-using existing imputed file"; return imputed_path
    end
    raw_df = DataFrame(XLSX.readtable(source_xlsx, "Sheet1"))
    column_names = String.(names(raw_df))[2:end]
    intensity_matrix = convert(Matrix{Union{Missing, Float64}}, Matrix(raw_df[:, 2:end]))
    fit_dropout_curves(intensity_matrix; column_names = column_names, log_transform = true,
                       min_detections_per_column = 5, output_path = curves_path)
    impute_mnar_from_paths(source_xlsx, curves_path; output_path = imputed_path,
                           manifest_path = manifest_path, seed = 42, log_transform = true)
    return imputed_path
end

const HAP40_DELTA_SRC = raw"C:\Users\Manuel\OneDrive\HAP40_interactome_enrichment\HAP40delta\HAP40_delta.xlsx"
const HAP40_K4_SRC    = raw"C:\Users\Manuel\OneDrive\HAP40_interactome_enrichment\HAP40-4K\HAP40_K4.xlsx"

function compute_and_save(label, src, sample_cols, control_cols, n_samples)
    arfile = joinpath(OUTBASE, label, "ar_cache.jld2")
    if get(ENV, "FORCE", "0") != "1" && load_result(arfile) !== nothing
        @info "$label: AR cache already present — skipping (set FORCE=1 to recompute)" arfile
        return
    end
    imp  = impute_hap40_dataset(src, label)
    raw  = load_data([src], sample_cols, control_cols, 1, 1, false;
                     normalisation_method = NORM, filter_insufficient_obs = false,
                     curate_interactive = false, imputation = :none)
    data = InteractionData[load_data([imp], sample_cols, control_cols, 1, 1, false;
                     normalisation_method = NORM, filter_insufficient_obs = false,
                     curate_interactive = false, imputation = :mnar)]
    # Locate the HAP40 (F8A1) bait row AFTER curation — refID must point at the
    # bait (it is NOT at row 1 in these files; row 1 is SYVN1). Curation reorders,
    # so find it dynamically rather than hardcoding. raw/imputed share order here
    # (same source, same curation), so the raw index is valid for both.
    poi = "9606.ENSP00000479624"
    ids = string.(BayesInteractomics.getIDs(raw))
    bait_idx = findfirst(id -> occursin("ENSP00000479624", id) ||
                               occursin(r"^F8A1$|^HAP40$"i, id) ||
                               occursin(r"F8A1|HAP40"i, id), ids)
    bait_idx === nothing && error("$label: HAP40/F8A1 bait not found among curated protein IDs")
    @info "$label: HAP40/F8A1 bait at curated index $bait_idx ($(ids[bait_idx]))"
    cfg = CONFIG(
        datafile = [src], sample_cols = sample_cols, control_cols = control_cols,
        poi = poi, n_controls = 6, n_samples = n_samples, refID = bait_idx,
        output = OutputFiles(joinpath(OUTBASE, label), image_ext = ".svg"),
        metalearner_path = "metalearners/HistGradientBoosting_tune.jld2",
        normalisation_method = NORM,
        plotHBMdists = false, plotlog2fc = false, plotregr = false, plotbayesrange = false,
        verbose = false, vc_legend_pos = :topleft, combination_method = :bma,
        run_diagnostics = get(ENV, "FAST_REGEN", "0") != "1",
        optimize_nu = true, curate_interactive = false, run_input_qc = true,
        mnar_variance_recovery = :inflation,
    )
    @info "$label: starting run_analysis (standalone)"
    _, ar = run_analysis(cfg, data, raw)
    mkpath(dirname(arfile)); save_result(ar, arfile)
    @info "$label: saved AR cache" arfile
    GC.gc()
end

# Single experiment per mutant — the Δ/4K samples are replicates of ONE
# experiment, so this is the honest structure (the HBM nparameters fix in
# models.jl makes single-experiment well-formed). 6 EGFP controls + the mutant
# samples. Single experiment ⇒ no dose axis ⇒ bf_correlation uninformative;
# detection + enrichment carry the signal.
compute_and_save("HAP40_delta", HAP40_DELTA_SRC, [Dict(1 => [8,9,10])], [Dict(1 => [2,3,4,5,6,7])], 3)
compute_and_save("HAP40_K4",    HAP40_K4_SRC,    [Dict(1 => [8,9])],    [Dict(1 => [2,3,4,5,6,7])], 2)
@info "HAP40 mutant ARs computed + saved. Next: examples/diff_6condition_from_cache.jl"
