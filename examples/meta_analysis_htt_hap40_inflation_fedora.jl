# =============================================================================
# meta_analysis_htt_hap40_inflation_fedora.jl
#   Fedora-host adaptation of meta_analysis_htt_hap40_inflation.jl
# =============================================================================
#
# Differences vs the original Windows script:
#   • BASEPATH points at /home/mseefelder/Schreibtisch/HTT_meta/
#   • HAP40 raw xlsx + metalearner referenced via absolute paths so the script
#     runs from any cwd (originals were relative — `data/HAP40_Strep.xlsx` —
#     and assumed cwd == BayesInteractomics repo root).
#   • BLAS thread count set to Sys.CPU_THREADS at startup. Julia thread count
#     is taken from --threads=auto at launch (see suggested invocation below).
#   • Cache reuse is handled by the existing idempotency logic:
#       - imputed HAP40 xlsx skip-via-mtime check (impute_hap40_dataset)
#       - dataset_phase75.xlsx skip-via-mtime check (merge_rep4b_into_rep4)
#       - JLD2 caches under BASEPATH/.bayesinteractomics_cache/ (curation,
#         BetaBernoulli, HBM regression, AnalysisResult) auto-keyed by
#         CONFIG hash + data hash
#
# Recommended invocation (32-core / 64-thread Fedora workstation):
#   julia --threads=auto --project=examples \
#         examples/meta_analysis_htt_hap40_inflation_fedora.jl
#
# Same locked design decisions as the original:
#   mnar_variance_recovery = :inflation   (all 4 conditions)
#   normalise_protocols    = false        (all 4 conditions)
#   protein-order alignment via the k-group
#        `differential_analysis(; conditions = (...))` wide DataFrame
#   HAP40 paths from examples/hap40_differential_interactome.jl
#   outputs under $BASEPATH/<condition>/
#   genotype-matched 140Q mIgG controls + Rep4B-into-Rep4 merge
#   NEW script (does NOT mutate examples/meta_analysis_workflow.jl)
#
# Outputs (under $BASEPATH):
#   wtHTT/, mHTT/, HAP40_Strep/, GST_HAP40/, diff_4condition/
#   imputed_data/{HAP40_Strep,GST_HAP40}_mnar.xlsx + manifest + dropout-curves
# =============================================================================

import Pkg
Pkg.activate(@__DIR__)
# Note: Pkg.resolve() / Pkg.precompile() intentionally omitted.
# The examples/Manifest.toml is the authoritative environment for this run
# (GLM 1.9.4 already resolved). Re-running resolve forces re-resolution
# against the current BayesInteractomics compat (capped at GLM 1.9.3) and
# fails. Trust the cached Manifest — first `using ...` line below will
# precompile whatever stale caches need it.

using LinearAlgebra
BLAS.set_num_threads(Sys.CPU_THREADS)
@info "Threading configuration" julia_threads=Threads.nthreads() blas_threads=BLAS.get_num_threads() cpu_threads=Sys.CPU_THREADS

using Flux, MLJ, MLJScikitLearnInterface, HDF5    # activates BayesInteractomicsMetalearnerExt
using GLM                                          # activates BayesInteractomicsImputationExt
using UMAP, Clustering                             # activates BayesInteractomicsEmbeddingsExt (UMAP plots)
using BayesInteractomics
using Graphs, SimpleWeightedGraphs, GraphPlot, Compose, Cairo  # network extension (consistent with hap40_differential_interactome.jl)
using XLSX, DataFrames

const BASEPATH = "/home/mseefelder/Schreibtisch/HTT_meta/"

# Absolute paths to the HAP40 source xlsx files inside the BayesInteractomics
# repo (the originals used `data/HAP40_Strep.xlsx` relative to cwd).
const HAP40_STREP_SRC = "/home/mseefelder/Github/BayesInteractomics/data/HAP40_Strep.xlsx"
const HAP40_GST_SRC   = "/home/mseefelder/Github/BayesInteractomics/data/GST_HAP40.xlsx"

# Absolute path to the metalearner artefact (originally `metalearners/HistGradientBoosting_tune.jld2`
# relative to the repo root).
const METALEARNER_PATH = "/home/mseefelder/Github/BayesInteractomics/metalearners/HistGradientBoosting_tune.jld2"

function count_samples(x, dummy)
    total = 0
    for i in x
        n = sum([length(y) for (_, y) in i])
        total += n
    end
    return total - dummy
end

# --------------------------------------------------------------------------- #
# HAP40 dropout-fit + MNAR-imputation block (Task 1)
# --------------------------------------------------------------------------- #
# For each HAP40 source file: fit per-column dropout curves, then MNAR-impute
# into a new xlsx that matches the source layout. Idempotent (re-run reuses
# existing outputs when the input hash is unchanged via the manifest mtime check).

function impute_hap40_dataset(source_xlsx::String, basename::String)
    imp_dir = joinpath(BASEPATH, "imputed_data")
    mkpath(imp_dir)

    curves_path   = joinpath(imp_dir, "$(basename)_dropout_curves.json")
    imputed_path  = joinpath(imp_dir, "$(basename)_mnar.xlsx")
    manifest_path = joinpath(imp_dir, "$(basename)_mnar_manifest.json")

    # Skip re-imputation if outputs are newer than source (cheap idempotency).
    if isfile(imputed_path) && isfile(manifest_path) && isfile(curves_path)
        if mtime(imputed_path) > mtime(source_xlsx)
            @info "HAP40 $basename: re-using existing imputed file (newer than source)" imputed_path
            return imputed_path
        end
    end

    @info "HAP40 $basename: loading source and fitting dropout curves" source_xlsx
    raw_df = DataFrame(XLSX.readtable(source_xlsx, "Sheet1"))
    all_names    = String.(names(raw_df))
    column_names = all_names[2:end]              # cols 2..end are intensities; col 1 is "Protein"
    raw_matrix   = Matrix(raw_df[:, 2:end])
    intensity_matrix = convert(Matrix{Union{Missing, Float64}}, raw_matrix)

    # Fit per-column dropout curves. log_transform=true matches the
    # HTT pipeline's accident-of-scale-consistency precedent. NaN curves for
    # columns with < 5 detections are tolerated.
    fit_dropout_curves(intensity_matrix;
        column_names = column_names,
        log_transform = true,
        min_detections_per_column = 5,
        output_path = curves_path,
    )

    # MNAR-impute via the path-driven orchestrator.
    @info "HAP40 $basename: MNAR-imputing" curves_path imputed_path
    result = impute_mnar_from_paths(source_xlsx, curves_path;
        output_path   = imputed_path,
        manifest_path = manifest_path,
        seed          = 42,
        log_transform = true,
    )

    @info "HAP40 $basename: imputation complete" output = result.output_path
    return imputed_path
end

hap40_strep_imputed = impute_hap40_dataset(HAP40_STREP_SRC, "HAP40_Strep")
hap40_gst_imputed   = impute_hap40_dataset(HAP40_GST_SRC,   "GST_HAP40")

# --------------------------------------------------------------------------- #
# Correction: Cerebellum-140Q-8WK-3E10-4E10 Rep4B (col 146) + Rep4
# (col 147) are technical replicates of the same biological sample. Merge them
# into a single mean-of-tech-rep column at position 147. Produces
# `dataset_phase75.xlsx` + `dataset_mnar_phase75.xlsx` derived files; leaves
# the originals untouched.
# --------------------------------------------------------------------------- #
function merge_rep4b_into_rep4(src_path::String, out_path::String;
                               col_rep4b::Int = 146, col_rep4::Int = 147)
    if isfile(out_path) && mtime(out_path) > mtime(src_path)
        @info "Rep4B merge: re-using existing $out_path (newer than source)"
        return out_path
    end
    @info "Rep4B merge: reading source" src_path
    df = DataFrame(XLSX.readtable(src_path, "Sheet1"))
    n = nrow(df)
    rep4b_col = df[:, col_rep4b]
    rep4_col  = df[:, col_rep4]
    merged = Vector{Union{Missing, Float64}}(undef, n)
    @inbounds for i in 1:n
        a = rep4b_col[i]; b = rep4_col[i]
        ma = ismissing(a); mb = ismissing(b)
        if ma && mb
            merged[i] = missing
        elseif ma
            merged[i] = Float64(b)
        elseif mb
            merged[i] = Float64(a)
        else
            merged[i] = (Float64(a) + Float64(b)) / 2.0   # geometric mean of raw (log-scale arithmetic)
        end
    end
    df[!, col_rep4] = merged
    mkpath(dirname(abspath(out_path)))
    XLSX.writetable(out_path, "Sheet1" => df; overwrite = true)
    @info "Rep4B merge: wrote derived file" out_path
    return out_path
end

dataset_phase75      = merge_rep4b_into_rep4(joinpath(BASEPATH, "dataset.xlsx"),
                                              joinpath(BASEPATH, "dataset_phase75.xlsx"))
dataset_mnar_phase75 = merge_rep4b_into_rep4(joinpath(BASEPATH, "imputed_data/dataset_mnar.xlsx"),
                                              joinpath(BASEPATH, "imputed_data/dataset_mnar_phase75.xlsx"))

# --------------------------------------------------------------------------- #
# HTT column definitions — verbatim copy from
# examples/meta_analysis_workflow.jl, verified by a pre-flight audit.
# --------------------------------------------------------------------------- #
# Dummy cols: 162, 163, 164, 165

# Wild-type HTT samples — n_dummy: 10 + 13 + 12 + 1 + 4 + 4 = 44
wt_s_grecco    = Dict(1 => [2,3,4,162], 2 => [5,6,7,162], 3 => [162,163,164,165], 4 => [162,163,164,165])
wt_s_gutierrez = Dict(1 => [29,30,31,162], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
wt_s_sap       = Dict(1 => [36,37,38,39], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
wt_s_grecco_2  = Dict(1 => [48,49,50,51], 2 => [52,53,54,55], 3 => [60,61,62,63], 4 => [64,65,66,162])
wt_s_grecco_3  = Dict(1 => [92,93,94,162], 2 => [95,96,97,162], 3 => [100,101,102,162], 4 => [103,104,105,162])
wt_s_grecco_4  = Dict(1 => [128,129,130,162], 2 => [131,132,133,162], 3 => [137,138,139,162], 4 => [140,141,142,162])

# Mutant HTT samples — correction: mut_s_grecco_4 protocol 1
# changed from [146,147,148,149] (Rep4B + Rep4 as two reps) to [147,148,149,162]
# because Rep4B has been merged into Rep4 via merge_rep4b_into_rep4 (col 147 now
# holds the row-wise mean of original cols 146 + 147). n_dummy: 10+16+12+1+2+4 = 45.
mut_s_grecco    = Dict(1 => [8,9,10,162], 2 => [11,12,13,162], 3 => [162,163,164,165], 4 => [162,163,164,165])
mut_s_gutierrez = Dict(1 => [162,163,164,165], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
mut_s_sap       = Dict(1 => [40,41,42,43], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
mut_s_grecco_2  = Dict(1 => [70,71,72,162], 2 => [73,74,75,76], 3 => [81,82,83,84], 4 => [85,86,87,88])
mut_s_grecco_3  = Dict(1 => [109,110,111,112], 2 => [113,114,115,116], 3 => [119,120,121,162], 4 => [122,123,124,162])
mut_s_grecco_4  = Dict(1 => [147,148,149,162], 2 => [150,151,152,162], 3 => [156,157,158,162], 4 => [159,160,161,162])

# wt (20Q) controls — n_dummy: 10 + 13 + 12 + 0 + 8 + 4 = 47
c_grecco       = Dict(1 => [14,15,16,162], 2 => [17,18,19,162], 3 => [162,163,164,165], 4 => [162,163,164,165])
c_gutierrez    = Dict(1 => [26,27,28,162], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
c_sap          = Dict(1 => [32,33,34,35], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
c_grecco_2_wt  = Dict(1 => [44,45,46,47], 2 => [44,45,46,47], 3 => [56,57,58,59], 4 => [56,57,58,59])
c_grecco_3_wt  = Dict(1 => [90, 91, 162, 163], 2 => [90, 91, 162, 163], 3 => [98,99,162,163], 4 => [98,99,162,163])
c_grecco_4_wt  = Dict(1 => [125,126,127,162], 2 => [125,126,127,162], 3 => [134,135,136,162], 4 => [134,135,136,162])

# Correction — genotype-matched 140Q mIgG controls for the
# Greco-preprint mutant analysis (Striatum / Cortex / Cerebellum).
# Source columns from the verified dataset.xlsx header audit:
#   Striatum 140Q-8WK  mIgG : cols 67,68,69     (Rep4, Rep3, Rep1 — Rep2 missing)
#   Striatum 140Q-40WK mIgG : cols 77,78,79,80  (Rep4, Rep3, Rep2, Rep1)
#   Cortex   140Q-8Wk  mIgG : cols 106,107,108  (Rep3, Rep2B, Rep1)
#   Cortex   140Q-40Wk mIgG : cols 117,118      (Rep3, Rep1)
#   Cerebellum 140Q-8WK  mIgG : cols 143,144,145 (Rep4, Rep3, Rep2)
#   Cerebellum 140Q-40WK mIgG : cols 153,154,155 (Rep4, Rep3, Rep2)
# Greco_2022 and Gutierrez and Sap retain their original control mapping
# (they include genotype-specific controls in c_grecco / c_gutierrez / c_sap already,
#  OR the bait pulldown vs Control_GFP is the genotype-agnostic reference for those studies).
c_grecco_2_mut = Dict(1 => [67,68,69,162], 2 => [67,68,69,162], 3 => [77,78,79,80], 4 => [77,78,79,80])
c_grecco_3_mut = Dict(1 => [106,107,108,162], 2 => [106,107,108,162], 3 => [117,118,162,163], 4 => [117,118,162,163])
c_grecco_4_mut = Dict(1 => [143,144,145,162], 2 => [143,144,145,162], 3 => [153,154,155,162], 4 => [153,154,155,162])
# mut_control_cols n_dummy: 10 + 13 + 12 + 2 + 6 + 4 = 47.

wt_sample_cols   = [wt_s_grecco, wt_s_gutierrez, wt_s_sap, wt_s_grecco_2, wt_s_grecco_3, wt_s_grecco_4]
mut_sample_cols  = [mut_s_grecco, mut_s_gutierrez, mut_s_sap, mut_s_grecco_2, mut_s_grecco_3, mut_s_grecco_4]
wt_control_cols  = [c_grecco, c_gutierrez, c_sap, c_grecco_2_wt,  c_grecco_3_wt,  c_grecco_4_wt]
mut_control_cols = [c_grecco, c_gutierrez, c_sap, c_grecco_2_mut, c_grecco_3_mut, c_grecco_4_mut]

# --------------------------------------------------------------------------- #
# HTT load — raw + MNAR-imputed; replay curation across the 4 loads.
# Correction: all 6 raw + imputed loads pull from the *_phase75
# derived files (col 147 = mean of Rep4 + Rep4B). Correction:
# wt loads use wt_control_cols (20Q mIgG); mut loads use mut_control_cols
# (140Q mIgG genotype-matched).
# --------------------------------------------------------------------------- #
raw_files = [dataset_phase75 for _ in 1:6]

@info "HTT: loading wt raw data (interactive curation off; replay path saved to cache)"
wt_raw_data = load_data(raw_files, wt_sample_cols, wt_control_cols, 1, 1, false,
                        normalise_protocols = false, curate_interactive = false)

replay_path = joinpath(BASEPATH, ".bayesinteractomics_cache", "dataset_phase75_curation_report.jld2")

@info "HTT: loading mutant raw data (replay) — using mut_control_cols (140Q mIgG-matched)"
mut_raw_data = load_data(raw_files, mut_sample_cols, mut_control_cols, 1, 1, false,
                         normalise_protocols = false, curate_replay = replay_path)

mnar_files = [dataset_mnar_phase75 for _ in 1:6]

@info "HTT: loading MNAR-imputed wt data (length-1 Vector for analyse-overload compatibility)"
wt_data  = InteractionData[load_data(mnar_files, wt_sample_cols,  wt_control_cols,  1, 1, false;
                                     normalise_protocols = false, curate_replay = replay_path,
                                     imputation = :mnar)]
@info "HTT: loading MNAR-imputed mut data (mut_control_cols)"
mut_data = InteractionData[load_data(mnar_files, mut_sample_cols, mut_control_cols, 1, 1, false;
                                     normalise_protocols = false, curate_replay = replay_path,
                                     imputation = :mnar)]

# --------------------------------------------------------------------------- #
# HAP40 column definitions (from examples/hap40_differential_interactome.jl) — Task 2/3
# --------------------------------------------------------------------------- #
hap40_strep_control_cols = [Dict(1 => [2,3,4], 2 => [5], 3 => [6,7])]
hap40_strep_sample_cols  = [Dict(1 => [8,9,10], 2 => [11,12,13], 3 => [14,15])]

gst_hap40_control_cols   = [Dict(1 => [2,3,4], 2 => [5,6,7], 3 => [8,9,10])]
gst_hap40_sample_cols    = [Dict(1 => [11,12,13], 2 => [14,15,16], 3 => [17,18,19])]

# --------------------------------------------------------------------------- #
# HAP40 load (Task 3)
# --------------------------------------------------------------------------- #
@info "HAP40_Strep: loading raw data"
hap40_strep_raw = load_data([HAP40_STREP_SRC], hap40_strep_sample_cols, hap40_strep_control_cols, 1, 1, false;
                            normalise_protocols = false, curate_interactive = false, imputation = :none)

@info "HAP40_Strep: loading MNAR-imputed data"
hap40_strep_data = InteractionData[load_data([hap40_strep_imputed], hap40_strep_sample_cols, hap40_strep_control_cols, 1, 1, false;
                                              normalise_protocols = false, curate_interactive = false, imputation = :mnar)]

@info "GST_HAP40: loading raw data"
hap40_gst_raw = load_data([HAP40_GST_SRC], gst_hap40_sample_cols, gst_hap40_control_cols, 1, 1, false;
                          normalise_protocols = false, curate_interactive = false, imputation = :none)

@info "GST_HAP40: loading MNAR-imputed data"
hap40_gst_data = InteractionData[load_data([hap40_gst_imputed], gst_hap40_sample_cols, gst_hap40_control_cols, 1, 1, false;
                                            normalise_protocols = false, curate_interactive = false, imputation = :mnar)]

####################################################################################
# CONFIG definitions (Task 2)
####################################################################################
wtHTT_config = CONFIG(
    sample_cols  = wt_sample_cols,
    control_cols = wt_control_cols,
    n_controls   = count_samples(wt_control_cols,    47),
    n_samples    = count_samples(wt_sample_cols,     44),
    refID        = 237,
    poi          = "9606.ENSP00000347184",          # STRING ID (matches encodings/9606.protein.info.v12.0.txt)

    datafile             = raw_files,
    output               = OutputFiles(joinpath(BASEPATH, "wtHTT"), image_ext = ".svg"),
    metalearner_path     = METALEARNER_PATH,
    normalise_protocols  = false,
    run_diagnostics      = true,

    # variance recovery via post-hoc inflation
    mnar_variance_recovery = :inflation,
)
wtHTT_config.output.results_file = joinpath(BASEPATH, "wtHTT", "wtHTT_meta_analysis_results.xlsx")

mHTT_config = CONFIG(
    sample_cols  = mut_sample_cols,
    control_cols = mut_control_cols,                # 140Q-matched
    n_controls   = count_samples(mut_control_cols,    47),
    n_samples    = count_samples(mut_sample_cols,     45),   # +1 dummy from Rep4B merge
    refID        = 237,
    poi          = "9606.ENSP00000347184",          # STRING ID (matches encodings/9606.protein.info.v12.0.txt)

    datafile             = raw_files,
    output               = OutputFiles(joinpath(BASEPATH, "mHTT"), image_ext = ".svg"),
    metalearner_path     = METALEARNER_PATH,
    normalise_protocols  = false,
    run_diagnostics      = true,

    mnar_variance_recovery = :inflation,
)
mHTT_config.output.results_file = joinpath(BASEPATH, "mHTT", "mHTT_meta_analysis_results.xlsx")

HAP40_Strep_config = CONFIG(
    datafile             = [HAP40_STREP_SRC],
    sample_cols          = hap40_strep_sample_cols,
    control_cols         = hap40_strep_control_cols,
    poi                  = "9606.ENSP00000479624",          # STRING ID (matches encodings/9606.protein.info.v12.0.txt)
    n_controls           = 6,
    n_samples            = 8,
    refID                = 1,

    output               = OutputFiles(joinpath(BASEPATH, "HAP40_Strep"), image_ext = ".svg"),
    metalearner_path     = METALEARNER_PATH,
    normalise_protocols  = false,
    plotHBMdists         = false,
    plotlog2fc           = false,
    plotregr              = false,
    plotbayesrange       = false,
    verbose              = false,
    vc_legend_pos        = :topleft,
    combination_method   = :bma,
    run_diagnostics      = true,
    optimize_nu          = true,
    curate_interactive   = false,
    run_input_qc         = true,                         # enable Data Quality tab for HAP40

    mnar_variance_recovery = :inflation,
)

HAP40_GST_config = CONFIG(
    datafile             = [HAP40_GST_SRC],
    sample_cols          = gst_hap40_sample_cols,
    control_cols         = gst_hap40_control_cols,
    poi                  = "9606.ENSP00000479624",          # STRING ID (matches encodings/9606.protein.info.v12.0.txt)
    n_controls           = 9,
    n_samples            = 9,
    refID                = 1,

    output               = OutputFiles(joinpath(BASEPATH, "GST_HAP40"), image_ext = ".svg"),
    metalearner_path     = METALEARNER_PATH,
    normalise_protocols  = false,
    plotHBMdists         = false,
    plotlog2fc           = false,
    plotregr             = false,
    plotbayesrange       = false,
    verbose              = false,
    vc_legend_pos        = :topleft,
    combination_method   = :bma,
    run_diagnostics      = true,
    optimize_nu          = true,
    curate_interactive   = false,
    run_input_qc         = true,                         # enable Data Quality tab for HAP40

    mnar_variance_recovery = :inflation,
)

####################################################################################
# Four independent run_analysis calls (Task 3)
# Idempotency: internal JLD2 caches (H0, BetaBernoulli, HBM regression,
# AnalysisResult) are auto-keyed by CONFIG hash + data hash. Re-running this
# script after a partial completion picks up cached artefacts and resumes from
# the latest uncached stage. Errors propagate (fail-fast).
####################################################################################
@info "wtHTT: starting run_analysis (~1-2 hr)"
_, ar_wtHTT       = run_analysis(wtHTT_config,        wt_data,         wt_raw_data)

@info "mHTT: starting run_analysis (~1-2 hr)"
_, ar_mHTT        = run_analysis(mHTT_config,         mut_data,        mut_raw_data)

@info "HAP40_Strep: starting run_analysis"
_, ar_hap40_strep = run_analysis(HAP40_Strep_config,  hap40_strep_data, hap40_strep_raw)

@info "GST_HAP40: starting run_analysis"
_, ar_hap40_gst   = run_analysis(HAP40_GST_config,    hap40_gst_data,   hap40_gst_raw)

####################################################################################
# k-group 4-condition differential (keyword-only overload)
####################################################################################
diff_4cond_base = joinpath(BASEPATH, "diff_4condition")
mkpath(diff_4cond_base)

@info "4-condition k-group differential analysis (wtHTT, mHTT, HAP40_Strep, GST_HAP40)"
diff_4cond = differential_analysis(;
    conditions = (
        wtHTT       = ar_wtHTT,
        mHTT        = ar_mHTT,
        HAP40_Strep = ar_hap40_strep,
        GST_HAP40   = ar_hap40_gst,
    ),
    contrasts = :all_pairs,
    config = DifferentialConfig(
        results_file        = joinpath(diff_4cond_base, "differential_results.xlsx"),
        volcano_file        = joinpath(diff_4cond_base, "differential_volcano.svg"),
        evidence_file       = joinpath(diff_4cond_base, "differential_evidence.svg"),
        scatter_file        = joinpath(diff_4cond_base, "differential_scatter.svg"),
        classification_file = joinpath(diff_4cond_base, "differential_classification.svg"),
        ma_file             = joinpath(diff_4cond_base, "differential_ma.svg"),
    ),
    multi_test_method = :bh,
    parallel_pairs    = :auto,
)

@info "Generating multi-condition differential report"
generate_differential_report(diff_4cond;
    output = joinpath(diff_4cond_base, "differential_report.html"))

@info "End-to-end run complete" outdir = BASEPATH
