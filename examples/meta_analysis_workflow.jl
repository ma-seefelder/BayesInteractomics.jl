# =============================================================================
# meta_analysis_workflow.jl — Multi-protocol meta-analysis with multiple imputation
# =============================================================================
#
# What this script demonstrates
#   • Six-protocol meta-analysis of HTT (Huntingtin) AP-MS datasets from
#     four published studies (grecco × 4, gutierrez, sap), with substantial
#     missing data (~40%) handled via multiple imputation
#   • load_data(...) multi-protocol API with normalise_protocols = false and
#     dummy-column padding (cols 162-165) for absent experiments
#   • Curation report replay (.bayesinteractomics_cache/dataset_curation_report.jld2)
#     so all imputed loads see the same curated protein set
#   • run_analysis(config, imputed_data, raw_data) overload pooling Bayes
#     factors across imputations
#   • differential_analysis(wt_config, mut_config, wt_imputed, wt_raw,
#                           mut_imputed, mut_raw; ...) on the imputed data
#
# Required input files
#   $BASEPATH/dataset.xlsx                       (raw, used for Beta-Bernoulli)
#   $BASEPATH/imputed_data/dataset_imp_{1..5}.xlsx
#   metalearners/HistGradientBoosting_tune.jld2
#
# Expected runtime
#   • Curation (interactive on first wt load, replayed for mut + imputed): ~5 min
#   • run_analysis on each (wt, mut) with 5 imputations: ~1-2 hr each
#   • Differential pass: ~5 min
#
# Cross-reference
#   docs/src/tutorial.md → Branch B — Multiple protocols / meta-analysis
#   docs/src/data_loading.md            for the multi-protocol API
#   docs/src/data_curation.md           for replay + bait_name tracking
#   docs/src/model_evaluation.md        for BMA + multiple-imputation pooling
# =============================================================================

import Pkg
Pkg.activate(@__DIR__)  # Activate examples environment
Pkg.resolve()
Pkg.precompile()

# Trigger packages for the optional extensions:
using Flux, MLJ, MLJScikitLearnInterface, HDF5   # activates BayesInteractomicsMetalearnerExt
using GLM                                         # activates BayesInteractomicsImputationExt
using BayesInteractomics

const BASEPATH = "C:/Users/Manuel/Desktop/HTT_meta/"

function count_samples(x, dummy)
    total = 0
    for i in x
        n = sum([length(y) for (_, y) in i])
        total += n
    end
    return total - dummy
end

# --------------------------------------------------------------------------- #
# Column definitions
# --------------------------------------------------------------------------- #
# dummy: 162, 163, 164, 165

# Wild-type HTT samples — n_dummy: 10 + 13 + 12 + 1 + 4 + 4 = 44
wt_s_grecco    = Dict(1 => [2,3,4,162], 2 => [5,6,7,162], 3 => [162,163,164,165], 4 => [162,163,164,165])
wt_s_gutierrez = Dict(1 => [29,30,31,162], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
wt_s_sap       = Dict(1 => [36,37,38,39], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
wt_s_grecco_2  = Dict(1 => [48,49,50,51], 2 => [52,53,54,55], 3 => [60,61,62,63], 4 => [64,65,66,162])
wt_s_grecco_3  = Dict(1 => [92,93,94,162], 2 => [95,96,97,162], 3 => [100,101,102,162], 4 => [103,104,105,162])
wt_s_grecco_4  = Dict(1 => [128,129,130,162], 2 => [131,132,133,162], 3 => [137,138,139,162], 4 => [140,141,142,162])

# Mutant HTT samples — n_dummy: 10 + 16 + 12 + 1 + 2 + 3 = 44
mut_s_grecco    = Dict(1 => [8,9,10,162], 2 => [11,12,13,162], 3 => [162,163,164,165], 4 => [162,163,164,165])
mut_s_gutierrez = Dict(1 => [162,163,164,165], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
mut_s_sap       = Dict(1 => [40,41,42,43], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
mut_s_grecco_2  = Dict(1 => [70,71,72,162], 2 => [73,74,75,76], 3 => [81,82,83,84], 4 => [85,86,87,88])
mut_s_grecco_3  = Dict(1 => [109,110,111,112], 2 => [113,114,115,116], 3 => [119,120,121,162], 4 => [122,123,124,162])
mut_s_grecco_4  = Dict(1 => [146,147,148,149], 2 => [150,151,152,162], 3 => [156,157,158,162], 4 => [159,160,161,162])

# Non-HTT controls — n_dummy: 10 + 13 + 12 + 0 + 8 + 4 = 47
c_grecco    = Dict(1 => [14,15,16,162], 2 => [17,18,19,162], 3 => [162,163,164,165], 4 => [162,163,164,165])
c_gutierrez = Dict(1 => [26,27,28,162], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
c_sap       = Dict(1 => [32,33,34,35], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
c_grecco_2  = Dict(1 => [44,45,46,47], 2 => [44,45,46,47], 3 => [56,57,58,59], 4 => [56,57,58,59])
c_grecco_3  = Dict(1 => [90, 91, 162, 163], 2 => [90, 91, 162, 163], 3 => [98,99,162,163], 4 => [98,99,162,163])
c_grecco_4  = Dict(1 => [125,126,127,162], 2 => [125,126,127,162], 3 => [134,135,136,162], 4 => [134,135,136,162])

wt_sample_cols  = [wt_s_grecco, wt_s_gutierrez, wt_s_sap, wt_s_grecco_2, wt_s_grecco_3, wt_s_grecco_4]
mut_sample_cols = [mut_s_grecco, mut_s_gutierrez, mut_s_sap, mut_s_grecco_2, mut_s_grecco_3, mut_s_grecco_4]
control_cols    = [c_grecco, c_gutierrez, c_sap, c_grecco_2, c_grecco_3, c_grecco_4]

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
raw_files = [joinpath(BASEPATH, "dataset.xlsx") for _ in 1:6]

# Load raw data — curation runs interactively once (wt) and is replayed for mut
# (curate_interactive=false so the full pipeline can run autonomously
#  on a freshly-cleared cache; revert to default true for production use.)
wt_raw_data = load_data(raw_files, wt_sample_cols, control_cols, 1, 1, false,
                        normalise_protocols = false, curate_interactive = false)

# Replay path: raw-data curation report saved by load_data above
replay_path = joinpath(BASEPATH, ".bayesinteractomics_cache", "dataset_curation_report.jld2")

mut_raw_data = load_data(raw_files, mut_sample_cols, control_cols, 1, 1, false,
                         normalise_protocols = false, curate_replay = replay_path)

# Load MNAR-imputed dataset, wrapped in length-1 Vector for analyse(imputed_vec, raw, ...) compatibility
# The multi-imputation `analyse` overload accepts m=1; grow the vector to m=2..3 when variance recovery ships.
mnar_files = [joinpath(BASEPATH, "imputed_data/dataset_mnar.xlsx") for _ in 1:6]
wt_data  = InteractionData[load_data(mnar_files, wt_sample_cols,  control_cols, 1, 1, false;
                                     normalise_protocols = false, curate_replay = replay_path,
                                     imputation = :mnar)]
mut_data = InteractionData[load_data(mnar_files, mut_sample_cols, control_cols, 1, 1, false;
                                     normalise_protocols = false, curate_replay = replay_path,
                                     imputation = :mnar)]
@info "Loaded MNAR-imputed dataset (length-1 vector for analyse compatibility)"

####################################################################################
# CONFIG definitions
####################################################################################
wtHTT_config = CONFIG(
    sample_cols  = wt_sample_cols,
    control_cols = control_cols,
    n_controls   = count_samples(control_cols,    47),
    n_samples    = count_samples(wt_sample_cols,  44),
    refID        = 237,
    poi          = "ENSP00000347184",

    datafile             = raw_files,
    output               = OutputFiles(joinpath(BASEPATH, "wtHTT"), image_ext = ".svg"),
    metalearner_path     = "metalearners/HistGradientBoosting_tune.jld2",
    normalise_protocols  = false,
    run_diagnostics         = true
)
wtHTT_config.output.results_file = joinpath(BASEPATH, "wtHTT", "wtHTT_meta_analysis_results.xlsx")

mHTT_config = CONFIG(
    sample_cols  = mut_sample_cols,
    control_cols = control_cols,
    n_controls   = count_samples(control_cols,     47),
    n_samples    = count_samples(mut_sample_cols,  44),
    refID        = 237,
    poi          = "ENSP00000347184",

    datafile             = raw_files,
    output               = OutputFiles(joinpath(BASEPATH, "mHTT"), image_ext = ".svg"),
    metalearner_path     = "metalearners/HistGradientBoosting_tune.jld2",
    normalise_protocols  = false,
    run_diagnostics         = true
)
mHTT_config.output.results_file = joinpath(BASEPATH, "mHTT", "mHTT_meta_analysis_results.xlsx")

####################################################################################
# Differential analysis: wild-type HTT vs. mutant HTT
####################################################################################
diff_base = joinpath(BASEPATH, "wtHTT_vs_mHTT")

# diff::DifferentialResult, result_A/result_B::AnalysisResult
(; diff, result_A, result_B) = differential_analysis(
    wtHTT_config, mHTT_config,
    wt_data, wt_raw_data,
    mut_data, mut_raw_data,
    condition_A = "wtHTT",
    condition_B = "mHTT",
    config = DifferentialConfig(
        results_file        = joinpath(diff_base, "differential_results.xlsx"),
        volcano_file        = joinpath(diff_base, "differential_volcano.svg"),
        evidence_file       = joinpath(diff_base, "differential_evidence.svg"),
        scatter_file        = joinpath(diff_base, "differential_scatter.svg"),
        classification_file = joinpath(diff_base, "differential_classification.svg"),
        ma_file             = joinpath(diff_base, "differential_ma.svg")
    )
)
