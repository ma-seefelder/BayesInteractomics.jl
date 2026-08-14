# =============================================================================
# meta_analysis_macro.jl — @interactomics version of meta_analysis_workflow.jl
# =============================================================================
#
# What this script demonstrates
#   • The same six-protocol HTT meta-analysis as meta_analysis_workflow.jl,
#     rewritten with the @interactomics macro DSL
#   • Column layouts via @protocol / @experiment — only real replicates listed,
#     dummy columns and missing experiments auto-padded by the macro
#   • n_samples / n_controls auto-computed (no manual count_samples helper)
#   • CONFIG construction, run_analysis, and differential_analysis all driven
#     by the macro — no boilerplate
#   • MNAR variance recovery handled in-process by run_analysis(config) via
#     mnar_variance_recovery = :multi_impute (no external imputed files)
#
# Required input files
#   $BASEPATH/dataset.xlsx
#   metalearners/HistGradientBoosting_tune.jld2
#
# Cross-reference
#   examples/meta_analysis_workflow.jl  — original (manual) version
#   docs/src/tutorial.md                — Branch B — Multiple protocols
# =============================================================================

import Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.precompile()

using Flux, MLJ, MLJScikitLearnInterface, HDF5   # BayesInteractomicsMetalearnerExt
using GLM                                         # BayesInteractomicsImputationExt
using BayesInteractomics

const BASEPATH = "C:/Users/Manuel/Desktop/HTT_meta/"
const FILE     = joinpath(BASEPATH, "dataset.xlsx")
const DIFF_DIR = joinpath(BASEPATH, "wtHTT_vs_mHTT")

# =========================================================================== #
# Differential meta-analysis: wild-type HTT vs. mutant HTT
# =========================================================================== #
#
# Six protocols from four studies (grecco ×4, gutierrez, sap).
# Each protocol has up to 4 experiments.  Only real replicates are listed;
# dummy=[162,163,164,165] is declared once per condition — the macro
# auto-pads both missing experiments and within-experiment width differences.
# =========================================================================== #

(; diff, result_A, result_B) = @interactomics begin

    # ── Wild-type HTT ────────────────────────────────────────────────────
    @condition wtHTT begin
        # Grecco — 2 experiments
        @protocol FILE begin
            @experiment samples=[2,3,4]       controls=[14,15,16]
            @experiment samples=[5,6,7]       controls=[17,18,19]
        end
        # Gutierrez — 1 experiment
        @protocol FILE begin
            @experiment samples=[29,30,31]    controls=[26,27,28]
        end
        # SAP — 1 experiment
        @protocol FILE begin
            @experiment samples=[36,37,38,39] controls=[32,33,34,35]
        end
        # Grecco 2 — 4 experiments (all real)
        @protocol FILE begin
            @experiment samples=[48,49,50,51] controls=[44,45,46,47]
            @experiment samples=[52,53,54,55] controls=[44,45,46,47]
            @experiment samples=[60,61,62,63] controls=[56,57,58,59]
            @experiment samples=[64,65,66]    controls=[56,57,58,59]
        end
        # Grecco 3 — 4 experiments
        @protocol FILE begin
            @experiment samples=[92,93,94]    controls=[90,91]
            @experiment samples=[95,96,97]    controls=[90,91]
            @experiment samples=[100,101,102] controls=[98,99]
            @experiment samples=[103,104,105] controls=[98,99]
        end
        # Grecco 4 — 4 experiments
        @protocol FILE begin
            @experiment samples=[128,129,130] controls=[125,126,127]
            @experiment samples=[131,132,133] controls=[125,126,127]
            @experiment samples=[137,138,139] controls=[134,135,136]
            @experiment samples=[140,141,142] controls=[134,135,136]
        end

        dummy = [162,163,164,165]
        bait    = "ENSP00000347184"
        bait_id = 237
        output  = OutputFiles(joinpath(BASEPATH, "wtHTT"), image_ext=".svg")
        metalearner_path       = "metalearners/HistGradientBoosting_tune.jld2"
        normalise_protocols    = false
        run_diagnostics        = true
        mnar_variance_recovery = :multi_impute
        mnar_m                 = 3
    end

    # ── Mutant HTT ───────────────────────────────────────────────────────
    @condition mHTT begin
        # Grecco — 2 experiments
        @protocol FILE begin
            @experiment samples=[8,9,10]      controls=[14,15,16]
            @experiment samples=[11,12,13]    controls=[17,18,19]
        end
        # Gutierrez — no real mut samples; controls still contribute
        @protocol FILE begin
            @experiment samples=[]            controls=[26,27,28]
        end
        # SAP — 1 experiment
        @protocol FILE begin
            @experiment samples=[40,41,42,43] controls=[32,33,34,35]
        end
        # Grecco 2 — 4 experiments (all real)
        @protocol FILE begin
            @experiment samples=[70,71,72]      controls=[44,45,46,47]
            @experiment samples=[73,74,75,76]   controls=[44,45,46,47]
            @experiment samples=[81,82,83,84]   controls=[56,57,58,59]
            @experiment samples=[85,86,87,88]   controls=[56,57,58,59]
        end
        # Grecco 3 — 4 experiments
        @protocol FILE begin
            @experiment samples=[109,110,111,112] controls=[90,91]
            @experiment samples=[113,114,115,116] controls=[90,91]
            @experiment samples=[119,120,121]     controls=[98,99]
            @experiment samples=[122,123,124]     controls=[98,99]
        end
        # Grecco 4 — 4 experiments
        @protocol FILE begin
            @experiment samples=[146,147,148,149] controls=[125,126,127]
            @experiment samples=[150,151,152]     controls=[125,126,127]
            @experiment samples=[156,157,158]     controls=[134,135,136]
            @experiment samples=[159,160,161]     controls=[134,135,136]
        end

        dummy = [162,163,164,165]
        bait    = "ENSP00000347184"
        bait_id = 237
        output  = OutputFiles(joinpath(BASEPATH, "mHTT"), image_ext=".svg")
        metalearner_path       = "metalearners/HistGradientBoosting_tune.jld2"
        normalise_protocols    = false
        run_diagnostics        = true
        mnar_variance_recovery = :multi_impute
        mnar_m                 = 3
    end

    # ── Differential comparison ──────────────────────────────────────────
    @compare wtHTT mHTT config=DifferentialConfig(
        results_file        = joinpath(DIFF_DIR, "differential_results.xlsx"),
        volcano_file        = joinpath(DIFF_DIR, "differential_volcano.svg"),
        evidence_file       = joinpath(DIFF_DIR, "differential_evidence.svg"),
        scatter_file        = joinpath(DIFF_DIR, "differential_scatter.svg"),
        classification_file = joinpath(DIFF_DIR, "differential_classification.svg"),
        ma_file             = joinpath(DIFF_DIR, "differential_ma.svg")
    )
end
