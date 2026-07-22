# =============================================================================
# hap40_differential_interactome.jl — Differential interactome (HAP40-Strep vs GST-HAP40)
# =============================================================================
#
# What this script demonstrates
#   • Two single-protocol analyses (HAP40-Strep, GST-HAP40) with BMA
#   • MNAR imputation + post-hoc variance recovery via inflation
#     (mnar_variance_recovery = :inflation)
#   • DNN-prior + MC-Dropout uncertainty (run_dnn_prior_mc_dropout)
#     → adds the "DNN Prior" tab + 5 prior_* columns per condition
#   • differential_analysis(...) — 6-arg imputed overload for the head-to-head
#     comparison (runs run_analysis(config, imputed, raw) per condition)
#   • DifferentialConfig with bfdr_threshold (per-condition FDR threshold)
#   • Result columns BFDR_A, BFDR_B, differential_BFDR, PEP_A, PEP_B, diff_PEP
#   • Auto-generated differential_report.html
#   • Optional Phase D-style docking on the gained interactors of condition A
#
# Required input files
#   data/HAP40_Strep.xlsx
#   data/GST_HAP40.xlsx
#   metalearners/HistGradientBoosting_tune.jld2
#   encodings/model-473-0.5414302830201915.jld2   (MC-Dropout DNN)
#
# Recommended invocation (from repo root so the relative data/, encodings/,
# metalearners/ paths resolve):
#   julia --threads=auto --project=examples examples/hap40_differential_interactome.jl
#
# Expected runtime
#   • MNAR impute + two analyses + MC-Dropout (K=30) + differential pass:
#     ~45-90 min cold on this workstation. Reuses analysis_cache_*.jld2 +
#     imputed *_mnar.xlsx (mtime-keyed) on re-run.
#
# Cross-reference
#   docs/src/tutorial.md → Branch C — Differential interactome (two conditions)
#   docs/src/differential_analysis.md   for the column inventory + thresholds
#   docs/src/reports.md                 for the differential HTML report
# =============================================================================

# For local package development, activate the examples environment.
import Pkg
Pkg.activate(@__DIR__)  # Activate examples environment
# NOTE: Pkg.resolve() / Pkg.precompile() intentionally omitted (Fedora-script
# precedent). The examples/Manifest.toml is the authoritative environment
# (GLM already resolved); re-running resolve forces re-resolution against the
# BayesInteractomics GLM compat cap and fails. The first `using ...` below
# precompiles whatever stale caches need it.

Threads.nthreads()
# Trigger packages for the optional extensions:
using Flux, MLJ, MLJScikitLearnInterface, HDF5   # activates BayesInteractomicsMetalearnerExt (+ MC-Dropout)
using GLM                                         # activates BayesInteractomicsImputationExt (MNAR)
using BayesInteractomics
using Graphs, SimpleWeightedGraphs, GraphPlot, Compose, Cairo  # Triggers network extension
using XLSX, DataFrames                            # for the dropout-fit / MNAR-impute block

# Base directory for all outputs (per-condition dirs + imputed_data/ live here).
const BASEPATH = "C:/Users/Manuel/Desktop/HAP40_interactome_enrichment"

# --------------------------------------------------------------------------- #
# MNAR-imputation block — fit per-column dropout curves, then
# MNAR-impute each HAP40 source into a new xlsx matching the source layout.
# Idempotent: re-run reuses existing outputs when the input is unchanged
# (mtime check against the source file). Mirrors the Fedora driver.
# --------------------------------------------------------------------------- #
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
    # HAP40 intensity scale. NaN curves for columns with < 5 detections are tolerated.
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

hap40_strep_imputed = impute_hap40_dataset("data/HAP40_Strep.xlsx", "HAP40_Strep")
hap40_gst_imputed   = impute_hap40_dataset("data/GST_HAP40.xlsx",   "GST_HAP40")

# --------------------------------------------------------------------------- #
# Column definitions (sample_cols then control_cols — load_data positional order)
# --------------------------------------------------------------------------- #
hap40_strep_control_cols = [Dict(1 => [2,3,4], 2 => [5], 3 => [6,7])]
hap40_strep_sample_cols  = [Dict(1 => [8,9,10], 2 => [11,12,13], 3 => [14,15])]

gst_hap40_control_cols   = [Dict(1 => [2,3,4], 2 => [5,6,7], 3 => [8,9,10])]
gst_hap40_sample_cols    = [Dict(1 => [11,12,13], 2 => [14,15,16], 3 => [17,18,19])]

# --------------------------------------------------------------------------- #
# Load raw (for Beta-Bernoulli detection) + MNAR-imputed (for HBM/regression).
# --------------------------------------------------------------------------- #
# NOTE: filter_insufficient_obs=false on BOTH raw and imputed loads.
# The default n_obs<2 exclusion would drop sparse proteins from the raw load that the
# filled imputed file keeps, leaving raw + imputed index-MISALIGNED — the multi-imputation
# run_analysis(config, imputed, raw) indexes both by the same protein index and now hard-errors
# on misalignment. Keeping all proteins on both sides preserves index alignment; the imputed
# values make the sparse proteins analysable.
@info "HAP40_Strep: loading raw data"
hap40_strep_raw = load_data(["data/HAP40_Strep.xlsx"], hap40_strep_sample_cols, hap40_strep_control_cols, 1, 1, false;
                            normalise_protocols = false, curate_interactive = false, imputation = :none,
                            filter_insufficient_obs = false)

@info "HAP40_Strep: loading MNAR-imputed data"
hap40_strep_data = InteractionData[load_data([hap40_strep_imputed], hap40_strep_sample_cols, hap40_strep_control_cols, 1, 1, false;
                                             normalise_protocols = false, curate_interactive = false, imputation = :mnar,
                                             filter_insufficient_obs = false)]

@info "GST_HAP40: loading raw data"
hap40_gst_raw = load_data(["data/GST_HAP40.xlsx"], gst_hap40_sample_cols, gst_hap40_control_cols, 1, 1, false;
                          normalise_protocols = false, curate_interactive = false, imputation = :none,
                          filter_insufficient_obs = false)

@info "GST_HAP40: loading MNAR-imputed data"
hap40_gst_data = InteractionData[load_data([hap40_gst_imputed], gst_hap40_sample_cols, gst_hap40_control_cols, 1, 1, false;
                                           normalise_protocols = false, curate_interactive = false, imputation = :mnar,
                                           filter_insufficient_obs = false)]

# ====================== HAP40-Strep config ====================================== #
HAP40_Strep_config = CONFIG(
    datafile = ["data/HAP40_Strep.xlsx"],
    control_cols = hap40_strep_control_cols,
    sample_cols = hap40_strep_sample_cols,
    poi = "9606.ENSP00000479624",
    normalise_protocols = false,
    output = BayesInteractomics.OutputFiles(joinpath(BASEPATH, "HAP40_Strep"), image_ext=".svg"),
    n_controls = 6,
    n_samples = 8,
    refID = 1,
    plotHBMdists = false,
    plotlog2fc = false,
    plotregr = false,
    plotbayesrange = false,
    verbose = false,
    vc_legend_pos = :topleft,
    metalearner_path = "metalearners/HistGradientBoosting_tune.jld2",
    combination_method = :bma,
    run_diagnostics = true,
    optimize_nu = true,
    curate_interactive = false,
    run_input_qc = true,                        # enable Data Quality tab

    # MNAR variance recovery via post-hoc inflation
    mnar_variance_recovery = :inflation,
    # DNN-prior + MC-Dropout uncertainty (default true; explicit here)
    run_dnn_prior_mc_dropout = true,
    dnn_prior_mc_k = 30,
)

# ====================== GST-HAP40 config ====================================== #
HAP40_GST_config = CONFIG(
    datafile = ["data/GST_HAP40.xlsx"],
    control_cols = gst_hap40_control_cols,
    sample_cols = gst_hap40_sample_cols,
    poi = "9606.ENSP00000479624",
    normalise_protocols = false,
    output = BayesInteractomics.OutputFiles(joinpath(BASEPATH, "GST_HAP40"), image_ext=".svg"),
    n_controls = 9,
    n_samples = 9,
    refID = 1,
    plotHBMdists = false,
    plotlog2fc = false,
    plotregr = false,
    plotbayesrange = false,
    verbose = false,
    vc_legend_pos = :topleft,
    metalearner_path = "metalearners/HistGradientBoosting_tune.jld2",
    combination_method = :bma,
    run_diagnostics = true,
    optimize_nu = true,
    curate_interactive = false,
    run_input_qc = true,                        # enable Data Quality tab

    mnar_variance_recovery = :inflation,
    run_dnn_prior_mc_dropout = true,
    dnn_prior_mc_k = 30,
)

# Differential analysis — 6-arg imputed overload. Runs run_analysis(config,
# imputed, raw) for each condition (honouring MNAR imputation + variance
# inflation + MC-Dropout), then the head-to-head differential. The differential
# HTML report auto-generates (config.generate_report_html default true).
#
# Result DataFrame columns of interest (see docs/src/differential_analysis.md):
#   BFDR_A, BFDR_B          — per-condition Storey BFDR
#   PEP_A, PEP_B            — per-condition Posterior Error Probability
#   differential_BFDR, diff_PEP, delta_log2fc, classification (GAINED / REDUCED / UNCHANGED / ...)
#   prior_mc_mean, prior_mc_std, prior_mc_ci_low, prior_mc_ci_high, prior_contribution
diff_result = differential_analysis(
    HAP40_Strep_config,
    HAP40_GST_config,
    hap40_strep_data, hap40_strep_raw,
    hap40_gst_data,   hap40_gst_raw,
    condition_A = "HAP40-Strep",
    condition_B = "GST-HAP40",
    config = DifferentialConfig(
        posterior_threshold    = 0.8,
        bfdr_threshold         = 0.05,        # max BFDR for differential significance
        delta_log2fc_threshold = 1.0,
        classification_method  = :posterior,
        # Raw Δlog2FC (NOT z-scored): z-scoring undoes the bait_anchor and re-inflates the
        # bait to a spurious top hit. bait_anchor defaults ON and zeroes the bait Δlog2FC.
        standardize_log2fc     = false,
        results_file        = joinpath(BASEPATH, "differential_results.xlsx"),
        volcano_file        = joinpath(BASEPATH, "differential_volcano.svg"),
        evidence_file       = joinpath(BASEPATH, "differential_evidence.svg"),
        scatter_file        = joinpath(BASEPATH, "differential_scatter.svg"),
        classification_file = joinpath(BASEPATH, "differential_classification.svg"),
        ma_file             = joinpath(BASEPATH, "differential_ma.svg")
    ),
    scatter_metric = :posterior_prob
)

@info "Differential run complete" outdir = BASEPATH

#=
# ====================== Structural Docking ====================================== #
# Post-analysis docking via AlphaFold Server to distinguish direct from indirect
# interactors. This is a three-phase workflow:
#   Phase 1: Generate JSON request files for AlphaFold Server (automated)
#   Phase 2: Upload JSONs to alphafoldserver.com, download result ZIPs (manual)
#   Phase 3: Parse results and update posteriors with docking Bayes factor (automated)
#
# HAP40 (F8W1N3 / ENSP00000479624) — 371 residues
# UniProt: https://www.uniprot.org/uniprot/F8W1N3

HAP40_SEQUENCE = "MAAAAAGLGGGGAGPGPEAGDFLARYRLVSNKLKKRFLRKPNVAEAGEQFGQLGRELRAQE" *
                 "CLPYAAWCQLAVARCQQALFHGPGEALALTEAARLFLRQERDARQRLVCPAAYGEPLQAAA" *
                 "SALGAAVRLHLELGQPAAAAALCLELAAALRDLGQPAAAAGHFQRAAQLQLPQLPLAALQA" *
                 "LGEAASCQLLARDYTGALAVFTRMQRLAREHGSHPVQSLPPPPPPAPQPGPGATPALPAAL" *
                 "LPPNSGSAAPSPAALGAFSDVLVRCEVSRVLLLLLLQPPPAKLLPEHAQTLEKYSWEAFDS" *
                 "HGQESSGQLPEELFLLLQSLVMATHEKDTEAIKSLQVEMWPLLTAEQNHLLHLVLQETIS" *
                 "PSGQGV"

# --- Phase 1: Generate docking request JSONs ---
# Use result_A from differential_analysis (HAP40-Strep AnalysisResult)
# Only high-confidence hits (posterior >= 0.8, PEP <= 0.01) are selected.
result_A = diff_result.result_A

docking_output = joinpath(BASEPATH, "docking_requests")

docking_config = DockingConfig(
    posterior_threshold = 0.8,      # Only dock high-confidence hits
    pep_threshold       = 0.01,
    max_pairs           = 50,       # Cap at 50 pairs (~2 days of AF Server)
    max_tokens_per_job  = 5000,     # AF Server limit: bait + prey residues
    max_jobs_per_batch  = 30,       # AF Server daily limit
    parse_full_data     = true,     # Parse full_data JSONs for Tier 2 pDockQ scoring
    request_output_dir  = docking_output,
    verbose             = true,
)

batch = generate_docking_requests(
    result_A.results,
    HAP40_SEQUENCE;
    bait_name   = "HAP40",
    output_dir  = docking_output,
    fasta_file  = "",              # Leave empty to auto-fetch from UniProt
    config      = docking_config,
)

println("Generated $(batch.n_requests) docking requests in $(batch.n_batches) batch(es)")
println("Upload guide: $(batch.guide_path)")
println("Skipped (cached): $(batch.n_skipped_cached)")
println("Skipped (too large): $(batch.n_skipped_too_large)")

# --- Phase 2: Manual ---
# 1. Go to https://alphafoldserver.com
# 2. Upload each .json file from docking_requests/batch_1/, batch_2/, ...
# 3. Download all result ZIP files
# 4. Place them in a single directory, e.g.:
docking_results_dir = joinpath(BASEPATH, "docking_results")

# --- Phase 3: Parse results and update posteriors ---
# Run this after downloading all result ZIPs from AlphaFold Server.
# Uncomment when results are available:

#=
docking = import_docking_results(docking_results_dir, result_A.results; config=docking_config)

println("Docked: $(docking.n_docked) / $(docking.n_total)")
println("Pending: $(docking.n_pending)")
println("Disordered (BF=1): $(docking.n_disordered)")

# Apply two-stage Bayesian update: P_combined = odds_ms * BF_dock / (1 + odds_ms * BF_dock)
updated_results = apply_docking_update(result_A.results, docking)

# Inspect docking scores for top hits
using DataFrames
docked = filter(r -> r.docking_status == "success", updated_results)
sort!(docked, :posterior_prob_combined, rev=true)
println("\nTop docked interactors:")
for row in first(eachrow(docked), 10)
    println("  $(row.Protein): P(MS)=$(round(row.posterior_prob_ms, digits=3)) → " *
            "P(combined)=$(round(row.posterior_prob_combined, digits=3)) " *
            "[BF_dock=$(round(row.bf_docking, digits=2)), " *
            "ipTM=$(round(row.iptm_best, digits=3)), " *
            "pDockQ=$(round(row.pdockq, digits=3)), " *
            "tier=$(row.calibration_tier)]")
end

# Regenerate report with docking data included
generate_report(updated_results, HAP40_Strep_config;
                analysis_result = result_A,
                docking_result  = docking)
=#
=#
