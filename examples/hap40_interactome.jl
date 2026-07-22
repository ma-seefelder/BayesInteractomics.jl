# =============================================================================
# hap40_interactome.jl — Single-bait HAP40 interactome analysis (with docking)
# =============================================================================
#
# What this script demonstrates
#   • Multi-protocol single-bait analysis (GST-HAP40 + HAP40-Strep merged)
#   • BMA evidence combination with diagnostics + nu optimization
#   • Three-phase AlphaFold Server docking workflow at the end:
#         1. Generate JSON requests (automated)
#         2. Upload to alphafoldserver.com (manual)
#         3. Parse result ZIPs and update posteriors (automated)
#
# Required input files
#   data/GST_HAP40.xlsx     — GST-tagged HAP40 AP-MS dataset
#   data/HAP40_Strep.xlsx   — Strep-tagged HAP40 AP-MS dataset
#   metalearners/HistGradientBoosting_tune.jld2  — pretrained metalearner
#   (Optional, for Phase 3) AlphaFold Server result ZIPs in a single folder
#
# Expected runtime
#   • Analysis (Phases 1-9 of run_analysis): ~10-30 min on 8 cores
#   • Docking request generation: <1 min
#   • Docking result parsing (after manual upload): ~1-5 min per 50 pairs
#
# Cross-reference
#   Documentation walkthrough: docs/src/tutorial.md  → Branch A — Single dataset (one protocol)
#                              docs/src/tutorial.md  → Branch D — With AlphaFold docking validation
#                              docs/src/docking.md   for docking BF tier formulas + C2Qscore
#                              docs/src/reports.md   for the 9-tab interactive HTML report
# =============================================================================

# For local package development, activate the examples environment
import Pkg
Pkg.activate(@__DIR__)  # Activate examples environment
Pkg.resolve()
Pkg.precompile()

Threads.nthreads()
# Trigger packages for the optional extensions:
using Flux, MLJ, MLJScikitLearnInterface, HDF5   # activates BayesInteractomicsMetalearnerExt
using GLM                                         # activates BayesInteractomicsImputationExt
using BayesInteractomics
using Graphs, SimpleWeightedGraphs, GraphPlot, Compose, Cairo  # Triggers network extension

# ====================== wtHAP40 ====================================== #
basepath = "C:/Users/Manuel/Desktop/HAP40_interactome_enrichment/wtHAP40/"


# HAP40-Strep: column 16,17,18 are dummy columns missing
wtHAP40_config = BayesInteractomics.CONFIG(
    datafile = ["data/GST_HAP40.xlsx","data/HAP40_Strep.xlsx"],
    control_cols = [
        Dict(1 => [2,3,4], 2 => [5,6,7], 3 => [8,9,10]),    # 9
        Dict(1 => [2,3,4], 2 => [5], 3 => [6,7])            # 6
    ],
    sample_cols = [
        Dict(1 => [11,12,13], 2 => [14,15,16], 3 => [17,18,19]), # 9
        Dict(1 => [8,9,10], 2 => [11,12,13], 3 => [14,15])       # 8
    ],
    poi = "9606.ENSP00000479624",
    normalise_protocols = false,
    output = BayesInteractomics.OutputFiles(basepath, image_ext=".svg"),
    n_controls = 15,
    n_samples = 17,
    refID = 1,
    plotHBMdists = false,
    plotlog2fc = false,
    plotregr = false,
    plotbayesrange = false,
    verbose = false,
    vc_legend_pos = :topleft,
    metalearner_path = "metalearners/HistGradientBoosting_tune.jld2",
    combination_method = :bma,        # BMA: LOO stacking over Copula + 3c-EM
    run_diagnostics    = true,        # PPC + WAIC + per-protein flags
    optimize_nu        = true,        # WAIC-driven Student-t nu optimization
    run_input_qc       = true,        # v1.1.5 input data QC gates
    run_validation     = true,        # mixture-model quality gates
    run_simulation     = true,        # parametric simulation + Platt calibration
    curate_interactive = false        # non-interactive curation (runs from cache replay)
)

# Run analysis (uses built-in caching via use_cache=true)
wtHAP40_results, wtHAP40_ar = run_analysis(wtHAP40_config)


# Finde die neueste Cache-Datei
#using JLD2
#cache_dir = "data/.bayesinteractomics_cache"
#cache_files = filter(f -> startswith(f, "analysis_cache_") && endswith(f, ".jld2"), readdir(cache_dir))
#cache_path = joinpath(cache_dir, sort(cache_files, by=f -> mtime(joinpath(cache_dir, f)), rev=true)[1])

#wtHAP40_ar = load_result(cache_path)
#wtHAP40_results = wtHAP40_ar.results

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
# Only high-confidence hits (posterior >= 0.8, PEP <= 0.01) are selected.

docking_output = "C:/Users/Manuel/Desktop/HAP40_interactome_enrichment/docking_requests"

docking_config = DockingConfig(
    posterior_threshold = 0.8,      # Only dock high-confidence hits
    pep_threshold       = 0.01,
    max_pairs           = 200,      # Cap at 100 pairs (~4 days of AF Server)
    max_tokens_per_job  = 5000,     # AF Server limit: bait + prey residues
    max_jobs_per_batch  = 30,       # AF Server daily limit
    parse_full_data     = true,     # Parse full_data JSONs for Tier 2 pDockQ scoring
    request_output_dir  = docking_output,
    verbose             = true,
)


batch = generate_docking_requests(
    wtHAP40_results,
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

docking_results_dir = "C:/Users/Manuel/Desktop/HAP40_interactome_enrichment/docking_results"
#=
# --- Phase 3: Parse results and update posteriors ---
# Run this after downloading all result ZIPs from AlphaFold Server.
# Uncomment when results are available:
=#

docking = import_docking_results(docking_results_dir, wtHAP40_results; config=docking_config)

println("Docked: $(docking.n_docked) / $(docking.n_total)")
println("Pending: $(docking.n_pending)")
println("Disordered (BF=1): $(docking.n_disordered)")

# Apply two-stage Bayesian update: P_combined = odds_ms * BF_dock / (1 + odds_ms * BF_dock)
updated_results = apply_docking_update(wtHAP40_results, docking)

# Inspect docking scores for top hits
using DataFrames
docked = filter(r -> r.docking_status == "success", updated_results)
sort!(docked, :posterior_prob_combined, rev=true)
println("\nTop docked interactors:")
for row in first(eachrow(docked), 10)
    println("  $(row.Protein): P(MS)=$(round(row.posterior_prob_ms, digits=3)) -> " *
            "P(combined)=$(round(row.posterior_prob_combined, digits=3)) " *
            "[BF_dock=$(round(row.bf_docking, digits=2)), " *
            "ipTM=$(round(row.iptm_best, digits=3)), " *
            "pDockQ=$(round(row.pdockq, digits=3)), " *
            "tier=$(row.calibration_tier)]")
end

# Regenerate report with docking data included — merge with existing sidecar
# so calibration, simulation, diagnostics data is preserved
sidecar = BayesInteractomics._sidecar_path(wtHAP40_config.output.report_file)
generate_report(updated_results, wtHAP40_config;
                docking_result  = docking,
                sidecar_path    = sidecar)
