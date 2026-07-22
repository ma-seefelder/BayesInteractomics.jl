# =============================================================================
# minimal_workflow.jl — The "if you read one example, read this" workflow
# =============================================================================
#
# What this script demonstrates (minimal end-to-end)
#   1. Loading data with STRING ID curation
#   2. Input data quality control (run_input_qc=true)
#   3. Bayesian analysis with BMA evidence combination (Copula + 3c-EM)
#   4. Parametric simulation + Platt calibration
#   5. Mixture-model validation gates
#   6. Opening the interactive HTML report
#
# Replace the four PLACEHOLDER values below with your own data layout, then run:
#     julia --project=examples examples/minimal_workflow.jl
#
# Cross-reference
#   docs/src/tutorial.md         → Branch A — Single dataset (one protocol)
#   docs/src/data_loading.md     for sample_cols / control_cols layout
#   docs/src/data_curation.md    for STRING curation + bait tracking
#   docs/src/data_quality_control.md         for run_input_qc gates
#   docs/src/simulation_calibration.md       for run_simulation + Platt
#   docs/src/model_evaluation.md             for BMA combination
#   docs/src/reports.md                      for the 9-tab HTML report
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__))
# Trigger packages for the optional extensions:
using Flux, MLJ, MLJScikitLearnInterface, HDF5   # activates BayesInteractomicsMetalearnerExt
using GLM                                         # activates BayesInteractomicsImputationExt
using BayesInteractomics

# --- 1. Specify input data --- PLACEHOLDER (replace with your own) -----------
data_file    = "path/to/your/data.xlsx"            # PLACEHOLDER: AP-MS data file (XLSX or CSV)
sample_cols  = [Dict(1 => [3, 4, 5])]               # PLACEHOLDER: bait sample replicate columns per protocol
control_cols = [Dict(1 => [6, 7, 8])]               # PLACEHOLDER: negative control replicate columns per protocol
bait_name    = "MYBAIT"                             # PLACEHOLDER: your bait protein name (used for refID tracking)

# --- 2. Load + curate (STRING ID resolution + protein-group splitting) -------
data, bait_idx = load_data(
    [data_file], sample_cols, control_cols;
    normalise_protocols = false,
    curate              = true,
    curate_interactive  = false,
    bait_name           = bait_name,
)

# --- 3. Configure the pipeline -----------------------------------------------
output = OutputFiles("./results", image_ext = ".svg")  # auto-fills all 30+ output paths

config = CONFIG(
    datafile              = [data_file],
    sample_cols           = sample_cols,
    control_cols          = control_cols,
    poi                   = bait_name,
    refID                 = bait_idx,
    n_controls            = sum(length.(values(control_cols[1]))),
    n_samples             = sum(length.(values(sample_cols[1]))),
    combination_method    = :bma,        # LOO stacking over Copula + 3c-EM (recommended)
    regression_likelihood = :robust_t,   # Student-t robust regression
    run_input_qc          = true,        # v1.1.5 input data QC gates
    run_validation        = true,        # mixture-model quality gates
    run_simulation        = true,        # parametric simulation + Platt calibration
    sim_n_synthetic       = 200,         # smaller grid for quick demo (default is 10_000)
    output                = output,
    generate_report_html  = true,
)

# --- 4. Run the pipeline -----------------------------------------------------
results, ar = run_analysis(config)

# --- 5. Inspect top hits + open the HTML report -----------------------------
top20 = first(sort(results, :posterior_prob; rev = true), 20)
show(top20[!, [:Protein, :posterior_prob, :PEP, :BFDR, :log2FC_mean]]; allrows = true)

println("\nInteractive HTML report: $(output.report_file)")
println("Open it in any browser — no server required.")
