using Documenter
using BayesInteractomics

# This ensures that the package is loaded from the source in the parent directory
push!(LOAD_PATH, "../src/")

makedocs(;
    modules=[BayesInteractomics],
    authors="Manuel Seefelder <manuel.seefelder@uni-ulm.de>",
    sitename="BayesInteractomics.jl",
    format=Documenter.HTML(;
        canonical="https://ma-seefelder.github.io/BayesInteractomics.jl",
        edit_link="main",
        assets=String[],
        size_threshold = 300 * 1024,  # 300 KiB (api.md is ~244 KiB)
        size_threshold_warn = 200 * 1024,
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "User Guide" => [
            "Data Loading" => "data_loading.md",
            "Data Curation" => "data_curation.md",
            "Data Quality Control" => "data_quality_control.md",
            "Imputation" => "imputation.md",
            "Analysis Pipeline" => "analysis.md",
            "Configuration" => "configuration.md",
            "Model Fitting" => "model_fitting.md",
            "Model Evaluation" => "model_evaluation.md",
            "Prior Engine (Metalearner)" => "metalearner.md",
            "Diagnostics" => "diagnostics.md",
            "Simulation & Calibration" => "simulation_calibration.md",
            "Prior Sensitivity" => "prior_sensitivity.md",
            "Differential Analysis" => "differential_analysis.md",
            "Bayesian Decision Risk" => "decision_risk.md",
            "Visualization" => "visualization.md",
            "Reports" => "reports.md",
            "Network Analysis" => "network_analysis.md",
            "Docking Integration" => "docking.md",
            "Optional Features and Extensions" => "optional_features.md",
        ],
        "Examples" => "examples.md",
        "Mathematical Background" => "mathematical_background.md",
        "API Reference" => "api.md",
    ],
    warnonly = [:missing_docs, :autodocs_block, :docs_block],
)

deploydocs(;
    repo="github.com/ma-seefelder/BayesInteractomics.jl",
    devbranch="dev",
)
