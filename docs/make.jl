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
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "User Guide" => [
            "Data Loading" => "data_loading.md",
            "Data Curation" => "data_curation.md",
            "Analysis Pipeline" => "analysis.md",
            "Model Fitting" => "model_fitting.md",
            "Model Evaluation" => "model_evaluation.md",
            "Diagnostics" => "diagnostics.md",
            "Differential Analysis" => "differential_analysis.md",
            "Visualization" => "visualization.md",
            "Reports" => "reports.md",
            "Network Analysis" => "network_analysis.md",
        ],
        "Examples" => "examples.md",
        "Mathematical Background" => "mathematical_background.md",
        "API Reference" => "api.md",
    ],
    warnonly = [:missing_docs, :cross_references, :autodocs_block, :docs_block],
)

deploydocs(;
    repo="github.com/ma-seefelder/BayesInteractomics.jl",
    devbranch="master",
)
