# Examples

This page showcases real-world analysis workflows using BayesInteractomics. All example scripts are available in the `examples/` directory of the package repository.

## Example 1: HAP40 Interactome Analysis

This example demonstrates analyzing the interactome of HAP40 (Huntingtin-associated protein 40) using data from multiple experimental protocols.

### Scientific Context

HAP40 is a protein that associates with Huntingtin (HTT) and plays important roles in neurodegenerative diseases. This analysis identifies HAP40-interacting proteins by combining evidence from two different affinity purification approaches:
- GST-tagged HAP40 pulldown
- Strep-tagged HAP40 pulldown

### Data Structure

The experiment includes:
- **Two protocols**: GST-HAP40 and HAP40-Strep
- **Three experiments per protocol**: Representing different biological conditions or replicates
- **Multiple replicates**: 3 control and 3 sample replicates per experiment

### Complete Workflow

```julia
using BayesInteractomics

# Define base directory for results
basepath = "path/to/HAP40_analysis"

# Configure analysis with two protocols
config = CONFIG(
    # Input files - one per protocol
    datafile = [
        "data/GST_HAP40.xlsx",
        "data/HAP40_Strep.xlsx"
    ],

    # Control sample column mappings
    # Protocol 1 (GST-HAP40): 3 experiments with 3 replicates each
    # Protocol 2 (HAP40-Strep): 3 experiments with varying replicates
    control_cols = [
        Dict(1 => [2,3,4], 2 => [5,6,7], 3 => [8,9,10]),    # Protocol 1: 9 controls
        Dict(1 => [2,3,4], 2 => [5], 3 => [6,7])            # Protocol 2: 6 controls
    ],

    # Sample column mappings
    sample_cols = [
        Dict(1 => [11,12,13], 2 => [14,15,16], 3 => [17,18,19]), # Protocol 1: 9 samples
        Dict(1 => [8,9,10], 2 => [11,12,13], 3 => [14,15])       # Protocol 2: 8 samples
    ],

    # Bait protein specification
    poi = "9606.ENSP00000479624",  # UniProt ID for HAP40
    refID = 1,                      # HAP40 is in first row of data

    # Total sample counts
    n_controls = 15,  # 9 + 6
    n_samples = 17,   # 9 + 8

    # Normalization - IMPORTANT for multiple protocols
    normalise_protocols = true,

    # Output files
    H0_file = joinpath(basepath, "copula_H0.xlsx"),
    results_file = joinpath(basepath, "final_results.xlsx"),
    volcano_file = joinpath(basepath, "volcano_plot.svg"),
    convergence_file = joinpath(basepath, "convergence.svg"),
    evidence_file = joinpath(basepath, "evidence.svg"),

    # Visualization options
    vc_legend_pos = :topleft,
    plotHBMdists = false,   # Set true to plot individual protein distributions
    plotlog2fc = false,
    plotregr = false,
    verbose = true
)

# Run complete analysis
results = run_analysis(config)

# Examine top interactions
top_interactions = first(
    sort(results, :Posterior_Probability, rev=true),
    20
)

println("Top 20 HAP40 interactors:")
println(top_interactions[:, [:Protein, :Posterior_Probability, :log2FC_mean,
                              :BF_enrichment, :BF_correlation, :BF_detection]])

# Filter for high-confidence hits
high_confidence = filter(
    row -> row.Posterior_Probability > 0.95 && row.log2FC_mean > 1.0,
    results
)

println("\nHigh-confidence interactions (P > 0.95, log2FC > 1): $(nrow(high_confidence))")
```

### Analyzing Individual Protocols

You can also analyze each protocol separately to understand protocol-specific effects:

```julia
# GST-HAP40 only
gst_config = CONFIG(
    datafile = ["data/GST_HAP40.xlsx"],
    control_cols = [Dict(1 => [2,3,4], 2 => [5,6,7], 3 => [8,9,10])],
    sample_cols = [Dict(1 => [11,12,13], 2 => [14,15,16], 3 => [17,18,19])],
    poi = "9606.ENSP00000479624",
    refID = 1,
    n_controls = 9,
    n_samples = 9,
    normalise_protocols = false,  # Single protocol
    H0_file = joinpath(basepath, "GST_HAP40/copula_H0.xlsx"),
    results_file = joinpath(basepath, "GST_HAP40/results.xlsx"),
    volcano_file = joinpath(basepath, "GST_HAP40/volcano.svg")
)

gst_results = run_analysis(gst_config)

# HAP40-Strep only
strep_config = CONFIG(
    datafile = ["data/HAP40_Strep.xlsx"],
    control_cols = [Dict(1 => [2,3,4], 2 => [5], 3 => [6,7])],
    sample_cols = [Dict(1 => [8,9,10], 2 => [11,12,13], 3 => [14,15])],
    poi = "9606.ENSP00000479624",
    refID = 1,
    n_controls = 6,
    n_samples = 8,
    normalise_protocols = false,
    H0_file = joinpath(basepath, "HAP40_Strep/copula_H0.xlsx"),
    results_file = joinpath(basepath, "HAP40_Strep/results.xlsx"),
    volcano_file = joinpath(basepath, "HAP40_Strep/volcano.svg")
)

strep_results = run_analysis(strep_config)

# Compare results across protocols
using DataFrames, StatsBase

# Merge results
merged = innerjoin(
    gst_results[:, [:Protein, :Posterior_Probability => :PP_GST]],
    strep_results[:, [:Protein, :Posterior_Probability => :PP_Strep]],
    on = :Protein
)

# Calculate correlation
cor_pp = cor(merged.PP_GST, merged.PP_Strep)
println("Correlation between protocol posterior probabilities: $(round(cor_pp, digits=3))")

# Identify protocol-consistent hits
consistent_hits = filter(
    row -> row.PP_GST > 0.9 && row.PP_Strep > 0.9,
    merged
)
println("Proteins with high confidence in both protocols: $(nrow(consistent_hits))")
```

## Example 2: Meta-Analysis with Multiple Imputation

This advanced example demonstrates combining data from multiple experiments with substantial missing data using multiple imputation.

### Scientific Context

This analysis examines the interactome of Huntingtin (HTT), the protein implicated in Huntington's disease. Data come from 6 different published studies, each using different AP-MS protocols. The combined dataset has substantial missing data (~40%), requiring multiple imputation for robust inference.

### Workflow Overview

1. Impute missing data using an appropriate method (e.g., MissForest, KNN)
2. Create M imputed datasets (typically M=5)
3. Run BayesInteractomics on each imputed dataset
4. Pool results following Rubin's rules

### Data Preparation

```julia
using BayesInteractomics

const BASEPATH = "path/to/HTT_meta_analysis"

# Helper function to count total samples across experiments
function count_samples(column_dicts, n_dummy)
    total = 0
    for dict in column_dicts
        n = sum(length(cols) for (_, cols) in dict)
        total += n
    end
    return total - n_dummy
end

# Define column mappings for 6 protocols (studies)
# Note: Dummy columns (162-165) are used for padding where experiments are missing

# Sample columns for each protocol
s_grecco    = Dict(1 => [2,3,4,162], 2 => [5,6,7,162], 3 => [162,163,164,165], 4 => [162,163,164,165])
s_gutierrez = Dict(1 => [29,30,31,162], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
s_sap       = Dict(1 => [36,37,38,39], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
s_grecco_2  = Dict(1 => [48,49,50,51], 2 => [52,53,54,55], 3 => [60,61,62,63], 4 => [64,65,66,162])
s_grecco_3  = Dict(1 => [92,93,94,162], 2 => [95,96,97,162], 3 => [100,101,102,162], 4 => [103,104,105,162])
s_grecco_4  = Dict(1 => [128,129,130,162], 2 => [131,132,133,162], 3 => [137,138,139,162], 4 => [140,141,142,162])

# Control columns for each protocol
c_grecco    = Dict(1 => [14,15,16,162], 2 => [17,18,19,162], 3 => [162,163,164,165], 4 => [162,163,164,165])
c_gutierrez = Dict(1 => [26,27,28,162], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
c_sap       = Dict(1 => [32,33,34,35], 2 => [162,163,164,165], 3 => [162,163,164,165], 4 => [162,163,164,165])
c_grecco_2  = Dict(1 => [44,45,46,47], 2 => [44,45,46,47], 3 => [56,57,58,59], 4 => [56,57,58,59])
c_grecco_3  = Dict(1 => [90, 91, 162, 163], 2 => [90, 91, 162, 163], 3 => [98,99,162,163], 4 => [98,99,162,163])
c_grecco_4  = Dict(1 => [125,126,127,162], 2 => [125,126,127,162], 3 => [134,135,136,162], 4 => [134,135,136,162])

sample_cols = [s_grecco, s_gutierrez, s_sap, s_grecco_2, s_grecco_3, s_grecco_4]
control_cols = [c_grecco, c_gutierrez, c_sap, c_grecco_2, c_grecco_3, c_grecco_4]

# Number of dummy columns used for padding
n_dummy_samples = 44
n_dummy_controls = 47
```

### Loading Imputed Datasets

```julia
# Load M=5 imputed datasets
imputed_data = InteractionData[]

for i in 1:5
    # Each imputation uses the same column mappings
    files = [
        joinpath(BASEPATH, "imputed_data/dataset_imp_$i.xlsx"),
        joinpath(BASEPATH, "imputed_data/dataset_imp_$i.xlsx"),
        joinpath(BASEPATH, "imputed_data/dataset_imp_$i.xlsx"),
        joinpath(BASEPATH, "imputed_data/dataset_imp_$i.xlsx"),
        joinpath(BASEPATH, "imputed_data/dataset_imp_$i.xlsx"),
        joinpath(BASEPATH, "imputed_data/dataset_imp_$i.xlsx")
    ]

    data = load_data(
        files,
        sample_cols,
        control_cols,
        normalise_protocols = true
    )

    push!(imputed_data, data)
    @info "Loaded imputed dataset $i"
end

# Also load raw data for Beta-Bernoulli model (uses detection, not intensity)
raw_data = load_data(
    [joinpath(BASEPATH, "dataset.xlsx") for _ in 1:6],
    sample_cols,
    control_cols,
    normalise_protocols = true
)
```

### Running Analysis on Imputed Data

```julia
# Define analysis configuration
analysis_config = CONFIG(
    sample_cols = sample_cols,
    control_cols = control_cols,
    n_controls = count_samples(control_cols, n_dummy_controls),
    n_samples = count_samples(sample_cols, n_dummy_samples),
    refID = 237,  # Row index of HTT in dataset
    poi = "ENSP00000347184",  # UniProt ID for HTT

    # File paths
    datafile = [joinpath(BASEPATH, "dataset.xlsx") for _ in 1:6],
    H0_file = joinpath(BASEPATH, "wtHTT/copula_H0.xlsx"),
    results_file = joinpath(BASEPATH, "wtHTT/results.xlsx"),
    volcano_file = joinpath(BASEPATH, "wtHTT/volcano_plot.svg"),
    convergence_file = joinpath(BASEPATH, "wtHTT/convergence.svg"),
    evidence_file = joinpath(BASEPATH, "wtHTT/evidence.svg"),

    # Analysis parameters
    normalise_protocols = true,
    verbose = true
)

# Run analysis with multiple imputation
# This will:
# 1. Run Beta-Bernoulli on raw_data (detection model doesn't use imputed values)
# 2. Run HBM and Regression on each imputed dataset
# 3. Pool Bayes factors across imputations
results = run_analysis(analysis_config, imputed_data, raw_data)

# Results now incorporate uncertainty from both:
# - Statistical inference (Bayesian posterior)
# - Missing data (imputation variance)

println("Analysis complete. Found $(nrow(filter(r -> r.Posterior_Probability > 0.95, results))) high-confidence interactions.")
```

### Comparing Wild-type vs Mutant HTT

The meta-analysis example also includes comparing interactomes of wild-type versus mutant HTT:

```julia
# Configure comparison by swapping sample/control assignments
# Wild-type HTT as control, Mutant HTT as sample
comparison_config = CONFIG(
    sample_cols = [...],      # Mutant HTT columns
    control_cols = [...],     # Wild-type HTT columns
    # ... other parameters same as above
)

comparison_results = run_analysis(comparison_config, imputed_data, raw_data)

# Identify differential interactors
# High probability in mutant, low in wild-type
differential = filter(
    row -> row.Posterior_Probability > 0.9 && row.log2FC_mean > 2.0,
    comparison_results
)

println("Proteins preferentially enriched with mutant HTT: $(nrow(differential))")
```

## Example 3: Custom Visualization

BayesInteractomics provides flexible plotting functions. Here are examples of customizing visualizations:

### Volcano Plot Customization

```julia
using BayesInteractomics, StatsPlots

# After running analysis
results = run_analysis(config)

# Extract data for plotting
log2fc = results.log2FC_mean
neg_log_p = -log10.(1 .- results.Posterior_Probability)
posterior_prob = results.Posterior_Probability

# Create custom volcano plot
volcano = scatter(
    log2fc,
    neg_log_p,
    xlabel = "Log2 Fold Change",
    ylabel = "-Log10(1 - Posterior Probability)",
    title = "HAP40 Interactome Volcano Plot",
    marker_z = posterior_prob,
    color = :viridis,
    colorbar_title = "Posterior Probability",
    markersize = 3,
    alpha = 0.7,
    legend = false
)

# Add threshold lines
hline!([2], linestyle=:dash, color=:red, label="P > 0.99")
vline!([-1, 1], linestyle=:dash, color=:blue, label="2-fold change")

# Annotate top hits
top_n = 10
top_proteins = first(sort(results, :Posterior_Probability, rev=true), top_n)
for row in eachrow(top_proteins)
    annotate!(row.log2FC_mean, -log10(1 - row.Posterior_Probability),
              text(row.Protein, 6, :left))
end

savefig(volcano, "custom_volcano_plot.svg")
```

### Evidence Distribution Plots

```julia
# Plot distribution of individual Bayes factors
using StatsPlots

p1 = histogram(
    log10.(results.BF_enrichment),
    xlabel = "Log10(BF Enrichment)",
    ylabel = "Count",
    title = "Enrichment Evidence",
    bins = 50,
    alpha = 0.7,
    color = :blue
)

p2 = histogram(
    log10.(results.BF_correlation),
    xlabel = "Log10(BF Correlation)",
    ylabel = "Count",
    title = "Correlation Evidence",
    bins = 50,
    alpha = 0.7,
    color = :green
)

p3 = histogram(
    log10.(results.BF_detection),
    xlabel = "Log10(BF Detection)",
    ylabel = "Count",
    title = "Detection Evidence",
    bins = 50,
    alpha = 0.7,
    color = :red
)

evidence_plot = plot(p1, p2, p3, layout=(1,3), size=(1200, 400))
savefig(evidence_plot, "evidence_distributions.svg")
```

### Comparing Results Across Conditions

```julia
# Compare interactomes between two conditions
using DataFrames, StatsPlots

# Load results from two conditions
wt_results = run_analysis(wt_config)
mut_results = run_analysis(mut_config)

# Merge on protein ID
comparison = innerjoin(
    wt_results[:, [:Protein, :Posterior_Probability => :PP_WT, :log2FC_mean => :FC_WT]],
    mut_results[:, [:Protein, :Posterior_Probability => :PP_MUT, :log2FC_mean => :FC_MUT]],
    on = :Protein
)

# Scatter plot comparing posterior probabilities
scatter(
    comparison.PP_WT,
    comparison.PP_MUT,
    xlabel = "Posterior Probability (WT)",
    ylabel = "Posterior Probability (Mutant)",
    title = "Interactome Comparison: WT vs Mutant",
    markersize = 3,
    alpha = 0.6,
    color = :steelblue,
    legend = false
)

# Add diagonal line
plot!([0, 1], [0, 1], linestyle=:dash, color=:red, label="Equal")

# Annotate differentially enriched
differential = filter(row -> abs(row.PP_WT - row.PP_MUT) > 0.5, comparison)
for row in eachrow(differential)
    annotate!(row.PP_WT, row.PP_MUT, text(row.Protein, 5, :left))
end

savefig("wt_vs_mutant_comparison.svg")
```

## Tips for Real Analyses

### Data Quality Checks

Before running the full analysis, perform quality checks:

```julia
# Load data
data = load_data(files, sample_cols, control_cols)

# Check data dimensions
println("Number of proteins: $(getNoProteins(data))")
println("Number of protocols: $(getNoProtocols(data))")

# Check missing data percentage
function missing_percentage(data::InteractionData)
    total_cells = 0
    missing_cells = 0

    for protocol in data
        for (exp_id, exp_data) in protocol.experiments
            total_cells += length(exp_data)
            missing_cells += sum(ismissing.(exp_data))
        end
    end

    return 100 * missing_cells / total_cells
end

println("Missing data: $(round(missing_percentage(data), digits=1))%")
```

### Batch Processing Multiple Baits

For analyzing multiple bait proteins in one dataset:

```julia
# List of bait proteins
baits = [
    (name="HAP40", id="ENSP00000479624", refID=1),
    (name="HTT", id="ENSP00000347184", refID=237),
    (name="PROTEIN_X", id="ENSP00000123456", refID=500)
]

# Analyze each bait
all_results = Dict()

for bait in baits
    println("Analyzing $(bait.name)...")

    config = CONFIG(
        # ... common parameters ...
        poi = bait.id,
        refID = bait.refID,
        results_file = "results_$(bait.name).xlsx"
    )

    results = run_analysis(config)
    all_results[bait.name] = results

    println("  Found $(nrow(filter(r -> r.Posterior_Probability > 0.95, results))) interactions")
end
```

## Further Resources

- See `examples/hap40_interactome.jl` for the complete HAP40 analysis script
- See `examples/meta_analysis_workflow.jl` for the full HTT meta-analysis with multiple imputation
- Check the [Tutorial](tutorial.md) for step-by-step guidance
- Refer to [Mathematical Background](mathematical_background.md) for details on the statistical models
- Browse [API Reference](analysis.md) for function documentation
