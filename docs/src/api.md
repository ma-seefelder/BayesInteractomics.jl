# API Reference

Complete documentation of all exported functions and types, organized by module.

## Pipeline & Configuration

```@docs
run_analysis
analyse
CONFIG
OutputFiles
```

## Data Loading & Curation

```@docs
load_data
InteractionData
Protocol
Protein
getProteinData
getSampleMatrix
getControlMatrix
curate_proteins
CurationReport
CurationEntry
CurationActionType
remove_contaminants
parse_protein_id
split_protein_groups
```

## Results & Caching

```@docs
BayesResult
AnalysisResult
CombinedBayesResult
save_result
load_result
check_cache
CacheStatus
set_bait_info!
getProteins
getBayesFactors
getPosteriorProbabilities
getQValues
getMeanLog2FC
getBaitProtein
```

## Statistical Models

### Beta-Bernoulli Detection Model

```@docs
betabernoulli
count_detections
prob_beta_greater
```

### Hierarchical Bayesian Model

```@docs
BayesFactorHBM
HBMResult
```

### Bayesian Regression

```@docs
BayesFactorRegression
RegressionResult
RobustRegressionResult
```

### Bayes Factor Computation

```@docs
calculate_bayes_factor
probability_of_direction
pd_to_p_value
log2FCStatistics
```

## Evidence Combination

### Copula Fitting

```@docs
combined_BF
fit_copula
compare_copulas
posterior_probability_from_bayes_factor
BayesFactorTriplet
PosteriorProbabilityTriplet
EMResult
```

### Latent Class

```@docs
LatentClassResult
```

### Model Comparison (WAIC)

```@docs
compute_waic
compare_regression_models
optimize_nu
WAICResult
ModelComparisonResult
```

## Multiple Imputation

```@docs
evaluate_imputed_fc_posteriors
```

## Diagnostics & Sensitivity

```@docs
model_diagnostics
sensitivity_analysis
DiagnosticsConfig
SensitivityConfig
SensitivityResult
CalibrationResult
ResidualResult
ProteinPPC
BetaBernoulliPPC
```

### Diagnostic Plots

```@docs
ppc_density_plot
ppc_pvalue_histogram
residual_qq_plot
scale_location_plot
residual_distribution_plot
calibration_plot
calibration_comparison_plot
pit_histogram_plot
nu_optimization_plot
bb_ppc_summary_plot
sensitivity_tornado_plot
sensitivity_heatmap
sensitivity_rank_correlation
```

## Differential Analysis

```@docs
differential_analysis
DifferentialConfig
DifferentialResult
InteractionClass
gained_interactions
lost_interactions
unchanged_interactions
significant_differential
export_differential
differential_volcano_plot
differential_evidence_plot
differential_scatter_plot
differential_classification_plot
differential_ma_plot
```

## Visualization

### Pipeline Plots

```@docs
plot_analysis
plot_results
evidence_plot
rank_rank_plot
volcano_plot
```

### Per-Protein Plots

```@docs
plot_inference_results
plot_log2fc
plot_regression
plot_bayesrange
write_txt
```

## Ranking

```@docs
Ranks
top
```

## Reports

```@docs
generate_report
generate_differential_report
```

## Network Analysis (Extension)

The following functions are stubs that are extended when `Graphs`, `SimpleWeightedGraphs`, `GraphPlot`, and `Compose` are loaded.

```@docs
build_network
network_statistics
centrality_measures
detect_communities
plot_network
save_network_plot
export_graphml
export_edgelist
export_node_attributes
```

## Index

```@index
```
