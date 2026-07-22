# API Reference

A complete cross-reference index of all exported functions and types. Each symbol's full docstring is rendered on the appropriate topical page; the entries below link to the canonical documentation home.

For an alphabetical list of every documented symbol, see the [Index](#Index) at the end of this page.

## Pipeline & Configuration

Documented in [Analysis Pipeline](@ref).

- [`run_analysis`](@ref)
- [`CONFIG`](@ref)
- [`OutputFiles`](@ref)
- [`AnalysisResult`](@ref)
- [`save_result`](@ref)
- [`load_result`](@ref)
- [`evaluate_imputed_fc_posteriors`](@ref)
- [`BMAResult`](@ref)
- [`LatentClassResult`](@ref)

## Data Loading & Curation

Documented in [Data Loading](@ref) and [Data Curation](@ref).

- [`load_data`](@ref)
- [`InteractionData`](@ref)
- [`Protocol`](@ref)
- [`Protein`](@ref)
- [`getProteinData`](@ref)
- [`curate_proteins`](@ref)
- [`CurationReport`](@ref)
- [`CurationEntry`](@ref)
- [`CurationActionType`](@ref)
- [`MergeCandidate`](@ref)
- [`remove_contaminants`](@ref)
- [`parse_protein_id`](@ref)
- [`split_protein_groups`](@ref)

The following additional curation symbols are exported. Their docstrings render below:

```@docs
CurationAPIError
CurationCache
MergeDecision
resolve_to_string_ids
merge_protein_rows
confirm_merges_interactive
save_curation_report
load_curation_report
check_bait_detected
set_bait_info!
```

Normalisation dispatch (exported):

```@docs
apply_normalisation
norm_median_of_ratios_id
```

## Intermediate Caching

Documented in [Analysis Pipeline](@ref).

- [`BetaBernoulliCache`](@ref)
- [`HBMRegressionCache`](@ref)
- [`H0Cache`](@ref)
- [`IntermediateCacheStatus`](@ref)
- [`save_betabernoulli_cache`](@ref)
- [`load_betabernoulli_cache`](@ref)
- [`save_hbm_regression_cache`](@ref)
- [`load_hbm_regression_cache`](@ref)
- [`save_h0_cache`](@ref)
- [`load_h0_cache`](@ref)

## Statistical Models

Documented in [Model Fitting](@ref).

- [`BayesFactorHBM`](@ref)
- [`BayesFactorRegression`](@ref)
- [`HBMResult`](@ref)
- [`HBMResultSingleProtocol`](@ref)
- [`HBMResultMultipleProtocols`](@ref)
- [`RobustRegressionResult`](@ref)
- [`RobustRegressionResultMultipleProtocols`](@ref)
- [`RobustRegressionResultSingleProtocol`](@ref)
- [`NuOptimizationResult`](@ref)
- [`enrichment`](@ref)
- [`precompute_enrichment_prior`](@ref)
- [`log2FC`](@ref)

```@docs
AnyRegressionResult
```

## Evidence Combination

Documented in [Model Evaluation](@ref).

- [`BayesInteractomics.combined_BF_latent_class`](@ref)
- [`LatentClassResult`](@ref)
- [`BMAResult`](@ref)
- [`bma_weights_plot`](@ref)
- [`compute_em_responsibilities`](@ref)
- [`DiscreteEmpirical`](@ref)

## Model Comparison (WAIC)

Documented in [Model Fitting](@ref).

- [`compute_waic`](@ref)
- [`compare_regression_models`](@ref)
- [`optimize_nu`](@ref)
- [`WAICResult`](@ref)
- [`ModelComparisonResult`](@ref)

## Simulation

Documented in [Simulation & Calibration](@ref).

- [`run_simulation`](@ref)
- [`SimulationResult`](@ref)

## Diagnostics & Sensitivity

Documented in [Diagnostics](@ref) and [Prior Sensitivity](@ref).

- [`model_diagnostics`](@ref)
- [`generate_diagnostics_report`](@ref)
- [`sensitivity_analysis`](@ref)
- [`generate_sensitivity_report`](@ref)
- [`DiagnosticsConfig`](@ref)
- [`SensitivityConfig`](@ref)
- [`SensitivityResult`](@ref)
- [`CalibrationResult`](@ref)
- [`ResidualResult`](@ref)
- [`EnhancedResidualResult`](@ref)
- [`PPCExtendedStatistics`](@ref)
- [`ProteinPPC`](@ref)
- [`BetaBernoulliPPC`](@ref)
- [`ProteinDiagnosticFlag`](@ref)
- [`PriorSetting`](@ref)
- [`sensitivity_rank_correlation`](@ref)

The input-QC five-check result types and the diagnostics result container are exported. Their docstrings render below:

```@docs
DiagnosticsResult
ScaleCheckResult
ReplicateCorrelationResult
MissingnessResult
IntensityShapeResult
```

### Diagnostic Plots

Documented in [Diagnostics](@ref).

- [`ppc_density_plot`](@ref)
- [`ppc_pvalue_histogram`](@ref)
- [`residual_qq_plot`](@ref)
- [`residual_distribution_plot`](@ref)
- [`calibration_plot`](@ref)
- [`pit_histogram_plot`](@ref)
- [`nu_optimization_plot`](@ref)
- [`bb_ppc_summary_plot`](@ref)

## Quality Gates & Copula Diagnostics

Documented in [Diagnostics](@ref).

- [`run_quality_gates`](@ref)
- [`QualityGateResult`](@ref)
- [`QualityGateCell`](@ref)
- [`KLContaminationResult`](@ref)
- [`ValidationResult`](@ref)
- [`compute_kl_contamination`](@ref)
- [`kl_h1_divergence`](@ref)
- [`kl_divergence_plot`](@ref)
- [`within_class_correlation`](@ref)
- [`within_class_correlation_plot`](@ref)
- [`agnostic_zone_analysis`](@ref)
- [`agnostic_zone_plot`](@ref)
- [`copula_bootstrap_ci`](@ref)
- [`copula_bootstrap_plot`](@ref)
- [`discordant_protein_analysis`](@ref)
- [`discordant_protein_plot`](@ref)
- [`component_assignment_plot`](@ref)
- [`em_convergence_plot`](@ref)

## Differential Analysis

Documented in [Differential Interaction Analysis](@ref).

- [`differential_analysis`](@ref)
- [`DifferentialConfig`](@ref)
- [`DifferentialResult`](@ref)
- [`InteractionClass`](@ref)
- [`gained_interactions`](@ref)
- [`lost_interactions`](@ref)
- [`unchanged_interactions`](@ref)
- [`significant_differential`](@ref)
- [`export_differential`](@ref)
- [`differential_volcano_plot`](@ref)
- [`differential_evidence_plot`](@ref)
- [`differential_scatter_plot`](@ref)
- [`differential_classification_plot`](@ref)
- [`differential_ma_plot`](@ref)

## Embeddings

Documented in [Differential Interaction Analysis](@ref).

The embedding subsystem (DNN-derived per-protein embeddings, sample UMAP/t-SNE, and pairwise condition similarity) exposes the following exported types. Their docstrings render below:

```@docs
EmbeddingsConfig
EmbeddingsResult
ConditionSimilarityResult
```

## Visualization

Documented in [Visualization Guide](@ref).

- [`BayesInteractomics.plot_analysis`](@ref)
- [`BayesInteractomics.plot_results`](@ref)
- [`BayesInteractomics.evidence_plot`](@ref)
- [`BayesInteractomics.rank_rank_plot`](@ref)
- [`BayesInteractomics.volcano_plot`](@ref)
- [`BayesInteractomics.plot_inference_results`](@ref)
- [`BayesInteractomics.plot_log2fc`](@ref)
- [`BayesInteractomics.plot_regression`](@ref)
- [`BayesInteractomics.plot_bayesrange`](@ref)

## Report Generation

Documented in [Reports](@ref).

- [`generate_report`](@ref)
- [`generate_differential_report`](@ref)

## Network Analysis (Extension)

Documented in [Network Analysis](@ref). The functions below are stubs that activate when `Graphs`, `SimpleWeightedGraphs`, `GraphPlot`, and `Compose` are loaded.

- [`build_network`](@ref)
- [`network_statistics`](@ref)
- [`centrality_measures`](@ref)
- [`detect_communities`](@ref)
- [`plot_network`](@ref)
- [`save_network_plot`](@ref)
- [`export_graphml`](@ref)
- [`export_edgelist`](@ref)
- [`export_node_attributes`](@ref)
- [`centrality_dataframe`](@ref)
- [`community_dataframe`](@ref)
- [`get_top_hubs`](@ref)
- [`edge_source_summary`](@ref)
- [`NetworkConfig`](@ref)
- [`NetworkPipelineResult`](@ref)
- [`AbstractNetworkResult`](@ref)
- [`run_network_analysis`](@ref)
- [`generate_network_report`](@ref)

## PPI Enrichment

Documented in [Network Analysis](@ref).

- [`PPIEnrichmentConfig`](@ref)
- [`enrich_network`](@ref)
- [`query_string_ppi`](@ref)
- [`clear_ppi_cache`](@ref)
- [`ppi_cache_info`](@ref)

## Docking Integration (Two-stage Bayes Update)

Documented in [Docking Integration](@ref).

- [`DockingConfig`](@ref)
- [`DockingCalibration`](@ref)
- [`DockingPairResult`](@ref)
- [`DockingResult`](@ref)
- [`DockingRequestBatch`](@ref)
- [`compute_bf_dock`](@ref)
- [`apply_docking_update`](@ref)
- [`default_calibration`](@ref)
- [`compute_pdockq`](@ref)
- [`compute_bf_from_pdockq`](@ref)
- [`compute_bf_from_iptm`](@ref)
- [`compute_c2qscore`](@ref)
- [`compute_bf_from_c2qscore`](@ref)
- [`docking_cache_key`](@ref)
- [`generate_docking_requests`](@ref)
- [`import_docking_results`](@ref)

## Internal (exported)

The following symbols are present in the package's export list but are **internal by convention** (underscore-prefixed). They are documented here for completeness and to keep the export surface fully accounted for, but they are **not part of the supported public API** — their signatures and behaviour may change without notice. Their docstrings render below:

```@docs
_fit_discrete_empirical_weighted
_jitter_discrete_to_uniform
_merge_diagnostics_to_results
_replace_detection_marginal
```

## Index

```@index
```
