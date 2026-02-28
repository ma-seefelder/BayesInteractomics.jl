#=
BayesInteractomics: A Julia package for the analysis of protein interactome data from Affinity-purification mass spectrometry (AP-MS) and proximity labelling experiments
# Version: 0.1.0

Copyright (C) 2024  Dr. rer. nat. Manuel Seefelder
E-Mail: manuel.seefelder@uni-ulm.de
Postal address: Department of Gene Therapy, University of Ulm, Helmholzstr. 8/1, 89081 Ulm, Germany

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
=#
module BayesInteractomics

# dependencies
import Base: vcat
import Base64
import CSV: read, File
import Dates: now, format
import Colors: distinguishable_colors
import DataFrames: AbstractDataFrame, DataFrame, rename!
import Distributions: Binomial, Cauchy, Gamma, MixtureModel, Normal, TDist, fit, params
import Random: AbstractRNG, GLOBAL_RNG, randperm
import MultivariateStats: PCA, PPCA, loadings
import Statistics: mean, median, quantile, std, var, cor
import StatsBase: Weights, sample
import StatsPlots
import ThreadsX
import XLSX: readtable, writetable
import QuadGK: quadgk

using Copulas
using LaTeXStrings
using LoggingExtras
using ProgressMeter
using RxInfer

# Core types and utilities
include("core/types.jl")
include("core/utils.jl")

# Diagnostics types (before results.jl so AnalysisResult can reference SensitivityResult)
include("diagnostics/types.jl")

# Data handling
include("data/loading.jl")
include("data/imputation.jl")

# Data curation (protein group splitting, synonym resolution, merging)
include("data/curation_types.jl")
include("data/string_api.jl")
include("data/curation.jl")

# Statistical inference
include("inference/betabernoulli.jl")
include("inference/models.jl")
include("inference/evaluation.jl")
include("inference/model_comparison.jl")
include("inference/visualization.jl")

# Evidence combination
include("combination/copula.jl")
include("combination/em_acceleration.jl")
include("combination/latent_class.jl")
include("combination/bma.jl")

# Diagnostics logic (must come after combination modules and before pipeline.jl)
include("diagnostics/sensitivity.jl")
include("diagnostics/sensitivity_plots.jl")
include("diagnostics/predictive_checks.jl")
include("diagnostics/residuals.jl")
include("diagnostics/calibration.jl")
include("diagnostics/diagnostic_plots.jl")

# Analysis workflows
include("analysis/pipeline.jl")
include("analysis/ranking.jl")
include("analysis/bfda.jl")

# Results and caching (must come after pipeline.jl for CONFIG)
include("core/results.jl")
include("core/intermediate_cache.jl")

# Visualization
include("visualization/plotting.jl")

# Differential interaction analysis
include("differential/types.jl")
include("differential/analysis.jl")
include("differential/visualization.jl")

# Machine learning
include("ml/pca.jl")
include("dnn/model.jl")
include("ml/metalearner.jl")

# Network analysis (stubs - extended by BayesInteractomicsNetworkExt)
include("network/stubs.jl")

# Interactive HTML report generation
include("reports/json_utils.jl")
include("reports/methods_generator.jl")
include("reports/report_generator.jl")

##############################################
# Public API
##############################################
# Data loading
public getNoExperiments, getExperiment, getIDs
public getIDs, getNames, getNoProtocols, getControls, getSamples
public getSampleMatrix, getControlMatrix
public getMatchedPositions, getExperimentPositions, getProtocolPositions

# Results and caching
public getProteins, getBayesFactors, getPosteriorProbabilities, getQValues
public getMeanLog2FC, getBaitProtein, getPosteriorProbs
public compute_config_hash, compute_data_hash
public compute_betabernoulli_hash, compute_hbm_regression_hash
public check_betabernoulli_cache, check_hbm_regression_cache
public get_betabernoulli_cache_filepath, get_hbm_regression_cache_filepath

# Model fitting
public getPosterior, getbfHBM, getbfRegression, getHBMstats, getregressionstats, clean_result

# Evidence combination
public combined_BF_latent_class, combined_BF_bma

# Model Evaluation
public log2FCStatistics, RegressionStatistics

# Model visualization
public write_txt

# Differential analysis accessors
public getDifferentialBayesFactors, getDifferentialPosteriors
public getDifferentialQValues, getClassifications, getDeltaLog2FC

# Exports
export load_data, run_analysis, BayesResult, getProteinData, CONFIG, OutputFiles
export Protocol, InteractionData, Protein
export BayesFactorHBM, BayesFactorRegression
export log2FC
export evaluate_imputed_fc_posteriors
export AbstractAnalysisResult, AnalysisResult, NetworkAnalysisResult
export save_result, load_result, CacheStatus
export BetaBernoulliCache, HBMRegressionCache, IntermediateCacheStatus
export save_betabernoulli_cache, load_betabernoulli_cache
export save_hbm_regression_cache, load_hbm_regression_cache
export set_bait_info!
export LatentClassResult, BMAResult

# Data curation
export curate_proteins, CurationReport, CurationActionType
export CurationEntry, MergeCandidate, MergeDecision, CurationCache
export CurationAPIError
export split_protein_groups, resolve_to_string_ids
export merge_protein_rows, confirm_merges_interactive
export save_curation_report, load_curation_report
export remove_contaminants, parse_protein_id

# Robust regression & model comparison
export RobustRegressionResult, AnyRegressionResult
export RobustRegressionResultMultipleProtocols, RobustRegressionResultSingleProtocol
export WAICResult, ModelComparisonResult, NuOptimizationResult
export compute_waic, compare_regression_models, optimize_nu

# Sensitivity analysis
export SensitivityConfig, SensitivityResult, PriorSetting
export sensitivity_analysis, generate_sensitivity_report
export sensitivity_tornado_plot, sensitivity_heatmap, sensitivity_rank_correlation

# Model diagnostics & posterior predictive checks
export DiagnosticsConfig, DiagnosticsResult
export ProteinPPC, BetaBernoulliPPC, ResidualResult, CalibrationResult
export EnhancedResidualResult, PPCExtendedStatistics, ProteinDiagnosticFlag
export model_diagnostics, generate_diagnostics_report, _merge_diagnostics_to_results
export ppc_density_plot, ppc_pvalue_histogram, pit_histogram_plot
export residual_qq_plot, residual_distribution_plot
export calibration_plot, calibration_comparison_plot, bb_ppc_summary_plot
export nu_optimization_plot

# Differential interaction analysis exports
export DifferentialConfig, DifferentialResult, InteractionClass
export GAINED, REDUCED, UNCHANGED, BOTH_NEGATIVE, CONDITION_A_SPECIFIC, CONDITION_B_SPECIFIC
export differential_analysis
export differential_volcano_plot, differential_evidence_plot, differential_scatter_plot
export differential_classification_plot, differential_ma_plot
export gained_interactions, lost_interactions, unchanged_interactions
export significant_differential, export_differential

# Interactive report exports
export generate_report, generate_differential_report

# Network analysis exports
export AbstractNetworkResult
export build_network, network_statistics, centrality_measures, detect_communities
export plot_network, save_network_plot
export export_graphml, export_edgelist, export_node_attributes
export centrality_dataframe, community_dataframe, get_top_hubs, edge_source_summary
export NetworkConfig, NetworkPipelineResult
export run_network_analysis, generate_network_report

# Documentation-required exports (symbols referenced in topic-page @docs blocks)
export analyse, check_cache
export betabernoulli
export probability_of_direction, pd_to_p_value
export combined_BF, fit_copula, compare_copulas, posterior_probability_from_bayes_factor
export scale_location_plot
export plot_analysis, plot_results, evidence_plot, rank_rank_plot, volcano_plot
export plot_inference_results, plot_log2fc, plot_regression, plot_bayesrange

# Prey-prey network enrichment exports
export PPIEnrichmentConfig
export enrich_network, query_string_ppi, clear_ppi_cache, ppi_cache_info

end
