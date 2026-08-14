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
import CSV
import Dates: now, format, DateTime
import Colors: distinguishable_colors
import DataFrames: AbstractDataFrame, DataFrame, rename!
import Distributions: Binomial, Cauchy, Gamma, LocationScale, MixtureModel, Normal, TDist, fit, params
import LinearAlgebra: I
import Random
import Random: AbstractRNG, GLOBAL_RNG, randperm
import MultivariateStats: PCA, PPCA, loadings
import Statistics: mean, median, quantile, std, var, cor
import StatsBase: Weights, sample, mad, corspearman
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
include("diagnostics/bb_mnar_codriven.jl")  # depends on BBMnarCodrivenConfig from types.jl

# Data handling
include("data/loading.jl")
include("data/imputation_stubs.jl")  # stub-and-override declarations for GLM-dependent imputation
include("data/imputation.jl")        # MAR/MICE wrapper — STAYS IN CORE (no GLM dep; per RESEARCH Q2 RESOLVED)
include("data/missingness.jl")       # _compute_per_protein_missingness helper (pre-imputation aggregation)
# dropout.jl + imputation_mnar.jl now live in
# ext/BayesInteractomicsImputationExt/. The imputation_stubs.jl above declares
# the empty function generics (fit_dropout_curves, save_dropout_fit, load_dropout_fit,
# impute_mnar, impute_mnar_from_paths, column_imputation_sigma) and the concrete DropoutFit struct; the
# extension attaches methods when the user loads `using GLM`. transition
# guard (@static if Base.find_package("GLM") ...) removed.

# Data curation (wrapped in submodule)
module Curation
    # Bring parent module's public symbols into scope.
    # NOTE: per deviation pattern — `using ..BayesInteractomics` would
    # only pull EXPORTED names, AND because the parent module is mid-definition
    # when this submodule is being compiled, its export list isn't fully
    # observable yet. The three curation files reference NO parent BayesInteractomics
    # types (they are fully self-contained), so we omit `import ..BayesInteractomics`
    # entirely. Internal cross-references between curation_types.jl, string_api.jl,
    # and curation.jl resolve inside this submodule's namespace.

    # Re-import external deps used by the three curation files at module scope.
    # Julia submodules do NOT inherit `using X` / `import X` from the parent —
    # every external package referenced inside the submodule must be brought
    # in explicitly here.
    import DataFrames: DataFrame, AbstractDataFrame, nrow, ncol, eachrow,
                       names as df_names, select!, insertcols!
    import Dates
    import Dates: DateTime, now, Millisecond, value
    import Downloads
    import SHA: sha256
    import CSV
    import Statistics: mean
    using JLD2
    # @info / @warn are core Base macros — no import needed.

    include("data/curation_types.jl")
    include("data/string_api.jl")
    include("data/curation.jl")
end  # module Curation

# Re-export the 16 public curation symbols at the top level so existing user
# code (`using BayesInteractomics; curate_proteins(...)`) and the existing
# `export` block below resolve unchanged.
using .Curation: curate_proteins, CurationReport, CurationActionType,
                 CurationEntry, MergeCandidate, MergeDecision, CurationCache,
                 CurationAPIError,
                 split_protein_groups, resolve_to_string_ids,
                 merge_protein_rows, confirm_merges_interactive,
                 save_curation_report, load_curation_report,
                 remove_contaminants, parse_protein_id

# Forward Curation-internal helpers + enum values to the parent namespace so
# existing test code (`BayesInteractomics.MergeCandidate`,
# `BayesInteractomics.find_merge_candidates`, `BayesInteractomics.CURATE_SPLIT`,
# etc.) resolves without modification. These are NOT exported — they remain
# implementation details — but they ARE reachable as `BayesInteractomics.X`.
using .Curation: CURATE_KEEP, CURATE_SPLIT, CURATE_MERGE, CURATE_RENAME,
                 CURATE_REMOVE, CURATE_UNMAPPED,
                 find_merge_candidates, find_bait_index, replay_merges,
                 save_curation_cache, load_curation_cache,
                 _curation_cache_key, _deduplicate_same_name_rows

# Input data quality control
include("qc/types.jl")
include("qc/scale_detection.jl")
include("qc/replicate_correlation.jl")
include("qc/missingness.jl")
include("qc/distribution_shape.jl")
include("qc/pca_separation.jl")
include("qc/run_qc.jl")

# Statistical inference
include("inference/betabernoulli.jl")
include("inference/models.jl")
include("inference/evaluation.jl")
include("inference/model_comparison.jl")
include("inference/visualization.jl")

# Evidence combination
include("combination/beta_mixture.jl")
include("combination/copula.jl")
include("combination/em_acceleration.jl")
include("combination/empirical_bayes.jl")
include("combination/latent_class.jl")
include("combination/bma.jl")

# Diagnostics logic (must come after combination modules and before pipeline.jl)
include("diagnostics/sensitivity.jl")
include("diagnostics/sensitivity_plots.jl")
include("diagnostics/predictive_checks.jl")
include("diagnostics/residuals.jl")
include("diagnostics/calibration.jl")
include("diagnostics/diagnostic_plots.jl")
include("diagnostics/copula_diagnostics.jl")
include("diagnostics/variance_inflation.jl")  # MNAR variance recovery (post-hoc CI widening; hard-lock; loads after core/types.jl + data/imputation_stubs.jl)

# Docking integration (wrapped in submodule)
#
# CRITICAL ORDERING: this `module Docking ... end` block MUST be defined
# BEFORE `include("analysis/pipeline.jl")` because `CONFIG.docking_config`
# (declared in pipeline.jl) is annotated `::Union{DockingConfig, Nothing}`
# and `pipeline.jl` also references `generate_docking_requests` unqualified.
# Both symbols are brought into parent scope via the `using .Docking:`
# re-export immediately below, which runs at compile time before pipeline.jl
# is included.
module Docking
    # Bring parent symbols referenced inside the docking files into scope.
    # NOTE: per Plans 07/08/09 deviation pattern — `using ..BayesInteractomics`
    # would only pull EXPORTED names, AND because the parent module is
    # mid-definition when this submodule is compiled, its export list isn't
    # fully observable yet. We rely on explicit `import ..BayesInteractomics:`
    # for every parent symbol referenced inside the submodule. Only `bfdr`
    # (defined in src/core/utils.jl, included earlier) is called from
    # `apply_docking_update` in docking/stubs.jl:480.
    import ..BayesInteractomics: bfdr

    # Re-import external deps used by the docking files at module scope.
    # Julia submodules do NOT inherit `using X` / `import X` from the parent —
    # every external package referenced inside the submodule must be brought
    # in explicitly here.
    import DataFrames: DataFrame, nrow, hasproperty, eachrow
    import JSON3
    import JLD2
    import Downloads
    import Dates: DateTime, now, Millisecond, Day, value, format
    import Statistics: mean, std
    using ProgressMeter

    include("docking/stubs.jl")
    # stubs.jl transitively includes cache.jl, sequence_retrieval.jl,
    # request_generator.jl, alphafold_db.jl, full_data_parser.jl, and
    # result_parser.jl (see its bottom-of-file include block).
end  # module Docking

# Re-export the 16 public docking symbols at the top level so existing user
# code (`using BayesInteractomics; DockingConfig`) and the `export` block
# at the bottom of this file resolve unchanged. This MUST appear before
# `include("analysis/pipeline.jl")` so the CONFIG struct definition
# (with `docking_config::Union{DockingConfig, Nothing}`) and the
# `generate_docking_requests(...)` call inside the pipeline both resolve
# against parent-level bindings.
using .Docking: DockingConfig, DockingCalibration, DockingPairResult,
                DockingResult, DockingRequestBatch,
                compute_bf_dock, apply_docking_update, default_calibration,
                compute_pdockq, compute_bf_from_pdockq, compute_bf_from_iptm,
                compute_bf_from_c2qscore, docking_cache_key, compute_c2qscore,
                generate_docking_requests, import_docking_results

# Forward Docking-internal helpers + constants to the parent namespace so
# existing test code (`using BayesInteractomics: _bayesian_update_log_odds`,
# `BayesInteractomics._compute_ipae`, `BayesInteractomics.C2QSCORE_AF3_WEIGHTS`,
# `BayesInteractomics._load_cached_pair`, `BayesInteractomics.FullDataScores`,
# etc.) resolves without modification. These are NOT exported — they remain
# implementation details — but they ARE reachable as `BayesInteractomics.X`.
using .Docking: _bayesian_update_log_odds, _derive_epsilon,
                C2QSCORE_AF3_WEIGHTS, C2QSCORE_AF3_BIAS,
                C2QSCORE_LOGISTIC_COEF, C2QSCORE_LOGISTIC_INTERCEPT,
                _compute_ipae,
                _load_cached_pair, _save_cached_pair,
                FullDataScores

# include EmbeddingsConfig / EmbeddingsResult /
# ConditionSimilarityResult type contracts BEFORE analysis/pipeline.jl because
# CONFIG has an `embeddings_config::EmbeddingsConfig` field, and BEFORE
# core/results.jl because AnalysisResult has an
# `embeddings::Union{Nothing, EmbeddingsResult}` field . The file has no
# upstream dependencies (pure type contracts + validators). The other
# files (stubs, sample/protein/condition compute) stay in their original
# location below — only the type contract file must move forward.
include("embeddings/embedding_types.jl")

# Analysis workflows
include("analysis/pipeline.jl")
include("analysis/ranking.jl")
include("analysis/bfda.jl")

# JSON utilities + Simulation included BEFORE core/results.jl
# AnalysisResult.simulation_result::Union{Nothing, SimulationResult} requires
# SimulationResult to be in scope at the time core/results.jl is parsed. The
# simulation engine has no upstream dependency on CONFIG, AnalysisResult, or
# the cache types — only LatentClassResult (in core/types.jl, included earlier),
# Optim, Distributions. json_utils.jl is moved alongside because simulation.jl's
# `_build_simulation_json` references json_object/json_number/json_array; though
# these are runtime-resolved, keeping the declaration order coherent reduces
# surprise. (Pre-Phase-69 order had simulation.jl AFTER results.jl, around
# line ~324; we relocated it here as part of to allow the typed field.)
include("reports/json_utils.jl")
include("simulation/simulation.jl")

# Results and caching (must come after pipeline.jl for CONFIG;
# also after simulation/simulation.jl for SimulationResult; : also
# after embeddings/embedding_types.jl for EmbeddingsResult)
include("core/results.jl")
include("core/intermediate_cache.jl")

# Visualization
include("visualization/plotting.jl")

# Similarity & Embeddings
# Type contracts (EmbeddingsConfig, EmbeddingsResult, ConditionSimilarityResult) +
# extension stub surface (fit_sample_umap / fit_sample_tsne / fit_protein_umap /
# fit_condition_clustering). Methods are attached by BayesInteractomicsEmbeddingsExt
# (registered in Project.toml [extensions]) when the user loads `using UMAP, Clustering`.
# Placed BEFORE `module Differential` so (k-group differential) can import
# ConditionSimilarityResult from the parent namespace at submodule compile time.
# embedding_types.jl include was relocated above (before core/results.jl);
# remaining embedding compute files load here.
include("embeddings/embedding_stubs.jl")
include("embeddings/sample_embedding.jl")
include("embeddings/protein_embedding.jl")
include("embeddings/condition_similarity.jl")

# Differential interaction analysis (wrapped in submodule)
#
# Forward declaration: `generate_differential_report` is defined inside
# `module Reports` (further down in this file), but it is CALLED from inside
# `module Differential` (in differential/analysis.jl when
# `config.generate_report_html == true`). To make both submodules attach
# methods to / consume the SAME binding, we declare an empty generic here
# at parent scope. The `Reports` submodule has been updated to
# `import ..BayesInteractomics: generate_differential_report` so its method
# extends this parent binding rather than creating a local `Reports.generate_differential_report`.
function generate_differential_report end

module Differential
    # Bring parent symbols referenced by the three differential files into scope.
    # NOTE: per Plans 07/08 deviation pattern — `using ..BayesInteractomics`
    # would only pull EXPORTED names, AND because the parent module is
    # mid-definition when this submodule is compiled, its export list isn't
    # fully observable yet. We rely on explicit
    # `import ..BayesInteractomics: <name>` for every parent symbol referenced
    # inside the submodule (types from core, accessors, the `pep`/`bfdr`
    # helpers, the `run_analysis` pipeline entry point, and the
    # `generate_differential_report` stub declared above).
    import ..BayesInteractomics: AbstractAnalysisResult, AnalysisResult,
                                 BayesResult, CONFIG,
                                 InteractionData,
                                 bfdr, pep,
                                 getProteins,
                                 isCalibrated,  # submodule extends parent's isCalibrated with DifferentialResult method
                                 run_analysis,
                                 generate_differential_report,
                                 ConditionSimilarityResult,  # DifferentialResult.condition_similarity field type
                                 EmbeddingsConfig,           # source embeddings cfg from upstream AR.config
                                 _compute_condition_similarity  # k×k similarity + dendrogram

    # Re-import external deps used by the three differential files at module
    # scope. Julia submodules do NOT inherit `using X` / `import X` from the
    # parent — every external package referenced inside the submodule must be
    # brought in explicitly here.
    import DataFrames: DataFrame, innerjoin, Not, nrow, outerjoin, rename!, select!,
                       names as df_names
    using Dates
    using StatsPlots
    import Statistics: mean, std, quantile
    import XLSX: writetable

    include("differential/types.jl")
    include("differential/analysis.jl")
    include("differential/laplace_omnibus.jl")
    include("differential/decision_risk.jl")   # helpers
    include("differential/visualization.jl")
end  # module Differential

# Re-export the 24 public differential symbols at the top level so existing
# user code (`using BayesInteractomics; differential_analysis(...)`) and the
# existing `export` block below resolve unchanged.
using .Differential: DifferentialConfig, DifferentialResult, InteractionClass,
                     GAINED, REDUCED, UNCHANGED, BOTH_NEGATIVE,
                     CONDITION_A_SPECIFIC, CONDITION_B_SPECIFIC,
                     differential_analysis,
                     differential_volcano_plot, differential_evidence_plot,
                     differential_scatter_plot, differential_classification_plot,
                     differential_ma_plot,
                     gained_interactions, lost_interactions, unchanged_interactions,
                     significant_differential, export_differential,
                     getDifferentialBayesFactors, getDifferentialPosteriors,
                     getDifferentialQValues, getDifferentialBFDR,
                     getDifferentialPEP,
                     getClassifications, getDeltaLog2FC,
                     condition_labels, getAnalyses,
                     # public symbols re-exported
                     DEFAULT_DIFFERENTIAL_LOSS,
                     DECISION_RISK_ACTIONS,
                     compute_decision_risk,
                     compute_decision_risk!

# Forward Differential-internal helpers to the parent namespace so existing
# test code (`using BayesInteractomics: _extract_copula_df`) keeps resolving
# without modification. These are NOT exported — they remain implementation
# details — but they ARE reachable as `BayesInteractomics.X`.
using .Differential: _extract_copula_df, _deduplicate_proteins,
                     _rename_columns, _safe_log10, _safe_ratio,
                     _to_float, _zscore

# Machine learning
include("ml/pca.jl")
include("ml/metalearner_stubs.jl")  # stub-and-override declaration for predict_metalearner
# dnn/model.jl + ml/metalearner.jl now live in
# ext/BayesInteractomicsMetalearnerExt/. The metalearner_stubs.jl above declares
# `function predict_metalearner end`; the extension attaches the method when the
# user loads `using Flux, MLJ, MLJScikitLearnInterface, HDF5`. transition
# guard (@static if Base.find_package("Flux") ...) removed.

# Network analysis (stubs - extended by BayesInteractomicsNetworkExt)
include("network/stubs.jl")

# include("reports/json_utils.jl") and include("simulation/simulation.jl")
# were relocated to BEFORE include("core/results.jl") above so that
# AnalysisResult.simulation_result::Union{Nothing, SimulationResult} can use the
# concrete type annotation. Originally these lived here (after metalearner_stubs.jl
# and network/stubs.jl) per 's " deviation [Rule 3 - blocking
# issue]" note: simulation.jl references json_number/json_string/json_array/json_object
# from json_utils.jl, so json_utils.jl must precede simulation.jl. That ordering
# constraint is preserved at the new location (json_utils → simulation → core/results).
# The ml/network stub includes that previously sat between json_utils.jl/simulation.jl
# and the Reports submodule remain in place above; nothing depends on the old position.

# Interactive HTML report generation (wrapped in submodule)
module Reports
    # Bring parent module's public symbols + internal types into scope.
    # NOTE: `using ..BayesInteractomics` would only pull exported names, AND
    # because the parent module is mid-definition when this submodule is
    # being compiled, its export list isn't fully observable yet. We rely
    # on explicit `import ..BayesInteractomics: <name>` for every parent
    # symbol referenced at the submodule's top level (function signatures,
    # type annotations, docstrings).
    import ..BayesInteractomics: CONFIG, OutputFiles,
                                 BayesResult, BayesFactorHBM, BayesFactorRegression,
                                 AbstractAnalysisResult, AnalysisResult,
                                 LatentClassResult, BMAResult,
                                 DifferentialResult, DifferentialConfig, InteractionClass,
                                 GAINED, REDUCED, UNCHANGED, BOTH_NEGATIVE,
                                 CONDITION_A_SPECIFIC, CONDITION_B_SPECIFIC,
                                 DockingResult, DockingPairResult,
                                 SensitivityResult,
                                 DiagnosticsResult, CalibrationResult, ProteinPPC,
                                 BetaBernoulliPPC, ResidualResult,
                                 InputQCResult,
                                 DropoutFit,
                                 # Accessors / helpers referenced inside the report files
                                 getProteins, getBayesFactors, getPosteriorProbabilities,
                                 getQValues, getBFDR, getMeanLog2FC, getBaitProtein,
                                 bfdr, pep, log2FC,
                                 RegressionStatistics,
                                 # JSON helpers (kept at parent level due to circular
                                 # dependency with simulation.jl — see note above)
                                 json_number, json_string, json_array, json_object,
                                 json_bool, encode_png_file,
                                 # Decision Risk per-row JSON helpers
                                 json_number_nan_safe, json_symbol_or_string,
                                 # Internal type used in report serialization
                                 DiscreteEmpirical,
                                 # Constants from src/core/types.jl referenced by report files
                                 JEFFREYS_SHIFT,
                                 # Cross-module helper defined in src/simulation/simulation.jl
                                 # but called from src/reports/report_generator.jl
                                 _build_simulation_json,
                                 # differential helpers consumed by
                                 # the per-condition tab JSON builders inside the
                                 # Reports submodule. `condition_labels` provides the
                                 # k-aware iteration label vector; `_extract_copula_df`
                                 # is the metalearner-stripping helper used by
                                 # `_build_diff_mixture_json` per Pitfall 3.
                                 condition_labels, _extract_copula_df,
                                 # parent-level stub for
                                 # `generate_differential_report` declared BEFORE
                                 # `module Differential`. Reports extends this
                                 # binding so that `module Differential`'s call
                                 # to `generate_differential_report(diff)`
                                 # resolves to the same method.
                                 generate_differential_report,
                                 # `_methods_embeddings_block` and the
                                 # five `_build_*_json` helpers reference
                                 # `EmbeddingsConfig` (resolution at parse time of
                                 # `config_or_emb isa EmbeddingsConfig`).
                                 EmbeddingsConfig

    # Re-import external deps the included files use at module scope.
    # (Julia submodules do NOT inherit `using X` / `import X` statements from
    # the parent — every external package referenced inside the submodule must
    # be brought in explicitly here.)
    using DataFrames: DataFrame, AbstractDataFrame, nrow, eachrow
    using Base64
    using CodecZlib: GzipCompressor, transcode
    import JSON3
    import Downloads
    import Dates
    import Dates: now, format, DateTime
    import Statistics: mean, median, quantile, std, var, cor
    import StatsBase: corspearman
    import Distributions
    import Distributions: Normal, Gamma, LogNormal, Weibull, pdf
    import Random
    import Random: MersenneTwister, randperm

    include("reports/methods_generator.jl")
    include("reports/report_generator.jl")
end  # module Reports

# Re-export the public Reports symbols at the top level so existing user code
# (`using BayesInteractomics; generate_report(...)`) works unchanged.
# NOTE: `generate_differential_report` is NOT re-listed here — Reports
# `import`s the parent stub (declared before `module Differential`) and
# extends it; the binding already lives at parent scope.
using .Reports: generate_report

# Forward Reports-internal helpers to the parent namespace so existing
# test code (`using BayesInteractomics: _build_protein_json`, etc.) keeps
# resolving without modification. These are NOT exported — they remain
# implementation details — but they ARE reachable as `BayesInteractomics.X`.
using .Reports: generate_methods_text, generate_methods_parameters,
                generate_reproducibility_block,
                _report_pkg_version, _build_structured_methods_data,
                _build_report_json, _build_meta_json, _build_results_json,
                _build_protein_json, _build_plots_json, _build_methods_json,
                _build_bma_summary_json, _build_mixture_model_json,
                _build_validation_json, _build_sensitivity_json,
                _build_qc_json, _build_diff_json, _build_diff_protein_json,
                # six new differential per-tab JSON helpers
                _build_diff_calibration_json, _build_diff_sensitivity_json,
                _build_diff_qc_json, _build_diff_mixture_json,
                _build_diff_methods_json, _build_diff_dbf_diagnostics_json,
                # Methods differential block + esc helper (tests)
                _methods_differential_block, _html_esc,
                # Mask-aware v2b Methods block +
                # the per-condition pct_imputed_cells chip helper.
                _methods_mask_aware_regression_block, _mask_aware_chip_html,
                # DNN Prior + MC-Dropout
                # Methods block (interactive report transparency).
                _methods_dnn_prior_block,
                # Embeddings + Condition Similarity JSON helpers
                # + Methods block (for warm-load + integration tests).
                _build_sample_embedding_json, _build_protein_embedding_json,
                _build_condition_matrix_json, _build_jaccard_json,
                _build_dendrogram_json, _methods_embeddings_block,
                _build_docking_json, _build_diagnostics_data_json,
                _build_kl_divergence_json, _build_copula_bootstrap_json,
                _build_discordant_json, _build_evidence_data_json,
                _build_non_detected_json,
                _add_marginal_fit_json!,
                # row-wise reconstruction helper used by
                # `_add_marginal_fit_json!`; exposed at parent scope so unit tests
                # can target it directly (`BayesInteractomics._reconstruct_lc_responsibilities`).
                _reconstruct_lc_responsibilities,
                # BB-exclusion-count helper used by
                # the defensive row-count guard inside `_add_marginal_fit_json!`;
                # forwarded to the parent namespace so the regression
                # tests can call it as
                # `BayesInteractomics.Reports._count_betabernoulli_excluded(...)`.
                _count_betabernoulli_excluded,
                _filter_detected, _report_float,
                _winsorize_quantile, _kde_density, _mc_combined_density,
                _sidecar_path, _write_sidecar, _merge_sidecar,
                _inline_molstar_bundle, _ensure_molstar_vendored,
                _molstar_vendor_dir,
                MOLSTAR_VERSION, MOLSTAR_JS_URL, MOLSTAR_CSS_URL

##############################################
# Public API
##############################################
# Data loading
public getNoExperiments, getExperiment, getIDs
public getIDs, getNames, getNoProtocols, getControls, getSamples
public getSampleMatrix, getControlMatrix
public getMatchedPositions, getExperimentPositions, getProtocolPositions

# Results and caching
public getProteins, getBayesFactors, getPosteriorProbabilities, getQValues, getBFDR
# getPEP / isCalibrated are exported (see export block below);
# Julia 1.12 disallows the same symbol appearing in both `public` and `export`.
public getMeanLog2FC, getBaitProtein, getPosteriorProbs
public bfdr, pep
public compute_config_hash, compute_data_hash, check_cache
public compute_betabernoulli_hash, compute_hbm_regression_hash
public check_betabernoulli_cache, check_hbm_regression_cache
public get_betabernoulli_cache_filepath, get_hbm_regression_cache_filepath

# Model fitting
public getPosterior, getbfHBM, getbfRegression, getHBMstats, getregressionstats, clean_result

# Evidence combination
public combined_BF_latent_class, combined_BF_bma

# Model Evaluation
public probability_of_direction, pd_to_p_value, log2FCStatistics, RegressionStatistics

# Model visualization
public plot_inference_results, plot_log2fc, plot_regression
public plot_bayesrange, write_txt

# Differential analysis accessors
public getDifferentialBayesFactors, getDifferentialPosteriors
public getDifferentialQValues, getDifferentialBFDR, getClassifications, getDeltaLog2FC
# getDifferentialPEP is exported (see export block below);
# Julia 1.12 disallows the same symbol appearing in both `public` and `export`.
public condition_labels, getAnalyses

# Macro DSL (@interactomics, @protocol)
include("macro.jl")

# Exports
export @interactomics, @protocol
export load_data, run_analysis, BayesResult, getProteinData, CONFIG, OutputFiles
export check_bait_detected
# Normalisation pipeline
export apply_normalisation, build_run_matrix, matrix_to_interactiondata, norm_median_of_ratios_id
# Multi-protocol scale-mismatch auto-detector
export detect_protocol_scale_mismatch
# Correct-order normalise-before-impute entry point
export normalise_then_impute
# Regression-safe per-condition bait-anchor correction
export bait_anchor_id
export Protocol, InteractionData, Protein
export BayesFactorHBM, BayesFactorRegression
export HBMResult, HBMResultSingleProtocol, HBMResultMultipleProtocols
export enrichment, precompute_enrichment_prior
export log2FC
export evaluate_imputed_fc_posteriors

# Dropout fit (per-column logistic dropout curves for MNAR-aware imputation)
export DropoutFit, fit_dropout_curves, save_dropout_fit, load_dropout_fit
export column_imputation_sigma

# MNAR-aware single imputation
export impute_mnar, impute_mnar_from_paths
export AbstractAnalysisResult, AnalysisResult, NetworkAnalysisResult
export save_result, load_result, CacheStatus
export BetaBernoulliCache, HBMRegressionCache, H0Cache, IntermediateCacheStatus
export save_betabernoulli_cache, load_betabernoulli_cache
export save_hbm_regression_cache, load_hbm_regression_cache
export save_h0_cache, load_h0_cache
export set_bait_info!
export LatentClassResult, BMAResult
export DiscreteEmpirical, _fit_discrete_empirical_weighted
export _replace_detection_marginal, _jitter_discrete_to_uniform
export estimate_dirichlet_eb, inv_digamma, build_prior_grid, _marginalize_over_priors

# Data curation
export curate_proteins, CurationReport, CurationActionType
export CurationEntry, MergeCandidate, MergeDecision, CurationCache
export CurationAPIError
export split_protein_groups, resolve_to_string_ids
export merge_protein_rows, confirm_merges_interactive
export save_curation_report, load_curation_report
export remove_contaminants, parse_protein_id

# Input data quality control
export InputQCResult, ScaleCheckResult, ReplicateCorrelationResult
export MissingnessResult, IntensityShapeResult, PCASeparationResult
export run_input_qc, worst_flag

# Robust regression & model comparison
export RobustRegressionResult, AnyRegressionResult
export RobustRegressionResultMultipleProtocols, RobustRegressionResultSingleProtocol
export WAICResult, ModelComparisonResult, NuOptimizationResult
export compute_waic, compare_regression_models, optimize_nu

# Sensitivity analysis
export SensitivityConfig, SensitivityResult, PriorSetting
export sensitivity_analysis, generate_sensitivity_report
export sensitivity_rank_correlation

# Model diagnostics & posterior predictive checks
export DiagnosticsConfig, DiagnosticsResult
export BBMnarCodrivenConfig
export ProteinPPC, BetaBernoulliPPC, ResidualResult, CalibrationResult
export EnhancedResidualResult, PPCExtendedStatistics, ProteinDiagnosticFlag
export model_diagnostics, generate_diagnostics_report, _merge_diagnostics_to_results
export ppc_density_plot, ppc_pvalue_histogram, pit_histogram_plot
export residual_qq_plot, residual_distribution_plot
export calibration_plot, bb_ppc_summary_plot
export nu_optimization_plot
export bma_weights_plot, compute_em_responsibilities
export QualityGateCell, QualityGateResult, KLContaminationResult, ValidationResult
export run_quality_gates, compute_kl_contamination
export kl_h1_divergence, kl_divergence_plot
export within_class_correlation, within_class_correlation_plot
export agnostic_zone_analysis, agnostic_zone_plot
export copula_bootstrap_ci, copula_bootstrap_plot
export discordant_protein_analysis, discordant_protein_plot
export component_assignment_plot, em_convergence_plot

# PEP / calibration accessor exports
export getPEP, getDifferentialPEP, isCalibrated

# Differential interaction analysis exports
export DifferentialConfig, DifferentialResult, InteractionClass
export GAINED, REDUCED, UNCHANGED, BOTH_NEGATIVE, CONDITION_A_SPECIFIC, CONDITION_B_SPECIFIC
export differential_analysis
export differential_volcano_plot, differential_evidence_plot, differential_scatter_plot
export differential_classification_plot, differential_ma_plot
export gained_interactions, lost_interactions, unchanged_interactions
export significant_differential, export_differential
# Decision Risk exports
export DEFAULT_DIFFERENTIAL_LOSS, DECISION_RISK_ACTIONS
export compute_decision_risk, compute_decision_risk!

# Interactive report exports
export generate_report, generate_differential_report

# Simulation exports
export run_simulation, SimulationResult

# Docking integration exports
export DockingConfig, DockingCalibration, DockingPairResult, DockingResult, DockingRequestBatch
export compute_bf_dock, apply_docking_update, default_calibration, compute_pdockq
export compute_bf_from_pdockq, compute_bf_from_iptm, compute_bf_from_c2qscore, docking_cache_key
export compute_c2qscore
export generate_docking_requests, import_docking_results

# Network analysis exports
export AbstractNetworkResult
export build_network, network_statistics, centrality_measures, detect_communities
export plot_network, save_network_plot
export export_graphml, export_edgelist, export_node_attributes
export centrality_dataframe, community_dataframe, get_top_hubs, edge_source_summary
export NetworkConfig, NetworkPipelineResult
export run_network_analysis, generate_network_report

# Prey-prey network enrichment exports
export PPIEnrichmentConfig
export enrich_network, query_string_ppi, clear_ppi_cache, ppi_cache_info

# Similarity & Embeddings type contracts
# Stub functions (fit_sample_umap / fit_sample_tsne / fit_protein_umap /
# fit_condition_clustering) are NOT exported — they are internal extension hooks
# matching the imputation_stubs / metalearner_stubs convention.
export EmbeddingsConfig, EmbeddingsResult, ConditionSimilarityResult

end
