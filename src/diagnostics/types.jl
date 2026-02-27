# Prior Sensitivity Analysis Types
# Provides types for systematic prior sensitivity sweeps

using Dates
using DataFrames

"""
    SensitivityConfig

Configuration for prior sensitivity analysis sweeps.

# Fields
- `bb_priors`: Grid of (α, β) pairs for Beta-Bernoulli prior sweep
- `em_prior_grid`: Grid of (α, β) NamedTuples for copula-EM prior sweep
- `lc_alpha_prior_grid`: Grid of Dirichlet alpha vectors for latent class prior sweep
- `n_top_proteins`: Number of top proteins to highlight in reports
"""
Base.@kwdef struct SensitivityConfig
    bb_priors::Vector{Tuple{Float64,Float64}} = [
        (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (5.0, 5.0), (10.0, 10.0)
    ]
    em_prior_grid::Vector{NamedTuple{(:α,:β), Tuple{Float64,Float64}}} = [
        (α=10.0, β=190.0), (α=25.0, β=175.0), (α=50.0, β=100.0)
    ]
    lc_alpha_prior_grid::Vector{Vector{Float64}} = [
        [20.0, 1.0], [10.0, 1.0], [5.0, 1.0]
    ]
    n_top_proteins::Int = 20
end

"""
    PriorSetting

Describes a single prior specification used in the sensitivity sweep.

# Fields
- `model`: Which model the prior applies to (`:betabernoulli`, `:copula_em`, `:latent_class`)
- `label`: Human-readable label (e.g. "BB(1.0,1.0)")
- `params`: NamedTuple of parameter values
"""
struct PriorSetting
    model::Symbol
    label::String
    params::NamedTuple
end

"""
    SensitivityResult

Complete results from a prior sensitivity analysis sweep.

# Fields
- `config`: The SensitivityConfig used for the sweep
- `prior_settings`: Vector of PriorSetting describing each column of the matrices
- `posterior_matrix`: Matrix of posterior probabilities (n_proteins × n_settings)
- `bf_matrix`: Matrix of combined Bayes factors (n_proteins × n_settings)
- `q_matrix`: Matrix of q-values (n_proteins × n_settings)
- `protein_names`: Vector of protein identifiers
- `baseline_index`: Column index of the baseline (default) prior setting
- `summary`: Per-protein summary statistics across all settings
- `classification_stability`: Per-protein classification stability across thresholds
- `timestamp`: When the analysis was run
"""
struct SensitivityResult
    config::SensitivityConfig
    prior_settings::Vector{PriorSetting}
    posterior_matrix::Matrix{Float64}
    bf_matrix::Matrix{Float64}
    q_matrix::Matrix{Float64}
    protein_names::Vector{String}
    baseline_index::Int
    summary::DataFrame
    classification_stability::DataFrame
    timestamp::DateTime
end

# ============================================================================ #
# Posterior Predictive Check & Model Diagnostics Types
# ============================================================================ #

"""
    DiagnosticsConfig

Configuration for posterior predictive checks and model diagnostics.

# Fields
- `n_ppc_draws::Int`: Number of posterior predictive draws per protein (default: 1000)
- `n_proteins_to_check::Int`: Number of proteins to run PPC on (default: 50)
- `ppc_protein_selection::Symbol`: Strategy for selecting proteins — `:top_and_random` selects
  top N/2 by combined BF plus N/2 random (default: `:top_and_random`)
- `residual_model::Symbol`: Which model residuals to compute — `:both`, `:hbm`, or `:regression` (default: `:both`)
- `calibration_bins::Int`: Number of bins for calibration assessment (default: 10)
- `n_top_display::Int`: Number of top proteins to show in reports (default: 20)
- `seed::Int`: Random seed for reproducibility (default: 42)
"""
Base.@kwdef mutable struct DiagnosticsConfig
    n_ppc_draws::Int = 1000
    n_proteins_to_check::Int = 50
    ppc_protein_selection::Symbol = :top_and_random
    residual_model::Symbol = :both
    calibration_bins::Int = 10
    n_top_display::Int = 20
    seed::Int = 42
    enhanced_residuals::Bool = true
end

"""
    ProteinPPC

Posterior predictive check results for a single protein (HBM or regression model).

# Fields
- `protein_name::String`: Protein identifier
- `model::Symbol`: Which model produced this PPC (`:hbm` or `:regression`)
- `observed::Vector{Float64}`: Observed data values
- `simulated::Matrix{Float64}`: Simulated replicate data (n_obs × n_draws)
- `pvalue_mean::Float64`: Bayesian p-value for mean test statistic
- `pvalue_sd::Float64`: Bayesian p-value for standard deviation test statistic
- `pvalue_log2fc::Float64`: Bayesian p-value for log2FC test statistic (HBM only; NaN for regression)
"""
struct ProteinPPC
    protein_name::String
    model::Symbol
    observed::Vector{Float64}
    simulated::Matrix{Float64}
    pvalue_mean::Float64
    pvalue_sd::Float64
    pvalue_log2fc::Float64
end

"""
    BetaBernoulliPPC

Posterior predictive check results for a single protein's Beta-Bernoulli detection model.

# Fields
- `protein_name::String`: Protein identifier
- `observed_k_sample::Int`: Observed detection count in samples
- `observed_k_control::Int`: Observed detection count in controls
- `simulated_k_sample::Vector{Int}`: Simulated sample detection counts (length = n_draws)
- `simulated_k_control::Vector{Int}`: Simulated control detection counts (length = n_draws)
- `pvalue_detection_diff::Float64`: Bayesian p-value for detection difference (sample - control)
"""
struct BetaBernoulliPPC
    protein_name::String
    observed_k_sample::Int
    observed_k_control::Int
    simulated_k_sample::Vector{Int}
    simulated_k_control::Vector{Int}
    pvalue_detection_diff::Float64
end

"""
    ResidualResult

Standardized residuals from HBM or regression model across all proteins.

# Fields
- `model::Symbol`: Which model produced these residuals (`:hbm` or `:regression`)
- `protein_names::Vector{String}`: Protein identifiers
- `residuals::Vector{Vector{Float64}}`: Per-protein residual vectors
- `mean_residuals::Vector{Float64}`: Mean residual per protein
- `pooled_residuals::Vector{Float64}`: All residuals concatenated
- `pooled_fitted::Vector{Float64}`: Fitted values parallel to `pooled_residuals` (for scale-location plots)
- `skewness::Float64`: Skewness of pooled residuals
- `kurtosis::Float64`: Excess kurtosis of pooled residuals
- `outlier_proteins::Vector{String}`: Proteins with |mean residual| > 2
"""
struct ResidualResult
    model::Symbol
    protein_names::Vector{String}
    residuals::Vector{Vector{Float64}}
    mean_residuals::Vector{Float64}
    pooled_residuals::Vector{Float64}
    pooled_fitted::Vector{Float64}
    skewness::Float64
    kurtosis::Float64
    outlier_proteins::Vector{String}
end

"""
    CalibrationResult

Calibration assessment comparing predicted posterior probabilities to observed agreement rates.

Note: Without a gold standard, "observed truth" is defined as internal consistency — a protein is
empirically positive if all three individual Bayes factors (enrichment, correlation, detection)
exceed 1.0. This is a self-consistency check, not true calibration.

# Fields
- `bin_midpoints::Vector{Float64}`: Midpoint of each probability bin
- `predicted_rate::Vector{Float64}`: Mean predicted posterior in each bin
- `observed_rate::Vector{Float64}`: Fraction of empirically positive proteins in each bin
- `bin_counts::Vector{Int}`: Number of proteins in each bin
- `ece::Float64`: Expected Calibration Error (weighted mean |predicted - observed|)
- `mce::Float64`: Maximum Calibration Error (max |predicted - observed|)
"""
struct CalibrationResult
    bin_midpoints::Vector{Float64}
    predicted_rate::Vector{Float64}
    observed_rate::Vector{Float64}
    bin_counts::Vector{Int}
    ece::Float64
    mce::Float64
end

"""
    EnhancedResidualResult

Randomized quantile residuals and PIT values from a Bayesian model.

Standard residuals use point estimates for posterior parameters, which can mask
model misspecification. Randomized quantile residuals integrate over the full
posterior predictive distribution, producing PIT (Probability Integral Transform)
values that should be Uniform(0,1) for a well-specified model.

# Fields
- `base::ResidualResult`: Standard residuals (for backwards compatibility)
- `quantile_residuals::Vector{Vector{Float64}}`: Per-protein quantile residuals
- `pit_values::Vector{Float64}`: Pooled PIT values (for histogram)
"""
struct EnhancedResidualResult
    base::ResidualResult
    quantile_residuals::Vector{Vector{Float64}}
    pit_values::Vector{Float64}
end

"""
    PPCExtendedStatistics

Extended posterior predictive check statistics for a single protein.

Beyond standard mean and SD checks, these statistics test for distributional
shape misspecification (skewness, kurtosis) and spread calibration (IQR/SD ratio).

# Fields
- `protein_name::String`: Protein identifier
- `model::Symbol`: Which model produced this PPC (`:hbm` or `:regression`)
- `pvalue_skewness::Float64`: Bayesian p-value for skewness test statistic
- `pvalue_kurtosis::Float64`: Bayesian p-value for kurtosis test statistic
- `pvalue_iqr_ratio::Float64`: Bayesian p-value for IQR/SD ratio test statistic
"""
struct PPCExtendedStatistics
    protein_name::String
    model::Symbol
    pvalue_skewness::Float64
    pvalue_kurtosis::Float64
    pvalue_iqr_ratio::Float64
end

"""
    ProteinDiagnosticFlag

Per-protein diagnostic summary with flags for potential issues.

Flags identify proteins that may need closer inspection due to residual outlier
behaviour or insufficient data for reliable inference.

# Fields
- `protein_name::String`: Protein identifier
- `n_observations::Int`: Number of observed data points
- `mean_residual::Float64`: Mean standardized residual
- `max_abs_residual::Float64`: Maximum |residual| across observations
- `is_residual_outlier::Bool`: `true` if |mean residual| > 2
- `is_low_data::Bool`: `true` if n_observations < 4
- `overall_flag::Symbol`: `:ok`, `:warning`, or `:fail`
"""
struct ProteinDiagnosticFlag
    protein_name::String
    n_observations::Int
    mean_residual::Float64
    max_abs_residual::Float64
    is_residual_outlier::Bool
    is_low_data::Bool
    overall_flag::Symbol
end

"""
    NuOptimizationResult

Result of optimizing the Student-t degrees-of-freedom parameter ν via Brent's method,
minimizing WAIC over [lower, upper].

# Fields
- `optimal_nu::Float64`: Best ν from Brent optimization
- `optimal_waic::WAICResult`: WAIC at optimal ν
- `normal_waic::WAICResult`: WAIC for Normal model (baseline comparison)
- `nu_trace::Vector{Float64}`: ν values evaluated during optimization
- `waic_trace::Vector{Float64}`: Corresponding WAIC values
- `delta_waic::Float64`: normal_waic.waic - optimal_waic.waic (positive = robust better)
- `delta_se::Float64`: SE of the difference
- `search_bounds::Tuple{Float64, Float64}`: (lower, upper) bounds used
"""
struct NuOptimizationResult
    optimal_nu::Float64
    optimal_waic::WAICResult
    normal_waic::WAICResult
    nu_trace::Vector{Float64}
    waic_trace::Vector{Float64}
    delta_waic::Float64
    delta_se::Float64
    search_bounds::Tuple{Float64, Float64}
end

"""
    DiagnosticsResult

Complete results from posterior predictive checks and model diagnostics.

# Fields
- `config::DiagnosticsConfig`: Configuration used for the diagnostics run
- `protein_ppcs::Vector{ProteinPPC}`: PPC results for selected proteins (HBM + regression)
- `bb_ppcs::Vector{BetaBernoulliPPC}`: PPC results for Beta-Bernoulli detection model
- `hbm_residuals::Union{ResidualResult, Nothing}`: Standardized residuals from HBM model
- `regression_residuals::Union{ResidualResult, Nothing}`: Standardized residuals from regression model
- `calibration::Union{CalibrationResult, Nothing}`: Strict calibration (all 3 BFs > 1.0)
- `calibration_relaxed::Union{CalibrationResult, Nothing}`: Relaxed calibration (2-of-3 BFs > 1.0)
- `calibration_enrichment_only::Union{CalibrationResult, Nothing}`: Enrichment-only calibration (BF_enrichment > 3.0)
- `enhanced_hbm_residuals::Union{EnhancedResidualResult, Nothing}`: Randomized quantile residuals from HBM model
- `enhanced_regression_residuals::Union{EnhancedResidualResult, Nothing}`: Randomized quantile residuals from regression model
- `ppc_extended::Union{Vector{PPCExtendedStatistics}, Nothing}`: Extended PPC statistics (skewness, kurtosis, IQR/SD)
- `protein_flags::Union{Vector{ProteinDiagnosticFlag}, Nothing}`: Per-protein diagnostic flags
- `model_comparison::Union{ModelComparisonResult, Nothing}`: WAIC comparison between Normal and robust regression models
- `nu_optimization::Union{NuOptimizationResult, Nothing}`: ν optimization results (Brent's method on WAIC)
- `summary::DataFrame`: Per-protein summary of diagnostics (protein name, p-values, flags)
- `timestamp::DateTime`: When the diagnostics were run
"""
struct DiagnosticsResult
    config::DiagnosticsConfig
    protein_ppcs::Vector{ProteinPPC}
    bb_ppcs::Vector{BetaBernoulliPPC}
    hbm_residuals::Union{ResidualResult, Nothing}
    regression_residuals::Union{ResidualResult, Nothing}
    calibration::Union{CalibrationResult, Nothing}
    calibration_relaxed::Union{CalibrationResult, Nothing}
    calibration_enrichment_only::Union{CalibrationResult, Nothing}
    enhanced_hbm_residuals::Union{EnhancedResidualResult, Nothing}
    enhanced_regression_residuals::Union{EnhancedResidualResult, Nothing}
    ppc_extended::Union{Vector{PPCExtendedStatistics}, Nothing}
    protein_flags::Union{Vector{ProteinDiagnosticFlag}, Nothing}
    model_comparison::Union{ModelComparisonResult, Nothing}
    nu_optimization::Union{NuOptimizationResult, Nothing}
    summary::DataFrame
    timestamp::DateTime
end

# ============================================================================ #
# Pretty-print show methods
# ============================================================================ #

function Base.show(io::IO, c::SensitivityConfig)
    println(io, "SensitivityConfig")
    println(io, "  BB priors          : $(length(c.bb_priors)) settings")
    println(io, "  EM prior grid      : $(length(c.em_prior_grid)) settings")
    println(io, "  LC alpha grid      : $(length(c.lc_alpha_prior_grid)) settings")
    print(io,   "  Top proteins shown : $(c.n_top_proteins)")
end

function Base.show(io::IO, p::PriorSetting)
    print(io, "PriorSetting(:$(p.model), \"$(p.label)\")")
end

function Base.show(io::IO, r::SensitivityResult)
    n_prot, n_set = size(r.posterior_matrix)
    println(io, "SensitivityResult")
    println(io, "───────────────────────────────────")
    println(io, "  Proteins           : $n_prot")
    println(io, "  Prior settings     : $n_set")
    println(io, "  Baseline index     : $(r.baseline_index)")
    if !isempty(r.summary)
        mean_range = round(mean(r.summary.range), digits=4)
        max_range  = round(maximum(r.summary.range), digits=4)
        println(io, "  Mean post. range   : $mean_range")
        println(io, "  Max post. range    : $max_range")
    end
    print(io,   "  Timestamp          : $(r.timestamp)")
end

function Base.show(io::IO, c::DiagnosticsConfig)
    println(io, "DiagnosticsConfig")
    println(io, "  PPC draws          : $(c.n_ppc_draws)")
    println(io, "  Proteins to check  : $(c.n_proteins_to_check)")
    println(io, "  Selection strategy : :$(c.ppc_protein_selection)")
    println(io, "  Residual model     : :$(c.residual_model)")
    println(io, "  Calibration bins   : $(c.calibration_bins)")
    println(io, "  Enhanced residuals : $(c.enhanced_residuals)")
    print(io,   "  Seed               : $(c.seed)")
end

function Base.show(io::IO, p::ProteinPPC)
    n_obs, n_draws = size(p.simulated)
    println(io, "ProteinPPC(\"$(p.protein_name)\", :$(p.model))")
    println(io, "  Observations       : $n_obs")
    println(io, "  PPC draws          : $n_draws")
    println(io, "  p-value (mean)     : $(round(p.pvalue_mean, digits=4))")
    println(io, "  p-value (SD)       : $(round(p.pvalue_sd, digits=4))")
    if !isnan(p.pvalue_log2fc)
        print(io, "  p-value (log2FC)   : $(round(p.pvalue_log2fc, digits=4))")
    else
        print(io, "  p-value (log2FC)   : N/A")
    end
end

function Base.show(io::IO, b::BetaBernoulliPPC)
    println(io, "BetaBernoulliPPC(\"$(b.protein_name)\")")
    println(io, "  Observed sample k  : $(b.observed_k_sample)")
    println(io, "  Observed control k : $(b.observed_k_control)")
    println(io, "  PPC draws          : $(length(b.simulated_k_sample))")
    print(io,   "  p-value (det diff) : $(round(b.pvalue_detection_diff, digits=4))")
end

function Base.show(io::IO, r::ResidualResult)
    n_prot = length(r.protein_names)
    n_pool = length(r.pooled_residuals)
    println(io, "ResidualResult(:$(r.model))")
    println(io, "  Proteins           : $n_prot")
    println(io, "  Pooled residuals   : $n_pool")
    println(io, "  Skewness           : $(round(r.skewness, digits=3))")
    println(io, "  Kurtosis (excess)  : $(round(r.kurtosis, digits=3))")
    print(io,   "  Outlier proteins   : $(length(r.outlier_proteins))")
end

function Base.show(io::IO, c::CalibrationResult)
    n_bins = length(c.bin_midpoints)
    total  = sum(c.bin_counts)
    println(io, "CalibrationResult")
    println(io, "  Bins               : $n_bins")
    println(io, "  Total proteins     : $total")
    println(io, "  ECE                : $(round(c.ece, digits=4))")
    print(io,   "  MCE                : $(round(c.mce, digits=4))")
end

function Base.show(io::IO, e::EnhancedResidualResult)
    n_pit = length(e.pit_values)
    println(io, "EnhancedResidualResult(:$(e.base.model))")
    println(io, "  Base residuals     : $(length(e.base.protein_names)) proteins")
    println(io, "  PIT values         : $n_pit")
    println(io, "  Skewness           : $(round(e.base.skewness, digits=3))")
    print(io,   "  Kurtosis (excess)  : $(round(e.base.kurtosis, digits=3))")
end

function Base.show(io::IO, p::PPCExtendedStatistics)
    println(io, "PPCExtendedStatistics(\"$(p.protein_name)\", :$(p.model))")
    println(io, "  p-value (skewness) : $(round(p.pvalue_skewness, digits=4))")
    println(io, "  p-value (kurtosis) : $(round(p.pvalue_kurtosis, digits=4))")
    print(io,   "  p-value (IQR/SD)   : $(round(p.pvalue_iqr_ratio, digits=4))")
end

function Base.show(io::IO, f::ProteinDiagnosticFlag)
    flag_str = f.overall_flag == :ok ? "ok" :
               f.overall_flag == :warning ? "warning" : "fail"
    println(io, "ProteinDiagnosticFlag(\"$(f.protein_name)\", :$flag_str)")
    println(io, "  Observations       : $(f.n_observations)")
    println(io, "  Mean residual      : $(round(f.mean_residual, digits=3))")
    println(io, "  Max |residual|     : $(round(f.max_abs_residual, digits=3))")
    println(io, "  Residual outlier   : $(f.is_residual_outlier)")
    print(io,   "  Low data           : $(f.is_low_data)")
end

function Base.show(io::IO, n::NuOptimizationResult)
    println(io, "NuOptimizationResult")
    println(io, "───────────────────────────────────")
    println(io, "  Optimal ν          : $(round(n.optimal_nu, digits=2))")
    println(io, "  Robust WAIC        : $(round(n.optimal_waic.waic, digits=2))")
    println(io, "  Normal WAIC        : $(round(n.normal_waic.waic, digits=2))")
    println(io, "  ΔWAIC              : $(round(n.delta_waic, digits=2)) ± $(round(n.delta_se, digits=2))")
    println(io, "  Search bounds      : [$(n.search_bounds[1]), $(n.search_bounds[2])]")
    print(io,   "  Evaluations        : $(length(n.nu_trace))")
end

function Base.show(io::IO, d::DiagnosticsResult)
    println(io, "DiagnosticsResult")
    println(io, "═══════════════════════════════════")
    println(io, "  Timestamp          : $(d.timestamp)")
    println(io)

    # PPC summary
    n_ppc = length(d.protein_ppcs)
    n_bb  = length(d.bb_ppcs)
    println(io, "Posterior Predictive Checks:")
    println(io, "  HBM/Regression PPC : $n_ppc proteins")
    println(io, "  Beta-Bernoulli PPC : $n_bb proteins")
    if n_ppc > 0
        pvals = [p.pvalue_mean for p in d.protein_ppcs]
        println(io, "  Mean p-val (mean)  : $(round(mean(pvals), digits=4))")
        extreme = count(p -> p < 0.05 || p > 0.95, pvals)
        println(io, "  Extreme p-values   : $extreme / $n_ppc")
    end
    println(io)

    # Residuals
    println(io, "Residuals:")
    if !isnothing(d.hbm_residuals)
        r = d.hbm_residuals
        println(io, "  HBM                : $(length(r.protein_names)) proteins, $(length(r.outlier_proteins)) outliers")
    else
        println(io, "  HBM                : not computed")
    end
    if !isnothing(d.regression_residuals)
        r = d.regression_residuals
        println(io, "  Regression         : $(length(r.protein_names)) proteins, $(length(r.outlier_proteins)) outliers")
    else
        println(io, "  Regression         : not computed")
    end
    if !isnothing(d.enhanced_hbm_residuals)
        println(io, "  Enhanced HBM       : $(length(d.enhanced_hbm_residuals.pit_values)) PIT values")
    end
    if !isnothing(d.enhanced_regression_residuals)
        println(io, "  Enhanced Regr.     : $(length(d.enhanced_regression_residuals.pit_values)) PIT values")
    end
    println(io)

    # Calibration
    println(io, "Calibration:")
    if !isnothing(d.calibration)
        println(io, "  Strict ECE         : $(round(d.calibration.ece, digits=4))")
    end
    if !isnothing(d.calibration_relaxed)
        println(io, "  Relaxed ECE        : $(round(d.calibration_relaxed.ece, digits=4))")
    end
    if !isnothing(d.calibration_enrichment_only)
        println(io, "  Enrichment ECE     : $(round(d.calibration_enrichment_only.ece, digits=4))")
    end
    if isnothing(d.calibration) && isnothing(d.calibration_relaxed) && isnothing(d.calibration_enrichment_only)
        println(io, "  not computed")
    end
    println(io)

    # Model comparison
    println(io, "Model Comparison:")
    if !isnothing(d.model_comparison)
        mc = d.model_comparison
        println(io, "  Normal WAIC        : $(round(mc.normal_waic.waic, digits=2))")
        if !isnothing(mc.robust_waic)
            println(io, "  Robust WAIC        : $(round(mc.robust_waic.waic, digits=2))")
        end
        println(io, "  ΔWAIC              : $(round(mc.delta_waic, digits=2)) ± $(round(mc.delta_se, digits=2))")
        println(io, "  Preferred          : :$(mc.preferred_model)")
    else
        println(io, "  not computed")
    end
    println(io)

    # ν optimization
    println(io, "ν Optimization:")
    if !isnothing(d.nu_optimization)
        nu = d.nu_optimization
        println(io, "  Optimal ν          : $(round(nu.optimal_nu, digits=2))")
        println(io, "  ΔWAIC              : $(round(nu.delta_waic, digits=2)) ± $(round(nu.delta_se, digits=2))")
        println(io, "  Evaluations        : $(length(nu.nu_trace))")
    else
        println(io, "  not computed")
    end
    println(io)

    # Flags
    if !isnothing(d.protein_flags)
        n_ok   = count(f -> f.overall_flag == :ok, d.protein_flags)
        n_warn = count(f -> f.overall_flag == :warning, d.protein_flags)
        n_fail = count(f -> f.overall_flag == :fail, d.protein_flags)
        println(io, "Protein Flags:")
        print(io,   "  ok=$n_ok  warning=$n_warn  fail=$n_fail")
    else
        print(io, "Protein Flags: not computed")
    end
end
