
# ----------------------- Regression and HBM Results ----------------------- #

"""
    AbstractInferenceResult

Abstract supertype for all inference result types from Bayesian models.

Subtypes include:
- [`RegressionResult`](@ref): Results from Bayesian linear regression models
- [`HBMResult`](@ref): Results from Hierarchical Bayesian Models

See also: [`BayesResult`](@ref), [`RegressionResultMultipleProtocols`](@ref), [`HBMResultMultipleProtocols`](@ref)
"""
abstract type AbstractInferenceResult end

"""
    RegressionResult <: AbstractInferenceResult

Abstract type for Bayesian regression model results analyzing dose-response correlation.

Concrete subtypes:
- [`RegressionResultMultipleProtocols`](@ref): Results when analyzing multiple protocols
- [`RegressionResultSingleProtocol`](@ref): Results when analyzing a single protocol

Each result contains posterior and prior inference results from RxInfer.jl.
"""
abstract type RegressionResult <: AbstractInferenceResult end

"""
    RobustRegressionResult <: AbstractInferenceResult

Abstract type for robust Bayesian regression model results using a Student-t likelihood
via Empirical Bayes.

The Student-t likelihood is implemented as:
  τ_i ~ Gamma(ν/2, scale = τ_base/(ν/2))
  y_i | μ, τ_i ~ Normal(μ, precision = τ_i)

where τ_base is a data-driven residual precision constant (Empirical Bayes) and
ν is the fixed degrees-of-freedom parameter controlling tail heaviness.
Marginal: y_i | μ ~ Student-t(ν, μ, τ_base).

Concrete subtypes:
- [`RobustRegressionResultMultipleProtocols`](@ref)
- [`RobustRegressionResultSingleProtocol`](@ref)
"""
abstract type RobustRegressionResult <: AbstractInferenceResult end

"""
    HBMResult <: AbstractInferenceResult

Abstract type for Hierarchical Bayesian Model results analyzing protein enrichment.

Concrete subtypes:
- [`HBMResultMultipleProtocols`](@ref): Results when analyzing multiple protocols
- [`HBMResultSingleProtocol`](@ref): Results when analyzing a single protocol

Each result contains posterior and prior inference results from RxInfer.jl.
"""
abstract type HBMResult <: AbstractInferenceResult end

"""
    RegressionResultMultipleProtocols <: RegressionResult

Stores regression model inference results for analyses involving multiple experimental protocols.

# Fields
- `posterior::InferenceResult`: Posterior distribution from variational inference (RxInfer.jl)
- `prior::InferenceResult`: Prior distribution used in the model

# Notes
Contains posterior samples for regression parameters including slopes (β₁) that measure
correlation between candidate protein and bait protein abundance across protocols.

See also: [`RegressionResultSingleProtocol`](@ref), [`RegressionModel`](@ref)
"""
struct RegressionResultMultipleProtocols <: RegressionResult
    posterior::InferenceResult
    prior::InferenceResult
end

"""
    RegressionResultSingleProtocol <: RegressionResult

Stores regression model inference results for analyses involving a single experimental protocol.

# Fields
- `posterior::InferenceResult`: Posterior distribution from variational inference (RxInfer.jl)
- `prior::InferenceResult`: Prior distribution used in the model

# Notes
Simpler structure than multiple protocol case, with regression parameters for a single protocol.

See also: [`RegressionResultMultipleProtocols`](@ref), [`RegressionModel`](@ref)
"""
struct RegressionResultSingleProtocol <: RegressionResult
    posterior::InferenceResult
    prior::InferenceResult
end

"""
    RobustRegressionResultMultipleProtocols <: RobustRegressionResult

Stores robust regression model results (Student-t likelihood via Empirical Bayes)
for analyses involving multiple experimental protocols.

# Fields
- `posterior::InferenceResult`: Posterior distribution from VMP inference
- `prior::InferenceResult`: Prior distribution used in the model
- `nu::Float64`: Student-t degrees of freedom (controls tail heaviness)
- `τ_base::Float64`: Data-driven residual precision constant (Empirical Bayes)
"""
struct RobustRegressionResultMultipleProtocols <: RobustRegressionResult
    posterior::InferenceResult
    prior::InferenceResult
    nu::Float64
    τ_base::Float64
end

"""
    RobustRegressionResultSingleProtocol <: RobustRegressionResult

Stores robust regression model results (Student-t likelihood via Empirical Bayes)
for analyses involving a single experimental protocol.

# Fields
- `posterior::InferenceResult`: Posterior distribution from VMP inference
- `prior::InferenceResult`: Prior distribution used in the model
- `nu::Float64`: Student-t degrees of freedom (controls tail heaviness)
- `τ_base::Float64`: Data-driven residual precision constant (Empirical Bayes)
"""
struct RobustRegressionResultSingleProtocol <: RobustRegressionResult
    posterior::InferenceResult
    prior::InferenceResult
    nu::Float64
    τ_base::Float64
end

"""
    AnyRegressionResult

Union type alias for either standard or robust regression results.
"""
const AnyRegressionResult = Union{RegressionResult, RobustRegressionResult}

"""
    WAICResult

Widely Applicable Information Criterion (WAIC) computed from VMP posteriors.

WAIC = -2 * (lppd - p_waic), where:
- lppd = Σᵢ log(1/S × Σₛ p(yᵢ | θˢ)) is the log pointwise predictive density
- p_waic = Σᵢ var_s(log p(yᵢ | θˢ)) is the effective number of parameters

# Fields
- `waic::Float64`: WAIC value (lower is better)
- `lppd::Float64`: Log pointwise predictive density
- `p_waic::Float64`: Effective number of parameters
- `pointwise_waic::Vector{Float64}`: Per-observation WAIC contributions
- `se::Float64`: Standard error of WAIC estimate
"""
struct WAICResult
    waic::Float64
    lppd::Float64
    p_waic::Float64
    pointwise_waic::Vector{Float64}
    se::Float64
end

function Base.show(io::IO, w::WAICResult)
    println(io, "WAICResult")
    println(io, "  WAIC    = $(round(w.waic, digits=2)) (SE = $(round(w.se, digits=2)))")
    println(io, "  lppd    = $(round(w.lppd, digits=2))")
    println(io, "  p_waic  = $(round(w.p_waic, digits=2))")
    print(io,   "  n_obs   = $(length(w.pointwise_waic))")
end

"""
    ModelComparisonResult

Result of comparing Normal vs. robust (Student-t) regression models via WAIC.

# Fields
- `normal_waic::WAICResult`: WAIC for the standard Normal regression model
- `robust_waic::Union{WAICResult, Nothing}`: WAIC for the robust model (Nothing if not computed)
- `delta_waic::Float64`: normal - robust (positive means robust is better)
- `delta_se::Float64`: Standard error of the WAIC difference
- `preferred_model::Symbol`: `:normal` or `:robust`
"""
struct ModelComparisonResult
    normal_waic::WAICResult
    robust_waic::Union{WAICResult, Nothing}
    delta_waic::Float64
    delta_se::Float64
    preferred_model::Symbol
end

function Base.show(io::IO, m::ModelComparisonResult)
    println(io, "ModelComparisonResult")
    println(io, "───────────────────────────────────")
    println(io, "  Normal WAIC  = $(round(m.normal_waic.waic, digits=2))")
    if !isnothing(m.robust_waic)
        println(io, "  Robust WAIC  = $(round(m.robust_waic.waic, digits=2))")
    else
        println(io, "  Robust WAIC  = not computed")
    end
    println(io, "  ΔWAIC        = $(round(m.delta_waic, digits=2)) ± $(round(m.delta_se, digits=2))")
    print(io,   "  Preferred    = :$(m.preferred_model)")
end

"""
    HBMResultMultipleProtocols <: HBMResult

Stores Hierarchical Bayesian Model inference results for analyses with multiple protocols.

# Fields
- `posterior::InferenceResult`: Posterior distribution from variational inference (RxInfer.jl)
- `prior::InferenceResult`: Prior distribution used in the model

# Notes
Contains posterior samples for log2 fold changes (log2FC) at both protocol and experiment levels,
capturing enrichment while accounting for between-protocol heterogeneity.

See also: [`HBMResultSingleProtocol`](@ref), [`HierarchicalBayesianModel`](@ref)
"""
struct HBMResultMultipleProtocols <: HBMResult
    posterior::InferenceResult
    prior::InferenceResult
end

"""
    HBMResultSingleProtocol <: HBMResult

Stores Hierarchical Bayesian Model inference results for single protocol analyses.

# Fields
- `posterior::InferenceResult`: Posterior distribution from variational inference (RxInfer.jl)
- `prior::InferenceResult`: Prior distribution used in the model

# Notes
Simpler hierarchical structure than multiple protocol case, with log2FC parameters
at the experiment level only.

See also: [`HBMResultMultipleProtocols`](@ref), [`HierarchicalBayesianModel`](@ref)
"""
struct HBMResultSingleProtocol <: HBMResult
    posterior::InferenceResult
    prior::InferenceResult
end

# ----------------------- BayesResult ----------------------- #

"""
    BayesResult

**Main output type** containing complete Bayesian analysis results for a single protein.

This structure holds Bayes factors, posterior statistics, and inference results from both
the Hierarchical Bayesian Model (HBM) and Bayesian regression model.

# Fields
- `bfHBM::Union{Matrix{Float64},Nothing}`: Bayes factors from HBM (enrichment model)
  - Matrix format: (protocols × comparison_types)
  - Each element is BF₁₀ for log2FC > 0 at that protocol
  - `nothing` if HBM failed to converge

- `bfRegression::Union{Vector{Float64},Nothing,Float64}`: Bayes factors from regression (correlation model)
  - Vector of BFs for each protocol (multiple protocols)
  - Single Float64 for single protocol
  - `nothing` if regression failed or not computed

- `HBM_stats::Dict{Symbol,Union{...}}`: Summary statistics from HBM posterior
  - `:empty` key present if no statistics available
  - Keys typically include: `:mean_log2FC`, `:sd_log2FC`, `:pd`, `:rope_percentage`, `:ess_bulk`, `:rhat`

- `regression_stats::Union{Dict{Symbol,...},Nothing}`: Summary statistics from regression posterior
  - Keys include: `:mean_slope`, `:sd_slope`, `:pd`, `:ess_bulk`, `:rhat`
  - `nothing` if regression not computed

- `hbm_result::Union{Nothing,HBMResult}`: Full HBM inference result from RxInfer.jl
  - Contains posterior and prior `InferenceResult` objects
  - `nothing` if model failed

- `regression_result::Union{Nothing,RegressionResult}`: Full regression inference result
  - Contains posterior and prior `InferenceResult` objects
  - `nothing` if not computed or failed

- `protein_name::String`: Identifier of the protein analyzed

# Accessor Functions
- `getProteinName(bf)`: Returns protein identifier
- `getbfHBM(bf)`: Returns enrichment Bayes factors
- `getbfRegression(bf)`: Returns correlation Bayes factors
- `getHBMstats(bf)`: Returns HBM summary statistics dictionary
- `getregressionstats(bf)`: Returns regression summary statistics
- `getPosterior(bf)`: Returns tuple of (HBM posterior, regression posterior)
- `getPrior(bf)`: Returns tuple of (HBM prior, regression prior)

# Examples
```julia
# After running analysis for a protein
result = BayesResult(...)

# Check enrichment evidence
bf_enrichment = getbfHBM(result)  # Matrix or Nothing
println("Enrichment BF: ", bf_enrichment)

# Get posterior statistics
stats = getHBMstats(result)
println("Mean log2FC: ", stats[:mean_log2FC])
println("Probability of direction: ", stats[:pd])
println("Rhat convergence: ", stats[:rhat])

# Check correlation evidence
bf_correlation = getbfRegression(result)
```

# Notes
- This is the intermediate result type before copula-based combination
- Missing or failed models result in `nothing` fields rather than errors
- Convergence diagnostics (ESS, Rhat) should be checked before trusting results
- Results are combined across proteins using [`combined_BF`](@ref) to produce final probabilities

See also: [`CombinedBayesResult`](@ref), [`HBMResult`](@ref), [`RegressionResult`](@ref),
[`BayesFactorTriplet`](@ref)
"""
struct BayesResult
    bfHBM::Union{Matrix{Float64},Nothing}
    bfRegression::Union{Vector{Float64},Nothing,Float64}
    HBM_stats::Dict{Symbol,Union{Vector{Vector{Float64}},Vector{Float64},Vector{String}}}
    regression_stats::Union{Dict{Symbol,Union{Float64,Vector{Vector{Float64}},Vector{Float64},Vector{String},String}},Nothing}
    hbm_result::Union{Nothing,HBMResult}
    regression_result::Union{Nothing,RegressionResult,RobustRegressionResult}
    protein_name::String
end

getPosterior(bf::BayesResult) = (bf.hbm_result.posterior, bf.regression_result.posterior)
getPrior(bf::BayesResult) = (bf.hbm_result.prior, bf.regression_result.prior)
getProteinName(bf::BayesResult) = bf.protein_name
getbfHBM(bf::BayesResult) = bf.bfHBM
getbfRegression(bf::BayesResult) = bf.bfRegression
getHBMstats(bf::BayesResult) = bf.HBM_stats
getregressionstats(bf::BayesResult) = bf.regression_stats

function Base.show(io::IO, bf::BayesResult)
    println(io, "BayesResult for protein   : ", bf.protein_name)
    println(io, "_______________________________________________")
    println(io, "")
    println(io, " - HBM Bayes Factors      : ", haskey(bf.HBM_stats, :empty) ? "not computed" : "size = $(size(bf.bfHBM))")
    println(io, " - Regression Bayes Fact. : ", isnothing(bf.bfRegression) ? "None" : "Available")
    println(io, " - Regression Stats       : ", isnothing(bf.regression_stats) ? "None" : "Available")
end


# ----------------------- Copula-related structures ----------------------- #
abstract type EvidenceTriplet end

function Base.show(io::IO, triplet::EvidenceTriplet)
    println(io, "$(typeof(triplet))")
    n_proteins = length(triplet.enrichment)
    println(io, "Number of proteins: $n_proteins")
end

# ------ BayesFactorTriplet ------ #

"""
    BayesFactorTriplet{T<:Real} <: EvidenceTriplet

Container for three complementary lines of evidence (Bayes factors) for protein interactions.

This structure holds Bayes factors from the three statistical models used in BayesInteractomics:
1. Enrichment model (HBM): Is the protein quantitatively enriched?
2. Correlation model (Regression): Does abundance correlate with bait?
3. Detection model (Beta-Bernoulli): Is detection rate higher in samples?

# Fields
- `enrichment::Vector{T}`: Bayes factors for enrichment (log2FC > 0)
- `correlation::Vector{T}`: Bayes factors for positive correlation (β₁ > 0)
- `detection::Vector{T}`: Bayes factors for detection rate (θ_sample > θ_control)

All vectors must have the same length (one element per protein).

# Constructor
```julia
BayesFactorTriplet(enrichment, correlation, detection)
```

Validates that:
- All three vectors have equal length
- Values are not all in [0,1] (warns if they appear to be probabilities)

# Methods
- `length(triplet)`: Number of proteins
- `log(triplet)`: Returns new triplet with log10-transformed Bayes factors

# Examples
```julia
# Create triplet for N proteins
bf_triplet = BayesFactorTriplet(
    enrichment = [10.5, 2.3, 0.8, ...],    # BF from HBM
    correlation = [5.2, 1.2, 0.5, ...],     # BF from regression
    detection = [50.0, 10.0, 0.3, ...]      # BF from Beta-Bernoulli
)

# Convert to log scale for visualization
log_bf = log(bf_triplet)

# Access individual evidence
println("Enrichment BFs: ", bf_triplet.enrichment)
```

# Notes
- Bayes factors > 1 support the alternative hypothesis (H₁: interaction)
- Bayes factors < 1 support the null hypothesis (H₀: no interaction)
- Use with [`combined_BF`](@ref) to integrate evidence via copulas
- Convert to probabilities using `BF/(1+BF)` assuming uniform prior

See also: [`PosteriorProbabilityTriplet`](@ref), [`combined_BF`](@ref), [`BayesResult`](@ref)
"""
struct BayesFactorTriplet{T<:Real} <: EvidenceTriplet
    enrichment::Vector{T}
    correlation::Vector{T}
    detection::Vector{T}

    function BayesFactorTriplet(enrichment::Vector{T}, correlation::Vector{T}, detection::Vector{T}) where {T<:Real}
        # 1. Validate the inputs
        _all_are_probabilities(enrichment) && @warn("Enrichment: Check that Bayes Factors are used and not posterior probabilities.")
        _all_are_probabilities(correlation) && @warn("Correlation: Check that Bayes Factors are used and not posterior probabilities.")
        _all_are_probabilities(detection) && @warn("Detection: Check that Bayes Factors are used and not posterior probabilities.")

        # Enforce equal lengths
        len_e = length(enrichment)
        if len_e != length(correlation) || len_e != length(detection)
            throw(DimensionMismatch("All vectors in BayesFactorTriplet must have the same length."))
        end

        return new{T}(enrichment, correlation, detection)
    end
end

function Base.length(triplet::BayesFactorTriplet)
    return length(triplet.enrichment)
end

function Base.log(p::BayesFactorTriplet)
    return BayesFactorTriplet(log10.(p.enrichment), log10.(p.correlation), log10.(p.detection))
end
# ------ PosteriorProbabilityTriplet ------ #

"""
    _all_are_probabilities(v::AbstractVector{<:Real})

Helper function checking if all elements in a vector are valid probabilities (between 0 and 1).

# Arguments
- `v::AbstractVector{<:Real}`: Vector to check

# Returns
- `Bool`: `true` if all elements are in [0, 1], `false` otherwise

# Notes
Used internally for input validation in [`PosteriorProbabilityTriplet`](@ref) and
warning generation in [`BayesFactorTriplet`](@ref).
"""
function _all_are_probabilities(v::AbstractVector{<:Real})
    return all(x -> 0 <= x <= 1, v)
end

"""
    PosteriorProbabilityTriplet{T<:Real} <: EvidenceTriplet

Container for posterior probabilities from three complementary lines of evidence.

Similar to [`BayesFactorTriplet`](@ref) but stores posterior probabilities (0-1 scale)
instead of Bayes factors. Used internally in copula fitting and EM algorithm.

# Fields
- `enrichment::Vector{T}`: Posterior probabilities for enrichment (0-1)
- `correlation::Vector{T}`: Posterior probabilities for positive correlation (0-1)
- `detection::Vector{T}`: Posterior probabilities for higher detection rate (0-1)

All vectors must have the same length and all values must be valid probabilities [0,1].

# Constructor
```julia
PosteriorProbabilityTriplet(enrichment, correlation, detection)
```

Validates that:
- All three vectors have equal length
- All values are in the range [0, 1]
- Throws `ArgumentError` if validation fails

# Methods
- `getindex(triplet, i)`: Extract probability triplet for protein i
- `squeeze(triplet; ϵ=eps(T))`: Squeezes probabilities away from 0 and 1 boundaries

# Examples
```julia
# Convert Bayes factors to probabilities (uniform prior)
bf_triplet = BayesFactorTriplet(...)
pp_triplet = PosteriorProbabilityTriplet(
    bf_triplet.enrichment ./ (1 .+ bf_triplet.enrichment),
    bf_triplet.correlation ./ (1 .+ bf_triplet.correlation),
    bf_triplet.detection ./ (1 .+ bf_triplet.detection)
)

# Squeeze away from boundaries for copula fitting
pp_squeezed = squeeze(pp_triplet, ϵ=1e-10)

# Extract single protein
protein_5_probs = pp_triplet[5]
```

# Notes
- Posterior probabilities assume uniform prior P(H₁) = 0.5
- Boundary values (exactly 0 or 1) can cause numerical issues in copula fitting
- Use [`squeeze`](@ref) to move boundary values slightly inward
- Used internally in [`em_fit_mixture`](@ref) and [`fit_copula`](@ref)

See also: [`BayesFactorTriplet`](@ref), [`squeeze`](@ref), [`combined_BF`](@ref)
"""
struct PosteriorProbabilityTriplet{T<:Real} <: EvidenceTriplet
    enrichment::Vector{T}
    correlation::Vector{T}
    detection::Vector{T}

    # --- Inner Constructor ---
    function PosteriorProbabilityTriplet(
        enrichment::Vector{T},
        correlation::Vector{T},
        detection::Vector{T}
    ) where {T<:Real}

        # 1. Validate the inputs
        _all_are_probabilities(enrichment) || throw(ArgumentError("All enrichment probabilities must be between 0 and 1."))
        _all_are_probabilities(correlation) || throw(ArgumentError("All correlation probabilities must be between 0 and 1."))
        _all_are_probabilities(detection) || throw(ArgumentError("All detection probabilities must be between 0 and 1."))

        len_e = length(enrichment)
        if len_e != length(correlation) || len_e != length(detection)
            throw(DimensionMismatch("All vectors in PosteriorProbabilityTriplet must have the same length."))
        end

        # 2. If validation passes, create the new object
        return new{T}(enrichment, correlation, detection)
    end
end

function Base.getindex(p::PosteriorProbabilityTriplet, i)
    return PosteriorProbabilityTriplet(p.enrichment[i], p.correlation[i], p.detection[i])
end


"""
    squeeze(vec::AbstractVector{T}; ϵ=eps(T)) where {T<:Real}

Squeeze probability values away from boundaries [0, 1] to (ϵ, 1-ϵ).

This function prevents numerical issues in copula fitting by ensuring no probability
is exactly 0 or 1, which can cause problems with log-transforms and inverse CDFs.

# Arguments
- `vec::AbstractVector{T}`: Vector of probabilities
- `ϵ::Real=eps(T)`: Small epsilon value determining boundary distance (default: machine epsilon)

# Returns
- `Vector{T}`: Squeezed probabilities in the range (ϵ, 1-ϵ)

# Formula
```
p_squeezed = p * (1 - 2ϵ) + ϵ
```

This linear transformation maps:
- 0 → ϵ
- 1 → 1-ϵ
- 0.5 → 0.5 (midpoint preserved)

# Examples
```julia
probs = [0.0, 0.5, 1.0, 0.99]
squeezed = squeeze(probs, ϵ=1e-10)
# Result: [1e-10, 0.5, 1.0 - 1e-10, 0.99 - tiny_amount]

# With default machine epsilon
squeezed = squeeze(probs)
```

# Notes
- Default ϵ is `eps(T)` (≈2.22e-16 for Float64)
- Larger ϵ values (e.g., 1e-10) may be needed for numerical stability in some copulas
- Applied automatically in [`squeeze(::PosteriorProbabilityTriplet)`](@ref)

See also: [`squeeze(::PosteriorProbabilityTriplet)`](@ref), [`fit_copula`](@ref)
"""
function squeeze(vec::AbstractVector{T}; ϵ=eps(T)) where {T<:Real}
    # Replace NaN with 0.5 (uninformative) before squeezing
    vec_safe = replace(vec, NaN => T(0.5), Inf => T(1.0), -Inf => T(0.0))
    return (vec_safe .* (1 - 2 * ϵ)) .+ ϵ
end

"""
    squeeze(p::PosteriorProbabilityTriplet{T}; ϵ=eps(T)) where {T<:Real}

Squeeze all probabilities in a triplet away from boundaries.

Applies [`squeeze`](@ref) to each evidence type (enrichment, correlation, detection)
in the probability triplet.

# Arguments
- `p::PosteriorProbabilityTriplet{T}`: Probability triplet
- `ϵ::Real=eps(T)`: Epsilon for boundary distance

# Returns
- `PosteriorProbabilityTriplet{T}`: New triplet with squeezed probabilities

# Examples
```julia
pp_triplet = PosteriorProbabilityTriplet(
    [0.0, 0.5, 1.0],
    [0.1, 0.5, 0.9],
    [0.0, 0.0, 1.0]
)

pp_squeezed = squeeze(pp_triplet, ϵ=1e-10)
# All 0s become 1e-10, all 1s become 1-1e-10
```

# Notes
- Essential preprocessing for copula fitting algorithms
- Used automatically in [`em_fit_mixture`](@ref)

See also: [`squeeze(::AbstractVector)`](@ref), [`PosteriorProbabilityTriplet`](@ref)
"""
function squeeze(p::PosteriorProbabilityTriplet{T}; ϵ=eps(T)) where {T<:Real}
    return PosteriorProbabilityTriplet(
        squeeze(p.enrichment, ϵ=ϵ),
        squeeze(p.correlation, ϵ=ϵ),
        squeeze(p.detection, ϵ=ϵ)
    )
end

# ------ EM results ------ #
"""
    EMResult(π0, π1, joint_H1, logs)

Holds the fitted parameters and convergence logs from the EM algorithm.
"""
struct EMResult
    π0::Float64
    π1::Float64
    joint_H1::SklarDist
    logs::DataFrame
    has_converged::Bool
end

function Base.show(io::IO, r::EMResult)
    println(io, "EMResult(π0=$(round(r.π0, digits=3)))")
    println(io, "------------------------------------")
    println(io, "algorithm has converged: $(r.has_converged)")
    println(io, "Convergence at $(r.logs[end, :iter]) iterations")
end


# ------ AbstractCombinationResult ------ #
"""
    AbstractCombinationResult

Abstract type for results from evidence combination methods.
Subtypes include `CombinedBayesResult` (copula-based) and `LatentClassResult` (VMP-based).
"""
abstract type AbstractCombinationResult end

"""
    get_bf(r::AbstractCombinationResult)

Extract combined Bayes factors from any combination result type.
"""
get_bf(r::AbstractCombinationResult) = r.bf


# ------ CombinedBayesResult ------ #
"""
    CombinedBayesResult(bf, posterior_prob, joint_H0, joint_H1, em_result, em_diagnostics)

The final output of the copula-based Bayesian interactomics analysis, containing
the combined Bayes Factors, posterior probabilities, and fitted mixture models.

# Fields
- `bf::Vector{Float64}`: Combined Bayes factors for each protein
- `posterior_prob::Vector{Float64}`: Posterior probabilities for each protein
- `joint_H0::SklarDist`: Joint distribution under null hypothesis
- `joint_H1::SklarDist`: Joint distribution under alternative hypothesis
- `em_result::EMResult`: Best EM fitting result
- `em_diagnostics::Union{DataFrame, Nothing}`: Diagnostics from EM restarts (nothing if n_restarts=1)
"""
struct CombinedBayesResult <: AbstractCombinationResult
    bf::Vector{Float64}
    posterior_prob::Vector{Float64}
    joint_H0::SklarDist
    joint_H1::SklarDist
    em_result::EMResult
    em_diagnostics::Union{DataFrame, Nothing}
end


# ------ LatentClassResult ------ #
"""
    LatentClassResult(bf, posterior_prob, class_parameters, mixing_weights, free_energy, converged, n_iterations)

The final output of the latent class (VMP-based) Bayesian interactomics analysis.

# Fields
- `bf::Vector{Float64}`: Combined Bayes factors for each protein
- `posterior_prob::Vector{Float64}`: Posterior probabilities for each protein
- `class_parameters::Dict{String, NamedTuple{(:mu, :sigma, :precision), Tuple{Float64,Float64,Float64}}}`:
   Parameters for background and interaction classes
- `mixing_weights::Vector{Float64}`: [P(background), P(interaction)]
- `free_energy::Vector{Float64}`: Free energy per VMP iteration
- `converged::Bool`: Whether the algorithm converged
- `n_iterations::Int`: Number of iterations performed
"""
struct LatentClassResult <: AbstractCombinationResult
    bf::Vector{Float64}
    posterior_prob::Vector{Float64}
    class_parameters::Dict{String, NamedTuple{(:mu, :sigma, :precision), Tuple{Float64,Float64,Float64}}}
    mixing_weights::Vector{Float64}
    free_energy::Vector{Float64}
    converged::Bool
    n_iterations::Int
end

function Base.show(io::IO, r::LatentClassResult)
    println(io, "LatentClassResult")
    println(io, "------------------------------------")
    println(io, "Converged: $(r.converged)")
    println(io, "Iterations: $(r.n_iterations)")
    println(io, "Mixing weights: π₀=$(round(r.mixing_weights[1], digits=3)), π₁=$(round(r.mixing_weights[2], digits=3))")
    println(io, "Background: μ=$(round(r.class_parameters["background"].mu, digits=3)), σ=$(round(r.class_parameters["background"].sigma, digits=3))")
    println(io, "Interaction: μ=$(round(r.class_parameters["interaction"].mu, digits=3)), σ=$(round(r.class_parameters["interaction"].sigma, digits=3))")
end


# ------ BMAResult ------ #
"""
    BMAResult(bf, posterior_prob, copula_result, latent_class_result,
              copula_bic, latent_class_bic, copula_weight, latent_class_weight)

Result of Bayesian Model Averaging (BMA) over copula and latent class combination methods.

BMA weights both methods by their BIC-approximated marginal likelihoods, producing
model-averaged posterior probabilities that properly propagate model uncertainty.

# Fields
- `bf::Vector{Float64}`: Model-averaged combined Bayes factors
- `posterior_prob::Vector{Float64}`: Model-averaged posterior probabilities
- `copula_result::CombinedBayesResult`: Full copula combination result
- `latent_class_result::LatentClassResult`: Full latent class combination result
- `copula_bic::Float64`: BIC for the copula model
- `latent_class_bic::Float64`: BIC for the latent class model
- `copula_weight::Float64`: BIC-derived model weight for copula
- `latent_class_weight::Float64`: BIC-derived model weight for latent class
"""
struct BMAResult <: AbstractCombinationResult
    bf::Vector{Float64}
    posterior_prob::Vector{Float64}
    copula_result::CombinedBayesResult
    latent_class_result::LatentClassResult
    copula_bic::Float64
    latent_class_bic::Float64
    copula_weight::Float64
    latent_class_weight::Float64
end

function Base.show(io::IO, r::BMAResult)
    println(io, "BMAResult")
    println(io, "------------------------------------")
    println(io, "Copula BIC:       $(round(r.copula_bic, digits=2))")
    println(io, "Latent class BIC: $(round(r.latent_class_bic, digits=2))")
    println(io, "Copula weight:    $(round(r.copula_weight, digits=4))")
    println(io, "LC weight:        $(round(r.latent_class_weight, digits=4))")
    println(io, "Proteins:         $(length(r.bf))")
end


# ----------------------- Ranks ----------------------- #
"""
    Ranks{I<:Integer, F<:AbstractFloat}

Stores posterior rank information for multiple entities (e.g., proteins or parameters).

# Fields
- `ranks::Matrix{I}`: Matrix of integer ranks, size = (entities × samples)
- `names::Vector{String}`: Names of the ranked entities
- `mean_ranks::Vector{F}`: Mean rank per entity
- `median_ranks::Vector{F}`: Median rank per entity

# Comments
If one iters over a `Ranks` object, it will iterate over the entities and returns the name of the protein and the ranks of the current entity
"""
struct Ranks{I<:Integer,F<:AbstractFloat}
    ranks::Matrix{I}
    names::Vector{String}
    mean_ranks::Vector{F}
    median_ranks::Vector{F}
end

function Ranks(ranks::Matrix{I}, names::Vector{String}) where {I<:Integer}
    mean_ranks = vec(mean(ranks, dims=2))
    median_ranks = vec(mapslices(median, ranks; dims=2))
    return Ranks{I,Float64}(ranks, names, mean_ranks, median_ranks)
end

getRanks(r::Ranks) = r.ranks
getNames(r::Ranks) = r.names
getMeanRanks(r::Ranks) = r.mean_ranks
getMedianRanks(r::Ranks) = r.median_ranks
Base.length(r::Ranks) = size(r.ranks, 1)
Base.getindex(r::Ranks, i::Integer) = r.ranks[i, :]
Base.iterate(r::Ranks, state=1) = state > length(r) ? nothing : ((r.names[state], r.ranks[state, :]), state + 1)


# ----------------------- Data containers ----------------------- #

# ---- Protocol

"""
    Protocol

    A mutable struct to store the data of a single experimental method or publication.

    Fields:
        - no_experiments: The number of experiments in the protocol.
        - protein_ids: A vector of protein IDs.
        - data: A dictionary with experiment indices as keys and data matrices as values.
                Data matrices have rows as proteins and columns as samples.

    Note: 
        - If one iters over a protocol, it will iterate over the experiments in the protocol and returns 
          the data matrix of the current experiment.
"""
mutable struct Protocol{F<:AbstractFloat,I<:Integer}
    no_experiments::I
    protein_ids::Vector{String}
    data::Dict{I,Matrix{Union{Missing,F}}}
end


# Interface
"""
    getNoExperiments(protocol::Protocol)
    Returns the number of experiments in the protocol
"""
getNoExperiments(protocol::Protocol) = protocol.no_experiments

"""
    getExperiment(protocol::Protocol, index::Integer)

    Returns the data matrix of the experiment with index `index` and throws an error if the index is out of bounds
"""
function getExperiment(protocol::Protocol, index::Integer)
    1 <= index <= getNoExperiments(protocol) || throw(BoundsError(protocol.data, index))
    data = get(protocol.data, index, nothing)
    data === nothing && throw(BoundsError(protocol.data, index))
    return data
end

# Iterator Interface
Base.getindex(x::Protocol, i) = getExperiment(x, i)
Base.firstindex(protocol::Protocol) = 1
Base.lastindex(protocol::Protocol) = length(protocol)
Base.length(protocol::Protocol) = getNoExperiments(protocol)
Base.IteratorEltype(protocol::Protocol) = Base.HasEltype()
Base.eltype(protocol::Protocol{F,I}) where {F<:AbstractFloat,I<:Integer} = Matrix{Union{Missing,F}}
Base.isdone(protocol::Protocol, index::Integer) = index > getNoExperiments(protocol)

getIDs(protocol::Protocol) = protocol.protein_ids
getProtocolData(protocol::Protocol) = protocol.data

function Base.iterate(protocol::Protocol, index::Integer=1)
    Base.isdone(protocol, index) && return nothing
    return getExperiment(protocol, index), index + 1
end

function Base.show(io::IO, protocol::Protocol)
    println(io, "Protocol with $(protocol.no_experiments) experiments and $(length(protocol.protein_ids)) proteins.")
end

# ---- InteractionData
abstract type AbstractInteractionData end


function getProtocolPositions(no_experiments::Vector{I}) where {I<:Integer}
    length(no_experiments) == 0 && throw(ArgumentError("'no_experiments' cannot be empty when calling getProtocolPositions: "))
    length(no_experiments) == 1 && return [2]

    protocolPositions::Vector{I} = [2]
    idx = 1
    while idx < length(no_experiments)
        push!(protocolPositions, protocolPositions[idx] + no_experiments[idx] + 1)
        idx += 1
    end
    return protocolPositions::Vector{I}
end

function getProtocolPositions(no_experiments::Dict{I,I}) where {I<:Integer}
    n = [no_experiments[i] for i in 1:length(no_experiments)]
    return getProtocolPositions(n)
end

"""
    getPositions(num_experiments::Dict{I,I}, no_parameters::I) where {I<:Integer}

    This function returns a vector of positions in paramter vectors where:
        - experiment_positions (Vector{Int}):   A vector of positions where parameters for individual experiments are stored.
        - protocol_positions (Vector{Int}):     A vector of positions where parameters for individual protocols are stored.

    Args:
        - num_experiments::Dict{I,I}:    The number of protocols.
        - no_parameters::I              The maximum number of experiments per protocol.

    Returns:
        - protocol_positions (Vector{Int}): A vector of positions where parameters for individual protocols are stored.
        - experiment_positions (Vector{Int}): A vector of positions where parameters for individual experiments are stored.
        - matched_positions (Vector{Int}): A vector of positions where the value shows the protocol index for each experiment.
"""
function getPositions(num_experiments::Dict{I,I}, no_parameters::I) where {I<:Integer}
    protocol_positions = getProtocolPositions(num_experiments)
    # experiment positions
    experiment_positions = setdiff(2:no_parameters, protocol_positions)
    # matched positions
    matched_positions = [protocol_positions[protocol] for protocol in 1:length(num_experiments) for _ in 1:num_experiments[protocol]]
    # return protocol_positions, experiment_positions, matched_positions
    return protocol_positions, experiment_positions, matched_positions
end

"""
    InteractionData

    A structure to store the data of multiple experimental methods or publications.

    Fields:
        - protein_IDs: A vector of protein IDs.
        - protein_names: A vector of protein names.
        - samples: A dictionary with protocol indices as keys and Protocol objects as values.
        - controls: A dictionary with protocol indices as keys and Protocol objects as values.
        - no_protocols: The number of protocols.
        - no_experiments: A dictionary with protocol indices as keys and the number of experiments as values.
        - no_parameters_HBM: The number of parameters for the HierarchicalBayesianModel.
        - no_parameters_Regression: The number of parameters for the Regression model.
        - experiment_positions: A vector of positions where parameters for individual experiments are stored.
        - protocol_positions: A vector of positions where parameters for individual protocols are stored.
        - matched_positions: A vector of positions where the value shows the protocol index for each experiment.

    Note:  By iterating over the InteractionData, one can access the sample and control data matrices of each protocol.

    ```julia
    for (protocol, sample, control) in interaction_data
        # do something
    end
    ```
"""
struct InteractionData{F<:AbstractFloat,I<:Integer} <: AbstractInteractionData
    protein_IDs::Vector{String}
    protein_names::Vector{String}
    samples::Dict{I,Protocol{F,I}}
    controls::Dict{I,Protocol{F,I}}

    no_protocols::I             # number of protocols
    no_experiments::Dict{I,I}  # number of experiments per protocol with protocol index as key
    no_parameters_HBM::I        # number of parameters for the HierarchicalBayesianModel
    no_parameters_Regression::I # number of parameters for the Regression model

    protocol_positions::Vector{I}
    experiment_positions::Vector{I}
    matched_positions::Vector{I}
end

# interface

getIDs(data::InteractionData) = data.protein_IDs
getNames(data::InteractionData) = data.protein_names
getNoProtocols(data::InteractionData) = data.no_protocols
getNoExperiments(data::InteractionData) = data.no_experiments

function getNoExperiments(data::InteractionData, index::Integer)
    haskey(data.no_experiments, index) || throw(ArgumentError("Protocol $index does not exist"))
    return data.no_experiments[index]
end

getControls(data::InteractionData) = data.controls
function getControls(data::InteractionData, index::Integer)
    haskey(data.controls, index) || throw(ArgumentError("Protocol $index does not exist"))
    return data.controls[index]
end

getSamples(data::InteractionData) = data.samples

function getSamples(data::InteractionData, index::Integer)
    haskey(data.samples, index) || throw(ArgumentError("Protocol $index does not exist"))
    return data.samples[index]
end

getExperimentPositions(data::InteractionData) = data.experiment_positions
getProtocolPositions(data::InteractionData) = unique(data.protocol_positions)
getMatchedPositions(data::InteractionData) = data.matched_positions


# iteration interface
Base.getindex(data::InteractionData, index::Integer) = Dict("controls" => getControls(data, index), "samples" => getSamples(data, index))



function Base.iterate(data::InteractionData, state::Tuple{Int,Int}=(1, 1))
    p, e = state
    p > data.no_protocols && return nothing
    e > getNoExperiments(data, p) && return iterate(data, (p + 1, 1))
    return ((p, e, getSamples(data, p)[e]), (p, e + 1))
end

# show
function Base.show(io::IO, data::InteractionData)
    num_protocols = getNoProtocols(data)
    num_proteins = length(getIDs(data))

    println(io, "🧬 InteractionData Summary")
    println(io, "─────────────────────────────")
    println(io, " • Number of Protocols     : $num_protocols")
    println(io, " • Number of Proteins      : $num_proteins")
    println(io, " • HBM Parameters          : $(data.no_parameters_HBM)")
    println(io, " • Regression Parameters   : $(data.no_parameters_Regression)")
    println(io, " • Total Experiments       : $(sum(values(data.no_experiments)))")
    println(io, " • Matched Positions       : $(length(data.matched_positions))")
    println(io)

    println(io, "Protocol Details:")
    println(io, "─────────────────────────────")
    for i in 1:num_protocols
        samples = getSamples(data, i)
        controls = getControls(data, i)
        num_exp = getNoExperiments(data, i)
        num_samples = size(getExperiment(samples, 1), 2)  # assuming column = sample
        num_controls = size(getExperiment(controls, 1), 2)

        println(io, " ▸ Protocol $i:")
        println(io, "   - # Experiments         : $num_exp")
        println(io, "   - Sample columns        : $num_samples")
        println(io, "   - Control columns       : $num_controls")
        println(io, "   - Protocol Param Pos    : $(data.protocol_positions[i])")
    end
    return nothing
end


"""
    DataFrame(data::InteractionData)

Convert an `InteractionData` object into a tidy `DataFrame`.

Each row in the resulting `DataFrame` corresponds to a single measurement (value) for a specific protein 
in a specific experiment of a specific protocol, and is labeled as either "Sample" or "Control".

# Arguments
- `data::InteractionData`: An object containing hierarchical experimental data (protocols → experiments → matrices of protein measurements).

# Returns
- A `DataFrame` with the following columns:
    - `:Protocol`   (`Int`)      — The index of the protocol.
    - `:Experiment` (`Int`)      — The index of the experiment within the protocol.
    - `:Protein`    (`String`)   — The protein identifier.
    - `:SampleType` (`String`)   — Either `"Sample"` or `"Control"`.
    - `:Value`      (`Union{Missing, Float64}`) — The measured value for that protein.

# Notes
- The function iterates over all protocols and all experiments within each protocol.
- Protein IDs are assumed to be shared across all experiments and protocols.
- Missing values are preserved in the output `DataFrame`.
"""
function DataFrame(data::InteractionData)
    df = DataFrame(
        Protocol=Int[],
        Experiment=Int[],
        Protein=String[],
        SampleType=String[],
        Value=Union{Missing,Float64}[]
    )

    protein_ids = getIDs(data)

    for protocol in 1:getNoProtocols(data)
        samples = getSamples(data, protocol)
        controls = getControls(data, protocol)
        num_experiments = getNoExperiments(data, protocol)

        for exp in 1:num_experiments
            sample_mat = samples[exp]
            control_mat = controls[exp]

            for (i, pid) in enumerate(protein_ids)
                for val in sample_mat[i, :]
                    push!(df, (protocol, exp, pid, "Sample", val))
                end
                for val in control_mat[i, :]
                    push!(df, (protocol, exp, pid, "Control", val))
                end
            end
        end
    end

    return df
end

function validate(data::InteractionData)
    boolean = true
    for i in 1:data.no_protocols
        if getIDs(data.samples[i]) != getIDs(data.controls[i])
            @warn "Protocol $i: Sample and control protein IDs mismatch"
            boolean = false
        end
    end
    return boolean
end

# --- Protein Struct ---
"""
    Protein

    A struct to store the data of a single protein.

    Fields:
        - id: The ID of the protein.
        - name: The name of the protein.
        - samples: A vector of Dictionaries with experiment indices as keys and the values for the samples.
        - controls: A vector of Dictionaries with experiment indices as keys and the values for the controls.
"""
struct Protein{F,I<:Integer}
    id::String
    name::String
    samples::Vector{Dict{I,Vector{F}}}
    controls::Vector{Dict{I,Vector{F}}}
end

function Base.show(io::IO, p::Protein)
    println(io, "Protein with ID $(p.id) and name $(p.name)")
    println(io, "Data from $(length(p.samples)) protocol(s) is available.")
end


"""
    getProteinData(data::InteractionData, protein_index::I) where I<:Integer

    Get the data for a specific protein from an InteractionData object.

    Args:
        - data (InteractionData): The InteractionData object.
        - protein_index (Integer): The index of the protein in the InteractionData object.

    Returns:
        - Protein: The data for the specific protein as a Protein struct.
"""
function getProteinData(data::InteractionData, protein_index::I) where I<:Integer
    F = eltype(getSamples(data, 1)[1])
    num_protocols = getNoProtocols(data)

    samples_by_protocol = Vector{Dict{I,Vector{Union{Missing,F}}}}(undef, num_protocols)
    controls_by_protocol = Vector{Dict{I,Vector{Union{Missing,F}}}}(undef, num_protocols)

    for p in 1:num_protocols
        sample_protocol = getSamples(data, p)
        control_protocol = getControls(data, p)

        num_experiments = getNoExperiments(sample_protocol)

        # Use comprehensions for a more concise construction
        samples_by_protocol[p] = Dict(e => sample_protocol[e][protein_index, :] for e in 1:num_experiments)
        controls_by_protocol[p] = Dict(e => control_protocol[e][protein_index, :] for e in 1:num_experiments)
    end

    return Protein(
        data.protein_IDs[protein_index],
        data.protein_names[protein_index],
        samples_by_protocol,
        controls_by_protocol
    )
end

getIDs(protein::Protein) = protein.id
getNames(protein::Protein) = protein.name
Base.length(protein::Protein) = length(protein.samples)

getControls(protein::Protein) = protein.controls
getControls(protein::Protein, index::Integer) = protein.controls[index]

getSamples(protein::Protein) = protein.samples
getSamples(protein::Protein, index::Integer) = protein.samples[index]


#check if two proteins are identical
function Base.:(==)(p1::Protein, p2::Protein)
    p1_controls, p2_controls = getControls(p1), getControls(p2)
    p1_samples, p2_samples = getSamples(p1), getSamples(p2)

    # replace missing with 0

    for i in 1:length(p1_controls)
        for j in 1:length(p1_controls[i])
            x1, x2 = p1_controls[i][j], p2_controls[i][j]
            y1, y2 = p1_samples[i][j], p2_samples[i][j]
            # replace missing with 0
            replace!(p1_controls[i][j], missing => 0)
            replace!(p2_controls[i][j], missing => 0)
            replace!(p1_samples[i][j], missing => 0)
            replace!(p2_samples[i][j], missing => 0)

            # check if the controls and samples are identical
            if x1 != x2 || y1 != y2
                return false
            end
        end
    end

    return true
end

function getMaxExperiments(protein::Protein)
    x = getSamples(protein)
    max_value::Int64 = 1
    for value ∈ x
        max_value = length(value) > max_value ? length(value) : max_value
    end
    return max_value
end

function getMaxSamples(protein::Protein)
    max_samples = 1
    for samples_in_protocol in getSamples(protein)
        for (_, sample_values) in samples_in_protocol
            max_samples = max(max_samples, length(sample_values))
        end
    end
    return max_samples
end

function getMatrix(protein::Protein, data::Vector{Dict{I,Vector{Union{Missing,F}}}}) where {F<:AbstractFloat,I<:Integer}
    # convert to a Array
    max_experiments, max_samples = getMaxExperiments(protein), getMaxSamples(protein)
    dims = (length(data), max_experiments, max_samples)
    x::Array{Union{Missing,F},3} = fill(missing, dims...)

    @inbounds for (sample, experiment) ∈ Iterators.product(1:length(data), 1:max_experiments)
        vals = get(data[sample], experiment, nothing)
        isnothing(vals) && continue
        x[sample, experiment, 1:length(vals)] .= vals
    end
    return x
end

"""
    getSampleMatrix(p::Protein)

Construct a 3D array from the sample values of a `Protein` across all protocols and experiments.

# Arguments
- `p::Protein`: A `Protein` struct containing sample and control data for each protocol and experiment.

# Returns
- A 3D array `Array{Union{Missing, Float64}, 3}` with dimensions `(num_protocols, max_experiments, max_samples)`, where:
    - `num_protocols`: Number of protocols.
    - `max_experiments`: Maximum number of experiments across protocols.
    - `max_samples`: Maximum number of replicate/sample values across all experiments.

# Notes
- Missing values are used to pad the array when sample counts are unequal.
- Use this to convert nested sample data into a dense tensor format for statistical modeling or visualization.
"""
getSampleMatrix(p::Protein) = getMatrix(p, getSamples(p))

"""
    getControlMatrix(p::Protein)

Construct a 3D array from the control values of a `Protein` across all protocols and experiments.

# Arguments
- `p::Protein`: A `Protein` struct containing sample and control data for each protocol and experiment.

# Returns
- A 3D array `Array{Union{Missing, Float64}, 3}` with dimensions `(num_protocols, max_experiments, max_samples)`, where:
    - `num_protocols`: Number of protocols.
    - `max_experiments`: Maximum number of experiments across protocols.
    - `max_samples`: Maximum number of replicate/sample values across all experiments.

# Notes
- Missing values are used to pad the array when sample counts are unequal.
- Use this to convert nested control data into a dense tensor format for statistical modeling or visualization.
"""
getControlMatrix(p::Protein) = getMatrix(p, getControls(p))

function Base.iterate(protein::Protein, index::Integer=1)
    index > length(getControls(protein)) && return nothing
    return_value = Dict("samples" => getSamples(protein, index), "controls" => getControls(protein, index))
    return return_value, index + 1
end