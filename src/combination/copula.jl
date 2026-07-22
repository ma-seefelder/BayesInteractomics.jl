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

This file comprises all functions to combine the individual partially dependent
Bayes Factors from the different statistical methods into a combined 
Bayes Factor using a copula. 
=#

# -------------------------------------------
# Define supported copula types 
# -------------------------------------------
const COPULA_TYPES = Dict(
        "ClaytonCopula" => ClaytonCopula,   # skewed towards lower tail, asymmetric
        "FrankCopula" => FrankCopula,       # neutral in tails, symmetric
        "GumbelCopula" => GumbelCopula,     # skewed towards upper tail, asymmetric
        "GaussianCopula" => GaussianCopula, # no tail skey, symmetric
        "EmpiricalCopula" => EmpiricalCopula,
        "JoeCopula" => JoeCopula
    )

# -------------------------------------------
# Evidence-stream selection
# -------------------------------------------
# Canonical order of the three copula evidence dimensions. The H0/H1 marginal
# tuples are ALWAYS packed as (enrichment, correlation, detection); the copula
# pseudo-obs matrix rows follow the same order. Dropping a stream (ABL-P2) removes
# its row + marginal while preserving the canonical order of the survivors.
const CANONICAL_STREAMS = (:enrichment, :correlation, :detection)

"""
    _active_streams(streams) -> Vector{Symbol}

Return the subset of `CANONICAL_STREAMS` present in `streams`, in canonical order
(enrichment, correlation, detection). The caller-supplied `streams` order is
ignored — set membership drives dimensionality, canonical order keeps the marginal
tuple and pseudo-obs rows aligned. Default `streams = collect(CANONICAL_STREAMS)`
yields all three (backward-compatible, byte-identical).
"""
function _active_streams(streams::AbstractVector{Symbol})
    return Symbol[s for s in CANONICAL_STREAMS if s in streams]
end


######################################################
# LocationShifted{T}: generic shifted distribution (defined in types.jl)
# LocationShiftedGamma = LocationShifted{Gamma{Float64}} (alias in types.jl)
######################################################

######################################################
# NaN-safe logpdf helper
######################################################

"""
    _safe_logpdf_vec(dist::SklarDist, p_triplets, lo, hi) -> Vector{Float64}

Compute `logpdf` for each column of `p_triplets` against `dist`, clamping to
`[lo, hi]`.  Standard `logpdf(SklarDist, x)` can return NaN when marginal CDFs
hit 0 or 1 exactly — the copula density is undefined there.

This helper decomposes the SklarDist logpdf into:
    logpdf(copula, u_squeezed) + Σ logpdf(marginal_i, x_i)
where `u_squeezed` = cdf values squeezed to `(ε, 1-ε)` before entering the copula.
This prevents NaN at the source rather than patching after the fact.
"""
function _safe_logpdf_vec(dist::SklarDist, p_triplets::AbstractMatrix, lo::Float64, hi::Float64;
                          gpd_tails::Union{Nothing, NamedTuple} = nothing,
                          streams::AbstractVector{Symbol} = collect(CANONICAL_STREAMS))
    cop   = dist.C
    margs = dist.m
    d     = length(margs)
    n     = size(p_triplets, 2)
    ε     = 1e-10
    vals  = Vector{Float64}(undef, n)
    u     = Vector{Float64}(undef, d)  # pre-allocate outside loop

    # Active streams in canonical order: row k of p_triplets / margs[k] is `active[k]`.
    # GPD tails are keyed by stream IDENTITY (not absolute row index) so a dropped
    # stream cannot mis-route the enrichment/correlation tail onto another dimension.
    active = _active_streams(streams)
    @assert length(active) == d "row count $d disagrees with active streams $(active)"
    gpd_e = (gpd_tails !== nothing && haskey(gpd_tails, :enrichment)) ? gpd_tails.enrichment : nothing
    gpd_c = (gpd_tails !== nothing && haskey(gpd_tails, :correlation)) ? gpd_tails.correlation : nothing

    @inbounds for j in 1:n
        # marginal log-densities + squeezed CDF for copula
        ll_marg = 0.0
        for k in 1:d
            x_k = p_triplets[k, j]
            s_k = active[k]
            if s_k === :enrichment && gpd_e !== nothing
                ll_marg += gpd_extended_logpdf(margs[k], Float64(x_k), gpd_e)
                u[k] = gpd_extended_cdf(margs[k], Float64(x_k), gpd_e)
            elseif s_k === :correlation && gpd_c !== nothing
                ll_marg += gpd_extended_logpdf(margs[k], Float64(x_k), gpd_c)
                u[k] = gpd_extended_cdf(margs[k], Float64(x_k), gpd_c)
            else
                ll_marg += logpdf(margs[k], x_k)
                u[k] = clamp(cdf(margs[k], x_k), ε, 1.0 - ε)
            end
        end
        ll_cop = logpdf(cop, u)
        v = ll_marg + ll_cop
        vals[j] = isnan(v) || !isfinite(v) ? lo : clamp(v, lo, hi)
    end
    return vals
end

######################################################
# Jittered CDF for DiscreteEmpirical (Denuit & Lambert 2005)
######################################################

"""
    _jitter_discrete_to_uniform(d::DiscreteEmpirical, values::AbstractVector{Float64};
                                 rng::AbstractRNG = Random.default_rng()) -> Vector{Float64}

Convert discrete detection values to pseudo-uniform observations suitable for copula fitting.

Implements the randomized CDF (Denuit & Lambert, 2005):
    F*(x) = F(x-) + U * P(X = x),  U ~ Uniform[0, 1]

where F(x-) = cdf(d, x) - pdf(d, x) is the left-limit of the CDF.
Results are clamped to (1e-6, 1 - 1e-6) to avoid copula boundary issues.

This allows discrete detection BFs to be used as the third pseudo-observation
dimension when fitting the copula in the combined_BF path.
"""
function _jitter_discrete_to_uniform(d::DiscreteEmpirical, values::AbstractVector{Float64};
                                      rng::AbstractRNG = Random.default_rng())
    result = similar(values)
    for i in eachindex(values)
        p_x = pdf(d, values[i])
        f_minus = cdf(d, values[i]) - p_x  # left-limit CDF = F(x-) = F(x) - P(X=x)
        u = rand(rng)
        result[i] = clamp(f_minus + u * p_x, 1e-6, 1.0 - 1e-6)
    end
    return result
end

######################################################
# Function definitions: Utils
######################################################

"""
    posterior_probability_from_bayes_factor(bf::BayesFactorTriplet) -> PosteriorProbabilityTriplet

Converts a `BayesFactorTriplet` into a `PosteriorProbabilityTriplet`.

This function takes a triplet of Bayes factor vectors (for enrichment, correlation, and detection)
and converts each vector into posterior probabilities, assuming prior odds of 1. It then returns
these new vectors wrapped in a `PosteriorProbabilityTriplet` struct.

# Arguments
- `bf::BayesFactorTriplet{R}`: A struct containing vectors of Bayes factors for enrichment, correlation, and detection.

# Returns
- `PosteriorProbabilityTriplet`: A struct containing the corresponding posterior probability vectors.
"""
function posterior_probability_from_bayes_factor(bf::BayesFactorTriplet{R}) where {R<:Real}
    return PosteriorProbabilityTriplet(
        posterior_probability_from_bayes_factor.(bf.enrichment),
        posterior_probability_from_bayes_factor.(bf.correlation),
        posterior_probability_from_bayes_factor.(bf.detection)
    )
end

posterior_probability_from_bayes_factor(p::PosteriorProbabilityTriplet) = p

"""
    posterior_probability_from_bayes_factor(bf)

    Converts a Bayes Factor to a posterior probability assuming prior odds = 1. 
    
    Edge cases:
    - If bf = Inf, returns 1.0
    - If bf = -Inf, returns 0.0
    - If bf = NaN, returns 0.5 (uninformative)
"""
function posterior_probability_from_bayes_factor(bf::R) where {R<:Real}
    if isnan(bf)
        return 0.5  # Uninformative when Bayes factor is undefined
    end
    if isfinite(bf)
        return bf / (1 + bf)
    end
    bf == Inf ? (p = 1.0) : (p = 0.0)
    return p
end



function weighted_resample(p_triplets, weights::Vector{T}, n::Int) where T
    idx = sample(1:length(weights), Weights(weights), n; replace=true)
    return p_triplets[:, idx]
end

######################################################
# Prior calibration for different experiment types
######################################################
"""
    EXPERIMENT_PRIORS

Prior hyperparameters for different experiment types.
Based on typical true positive rates in the literature.

Keys:
- `:APMS`: AP-MS experiments (~10% expected interactions)
- `:BioID`: BioID experiments (~20% expected interactions)
- `:TurboID`: TurboID experiments (~25% expected interactions)
- `:default`: Default prior (~12.5% expected, conservative)
- `:permissive`: Permissive prior (~33% expected)
- `:stringent`: Stringent prior (~5% expected)
"""
const EXPERIMENT_PRIORS = Dict{Symbol, NamedTuple{(:α, :β), Tuple{Float64, Float64}}}(
    :APMS => (α = 20.0, β = 180.0),       # ~10% expected interactions
    :BioID => (α = 30.0, β = 120.0),      # ~20% expected interactions
    :TurboID => (α = 40.0, β = 110.0),    # ~25% expected interactions
    :default => (α = 25.0, β = 175.0),    # ~12.5% expected (conservative)
    :permissive => (α = 50.0, β = 100.0), # ~33% expected
    :stringent => (α = 10.0, β = 190.0),  # ~5% expected
)

"""
    get_prior_hyperparameters(experiment_type::Symbol) -> NamedTuple

Get Beta prior hyperparameters for π₁ based on experiment type.

# Arguments
- `experiment_type::Symbol`: One of `:APMS`, `:BioID`, `:TurboID`, `:default`, `:permissive`, `:stringent`

# Returns
- `NamedTuple{(:α, :β)}`: Prior hyperparameters for Beta distribution on π₁
"""
function get_prior_hyperparameters(experiment_type::Symbol)
    if haskey(EXPERIMENT_PRIORS, experiment_type)
        return EXPERIMENT_PRIORS[experiment_type]
    else
        @warn "Unknown experiment type '$experiment_type', using default prior"
        return EXPERIMENT_PRIORS[:default]
    end
end

"""
    estimate_prior_empirical_bayes(p::PosteriorProbabilityTriplet, joint_H0;
                                    grid_size=20) -> NamedTuple

Estimate prior hyperparameters using empirical Bayes (marginal likelihood maximization).

# Arguments
- `p::PosteriorProbabilityTriplet`: Posterior probabilities from the three models
- `joint_H0`: The fitted H0 distribution (SklarDist)
- `grid_size::Int=20`: Grid resolution for hyperparameter search

# Returns
- `NamedTuple{(:α, :β, :expected_π1)}`: Estimated hyperparameters and implied expected π₁
"""
function estimate_prior_empirical_bayes(p::PosteriorProbabilityTriplet,
                                         joint_H0;
                                         grid_size::Int = 20)
    # Grid search over α, β combinations
    α_grid = range(5.0, 100.0, length=grid_size)
    β_grid = range(50.0, 500.0, length=grid_size)

    best_ml = -Inf
    best_α, best_β = 25.0, 175.0

    # Squeeze to avoid logpdf issues at boundaries
    p_squeezed = squeeze(p, ϵ=1e-10)
    p_triplets = hcat(p_squeezed.enrichment, p_squeezed.correlation, p_squeezed.detection)'

    for α in α_grid, β in β_grid
        # Expected π₁ under this prior
        π1_prior = α / (α + β)
        π0_prior = 1 - π1_prior

        # Marginal likelihood approximation
        # P(data | α, β) ≈ ∫ P(data | π) P(π | α, β) dπ
        # Use prior mean as point estimate
        log_lik_H0 = _safe_logpdf_vec(joint_H0, p_triplets, -700.0, 700.0)

        # Filter out non-finite values
        finite_mask = isfinite.(log_lik_H0)
        if sum(finite_mask) < 10
            continue
        end

        # Approximate marginal likelihood using finite values only
        ml = sum(log.(π0_prior .* exp.(clamp.(log_lik_H0[finite_mask], -700.0, 700.0)) .+ π1_prior))

        if isfinite(ml) && ml > best_ml
            best_ml = ml
            best_α, best_β = α, β
        end
    end

    return (α = best_α, β = best_β, expected_π1 = best_α / (best_α + best_β))
end

######################################################
# Weighted fitting helpers for H1 re-fitting
######################################################
"""
    safe_damped_beta(fit::Beta, prev::Beta, α_damp::Float64) -> Beta

Compute damped Beta parameters with safety checks for NaN/Inf values.

Returns: `Beta(α_damp * fit.α + (1-α_damp) * prev.α, α_damp * fit.β + (1-α_damp) * prev.β)`
with fallbacks to previous or default Beta(2,2) if computation fails.
"""
function safe_damped_beta(fit::Beta, prev::Beta, α_damp::Float64)
    fit_α, fit_β = params(fit)
    prev_α, prev_β = params(prev)

    # Check for invalid fitted parameters - use previous if fitted is bad
    if !isfinite(fit_α) || !isfinite(fit_β) || fit_α <= 0 || fit_β <= 0
        return prev  # Keep previous distribution
    end

    # Check for invalid previous parameters - use fitted if previous is bad
    if !isfinite(prev_α) || !isfinite(prev_β) || prev_α <= 0 || prev_β <= 0
        return fit  # Use fitted distribution
    end

    # Compute damped parameters
    new_α = α_damp * fit_α + (1 - α_damp) * prev_α
    new_β = α_damp * fit_β + (1 - α_damp) * prev_β

    # Final safety check
    if !isfinite(new_α) || !isfinite(new_β) || new_α <= 0 || new_β <= 0
        return prev  # Fallback to previous
    end

    return Beta(new_α, new_β)
end

"""
    fit_beta_weighted(x::Vector{Float64}, w::Vector{Float64};
                      prior_α::Float64=2.0, prior_β::Float64=2.0,
                      min_n_eff::Float64=5.0) -> Beta

Fit Beta distribution using weighted method of moments with regularization.

Uses effective sample size to determine shrinkage toward a Beta(2,2) prior,
preventing degenerate estimates when data is sparse or weights are concentrated.
When variance is too large for valid Beta parameters, shrinks toward 50% of
theoretical maximum variance.

# Arguments
- `x::Vector{Float64}`: Data values (should be in (0,1))
- `w::Vector{Float64}`: Weights (non-negative)

# Keywords
- `prior_α::Float64=2.0`: Prior Beta α parameter for shrinkage
- `prior_β::Float64=2.0`: Prior Beta β parameter for shrinkage
- `min_n_eff::Float64=5.0`: Minimum effective sample size before full shrinkage to prior

# Returns
- `Beta`: Fitted Beta distribution (regularized, never uniform fallback)
"""
function fit_beta_weighted(x::Vector{Float64}, w::Vector{Float64};
                           prior_α::Float64=2.0, prior_β::Float64=2.0,
                           min_n_eff::Float64=5.0)
    # Squeeze values away from boundaries to avoid numerical issues
    x = clamp.(x, 1e-10, 1.0 - 1e-10)

    # Handle NaN/Inf in weights - replace with 0
    w = replace(w, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
    w = clamp.(w, 0.0, 1.0)

    # Normalize weights
    w_sum = sum(w)
    if !isfinite(w_sum) || w_sum ≤ 0
        # No valid weights - return prior
        return Beta(prior_α, prior_β)
    end
    w_norm = w ./ w_sum

    # Effective sample size (Kish's formula)
    n_eff = w_sum^2 / sum(w.^2)
    if !isfinite(n_eff)
        n_eff = 0.0
    end

    # Compute shrinkage factor based on effective sample size
    # shrinkage = 1 means use data, shrinkage = 0 means use prior
    shrinkage = clamp(n_eff / (n_eff + min_n_eff), 0.0, 1.0)

    # Weighted mean and variance
    μ_data = sum(w_norm .* x)
    σ²_data = sum(w_norm .* (x .- μ_data).^2)

    # Check for NaN in computed values
    if !isfinite(μ_data)
        μ_data = 0.5  # Neutral mean
    end
    if !isfinite(σ²_data)
        σ²_data = 0.0
    end

    # Clamp μ to valid range
    μ_data = clamp(μ_data, 0.01, 0.99)

    # Theoretical maximum variance for Beta: μ(1-μ)
    max_var = μ_data * (1 - μ_data)

    # If variance is too large (≥ max), shrink to 50% of max
    if σ²_data >= max_var
        σ²_data = 0.5 * max_var
    end

    # If variance is too small or zero, use prior-implied variance
    prior_mean = prior_α / (prior_α + prior_β)
    prior_var = (prior_α * prior_β) / ((prior_α + prior_β)^2 * (prior_α + prior_β + 1))
    if σ²_data ≤ 1e-10
        σ²_data = prior_var
    end

    # Shrink mean toward prior mean
    μ = shrinkage * μ_data + (1 - shrinkage) * prior_mean

    # Shrink variance toward prior variance
    σ² = shrinkage * σ²_data + (1 - shrinkage) * prior_var

    # Recompute max_var for shrunk mean and ensure valid variance
    max_var_shrunk = μ * (1 - μ)
    σ² = min(σ², 0.99 * max_var_shrunk)  # Ensure valid for method of moments
    σ² = max(σ², 1e-10)  # Ensure positive

    # Method of moments estimators
    common = μ * (1 - μ) / σ² - 1
    α = μ * common
    β = (1 - μ) * common

    # Ensure valid parameters (minimum 0.1 for numerical stability)
    α = max(α, 0.1)
    β = max(β, 0.1)

    # Final safety check - fallback to prior if still invalid
    if !isfinite(α) || !isfinite(β)
        return Beta(prior_α, prior_β)
    end

    return Beta(α, β)
end

"""
    fit_beta_safe(x::AbstractVector{<:Real}) -> Beta

Fit Beta distribution with safe handling of edge cases.
Uses method of moments with fallbacks for low variance or boundary values.

# Arguments
- `x::AbstractVector{<:Real}`: Data values (should be in (0,1))

# Returns
- `Beta`: Fitted Beta distribution (falls back to Beta(2,2) if fitting fails)
"""
function fit_beta_safe(x::AbstractVector{<:Real})
    # Filter out NaN and Inf values first
    x_valid = filter(isfinite, x)

    if length(x_valid) < 2
        return Beta(2.0, 2.0)  # Safe fallback
    end

    # Squeeze values away from boundaries
    x_safe = clamp.(x_valid, 1e-10, 1.0 - 1e-10)

    n = length(x_safe)
    if n < 2
        return Beta(2.0, 2.0)  # Safe fallback
    end

    # Compute mean and variance
    μ = mean(x_safe)
    σ² = var(x_safe)

    # Check for NaN in computed statistics
    if !isfinite(μ) || !isfinite(σ²)
        return Beta(2.0, 2.0)  # Safe fallback
    end

    # Clamp μ to valid range
    μ = clamp(μ, 0.01, 0.99)

    # Check for valid variance (must be positive and less than theoretical maximum)
    if σ² <= 1e-10 || σ² >= μ * (1 - μ)
        # Low or invalid variance: return distribution centered at mean with moderate spread
        # Use a Beta with mode at μ and reasonable concentration
        α = 2.0 + 10.0 * μ
        β = 2.0 + 10.0 * (1 - μ)
        return Beta(α, β)
    end

    # Method of moments estimators
    common = μ * (1 - μ) / σ² - 1
    α = μ * common
    β = (1 - μ) * common

    # Ensure valid parameters (minimum 0.1 for numerical stability)
    α = max(α, 0.1)
    β = max(β, 0.1)

    # Final check for NaN/Inf
    if !isfinite(α) || !isfinite(β)
        return Beta(2.0, 2.0)  # Safe fallback
    end

    return Beta(α, β)
end

"""
    fit_copula_weighted(p::PosteriorProbabilityTriplet, w::Vector{Float64};
                        n_eff_threshold=50.0, n_resample=10_000)

Fit copula using importance-weighted pseudo-observations via resampling.

# Arguments
- `p::PosteriorProbabilityTriplet`: Posterior probabilities
- `w::Vector{Float64}`: Weights for each observation
- `n_eff_threshold::Float64=50.0`: Minimum effective sample size to attempt fitting
- `n_resample::Int=10_000`: Number of resamples for weighted fitting

# Returns
- `Copula` or `nothing`: Fitted copula, or nothing if effective sample size is too small
"""
function fit_copula_weighted(p::PosteriorProbabilityTriplet, w::Vector{Float64};
                              n_eff_threshold::Float64 = 50.0,
                              n_resample::Int = 10_000,
                              copula_family::Union{Nothing, Type} = nothing)
    # Handle NaN/Inf in weights - replace with 0
    w = replace(w, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
    w = clamp.(w, 0.0, 1.0)

    # Check effective sample size
    w_sum = sum(w)
    if !isfinite(w_sum) || w_sum ≤ 0
        return nothing
    end
    n_eff = w_sum^2 / sum(w.^2)
    if !isfinite(n_eff) || n_eff < n_eff_threshold
        return nothing  # Signal to keep previous copula
    end

    # Weighted resampling approach (more stable than direct weighted MLE)
    w_normalized = w ./ w_sum
    # Final check for any remaining NaN (shouldn't happen but be safe)
    if any(isnan, w_normalized)
        return nothing
    end
    idx = sample(1:length(w), Weights(w_normalized), n_resample; replace=true)

    p_resampled = PosteriorProbabilityTriplet(
        p.enrichment[idx],
        p.correlation[idx],
        p.detection[idx]
    )

    try
        if copula_family !== nothing
            return fit_copula(copula_family, p_resampled)
        else
            return fit_copula(p_resampled)
        end
    catch e
        @warn "Weighted copula fitting failed: $e"
        return nothing
    end
end

"""
    computeH0_BayesFactors(data; kwargs...) -> DataFrame

Estimate the null distribution of Bayes Factors for protein interaction data using permutation-based resampling.

This function:
1. Randomly permutes the sample/control labels in the input `data` to destroy any true signal (generating a null distribution).
2. Computes Bayes Factors under three different statistical models:
   - Detection Bayes Factors (`bf_detected`) using a Beta-Bernoulli model.
   - Correlation Bayes Factors (`bf_correlation`) using a regression model.
   - Enrichment Bayes Factors (`bf_enrichment`) using a hierarchical Bayesian model.
3. Returns the results in a tidy `DataFrame` and optionally writes it to an Excel file.
s
# Arguments
- `data::InteractionData`: The input dataset containing sample and control measurements.

# Keywords
- `n_controls::Int=0`: Number of controls
- `n_samples::Int=0`: Number of samples
- `refID::Int=1`: ID of the reference (bait) protein.
- `n::Int=10_000`: The total number of permuted protein measurements to generate for building the null distribution. A larger `n` results in a more stable estimation at the cost of longer computation time.

# Returns
- `H0::DataFrame` :
    A `DataFrame` with columns:
    - `:bf_enrichment` — Bayes Factors from the hierarchical model (HBM).
    - `:bf_correlation` — Bayes Factors from the regression model.
    - `:bf_detected` — Bayes Factors from the Beta-Bernoulli detection model.

# Notes
- The label permutation is **done once** at the start (not separately per model).
- Bayes Factors are computed in parallel across proteins using multiple threads.
"""
function computeH0_BayesFactors(data; n_controls = 0, n_samples = 0, refID = 1, n::Int = 8_000,
    regression_likelihood::Symbol = :robust_t,
    student_t_nu::Float64 = 5.0,
    hbm_iterations::Int = 75,
    regression_iterations::Int = 50,
    regression_bf_threshold::Float64 = 0.1,
    h0_cache_file::String = "",
    jzs_r_scale::Float64 = 0.0,
    regression_min_posterior_var::Float64 = 0.0,
    detected_mask::Union{BitVector, Nothing} = nothing)
    n_proteins = length(getIDs(data))

    # ------------------------------------ #
    # permute data
    # ------------------------------------ #
    n_datasets = div(n, n_proteins)
    permuted_data = permuteLabels(data, refID)

    for _ in 2:n_datasets
        permuted_data = vcat(permuted_data, permuteLabels(data))
    end

    n_proteins = length(getIDs(permuted_data))

    # Non-detected proteins have all-missing values in both samples and controls.
    # Permuting labels preserves all-missing rows, so permuted_data.detected correctly
    # identifies non-detected proteins without needing the original detected_mask.
    perm_detected = permuted_data.detected

    # ------------------------------------ #
    # Beta-Bernoulli model
    # ------------------------------------ #
    bf_detected = zeros(Float64, n_proteins)

    p = Progress(
        n_proteins, desc="Step 1: Computing Beta-Bernoulli Bayes factors...",
        showspeed=true,
        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
        barlen = 20
        )

    Threads.@threads for i in 1:n_proteins
        if !perm_detected[i]
            bf_detected[i] = 1.0
            ProgressMeter.next!(p)
            continue
        end
        b, _, _ = betabernoulli(permuted_data, i, n_controls, n_samples)
        if ismissing(b)
            bf_detected[i] = 1.0
        else
            bf_detected[i] = b
        end
        ProgressMeter.next!(p)
    end
    finish!(p)

    # ------------------------------------ #
    # Precompute priors for H0 (mirrors analyse() — avoids broken per-protein prior computation)
    # Hyperparameters (τ0, μ0) and data structure are invariant to label permutation,
    # so we compute them from the original `data` where τ0 is known to be stable.
    # ------------------------------------ #
    τ_dist = τ0(data)
    a_0_h0, b_0_h0 = τ_dist.α, τ_dist.θ
    μ_0_h0, σ_0_h0 = μ0(data)

    robust_tau_base_h0 = regression_likelihood == :robust_t ? estimate_regression_tau_base(data, refID) : NaN

    cached_hbm_prior_h0 = precompute_enrichment_prior(data; μ_0=μ_0_h0, σ_0=σ_0_h0, a_0=a_0_h0, b_0=b_0_h0)
    if getNoProtocols(data) == 1
        cached_regression_prior_h0 = if regression_likelihood == :robust_t
            if jzs_r_scale > 0.0
                precompute_regression_one_protocol_robust_jzs_prior(data, refID, μ_0_h0, σ_0_h0; nu=student_t_nu, τ_base=robust_tau_base_h0, jzs_r_scale=jzs_r_scale)
            else
                precompute_regression_one_protocol_robust_prior(data, refID, μ_0_h0, σ_0_h0; nu=student_t_nu, τ_base=robust_tau_base_h0)
            end
        else
            precompute_regression_one_protocol_prior(data, refID, μ_0_h0, σ_0_h0)
        end
    else
        cached_regression_prior_h0 = if regression_likelihood == :robust_t
            if jzs_r_scale > 0.0
                precompute_regression_multi_protocol_robust_jzs_prior(data, refID, μ_0_h0, σ_0_h0; nu=student_t_nu, τ_base=robust_tau_base_h0, jzs_r_scale=jzs_r_scale)
            else
                precompute_regression_multi_protocol_robust_prior(data, refID, μ_0_h0, σ_0_h0; nu=student_t_nu, τ_base=robust_tau_base_h0)
            end
        else
            precompute_regression_multi_protocol_prior(data, refID, μ_0_h0, σ_0_h0)
        end
    end

    # ------------------------------------ #
    # hierarchical & regression model
    # ------------------------------------ #
    bf_correlation = zeros(Float64, n_proteins)
    bf_enrichment = zeros(Float64, n_proteins)

    p = Progress(
        n_proteins, desc="Step 2: Computing hierarchical and regression Bayes factors...",
        showspeed=true,
        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
        barlen = 20, dt = 10
        )

    Threads.@threads for i in 1:n_proteins
        # Skip non-detected proteins (all-missing in samples — permutation preserves this)
        if !perm_detected[i]
            ProgressMeter.next!(p)
            continue
        end

        # check that the protein is not the bait protein
        protein_data   = getProteinData(permuted_data, i)
        reference_data = getProteinData(permuted_data, refID)

        if reference_data == protein_data
            ProgressMeter.next!(p)
            continue
        end

        try
            result = main(
                permuted_data, i, refID,
                plotHBMdists = false, plotlog2fc = false, plotregr = false,
                plotbayesrange = false, csv_file = nothing,
                writecsv = false, verbose = false, computeHBM = true,
                μ_0 = μ_0_h0, σ_0 = σ_0_h0, a_0 = a_0_h0, b_0 = b_0_h0,
                cached_hbm_prior = cached_hbm_prior_h0,
                cached_regression_prior = cached_regression_prior_h0,
                regression_likelihood = regression_likelihood,
                student_t_nu = student_t_nu,
                robust_tau_base = robust_tau_base_h0,
                hbm_iterations = hbm_iterations,
                regression_iterations = regression_iterations,
                h0_mode = true,
                regression_bf_threshold = regression_bf_threshold,
                jzs_r_scale = jzs_r_scale,
                global_tau_base = robust_tau_base_h0,
                regression_min_posterior_var = regression_min_posterior_var
            )
            bf_correlation[i] = result.bfRegression[1]
            bf_enrichment[i] = result.bfHBM[1]
        catch e
            @warn "H0 computation failed for permuted protein $i: $e"
        end
        ProgressMeter.next!(p)
    end
    finish!(p)

    n_successful = sum(bf_correlation .!= 0)
    @info "H0 computation: $n_successful / $n_proteins permuted proteins computed successfully"
    if n_successful == 0
        error("""
        computeH0_BayesFactors: All permuted protein computations failed (bf_correlation = 0 for all proteins).
        This usually means the 'main()' function throws for the permuted data.
        Run with verbose=true or increase the log level to :Debug to see individual errors.
        The H0 file has NOT been written. Please investigate the data and retry.
        """)
    end

    # ------------------------------------ #
    # write to file
    # ------------------------------------ #
    H0 = DataFrame(
        bf_enrichment  = bf_enrichment,
        bf_correlation = bf_correlation,
        bf_detected    = bf_detected
    )

    # delete invalid rows:
    # 1. all rows where bf_correlation == 0 (as they could not be computed)
    # 2. the bait protein as it will always show strong correlation (correlation against itself)
    H0 = H0[H0.bf_correlation .!= 0, :]
    H0 = H0[setdiff(1:size(H0,1), refID), :]

    if nrow(H0) < 100
        @warn "H0 DataFrame has only $(nrow(H0)) rows after filtering. Results may be unreliable."
    end

    # Note: H0Cache (log-BF format) is saved by precompute_h0, not here.
    # computeH0_BayesFactors returns the raw DataFrame for backward compat.

    return H0
end


######################################################
# Copula fitting logic
######################################################

"""
    copula_nparams(cop_type) -> Int

Return number of parameters for a copula family (3-dimensional case).

# Arguments
- `cop_type::Type`: Copula type from Copulas.jl

# Returns
- `Int`: Number of parameters in the copula family
"""
function copula_nparams(cop_type::Type)
    # 3-dimensional copulas
    if cop_type <: ClaytonCopula
        return 1  # θ
    elseif cop_type <: FrankCopula
        return 1  # θ
    elseif cop_type <: GumbelCopula
        return 1  # θ
    elseif cop_type <: JoeCopula
        return 1  # θ
    elseif cop_type <: GaussianCopula
        return 3  # Correlation matrix: ρ₁₂, ρ₁₃, ρ₂₃
    elseif cop_type <: EmpiricalCopula
        return 0  # Non-parametric
    else
        return 1  # Default assumption
    end
end

"""
    compare_copulas(p::EvidenceTriplet; criterion::Symbol=:BIC)

Fits multiple copulas to posterior probabilities and compares their fit
using the specified criterion. Returns a sorted DataFrame.

If `p` is a `BayesFactorTriplet`, it is converted to posterior probabilities
(assuming prior odds = 1).

# Arguments
- `p::EvidenceTriplet`: Vector of posterior probabilities or Bayes Factors

# Keywords
- `criterion::Symbol=:BIC`: Selection criterion (`:BIC`, `:AIC`, or `:loglik`)

# Returns
- `DataFrame`: Sorted comparison with columns `Family`, `LogLik`, `BIC`, `AIC`
"""
function compare_copulas(p::EvidenceTriplet; criterion::Symbol = :BIC)
    if p isa BayesFactorTriplet
        p = posterior_probability_from_bayes_factor(p)
    end

    if isa(p, PosteriorProbabilityTriplet) == false
        throw(ArgumentError("p must be of type EvidenceTriplet or BayesFactorTriplet."))
    end

    u = hcat(p.enrichment, p.correlation, p.detection)'
    n = size(u, 2)  # Sample size
    error_only_logger = MinLevelLogger(current_logger(), Logging.Error);

    # Fit each copula and record log-likelihood, BIC, AIC
    results = DataFrame(Family=String[], LogLik=Float64[], BIC=Float64[], AIC=Float64[])
    for (copula_name, fam) ∈ COPULA_TYPES
        try
            with_logger(error_only_logger) do
                cop = fit(fam, u)
                ll = loglikelihood(cop, u)
                k = copula_nparams(fam)

                bic = -2 * ll + k * log(n)
                aic = -2 * ll + 2 * k

                push!(results, (copula_name, ll, bic, aic))
            end
        catch e
        end
    end

    # Sort by selected criterion (lower is better for BIC/AIC, higher for loglik)
    if criterion == :BIC
        sort!(results, :BIC)
    elseif criterion == :AIC
        sort!(results, :AIC)
    else  # :loglik or any other
        sort!(results, :LogLik, rev=true)
    end

    return results
end

function fit_copula(copula, p::EvidenceTriplet)
    if p isa BayesFactorTriplet
        p = posterior_probability_from_bayes_factor(p)
    end

    if isa(p, PosteriorProbabilityTriplet) == false
        throw(ArgumentError("p must be of type EvidenceTriplet or BayesFactorTriplet."))
    end

    u = hcat(p.enrichment, p.correlation, p.detection)'

    # Clamp values to strictly within (0,1) to prevent singularities in copula fitting
    # Some copulas (Clayton, Gumbel, Joe) have singularities at exactly 0 or 1
    clamp!(u, nextfloat(0.0), prevfloat(1.0))

    return fit(copula, u)
end

"""
    fit_copula(p::EvidenceTriplet; searchBestCopula=true, copula=FrankCopula, criterion=:BIC)

Fit a copula to the posterior probability triplet.

# Arguments
- `p::EvidenceTriplet`: Evidence triplet (Bayes factors or posterior probabilities)

# Keywords
- `searchBestCopula::Bool=true`: If true, select best copula family automatically
- `copula=FrankCopula`: Copula type to use if `searchBestCopula=false`
- `criterion::Symbol=:BIC`: Selection criterion when searching (`:BIC`, `:AIC`, `:loglik`)

# Returns
- Fitted copula object
"""
function fit_copula(p::EvidenceTriplet; searchBestCopula = true, copula = FrankCopula, criterion::Symbol = :BIC)
    # If searchBestCopula = false, use copula as input
    if searchBestCopula == false
        return fit_copula(copula, p)
    end

    compared_copulas = compare_copulas(p; criterion=criterion)
    best_copula_name = compared_copulas[1, :Family]

    if haskey(COPULA_TYPES, best_copula_name)
        SelectedCopula = COPULA_TYPES[best_copula_name]
        return fit_copula(SelectedCopula, p)
    else
        @error "Best copula '$best_copula_name' not found in registry."
        return nothing
    end
end

######################################################
# Joint Bayes Factors modelling
######################################################

"""
    combined_BF(bf::BayesFactorTriplet, refID::Int64; kwargs...)

Combine individual Bayes factors using copula-based mixture model.

# Arguments
- `bf::BayesFactorTriplet`: Bayes factors from the three models
- `refID::Int64`: Index of the reference (bait) protein

# Keywords
- `H0_file::String`: Path to legacy XLSX H0 file (fallback, read-only)
- `h0_cache_file::String=""`: Path to JLD2 H0 cache (preferred)
- `max_iter::Int=1000`: Maximum EM iterations
- `init_π0::Float64=0.80`: Initial π₀ (null proportion)
- `prior::Union{Symbol, NamedTuple}=:default`: Prior for π₁ (:APMS, :BioID, :TurboID, :default, :permissive, :stringent, :empirical_bayes, or custom (α=, β=))
- `n_restarts::Int=20`: Number of EM restarts (set to 1 to disable)
- `copula_criterion::Symbol=:BIC`: Copula selection criterion (:BIC, :AIC, :loglik)
- `h1_refitting::Bool=true`: Enable weighted H1 updates in M-step
- `burn_in::Int=10`: Number of iterations before starting H1 re-fitting
- `use_acceleration::Bool=true`: Enable SQUAREM acceleration for faster convergence
- `verbose::Bool=true`: Print progress information

# Returns
- `CombinedBayesResult`: Combined Bayes factors and posterior probabilities
"""

# ---- Tier 1b: H0 goodness-of-fit diagnostic helpers ---- #

# Kolmogorov-Smirnov statistic between an empirical vector and a fitted distribution.
function _ks_statistic(data::AbstractVector{<:Real}, dist)
    sorted = sort(data)
    n = length(sorted)
    D = 0.0
    for (i, x) in enumerate(sorted)
        D = max(D, abs(i / n - cdf(dist, x)), abs((i - 1) / n - cdf(dist, x)))
    end
    return D
end

"""
    _reconstruct_h0_data(bf::BayesFactorTriplet, H0_file::String; h0_cache_file::String="") -> PosteriorProbabilityTriplet

Reconstruct the H0 Bayes factor data for diagnostic plotting.
Loads H0 from legacy XLSX, augments with weak-evidence proteins, returns BayesFactorTriplet.

Note: This is a backward-compat helper for diagnostic plots. The main combination pathway
now uses log-BF scale via precompute_h0 with phase1_result.
"""
function _reconstruct_h0_data(bf::BayesFactorTriplet, H0_file::String; h0_cache_file::String="")
    H0 = nothing
    if !isempty(H0_file) && isfile(H0_file)
        H0 = DataFrame(readtable(H0_file, "Sheet1"))
    end
    if isnothing(H0) || nrow(H0) == 0
        @warn "_reconstruct_h0_data: No H0 data found, returning empty BayesFactorTriplet"
        return BayesFactorTriplet(Float64[], Float64[], Float64[])
    end

    bf_H0 = BayesFactorTriplet(
        Vector{Float64}(H0.bf_enrichment),
        Vector{Float64}(H0.bf_correlation),
        Vector{Float64}(H0.bf_detected)
    )
    idx_H0 = findall(i -> bf.enrichment[i] <= 1.0 && bf.correlation[i] <= 1.0, 1:length(bf.enrichment))
    append!(bf_H0.enrichment,  bf.enrichment[idx_H0])
    append!(bf_H0.correlation, bf.correlation[idx_H0])
    append!(bf_H0.detection,   bf.detection[idx_H0])
    return bf_H0
end

"""
    extract_marginals(joint::SklarDist) -> Tuple

Extract the marginal distributions from a fitted SklarDist joint distribution.
Returns a tuple of the marginal distributions (enrichment, correlation, detection).
Works for both H0 and H1 joint distributions.
"""
function extract_marginals(joint)
    return joint.m
end

"""
    extract_h0_marginals(joint_H0) -> Tuple

Deprecated: use `extract_marginals` instead.
"""
extract_h0_marginals(joint_H0) = extract_marginals(joint_H0)

"""
    GPDTailInfo

Stores fitted GPD (Generalized Pareto Distribution) for upper and lower tails
of a marginal distribution. Used to extend CDF/PDF beyond the bulk of the H0 fit,
preventing CDF saturation that causes the copula BF ceiling.

Only applied to enrichment and correlation marginals (NOT detection, which is discrete).
"""
struct GPDTailInfo
    gpd_upper::Union{Nothing, GeneralizedPareto}  # GPD for upper tail
    gpd_lower::Union{Nothing, GeneralizedPareto}  # GPD for lower tail (on negated exceedances)
    threshold_upper::Float64  # 90th percentile of H0 data
    threshold_lower::Float64  # 10th percentile of H0 data
end

"""
    fit_gpd_mom(exceedances::Vector{Float64}) -> Union{Nothing, GeneralizedPareto}

Fit a Generalized Pareto Distribution to exceedances using method of moments.
Returns nothing if fewer than 20 exceedances or degenerate data (zero variance).

Shape parameter xi is clamped to [-0.5, 0.5], scale sigma must be > 1e-6.
"""
function fit_gpd_mom(exceedances::Vector{Float64})
    length(exceedances) >= 20 || return nothing

    m = mean(exceedances)
    v = var(exceedances)
    v < 1e-12 && return nothing  # near-zero variance guard
    m <= 0.0 && return nothing   # exceedances must be positive

    xi_raw = 0.5 * (1.0 - m^2 / v)
    # Only use MoM result within valid range; outside, return nothing to avoid
    # fitting an inconsistent (xi, sigma) pair whose moments do not match the data.
    abs(xi_raw) > 0.5 && return nothing
    xi = xi_raw
    sigma = m * (1.0 - xi)
    sigma > 1e-6 || return nothing

    return GeneralizedPareto(0.0, sigma, xi)
end

"""
    _fit_gpd_tails(data::Vector{Float64}, marg) -> GPDTailInfo

Fit GPD tails to a marginal's H0 data at the 90th (upper) and 10th (lower) percentiles.
"""
function _fit_gpd_tails(data::Vector{Float64}, marg)
    thresh_upper = quantile(data, 0.9)
    thresh_lower = quantile(data, 0.1)

    upper_exc = Float64[x - thresh_upper for x in data if x > thresh_upper]
    lower_exc = Float64[thresh_lower - x for x in data if x < thresh_lower]

    gpd_upper = fit_gpd_mom(upper_exc)
    gpd_lower = fit_gpd_mom(lower_exc)

    return GPDTailInfo(gpd_upper, gpd_lower, thresh_upper, thresh_lower)
end

"""
    gpd_extended_cdf(marg, x, gpd_tail_info::Union{Nothing, GPDTailInfo}) -> Float64

Compute CDF using GPD tail extrapolation for values beyond the bulk distribution.
Falls back to clamped `cdf(marg, x)` when gpd_tail_info is nothing or GPD is not fitted.
"""
function gpd_extended_cdf(marg, x::Float64, gpd_tail_info::Union{Nothing, GPDTailInfo})
    eps_cdf = 1e-10
    if gpd_tail_info === nothing
        return clamp(cdf(marg, x), eps_cdf, 1.0 - eps_cdf)
    end

    tu = gpd_tail_info.threshold_upper
    tl = gpd_tail_info.threshold_lower

    if x > tu && gpd_tail_info.gpd_upper !== nothing
        # Upper tail: F(x) = F(tu) + (1 - F(tu)) * F_gpd(x - tu)
        F_thresh = cdf(marg, tu)
        F_gpd = cdf(gpd_tail_info.gpd_upper, x - tu)
        val = F_thresh + (1.0 - F_thresh) * F_gpd
        return clamp(val, eps_cdf, 1.0 - eps_cdf)
    elseif x < tl && gpd_tail_info.gpd_lower !== nothing
        # Lower tail: F(x) = F(tl) * (1 - F_gpd(tl - x))
        F_thresh = cdf(marg, tl)
        F_gpd = cdf(gpd_tail_info.gpd_lower, tl - x)
        val = F_thresh * (1.0 - F_gpd)
        return clamp(val, eps_cdf, 1.0 - eps_cdf)
    else
        # Bulk: standard CDF
        return clamp(cdf(marg, x), eps_cdf, 1.0 - eps_cdf)
    end
end

"""
    gpd_extended_logpdf(marg, x, gpd_tail_info::Union{Nothing, GPDTailInfo}) -> Float64

Compute log-PDF using GPD tail extrapolation for values beyond the bulk distribution.
Falls back to `logpdf(marg, x)` when gpd_tail_info is nothing or GPD not fitted.

For upper tail: `log(1 - F(tu)) + logpdf(gpd, x - tu)`
For lower tail: `log(F(tl)) + logpdf(gpd, tl - x)`
"""
function gpd_extended_logpdf(marg, x::Float64, gpd_tail_info::Union{Nothing, GPDTailInfo})
    if gpd_tail_info === nothing
        return logpdf(marg, x)
    end

    tu = gpd_tail_info.threshold_upper
    tl = gpd_tail_info.threshold_lower

    if x > tu && gpd_tail_info.gpd_upper !== nothing
        val = log(max(1.0 - cdf(marg, tu), 1e-300)) + logpdf(gpd_tail_info.gpd_upper, x - tu)
        return (isnan(val) || !isfinite(val)) ? logpdf(marg, x) : val
    elseif x < tl && gpd_tail_info.gpd_lower !== nothing
        val = log(max(cdf(marg, tl), 1e-300)) + logpdf(gpd_tail_info.gpd_lower, tl - x)
        return (isnan(val) || !isfinite(val)) ? logpdf(marg, x) : val
    else
        return logpdf(marg, x)
    end
end

"""
    PrecomputedH0

Contains log-BF vectors for H0 proteins, fitted Normal/LocationScale marginals,
the copula, KS diagnostics, and the full SklarDist joint distribution.

Pass to `combined_BF` via the `precomputed_h0` kwarg to avoid redundant
H0 computation when `combined_BF` is called multiple times.
"""
struct PrecomputedH0
    log_bf_h0::NamedTuple{(:enrichment, :correlation, :detection),
                          Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}
    marginals::Tuple{UnivariateDistribution, UnivariateDistribution, UnivariateDistribution}
    ks_results::NamedTuple{(:enrichment, :correlation, :detection),
                           Tuple{Float64, Float64, Float64}}
    marginal_upgraded::NamedTuple{(:enrichment, :correlation, :detection),
                                  Tuple{Bool, Bool, Bool}}
    copula_h0::Any  # Copula object from Copulas.jl
    copula_family::String
    joint_H0::SklarDist
    gpd_tails::NamedTuple{(:enrichment, :correlation), Tuple{Union{Nothing,GPDTailInfo}, Union{Nothing,GPDTailInfo}}}
end

"""
    precompute_h0(bf::BayesFactorTriplet, phase1_result::LatentClassResult; kwargs...) -> PrecomputedH0

Compute PrecomputedH0 from BayesFactorTriplet + Phase 1 EM assignments.

1. Log-transform BFs
2. Extract H0 indices from Phase 1 responsibilities via MAP
3. Fit Normal marginals on H0 data, with KS auto-upgrade to LocationScale(TDist)
4. Build pseudo-observations and fit best copula
5. Return PrecomputedH0

# Arguments
- `bf::BayesFactorTriplet`: Individual Bayes factors
- `phase1_result::LatentClassResult`: Phase 1 EM result with responsibilities

# Keywords
- `ks_threshold::Float64=0.15`: KS threshold for marginal auto-upgrade
- `copula_criterion::Symbol=:BIC`: Copula selection criterion
- `h0_cache_file::String=""`: Path to JLD2 cache (optional)
- `verbose::Bool=true`: Print diagnostic info
"""
function precompute_h0(bf::BayesFactorTriplet, phase1_result::LatentClassResult;
                        ks_threshold::Float64 = 0.15,
                        copula_criterion::Symbol = :BIC,
                        copula_family::Union{Nothing, Type} = nothing,
                        streams::AbstractVector{Symbol} = collect(CANONICAL_STREAMS),
                        h0_cache_file::String = "",
                        verbose::Bool = true)
    # ABL-P2: a copula needs >=2 evidence streams. This is the FIRST build site
    # reached on the combined_BF path; reject a degenerate (<2-stream) config loudly
    # before any pseudo-obs / fit work.
    active = _active_streams(streams)
    @assert length(active) >= 2 "copula needs >=2 evidence streams; got $(streams)"
    eps_bf = 1e-300
    log_bf_e = log.(max.(bf.enrichment, eps_bf))
    log_bf_c = log.(max.(bf.correlation, eps_bf))
    log_bf_d = log.(max.(bf.detection, eps_bf))
    n = length(log_bf_e)

    # Extract H0 indices from Phase 1 responsibilities via MAP
    resp = phase1_result.responsibilities
    if resp === nothing
        error("precompute_h0 requires Phase 1 LatentClassResult with responsibilities matrix")
    end

    h0_idx = Int[]
    for i in 1:n
        k = argmax(@view resp[i, :])
        if k == 1
            push!(h0_idx, i)
        end
    end

    # Guard: if too few H0 proteins, use all non-H1 proteins
    if length(h0_idx) < 30
        @warn "Too few H0 proteins ($(length(h0_idx)) < 30); expanding to include agnostic proteins"
        h0_idx = Int[]
        for i in 1:n
            k = argmax(@view resp[i, :])
            if k != 3  # not H1
                push!(h0_idx, i)
            end
        end
    end

    verbose && @info "precompute_h0: $(length(h0_idx)) H0 proteins"

    # Fit Normal marginals on H0 data
    marg_e = fit(Normal, log_bf_e[h0_idx])
    marg_c = fit(Normal, log_bf_c[h0_idx])
    marg_d = fit(Normal, log_bf_d[h0_idx])

    # KS check + auto-upgrade per marginal
    marg_e, ks_e, upgraded_e = _fit_with_ks_check(log_bf_e[h0_idx], marg_e, ks_threshold)
    marg_c, ks_c, upgraded_c = _fit_with_ks_check(log_bf_c[h0_idx], marg_c, ks_threshold)
    marg_d, ks_d, upgraded_d = _fit_with_ks_check(log_bf_d[h0_idx], marg_d, ks_threshold)

    # Build DiscreteEmpirical for detection dimension from H0 proteins
    # Detection BFs have finite discrete support (count-based); DiscreteEmpirical captures this exactly
    disc_det_H0 = DiscreteEmpirical(log_bf_d[h0_idx])

    if verbose
        @info "H0 marginal fit (log-BF scale):" *
              " KS_e=$(round(ks_e, digits=4))$(upgraded_e ? "[TDist]" : "[Normal]")" *
              " KS_c=$(round(ks_c, digits=4))$(upgraded_c ? "[TDist]" : "[Normal]")" *
              " KS_d=$(round(ks_d, digits=4))$(upgraded_d ? "[TDist]" : "[Normal]")" *
              " (detection: DiscreteEmpirical with $(length(disc_det_H0.values)) support points)"
    end

    # Fit GPD tails for enrichment and correlation (NOT detection -- discrete distribution)
    gpd_e = _fit_gpd_tails(log_bf_e[h0_idx], marg_e)
    gpd_c = _fit_gpd_tails(log_bf_c[h0_idx], marg_c)
    gpd_tails_h0 = (enrichment=gpd_e, correlation=gpd_c)
    if verbose
        _gpd_status(name, info) = info.gpd_upper !== nothing ? "fitted" : "none"
        @info "GPD tails: enrichment=$(_gpd_status("e", gpd_e)), correlation=$(_gpd_status("c", gpd_c))"
    end

    # Build pseudo-observations (detection uses jittered CDF for discrete values)
    # u_h0 rows = ACTIVE dimensions in canonical order. With all three streams this is
    # the legacy 3 x n matrix (enrichment, correlation, detection); a dropped stream
    # removes exactly its row (ABL-P2). Detection uses the jittered discrete CDF.
    u_det_h0 = _jitter_discrete_to_uniform(disc_det_H0, log_bf_d[h0_idx])
    u_enrich_h0 = [gpd_extended_cdf(marg_e, x, gpd_e) for x in log_bf_e[h0_idx]]
    u_corr_h0 = [gpd_extended_cdf(marg_c, x, gpd_c) for x in log_bf_c[h0_idx]]
    _h0_rows = Dict(
        :enrichment  => reshape(u_enrich_h0, 1, :),
        :correlation => reshape(u_corr_h0, 1, :),
        :detection   => reshape(u_det_h0, 1, :),
    )
    u_h0 = reduce(vcat, [_h0_rows[s] for s in active])  # length(active) x n matrix

    # Marginal tuple matches the active rows (canonical order).
    _h0_margs = Dict(:enrichment => marg_e, :correlation => marg_c, :detection => marg_d)
    active_margs = Tuple(_h0_margs[s] for s in active)

    # Fit best copula (force_family = copula_family forces a fixed family; nothing = BIC)
    copula_h0, family_name = _fit_best_copula_logbf(u_h0; criterion=copula_criterion, force_family=copula_family)
    verbose && @info "H0 copula (log-BF): $family_name"

    # Build SklarDist (use Normal for detection for backward compat with SklarDist CDF evaluation)
    joint_H0 = SklarDist(copula_h0, active_margs)

    ks_results = (enrichment=ks_e, correlation=ks_c, detection=ks_d)
    marginal_upgraded = (enrichment=upgraded_e, correlation=upgraded_c, detection=upgraded_d)

    return PrecomputedH0(
        (enrichment=log_bf_e[h0_idx], correlation=log_bf_c[h0_idx], detection=log_bf_d[h0_idx]),
        (marg_e, marg_c, marg_d),
        ks_results,
        marginal_upgraded,
        copula_h0,
        family_name,
        joint_H0,
        gpd_tails_h0
    )
end

# Legacy 2-arg precompute_h0 for backward compat (deprecated)
function precompute_h0(bf::BayesFactorTriplet, H0_file::String; h0_cache_file::String="", verbose::Bool=true)
    @warn "precompute_h0(bf, H0_file) is deprecated. Use precompute_h0(bf, phase1_result) instead."
    # Cannot proceed without phase1_result -- error out
    error("precompute_h0 now requires a LatentClassResult (phase1_result) argument. " *
          "Run Phase 1 EM first via combined_BF_latent_class, then pass the result.")
end

"""
    _fit_normal_weighted(data, w; sigma_floor=0.01) -> Normal

Fit a Normal distribution using weighted data.
"""
function _fit_normal_weighted(data::AbstractVector{Float64}, w::AbstractVector{Float64};
                               sigma_floor::Float64 = 0.01)
    sw = sum(w)
    sw < 1e-10 && return Normal(0.0, 1.0)
    mu = sum(w .* data) / sw
    var_val = sum(w .* (data .- mu).^2) / sw
    sigma = max(sqrt(var_val), sigma_floor)
    return Normal(mu, sigma)
end

######################################################
# Weighted MLE helpers for H1 enrichment families
######################################################

"""
    _fit_lognormal_weighted(shifted_data, w) -> LogNormal

Analytical weighted MLE for LogNormal distribution.
`shifted_data` must be strictly positive (pre-shifted by JEFFREYS_SHIFT).
"""
function _fit_lognormal_weighted(shifted_data::AbstractVector{Float64}, w::AbstractVector{Float64})
    sw = sum(w)
    sw < 1e-10 && return LogNormal(0.5, 1.0)
    log_data = log.(max.(shifted_data, 1e-10))
    mu_w = sum(w .* log_data) / sw
    var_w = sum(w .* (log_data .- mu_w).^2) / sw
    mu_w  = clamp(mu_w, -2.0, 5.0)
    sigma_w = clamp(sqrt(max(var_w, 1e-6)), 0.05, 3.0)
    return LogNormal(mu_w, sigma_w)
end

"""
    _fit_weibull_weighted(shifted_data, w) -> Weibull

Optim L-BFGS weighted MLE for Weibull distribution.
`shifted_data` must be strictly positive (pre-shifted by JEFFREYS_SHIFT).
"""
function _fit_weibull_weighted(shifted_data::AbstractVector{Float64}, w::AbstractVector{Float64})
    sw = sum(w)
    sw < 1e-10 && return Weibull(1.5, 2.0)

    # Starting point: unweighted fit
    w0 = try
        wfit = Distributions.fit(Weibull, shifted_data)
        [log(shape(wfit)), log(scale(wfit))]
    catch
        [log(1.5), log(2.0)]
    end

    # Negative weighted log-likelihood (log-parameterized)
    function neg_wll(p)
        alpha = exp(p[1])  # shape
        theta = exp(p[2])  # scale
        ll = sum(w .* logpdf.(Weibull(alpha, theta), shifted_data))
        return isfinite(ll) ? -ll : 1e20
    end

    result = try
        Optim.optimize(neg_wll, w0, Optim.LBFGS(), Optim.Options(iterations=200, g_tol=1e-6))
    catch
        return Weibull(1.5, 2.0)
    end

    alpha = clamp(exp(result.minimizer[1]), 0.5, 20.0)
    theta = clamp(exp(result.minimizer[2]), 0.05, 20.0)
    return Weibull(alpha, theta)
end

"""
    _compute_weighted_bic(weighted_ll, n_params, w) -> Float64

Compute BIC using Kish effective sample size.
`BIC = -2 * weighted_ll + n_params * log(n_eff)` where `n_eff = (sum(w))^2 / sum(w^2)`.
"""
function _compute_weighted_bic(weighted_ll::Float64, n_params::Int, w::AbstractVector{Float64})
    n_eff = sum(w)^2 / sum(w .^ 2)
    return -2.0 * weighted_ll + n_params * log(max(n_eff, 1.0))
end

"""
    _select_h1_family_bic(shifted_data, w_above, current_family) -> (Symbol, Dict{Symbol,Float64})

Compare Gamma, LogNormal, and Weibull on `shifted_data` (already shifted by JEFFREYS_SHIFT)
using BIC with responsibility weights `w_above`. Returns the best family and BIC table.

Requires `n_eff >= 5.0`; otherwise returns `(current_family, all-Inf dict)`.
"""
function _select_h1_family_bic(shifted_data::AbstractVector{Float64},
                                w_above::AbstractVector{Float64},
                                current_family::Symbol)
    n_eff = sum(w_above)^2 / max(sum(w_above .^ 2), 1e-20)
    inf_bic = Dict{Symbol,Float64}(:gamma => Inf, :lognormal => Inf, :weibull => Inf)

    if n_eff < 5.0 || length(shifted_data) < 5
        return (current_family, inf_bic)
    end

    bic_table = Dict{Symbol,Float64}()

    # Gamma (2 params)
    bic_table[:gamma] = try
        gfit = Distributions.fit(Gamma, shifted_data, w_above)
        alpha_g = clamp(shape(gfit), 0.5, 50.0)
        theta_g = clamp(scale(gfit), 0.05, 20.0)
        wll = sum(w_above .* logpdf.(Gamma(alpha_g, theta_g), shifted_data))
        _compute_weighted_bic(wll, 2, w_above)
    catch
        Inf
    end

    # LogNormal (2 params)
    bic_table[:lognormal] = try
        lnfit = _fit_lognormal_weighted(shifted_data, w_above)
        wll = sum(w_above .* logpdf.(lnfit, shifted_data))
        _compute_weighted_bic(wll, 2, w_above)
    catch
        Inf
    end

    # Weibull (2 params)
    bic_table[:weibull] = try
        wfit = _fit_weibull_weighted(shifted_data, w_above)
        wll = sum(w_above .* logpdf.(wfit, shifted_data))
        _compute_weighted_bic(wll, 2, w_above)
    catch
        Inf
    end

    @debug "H1 family BIC scores" bic_table...

    # Select family with lowest BIC
    best_family = current_family
    best_bic = get(bic_table, current_family, Inf)
    for (fam, bic) in bic_table
        if isfinite(bic) && bic < best_bic
            best_bic = bic
            best_family = fam
        end
    end

    return (best_family, bic_table)
end

"""
    _reinit_h1_params_for_family(family, shifted_data, w_above) -> (Float64, Float64)

Fit the selected family to H1 pool data and return (param1, param2).
- Gamma: (shape, scale)
- LogNormal: (mu, sigma)
- Weibull: (shape, scale)
"""
function _reinit_h1_params_for_family(family::Symbol,
                                       shifted_data::AbstractVector{Float64},
                                       w_above::AbstractVector{Float64})
    try
        if family == :lognormal
            lnfit = _fit_lognormal_weighted(shifted_data, w_above)
            return (lnfit.μ, lnfit.σ)
        elseif family == :weibull
            wfit = _fit_weibull_weighted(shifted_data, w_above)
            return (shape(wfit), scale(wfit))
        else  # :gamma
            gfit = Distributions.fit(Gamma, shifted_data, w_above)
            return (clamp(shape(gfit), 0.5, 50.0), clamp(scale(gfit), 0.05, 20.0))
        end
    catch
        # Fallback defaults
        if family == :lognormal
            return (0.5, 1.0)
        elseif family == :weibull
            return (1.5, 2.0)
        else
            return (2.0, 2.0)
        end
    end
end

######################################################

"""
    _fit_shifted_h1_marginal(data; fallback_alpha=2.0, fallback_theta=2.0, family=:gamma)

Fit a LocationShifted{T} marginal to data (unweighted). Data values above
JEFFREYS_SHIFT are shifted and fit to the specified family distribution via MLE.

Supported families: `:gamma` (default), `:lognormal`, `:weibull`.
"""
function _fit_shifted_h1_marginal(data::AbstractVector{<:Real};
                                   fallback_alpha::Float64=2.0, fallback_theta::Float64=2.0,
                                   family::Symbol=:gamma)
    above_c = data .> JEFFREYS_SHIFT
    if count(above_c) >= 5
        shifted = max.(data[above_c] .- JEFFREYS_SHIFT, 1e-10)
        try
            if family == :lognormal
                return LocationShifted(_fit_lognormal_weighted(shifted, ones(length(shifted))), JEFFREYS_SHIFT)
            elseif family == :weibull
                return LocationShifted(_fit_weibull_weighted(shifted, ones(length(shifted))), JEFFREYS_SHIFT)
            else  # :gamma (default)
                gfit = Distributions.fit(Gamma, shifted)
                alpha_h1 = clamp(shape(gfit), 0.5, 50.0)
                theta_h1 = clamp(scale(gfit), 0.05, 20.0)
                return LocationShifted(Gamma(alpha_h1, theta_h1), JEFFREYS_SHIFT)
            end
        catch
            return LocationShifted(Gamma(fallback_alpha, fallback_theta), JEFFREYS_SHIFT)
        end
    else
        return LocationShifted(Gamma(fallback_alpha, fallback_theta), JEFFREYS_SHIFT)
    end
end

# Deprecated alias — calls the new canonical name
_fit_shifted_gamma_marginal(data::AbstractVector{<:Real}; fallback_alpha::Float64=2.0, fallback_theta::Float64=2.0) =
    _fit_shifted_h1_marginal(data; fallback_alpha=fallback_alpha, fallback_theta=fallback_theta, family=:gamma)

"""
    _fit_shifted_h1_marginal_weighted(data, weights; fallback_alpha=2.0, fallback_theta=2.0, family=:gamma)

Fit a LocationShifted{T} marginal to data with responsibility weights (M-step).

Supported families: `:gamma` (default), `:lognormal`, `:weibull`.
"""
function _fit_shifted_h1_marginal_weighted(data::AbstractVector{<:Real}, w::AbstractVector{Float64};
                                            fallback_alpha::Float64=2.0, fallback_theta::Float64=2.0,
                                            family::Symbol=:gamma)
    above_c_mask = data .> JEFFREYS_SHIFT
    shifted = max.(data[above_c_mask] .- JEFFREYS_SHIFT, 1e-10)
    w_shifted = w[above_c_mask]
    if sum(w_shifted) > 1e-10 && count(above_c_mask) >= 5
        try
            if family == :lognormal
                return LocationShifted(_fit_lognormal_weighted(shifted, w_shifted), JEFFREYS_SHIFT)
            elseif family == :weibull
                return LocationShifted(_fit_weibull_weighted(shifted, w_shifted), JEFFREYS_SHIFT)
            else  # :gamma (default)
                gfit = Distributions.fit(Gamma, shifted, w_shifted)
                alpha_h1 = clamp(shape(gfit), 0.5, 50.0)
                theta_h1 = clamp(scale(gfit), 0.05, 20.0)
                return LocationShifted(Gamma(alpha_h1, theta_h1), JEFFREYS_SHIFT)
            end
        catch
            return LocationShifted(Gamma(fallback_alpha, fallback_theta), JEFFREYS_SHIFT)
        end
    else
        return LocationShifted(Gamma(fallback_alpha, fallback_theta), JEFFREYS_SHIFT)
    end
end

# Deprecated alias — calls the new canonical name
_fit_shifted_gamma_marginal_weighted(data::AbstractVector{<:Real}, w::AbstractVector{Float64}; fallback_alpha::Float64=2.0, fallback_theta::Float64=2.0) =
    _fit_shifted_h1_marginal_weighted(data, w; fallback_alpha=fallback_alpha, fallback_theta=fallback_theta, family=:gamma)

function combined_BF(bf::BayesFactorTriplet, refID::Int64;
                     phase1_result::LatentClassResult,
                     ks_threshold::Float64 = 0.15,
                     copula_criterion::Symbol = :BIC,
                     n_restarts::Int = 20,
                     max_iter::Int = 200,
                     burn_in::Int = 10,
                     use_acceleration::Bool = true,
                     verbose::Bool = true,
                     precomputed_h0::Union{Nothing, PrecomputedH0} = nothing,
                     # Legacy kwargs (ignored with deprecation warning)
                     H0_file::String = "",
                     h0_cache_file::String = "",
                     prior::Union{Symbol, NamedTuple} = :default,
                     copula_family::Union{Nothing, Type} = nothing,
                     h1_copula_family::Union{Nothing, Type} = nothing,
                     streams::AbstractVector{Symbol} = collect(CANONICAL_STREAMS),
                     h1_refitting::Bool = true,
                     init_pi0::Float64 = 0.80,
                     copula_dirichlet_prior::Vector{Float64} = [5.0, 2.0, 1.0])

    # ABL-P2: active evidence streams in canonical order. With all three this is the
    # legacy 3-D copula (byte-identical); dropping a stream re-fits a lower-D copula.
    active = _active_streams(streams)
    @assert length(active) >= 2 "copula needs >=2 evidence streams; got $(streams)"
    d_active = length(active)
    # Per-stream log-BF vectors keyed by canonical symbol (restricted to active below).
    # 1. Log-transform BFs
    eps_bf = 1e-300
    log_bf_e = log.(max.(bf.enrichment, eps_bf))
    log_bf_c = log.(max.(bf.correlation, eps_bf))
    log_bf_d = log.(max.(bf.detection, eps_bf))
    n = length(log_bf_e)
    _logbf_by_stream = Dict(:enrichment => log_bf_e, :correlation => log_bf_c, :detection => log_bf_d)
    # log_bf_matrix rows follow the ACTIVE streams in canonical order (d_active x n).
    log_bf_matrix = reduce(vcat, [reshape(_logbf_by_stream[s], 1, :) for s in active])

    # 2. Extract Phase 1 MAP assignments
    resp = phase1_result.responsibilities
    if resp === nothing
        error("combined_BF requires Phase 1 LatentClassResult with responsibilities matrix")
    end
    h0_idx = Int[]; h1_idx = Int[]; ag_idx = Int[]
    for i in 1:n
        k = argmax(@view resp[i, :])
        if k == 1; push!(h0_idx, i)
        elseif k == 3; push!(h1_idx, i)
        else; push!(ag_idx, i)
        end
    end
    verbose && @info "Phase 1 assignments: H0=$(length(h0_idx)), agnostic=$(length(ag_idx)), H1=$(length(h1_idx))"

    # 3. H0 precomputation
    if precomputed_h0 !== nothing
        h0_data = precomputed_h0
    else
        h0_data = precompute_h0(bf, phase1_result;
                                 ks_threshold=ks_threshold,
                                 copula_criterion=copula_criterion,
                                 copula_family=copula_family,
                                 streams=streams,
                                 verbose=verbose)
    end
    joint_H0 = h0_data.joint_H0
    gpd_tails_em = hasfield(typeof(h0_data), :gpd_tails) ? h0_data.gpd_tails : nothing

    # 4. Initialize 3 components
    min_log_exp = -700.0
    max_log_exp = 700.0

    # H0 component: from precomputed
    cop_H0 = h0_data.copula_h0

    # Agnostic component: independence copula + Normal marginals near 0
    if length(ag_idx) >= 10
        ag_marg_e = fit(Normal, log_bf_e[ag_idx])
        ag_marg_c = fit(Normal, log_bf_c[ag_idx])
        ag_marg_d = fit(Normal, log_bf_d[ag_idx])
    else
        ag_marg_e = Normal(0.0, 0.5)
        ag_marg_c = Normal(0.0, 0.5)
        ag_marg_d = Normal(0.0, 0.5)
    end
    _ag_margs_by_stream = Dict(:enrichment => ag_marg_e, :correlation => ag_marg_c, :detection => ag_marg_d)
    cop_ag = GaussianCopula(1.0 * I(d_active))
    joint_ag = SklarDist(cop_ag, Tuple(_ag_margs_by_stream[s] for s in active))

    # H1 component: fit from H1-assigned proteins
    # H1 enrichment uses LocationShiftedGamma: zero density below JEFFREYS_SHIFT
    # Force-family override (ABL-P3): when an H1 family is forced AND there are enough
    # H1 proteins to fit marginals + the forced copula, take the full fitting path so the
    # forced family actually binds (otherwise the small-H1 branch hardcodes IndependentCopula).
    force_h1_path = h1_copula_family !== nothing && length(h1_idx) >= 5
    if length(h1_idx) < 50 && !force_h1_path
        verbose && length(h1_idx) > 0 && @warn "Few H1 proteins ($(length(h1_idx))); using independence copula for H1"
        h1_marg_e = _fit_shifted_gamma_marginal(log_bf_e[h1_idx])
        h1_marg_c = length(h1_idx) >= 5 ? fit(Normal, log_bf_c[h1_idx]) : Normal(1.0, 2.0)
        h1_marg_d = length(h1_idx) >= 5 ? fit(Normal, log_bf_d[h1_idx]) : Normal(1.0, 2.0)
        cop_H1 = GaussianCopula(1.0 * I(d_active))
        h1_family = "IndependentCopula"
    else
        # LocationShiftedGamma for enrichment H1
        h1_marg_e = _fit_shifted_gamma_marginal(log_bf_e[h1_idx])
        # Symmetric Normal for correlation and detection (with KS check)
        h1_marg_c = fit(Normal, log_bf_c[h1_idx])
        h1_marg_d = fit(Normal, log_bf_d[h1_idx])
        h1_marg_c, _, _ = _fit_with_ks_check(log_bf_c[h1_idx], h1_marg_c, ks_threshold)
        h1_marg_d, _, _ = _fit_with_ks_check(log_bf_d[h1_idx], h1_marg_d, ks_threshold)
        u_h1 = _build_pseudo_obs(log_bf_e[h1_idx], log_bf_c[h1_idx], log_bf_d[h1_idx],
                                  h1_marg_e, h1_marg_c, h1_marg_d; streams=streams)
        cop_H1, h1_family = _fit_best_copula_logbf(u_h1; criterion=copula_criterion, force_family=h1_copula_family)
    end
    _h1_margs_by_stream = Dict(:enrichment => h1_marg_e, :correlation => h1_marg_c, :detection => h1_marg_d)
    joint_H1 = SklarDist(cop_H1, Tuple(_h1_margs_by_stream[s] for s in active))
    verbose && @info "H1 copula: $(h1_family)"

    # 5. Mixing weights warm start from Phase 1
    pi_H0_init = phase1_result.mixing_weights[1]
    pi_ag_init = length(phase1_result.mixing_weights) >= 3 ? phase1_result.mixing_weights[2] : 0.15
    pi_H1_init = length(phase1_result.mixing_weights) >= 3 ? phase1_result.mixing_weights[3] : phase1_result.mixing_weights[end]
    dirichlet_prior = copula_dirichlet_prior

    # 6. Multi-restart EM loop
    best_ll = -Inf
    best_pi_H0 = pi_H0_init
    best_pi_ag = pi_ag_init
    best_pi_H1 = pi_H1_init
    best_joint_H0 = joint_H0
    best_joint_ag = joint_ag
    best_joint_H1 = joint_H1
    best_converged = false
    best_n_iter = 0

    # Pre-allocate diagnostics
    diag_restart = Int[]
    diag_init_pi0 = Float64[]
    diag_init_method = String[]
    diag_final_pi0 = Float64[]
    diag_final_pi1 = Float64[]
    diag_ll = Float64[]
    diag_iterations = Int[]
    diag_converged = Bool[]
    diag_status = String[]

    for restart in 1:n_restarts
        restart_start_time = time()
        # Initialize with different strategies
        if restart == 1
            # Strategy 0: Phase 1 warm-start
            pi_h0 = pi_H0_init; pi_ag_r = pi_ag_init; pi_h1 = pi_H1_init
            method_name = "phase1_warmstart"
            cur_joint_H0 = joint_H0
            cur_joint_ag = joint_ag
            cur_joint_H1 = joint_H1
        elseif restart <= 5
            # Strategy 1-4: Quantile-based shifts
            offsets = [0.05, -0.05, 0.10, -0.10]
            offset = offsets[restart - 1]
            pi_h0 = clamp(pi_H0_init + offset, 0.1, 0.95)
            pi_ag_r = clamp(pi_ag_init - offset * 0.5, 0.01, 0.5)
            pi_h1 = 1.0 - pi_h0 - pi_ag_r
            pi_h1 = max(pi_h1, 0.01)
            s = pi_h0 + pi_ag_r + pi_h1; pi_h0 /= s; pi_ag_r /= s; pi_h1 /= s
            method_name = "quantile_shift"
            cur_joint_H0 = joint_H0; cur_joint_ag = joint_ag; cur_joint_H1 = joint_H1
        else
            # Strategy 5+: Random perturbations
            pi_h0 = clamp(pi_H0_init + randn() * 0.1, 0.1, 0.95)
            pi_ag_r = clamp(pi_ag_init + randn() * 0.05, 0.01, 0.5)
            pi_h1 = 1.0 - pi_h0 - pi_ag_r
            pi_h1 = max(pi_h1, 0.01)
            s = pi_h0 + pi_ag_r + pi_h1; pi_h0 /= s; pi_ag_r /= s; pi_h1 /= s
            method_name = "random"
            cur_joint_H0 = joint_H0; cur_joint_ag = joint_ag; cur_joint_H1 = joint_H1
        end

        try
            converged_r = false
            n_iter_r = 0
            ll_r = -Inf
            copula_refitted = false
            patience_counter = 0
            prev_ll_restart = -Inf

            for iter in 1:max_iter
                # E-step: compute responsibilities via log-sum-exp
                ll_h0_v = _safe_logpdf_vec(cur_joint_H0, log_bf_matrix, min_log_exp, max_log_exp; gpd_tails=gpd_tails_em, streams=streams)
                ll_ag_v = _safe_logpdf_vec(cur_joint_ag, log_bf_matrix, min_log_exp, max_log_exp; gpd_tails=gpd_tails_em, streams=streams)
                ll_h1_v = _safe_logpdf_vec(cur_joint_H1, log_bf_matrix, min_log_exp, max_log_exp; gpd_tails=gpd_tails_em, streams=streams)

                # Exclude bait
                ll_h0_v[refID] = min_log_exp
                ll_ag_v[refID] = min_log_exp
                ll_h1_v[refID] = min_log_exp

                log_pi_h0 = log(max(pi_h0, 1e-300))
                log_pi_ag = log(max(pi_ag_r, 1e-300))
                log_pi_h1 = log(max(pi_h1, 1e-300))

                r_H0 = Vector{Float64}(undef, n)
                r_ag = Vector{Float64}(undef, n)
                r_H1 = Vector{Float64}(undef, n)
                total_ll = 0.0
                for j in 1:n
                    a = log_pi_h0 + ll_h0_v[j]
                    b = log_pi_ag + ll_ag_v[j]
                    c = log_pi_h1 + ll_h1_v[j]
                    mx = max(a, b, c)
                    denom = mx + log(exp(a - mx) + exp(b - mx) + exp(c - mx))
                    total_ll += denom
                    r_H0[j] = exp(a - denom)
                    r_ag[j] = exp(b - denom)
                    r_H1[j] = exp(c - denom)
                end

                # M-step: mixing weights with Dirichlet prior
                n_h0_w = sum(r_H0) + dirichlet_prior[1] - 1.0
                n_ag_w = sum(r_ag) + dirichlet_prior[2] - 1.0
                n_h1_w = sum(r_H1) + dirichlet_prior[3] - 1.0
                total_w = n_h0_w + n_ag_w + n_h1_w
                pi_h0_new = max(n_h0_w / total_w, 1e-6)
                pi_ag_new = max(n_ag_w / total_w, 1e-6)
                pi_h1_new = max(n_h1_w / total_w, 1e-6)
                s_w = pi_h0_new + pi_ag_new + pi_h1_new
                pi_h0_new /= s_w; pi_ag_new /= s_w; pi_h1_new /= s_w

                # M-step: refit Normal marginals per component (rows = active streams)
                margs_h0 = Tuple([_fit_normal_weighted(Vector(log_bf_matrix[d, :]), r_H0) for d in 1:d_active])
                margs_ag = Tuple([_fit_normal_weighted(Vector(log_bf_matrix[d, :]), r_ag) for d in 1:d_active])
                # H1: the enrichment row uses LocationShiftedGamma, the other active
                # streams use a symmetric Normal. Keyed by stream identity (active[d]),
                # NOT by absolute row index, so a dropped stream cannot misroute the
                # Gamma marginal onto correlation/detection.
                margs_h1_list = Any[nothing for _ in 1:d_active]
                for d in 1:d_active
                    data_d = Vector(log_bf_matrix[d, :])
                    if active[d] === :enrichment
                        # LocationShiftedGamma for enrichment H1
                        margs_h1_list[d] = _fit_shifted_gamma_marginal_weighted(data_d, r_H1)
                    else
                        margs_h1_list[d] = _fit_normal_weighted(data_d, r_H1)
                    end
                end
                margs_h1 = Tuple(margs_h1_list)

                cur_joint_H0 = SklarDist(cur_joint_H0.C, margs_h0)
                cur_joint_ag = SklarDist(cur_joint_ag.C, margs_ag)
                cur_joint_H1 = SklarDist(cur_joint_H1.C, margs_h1)

                # Copula BIC refit once after burn-in
                if iter == burn_in && !copula_refitted
                    copula_refitted = true
                    # Map active rows back to canonical (enrichment, correlation, detection)
                    # slots so _build_pseudo_obs receives each stream's vector/marginal in
                    # its positional slot. Absent streams get an empty vector + placeholder
                    # marginal that _build_pseudo_obs never touches (it only fills active rows).
                    _row_of = Dict(active[d] => d for d in 1:d_active)
                    _placeholder = Normal(0.0, 1.0)
                    _vec_for(s, idx) = haskey(_row_of, s) ? [log_bf_matrix[_row_of[s], j] for j in idx] : Float64[]
                    _marg_for(margs, s) = haskey(_row_of, s) ? margs[_row_of[s]] : _placeholder
                    # Refit H0 copula
                    u_h0_r = _build_pseudo_obs(
                        _vec_for(:enrichment, h0_idx),
                        _vec_for(:correlation, h0_idx),
                        _vec_for(:detection, h0_idx),
                        _marg_for(margs_h0, :enrichment), _marg_for(margs_h0, :correlation),
                        _marg_for(margs_h0, :detection); gpd_tails=gpd_tails_em, streams=streams)
                    try
                        new_cop_h0, _ = _fit_best_copula_logbf(u_h0_r; criterion=copula_criterion, force_family=copula_family)
                        cur_joint_H0 = SklarDist(new_cop_h0, margs_h0)
                    catch; end
                    # Refit H1 copula
                    if length(h1_idx) >= 50
                        u_h1_r = _build_pseudo_obs(
                            _vec_for(:enrichment, h1_idx),
                            _vec_for(:correlation, h1_idx),
                            _vec_for(:detection, h1_idx),
                            _marg_for(margs_h1, :enrichment), _marg_for(margs_h1, :correlation),
                            _marg_for(margs_h1, :detection); gpd_tails=gpd_tails_em, streams=streams)
                        try
                            new_cop_h1, _ = _fit_best_copula_logbf(u_h1_r; criterion=copula_criterion, force_family=h1_copula_family)
                            cur_joint_H1 = SklarDist(new_cop_h1, margs_h1)
                        catch; end
                    end
                    # Agnostic always independence copula
                end

                # Convergence check
                n_iter_r = iter
                if iter > 10 && isfinite(ll_r)
                    ll_change = abs(total_ll - ll_r) / max(abs(total_ll), 1.0)
                    if ll_change < 1e-6
                        converged_r = true
                        pi_h0 = pi_h0_new; pi_ag_r = pi_ag_new; pi_h1 = pi_h1_new
                        ll_r = total_ll
                        break
                    end
                end

                # Patience-based early stopping for stagnant restarts
                if iter > 10 && isfinite(total_ll) && isfinite(prev_ll_restart)
                    rel_improvement = (total_ll - prev_ll_restart) / max(abs(total_ll), 1.0)
                    if rel_improvement < 1e-4
                        patience_counter += 1
                        if patience_counter >= 20
                            # Stagnant — accept current state and move on
                            converged_r = true
                            pi_h0 = pi_h0_new; pi_ag_r = pi_ag_new; pi_h1 = pi_h1_new
                            ll_r = total_ll
                            break
                        end
                    else
                        patience_counter = 0
                    end
                end
                prev_ll_restart = total_ll

                # Restart quality gate: skip clearly worse restarts early
                if iter == 30 && isfinite(best_ll) && isfinite(total_ll) && (total_ll < best_ll - 50.0)
                    break
                end

                pi_h0 = pi_h0_new; pi_ag_r = pi_ag_new; pi_h1 = pi_h1_new
                ll_r = total_ll
            end

            # Record diagnostics
            restart_elapsed = time() - restart_start_time
            verbose && println("  Restart $restart: $(n_iter_r) iters, $(round(restart_elapsed, digits=1))s, LL=$(round(ll_r, digits=2)), $(converged_r ? "converged" : "not converged")")
            push!(diag_restart, restart)
            push!(diag_init_pi0, pi_H0_init)
            push!(diag_init_method, method_name)
            push!(diag_final_pi0, pi_h0)
            push!(diag_final_pi1, pi_h1)
            push!(diag_ll, ll_r)
            push!(diag_iterations, n_iter_r)
            push!(diag_converged, converged_r)
            push!(diag_status, "success")

            if isfinite(ll_r) && ll_r > best_ll
                best_ll = ll_r
                best_pi_H0 = pi_h0; best_pi_ag = pi_ag_r; best_pi_H1 = pi_h1
                best_joint_H0 = cur_joint_H0; best_joint_ag = cur_joint_ag; best_joint_H1 = cur_joint_H1
                best_converged = converged_r
                best_n_iter = n_iter_r
                verbose && println("  Restart $restart: new best log-likelihood = $(round(ll_r, digits=2))")
            end
        catch e
            push!(diag_restart, restart)
            push!(diag_init_pi0, pi_H0_init)
            push!(diag_init_method, method_name)
            push!(diag_final_pi0, NaN)
            push!(diag_final_pi1, NaN)
            push!(diag_ll, NaN)
            push!(diag_iterations, 0)
            push!(diag_converged, false)
            s = string(e)
            push!(diag_status, length(s) > 200 ? s[1:prevind(s, 200)] * "..." : s)
        end
    end

    # 7. Optional SQUAREM acceleration of best result
    # NOTE: em_fit_mixture_accelerated_logbf is hardcoded to the 3-D copula
    # (margs[1],margs[2],disc + log_bf_matrix[3,:]). Under an ABL-P2 stream drop the
    # copula is <3-D, so acceleration is skipped (the un-accelerated best restart is
    # already a valid fit). With all three streams this runs exactly as before
    # (byte-identical default path).
    if use_acceleration && !best_converged && d_active == 3
        verbose && @info "Running SQUAREM acceleration on best restart..."
        try
            acc_pi_H0, acc_pi_ag, acc_pi_H1, acc_jH0, acc_jag, acc_jH1, acc_conv, _ =
                em_fit_mixture_accelerated_logbf(
                    best_pi_H0, best_pi_ag, best_pi_H1,
                    best_joint_H0, best_joint_ag, best_joint_H1,
                    log_bf_matrix, refID;
                    max_iter=500, burn_in=5, dirichlet_prior=copula_dirichlet_prior,
                    verbose=verbose)
            acc_ll = compute_log_likelihood_3c(acc_pi_H0, acc_pi_ag, acc_pi_H1,
                acc_jH0, acc_jag, acc_jH1, log_bf_matrix, min_log_exp, max_log_exp)
            if isfinite(acc_ll) && acc_ll > best_ll
                best_pi_H0 = acc_pi_H0; best_pi_ag = acc_pi_ag; best_pi_H1 = acc_pi_H1
                best_joint_H0 = acc_jH0; best_joint_ag = acc_jag; best_joint_H1 = acc_jH1
                best_converged = acc_conv
                best_ll = acc_ll
                verbose && @info "SQUAREM improved log-likelihood to $(round(acc_ll, digits=2))"
            end
        catch e
            @warn "SQUAREM acceleration failed: $e"
        end
    end

    # 8. Compute combined BF from final EM result
    ll_H0_final = _safe_logpdf_vec(best_joint_H0, log_bf_matrix, min_log_exp, max_log_exp; gpd_tails=gpd_tails_em, streams=streams)
    ll_ag_final = _safe_logpdf_vec(best_joint_ag, log_bf_matrix, min_log_exp, max_log_exp; gpd_tails=gpd_tails_em, streams=streams)
    ll_H1_final = _safe_logpdf_vec(best_joint_H1, log_bf_matrix, min_log_exp, max_log_exp; gpd_tails=gpd_tails_em, streams=streams)

    # Bait protein: robust quantile-based values
    non_ref = setdiff(1:n, refID)
    finite_H0 = filter(isfinite, ll_H0_final[non_ref])
    finite_H1 = filter(isfinite, ll_H1_final[non_ref])
    ll_H0_final[refID] = isempty(finite_H0) ? min_log_exp : quantile(finite_H0, 0.001)
    ll_H1_final[refID] = isempty(finite_H1) ? max_log_exp : quantile(finite_H1, 0.999)
    ll_ag_final[refID] = min_log_exp

    # f_not_H1 = (pi_H0 * f_H0 + pi_ag * f_ag) / (pi_H0 + pi_ag)
    denom_weight = best_pi_H0 + best_pi_ag
    combined_bf = Vector{Float64}(undef, n)
    posterior_prob = Vector{Float64}(undef, n)
    for j in 1:n
        log_pi_h0 = log(max(best_pi_H0, 1e-300))
        log_pi_ag = log(max(best_pi_ag, 1e-300))
        log_pi_h1 = log(max(best_pi_H1, 1e-300))

        a = log_pi_h0 + ll_H0_final[j]
        b = log_pi_ag + ll_ag_final[j]
        c = log_pi_h1 + ll_H1_final[j]

        mx_ab = max(a, b)
        log_f_not_H1 = mx_ab + log(exp(a - mx_ab) + exp(b - mx_ab)) - log(denom_weight)
        log_BF_j = clamp(ll_H1_final[j] - log_f_not_H1, -46.0, 46.0)
        combined_bf[j] = exp(log_BF_j)

        # Posterior probability via softmax
        mx_all = max(a, b, c)
        total_density = mx_all + log(exp(a - mx_all) + exp(b - mx_all) + exp(c - mx_all))
        posterior_prob[j] = exp(c - total_density)
    end

    # 9. Post-hoc monotonicity constraint
    for i in eachindex(combined_bf)
        if bf.enrichment[i] < 1.0 && bf.correlation[i] < 1.0
            max_individual = max(bf.enrichment[i], bf.correlation[i], bf.detection[i])
            combined_bf[i] = min(combined_bf[i], max_individual)
        end
    end

    # 10. Agnostic shrinkage: smooth interpolation toward BF=1.0 proportional to agnostic weight
    # Recompute soft responsibilities from best restart's final log-likelihoods
    for j in 1:n
        a = log(max(best_pi_H0, 1e-300)) + ll_H0_final[j]
        b = log(max(best_pi_ag, 1e-300)) + ll_ag_final[j]
        c = log(max(best_pi_H1, 1e-300)) + ll_H1_final[j]
        mx = max(a, b, c)
        denom = mx + log(exp(a - mx) + exp(b - mx) + exp(c - mx))
        w_ag = exp(b - denom)  # posterior probability of agnostic class

        # Geometric shrinkage: log_bf_final = (1 - w_ag) * log_bf_copula
        log_bf = log(max(combined_bf[j], 1e-300))
        log_bf_shrunk = (1.0 - w_ag) * log_bf
        combined_bf[j] = exp(clamp(log_bf_shrunk, -46.0, 46.0))

        # Recompute posterior from shrunk BF
        prior_odds_val = best_pi_H1 / max(1.0 - best_pi_H1, 1e-300)
        odds = combined_bf[j] * prior_odds_val
        posterior_prob[j] = odds / (1.0 + odds)
    end

    # 11. Build EMResult with 3-component fields
    logs_df = DataFrame(iter=Int[best_n_iter], pi0=Float64[best_pi_H0], pi1=Float64[best_pi_H1], ll=Float64[best_ll])
    em_result = EMResult(best_pi_H0, best_pi_H1, best_pi_ag,
        best_joint_H1, best_joint_H0, best_joint_ag, logs_df, best_converged)

    # 12. Build em_diagnostics DataFrame
    em_diagnostics = DataFrame(
        restart = diag_restart,
        init_pi0 = diag_init_pi0,
        init_method = diag_init_method,
        final_pi0 = diag_final_pi0,
        final_pi1 = diag_final_pi1,
        log_likelihood = diag_ll,
        iterations = diag_iterations,
        converged = diag_converged,
        status = diag_status
    )

    verbose && @info "Best EM result: pi_H0=$(round(best_pi_H0, digits=3)), pi_ag=$(round(best_pi_ag, digits=3)), pi_H1=$(round(best_pi_H1, digits=3)), ll=$(round(best_ll, digits=2))"

    # 13. Return CombinedBayesResult with KS diagnostics
    return CombinedBayesResult(
        combined_bf,
        posterior_prob,
        best_joint_H0,
        best_joint_H1,
        em_result,
        em_diagnostics,
        h0_data.copula_family,
        h1_family,
        h0_data.ks_results,
        h0_data.marginal_upgraded,
        length(h0_idx),
        length(h1_idx),
        length(ag_idx)
    )
end

######################################################
# Mixture Estimation (EM algorithm)
######################################################

"""
    _kmeans2(X::AbstractMatrix{<:Real}; maxiter::Int=100) -> (assignments::Vector{Int},)

Minimal k=2 k-means clustering (Lloyd's algorithm).

Operates on a `d×n` matrix where each column is a data point.
Returns a NamedTuple with an `assignments` vector of 1s and 2s.

Used exclusively for EM initialisation — replaces the external Clustering.jl dependency.
"""
function _kmeans2(X::AbstractMatrix{<:Real}; maxiter::Int=100)
    d, n = size(X)
    # Initialise centroids via k-means++ seeding for deterministic spread
    i1 = rand(1:n)
    c1 = X[:, i1]
    # Pick second centroid proportional to squared distance from first
    dists = [sum((X[:, j] .- c1).^2) for j in 1:n]
    dsum = sum(dists)
    if dsum > 0
        cumprob = cumsum(dists ./ dsum)
        r = rand()
        i2 = searchsortedfirst(cumprob, r)
        i2 = clamp(i2, 1, n)
    else
        i2 = mod1(i1 + 1, n)  # degenerate case: all points identical
    end
    c2 = X[:, i2]

    assignments = zeros(Int, n)
    for _ in 1:maxiter
        # Assignment step
        changed = false
        for j in 1:n
            d1 = sum((X[:, j] .- c1).^2)
            d2 = sum((X[:, j] .- c2).^2)
            new_a = d1 <= d2 ? 1 : 2
            if new_a != assignments[j]
                assignments[j] = new_a
                changed = true
            end
        end
        !changed && break
        # Update step
        mask1 = assignments .== 1
        mask2 = assignments .== 2
        any(mask1) && (c1 = vec(mean(X[:, mask1], dims=2)))
        any(mask2) && (c2 = vec(mean(X[:, mask2], dims=2)))
    end
    return (assignments=assignments,)
end

"""
    get_H1_initialization_set(p::PosteriorProbabilityTriplet; method=:quantile) -> Vector{Int}

Return indices for H1 initialization using different strategies.

# Arguments
- `p::PosteriorProbabilityTriplet`: Posterior probabilities

# Keywords
- `method::Symbol=:quantile`: Initialization method (:quantile, :kmeans, :random_top20, :empirical_bayes)

# Returns
- `Vector{Int}`: Indices of proteins to use for H1 initialization

# Notes
The `:empirical_bayes` method uses the same selection strategy as `:quantile`, but is intended
for use with `em_restart_diagnostics` where π₀ is estimated via `estimate_prior_empirical_bayes`.
"""
function get_H1_initialization_set(p::PosteriorProbabilityTriplet; method::Symbol = :quantile)
    if method == :quantile
        # Use existing quantile-based initialization with NaN handling
        mean_strength = @. p.enrichment * p.correlation * p.detection
        mean_strength = replace(mean_strength, NaN => 0.0)
        valid_strengths = filter(isfinite, mean_strength)
        quantile_threshold = isempty(valid_strengths) ? 0.5 : quantile(valid_strengths, 0.95)
        return find_H1_initialization_set(mean_strength, p, quantile_threshold)

    elseif method == :kmeans
        # K-means with k=2, return cluster with higher mean
        # Replace NaN with 0.5 for k-means
        enrich_safe = replace(p.enrichment, NaN => 0.5)
        corr_safe = replace(p.correlation, NaN => 0.5)
        detect_safe = replace(p.detection, NaN => 0.5)
        X = hcat(enrich_safe, corr_safe, detect_safe)'  # 3×n matrix
        try
            result = _kmeans2(X; maxiter=100)

            # Find which cluster has higher average posterior
            cluster1_mask = result.assignments .== 1
            cluster2_mask = result.assignments .== 2

            if sum(cluster1_mask) == 0 || sum(cluster2_mask) == 0
                # Fallback if one cluster is empty
                return get_H1_initialization_set(p; method=:quantile)
            end

            cluster1_mean = mean((enrich_safe[cluster1_mask] .+
                                  corr_safe[cluster1_mask] .+
                                  detect_safe[cluster1_mask]) ./ 3)
            cluster2_mean = mean((enrich_safe[cluster2_mask] .+
                                  corr_safe[cluster2_mask] .+
                                  detect_safe[cluster2_mask]) ./ 3)

            h1_cluster = cluster1_mean > cluster2_mean ? 1 : 2
            idx = findall(result.assignments .== h1_cluster)

            # Ensure minimum number of proteins
            if length(idx) < 50
                return get_H1_initialization_set(p; method=:quantile)
            end
            return idx
        catch e
            @warn "K-means initialization failed: $e, falling back to quantile"
            return get_H1_initialization_set(p; method=:quantile)
        end

    elseif method == :random_top20
        # Random sample from top 20% by mean posterior with NaN handling
        enrich_safe = replace(p.enrichment, NaN => 0.5)
        corr_safe = replace(p.correlation, NaN => 0.5)
        detect_safe = replace(p.detection, NaN => 0.5)
        mean_p = (enrich_safe .+ corr_safe .+ detect_safe) ./ 3
        valid_mean_p = filter(isfinite, mean_p)
        threshold = isempty(valid_mean_p) ? 0.5 : quantile(valid_mean_p, 0.80)
        candidates = findall(mean_p .>= threshold)
        n_select = min(100, length(candidates))
        if n_select < 50
            return get_H1_initialization_set(p; method=:quantile)
        end
        return sample(candidates, n_select; replace=false)

    elseif method == :empirical_bayes
        # Use same selection as quantile for H1 set,
        # but π0 will be estimated separately via estimate_prior_empirical_bayes
        return get_H1_initialization_set(p; method=:quantile)

    else
        @warn "Unknown initialization method: $method, falling back to quantile"
        return get_H1_initialization_set(p; method=:quantile)
    end
end

function find_H1_initialization_set(
    mean_strength::Vector{Float64},
    p::PosteriorProbabilityTriplet,
    quantile_threshold::Float64 = 0.95,
    starting_threshold::Float64 = 0.999,
    min_proteins::Int64 = 50
    )

    iter = 1
    n_proteins = 0
    threshold = starting_threshold
    idx = Int64[]
    while n_proteins < min_proteins
        n_proteins, idx = get_H1_length(mean_strength, p, quantile_threshold, threshold)
        if n_proteins >= min_proteins
            @debug "A threshold of $threshold has been employed on the mean strength to build H1."
        end

        iter += 1
        threshold -= 0.001

        # Safety valve to prevent infinite loop
        if threshold < 0.5
            @warn "Could not find enough proteins for H1 initialization, using top $(length(idx)) proteins"
            break
        end
    end

    return idx
end

function get_H1_length(
    mean_strength::Vector{F}, 
    p::PosteriorProbabilityTriplet, 
    quantile_threshold::Float64 = 0.95, 
    threshold::Float64 = 0.95
    ) where {F<:AbstractFloat}

    # --- mean strength filter --- #
    # all protein with a mean strength about the quantile_threshold and the absolute threshold
    idx_init = findall(x -> x > threshold && x > quantile_threshold, mean_strength)
    
    # --- BF_detection filter --- #
    # remove proteins with evidence for a less frequent detection in samples with the bait protein
    # threshold is a BF_detection below 1/3
    negative_detection_evidence = findall(x -> x < 0.5/3, p.detection[idx_init])

    # add proteins with a BF_detection above 10 
    function _add_proteins(p::PosteriorProbabilityTriplet{F} ) 
        idx_detection = findall(x -> x > 10/11, p.detection)
        # remove proteins with negative enrichment evidence
        idx_enrichment = findall(x -> x < 0.7, p.enrichment[idx_detection])
        deleteat!(idx_detection, idx_enrichment)
        # remove proteins with negative correlation evidence
        idx_correlation = findall(x -> x < 0.7, p.correlation[idx_detection])
        deleteat!(idx_detection, idx_correlation)
        return idx_detection
    end

    added_proteins_detection = _add_proteins(p)
    idx_init = setdiff(idx_init, negative_detection_evidence)

    idx_init = union(idx_init, added_proteins_detection)

    return length(idx_init), idx_init
end

"""
    hasEMconverged(logs; tol=1e-4, window=5, π_tol=1e-4) -> Bool

Multi-criteria convergence detection for EM algorithm.

Checks three convergence criteria:
1. **Smoothed log-likelihood change**: Compares window-averaged log-likelihoods
2. **Parameter stability**: Checks if π₁ range over window is below threshold
3. **Oscillation detection**: Detects small-amplitude oscillations with many sign changes

Returns `true` if any criterion indicates convergence.

# Arguments
- `logs::DataFrame`: EM iteration logs with columns `:ll` and `:pi1`
- `tol::Float64=1e-4`: Tolerance for relative log-likelihood change
- `window::Int=5`: Window size for smoothing and stability checks
- `π_tol::Float64=1e-4`: Tolerance for π₁ parameter stability

# Returns
- `Bool`: `true` if EM has converged, `false` otherwise
"""
function hasEMconverged(logs; tol::Float64=1e-4, window::Int=5, π_tol::Float64=1e-4)
    niter = size(logs, 1)

    # Need at least 2*window iterations to compute smoothed change
    if niter < max(5, 2 * window)
        return false
    end

    # Criterion 1: Smoothed log-likelihood change
    recent_lls = logs[(niter - window + 1):niter, :ll]
    prev_lls = logs[(niter - 2 * window + 1):(niter - window), :ll]

    if all(isfinite.(recent_lls)) && all(isfinite.(prev_lls))
        mean_recent = mean(recent_lls)
        mean_prev = mean(prev_lls)
        if abs(mean_prev) > eps(Float64)
            ll_change = abs(mean_recent - mean_prev) / abs(mean_prev)
            if ll_change < tol
                return true
            end
        end
    end

    # Criterion 2: Parameter stability (π₁ range over window)
    recent_π1 = logs[(niter - window + 1):niter, :pi1]
    if all(isfinite.(recent_π1))
        π1_range = maximum(recent_π1) - minimum(recent_π1)
        if π1_range < π_tol
            return true
        end
    end

    # Criterion 3: Oscillation detection (many sign changes with small amplitude)
    if niter >= 10
        ll_recent = logs[(niter - 9):niter, :ll]
        if all(isfinite.(ll_recent))
            ll_diffs = diff(ll_recent)
            signs = sign.(ll_diffs)
            # Count sign changes (direction reversals)
            sign_changes = sum(signs[1:end-1] .!= signs[2:end])

            # If 6+ sign changes in 9 steps, likely oscillating
            if sign_changes >= 6
                amplitude = maximum(abs.(ll_diffs))
                # If amplitude is small relative to current ll, declare converged
                if abs(logs[niter, :ll]) > eps(Float64)
                    if amplitude < abs(logs[niter, :ll]) * tol * 10
                        return true
                    end
                end
            end
        end
    end

    return false
end


# em_fit_mixture for BayesFactorTriplet (legacy stub -- error out)
function em_fit_mixture(p::BayesFactorTriplet, joint_H0::SklarDist; max_iter=500, init_pi0=0.80)
    @error("em_fit_mixture is not implemented for BayesFactorTriplet. Use combined_BF with phase1_result instead.")
end

# --- Old probability-scale EM code removed ---
# em_fit_mixture(p::PosteriorProbabilityTriplet, ...) has been replaced by
# the 3-component copula EM inside combined_BF(bf, refID; phase1_result=...).
#
# em_fit_mixture_robust(p::PosteriorProbabilityTriplet, ...) has been replaced by
# the multi-restart logic inside combined_BF.
#
# em_restart_diagnostics has been removed (used old EM).
# em_fit_mixture_accelerated has been replaced by em_fit_mixture_accelerated_logbf in em_acceleration.jl.


"""
    summarize_em_diagnostics(diag::DataFrame)

Print a summary of EM restart diagnostics with robust handling of NaN values.

# Arguments
- `diag::DataFrame`: Output from `em_restart_diagnostics`

# Notes
- All statistics (mean, std, min, max, range) are computed after filtering out NaN values
- This ensures that failed restarts with NaN entries don't affect the summary
- If all values are NaN for log-likelihood or π₁, returns a minimal result

# Returns
- `NamedTuple`: Summary statistics including:
  - `n_successful`: Number of successful restarts (before NaN filtering)
  - `n_converged`: Number of converged restarts
  - `best_ll`: Best log-likelihood (among valid values)
  - `worst_ll`: Worst log-likelihood (among valid values)
  - `ll_range`: Range of log-likelihoods (among valid values)
  - `π1_mean`: Mean of final π₁ values (ignoring NaN)
  - `π1_std`: Standard deviation of π₁ values (ignoring NaN)
  - `π1_range`: Range of π₁ values (ignoring NaN)
  - `is_robust`: Whether results are considered robust (π₁ std < 0.05 and ll_range < 100)
"""
function summarize_em_diagnostics(diag::DataFrame)
    successful = diag[diag.status .== "success", :]
    n_successful = nrow(successful)

    if n_successful == 0
        println("⚠️  All restarts failed!")
        return (n_successful = 0, n_converged = 0, is_robust = false)
    end

    n_converged = sum(successful.converged)
    lls = successful.log_likelihood
    π1s = successful.final_pi1

    # Filter out NaN values to ignore failed restarts from statistics
    lls_valid = lls[isfinite.(lls)]
    π1s_valid = π1s[isfinite.(π1s)]

    # Check if we have any valid values after filtering
    if isempty(lls_valid) || isempty(π1s_valid)
        println("⚠️  No valid log-likelihoods or π₁ values found!")
        return (n_successful = n_successful, n_converged = n_converged, is_robust = false)
    end

    worst_ll, best_ll = extrema(lls_valid)
    relative_ll_deviation = abs(best_ll - worst_ll) / mean(lls_valid)

    π1_mean = mean(π1s_valid)
    π1_std = std(π1s_valid)
    π1_min, π1_max = extrema(π1s_valid)
    ll_range = best_ll - worst_ll
    # Robustness criterion: π₁ std < 0.05 and relative_ll_deviation < 0.05
    is_robust = π1_std < 0.05 && relative_ll_deviation < 0.05

    println("═══════════════════════════════════════════════════════")
    println("           EM Restart Diagnostics Summary              ")
    println("═══════════════════════════════════════════════════════")
    println()
    println("Restarts: $n_successful successful, $n_converged converged")
    println()
    println("Log-likelihood:")
    println("  Best:  $(round(best_ll, digits=2))")
    println("  Worst: $(round(worst_ll, digits=2))")
    println("  Range: $(round(ll_range, digits=2))")
    println()
    println("π₁ (interaction proportion):")
    println("  Mean:  $(round(π1_mean, digits=4))")
    println("  Std:   $(round(π1_std, digits=4))")
    println("  Range: [$(round(π1_min, digits=4)), $(round(π1_max, digits=4))]")
    println()

    if is_robust
        println("✓ Results appear ROBUST (low variability across restarts)")
    else
        if π1_std >= 0.05
            println("⚠️  HIGH VARIABILITY in π₁ estimates (std = $(round(π1_std, digits=4)))")
            println("    Consider: more restarts, different priors, or data quality check")
        end
        if relative_ll_deviation >= 0.05
            println("⚠️  LARGE LOG-LIKELIHOOD SPREAD (relative deviation = $(round(relative_ll_deviation, digits=2)))")
            println("    EM may be converging to different local optima")
        end
    end
    println()
    println("═══════════════════════════════════════════════════════")

    return (
        n_successful = n_successful,
        n_converged = n_converged,
        best_ll = best_ll,
        worst_ll = worst_ll,
        ll_range = ll_range,
        π1_mean = π1_mean,
        π1_std = π1_std,
        π1_range = (π1_min, π1_max),
        is_robust = is_robust
    )
end

"""
    plot_em_diagnostics(diag::DataFrame)

Create diagnostic plots for EM restart analysis.

Returns a combined plot with:
1. Log-likelihood by restart (with best marked)
2. Final π₁ by restart (with mean line)
3. π₁ vs log-likelihood scatter
4. Iterations to convergence by method

Requires StatsPlots to be loaded.
"""
function plot_em_diagnostics(diag::DataFrame)
    successful = diag[diag.status .== "success", :]

    if nrow(successful) == 0
        error("No successful restarts to plot")
    end

    # Plot 1: Log-likelihood by restart
    best_idx = argmax(successful.log_likelihood)
    plt1 = StatsPlots.scatter(successful.restart, successful.log_likelihood,
        xlabel = "Restart", ylabel = "Log-likelihood",
        label = nothing, markersize = 6,
        title = "Log-likelihood by Restart"
    )
    StatsPlots.scatter!(plt1, [successful.restart[best_idx]], [successful.log_likelihood[best_idx]],
        markersize = 10, markershape = :star5, color = :red, label = "Best")

    # Plot 2: π₁ by restart
    π1_mean = mean(successful.final_pi1)
    plt2 = StatsPlots.scatter(successful.restart, successful.final_pi1,
        xlabel = "Restart", ylabel = "π₁",
        label = nothing, markersize = 6,
        title = "π₁ by Restart"
    )
    StatsPlots.hline!(plt2, [π1_mean], linestyle = :dash, color = :red,
        label = "Mean ($(round(π1_mean, digits=3)))")

    # Plot 3: π₁ vs log-likelihood
    plt3 = StatsPlots.scatter(successful.final_pi1, successful.log_likelihood,
        xlabel = "π₁", ylabel = "Log-likelihood",
        label = nothing, markersize = 6,
        title = "π₁ vs Log-likelihood"
    )

    # Plot 4: Iterations by method
    method_colors = Dict(:quantile => :blue, :kmeans => :green, :random_top20 => :orange)
    plt4 = StatsPlots.scatter(successful.restart, successful.iterations,
        xlabel = "Restart", ylabel = "Iterations",
        group = successful.init_method,
        markersize = 6,
        title = "Iterations to Convergence"
    )

    return StatsPlots.plot(plt1, plt2, plt3, plt4, layout = (2, 2), size = (900, 700))
end


"""
    EMconvergenceDiagnosticPlot(result::EMResult)

    Plot diagnostics for EM convergence
        This function plots the log-likelihood, π0, and π1 over iterations.

    Args:
        result: EMResult

    Returns:
        Plots
"""
function EMconvergenceDiagnosticPlot(result::EMResult)
    return EMconvergenceDiagnosticPlot(result.logs)
end

"""
    EMconvergenceDiagnosticPlot(logs)

    Plot diagnostics for EM convergence
        This function plots the log-likelihood, π0, and π1 over iterations.

    Args:
        logs: DataFrame with columns `iter`, `π0`, `π1`, and `ll`

    Returns:
        Plots
"""
function EMconvergenceDiagnosticPlot(logs)
    plt1 = StatsPlots.plot(
        logs.iter[2:end], logs.ll[2:end],
        seriestype = :line, legend = true, label = nothing,
        xlabel = "Iteration", ylabel = "Log-likelihood",
        foreground_color_legend = nothing, background_color_legend = nothing
    )

    plt2 = StatsPlots.plot(
        logs.iter[2:end], logs.pi0[2:end],
        seriestype = :line, legend = true, label = nothing,
        xlabel = "Iteration", ylabel = "π0",
        foreground_color_legend = nothing, background_color_legend = nothing
    )

    plt3 = StatsPlots.plot(
        logs.iter[2:end], logs.pi1[2:end],
        seriestype = :line, legend = true, label = nothing,
        xlabel = "Iteration", ylabel = "π1",
        foreground_color_legend = nothing, background_color_legend = nothing
    )

    return StatsPlots.plot(plt1, plt2, plt3, layout = (3,1), size = (600, 600))
end


######################################################
# Log-BF scale copula combination pathway
######################################################

# --- Copula families for log-BF pathway (4 target families, per user decision) ---
const LOGBF_COPULA_FAMILIES = Dict(
    "ClaytonCopula" => ClaytonCopula,
    "FrankCopula" => FrankCopula,
    "GumbelCopula" => GumbelCopula,
    "GaussianCopula" => GaussianCopula
)

# PrecomputedH0 has been merged into PrecomputedH0 (see above)

"""
    _fit_with_ks_check(data, initial_fit, ks_threshold) -> (dist, ks_stat, was_upgraded)

Fit a Normal marginal and check KS uniformity of PIT values. If KS > threshold,
auto-upgrade to LocationScale(mu, sigma, TDist(nu)) by grid search over nu.

# Arguments
- `data::AbstractVector{<:Real}`: Log-BF data for one marginal
- `initial_fit`: Initial Normal distribution fit
- `ks_threshold::Float64`: KS threshold for triggering upgrade (default 0.15)

# Returns
- `(dist, ks_stat, was_upgraded)`: Fitted distribution, KS statistic, upgrade flag
"""
function _fit_with_ks_check(data::AbstractVector{<:Real}, initial_fit, ks_threshold::Float64)
    # PIT-transform data with initial Normal
    u = cdf.(initial_fit, data)
    u_clamped = clamp.(u, 1e-10, 1.0 - 1e-10)
    ks = _ks_statistic(u_clamped, Distributions.Uniform(0.0, 1.0))

    if ks <= ks_threshold
        return (initial_fit, ks, false)
    end

    # Grid search over TDist degrees of freedom
    mu_data = mean(data)
    sigma_data = std(data)
    sigma_data = max(sigma_data, 1e-6)  # guard against zero std

    best_ks = ks
    best_dist = initial_fit
    best_upgraded = false

    for nu in [3, 4, 5, 7, 10, 15, 20, 30]
        candidate = LocationScale(mu_data, sigma_data, TDist(nu))
        u_candidate = clamp.(cdf.(candidate, data), 1e-10, 1.0 - 1e-10)
        ks_candidate = _ks_statistic(u_candidate, Distributions.Uniform(0.0, 1.0))
        if ks_candidate < best_ks
            best_ks = ks_candidate
            best_dist = candidate
            best_upgraded = true
        end
    end

    if !best_upgraded
        @warn "KS auto-upgrade: no TDist improved over Normal (KS=$ks); keeping Normal"
    end

    return (best_dist, best_ks, best_upgraded)
end

"""
    _build_pseudo_obs(log_bf_e, log_bf_c, log_bf_d, marg_e, marg_c, marg_d) -> Matrix{Float64}

Construct a 3 x n matrix of PIT-transformed pseudo-observations for copula fitting.
"""
function _build_pseudo_obs(log_bf_e::AbstractVector, log_bf_c::AbstractVector,
                            log_bf_d::AbstractVector,
                            marg_e, marg_c, marg_d;
                            gpd_tails::Union{Nothing, NamedTuple} = nothing,
                            streams::AbstractVector{Symbol} = collect(CANONICAL_STREAMS))
    eps_u = 1e-10

    gpd_e = (gpd_tails !== nothing && haskey(gpd_tails, :enrichment)) ? gpd_tails.enrichment : nothing
    gpd_c = (gpd_tails !== nothing && haskey(gpd_tails, :correlation)) ? gpd_tails.correlation : nothing

    # ABL-P2: emit only the active rows (canonical order). With all three streams
    # this is byte-identical to the legacy 3-row matrix; dropping a stream removes
    # exactly its row + marginal. `n` is taken from a PRESENT active stream's vector
    # (the dropped stream's positional vector may be empty under a drop).
    active = _active_streams(streams)
    _vec_by_stream = Dict(:enrichment => log_bf_e, :correlation => log_bf_c, :detection => log_bf_d)
    n = length(_vec_by_stream[active[1]])
    u = Matrix{Float64}(undef, length(active), n)
    for (k, s) in enumerate(active)
        if s === :enrichment
            for j in 1:n
                u[k, j] = gpd_e !== nothing ? gpd_extended_cdf(marg_e, Float64(log_bf_e[j]), gpd_e) :
                           clamp(cdf(marg_e, log_bf_e[j]), eps_u, 1.0 - eps_u)
            end
        elseif s === :correlation
            for j in 1:n
                u[k, j] = gpd_c !== nothing ? gpd_extended_cdf(marg_c, Float64(log_bf_c[j]), gpd_c) :
                           clamp(cdf(marg_c, log_bf_c[j]), eps_u, 1.0 - eps_u)
            end
        else  # :detection -- never GPD
            for j in 1:n
                u[k, j] = clamp(cdf(marg_d, log_bf_d[j]), eps_u, 1.0 - eps_u)
            end
        end
    end
    return u
end

"""
    _compare_copulas_logbf(u; criterion=:BIC) -> DataFrame

BIC-based copula family comparison on a raw 3 x n pseudo-observation matrix.
Uses only the 4 target families: Clayton, Frank, Gumbel, Gaussian.

# Arguments
- `u::AbstractMatrix{Float64}`: 3 x n pseudo-observation matrix
- `criterion::Symbol`: Selection criterion (`:BIC`, `:AIC`, or `:loglik`)

# Returns
- `DataFrame`: Sorted comparison with columns `Family`, `LogLik`, `BIC`, `AIC`
"""
function _compare_copulas_logbf(u::AbstractMatrix{Float64}; criterion::Symbol = :BIC)
    n = size(u, 2)
    results = DataFrame(Family=String[], LogLik=Float64[], BIC=Float64[], AIC=Float64[])
    error_only_logger = MinLevelLogger(current_logger(), Logging.Error)
    for (name, fam) in LOGBF_COPULA_FAMILIES
        try
            with_logger(error_only_logger) do
                cop = fit(fam, u)
                ll = loglikelihood(cop, u)
                k = copula_nparams(fam)
                bic = -2 * ll + k * log(n)
                aic = -2 * ll + 2 * k
                push!(results, (name, ll, bic, aic))
            end
        catch e
            # Skip families that fail to fit
        end
    end
    sort!(results, criterion == :BIC ? :BIC : (criterion == :AIC ? :AIC : :LogLik),
          rev = (criterion != :BIC && criterion != :AIC))
    return results
end

"""
    _fit_best_copula_logbf(u; criterion=:BIC, min_samples=50, force_family=nothing) -> (copula, family_name)

Fit the best copula from the 4 target families. Falls back to independence copula
(GaussianCopula with identity correlation) if fewer than `min_samples` observations
or all families fail to fit.

When `force_family` is set (ablation knob), the BIC selection is
bypassed entirely and the requested family is fitted directly. The `n < min_samples`
independence-copula guard still takes precedence (a forced family on too-few samples is
degenerate). The default `force_family = nothing` reaches the unchanged BIC path, so the
byte-identity gate is preserved.

# Arguments
- `u::AbstractMatrix{Float64}`: 3 x n pseudo-observation matrix
- `criterion::Symbol`: Selection criterion
- `min_samples::Int`: Minimum sample size for fitting
- `force_family::Union{Nothing, Type}`: Force a fixed copula family (e.g. `FrankCopula`)
  instead of BIC selection; `nothing` (default) = BIC over `LOGBF_COPULA_FAMILIES`.

# Returns
- `(copula, family_name)`: Fitted copula and its family name string
"""
function _fit_best_copula_logbf(u::AbstractMatrix{Float64};
                                 criterion::Symbol = :BIC,
                                 min_samples::Int = 50,
                                 force_family::Union{Nothing, Type} = nothing)
    n = size(u, 2)
    if force_family !== nothing
        # Ablation knob (ABL-P3): bypass BIC, fit the requested family directly. A
        # forced family is an explicit user override, so it takes precedence over the
        # `min_samples` BIC-reliability gate -- but we still need at least a handful of
        # points to estimate the copula parameter. Below that floor (or if the fit
        # errors) fall back to the independence copula like the BIC path does.
        force_min = min(min_samples, 5)
        if n < force_min
            @warn "Too few observations ($n < $force_min) to force copula family; using independence copula"
            return GaussianCopula(1.0 * I(size(u, 1))), "IndependentCopula"
        end
        error_only_logger = MinLevelLogger(current_logger(), Logging.Error)
        cop = try
            with_logger(error_only_logger) do
                fit(force_family, u)
            end
        catch e
            @warn "Forced copula family fit failed ($(force_family)); using independence copula" exception=e
            nothing
        end
        cop === nothing && return GaussianCopula(1.0 * I(size(u, 1))), "IndependentCopula"
        # Resolve the family name the SAME way the BIC path does -- via the
        # LOGBF_COPULA_FAMILIES string key, NOT nameof(). The Archimedean families
        # (FrankCopula/ClaytonCopula/GumbelCopula) are UnionAll aliases whose nameof
        # collapses to "ArchimedeanCopula"; only the map key carries the family label.
        forced_name = nothing
        for (k, v) in LOGBF_COPULA_FAMILIES
            if v === force_family
                forced_name = k
                break
            end
        end
        return cop, (forced_name === nothing ? string(nameof(force_family)) : forced_name)
    end
    if n < min_samples
        @warn "Too few observations ($n < $min_samples) for copula fitting; using independence copula"
        return GaussianCopula(1.0 * I(size(u, 1))), "IndependentCopula"
    end
    results = _compare_copulas_logbf(u; criterion=criterion)
    if nrow(results) == 0
        @warn "All copula families failed to fit; using independence copula"
        return GaussianCopula(1.0 * I(size(u, 1))), "IndependentCopula"
    end
    best_name = results.Family[1]
    best_fam = LOGBF_COPULA_FAMILIES[best_name]
    error_only_logger = MinLevelLogger(current_logger(), Logging.Error)
    cop = with_logger(error_only_logger) do
        fit(best_fam, u)
    end
    return cop, best_name
end



