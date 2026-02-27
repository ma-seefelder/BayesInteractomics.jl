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
        log_lik_H0 = logpdf.(Ref(joint_H0), eachcol(p_triplets))

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
                              n_resample::Int = 10_000)
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
        return fit_copula(p_resampled)
    catch e
        @warn "Weighted copula fitting failed: $e"
        return nothing
    end
end

function H0_BF_enrichment(t₀::Float64 = 10.0, α₀ = 0.05)
    # Calibrate σ so that tail probability matches α₀
    σ = log(t₀) / quantile(Normal(), 1 - α₀)
    μ = 0.0                     
    bf_null_dist = LogNormal(μ, σ) 
    
    isapprox(1 - cdf(bf_null_dist, t₀), α₀; atol = 1e-3) || error("Calibration failed")
    isapprox(median(bf_null_dist), 1.0; atol = 1e-12) || error("Calibration failed")
    mean(bf_null_dist) <= 1.0 && error("Calibration failed")
    
    return bf_null_dist
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
- `savefile::String="copula_H0.xlsx"`: Path to the Excel file where the results will be written. If the file already exists, an error is raised.
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
function computeH0_BayesFactors(data; savefile = "copula_H0.xlsx", n_controls = 0, n_samples = 0, refID = 1, n::Int = 25_000,
    regression_likelihood::Symbol = :robust_t,
    student_t_nu::Float64 = 5.0)
    isfile(savefile) && @error "The file $savefile already exists"
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

    if getNoProtocols(data) == 1
        cached_hbm_prior_h0 = precompute_HBM_single_protocol_prior(data; μ_0=μ_0_h0, σ_0=σ_0_h0, a_0=a_0_h0, b_0=b_0_h0)
        cached_regression_prior_h0 = if regression_likelihood == :robust_t
            precompute_regression_one_protocol_robust_prior(data, refID, μ_0_h0, σ_0_h0; nu=student_t_nu, τ_base=robust_tau_base_h0)
        else
            precompute_regression_one_protocol_prior(data, refID, μ_0_h0, σ_0_h0)
        end
    else
        cached_hbm_prior_h0 = precompute_HBM_prior(data; μ_0=μ_0_h0, σ_0=σ_0_h0, a_0=a_0_h0, b_0=b_0_h0)
        cached_regression_prior_h0 = if regression_likelihood == :robust_t
            precompute_regression_multi_protocol_robust_prior(data, refID, μ_0_h0, σ_0_h0; nu=student_t_nu, τ_base=robust_tau_base_h0)
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
                robust_tau_base = robust_tau_base_h0
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
        @warn "H0 DataFrame has only $(nrow(H0)) rows after filtering. Results may be unreliable. Consider deleting $(savefile) and rerunning."
    end

    writetable(savefile,  H0)
    return H0
end


#H0_bf = rand(LogNormal(0, 1), 100_000, 3)
#CSV.write("copula_H0.csv", DataFrame(H0_bf, :auto))

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
- `H0_file::String="copula_H0.xlsx"`: Path to precomputed H0 Bayes factors
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
function combined_BF(bf::BayesFactorTriplet, refID::Int64;
                     H0_file = "copula_H0.xlsx",
                     max_iter = 1000,
                     init_π0 = 0.80,
                     prior::Union{Symbol, NamedTuple} = :default,
                     n_restarts::Int = 20,
                     copula_criterion::Symbol = :BIC,
                     h1_refitting::Bool = true,
                     burn_in::Int = 10,
                     use_acceleration::Bool = true,
                     verbose::Bool = true)
    # --- load permuted data: H0 --- #
    H0 = DataFrame(readtable(H0_file, "Sheet1"))

    if nrow(H0) == 0
        error("""
        combined_BF: The H0 file '$H0_file' is empty (0 rows after filtering).
        This typically means the permuted null-distribution computation in computeH0_BayesFactors failed for all proteins.
        Please delete '$H0_file' and rerun the analysis to trigger a fresh H0 computation.
        Check the log output for 'H0 computation failed' messages to identify the root cause.
        """)
    end

    bf_H0 = BayesFactorTriplet(
        Vector{Float64}(H0.bf_enrichment),
        Vector{Float64}(H0.bf_correlation),
        Vector{Float64}(H0.bf_detected)
    )

    # --- add proteins with a BF-enrichment below 3 and bf_correlation below 1.0 to H0--- #
    idx_H0 = Int64[]
    append!(idx_H0, findall(x -> x <= 3, bf.enrichment))
    append!(idx_H0, findall(x -> x <= 1.0, bf.correlation))
    unique!(idx_H0)

    append!(bf_H0.enrichment,  bf.enrichment[idx_H0])
    append!(bf_H0.correlation, bf.correlation[idx_H0])
    append!(bf_H0.detection,   bf.detection[idx_H0])

    # convert to posterior probabilities
    p_H0                = posterior_probability_from_bayes_factor(bf_H0)
    p                   = posterior_probability_from_bayes_factor(bf)

    # --- clamp p-values to avoid numeric instability --- #
    ϵ                   = eps(Float64)
    p_H0                = squeeze(p_H0, ϵ = ϵ)
    p                   = squeeze(p, ϵ = ϵ)

    #--- fit copula for the null-hypothesis: no interaction
    copula_H0           = fit_copula(p_H0; criterion=copula_criterion)
    H0_marg1            = fit(Beta, p_H0.enrichment)
    H0_marg2            = fit(Beta, p_H0.correlation)
    H0_marg3            = fit(Beta, p_H0.detection)
    joint_H0            = SklarDist(copula_H0, (H0_marg1, H0_marg2, H0_marg3))

    # Handle empirical Bayes prior estimation
    actual_prior = prior
    if prior == :empirical_bayes
        verbose && @info "Estimating prior using empirical Bayes..."
        eb_result = estimate_prior_empirical_bayes(p, joint_H0)
        actual_prior = (α = eb_result.α, β = eb_result.β)
        verbose && @info "Empirical Bayes prior: α=$(eb_result.α), β=$(eb_result.β), expected π₁=$(round(eb_result.expected_π1, digits=3))"
    end

    #--- fit copula for the alternate-hypothesis: interaction --- #
    em_diagnostics = nothing
    if n_restarts > 1
        em, em_diagnostics = em_fit_mixture_robust(p, joint_H0, refID;
                                    n_restarts = n_restarts,
                                    max_iter = max_iter,
                                    init_π0 = init_π0,
                                    prior = actual_prior,
                                    h1_refitting = h1_refitting,
                                    burn_in = burn_in,
                                    copula_criterion = copula_criterion,
                                    use_acceleration = use_acceleration,
                                    verbose = verbose)
    else
        # Use accelerated EM if enabled
        if use_acceleration
            em = em_fit_mixture_accelerated(p, joint_H0, refID;
                                use_acceleration = true,
                                max_iter = max_iter,
                                init_π0 = init_π0,
                                prior = actual_prior,
                                h1_refitting = h1_refitting,
                                burn_in = burn_in,
                                copula_criterion = copula_criterion,
                                verbose = verbose)
        else
            em = em_fit_mixture(p, joint_H0, refID;
                                max_iter = max_iter,
                                init_π0 = init_π0,
                                prior = actual_prior,
                                h1_refitting = h1_refitting,
                                burn_in = burn_in,
                                copula_criterion = copula_criterion,
                                verbose = verbose)
        end
    end
    prior_odds          = em.π1 /  em.π0

    # --- predict Bayes Factors --- #
    p_triplets                  = hcat(p.enrichment, p.correlation, p.detection)'
    log_likelihood_H1           = logpdf.(Ref(em.joint_H1), eachcol(p_triplets))
    log_likelihood_H0           = logpdf.(Ref(joint_H0), eachcol(p_triplets))
    # converte bait protein to min and max values
    log_likelihood_H0[refID] = findmin(log_likelihood_H0[isfinite.(log_likelihood_H0)])[1]
    log_likelihood_H1[refID] = findmax(log_likelihood_H1[isfinite.(log_likelihood_H1)])[1]
    # compute log_BF
    log_BF                      = log_likelihood_H1 .- log_likelihood_H0
    min_log_bf                  = log(floatmin(Float64)) # minimal possible value for Float64
    max_log_bf                  = log(floatmax(Float64)) # maximal possible value for Float64
    log_BF                      = clamp.(log_BF, min_log_bf, max_log_bf)
    BF                          = exp.(log_BF)
    # compute posterior probabilities: posterior_odds = BF × prior_odds
    posterior_odds              = BF .* prior_odds
    posterior_prob              = posterior_odds ./ (1 .+ posterior_odds)
    # return results
    return CombinedBayesResult(BF, posterior_prob, joint_H0, em.joint_H1, em, em_diagnostics)
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
        idx_enrichment = findall(x -> x < 0.5 || x < 0.5, p.enrichment[idx_detection])
        deleteat!(idx_detection, idx_enrichment)
        # remove proteins with negative correlation evidence
        idx_correlation = findall(x -> x < 0.5, p.correlation[idx_detection])
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
- `logs::DataFrame`: EM iteration logs with columns `:ll` and `:π1`
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
    recent_π1 = logs[(niter - window + 1):niter, :π1]
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


function em_fit_mixture(p::BayesFactorTriplet, joint_H0::SklarDist; max_iter=500, init_pi0=0.80)
    @error("em_fit_mixture is not implemented for BayesFactorTriplet. Convert to PosteriorProbabilityTriplet before")
end

"""
    em_fit_mixture(p, joint_H0, refID; kwargs...)

Fit a two-component mixture model using the EM algorithm.

# Arguments
- `p::PosteriorProbabilityTriplet`: Posterior probabilities from the three models
- `joint_H0::SklarDist`: Joint distribution under the null hypothesis
- `refID::Int64`: Index of the reference (bait) protein

# Keywords
- `max_iter::Int=5000`: Maximum number of EM iterations
- `init_π0::Float64=0.80`: Initial value for π₀ (null proportion)
- `prior::Union{Symbol, NamedTuple}=:default`: Prior for π₁ (:APMS, :BioID, :TurboID, :default, :permissive, :stringent, or custom (α=, β=))
- `h1_refitting::Bool=true`: Enable weighted H1 updates in M-step
- `burn_in::Int=10`: Number of iterations before starting H1 re-fitting
- `copula_criterion::Symbol=:BIC`: Copula selection criterion for H1 re-fitting
- `init_set::Union{Vector{Int}, Nothing}=nothing`: Custom initialization set for H1
- `damping_initial::Float64=0.3`: Initial damping factor for H1 updates (0=no damping, 1=full damping)
- `damping_final::Float64=0.8`: Final damping factor for H1 updates (increases with iterations)
- `verbose::Bool=true`: Print progress information

# Returns
- `EMResult`: EM fitting results including π₀, π₁, joint_H1, logs, and convergence status

# Notes
Damping is applied to H1 marginal updates to prevent oscillations. The damping factor
interpolates linearly from `damping_initial` to `damping_final` as iterations progress
from burn-in to max_iter. Higher damping means slower, more stable updates.
"""
function em_fit_mixture(p::PosteriorProbabilityTriplet, joint_H0::SklarDist, refID::Int64;
                        max_iter::Int = 5000,
                        init_π0::Float64 = 0.80,
                        prior::Union{Symbol, NamedTuple} = :default,
                        h1_refitting::Bool = true,
                        burn_in::Int = 10,
                        copula_criterion::Symbol = :BIC,
                        init_set::Union{Vector{Int}, Nothing} = nothing,
                        damping_initial::Float64 = 0.3,
                        damping_final::Float64 = 0.8,
                        verbose::Bool = true)
    ϵ = eps(Float64)
    π0 = init_π0
    π1 = 1.0 - π0

    # Get prior hyperparameters
    if prior isa Symbol
        prior_params = get_prior_hyperparameters(prior)
    else
        prior_params = prior
    end
    α_prior, β_prior = prior_params.α, prior_params.β

    # Float64 limits for numerical stability:
    # exp(709) ≈ 1.8e308 (near floatmax), exp(-745) ≈ 5e-324 (near floatmin)
    max_log_exp = 709.0   # Maximum safe input for exp()
    min_log_exp = -745.0  # Minimum safe input for exp()
    # ----------------------------------------------------------------------------- #
    # Step 1: Initial estimate of f1 from strong signal proteins
    # ----------------------------------------------------------------------------- #
    if init_set === nothing
        # Compute mean strength, replacing NaN with 0 (no signal)
        mean_strength = @. p.enrichment * p.correlation * p.detection
        mean_strength = replace(mean_strength, NaN => 0.0)

        # Filter valid values for quantile computation
        valid_strengths = filter(isfinite, mean_strength)
        if isempty(valid_strengths) || length(valid_strengths) < 10
            # Fallback: use default threshold if too few valid values
            quantile_threshold = 0.5
        else
            quantile_threshold = quantile(valid_strengths, 0.95)
        end
        idx_init = find_H1_initialization_set(mean_strength, p, quantile_threshold)
    else
        idx_init = init_set
    end

    # ------------------------- #
    # Fit marginals and copulas
    # ------------------------- #
    # Check if we have enough proteins for initialization
    if isempty(idx_init) || length(idx_init) < 5
        @warn "Insufficient proteins for H1 initialization ($(length(idx_init))), using default distributions"
        marg1 = Beta(2.0, 2.0)
        marg2 = Beta(2.0, 2.0)
        marg3 = Beta(2.0, 2.0)
        copula_H1 = FrankCopula(3, 1.0)
        joint_H1 = SklarDist(copula_H1, (marg1, marg2, marg3))
    else
        # Squeeze probabilities away from boundaries to avoid numerical issues in Beta fitting
        p_init = squeeze(p[idx_init], ϵ=1e-10)
        # Use safe Beta fitting that handles low-variance edge cases
        marg1 = fit_beta_safe(p_init.enrichment)
        marg2 = fit_beta_safe(p_init.correlation)
        marg3 = fit_beta_safe(p_init.detection)

        # Fit copula with fallback to FrankCopula if fitting fails
        copula_H1 = try
            fit_copula(p_init; criterion=copula_criterion)
        catch e
            @warn "Copula fitting failed: $e, using default FrankCopula"
            FrankCopula(3, 1.0)
        end
        joint_H1 = SklarDist(copula_H1, (marg1, marg2, marg3))
    end

    # Squeeze probabilities for EM evaluation to avoid logpdf issues at boundaries
    p_squeezed = squeeze(p, ϵ=1e-10)
    p_triplets = hcat(p_squeezed.enrichment, p_squeezed.correlation, p_squeezed.detection)'
    # Initialize log-likelihood
    prev_ll = -Inf
    logs = DataFrame(iter = Int[0], π0=Float64[π0], π1=Float64[π1], ll=Float64[prev_ll])

    ##############
    # EM-loop
    ##############
    progress = nothing
    if verbose
        progress = Progress(
            max_iter, desc="Fitting EM model...",
            showspeed=true,
            barglyphs=BarGlyphs('|','█', [' ' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen = 20, dt = 1
            )
    end

    for iter in 1:max_iter
        # ---------------------------------------------------------------- #
        # --- E-Step: Calculate responsibilities (weights) --- #
        # ---------------------------------------------------------------- #
        # Clamp log-densities to prevent overflow in subsequent exp() operations
        f0_vals = clamp.(logpdf.(Ref(joint_H0), eachcol(p_triplets)), min_log_exp, max_log_exp)
        f1_vals = clamp.(logpdf.(Ref(joint_H1), eachcol(p_triplets)), min_log_exp, max_log_exp)

        !isfinite(f0_vals[refID]) && (f0_vals[refID] = min_log_exp)
        !isfinite(f1_vals[refID]) && (f1_vals[refID] = min_log_exp)

        # Compute log-weights with careful clamping to prevent overflow
        log_π0 = log(π0)
        log_π1 = log(π1)

        # Use clamped exp() to avoid overflow: log(π0·exp(f0) + π1·exp(f1))
        log_weights = log_π1 .+ f1_vals
        log_weights .-= @. log(
            π0 * exp(clamp(log_π0 + f0_vals, min_log_exp, max_log_exp)) +
            π1 * exp(clamp(log_π1 + f1_vals, min_log_exp, max_log_exp))
        )

        # Convert log-weights to weights with safe exp() and clamping
        w = exp.(clamp.(log_weights, -7.0, max_log_exp))
        w = clamp.(w, ϵ, 1-ϵ)
        replace!(w, Inf => 1.0 - ϵ, -Inf => ϵ)

        # log_marginal_likelihood
        log_marginal_likelihood = logsumexp.(log_π0 .+ f0_vals, log_π1 .+ f1_vals)

        # ---------------------------------------------------------------- #
        # --- M-Step: update π and optionally fit f1 using weighted data - #
        # ---------------------------------------------------------------- #

        # 1. Update H1 distribution parameters (after burn-in period)
        if h1_refitting && iter > burn_in
            # Compute progress-dependent damping factor (increases over iterations)
            progress = min(1.0, (iter - burn_in) / max(1, max_iter - burn_in))
            α_damp = damping_initial + progress * (damping_final - damping_initial)

            # Re-fit marginals with weights (using squeezed data)
            marg1_fit = fit_beta_weighted(p_squeezed.enrichment, w)
            marg2_fit = fit_beta_weighted(p_squeezed.correlation, w)
            marg3_fit = fit_beta_weighted(p_squeezed.detection, w)

            # Get current marginal parameters
            current_margs = joint_H1.m
            prev_marg1 = current_margs[1]
            prev_marg2 = current_margs[2]
            prev_marg3 = current_margs[3]

            # Apply damped updates with safety checks
            marg1_new = safe_damped_beta(marg1_fit, prev_marg1, α_damp)
            marg2_new = safe_damped_beta(marg2_fit, prev_marg2, α_damp)
            marg3_new = safe_damped_beta(marg3_fit, prev_marg3, α_damp)

            # Re-fit copula (only if sufficient effective samples)
            cop_new = fit_copula_weighted(p_squeezed, w)

            if cop_new !== nothing
                copula_H1 = cop_new
                joint_H1 = SklarDist(cop_new, (marg1_new, marg2_new, marg3_new))
            else
                # Only update marginals, keep copula structure
                joint_H1 = SklarDist(copula_H1, (marg1_new, marg2_new, marg3_new))
            end
        end

        # 2. Update mixture proportions (π) with Beta prior (MAP estimate)
        sum_weights = sum(w)
        N_weights = length(w)

        π1_new = (sum_weights + α_prior - 1) / (N_weights + α_prior + β_prior - 2)
        π0_new = 1.0 - π1_new

        # ---------------------------------- #
        # ------- Convergence check -------- #
        # ---------------------------------- #
        ll = sum(log_marginal_likelihood)
        push!(logs, (iter, π0_new, π1_new, ll))

        # Check for convergence
        if hasEMconverged(logs, tol = 1e-3)
            break
        end

        # --------------------------------------------------------------- #
        # ---------- Update parameters for next iteration --------------- #
        # --------------------------------------------------------------- #
        π0 = max(π0_new, 1e-6)
        π1 = π1_new
        prev_ll = ll

        verbose && !isnothing(progress) && ProgressMeter.next!(progress)
    end
    verbose && !isnothing(progress) && finish!(progress)

    iter = logs[end, :iter]
    has_converged = iter < max_iter

    # Oscillation-aware termination: if not converged, check for oscillation and average
    if !has_converged && size(logs, 1) >= 20
        # Check for oscillation in last 20 iterations
        ll_recent = logs[(end - 19):end, :ll]
        if all(isfinite.(ll_recent))
            ll_diffs = diff(ll_recent)
            signs = sign.(ll_diffs)
            sign_changes = sum(signs[1:end-1] .!= signs[2:end])

            # If 12+ sign changes in 19 differences, we're oscillating
            if sign_changes >= 12
                # Average π parameters over last 20 iterations
                π0_avg = mean(logs[(end - 19):end, :π0])
                π1_avg = mean(logs[(end - 19):end, :π1])

                # Use averaged parameters
                π0 = π0_avg
                π1 = π1_avg
                has_converged = true  # Consider this as converged via averaging

                verbose && @info "EM oscillating at max_iter, using averaged parameters (π₁=$(round(π1_avg, digits=4)))"
            end
        end
    end

    verbose && has_converged && iter < max_iter && println("EM converged at iteration $iter")
    !has_converged && @warn "EM did not converge"

    return EMResult(π0, π1, joint_H1, logs, has_converged)
end

"""
    em_fit_mixture_robust(p, joint_H0, refID; n_restarts=20, use_acceleration=true, kwargs...)

Run EM with multiple restarts and return best result by log-likelihood with diagnostics.

This function runs the EM algorithm with different initialization strategies,
collects diagnostic information from all runs, and returns both the best result
and the diagnostics DataFrame.

# Arguments
- `p::PosteriorProbabilityTriplet`: Posterior probabilities from the three models
- `joint_H0::SklarDist`: Joint distribution under the null hypothesis
- `refID::Int`: Index of the reference (bait) protein

# Keywords
- `n_restarts::Int=20`: Number of restarts (default: 10 for robustness)
- `use_acceleration::Bool=true`: Enable SQUAREM acceleration for faster convergence
- `verbose::Bool=true`: Print progress information
- All other keywords are passed to `em_fit_mixture`

# Returns
- `Tuple{EMResult, DataFrame}`: Best EM fitting result by log-likelihood and diagnostics DataFrame
"""
function em_fit_mixture_robust(p::PosteriorProbabilityTriplet,
                                joint_H0::SklarDist,
                                refID::Int;
                                n_restarts::Int = 20,
                                use_acceleration::Bool = true,
                                verbose::Bool = true,
                                kwargs...)
    best_result = nothing
    best_ll = -Inf
    # Compute empirical Bayes estimate for π0 once (used by :empirical_bayes strategies)
    eb_result = estimate_prior_empirical_bayes(p, joint_H0)
    eb_π0 = 1 - eb_result.expected_π1
    # Define initialization strategies
    init_strategies = [
        # Original quantile strategies
        (π0 = 0.80, method = :quantile),
        (π0 = 0.70, method = :quantile),
        (π0 = 0.90, method = :quantile),
        (π0 = 0.65, method = :quantile),
        (π0 = 0.95, method = :quantile),
        # Additional quantile coverage
        (π0 = 0.72, method = :quantile),
        (π0 = 0.88, method = :quantile),
        # Original kmeans strategies
        (π0 = 0.75, method = :kmeans),
        (π0 = 0.85, method = :kmeans),
        (π0 = 0.78, method = :kmeans),
        # Additional kmeans with lower π0 coverage
        (π0 = 0.60, method = :kmeans),
        (π0 = 0.55, method = :kmeans),
        (π0 = 0.50, method = :kmeans),
        # Original random_top20 strategies
        (π0 = 0.80, method = :random_top20),
        (π0 = 0.75, method = :random_top20),
        # Additional random_top20 coverage
        (π0 = 0.70, method = :random_top20),
        (π0 = 0.85, method = :random_top20),
        # Empirical Bayes strategies (π0 determined from data)
        (π0 = eb_π0, method = :empirical_bayes),
        (π0 = clamp(eb_π0 + 0.05, 0.0, 1.0), method = :empirical_bayes),
        (π0 = clamp(eb_π0 - 0.05, 0.0, 1.0), method = :empirical_bayes),
    ]


    # Limit to requested number of restarts
    strategies_to_use = init_strategies[1:min(n_restarts, length(init_strategies))]

    # Initialize diagnostics DataFrame
    diagnostics = DataFrame(
        restart = Int[],
        init_π0 = Float64[],
        init_method = Symbol[],
        final_π0 = Float64[],
        final_π1 = Float64[],
        log_likelihood = Float64[],
        iterations = Int[],
        converged = Bool[],
        status = String[]
    )

    for (i, strategy) in enumerate(strategies_to_use)
        verbose && println("EM restart $i/$(length(strategies_to_use)) with π₀=$(strategy.π0), method=$(strategy.method)")

        try
            init_set = get_H1_initialization_set(p; method=strategy.method)

            # Use accelerated EM if enabled
            if use_acceleration
                result = em_fit_mixture_accelerated(p, joint_H0, refID;
                                        use_acceleration=true,
                                        init_π0=strategy.π0,
                                        init_set=init_set,
                                        verbose=false,  # Suppress per-restart progress
                                        kwargs...)
            else
                result = em_fit_mixture(p, joint_H0, refID;
                                        init_π0=strategy.π0,
                                        init_set=init_set,
                                        verbose=false,  # Suppress per-restart progress
                                        kwargs...)
            end

            final_ll = result.logs[end, :ll]

            # Record diagnostics
            push!(diagnostics, (
                restart = i,
                init_π0 = strategy.π0,
                init_method = strategy.method,
                final_π0 = result.π0,
                final_π1 = result.π1,
                log_likelihood = final_ll,
                iterations = result.logs[end, :iter],
                converged = result.has_converged,
                status = "success"
            ))

            if isfinite(final_ll) && final_ll > best_ll
                best_ll = final_ll
                best_result = result
                verbose && println("  → New best log-likelihood: $(round(final_ll, digits=2))")
            end
        catch e
            verbose && @warn "Restart $i failed: $e"
            # Record failed restart in diagnostics
            push!(diagnostics, (
                restart = i,
                init_π0 = strategy.π0,
                init_method = strategy.method,
                final_π0 = NaN,
                final_π1 = NaN,
                log_likelihood = NaN,
                iterations = 0,
                converged = false,
                status = string(e)
            ))
        end
    end

    if best_result === nothing
        error("All EM restarts failed")
    end

    verbose && println("Best EM result: log-likelihood = $(round(best_ll, digits=2))")
    return best_result, diagnostics
end

"""
    em_restart_diagnostics(p, joint_H0, refID; n_restarts=20, kwargs...)

Run EM with multiple restarts and return diagnostic information about all runs.

This function helps assess the robustness of EM fitting by returning detailed
information about each restart, allowing you to check for convergence to
different local optima.

# Arguments
- `p::PosteriorProbabilityTriplet`: Posterior probabilities from the three models
- `joint_H0::SklarDist`: Joint distribution under the null hypothesis
- `refID::Int`: Index of the reference (bait) protein

# Keywords
- `n_restarts::Int=20`: Number of restarts (up to 20 strategies available)
- All other keywords are passed to `em_fit_mixture`

# Returns
- `DataFrame`: Diagnostic table with columns:
  - `restart`: Restart number
  - `init_π0`: Initial π₀ value
  - `init_method`: Initialization method (:quantile, :kmeans, :random_top20, :empirical_bayes)
  - `final_π0`: Final π₀ value
  - `final_π1`: Final π₁ value
  - `log_likelihood`: Final log-likelihood
  - `iterations`: Number of iterations
  - `converged`: Whether EM converged
  - `status`: "success" or error message

# Initialization Strategies
The function uses 20 different initialization strategies:
- `:quantile` (7 strategies): π₀ ∈ {0.65, 0.70, 0.72, 0.80, 0.88, 0.90, 0.95}
- `:kmeans` (6 strategies): π₀ ∈ {0.50, 0.55, 0.60, 0.75, 0.78, 0.85}
- `:random_top20` (4 strategies): π₀ ∈ {0.70, 0.75, 0.80, 0.85}
- `:empirical_bayes` (3 strategies): π₀ estimated from data via `estimate_prior_empirical_bayes`,
  with perturbations of ±0.05

# Example
```julia
# Run diagnostics with all 20 strategies
diag = em_restart_diagnostics(p, joint_H0, refID)

# Check variability in π₁ estimates
println("π₁ range: ", extrema(diag.final_π1))
println("π₁ std: ", std(diag.final_π1))

# Check log-likelihood spread
println("Log-lik range: ", extrema(diag.log_likelihood))

# Verify empirical_bayes method is included
@assert :empirical_bayes in diag.init_method

# Plot convergence across restarts
using StatsPlots
@df diag scatter(:restart, :log_likelihood, xlabel="Restart", ylabel="Log-likelihood")
```
"""
function em_restart_diagnostics(p::PosteriorProbabilityTriplet,
                                 joint_H0::SklarDist,
                                 refID::Int;
                                 n_restarts::Int = 20,
                                 kwargs...)
    # Compute empirical Bayes estimate for π0 once (used by :empirical_bayes strategies)
    eb_result = estimate_prior_empirical_bayes(p, joint_H0)
    eb_π0 = 1 - eb_result.expected_π1

    # Define initialization strategies (~20 total)
    # Uses a mix of methods and π0 values covering 0.50-0.95 range
    init_strategies = [
        # Original quantile strategies
        (π0 = 0.80, method = :quantile),
        (π0 = 0.70, method = :quantile),
        (π0 = 0.90, method = :quantile),
        (π0 = 0.65, method = :quantile),
        (π0 = 0.95, method = :quantile),
        # Additional quantile coverage
        (π0 = 0.72, method = :quantile),
        (π0 = 0.88, method = :quantile),
        # Original kmeans strategies
        (π0 = 0.75, method = :kmeans),
        (π0 = 0.85, method = :kmeans),
        (π0 = 0.78, method = :kmeans),
        # Additional kmeans with lower π0 coverage
        (π0 = 0.60, method = :kmeans),
        (π0 = 0.55, method = :kmeans),
        (π0 = 0.50, method = :kmeans),
        # Original random_top20 strategies
        (π0 = 0.80, method = :random_top20),
        (π0 = 0.75, method = :random_top20),
        # Additional random_top20 coverage
        (π0 = 0.70, method = :random_top20),
        (π0 = 0.85, method = :random_top20),
        # Empirical Bayes strategies (π0 determined from data)
        (π0 = eb_π0, method = :empirical_bayes),
        (π0 = clamp(eb_π0 + 0.05, 0.0, 1.0), method = :empirical_bayes),
        (π0 = clamp(eb_π0 - 0.05, 0.0, 1.0), method = :empirical_bayes),
    ]

    strategies_to_use = init_strategies[1:min(n_restarts, length(init_strategies))]

    # Initialize results DataFrame
    results = DataFrame(
        restart = Int[],
        init_π0 = Float64[],
        init_method = Symbol[],
        final_π0 = Float64[],
        final_π1 = Float64[],
        log_likelihood = Float64[],
        iterations = Int[],
        converged = Bool[],
        status = String[]
    )

    for (i, strategy) in enumerate(strategies_to_use)
        try
            init_set = get_H1_initialization_set(p; method=strategy.method)
            result = em_fit_mixture(p, joint_H0, refID;
                                    init_π0=strategy.π0,
                                    init_set=init_set,
                                    verbose=false,
                                    kwargs...)

            push!(results, (
                restart = i,
                init_π0 = strategy.π0,
                init_method = strategy.method,
                final_π0 = result.π0,
                final_π1 = result.π1,
                log_likelihood = result.logs[end, :ll],
                iterations = result.logs[end, :iter],
                converged = result.has_converged,
                status = "success"
            ))
        catch e
            push!(results, (
                restart = i,
                init_π0 = strategy.π0,
                init_method = strategy.method,
                final_π0 = NaN,
                final_π1 = NaN,
                log_likelihood = NaN,
                iterations = 0,
                converged = false,
                status = string(e)
            ))
        end
    end

    return results
end

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
    π1s = successful.final_π1

    # Filter out NaN values to ignore failed restarts from statistics
    lls_valid = lls[isfinite.(lls)]
    π1s_valid = π1s[isfinite.(π1s)]

    # Check if we have any valid values after filtering
    if isempty(lls_valid) || isempty(π1s_valid)
        println("⚠️  No valid log-likelihoods or π₁ values found!")
        return (n_successful = n_successful, n_converged = n_converged, is_robust = false)
    end

    best_ll = maximum(lls_valid)
    worst_ll = minimum(lls_valid)
    ll_range = best_ll - worst_ll

    π1_mean = mean(π1s_valid)
    π1_std = std(π1s_valid)
    π1_min, π1_max = extrema(π1s_valid)

    # Robustness criterion: π₁ std < 0.05 and ll_range < 100
    is_robust = π1_std < 0.05 && ll_range < 100

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
            println("   Consider: more restarts, different priors, or data quality check")
        end
        if ll_range >= 100
            println("⚠️  LARGE LOG-LIKELIHOOD SPREAD (range = $(round(ll_range, digits=2)))")
            println("   EM may be converging to different local optima")
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
    π1_mean = mean(successful.final_π1)
    plt2 = StatsPlots.scatter(successful.restart, successful.final_π1,
        xlabel = "Restart", ylabel = "π₁",
        label = nothing, markersize = 6,
        title = "π₁ by Restart"
    )
    StatsPlots.hline!(plt2, [π1_mean], linestyle = :dash, color = :red,
        label = "Mean ($(round(π1_mean, digits=3)))")

    # Plot 3: π₁ vs log-likelihood
    plt3 = StatsPlots.scatter(successful.final_π1, successful.log_likelihood,
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
        logs.iter[2:end], logs.π0[2:end],
        seriestype = :line, legend = true, label = nothing,
        xlabel = "Iteration", ylabel = "π0",
        foreground_color_legend = nothing, background_color_legend = nothing
    )

    plt3 = StatsPlots.plot(
        logs.iter[2:end], logs.π1[2:end],
        seriestype = :line, legend = true, label = nothing,
        xlabel = "Iteration", ylabel = "π1",
        foreground_color_legend = nothing, background_color_legend = nothing
    )

    return StatsPlots.plot(plt1, plt2, plt3, layout = (3,1), size = (600, 600))
end

