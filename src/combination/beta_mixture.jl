#=
Beta Mixture Model for marginal distributions.

Replaces single-Beta marginal fitting with BIC-selected Beta mixture models
(1 to max_K components) for improved capture of multimodal posterior probability
distributions. Uses Distributions.MixtureModel for SklarDist compatibility.

Copyright (C) 2024  Dr. rer. nat. Manuel Seefelder

=#

using Optim: Optim
using SpecialFunctions: digamma, logbeta

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
const _BETA_MIX_PARAM_LB = 0.01   # lower bound for α, β
const _BETA_MIX_PARAM_UB = 1000.0 # upper bound for α, β
const _BETA_MIX_REG      = 1e-4   # L2 regularization strength
const _BETA_MIX_EPS      = 1e-10  # boundary squeeze epsilon

# ──────────────────────────────────────────────────────────────
# Pack / unpack: transform between constrained mixture params
# and unconstrained θ vector for L-BFGS
# ──────────────────────────────────────────────────────────────

"""
    _pack_params(weights, components) -> Vector{Float64}

Pack Beta mixture parameters into an unconstrained vector for L-BFGS.
Layout: [η₁..η_{K-1}, log(α₁), log(β₁), ..., log(α_K), log(β_K)]
where η are log-ratio weights relative to the last component.
"""
function _pack_params(weights::Vector{Float64}, components::Vector{Beta{Float64}})
    K = length(weights)
    θ = Vector{Float64}(undef, 3K - 1)
    if K > 1
        log_w = log.(max.(weights, 1e-15))
        @inbounds for k in 1:(K-1)
            θ[k] = log_w[k] - log_w[K]
        end
    end
    @inbounds for k in 1:K
        α_k, β_k = params(components[k])
        θ[K - 1 + 2(k - 1) + 1] = log(max(α_k, _BETA_MIX_PARAM_LB))
        θ[K - 1 + 2(k - 1) + 2] = log(max(β_k, _BETA_MIX_PARAM_LB))
    end
    return θ
end

"""
    _unpack_params(θ, K) -> (weights, components)

Unpack unconstrained θ vector into mixture weights and Beta components.
"""
function _unpack_params(θ::AbstractVector, K::Int)
    # Weights via softmax
    if K > 1
        η = Vector{Float64}(undef, K)
        @inbounds for k in 1:(K-1)
            η[k] = θ[k]
        end
        η[K] = 0.0
        max_η = maximum(η)
        exp_η = exp.(η .- max_η)
        weights = exp_η ./ sum(exp_η)
    else
        weights = [1.0]
    end

    # Components with clamped parameters
    components = Vector{Beta{Float64}}(undef, K)
    @inbounds for k in 1:K
        α_k = clamp(exp(θ[K - 1 + 2(k - 1) + 1]), _BETA_MIX_PARAM_LB, _BETA_MIX_PARAM_UB)
        β_k = clamp(exp(θ[K - 1 + 2(k - 1) + 2]), _BETA_MIX_PARAM_LB, _BETA_MIX_PARAM_UB)
        components[k] = Beta(α_k, β_k)
    end
    return weights, components
end

# ──────────────────────────────────────────────────────────────
# Negative log-likelihood
# ──────────────────────────────────────────────────────────────

"""
    _negloglik(θ, data, K; λ_reg) -> Float64

Negative log-likelihood of a K-component Beta mixture on `data`,
with L2 regularization on log-shape parameters to prevent degeneracy.
"""
function _negloglik(θ::AbstractVector, data::Vector{Float64}, K::Int;
                    λ_reg::Float64 = _BETA_MIX_REG)
    weights, components = _unpack_params(θ, K)
    n = length(data)
    nll = 0.0
    @inbounds for i in 1:n
        x = data[i]
        # log-sum-exp for numerical stability
        lse = -Inf
        for k in 1:K
            lp = log(weights[k]) + logpdf(components[k], x)
            if lp > lse
                # log-sum-exp accumulation
                lse = lp + log1p(exp(lse - lp))
            else
                lse = lse + log1p(exp(lp - lse))
            end
        end
        nll -= lse
    end
    # L2 regularization on log(α), log(β)
    offset = K > 1 ? K - 1 : 0
    reg = 0.0
    @inbounds for j in (offset + 1):length(θ)
        reg += θ[j]^2
    end
    nll += λ_reg * reg
    return nll
end

"""
    _negloglik_weighted(θ, data, w_norm, K; λ_reg) -> Float64

Weighted negative log-likelihood for EM M-step fitting.
"""
function _negloglik_weighted(θ::AbstractVector, data::Vector{Float64},
                              w_norm::Vector{Float64}, K::Int;
                              λ_reg::Float64 = _BETA_MIX_REG)
    weights, components = _unpack_params(θ, K)
    n = length(data)
    nll = 0.0
    @inbounds for i in 1:n
        x = data[i]
        wi = w_norm[i]
        wi <= 0 && continue
        lse = -Inf
        for k in 1:K
            lp = log(weights[k]) + logpdf(components[k], x)
            if lp > lse
                lse = lp + log1p(exp(lse - lp))
            else
                lse = lse + log1p(exp(lp - lse))
            end
        end
        nll -= wi * lse
    end
    offset = K > 1 ? K - 1 : 0
    reg = 0.0
    @inbounds for j in (offset + 1):length(θ)
        reg += θ[j]^2
    end
    nll += λ_reg * reg
    return nll
end

# ──────────────────────────────────────────────────────────────
# Analytical gradient closures (replaces ForwardDiff)
# ──────────────────────────────────────────────────────────────

"""
    _make_negloglik_fg(data, K; λ_reg) -> fg!

Return a closure `fg!(F, G, θ)` for `Optim.only_fg!` that computes the negative
log-likelihood and its analytical gradient simultaneously.  Uses digamma functions
instead of ForwardDiff for ~5-10× speedup.
"""
function _make_negloglik_fg(data::Vector{Float64}, K::Int;
                            λ_reg::Float64 = _BETA_MIX_REG)
    n = length(data)
    # Precompute log(x) and log(1-x) — constant across optimizer iterations
    log_x  = log.(data)
    log1mx = log.(1.0 .- data)

    # Workspace buffers (captured by closure, reused every call)
    η_buf    = Vector{Float64}(undef, K)
    exp_η    = Vector{Float64}(undef, K)
    w_buf    = Vector{Float64}(undef, K)
    α_buf    = Vector{Float64}(undef, K)
    β_buf    = Vector{Float64}(undef, K)
    ψ_αβ    = Vector{Float64}(undef, K)  # digamma(α_k + β_k)
    ψ_α     = Vector{Float64}(undef, K)  # digamma(α_k)
    ψ_β     = Vector{Float64}(undef, K)  # digamma(β_k)
    logB    = Vector{Float64}(undef, K)  # logbeta(α_k, β_k)
    lp_buf  = Vector{Float64}(undef, K)  # log(w_k * pdf_k(x_i))
    r_buf   = Vector{Float64}(undef, K)  # responsibilities r_{ik}
    clamped_lo = Vector{Bool}(undef, K)   # whether α_k or β_k was clamped low
    clamped_hi_α = Vector{Bool}(undef, K)
    clamped_hi_β = Vector{Bool}(undef, K)
    clamped_lo_α = Vector{Bool}(undef, K)
    clamped_lo_β = Vector{Bool}(undef, K)

    offset = K > 1 ? K - 1 : 0

    function fg!(F, G, θ)
        # ── Unpack weights via softmax ──
        if K > 1
            @inbounds for k in 1:(K-1)
                η_buf[k] = θ[k]
            end
            η_buf[K] = 0.0
            max_η = maximum(@view η_buf[1:K])
            s = 0.0
            @inbounds for k in 1:K
                exp_η[k] = exp(η_buf[k] - max_η)
                s += exp_η[k]
            end
            @inbounds for k in 1:K
                w_buf[k] = exp_η[k] / s
            end
        else
            w_buf[1] = 1.0
        end

        # ── Unpack shape params with clamp tracking ──
        @inbounds for k in 1:K
            raw_α = exp(θ[offset + 2(k-1) + 1])
            raw_β = exp(θ[offset + 2(k-1) + 2])
            α_buf[k] = clamp(raw_α, _BETA_MIX_PARAM_LB, _BETA_MIX_PARAM_UB)
            β_buf[k] = clamp(raw_β, _BETA_MIX_PARAM_LB, _BETA_MIX_PARAM_UB)
            clamped_lo_α[k] = raw_α < _BETA_MIX_PARAM_LB
            clamped_hi_α[k] = raw_α > _BETA_MIX_PARAM_UB
            clamped_lo_β[k] = raw_β < _BETA_MIX_PARAM_LB
            clamped_hi_β[k] = raw_β > _BETA_MIX_PARAM_UB
        end

        # ── Precompute digamma / logbeta per component ──
        @inbounds for k in 1:K
            ψ_α[k]  = digamma(α_buf[k])
            ψ_β[k]  = digamma(β_buf[k])
            ψ_αβ[k] = digamma(α_buf[k] + β_buf[k])
            logB[k]  = logbeta(α_buf[k], β_buf[k])
        end

        need_grad = G !== nothing
        if need_grad
            fill!(G, 0.0)
        end

        nll = 0.0

        @inbounds for i in 1:n
            lxi  = log_x[i]
            l1xi = log1mx[i]

            # ── log p(x_i | k) for each component + log-sum-exp ──
            max_lp = -Inf
            for k in 1:K
                lp_buf[k] = log(w_buf[k]) +
                            (α_buf[k] - 1.0) * lxi +
                            (β_buf[k] - 1.0) * l1xi -
                            logB[k]
                if lp_buf[k] > max_lp
                    max_lp = lp_buf[k]
                end
            end
            lse = 0.0
            for k in 1:K
                lse += exp(lp_buf[k] - max_lp)
            end
            log_mix = max_lp + log(lse)
            nll -= log_mix

            if need_grad
                # ── Responsibilities r_{ik} ──
                for k in 1:K
                    r_buf[k] = exp(lp_buf[k] - log_mix)
                end

                # ── Weight gradients (softmax parameterization) ──
                if K > 1
                    for k in 1:(K-1)
                        G[k] -= (r_buf[k] - w_buf[k])
                    end
                end

                # ── Shape parameter gradients ──
                for k in 1:K
                    r_ik = r_buf[k]
                    # ∂NLL/∂(log α_k): zero if clamped
                    if !(clamped_lo_α[k] || clamped_hi_α[k])
                        G[offset + 2(k-1) + 1] -= r_ik * α_buf[k] *
                            (lxi - ψ_α[k] + ψ_αβ[k])
                    end
                    # ∂NLL/∂(log β_k): zero if clamped
                    if !(clamped_lo_β[k] || clamped_hi_β[k])
                        G[offset + 2(k-1) + 2] -= r_ik * β_buf[k] *
                            (l1xi - ψ_β[k] + ψ_αβ[k])
                    end
                end
            end
        end

        # ── L2 regularization ──
        reg = 0.0
        @inbounds for j in (offset + 1):length(θ)
            reg += θ[j]^2
        end
        nll += λ_reg * reg

        if need_grad
            @inbounds for j in (offset + 1):length(θ)
                G[j] += 2 * λ_reg * θ[j]
            end
        end

        return nll
    end

    return fg!
end

"""
    _make_negloglik_weighted_fg(data, w_norm, K; λ_reg) -> fg!

Weighted version of `_make_negloglik_fg` for EM M-step fitting.
Each data point contribution is scaled by `w_norm[i]`.
"""
function _make_negloglik_weighted_fg(data::Vector{Float64}, w_norm::Vector{Float64},
                                     K::Int; λ_reg::Float64 = _BETA_MIX_REG)
    n = length(data)
    log_x  = log.(data)
    log1mx = log.(1.0 .- data)

    η_buf    = Vector{Float64}(undef, K)
    exp_η    = Vector{Float64}(undef, K)
    w_buf    = Vector{Float64}(undef, K)
    α_buf    = Vector{Float64}(undef, K)
    β_buf    = Vector{Float64}(undef, K)
    ψ_αβ    = Vector{Float64}(undef, K)
    ψ_α     = Vector{Float64}(undef, K)
    ψ_β     = Vector{Float64}(undef, K)
    logB    = Vector{Float64}(undef, K)
    lp_buf  = Vector{Float64}(undef, K)
    r_buf   = Vector{Float64}(undef, K)
    clamped_hi_α = Vector{Bool}(undef, K)
    clamped_hi_β = Vector{Bool}(undef, K)
    clamped_lo_α = Vector{Bool}(undef, K)
    clamped_lo_β = Vector{Bool}(undef, K)

    offset = K > 1 ? K - 1 : 0

    function fg!(F, G, θ)
        # ── Unpack weights ──
        if K > 1
            @inbounds for k in 1:(K-1)
                η_buf[k] = θ[k]
            end
            η_buf[K] = 0.0
            max_η = maximum(@view η_buf[1:K])
            s = 0.0
            @inbounds for k in 1:K
                exp_η[k] = exp(η_buf[k] - max_η)
                s += exp_η[k]
            end
            @inbounds for k in 1:K
                w_buf[k] = exp_η[k] / s
            end
        else
            w_buf[1] = 1.0
        end

        # ── Unpack shape params ──
        @inbounds for k in 1:K
            raw_α = exp(θ[offset + 2(k-1) + 1])
            raw_β = exp(θ[offset + 2(k-1) + 2])
            α_buf[k] = clamp(raw_α, _BETA_MIX_PARAM_LB, _BETA_MIX_PARAM_UB)
            β_buf[k] = clamp(raw_β, _BETA_MIX_PARAM_LB, _BETA_MIX_PARAM_UB)
            clamped_lo_α[k] = raw_α < _BETA_MIX_PARAM_LB
            clamped_hi_α[k] = raw_α > _BETA_MIX_PARAM_UB
            clamped_lo_β[k] = raw_β < _BETA_MIX_PARAM_LB
            clamped_hi_β[k] = raw_β > _BETA_MIX_PARAM_UB
        end

        @inbounds for k in 1:K
            ψ_α[k]  = digamma(α_buf[k])
            ψ_β[k]  = digamma(β_buf[k])
            ψ_αβ[k] = digamma(α_buf[k] + β_buf[k])
            logB[k]  = logbeta(α_buf[k], β_buf[k])
        end

        need_grad = G !== nothing
        if need_grad
            fill!(G, 0.0)
        end

        nll = 0.0

        @inbounds for i in 1:n
            wi = w_norm[i]
            wi <= 0 && continue

            lxi  = log_x[i]
            l1xi = log1mx[i]

            max_lp = -Inf
            for k in 1:K
                lp_buf[k] = log(w_buf[k]) +
                            (α_buf[k] - 1.0) * lxi +
                            (β_buf[k] - 1.0) * l1xi -
                            logB[k]
                if lp_buf[k] > max_lp
                    max_lp = lp_buf[k]
                end
            end
            lse = 0.0
            for k in 1:K
                lse += exp(lp_buf[k] - max_lp)
            end
            log_mix = max_lp + log(lse)
            nll -= wi * log_mix

            if need_grad
                for k in 1:K
                    r_buf[k] = exp(lp_buf[k] - log_mix)
                end

                if K > 1
                    for k in 1:(K-1)
                        G[k] -= wi * (r_buf[k] - w_buf[k])
                    end
                end

                for k in 1:K
                    r_ik = wi * r_buf[k]
                    if !(clamped_lo_α[k] || clamped_hi_α[k])
                        G[offset + 2(k-1) + 1] -= r_ik * α_buf[k] *
                            (lxi - ψ_α[k] + ψ_αβ[k])
                    end
                    if !(clamped_lo_β[k] || clamped_hi_β[k])
                        G[offset + 2(k-1) + 2] -= r_ik * β_buf[k] *
                            (l1xi - ψ_β[k] + ψ_αβ[k])
                    end
                end
            end
        end

        # ── L2 regularization ──
        reg = 0.0
        @inbounds for j in (offset + 1):length(θ)
            reg += θ[j]^2
        end
        nll += λ_reg * reg

        if need_grad
            @inbounds for j in (offset + 1):length(θ)
                G[j] += 2 * λ_reg * θ[j]
            end
        end

        return nll
    end

    return fg!
end

# ──────────────────────────────────────────────────────────────
# Initialization strategies
# ──────────────────────────────────────────────────────────────

"""
    _init_quantile(data, K) -> (weights, components)

Quantile-based initialization: partition data into K equal-probability bins
and fit a single Beta to each bin via method of moments.
"""
function _init_quantile(data::Vector{Float64}, K::Int)
    if K == 1
        return [1.0], [fit_beta_safe(data)]
    end
    sorted = sort(data)
    n = length(sorted)
    weights = fill(1.0 / K, K)
    components = Vector{Beta{Float64}}(undef, K)
    for k in 1:K
        lo = max(1, round(Int, (k - 1) / K * n) + 1)
        hi = min(n, round(Int, k / K * n))
        hi < lo && (hi = lo)
        subset = sorted[lo:hi]
        components[k] = fit_beta_safe(subset)
    end
    return weights, components
end

"""
    _init_random(data, K, rng) -> (weights, components)

Random initialization: sample weights from Dirichlet(1,...,1) and
parameters from LogNormal(0,1).
"""
function _init_random(data::Vector{Float64}, K::Int, rng::AbstractRNG)
    if K == 1
        return [1.0], [fit_beta_safe(data)]
    end
    # Random weights (Dirichlet via Gamma)
    raw_w = [randexp(rng) for _ in 1:K]
    weights = raw_w ./ sum(raw_w)
    # Random components based on data mean with jitter
    μ = clamp(mean(data), 0.05, 0.95)
    components = Vector{Beta{Float64}}(undef, K)
    for k in 1:K
        α = clamp(exp(randn(rng) + log(2.0)), 0.1, 50.0)
        β = clamp(exp(randn(rng) + log(2.0)), 0.1, 50.0)
        components[k] = Beta(α, β)
    end
    return weights, components
end

# ──────────────────────────────────────────────────────────────
# Core fitting: fit_beta_mixture
# ──────────────────────────────────────────────────────────────

"""
    fit_beta_mixture(data; max_K=3, n_starts=6, verbose=false) -> MixtureModel | Beta

Fit a Beta mixture model to data on (0,1) using BIC for model selection.
Tries K=1..max_K components with multiple L-BFGS restarts per K.

Returns a `MixtureModel` (Distributions.jl) that is fully compatible with
`SklarDist` from Copulas.jl. When K=1 is selected, returns a single `Beta`.

# Arguments
- `data`: Vector of values in (0,1)

# Keywords
- `max_K::Int=3`: Maximum number of mixture components
- `n_starts::Int=6`: Number of random restarts per K value
- `verbose::Bool=false`: Print fitting details
"""
function fit_beta_mixture(data::AbstractVector{<:Real}; max_K::Int = 3,
                           n_starts::Int = 6, verbose::Bool = false)
    # Preprocess: filter and squeeze
    x = Float64[v for v in data if isfinite(v)]
    length(x) < 2 && return fit_beta_safe(data)
    x = clamp.(x, _BETA_MIX_EPS, 1.0 - _BETA_MIX_EPS)
    n = length(x)

    # Minimum data requirement: at least 5 observations per free parameter
    # K=1: 2 params, K=2: 5 params, K=3: 8 params
    best_bic = Inf
    best_result = nothing  # (weights, components, K)

    for K in 1:max_K
        n_params = 3K - 1
        n < 5 * n_params && continue

        fg! = _make_negloglik_fg(x, K)

        for s in 1:n_starts
            # Choose initialization strategy
            if s == 1
                # First start: quantile-based (deterministic)
                w_init, c_init = _init_quantile(x, K)
            else
                # Remaining: random
                rng = Random.MersenneTwister(hash((K, s, 42)))
                w_init, c_init = _init_random(x, K, rng)
            end

            θ_init = _pack_params(w_init, c_init)

            try
                result = Optim.optimize(
                    Optim.only_fg!(fg!),
                    θ_init,
                    Optim.LBFGS(),
                    Optim.Options(iterations = 500, g_tol = 1e-8,
                                  show_trace = false)
                )

                nll = Optim.minimum(result)
                isfinite(nll) || continue
                bic = 2 * nll + n_params * log(n)

                if bic < best_bic
                    best_bic = bic
                    w_best, c_best = _unpack_params(Optim.minimizer(result), K)
                    best_result = (w_best, c_best, K)
                end
            catch e
                verbose && @warn "Beta mixture K=$K start=$s failed" exception = e
            end
        end

        verbose && best_result !== nothing && best_result[3] == K &&
            @info "K=$K BIC=$(round(best_bic, digits=2))"
    end

    # Fallback
    if best_result === nothing
        return fit_beta_safe(data)
    end

    w, c, K = best_result

    # Prune negligible components (weight < 0.01)
    keep = findall(w .>= 0.01)
    if length(keep) < K && length(keep) >= 1
        w = w[keep]
        c = c[keep]
        w ./= sum(w)
        K = length(keep)
    end

    # K=1: return plain Beta for maximum backward compatibility
    if K == 1
        return c[1]
    end

    return MixtureModel(c, w)
end

# ──────────────────────────────────────────────────────────────
# Weighted fitting: fit_beta_mixture_weighted (for EM M-step)
# ──────────────────────────────────────────────────────────────

"""
    fit_beta_mixture_weighted(x, w; max_K=3, n_starts=4, prior_mix=nothing)

Fit a Beta mixture model using weighted data (for the EM M-step).
Uses Kish's effective sample size for BIC computation and automatic
fallback to prior when data is insufficient.

# Arguments
- `x::Vector{Float64}`: Data values in (0,1)
- `w::Vector{Float64}`: Non-negative weights (e.g., EM responsibilities)

# Keywords
- `max_K::Int=3`: Maximum components (BIC with n_eff prevents overfitting)
- `n_starts::Int=4`: Number of restarts per K
- `prior_mix`: Fallback distribution when fitting fails
"""
function fit_beta_mixture_weighted(x::Vector{Float64}, w::Vector{Float64};
                                    max_K::Int = 3, n_starts::Int = 4,
                                    prior_mix = nothing)
    fallback = prior_mix !== nothing ? prior_mix : Beta(2.0, 2.0)

    # Preprocess
    x = clamp.(x, _BETA_MIX_EPS, 1.0 - _BETA_MIX_EPS)
    w = clamp.(replace(w, NaN => 0.0, Inf => 0.0, -Inf => 0.0), 0.0, 1.0)
    w_sum = sum(w)
    (!isfinite(w_sum) || w_sum <= 0) && return fallback
    w_norm = w ./ w_sum

    # Effective sample size (Kish)
    n_eff = w_sum^2 / sum(w .^ 2)
    (!isfinite(n_eff) || n_eff < 10) && return fallback

    best_bic = Inf
    best_result = nothing

    for K in 1:max_K
        n_params = 3K - 1
        n_eff < 5 * n_params && continue

        fg! = _make_negloglik_weighted_fg(x, w_norm, K)

        for s in 1:n_starts
            if s == 1
                w_init, c_init = _init_quantile(x, K)
            else
                rng = Random.MersenneTwister(hash((K, s, 99)))
                w_init, c_init = _init_random(x, K, rng)
            end
            θ_init = _pack_params(w_init, c_init)

            try
                result = Optim.optimize(
                    Optim.only_fg!(fg!),
                    θ_init,
                    Optim.LBFGS(),
                    Optim.Options(iterations = 300, g_tol = 1e-6,
                                  show_trace = false)
                )
                nll = Optim.minimum(result)
                isfinite(nll) || continue
                bic = 2 * nll + n_params * log(n_eff)

                if bic < best_bic
                    best_bic = bic
                    w_best, c_best = _unpack_params(Optim.minimizer(result), K)
                    best_result = (w_best, c_best, K)
                end
            catch
            end
        end
    end

    if best_result === nothing
        return fallback
    end

    w_r, c_r, K = best_result
    keep = findall(w_r .>= 0.01)
    if length(keep) < K && length(keep) >= 1
        w_r = w_r[keep]; c_r = c_r[keep]
        w_r ./= sum(w_r); K = length(keep)
    end

    return K == 1 ? c_r[1] : MixtureModel(c_r, w_r)
end

# ──────────────────────────────────────────────────────────────
# Damping for mixture marginals in EM
# ──────────────────────────────────────────────────────────────

"""
    safe_damped_mixture(fit_mix, prev_mix, α_damp) -> Distribution

Damped update for mixture (or single Beta) marginals.
Handles both same-K (component-wise interpolation) and different-K cases.
Falls back gracefully when parameters are invalid.
"""
function safe_damped_mixture(fit_mix, prev_mix, α_damp::Float64)
    # Single Beta fast path (backward compatible with safe_damped_beta)
    if fit_mix isa Beta && prev_mix isa Beta
        return safe_damped_beta(fit_mix, prev_mix, α_damp)
    end

    # Extract component count
    K_fit  = fit_mix isa MixtureModel ? ncomponents(fit_mix) : 1
    K_prev = prev_mix isa MixtureModel ? ncomponents(prev_mix) : 1

    # Validate fit_mix
    if !_valid_mixture(fit_mix)
        return prev_mix
    end
    if !_valid_mixture(prev_mix)
        return fit_mix
    end

    # Same K: component-wise interpolation (sort by mean for alignment)
    if K_fit == K_prev && K_fit > 1
        return _damp_same_k(fit_mix, prev_mix, α_damp)
    end

    # Different K or one is single Beta: use the fitted one
    # (damping is conceptually at the density level via the EM schedule)
    return fit_mix
end

function _valid_mixture(d)
    try
        μ = mean(d)
        return isfinite(μ) && 0 < μ < 1
    catch
        return false
    end
end

function _damp_same_k(fit_mix::MixtureModel, prev_mix::MixtureModel, α_damp::Float64)
    K = ncomponents(fit_mix)
    # Sort both by component mean for alignment
    idx_fit  = sortperm([mean(component(fit_mix, k)) for k in 1:K])
    idx_prev = sortperm([mean(component(prev_mix, k)) for k in 1:K])

    new_comps = Vector{Beta{Float64}}(undef, K)
    new_weights = Vector{Float64}(undef, K)

    for i in 1:K
        kf = idx_fit[i]
        kp = idx_prev[i]

        # Interpolate weights
        wf = probs(fit_mix)[kf]
        wp = probs(prev_mix)[kp]
        new_weights[i] = α_damp * wf + (1 - α_damp) * wp

        # Interpolate Beta parameters
        αf, βf = params(component(fit_mix, kf))
        αp, βp = params(component(prev_mix, kp))
        α_new = max(α_damp * αf + (1 - α_damp) * αp, _BETA_MIX_PARAM_LB)
        β_new = max(α_damp * βf + (1 - α_damp) * βp, _BETA_MIX_PARAM_LB)
        new_comps[i] = Beta(α_new, β_new)
    end

    new_weights ./= sum(new_weights)
    return MixtureModel(new_comps, new_weights)
end

# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

"""
    mixture_ncomponents(d) -> Int

Return the number of components in a distribution (1 for non-mixtures).
"""
mixture_ncomponents(d::MixtureModel) = ncomponents(d)
mixture_ncomponents(d) = 1

"""
    mixture_info_string(d) -> String

Short description of a fitted marginal for diagnostic logging.
"""
function mixture_info_string(d::MixtureModel)
    K = ncomponents(d)
    parts = String[]
    for k in 1:K
        α_k, β_k = params(component(d, k))
        w_k = probs(d)[k]
        push!(parts, "$(round(w_k,digits=2))·Beta($(round(α_k,digits=1)),$(round(β_k,digits=1)))")
    end
    return join(parts, " + ")
end
function mixture_info_string(d::Beta)
    α, β = params(d)
    return "Beta($(round(α,digits=1)),$(round(β,digits=1)))"
end
mixture_info_string(d) = string(typeof(d))
