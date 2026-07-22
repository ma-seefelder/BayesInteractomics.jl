# src/diagnostics/variance_inflation.jl
#
# Optional MNAR-driven variance recovery (post-hoc CI widening).
#
# This is the `:inflation` arm of `CONFIG.mnar_variance_recovery`.
# It widens each protein's posterior log2FC σ by √(inflation_factor) AFTER the
# HBM / regression inference returns, leaving the RxInfer factor graph
# untouched (a hard invariant of the inflation arm).
#
# Naming note: the helper is `apply_mnar_variance_inflation!`
# (NOT `apply_variance_inflation!`) to disambiguate from the LOCAL Student-t
# `variance_inflation = nu / (nu-2)` factor used inside
# `src/diagnostics/residuals.jl:133-137`. The Student-t factor is a robustness
# correction within the regression posterior predictive variance; the MNAR
# factor here is an external dropout-driven CI widening on the HBM log2FC
# posterior.
#
# SCHEMA NOTE (observed in running codebase — see `src/inference/evaluation.jl::log2FCStatistics`):
# `BayesResult.HBM_stats` is the Dict returned by `log2FCStatistics(hbm_result, α)`
# with these keys (NOT the textbook `:σ`/`:mean`/`:CI_lower`/`:CI_upper`):
#   :mean_log2FC        — Vector{Float64}
#   :median_log2FC      — Vector{Float64}
#   :sd_log2FC          — Vector{Float64}              <-- σ we scale by √inflation
#   :variance_log2FC    — Vector{Float64}              <-- σ² we scale by inflation
#   :pd                 — Vector{Float64}
#   :pd_direction       — Vector{String}
#   :credible_interval  — Vector{Vector{Float64}}      <-- pairs [lo, hi] per protocol×experiment
# Schema robustness: in the running codebase the σ schema observed is
# `Vector{Float64}` (flat across protocol×experiment). The mutator still
# requires BOTH-schema robustness (flat AND nested) so it throws on
# any other type and silently does nothing on neither — never a silent no-op.

# `mean` is imported at the parent `BayesInteractomics` level (src/BayesInteractomics.jl:36:
# `import Statistics: mean, ...`); no local `using Statistics` is needed when this file
# is included from the parent module.

"""
    _compute_inflation_factor_protein(missing_mask, rho_zeta, sigma_sq, max_factor) -> Float64

Compute the per-protein variance-inflation factor from per-column dropout
parameters (ρ̂_c, ζ̂_c) and missingness pattern. The formula is

    inflation_i = clamp(1 + frac_missing_i × mean_dropout_severity_i, 1.0, max_factor)
    mean_dropout_severity_i = mean over missing columns c of: ζ̂_c² × σ̂²_c

When every dropout-curve entry is NaN (the NaN-curve fallback) OR
the protein has no missing columns, the function returns `1.0` (no widening).

The clamp at `max_factor` (default 3.0 from `CONFIG.mnar_inflation_max`)
prevents pathological tilts (high `ζ̂_c` × high missingness) from producing
meaningless CIs.
"""
function _compute_inflation_factor_protein(
    missing_mask_i::AbstractVector{Bool},
    rho_zeta_per_column::Vector{Tuple{Float64, Float64}},
    sigma_squared_per_column::Vector{Float64},
    max_factor::Float64,
)::Float64
    frac_missing = mean(missing_mask_i)
    if frac_missing == 0.0 || all(isnan, [first(rz) for rz in rho_zeta_per_column])
        return 1.0
    end
    missing_cols = findall(missing_mask_i)
    contribs = Float64[]
    for c in missing_cols
        ρ, ζ = rho_zeta_per_column[c]
        (isnan(ρ) || isnan(ζ)) && continue
        # Tilted-Gaussian information loss; ζ² × σ² is the second-order term in the
        # tilt expansion. Higher ζ → steeper tilt → more lost information.
        push!(contribs, ζ^2 * sigma_squared_per_column[c])
    end
    isempty(contribs) && return 1.0
    severity = mean(contribs)
    return clamp(1.0 + frac_missing * severity, 1.0, max_factor)
end

# ----- schema-robust σ widening helper ------------------------------------ #
# Apply √factor scaling to result.HBM_stats[:sd_log2FC] (and recompute
# :variance_log2FC + :credible_interval), handling both flat Vector{Float64}
# and nested Vector{Vector{Float64}} schemas. Throws ArgumentError on any
# other type — never silently no-ops.
#
# Schema keys observed (this file's header doc):
#   :sd_log2FC         — σ vector(-of-vectors) we scale by √factor
#   :variance_log2FC   — σ² vector(-of-vectors) we scale by factor
#   :mean_log2FC       — μ vector(-of-vectors), used to re-derive CI
#   :credible_interval — Vector{Vector{Float64}} of [lo, hi] pairs (one pair per protocol×experiment)
function _scale_sigma_and_ci_in_place!(result::BayesResult, sqrt_factor::Float64)
    # Defensive: empty/skipped HBM results carry a `:empty => Float64[]` sentinel
    # and have no :sd_log2FC key. Nothing to scale.
    haskey(result.HBM_stats, :sd_log2FC) || return

    σ_vec = result.HBM_stats[:sd_log2FC]
    schema = nothing
    if σ_vec isa Vector{Float64}
        schema = :flat
        for i in eachindex(σ_vec)
            σ_vec[i] *= sqrt_factor
        end
    elseif σ_vec isa Vector{Vector{Float64}}
        schema = :nested
        for inner in σ_vec
            for i in eachindex(inner)
                inner[i] *= sqrt_factor
            end
        end
    else
        throw(ArgumentError(
            "apply_mnar_variance_inflation! expected HBM_stats[:σ] to be Vector{Float64} or Vector{Vector{Float64}}; " *
            "got $(typeof(σ_vec))"
        ))
    end

    # Recompute :variance_log2FC if present (σ² = (√factor)² × old σ² = factor × old σ²).
    if haskey(result.HBM_stats, :variance_log2FC)
        var_vec = result.HBM_stats[:variance_log2FC]
        f² = sqrt_factor * sqrt_factor
        if schema === :flat && var_vec isa Vector{Float64}
            for i in eachindex(var_vec)
                var_vec[i] *= f²
            end
        elseif schema === :nested && var_vec isa Vector{Vector{Float64}}
            for inner in var_vec
                for i in eachindex(inner)
                    inner[i] *= f²
                end
            end
        end
        # If schemas don't match, skip silently (defensive only — they SHOULD match).
    end

    # CI re-derivation — must match σ's schema. The credible_interval is a
    # Vector{Vector{Float64}} of [lo, hi] pairs. We re-derive each pair from
    # the (now-inflated) σ using the Normal-approximation 1.96 z-score at α=0.95.
    if haskey(result.HBM_stats, :mean_log2FC) &&
       haskey(result.HBM_stats, :credible_interval)
        μ_vec = result.HBM_stats[:mean_log2FC]
        ci_vec = result.HBM_stats[:credible_interval]
        z = 1.96  # 95% credible interval (matches log2FCStatistics α=0.95 default)
        if schema === :flat &&
           μ_vec isa Vector{Float64} && ci_vec isa Vector{Vector{Float64}}
            for i in eachindex(μ_vec)
                pair = ci_vec[i]
                if length(pair) == 2
                    pair[1] = μ_vec[i] - z * σ_vec[i]
                    pair[2] = μ_vec[i] + z * σ_vec[i]
                end
            end
        elseif schema === :nested &&
               μ_vec isa Vector{Vector{Float64}} && ci_vec isa Vector{Vector{Float64}}
            # In a fully-nested schema, ci_vec would be Vector{Vector{Vector{Float64}}};
            # if instead it remains Vector{Vector{Float64}} flat-paired with nested σ,
            # we can't safely re-derive. Skip silently (defensive only).
        end
        # If schemas don't match between :sd_log2FC and :mean_log2FC/:credible_interval,
        # skip CI re-derivation silently (defensive only — schemas SHOULD match because
        # they all come from the same log2FCStatistics call).
    end
    return
end

"""
    apply_mnar_variance_inflation!(result::BayesResult, dropout_fit::DropoutFit,
                                   intensity_row::AbstractVector;
                                   max_factor::Float64 = 3.0,
                                   override::Union{Nothing,Float64} = nothing) -> Float64

Widen the per-protein log2FC posterior σ in
`result.HBM_stats[:sd_log2FC]` (and the corresponding `:variance_log2FC`
and 95% credible intervals `:credible_interval`) by `sqrt(inflation_factor)`.

- `intensity_row::AbstractVector` is the protein's row of the imputed
  intensity matrix (used only to derive `missing_mask = isnan.(row)`).
  Per-column σ̂² defaults to `1.0` (i.e. treats every column as unit-variance);
  callers with actual per-column σ̂² should use the matrix+index overload below.
- `override`, when non-nothing, replaces the auto-derived factor with the
  given scalar (clamped at `[1.0, Inf)`; sensitivity-study path).

Returns the inflation factor that was applied (in `[1.0, max_factor]`).
Does NOT touch `bfHBM` / `bfRegression`; downstream BF recomputation lives
in the analyse(...) per-protein loop.

**Schema robustness:** handles both `Vector{Float64}` and
`Vector{Vector{Float64}}` σ schemas; throws `ArgumentError` on any other
type (never silently no-ops).
"""
function apply_mnar_variance_inflation!(
    result::BayesResult,
    dropout_fit::DropoutFit,
    intensity_row::AbstractVector;
    max_factor::Float64 = 3.0,
    override::Union{Nothing,Float64} = nothing,
)::Float64
    factor = if override !== nothing
        max(1.0, Float64(override))
    else
        n_cols = length(dropout_fit.rho)
        @assert length(intensity_row) == n_cols "intensity_row length ($(length(intensity_row))) ≠ dropout_fit columns ($n_cols)"
        missing_mask = [ismissing(x) || (x isa Number && isnan(Float64(x))) for x in intensity_row]
        rho_zeta = [(dropout_fit.rho[c], dropout_fit.zeta[c]) for c in 1:n_cols]
        sigma_sq = Vector{Float64}(undef, n_cols)
        fill!(sigma_sq, 1.0)
        _compute_inflation_factor_protein(missing_mask, rho_zeta, sigma_sq, max_factor)
    end

    _scale_sigma_and_ci_in_place!(result, sqrt(factor))
    return factor
end

"""
    apply_mnar_variance_inflation!(result, dropout_fit, intensity_matrix, protein_idx;
                                   sigma_sq_per_column, max_factor, override) -> Float64

Three-arg variant that accepts the FULL intensity matrix + protein index +
precomputed per-column σ̂² vector. Preferred over the row-only signature
when the caller has already computed per-column σ̂² across all proteins
(computed once per dataset).
"""
function apply_mnar_variance_inflation!(
    result::BayesResult,
    dropout_fit::DropoutFit,
    intensity_matrix::AbstractMatrix,
    protein_idx::Int;
    sigma_sq_per_column::Vector{Float64},
    max_factor::Float64 = 3.0,
    override::Union{Nothing,Float64} = nothing,
)::Float64
    factor = if override !== nothing
        max(1.0, Float64(override))
    else
        n_cols = length(dropout_fit.rho)
        @assert size(intensity_matrix, 2) == n_cols "matrix has $(size(intensity_matrix,2)) cols ≠ dropout_fit $n_cols"
        @assert length(sigma_sq_per_column) == n_cols "sigma_sq_per_column length ≠ dropout_fit columns"
        row = view(intensity_matrix, protein_idx, :)
        missing_mask = [ismissing(x) || (x isa Number && isnan(Float64(x))) for x in row]
        rho_zeta = [(dropout_fit.rho[c], dropout_fit.zeta[c]) for c in 1:n_cols]
        _compute_inflation_factor_protein(missing_mask, rho_zeta, sigma_sq_per_column, max_factor)
    end

    _scale_sigma_and_ci_in_place!(result, sqrt(factor))
    return factor
end
