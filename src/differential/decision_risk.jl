# src/differential/decision_risk.jl
#
# Bayesian Decision Risk helpers.
#
# Implements:
#   DEFAULT_DIFFERENTIAL_LOSS    — 4×4 asymmetric loss matrix, zero diagonal
#   DECISION_RISK_ACTIONS        — action order vector
#   compute_decision_risk!       — public, in-place; appends 6 columns to df
#   compute_decision_risk        — public, pure; returns NamedTuple
#   _expected_risk_row           — internal, per-row risk integral
#
# Posterior input: γ-PEP columns (pep_gained, pep_reduced, pep_unchanged,
# pep_both_negative), renormalised: P(k) = (1 - pep_k) / Σ_j (1 - pep_j).
# Degenerate (Σ < 1e-12): uniform fallback + one @warn per call (maxlog=1).
#
# Coverage: CONDITION_A_SPECIFIC / CONDITION_B_SPECIFIC rows fall outside the
# 4-action loss matrix → decision_risk = NaN, four risk_<class> = NaN,
# optimal_call = :condition_a_specific (or :condition_b_specific). NaN, NOT Missing,
# keeps column eltype Float64.
#
# The default loss matrix:
#                  truth:gained  truth:reduced  truth:unchanged  truth:bothneg
# action:gained          0            10              3              3
# action:reduced        10             0              3              3
# action:unchanged       5             5              0              1
# action:bothneg         5             5              1              0
#
# Per-cell justification:
#   - Diagonal = 0: correct call costs nothing.
#   - Direction-flip = 10: calling `gained` when truth is `reduced` is the most
#     expensive error (2× missed-hit because reversing a published direction is
#     more costly than missing one entirely).
#   - Over-claim = 3: calling `gained`/`reduced` when truth is `unchanged` wastes
#     follow-up resources but doesn't actively mislead.
#   - Missed-hit = 5: calling `unchanged`/`bothneg` when truth is `gained`/`reduced`
#     permanently buries a real signal.
#   - Conservative-default = 1: within-quadrant slip in the no-interaction region.

import DataFrames: DataFrame
import Statistics: median, quantile
# InteractionClass enum (GAINED, REDUCED, UNCHANGED, BOTH_NEGATIVE,
# CONDITION_A_SPECIFIC, CONDITION_B_SPECIFIC) is already in scope inside
# `module Differential` via `include("differential/types.jl")`. Do NOT
# re-import here.

# ─────────────────────────────────────────────────────────────────────────────
# Default 4×4 asymmetric loss matrix (rows = action, cols = truth state)
# ─────────────────────────────────────────────────────────────────────────────

"""
    DEFAULT_DIFFERENTIAL_LOSS::Matrix{Float64}

Default 4×4 asymmetric loss matrix for Bayesian Decision Risk.

Rows index the candidate ACTION; columns index the underlying TRUTH state. Both axes follow
the order `DECISION_RISK_ACTIONS = [:gained, :reduced, :unchanged, :both_negative]`. The
matrix encodes the relative cost of each (action, truth) pair:

| action ↓ / truth → | `:gained` | `:reduced` | `:unchanged` | `:both_negative` |
|--------------------|-----------|------------|--------------|------------------|
| `:gained`          | 0         | 10         | 3            | 3                |
| `:reduced`         | 10        | 0          | 3            | 3                |
| `:unchanged`       | 5         | 5          | 0            | 1                |
| `:both_negative`   | 5         | 5          | 1            | 0                |

Direction-flip = 10 (2× a missed hit), over-claim a real hit as unchanged = 3, missed-hit
(truth is gained/reduced but we called it unchanged/both_negative) = 5, conservative-default
confusion between `:unchanged` and `:both_negative` = 1. Zero diagonal.

Override at call site via `DifferentialConfig(; loss_matrix = my_custom_matrix)` or the
`loss_matrix=` kwarg to [`differential_analysis`](@ref). The constructor validates the
custom matrix: must be 4×4, zero diagonal, all entries ≥ 0, all entries finite.

See also: [`DECISION_RISK_ACTIONS`](@ref), [`compute_decision_risk!`](@ref),
[`compute_decision_risk`](@ref).
"""
const DEFAULT_DIFFERENTIAL_LOSS::Matrix{Float64} = [
    0.0  10.0  3.0  3.0;
   10.0   0.0  3.0  3.0;
    5.0   5.0  0.0  1.0;
    5.0   5.0  1.0  0.0;
]

"""
    DECISION_RISK_ACTIONS::Vector{Symbol}

Canonical action-order vector for Bayesian Decision Risk.

`[:gained, :reduced, :unchanged, :both_negative]` — the four candidate actions a user can
take in response to per-pair differential evidence. The vector indexes BOTH axes of
[`DEFAULT_DIFFERENTIAL_LOSS`](@ref) (rows = action, cols = truth state) AND the per-row
risk integral `_expected_risk_row(posterior, L)` in `src/differential/decision_risk.jl`.

Order is LOAD-BEARING — `compute_decision_risk!` writes the `optimal_call` column by
`DECISION_RISK_ACTIONS[argmin over the four risk_<class> columns]`. Reordering breaks the
optimal_call ordering contract.

See also: [`DEFAULT_DIFFERENTIAL_LOSS`](@ref), [`compute_decision_risk!`](@ref).
"""
const DECISION_RISK_ACTIONS::Vector{Symbol} = [:gained, :reduced, :unchanged, :both_negative]

# ─────────────────────────────────────────────────────────────────────────────
# Loss-matrix validation — applied at both public entry points
# ─────────────────────────────────────────────────────────────────────────────

"""
    _validate_loss_matrix(L::Matrix{Float64})

Throw `ArgumentError` if `L` is not a 4×4 Float64 matrix with zero diagonal,
non-negative entries, all finite. Called at the top of `compute_decision_risk!`
and `compute_decision_risk` (kwarg overrides bypass the `DifferentialConfig`
constructor validation, so this defensive late-bind check is required).
"""
function _validate_loss_matrix(L::Matrix{Float64})
    if size(L) != (4, 4)
        throw(ArgumentError(
            "loss_matrix must be 4×4 Float64 with zero diagonal, non-negative entries, all finite; " *
            "got size $(size(L)) (expected (4, 4))"))
    end
    for i in 1:4
        if L[i, i] != 0.0
            throw(ArgumentError(
                "loss_matrix must be 4×4 Float64 with zero diagonal, non-negative entries, all finite; " *
                "got nonzero diagonal at L[$i, $i] = $(L[i, i])"))
        end
    end
    for i in 1:4, j in 1:4
        v = L[i, j]
        if !isfinite(v)
            throw(ArgumentError(
                "loss_matrix must be 4×4 Float64 with zero diagonal, non-negative entries, all finite; " *
                "got non-finite entry at L[$i, $j] = $v"))
        end
        if v < 0.0
            throw(ArgumentError(
                "loss_matrix must be 4×4 Float64 with zero diagonal, non-negative entries, all finite; " *
                "got negative entry at L[$i, $j] = $v"))
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal helper — per-row risk integral
# ─────────────────────────────────────────────────────────────────────────────

"""
    _expected_risk_row(posterior::AbstractVector{Float64}, L::Matrix{Float64})
        -> NTuple{4, Float64}

Per-row expected loss integral. Returns `(risk_gained, risk_reduced,
risk_unchanged, risk_both_negative)` where `risk_a = Σ_k L[a, k] * posterior[k]`.

Caller is responsible for ensuring `posterior` is already renormalised (sums to
1.0) and `length(posterior) == 4`. No defensive checks here — internal use only.
"""
function _expected_risk_row(posterior::AbstractVector{Float64}, L::Matrix{Float64})::NTuple{4, Float64}
    r1 = L[1,1]*posterior[1] + L[1,2]*posterior[2] + L[1,3]*posterior[3] + L[1,4]*posterior[4]
    r2 = L[2,1]*posterior[1] + L[2,2]*posterior[2] + L[2,3]*posterior[3] + L[2,4]*posterior[4]
    r3 = L[3,1]*posterior[1] + L[3,2]*posterior[2] + L[3,3]*posterior[3] + L[3,4]*posterior[4]
    r4 = L[4,1]*posterior[1] + L[4,2]*posterior[2] + L[4,3]*posterior[3] + L[4,4]*posterior[4]
    return (r1, r2, r3, r4)
end

# ─────────────────────────────────────────────────────────────────────────────
# Pure public: compute_decision_risk
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_decision_risk(pep_gained, pep_reduced, pep_unchanged, pep_both_negative,
                          classification;
                          loss_matrix::Matrix{Float64} = DEFAULT_DIFFERENTIAL_LOSS)
        -> NamedTuple

Pure (non-mutating) helper. Given four γ-PEP column vectors and
a classification column (`InteractionClass` enum values), compute per-row:

- `optimal_call::Symbol` — Bayes-optimal action minimising expected loss
- `decision_risk::Float64` — expected loss of the optimal action
- `risk_gained, risk_reduced, risk_unchanged, risk_both_negative::Float64`
  — expected loss of each candidate action
- `loss_matrix_default::Bool` — `true` iff `loss_matrix == DEFAULT_DIFFERENTIAL_LOSS`

Returns a `NamedTuple` with these seven keys, each a `Vector{T}` of length `n`.

# Posterior renormalisation
For each row, `P(k) = (1 - pep_k) / Σ_j (1 - pep_j)` over the four states.
If `Σ < 1e-12`, posterior falls back to uniform `[0.25, 0.25, 0.25, 0.25]` and a
single `@warn ... maxlog=1` fires for the call.

# 6-class enum coverage
`CONDITION_A_SPECIFIC` / `CONDITION_B_SPECIFIC` rows fall outside the 4-action
loss matrix:
- `decision_risk = NaN`, all four `risk_<class> = NaN`
- `optimal_call = :condition_a_specific` (or `:condition_b_specific`)
- `loss_matrix_default` still reflects the matrix-vs-default comparison

NaN (not `missing`) keeps the column eltype `Float64` — avoids
`Union{Missing, Float64}` cascading through downstream code.

# Defensive `missing` handling
γ-PEP elements that are `missing` are treated as `1.0` (effectively zero
probability mass in the unnormalised posterior); negative drift from numerical
rounding is clamped to `0.0` before normalisation.
"""
function compute_decision_risk(pep_gained::AbstractVector,
                                pep_reduced::AbstractVector,
                                pep_unchanged::AbstractVector,
                                pep_both_negative::AbstractVector,
                                classification::AbstractVector;
                                loss_matrix::Matrix{Float64} = DEFAULT_DIFFERENTIAL_LOSS)
    _validate_loss_matrix(loss_matrix)

    n = length(pep_gained)
    (length(pep_reduced)       == n) || throw(ArgumentError(
        "compute_decision_risk: pep_reduced length mismatch ($(length(pep_reduced)) vs $n)"))
    (length(pep_unchanged)     == n) || throw(ArgumentError(
        "compute_decision_risk: pep_unchanged length mismatch ($(length(pep_unchanged)) vs $n)"))
    (length(pep_both_negative) == n) || throw(ArgumentError(
        "compute_decision_risk: pep_both_negative length mismatch ($(length(pep_both_negative)) vs $n)"))
    (length(classification)    == n) || throw(ArgumentError(
        "compute_decision_risk: classification length mismatch ($(length(classification)) vs $n)"))

    is_default::Bool = (loss_matrix == DEFAULT_DIFFERENTIAL_LOSS)

    optimal_call_col       = Vector{Symbol}(undef, n)
    decision_risk_col      = Vector{Float64}(undef, n)
    risk_gained_col        = Vector{Float64}(undef, n)
    risk_reduced_col       = Vector{Float64}(undef, n)
    risk_unchanged_col     = Vector{Float64}(undef, n)
    risk_both_negative_col = Vector{Float64}(undef, n)
    loss_default_col       = fill(is_default, n)

    did_warn_degenerate = false

    # accumulate per-row Σ(1 − γ-PEP) for the
    # end-of-call summary `@info`. Unconditional (NOT gated on
    # ENV["GSD_DECISION_RISK_TELEMETRY"]) so the summary always has data; the
    # per-row ENV-gated `@info` at L240-244 below remains the noisy debug
    # channel. Measured 5.08%
    # degenerate << 50% threshold → informational only; no algorithm change.
    # Future production datasets crossing the 50% band will reuse this
    # telemetry signature.
    z_history = Float64[]
    sizehint!(z_history, n)

    @inbounds for i in 1:n
        cls = classification[i]
        # CONDITION_A_SPECIFIC / CONDITION_B_SPECIFIC rows fall outside the
        # 4-action loss matrix — emit NaN sentinels and preserve the MAP enum.
        if cls === CONDITION_A_SPECIFIC
            optimal_call_col[i]       = :condition_a_specific
            decision_risk_col[i]      = NaN
            risk_gained_col[i]        = NaN
            risk_reduced_col[i]       = NaN
            risk_unchanged_col[i]     = NaN
            risk_both_negative_col[i] = NaN
            continue
        elseif cls === CONDITION_B_SPECIFIC
            optimal_call_col[i]       = :condition_b_specific
            decision_risk_col[i]      = NaN
            risk_gained_col[i]        = NaN
            risk_reduced_col[i]       = NaN
            risk_unchanged_col[i]     = NaN
            risk_both_negative_col[i] = NaN
            continue
        end

        # GAINED / REDUCED / UNCHANGED / BOTH_NEGATIVE row
        pg_v  = pep_gained[i]
        pr_v  = pep_reduced[i]
        pu_v  = pep_unchanged[i]
        pbn_v = pep_both_negative[i]

        # Missing → treat as 1.0 (zero unnormalised probability mass)
        pg  = ismissing(pg_v)  ? 1.0 : Float64(pg_v)
        pr  = ismissing(pr_v)  ? 1.0 : Float64(pr_v)
        pu  = ismissing(pu_v)  ? 1.0 : Float64(pu_v)
        pbn = ismissing(pbn_v) ? 1.0 : Float64(pbn_v)

        # Unnormalised joint-state probabilities; clamp ≥ 0 (numerical drift guard).
        u1 = max(0.0, 1.0 - pg)
        u2 = max(0.0, 1.0 - pr)
        u3 = max(0.0, 1.0 - pu)
        u4 = max(0.0, 1.0 - pbn)
        Z  = u1 + u2 + u3 + u4

        # unconditional accumulation for the end-of-call
        # summary roll-up. Excludes CONDITION_A/B_SPECIFIC rows (those `continue`
        # above before reaching this point) — `z_history` reflects only rows
        # that exercise the 4-action loss matrix.
        push!(z_history, Z)

        # debug-flag-gated per-row Σ(1−γ-PEP) telemetry. OFF
        # by default. Enable by setting ENV["GSD_DECISION_RISK_TELEMETRY"] = "1"
        # before running `differential_analysis` / `compute_decision_risk!`. Used
        # to confirm or rule out the degenerate-uniform branch as the cause of
        # all-zero decision_risk for k≥3. Kept as a permanent diagnostic per
        # W1 Gate (zero allocation overhead when ENV var unset — single
        # `get(ENV, ...)` call returning ""). `maxlog=Inf` here (per-row, not
        # throttled — we want the full Σ distribution for diagnosis); the
        # production `@warn` for the degenerate fallback below stays `maxlog=1`.
        if get(ENV, "GSD_DECISION_RISK_TELEMETRY", "") == "1"
            @info "[decision_risk telemetry] Σ(1−γ-PEP)" row=i Σ=Z degenerate=(Z < 1e-12) pep_gained=pg pep_reduced=pr pep_unchanged=pu pep_both_negative=pbn
        end

        local p1::Float64, p2::Float64, p3::Float64, p4::Float64
        if Z < 1e-12
            # degenerate fallback: uniform 0.25 each; one @warn per call.
            if !did_warn_degenerate
                @warn "Decision Risk: degenerate posterior (Σ(1 - γ-PEP) < 1e-12). " *
                      "Falling back to uniform 0.25 each. Check upstream γ-PEP computation." maxlog=1
                did_warn_degenerate = true
            end
            p1 = 0.25
            p2 = 0.25
            p3 = 0.25
            p4 = 0.25
        else
            p1 = u1 / Z
            p2 = u2 / Z
            p3 = u3 / Z
            p4 = u4 / Z
        end

        posterior = (p1, p2, p3, p4)
        (r1, r2, r3, r4) = _expected_risk_row(collect(posterior), loss_matrix)

        risk_gained_col[i]        = r1
        risk_reduced_col[i]       = r2
        risk_unchanged_col[i]     = r3
        risk_both_negative_col[i] = r4

        # argmin over the four risks. `argmin` on a tuple returns the linear index.
        risks_tuple = (r1, r2, r3, r4)
        opt_idx     = argmin(risks_tuple)
        optimal_call_col[i]  = DECISION_RISK_ACTIONS[opt_idx]
        decision_risk_col[i] = risks_tuple[opt_idx]
    end

    # end-of-call summary roll-up. Emits
    # exactly once per `compute_decision_risk!` call (the in-place wrapper
    # delegates the row loop here). `maxlog=1` is the call-site idempotency
    # guard. CONDITION_A/B_SPECIFIC rows are excluded from
    # `z_history` (they short-circuit above), so `n` here = count of rows that
    # exercised the 4-action loss matrix, which may be < length(classification).
    # Measured 5.08% degenerate
    # << 50% threshold → informational only; uniform fallback retained.
    let n_summary = length(z_history),
        n_degen   = count(z -> z < 1e-12, z_history),
        z_finite  = filter(isfinite, z_history)

        if n_summary > 0
            frac_degen = round(n_degen / n_summary; digits = 4)
            Z_min     = isempty(z_finite) ? NaN : minimum(z_finite)
            Z_median  = isempty(z_finite) ? NaN : median(z_finite)
            Z_p95     = isempty(z_finite) ? NaN : quantile(z_finite, 0.95)
            Z_max     = isempty(z_finite) ? NaN : maximum(z_finite)
            @info "[decision_risk summary] n=$(n_summary) degenerate=$(n_degen) frac_degen=$(frac_degen) Z_min=$(Z_min) Z_median=$(Z_median) Z_p95=$(Z_p95) Z_max=$(Z_max)" maxlog=1
        end
    end

    return (
        optimal_call       = optimal_call_col,
        decision_risk      = decision_risk_col,
        risk_gained        = risk_gained_col,
        risk_reduced       = risk_reduced_col,
        risk_unchanged     = risk_unchanged_col,
        risk_both_negative = risk_both_negative_col,
        loss_matrix_default = loss_default_col,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# In-place public: compute_decision_risk!
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_decision_risk!(df::DataFrame; loss_matrix::Matrix{Float64} = DEFAULT_DIFFERENTIAL_LOSS)
        -> DataFrame

Mutates `df` in place by appending the six Decision Risk columns (and the
provenance flag) computed via [`compute_decision_risk`](@ref):

`optimal_call`, `decision_risk`, `risk_gained`, `risk_reduced`,
`risk_unchanged`, `risk_both_negative`, `loss_matrix_default`.

`df` must already carry the γ-PEP columns (`pep_gained`,
`pep_reduced`, `pep_unchanged`, `pep_both_negative`) and a `classification`
column of `InteractionClass` values; missing required columns throw
`ArgumentError`.

Returns the (mutated) `df` for chaining.
"""
function compute_decision_risk!(df::DataFrame; loss_matrix::Matrix{Float64} = DEFAULT_DIFFERENTIAL_LOSS)
    _validate_loss_matrix(loss_matrix)

    required_cols = [:pep_gained, :pep_reduced, :pep_unchanged, :pep_both_negative, :classification]
    missing_cols  = filter(c -> !hasproperty(df, c), required_cols)
    isempty(missing_cols) || throw(ArgumentError(
        "compute_decision_risk!: DataFrame missing required columns: $(missing_cols). " *
        "γ-PEP columns and `classification` must be populated before Decision Risk."))

    nt = compute_decision_risk(
        df.pep_gained, df.pep_reduced, df.pep_unchanged,
        df.pep_both_negative, df.classification;
        loss_matrix = loss_matrix,
    )

    df[!, :optimal_call]        = nt.optimal_call
    df[!, :decision_risk]       = nt.decision_risk
    df[!, :risk_gained]         = nt.risk_gained
    df[!, :risk_reduced]        = nt.risk_reduced
    df[!, :risk_unchanged]      = nt.risk_unchanged
    df[!, :risk_both_negative]  = nt.risk_both_negative
    df[!, :loss_matrix_default] = nt.loss_matrix_default
    return df
end
