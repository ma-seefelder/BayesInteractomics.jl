# ext/BayesInteractomicsMetalearnerExt/metalearner_calibration.jl
#
# Metalearner-level recalibration: a post-hoc
# calibration map that wraps the self-contained MLJ.Stack output so the shipped
# artefact emits CALIBRATED P(true). The metalearner output is consumed as
# evidence/prior in the Bayesian integration UPSTREAM of the pipeline's own
# Platt step, so the artefact itself must be well-calibrated.
#
# `MetalearnerCalibrator` supports two methods, both pure-Julia + serialisable as
# plain fields into the `<artefact>.meta.jld2` sidecar (no Optim / external model
# objects to deserialise):
#
#   :platt    — calibrated = logistic(a · logit(raw) + b). 2-param logit-space
#               logistic, fit by Newton on the binary cross-entropy. Cheap,
#               smooth, but limited flexibility (can't fix non-monotone-in-logit
#               miscalibration).
#   :isotonic — calibrated = piecewise-constant non-decreasing step function fit
#               by Pool-Adjacent-Violators (PAV) on (raw, label) pairs. More
#               flexible; reliably closes large ECE gaps. Stored as sorted
#               breakpoints `x_knots` + fitted values `y_knots`; applied by
#               right-continuous interpolation with flat extrapolation.
#
# `:identity` is the no-op fallback (a=1,b=0 / single knot) used for legacy
# artefacts so the :legacy_8feat byte-identical contract is never touched.

# ---- Struct (plain fields → JLD2-serialisable in the sidecar) ----------------
struct MetalearnerCalibrator
    method::Symbol               # :platt | :isotonic | :identity
    # Platt params (used when method === :platt)
    a::Float64
    b::Float64
    # Isotonic breakpoints (used when method === :isotonic); empty otherwise
    x_knots::Vector{Float64}
    y_knots::Vector{Float64}
    n_training::Int
end

# Identity calibrator — never alters the input (legacy / fallback).
_identity_calibrator() = MetalearnerCalibrator(:identity, 1.0, 0.0, Float64[], Float64[], 0)

# ---- Numerics ----------------------------------------------------------------
_logit(p::Real)    = (q = clamp(float(p), 1e-12, 1 - 1e-12); log(q / (1 - q)))
_logistic(z::Real) = 1 / (1 + exp(-z))

"""
    _per_bin_ece(probs, labels; nbins=10) -> Float64

Per-bin ECE, 10 equal-FREQUENCY bins, size-weighted — identical recipe to the
held-out 50k-validation `per_bin_ece` so the in-fit ECE matches the gate.
`labels ∈ {0,1}`.
"""
function _per_bin_ece(probs::AbstractVector{<:Real}, labels::AbstractVector{<:Real}; nbins::Int = 10)
    n = length(probs)
    n == 0 && return 0.0
    order = sortperm(probs)
    p = collect(float.(probs))[order]
    y = collect(float.(labels))[order]
    ece = 0.0
    start = 1
    for b in 1:nbins
        stop = round(Int, b * n / nbins)
        stop < start && continue
        idx = start:stop
        bin_p = sum(@view p[idx]) / length(idx)
        bin_y = sum(@view y[idx]) / length(idx)
        ece += (length(idx) / n) * abs(bin_p - bin_y)
        start = stop + 1
    end
    return ece
end

# ---- Platt fit (Newton on logit-space logistic BCE) --------------------------
function _fit_platt(raw::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(raw)
    n == 0 && return (1.0, 0.0)
    x = _logit.(raw)
    yf = float.(y)
    a, b = 1.0, 0.0
    for _ in 1:200
        # Gradient + Hessian of mean BCE w.r.t. (a, b).
        g1 = 0.0; g2 = 0.0
        h11 = 0.0; h12 = 0.0; h22 = 0.0
        for i in 1:n
            z = a * x[i] + b
            p = _logistic(z)
            r = p - yf[i]
            w = p * (1 - p)
            g1 += r * x[i]; g2 += r
            h11 += w * x[i] * x[i]; h12 += w * x[i]; h22 += w
        end
        # Ridge for numerical stability of the 2×2 solve.
        h11 += 1e-9; h22 += 1e-9
        det = h11 * h22 - h12 * h12
        abs(det) < 1e-18 && break
        da = (g1 * h22 - g2 * h12) / det
        db = (h11 * g2 - h12 * g1) / det
        a -= da; b -= db
        (abs(da) + abs(db)) < 1e-10 && break
    end
    return (a, b)
end

# ---- Isotonic fit (regularised Pool-Adjacent-Violators) ----------------------
# Returns sorted x breakpoints + the PAV-fitted non-decreasing y values.
#
# REGULARISATION: an unregularised PAV on the raw (raw, y)
# pairs over-fragments on small calibration sets (the val split, n≈3.7k) — the
# step function memorises val and generalises poorly to test (val ECE 0.003 but
# test ECE 0.036). To regularise, we first AGGREGATE the data into `nbins` equal-
# frequency bins (each bin → mean raw x, mean label y, count), then run PAV on
# the `nbins` bin centroids with count weights. Fewer, count-weighted blocks ⇒
# a smoother map that transfers to the test split. `nbins` is capped so each bin
# holds ≥ `min_per_bin` points.
function _fit_isotonic(raw::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                       nbins::Int = 50, min_per_bin::Int = 30)
    n = length(raw)
    n == 0 && return (Float64[], Float64[])
    order = sortperm(collect(float.(raw)))
    xs = collect(float.(raw))[order]
    ys = collect(float.(y))[order]

    # Aggregate into ≤ nbins equal-frequency bins (≥ min_per_bin each).
    k = max(1, min(nbins, fld(n, max(min_per_bin, 1))))
    bx = Float64[]; by = Float64[]; bw = Int[]
    start = 1
    for b in 1:k
        stop = b == k ? n : round(Int, b * n / k)
        stop < start && continue
        idx = start:stop
        push!(bx, sum(@view xs[idx]) / length(idx))
        push!(by, sum(@view ys[idx]) / length(idx))
        push!(bw, length(idx))
        start = stop + 1
    end

    # Weighted PAV on the bin centroids: blocks of (value, weight, x-right-edge),
    # merge while monotonicity is violated. Weighted mean per block.
    blk_val = Float64[]; blk_w = Int[]; blk_xend = Float64[]
    for i in eachindex(by)
        push!(blk_val, by[i]); push!(blk_w, bw[i]); push!(blk_xend, bx[i])
        while length(blk_val) > 1 && blk_val[end-1] > blk_val[end]
            w = blk_w[end-1] + blk_w[end]
            v = (blk_val[end-1] * blk_w[end-1] + blk_val[end] * blk_w[end]) / w
            xe = blk_xend[end]
            pop!(blk_val); pop!(blk_w); pop!(blk_xend)
            blk_val[end] = v; blk_w[end] = w; blk_xend[end] = xe
        end
    end
    # Right-continuous step: each block's right-edge x is the knot, block value y.
    return (copy(blk_xend), copy(blk_val))
end

# ---- Apply -------------------------------------------------------------------
"""
    apply_calibrator(cal::MetalearnerCalibrator, p) -> Float64

Map a single raw Stack P(true) → calibrated P(true), clamped to [1e-9, 1-1e-9].
`:identity` returns the input unchanged (modulo clamp).
"""
function apply_calibrator(cal::MetalearnerCalibrator, p::Real)::Float64
    pf = float(p)
    if cal.method === :identity
        return pf
    elseif cal.method === :platt
        return clamp(_logistic(cal.a * _logit(pf) + cal.b), 1e-9, 1 - 1e-9)
    elseif cal.method === :isotonic
        isempty(cal.x_knots) && return pf
        # Right-continuous step: smallest knot with x_knot >= p. Flat extrapolation.
        if pf <= cal.x_knots[1]
            return clamp(cal.y_knots[1], 1e-9, 1 - 1e-9)
        elseif pf >= cal.x_knots[end]
            return clamp(cal.y_knots[end], 1e-9, 1 - 1e-9)
        end
        # binary search for first knot >= pf
        lo, hi = 1, length(cal.x_knots)
        while lo < hi
            mid = (lo + hi) >>> 1
            if cal.x_knots[mid] < pf
                lo = mid + 1
            else
                hi = mid
            end
        end
        return clamp(cal.y_knots[lo], 1e-9, 1 - 1e-9)
    else
        return pf
    end
end

apply_calibrator(cal::MetalearnerCalibrator, p::AbstractVector) =
    [apply_calibrator(cal, x) for x in p]

# ---- Fit (Platt-first, isotonic-fallback; selected on a held-out split) ------
"""
    fit_metalearner_calibrator(raw_fit, y_fit, raw_eval, y_eval; ece_target=0.035)
        -> (cal::MetalearnerCalibrator, info::NamedTuple)

Fit a calibration map on `(raw_fit, y_fit)` (data the Stack did NOT train on),
then SELECT the method by per-bin ECE on the independent `(raw_eval, y_eval)`
hold-out:

- Fit Platt; if its eval ECE ≤ `ece_target`, keep Platt.
- Otherwise fit isotonic; keep whichever (Platt vs isotonic) gives the lower eval
  ECE (isotonic almost always wins on large gaps).

Returns the chosen calibrator + an `info` NamedTuple with the raw/Platt/isotonic
eval ECEs and the chosen method, for the SUMMARY + the sidecar provenance.
"""
function fit_metalearner_calibrator(raw_fit::AbstractVector{<:Real}, y_fit::AbstractVector{<:Real},
                                    raw_eval::AbstractVector{<:Real}, y_eval::AbstractVector{<:Real};
                                    ece_target::Float64 = 0.035)
    ece_raw = _per_bin_ece(raw_eval, y_eval)

    a, b = _fit_platt(raw_fit, y_fit)
    platt = MetalearnerCalibrator(:platt, a, b, Float64[], Float64[], length(raw_fit))
    ece_platt = _per_bin_ece(apply_calibrator(platt, raw_eval), y_eval)

    if ece_platt <= ece_target
        return platt, (chosen = :platt, ece_raw = ece_raw, ece_platt = ece_platt,
                       ece_isotonic = NaN, ece_target = ece_target)
    end

    xk, yk = _fit_isotonic(raw_fit, y_fit)
    iso = MetalearnerCalibrator(:isotonic, 1.0, 0.0, xk, yk, length(raw_fit))
    ece_iso = _per_bin_ece(apply_calibrator(iso, raw_eval), y_eval)

    chosen, cal = ece_iso <= ece_platt ? (:isotonic, iso) : (:platt, platt)
    return cal, (chosen = chosen, ece_raw = ece_raw, ece_platt = ece_platt,
                 ece_isotonic = ece_iso, ece_target = ece_target)
end
