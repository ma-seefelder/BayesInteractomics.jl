# bb_mnar_codriven per-protein diagnostic flag.
#
# Companion to `BBMnarCodrivenConfig` (in src/diagnostics/types.jl). The struct
# defines the thresholds; this file defines the per-protein flag computation.
# Both are intentionally co-located in src/diagnostics/ for namespace locality.

"""
    _compute_bb_mnar_codriven(bf_detected, bf_combined, missing_fraction, cfg)
        -> Vector{Bool}

Per-protein diagnostic flag: element `i` is `true` when ALL THREE conditions are
strictly above their thresholds:

    bb_mnar_codriven_i := (bf_detected_i  > cfg.bb_bf_threshold)
                       ∧ (bf_combined_i   > cfg.hbm_bf_threshold)
                       ∧ (missing_fraction_i > cfg.missing_fraction_threshold)

`bf_combined` is the post-MNAR BMA combined Bayes factor (the `BF` column on
`copula_results`), NOT the standalone HBM enrichment factor. The boundary
case at exactly the threshold value returns `false` (strict `>`, NOT `>=`).

Missing inputs coalesce to `0.0` so the corresponding inequality fails and
the flag remains `false` (proteins with zero detections have `missing` in
some upstream paths).

Returns a `Vector{Bool}` of length `length(bf_detected)`.
"""
function _compute_bb_mnar_codriven(
    bf_detected::AbstractVector,
    bf_combined::AbstractVector,
    missing_fraction::AbstractVector,
    cfg::BBMnarCodrivenConfig,
)
    n = length(bf_detected)
    @assert length(bf_combined)     == n "bf_combined length mismatch"
    @assert length(missing_fraction) == n "missing_fraction length mismatch"

    flag = Vector{Bool}(undef, n)
    for i in 1:n
        # Strict `>` (NOT `>=`); boundary values do NOT flag.
        bb  = coalesce(bf_detected[i],      0.0) > cfg.bb_bf_threshold
        hbm = coalesce(bf_combined[i],      0.0) > cfg.hbm_bf_threshold
        mf  = coalesce(missing_fraction[i], 0.0) > cfg.missing_fraction_threshold
        flag[i] = bb && hbm && mf
    end
    return flag
end
