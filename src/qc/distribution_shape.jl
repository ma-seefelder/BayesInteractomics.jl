"""
    Intensity distribution shape check for input data quality control.

Detects three types of artifacts in intensity distributions:
- Bimodality (platykurtic kurtosis suggests two modes)
- Spike at zero/minimum (zero-inflation artifact)
- Heavy right tail (leptokurtic, suggests non-log or contamination)
"""

"""
    check_intensity_shape(data::InteractionData) -> IntensityShapeResult

Check intensity distribution shape for each replicate across all protocols,
experiments, and groups.

Flagging rules:
- Bimodality: `:warning` if excess kurtosis < -1.2 (advisory only, no `:fail`)
- Spike at zero/min: `:warning` if spike_fraction > 0.20, `:fail` if > 0.40
- Heavy right tail: `:warning` if excess kurtosis > 7.0 (advisory only, no `:fail`)

Skewness is computed and stored but not used for flagging.
Replicates with < 10 non-missing values are skipped (too few for distribution statistics).
"""
function check_intensity_shape(data::InteractionData)::IntensityShapeResult
    checks = IntensityShapeCheck[]

    for proto_idx in 1:data.no_protocols
        sample_proto = data.samples[proto_idx]
        control_proto = data.controls[proto_idx]

        for exp_idx in 1:getNoExperiments(sample_proto)
            for (group_sym, proto) in ((:sample, sample_proto), (:control, control_proto))
                mat = getExperiment(proto, exp_idx)
                n_cols = size(mat, 2)

                for col in 1:n_cols
                    vals = collect(skipmissing(mat[:, col]))

                    # Skip if too few values for distribution statistics
                    if length(vals) < 10
                        continue
                    end

                    n_values = length(vals)

                    # Compute distribution statistics (qualified to avoid import conflicts)
                    excess_kurt = StatsBase.kurtosis(vals)
                    skew_val = StatsBase.skewness(vals)

                    # Bimodality check: platykurtic kurtosis suggests two modes
                    bimodality_flag = excess_kurt < -1.2 ? :warning : :ok

                    # Spike at zero/minimum check
                    min_val = minimum(vals)
                    spike_count = count(v -> v == min_val || v == 0.0, vals)
                    spike_frac = spike_count / n_values

                    spike_flag = if spike_frac > 0.40
                        :fail
                    elseif spike_frac > 0.20
                        :warning
                    else
                        :ok
                    end

                    # Heavy right tail check: strongly leptokurtic
                    tail_flag = excess_kurt > 7.0 ? :warning : :ok

                    # Per-replicate flag is worst of all three
                    flag = worst_flag(bimodality_flag, spike_flag, tail_flag)

                    push!(checks, IntensityShapeCheck(
                        proto_idx, exp_idx, group_sym, col,
                        n_values, excess_kurt, skew_val, spike_frac,
                        bimodality_flag, spike_flag, tail_flag, flag
                    ))
                end
            end
        end
    end

    overall = isempty(checks) ? :ok : worst_flag((c.flag for c in checks)...)
    return IntensityShapeResult(checks, overall)
end
