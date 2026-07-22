"""
    Missingness asymmetry check for input data quality control.

Detects replicates with disproportionately more missing values than their
peers within the same experiment group.
"""

"""
    check_missingness(data::InteractionData) -> MissingnessResult

Check for missingness asymmetry across replicates within each experiment group.

For each protocol, experiment, and group (sample/control):
- Computes per-replicate missing fraction (missing values / total proteins)
- Computes median missing fraction across replicates
- Computes max_ratio = max(missing_fractions) / median_fraction

Flagging rules:
- `:fail` if max_ratio > 3.0
- `:warning` if max_ratio > 2.0
- `:ok` otherwise

Edge cases:
- Single replicate: always `:ok` (no asymmetry possible)
- All replicates have zero missing: max_ratio = 1.0, flag `:ok`
- Median is zero but some replicates have missing: max_ratio = Inf
"""
function check_missingness(data::InteractionData)::MissingnessResult
    checks = ReplicateMissingness[]

    for proto_idx in 1:data.no_protocols
        sample_proto = data.samples[proto_idx]
        control_proto = data.controls[proto_idx]

        for exp_idx in 1:getNoExperiments(sample_proto)
            for (group_sym, proto) in ((:sample, sample_proto), (:control, control_proto))
                mat = getExperiment(proto, exp_idx)
                n_rows, n_cols = size(mat)

                # Single replicate: no asymmetry possible
                if n_cols == 1
                    frac = n_rows > 0 ? count(ismissing, mat[:, 1]) / n_rows : 0.0
                    push!(checks, ReplicateMissingness(
                        proto_idx, exp_idx, group_sym,
                        [frac], frac, 1.0, :ok
                    ))
                    continue
                end

                # Compute per-replicate missing fractions
                missing_fractions = Float64[]
                for col in 1:n_cols
                    frac = n_rows > 0 ? count(ismissing, mat[:, col]) / n_rows : 0.0
                    push!(missing_fractions, frac)
                end

                med_frac = median(missing_fractions)

                # Compute max_ratio (high-outlier detection only: flags replicates
                # with disproportionately MORE missing values than the median.
                # Low-outlier detection is out of scope -- a replicate with fewer
                # missing values than peers is not considered anomalous.)
                if med_frac == 0.0
                    # If any replicate has missing values while median is zero
                    max_ratio = any(f -> f > 0.0, missing_fractions) ? Inf : 1.0
                else
                    max_ratio = maximum(missing_fractions) / med_frac
                end

                # Asymmetry-ratio thresholds
                flag = if max_ratio > 3.0
                    :fail
                elseif max_ratio > 2.0
                    :warning
                else
                    :ok
                end

                push!(checks, ReplicateMissingness(
                    proto_idx, exp_idx, group_sym,
                    missing_fractions, med_frac, max_ratio, flag
                ))
            end
        end
    end

    overall = worst_flag((c.flag for c in checks)...)
    return MissingnessResult(checks, overall)
end
