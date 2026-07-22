"""
    Scale detection check for input data quality control.

Detects whether intensity values appear to be on linear rather than log scale
by checking maximum values across all protocols, experiments, and groups.
"""

"""
    check_scale(data::InteractionData) -> ScaleCheckResult

Check whether intensity data appears to be on log scale.

Iterates all protocols, experiments, and both sample/control groups.
Flags `:warning` if max intensity > 1000 (suggests linear scale),
otherwise `:ok`. There is no `:fail` threshold for scale detection.

Handles missing values via `skipmissing` and guards against empty matrices.
"""
function check_scale(data::InteractionData)::ScaleCheckResult
    protocol_results = ProtocolScaleCheck[]

    for proto_idx in 1:data.no_protocols
        max_val = -Inf
        sample_proto = data.samples[proto_idx]
        control_proto = data.controls[proto_idx]

        for exp_idx in 1:getNoExperiments(sample_proto)
            sample_mat = getExperiment(sample_proto, exp_idx)
            control_mat = getExperiment(control_proto, exp_idx)

            for mat in (sample_mat, control_mat)
                vals = skipmissing(mat)
                if !isempty(vals)
                    max_val = max(max_val, maximum(vals))
                end
            end
        end

        # If max_val is still -Inf (all matrices empty), treat as :ok
        if max_val == -Inf
            max_val = 0.0
        end

        flag = max_val > 1000 ? :warning : :ok  # no :fail threshold for scale detection
        push!(protocol_results, ProtocolScaleCheck(proto_idx, max_val, flag))
    end

    overall = worst_flag((r.flag for r in protocol_results)...)
    return ScaleCheckResult(protocol_results, overall)
end
