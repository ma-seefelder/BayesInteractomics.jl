"""
    Replicate correlation check for input data quality control.

Computes pairwise Spearman correlations between replicates within each
experiment group to detect outlier replicates.
"""

"""
    pairwise_spearman_with_count(mat::Matrix{Union{Missing, Float64}}) -> (Matrix{Float64}, Matrix{Int})

Compute pairwise Spearman correlations and shared protein counts for a
proteins-by-replicates matrix.

Returns:
- `cors`: Pairwise Spearman correlation matrix (NaN where < 3 shared proteins)
- `shared`: Number of shared non-missing proteins per pair (diagonal = non-missing count per replicate)
"""
function pairwise_spearman_with_count(mat::Matrix{Union{Missing, Float64}})
    n_cols = size(mat, 2)
    cors = Matrix{Float64}(undef, n_cols, n_cols)
    shared = Matrix{Int}(undef, n_cols, n_cols)

    for i in 1:n_cols, j in i:n_cols
        mask = .!ismissing.(mat[:, i]) .& .!ismissing.(mat[:, j])
        n_shared = count(mask)
        shared[i, j] = shared[j, i] = n_shared

        if i == j
            cors[i, j] = 1.0
            shared[i, j] = count(!ismissing, mat[:, i])
        elseif n_shared < 3
            cors[i, j] = cors[j, i] = NaN
        else
            x = Float64.(mat[mask, i])
            y = Float64.(mat[mask, j])
            cors[i, j] = cors[j, i] = corspearman(x, y)
        end
    end

    return cors, shared
end

"""
    check_replicate_correlation(data::InteractionData) -> ReplicateCorrelationResult

Check pairwise Spearman correlation between replicates within each experiment group.

Flagging rules:
- `:ok` if min pairwise correlation >= 0.80
- `:warning` if min correlation in [0.60, 0.80)
- `:fail` if min correlation < 0.60

Edge cases:
- Single replicate: always `:ok` (no pairs to compare)
- All pairs have < 3 shared proteins: `:warning` with min_correlation = NaN
"""
function check_replicate_correlation(data::InteractionData)::ReplicateCorrelationResult
    checks = ReplicateCorrelation[]

    for proto_idx in 1:data.no_protocols
        sample_proto = data.samples[proto_idx]
        control_proto = data.controls[proto_idx]

        for exp_idx in 1:getNoExperiments(sample_proto)
            for (group_sym, proto) in ((:sample, sample_proto), (:control, control_proto))
                mat = getExperiment(proto, exp_idx)
                n_cols = size(mat, 2)

                # Single replicate: no pairs to compare
                if n_cols == 1
                    cors = ones(Float64, 1, 1)
                    shared = Matrix{Int}(undef, 1, 1)
                    shared[1, 1] = count(!ismissing, mat[:, 1])
                    push!(checks, ReplicateCorrelation(
                        proto_idx, exp_idx, group_sym,
                        cors, shared, 1, 1.0, :ok
                    ))
                    continue
                end

                cors, shared = pairwise_spearman_with_count(mat)

                # Compute minimum off-diagonal correlation (ignoring NaN)
                off_diag = [cors[i, j] for i in 1:n_cols for j in (i+1):n_cols if !isnan(cors[i, j])]

                if isempty(off_diag)
                    # All pairs have < 3 shared proteins
                    min_cor = NaN
                    flag = :warning
                else
                    min_cor = minimum(off_diag)
                    flag = if min_cor < 0.60
                        :fail
                    elseif min_cor < 0.80
                        :warning
                    else
                        :ok
                    end
                end

                push!(checks, ReplicateCorrelation(
                    proto_idx, exp_idx, group_sym,
                    cors, shared, n_cols, min_cor, flag
                ))
            end
        end
    end

    overall = worst_flag((c.flag for c in checks)...)
    return ReplicateCorrelationResult(checks, overall)
end
