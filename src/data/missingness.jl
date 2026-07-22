"""
    _compute_per_protein_missingness(raw_data::InteractionData) -> Vector{Float64}

Per-protein fraction of missing sample observations, aggregated across all
protocols and sample experiments. Computed BEFORE imputation. Returns
NaN-free `Vector{Float64}` of length `n_proteins`; proteins with zero
sample slots → 1.0 (guarded against division by zero).

Mirrors the `count_detections` aggregation idiom at
`src/inference/betabernoulli.jl:128-148` (and the canonical
`for p in 1:data.no_protocols; sp = data.samples[p]; for e in
1:sp.no_experiments; mat = sp.data[e]` walk at
`src/analysis/pipeline.jl:192-216`) but counts `ismissing` slots instead
of detections and divides by total sample slots.

Provides the `missing_fraction` column on `copula_results` that feeds the
`bb_mnar_codriven` diagnostic flag. Controls are intentionally excluded —
only the SAMPLE side enters the diagnostic.
"""
function _compute_per_protein_missingness(raw_data::InteractionData)::Vector{Float64}
    n_proteins = length(raw_data.protein_IDs)
    missing_frac = Vector{Float64}(undef, n_proteins)

    for idx in 1:n_proteins
        total_slots = 0
        n_missing   = 0
        # Walk all protocols' SAMPLE matrices (controls intentionally excluded)
        for p in 1:raw_data.no_protocols
            sample_protocol = raw_data.samples[p]
            for e in 1:sample_protocol.no_experiments
                mat = sample_protocol.data[e]   # (n_proteins, n_replicates) Matrix{Union{Missing,Float64}}
                row = @view mat[idx, :]
                total_slots += length(row)
                n_missing   += count(ismissing, row)
            end
        end
        missing_frac[idx] = total_slots == 0 ? 1.0 : n_missing / total_slots
    end
    return missing_frac
end
