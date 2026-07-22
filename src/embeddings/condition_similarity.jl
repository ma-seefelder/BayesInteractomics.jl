# src/embeddings/condition_similarity.jl
# Similarity & Embeddings: condition-level k×k matrices + dendrogram.
# Pattern source: src/differential/analysis.jl (pairwise innerjoin shared-protein pattern).

using DataFrames
using Statistics
using StatsBase

_safe_log10_cond(x::Real) = log10(max(Float64(x), 1e-12))

"""
    _build_condition_extract(ar) ->
        (protein_ids::Vector{String}, bf::Vector{Float64}, log2fc::Vector{Float64},
         posterior::Vector{Float64})

Pull the four per-protein vectors out of an AnalysisResult for downstream k×k computation.
Reads `ar.copula_results` (canonical BMA wiring). `:BF` is the BMA-combined BF.
"""
function _build_condition_extract(ar)
    df = ar.copula_results
    log2fc_col = hasproperty(df, :log2FC) ? :log2FC :
                 hasproperty(df, :log2FC_mean) ? :log2FC_mean :
                 hasproperty(df, :mean_log2FC) ? :mean_log2FC :
                 error("[Embeddings] copula_results must have :log2FC, :log2FC_mean, or :mean_log2FC column")

    ids = String[]
    bf  = Float64[]
    l2  = Float64[]
    pp  = Float64[]
    for i in 1:nrow(df)
        b  = df[i, :BF]
        p  = df[i, :posterior_prob]
        l  = df[i, log2fc_col]
        l_scalar = (l isa AbstractVector) ? (isempty(l) ? missing : mean(skipmissing(l))) : l
        if any(ismissing, (b, p, l_scalar))
            continue
        end
        push!(ids, String(df[i, :Protein]))
        push!(bf,  Float64(b))
        push!(l2,  Float64(l_scalar))
        push!(pp,  Float64(p))
    end
    return (protein_ids = ids, bf = bf, log2fc = l2, posterior = pp)
end

"""
    _pairwise_intersection(e_i, e_j) ->
        (n_shared, log10_bf_i, log10_bf_j, log2fc_i, log2fc_j, pp_i, pp_j)

Returns the four pairs of vectors (each length n_shared) for the intersection of
detected proteins between conditions i and j.
"""
function _pairwise_intersection(e_i, e_j)
    shared = intersect(e_i.protein_ids, e_j.protein_ids)
    if isempty(shared)
        z = Float64[]
        return (0, z, z, z, z, z, z)
    end
    idx_i = indexin(shared, e_i.protein_ids)
    idx_j = indexin(shared, e_j.protein_ids)
    l10_i = [_safe_log10_cond(e_i.bf[k]) for k in idx_i]
    l10_j = [_safe_log10_cond(e_j.bf[k]) for k in idx_j]
    l2_i  = [e_i.log2fc[k] for k in idx_i]
    l2_j  = [e_j.log2fc[k] for k in idx_j]
    pp_i  = [e_i.posterior[k] for k in idx_i]
    pp_j  = [e_j.posterior[k] for k in idx_j]
    return (length(shared), l10_i, l10_j, l2_i, l2_j, pp_i, pp_j)
end

"""
    _top_k_set(e, top_k::Int) -> Set{String}

Top-K ranking: posterior_prob desc, BF desc tiebreak, alphabetical final tiebreak.
Shrinks to actual_k = min(top_k, n_detected) and @info-emits the shrink.
"""
function _top_k_set(e, top_k::Int)
    n = length(e.protein_ids)
    actual_k = min(top_k, n)
    if actual_k < top_k
        @info "[Embeddings] Jaccard@Top-K: condition has only $n proteins (< top_k=$top_k); using actual_k=$actual_k" maxlog=1
    end
    n == 0 && return Set{String}()
    ord = sortperm(collect(zip(e.posterior, e.bf, e.protein_ids));
                   by = t -> (-t[1], -t[2], t[3]))
    return Set(e.protein_ids[ord[1:actual_k]])
end

"""
    _compute_condition_similarity(ars::AbstractVector, cfg::EmbeddingsConfig) ->
        Union{Nothing, ConditionSimilarityResult}

Compute k×k Spearman, Pearson(log2FC), Pearson(posterior), Jaccard@TopK plus the UPGMA
dendrogram for `k = length(ars)` analysis results.

Returns `nothing` when `k < 2` (single-bait degenerate).
"""
function _compute_condition_similarity(ars::AbstractVector, cfg::EmbeddingsConfig)
    k = length(ars)
    k < 2 && return nothing
    !cfg.run_embeddings && return nothing

    # Build per-condition extracts once.
    extracts = [_build_condition_extract(ar) for ar in ars]
    labels   = [getfield(ar, :bait_protein) === nothing ? "cond_$i" :
                String(getfield(ar, :bait_protein))
                for (i, ar) in enumerate(ars)]

    spearman    = Matrix{Float64}(undef, k, k)
    pear_log2fc = Matrix{Float64}(undef, k, k)
    pear_pp     = Matrix{Float64}(undef, k, k)
    jaccard     = Matrix{Float64}(undef, k, k)
    n_shared    = Matrix{Int}(undef, k, k)

    # Top-K sets cached once per condition.
    top_sets = [_top_k_set(extracts[i], cfg.top_k_jaccard) for i in 1:k]

    for i in 1:k, j in i:k
        if i == j
            spearman[i, j]    = 1.0
            pear_log2fc[i, j] = 1.0
            pear_pp[i, j]     = 1.0
            jaccard[i, j]     = 1.0
            n_shared[i, j]    = length(extracts[i].protein_ids)
        else
            (n_int, l10_i, l10_j, l2_i, l2_j, pp_i, pp_j) =
                _pairwise_intersection(extracts[i], extracts[j])
            n_shared[i, j] = n_int
            n_shared[j, i] = n_int
            if n_int < 2
                spearman[i, j]    = 0.0
                pear_log2fc[i, j] = 0.0
                pear_pp[i, j]     = 0.0
            else
                spearman[i, j]    = StatsBase.corspearman(l10_i, l10_j)
                pear_log2fc[i, j] = cor(l2_i, l2_j)
                pear_pp[i, j]     = cor(pp_i, pp_j)
            end
            spearman[j, i]    = spearman[i, j]
            pear_log2fc[j, i] = pear_log2fc[i, j]
            pear_pp[j, i]     = pear_pp[i, j]

            # Jaccard@TopK
            numer = length(intersect(top_sets[i], top_sets[j]))
            denom = length(union(top_sets[i], top_sets[j]))
            jaccard[i, j] = denom == 0 ? 0.0 : numer / denom
            jaccard[j, i] = jaccard[i, j]
        end
    end

    # hclust on D = clamp(1 - spearman, 0, 2). Extension-gated.
    D = clamp.(1.0 .- spearman, 0.0, 2.0)
    merges  = zeros(Int, max(k - 1, 0), 2)
    heights = Float64[]
    order   = collect(1:k)

    if _embeddings_extension_loaded()
        try
            d = fit_condition_clustering(D)
            merges  = Matrix{Int}(d.merges)
            heights = Vector{Float64}(d.heights)
            order   = Vector{Int}(d.order)
        catch e
            @warn "[Embeddings] condition hclust failed: $e" maxlog=1
        end
    end

    return ConditionSimilarityResult(
        labels,
        spearman, pear_log2fc, pear_pp, jaccard,
        n_shared,
        cfg.top_k_jaccard,
        merges, heights, order,
        :average,
    )
end
