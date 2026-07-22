# ============================================================================
# Group-disjoint cross-validation for the production MLJ.Stack blender ONLY.
#
# A hand-rolled, DEPENDENCY-FREE fold builder. Uses only `Base` +
# `Statistics`/`LinearAlgebra` (NO Clustering.jl / Distances.jl / Flux) so the
# file `include`s cleanly in the in-process test and adds NO
# supply-chain surface (no package-legitimacy checkpoint required).
#
# Why group-disjoint CV: the StratifiedCV out-of-fold (OOF) adjudication inside
# `MLJ.Stack` can leak near-duplicate rows across folds (two rows with almost
# identical 14/15-dim input features landing one in train, one in test of the
# same fold). The blender then trains on leakage-inflated OOF estimates. Grouping
# rows by input-feature similarity and assigning WHOLE groups to a single fold
# makes the OOF folds group-disjoint, removing that leakage channel.
#
# Public API:
#
#   assign_groups(X::AbstractMatrix; k, seed, max_iter) -> Vector{Int}
#       One integer group label per ROW of X, via a deterministic hand-written
#       k-means over the column-standardised input matrix.
#
#   group_disjoint_folds(groups::AbstractVector{<:Integer},
#                        [y::AbstractVector];
#                        nfolds, group_centroids) -> Vector{Tuple{Vector{Int},Vector{Int}}}
#       Assigns whole groups to single test folds (round-robin by descending
#       group size for balance). For each fold: test = rows whose group is
#       assigned to that fold, train = complement. Optional `y` triggers the
#       single-class fallback: any test fold missing a class has its
#       smallest cluster merged into the nearest-centroid neighbouring fold.
# ============================================================================

using Statistics: mean, std
using LinearAlgebra: norm
import Random

# ----------------------------------------------------------------------------
# Column standardisation — zero-mean / unit-std per feature so the Euclidean
# distance used by the hand-rolled k-means is not dominated by high-variance
# columns. A near-constant column (std ≈ 0) is left unscaled (divide by 1) to
# avoid blow-up.
# ----------------------------------------------------------------------------
function _standardise(X::AbstractMatrix{<:Real})
    n, d = size(X)
    Xs = Matrix{Float64}(undef, n, d)
    @inbounds for j in 1:d
        col = @view X[:, j]
        μ = mean(col)
        σ = std(col)
        σ = (σ > 1e-12) ? σ : 1.0
        for i in 1:n
            Xs[i, j] = (col[i] - μ) / σ
        end
    end
    return Xs
end

# ----------------------------------------------------------------------------
# assign_groups — deterministic hand-rolled k-means (Lloyd's algorithm) over the
# standardised rows of X. Returns one group label in 1:k_eff per row.
#
# `k` defaults to a small multiple of a typical fold count so that
# `group_disjoint_folds` has more groups than folds to distribute (round-robin
# balance needs k ≫ nfolds). k is clamped to `[1, n]`.
#
# Determinism: a fixed `seed` plus a deterministic k-means++ style seeding
# (farthest-point after a seeded first centre) — no global RNG state leak.
# ----------------------------------------------------------------------------
function assign_groups(X::AbstractMatrix{<:Real};
                       k::Integer = 0,
                       seed::Integer = 42,
                       max_iter::Integer = 50)
    n = size(X, 1)
    n == 0 && return Int[]

    # Default k: a small multiple of 5 (the typical STACK_NFOLDS), clamped to n.
    k_eff = k > 0 ? Int(k) : min(n, max(2, 3 * 5))
    k_eff = clamp(k_eff, 1, n)

    Xs = _standardise(X)

    # ---- k-means++-style deterministic seeding (farthest-point) ----
    rng = Random.MersenneTwister(seed)
    centres = Vector{Vector{Float64}}()
    first_idx = (Random.rand(rng, 1:n))
    push!(centres, Vector{Float64}(@view Xs[first_idx, :]))
    while length(centres) < k_eff
        # distance of each row to its nearest existing centre
        best_d = fill(Inf, n)
        @inbounds for i in 1:n
            ri = @view Xs[i, :]
            for c in centres
                dd = 0.0
                @inbounds for j in eachindex(ri)
                    δ = ri[j] - c[j]
                    dd += δ * δ
                end
                best_d[i] = min(best_d[i], dd)
            end
        end
        # pick the farthest row as the next centre (deterministic; ties → first)
        nxt = argmax(best_d)
        # if the farthest point is already at distance 0 (duplicates dominate),
        # fall back to a seeded random unused index to avoid degenerate centres
        if best_d[nxt] <= 0.0
            nxt = Random.rand(rng, 1:n)
        end
        push!(centres, Vector{Float64}(@view Xs[nxt, :]))
    end

    # ---- Lloyd iterations ----
    labels = ones(Int, n)
    for _ in 1:max_iter
        changed = false
        # assignment step
        @inbounds for i in 1:n
            ri = @view Xs[i, :]
            best_k, best_dd = 1, Inf
            for (ci, c) in enumerate(centres)
                dd = 0.0
                @inbounds for j in eachindex(ri)
                    δ = ri[j] - c[j]
                    dd += δ * δ
                end
                if dd < best_dd
                    best_dd = dd
                    best_k = ci
                end
            end
            if labels[i] != best_k
                labels[i] = best_k
                changed = true
            end
        end
        # update step
        d = size(Xs, 2)
        sums = [zeros(Float64, d) for _ in 1:k_eff]
        counts = zeros(Int, k_eff)
        @inbounds for i in 1:n
            li = labels[i]
            counts[li] += 1
            ri = @view Xs[i, :]
            s = sums[li]
            for j in 1:d
                s[j] += ri[j]
            end
        end
        @inbounds for ci in 1:k_eff
            if counts[ci] > 0
                centres[ci] = sums[ci] ./ counts[ci]
            end
            # empty cluster: leave its centre where it was (will simply attract
            # no points; harmless for the downstream fold builder)
        end
        changed || break
    end

    # ---- Relabel to a contiguous 1:m range over the groups that are non-empty,
    # so downstream round-robin sees dense labels. ----
    present = sort(unique(labels))
    remap = Dict(g => i for (i, g) in enumerate(present))
    return Int[remap[l] for l in labels]
end

# ----------------------------------------------------------------------------
# Centroid of each group in standardised space — used by the single-class
# fallback to find the "nearest neighbouring fold" for a merge.
# ----------------------------------------------------------------------------
function _group_centroids(Xs::AbstractMatrix{<:Real}, groups::AbstractVector{<:Integer})
    gids = sort(unique(groups))
    d = size(Xs, 2)
    cents = Dict{Int,Vector{Float64}}()
    for g in gids
        idx = findall(==(g), groups)
        c = zeros(Float64, d)
        @inbounds for i in idx
            ri = @view Xs[i, :]
            for j in 1:d
                c[j] += ri[j]
            end
        end
        cents[g] = c ./ max(length(idx), 1)
    end
    return cents
end

# ----------------------------------------------------------------------------
# group_disjoint_folds — assign WHOLE groups to single test folds.
#
# Round-robin by descending group size: the largest group goes to fold 1, next
# to fold 2, … wrapping around, so each fold receives a balanced row count and
# NO group is split across folds. For each fold, test = rows of the groups it
# owns; train = every other row. This guarantees:
#   - partition per fold (train ∪ test == 1:n, train ∩ test == ∅)
#   - group-disjointness (a group lives wholly in exactly one test fold, hence
#     never spans train+test of that fold)
#
# Single-class fallback (only when `y` is supplied): after assignment,
# any test fold missing a class has its SMALLEST owned group re-assigned to the
# nearest-centroid neighbouring fold, then the fold class-presence is re-checked.
# `group_centroids` (a Dict group→centroid in the same space used to cluster)
# enables the nearest-neighbour merge; pass it from the caller when `y` is used.
# ----------------------------------------------------------------------------
function group_disjoint_folds(groups::AbstractVector{<:Integer},
                              y::Union{Nothing,AbstractVector} = nothing;
                              nfolds::Integer = 5,
                              group_centroids::Union{Nothing,AbstractDict} = nothing)
    n = length(groups)
    n == 0 && return Tuple{Vector{Int},Vector{Int}}[]
    nf = clamp(Int(nfolds), 1, n)

    gids = sort(unique(groups))
    # rows per group + group sizes
    rows_of = Dict(g => findall(==(g), groups) for g in gids)
    # sort groups by descending size for round-robin balance (ties → group id)
    ordered = sort(gids; by = g -> (-length(rows_of[g]), g))

    # fold_of_group[g] = which fold (1:nf) owns group g
    fold_of_group = Dict{Int,Int}()
    for (i, g) in enumerate(ordered)
        fold_of_group[g] = ((i - 1) % nf) + 1
    end

    # ---- single-class fallback (requires y) ----
    if y !== nothing
        @assert length(y) == n "y length ($(length(y))) must match groups length ($n)"
        # iterate a bounded number of repair passes
        for _pass in 1:(nf + length(gids))
            # find a fold whose test rows are single-class
            offending = 0
            for f in 1:nf
                test_rows = _fold_test_rows(groups, fold_of_group, f)
                isempty(test_rows) && continue
                if length(unique(@view y[test_rows])) < 2
                    offending = f
                    break
                end
            end
            offending == 0 && break  # all folds carry ≥2 classes

            # groups owned by the offending fold, smallest first
            owned = [g for g in gids if fold_of_group[g] == offending]
            isempty(owned) && break
            sort!(owned; by = g -> (length(rows_of[g]), g))
            smallest = owned[1]

            # nearest neighbouring fold by centroid distance (if centroids given);
            # otherwise round-robin to the next fold.
            target_fold = _nearest_fold(smallest, offending, fold_of_group, gids,
                                        rows_of, group_centroids, nf)
            target_fold == offending && break  # nowhere to move → give up gracefully
            fold_of_group[smallest] = target_fold
        end
    end

    # ---- materialise folds ----
    folds = Vector{Tuple{Vector{Int},Vector{Int}}}(undef, nf)
    for f in 1:nf
        test_rows = sort(_fold_test_rows(groups, fold_of_group, f))
        train_rows = sort(setdiff(1:n, test_rows))
        folds[f] = (Vector{Int}(train_rows), Vector{Int}(test_rows))
    end
    return folds
end

# rows belonging to groups owned by fold `f`
function _fold_test_rows(groups::AbstractVector{<:Integer},
                         fold_of_group::AbstractDict, f::Integer)
    rows = Int[]
    @inbounds for i in eachindex(groups)
        if get(fold_of_group, Int(groups[i]), 0) == f
            push!(rows, i)
        end
    end
    return rows
end

# choose the destination fold for a merged group: nearest other fold by centroid
# distance when centroids are available; otherwise the next fold cyclically.
function _nearest_fold(g::Integer, src_fold::Integer, fold_of_group::AbstractDict,
                       gids, rows_of, group_centroids, nf::Integer)
    if group_centroids === nothing || !haskey(group_centroids, g)
        return (src_fold % nf) + 1
    end
    cg = group_centroids[g]
    best_fold, best_d = src_fold, Inf
    for f in 1:nf
        f == src_fold && continue
        # mean distance from g's centroid to the centroids of groups already in f.
        # Exclude group g ITSELF (a group id), then keep groups currently assigned
        # to fold f. Comparing `fold_of_group[h] == g` would compare a FOLD index
        # against a GROUP id (wrong domain).
        members = [h for h in gids if h == g ? false : (fold_of_group[h] == f)]
        isempty(members) && continue
        dd = 0.0
        cnt = 0
        for h in members
            haskey(group_centroids, h) || continue
            dd += norm(cg .- group_centroids[h])
            cnt += 1
        end
        cnt == 0 && continue
        avg = dd / cnt
        if avg < best_d
            best_d = avg
            best_fold = f
        end
    end
    return best_fold == src_fold ? (src_fold % nf) + 1 : best_fold
end
