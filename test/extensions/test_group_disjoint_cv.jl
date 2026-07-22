# RED scaffold: group-disjoint CV folds.
#
# Idiom A: a single in-process @testitem. NO `using Flux` — the fold builder is a
# dependency-free greedy clustering (`assign_groups`) + a fold splitter
# (`group_disjoint_folds`) over the 14/15-column input matrix, shipped in
# `metalearners/group_disjoint_cv.jl`.
#
# RED until the fold builder lands: the `include` of the not-yet-existing file raises a
# file-not-found (or UndefVarError once the file exists but a helper is missing).
# No @test_skip — the gate is proven by a hard failure.
#
# Contracts asserted (the fold builder must satisfy ALL):
#   - each fold is a Tuple{Vector{Int}, Vector{Int}}  (train_idx, test_idx)
#   - train ∪ test == every row, train ∩ test == ∅  (partition per fold)
#   - no group label appears in BOTH the train and test of the same fold
#     (the group-disjointness contract — leakage prevention)
#   - every test fold contains BOTH classes (Open-Q3 class-balance fallback)

@testitem "group-disjoint folds (disjointness + class balance)" begin
    using Test

    # TestItemRunner cd's into this file's directory; anchor to repo root.
    repo_root = dirname(dirname(@__DIR__))
    include(joinpath(repo_root, "metalearners", "group_disjoint_cv.jl"))

    # Build a tiny labelled input matrix: ~20 rows, 14 cols, 3 obvious clusters,
    # both classes present. Cluster centres are far apart so any reasonable
    # grouping recovers 3 groups; within-cluster jitter is tiny.
    ncols = 14
    centres = [fill(0.0, ncols), fill(10.0, ncols), fill(20.0, ncols)]
    rows = Vector{Vector{Float64}}()
    labels = Int[]
    for (ci, c) in enumerate(centres)
        for j in 1:7                                   # 7 rows per cluster → 21 rows
            push!(rows, c .+ (j * 1e-3))               # deterministic tiny jitter
            push!(labels, (j % 2 == 0) ? 1 : 0)        # both classes within each cluster
        end
    end
    X = permutedims(hcat(rows...))                     # 21 × 14
    y = labels
    nrows = size(X, 1)

    groups = assign_groups(X)
    @test groups isa AbstractVector{<:Integer}
    @test length(groups) == nrows

    folds = group_disjoint_folds(groups; nfolds = 3)
    @test folds isa AbstractVector
    @test length(folds) == 3

    for (train_idx, test_idx) in folds
        @test train_idx isa Vector{Int}
        @test test_idx isa Vector{Int}

        # Partition contract: train ∪ test == all rows, train ∩ test == ∅.
        @test sort(vcat(train_idx, test_idx)) == collect(1:nrows)
        @test isempty(intersect(train_idx, test_idx))

        # Group-disjointness: no group spans train AND test of the same fold.
        train_groups = Set(groups[train_idx])
        test_groups = Set(groups[test_idx])
        @test isempty(intersect(train_groups, test_groups))

        # Class-balance (Open-Q3): every test fold carries both classes.
        @test Set(y[test_idx]) == Set([0, 1])
    end
end
