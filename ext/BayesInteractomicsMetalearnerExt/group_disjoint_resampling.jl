# MLJ.ResamplingStrategy wrapper around the group-disjoint folds. MUST live in
# the extension (not Main / the train script) so a self-contained MLJ.Stack that
# embeds it in its `resampling` field deserialises in a FRESH process — the
# saved-Stack ship trap.
#
# (A script-local `Main.GroupDisjointFolds` serialises into the saved Stack's
# `resampling` field; a fresh child process that loads the artefact has no such
# type and dies with `UndefVarError: GroupDisjointFolds`. Hosting it here — a
# stable, always-loaded namespace — mirrors how the Stack's EvoTrees/KNN base
# learner types already resolve on reload.)
#
# `MLJ.ResamplingStrategy` is re-exported into the `MLJ` namespace; the generic
# `train_test_pairs` is reachable only as `MLJ.MLJBase.train_test_pairs` — that
# is the exact function MLJ.Stack's prefit dispatches on (the 4-arg form).
struct GroupDisjointFolds <: MLJ.ResamplingStrategy
    folds::Vector{Tuple{Vector{Int},Vector{Int}}}
end
# 4-arg form is the one MLJ.Stack's prefit invokes.
MLJ.MLJBase.train_test_pairs(r::GroupDisjointFolds, rows, X, y) = r.folds
# 2-arg form for generic MLJ resampling call sites.
MLJ.MLJBase.train_test_pairs(r::GroupDisjointFolds, rows)       = r.folds
