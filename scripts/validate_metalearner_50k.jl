# scripts/validate_phase77_50k.jl
#
# End-to-end production calibration gate (success criterion 7). Runs the
# metalearner inference path on a large labeled fixture and measures per-bin
# ECE (10 equal-frequency bins, size-weighted) on a stratified hold-out, plus
# wall-clock per run.
#
# GATES (per ROADMAP success criterion 7), for BOTH variants:
#   - per-bin ECE ≤ 0.035 on the stratified hold-out
#   - wall-clock ≤ 15 min CPU
#
# TWO RUNS:
#   - metalearner_use_mc_dropout = false → metalearners/metalearner_tr_ddi.jld2   (14-feat)
#   - metalearner_use_mc_dropout = true  → metalearners/metalearner_tr_ddi_mc.jld2 (15-feat)
#
# ============================================================================
# FIXTURE CONSTRUCTION (documented precisely — Task 3 is human-gated BECAUSE the
# fixture is underspecified by the ROADMAP):
#
# A genuinely-labeled 50k-pair set does NOT exist in this repository. The largest
# labeled set we can assemble WITHOUT fabricating labels is the union of the
# three production training/validation/test HDF5 splits:
#
#     train_data.h5 : 13,029 labeled pairs
#     val_data.h5   :  3,722 labeled pairs
#     test_data.h5  :  1,862 labeled pairs
#     ---------------------------------------
#     TOTAL         : 18,613 real labeled pairs
#
# We therefore use the FULL 18,613-pair labeled pool as the end-to-end fixture
# (option (b) of the orchestrator brief — "use the largest labeled set you can
# assemble"). Each pair carries REAL features:
#   - 7 STRING in-species channels + 1 DNN score   (getMetaLearnerDataset)
#   - 4 STRING transferred + 2 Pfam DDI columns     (Spike-009 feature lookup)
#   - mc_std (15-feat variant only)                 (mc_dropout_batch, K=30)
# and a REAL {0,1} interaction label from the HDF5 `features_labels` matrix.
#
# STRATIFIED HOLD-OUT for the ECE gate:
#   The metalearner blender was fit on train+val (16,751 pairs). The test split
#   (1,862 pairs) is the GENUINE never-seen-in-blender-fit hold-out, so the
#   primary ECE gate is computed on the TEST split. We also report the full-pool
#   ECE for context (optimistic — includes the blender's own training rows).
#   The test split preserves the production class balance (positive rate ≈ 0.346,
#   matching 04-SUMMARY.md meta.test_positive_rate), i.e. it is already
#   stratified by the upstream train/val/test partition.
#
# This replicates the EXACT production inference arithmetic used by the
# predict_metalearner :tr_ddi / :tr_ddi_mc branches (same 14/15-col canonical
# feature order, same MLJ.predict → MLJ.pdf.(·, Ref(1.0)) posterior extraction,
# same Spike-009 lookup, same K=30 mc_std) — so the calibration measured here is
# the calibration the shipping pipeline produces.
#
# A run_analysis() 50k-pair fixture is NOT used: run_analysis drives the
# metalearner via the bait-proteome `prediction_data` path (one bait × whole
# proteome), which has NO ground-truth interaction labels and therefore cannot
# yield a label-based ECE. Driving the metalearner inference arithmetic directly
# on the labeled HDF5 pool is the only way to compute a TRUE per-bin ECE.
#
# SHIPPING-PATH MEASUREMENT (post-METAL-INFER-FIX): the shipped artefacts are now
# self-contained MLJ.Stack machines that predict directly on the RAW 14/15-col
# feature frame — exactly the call `predict_metalearner` makes at inference. So
# the ECE measured here IS the calibration the production pipeline produces (no
# more reconstructed-stack proxy / default-HP base learners as in the pre-fix run).
# ============================================================================
#
# ENV overrides (gitignored inputs live in the MAIN repo when run from a worktree):
#   SPIKE_MODEL / SPIKE_MODEL_473  → encodings/model-473-0.5414302830201915.jld2
#   SPIKE_TRAIN_DATA               → encodings/train_data.h5
#   SPIKE_VAL_DATA                 → encodings/val_data.h5
#   SPIKE_TEST_DATA                → encodings/test_data.h5
#   SPIKE_009_DIR                  → feature-builder source dir
#   METALEARNER_LOOKUP_CACHE       → lookup cache path (optional)
#
# Run:
#   julia --project=examples scripts/validate_phase77_50k.jl

using Flux, MLJ, MLJScikitLearnInterface, HDF5
using BayesInteractomics
using DataFrames, JLD2, Statistics, Printf

const ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)
ext === nothing && error("BayesInteractomicsMetalearnerExt not loaded — run with `julia --project=examples`.")

# ---- Path resolution (ENV overrides honoured) -------------------------------
const TRAIN_H5 = get(ENV, "SPIKE_TRAIN_DATA", "encodings/train_data.h5")
const VAL_H5   = get(ENV, "SPIKE_VAL_DATA",   "encodings/val_data.h5")
const TEST_H5  = get(ENV, "SPIKE_TEST_DATA",  "encodings/test_data.h5")
const MODEL    = get(ENV, "SPIKE_MODEL",      "encodings/model-473-0.5414302830201915.jld2")
const MC_MODEL = get(ENV, "SPIKE_MODEL_473",  MODEL)
const FEATURE_SRC_ROOT = get(ENV, "SPIKE_009_DIR", get(ENV, "BAYESINT_FEATURE_SRC_ROOT", ""))
const LOOKUP_CACHE = get(ENV, "METALEARNER_LOOKUP_CACHE",
    ".bayesinteractomics_cache/metalearner_features/lookup.jld2")

for (nm, p) in (("train", TRAIN_H5), ("val", VAL_H5), ("test", TEST_H5), ("model", MODEL))
    isfile(p) || error("Missing $nm input: $p (set the SPIKE_* env override).")
end

# ---- Gate thresholds --------------------------------------------------------
const ECE_GATE     = 0.035
const WALLCLOCK_S  = 15 * 60   # 15 min CPU

# ---- Per-bin ECE: 10 equal-frequency bins, size-weighted --------------------
function per_bin_ece(probs::AbstractVector{<:Real}, labels::AbstractVector{<:Real}; nbins::Int = 10)
    n = length(probs)
    @assert n == length(labels) "probs/labels length mismatch"
    order = sortperm(probs)
    p = probs[order]
    y = labels[order]
    # Equal-frequency edges: split into nbins contiguous chunks of ~equal size.
    ece = 0.0
    start = 1
    for b in 1:nbins
        stop = round(Int, b * n / nbins)
        stop < start && continue
        idx = start:stop
        bin_p = mean(@view p[idx])
        bin_y = mean(@view y[idx])
        w = length(idx) / n
        ece += w * abs(bin_p - bin_y)
        start = stop + 1
    end
    return ece
end

# ---- TR+DDI enrichment (mirrors train_all_metalearners.jl::_enrich_with_tr_ddi!) ----
const TR_DDI_DEFAULT = (
    neighborhood_tr = 0.0, experiments_tr = 0.0, database_tr = 0.0,
    textmining_tr = 0.0, ddi_n_known = 0, ddi_has_known = false,
)

function enrich_with_tr_ddi!(df::DataFrame, lookup::Dict)
    nrows = nrow(df)
    cols = (neighborhood_tr = Vector{Float64}(undef, nrows),
            experiments_tr  = Vector{Float64}(undef, nrows),
            database_tr     = Vector{Float64}(undef, nrows),
            textmining_tr   = Vector{Float64}(undef, nrows),
            ddi_n_known     = Vector{Int}(undef, nrows),
            ddi_has_known   = Vector{Bool}(undef, nrows))
    n_default = 0
    for i in 1:nrows
        a = String(df[i, :Protein1]); b = String(df[i, :Protein2])
        key = a <= b ? (a, b) : (b, a)
        nt = get(lookup, key, TR_DDI_DEFAULT)
        nt === TR_DDI_DEFAULT && (n_default += 1)
        cols.neighborhood_tr[i] = Float64(nt.neighborhood_tr)
        cols.experiments_tr[i]  = Float64(nt.experiments_tr)
        cols.database_tr[i]     = Float64(nt.database_tr)
        cols.textmining_tr[i]   = Float64(nt.textmining_tr)
        cols.ddi_n_known[i]     = Int(nt.ddi_n_known)
        cols.ddi_has_known[i]   = Bool(nt.ddi_has_known)
    end
    df[!, :neighborhood_tr] = cols.neighborhood_tr
    df[!, :experiments_tr]  = cols.experiments_tr
    df[!, :database_tr]     = cols.database_tr
    df[!, :textmining_tr]   = cols.textmining_tr
    df[!, :ddi_n_known]     = cols.ddi_n_known
    df[!, :ddi_has_known]   = cols.ddi_has_known
    return n_default
end

# ---- mc_std per H5 (mirrors train_all_metalearners.jl::_compute_mc_std_for_h5) ----
function compute_mc_std(h5_path::AbstractString; K::Int = 30, embedding_dim::Int = 3072)
    model = ext.getDNNModel(11, ext._define_layers(512, 11),
                            ext._define_activations("relu", 11),
                            0.6641025641025641)
    Flux.loadmodel!(model, JLD2.load(MC_MODEL, "model_state"))
    model = model |> Flux.cpu
    fl = HDF5.h5open(h5_path, "r") do f; HDF5.read(f, "features_labels"); end
    X = Matrix{Float32}(fl[:, 1:embedding_dim]')   # (3072, n_pairs)
    mc = ext.mc_dropout_batch(model, X; K = K)
    return sqrt.(Float64.(mc.var))
end

# ---- Build the production feature frame (14 or 15 cols) ----------------------
const BASE_COLS = [:neighborhood, :fusion, :phylogenetic, :coexpression,
                   :experimental, :database, :textmining, :DNN,
                   :neighborhood_tr, :experiments_tr, :database_tr,
                   :textmining_tr, :ddi_n_known, :ddi_has_known]

function production_frame(df::DataFrame, use_mc::Bool)::DataFrame
    cols = use_mc ? vcat(BASE_COLS, [:mc_std]) : BASE_COLS
    out = DataFrame()
    for c in cols
        col = df[!, c]
        out[!, c] = c === :ddi_n_known ? Int.(col) :
                    c === :ddi_has_known ? Bool.(col) : Float64.(col)
    end
    return out
end

# ---- Run one variant (false → tr_ddi ; true → tr_ddi_mc) --------------------
function run_variant(use_mc::Bool, lookup::Dict)
    variant = use_mc ? "tr_ddi_mc" : "tr_ddi"
    artefact = use_mc ? "metalearners/metalearner_tr_ddi_mc.jld2" :
                        "metalearners/metalearner_tr_ddi.jld2"
    @info "=== variant=$variant artefact=$artefact ==="

    t0 = time()

    # 1. Load the labeled feature DataFrames (real features + real labels).
    train_df = ext.getMetaLearnerDataset(TRAIN_H5, MODEL)
    val_df   = ext.getMetaLearnerDataset(VAL_H5,   MODEL)
    test_df  = ext.getMetaLearnerDataset(TEST_H5,  MODEL)

    # 2. Enrich with the 6 Spike-009 TR+DDI columns.
    enrich_with_tr_ddi!(train_df, lookup)
    enrich_with_tr_ddi!(val_df, lookup)
    enrich_with_tr_ddi!(test_df, lookup)

    # 3. mc_std (15-feat variant only).
    if use_mc
        train_df[!, :mc_std] = compute_mc_std(TRAIN_H5)
        val_df[!, :mc_std]   = compute_mc_std(VAL_H5)
        test_df[!, :mc_std]  = compute_mc_std(TEST_H5)
    end

    pool_df = vcat(train_df, val_df, test_df)
    n_pool  = nrow(pool_df)
    n_test  = nrow(test_df)

    # 4. Load the schema-tagged metalearner machine.
    loaded = ext.load_metalearner_with_schema(artefact)
    meta = loaded.mach
    schema_cols = loaded.schema_columns
    expected_n = use_mc ? 15 : 14
    @assert length(schema_cols) == expected_n "schema col count mismatch: $(length(schema_cols))"

    # 5. Build the canonical feature frame + predict.
    pool_feat = production_frame(pool_df, use_mc)
    test_feat = production_frame(test_df, use_mc)
    @assert ncol(pool_feat) == expected_n

    pool_y = Float64.(pool_df.label)
    test_y = Float64.(test_df.label)

    # METAL-INFER-FIX: the shipped artefact is a
    # SELF-CONTAINED MLJ.Stack predicting on the RAW 14/15-col frame — exactly
    # the `predict_metalearner` inference path. Recalibration: then
    # apply the embedded `loaded.calibrator` (Platt/isotonic), reproducing the
    # full shipping path that `predict_metalearner` runs. Base.invokelatest:
    # load_metalearner_with_schema `Base.require`s the Stack base-learner
    # packages at runtime (extending the world); re-resolve dispatch.
    pool_raw = MLJ.pdf.(Base.invokelatest(MLJ.predict, meta, pool_feat), Ref(1.0))
    test_raw = MLJ.pdf.(Base.invokelatest(MLJ.predict, meta, test_feat), Ref(1.0))
    pool_pred = ext.apply_calibrator(loaded.calibrator, pool_raw)
    test_pred = ext.apply_calibrator(loaded.calibrator, test_raw)
    @info "Calibrator applied" variant=variant method=loaded.calibrator.method

    elapsed = time() - t0

    # 6. Per-bin ECE (10 equal-frequency, size-weighted). Hold-out = test split.
    ece_test = per_bin_ece(test_pred, test_y; nbins = 10)
    ece_pool = per_bin_ece(pool_pred, pool_y; nbins = 10)

    pass_ece  = ece_test <= ECE_GATE
    pass_wall = elapsed <= WALLCLOCK_S

    @info "RESULT $variant" n_pool=n_pool n_test=n_test test_pos_rate=round(mean(test_y), digits=4) ece_test=round(ece_test, digits=5) ece_pool=round(ece_pool, digits=5) wallclock_s=round(elapsed, digits=1) wallclock_min=round(elapsed/60, digits=2)
    @info "GATES  $variant" ece_gate=ECE_GATE pass_ece=pass_ece wallclock_gate_s=WALLCLOCK_S pass_wallclock=pass_wall

    return (; variant, n_pool, n_test, ece_test, ece_pool,
            wallclock_s = elapsed, pass_ece, pass_wall)
end

# ---- Main -------------------------------------------------------------------
@info "Building/loading TR+DDI feature lookup" spike_dir=FEATURE_SRC_ROOT cache=LOOKUP_CACHE
lookup = ext.get_or_build_feature_lookup(; spike_dir = FEATURE_SRC_ROOT, cache_path = LOOKUP_CACHE)
@info "Feature lookup ready" n_pairs=length(lookup)

function _safe_run(use_mc)
    try
        return run_variant(use_mc, lookup)
    catch e
        variant = use_mc ? "tr_ddi_mc" : "tr_ddi"
        @error "BLOCKER: variant $variant could not run end-to-end" exception=(e, catch_backtrace())
        return (; variant, n_pool = 0, n_test = 0, ece_test = NaN, ece_pool = NaN,
                wallclock_s = NaN, pass_ece = false, pass_wall = false, blocked = true)
    end
end

r_false = _safe_run(false)   # tr_ddi
r_true  = _safe_run(true)    # tr_ddi_mc

println("\n" * "="^78)
println("END-TO-END CALIBRATION GATE SUMMARY")
println("="^78)
println(@sprintf("Fixture: %d real labeled pairs (train+val+test pool); hold-out = test split (%d pairs)",
                 r_false.n_pool, r_false.n_test))
println(@sprintf("ECE gate ≤ %.3f ; wall-clock gate ≤ %d s (15 min) — both per variant\n", ECE_GATE, WALLCLOCK_S))
for r in (r_false, r_true)
    println(@sprintf("  %-10s : %5.1f s (%.2f min) — wall %s | ECE_test %.5f — ECE %s | ECE_pool %.5f",
        r.variant, r.wallclock_s, r.wallclock_s/60,
        r.pass_wall ? "PASS" : "FAIL", r.ece_test, r.pass_ece ? "PASS" : "FAIL", r.ece_pool))
end
all_pass = r_false.pass_ece && r_false.pass_wall && r_true.pass_ece && r_true.pass_wall
println("\n" * (all_pass ? "ALL GATES PASS" : "ONE OR MORE GATES FAIL — see numbers above"))
println("="^78)

# Non-zero exit on gate failure so CI / orchestrator sees it; numbers already printed.
exit(all_pass ? 0 : 1)
