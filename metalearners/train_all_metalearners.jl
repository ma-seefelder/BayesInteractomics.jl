#!/usr/bin/env julia
# ============================================================================
# Production training script for the TR+DDI metalearner.
#
# This script is the canonical entry point for re-training the production
# metalearner:
#
#   - 14-feature schema (8 baseline STRING/DNN features + 6 TR+DDI features)
#     — the TR+DDI schema winner C6_TR_DDI (AUC 0.8806, ECE 0.0308, MCC 0.593)
#   - 6-candidate level-1 base-learner pool: HistGradientBoostingClassifier,
#     EvoTreesClassifier, LogisticClassifier, KNNClassifier,
#     RandomForestClassifier, ExtraTreesClassifier
#   - LR_L2 level-2 blender with Sill-2009 feature-weighted stacking
#     (original 14 features fed at BOTH level-1 AND level-2; feature-weighted
#     stacking winner, ECE 0.0259)
#
# DROPPED from the earlier candidate pool: the boosted-stumps classifier
# and the naive-Bayes classifier. Both exhibit structural mis-calibration
# that survives hyperparameter tuning: ECE 0.12 (boosted stumps) and ECE 0.22
# (naive Bayes) on the original 8-feature schema. The existing
# `metalearners/GaussianNBC_tune.jld2` and the existing
# `metalearners/ensemble.jld2` (which references it) STAY on disk for
# back-compat — this script just stops training them.
#
# Tuning measure: defaults are sourced from a tuning-measure verdict file (the
# final line `measure = :{brier,logloss}` is the grep target). The environment
# variable `METALEARNER_TUNING ∈ {brier, logloss}` overrides the verdict-derived
# default. Verdict at time of writing: `measure = :brier` (ΔECE = -0.00182,
# ΔMCC = +0.000625 — both calibration gates FAIL by ~3-8x; Brier kept).
#
# Schema dispatch: the environment variable
# `METALEARNER_SCHEMA ∈ {tr_ddi, tr_ddi_mc}` selects the 14-feature production
# default (`tr_ddi`) OR the 15-feature MC-Dropout variant (`tr_ddi_mc`). The
# 15-feature mode invokes the MC-Dropout batch helper (K=30) on the training
# data's embedding matrix and appends the resulting `mc_std` column to BOTH
# the level-1 base training input AND the level-2 blender input (features at
# all stack levels). Run the script TWICE in sequence to produce both
# production artefacts:
#
#   METALEARNER_SCHEMA=tr_ddi    julia --project=examples metalearners/train_all_metalearners.jl
#   METALEARNER_SCHEMA=tr_ddi_mc julia --project=examples metalearners/train_all_metalearners.jl
#
# Output (`METALEARNER_SCHEMA=tr_ddi`):
#   metalearners/metalearner_tr_ddi.jld2     — 14-feat blender + schema sidecar
#
# Output (`METALEARNER_SCHEMA=tr_ddi_mc`):
#   metalearners/metalearner_tr_ddi_mc.jld2  — 15-feat (incl. mc_std) blender + schema sidecar
# ============================================================================

println("="^80)
println("Metalearner training pipeline — TR+DDI schema")
println("="^80)

# Activate the environment — honour an already-active project (e.g.
# `julia --project=examples …`) and only fall back to the worktree root when
# none was specified on the command line. The earlier unconditional
# `Pkg.activate(".")` masked any user-supplied `--project=…` and forced the
# script to look up Flux / MLJ in the worktree root, which does NOT have them
# as dependencies (they live in `examples/Project.toml` as trigger packages
# for `BayesInteractomicsMetalearnerExt`).
using Pkg
let active = Base.active_project()
    if active === nothing || basename(active) != "Project.toml" || dirname(active) == pwd()
        # No --project=… was specified OR the active project IS the worktree
        # root. Either way fall back to the historical default.
        Pkg.activate(".")
    else
        @info "Using already-active Julia project" path=active
    end
end

# Trigger packages for the BayesInteractomicsMetalearnerExt extension.
using Flux, MLJ, MLJScikitLearnInterface, HDF5
using BayesInteractomics
using DataFrames
using Random
using Statistics
using JLD2  # for JLD2.load inside the MC-Dropout helper (Step 4b)

# Dependency-free group-disjoint CV fold builder for the production MLJ.Stack
# blender ONLY. `include`d (not a package dep) so the Stack resampler at the
# single production site below can swap from the stratified-CV default to
# explicit group-disjoint folds — no other resampler touched.
include(joinpath(@__DIR__, "group_disjoint_cv.jl"))

# Resolve the extension explicitly so calls read as `ext.fit_*`.
ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)
if ext === nothing
    error("BayesInteractomicsMetalearnerExt is not loaded. Ensure Flux + MLJ + MLJScikitLearnInterface + HDF5 are all available in the active project.")
end

# ----------------------------------------------------------------------------
# Tuning measure resolution.
#
# Source of truth: an optional tuning-measure verdict file whose final line
# reads `measure = :brier` OR `measure = :logloss`. The METALEARNER_TUNING env
# var overrides the verdict, primarily for testing log-loss.
# ----------------------------------------------------------------------------

const VERDICT_FILE = get(ENV, "METALEARNER_TUNING_VERDICT", "")

function _read_measure_verdict(path::AbstractString)::String
    if !isfile(path)
        @warn "Tuning-measure verdict file missing — defaulting to brier" path = path
        return "brier"
    end
    last_line = ""
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        last_line = String(s)
    end
    m = match(r"^measure = :(brier|logloss)\s*$", last_line)
    if m === nothing
        @warn "Tuning-measure verdict file present but final non-empty line does not match `measure = :{brier|logloss}` — defaulting to brier" last_line = last_line
        return "brier"
    end
    return m.captures[1]
end

measure_verdict = _read_measure_verdict(VERDICT_FILE)
tuning_env = get(ENV, "METALEARNER_TUNING", measure_verdict)
measure = tuning_env == "logloss" ? MLJ.LogLoss() : MLJ.BrierLoss()
@info "Tuning measure" METALEARNER_TUNING=tuning_env measure_verdict=measure_verdict measure=measure

# ----------------------------------------------------------------------------
# Schema dispatch.
#
# METALEARNER_SCHEMA ∈ {tr_ddi, tr_ddi_mc} controls the schema of the final
# blender artefact. The 15-feat variant inserts `mc_std` (MC-Dropout K=30) at
# the end of the production column order and writes to
# `metalearners/metalearner_tr_ddi_mc.jld2`.
# ----------------------------------------------------------------------------

schema_env = get(ENV, "METALEARNER_SCHEMA", "tr_ddi")
if schema_env ∉ ("tr_ddi", "tr_ddi_mc")
    error("METALEARNER_SCHEMA must be one of {tr_ddi, tr_ddi_mc}; got '$(schema_env)'")
end
schema_tag = Symbol(schema_env)
final_output_path = schema_tag === :tr_ddi_mc ?
    "metalearners/metalearner_tr_ddi_mc.jld2" :
    "metalearners/metalearner_tr_ddi.jld2"
@info "Schema" METALEARNER_SCHEMA=schema_env schema_tag=schema_tag final_output_path=final_output_path

# Define paths. SPIKE_TRAIN_DATA / SPIKE_VAL_DATA / SPIKE_TEST_DATA / SPIKE_MODEL_473
# overrides are honoured so this script can run from a worktree where the
# large H5 / model-473 files are not local (they are .gitignored).
dtrain_path = get(ENV, "SPIKE_TRAIN_DATA", "encodings/train_data.h5")
dval_path   = get(ENV, "SPIKE_VAL_DATA",   "encodings/val_data.h5")
dtest_path  = get(ENV, "SPIKE_TEST_DATA",  "encodings/test_data.h5")
# model-473 is the production DNN model (3072-dim input matches the H5 embedding
# dim; see ext/BayesInteractomicsMetalearnerExt/metalearner.jl::MODELPATH). An
# earlier default of model-927 has a 1024-dim input head and crashes with
# DimensionMismatch on the 3072-dim training data.
model_path  = get(ENV, "SPIKE_MODEL",      "encodings/model-473-0.5414302830201915.jld2")
# Same model is consumed for MC-Dropout passes (no separate file needed). Kept
# as a distinct env var so future runs that want to mix-and-match the
# deterministic and MC backbones can do so.
mc_dropout_model_path = get(ENV, "SPIKE_MODEL_473", "encodings/model-473-0.5414302830201915.jld2")

# Verify input files exist
println("\nVerifying training data files...")
required_files = [
    ("Training data", dtrain_path),
    ("Validation data", dval_path),
    ("Test data", dtest_path),
    ("DNN model", model_path),
]
if schema_tag === :tr_ddi_mc
    push!(required_files, ("MC-Dropout DNN model (model-473)", mc_dropout_model_path))
end
for (name, path) in required_files
    if isfile(path)
        println("  ✓ $name: $path")
    else
        error("  ✗ Missing $name: $path")
    end
end

# Create metalearners directory if it doesn't exist
if !isdir("metalearners")
    mkdir("metalearners")
    println("\nCreated metalearners/ directory")
else
    println("\nUsing existing metalearners/ directory")
end

# ----------------------------------------------------------------------------
# Step (3) — Load the 8 baseline STRING/DNN features from the HDF5 files.
#
# ext.getMetaLearnerDataset returns a DataFrame with columns
#   [Protein1, Protein2, neighborhood, fusion, phylogenetic, coexpression,
#    experimental, database, textmining, DNN, label]
# (see ext/BayesInteractomicsMetalearnerExt/metalearner.jl::getMetaLearnerDataset)
# ----------------------------------------------------------------------------

println("\n" * "="^80)
println("Step 3: load 8 baseline STRING + DNN features")
println("="^80)

val_data_raw   = ext.getMetaLearnerDataset(dval_path,   model_path)
train_data_raw = ext.getMetaLearnerDataset(dtrain_path, model_path)
test_data_raw  = ext.getMetaLearnerDataset(dtest_path,  model_path)
# Concatenate train + val for the production fit, matching the earlier
# pattern of train/val pooling under the StratifiedCV resampler (the existing
# fit_* used val_data only; we now pool train + val for the production schema
# so the 6 TR+DDI columns get exposed to the wider class-balance distribution).
data_baseline = vcat(train_data_raw, val_data_raw)
@info "Baseline data loaded" n_train=nrow(train_data_raw) n_val=nrow(val_data_raw) n_test=nrow(test_data_raw) n_pooled=nrow(data_baseline)

# ----------------------------------------------------------------------------
# Step (4) — Append the 6 TR+DDI columns from the feature lookup.
#
# Production column order (14 columns):
#   [neighborhood, fusion, phylogenetic, coexpression, experimental, database,
#    textmining, DNN,                          ← 8 baseline (preprocess_data slice)
#    neighborhood_tr, experiments_tr, database_tr, textmining_tr,
#    ddi_n_known, ddi_has_known]               ← 6 TR+DDI features
# ----------------------------------------------------------------------------

println("\n" * "="^80)
println("Step 4: append 6 TR+DDI columns from the feature lookup")
println("="^80)

lookup = let
    # Build the TR/DDI lookup over ONLY the split's pairs, grouped by taxon,
    # reading the per-species STRING/UniProt/3did files staged under
    # `encodings/`. This REPLACES the human-only `get_or_build_feature_lookup`
    # call (which featurised only species "9606" from the human-only cache dir)
    # — running the refit with that path would yield another human-only
    # artefact, defeating the multi-species goal.
    #
    # We do NOT call `build_feature_lookup_multispecies` here: that enumerates
    # ENTIRE-proteome pairs from each links.full (29× millions of pairs, mostly
    # never queried by `_enrich_with_tr_ddi!`). The split-pairs-restricted route
    # is correct (the enrich only looks up split pairs; misses fall back to
    # TR_DDI_DEFAULT) and bounded.
    encodings_dir = get(ENV, "SPECIES_ENCODINGS_DIR", "encodings")
    # abspath: _included_spike009_modules does Base.include(@__MODULE__, path),
    # which resolves a RELATIVE path against the including file's dir
    # (metalearners/), not the repo root — yielding a wrong nested path and a
    # SystemError. An absolute path (resolved against the repo-root cwd the
    # refit is launched from) makes the nested include unambiguous.
    spike_src_dir = abspath(get(ENV, "SPIKE_009_SRC",
        get(ENV, "BAYESINT_FEATURE_SRC_DIR",
            joinpath(dirname(@__DIR__), "metalearners", "feature_builders"))))
    @info "Building multi-species split-pairs feature lookup" encodings_dir=encodings_dir spike_src_dir=spike_src_dir

    # Collect every (Protein1, Protein2) across the split (data_baseline =
    # train ⧺ val, plus test_data_raw). Both DataFrames already exist above —
    # do NOT reload data.
    split_pairs = Tuple{String,String}[]
    for df in (data_baseline, test_data_raw)
        for i in 1:nrow(df)
            push!(split_pairs, (String(df[i, :Protein1]), String(df[i, :Protein2])))
        end
    end
    unique!(split_pairs)

    # Species of a STRING ID = substring before the FIRST '.' (same rule as
    # enumerate_split_species in download_species_data.jl). STRING links are
    # within-species, so a pair's two proteins share the taxon; key the pair's
    # species by Protein1's prefix.
    species_prefix(id::AbstractString) = String(split(String(id), ".")[1])

    # Group split pairs by species.
    pairs_by_species = Dict{String, Vector{Tuple{String,String}}}()
    for (a, b) in split_pairs
        sp = species_prefix(a)
        push!(get!(pairs_by_species, sp, Tuple{String,String}[]), (a, b))
    end

    # Include the spike-009 builder modules ONCE and share the handle across
    # every per-species call (avoids re-including 29×).
    shared_mods = ext._included_spike009_modules(spike_src_dir)

    merged = Dict{Tuple{String,String}, NamedTuple}()
    for sp in sort!(collect(keys(pairs_by_species)))
        sp_pairs = pairs_by_species[sp]
        n_sp = length(sp_pairs)
        # n × 2 pairs matrix for this species.
        pairs_matrix = Matrix{String}(undef, n_sp, 2)
        species_protein_set = Set{String}()
        for (i, (a, b)) in enumerate(sp_pairs)
            pairs_matrix[i, 1] = a
            pairs_matrix[i, 2] = b
            push!(species_protein_set, a); push!(species_protein_set, b)
        end
        @info "Featurising split pairs for species" species=sp n_pairs=n_sp
        sp_lookup = ext.featurise_pairs_onthefly(pairs_matrix, sp;
            source_dir   = encodings_dir,
            spike_src_dir = spike_src_dir,
            restrict_to  = species_protein_set,
            mods         = shared_mods)
        merge!(merged, sp_lookup)
    end
    merged
end
@info "Feature lookup loaded" n_pairs=length(lookup) n_species=length(Set(String(split(String(k[1]), ".")[1]) for k in keys(lookup)))

# Default tuple for pairs missing from the lookup (STRING+Pfam+UniProt
# coverage gap). Order matches METALEARNER_FEATURE_LOOKUP_COLUMNS.
const TR_DDI_DEFAULT = (
    neighborhood_tr = 0.0,
    experiments_tr  = 0.0,
    database_tr     = 0.0,
    textmining_tr   = 0.0,
    ddi_n_known     = 0,
    ddi_has_known   = false,
)

function _enrich_with_tr_ddi!(df::DataFrame, lookup::Dict)
    n_imputed_default = 0
    nrows = nrow(df)
    neighborhood_tr = Vector{Float64}(undef, nrows)
    experiments_tr  = Vector{Float64}(undef, nrows)
    database_tr     = Vector{Float64}(undef, nrows)
    textmining_tr   = Vector{Float64}(undef, nrows)
    ddi_n_known     = Vector{Int}(undef,     nrows)
    ddi_has_known   = Vector{Bool}(undef,    nrows)

    for i in 1:nrows
        a = String(df[i, :Protein1])
        b = String(df[i, :Protein2])
        key = a <= b ? (a, b) : (b, a)
        nt = get(lookup, key, TR_DDI_DEFAULT)
        if nt === TR_DDI_DEFAULT
            n_imputed_default += 1
        end
        neighborhood_tr[i] = Float64(nt.neighborhood_tr)
        experiments_tr[i]  = Float64(nt.experiments_tr)
        database_tr[i]     = Float64(nt.database_tr)
        textmining_tr[i]   = Float64(nt.textmining_tr)
        ddi_n_known[i]     = Int(nt.ddi_n_known)
        ddi_has_known[i]   = Bool(nt.ddi_has_known)
    end

    df[!, :neighborhood_tr] = neighborhood_tr
    df[!, :experiments_tr]  = experiments_tr
    df[!, :database_tr]     = database_tr
    df[!, :textmining_tr]   = textmining_tr
    df[!, :ddi_n_known]     = ddi_n_known
    df[!, :ddi_has_known]   = ddi_has_known
    return n_imputed_default
end

n_imputed_train = _enrich_with_tr_ddi!(data_baseline, lookup)
n_imputed_test  = _enrich_with_tr_ddi!(test_data_raw, lookup)
@info "TR+DDI columns appended" n_imputed_default_train=n_imputed_train n_imputed_default_test=n_imputed_test

# ----------------------------------------------------------------------------
# Step (4b) — Append `mc_std` column when METALEARNER_SCHEMA = tr_ddi_mc.
#
# +MC beats no-MC on AUC + MCC + Sens + F1 + Sharpness when `mc_std` is fed at
# BOTH level-1 base training AND level-2 blender training (features at all
# stack levels). K=30.
#
# The MC-Dropout pass loads embeddings from each H5 file directly (mirroring
# getMetaLearnerDataset's read order), runs mc_dropout_batch on the
# (3072, n_pairs)-shaped matrix, and derives mc_std = sqrt.(var) per the actual
# return shape of mc_dropout_batch (`(samples, mean, var, baseline)`).
# ----------------------------------------------------------------------------

function _compute_mc_std_for_h5(h5_path::AbstractString, mc_model_path::AbstractString;
                                K::Int = 30, embedding_dim::Int = 3072)
    # Load + load model-473 fresh (mirrors mc_dropout.jl::compute_mc_prior!).
    model = ext.getDNNModel(11, ext._define_layers(512, 11),
                            ext._define_activations("relu", 11),
                            0.6641025641025641)
    Flux.loadmodel!(model, JLD2.load(mc_model_path, "model_state"))
    model = model |> Flux.cpu

    features_and_labels = HDF5.h5open(h5_path, "r") do f
        HDF5.read(f, "features_labels")
    end
    # features_and_labels is (n_pairs, embedding_dim+1) — drop the trailing label col.
    n_pairs = size(features_and_labels, 1)
    X = Matrix{Float32}(features_and_labels[:, 1:embedding_dim]')  # (3072, n_pairs)
    @info "MC-Dropout input shape" h5_path = h5_path n_pairs = n_pairs embedding_dim = embedding_dim
    mc_result = ext.mc_dropout_batch(model, X; K = K)
    return sqrt.(Float64.(mc_result.var))
end

if schema_tag === :tr_ddi_mc
    # We need MC-Dropout std per pair in the SAME row order as data_baseline +
    # test_data_raw. data_baseline = vcat(train_data_raw, val_data_raw), so we
    # compute std vectors for train + val + test individually and concatenate.
    println("\n" * "="^80)
    println("Step 4b: compute mc_std via the MC-Dropout batch helper (K=30)")
    println("="^80)

    std_train = _compute_mc_std_for_h5(dtrain_path, mc_dropout_model_path)
    std_val   = _compute_mc_std_for_h5(dval_path,   mc_dropout_model_path)
    std_test  = _compute_mc_std_for_h5(dtest_path,  mc_dropout_model_path)

    @assert length(std_train) == nrow(train_data_raw) "mc_std train length mismatch: $(length(std_train)) vs $(nrow(train_data_raw))"
    @assert length(std_val)   == nrow(val_data_raw)   "mc_std val length mismatch: $(length(std_val)) vs $(nrow(val_data_raw))"
    @assert length(std_test)  == nrow(test_data_raw)  "mc_std test length mismatch: $(length(std_test)) vs $(nrow(test_data_raw))"

    data_baseline[!, :mc_std] = vcat(std_train, std_val)
    test_data_raw[!, :mc_std] = std_test
    @info "mc_std column appended" n_train_val=length(data_baseline.mc_std) n_test=length(test_data_raw.mc_std)
end

# Build the production DataFrame in fit order. 14 columns for `:tr_ddi`, 15
# columns (incl. `mc_std`) for `:tr_ddi_mc`. The ext.preprocess_data helper
# drops the first 2 columns (Protein1/Protein2) and casts the 8 baseline
# columns to Float64; we replicate it inline so the downstream production
# column order is unambiguous.
function _build_production_frame(df::DataFrame, schema_tag::Symbol)::DataFrame
    base_cols = [
        :neighborhood, :fusion, :phylogenetic, :coexpression, :experimental,
        :database, :textmining, :DNN,
        :neighborhood_tr, :experiments_tr, :database_tr, :textmining_tr,
        :ddi_n_known, :ddi_has_known,
    ]
    cols = schema_tag === :tr_ddi_mc ? vcat(base_cols, [:mc_std]) : base_cols
    out = DataFrame()
    for c in cols
        col = df[!, c]
        # ddi_n_known stays Int; ddi_has_known stays Bool; all others Float64.
        if c === :ddi_n_known
            out[!, c] = Int.(col)
        elseif c === :ddi_has_known
            out[!, c] = Bool.(col)
        else
            out[!, c] = Float64.(col)
        end
    end
    return out
end

data_prod      = _build_production_frame(data_baseline, schema_tag)
test_data_prod = _build_production_frame(test_data_raw, schema_tag)
# Coerce labels to Multiclass{2} — BrierLoss / LogLoss measures in MLJ.TunedModel
# require categorical targets. The pre-Phase-77 preprocess_data noted Float64
# labels "work with classifiers and predictions" but that path bypasses
# TunedModel's measure-driven HP selection, which DOES call `int()` on the
# target broadcast and rejects raw Float64.
target         = MLJ.coerce(Int.(data_baseline.label), MLJ.Multiclass)
target_test    = MLJ.coerce(Int.(test_data_raw.label), MLJ.Multiclass)
@info "Production frame built" n_rows=nrow(data_prod) n_cols=ncol(data_prod) schema_tag=schema_tag

# Production column-name list — used by save_metalearner_with_schema (Step 8).
const SCHEMA_COLUMNS_TR_DDI = String[
    "neighborhood", "fusion", "phylogenetic", "coexpression", "experimental",
    "database", "textmining", "DNN",
    "neighborhood_tr", "experiments_tr", "database_tr", "textmining_tr",
    "ddi_n_known", "ddi_has_known",
]
@assert length(SCHEMA_COLUMNS_TR_DDI) == 14 "Schema column list must be 14 entries"

# 15-column variant carries `mc_std` at the end (MC-Dropout std).
const SCHEMA_COLUMNS_TR_DDI_MC = vcat(SCHEMA_COLUMNS_TR_DDI, ["mc_std"])
@assert length(SCHEMA_COLUMNS_TR_DDI_MC) == 15 "Schema column list (mc) must be 15 entries"

# Pick the schema-column list matching the active dispatch.
final_schema_columns = schema_tag === :tr_ddi_mc ? SCHEMA_COLUMNS_TR_DDI_MC : SCHEMA_COLUMNS_TR_DDI
expected_ncols = schema_tag === :tr_ddi_mc ? 15 : 14
@assert ncol(data_prod) == expected_ncols "Production frame must be $(expected_ncols) columns for schema $(schema_tag); got $(ncol(data_prod))"

# ----------------------------------------------------------------------------
# Metalearner-level recalibration — PER-SPLIT production
# frames for an honest 3-way data split (no optimistic bias):
#   - Stack fits on `train` ONLY.
#   - The post-hoc calibrator fits on the Stack's predictions over `val`
#     (data the Stack NEVER trained on).
#   - ECE is evaluated on `test` (untouched by both Stack fit AND calibrator fit).
# `data_baseline` (= train ⧺ val, already TR+DDI- and mc_std-enriched in row
# order) is split back into its train / val halves by row count.
# ----------------------------------------------------------------------------
const _N_TRAIN = nrow(train_data_raw)
train_data_baseline = data_baseline[1:_N_TRAIN, :]
val_data_baseline   = data_baseline[(_N_TRAIN + 1):end, :]
@assert nrow(val_data_baseline) == nrow(val_data_raw) "val split row mismatch: $(nrow(val_data_baseline)) vs $(nrow(val_data_raw))"

train_prod = _build_production_frame(train_data_baseline, schema_tag)
val_prod   = _build_production_frame(val_data_baseline,   schema_tag)
target_train = MLJ.coerce(Int.(train_data_baseline.label), MLJ.Multiclass)
y_val_int    = Int.(val_data_baseline.label)
y_test_int   = Int.(test_data_raw.label)
@info "Per-split frames built (calibration discipline)" n_train=nrow(train_prod) n_val=nrow(val_prod) n_test=nrow(test_data_prod)

# ----------------------------------------------------------------------------
# Steps (5)–(7) — Single self-contained `MLJ.Stack`
# (inference-correctness fix, Option 1).
#
# RATIONALE (the defect this replaces):
#   The pre-fix path trained 6 base learners separately (Step 5), computed
#   manual OOF predictions (Step 6), then fit a BARE LR_L2 blender on the
#   20/21-column `[OOF | features]` matrix (Step 7) and saved ONLY that bare
#   blender. At inference, `predict_metalearner` calls `MLJ.predict(meta,
#   data_14)` on the RAW 14/15-column frame — but the bare blender expects 20
#   columns, so it raised "X has 14 features, but LogisticRegression is
#   expecting 20 features". The 6 base-learner tunes were never committed, so
#   the shipped artefact was unusable for inference.
#
# THE FIX:
#   Build ONE `MLJ.Stack`. The Stack internally (a) cross-fits the 6 base
#   learners with `resampling = StratifiedCV(nfolds=5)` to produce out-of-fold
#   adjudication features, (b) feeds `[OOF | original features]` to the LR_L2
#   metalearner (Sill-2009 feature-weighted stacking — the Stack appends the
#   original input features to the OOF adjudicators by construction), and
#   (c) at predict-time runs the full base→blender chain. A `machine(stack,
#   raw_feats, target)` is fit and SAVED whole, so `MLJ.predict(mach,
#   raw_14col_df)` works end-to-end with NO manual OOF reconstruction and NO
#   separate base-learner artefacts.
#
# WALL-CLOCK NOTE: the 6 base learners are fixed-HP (sensible defaults / the
# Spike-008/009 best-known HPs), NOT nested `TunedModel`s. Nesting 6 inner
# LatinHypercube searches inside the Stack's own 5-fold CV would multiply the
# fit cost ~6×20×5 and blow the budget. The LR_L2 metalearner is a plain
# fixed-HP `LogisticClassifier(penalty="l2")` for the same reason. This trades
# a small amount of inner-HP optimality for a self-contained, inference-correct
# artefact — exactly the METAL-INFER-FIX goal.
# ----------------------------------------------------------------------------

println("\n" * "="^80)
println("Steps 5–7: build + fit self-contained MLJ.Stack ($(schema_tag) schema)")
println("="^80)

const _schema_suffix = string(schema_tag)  # "tr_ddi" or "tr_ddi_mc"
const STACK_NFOLDS = 5   # internal cross-fitting folds for OOF adjudication

# Pre-load all candidate model types at top-level scope (world-age safety —
# `MLJ.@load` extends the world; pre-loading after `using MLJ` sidesteps the
# MethodError that a deferred load inside a function would trigger).
const _HGB_T = MLJ.@load HistGradientBoostingClassifier pkg=MLJScikitLearnInterface verbosity=0
const _LR_T  = MLJ.@load LogisticClassifier             pkg=MLJScikitLearnInterface verbosity=0
const _ET_T  = MLJ.@load EvoTreeClassifier              pkg=EvoTrees                 verbosity=0
const _KNN_T = MLJ.@load KNNClassifier                  pkg=NearestNeighborModels    verbosity=0
const _RF_T  = MLJ.@load RandomForestClassifier         pkg=DecisionTree             verbosity=0
const _XT_T  = MLJ.@load ExtraTreesClassifier           pkg=MLJScikitLearnInterface  verbosity=0

# Fixed-HP base learners (the 6-candidate pool — AdaBoost + GaussianNB
# dropped). HPs are sensible production defaults / best-known.
#
# ARTEFACT-SIZE NOTE: a self-contained Stack serialises ALL fitted base learners
# (one full refit + the cross-fit copies). With n_trees / n_estimators = 200 the
# RandomForest + ExtraTrees ensembles dominate and the saved Stack balloons to
# ~1 GB — too large to commit. We cap the tree ensembles (n=50, bounded depth)
# to keep the shipped artefact in the low-hundreds-of-MB range. This is the
# size/optimality trade-off called out in the Steps 5–7 rationale; the
# discrimination loss from 200→50 trees is small on a 14/15-feature stack.
hgb_base = _HGB_T(max_iter = 100, learning_rate = 0.1,
                  max_leaf_nodes = 31, l2_regularization = 0.0)
et_base  = _ET_T(eta = 0.1, max_depth = 6, nrounds = 100)
lr_base  = _LR_T(penalty = "l2", max_iter = 1_000)
knn_base = _KNN_T(K = 15)
rf_base  = _RF_T(n_trees = 50, max_depth = 12)
xt_base  = _XT_T(n_estimators = 50, max_depth = 12)

# Level-2 metalearner: fixed-HP L2-regularised logistic (LR_L2).
blender = _LR_T(penalty = "l2", max_iter = 1_000)

# ----------------------------------------------------------------------------
# Group-disjoint CV for the blender OOF.
#
# Cluster the 14/15-col `train_prod` input matrix and assign WHOLE clusters to
# single folds, so no near-duplicate row spans the train+test of any OOF fold.
# This is the ONLY production resampler swap — the stratified-CV sites inside
# the fit_* tuning helpers are intentionally left untouched.
# `target_train` (Multiclass{2}) drives the single-class fold fallback;
# the group centroids let the fallback merge into the nearest neighbouring fold.
const _GD_X        = Matrix(train_prod)
const _GD_GROUPS   = assign_groups(_GD_X)
const _GD_CENTS    = _group_centroids(_standardise(_GD_X), _GD_GROUPS)
const folds        = group_disjoint_folds(_GD_GROUPS, target_train;
                                          nfolds = STACK_NFOLDS,
                                          group_centroids = _GD_CENTS)
# Surface fold sizes + per-fold class counts so a refit run shows any single-
# class fallback firing.
let
    y_train_int = Int.(MLJ.int(target_train)) .- 1  # MLJ.int is 1-based; map to {0,1}
    for (fi, (tr, te)) in enumerate(folds)
        c0 = count(==(0), @view y_train_int[te])
        c1 = count(==(1), @view y_train_int[te])
        @info "Group-disjoint OOF fold" fold = fi n_train = length(tr) n_test = length(te) test_class0 = c0 test_class1 = c1
    end
    @info "Group-disjoint CV wired into the production Stack" n_groups = length(unique(_GD_GROUPS)) nfolds = STACK_NFOLDS
end

# ----------------------------------------------------------------------------
# Wrap the precomputed group-disjoint folds in a real `ResamplingStrategy`.
#
# `MLJ.Stack`'s prefit calls `train_test_pairs(resampling, rows, X, y)` (4-arg
# form), which only has a method for `ResamplingStrategy` subtypes. Passing the
# bare `Vector{Tuple{Vector{Int},Vector{Int}}}` returned by
# `group_disjoint_folds` (which has no such method) raises a MethodError inside
# the Stack fit. `ext.GroupDisjointFolds` is a thin strategy that returns the
# precomputed folds for both the 4-arg (Stack) and 2-arg (generic) call sites.
#
# CRITICAL: the type is hosted in the EXTENSION (`group_disjoint_resampling.jl`),
# NOT here in `Main`. A self-contained `MLJ.Stack` serialises its `resampling`
# field; a `Main.GroupDisjointFolds` would be undefined in the fresh child
# process that load+predicts the shipped artefact (ship trap →
# `UndefVarError`). The extension is a stable, always-loaded namespace, so the
# serialised type resolves on reload — exactly like the Stack's EvoTrees/KNN
# base-learner types already do. It is also NOT in `group_disjoint_cv.jl`
# (intentionally MLJ-free; its unit test includes it without MLJ loaded).

# Calibration-discipline note: the Stack fits on `train_prod`
# ONLY (NOT train+val). `val_prod` is reserved for fitting the post-hoc
# calibrator (Stack-unseen), and `test_data_prod` for the honest ECE gate.
stack = MLJ.Stack(;
    metalearner = blender,
    resampling  = ext.GroupDisjointFolds(folds),   # group-disjoint OOF — type hosted in the extension for fresh-process deserialisation
    measure     = measure,
    hgb         = hgb_base,
    evotrees    = et_base,
    logistic    = lr_base,
    knn         = knn_base,
    rforest     = rf_base,
    extratrees  = xt_base,
)

final_mach = let
    t0 = time()
    mach = MLJ.machine(stack, train_prod, target_train)
    @info "Fitting MLJ.Stack on TRAIN ONLY (6 fixed-HP base learners + LR_L2 metalearner, $(STACK_NFOLDS)-fold internal CV)..."
    MLJ.fit!(mach; verbosity = 1)
    @info "Stack fit complete" wall_seconds = round(time() - t0, digits = 1) n_train = nrow(train_prod)
    mach
end

# ----------------------------------------------------------------------------
# Step (7b) — Fit the post-hoc calibrator on the Stack's predictions over `val`
# (Stack-unseen). Platt-first, isotonic-fallback; method selected by per-bin ECE
# on `val`. This makes the SHIPPED artefact emit calibrated P(true), required
# because the metalearner output feeds the Bayesian integration upstream of the
# pipeline's own Platt step.
# ----------------------------------------------------------------------------
println("\n" * "="^80)
println("Step 7b: fit post-hoc calibrator (Platt→isotonic) on the val split")
println("="^80)

raw_val  = MLJ.pdf.(MLJ.predict(final_mach, val_prod), Ref(1))
calibrator, cal_info = ext.fit_metalearner_calibrator(
    raw_val, y_val_int, raw_val, y_val_int; ece_target = 0.035)
@info "Calibrator fit" chosen=cal_info.chosen ece_raw_val=round(cal_info.ece_raw, digits=5) ece_platt_val=round(cal_info.ece_platt, digits=5) ece_isotonic_val=(isnan(cal_info.ece_isotonic) ? "n/a" : round(cal_info.ece_isotonic, digits=5)) n_val=length(raw_val)

# ----------------------------------------------------------------------------
# Step (8) — Persist the final metalearner via the schema-aware sidecar helper.
# This OVERWRITES whatever fit_LR_L2_blender wrote at the same path with a
# fresh MLJ.save + a sibling `<path>.meta.jld2` carrying schema_tag and the
# 14- or 15-column schema_columns list.
# ----------------------------------------------------------------------------

println("\n" * "="^80)
println("Step 8: persist final metalearner with schema-aware sidecar metadata")
println("="^80)

ext.save_metalearner_with_schema(
    final_mach,
    final_output_path;
    schema_tag    = schema_tag,
    schema_columns = final_schema_columns,
    calibrator    = calibrator,   # embed the post-hoc calibrator
)

# ----------------------------------------------------------------------------
# Step (9) — Summary line.
# ----------------------------------------------------------------------------

@info "Metalearner trained" path=final_output_path schema=schema_tag n_features=expected_ncols candidates=6 measure=measure

# ----------------------------------------------------------------------------
# Step (9b) — Held-out test-set evaluation for the SUMMARY.
#
# Builds the level-2 blender input on test_data_prod via the same recipe used
# at production fit time:
#   1. For each candidate, predict on test using the production-tuned mach.
#   2. Stack the 6 predictions as test OOF columns.
#   3. Concat with test_data_prod (14 or 15 features).
#   4. Run the blender, compute ECE + MCC + AUC.
# ----------------------------------------------------------------------------

println("\n" * "="^80)
println("Step 9b: held-out test-set evaluation")
println("="^80)

let
    n_test = nrow(test_data_prod)
    # Stack predicts directly on the RAW test features — same call path
    # `predict_metalearner` uses at inference. RAW vs CALIBRATED comparison:
    # the shipped artefact applies `calibrator` so the inference
    # P(true) matches `p_test_cal` below.
    ŷ_test = MLJ.predict(final_mach, test_data_prod)
    p_test_raw = MLJ.pdf.(ŷ_test, Ref(1))
    p_test_cal = ext.apply_calibrator(calibrator, p_test_raw)
    y_true = Int.(test_data_raw.label)

    # AUC (rank-based — calibration is monotone for Platt and weakly monotone for
    # isotonic, so AUC is essentially unchanged; report on calibrated probs).
    auc_val = try MLJ.auc(ŷ_test, target_test) catch e; NaN end

    # MCC on calibrated probs (predicted class = 1 if p>=0.5 else 0).
    mcc_val = try
        ŷ_hard = MLJ.coerce([p >= 0.5 ? 1 : 0 for p in p_test_cal], MLJ.Multiclass)
        MLJ.mcc(ŷ_hard, target_test)
    catch e
        NaN
    end

    # Per-bin ECE — 10 equal-FREQUENCY bins, size-weighted (matches the gate in
    # scripts/validate_phase77_50k.jl and the in-extension _per_bin_ece).
    ece_raw = ext._per_bin_ece(p_test_raw, y_true)
    ece_cal = ext._per_bin_ece(p_test_cal, y_true)
    brier_val = sum((p_test_cal .- y_true) .^ 2) / length(p_test_cal)

    @info "Held-out test metrics (calibrated)" schema=schema_tag n_test=n_test AUC=auc_val MCC=mcc_val ECE_raw=round(ece_raw, digits=5) ECE_cal=round(ece_cal, digits=5) Brier=brier_val calibrator=cal_info.chosen
    println("AUC=$(auc_val), MCC=$(mcc_val), ECE_raw=$(ece_raw), ECE_cal=$(ece_cal), Brier=$(brier_val), calibrator=$(cal_info.chosen)")

    # Persist metrics for SUMMARY consumption.
    metrics_path = "metalearners/$(schema_tag)_test_metrics.txt"
    open(metrics_path, "w") do io
        println(io, "# Held-out test metrics for schema $(schema_tag) (calibrated)")
        println(io, "schema_tag = :$(schema_tag)")
        println(io, "n_features = $(expected_ncols)")
        println(io, "n_test = $(n_test)")
        println(io, "AUC = $(auc_val)")
        println(io, "MCC = $(mcc_val)")
        println(io, "ECE_raw_per_bin = $(ece_raw)")
        println(io, "ECE_calibrated_per_bin = $(ece_cal)")
        println(io, "Brier = $(brier_val)")
        println(io, "tuning_measure = $(measure)")
        println(io, "model = MLJ.Stack (6 fixed-HP base learners + LR_L2 metalearner) + post-hoc calibrator")
        println(io, "calibrator_method = $(cal_info.chosen)")
        println(io, "calibrator_ece_val_raw = $(cal_info.ece_raw)")
        println(io, "calibrator_ece_val_platt = $(cal_info.ece_platt)")
        println(io, "calibrator_ece_val_isotonic = $(cal_info.ece_isotonic)")
        println(io, "stack_nfolds = $(STACK_NFOLDS)")
        println(io, "data_split = Stack on train; calibrator on val; ECE on test")
    end
    @info "Test metrics persisted" path=metrics_path
end

println("\n" * "="^80)
println("TRAINING COMPLETE")
println("="^80)

# Verify outputs. The self-contained MLJ.Stack ships as a SINGLE artefact
# (+ schema sidecar) — no separate per-candidate base-learner tunes anymore
# (METAL-INFER-FIX: the Stack carries the base learners internally).
saved_models = [
    final_output_path,
    final_output_path * ".meta.jld2",
]
for f in saved_models
    if isfile(f)
        filesize_mb = filesize(f) / (1024^2)
        println("  ✓ $f ($(round(filesize_mb, digits=2)) MB)")
    else
        println("  ✗ Missing: $f")
    end
end

println("\nTraining pipeline complete.")
println("="^80)
