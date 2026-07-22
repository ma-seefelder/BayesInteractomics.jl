# scripts/eval_species_metalearner.jl
#
# Per-species stratified AUC/ECE eval.
#
# Proves the species-agnostic rebuild by computing per-species AUC + ECE on
# the EXISTING test split, running the inference loop TWICE:
#   - once against the NEW multi-species blender  (metalearners/metalearner_tr_ddi*.jld2)
#   - once against the KEPT human-only blender     (metalearners/_human_only/metalearner_tr_ddi*.jld2)
# Both artefacts are scored with the IDENTICAL production inference arithmetic
# (same canonical 14/15-col feature frame, same MLJ.pdf.(·, Ref(1.0)) extraction,
# same embedded calibrator) — only the artefact path differs. The lift
# (AUC_new − AUC_human) is isolated per species so a single pooled number cannot
# mask a non-human regression.
#
# ============================================================================
# WHAT IS SCORED (TEST SPLIT ONLY, no re-split):
#
# The labeled pool is the union of the three production train/val/test HDF5 splits
# (same assembly as scripts/validate_phase77_50k.jl). The TR+DDI feature lookup is
# built over the SPLIT pairs grouped by taxon (same multi-species route as
# metalearners/train_all_metalearners.jl Step 4 — split-pairs-restricted, NOT the
# whole-proteome enumeration). BUT the per-species AUC/ECE metrics are computed on
# the TEST-split rows ONLY (encodings/test_data.h5) — the genuine
# never-seen-in-blender-fit hold-out. The train/val pools are loaded only so the
# multi-species feature lookup covers every test pair's species; no metric is
# computed on train/val rows (T-77.2-08-LEAK mitigation).
#
# Production inference arithmetic (identical to predict_metalearner :tr_ddi /
# :tr_ddi_mc and to validate_phase77_50k.jl):
#   - 7 STRING in-species channels + 1 DNN score   (getMetaLearnerDataset)
#   - 4 STRING transferred + 2 Pfam DDI columns     (split-pairs feature lookup)
#   - mc_std (15-feat :tr_ddi_mc variant only)      (mc_dropout_batch, K=30)
#   - MLJ.predict → MLJ.pdf.(·, Ref(1.0)) → embedded calibrator (apply_calibrator)
# and a REAL {0,1} interaction label from the HDF5 `features_labels` matrix.
#
# Species of a STRING ID = substring before the FIRST '.' (split(id, ".")[1]) —
# the same rule download_species_data.jl / train_all_metalearners.jl Step 4 use.
# ============================================================================
#
# ENV overrides (gitignored inputs live in the MAIN repo when run from a worktree):
#   EVAL_SCHEMA                    → tr_ddi | tr_ddi_mc   (default tr_ddi_mc — production default)
#   SPIKE_MODEL / SPIKE_MODEL_473  → encodings/model-473-0.5414302830201915.jld2
#   SPIKE_TRAIN_DATA               → encodings/train_data.h5
#   SPIKE_VAL_DATA                 → encodings/val_data.h5
#   SPIKE_TEST_DATA                → encodings/test_data.h5
#   SPECIES_ENCODINGS_DIR          → encodings              (per-species STRING/UniProt/3did dir)
#   SPIKE_009_SRC                  → feature-builder source dir
#   EVAL_NEW_DIR                   → metalearners            (new multi-species artefacts)
#   EVAL_HUMAN_DIR                 → metalearners/_human_only (kept human-only baseline)
#
# Run (production default — :tr_ddi_mc):
#   julia --project=examples --threads=2 scripts/eval_species_metalearner.jl
# Run the 14-feat variant:
#   EVAL_SCHEMA=tr_ddi julia --project=examples --threads=2 scripts/eval_species_metalearner.jl

using Flux, MLJ, MLJScikitLearnInterface, HDF5
using BayesInteractomics
using DataFrames, JLD2, Statistics, Printf

const ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)
ext === nothing && error("BayesInteractomicsMetalearnerExt not loaded — run with `julia --project=examples`.")

# Canonical ECE recipe — reuses the extension's `_per_bin_ece` so the eval
# matches the training-time / validate_phase77_50k.jl gate exactly (10 equal-
# FREQUENCY bins, size-weighted). Do NOT re-derive ECE.
const per_bin_ece = ext._per_bin_ece

# ---- Schema selection (ENV) -------------------------------------------------
const SCHEMA = let s = lowercase(get(ENV, "EVAL_SCHEMA", "tr_ddi_mc"))
    s ∈ ("tr_ddi", "tr_ddi_mc") || error("EVAL_SCHEMA must be tr_ddi or tr_ddi_mc; got $s")
    Symbol(s)
end
const USE_MC = SCHEMA === :tr_ddi_mc

# ---- Path resolution (ENV overrides honoured) -------------------------------
const TRAIN_H5 = get(ENV, "SPIKE_TRAIN_DATA", "encodings/train_data.h5")
const VAL_H5   = get(ENV, "SPIKE_VAL_DATA",   "encodings/val_data.h5")
const TEST_H5  = get(ENV, "SPIKE_TEST_DATA",  "encodings/test_data.h5")
const MODEL    = get(ENV, "SPIKE_MODEL",      "encodings/model-473-0.5414302830201915.jld2")
const MC_MODEL = get(ENV, "SPIKE_MODEL_473",  MODEL)
const ENCODINGS_DIR = get(ENV, "SPECIES_ENCODINGS_DIR", "encodings")
# abspath: _included_spike009_modules does Base.include against the including
# file's dir, so a relative spike-src dir would resolve wrong — abspath it.
const FEATURE_SRC_DIR = abspath(get(ENV, "SPIKE_009_SRC",
    get(ENV, "BAYESINT_FEATURE_SRC_DIR",
        joinpath(dirname(@__DIR__), "metalearners", "feature_builders"))))

# Artefact directories: new multi-species refit vs the kept human-only baseline.
const NEW_DIR   = get(ENV, "EVAL_NEW_DIR",   "metalearners")
const HUMAN_DIR = get(ENV, "EVAL_HUMAN_DIR", joinpath("metalearners", "_human_only"))
_artefact_name() = USE_MC ? "metalearner_tr_ddi_mc.jld2" : "metalearner_tr_ddi.jld2"
const NEW_ARTEFACT   = joinpath(NEW_DIR,   _artefact_name())
const HUMAN_ARTEFACT = joinpath(HUMAN_DIR, _artefact_name())

const HUMAN_TAXON = "9606"   # the human taxon prefix

# ---- Species of a STRING ID = prefix before the first '.' --------------------
species_prefix(id::AbstractString) = String(split(String(id), ".")[1])

# ---- Hand-rolled rank-based AUC (Mann–Whitney U / no new package) -----------
# AUC = P(score(positive) > score(negative)), ties counted as 0.5. Computed via
# the average-rank identity: U = R_pos − n_pos(n_pos+1)/2, AUC = U / (n_pos·n_neg).
# Returns NaN when one class is absent (AUC undefined).
function rank_auc(scores::AbstractVector{<:Real}, labels::AbstractVector{<:Real})
    n = length(scores)
    @assert n == length(labels) "scores/labels length mismatch"
    n == 0 && return NaN
    pos = count(==(1), Int.(round.(labels)))
    neg = n - pos
    (pos == 0 || neg == 0) && return NaN
    # Average ranks (ascending) with tie handling.
    order = sortperm(scores)
    ranks = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        j = i
        while j < n && scores[order[j + 1]] == scores[order[i]]
            j += 1
        end
        avg = (i + j) / 2.0   # average of the 1-based rank positions in [i, j]
        for k in i:j
            ranks[order[k]] = avg
        end
        i = j + 1
    end
    rank_sum_pos = 0.0
    for idx in 1:n
        if Int(round(labels[idx])) == 1
            rank_sum_pos += ranks[idx]
        end
    end
    U = rank_sum_pos - pos * (pos + 1) / 2.0
    return U / (pos * neg)
end

# ---- Build the multi-species TR+DDI lookup over the SPLIT pairs ---------------
# Mirrors metalearners/train_all_metalearners.jl Step 4 (split-pairs-restricted,
# grouped by taxon). We featurise every pair across train+val+test so that every
# test pair's species is covered; metrics are later restricted to test rows.
function build_split_pairs_lookup(pool_df::DataFrame)
    split_pairs = Tuple{String,String}[]
    for i in 1:nrow(pool_df)
        push!(split_pairs, (String(pool_df[i, :Protein1]), String(pool_df[i, :Protein2])))
    end
    unique!(split_pairs)

    pairs_by_species = Dict{String, Vector{Tuple{String,String}}}()
    for (a, b) in split_pairs
        sp = species_prefix(a)
        push!(get!(pairs_by_species, sp, Tuple{String,String}[]), (a, b))
    end

    # Include the spike-009 builder modules ONCE; share across per-species calls.
    shared_mods = ext._included_spike009_modules(FEATURE_SRC_DIR)

    merged = Dict{Tuple{String,String}, NamedTuple}()
    for sp in sort!(collect(keys(pairs_by_species)))
        sp_pairs = pairs_by_species[sp]
        n_sp = length(sp_pairs)
        pairs_matrix = Matrix{String}(undef, n_sp, 2)
        species_protein_set = Set{String}()
        for (i, (a, b)) in enumerate(sp_pairs)
            pairs_matrix[i, 1] = a
            pairs_matrix[i, 2] = b
            push!(species_protein_set, a); push!(species_protein_set, b)
        end
        @info "Featurising split pairs for species" species=sp n_pairs=n_sp
        try
            sp_lookup = ext.featurise_pairs_onthefly(pairs_matrix, sp;
                source_dir   = ENCODINGS_DIR,
                spike_src_dir = FEATURE_SRC_DIR,
                restrict_to  = species_protein_set,
                mods         = shared_mods)
            merge!(merged, sp_lookup)
        catch e
            # A sparse species missing source files must NOT abort the eval —
            # its pairs zero-impute (TR_DDI_DEFAULT) and still appear in the table.
            @warn "Featurisation failed for species $sp; its pairs zero-impute." exception=e
        end
    end
    return merged
end

# ---- TR+DDI enrichment (mirrors train_all_metalearners.jl::_enrich_with_tr_ddi!) ----
const TR_DDI_DEFAULT = (
    neighborhood_tr = 0.0, experiments_tr = 0.0, database_tr = 0.0,
    textmining_tr = 0.0, ddi_n_known = 0, ddi_has_known = false,
)

function enrich_with_tr_ddi!(df::DataFrame, lookup::Dict)
    nrows = nrow(df)
    neighborhood_tr = Vector{Float64}(undef, nrows)
    experiments_tr  = Vector{Float64}(undef, nrows)
    database_tr     = Vector{Float64}(undef, nrows)
    textmining_tr   = Vector{Float64}(undef, nrows)
    ddi_n_known     = Vector{Int}(undef,     nrows)
    ddi_has_known   = Vector{Bool}(undef,    nrows)
    n_default = 0
    for i in 1:nrows
        a = String(df[i, :Protein1]); b = String(df[i, :Protein2])
        key = a <= b ? (a, b) : (b, a)   # canonical lexicographic key
        nt = get(lookup, key, TR_DDI_DEFAULT)
        nt === TR_DDI_DEFAULT && (n_default += 1)
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

# ---- Build the canonical production feature frame (14 or 15 cols) ------------
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

# ---- Production inference: raw Stack predict → embedded calibrator -----------
# IDENTICAL arithmetic to predict_metalearner :tr_ddi/:tr_ddi_mc and
# validate_phase77_50k.jl. Base.invokelatest: load_metalearner_with_schema
# `Base.require`s the Stack base-learner packages at runtime (extends the world);
# re-resolve dispatch (Julia 1.12 world-age).
function predict_calibrated(artefact_path::AbstractString, feat::DataFrame, use_mc::Bool)
    isfile(artefact_path) || error("Artefact not found: $artefact_path")
    loaded = ext.load_metalearner_with_schema(artefact_path)
    meta = loaded.mach
    expected_n = use_mc ? 15 : 14
    @assert length(loaded.schema_columns) == expected_n "schema col count mismatch for $artefact_path: $(length(loaded.schema_columns)) (expected $expected_n)"
    @assert ncol(feat) == expected_n "feature frame col count mismatch: $(ncol(feat)) (expected $expected_n)"
    raw = MLJ.pdf.(Base.invokelatest(MLJ.predict, meta, feat), Ref(1.0))
    # apply_calibrator is defined at extension LOAD time (not runtime-included),
    # so no world-age guard is needed (unlike MLJ.predict on the Stack). The
    # AbstractVector method maps the embedded calibrator over the raw scores.
    pred = ext.apply_calibrator(loaded.calibrator, raw)
    return pred, loaded.calibrator.method
end

# ---- Main -------------------------------------------------------------------
function main()
    println("="^78)
    println("Per-species stratified AUC/ECE eval (new vs human-only)")
    println("="^78)
    println(@sprintf("  schema = %s  (use_mc=%s)", SCHEMA, USE_MC))
    println(@sprintf("  NEW   artefact = %s", NEW_ARTEFACT))
    println(@sprintf("  HUMAN artefact = %s", HUMAN_ARTEFACT))
    println(@sprintf("  test split (metrics restricted to this) = %s", TEST_H5))
    println("="^78)

    for (nm, p) in (("train", TRAIN_H5), ("val", VAL_H5), ("test", TEST_H5),
                    ("model", MODEL), ("new artefact", NEW_ARTEFACT),
                    ("human artefact", HUMAN_ARTEFACT))
        isfile(p) || error("Missing $nm input: $p (set the corresponding ENV override).")
    end

    # 1. Load the labeled feature DataFrames (real features + real labels).
    @info "Loading labeled splits" train=TRAIN_H5 val=VAL_H5 test=TEST_H5
    train_df = ext.getMetaLearnerDataset(TRAIN_H5, MODEL)
    val_df   = ext.getMetaLearnerDataset(VAL_H5,   MODEL)
    test_df  = ext.getMetaLearnerDataset(TEST_H5,  MODEL)
    pool_df  = vcat(train_df, val_df, test_df)

    # 2. Build the multi-species TR+DDI lookup over the SPLIT pairs (grouped by
    #    taxon) and enrich BOTH pool and test (test is the scored subset).
    @info "Building multi-species split-pairs TR+DDI lookup" encodings=ENCODINGS_DIR spike_src=FEATURE_SRC_DIR
    lookup = build_split_pairs_lookup(pool_df)
    @info "Feature lookup ready" n_pairs=length(lookup)
    n_def = enrich_with_tr_ddi!(test_df, lookup)
    @info "TR+DDI enriched test split" n_test=nrow(test_df) n_imputed_default=n_def

    # 3. mc_std for the test split (15-feat variant only).
    if USE_MC
        @info "Computing mc_std (K=30) for the test split"
        test_df[!, :mc_std] = compute_mc_std(TEST_H5)
    end

    # 4. Build the canonical feature frame ONCE; both artefacts score the SAME frame.
    test_feat = production_frame(test_df, USE_MC)
    test_y    = Float64.(test_df.label)
    test_sp   = species_prefix.(String.(test_df.Protein1))

    # 5. Run the production inference TWICE (new + human-only, identical arithmetic).
    @info "Scoring test split with NEW artefact" artefact=NEW_ARTEFACT
    pred_new, cal_new = predict_calibrated(NEW_ARTEFACT, test_feat, USE_MC)
    @info "Scoring test split with HUMAN-ONLY artefact" artefact=HUMAN_ARTEFACT
    pred_human, cal_human = predict_calibrated(HUMAN_ARTEFACT, test_feat, USE_MC)
    @info "Calibrators applied" new_method=cal_new human_method=cal_human

    # 6. Per-species AUC + ECE (test rows only) for BOTH artefacts.
    species = sort!(unique(test_sp))
    rows = NamedTuple[]
    for sp in species
        idx = findall(==(sp), test_sp)
        n   = length(idx)
        yv  = test_y[idx]
        auc_new   = rank_auc(pred_new[idx],   yv)
        auc_human = rank_auc(pred_human[idx], yv)
        ece_new   = per_bin_ece(pred_new[idx],   yv)
        ece_human = per_bin_ece(pred_human[idx], yv)
        lift = (isnan(auc_new) || isnan(auc_human)) ? NaN : auc_new - auc_human
        push!(rows, (; species = sp, n_test_pairs = n,
                     AUC_new = auc_new, AUC_human = auc_human,
                     ECE_new = ece_new, ECE_human = ece_human, AUC_lift = lift))
    end

    # 7a. Machine-parseable delimited table (TSV; one header + one row per species).
    println()
    println("SPECIES_TABLE_BEGIN")
    println(join(["species", "n_test_pairs", "AUC_new", "AUC_human",
                  "ECE_new", "ECE_human", "AUC_lift"], "\t"))
    fmt(x) = isnan(x) ? "NaN" : @sprintf("%.6f", x)
    for r in rows
        println(join([r.species, string(r.n_test_pairs),
                      fmt(r.AUC_new), fmt(r.AUC_human),
                      fmt(r.ECE_new), fmt(r.ECE_human), fmt(r.AUC_lift)], "\t"))
    end
    println("SPECIES_TABLE_END")

    # 7b. Readable aligned view (human-friendly; not parsed).
    println()
    println("Per-species AUC/ECE (test split; new vs human-only) — schema=$SCHEMA")
    @printf("%-10s %8s %9s %9s %9s %9s %9s\n",
            "species", "n", "AUC_new", "AUC_hum", "ECE_new", "ECE_hum", "AUC_lift")
    println("-"^76)
    for r in rows
        @printf("%-10s %8d %9s %9s %9s %9s %9s\n",
                r.species, r.n_test_pairs,
                fmt(r.AUC_new), fmt(r.AUC_human),
                fmt(r.ECE_new), fmt(r.ECE_human), fmt(r.AUC_lift))
    end

    # 8. Non-human lift summary (species != 9606, with a defined lift).
    nonhuman_lifts = Float64[r.AUC_lift for r in rows
                             if r.species != HUMAN_TAXON && !isnan(r.AUC_lift)]
    println()
    if isempty(nonhuman_lifts)
        println("NONHUMAN_LIFT_SUMMARY\tn_species=0\tmean_lift=NaN\tmedian_lift=NaN\tn_nonneg=0")
        println("  (no non-human species with a defined AUC_lift — both classes present required)")
    else
        ml  = mean(nonhuman_lifts)
        mdl = median(nonhuman_lifts)
        n_nonneg = count(>=(0.0), nonhuman_lifts)
        println(@sprintf("NONHUMAN_LIFT_SUMMARY\tn_species=%d\tmean_lift=%.6f\tmedian_lift=%.6f\tn_nonneg=%d",
                         length(nonhuman_lifts), ml, mdl, n_nonneg))
        println(@sprintf("  non-human species (≠9606): n=%d, mean AUC_lift=%.6f, median=%.6f, %d/%d non-negative",
                         length(nonhuman_lifts), ml, mdl, n_nonneg, length(nonhuman_lifts)))
    end

    # 9. Human continuity: report 9606 AUC/ECE unchanged within tolerance.
    hrow = findfirst(r -> r.species == HUMAN_TAXON, rows)
    if hrow !== nothing
        r = rows[hrow]
        println(@sprintf("HUMAN_CONTINUITY\tAUC_new=%s\tAUC_human=%s\tECE_new=%s\tECE_human=%s",
                         fmt(r.AUC_new), fmt(r.AUC_human), fmt(r.ECE_new), fmt(r.ECE_human)))
    end

    println()
    println("EVAL_DONE")
    return 0
end

# Run only as a script (clean for orchestrator `include`-ing).
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
