# test/combination/test_ablation_knobs.jl
#
# Ablation-knob test scaffold.
#
# Filename contains "ablation" so the VALIDATION filter
#   `occursin("ablation", ti.filename)` selects every testitem here.
#
# Two classes of testitem live in this file:
#   * GATE / A1  -- GREEN on the clean (pre-patch) tree. These pin the
#                   byte-identity baseline and the cache-key independence finding.
#   * ABL-P2 / ABL-P3 -- intentionally RED on the clean tree (marked
#                   `# RED until the feature lands`). They reference the kwargs
#                   (`streams`, `copula_family`, `h1_copula_family`) so the
#                   implementing work SEES them flip from RED to a real pass.
#                   They are NOT @test_broken: the implementing work must watch
#                   them turn green.
#
# Determinism: every testitem that runs the combination rebuilds the SAME
# fixture triplet (build_ablation_fixture) and seeds the global RNG with the
# committed ABLATION_FIXTURE_SEED *immediately before* combining, exactly as the
# reference was captured.

# ════════════════════════════════════════════════════════════════════════════
# GATE -- byte-identical default `full` path (GREEN on the clean tree)
# ════════════════════════════════════════════════════════════════════════════
@testitem "ablation byte-identical full default path" begin
    using BayesInteractomics
    using Random
    include(joinpath(@__DIR__, "..", "fixtures", "ablation_full_reference.jl"))

    triplet = build_ablation_fixture()
    refID = 1
    Random.seed!(ABLATION_FIXTURE_SEED)
    result = BayesInteractomics.combined_BF_bma(triplet, refID; verbose=false)

    # Same tree that produced the reference -> must reproduce it, EXCEPT the bait
    # (refID) cell. The frozen reference was captured BEFORE the E-5 fix, which
    # clamps the refID protein's default-`:bma`-path posterior/BF to the column
    # maximum (bait special-treatment, now consistent with the legacy-2c and
    # explicit-3c paths). E-5 is an intentional, reviewed change to default bait
    # handling that postdates the freeze — protein 1 (= refID) legitimately jumps
    # from its pre-E-5 marginalised value to the column max. So the bait cell is
    # asserted separately (== maximum), not against the stale frozen literal; every
    # NON-bait cell must still byte-reproduce. posterior_prob is bounded [0,1] so
    # atol is fine; bf spans 1e9..1e18, where atol=1e-9 demands ~27 significant
    # digits (impossible in Float64) -> use rtol for the large-magnitude bf.
    nonbait = setdiff(1:length(result.posterior_prob), refID)
    @test isapprox(result.posterior_prob[nonbait], REFERENCE_POSTERIOR_FULL[nonbait]; atol=1e-9)
    @test isapprox(result.bf[nonbait],             REFERENCE_BF_FULL[nonbait];        rtol=1e-6)
    # Bait cell (E-5): clamped to the column maximum for both posterior and bf.
    @test result.posterior_prob[refID] == maximum(result.posterior_prob)
    @test result.bf[refID]             == maximum(result.bf)
    @test length(result.posterior_prob) == REFERENCE_FIXTURE_N
end

# ════════════════════════════════════════════════════════════════════════════
# ABL-P2 -- evidence_streams (drop one evidence stream)
# ════════════════════════════════════════════════════════════════════════════

# RED until evidence_streams lands. On the clean tree `combined_BF_bma`
# swallows the unknown `streams` kwarg via its `kwargs...` catch-all, so the H0
# copula is still fitted in 3-D -> the 2-D assertion fails (RED-by-failure).
@testitem "ablation evidence_streams drops to 2-D copula (ABL-P2)" begin
    using BayesInteractomics
    using Random
    include(joinpath(@__DIR__, "..", "fixtures", "ablation_full_reference.jl"))

    triplet = build_ablation_fixture()
    refID = 1
    Random.seed!(ABLATION_FIXTURE_SEED)
    result = BayesInteractomics.combined_BF_bma(triplet, refID;
                                                streams = [:detection, :correlation],
                                                verbose = false)

    # The H0 copula's marginal tuple must drop to 2 dimensions when one stream
    # is excluded. joint_H0 is a SklarDist whose marginal tuple length == #streams.
    joint_H0 = result.copula_result.joint_H0
    @test length(joint_H0.m) == 2   # RED until evidence_streams lands
end

# RED until evidence_streams lands. `@assert length(cols) >= 2` should reject a single-stream
# config loudly. On the clean tree the `streams` kwarg is swallowed -> no assert
# fires -> @test_throws fails (RED-by-failure).
@testitem "ablation single-stream trips assert (ABL-P2)" begin
    using BayesInteractomics
    using Random
    include(joinpath(@__DIR__, "..", "fixtures", "ablation_full_reference.jl"))

    triplet = build_ablation_fixture()
    refID = 1
    Random.seed!(ABLATION_FIXTURE_SEED)
    # RED until evidence_streams lands: single-stream copula must throw AssertionError.
    @test_throws AssertionError BayesInteractomics.combined_BF_bma(
        triplet, refID; streams = [:detection], verbose = false)
end

# RED until evidence_streams lands. Exclude-only semantics: dropping a stream from the copula
# must leave the upstream BF input vectors numerically untouched (still all 3).
@testitem "ablation exclude-only leaves upstream BF vectors intact (ABL-P2)" begin
    using BayesInteractomics
    using Random
    include(joinpath(@__DIR__, "..", "fixtures", "ablation_full_reference.jl"))

    triplet = build_ablation_fixture()
    refID = 1
    # Snapshot the input vectors before the drop run.
    e0 = copy(triplet.enrichment)
    c0 = copy(triplet.correlation)
    d0 = copy(triplet.detection)

    Random.seed!(ABLATION_FIXTURE_SEED)
    result = BayesInteractomics.combined_BF_bma(triplet, refID;
                                                streams = [:detection, :correlation],
                                                verbose = false)

    # Inputs must be byte-unchanged after a drop run.
    @test triplet.enrichment == e0
    @test triplet.correlation == c0
    @test triplet.detection == d0
    # And the drop must actually have happened (2-D H0 copula). RED until evidence_streams lands.
    @test length(result.copula_result.joint_H0.m) == 2
end

# Regression (v1.2.1 audit ABL-P2 WARNING): the `:copula` combination path calls
# `combined_BF` directly (not `combined_BF_bma`). `combined_BF` honouring `streams`
# is already covered transitively above (combined_BF_bma forwards it), but the
# multi-protocol `analyse(::Vector{InteractionData})` `:copula` branch previously
# OMITTED `streams = streams` at the call site, silently dropping the evidence_streams
# override on that path. A function-level unit test cannot catch a call-site omission,
# so pin both `:copula` call sites at the source level (same idiom as the A1 cache-key
# test): every bare `combined_BF(` invocation in pipeline.jl MUST forward `streams`.
@testitem "ablation both :copula call sites thread streams (ABL-P2)" begin
    pipeline_path = joinpath(@__DIR__, "..", "..", "src", "analysis", "pipeline.jl")
    src = read(pipeline_path, String)

    # Collect each `combinedResult = combined_BF(` arg block (the bare-copula call
    # sites; `combined_BF_bma(` is a different name and is excluded by the `(`).
    starts = collect(eachmatch(r"combinedResult = combined_BF\("m, src))
    @test length(starts) == 2   # single-protocol + multi-protocol :copula branches

    for m in starts
        # Slice from the call to the next `)\n` that closes the kwarg block.
        rest = src[m.offset:end]
        close_idx = findfirst(r"\n        \)", rest)
        block = rest[1:close_idx.stop]
        @test occursin("streams = streams", block)   # ABL-P2: override must reach combined_BF
    end
end

# ════════════════════════════════════════════════════════════════════════════
# ABL-P3 -- copula_family / h1_copula_family (force a fixed copula family)
# ════════════════════════════════════════════════════════════════════════════

# RED until copula_family plumbing lands. On the clean tree the kwargs are
# swallowed by `kwargs...` -> family stays BIC-selected -> assertion fails
# (RED-by-failure, NOT MethodError).
@testitem "ablation copula_family=FrankCopula forces Frank at H0 AND H1 (ABL-P3)" begin
    using BayesInteractomics
    using Random
    include(joinpath(@__DIR__, "..", "fixtures", "ablation_full_reference.jl"))

    triplet = build_ablation_fixture()
    refID = 1
    Random.seed!(ABLATION_FIXTURE_SEED)
    result = BayesInteractomics.combined_BF_bma(triplet, refID;
                                                copula_family    = BayesInteractomics.FrankCopula,
                                                h1_copula_family = BayesInteractomics.FrankCopula,
                                                verbose = false)

    cop = result.copula_result
    # Forcing Frank must override BIC at BOTH H0 and H1 regardless of the data.
    @test cop.h0_copula_family == "FrankCopula"   # RED until copula_family plumbing lands
    @test cop.h1_copula_family == "FrankCopula"   # RED until copula_family plumbing lands
end

# RED until copula_family plumbing lands. Each of the four target families must be forceable through
# the DEFAULT :bma path. On the clean tree all four assertions fail (kwargs
# swallowed -> BIC selection).
@testitem "ablation all four families forceable through :bma default path (ABL-P3)" begin
    using BayesInteractomics
    using Random
    include(joinpath(@__DIR__, "..", "fixtures", "ablation_full_reference.jl"))

    refID = 1
    families = (
        ("GaussianCopula", BayesInteractomics.GaussianCopula),
        ("FrankCopula",    BayesInteractomics.FrankCopula),
        ("ClaytonCopula",  BayesInteractomics.ClaytonCopula),
        ("GumbelCopula",   BayesInteractomics.GumbelCopula),
    )
    for (name, fam) in families
        triplet = build_ablation_fixture()
        Random.seed!(ABLATION_FIXTURE_SEED)
        result = BayesInteractomics.combined_BF_bma(triplet, refID;
                                                    copula_family    = fam,
                                                    h1_copula_family = fam,
                                                    verbose = false)
        cop = result.copula_result
        @test cop.h0_copula_family == name   # RED until copula_family plumbing lands
        @test cop.h1_copula_family == name   # RED until copula_family plumbing lands
    end
end

# ════════════════════════════════════════════════════════════════════════════
# ABL-P1 -- use_metalearner_prior (gate the DNN-prior update_posterior_prob! step)
# ════════════════════════════════════════════════════════════════════════════
#
# The metalearner extension (Flux/MLJ/MLJScikitLearnInterface/HDF5) is generally NOT
# loaded in the test environment, so we cannot invoke the real `update_posterior_prob!`
# positive branch here. Instead we test the GUARD LOGIC in isolation:
# the boolean short-circuit `config.use_metalearner_prior && ml_status === :loaded &&
# !isnothing(meta_data)` MUST be false whenever `use_metalearner_prior` is false,
# regardless of `ml_status` / `meta_data`, and equals the legacy two-term guard when
# `use_metalearner_prior` is true (default). We also pin the CONFIG default + assert the
# guard prefix is present at BOTH source sites.
@testitem "ablation use_metalearner_prior gates DNN prior (ABL-P1)" begin
    using BayesInteractomics

    # Default preserves behaviour: the field is true out of the box.
    cfg_default = CONFIG(datafile = String[], control_cols = [], sample_cols = [],
                         poi = "X", n_controls = 1, n_samples = 1, refID = 1,
                         output = OutputFiles("tmp_abl_p1"))
    @test getfield(cfg_default, :use_metalearner_prior) === true

    # Guard-logic truth table (the exact short-circuit used at both pipeline sites).
    guard(use_prior, ml_status, meta_data) =
        use_prior && ml_status === :loaded && !isnothing(meta_data)

    # use_metalearner_prior = false -> branch NEVER taken, whatever ml_status / meta_data.
    @test guard(false, :loaded, Dict(:x => 1)) === false
    @test guard(false, :extension_not_loaded, nothing) === false
    @test guard(false, :loaded, nothing) === false

    # use_metalearner_prior = true (default) -> reduces to the legacy two-term guard:
    # applied only when the extension is loaded AND meta_data is present.
    @test guard(true, :loaded, Dict(:x => 1)) === true                 # applied (default path)
    @test guard(true, :extension_not_loaded, Dict(:x => 1)) === false  # not loaded -> fallback
    @test guard(true, :loaded, nothing) === false                      # no meta_data -> fallback

    # Both pipeline guard sites must source-contain the `config.use_metalearner_prior &&`
    # prefix immediately before the `ml_status === :loaded` condition.
    src = read(joinpath(@__DIR__, "..", "..", "src", "analysis", "pipeline.jl"), String)
    @test count("config.use_metalearner_prior && ml_status === :loaded", src) >= 2
end

# ════════════════════════════════════════════════════════════════════════════
# A1 -- cache-key independence from fieldnames(CONFIG) (GREEN on the clean tree)
# ════════════════════════════════════════════════════════════════════════════
# Verifies the cache-key independence assumption: no JLD2 cache-key hash in
# src/core/intermediate_cache.jl iterates `fieldnames(CONFIG)` or serialises the
# whole CONFIG struct. Therefore adding the 4 new CONFIG fields
# (evidence_streams, copula_family, h1_copula_family, use_metalearner_prior)
# cannot invalidate any intermediate cache. If a future edit introduces a
# whole-struct hash, this testitem flips RED and surfaces the gate risk.
@testitem "ablation cache-key independence (A1)" begin
    cache_src_path = joinpath(@__DIR__, "..", "..", "src", "core", "intermediate_cache.jl")
    @test isfile(cache_src_path)
    src = read(cache_src_path, String)

    # No cache-key hash may iterate fieldnames(CONFIG) or hash a bare CONFIG.
    @test !occursin("fieldnames(CONFIG)", src)
    @test !occursin("fieldnames(config)", src)

    # The four new CONFIG field symbols must NOT appear in any cache-key
    # construction (they don't exist on the clean tree, and the cache hashes
    # select only data/refID/likelihood/nu/imputation fields explicitly).
    for sym in ("evidence_streams", "copula_family", "h1_copula_family", "use_metalearner_prior")
        @test !occursin(sym, src)
    end
end
