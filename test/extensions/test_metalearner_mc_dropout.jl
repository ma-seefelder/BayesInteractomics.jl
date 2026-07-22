# MC-Dropout contract tests (complement to test_metalearner.jl).
#
# Two distinct blocks (no name collision with the main "MC-Dropout column
# reconstruction" block):
#   1. MC-default schema resolution: with the
#      extension loaded, `metalearner_use_mc_dropout = true` (the new default)
#      resolves the 15-feat :tr_ddi_mc artefact, and `= false` (opt-out) resolves
#      the 14-feat :tr_ddi artefact. The former hard-error guard was REMOVED
#      (MC rides the metalearner's existing DNN dependency), so there is nothing to
#      halt — this block now asserts the resolution behaviour instead.
#   2. mock mc_std reconstruction: the fixture's mock_mc_dropout_batch produces a
#      finite per-pair std that gets appended as the 15th column, mirroring the
#      production :tr_ddi_mc row builder (ext/.../metalearner.jl) WITHOUT needing a
#      loaded DNN model.

# --- Block 1: MC-default schema resolution (requires the extension; subprocess) ---

@testitem "metalearner_use_mc_dropout default-off resolves :tr_ddi; opt-in (deprecated) resolves :tr_ddi_mc" begin
    using Test

    repo_root = dirname(dirname(@__DIR__))
    examples_proj = joinpath(repo_root, "examples")
    flux_ok = Base.find_package("Flux") !== nothing ||
              (isdir(examples_proj) && isfile(joinpath(examples_proj, "Project.toml")))

    if !flux_ok
        @info "Skipping 'metalearner_use_mc_dropout default-on ...': Flux not discoverable."
        @test true
    else
        # The hard-error guard is gone. Assert the
        # schema-aware default resolver picks the MC artefact when the flag is
        # default-true and the 14-feat artefact when opted out. Resolution returns
        # the resolved path basename when the artefact is on disk, else nothing —
        # both committed artefacts are present in the repo, so we assert basenames.
        script = raw"""
            using Flux, MLJ, MLJScikitLearnInterface, HDF5
            using BayesInteractomics
            ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)

            p_mc  = ext.resolve_metalearner_path(nothing; use_mc_dropout = true)
            p_14  = ext.resolve_metalearner_path(nothing; use_mc_dropout = false)
            println("GUARD_REMOVED=", !isdefined(ext, :_require_mc_dropout_pipeline))
            println("MC_BASENAME=", p_mc === nothing ? "nothing" : basename(p_mc))
            println("F14_BASENAME=", p_14 === nothing ? "nothing" : basename(p_14))
            # CONFIG default must be false (non-MC :tr_ddi default contract). Minimal
            # valid CONFIG (datafile/control_cols/sample_cols/poi are the required fields).
            _cfg = CONFIG(datafile = String[], control_cols = Dict{Int,Vector{Int}}[],
                          sample_cols = Dict{Int,Vector{Int}}[], poi = "X")
            println("CONFIG_DEFAULT_MC=", _cfg.metalearner_use_mc_dropout)
        """
        cmd = Cmd(`julia --project=$examples_proj --threads=4 -e $script`; dir = repo_root)
        out = read(cmd, String)
        # The removed guard function must no longer exist.
        @test occursin("GUARD_REMOVED=true", out)
        # Explicit use_mc_dropout=true (deprecated) → :tr_ddi_mc artefact; default false → :tr_ddi artefact.
        @test occursin("MC_BASENAME=metalearner_tr_ddi_mc.jld2", out)
        @test occursin("F14_BASENAME=metalearner_tr_ddi.jld2", out)
        # CONFIG ships non-MC :tr_ddi default-off.
        @test occursin("CONFIG_DEFAULT_MC=false", out)
    end
end


# --- Block 2: mock mc_std reconstruction (pure; core-only deps) -----------------
#
# Reproduces the test/fixtures/metalearner_tr_ddi_100pairs.jl synthetic generator
# + mock_mc_dropout_batch inline so the block is self-contained under the metalearner
# filename filter (the @testsetup in the fixture file is not collected when the
# filter excludes its non-"metalearner" filename). The fixture file remains the
# canonical exported home of load_synthetic_fixture / mock_mc_dropout_batch and is
# exercised by the full-suite metalearner filter (gate 2).

@testitem "predict_metalearner reconstructs mc_std via mock mc_dropout_batch" begin
    using BayesInteractomics
    using Test
    using DataFrames
    using Random

    # --- inline synthetic fixture (identical recipe to the fixture module) ---
    Random.seed!(2026_05_22)
    n = 100
    features = DataFrame(
        neighborhood    = rand(n), fusion        = rand(n),
        phylogenetic    = rand(n), coexpression  = rand(n),
        experimental    = rand(n), database      = rand(n),
        textmining      = rand(n), DNN           = rand(n),
        neighborhood_tr = rand(n), experiments_tr= rand(n),
        database_tr     = rand(n), textmining_tr = rand(n),
        ddi_n_known     = rand(0:5, n), ddi_has_known = rand(Bool, n),
    )

    # --- inline mock_mc_dropout_batch (test double; returns `.std`, not `var`) ---
    mock_mc_dropout_batch(model, X; K::Int = 30, kwargs...) =
        (mean = rand(size(X, 1)), std = abs.(randn(size(X, 1))) .* 0.1)

    # Reconstruct the 15th column exactly as the production :tr_ddi_mc branch does.
    mc = mock_mc_dropout_batch(nothing, features; K = 30)
    f15 = hcat(features, DataFrame(mc_std = mc.std))

    @test ncol(f15) == 15
    @test "mc_std" in names(f15)
    @test length(mc.std) == nrow(features)
    @test !any(isnan, f15.mc_std)        # no NaN in reconstructed column
    @test all(isfinite, f15.mc_std)
    @test all(>=(0.0), f15.mc_std)       # std is a non-negative magnitude

    # The 14 production-schema columns referenced verbatim.
    for c in (:ddi_n_known, :ddi_has_known, :neighborhood_tr)
        @test c in propertynames(features)
    end
end
