# Fresh-process load+predict ship-verify
# harness. Encodes the metalearner trap ("Production
# metalearner = self-contained MLJ.Stack (not bare blender)"): the first
# ship saved only the bare blender and failed at inference because `predict` was
# fed RAW features to a blender expecting `[OOF preds | features]`. Fit-then-
# predict-in-process tests missed it. The ONLY reliable verification is to
# load+predict a shipped artefact in a FRESH process.
#
# Idiom B: subprocess-activation @testitem mirroring the Cmd wiring of
# test_metalearner_back_compat.jl. The artefact is a self-contained
# MLJ.Stack whose embedded base-learner structs (EvoTrees / KNN / DecisionTree)
# only deserialise when their defining packages are loaded — so the child runs
# under examples/ and `load_metalearner_with_schema` calls
# `_ensure_stack_base_learners_loaded()` internally before `MLJ.machine(path)`.
# `Base.invokelatest(MLJ.predict, …)` re-resolves dispatch after that runtime
# `Base.require` extends the world.
#
# Parameterised artefact (SAM_REFIT_ARTEFACT, default
# metalearners/metalearner_tr_ddi.jld2) so the harness:
#   - runs against the CURRENT 14-feat artefact NOW (proving the mechanism), and
#   - retargets the refit artefact by `ENV["SAM_REFIT_ARTEFACT"]=…`.
#
# This test is GREEN on the current artefact (the harness must work before the
# refit lands) — distinct from the RED inference-axis scaffolds elsewhere in this suite.

@testitem "Fresh-process load+predict yields a finite probability in [0,1]" begin
    using Test

    repo_root = dirname(dirname(@__DIR__))
    examples_proj = joinpath(repo_root, "examples")
    flux_ok = Base.find_package("Flux") !== nothing ||
              (isdir(examples_proj) && isfile(joinpath(examples_proj, "Project.toml")))

    artefact_rel = get(ENV, "SAM_REFIT_ARTEFACT",
                       joinpath("metalearners", "metalearner_tr_ddi.jld2"))

    if !flux_ok
        @info "Skipping 'Fresh-process load+predict': Flux not discoverable."
        @test true
    else
        # Pass the (relative) artefact path into the child via ENV. The child
        # resolves it via the extension's `resolve_metalearner_path` (dev tree OR
        # the lazily-downloaded models artifact) and prints SKIP when it is
        # genuinely absent from both — so this runs on CI without the files on disk.
        script = raw"""
            using Flux, MLJ, MLJScikitLearnInterface, HDF5, JLD2, DataFrames
            using LazyArtifacts
            using BayesInteractomics
            ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)

            artefact_rel = ENV["SAM_REFIT_ARTEFACT_CHILD"]
            artefact = ext.resolve_metalearner_path(artefact_rel)
            if artefact === nothing || !isfile(artefact)
                println("SKIP=artefact_absent")
                exit(0)
            end

            # load_metalearner_with_schema pre-loads the Stack base-learner packages
            # (_ensure_stack_base_learners_loaded) before MLJ.machine(path) so the
            # self-contained Stack deserialises cleanly in this fresh world.
            loaded = ext.load_metalearner_with_schema(artefact)
            cols   = Symbol.(loaded.schema_columns)

            # Fixed deterministic input row over the artefact's own schema columns.
            # ddi_n_known is Int, ddi_has_known is Bool — match production eltypes
            # (the 50k-validation production_frame); all others Float64 = 0.5.
            row = DataFrame()
            for c in cols
                row[!, c] = c === :ddi_n_known   ? [1] :
                            c === :ddi_has_known ? [true] :
                                                   [0.5]
            end

            # Base.invokelatest: load_metalearner_with_schema Base.require'd the
            # base-learner packages at runtime, extending the world; re-resolve.
            preds = Base.invokelatest(MLJ.predict, loaded.mach, row)
            p = MLJ.pdf.(preds, Ref(1.0))[1]

            finite_in_unit = isfinite(p) && (0.0 <= p <= 1.0)
            println("SCHEMA_N=", length(cols))
            println("P=", p)
            println("MATCH=", finite_in_unit)
        """
        env = copy(ENV)
        env["SAM_REFIT_ARTEFACT_CHILD"] = artefact_rel
        cmd = Cmd(Cmd(`julia --project=$examples_proj --threads=4 -e $script`); dir = repo_root, env = env)
        out = read(cmd, String)
        if occursin("SKIP=artefact_absent", out)
            @info "Skipping 'Fresh-process load+predict': artefact absent from repo tree and models artifact" artefact_rel
            @test true
        else
            @test occursin("MATCH=true", out)
        end
    end
end
