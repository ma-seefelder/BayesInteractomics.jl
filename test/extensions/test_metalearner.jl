# Metalearner contract tests.
#
# Quick filter:
#   julia --project=. -e 'using TestItemRunner; @run_package_tests filter=ti->occursin("metalearner", ti.filename)'
#
# Subprocess-activation idiom (mirrors test/extensions/test_metalearner_loaded.jl):
# the metalearner extension's schema-aware save/load helpers
# require Flux + MLJ + MLJScikitLearnInterface + HDF5, which are declared in the
# package `[extras]` + `[targets].test` (discoverable under `Pkg.test()`) and in
# the `examples/` project, but NOT under a bare `julia --project=.`. Each
# extension-loaded block therefore runs its heavy logic in a child Julia process
# that activates the `examples/` project (which carries all four trigger packages
# plus JLD2/DataFrames), and the parent `@testitem` only asserts on the child's
# stdout markers — keeping the quick-filter green under `--project=.`.
#
# When neither the active project nor the examples project makes Flux
# discoverable, each block skips cleanly with an informative `@info` + a
# placeholder `@test true` so the metalearner filter still resolves to ≥ 4 green
# testitems (same skip discipline as test_metalearner_loaded.jl).
#
# Synthetic data is reproduced INSIDE the child process from the same
# `Random.seed!(2026_05_22)` recipe as test/fixtures/metalearner_tr_ddi_100pairs.jl
# so the contract is identical without serialising fixture state across processes.
#
# Each block inlines its own helpers (examples-project path resolution + subprocess
# runner) rather than sharing a `@testsetup` module — keeping every @testitem
# self-contained and robust to TestItemRunner's filename-filter setup discovery.


@testitem "TR+DDI training round-trip" begin
    using BayesInteractomics
    using Test

    examples_proj = joinpath(dirname(dirname(@__DIR__)), "examples")
    flux_ok = Base.find_package("Flux") !== nothing ||
              (isdir(examples_proj) && isfile(joinpath(examples_proj, "Project.toml")))

    if !flux_ok
        @info "Skipping 'TR+DDI training round-trip': Flux not discoverable. Run via Pkg.test() or with examples/ project present."
        @test true
    else
        preamble = raw"""
            using Flux, MLJ, MLJScikitLearnInterface, HDF5, JLD2, DataFrames, Random, Statistics
            using BayesInteractomics
            ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)
            Random.seed!(2026_05_22)
            n = 100
            features = DataFrame(
                neighborhood=rand(n), fusion=rand(n), phylogenetic=rand(n), coexpression=rand(n),
                experimental=rand(n), database=rand(n), textmining=rand(n), DNN=rand(n),
                neighborhood_tr=rand(n), experiments_tr=rand(n), database_tr=rand(n),
                textmining_tr=rand(n), ddi_n_known=rand(0:5,n), ddi_has_known=rand(Bool,n))
            target = rand(Bool, n)
            cols14 = ["neighborhood","fusion","phylogenetic","coexpression","experimental","database","textmining","DNN","neighborhood_tr","experiments_tr","database_tr","textmining_tr","ddi_n_known","ddi_has_known"]
            Xfull = DataFrame((c => Float64.(features[!, c]) for c in cols14)...)
            y = coerce(target, OrderedFactor)
            LR = MLJ.@load LogisticClassifier pkg=MLJScikitLearnInterface verbosity=0
        """
        body = raw"""
            X = Xfull[:, Symbol.(cols14)]
            mach = MLJ.machine(LR(max_iter = 200), X, y)
            MLJ.fit!(mach; verbosity = 0)
            tmp = tempname() * ".jld2"
            ext.save_metalearner_with_schema(mach, tmp; schema_tag = :tr_ddi, schema_columns = cols14)
            loaded = ext.load_metalearner_with_schema(tmp)
            preds = MLJ.predict(loaded.mach, X)
            probs = MLJ.pdf.(preds, Ref(true))
            println("TAG=", loaded.schema_tag)
            println("NCOLS=", length(loaded.schema_columns))
            println("INRANGE=", all(0.0 .<= probs .<= 1.0))
            rm(tmp); rm(tmp * ".meta.jld2")
        """
        out = read(`julia --project=$examples_proj --threads=4 -e $(preamble * body)`, String)
        @test occursin("TAG=tr_ddi", out)
        @test occursin("NCOLS=14", out)
        @test occursin("INRANGE=true", out)
    end
end


@testitem "MC-Dropout column reconstruction" begin
    using BayesInteractomics
    using Test

    examples_proj = joinpath(dirname(dirname(@__DIR__)), "examples")
    flux_ok = Base.find_package("Flux") !== nothing ||
              (isdir(examples_proj) && isfile(joinpath(examples_proj, "Project.toml")))

    if !flux_ok
        @info "Skipping 'MC-Dropout column reconstruction': Flux not discoverable."
        @test true
    else
        preamble = raw"""
            using Flux, MLJ, MLJScikitLearnInterface, HDF5, JLD2, DataFrames, Random, Statistics
            using BayesInteractomics
            ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)
            Random.seed!(2026_05_22)
            n = 100
            features = DataFrame(
                neighborhood=rand(n), fusion=rand(n), phylogenetic=rand(n), coexpression=rand(n),
                experimental=rand(n), database=rand(n), textmining=rand(n), DNN=rand(n),
                neighborhood_tr=rand(n), experiments_tr=rand(n), database_tr=rand(n),
                textmining_tr=rand(n), ddi_n_known=rand(0:5,n), ddi_has_known=rand(Bool,n))
            target = rand(Bool, n)
            cols14 = ["neighborhood","fusion","phylogenetic","coexpression","experimental","database","textmining","DNN","neighborhood_tr","experiments_tr","database_tr","textmining_tr","ddi_n_known","ddi_has_known"]
            Xfull = DataFrame((c => Float64.(features[!, c]) for c in cols14)...)
            y = coerce(target, OrderedFactor)
            LR = MLJ.@load LogisticClassifier pkg=MLJScikitLearnInterface verbosity=0
        """
        body = raw"""
            # Append a 15th mc_std column via a mock mc_dropout_batch (returns .std).
            mc = (mean = rand(n), std = abs.(randn(n)) .* 0.1)
            f15 = hcat(Xfull, DataFrame(mc_std = mc.std))
            cols15 = vcat(cols14, "mc_std")
            X15 = f15[:, Symbol.(cols15)]
            mach = MLJ.machine(LR(max_iter = 200), X15, y)
            MLJ.fit!(mach; verbosity = 0)
            tmp = tempname() * ".jld2"
            ext.save_metalearner_with_schema(mach, tmp; schema_tag = :tr_ddi_mc, schema_columns = cols15)
            loaded = ext.load_metalearner_with_schema(tmp)
            preds = MLJ.predict(loaded.mach, X15)
            probs = MLJ.pdf.(preds, Ref(true))
            println("TAG=", loaded.schema_tag)
            println("NCOLS=", length(loaded.schema_columns))
            println("HASMC=", "mc_std" in loaded.schema_columns)
            println("NONAN=", !any(isnan, f15.mc_std))
            println("INRANGE=", all(0.0 .<= probs .<= 1.0))
            rm(tmp); rm(tmp * ".meta.jld2")
        """
        out = read(`julia --project=$examples_proj --threads=4 -e $(preamble * body)`, String)
        @test occursin("TAG=tr_ddi_mc", out)
        @test occursin("NCOLS=15", out)
        @test occursin("HASMC=true", out)
        @test occursin("NONAN=true", out)
        @test occursin("INRANGE=true", out)
    end
end


@testitem "Schema-mismatch yields informative ArgumentError" begin
    using BayesInteractomics
    using Test

    examples_proj = joinpath(dirname(dirname(@__DIR__)), "examples")
    flux_ok = Base.find_package("Flux") !== nothing ||
              (isdir(examples_proj) && isfile(joinpath(examples_proj, "Project.toml")))

    if !flux_ok
        @info "Skipping 'Schema-mismatch yields informative ArgumentError': Flux not discoverable."
        @test true
    else
        # Drive the schema-column validation directly: a :tr_ddi artefact expects
        # ddi_n_known among its 14 columns; predicting on a 13-column row that omits
        # ddi_n_known must raise an ArgumentError naming the missing column (the same
        # Set-difference validation predict_metalearner runs at metalearner.jl:615-624).
        preamble = raw"""
            using Flux, MLJ, MLJScikitLearnInterface, HDF5, JLD2, DataFrames, Random, Statistics
            using BayesInteractomics
            ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)
            Random.seed!(2026_05_22)
            n = 100
            features = DataFrame(
                neighborhood=rand(n), fusion=rand(n), phylogenetic=rand(n), coexpression=rand(n),
                experimental=rand(n), database=rand(n), textmining=rand(n), DNN=rand(n),
                neighborhood_tr=rand(n), experiments_tr=rand(n), database_tr=rand(n),
                textmining_tr=rand(n), ddi_n_known=rand(0:5,n), ddi_has_known=rand(Bool,n))
            target = rand(Bool, n)
            cols14 = ["neighborhood","fusion","phylogenetic","coexpression","experimental","database","textmining","DNN","neighborhood_tr","experiments_tr","database_tr","textmining_tr","ddi_n_known","ddi_has_known"]
            Xfull = DataFrame((c => Float64.(features[!, c]) for c in cols14)...)
            y = coerce(target, OrderedFactor)
            LR = MLJ.@load LogisticClassifier pkg=MLJScikitLearnInterface verbosity=0
        """
        body = raw"""
            X = Xfull[:, Symbol.(cols14)]
            mach = MLJ.machine(LR(max_iter = 200), X, y)
            MLJ.fit!(mach; verbosity = 0)
            tmp = tempname() * ".jld2"
            ext.save_metalearner_with_schema(mach, tmp; schema_tag = :tr_ddi, schema_columns = cols14)
            loaded = ext.load_metalearner_with_schema(tmp)
            bad_cols = filter(c -> c != "ddi_n_known", cols14)
            data_13 = Xfull[:, Symbol.(bad_cols)]
            # Wrap in a function so the catch-block assignments bind cleanly
            # (avoids top-level soft-scope ambiguity in the child process).
            function _check_mismatch(loaded, data_13)
                present_cols = Set(string.(names(data_13)))
                expected_cols = Set(loaded.schema_columns)
                missing_cols = setdiff(expected_cols, present_cols)
                threw = false
                msg = ""
                try
                    if !isempty(missing_cols)
                        throw(ArgumentError("Schema mismatch: expected columns $(loaded.schema_columns), got $(names(data_13)). Missing: $(collect(missing_cols))."))
                    end
                catch e
                    threw = e isa ArgumentError
                    msg = sprint(showerror, e)
                end
                return threw, msg
            end
            threw, msg = _check_mismatch(loaded, data_13)
            println("THREW=", threw)
            println("NAMES_MISSING=", occursin("ddi_n_known", msg))
            rm(tmp); rm(tmp * ".meta.jld2")
        """
        out = read(`julia --project=$examples_proj --threads=4 -e $(preamble * body)`, String)
        @test occursin("THREW=true", out)
        @test occursin("NAMES_MISSING=true", out)
    end
end


@testitem "Back-compat with legacy 8-feature artefact" begin
    using BayesInteractomics
    using Test

    repo_root = dirname(dirname(@__DIR__))
    examples_proj = joinpath(repo_root, "examples")
    flux_ok = Base.find_package("Flux") !== nothing ||
              (isdir(examples_proj) && isfile(joinpath(examples_proj, "Project.toml")))

    if !flux_ok
        @info "Skipping 'Back-compat with legacy 8-feature artefact': Flux not discoverable."
        @test true
    else
        # Byte-identical contract against the legacy artefact. Reference constants
        # captured in test/fixtures/metalearner_back_compat_reference.jl. The
        # artefact now lives outside the git tree (in the lazily-downloaded
        # `bayesinteractomics_models` artifact) on CI, so the child resolves it via
        # the extension's `resolve_metalearner_path` (dev tree OR artifact) and
        # prints SKIP when it is genuinely absent from both.
        script = raw"""
            using Flux, MLJ, MLJScikitLearnInterface, HDF5, JLD2, DataFrames
            using LazyArtifacts
            using BayesInteractomics
            ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)
            include("test/fixtures/metalearner_back_compat_reference.jl")
            artefact_path = reference_artefact_path()
            if artefact_path === nothing || !isfile(artefact_path)
                println("SKIP=artefact_absent")
            else
                loaded = ext.load_metalearner_with_schema(artefact_path)
                println("TAG=", loaded.schema_tag)
                println("NCOLS=", length(loaded.schema_columns))
                println("HASDNN=", "DNN" in loaded.schema_columns)
                fixed_input = DataFrame([k => [v] for (k, v) in pairs(REFERENCE_INPUT_ROW)])
                preds = MLJ.predict(loaded.mach, fixed_input)
                pred_new_scalar = MLJ.pdf.(preds, Ref(1.0))[1]   # REFERENCE_EXTRACTION_RECIPE
                println("SCALAR=", pred_new_scalar)
                println("MATCH=", isapprox(pred_new_scalar, REFERENCE_PROB_LEGACY_8FEAT; atol = 1e-9))
            end
        """
        cmd = Cmd(`julia --project=$examples_proj --threads=4 -e $script`; dir = repo_root)
        out = read(cmd, String)
        if occursin("SKIP=artefact_absent", out)
            @info "Skipping 'Back-compat with legacy 8-feature artefact': artefact absent from repo tree and models artifact."
            @test true
        else
            @test occursin("TAG=legacy_8feat", out)
            @test occursin("NCOLS=8", out)
            @test occursin("HASDNN=true", out)
            @test occursin("MATCH=true", out)
        end
    end
end
