# Schema-IO round-trip test (complement to test_metalearner.jl).
#
# Verifies the save/load contract in isolation: a metalearner persisted
# via `save_metalearner_with_schema(..., schema_tag, schema_columns)` round-trips
# its schema_tag + schema_columns through `load_metalearner_with_schema` for BOTH
# the :tr_ddi (14-col) and :tr_ddi_mc (15-col) tags, and rejects the reserved
# :legacy_8feat sentinel on the save side.
#
# This is distinct from the four main "TR+DDI training round-trip" /
# "MC-Dropout column reconstruction" blocks — those exercise fit+predict; this
# block exercises ONLY the metadata persistence contract. No two @testitem names
# collide across the metalearner files.
#
# Subprocess-activation idiom: same as test_metalearner.jl — the helpers
# require Flux+MLJ which are not discoverable under bare --project=. (they live in
# [extras]/[targets].test and examples/). When unavailable the block skips cleanly.

@testitem "save+load round-trip carries schema_tag + schema_columns" begin
    using Test

    repo_root = dirname(dirname(@__DIR__))
    examples_proj = joinpath(repo_root, "examples")
    flux_ok = Base.find_package("Flux") !== nothing ||
              (isdir(examples_proj) && isfile(joinpath(examples_proj, "Project.toml")))

    if !flux_ok
        @info "Skipping 'save+load round-trip carries schema_tag + schema_columns': Flux not discoverable."
        @test true
    else
        script = raw"""
            using Flux, MLJ, MLJScikitLearnInterface, HDF5, JLD2, DataFrames, Random
            using BayesInteractomics
            ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)

            Random.seed!(2026_05_22)
            n = 60
            cols14 = ["neighborhood","fusion","phylogenetic","coexpression",
                      "experimental","database","textmining","DNN",
                      "neighborhood_tr","experiments_tr","database_tr",
                      "textmining_tr","ddi_n_known","ddi_has_known"]
            X14 = DataFrame((c => rand(n) for c in cols14)...)
            y = coerce(rand(Bool, n), OrderedFactor)
            LR = MLJ.@load LogisticClassifier pkg=MLJScikitLearnInterface verbosity=0

            # --- :tr_ddi round-trip ---
            m14 = MLJ.machine(LR(max_iter = 150), X14, y); MLJ.fit!(m14; verbosity = 0)
            t14 = tempname() * ".jld2"
            ret14 = ext.save_metalearner_with_schema(m14, t14; schema_tag = :tr_ddi, schema_columns = cols14)
            l14 = ext.load_metalearner_with_schema(t14)
            println("RET_IS_PRIMARY=", ret14 == t14)
            println("SIDECAR_EXISTS=", isfile(t14 * ".meta.jld2"))
            println("TAG14=", l14.schema_tag)
            println("COLS14_MATCH=", l14.schema_columns == cols14)

            # --- :tr_ddi_mc round-trip ---
            cols15 = vcat(cols14, "mc_std")
            X15 = hcat(X14, DataFrame(mc_std = rand(n)))
            m15 = MLJ.machine(LR(max_iter = 150), X15, y); MLJ.fit!(m15; verbosity = 0)
            t15 = tempname() * ".jld2"
            ext.save_metalearner_with_schema(m15, t15; schema_tag = :tr_ddi_mc, schema_columns = cols15)
            l15 = ext.load_metalearner_with_schema(t15)
            println("TAG15=", l15.schema_tag)
            println("COLS15_MATCH=", l15.schema_columns == cols15)

            # --- reserved sentinel rejected on save ---
            function _probe_reject(m14, cols14)
                rejected = false
                try
                    ext.save_metalearner_with_schema(m14, tempname() * ".jld2";
                        schema_tag = :legacy_8feat, schema_columns = cols14)
                catch e
                    rejected = e isa ArgumentError
                end
                return rejected
            end
            println("LEGACY_REJECTED=", _probe_reject(m14, cols14))

            rm(t14); rm(t14 * ".meta.jld2"); rm(t15); rm(t15 * ".meta.jld2")
        """
        cmd = Cmd(`julia --project=$examples_proj --threads=4 -e $script`; dir = repo_root)
        out = read(cmd, String)
        @test occursin("RET_IS_PRIMARY=true", out)
        @test occursin("SIDECAR_EXISTS=true", out)
        @test occursin("TAG14=tr_ddi", out)
        @test occursin("COLS14_MATCH=true", out)
        @test occursin("TAG15=tr_ddi_mc", out)
        @test occursin("COLS15_MATCH=true", out)
        @test occursin("LEGACY_REJECTED=true", out)
    end
end
