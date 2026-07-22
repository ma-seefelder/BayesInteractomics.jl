# RED scaffold: on-the-fly featuriser.
#
# Idiom B: subprocess-activation @testitem. The re-enabled on-the-fly featuriser
# (the slow path in feature_lookup_cache.jl that currently `throw`s because it
# lacks a pair list) needs the Metalearner extension loaded, so the heavy logic
# runs in a child Julia process under examples/. When Flux is undiscoverable the
# block skips cleanly (mandatory clean-skip).
#
# RED until the slow path is re-enabled: the on-the-fly featuriser entry point
# (`ext.featurise_pair_onthefly` over a single novel (poi, prey) pair with a
# `restrict_to` STRING-ID set) is not yet re-enabled, so the child script errors
# before printing `MATCH=true`. When Flux IS discoverable the outer @test FAILS
# (no MATCH line) — the gate is proven by a hard failure, not skipped.
#
# Fixture: test/fixtures/metalearner_multispecies/ (EXPECTED.md). The featuriser
# must return ALL 6 columns (4 TR + ddi_n_known + ddi_has_known) for the novel
# 10090 mouse pair, matching the hand-computed non-zero expectations.

@testitem "on-the-fly featuriser returns 6 non-zero cols for a novel 10090 pair" begin
    using Test

    repo_root = dirname(dirname(@__DIR__))
    examples_proj = joinpath(repo_root, "examples")
    flux_ok = Base.find_package("Flux") !== nothing ||
              (isdir(examples_proj) && isfile(joinpath(examples_proj, "Project.toml")))

    if !flux_ok
        @info "Skipping 'on-the-fly featuriser': Flux not discoverable."
        @test true
    else
        @test isdir(joinpath(repo_root, "test", "fixtures", "metalearner_multispecies"))

        script = raw"""
            using Flux, MLJ, MLJScikitLearnInterface, HDF5, JLD2, DataFrames
            using BayesInteractomics
            ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)

            fixdir = joinpath("test", "fixtures", "metalearner_multispecies")

            # Novel (poi, prey) 10090 pair; restrict_to = the two STRING IDs.
            poi  = "10090.ENSMUSP00000B1"
            prey = "10090.ENSMUSP00000B2"
            restrict = Set([poi, prey])

            # Entry point: the re-enabled slow path featurises this
            # single pair on the fly from the fixture source files.
            rec = ext.featurise_pair_onthefly(
                poi, prey;
                source_dir = fixdir,
                species = "10090",
                restrict_to = restrict,
            )

            # Must return ALL 6 production-schema columns (EXPECTED.md 10090 pair).
            cols = (:neighborhood_tr, :experiments_tr, :database_tr,
                    :textmining_tr, :ddi_n_known, :ddi_has_known)
            has_all = all(c -> hasproperty(rec, c), cols)

            ok = has_all &&
                 isapprox(rec.neighborhood_tr, 110.0; atol = 1e-9) &&
                 isapprox(rec.experiments_tr, 120.0; atol = 1e-9) &&
                 isapprox(rec.database_tr,    130.0; atol = 1e-9) &&
                 isapprox(rec.textmining_tr,  140.0; atol = 1e-9) &&
                 isapprox(rec.ddi_n_known,      1.0; atol = 1e-9) &&
                 isapprox(rec.ddi_has_known,    1.0; atol = 1e-9)

            println("NCOLS=", count(c -> hasproperty(rec, c), cols))
            println("MATCH=", ok)
        """
        cmd = Cmd(`julia --project=$examples_proj --threads=4 -e $script`; dir = repo_root)
        out = read(cmd, String)
        @test occursin("MATCH=true", out)
    end
end
