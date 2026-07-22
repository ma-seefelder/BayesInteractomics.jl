# RED scaffold: per-species feature lookup.
#
# Idiom B: subprocess-activation @testitem. The per-species TR/DDI lookup builder
# needs the Metalearner extension loaded (Flux+MLJ+HDF5), which is not
# discoverable under bare `--project=.` — so the heavy logic runs in a child
# Julia process under examples/. When Flux is undiscoverable the block skips
# cleanly (mandatory clean-skip).
#
# RED until the builder lands: the per-species lookup-build entry point
# (`ext.build_feature_lookup_for_species` / the generalised
# METALEARNER_FEATURE_LOOKUP_SOURCE_FILES loop over species ids) does not yet
# exist, so the child script errors before printing `MATCH=true`. When Flux IS
# discoverable the outer @test therefore FAILS (no MATCH line) — the gate
# is proven by a hard failure, not skipped.
#
# Fixture: test/fixtures/metalearner_multispecies/ (2 species, hand-computed
# expectations in EXPECTED.md). The 10090 (mouse) pair MUST resolve to its
# non-zero TR/DDI NamedTuple — proving sparse non-human species are featurised,
# not zero-filled.

@testitem "per-species feature lookup resolves the 10090 pair to non-zero TR/DDI" begin
    using Test

    repo_root = dirname(dirname(@__DIR__))
    examples_proj = joinpath(repo_root, "examples")
    flux_ok = Base.find_package("Flux") !== nothing ||
              (isdir(examples_proj) && isfile(joinpath(examples_proj, "Project.toml")))

    if !flux_ok
        @info "Skipping 'per-species feature lookup': Flux not discoverable."
        @test true
    else
        @test isdir(joinpath(repo_root, "test", "fixtures", "metalearner_multispecies"))

        script = raw"""
            using Flux, MLJ, MLJScikitLearnInterface, HDF5, JLD2, DataFrames
            using BayesInteractomics
            ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)

            fixdir = joinpath("test", "fixtures", "metalearner_multispecies")

            # Expected (EXPECTED.md, 10090 mouse pair):
            #   neighborhood_tr=110, experiments_tr=120, database_tr=130,
            #   textmining_tr=140, ddi_n_known=1, ddi_has_known=1
            poi  = "10090.ENSMUSP00000B1"
            prey = "10090.ENSMUSP00000B2"

            # Entry point: build the lookup for BOTH species pointing
            # the per-species SOURCE_FILES at the fixture dir, then query the 10090
            # canonical pair key. (Helper name/signature is what the builder ships; the
            # scaffold just needs SOME callable that returns the production NamedTuple.)
            lookup = ext.build_feature_lookup_multispecies(
                fixdir;
                species = ["9606", "10090"],
            )

            key = (poi <= prey) ? (poi, prey) : (prey, poi)
            rec = lookup[key]

            ok = isapprox(rec.neighborhood_tr, 110.0; atol = 1e-9) &&
                 isapprox(rec.experiments_tr, 120.0; atol = 1e-9) &&
                 isapprox(rec.database_tr,    130.0; atol = 1e-9) &&
                 isapprox(rec.textmining_tr,  140.0; atol = 1e-9) &&
                 isapprox(rec.ddi_n_known,      1.0; atol = 1e-9) &&
                 isapprox(rec.ddi_has_known,    1.0; atol = 1e-9)

            println("MATCH=", ok)
        """
        cmd = Cmd(`julia --project=$examples_proj --threads=4 -e $script`; dir = repo_root)
        out = read(cmd, String)
        @test occursin("MATCH=true", out)
    end
end
