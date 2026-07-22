# RED scaffold: extracted combine_embeddings.
#
# Idiom B: subprocess-activation @testitem. The extracted
# `combine_embeddings(::HDF5.Group, ::HDF5.Group, ::String; output_file)` method
# needs the Metalearner extension loaded (Flux+MLJ+HDF5), which is not
# discoverable under bare `--project=.` — so the heavy logic runs in a child
# Julia process under examples/. When Flux is undiscoverable the block skips
# cleanly (mandatory clean-skip).
#
# RED until the helper is lifted: today the `combine_embeddings` symbol is referenced inside
# `prediction_data` (metalearner.jl:253-285, the `if !isfile("encodings/emb_…")`
# branch) but is NOT a method reachable from the extension's public surface — the
# call throws `UndefVarError(:combine_embeddings)` because the helper lives only
# in the offline `scripts/dnn_training/generate_dataset.jl` builder, never
# included into the runtime extension. Lifting the helper into the
# extension lets `prediction_data` build per-species embeddings on-the-fly.
# Until then `ext.combine_embeddings` is undefined, the child script errors before
# printing `MATCH=true`, and (when Flux IS discoverable) the outer @test FAILS —
# the gate is proven by a hard failure, not skipped.
#
# Fixtures: test/fixtures/metalearner_multispecies/emb_seq_mini.h5 (1024×2) +
# emb_net_mini.h5 (512×2), each carrying a `proteins` String dataset (2 ids) and
# an `embeddings` Float matrix. The extracted combine_embeddings must vcat the
# 1024-dim seq embedding with the 512-dim net embedding → a 1536×2 matrix written
# under an HDF5 group named by the species string.

@testitem "combine_embeddings concatenates seq+net to 1536×2 from tiny HDF5 groups" begin
    using Test

    repo_root = dirname(dirname(@__DIR__))
    examples_proj = joinpath(repo_root, "examples")
    flux_ok = Base.find_package("Flux") !== nothing ||
              (isdir(examples_proj) && isfile(joinpath(examples_proj, "Project.toml")))

    if !flux_ok
        @info "Skipping 'combine_embeddings': Flux not discoverable."
        @test true
    else
        fixdir = joinpath(repo_root, "test", "fixtures", "metalearner_multispecies")
        @test isfile(joinpath(fixdir, "emb_seq_mini.h5"))
        @test isfile(joinpath(fixdir, "emb_net_mini.h5"))

        script = raw"""
            using Flux, MLJ, MLJScikitLearnInterface, HDF5, JLD2, DataFrames
            using BayesInteractomics
            ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsMetalearnerExt)

            fixdir  = joinpath("test", "fixtures", "metalearner_multispecies")
            seq_h5  = joinpath(fixdir, "emb_seq_mini.h5")
            net_h5  = joinpath(fixdir, "emb_net_mini.h5")
            species = "10090"

            # Stage the two mini embeddings into one tmp file as named groups,
            # exactly as prediction_data (metalearner.jl:262-276) does before
            # calling combine_embeddings.
            tmp_dir = mktempdir()
            tmp_in  = joinpath(tmp_dir, "tmp_in.h5")
            out_h5  = joinpath(tmp_dir, "emb_$(species).h5")

            seq_emb  = HDF5.h5open(seq_h5, "r") do f; HDF5.read(f, "embeddings"); end
            seq_prot = HDF5.h5open(seq_h5, "r") do f; HDF5.read(f, "proteins");   end
            net_emb  = HDF5.h5open(net_h5, "r") do f; HDF5.read(f, "embeddings"); end
            net_prot = HDF5.h5open(net_h5, "r") do f; HDF5.read(f, "proteins");   end

            f = HDF5.h5open(tmp_in, "w")
            g1 = HDF5.create_group(f, "$(species)_seq")
            g2 = HDF5.create_group(f, "$(species)_net")
            g1["embeddings"] = seq_emb; g1["proteins"] = seq_prot
            g2["embeddings"] = net_emb; g2["proteins"] = net_prot

            # Entry point: the extracted group-method.
            ext.combine_embeddings(
                f["$(species)_seq"], f["$(species)_net"], String(species),
                output_file = out_h5,
            )
            close(f)

            emb = HDF5.h5open(out_h5, "r") do file; HDF5.read(file, "$species/embeddings"); end
            prot = HDF5.h5open(out_h5, "r") do file; HDF5.read(file, "$species/proteins"); end

            dims_ok    = size(emb, 1) == 1536 && size(emb, 2) == 2
            aligned_ok = length(prot) == 2 && all(p -> startswith(String(p), "10090."), prot)

            println("MATCH=", dims_ok && aligned_ok)
        """
        cmd = Cmd(`julia --project=$examples_proj --threads=4 -e $script`; dir = repo_root)
        out = read(cmd, String)
        @test occursin("MATCH=true", out)
    end
end
