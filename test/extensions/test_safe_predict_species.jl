# Species-derived path forwarding through _safe_predict_metalearner (RED scaffold).
#
# Idiom B: subprocess-activation @testitem. The species-forwarding wiring lives in
# the always-loaded pipeline (`_safe_predict_metalearner`, src/analysis/pipeline.jl)
# but the assertion exercises the Metalearner extension's `predict_metalearner`
# keyword surface (links / protein_info / embeddings_seq / embeddings_net), so the
# child process activates examples/. When Flux is undiscoverable the block skips
# cleanly.
#
# What this guards (the species-forwarding gap, today RED):
#   `_safe_predict_metalearner(config)` currently forwards only `config.poi`,
#   `config.output.prior_file`, `config.metalearner_path`, and
#   `config.metalearner_use_mc_dropout` to `predict_metalearner`. It does NOT
#   forward `config.species`, so for a non-human species (e.g. 10090 mouse) the
#   call silently falls back to the hardcoded 9606 `links` / `protein_info` /
#   `embeddings_*` defaults baked into `predict_metalearner`'s signature — i.e.
#   mouse data would be scored against human STRING features. The forwarding path must make
#   `_safe_predict_metalearner` derive the per-species filenames from
#   `config.species` and forward them.
#
# Test strategy: monkeypatch `BayesInteractomics.predict_metalearner` in the child
# process with a capture stub that records the forwarded `links` / `protein_info`
# / `embeddings_seq` / `embeddings_net` kwargs into a Ref, then returns the
# canonical Variante-B 4-slot payload shape that `_safe_predict_metalearner`
# expects to unpack `(predictions, model, embedding_matrix)` from (status is
# appended by the wrapper). Two CONFIGs are exercised:
#
#   species = 10090 → forwarded `links` MUST contain "10090" and MUST NOT be the
#                     bare hardcoded 9606 literal (de-hardcode).
#   species = 9606  → forwarded `links` / `protein_info` MUST be byte-identical to
#                     the current human literals (byte-identity):
#                       encodings/9606.protein.links.detailed.v12.0.onlyAB.txt
#                       encodings/9606.protein.info.v12.0.txt
#
# RED until forwarding lands: with species un-forwarded, the 10090 CONFIG forwards the
# 9606 literal → the "10090 in links" assertion FAILS. The child prints
# MATCH=false (or errors), so the outer @test FAILS when Flux is discoverable.

@testitem "_safe_predict_metalearner forwards species-derived paths (9606 byte-identical)" begin
    using Test

    repo_root = dirname(dirname(@__DIR__))
    examples_proj = joinpath(repo_root, "examples")
    flux_ok = Base.find_package("Flux") !== nothing ||
              (isdir(examples_proj) && isfile(joinpath(examples_proj, "Project.toml")))

    if !flux_ok
        @info "Skipping '_safe_predict_metalearner species forwarding': Flux not discoverable."
        @test true
    else
        script = raw"""
            using Flux, MLJ, MLJScikitLearnInterface, HDF5, JLD2, DataFrames
            using BayesInteractomics

            # Capture stub: record the species-sensitive kwargs the pipeline forwards.
            const FORWARDED = Ref{Any}(nothing)
            function BayesInteractomics.predict_metalearner(poi::String;
                    links = "encodings/9606.protein.links.detailed.v12.0.onlyAB.txt",
                    protein_info = "encodings/9606.protein.info.v12.0.txt",
                    embeddings_seq = "encodings/9606.protein.sequence.embeddings.v12.0.h5",
                    embeddings_net = "encodings/9606.protein.network.embeddings.v12.0.h5",
                    kwargs...)
                FORWARDED[] = (; links, protein_info, embeddings_seq, embeddings_net)
                # Variante-B 3-slot return (predictions, model, embedding_matrix);
                # _safe_predict_metalearner appends the :loaded status.
                return DataFrame(Protein = ["X"], MetaClassifier = [0.5]), :mock_model, nothing
            end

            sp = BayesInteractomics

            function forwarded_for(species_id)
                cfg = CONFIG(
                    datafile     = ["dummy.xlsx"],
                    control_cols = [Dict(1 => [2, 3, 4])],
                    sample_cols  = [Dict(1 => [5, 6, 7])],
                    poi          = "9606.ENSP00000479624",
                    n_controls   = 3,
                    n_samples    = 3,
                    refID        = 1,
                    output       = OutputFiles("results_mock"),
                    species      = species_id,
                )
                FORWARDED[] = nothing
                ret = sp._safe_predict_metalearner(cfg)
                return (ret, FORWARDED[])
            end

            ret_h,  fwd_h  = forwarded_for(9606)
            ret_m,  fwd_m  = forwarded_for(10090)

            # Variante-B 4-tuple return contract preserved (predictions, model, embedding_matrix, status).
            tuple_ok = (ret_h isa Tuple) && length(ret_h) == 4 && ret_h[4] === :loaded

            # 10090 forwards a species-derived links path (not the 9606 literal).
            mouse_ok = fwd_m !== nothing &&
                       occursin("10090", fwd_m.links) &&
                       fwd_m.links != "encodings/9606.protein.links.detailed.v12.0.onlyAB.txt"

            # 9606 reproduces the human literal filenames byte-for-byte.
            human_ok = fwd_h !== nothing &&
                       fwd_h.links == "encodings/9606.protein.links.detailed.v12.0.onlyAB.txt" &&
                       fwd_h.protein_info == "encodings/9606.protein.info.v12.0.txt"

            println("TUPLE_OK=", tuple_ok)
            println("MOUSE_OK=", mouse_ok)
            println("HUMAN_OK=", human_ok)
            println("MATCH=", tuple_ok && mouse_ok && human_ok)
        """
        cmd = Cmd(`julia --project=$examples_proj --threads=4 -e $script`; dir = repo_root)
        out = read(cmd, String)
        @test occursin("MATCH=true", out)
    end
end
