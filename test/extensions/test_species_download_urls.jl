# Per-species download URL builder (RED scaffold).
#
# Idiom A: a single in-process @testitem. NO `using Flux`, NO network. This test
# pins the byte-exact URL/filename contract for the species-parameterised
# download script (the script ships as `metalearners/download_species_data.jl` with the
# pure helpers behind a `if abspath(PROGRAM_FILE) == @__FILE__` guard so they are
# `include`-able without triggering any download).
#
# RED until the script ships: the `include` of the not-yet-existing script raises a
# file-not-found (or, once the file exists but a helper is renamed, an
# UndefVarError). Either way the testitem FAILS LOUDLY — there is no @test_skip
# branch — the contract is proven by a hard failure, not a skip.
#
# Byte-identical contract: for sp == "9606" the builder MUST reproduce the EXACT
# literals the human-only download script emits today, so the human path stays
# byte-identical.

@testitem "Per-species download URL builder (byte-identical for 9606)" begin
    using Test

    # TestItemRunner cd's into this file's directory; anchor to repo root.
    repo_root = dirname(dirname(@__DIR__))
    script_path = joinpath(repo_root, "metalearners", "download_species_data.jl")

    # RED gate: until the script ships (with includable pure helpers),
    # this include fails — proving the scaffold. Do NOT guard with @test_skip.
    include(script_path)

    # --- Non-human species (10090, mouse): every URL must carry the species id ---
    sp = "10090"

    links_full_url = string_links_full_url(sp)
    @test links_full_url ==
        "https://stringdb-downloads.org/download/protein.links.full.v12.0/10090.protein.links.full.v12.0.txt.gz"

    aliases_url = string_aliases_url(sp)
    @test aliases_url ==
        "https://stringdb-downloads.org/download/protein.aliases.v12.0/10090.protein.aliases.v12.0.txt.gz"

    # The pre-processed inference baseline target name:
    # the `onlyAB` detailed file is a project-local derived artefact, named per species.
    @test string_detailed_onlyab_target(sp) ==
        "10090.protein.links.detailed.v12.0.onlyAB.txt"

    uniprot_url = uniprot_sl_pfam_url(sp)
    @test uniprot_url ==
        "https://rest.uniprot.org/uniprotkb/stream?query=organism_id%3A10090&format=tsv&fields=accession%2Cid%2Creviewed%2Ccc_subcellular_location%2Cxref_pfam"
    @test occursin("organism_id%3A10090", uniprot_url)

    # --- 9606 must reproduce the human-path literals byte-for-byte ---
    @test string_links_full_url("9606") ==
        "https://stringdb-downloads.org/download/protein.links.full.v12.0/9606.protein.links.full.v12.0.txt.gz"
    @test string_aliases_url("9606") ==
        "https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz"
    @test uniprot_sl_pfam_url("9606") ==
        "https://rest.uniprot.org/uniprotkb/stream?query=organism_id%3A9606&format=tsv&fields=accession%2Cid%2Creviewed%2Ccc_subcellular_location%2Cxref_pfam"

    # --- Species enumeration from a STRING-prefixed protein-ID matrix ---
    # Rule: species id == split(protein_id, ".")[1]. A fake n×2 String matrix of
    # STRING IDs must enumerate to the unique taxon-prefix set.
    proteins = ["9606.ENSP1" "10090.ENSMUSP1";
                "9606.ENSP2" "9606.ENSP3";
                "10090.ENSMUSP2" "10090.ENSMUSP3"]
    @test enumerate_species(proteins) == Set(["9606", "10090"])
end
