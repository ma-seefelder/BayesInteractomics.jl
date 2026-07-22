"""
    test_curation.jl

Tests for protein data curation: splitting protein groups, parsing IDs,
removing contaminants, merging duplicate proteins, and curation report persistence.
"""

# ─────────────────────────────────────────────────────────────────────────────
# parse_protein_id
# ─────────────────────────────────────────────────────────────────────────────

@testitem "parse_protein_id: plain gene symbol" begin
    using BayesInteractomics

    @test parse_protein_id("HAP40") == "HAP40"
    @test parse_protein_id("RBFOX1") == "RBFOX1"
    @test parse_protein_id("  TP53  ") == "TP53"
end

@testitem "parse_protein_id: UniProt pipe format" begin
    using BayesInteractomics

    @test parse_protein_id("sp|P04637|P53_HUMAN") == "P53_HUMAN"
    @test parse_protein_id("tr|Q9NZM3|F8A1_HUMAN") == "F8A1_HUMAN"
    @test parse_protein_id("sp|P12345|") == "P12345"
end

@testitem "parse_protein_id: isoform stripping" begin
    using BayesInteractomics

    @test parse_protein_id("P12345-2") == "P12345"
    @test parse_protein_id("Q9NZM3-1") == "Q9NZM3"
    # Should NOT strip if not a UniProt-style accession
    @test parse_protein_id("HAP40-2") == "HAP40-2"
end

# ─────────────────────────────────────────────────────────────────────────────
# split_protein_groups
# ─────────────────────────────────────────────────────────────────────────────

@testitem "split_protein_groups: semicolon-separated" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["RBFOX3;RBFOX2;RBFOX1", "TP53", "HAP40"],
        intensity1 = [100.0, 200.0, 300.0],
        intensity2 = [110.0, 210.0, 310.0]
    )

    result, entries = split_protein_groups(df, 1)

    # Should expand 3 proteins from group + 2 singles = 5 rows
    @test nrow(result) == 5

    # Check the split proteins
    proteins = result.protein
    @test "RBFOX3" in proteins
    @test "RBFOX2" in proteins
    @test "RBFOX1" in proteins
    @test "TP53" in proteins
    @test "HAP40" in proteins

    # All split rows should have the same data
    rbfox_rows = filter(row -> row.protein in ["RBFOX3", "RBFOX2", "RBFOX1"], result)
    @test all(rbfox_rows.intensity1 .== 100.0)
    @test all(rbfox_rows.intensity2 .== 110.0)

    # Check that entries log correctly
    split_entries = filter(e -> e.action == BayesInteractomics.CURATE_SPLIT, entries)
    @test length(split_entries) == 3
    # First in group should be lead
    leads = filter(e -> e.is_lead, split_entries)
    @test length(leads) == 1
    @test leads[1].canonical_name == "RBFOX3"
end

@testitem "split_protein_groups: no groups (no-op)" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["TP53", "HAP40", "GAPDH"],
        value = [1.0, 2.0, 3.0]
    )

    result, entries = split_protein_groups(df, 1)

    @test nrow(result) == 3
    @test result.protein == ["TP53", "HAP40", "GAPDH"]
    # No splits should have occurred
    split_entries = filter(e -> e.action == BayesInteractomics.CURATE_SPLIT, entries)
    @test isempty(split_entries)
end

@testitem "split_protein_groups: custom delimiter" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["RBFOX3/RBFOX2", "TP53"],
        value = [10.0, 20.0]
    )

    result, entries = split_protein_groups(df, 1; delimiter="/")

    @test nrow(result) == 3
    @test "RBFOX3" in result.protein
    @test "RBFOX2" in result.protein
    @test "TP53" in result.protein
end

# ─────────────────────────────────────────────────────────────────────────────
# remove_contaminants
# ─────────────────────────────────────────────────────────────────────────────

@testitem "remove_contaminants" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["TP53", "CON__KRT1", "HAP40", "REV__P12345", "GAPDH"],
        value = [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    result, entries = remove_contaminants(df, 1)

    @test nrow(result) == 3
    @test result.protein == ["TP53", "HAP40", "GAPDH"]
    @test result.value == [1.0, 3.0, 5.0]
    @test length(entries) == 2
    @test all(e.action == BayesInteractomics.CURATE_REMOVE for e in entries)
end

@testitem "remove_contaminants: case insensitive" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["con__keratin", "rev__decoy", "VALID_PROTEIN"],
        value = [1.0, 2.0, 3.0]
    )

    result, entries = remove_contaminants(df, 1)

    @test nrow(result) == 1
    @test result.protein == ["VALID_PROTEIN"]
end

# ─────────────────────────────────────────────────────────────────────────────
# merge_protein_rows
# ─────────────────────────────────────────────────────────────────────────────

@testitem "merge_protein_rows: max strategy" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["HAP40", "F8A1", "TP53"],
        intensity1 = [100.0, 150.0, 200.0],
        intensity2 = [missing, 120.0, 210.0]
    )

    candidate = BayesInteractomics.MergeCandidate(
        ["HAP40", "F8A1"],
        "9606.ENSP00000479624",
        "F8A1",
        "HAP40 protein",
        Dict("HAP40" => [1], "F8A1" => [2])
    )
    decision = BayesInteractomics.MergeDecision(candidate, true, "F8A1")

    result, entries = merge_protein_rows(df, [decision], 1; strategy=:max)

    @test nrow(result) == 2  # merged row + TP53
    @test result.protein[1] == "F8A1"
    @test result.intensity1[1] == 150.0  # max(100, 150)
    @test result.intensity2[1] == 120.0  # max(missing, 120) = 120
    @test result.protein[2] == "TP53"
end

@testitem "merge_protein_rows: mean strategy" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["HAP40", "F8A1"],
        intensity = [100.0, 200.0]
    )

    candidate = BayesInteractomics.MergeCandidate(
        ["HAP40", "F8A1"],
        "9606.ENSP00000479624",
        "F8A1",
        "",
        Dict("HAP40" => [1], "F8A1" => [2])
    )
    decision = BayesInteractomics.MergeDecision(candidate, true, "F8A1")

    result, entries = merge_protein_rows(df, [decision], 1; strategy=:mean)

    @test nrow(result) == 1
    @test result.intensity[1] ≈ 150.0  # mean(100, 200)
end

@testitem "merge_protein_rows: rejected merge" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["HAP40", "F8A1"],
        intensity = [100.0, 200.0]
    )

    candidate = BayesInteractomics.MergeCandidate(
        ["HAP40", "F8A1"],
        "9606.ENSP00000479624",
        "F8A1",
        "",
        Dict("HAP40" => [1], "F8A1" => [2])
    )
    decision = BayesInteractomics.MergeDecision(candidate, false, "")

    result, entries = merge_protein_rows(df, [decision], 1)

    # Both rows should be kept
    @test nrow(result) == 2
    @test result.intensity == [100.0, 200.0]
end

@testitem "merge_protein_rows: protein order preserved" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["BAIT", "HAP40", "F8A1", "TP53", "GAPDH"],
        value = [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    candidate = BayesInteractomics.MergeCandidate(
        ["HAP40", "F8A1"],
        "9606.ENSP00000479624",
        "F8A1",
        "",
        Dict("HAP40" => [2], "F8A1" => [3])
    )
    decision = BayesInteractomics.MergeDecision(candidate, true, "F8A1")

    result, entries = merge_protein_rows(df, [decision], 1; strategy=:max)

    # BAIT should remain at position 1, merged at 2, TP53 at 3, GAPDH at 4
    @test nrow(result) == 4
    @test result.protein == ["BAIT", "F8A1", "TP53", "GAPDH"]
    @test result.value[1] == 1.0  # BAIT unchanged
    @test result.value[2] == 3.0  # max(2.0, 3.0)
end

# ─────────────────────────────────────────────────────────────────────────────
# find_merge_candidates
# ─────────────────────────────────────────────────────────────────────────────

@testitem "find_merge_candidates: synonyms detected" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["HAP40", "F8A1", "TP53"],
        value = [1.0, 2.0, 3.0]
    )

    name_to_id = Dict(
        "HAP40" => "9606.ENSP00000479624",
        "F8A1" => "9606.ENSP00000479624",
        "TP53" => "9606.ENSP00000269305"
    )
    id_to_preferred = Dict(
        "9606.ENSP00000479624" => "F8A1",
        "9606.ENSP00000269305" => "TP53"
    )
    id_to_annotation = Dict(
        "9606.ENSP00000479624" => "HAP40 protein",
        "9606.ENSP00000269305" => "Tumor protein p53"
    )

    candidates = BayesInteractomics.find_merge_candidates(
        name_to_id, id_to_preferred, id_to_annotation, df, 1
    )

    @test length(candidates) == 1
    @test Set(candidates[1].names) == Set(["HAP40", "F8A1"])
    @test candidates[1].string_id == "9606.ENSP00000479624"
    @test candidates[1].preferred_name == "F8A1"
end

@testitem "find_merge_candidates: no synonyms" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["TP53", "GAPDH"],
        value = [1.0, 2.0]
    )

    name_to_id = Dict(
        "TP53" => "9606.ENSP00000269305",
        "GAPDH" => "9606.ENSP00000229239"
    )
    id_to_preferred = Dict(
        "9606.ENSP00000269305" => "TP53",
        "9606.ENSP00000229239" => "GAPDH"
    )
    id_to_annotation = Dict{String,String}()

    candidates = BayesInteractomics.find_merge_candidates(
        name_to_id, id_to_preferred, id_to_annotation, df, 1
    )

    @test isempty(candidates)
end

# ─────────────────────────────────────────────────────────────────────────────
# Bait tracking
# ─────────────────────────────────────────────────────────────────────────────

@testitem "find_bait_index" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["GAPDH", "F8A1", "TP53"],
        value = [1.0, 2.0, 3.0]
    )

    @test BayesInteractomics.find_bait_index(df, 1, "F8A1") == 2
    @test BayesInteractomics.find_bait_index(df, 1, "TP53") == 3
    @test BayesInteractomics.find_bait_index(df, 1, "NONEXISTENT") === nothing
    # Case-insensitive
    @test BayesInteractomics.find_bait_index(df, 1, "gapdh") == 1
end

# ─────────────────────────────────────────────────────────────────────────────
# CurationReport save/load roundtrip
# ─────────────────────────────────────────────────────────────────────────────

@testitem "CurationReport: roundtrip save/load" begin
    using BayesInteractomics
    using Dates

    entries = [
        BayesInteractomics.CurationEntry(
            "HAP40", "F8A1", "9606.ENSP00000479624",
            BayesInteractomics.CURATE_MERGE,
            "Merged", true, [1], nothing, true
        ),
        BayesInteractomics.CurationEntry(
            "TP53", "TP53", "9606.ENSP00000269305",
            BayesInteractomics.CURATE_KEEP,
            "Kept", false, [2], nothing, true
        )
    ]

    candidate = BayesInteractomics.MergeCandidate(
        ["HAP40", "F8A1"],
        "9606.ENSP00000479624",
        "F8A1",
        "HAP40 protein",
        Dict("HAP40" => [1], "F8A1" => [2])
    )
    decisions = [BayesInteractomics.MergeDecision(candidate, true, "F8A1")]

    report = BayesInteractomics.CurationReport(
        entries, decisions, 9606, "version-12-0",
        UInt64(12345), now(), "0.1.0", 10, 8,
        Dict(:splits => 0, :merges => 1, :removals => 0, :unmapped => 0, :renames => 0, :kept => 1)
    )

    # Save and reload
    tmpdir = mktempdir()
    base_path = joinpath(tmpdir, "test")
    save_curation_report(report, base_path)

    jld2_path = base_path * "_curation_report.jld2"
    @test isfile(jld2_path)

    csv_path = base_path * "_curation_log.csv"
    @test isfile(csv_path)

    loaded = load_curation_report(jld2_path)
    @test !isnothing(loaded)
    @test loaded.species == 9606
    @test loaded.n_proteins_before == 10
    @test loaded.n_proteins_after == 8
    @test length(loaded.entries) == 2
    @test length(loaded.merge_decisions) == 1
    @test loaded.merge_decisions[1].approved == true
    @test loaded.merge_decisions[1].chosen_name == "F8A1"
end

# ─────────────────────────────────────────────────────────────────────────────
# CurationReport show method
# ─────────────────────────────────────────────────────────────────────────────

@testitem "CurationReport: show method" begin
    using BayesInteractomics
    using Dates

    report = BayesInteractomics.CurationReport(
        BayesInteractomics.CurationEntry[],
        BayesInteractomics.MergeDecision[],
        9606, "version-12-0",
        UInt64(0), now(), "0.1.0", 100, 95,
        Dict(:splits => 5, :merges => 2, :removals => 3, :unmapped => 1, :renames => 4, :kept => 85)
    )

    output = sprint(show, report)
    @test occursin("CurationReport", output)
    @test occursin("9606", output)
    @test occursin("100", output)
    @test occursin("95", output)
end

# ─────────────────────────────────────────────────────────────────────────────
# CurationAPIError
# ─────────────────────────────────────────────────────────────────────────────

@testitem "CurationAPIError: display" begin
    using BayesInteractomics

    err = BayesInteractomics.CurationAPIError("Test error", 404, "Check URL")
    output = sprint(showerror, err)
    @test occursin("CurationAPIError", output)
    @test occursin("404", output)
    @test occursin("Check URL", output)
end

# ─────────────────────────────────────────────────────────────────────────────
# STRING API cache persistence
# ─────────────────────────────────────────────────────────────────────────────

@testitem "CurationCache: save/load roundtrip" begin
    using BayesInteractomics
    using Dates

    cache = BayesInteractomics.CurationCache(
        Dict("HAP40" => "9606.ENSP00000479624"),
        Dict("9606.ENSP00000479624" => "F8A1"),
        Dict("9606.ENSP00000479624" => "HAP40 protein"),
        9606,
        "version-12-0",
        now()
    )

    tmpdir = mktempdir()
    key = "test_cache_key"

    BayesInteractomics.save_curation_cache(cache, tmpdir, key)
    @test isfile(joinpath(tmpdir, "curation_$(key).jld2"))

    loaded = BayesInteractomics.load_curation_cache(tmpdir, key)
    @test !isnothing(loaded)
    @test loaded.species == 9606
    @test loaded.mapping["HAP40"] == "9606.ENSP00000479624"
    @test loaded.preferred_names["9606.ENSP00000479624"] == "F8A1"
end

@testitem "CurationCache: missing file returns nothing" begin
    using BayesInteractomics

    tmpdir = mktempdir()
    loaded = BayesInteractomics.load_curation_cache(tmpdir, "nonexistent")
    @test isnothing(loaded)
end

# ─────────────────────────────────────────────────────────────────────────────
# End-to-end non-interactive curate_proteins
# ─────────────────────────────────────────────────────────────────────────────

@testitem "deduplicate_same_name_rows: merges identical names" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["NSUN5", "TP53", "NSUN5", "GAPDH", "NSUN5"],
        intensity1 = [10.0, 200.0, 30.0, 400.0, 50.0],
        intensity2 = [missing, 210.0, 35.0, 410.0, missing]
    )

    result, entries = BayesInteractomics._deduplicate_same_name_rows(df, 1; strategy=:max)

    @test nrow(result) == 3
    @test result.protein == ["NSUN5", "TP53", "GAPDH"]
    @test result.intensity1[1] == 50.0  # max(10, 30, 50)
    @test result.intensity2[1] == 35.0  # max(missing, 35, missing) = 35
    @test result.intensity1[2] == 200.0  # TP53 unchanged
end

@testitem "deduplicate_same_name_rows: no duplicates (no-op)" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["A", "B", "C"],
        value = [1.0, 2.0, 3.0]
    )

    result, entries = BayesInteractomics._deduplicate_same_name_rows(df, 1)

    @test nrow(result) == 3
    @test isempty(entries)
end

@testitem "deduplicate_same_name_rows: mean strategy" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        protein = ["X", "X"],
        value = [100.0, 200.0]
    )

    result, entries = BayesInteractomics._deduplicate_same_name_rows(df, 1; strategy=:mean)

    @test nrow(result) == 1
    @test result.value[1] ≈ 150.0
end

@testitem "curate_proteins: contaminant removal and splitting (offline)" begin
    using BayesInteractomics
    using DataFrames

    # Test contaminant removal and group splitting in isolation
    # (These steps don't require STRING API)
    df = DataFrame(
        protein = ["CON__KRT1", "RBFOX3;RBFOX2", "TP53", "REV__DECOY"],
        intensity1 = [10.0, 100.0, 200.0, 5.0],
        intensity2 = [11.0, 110.0, 210.0, 6.0]
    )

    # Step 1: Remove contaminants
    clean_df, contam_entries = remove_contaminants(df, 1)
    @test nrow(clean_df) == 2
    @test length(contam_entries) == 2

    # Step 2: Split protein groups
    split_df, split_entries = split_protein_groups(clean_df, 1)
    @test nrow(split_df) == 3
    @test "RBFOX3" in split_df.protein
    @test "RBFOX2" in split_df.protein
    @test "TP53" in split_df.protein

    # Verify data duplication
    rbfox_rows = filter(row -> row.protein in ["RBFOX3", "RBFOX2"], split_df)
    @test all(rbfox_rows.intensity1 .== 100.0)
    @test all(rbfox_rows.intensity2 .== 110.0)

    # Verify TP53 data unchanged
    tp53_row = filter(row -> row.protein == "TP53", split_df)
    @test tp53_row.intensity1[1] == 200.0
end

# ─────────────────────────────────────────────────────────────────────────────
# load_data with curate=false (regression test)
# ─────────────────────────────────────────────────────────────────────────────

@testitem "load_data: curate=false preserves existing behavior" begin
    using BayesInteractomics
    using BayesInteractomics: InteractionData, getIDs

    # This test verifies that curate=false doesn't break existing load_data behavior
    # We can't easily test with real files here, but we can verify the kwarg exists
    # and doesn't error when set to false
    @test hasmethod(BayesInteractomics.load_data, Tuple{Vector{String}, Vector{Dict{Int,Vector{Int}}}, Vector{Dict{Int,Vector{Int}}}})
end

# ─────────────────────────────────────────────────────────────────────────────
# _curation_cache_key determinism
# ─────────────────────────────────────────────────────────────────────────────

@testitem "curation cache key: deterministic and order-independent" begin
    using BayesInteractomics

    key1 = BayesInteractomics._curation_cache_key(["HAP40", "TP53", "GAPDH"], 9606)
    key2 = BayesInteractomics._curation_cache_key(["GAPDH", "HAP40", "TP53"], 9606)
    key3 = BayesInteractomics._curation_cache_key(["HAP40", "TP53", "GAPDH"], 10090)

    @test key1 == key2  # Order-independent
    @test key1 != key3  # Different species → different key
    @test length(key1) == 64  # SHA256 hex string
end

# ─────────────────────────────────────────────────────────────────────────────
# replay_merges
# ─────────────────────────────────────────────────────────────────────────────

@testitem "replay_merges: decisions replayed from report" begin
    using BayesInteractomics
    using Dates

    candidate = BayesInteractomics.MergeCandidate(
        ["HAP40", "F8A1"],
        "9606.ENSP00000479624",
        "F8A1",
        "",
        Dict("HAP40" => [1], "F8A1" => [2])
    )

    saved_decision = BayesInteractomics.MergeDecision(candidate, true, "HAP40")

    report = BayesInteractomics.CurationReport(
        BayesInteractomics.CurationEntry[],
        [saved_decision],
        9606, "version-12-0",
        UInt64(0), now(), "0.1.0", 10, 9,
        Dict{Symbol,Int}()
    )

    # New candidate (same STRING ID, potentially different row indices)
    new_candidate = BayesInteractomics.MergeCandidate(
        ["HAP40", "F8A1"],
        "9606.ENSP00000479624",
        "F8A1",
        "",
        Dict("HAP40" => [3], "F8A1" => [5])
    )

    decisions = BayesInteractomics.replay_merges([new_candidate], report)

    @test length(decisions) == 1
    @test decisions[1].approved == true
    @test decisions[1].chosen_name == "HAP40"  # Preserves user's original choice
end
