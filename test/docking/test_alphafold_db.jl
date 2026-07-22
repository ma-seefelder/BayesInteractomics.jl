# ═══════════════════════════════════════════════════════════════════════════════
# Tests for AlphaFold DB API client and structural dockability scoring
# ═══════════════════════════════════════════════════════════════════════════════

@testitem "compute_dockability" begin
    using BayesInteractomics

    # Formula: mean_pLDDT / 100 * (1 - frac_disordered)
    @test BayesInteractomics.Docking.compute_dockability(80.0, 0.1) ≈ 0.72
    @test BayesInteractomics.Docking.compute_dockability(100.0, 0.0) ≈ 1.0
    @test BayesInteractomics.Docking.compute_dockability(0.0, 1.0) ≈ 0.0
    @test BayesInteractomics.Docking.compute_dockability(50.0, 0.5) ≈ 0.25
    @test BayesInteractomics.Docking.compute_dockability(72.35, 0.18) ≈ (72.35 / 100.0) * (1.0 - 0.18)
    # Boundary: all confident residues
    @test BayesInteractomics.Docking.compute_dockability(95.0, 0.0) ≈ 0.95
end

@testitem "pLDDT cache round-trip" begin
    using BayesInteractomics
    using JLD2
    import Dates

    # Use a temp directory to avoid polluting real cache
    test_dir = mktempdir()
    cache_path = joinpath(test_dir, "TEST_PROTEIN.jld2")

    # Save
    JLD2.jldsave(cache_path;
        mean_plddt=72.5,
        frac_disordered=0.15,
        n_residues=393,
        timestamp=string(Dates.now()))

    # Verify round-trip via direct JLD2 load
    data = JLD2.load(cache_path)
    @test data["mean_plddt"] ≈ 72.5
    @test data["frac_disordered"] ≈ 0.15
    @test data["n_residues"] == 393
end

@testitem "pLDDT cache miss returns nothing" begin
    using BayesInteractomics

    result = BayesInteractomics.Docking._load_plddt_cache("NONEXISTENT_PROTEIN_ZZZZZ")
    @test result === nothing
end

@testitem "fetch_alphafold_plddt empty ID" begin
    using BayesInteractomics

    result = BayesInteractomics.Docking.fetch_alphafold_plddt("")
    @test result === nothing

    result2 = BayesInteractomics.Docking.fetch_alphafold_plddt("   ")
    @test result2 === nothing
end

@testitem "fetch_dockability_scores empty DataFrame" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(Protein=String[])
    scores, uniprot_map = BayesInteractomics.Docking.fetch_dockability_scores(df; resolve_uniprot=false)
    @test isempty(scores)
    @test isempty(uniprot_map)
end

@testitem "fetch_dockability_scores missing UniProt falls back to 0.5" begin
    using BayesInteractomics
    using DataFrames

    # DataFrame without uniprot_id column -- all proteins get neutral 0.5
    df = DataFrame(Protein=["ProtA", "ProtB"])
    scores, uniprot_map = BayesInteractomics.Docking.fetch_dockability_scores(df; resolve_uniprot=false)
    @test scores["ProtA"] == 0.5
    @test scores["ProtB"] == 0.5
    @test isempty(uniprot_map)  # no cache hits for fake names
end

@testitem "rate limiter constant exists" begin
    using BayesInteractomics

    @test BayesInteractomics.Docking.ALPHAFOLD_RATE_LIMIT_MS == 500
    @test BayesInteractomics.Docking.ALPHAFOLD_TIMEOUT_S == 30
end

# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests for composite ranking in _filter_candidates
# ═══════════════════════════════════════════════════════════════════════════════

@testitem "composite score ranking: high pLDDT outranks low pLDDT at equal posterior" begin
    using BayesInteractomics

    # Two proteins with identical posterior but different structural quality
    # Without UniProt IDs, both get neutral dockability=0.5, so same composite
    # Test the scoring formula directly instead
    w = 0.3
    # Protein A: high pLDDT (dockability=0.9), posterior=0.85
    score_A = (1.0 - w) * 0.85 + w * 0.9
    # Protein B: low pLDDT (dockability=0.2), posterior=0.85
    score_B = (1.0 - w) * 0.85 + w * 0.2
    @test score_A > score_B

    # Also verify formula matches the dockability definition
    dock_A = BayesInteractomics.Docking.compute_dockability(90.0, 0.0)  # 0.9
    dock_B = BayesInteractomics.Docking.compute_dockability(30.0, 0.6)  # 0.12
    @test dock_A > dock_B
    @test (1.0 - w) * 0.85 + w * dock_A > (1.0 - w) * 0.85 + w * dock_B
end

@testitem "composite score with dockability_weight=0 equals posterior ranking" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        Protein=["X", "Y", "Z"],
        posterior_prob=[0.99, 0.85, 0.92],
        PEP=[0.001, 0.01, 0.005],
        is_detected=[true, true, true]
    )
    config = DockingConfig(posterior_threshold=0.8, pep_threshold=0.01, dockability_weight=0.0)
    candidates = BayesInteractomics.Docking._filter_candidates(df, config)

    # With w=0.0, composite = 1.0*posterior + 0.0*dockability = posterior
    @test candidates.Protein[1] == "X"  # 0.99
    @test candidates.Protein[2] == "Z"  # 0.98
    @test candidates.Protein[3] == "Y"  # 0.96
end

@testitem "_filter_candidates output has dockability columns" begin
    using BayesInteractomics
    using DataFrames

    df = DataFrame(
        Protein=["A"],
        posterior_prob=[0.9],
        PEP=[0.01],
        is_detected=[true]
    )
    config = DockingConfig(posterior_threshold=0.8, pep_threshold=0.01)
    candidates = BayesInteractomics.Docking._filter_candidates(df, config)

    @test hasproperty(candidates, :composite_score)
    @test hasproperty(candidates, :mean_plddt_monomer)
    @test hasproperty(candidates, :frac_disordered_monomer)
    @test nrow(candidates) == 1  # Not filtered out
end

@testitem "all PEP-passing candidates retained (deprioritized not filtered)" begin
    using BayesInteractomics
    using DataFrames

    # 5 proteins all passing threshold -- none should be removed
    df = DataFrame(
        Protein=["A", "B", "C", "D", "E"],
        posterior_prob=[0.99, 0.995, 0.992, 0.997, 0.991],
        PEP=[0.001, 0.002, 0.005, 0.008, 0.01],
        is_detected=[true, true, true, true, true]
    )
    config = DockingConfig(posterior_threshold=0.8, pep_threshold=0.01)
    candidates = BayesInteractomics.Docking._filter_candidates(df, config)
    @test nrow(candidates) == 5  # All retained, just reordered
end

# ═══════════════════════════════════════════════════════════════════════════════
# UniProt ID resolution tests
# ═══════════════════════════════════════════════════════════════════════════════

@testitem "UniProt mapping cache round-trip" begin
    using BayesInteractomics, JLD2

    # Test cache save/load via JLD2 directly
    cache_dir = mktempdir()
    mapping = Dict("9606_TP53" => "P04637", "9606_BRCA1" => "P38398")

    cache_path = joinpath(cache_dir, "mapping.jld2")
    JLD2.save(cache_path, "mapping", mapping)
    loaded = JLD2.load(cache_path, "mapping")
    @test loaded["9606_TP53"] == "P04637"
    @test loaded["9606_BRCA1"] == "P38398"
    @test length(loaded) == 2
end

@testitem "DockingConfig species default" begin
    using BayesInteractomics
    dc = DockingConfig()
    @test dc.species == 9606
    dc2 = DockingConfig(species=10090)  # mouse
    @test dc2.species == 10090
end

@testitem "fetch_dockability_scores returns Tuple with uniprot_map" begin
    using BayesInteractomics, DataFrames

    # Create minimal candidates DataFrame
    df = DataFrame(
        Protein = ["ProtA", "ProtB", "ProtC"],
        posterior_prob = [0.95, 0.90, 0.85],
        BFDR = [0.001, 0.01, 0.02],
        is_detected = [true, true, true],
        bayes_factor = [100.0, 50.0, 20.0]
    )

    # Test fetch_dockability_scores with resolve_uniprot=false (no API calls)
    scores, uniprot_map = BayesInteractomics.Docking.fetch_dockability_scores(df; resolve_uniprot=false)

    # Without resolve, all proteins get neutral 0.5
    @test all(v == 0.5 for v in values(scores))

    # Verify uniprot_map is returned (empty since no cached mappings)
    @test uniprot_map isa Dict{String, String}

    # Verify uniprot_id column can be added from uniprot_map
    df.uniprot_id = map(row -> get(uniprot_map, row.Protein, ""), eachrow(df))
    @test hasproperty(df, :uniprot_id)
    @test all(row -> row.uniprot_id isa String, eachrow(df))
end

@testitem "_filter_candidates adds uniprot_id column" begin
    using BayesInteractomics, DataFrames

    # Both proteins must PASS the PEP filter (PEP <= pep_threshold) for the 2-candidate
    # assertion below to hold — _filter_candidates removes PEP-failing rows (see request_generator.jl).
    df = DataFrame(
        Protein=["A", "B"],
        posterior_prob=[0.9, 0.85],
        PEP=[0.01, 0.01],
        is_detected=[true, true]
    )
    config = DockingConfig(posterior_threshold=0.8, pep_threshold=0.01)
    candidates = BayesInteractomics.Docking._filter_candidates(df, config)

    @test hasproperty(candidates, :uniprot_id)
    @test all(row -> row.uniprot_id isa String, eachrow(candidates))
    @test nrow(candidates) == 2
end
