"""
    test_ppi_enrichment.jl

Tests for prey-prey network enrichment functionality.
Unit tests for Bayesian weighting run without network access.
Integration tests that hit the STRING API are guarded by ENV variable.
"""

using TestItemRunner
using DataFrames

# Helper to create a mock star-topology InteractionNetwork for enrichment tests
# NOTE: This function requires BayesInteractomics, Graphs, SimpleWeightedGraphs,
# GraphPlot, Compose, and DataFrames to already be loaded in the calling scope.
function create_mock_network_for_enrichment()
    mock_results = DataFrame(
        Protein = ["HTT", "HAP40", "CCT2", "CCT5", "CCT7", "TCP1", "VCP"],
        PosteriorProbability = [0.99, 0.95, 0.92, 0.88, 0.85, 0.80, 0.75],
        BayesFactor = [500.0, 200.0, 100.0, 80.0, 60.0, 40.0, 20.0],
        q_value = [0.0001, 0.001, 0.002, 0.005, 0.008, 0.01, 0.02],
        mean_log2FC = [5.0, 4.5, 3.8, 3.2, 2.8, 2.5, 2.0]
    )

    ar = NetworkAnalysisResult(mock_results, bait_protein="HTT", bait_index=1)
    net = build_network(ar, posterior_threshold=0.5, q_threshold=0.05)
    return net
end

# ============================================================================
# Unit tests — Bayesian weight computation (no network access needed)
# ============================================================================

@testitem "Bayesian weight: score 0 returns near prior" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

    # Access internal function via the extension module
    ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsNetworkExt)
    compute_weight = ext._compute_prey_prey_weight

    # STRING API v12+ returns scores as floats in [0, 1]
    w = compute_weight(0.001, 0.05, 0.002)
    # With STRING score ~0, BF should be very small, posterior near but below prior
    @test w < 0.1
    @test w >= 0.0
end

@testitem "Bayesian weight: score ~1.0 returns near 1.0" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

    ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsNetworkExt)
    compute_weight = ext._compute_prey_prey_weight

    # STRING scores are floats in [0, 1]
    w = compute_weight(0.999, 0.05, 0.002)
    @test w > 0.99
    @test w <= 1.0
end

@testitem "Bayesian weight: score 0.7 returns reasonable value" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

    ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsNetworkExt)
    compute_weight = ext._compute_prey_prey_weight

    w = compute_weight(0.7, 0.05, 0.002)
    # High confidence STRING score with co-purification prior should be high
    @test w > 0.5
    @test w < 1.0
end

@testitem "Bayesian weight: monotonicity" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

    ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsNetworkExt)
    compute_weight = ext._compute_prey_prey_weight

    # STRING scores on [0, 1] scale
    scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    weights = [compute_weight(s, 0.05, 0.002) for s in scores]

    # Weights should be strictly monotonically increasing
    for i in 2:length(weights)
        @test weights[i] > weights[i-1]
    end
end

@testitem "Bayesian weight: higher prior increases posterior" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

    ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsNetworkExt)
    compute_weight = ext._compute_prey_prey_weight

    w_low_prior = compute_weight(0.5, 0.01, 0.002)
    w_high_prior = compute_weight(0.5, 0.10, 0.002)

    @test w_high_prior > w_low_prior
end

@testitem "Per-channel weight computation" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

    ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsNetworkExt)
    compute_channels = ext._compute_prey_prey_weight_channels

    # STRING channel scores are also on [0, 1] scale
    channel_scores = Dict(:experimental => 0.6, :database => 0.4)
    channels = [:experimental, :database]

    w = compute_channels(channel_scores, channels, 0.05, 0.002)
    @test w > 0.0
    @test w <= 1.0

    # With more evidence channels, posterior should be high
    @test w > 0.5
end

@testitem "Per-channel weight: empty channels" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

    ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsNetworkExt)
    compute_channels = ext._compute_prey_prey_weight_channels

    # All zero scores
    channel_scores = Dict(:experimental => 0.0, :database => 0.0)
    channels = [:experimental, :database]

    w = compute_channels(channel_scores, channels, 0.05, 0.002)
    # With no evidence, posterior equals prior
    @test isapprox(w, 0.05, atol=0.01)
end

# ============================================================================
# Unit tests — Caching utilities (no network access)
# ============================================================================

@testitem "Cache key determinism" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

    ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsNetworkExt)
    cache_key = ext._cache_key

    proteins = ["ProteinA", "ProteinB", "ProteinC"]

    # Same inputs produce same key
    key1 = cache_key(proteins, 9606, :physical)
    key2 = cache_key(proteins, 9606, :physical)
    @test key1 == key2

    # Order-independent
    key3 = cache_key(reverse(proteins), 9606, :physical)
    @test key1 == key3

    # Different inputs produce different keys
    key4 = cache_key(proteins, 10090, :physical)  # Different species
    @test key1 != key4

    key5 = cache_key(proteins, 9606, :functional)  # Different network type
    @test key1 != key5
end

@testitem "Form encoding" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

    ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsNetworkExt)
    form_encode = ext._form_encode

    params = Dict("key1" => "value1", "key2" => "value2")
    encoded = form_encode(params)

    # Should contain both key=value pairs
    @test occursin("key1=value1", encoded)
    @test occursin("key2=value2", encoded)
    @test occursin("&", encoded)
end

@testitem "PPIEnrichmentConfig defaults" begin
    using BayesInteractomics

    config = PPIEnrichmentConfig()
    @test config.species == 9606
    @test config.min_string_score == 700
    @test config.network_type == :physical
    @test config.co_purification_prior == 0.05
    @test config.string_prior == 0.002
    @test config.channels == [:combined]
    @test config.use_bayesian_weighting == true
    @test config.cache_dir == ""
    @test config.offline_file == ""
    @test config.caller_identity == "BayesInteractomics.jl"
end

@testitem "PPIEnrichmentConfig custom values" begin
    using BayesInteractomics

    config = PPIEnrichmentConfig(
        species = 10090,
        min_string_score = 400,
        network_type = :functional,
        co_purification_prior = 0.10
    )
    @test config.species == 10090
    @test config.min_string_score == 400
    @test config.network_type == :functional
    @test config.co_purification_prior == 0.10
end

# ============================================================================
# Unit tests — enrich_network with empty/minimal networks
# ============================================================================

@testitem "enrich_network: empty network returns unchanged" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose
    using DataFrames

    # Build empty network
    mock_results = DataFrame(
        Protein = ["A"],
        PosteriorProbability = [0.1],  # Below threshold
        BayesFactor = [0.5],
        q_value = [0.5],
        mean_log2FC = [0.1]
    )
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5, q_threshold=0.05)

    # Should return without error even for empty network
    result = enrich_network(net, species=9606, verbose=false)
    @test nv(result.graph) == nv(net.graph)
end

@testitem "enrich_network: single prey returns with edge_source" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose
    using DataFrames

    mock_results = DataFrame(
        Protein = ["PreyA"],
        PosteriorProbability = [0.95],
        BayesFactor = [100.0],
        q_value = [0.001],
        mean_log2FC = [3.0]
    )
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait", bait_index=1)
    net = build_network(ar, posterior_threshold=0.5, q_threshold=0.05)

    # Only 1 prey — no prey-prey possible, should return with edge_source
    result = enrich_network(net, species=9606, verbose=false)
    @test hasproperty(result.edge_attributes, :edge_source)
end

@testitem "enrich_network: parameter validation" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose
    using DataFrames

    mock_results = DataFrame(
        Protein = ["A", "B"],
        PosteriorProbability = [0.95, 0.90],
        BayesFactor = [100.0, 50.0],
        q_value = [0.001, 0.01],
        mean_log2FC = [3.0, 2.5]
    )
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait", bait_index=1)
    net = build_network(ar, posterior_threshold=0.5, q_threshold=0.05)

    # Invalid network_type
    @test_throws ErrorException enrich_network(net, network_type=:invalid, verbose=false)

    # Invalid min_string_score
    @test_throws ErrorException enrich_network(net, min_string_score=-1, verbose=false)
    @test_throws ErrorException enrich_network(net, min_string_score=1001, verbose=false)
end

# ============================================================================
# Unit tests — edge_source_summary
# ============================================================================

@testitem "edge_source_summary: basic network" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose
    using DataFrames

    mock_results = DataFrame(
        Protein = ["A", "B", "C"],
        PosteriorProbability = [0.95, 0.90, 0.85],
        BayesFactor = [100.0, 50.0, 20.0],
        q_value = [0.001, 0.01, 0.02],
        mean_log2FC = [3.0, 2.5, 2.0]
    )
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait", bait_index=1)
    net = build_network(ar, posterior_threshold=0.5, q_threshold=0.05)

    ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsNetworkExt)
    summary = ext.edge_source_summary(net)

    # Without enrichment, all edges are experimental
    @test haskey(summary, "experimental")
    @test summary["experimental"] == ne(net.graph)
end

# ============================================================================
# Integration tests — STRING API (guarded by environment variable)
# ============================================================================

@testitem "STRING API: query real proteins" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose
    using DataFrames

    if get(ENV, "BAYESINTERACTOMICS_TEST_NETWORK", "false") != "true"
        @test true  # Skip silently
        return
    end

    proteins = ["TP53", "MDM2", "BRCA1"]
    result = query_string_ppi(proteins, 9606; network_type=:physical)

    @test hasproperty(result, :species)
    @test result.species == 9606
    @test result.n_query_proteins == 3
    @test length(result.protein_mapping) > 0

    # TP53-MDM2 is one of the best-known interactions
    @test nrow(result.interactions) > 0
end

@testitem "STRING API: full enrichment pipeline" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose
    using DataFrames

    if get(ENV, "BAYESINTERACTOMICS_TEST_NETWORK", "false") != "true"
        @test true  # Skip silently
        return
    end

    # Use well-known interacting proteins for reliable STRING results
    mock_results = DataFrame(
        Protein = ["TP53", "MDM2", "MDM4", "CDKN1A", "BAX", "BCL2"],
        PosteriorProbability = [0.99, 0.95, 0.92, 0.88, 0.85, 0.80],
        BayesFactor = [500.0, 200.0, 100.0, 80.0, 60.0, 40.0],
        q_value = [0.0001, 0.001, 0.002, 0.005, 0.008, 0.01],
        mean_log2FC = [5.0, 4.5, 3.8, 3.2, 2.8, 2.5]
    )
    ar = NetworkAnalysisResult(mock_results, bait_protein="TP53", bait_index=1)
    net = build_network(ar, posterior_threshold=0.5, q_threshold=0.05)

    # Run enrichment with real STRING API — use low threshold for testing
    enriched = enrich_network(net,
        species = 9606,
        min_string_score = 150,  # Very low threshold for testing
        network_type = :physical,
        verbose = true
    )

    # Basic checks
    @test nv(enriched.graph) == nv(net.graph)  # Same nodes
    @test ne(enriched.graph) >= ne(net.graph)  # At least as many edges

    # edge_source column always exists
    @test hasproperty(enriched.edge_attributes, :edge_source)
    @test hasproperty(enriched.edge_attributes, :string_score)

    # Original edges are experimental
    n_original = ne(net.graph)
    if n_original > 0
        @test any(enriched.edge_attributes.edge_source .== "experimental")
    end

    # If new edges were added, check enrichment-specific attributes
    if ne(enriched.graph) > ne(net.graph)
        @test any(enriched.edge_attributes.edge_source .== "public_ppi")
        @test hasproperty(enriched.node_attributes, :string_id)
        @test hasproperty(enriched.node_attributes, :n_prey_prey_edges)
        @test haskey(enriched.threshold_used, :min_string_score)
        @test haskey(enriched.threshold_used, :string_species)
    end

    # Downstream functions work on enriched network
    stats = network_statistics(enriched)
    @test stats.n_nodes > 0

    cm = centrality_measures(enriched)
    @test length(cm.protein_names) == nv(enriched.graph)

    communities = detect_communities(enriched, algorithm=:louvain)
    @test communities.n_communities >= 1
end

@testitem "STRING API: enriched network export" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose
    using DataFrames

    if get(ENV, "BAYESINTERACTOMICS_TEST_NETWORK", "false") != "true"
        @test true  # Skip silently
        return
    end

    mock_results = DataFrame(
        Protein = ["HTT", "HAP40", "CCT2", "CCT5", "CCT7", "TCP1", "VCP"],
        PosteriorProbability = [0.99, 0.95, 0.92, 0.88, 0.85, 0.80, 0.75],
        BayesFactor = [500.0, 200.0, 100.0, 80.0, 60.0, 40.0, 20.0],
        q_value = [0.0001, 0.001, 0.002, 0.005, 0.008, 0.01, 0.02],
        mean_log2FC = [5.0, 4.5, 3.8, 3.2, 2.8, 2.5, 2.0]
    )
    ar = NetworkAnalysisResult(mock_results, bait_protein="HTT", bait_index=1)
    net = build_network(ar, posterior_threshold=0.5, q_threshold=0.05)
    enriched = enrich_network(net, species=9606, min_string_score=400, verbose=false)

    # Export to GraphML and verify edge_source attribute
    tmpfile = tempname() * ".graphml"
    try
        export_graphml(enriched, tmpfile)
        content = read(tmpfile, String)

        @test occursin("edge_source", content)
        @test occursin("experimental", content)

        if ne(enriched.graph) > ne(net.graph)
            @test occursin("public_ppi", content)
        end
    finally
        rm(tmpfile; force=true)
    end

    # Export edge list and verify
    tmpfile2 = tempname() * ".csv"
    try
        export_edgelist(enriched, tmpfile2)
        @test isfile(tmpfile2)
    finally
        rm(tmpfile2; force=true)
    end
end

@testitem "STRING API: cache operations" begin
    using BayesInteractomics
    using Graphs, SimpleWeightedGraphs, GraphPlot, Compose

    if get(ENV, "BAYESINTERACTOMICS_TEST_NETWORK", "false") != "true"
        @test true  # Skip silently
        return
    end

    # Use temp directory for test cache
    test_cache = joinpath(tempdir(), "bayesinteractomics_test_cache_$(rand(1000:9999))")

    try
        # First query — should hit STRING API
        proteins = ["TP53", "MDM2"]
        result1 = query_string_ppi(proteins, 9606; cache_dir=test_cache)

        # Check cache info
        info = ppi_cache_info(cache_dir=test_cache)
        @test info.n_entries >= 1

        # Second query — should hit cache
        result2 = query_string_ppi(proteins, 9606; cache_dir=test_cache)
        @test result2.n_query_proteins == result1.n_query_proteins

        # Clear cache
        clear_ppi_cache(cache_dir=test_cache)
        info2 = ppi_cache_info(cache_dir=test_cache)
        @test info2.n_entries == 0
    finally
        rm(test_cache; force=true, recursive=true)
    end
end
