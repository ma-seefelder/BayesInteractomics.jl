# test/embeddings/test_embedding_types.jl
# Similarity & Embeddings: type + config round-trip tests.
# Contract: EmbeddingsResult / ConditionSimilarityResult JLD2 round-trip,
# CACHE_VERSION 23 → 24.

@testitem "EmbeddingsConfig defaults" begin
    using BayesInteractomics
    cfg = EmbeddingsConfig()
    @test cfg.method === :umap
    @test cfg.seed == 42
    @test cfg.supervised === false
    @test cfg.n_neighbors == 15
    @test cfg.min_dist == 0.1
    @test cfg.top_k_jaccard == 50
    @test cfg.run_embeddings === true
end

@testitem "_validate_embeddings_config rejects invalid fields" begin
    using BayesInteractomics
    @test BayesInteractomics._validate_embeddings_config(EmbeddingsConfig()) === nothing
    @test_throws ArgumentError BayesInteractomics._validate_embeddings_config(EmbeddingsConfig(method=:bogus))
    @test_throws ArgumentError BayesInteractomics._validate_embeddings_config(EmbeddingsConfig(seed=-1))
    @test_throws ArgumentError BayesInteractomics._validate_embeddings_config(EmbeddingsConfig(n_neighbors=1))
    @test_throws ArgumentError BayesInteractomics._validate_embeddings_config(EmbeddingsConfig(min_dist=-0.01))
    @test_throws ArgumentError BayesInteractomics._validate_embeddings_config(EmbeddingsConfig(top_k_jaccard=0))
end

@testitem "_config_snapshot returns expected NamedTuple keys" begin
    using BayesInteractomics
    snap = BayesInteractomics._config_snapshot(EmbeddingsConfig())
    @test keys(snap) == (:method, :seed, :n_neighbors, :min_dist, :supervised, :top_k_jaccard)
    @test snap.method === :umap
end

@testitem "EmbeddingsResult positional construction" begin
    using BayesInteractomics
    snap = BayesInteractomics._config_snapshot(EmbeddingsConfig())
    er = EmbeddingsResult(
        zeros(4, 2), [12.0, 8.0],
        (condition=String[], replicate=Int[], experiment=Int[], protocol=Int[]),
        :complete_case,
        nothing, nothing, nothing, Symbol[], String[],
        snap,
    )
    @test er.sample_filter_level === :complete_case
    @test er.sample_umap_coords === nothing
    @test er.protein_umap_coords === nothing
    @test er.config_snapshot === snap
end

@testitem "ConditionSimilarityResult positional construction" begin
    using BayesInteractomics
    cs = ConditionSimilarityResult(
        ["A", "B"],
        [1.0 0.5; 0.5 1.0],
        [1.0 0.3; 0.3 1.0],
        [1.0 0.7; 0.7 1.0],
        [1.0 0.2; 0.2 1.0],
        [5 5; 5 5],
        50,
        zeros(Int, 1, 2), [0.5], [1, 2],
        :average,
    )
    @test cs.linkage === :average
    @test size(cs.spearman_log10_bf) == (2, 2)
    @test cs.top_k_used == 50
end

@testitem "CACHE_VERSION bumped 23 -> 24" begin
    using BayesInteractomics
    @test BayesInteractomics.CACHE_VERSION == 26
end

@testitem "AnalysisResult.embeddings JLD2 round-trip" begin
    using BayesInteractomics
    using DataFrames, Dates
    # Inline minimal AnalysisResult (cannot include test/fixtures/test_fixtures.jl from
    # inside a @testitem — that file uses TestItemRunner @testsetup/@testmodule macros
    # which aren't visible at testitem eval time). Build the AnalysisResult directly.
    snap = BayesInteractomics._config_snapshot(EmbeddingsConfig())
    er = EmbeddingsResult(
        zeros(4, 2), [12.0, 8.0],
        (condition=String[], replicate=Int[], experiment=Int[], protocol=Int[]),
        :complete_case,
        nothing, nothing, nothing, Symbol[], String[],
        snap,
    )
    empty_df = DataFrame()
    ar = BayesInteractomics.AnalysisResult(
        empty_df, empty_df, nothing, nothing, nothing, nothing, nothing, :bma,
        nothing, nothing, UInt64(0), UInt64(0), now(), "test",
        "BAIT_WT", 1, nothing, nothing, nothing, :loaded,
        nothing, nothing,
        false,           # is_calibrated
        er,              # embeddings
    )
    tmp = tempname() * ".jld2"
    BayesInteractomics.save_result(ar, tmp)
    loaded = BayesInteractomics.load_result(tmp)
    @test loaded !== nothing
    @test loaded.embeddings isa EmbeddingsResult
    @test loaded.embeddings.sample_filter_level === ar.embeddings.sample_filter_level
    rm(tmp; force=true)
end

@testitem "Legacy CACHE_VERSION 23 cache returns nothing from load_result" begin
    using BayesInteractomics, JLD2
    tmp = tempname() * ".jld2"
    jldsave(tmp; cache_version = 23, copula_results = nothing)
    loaded = BayesInteractomics.load_result(tmp)
    @test loaded === nothing
    rm(tmp; force=true)
end
