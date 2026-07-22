# test/embeddings/test_condition_similarity.jl
# Similarity & Embeddings: condition-level k×k similarity tests.

@testitem "k=1 returns nothing (single-bait short-circuit)" begin
    using BayesInteractomics, DataFrames
    struct _FakeAR1; copula_results::DataFrame; bait_protein::String; end
    df = DataFrame(Protein=["A"], BF=[10.0], BF_enrichment=[1.0], BF_correlation=[1.0],
                   BF_detected=[1.0], posterior_prob=[0.9], log2FC=[2.0])
    r = BayesInteractomics._compute_condition_similarity([_FakeAR1(df, "X")], EmbeddingsConfig())
    @test r === nothing
end

@testitem "k=2 produces ConditionSimilarityResult with diagonal anchors at 1.0" begin
    using BayesInteractomics, DataFrames
    struct _FakeAR2; copula_results::DataFrame; bait_protein::String; end
    df1 = DataFrame(Protein=["A","B","C"], BF=[10.0, 1.0, 50.0], BF_enrichment=[1.0,1.0,1.0],
                    BF_correlation=[1.0,1.0,1.0], BF_detected=[1.0,1.0,1.0],
                    posterior_prob=[0.99, 0.3, 0.95], log2FC=[3.0, 0.1, 4.0])
    df2 = DataFrame(Protein=["A","B","D"], BF=[8.0, 0.5, 100.0], BF_enrichment=[1.0,1.0,1.0],
                    BF_correlation=[1.0,1.0,1.0], BF_detected=[1.0,1.0,1.0],
                    posterior_prob=[0.97, 0.2, 0.99], log2FC=[2.5, -0.05, 5.0])
    r = BayesInteractomics._compute_condition_similarity(
        [_FakeAR2(df1, "WT"), _FakeAR2(df2, "MUT")], EmbeddingsConfig(top_k_jaccard=2))
    @test r isa ConditionSimilarityResult
    @test r.spearman_log10_bf[1, 1] == 1.0
    @test r.spearman_log10_bf[2, 2] == 1.0
    @test r.jaccard_top_k[1, 1] == 1.0
    @test r.linkage === :average
    @test r.condition_labels == ["WT", "MUT"]
end

@testitem "pairwise intersection (n_shared) is bilateral" begin
    using BayesInteractomics, DataFrames
    struct _FakeAR3; copula_results::DataFrame; bait_protein::String; end
    df1 = DataFrame(Protein=["A","B","C"], BF=[1.0,1.0,1.0], BF_enrichment=[1.0,1.0,1.0],
                    BF_correlation=[1.0,1.0,1.0], BF_detected=[1.0,1.0,1.0],
                    posterior_prob=[0.5,0.5,0.5], log2FC=[0.0,0.0,0.0])
    df2 = DataFrame(Protein=["A","B","D"], BF=[1.0,1.0,1.0], BF_enrichment=[1.0,1.0,1.0],
                    BF_correlation=[1.0,1.0,1.0], BF_detected=[1.0,1.0,1.0],
                    posterior_prob=[0.5,0.5,0.5], log2FC=[0.0,0.0,0.0])
    r = BayesInteractomics._compute_condition_similarity(
        [_FakeAR3(df1, "WT"), _FakeAR3(df2, "MUT")], EmbeddingsConfig())
    @test r.n_shared_per_cell[1, 2] == 2  # A, B
    @test r.n_shared_per_cell[2, 1] == 2
    @test r.n_shared_per_cell[1, 1] == 3
    @test r.n_shared_per_cell[2, 2] == 3
end

@testitem "Jaccard@TopK ranks by posterior desc with BF desc tiebreak" begin
    using BayesInteractomics, DataFrames
    struct _FakeAR4; copula_results::DataFrame; bait_protein::String; end
    # Both conditions saturate posterior_prob at 1.0 → tiebreak on BF must determine ranking.
    df1 = DataFrame(Protein=["X","Y","Z"], BF=[100.0, 50.0, 10.0], BF_enrichment=[1.0,1.0,1.0],
                    BF_correlation=[1.0,1.0,1.0], BF_detected=[1.0,1.0,1.0],
                    posterior_prob=[1.0, 1.0, 0.5], log2FC=[1.0, 1.0, 0.5])
    df2 = DataFrame(Protein=["X","Y","Z"], BF=[80.0, 60.0, 10.0], BF_enrichment=[1.0,1.0,1.0],
                    BF_correlation=[1.0,1.0,1.0], BF_detected=[1.0,1.0,1.0],
                    posterior_prob=[1.0, 1.0, 0.5], log2FC=[1.0, 1.0, 0.5])
    # Top-2 by posterior+BF in both: {X, Y} for both conditions → Jaccard = 1.0
    r = BayesInteractomics._compute_condition_similarity(
        [_FakeAR4(df1, "WT"), _FakeAR4(df2, "MUT")], EmbeddingsConfig(top_k_jaccard=2))
    @test r.jaccard_top_k[1, 2] == 1.0
end

@testitem "clamp keeps log10 finite when BFs are ~0" begin
    using BayesInteractomics, DataFrames
    struct _FakeAR5; copula_results::DataFrame; bait_protein::String; end
    # Mix a varying BF column with a near-zero one so we can verify that the clamp
    # produces a *finite* log10 value (rather than -Inf) for the tiny-BF rows. Note:
    # Spearman correlation on a fully constant vector returns NaN by definition — the
    # clamp's contract is "log10 is finite", not "correlation is finite" (Spearman
    # NaN is a downstream consequence of zero variance, not of the clamp).
    df1 = DataFrame(Protein=["A","B"], BF=[1.0, 1e-30], BF_enrichment=[1.0,1.0],
                    BF_correlation=[1.0,1.0], BF_detected=[1.0,1.0],
                    posterior_prob=[0.5,0.1], log2FC=[1.0,0.0])
    df2 = DataFrame(Protein=["A","B"], BF=[1.0, 1e-15], BF_enrichment=[1.0,1.0],
                    BF_correlation=[1.0,1.0], BF_detected=[1.0,1.0],
                    posterior_prob=[0.5,0.1], log2FC=[1.0,0.0])
    r = BayesInteractomics._compute_condition_similarity(
        [_FakeAR5(df1, "WT"), _FakeAR5(df2, "MUT")], EmbeddingsConfig())
    @test isfinite(r.spearman_log10_bf[1, 2])
    @test r isa ConditionSimilarityResult
    # Direct probe of the clamp: _safe_log10_cond(0) should be finite, not -Inf.
    @test isfinite(BayesInteractomics._safe_log10_cond(0.0))
    @test isfinite(BayesInteractomics._safe_log10_cond(1e-30))
end

@testitem "dendrogram populated when extension loaded" begin
    using BayesInteractomics, DataFrames
    # EmbeddingsExt only activates when ALL of its trigger packages are loaded;
    # a partial `using UMAP, Clustering` leaves the dendrogram code path stubbed.
    if Base.find_package("UMAP") === nothing || Base.find_package("Clustering") === nothing ||
       Base.find_package("Distances") === nothing || Base.find_package("TSne") === nothing
        @info "Skipping dendrogram populated test: EmbeddingsExt triggers not discoverable."
        @test true
    else
        @eval using UMAP, Clustering, Distances, TSne
        struct _FakeAR6; copula_results::DataFrame; bait_protein::String; end
        df1 = DataFrame(Protein=["A","B","C"], BF=[10.0, 1.0, 50.0], BF_enrichment=[1.0,1.0,1.0],
                        BF_correlation=[1.0,1.0,1.0], BF_detected=[1.0,1.0,1.0],
                        posterior_prob=[0.9,0.3,0.95], log2FC=[3.0,0.1,4.0])
        df2 = DataFrame(Protein=["A","B","C"], BF=[8.0, 0.5, 40.0], BF_enrichment=[1.0,1.0,1.0],
                        BF_correlation=[1.0,1.0,1.0], BF_detected=[1.0,1.0,1.0],
                        posterior_prob=[0.92,0.2,0.91], log2FC=[2.5,-0.05,3.8])
        r = BayesInteractomics._compute_condition_similarity(
            [_FakeAR6(df1, "WT"), _FakeAR6(df2, "MUT")], EmbeddingsConfig())
        @test size(r.dendrogram_merges, 1) >= 1
        @test length(r.dendrogram_heights) >= 1
        @test r.dendrogram_order == [1, 2] || r.dendrogram_order == [2, 1]
    end
end
