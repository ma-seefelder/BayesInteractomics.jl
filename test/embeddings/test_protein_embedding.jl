# test/embeddings/test_protein_embedding.jl
# Similarity & Embeddings: protein-level UMAP feature assembly tests.

@testitem "_assemble_protein_features builds 5-column matrix in canonical order" begin
    using BayesInteractomics, DataFrames
    df = DataFrame(Protein=["A","B"], BF_enrichment=[10.0, 1e-15], BF_correlation=[5.0, 1.0],
                   BF_detected=[2.0, 0.5], posterior_prob=[0.99, 0.1], log2FC=[3.0, -0.1])
    F, kept, ids = BayesInteractomics._assemble_protein_features(df)
    @test size(F) == (2, 5)
    @test ids == ["A", "B"]
    # Column 1 = log10_bf_enrichment, with the 1e-12 floor.
    @test F[2, 1] == log10(1e-12)
    @test F[1, 4] ≈ 0.99
    @test F[1, 5] ≈ 3.0
end

@testitem "numerical safety clamp on log10(BF)" begin
    using BayesInteractomics, DataFrames
    df = DataFrame(Protein=["A"], BF_enrichment=[0.0], BF_correlation=[1e-30],
                   BF_detected=[1.0], posterior_prob=[0.5], log2FC=[0.0])
    F, _, _ = BayesInteractomics._assemble_protein_features(df)
    @test F[1, 1] == log10(1e-12)
    @test F[1, 2] == log10(1e-12)
end

@testitem "z-score per column has mean 0 std 1" begin
    using BayesInteractomics, Statistics
    F = randn(50, 5)
    F_z = BayesInteractomics._zscore_columns(F)
    for j in 1:5
        @test abs(mean(F_z[:, j])) < 1e-10
        @test abs(std(F_z[:, j]) - 1.0) < 1e-10
    end
end

@testitem "zero-std column returned as zeros" begin
    using BayesInteractomics
    F = hcat(randn(20), zeros(20), randn(20), randn(20), randn(20))
    F_z = BayesInteractomics._zscore_columns(F)
    @test all(F_z[:, 2] .== 0.0)
end

@testitem "missing feature column drops row" begin
    using BayesInteractomics, DataFrames
    df = DataFrame(Protein=["A","B","C"],
                   BF_enrichment=Union{Missing,Float64}[10.0, missing, 5.0],
                   BF_correlation=[1.0, 1.0, 1.0],
                   BF_detected=[1.0, 1.0, 1.0],
                   posterior_prob=[0.9, 0.5, 0.6],
                   log2FC=[2.0, 0.0, 1.0])
    F, kept, ids = BayesInteractomics._assemble_protein_features(df)
    @test size(F, 1) == 2
    @test ids == ["A", "C"]
    @test kept == [1, 3]
end

@testitem "class labels fallback: 2-class from posterior_prob when lc_result nothing" begin
    using BayesInteractomics, DataFrames
    df = DataFrame(Protein=["A","B"], BF_enrichment=[10.0, 1.0], BF_correlation=[1.0, 1.0],
                   BF_detected=[1.0, 1.0], posterior_prob=[0.99, 0.3], log2FC=[3.0, 0.1])
    _, kept, _ = BayesInteractomics._assemble_protein_features(df)
    labels = BayesInteractomics._derive_protein_class_labels(df, nothing, kept)
    @test labels == [:H1, :H0]
end

@testitem "differential :classification column overrides lc_result" begin
    using BayesInteractomics, DataFrames
    df = DataFrame(Protein=["A","B"], BF_enrichment=[10.0, 1.0], BF_correlation=[1.0, 1.0],
                   BF_detected=[1.0, 1.0], posterior_prob=[0.99, 0.3], log2FC=[3.0, 0.1],
                   classification=[:GAINED, :UNCHANGED])
    _, kept, _ = BayesInteractomics._assemble_protein_features(df)
    labels = BayesInteractomics._derive_protein_class_labels(df, nothing, kept)
    @test labels == [:GAINED, :UNCHANGED]
end

@testitem "supervised=true emits @warn and falls back to unsupervised" begin
    using BayesInteractomics
    # EmbeddingsExt only activates when ALL of its trigger packages are loaded;
    # the supervised→unsupervised warn path lives in the extension, so the full
    # trigger set (UMAP, Clustering, Distances, TSne) must be present.
    if Base.find_package("UMAP") === nothing || Base.find_package("Clustering") === nothing ||
       Base.find_package("Distances") === nothing || Base.find_package("TSne") === nothing
        @info "Skipping supervised=true warn test: EmbeddingsExt triggers not discoverable."
        @test true
    else
        @eval using UMAP, Clustering, Distances, TSne
        F_z = randn(40, 5)
        classes = Symbol[i % 2 == 0 ? :H1 : :H0 for i in 1:40]
        @test_logs (:warn, r"supervised UMAP requires UMAP.jl >= 0.2") match_mode=:any begin
            coords = BayesInteractomics.fit_protein_umap(F_z, 10, 0.1; supervised=true, class_labels=classes)
            @test size(coords) == (40, 2)
        end
    end
end
