# test/embeddings/test_extension_load.jl
# Similarity & Embeddings: extension load smoke tests.
# Note: TestItemRunner isolates each item, but Julia's package loading is process-global.
# We cannot "unload" UMAP/Clustering once loaded in a session, so we test the warm-load
# state and assume the cold-load state is exercised by any test that does NOT import them.

@testitem "Extension loads: fit_sample_umap has ≥1 method after using UMAP, Clustering" begin
    using BayesInteractomics
    # BayesInteractomicsEmbeddingsExt triggers on ALL of UMAP, Clustering, Distances, TSne
    # (see Project.toml [extensions]); loading a subset does NOT activate it.
    if any(p -> Base.find_package(p) === nothing, ("UMAP", "Clustering", "Distances", "TSne"))
        @info "Skipping extension warm-load test: embeddings trigger packages not discoverable."
        @test true
    else
        @eval using UMAP, Clustering, Distances, TSne
        @test BayesInteractomics._embeddings_extension_loaded() === true
        @test !isempty(methods(BayesInteractomics.fit_sample_umap))
        @test !isempty(methods(BayesInteractomics.fit_protein_umap))
        @test !isempty(methods(BayesInteractomics.fit_condition_clustering))
    end
end

@testitem "EMBEDDINGS_STATUS_VALUES sentinel set is locked" begin
    using BayesInteractomics
    @test :loaded in BayesInteractomics.EMBEDDINGS_STATUS_VALUES
    @test :pca_only in BayesInteractomics.EMBEDDINGS_STATUS_VALUES
    @test :failed in BayesInteractomics.EMBEDDINGS_STATUS_VALUES
    @test :not_run in BayesInteractomics.EMBEDDINGS_STATUS_VALUES
end

@testitem "_require_embeddings_extension throws for :tsne when TSne absent" begin
    using BayesInteractomics
    # If TSne is NOT loaded (i.e. the optional-within-optional path absent), the call throws.
    # Otherwise (TSne IS loaded in this session), the call returns nothing.
    # Branch on the methods-empty probe.
    if isempty(methods(BayesInteractomics.fit_sample_tsne))
        @test_throws ArgumentError BayesInteractomics._require_embeddings_extension(:tsne)
    else
        @test BayesInteractomics._require_embeddings_extension(:tsne) === nothing
    end
end
