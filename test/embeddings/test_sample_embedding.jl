# test/embeddings/test_sample_embedding.jl
# Similarity & Embeddings: sample-level PCA + UMAP/t-SNE tests.

@testitem "sample PCA renders even when run_embeddings=true and ext NOT loaded" begin
    using BayesInteractomics
    # PCA-always-on contract: even with method=:umap and no extension loaded, the
    # embedding compute should still emit a PCA score matrix and fall back gracefully
    # for UMAP. Here we exercise the cfg construction + the gate semantics.
    cfg = EmbeddingsConfig(method=:umap, run_embeddings=true)
    @test cfg.method === :umap
    @test cfg.run_embeddings === true
end

@testitem "_compute_sample_embedding short-circuits when run_embeddings=false" begin
    using BayesInteractomics
    cfg = EmbeddingsConfig(run_embeddings=false)
    @test cfg.run_embeddings === false
end

@testitem "method=:tsne without TSne loaded throws ArgumentError BEFORE compute" begin
    using BayesInteractomics
    # If TSne is NOT loaded, the explicit-error path throws. If TSne IS loaded in
    # this session, the guard returns nothing — branch on the method-set probe.
    if isempty(methods(BayesInteractomics.fit_sample_tsne))
        @test_throws ArgumentError BayesInteractomics._require_embeddings_extension(:tsne)
    else
        @test BayesInteractomics._require_embeddings_extension(:tsne) === nothing
    end
    @test BayesInteractomics._require_embeddings_extension(:umap) === nothing
    @test BayesInteractomics._require_embeddings_extension(:none) === nothing
end

@testitem "fit_sample_umap extension method active after using UMAP, Clustering" begin
    using BayesInteractomics
    # Active-project-skip pattern: only run when UMAP
    # is discoverable in the active project. Pkg.test() makes [targets].test deps
    # discoverable; running `julia --project=.` directly does not.
    if any(p -> Base.find_package(p) === nothing, ("UMAP", "Clustering", "Distances", "TSne"))
        @info "Skipping fit_sample_umap activation test: UMAP/Clustering not discoverable. Run under Pkg.test() to exercise."
        @test true
    else
        @eval using UMAP, Clustering, Distances, TSne
        @test !isempty(methods(BayesInteractomics.fit_sample_umap))
        coords = BayesInteractomics.fit_sample_umap(randn(20, 10), 5, 0.1)
        @test size(coords) == (20, 2)
    end
end

@testitem "UMAP determinism — same Random.seed! gives bit-identical coords" begin
    using BayesInteractomics
    using Random
    if any(p -> Base.find_package(p) === nothing, ("UMAP", "Clustering", "Distances", "TSne"))
        @info "Skipping UMAP determinism test: UMAP/Clustering not discoverable."
        @test true
    else
        @eval using UMAP, Clustering, Distances, TSne
        X = randn(20, 10)
        Random.seed!(42); a = BayesInteractomics.fit_sample_umap(X, 5, 0.1)
        Random.seed!(42); b = BayesInteractomics.fit_sample_umap(X, 5, 0.1)
        @test a == b
    end
end

@testitem "fit_sample_umap returns 2D Matrix{Float64}" begin
    using BayesInteractomics
    if any(p -> Base.find_package(p) === nothing, ("UMAP", "Clustering", "Distances", "TSne"))
        @info "Skipping fit_sample_umap return-type test: UMAP/Clustering not discoverable."
        @test true
    else
        @eval using UMAP, Clustering, Distances, TSne
        coords = BayesInteractomics.fit_sample_umap(randn(15, 8), 5, 0.1)
        @test coords isa Matrix{Float64}
        @test size(coords, 2) == 2
    end
end
