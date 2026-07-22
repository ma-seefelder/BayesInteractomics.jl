# test/reports/test_methods_embeddings_pooled_text.jl
# Locks the Methods-tab sentence describing the pooled-mean post-imputation
# matrix used for sample PCA/UMAP inside `_methods_embeddings_block(cfg)`.

@testitem "Methods-tab HTML contains pooled-mean sentence when run_embeddings=true" begin
    using BayesInteractomics

    cfg = EmbeddingsConfig(run_embeddings = true)
    html = BayesInteractomics._methods_embeddings_block(cfg)

    # New contract.
    @test occursin("pooled-mean", html)
    @test occursin("post-imputation", html)
    @test occursin("between-imputation variance", html)
    # Existing copy must not regress.
    @test occursin("PCA on the post-imputation log-intensity matrix", html)
end

@testitem "Methods-tab HTML is empty when run_embeddings=false (no-regression)" begin
    using BayesInteractomics

    cfg = EmbeddingsConfig(run_embeddings = false)
    html = BayesInteractomics._methods_embeddings_block(cfg)

    @test html == ""
end
