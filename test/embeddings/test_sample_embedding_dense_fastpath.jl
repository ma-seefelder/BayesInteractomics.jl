# test/embeddings/test_sample_embedding_dense_fastpath.jl
# RED scaffold: locks the dense fast-path contract in _compute_sample_embedding.
# These tests are expected to fail until the dense-input bypass branch is added.
# The recommended sentinel for the dense path is :complete
# (downstream JS treats it as "no caveats"). If a different sentinel is chosen, like
# :complete_post_imputation, update this test in lock-step.

@testitem "_compute_sample_embedding skips _filter_and_impute on dense Union{Missing,Float64} input with no actual missings" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData

    # Build an InteractionData with eltype Union{Missing, Float64} but NO actual missings —
    # mimics a pooled-imputed result. Use enough proteins (>=20) so a hypothetical cascade
    # filter wouldn't trip the :skipped branch on its own.
    n_proteins  = 25
    n_replicates = 3
    protein_ids   = ["P$i" for i in 1:n_proteins]
    protein_names = protein_ids

    # Random-but-finite values (no actual missings present).
    sample_mat = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_replicates)
    for i in 1:n_proteins, j in 1:n_replicates
        sample_mat[i, j] = float(10 + i + j * 0.1)
    end
    control_mat = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_replicates)
    for i in 1:n_proteins, j in 1:n_replicates
        control_mat[i, j] = float(5 + i + j * 0.1)
    end

    sample_proto  = Protocol(1, protein_ids, Dict(1 => sample_mat))
    control_proto = Protocol(1, protein_ids, Dict(1 => control_mat))
    data = InteractionData(
        protein_ids, protein_names,
        Dict(1 => sample_proto), Dict(1 => control_proto),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1],
        trues(n_proteins),
    )

    cfg = EmbeddingsConfig(run_embeddings = true, method = :none)
    result = BayesInteractomics._compute_sample_embedding(data, cfg)

    # Dense input must produce non-empty PCA scores via the bypass path.
    @test size(result.sample_pca_scores, 1) > 0
    @test size(result.sample_pca_scores, 2) > 0
    # Sentinel: :complete per the dense-path recommendation.
    # If a different sentinel is chosen (e.g. :complete_post_imputation), update here.
    @test result.sample_filter_level === :complete
end

@testitem "_compute_sample_embedding still hits cascade filter when actual missings present" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData

    # Same fixture shape but introduce one `missing` cell. The dense fast-path
    # must NOT fire; the cascade-filter path (:complete / :non_missing_80pct /
    # :non_missing_50pct / :skipped) is the expected level.
    n_proteins  = 25
    n_replicates = 3
    protein_ids   = ["P$i" for i in 1:n_proteins]
    protein_names = protein_ids

    sample_mat = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_replicates)
    for i in 1:n_proteins, j in 1:n_replicates
        sample_mat[i, j] = float(10 + i + j * 0.1)
    end
    # Single actual missing.
    sample_mat[1, 1] = missing

    control_mat = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_replicates)
    for i in 1:n_proteins, j in 1:n_replicates
        control_mat[i, j] = float(5 + i + j * 0.1)
    end

    sample_proto  = Protocol(1, protein_ids, Dict(1 => sample_mat))
    control_proto = Protocol(1, protein_ids, Dict(1 => control_mat))
    data = InteractionData(
        protein_ids, protein_names,
        Dict(1 => sample_proto), Dict(1 => control_proto),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1],
        trues(n_proteins),
    )

    cfg = EmbeddingsConfig(run_embeddings = true, method = :none)
    result = BayesInteractomics._compute_sample_embedding(data, cfg)

    # When actual missings are present, the cascade path must have fired —
    # one of the cascade-derived sentinel values returned by filter_complete_case
    # (src/qc/pca_separation.jl: :complete_case / :threshold_80 / :threshold_50) or :skipped.
    # NOT :complete (that sentinel is reserved for the dense fast-path).
    @test result.sample_filter_level in (:complete_case, :threshold_80, :threshold_50, :skipped)
    @test result.sample_filter_level !== :complete
end
