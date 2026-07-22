# test/integration/test_imputed_embeddings_smoke.jl
# Integration smoke check: on an MNAR-heavy fixture the raw cascade
# would produce <20 surviving proteins. With pooled-imputed data the embedding path
# must produce non-empty sample_pca_scores.

@testitem "integration: _compute_embeddings on pooled-imputed InteractionData produces non-empty sample_pca_scores" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData

    # Build a synthetic MNAR-heavy raw fixture: 30 proteins, 2 protocols, 4 samples
    # per protocol, ~70% missingness arranged so NO protein has full coverage in
    # BOTH protocols (the cascade filter's :complete / 80% / 50% tiers would all
    # fall below the 20-protein minimum and trip :skipped).
    n_proteins  = 30
    n_reps      = 4
    protein_ids   = ["P$i" for i in 1:n_proteins]
    protein_names = protein_ids

    function make_raw_protocol(missing_phase::Int)
        # missing_phase ∈ {1, 2}: protein i has its sample-matrix mostly missing in
        # protocol `missing_phase` (odd / even rows). Ensures disjoint detected sets.
        sample_mat = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_reps)
        for i in 1:n_proteins, j in 1:n_reps
            sample_mat[i, j] = ((i + missing_phase) % 2 == 0) ? missing : float(10 + i + j * 0.1)
        end
        control_mat = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_reps)
        for i in 1:n_proteins, j in 1:n_reps
            control_mat[i, j] = float(5 + i + j * 0.1)
        end
        return (sample_mat, control_mat)
    end

    function build_imputed_id(seed_offset::Float64)
        # M imputations: dense (no missings), constant offset added so the
        # element-wise mean across M is the base + mean(offsets).
        s1 = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_reps)
        s2 = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_reps)
        c1 = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_reps)
        c2 = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_reps)
        for i in 1:n_proteins, j in 1:n_reps
            s1[i, j] = float(10 + i + j * 0.1) + seed_offset
            s2[i, j] = float(12 + i + j * 0.1) + seed_offset
            c1[i, j] = float(5 + i + j * 0.1) + seed_offset
            c2[i, j] = float(6 + i + j * 0.1) + seed_offset
        end
        sample_p1  = Protocol(1, protein_ids, Dict(1 => s1))
        sample_p2  = Protocol(1, protein_ids, Dict(1 => s2))
        control_p1 = Protocol(1, protein_ids, Dict(1 => c1))
        control_p2 = Protocol(1, protein_ids, Dict(1 => c2))
        return InteractionData(
            protein_ids, protein_names,
            Dict(1 => sample_p1, 2 => sample_p2),
            Dict(1 => control_p1, 2 => control_p2),
            2, Dict(1 => 1, 2 => 1),
            5, 3,
            [2, 4], [3, 5], [1, 1],
            trues(n_proteins),
        )
    end

    imputed = [build_imputed_id(0.0), build_imputed_id(1.0), build_imputed_id(2.0)]

    # The pooled InteractionData should land here.
    pooled = BayesInteractomics._pool_imputed_matrix(imputed)

    # Sample embedding directly: no AnalysisResult plumbing needed for the smoke check.
    cfg = EmbeddingsConfig(run_embeddings = true, method = :none)
    result = BayesInteractomics._compute_sample_embedding(pooled, cfg)

    @test size(result.sample_pca_scores, 1) > 0
    @test size(result.sample_pca_scores, 2) > 0
end
