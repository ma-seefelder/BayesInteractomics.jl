# test/reports/test_marginal_fit_row_wise.jl
# Locks the row-wise reconstruction contract for _add_marginal_fit_json!. These
# tests consume the pure helper:
#
#   BayesInteractomics._reconstruct_lc_responsibilities(results_df, lc) -> Matrix{Float64}
#
# Priority chain: LC.responsibilities row (when Protein name in lc.protein_names) →
# [P_H0, P_agnostic, P_H1] from results_df (always populated post-BMA) → [0, 0, 0].

@testitem "row-wise reconstruction copies LC row when protein name found in lc.protein_names" begin
    using BayesInteractomics
    using BayesInteractomics: LatentClassResult
    using DataFrames

    results_df = DataFrame(
        Protein         = ["P1", "P2", "P3"],
        bf_enrichment   = [1.0, 2.0, 3.0],
        bf_correlation  = [1.0, 2.0, 3.0],
        bf_detected     = [1.0, 2.0, 3.0],
        P_H0            = [0.7, 0.5, 0.6],
        P_agnostic      = [0.2, 0.3, 0.3],
        P_H1            = [0.1, 0.2, 0.1],
    )

    # Build a minimal real LatentClassResult covering proteins P1 and P2 only.
    responsibilities = [0.9 0.05 0.05; 0.1 0.1 0.8]
    lc = LatentClassResult(
        [0.5, 0.5],                                                     # bf
        [0.5, 0.5],                                                     # posterior_prob
        Dict{String, NamedTuple{(:mu, :sigma, :precision), Tuple{Float64,Float64,Float64}}}(),  # class_parameters
        [1/3, 1/3, 1/3],                                                # mixing_weights
        Float64[],                                                      # free_energy
        true,                                                           # converged
        1,                                                              # n_iterations
        responsibilities,                                               # responsibilities
        nothing,                                                        # all_restart_traces
        1.0,                                                            # alpha_enrichment_h1
        1.0,                                                            # theta_enrichment_h1
        1.0,                                                            # h1_enrichment_sd
        :gamma,                                                         # h1_enrichment_family
        Dict{Symbol, Float64}(),                                        # h1_bic_scores
        nothing,                                                        # em_diagnostics
        nothing, nothing, nothing,                                      # disc_H0/ag/H1
        nothing, nothing, nothing,                                      # per_step_ll_traces, n_step_halving_reverts, per_dimension_params
    )
    # Override protein_names (the 21-arg ctor defaults it to nothing per types.jl:1016).
    @test lc.responsibilities == responsibilities
    # protein_names is the LAST field; we need it populated. Rebuild via the
    # full-field setter pattern is unavailable for `struct`, so we reach into the
    # responsibility-helper via a small ad-hoc NamedTuple stand-in is NOT acceptable —
    # instead the helper accepts duck-typed LC-like objects in its signature.
    # Simply assert the symbol resolves.

    # Helper signature: (results_df, lc_like) -> Matrix{Float64} of size (nrow(results_df), 3).
    # lc_like must expose `.protein_names::Vector{String}` and `.responsibilities::Matrix{Float64}`.
    lc_like = (protein_names = ["P1", "P2"], responsibilities = responsibilities)
    mat = BayesInteractomics._reconstruct_lc_responsibilities(results_df, lc_like)

    @test size(mat) == (3, 3)
    @test mat[1, :] ≈ [0.9, 0.05, 0.05]
    @test mat[2, :] ≈ [0.1, 0.1, 0.8]
end

@testitem "row-wise fallback to [P_H0, P_agnostic, P_H1] when protein name absent from lc.protein_names" begin
    using BayesInteractomics
    using DataFrames

    results_df = DataFrame(
        Protein         = ["P1", "P2", "P3"],
        bf_enrichment   = [1.0, 2.0, 3.0],
        bf_correlation  = [1.0, 2.0, 3.0],
        bf_detected     = [1.0, 2.0, 3.0],
        P_H0            = [0.7, 0.5, 0.6],
        P_agnostic      = [0.2, 0.3, 0.3],
        P_H1            = [0.1, 0.2, 0.1],
    )

    # LC covers P1 only.
    lc_like = (
        protein_names    = ["P1"],
        responsibilities = reshape([0.9, 0.05, 0.05], 1, 3),
    )

    mat = BayesInteractomics._reconstruct_lc_responsibilities(results_df, lc_like)

    @test size(mat) == (3, 3)
    @test mat[1, :] ≈ [0.9, 0.05, 0.05]
    @test mat[2, :] ≈ [0.5, 0.3, 0.2]   # row-2 falls back to results_df.[P_H0, P_agnostic, P_H1]
    @test mat[3, :] ≈ [0.6, 0.3, 0.1]   # row-3 likewise
end

@testitem "row stays [0, 0, 0] when neither LC nor P_* columns provide data for that row" begin
    using BayesInteractomics
    using DataFrames

    results_df = DataFrame(
        Protein         = ["P1", "P2", "P3"],
        bf_enrichment   = [1.0, 2.0, 3.0],
        bf_correlation  = [1.0, 2.0, 3.0],
        bf_detected     = [1.0, 2.0, 3.0],
        # Make all three P_* columns admit `missing` and put missings on row 3.
        P_H0            = Union{Missing, Float64}[0.7, 0.5, missing],
        P_agnostic      = Union{Missing, Float64}[0.2, 0.3, missing],
        P_H1            = Union{Missing, Float64}[0.1, 0.2, missing],
    )

    lc_like = (
        protein_names    = ["P1"],
        responsibilities = reshape([0.9, 0.05, 0.05], 1, 3),
    )

    mat = BayesInteractomics._reconstruct_lc_responsibilities(results_df, lc_like)

    @test size(mat) == (3, 3)
    @test mat[1, :] ≈ [0.9, 0.05, 0.05]
    @test mat[3, :] ≈ [0.0, 0.0, 0.0]
end
