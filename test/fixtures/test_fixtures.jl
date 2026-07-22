"""
    test_fixtures.jl

Shared test fixtures and helper functions for BayesInteractomics tests.

This file provides @testsetup modules that generate common test data
used across multiple test files.
"""

using BayesInteractomics
using BayesInteractomics: Protocol, InteractionData, BayesFactorTriplet, PosteriorProbabilityTriplet,
    BayesResult, getNoExperiments, getExperiment, getIDs, getNoProtocols, getControls, getSamples
using Distributions
using Random
using Statistics

@testsetup module StatisticalFixtures
    using BayesInteractomics
    using Distributions
    using Random

    """
        create_enriched_data(enrichment_fc::Float64, bait_index::Int, n_proteins::Int = 5,
                            n_experiments::Int = 3, n_replicates::Int = 3)

    Create synthetic protein abundance data with known enrichment pattern.

    The bait protein (at index bait_index) and a candidate protein (at index 1, unless bait_index=1)
    show enrichment in samples vs controls.
    """
    function create_enriched_data(enrichment_fc::Float64, bait_index::Int;
                                  n_proteins::Int = 5, n_experiments::Int = 3, n_replicates::Int = 3)
        Random.seed!(42)

        # Create control data (baseline)
        control_data = Dict{Int, Matrix{Union{Missing, Float64}}}()
        sample_data = Dict{Int, Matrix{Union{Missing, Float64}}}()

        for exp in 1:n_experiments
            # Controls: all proteins have similar abundance
            control_mat = randn(n_proteins, n_replicates) .+ 8.0  # log2 scale, mean ~8

            # Samples: bait and enriched protein have higher abundance
            sample_mat = copy(control_mat)
            sample_mat[bait_index, :] .+= enrichment_fc  # Bait is enriched

            # Enrich a second protein if not the bait
            enriched_idx = (bait_index == 1) ? 2 : 1
            sample_mat[enriched_idx, :] .+= enrichment_fc / 2

            control_data[exp] = control_mat
            sample_data[exp] = sample_mat
        end

        return control_data, sample_data
    end

    """
        mock_normal_inference_result(means::Vector, sds::Vector)

    Create a mock Normal distribution for testing BF calculations.
    """
    function mock_normal_inference_result(means::Vector, sds::Vector)
        @assert length(means) == length(sds) "means and sds must have same length"
        return [Normal(means[i], sds[i]) for i in eachindex(means)]
    end

    """
        create_synthetic_bayes_factor_triplet(n_proteins::Int)

    Create a synthetic triplet of Bayes factors for testing copula functions.
    """
    function create_synthetic_bayes_factor_triplet(n_proteins::Int = 5)
        Random.seed!(42)

        # Create Bayes factors > 1 (supporting H1) for some proteins
        enrichment_bf = vcat(10.0 .+ rand(2) .* 40, 0.2 .+ rand(3) .* 0.3)
        correlation_bf = vcat(5.0 .+ rand(2) .* 25, 0.1 .+ rand(3) .* 0.4)
        detection_bf = vcat(20.0 .+ rand(2) .* 80, 0.05 .+ rand(3) .* 0.15)

        return BayesFactorTriplet(enrichment_bf, correlation_bf, detection_bf)
    end
end

@testmodule DifferentialFixtures begin
    using BayesInteractomics
    using BayesInteractomics: DifferentialResult, DifferentialConfig, InteractionClass,
                              AnalysisResult, GAINED, REDUCED, UNCHANGED, BOTH_NEGATIVE,
                              EmbeddingsResult, ConditionSimilarityResult
    using DataFrames, Dates, Random

    """
        create_two_condition_result(; analyses_populated::Bool = true,
                                      tmpdir::String = mktempdir()) -> DifferentialResult

    Fixture: build a synthetic 5-protein 2-condition `DifferentialResult`
    for tab-payload tests. When `analyses_populated=true`, attaches two minimal
    `AnalysisResult` objects (with the `simulation_result` + `config` fields
    defaulting to `nothing`) so per-condition tab JSON builders can iterate
    `diff.analyses` without erroring; the per-condition builders then return `"null"`
    or empty payloads — graceful degradation honours the per-tab `hasproperty` /
    `=== nothing` guards in `_build_diff_*_json` helpers.

    The `results_df` includes the `dbf_diagnostic::Symbol` column, with a
    mix of `:ok`, `:saturated`, and `:model_disagreement` values so downstream tests
    can exercise multiple branches of the diagnostic legend.

    Extension: three new kwargs
    `is_calibrated_A`, `is_calibrated_B`, `missing_fraction_high` make the
    mixed-calibration scenario and the bb_mnar_codriven diagnostic
    reachable from tests. Defaults preserve prior behaviour.

    These kwargs are no-ops at the constructor layer until `AnalysisResult` /
    `DifferentialResult` gain the matching fields, at which point `hasfield(...)`
    guards switch over automatically.
    """
    function create_two_condition_result(; analyses_populated::Bool = true,
                                            tmpdir::String = mktempdir(),
                                            is_calibrated_A::Bool = false,
                                            is_calibrated_B::Bool = false,
                                            missing_fraction_high::Bool = false,
                                            embeddings_populated::Bool = true)
        # Fixture extension: minimal EmbeddingsResult + ConditionSimilarityResult builders
        function _mk_embeddings(n_samples::Int, n_proteins::Int)
            pca_scores = randn(n_samples, 2)
            var_exp = [12.4, 8.7]
            labels = (
                condition  = String["sample","sample","sample","sample","control","control","control","control"][1:n_samples],
                replicate  = Int[1, 2, 3, 4, 1, 2, 3, 4][1:n_samples],
                experiment = Int[1, 1, 1, 1, 1, 1, 1, 1][1:n_samples],
                protocol   = Int[1, 1, 1, 1, 1, 1, 1, 1][1:n_samples],
            )
            umap_coords    = randn(n_samples, 2)
            protein_umap   = randn(n_proteins, 2)
            protein_classes = Symbol[i % 3 == 0 ? :H0 : (i % 3 == 1 ? :Agnostic : :H1) for i in 1:n_proteins]
            protein_ids    = String["P$i" for i in 1:n_proteins]
            snapshot = (method=:umap, seed=42, n_neighbors=15, min_dist=0.1, supervised=false, top_k_jaccard=50)
            return BayesInteractomics.EmbeddingsResult(
                pca_scores, var_exp, labels, :complete_case,
                umap_coords, nothing, protein_umap, protein_classes, protein_ids,
                snapshot,
            )
        end

        function _mk_cond_similarity()
            labels = ["WT", "Mutant"]
            sp  = [1.0 0.73; 0.73 1.0]
            p2  = [1.0 0.65; 0.65 1.0]
            pp  = [1.0 0.80; 0.80 1.0]
            jac = [1.0 0.32; 0.32 1.0]
            n_shared = [5 5; 5 5]
            return BayesInteractomics.ConditionSimilarityResult(
                labels, sp, p2, pp, jac, n_shared,
                50,
                zeros(Int, 1, 2), [0.27], [1, 2],
                :average,
            )
        end
        Random.seed!(69)
        dcfg = DifferentialConfig(
            volcano_file        = joinpath(tmpdir, "vol.png"),
            evidence_file       = joinpath(tmpdir, "ev.png"),
            scatter_file        = joinpath(tmpdir, "sc.png"),
            classification_file = joinpath(tmpdir, "cl.png"),
            ma_file             = joinpath(tmpdir, "ma.png"),
            results_file        = joinpath(tmpdir, "diff_results.xlsx"),
        )

        results_df = DataFrame(
            Protein                = ["P1", "P2", "P3", "P4", "P5"],
            bf_A                   = [200.0, 50.0, 10.0, 0.5, 1e19],   # P5 triggers :saturated
            bf_B                   = [1.0,   1.0,  10.0, 1.0, 1.0],
            BF_A                   = [200.0, 50.0, 10.0, 0.5, 1e19],
            BF_B                   = [1.0,   1.0,  10.0, 1.0, 1.0],
            log2fc_A               = [3.0,   1.5,  0.1, -2.0, 0.0],
            log2fc_B               = [0.5,   2.0,  0.0, -0.2, 0.0],
            delta_log2fc           = [2.5,  -0.5,  0.1, -1.8, 0.0],
            dbf                    = [200.0, 50.0, 1.0,  0.5, 1e19],
            log10_dbf              = [2.3,   1.7,  0.0, -0.3, 19.0],
            dbf_enrichment         = [150.0, 40.0, 1.0,  0.5, 1.0],
            dbf_correlation        = [1.3,   1.2,  1.0,  1.0, 1.0],
            dbf_detected           = [1.0,   1.04, 1.0,  1.0, 1e19],
            bf_em_A                = Union{Missing,Float64}[200.0, 50.0, 10.0, 0.5, 1e19],
            bf_em_B                = Union{Missing,Float64}[1.0,   1.0,  10.0, 1.0, 1.0],
            bf_copula_A            = Union{Missing,Float64}[2.0,   50.0, 10.0, 0.5, 1e19],
            bf_copula_B            = Union{Missing,Float64}[1.0,   1.0,  10.0, 100.0, 1.0],
            posterior_A            = [0.99, 0.95, 0.50, 0.10, 0.99],
            posterior_B            = [0.50, 0.50, 0.50, 0.10, 0.50],
            delta_posterior        = [0.49, 0.45, 0.0, 0.0, 0.49],
            BFDR_A                 = [0.001, 0.01, 0.50, 0.90, 0.001],
            BFDR_B                 = [0.50, 0.50, 0.50, 0.90, 0.50],
            PEP_A                  = [0.01, 0.05, 0.50, 0.90, 0.01],
            PEP_B                  = [0.50, 0.50, 0.50, 0.90, 0.50],
            differential_posterior = [0.99, 0.95, 0.50, 0.10, 0.99],
            differential_BFDR      = [0.001, 0.01, 0.50, 0.90, 0.001],
            diff_PEP               = [0.01, 0.05, 0.50, 0.90, 0.01],
            classification         = InteractionClass[GAINED, REDUCED, UNCHANGED, UNCHANGED, GAINED],
            dbf_diagnostic         = Symbol[:ok, :model_disagreement, :ok, :ok, :saturated],
        )

        if !analyses_populated
            return DifferentialResult(
                results_df, "WT", "Mutant", dcfg,
                5, 5, 5, 0, 0,
                now(), 2, 1, 2, 0,
            )
        end

        # embeddings/condition_similarity fixtures (constructed once per call)
        embeddings_A          = embeddings_populated ? _mk_embeddings(8, 5) : nothing
        embeddings_B          = embeddings_populated ? _mk_embeddings(8, 5) : nothing
        condition_similarity  = embeddings_populated ? _mk_cond_similarity() : nothing

        # Build two minimal AnalysisResults (22-arg constructor).
        # simulation_result + config are kept as `nothing` so per-condition Calibration
        # / Methods builders return "null" / skip — that's the correct unit-test signal.
        # Tests that need richer per-condition payloads stub these locally.
        empty_df  = DataFrame()
        copula_df = DataFrame(
            Protein        = results_df.Protein,
            BF             = results_df.bf_A,
            log2FC         = results_df.log2fc_A,
            mean_log2FC    = results_df.log2fc_A,        # omnibus consumer name lock
            sd_log2FC      = fill(0.5, nrow(results_df)),# per-protein σ for Laplace omnibus
            BF_enrichment  = [5.0, 4.0, 3.0, 2.0, 1.0],
            BF_correlation = [2.0, 2.0, 1.0, 1.0, 1.0],
            BF_detected    = [1.0, 1.0, 1.0, 1.0, 1.0],
            posterior_prob = [0.99, 0.95, 0.5, 0.1, 0.99],
            Component      = [1, 2, 3, 3, 1],
        )
        # Inject high-missingness into copula_df when requested.
        # Static condition labels ("WT", "Mutant") avoid the T-70-04-02 XSS path —
        # nothing user-controlled flows into the fixture's DataFrame.
        if missing_fraction_high
            mf = fill(0.1, nrow(copula_df))
            mf[4:end] .= 0.7
            if hasproperty(copula_df, :missing_fraction)
                copula_df.missing_fraction = mf
            else
                insertcols!(copula_df, :missing_fraction => mf)
            end
        end

        # Canonical 23-arg AnalysisResult form (`is_calibrated::Bool` is the
        # final positional field). The earlier `hasfield` guard is no longer
        # needed.
        function _mk_ar(label::String, df::DataFrame, is_calibrated::Bool, embeddings)
            AnalysisResult(
                df, empty_df, nothing, nothing, nothing, nothing, nothing, :bma,
                nothing, nothing, UInt64(0), UInt64(0), now(), "test",
                label, 1, nothing, nothing, nothing, :loaded,
                nothing, nothing,
                is_calibrated,           # canonical 23-arg form
                embeddings,              # 24th positional arg
            )
        end
        ar_A = _mk_ar("BAIT_WT",  copula_df, is_calibrated_A, embeddings_A)
        ar_B = _mk_ar("BAIT_MUT", copula_df, is_calibrated_B, embeddings_B)

        # Canonical 17-arg DifferentialResult form (`is_calibrated_A::Bool` /
        # `is_calibrated_B::Bool` follow `analyses`).
        return DifferentialResult(
            results_df, "WT", "Mutant", dcfg,
            5, 5, 5, 0, 0,
            now(), 2, 1, 2, 0,
            [ar_A, ar_B],                       # analyses
            is_calibrated_A, is_calibrated_B,
            condition_similarity,               # 18th positional arg
        )
    end

    """
        create_three_condition_result(; n_proteins::Int = 6, tmpdir::String = mktempdir())
            -> NamedTuple{(:ar_wt, :ar_mut1, :ar_mut2)}

    Synthetic 3-condition AR triple with controlled ground-truth
    GAINED/REDUCED/UNCHANGED/BOTH_NEGATIVE patterns across the three pairwise
    contrasts (wt-vs-mut1, wt-vs-mut2, mut1-vs-mut2). Returns the AR triple as a
    NamedTuple ready for:

        differential_analysis(; conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1, mut2 = fx.ar_mut2))

    Ground-truth layout (n_proteins = 6, one protein per pattern):

    | Protein | wt BF | mut1 BF | mut2 BF | wt-vs-mut1 | wt-vs-mut2 | mut1-vs-mut2 |
    |---------|-------|---------|---------|------------|------------|--------------|
    | P1      | 200   | 1.0     | 1.0     | GAINED     | GAINED     | UNCHANGED    |
    | P2      | 1.0   | 200     | 1.0     | REDUCED    | UNCHANGED  | GAINED       |
    | P3      | 1.0   | 1.0     | 200     | UNCHANGED  | REDUCED    | REDUCED      |
    | P4      | 200   | 200     | 200     | UNCHANGED  | UNCHANGED  | UNCHANGED    |
    | P5      | 0.1   | 0.1     | 0.1     | BOTH_NEG   | BOTH_NEG   | BOTH_NEG     |
    | P6      | 200   | 200     | 1.0     | UNCHANGED  | GAINED     | GAINED       |

    All three ARs use the SAME bait label `"BAIT"` so the bait-mismatch @warn
    does NOT fire under default conditions. Tests exercising the bait-mismatch
    path should flip ONE label after fixture construction.

    This fixture ships at n_proteins=6 minimum for fast quick-filter runtime; the
    `n_proteins` kwarg may be clamped lower (1..6) for sub-pattern testitems.
    Downstream plans MAY add wider fixtures (e.g. n_proteins=20+) for power
    tests — this fixture is the canonical 6-row ground-truth source.
    """
    function create_three_condition_result(; n_proteins::Int = 6, tmpdir::String = mktempdir())
        Random.seed!(72)
        n = clamp(n_proteins, 1, 6)
        bf_wt        = [200.0, 1.0,   1.0,   200.0, 0.1,  200.0][1:n]
        bf_mut1      = [1.0,   200.0, 1.0,   200.0, 0.1,  200.0][1:n]
        bf_mut2      = [1.0,   1.0,   200.0, 200.0, 0.1,  1.0  ][1:n]
        log2fc_wt    = [3.0,   0.5,   0.5,   3.0,  -1.0,  3.0  ][1:n]
        log2fc_mut1  = [0.5,   3.0,   0.5,   3.0,  -1.0,  3.0  ][1:n]
        log2fc_mut2  = [0.5,   0.5,   3.0,   3.0,  -1.0,  0.5  ][1:n]

        function _mk_copula_df(bf::Vector{Float64}, log2fc::Vector{Float64};
                                sd_log2fc::Union{Vector{Float64}, Nothing} = nothing)
            # Column names match the canonical `analyse()` pipeline output so the
            # 2-group `differential_analysis(::AR, ::AR)` path (invoked from
            # `_runpair!`) finds the lowercase `bf_enrichment_A` / `mean_log2FC_A`
            # / etc. columns it expects after `_rename_columns`.
            sd = sd_log2fc === nothing ? fill(0.5, n) : sd_log2fc
            return DataFrame(
                Protein         = ["P$i" for i in 1:n],
                BF              = bf,
                bf_enrichment   = bf,
                bf_correlation  = ones(n),
                bf_detected     = ones(n),
                mean_log2FC     = log2fc,
                sd_log2FC       = sd,                     # per-protein σ for Laplace omnibus
                posterior_prob  = bf ./ (1 .+ bf),
                Component       = fill(1, n),
            )
        end

        # 24-arg canonical AnalysisResult ctor.
        # The bait label "BAIT" is shared so the bait-mismatch @warn does NOT fire.
        function _mk_ar(label::String, copula_df::DataFrame)
            return AnalysisResult(
                copula_df, DataFrame(),                  # copula_results, df_hierarchical
                nothing, nothing, nothing,                # em, joint_H0, joint_H1
                nothing, nothing, :bma,                   # latent_class_result, bma_result, combination_method
                nothing, nothing,                         # sensitivity, diagnostics
                UInt64(0), UInt64(0),                     # hash_a, hash_b
                now(), "test",                            # timestamp, version_string
                label, 1,                                 # bait_protein, bait_index
                nothing, nothing, nothing, :loaded,       # sensitivity_result, validation_result, input_qc, metalearner_status
                nothing, nothing,                         # simulation_result, config
                false,                                    # is_calibrated
                nothing,                                  # embeddings
            )
        end

        ar_wt   = _mk_ar("BAIT", _mk_copula_df(bf_wt,   log2fc_wt))
        ar_mut1 = _mk_ar("BAIT", _mk_copula_df(bf_mut1, log2fc_mut1))
        ar_mut2 = _mk_ar("BAIT", _mk_copula_df(bf_mut2, log2fc_mut2))
        return (ar_wt = ar_wt, ar_mut1 = ar_mut1, ar_mut2 = ar_mut2)
    end

    """
        create_four_condition_result(; n_proteins::Int = 50, tmpdir::String = mktempdir())
            -> NamedTuple{(:ar_wt, :ar_mut1, :ar_mut2, :ar_mut3)}

    Synthetic 4-condition AR fixture for k-group Laplace omnibus +
    small-multiples report tests. Returns 4 AnalysisResult instances ready for:

        differential_analysis(; conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                                              mut2 = fx.ar_mut2, mut3 = fx.ar_mut3))

    Ground-truth blocks (locked at `n = max(n_proteins, 50)`; randn() draws require
    the full 50-row block layout):

      1..10  — all-enriched (high BF in all 4 conditions; +3.0 log2FC mean shift)
      11..20 — single-condition shift (high BF only in wt; +3.0 log2FC only in wt)
      21..25 — wt+mut1 pair-specific (high BF in wt+mut1 only)
      26..30 — mut2+mut3 pair-specific (high BF in mut2+mut3 only)
      31..n  — background (low BF ~0.5, near-zero log2FC ~0.1·N(0,1))

    σ layout (per-condition, identical shape across conditions):
      sd[1]  = 1e-6  → zero-σ clamp driver
      sd[11] = 0.05  → BF→∞ driver (paired with +3.0 wt-only shift; ≥4σ)
      others ∈ {0.3, 0.5, 0.7} (uniform random pool)

    All four ARs share the bait label `"BAIT"` so the bait-mismatch @warn does
    NOT fire. Deterministic seed `Random.seed!(20260514)`.

    This fixture ships at the 50-row default for full ground-truth coverage; the
    `n_proteins` kwarg is clamped to `max(n_proteins, 50)` because the block
    structure (10 + 10 + 10 + 20 = 50) is the lock.
    """
    function create_four_condition_result(; n_proteins::Int = 50, tmpdir::String = mktempdir())
        Random.seed!(20260514)
        n = max(n_proteins, 50)  # block structure (10+10+10+20) requires ≥50

        function _block_bf(positive_idx::Vector{Int})
            bf = fill(0.5, n)
            for i in positive_idx
                bf[i] = max(100.0 + 50.0 * randn(), 5.0)  # floor at 5.0
            end
            return bf
        end

        function _block_log2fc(positive_idx::Vector{Int}, magnitude::Float64)
            lfc = 0.1 .* randn(n)
            for i in positive_idx
                lfc[i] = magnitude + 0.2 * randn()
            end
            return lfc
        end

        function _block_sd()
            sd = rand([0.3, 0.5, 0.7], n)
            sd[1]  = 1e-6   # zero-σ clamp driver
            sd[11] = 0.05   # BF→∞ driver (paired with shifted mean below)
            return sd
        end

        # Ground-truth index blocks.
        all_enriched   = collect(1:10)
        wt_only        = collect(11:20)
        wt_mut1_pair   = collect(21:25)
        mut2_mut3_pair = collect(26:30)

        bf_wt   = _block_bf(vcat(all_enriched, wt_only, wt_mut1_pair))
        bf_mut1 = _block_bf(vcat(all_enriched, wt_mut1_pair))
        bf_mut2 = _block_bf(vcat(all_enriched, mut2_mut3_pair))
        bf_mut3 = _block_bf(vcat(all_enriched, mut2_mut3_pair))

        log2fc_wt   = _block_log2fc(vcat(all_enriched, wt_only, wt_mut1_pair), 3.0)
        log2fc_mut1 = _block_log2fc(vcat(all_enriched, wt_mut1_pair), 3.0)
        log2fc_mut2 = _block_log2fc(vcat(all_enriched, mut2_mut3_pair), 3.0)
        log2fc_mut3 = _block_log2fc(vcat(all_enriched, mut2_mut3_pair), 3.0)

        sd_wt   = _block_sd()
        sd_mut1 = _block_sd()
        sd_mut2 = _block_sd()
        sd_mut3 = _block_sd()

        # Local _mk_copula_df closure (mirrors create_three_condition_result helper
        # extended with sd_log2fc kwarg). Kept inside this factory to avoid touching
        # the sibling fixture's hoist surface.
        function _mk_copula_df(bf::Vector{Float64}, log2fc::Vector{Float64};
                                sd_log2fc::Union{Vector{Float64}, Nothing} = nothing)
            sd = sd_log2fc === nothing ? fill(0.5, n) : sd_log2fc
            return DataFrame(
                Protein         = ["P$i" for i in 1:n],
                BF              = bf,
                bf_enrichment   = bf,
                bf_correlation  = ones(n),
                bf_detected     = ones(n),
                mean_log2FC     = log2fc,
                sd_log2FC       = sd,                     # per-protein σ for Laplace omnibus
                posterior_prob  = bf ./ (1 .+ bf),
                Component       = fill(1, n),
            )
        end

        # 24-arg canonical AnalysisResult ctor (mirrors create_three_condition_result).
        function _mk_ar(label::String, copula_df::DataFrame)
            return AnalysisResult(
                copula_df, DataFrame(),                  # copula_results, df_hierarchical
                nothing, nothing, nothing,                # em, joint_H0, joint_H1
                nothing, nothing, :bma,                   # latent_class_result, bma_result, combination_method
                nothing, nothing,                         # sensitivity, diagnostics
                UInt64(0), UInt64(0),                     # hash_a, hash_b
                now(), "test",                            # timestamp, version_string
                label, 1,                                 # bait_protein, bait_index
                nothing, nothing, nothing, :loaded,       # sensitivity_result, validation_result, input_qc, metalearner_status
                nothing, nothing,                         # simulation_result, config
                false,                                    # is_calibrated
                nothing,                                  # embeddings
            )
        end

        ar_wt   = _mk_ar("BAIT", _mk_copula_df(bf_wt,   log2fc_wt;   sd_log2fc = sd_wt))
        ar_mut1 = _mk_ar("BAIT", _mk_copula_df(bf_mut1, log2fc_mut1; sd_log2fc = sd_mut1))
        ar_mut2 = _mk_ar("BAIT", _mk_copula_df(bf_mut2, log2fc_mut2; sd_log2fc = sd_mut2))
        ar_mut3 = _mk_ar("BAIT", _mk_copula_df(bf_mut3, log2fc_mut3; sd_log2fc = sd_mut3))
        return (ar_wt = ar_wt, ar_mut1 = ar_mut1, ar_mut2 = ar_mut2, ar_mut3 = ar_mut3)
    end
end  # @testmodule DifferentialFixtures

# NOTE: the `DataStructureFixtures` @testsetup was moved to test/fixtures/data_structure_fixtures.jl.
# When it lived here (immediately after the large DifferentialFixtures @testmodule) TestItemRunner did
# not register it, so test_h0_sampling.jl errored with "Test setup DataStructureFixtures is not defined".
