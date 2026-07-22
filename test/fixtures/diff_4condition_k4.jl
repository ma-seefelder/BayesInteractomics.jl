"""
    diff_4condition_k4.jl

k=4 differential analysis fixture (`DifferentialFixturesK4`).

Provides a deterministic 4-condition (k=4) `(wt, mut1, mut2, mut3)` fixture
suitable for `differential_analysis(; conditions=..., contrasts=:all_pairs)`
which yields 6 pairs. Used across the differential test plans via
`setup=[DifferentialFixturesK4]`.

The actual ground-truth block construction (and AR ctor cascade) is shared
with the `create_four_condition_result` helper already hosted
inside `DifferentialFixtures` (see `test/fixtures/test_fixtures.jl`); this
file re-exports that helper under the locked `DifferentialFixturesK4` name
so downstream testitems can opt into the same deterministic
seed (`Random.seed!(20260514)`) and 50-row block layout without depending
on the broader `DifferentialFixtures` API surface.

Seed: `Random.seed!(20260514)` (inherited from `create_four_condition_result`).
n_proteins: 50 (block layout 10+10+10+20 is locked).

# Locks honoured
- BMA terminology: "Copula" + "3c-EM" verbatim.
- FDR terminology: BFDR / PEP / local_fdr.
- CACHE_VERSION = 24; not bumped by this fixture.
- `DifferentialResult` 20-arg ctor (byte-equality).
"""

@testmodule DifferentialFixturesK4 begin
    using BayesInteractomics
    using BayesInteractomics: DifferentialResult, DifferentialConfig, AnalysisResult,
                              InteractionClass, GAINED, REDUCED, UNCHANGED, BOTH_NEGATIVE
    using DataFrames, Dates, Random

    """
        create_four_condition_result(; n_proteins::Int = 50, tmpdir::String = mktempdir())
            -> NamedTuple{(:ar_wt, :ar_mut1, :ar_mut2, :ar_mut3)}

    Wrapper around the base four-condition fixture. Returns 4
    `AnalysisResult` instances with the ground-truth block layout

      1..10  — all-enriched
      11..20 — wt-only
      21..25 — wt+mut1 pair-specific
      26..30 — mut2+mut3 pair-specific
      31..n  — background

    suitable for:

        diff = differential_analysis(;
            conditions = (wt = fx.ar_wt, mut1 = fx.ar_mut1,
                          mut2 = fx.ar_mut2, mut3 = fx.ar_mut3),
            contrasts  = :all_pairs,   # 6 pairs
        )

    Seed `Random.seed!(20260514)`. Bait label is shared as `"BAIT"` across
    all 4 ARs so the bait-mismatch `@warn` does NOT fire.
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
            sd[11] = 0.05   # BF→∞ driver (paired with shifted mean)
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

        # 24-arg canonical AnalysisResult ctor (mirrors create_three_condition_result
        # / create_four_condition_result in DifferentialFixtures).
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
end  # @testmodule DifferentialFixturesK4
