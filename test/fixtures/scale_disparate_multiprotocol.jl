"""
    scale_disparate_multiprotocol.jl

Deterministic 2-protocol synthetic
`InteractionData` fixture for the normalisation de-saturation proof.

# What it provides

`load_fixture(; matched::Bool=false)` returns a NamedTuple
`(raw, imputed, column_imputation_sigma_sq, refID)` where:

- `matched=false` (default — the SCALE-DISPARATE arm): protocol 2 sits at a
  large per-protein cross-protocol baseline OFFSET (+`OFFSET_LOG2` ≈ 8 log2)
  versus protocol 1. This is the saturation driver — the SAME protein
  is measured at a different absolute baseline in the two pulldowns
  (GST-tag vs Strep-tag analogue). The hierarchical multi-protocol regression
  pools the two protocols through the shared slope `μ_α`; an un-removed
  per-protein cross-protocol offset becomes a spurious slope (|μ_α| ≫ threshold,
  tight σ → bf_correlation pinned at the 1e6 ceiling). Column-scaling
  (`median_of_ratios`) does NOT touch this offset; per-protein ROW-CENTERING
  removes it.

- `matched=true` (the MATCHED-LEVEL arm): both protocols share the baseline
  (real HAP40 GST/Strep matched-bait-level analogue, ~30 vs ~30.1 log2). No
  cross-protocol offset → `:auto` must NOT over-correct → no saturation under
  ANY normalisation.

# Sourcing decision (Option c — synthetic)

The locked-narrow real-HAP40 arm quartiles (A ~38 % / B 0 % /
C ~38 % / F 0 %) require the user's real HAP40 GST+Strep XLSX, which lives at a
user-only path and is NOT committed to this repo (same constraint as the
`hap40_strep_slice_mnar.jl` fixture — Option a). The chosen sourcing
decision is Option (c): a fully self-contained synthetic
scale-disparate multi-protocol fixture that exhibits the SAME
saturation→de-saturation behaviour as the real data — saturated under
`:none` / `:median_of_ratios`, ~0 % under `:row_center` / `:both`. The
locked-narrow real-HAP40 confirmation is a final user-run checkpoint.

# Deterministic seed

`Random.seed!(SEED)` at the top of `load_fixture`. Re-running the loader in the
same Julia session produces a byte-identical `InteractionData` (asserted in
`test_normalisation.jl`).
"""


@testmodule ScaleDisparateMultiprotocol begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, getPositions, getIDs,
        getNoProtocols
    using Random
    using Statistics

    export load_fixture

    const SEED         = 2026_05_25
    const OFFSET_LOG2  = 10.0  # protocol-2 cross-protocol baseline offset MEAN (the saturation driver)
    const OFFSET_SD    = 6.0   # per-protein spread of that offset — large so a single per-COLUMN
                               # median-of-ratios size factor canNOT remove it (only per-protein
                               # row-centering can), reproducing the orthogonal-axes result.

    """
        load_fixture(; matched::Bool=false) -> NamedTuple

    Returns `(raw, imputed, column_imputation_sigma_sq, refID)`.

    - `raw::InteractionData` — 2-protocol fixture with ~30 % MNAR-like missings
       injected. Protocol 2 carries the +`OFFSET_LOG2` per-protein cross-protocol
       baseline offset when `matched=false`; no offset when `matched=true`.
    - `imputed::InteractionData` — same shape with missings filled by a
       tilted-Gaussian MNAR draw (σ_imp = 0.5 log2).
    - `column_imputation_sigma_sq::Dict{Tuple{Int,Int,Int}, Float64}` — per-cell
       σ²_imp lookup keyed by `(protocol, experiment, cat-axis index)`; every cell
       maps to `σ_imp² = 0.25`.
    - `refID::Int = 1` — the bait/reference protein row.

    Deterministic: same call → identical output in the same Julia session.
    """
    function load_fixture(; matched::Bool=false)
        Random.seed!(SEED)

        n_proteins  = 60
        n_protocols = 2
        n_exp_per_protocol      = 1
        n_samples_per_protocol  = [3, 3]
        n_controls_per_protocol = [3, 3]

        protein_ids   = ["P$(lpad(i, 3, '0'))" for i in 1:n_proteins]
        protein_names = ["Protein_$(lpad(i, 3, '0'))" for i in 1:n_proteins]

        n_interactors = 12   # 20 % interactor prevalence
        bait_idx  = 1
        bait_mean = 12.0
        σ_imp     = 0.5

        # Per-protein baseline drawn ONCE (shared row position across protocols).
        protein_baseline = Float64[8.0 + 0.5 * randn() for _ in 1:n_proteins]

        # The cross-protocol offset: applied to the NON-bait proteins in
        # protocol 2 ONLY (both sample AND control cells of each protein — so it
        # cancels in sample−control → log2FC invariant, the HBM-safe fact),
        # while the bait REFERENCE row stays UNSHIFTED. This breaks each protein's
        # protein-vs-bait relationship across the two protocols → a spurious slope in
        # the hierarchical regression that pools the protocols through the shared μ_α
        # → bf_correlation pins at the 1e6 ceiling. Per-protein varying (mean +
        # Gaussian) so it is a genuine per-protein cross-protocol gap (not a single
        # constant the per-protocol OLS could absorb). All-zero when `matched=true`.
        protein_offset_p2 = Float64[
            matched ? 0.0 : (OFFSET_LOG2 + OFFSET_SD * randn()) for _ in 1:n_proteins
        ]
        # The offset for (protein, protocol): bait reference is never shifted.
        _offset(prot::Int, p::Int) = (p == 2 && prot != bait_idx) ? protein_offset_p2[prot] : 0.0

        function _build_sample(p::Int, n_s::Int)
            mat = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_s)
            for prot in 1:n_proteins
                off = _offset(prot, p)
                if prot == bait_idx
                    for s in 1:n_s
                        mat[prot, s] = bait_mean + 0.3 * randn()
                    end
                elseif prot <= 1 + n_interactors
                    # Genuine interactor — correlates with bait (enrichment ≈ 3.2 log2).
                    enrichment = 0.8 * (bait_mean - 8.0)
                    for s in 1:n_s
                        mat[prot, s] = protein_baseline[prot] + off + enrichment + 0.4 * randn()
                    end
                else
                    # Noise protein — flat enrichment.
                    for s in 1:n_s
                        mat[prot, s] = protein_baseline[prot] + off + 0.4 * randn()
                    end
                end
            end
            return mat
        end

        function _build_control(p::Int, n_c::Int)
            mat = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_c)
            for prot in 1:n_proteins
                off = _offset(prot, p)
                if prot == bait_idx
                    for c in 1:n_c
                        mat[prot, c] = 6.0 + 0.4 * randn()
                    end
                else
                    for c in 1:n_c
                        mat[prot, c] = protein_baseline[prot] + off + 0.4 * randn()
                    end
                end
            end
            return mat
        end

        function _inject_and_impute!(raw_m, imp_m)
            # MNAR-like: low intensity → higher missingness probability. Protocol-2
            # offset cells sit higher and are rarely missing — realistic and irrelevant
            # to the cross-protocol-offset saturation contract.
            ρ = 5.0; ζ = -0.8
            for I in eachindex(raw_m)
                y = raw_m[I]
                p_miss = 1.0 / (1.0 + exp(-(ρ + ζ * y)))
                if rand() < p_miss
                    raw_m[I] = missing
                    imp_m[I] = y + σ_imp * randn() - 0.5
                end
            end
            return nothing
        end

        samples_raw_dict  = Dict{Int, Protocol{Float64, Int}}()
        samples_imp_dict  = Dict{Int, Protocol{Float64, Int}}()
        controls_raw_dict = Dict{Int, Protocol{Float64, Int}}()
        controls_imp_dict = Dict{Int, Protocol{Float64, Int}}()
        no_exp_dict       = Dict{Int, Int}()

        for p in 1:n_protocols
            n_s = n_samples_per_protocol[p]
            n_c = n_controls_per_protocol[p]

            sample_raw = Dict{Int, Matrix{Union{Missing, Float64}}}()
            sample_imp = Dict{Int, Matrix{Union{Missing, Float64}}}()
            ctrl_raw   = Dict{Int, Matrix{Union{Missing, Float64}}}()
            ctrl_imp   = Dict{Int, Matrix{Union{Missing, Float64}}}()

            for e in 1:n_exp_per_protocol
                s_full = _build_sample(p, n_s)
                c_full = _build_control(p, n_c)
                s_imp  = copy(s_full)
                c_imp  = copy(c_full)
                _inject_and_impute!(s_full, s_imp)
                _inject_and_impute!(c_full, c_imp)
                sample_raw[e] = s_full
                sample_imp[e] = s_imp
                ctrl_raw[e]   = c_full
                ctrl_imp[e]   = c_imp
            end

            samples_raw_dict[p]  = Protocol(n_exp_per_protocol, protein_ids, sample_raw)
            samples_imp_dict[p]  = Protocol(n_exp_per_protocol, protein_ids, sample_imp)
            controls_raw_dict[p] = Protocol(n_exp_per_protocol, protein_ids, ctrl_raw)
            controls_imp_dict[p] = Protocol(n_exp_per_protocol, protein_ids, ctrl_imp)
            no_exp_dict[p] = n_exp_per_protocol
        end

        no_hbm = 1 + n_protocols + n_protocols * n_exp_per_protocol
        no_reg = 1 + n_protocols
        pp, ep, mp = getPositions(no_exp_dict, no_hbm)

        raw_data = InteractionData(
            protein_ids, protein_names,
            samples_raw_dict, controls_raw_dict,
            n_protocols, no_exp_dict,
            no_hbm, no_reg, ep, pp, mp,
            trues(n_proteins),
        )
        imputed_data = InteractionData(
            protein_ids, protein_names,
            samples_imp_dict, controls_imp_dict,
            n_protocols, no_exp_dict,
            no_hbm, no_reg, ep, pp, mp,
            trues(n_proteins),
        )

        column_imputation_sigma_sq = Dict{Tuple{Int, Int, Int}, Float64}()
        sigma_imp_sq = σ_imp^2
        max_cells_per_exp = maximum(n_samples_per_protocol[p] + n_controls_per_protocol[p]
                                    for p in 1:n_protocols)
        for p in 1:n_protocols, e in 1:n_exp_per_protocol, s in 1:max_cells_per_exp
            column_imputation_sigma_sq[(p, e, s)] = sigma_imp_sq
        end

        return (
            raw                        = raw_data,
            imputed                    = imputed_data,
            column_imputation_sigma_sq = column_imputation_sigma_sq,
            refID                      = bait_idx,
        )
    end

    """
        load_single_protocol_fixture() -> NamedTuple

    Deterministic single-protocol (`no_protocols == 1`) synthetic `InteractionData`
    for the "single-protocol unaffected" contract — `:auto` must resolve to `:none`
    (no cross-protocol offset exists to correct). Returns `(raw, refID)`.
    """
    function load_single_protocol_fixture()
        Random.seed!(SEED + 1)

        n_proteins = 40
        n_exp      = 2
        n_samples  = 3
        n_controls = 2
        bait_idx   = 1
        bait_mean  = 12.0
        n_interactors = 8

        protein_ids   = ["P$(lpad(i, 3, '0'))" for i in 1:n_proteins]
        protein_names = ["Protein_$(lpad(i, 3, '0'))" for i in 1:n_proteins]
        baseline = Float64[8.0 + 0.5 * randn() for _ in 1:n_proteins]

        function _sample(n_s)
            mat = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_s)
            for prot in 1:n_proteins
                if prot == bait_idx
                    for s in 1:n_s; mat[prot, s] = bait_mean + 0.3 * randn(); end
                elseif prot <= 1 + n_interactors
                    enr = 0.8 * (bait_mean - 8.0)
                    for s in 1:n_s; mat[prot, s] = baseline[prot] + enr + 0.4 * randn(); end
                else
                    for s in 1:n_s; mat[prot, s] = baseline[prot] + 0.4 * randn(); end
                end
            end
            return mat
        end
        function _control(n_c)
            mat = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_c)
            for prot in 1:n_proteins
                base = prot == bait_idx ? 6.0 : baseline[prot]
                for c in 1:n_c; mat[prot, c] = base + 0.4 * randn(); end
            end
            return mat
        end

        sraw = Dict{Int, Matrix{Union{Missing, Float64}}}()
        craw = Dict{Int, Matrix{Union{Missing, Float64}}}()
        for e in 1:n_exp
            sraw[e] = _sample(n_samples)
            craw[e] = _control(n_controls)
        end
        samples_dict  = Dict(1 => Protocol(n_exp, protein_ids, sraw))
        controls_dict = Dict(1 => Protocol(n_exp, protein_ids, craw))
        no_exp_dict   = Dict(1 => n_exp)
        no_hbm = 1 + 1 + 1 * n_exp
        no_reg = 1 + 1
        pp, ep, mp = getPositions(no_exp_dict, no_hbm)
        raw_data = InteractionData(
            protein_ids, protein_names,
            samples_dict, controls_dict,
            1, no_exp_dict,
            no_hbm, no_reg, ep, pp, mp,
            trues(n_proteins),
        )
        return (raw = raw_data, refID = bait_idx)
    end
end
