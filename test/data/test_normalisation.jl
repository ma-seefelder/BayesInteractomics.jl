"""
    test_normalisation.jl

TestItemRunner blocks for the normalisation
method correctness + back-compat byte-equality anchors + log2FC invariance +
single-protocol-unaffected contracts.

Testitems:
  1. median_of_ratios correctness + missing-aware + log2 round-trip
  2. :none byte-identical to normalise_protocols=false (apply_normalisation + load_data forms)
  3. :row_center byte-identical to normalise_protocols=true (apply_normalisation + load_data forms)
  4. row-centering leaves log2FC invariant (HBM-safe)
  5. single-protocol unaffected (:auto resolves to :none; byte-identical to no-normalisation)

The committed scale-disparate / matched fixture is `test/fixtures/scale_disparate_multiprotocol.jl`.
The committed `dummy_data.csv` (3-protocol) is used for the load_data byte-equality forms.
"""


@testitem "median_of_ratios correctness + missing-aware + log2 round-trip" setup=[ScaleDisparateMultiprotocol] begin
    using BayesInteractomics
    using BayesInteractomics: norm_median_of_ratios_id, build_run_matrix, getIDs,
        getSamples, getControls
    using Statistics

    fx = ScaleDisparateMultiprotocol.load_fixture(matched=false)
    d  = fx.raw

    normed = norm_median_of_ratios_id(d)

    # Flatten both to run-matrices for cell-wise comparison.
    X0, meta0, _ = build_run_matrix(d)
    X1, meta1, _ = build_run_matrix(normed)

    @test size(X0) == size(X1)
    @test meta0 == meta1   # deterministic column layout preserved

    # (a) Missings preserved 1:1 — a cell is missing in the output iff missing in input.
    mismatched_missing = count(i -> ismissing(X0[i]) != ismissing(X1[i]), eachindex(X0))
    @test mismatched_missing == 0

    # (b) Zero NaN/Inf in the observed (non-missing) output cells.
    obs_out = Float64[Float64(x) for x in X1 if !ismissing(x)]
    @test !isempty(obs_out)
    @test all(isfinite, obs_out)

    # (c) Output stays in a sane log2 band (no scale explosion / collapse).
    @test all(v -> -10.0 <= v <= 40.0, obs_out)

    # (d) Composition-robustness: inject a KNOWN per-column loading factor on the
    #     linear scale (+Δ log2 on every observed cell of one set of columns) and
    #     confirm median_of_ratios recovers / equalises the per-column observed
    #     medians (the size factors absorb the synthetic loading difference).
    # Build a copy of the matrix with a +1.5 log2 loading bump on the first half of columns.
    Xb = copy(X0)
    nc = size(Xb, 2)
    bumped_cols = 1:(nc ÷ 2)
    for j in bumped_cols, i in 1:size(Xb, 1)
        ismissing(Xb[i, j]) || (Xb[i, j] = Float64(Xb[i, j]) + 1.5)
    end
    # Round-trip the bumped matrix into an InteractionData (mirror d), normalise.
    d_bumped = BayesInteractomics.matrix_to_interactiondata(d, Xb, meta0)
    d_bumped_norm = norm_median_of_ratios_id(d_bumped)
    Xbn, _, _ = build_run_matrix(d_bumped_norm)

    colmed(M, j) = (o = Float64[Float64(x) for x in @view(M[:, j]) if !ismissing(x)];
                    isempty(o) ? NaN : median(o))
    pre_meds  = [colmed(Xb,  j) for j in 1:nc]
    post_meds = [colmed(Xbn, j) for j in 1:nc]
    # The injected +1.5 loading spread should SHRINK after median-of-ratios scaling.
    pre_spread  = maximum(filter(!isnan, pre_meds))  - minimum(filter(!isnan, pre_meds))
    post_spread = maximum(filter(!isnan, post_meds)) - minimum(filter(!isnan, post_meds))
    @test post_spread < pre_spread
end


@testitem ":none byte-identical to normalise_protocols=false" setup=[ScaleDisparateMultiprotocol] begin
    using BayesInteractomics
    using BayesInteractomics: apply_normalisation

    fx = ScaleDisparateMultiprotocol.load_fixture(matched=false)
    d  = fx.raw

    # apply_normalisation form: :none returns the data unchanged (content-equal,
    # and in fact the identical object — identity is the strongest byte-equality).
    @test isequal(apply_normalisation(d, :none), d)

    # load_data form: normalisation_method=:none ≡ normalise_protocols=false.
    control_cols = Dict(1 => [2, 3, 4], 2 => [5], 3 => [6, 7])
    sample_cols  = Dict(1 => [8, 9, 10], 2 => [11, 12, 13], 3 => [14, 15])
    csv_file = joinpath(dirname(@__DIR__), "dummy_data", "dummy_data.csv")

    d_method = load_data([csv_file], [sample_cols], [control_cols], 1, 1;
                         normalisation_method=:none, curate=false, imputation=:none)
    d_bool   = load_data([csv_file], [sample_cols], [control_cols], 1, 1;
                         normalise_protocols=false, curate=false, imputation=:none)
    @test isequal(d_method, d_bool)
end


@testitem ":row_center byte-identical to normalise_protocols=true" setup=[ScaleDisparateMultiprotocol] begin
    using BayesInteractomics
    using BayesInteractomics: apply_normalisation, normalize

    fx = ScaleDisparateMultiprotocol.load_fixture(matched=false)
    d  = fx.raw

    # apply_normalisation form: :row_center ≡ the existing normalize().
    @test isequal(apply_normalisation(d, :row_center), normalize(d))

    # load_data form: normalisation_method=:row_center ≡ normalise_protocols=true.
    control_cols = Dict(1 => [2, 3, 4], 2 => [5], 3 => [6, 7])
    sample_cols  = Dict(1 => [8, 9, 10], 2 => [11, 12, 13], 3 => [14, 15])
    csv_file = joinpath(dirname(@__DIR__), "dummy_data", "dummy_data.csv")

    d_method = load_data([csv_file], [sample_cols], [control_cols], 1, 1;
                         normalisation_method=:row_center, curate=false, imputation=:none)
    d_bool   = load_data([csv_file], [sample_cols], [control_cols], 1, 1;
                         normalise_protocols=true, curate=false, imputation=:none)
    @test isequal(d_method, d_bool)
end


@testitem "row-centering leaves log2FC invariant (HBM-safe)" setup=[ScaleDisparateMultiprotocol] begin
    using BayesInteractomics
    using BayesInteractomics: apply_normalisation, build_run_matrix, getIDs
    using Statistics

    fx = ScaleDisparateMultiprotocol.load_fixture(matched=false)
    d  = fx.raw
    rc = apply_normalisation(d, :row_center)

    # Per-protein per-protocol enrichment = mean(sample) − mean(control).
    # Row-centering subtracts a per-(protein,protocol,exp) constant from BOTH
    # sample and control cells → cancels in the contrast → log2FC invariant
    # (HBM-safe).
    function enrichment_vec(data)
        X, meta, ids = build_run_matrix(data)
        np = length(ids)
        protocols = sort(unique(m.protocol for m in meta))
        out = Float64[]
        for p in protocols
            scol = [j for (j, m) in enumerate(meta) if m.protocol == p && m.group == :sample]
            ccol = [j for (j, m) in enumerate(meta) if m.protocol == p && m.group == :control]
            for i in 1:np
                so = Float64[Float64(x) for x in @view(X[i, scol]) if !ismissing(x)]
                co = Float64[Float64(x) for x in @view(X[i, ccol]) if !ismissing(x)]
                (isempty(so) || isempty(co)) && continue
                push!(out, mean(so) - mean(co))
            end
        end
        return out
    end

    e0 = enrichment_vec(d)
    e1 = enrichment_vec(rc)
    @test length(e0) == length(e1)
    @test !isempty(e0)
    @test maximum(abs.(e0 .- e1)) <= 1e-9
end


@testitem "single-protocol unaffected (:auto -> :none, byte-identical)" setup=[ScaleDisparateMultiprotocol] begin
    using BayesInteractomics
    using BayesInteractomics: apply_normalisation, detect_protocol_scale_mismatch,
        _resolve_normalisation_method

    fx = ScaleDisparateMultiprotocol.load_single_protocol_fixture()
    d  = fx.raw

    # :auto on single-protocol data must NOT fire the detector.
    @test detect_protocol_scale_mismatch(d; refID=1) == false

    # The load_data :auto resolution applies :none when the detector returns false,
    # so the data is byte-identical to the no-normalisation path. We assert the
    # equivalent dispatcher outcome directly: :auto would resolve to :none here
    # (detector false), and apply_normalisation(d, :none) === d.
    @test isequal(apply_normalisation(d, :none), d)

    # _resolve_normalisation_method keeps :auto as :auto (the detector runs in
    # load_data, not the resolver) — the resolver maps only the legacy bool. The
    # single-protocol "unaffected" contract is the detector returning false above.
    @test _resolve_normalisation_method(:none, false) == :none
    @test _resolve_normalisation_method(:none, true)  == :row_center
end
