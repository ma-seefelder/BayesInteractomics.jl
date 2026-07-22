@testitem "prepare_regression_data is_imputed mask" tags=[:mask_aware] begin
    using BayesInteractomics
    using BayesInteractomics: prepare_regression_data, Protocol, InteractionData

    # ---------------------------------------------------------------------
    # Fixture: 1 protocol, 1 experiment, 3 samples + 1 control, 3 proteins.
    # Inject missings into RAW data at known cells:
    #   - protein 1 (the protein of interest), sample column 2  -> missing
    #   - protein 2 (the reference protein), sample column 1    -> missing
    # The IMPUTED data fills both with concrete Float64 values (e.g., 1.0).
    # ---------------------------------------------------------------------
    protein_ids = ["P1", "P2", "P3"]
    protein_names = ["Protein1", "Protein2", "Protein3"]

    # RAW sample matrix (proteins x samples) — explicit Union{Missing,Float64}
    raw_sample::Matrix{Union{Missing, Float64}} = [
        10.0 missing  12.0;   # protein 1: col 2 missing
        missing 9.0   10.0;   # protein 2: col 1 missing
        5.0    6.0    7.0     # protein 3: no missing
    ]
    raw_control::Matrix{Union{Missing, Float64}} = [
        8.0 9.0 10.0;
        8.0 9.0 10.0;
        5.0 6.0 7.0
    ]

    # IMPUTED matrices: same shape, missings replaced by 1.0
    imp_sample::Matrix{Union{Missing, Float64}} = [
        10.0 1.0  12.0;
        1.0  9.0  10.0;
        5.0  6.0  7.0
    ]
    imp_control::Matrix{Union{Missing, Float64}} = [
        8.0 9.0 10.0;
        8.0 9.0 10.0;
        5.0 6.0 7.0
    ]

    function make_data(sample_mat, control_mat)
        p_sample  = Protocol(1, protein_ids, Dict(1 => sample_mat))
        p_control = Protocol(1, protein_ids, Dict(1 => control_mat))
        return InteractionData(
            protein_ids, protein_names,
            Dict(1 => p_sample),
            Dict(1 => p_control),
            1, Dict(1 => 1),
            3, 2,
            [2], [3], [1],
            trues(3),
        )
    end

    raw     = make_data(raw_sample, raw_control)
    imputed = make_data(imp_sample, imp_control)

    # =====================================================================
    # Test 1: legacy 2-arg call — 3-field NamedTuple, all-false mask
    # =====================================================================
    result_legacy = prepare_regression_data(imputed, 1, 2)
    @test result_legacy isa NamedTuple
    @test :is_imputed in keys(result_legacy)
    @test length(keys(result_legacy)) == 3
    @test result_legacy.is_imputed isa AbstractArray{Bool}
    @test all(.!result_legacy.is_imputed)

    # =====================================================================
    # Test 2: raw_data kwarg — mask matches cell-wise OR of raw missings
    # =====================================================================
    result = prepare_regression_data(imputed, 1, 2; raw_data=raw)
    @test result.is_imputed isa AbstractArray{Bool}
    @test size(result.is_imputed) == size(result.sample)

    # Ground truth: rebuild raw_sample_full + raw_reference_full the same way
    # the function does and verify per-cell OR.
    # Shape is (n_protocols=1, n_experiments_after_cat=2, n_replicates=3).
    # Sample-side lives at experiment dim index 1; control-side at index 2.
    expected = Array{Bool, 3}(undef, 1, 2, 3)
    for rep in 1:3
        # sample-side (experiment slot 1): protein 1 row, col `rep` of raw_sample
        expected[1, 1, rep] = ismissing(raw_sample[1, rep]) || ismissing(raw_sample[2, rep])
        # control-side (experiment slot 2): protein 1 row, col `rep` of raw_control
        expected[1, 2, rep] = ismissing(raw_control[1, rep]) || ismissing(raw_control[2, rep])
    end
    @test result.is_imputed == expected

    # Spot-check at least 2 distinct cells: one TRUE, one FALSE.
    # Cell (1, 1, 2): sample-side rep 2 — protein 1 sample col 2 is missing in raw -> TRUE.
    @test result.is_imputed[1, 1, 2] == true
    # Cell (1, 1, 1): sample-side rep 1 — protein 1 sample col 1 NOT missing AND
    # protein 2 sample col 1 IS missing -> cell-wise OR -> TRUE
    @test result.is_imputed[1, 1, 1] == true
    # Cell (1, 1, 3): sample-side rep 3 — both proteins have finite values -> FALSE
    @test result.is_imputed[1, 1, 3] == false
    # Cell (1, 2, 1): control-side rep 1 — both proteins have finite control values -> FALSE
    @test result.is_imputed[1, 2, 1] == false

    # =====================================================================
    # Test 3: raw_data=nothing identical to legacy 2-arg
    # =====================================================================
    result_nothing = prepare_regression_data(imputed, 1, 2; raw_data=nothing)
    @test result_nothing.is_imputed == result_legacy.is_imputed
    @test result_nothing.sample == result_legacy.sample
    @test result_nothing.reference == result_legacy.reference

    # =====================================================================
    # Test 4: size invariants
    # =====================================================================
    @test size(result.is_imputed) == size(result.sample) == size(result.reference)

    # =====================================================================
    # Test 5: positional unpacking still binds first two fields
    # =====================================================================
    s, r = prepare_regression_data(imputed, 1, 2)
    @test s == result_legacy.sample
    @test r == result_legacy.reference
end
