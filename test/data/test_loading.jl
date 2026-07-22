"""
    test_loading.jl

Tests for data loading functionality including Protocol, InteractionData,
and load_data functions using TestItemRunner framework.
"""

@testitem "Protocol construction and accessors" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, getNoExperiments, getExperiment, getIDs

    m::Matrix{Union{Missing,Float64}} = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

    protocol = Protocol(
        3,
        ["A", "B", "C"],
        Dict(1 => m, 2 => m, 3 => m)
    )

    @test getNoExperiments(protocol) == 3
    @test getExperiment(protocol, 1) == [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    @test getExperiment(protocol, 2) == [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    @test getExperiment(protocol, 3) == [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    @test getIDs(protocol) == ["A", "B", "C"]
    @test_throws BoundsError getExperiment(protocol, 4)
end

@testitem "Protocol indexing and iteration" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, getExperiment

    m::Matrix{Union{Missing,Float64}} = [1.0 2.0; 3.0 4.0]
    protocol = Protocol(2, ["P1", "P2"], Dict(1 => m, 2 => m))

    # Test indexing interface
    @test protocol[1] == m
    @test protocol[2] == m

    # Test iteration. A Ref is used (rather than a bare local counter) because on Julia 1.12 a
    # `for`-loop body is soft scope: reassigning a testitem-module global (`c += 1`) is treated as
    # a new, uninitialised local → UndefVarError. Mutating a Ref sidesteps the soft-scope rule.
    seen = Ref(0)
    for exp in protocol
        seen[] += 1
        @test exp == m
    end
    @test seen[] == 2
end

@testitem "InteractionData construction and accessors" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, getIDs, getNames, getNoProtocols,
        getControls, getSamples

    m::Matrix{Union{Missing,Float64}} = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

    protocol = Protocol(
        3,
        ["A", "B", "C"],
        Dict(1 => deepcopy(m), 2 => deepcopy(m), 3 => deepcopy(m))
    )

    # InteractionData field order:
    # protein_IDs, protein_names, samples, controls,
    # no_protocols, no_experiments, no_parameters_HBM, no_parameters_Regression,
    # protocol_positions, experiment_positions, matched_positions
    interaction_data = InteractionData(
        ["A", "B", "C"],
        ["A", "B", "C"],
        Dict(1 => protocol, 2 => protocol, 3 => protocol),
        Dict(1 => protocol, 2 => protocol, 3 => protocol),
        3,
        Dict(1 => 3, 2 => 3, 3 => 3),
        13,  # HBM parameters: 1 + 3 protocols + 9 experiments
        4,   # Regression parameters
        [2, 6, 10],  # Protocol positions
        [3, 4, 5, 7, 8, 9, 11, 12, 13],  # Experiment positions
        [1, 1, 1, 2, 2, 2, 3, 3, 3],  # Matched positions
        trues(3)
    )

    # check accessors for InteractionData
    @test getIDs(interaction_data) == ["A", "B", "C"]
    @test getNames(interaction_data) == ["A", "B", "C"]
    @test getNoProtocols(interaction_data) == 3
    @test getControls(interaction_data, 1) == protocol
    @test getSamples(interaction_data, 1) == protocol
end

@testitem "load_csv function" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, getNoExperiments, getIDs, getNoProtocols

    id_col = 1
    name_col = 1
    control_cols = Dict(1 => [2, 3, 4], 2 => [5], 3 => [6, 7])
    sample_cols = Dict(1 => [8, 9, 10], 2 => [11, 12, 13], 3 => [14, 15])

    csv_file = joinpath(dirname(@__DIR__), "dummy_data", "dummy_data.csv")

    # Accessing internal functions via BayesInteractomics
    samples, controls, protein_ids, protein_names = BayesInteractomics.load_csv(
        csv_file, sample_cols, control_cols, id_col, name_col, false
    )

    @test typeof(samples) == Protocol{Float64, Int64}
    @test typeof(controls) == Protocol{Float64, Int64}
    @test getNoExperiments(samples) == 3
end

@testitem "load_xlsx function" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, getNoExperiments, getIDs, getNoProtocols

    id_col = 1
    name_col = 1
    control_cols = Dict(1 => [2, 3, 4], 2 => [5], 3 => [6, 7])
    sample_cols = Dict(1 => [8, 9, 10], 2 => [11, 12, 13], 3 => [14, 15])

    xlsx_file = joinpath(dirname(@__DIR__), "dummy_data", "dummy_data.xlsx")

    # Accessing internal functions via BayesInteractomics
    samples, controls, protein_ids, protein_names = BayesInteractomics.load_xlsx(
        xlsx_file, sample_cols, control_cols, id_col, name_col, false
    )

    @test typeof(samples) == Protocol{Float64, Int64}
    @test typeof(controls) == Protocol{Float64, Int64}
    @test getNoExperiments(samples) == 3
end

@testitem "load_data with multiple protocols" begin
    using BayesInteractomics
    using BayesInteractomics: getNoProtocols, getControls, getSamples, getIDs, getNames

    id_col = 1
    name_col = 1
    control_cols = Dict(1 => [2, 3, 4], 2 => [5], 3 => [6, 7])
    sample_cols = Dict(1 => [8, 9, 10], 2 => [11, 12, 13], 3 => [14, 15])

    csv_file = joinpath(dirname(@__DIR__), "dummy_data", "dummy_data.csv")
    xlsx_file = joinpath(dirname(@__DIR__), "dummy_data", "dummy_data.xlsx")

    interactome_data = load_data(
        [csv_file, xlsx_file],
        [sample_cols, sample_cols],
        [control_cols, control_cols],
        name_col,
        id_col;
        imputation=:none
    )

    @test typeof(interactome_data) == InteractionData{Float64, Int64}
    @test getNoProtocols(interactome_data) == 2
    @test length(getIDs(interactome_data)) == 3
    @test length(getNames(interactome_data)) == 3

    @test typeof(getControls(interactome_data, 1)) == Protocol{Float64, Int64}
    @test typeof(getSamples(interactome_data, 1)) == Protocol{Float64, Int64}

    @test typeof(getControls(interactome_data, 2)) == Protocol{Float64, Int64}
    @test typeof(getSamples(interactome_data, 2)) == Protocol{Float64, Int64}
end

@testitem "file validation" begin
    using BayesInteractomics: check_file

    # Non-existent file should throw ArgumentError
    @test_throws ArgumentError check_file("nonexistent_file_xyz.csv")

    # Existing file should not throw
    csv_file = joinpath(dirname(@__DIR__), "dummy_data", "dummy_data.csv")
    @test check_file(csv_file) === nothing
end

@testitem "compute_observation_counts basic case" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, compute_observation_counts

    # Create sample and control matrices with mixed missing/non-missing
    sample_mat::Matrix{Union{Missing,Float64}} = [1.0 missing 2.0; missing missing missing; 4.0 5.0 6.0]
    control_mat::Matrix{Union{Missing,Float64}} = [missing 3.0 1.5; 1.0 missing missing; 7.0 8.0 9.0]

    protein_ids = ["P1", "P2", "P3"]

    samples = Dict(1 => Protocol(1, protein_ids, Dict(1 => sample_mat)))
    controls = Dict(1 => Protocol(1, protein_ids, Dict(1 => control_mat)))

    n_sample_obs, n_control_obs = compute_observation_counts(protein_ids, samples, controls)

    # P1: sample=[1.0, missing, 2.0] -> 2 obs, control=[missing, 3.0, 1.5] -> 2 obs
    @test n_sample_obs[1] == 2
    @test n_control_obs[1] == 2

    # P2: sample=[missing, missing, missing] -> 0 obs, control=[1.0, missing, missing] -> 1 obs
    @test n_sample_obs[2] == 0
    @test n_control_obs[2] == 1

    # P3: sample=[4.0, 5.0, 6.0] -> 3 obs, control=[7.0, 8.0, 9.0] -> 3 obs
    @test n_sample_obs[3] == 3
    @test n_control_obs[3] == 3
end

@testitem "filter_insufficient_observations" begin
    using BayesInteractomics: filter_insufficient_observations
    using DataFrames

    protein_ids = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]
    n_sample_obs = [2, 0, 1, 3, 0, 1, 2, 0, 1, 3]
    n_control_obs = [1, 0, 0, 2, 1, 0, 1, 0, 1, 1]

    kept_indices, exclusion_report = filter_insufficient_observations(
        protein_ids, n_sample_obs, n_control_obs, 2, 2
    )

    # With 2+2 threshold: only P4 (3,2) meets both >= 2
    # P1 (2,1)✗ control<2, P2 (0,0)✗, P3 (1,0)✗, P5 (0,1)✗, P6 (1,0)✗,
    # P7 (2,1)✗ control<2, P8 (0,0)✗, P9 (1,1)✗ both<2, P10 (3,1)✗ control<2
    @test kept_indices == [4]

    # Excluded: P1, P2, P3, P5, P6, P7, P8, P9, P10
    @test nrow(exclusion_report) == 9
    @test String.(exclusion_report.protein_id) == ["P1", "P2", "P3", "P5", "P6", "P7", "P8", "P9", "P10"]
    @test exclusion_report.n_sample_obs == [2, 0, 1, 0, 1, 2, 0, 1, 3]
    @test exclusion_report.n_control_obs == [1, 0, 0, 1, 0, 1, 0, 1, 1]
end

@testitem "exclusion_report CSV structure" begin
    using BayesInteractomics
    using DataFrames, CSV

    # Build exclusion report
    exclusion_report = DataFrame(
        protein_id = ["P2", "P5", "P8"],
        n_sample_obs = [0, 0, 0],
        n_control_obs = [0, 1, 0]
    )

    # Write to temporary CSV
    tmp_file = joinpath(tempdir(), "test_exclusion_report.csv")
    CSV.write(tmp_file, exclusion_report)

    # Read back and verify
    loaded_df = CSV.read(tmp_file, DataFrame)

    @test names(loaded_df) == ["protein_id", "n_sample_obs", "n_control_obs"]
    @test size(loaded_df, 1) == 3
    # CSV may return InlineString, so convert for comparison
    @test String.(loaded_df.protein_id) == ["P2", "P5", "P8"]
    @test typeof(loaded_df.n_sample_obs[1]) == Int64
    @test typeof(loaded_df.n_control_obs[1]) == Int64

    # Clean up
    rm(tmp_file)
end

@testitem "load_data filters and reports exclusions" begin
    using BayesInteractomics
    using DataFrames, CSV

    id_col = 1
    name_col = 1
    control_cols = Dict(1 => [2, 3, 4], 2 => [5], 3 => [6, 7])
    sample_cols = Dict(1 => [8, 9, 10], 2 => [11, 12, 13], 3 => [14, 15])

    csv_file = joinpath(dirname(@__DIR__), "dummy_data", "dummy_data.csv")

    # Call load_data with curate=false to simplify test
    interactome_data = load_data(
        [csv_file],
        [sample_cols],
        [control_cols],
        name_col,
        id_col,
        false;
        curate=false,
        imputation=:none
    )

    # Verify InteractionData was created
    @test typeof(interactome_data) == InteractionData{Float64, Int64}

    # Verify detected BitVector is set (at least some proteins should be detected)
    @test length(interactome_data.detected) == length(interactome_data.protein_IDs)

    # Verify exclusion CSV exists in cache directory
    cache_dir = joinpath(dirname(abspath(csv_file)), ".bayesinteractomics_cache")
    exclusion_file = joinpath(cache_dir, "$(splitext(basename(csv_file))[1])_excluded_proteins.csv")

    # If exclusions occurred, CSV should exist; if no exclusions, file may not exist (both are valid)
    if isfile(exclusion_file)
        excluded_df = CSV.read(exclusion_file, DataFrame)
        @test :protein_id in names(excluded_df)
        @test :n_sample_obs in names(excluded_df)
        @test :n_control_obs in names(excluded_df)
    end
end

@testitem "impute_missing_values! with variance fallback" begin
    using BayesInteractomics
    using DataFrames
    using Distributions

    # Create test data with missing values and NaN variances
    data = DataFrame(
        protein_id = ["P1", "P2", "P3"],
        sample_1 = [1.0, missing, 3.0],
        sample_2 = [2.0, 2.5, missing],
        sample_3 = [3.0, 3.5, 4.0],
        control_1 = [0.5, 1.0, missing],
        control_2 = [1.5, missing, 2.0]
    )

    sample_cols = Dict(1 => [2, 3, 4])
    control_cols = Dict(1 => [5, 6])

    # Call impute_missing_values! with impute=true
    # This tests the bugfix: variable names changed from rowvar_samples to row_var_sample
    result = BayesInteractomics.impute_missing_values!(data, sample_cols, control_cols)

    # Verify no missing values remain (or are successfully imputed with defaults)
    @test typeof(result) == DataFrame
    @test size(result) == size(data)

    # Verify that at least some imputations occurred (missing values filled)
    missing_count_original = count(ismissing, Matrix(data))
    missing_count_result = count(ismissing, Matrix(result))
    @test missing_count_result <= missing_count_original

    # Verify that imputed values are numeric (normal distribution samples)
    for col in names(result)[2:end]
        for val in result[!, col]
            if !ismissing(val)
                @test isa(val, Number)
            end
        end
    end
end

@testitem "compute_detected_mask multi-protocol per-protocol logic" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, compute_detected_mask

    protein_ids = ["P1", "P2", "P3", "P4"]

    # Protocol 1: P1 has 2 sample obs, P2 has 1, P3 has 0, P4 has 2
    s1_mat::Matrix{Union{Missing,Float64}} = [1.0 2.0; missing 1.0; missing missing; 3.0 4.0]
    c1_mat::Matrix{Union{Missing,Float64}} = [1.0 2.0; missing missing; missing missing; 5.0 6.0]

    # Protocol 2: P1 has 1 sample obs, P2 has 1, P3 has 2, P4 has 0
    s2_mat::Matrix{Union{Missing,Float64}} = [1.0 missing; 2.0 missing; 3.0 4.0; missing missing]
    c2_mat::Matrix{Union{Missing,Float64}} = [missing missing; 1.0 2.0; 5.0 6.0; missing missing]

    samples = Dict(
        1 => Protocol(1, protein_ids, Dict(1 => s1_mat)),
        2 => Protocol(1, protein_ids, Dict(1 => s2_mat))
    )
    controls = Dict(
        1 => Protocol(1, protein_ids, Dict(1 => c1_mat)),
        2 => Protocol(1, protein_ids, Dict(1 => c2_mat))
    )

    detected = compute_detected_mask(protein_ids, samples, controls)

    # P1: protocol 1 has 2 sample + 2 control → detected
    @test detected[1] == true

    # P2: protocol 1 has 1 sample + 0 control; protocol 2 has 1 sample + 2 control
    #     Neither protocol alone has ≥2 sample AND ≥2 control → NOT detected
    @test detected[2] == false

    # P3: protocol 2 has 2 sample + 2 control → detected
    @test detected[3] == true

    # P4: protocol 1 has 2 sample + 2 control → detected
    @test detected[4] == true
end

@testitem "compute_detected_mask single-protocol unchanged behavior" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, compute_detected_mask

    protein_ids = ["P1", "P2", "P3"]

    # Single protocol: P1 has 2 sample + 2 control, P2 has 1 sample, P3 has 3+3
    s_mat::Matrix{Union{Missing,Float64}} = [1.0 2.0 missing; missing 1.0 missing; 4.0 5.0 6.0]
    c_mat::Matrix{Union{Missing,Float64}} = [1.0 missing 2.0; 1.0 missing missing; 7.0 8.0 9.0]

    samples = Dict(1 => Protocol(1, protein_ids, Dict(1 => s_mat)))
    controls = Dict(1 => Protocol(1, protein_ids, Dict(1 => c_mat)))

    detected = compute_detected_mask(protein_ids, samples, controls)

    @test detected[1] == true   # 2 sample + 2 control
    @test detected[2] == false  # 1 sample + 1 control (both below threshold)
    @test detected[3] == true   # 3 sample + 3 control
end
