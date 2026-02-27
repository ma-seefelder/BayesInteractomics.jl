using BayesInteractomics
using Test
using DataFrames
using Dates

@testitem "Intermediate cache hash functions" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, compute_betabernoulli_hash, compute_hbm_regression_hash
    using DataFrames

    # Create minimal test data
    mat = Union{Missing, Float64}[1.0 2.0; 3.0 4.0; 5.0 6.0]
    protocol = Protocol(
        1,  # no_experiments
        ["Protein1", "Protein2", "Protein3"],
        Dict(1 => mat)
    )
    data = InteractionData(
        ["Protein1", "Protein2", "Protein3"],  # protein_IDs
        ["Protein1", "Protein2", "Protein3"],  # protein_names
        Dict(1 => protocol),  # samples
        Dict(1 => protocol),  # controls
        1, Dict(1 => 1),  # no_protocols, no_experiments
        3, 2,  # no_parameters_HBM, no_parameters_Regression
        [2], [3], [1]  # positions
    )

    # Test Beta-Bernoulli hash
    h1 = compute_betabernoulli_hash(data, 3, 3)
    h2 = compute_betabernoulli_hash(data, 3, 3)
    h3 = compute_betabernoulli_hash(data, 4, 3)  # Different n_controls

    @test h1 == h2  # Same parameters should give same hash
    @test h1 != h3  # Different parameters should give different hash

    # Test HBM+Regression hash
    h4 = compute_hbm_regression_hash(data, 1, :normal, 5.0)
    h5 = compute_hbm_regression_hash(data, 1, :normal, 5.0)
    h6 = compute_hbm_regression_hash(data, 2, :normal, 5.0)  # Different refID
    h7 = compute_hbm_regression_hash(data, 1, :robust_t, 5.0)  # Different likelihood
    h8 = compute_hbm_regression_hash(data, 1, :robust_t, 7.0)  # Different nu

    @test h4 == h5  # Same parameters should give same hash
    @test h4 != h6  # Different refID should give different hash
    @test h4 != h7  # Different likelihood should give different hash
    @test h7 != h8  # Different nu should give different hash
end

@testitem "BetaBernoulliCache save/load round-trip" begin
    using BayesInteractomics
    using Dates
    using JLD2

    # Create test cache
    bf_detected = [1.0, 2.0, 3.0]
    protein_ids = ["P1", "P2", "P3"]
    n_controls = 3
    n_samples = 3
    data_hash = UInt64(12345)
    timestamp = now()
    pkg_version = "0.1.0"

    cache = BetaBernoulliCache(
        bf_detected, protein_ids, n_controls, n_samples,
        data_hash, timestamp, pkg_version
    )

    # Save and load
    temp_file = tempname() * ".jld2"
    try
        save_betabernoulli_cache(cache, temp_file)
        loaded = load_betabernoulli_cache(temp_file)

        @test !isnothing(loaded)
        @test loaded.bf_detected == bf_detected
        @test loaded.protein_ids == protein_ids
        @test loaded.n_controls == n_controls
        @test loaded.n_samples == n_samples
        @test loaded.data_hash == data_hash
        @test loaded.package_version == pkg_version
    finally
        isfile(temp_file) && rm(temp_file)
    end
end

@testitem "HBMRegressionCache save/load round-trip" begin
    using BayesInteractomics
    using DataFrames
    using Dates
    using JLD2

    # Create test cache
    df = DataFrame(
        Protein = ["P1", "P2", "P3"],
        BF_log2FC = [1.0, 2.0, 3.0],
        bf_slope = [0.5, 1.0, 1.5]
    )
    bf_enrichment = [1.0, 2.0, 3.0]
    bf_correlation = [0.5, 1.0, 1.5]
    protein_ids = ["P1", "P2", "P3"]
    refID = 1
    data_hash = UInt64(12345)
    timestamp = now()
    pkg_version = "0.1.0"

    regression_likelihood = :robust_t
    student_t_nu = 5.0

    cache = HBMRegressionCache(
        df, bf_enrichment, bf_correlation, protein_ids,
        refID, regression_likelihood, student_t_nu,
        data_hash, timestamp, pkg_version
    )

    # Save and load
    temp_file = tempname() * ".jld2"
    try
        save_hbm_regression_cache(cache, temp_file)
        loaded = load_hbm_regression_cache(temp_file)

        @test !isnothing(loaded)
        @test loaded.df_hierarchical == df
        @test loaded.bf_enrichment == bf_enrichment
        @test loaded.bf_correlation == bf_correlation
        @test loaded.protein_ids == protein_ids
        @test loaded.refID == refID
        @test loaded.regression_likelihood == regression_likelihood
        @test loaded.student_t_nu == student_t_nu
        @test loaded.data_hash == data_hash
        @test loaded.package_version == pkg_version
    finally
        isfile(temp_file) && rm(temp_file)
    end
end

@testitem "Beta-Bernoulli cache validation" begin
    using BayesInteractomics
    using BayesInteractomics: INTERMEDIATE_CACHE_HIT, INTERMEDIATE_CACHE_MISS_PARAMS, INTERMEDIATE_CACHE_MISS_NO_FILE, Protocol, InteractionData
    using BayesInteractomics: BetaBernoulliCache, save_betabernoulli_cache, check_betabernoulli_cache, compute_data_hash, getIDs
    using Dates

    # Create test data
    mat = Union{Missing, Float64}[1.0 2.0; 3.0 4.0; 5.0 6.0]
    protocol = Protocol(
        1,  # no_experiments
        ["Protein1", "Protein2", "Protein3"],
        Dict(1 => mat)
    )
    data = InteractionData(
        ["Protein1", "Protein2", "Protein3"],  # protein_IDs
        ["Protein1", "Protein2", "Protein3"],  # protein_names
        Dict(1 => protocol),  # samples
        Dict(1 => protocol),  # controls
        1, Dict(1 => 1),  # no_protocols, no_experiments
        3, 2,  # no_parameters_HBM, no_parameters_Regression
        [2], [3], [1]  # positions
    )

    n_controls = 3
    n_samples = 3

    # Create and save cache
    bf_detected = [1.0, 2.0, 3.0]
    cache = BetaBernoulliCache(
        bf_detected,
        getIDs(data),
        n_controls,
        n_samples,
        compute_data_hash(data),
        now(),
        "0.1.0"
    )

    temp_file = tempname() * ".jld2"
    try
        save_betabernoulli_cache(cache, temp_file)

        # Test cache hit
        status, cached = check_betabernoulli_cache(temp_file, data, n_controls, n_samples)
        @test status == INTERMEDIATE_CACHE_HIT
        @test !isnothing(cached)
        @test cached.bf_detected == bf_detected

        # Test parameter mismatch
        status, cached = check_betabernoulli_cache(temp_file, data, 4, n_samples)
        @test status == INTERMEDIATE_CACHE_MISS_PARAMS
        @test isnothing(cached)

        # Test non-existent file
        status, cached = check_betabernoulli_cache("nonexistent.jld2", data, n_controls, n_samples)
        @test status == INTERMEDIATE_CACHE_MISS_NO_FILE
        @test isnothing(cached)
    finally
        isfile(temp_file) && rm(temp_file)
    end
end

@testitem "HBM+Regression cache validation" begin
    using BayesInteractomics
    using BayesInteractomics: INTERMEDIATE_CACHE_HIT, INTERMEDIATE_CACHE_MISS_PARAMS, INTERMEDIATE_CACHE_MISS_NO_FILE, Protocol, InteractionData
    using BayesInteractomics: HBMRegressionCache, save_hbm_regression_cache, check_hbm_regression_cache, compute_data_hash, getIDs
    using DataFrames
    using Dates

    # Create test data
    mat = Union{Missing, Float64}[1.0 2.0; 3.0 4.0; 5.0 6.0]
    protocol = Protocol(
        1,  # no_experiments
        ["Protein1", "Protein2", "Protein3"],
        Dict(1 => mat)
    )
    data = InteractionData(
        ["Protein1", "Protein2", "Protein3"],  # protein_IDs
        ["Protein1", "Protein2", "Protein3"],  # protein_names
        Dict(1 => protocol),  # samples
        Dict(1 => protocol),  # controls
        1, Dict(1 => 1),  # no_protocols, no_experiments
        3, 2,  # no_parameters_HBM, no_parameters_Regression
        [2], [3], [1]  # positions
    )

    refID = 1

    # Create and save cache
    df = DataFrame(
        Protein = getIDs(data),
        BF_log2FC = [1.0, 2.0, 3.0],
        bf_slope = [0.5, 1.0, 1.5]
    )
    regression_likelihood = :robust_t
    student_t_nu = 5.0

    cache = HBMRegressionCache(
        df,
        [1.0, 2.0, 3.0],
        [0.5, 1.0, 1.5],
        getIDs(data),
        refID,
        regression_likelihood,
        student_t_nu,
        compute_data_hash(data),
        now(),
        "0.1.0"
    )

    temp_file = tempname() * ".jld2"
    try
        save_hbm_regression_cache(cache, temp_file)

        # Test cache hit
        status, cached = check_hbm_regression_cache(temp_file, data, refID, :robust_t, 5.0)
        @test status == INTERMEDIATE_CACHE_HIT
        @test !isnothing(cached)
        @test cached.bf_enrichment == [1.0, 2.0, 3.0]

        # Test parameter mismatch: different refID
        status, cached = check_hbm_regression_cache(temp_file, data, 2, :robust_t, 5.0)
        @test status == INTERMEDIATE_CACHE_MISS_PARAMS
        @test isnothing(cached)

        # Test parameter mismatch: different likelihood
        status, cached = check_hbm_regression_cache(temp_file, data, refID, :normal, 5.0)
        @test status == INTERMEDIATE_CACHE_MISS_PARAMS
        @test isnothing(cached)

        # Test parameter mismatch: different nu
        status, cached = check_hbm_regression_cache(temp_file, data, refID, :robust_t, 7.0)
        @test status == INTERMEDIATE_CACHE_MISS_PARAMS
        @test isnothing(cached)

        # Test non-existent file
        status, cached = check_hbm_regression_cache("nonexistent.jld2", data, refID, :robust_t, 5.0)
        @test status == INTERMEDIATE_CACHE_MISS_NO_FILE
        @test isnothing(cached)
    finally
        isfile(temp_file) && rm(temp_file)
    end
end

@testitem "Cache file path generation" begin
    using BayesInteractomics
    using BayesInteractomics: CONFIG, get_betabernoulli_cache_filepath, get_hbm_regression_cache_filepath

    # Create mock CONFIG
    config = CONFIG(
        datafile = [joinpath(tempdir(), "test_data.xlsx")],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols = [Dict(1 => [4,5,6])],
        poi = "TestProtein",
        n_controls = 3,
        n_samples = 3,
        refID = 1
    )

    # Test Beta-Bernoulli cache path
    bb_path = get_betabernoulli_cache_filepath(config)
    @test occursin(".bayesinteractomics_cache", bb_path)
    @test occursin("betabernoulli_", bb_path)
    @test endswith(bb_path, ".jld2")

    # Test HBM+Regression cache path
    hbm_path = get_hbm_regression_cache_filepath(config)
    @test occursin(".bayesinteractomics_cache", hbm_path)
    @test occursin("hbm_regression_", hbm_path)
    @test occursin("_ref1", hbm_path)
    @test occursin("_robust_t", hbm_path)
    @test occursin("_nu5.0", hbm_path)
    @test endswith(hbm_path, ".jld2")

    # Paths should be different
    @test bb_path != hbm_path

    # Changing refID should change HBM path but not BB path
    config2 = deepcopy(config)
    config2.refID = 2
    bb_path2 = get_betabernoulli_cache_filepath(config2)
    hbm_path2 = get_hbm_regression_cache_filepath(config2)

    @test bb_path == bb_path2  # Beta-Bernoulli path unchanged
    @test hbm_path != hbm_path2  # HBM path changed

    # Changing likelihood should change HBM path
    config3 = deepcopy(config)
    config3.regression_likelihood = :normal
    hbm_path3 = get_hbm_regression_cache_filepath(config3)
    @test hbm_path != hbm_path3
    @test occursin("_normal", hbm_path3)
    @test !occursin("_nu", hbm_path3)  # nu not included for :normal

    # Changing nu should change HBM path
    config4 = deepcopy(config)
    config4.student_t_nu = 7.0
    hbm_path4 = get_hbm_regression_cache_filepath(config4)
    @test hbm_path != hbm_path4
    @test occursin("_nu7.0", hbm_path4)
end

@testitem "Invalid cache version handling" begin
    using BayesInteractomics
    using JLD2
    using Dates

    # Create cache with wrong version
    temp_file = tempname() * ".jld2"
    try
        jldsave(temp_file; compress=true,
            cache_version = 999,  # Wrong version
            bf_detected = [1.0, 2.0],
            protein_ids = ["P1", "P2"],
            n_controls = 3,
            n_samples = 3,
            data_hash = UInt64(123),
            timestamp = now(),
            package_version = "0.1.0"
        )

        # Should return nothing due to version mismatch
        loaded = load_betabernoulli_cache(temp_file)
        @test isnothing(loaded)
    finally
        isfile(temp_file) && rm(temp_file)
    end
end
