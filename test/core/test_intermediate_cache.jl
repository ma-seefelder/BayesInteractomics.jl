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
        [2], [3], [1],  # positions
        trues(3)
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
        data_hash, timestamp, pkg_version,
        :mnar  # imputation_method
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
        @test loaded.imputation_method === :mnar
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

    regression_bf_threshold = 0.0

    cache = HBMRegressionCache(
        df, bf_enrichment, bf_correlation, protein_ids,
        refID, regression_likelihood, student_t_nu,
        regression_bf_threshold,
        data_hash, timestamp, pkg_version,
        :mnar  # imputation_method
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
        @test loaded.regression_bf_threshold == regression_bf_threshold
        @test loaded.data_hash == data_hash
        @test loaded.package_version == pkg_version
        @test loaded.imputation_method === :mnar
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
        [2], [3], [1],  # positions
        trues(3)
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
        "0.1.0",
        :mnar  # imputation_method
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
        [2], [3], [1],  # positions
        trues(3)
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

    regression_bf_threshold = 0.0

    cache = HBMRegressionCache(
        df,
        [1.0, 2.0, 3.0],
        [0.5, 1.0, 1.5],
        getIDs(data),
        refID,
        regression_likelihood,
        student_t_nu,
        regression_bf_threshold,
        compute_data_hash(data),
        now(),
        "0.1.0",
        :mnar  # imputation_method
    )

    temp_file = tempname() * ".jld2"
    try
        save_hbm_regression_cache(cache, temp_file)

        # Test cache hit
        status, cached = check_hbm_regression_cache(temp_file, data, refID, :robust_t, 5.0, 0.0)
        @test status == INTERMEDIATE_CACHE_HIT
        @test !isnothing(cached)
        @test cached.bf_enrichment == [1.0, 2.0, 3.0]
        @test cached.regression_bf_threshold == 0.0

        # Test parameter mismatch: different refID
        status, cached = check_hbm_regression_cache(temp_file, data, 2, :robust_t, 5.0, 0.0)
        @test status == INTERMEDIATE_CACHE_MISS_PARAMS
        @test isnothing(cached)

        # Test parameter mismatch: different likelihood
        status, cached = check_hbm_regression_cache(temp_file, data, refID, :normal, 5.0, 0.0)
        @test status == INTERMEDIATE_CACHE_MISS_PARAMS
        @test isnothing(cached)

        # Test parameter mismatch: different nu
        status, cached = check_hbm_regression_cache(temp_file, data, refID, :robust_t, 7.0, 0.0)
        @test status == INTERMEDIATE_CACHE_MISS_PARAMS
        @test isnothing(cached)

        # Test parameter mismatch: different regression_bf_threshold
        status, cached = check_hbm_regression_cache(temp_file, data, refID, :robust_t, 5.0, 0.3)
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

@testitem "H0Cache save/load round-trip" begin
    using BayesInteractomics
    using BayesInteractomics: H0Cache, save_h0_cache, load_h0_cache
    using Dates
    using JLD2

    # H0Cache was redesigned (Option 2A hybrid H0): it now stores log-BF vectors + fitted marginal
    # parameters + KS diagnostics, not a DataFrame. Construct via the current 20-field struct.
    cache = H0Cache(
        [1.0, 2.0, 3.0],              # log_bf_enrichment
        [0.5, 1.0, 1.5],              # log_bf_correlation
        [0.8, 1.2, 0.9],              # log_bf_detection
        -1.0, 0.5, 0.0,              # marginal_enrichment mu/sigma/nu
        -0.8, 0.4, 0.0,              # marginal_correlation mu/sigma/nu
        -1.2, 0.6, 0.0,              # marginal_detection mu/sigma/nu
        0.05, 0.04, 0.06,           # ks_enrichment/correlation/detection
        3,                           # n_h0_proteins
        UInt64(12345), now(), "0.1.0", :mnar
    )

    temp_file = tempname() * ".jld2"
    try
        save_h0_cache(cache, temp_file)
        loaded = load_h0_cache(temp_file)

        @test !isnothing(loaded)
        @test loaded.log_bf_enrichment == [1.0, 2.0, 3.0]
        @test loaded.log_bf_correlation == [0.5, 1.0, 1.5]
        @test loaded.log_bf_detection == [0.8, 1.2, 0.9]
        @test loaded.marginal_enrichment_mu == -1.0
        @test loaded.marginal_correlation_sigma == 0.4
        @test loaded.marginal_detection_nu == 0.0
        @test loaded.ks_correlation == 0.04
        @test loaded.n_h0_proteins == 3
        @test loaded.data_hash == UInt64(12345)
        @test loaded.package_version == "0.1.0"
        @test loaded.imputation_method == :mnar
    finally
        isfile(temp_file) && rm(temp_file)
    end
end

@testitem "H0 cache validation" begin
    using BayesInteractomics
    using BayesInteractomics: INTERMEDIATE_CACHE_HIT, INTERMEDIATE_CACHE_MISS_PARAMS,
        INTERMEDIATE_CACHE_MISS_DATA, INTERMEDIATE_CACHE_MISS_NO_FILE
    using BayesInteractomics: Protocol, InteractionData, H0Cache, save_h0_cache, check_h0_cache, compute_data_hash
    using Dates

    # Create test data + its content hash
    mat = Union{Missing, Float64}[1.0 2.0; 3.0 4.0; 5.0 6.0]
    protocol = Protocol(
        1, ["Protein1", "Protein2", "Protein3"], Dict(1 => mat)
    )
    data = InteractionData(
        ["Protein1", "Protein2", "Protein3"],
        ["Protein1", "Protein2", "Protein3"],
        Dict(1 => protocol), Dict(1 => protocol),
        1, Dict(1 => 1), 3, 2, [2], [3], [1],
        trues(3)
    )
    dh = compute_data_hash(data)

    # Current H0Cache is the 20-field log-BF struct; validation is keyed on (data_hash, imputation_method).
    cache = H0Cache(
        [1.0, 2.0, 3.0], [0.5, 1.0, 1.5], [0.8, 1.2, 0.9],
        -1.0, 0.5, 0.0, -0.8, 0.4, 0.0, -1.2, 0.6, 0.0,
        0.05, 0.04, 0.06,
        3, dh, now(), "0.1.0", :mnar
    )

    temp_file = tempname() * ".jld2"
    try
        save_h0_cache(cache, temp_file)

        # Cache hit: matching data hash + imputation method
        status, cached = check_h0_cache(temp_file, dh, :mnar)
        @test status == INTERMEDIATE_CACHE_HIT
        @test !isnothing(cached)
        @test cached.log_bf_enrichment == [1.0, 2.0, 3.0]
        @test cached.n_h0_proteins == 3

        # Data-hash mismatch → data miss
        status, _ = check_h0_cache(temp_file, dh + UInt64(1), :mnar)
        @test status == INTERMEDIATE_CACHE_MISS_DATA

        # imputation_method mismatch → params miss
        status, _ = check_h0_cache(temp_file, dh, :none)
        @test status == INTERMEDIATE_CACHE_MISS_PARAMS

        # Non-existent file → no-file miss
        status, _ = check_h0_cache("nonexistent.jld2", dh, :mnar)
        @test status == INTERMEDIATE_CACHE_MISS_NO_FILE
    finally
        isfile(temp_file) && rm(temp_file)
    end
end

@testitem "H0 cache file path generation" begin
    using BayesInteractomics
    using BayesInteractomics: get_h0_cache_filepath

    config = CONFIG(
        datafile = [joinpath(tempdir(), "test_data.xlsx")],
        control_cols = [Dict(1 => [1,2,3])],
        sample_cols = [Dict(1 => [4,5,6])],
        poi = "TestProtein",
        n_controls = 3,
        n_samples = 3,
        refID = 1
    )

    h0_path = get_h0_cache_filepath(config)
    @test occursin(".bayesinteractomics_cache", h0_path)
    @test occursin("h0_", h0_path)
    @test occursin("_ref1", h0_path)
    @test occursin("_robust_t", h0_path)
    @test endswith(h0_path, ".jld2")

    # Changing refID should change the path
    config2 = deepcopy(config)
    config2.refID = 2
    h0_path2 = get_h0_cache_filepath(config2)
    @test h0_path != h0_path2
    @test occursin("_ref2", h0_path2)

    # Changing regression_bf_threshold doesn't change path (validated at check time)
    # but changing regression_likelihood does
    config3 = deepcopy(config)
    config3.regression_likelihood = :normal
    h0_path3 = get_h0_cache_filepath(config3)
    @test h0_path != h0_path3
    @test occursin("_normal", h0_path3)
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

@testitem "Old cache version 7 rejected after nu-bound bump" begin
    using BayesInteractomics
    using JLD2
    using Dates

    # Confirm the version was bumped to 16 (H1 enrichment hard clamps, gate removal)
    @test BayesInteractomics.INTERMEDIATE_CACHE_VERSION == 16

    # Create a cache file written with the old version (7)
    temp_file = tempname() * ".jld2"
    try
        jldsave(temp_file; compress=true,
            cache_version = 7,  # Old version — must be rejected
            bf_detected = [1.0, 2.0],
            protein_ids = ["P1", "P2"],
            n_controls = 3,
            n_samples = 3,
            data_hash = UInt64(456),
            timestamp = string(now()),
            package_version = "0.1.0"
        )

        # load_betabernoulli_cache must return nothing for stale version-7 caches
        loaded = load_betabernoulli_cache(temp_file)
        @test isnothing(loaded)
    finally
        isfile(temp_file) && rm(temp_file)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Cache imputation_method discrimination + coexistence
# Per-cache `imputation_method` field; MICE/MNAR caches coexist.
# ─────────────────────────────────────────────────────────────────────────────

@testitem "Cache hash discriminates by imputation method (BB + HBM)" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, compute_betabernoulli_hash, compute_hbm_regression_hash

    mat = Union{Missing, Float64}[1.0 2.0; 3.0 4.0; 5.0 6.0]
    protocol = Protocol(1, ["Protein1", "Protein2", "Protein3"], Dict(1 => mat))
    data = InteractionData(
        ["Protein1", "Protein2", "Protein3"],
        ["Protein1", "Protein2", "Protein3"],
        Dict(1 => protocol), Dict(1 => protocol),
        1, Dict(1 => 1),
        3, 2,
        [2], [3], [1],
        trues(3),
    )

    # Beta-Bernoulli
    h_mnar = compute_betabernoulli_hash(data, 3, 3, :mnar)
    h_mar  = compute_betabernoulli_hash(data, 3, 3, :mar)
    h_none = compute_betabernoulli_hash(data, 3, 3, :none)
    @test h_mnar != h_mar
    @test h_mnar != h_none
    @test h_mar != h_none

    # HBM regression
    h2_mnar = compute_hbm_regression_hash(data, 1, :robust_t, 5.0, :mnar)
    h2_mar  = compute_hbm_regression_hash(data, 1, :robust_t, 5.0, :mar)
    @test h2_mnar != h2_mar
end

@testitem "Cache schema includes imputation_method field (BB + HBM + H0)" begin
    using BayesInteractomics
    @test :imputation_method in fieldnames(BayesInteractomics.BetaBernoulliCache)
    @test :imputation_method in fieldnames(BayesInteractomics.HBMRegressionCache)
    @test :imputation_method in fieldnames(BayesInteractomics.H0Cache)
end

@testitem "BetaBernoulliCache JLD2 round-trip preserves imputation_method" begin
    using BayesInteractomics
    using Dates

    cache = BayesInteractomics.BetaBernoulliCache(
        [1.0, 2.0, 3.0],          # bf_detected
        ["P1", "P2", "P3"],       # protein_ids
        3,                         # n_controls
        3,                         # n_samples
        UInt64(0xDEADBEEF),       # data_hash
        now(),                     # timestamp
        "1.2.0",                   # package_version
        :mnar,                     # imputation_method
    )

    tmp = tempname() * ".jld2"
    try
        BayesInteractomics.save_betabernoulli_cache(cache, tmp)
        loaded = BayesInteractomics.load_betabernoulli_cache(tmp)

        @test loaded !== nothing
        @test loaded.imputation_method === :mnar
        @test loaded.bf_detected == cache.bf_detected
        @test loaded.protein_ids == cache.protein_ids
    finally
        isfile(tmp) && rm(tmp)
    end
end

@testitem "MICE and MNAR caches coexist on disk under distinct filenames" begin
    using BayesInteractomics
    using Dates

    cache_mnar = BayesInteractomics.BetaBernoulliCache(
        [1.0, 2.0, 3.0], ["P1", "P2", "P3"], 3, 3, UInt64(0x1), now(), "1.2.0", :mnar,
    )
    cache_mar = BayesInteractomics.BetaBernoulliCache(
        [10.0, 20.0, 30.0], ["P1", "P2", "P3"], 3, 3, UInt64(0x1), now(), "1.2.0", :mar,
    )

    tmp = mktempdir()
    fp_mnar = joinpath(tmp, "betabernoulli_xx_mnar.jld2")
    fp_mar  = joinpath(tmp, "betabernoulli_xx_mar.jld2")

    BayesInteractomics.save_betabernoulli_cache(cache_mnar, fp_mnar)
    BayesInteractomics.save_betabernoulli_cache(cache_mar,  fp_mar)

    @test isfile(fp_mnar)
    @test isfile(fp_mar)

    loaded_mnar = BayesInteractomics.load_betabernoulli_cache(fp_mnar)
    loaded_mar  = BayesInteractomics.load_betabernoulli_cache(fp_mar)
    @test loaded_mnar.imputation_method === :mnar
    @test loaded_mar.imputation_method === :mar
    @test loaded_mnar.bf_detected != loaded_mar.bf_detected  # distinct payloads survive
end

@testitem "check_betabernoulli_cache returns INTERMEDIATE_CACHE_MISS_PARAMS when imputation method differs" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, INTERMEDIATE_CACHE_HIT, INTERMEDIATE_CACHE_MISS_PARAMS
    using BayesInteractomics: BetaBernoulliCache, save_betabernoulli_cache, check_betabernoulli_cache, compute_data_hash, getIDs
    using Dates

    # Build minimal InteractionData with a deterministic data hash (avoid load_data complexity)
    mat = Union{Missing, Float64}[1.0 2.0; 3.0 4.0; 5.0 6.0]
    protocol = Protocol(1, ["P1", "P2", "P3"], Dict(1 => mat))
    data = InteractionData(
        ["P1", "P2", "P3"], ["P1", "P2", "P3"],
        Dict(1 => protocol), Dict(1 => protocol),
        1, Dict(1 => 1), 3, 2,
        [2], [3], [1], trues(3),
    )

    tmp = mktempdir()
    fp = joinpath(tmp, "bb_test.jld2")

    # Save a :mar-tagged cache with the same data_hash + protein_ids the check will compute
    cache = BetaBernoulliCache(
        [1.0, 2.0, 3.0],
        getIDs(data),
        3, 3,
        compute_data_hash(data),
        now(), "1.2.0",
        :mar,
    )
    save_betabernoulli_cache(cache, fp)

    # Request :mnar — must miss on params (imputation_method mismatch)
    status, _ = check_betabernoulli_cache(fp, data, 3, 3, :mnar)
    @test status == INTERMEDIATE_CACHE_MISS_PARAMS

    # Request :mar — must hit
    status_hit, cached_hit = check_betabernoulli_cache(fp, data, 3, 3, :mar)
    @test status_hit == INTERMEDIATE_CACHE_HIT
    @test cached_hit.imputation_method === :mar
end
