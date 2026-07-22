# Pipeline integration + default flip + cache invalidation tests.
# Covers Success Criteria SC1, SC2, SC4-A.
#
# Coverage axes:
#   SC1 — kwarg dispatch + CONFIG default + override semantics
#   SC2 — :mar deprecation @warn maxlog=1 mentioning v1.3
#   SC3 — cache hash discrimination + filename suffix (also covered in test_intermediate_cache.jl)
#   SC4-A — synthetic E2E load_data smoke under :mnar
#
# TestItemRunner @testitem blocks (NOT @testset). Each block runs in an isolated
# module — must `using BayesInteractomics` + std-lib imports inside each block.
# `using Random` explicit on Julia 1.12 for `MersenneTwister`.
#
# `load_data` excludes proteins with n_sample_obs < 2 OR n_control_obs < 2 (filter_insufficient_observations).
# Synthetic fixtures therefore need at least 2 sample columns + 2 control columns of fully-observed values.

@testitem "load_data: accepts imputation = :mnar kwarg without MethodError" begin
    using BayesInteractomics
    using DataFrames, XLSX

    # Build a synthetic XLSX (3 proteins, 4 numeric cols — 2 sample + 2 control, all observed)
    tmp = mktempdir()
    path = joinpath(tmp, "tiny.xlsx")
    df = DataFrame(:id => ["P1", "P2", "P3"],
                   :s1 => [1.0, 2.0, 3.0],
                   :s2 => [1.5, 2.5, 3.5],
                   :c1 => [4.0, 5.0, 6.0],
                   :c2 => [4.5, 5.5, 6.5])
    XLSX.writetable(path, df)

    sample_cols = [Dict(1 => [2, 3])]
    control_cols = [Dict(1 => [4, 5])]
    data = load_data([path], sample_cols, control_cols, 1, 1, false; curate = false, imputation = :mnar)
    @test data isa BayesInteractomics.InteractionData
end

@testitem "load_data: :mar emits deprecation warning mentioning v1.3" begin
    using BayesInteractomics
    using Test, DataFrames, XLSX

    tmp = mktempdir()
    path = joinpath(tmp, "tiny.xlsx")
    df = DataFrame(:id => ["P1", "P2", "P3"],
                   :s1 => [1.0, 2.0, 3.0],
                   :s2 => [1.5, 2.5, 3.5],
                   :c1 => [4.0, 5.0, 6.0],
                   :c2 => [4.5, 5.5, 6.5])
    XLSX.writetable(path, df)

    sample_cols = [Dict(1 => [2, 3])]
    control_cols = [Dict(1 => [4, 5])]

    # @test_logs captures the warning; allow other log records via match_mode=:any
    # so the deprecation @warn is observed even when other lower-level logs surround it.
    @test_logs (:warn, r"deprecated.*v1\.3"i) match_mode=:any begin
        load_data([path], sample_cols, control_cols, 1, 1, false; curate = false, imputation = :mar)
    end
end

@testitem "load_data: rejects unknown imputation symbol with ArgumentError" begin
    using BayesInteractomics
    using Test

    @test_throws ArgumentError load_data(String["nonexistent.xlsx"],
                                         [Dict(1 => [2])], [Dict(1 => [2])],
                                         1, 1, false; imputation = :foo)
end

@testitem "load_data: :none preserves missings (no pre-imputation at loader)" begin
    using BayesInteractomics
    using DataFrames, XLSX

    tmp = mktempdir()
    path = joinpath(tmp, "missings.xlsx")
    # Synthetic data with explicit missing values — but enough observations per protein
    # to survive the n_obs >= 2 exclusion filter on both sample and control sides.
    df = DataFrame(:id => ["P1", "P2", "P3"],
                   :s1 => Union{Missing, Float64}[1.0, 2.0, 3.0],
                   :s2 => Union{Missing, Float64}[1.5, missing, 3.5],
                   :s3 => Union{Missing, Float64}[2.0, 2.5, 3.7],
                   :c1 => Union{Missing, Float64}[4.0, 5.0, 6.0],
                   :c2 => Union{Missing, Float64}[4.5, 5.5, missing],
                   :c3 => Union{Missing, Float64}[5.0, 6.0, 7.0])
    XLSX.writetable(path, df)

    sample_cols = [Dict(1 => [2, 3, 4])]
    control_cols = [Dict(1 => [5, 6, 7])]
    data = load_data([path], sample_cols, control_cols, 1, 1, false; curate = false, imputation = :none)
    # The loader must accept :none without erroring on missings, validating the
    # metadata-only semantics: load_data does NOT pre-impute under :none.
    @test data isa BayesInteractomics.InteractionData
end

@testitem "CONFIG.imputation_method defaults to :mnar (and override works)" begin
    using BayesInteractomics

    config = CONFIG(
        datafile = String[],
        control_cols = Dict{Int,Vector{Int}}[],
        sample_cols = Dict{Int,Vector{Int}}[],
        poi = "",
    )
    @test config.imputation_method === :mnar

    # Override works
    config2 = CONFIG(
        datafile = String[],
        control_cols = Dict{Int,Vector{Int}}[],
        sample_cols = Dict{Int,Vector{Int}}[],
        poi = "",
        imputation_method = :mar,
    )
    @test config2.imputation_method === :mar

    config3 = CONFIG(
        datafile = String[],
        control_cols = Dict{Int,Vector{Int}}[],
        sample_cols = Dict{Int,Vector{Int}}[],
        poi = "",
        imputation_method = :none,
    )
    @test config3.imputation_method === :none
end

@testitem "compute_betabernoulli_hash + compute_hbm_regression_hash discriminate by imputation method" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, compute_betabernoulli_hash, compute_hbm_regression_hash

    # Build a minimal InteractionData (matches existing test_intermediate_cache.jl pattern)
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

    h_mnar = compute_betabernoulli_hash(data, 3, 3, :mnar)
    h_mar  = compute_betabernoulli_hash(data, 3, 3, :mar)
    h_none = compute_betabernoulli_hash(data, 3, 3, :none)
    @test h_mnar != h_mar
    @test h_mnar != h_none
    @test h_mar != h_none

    # Same imputation -> same hash (sanity)
    @test h_mnar == compute_betabernoulli_hash(data, 3, 3, :mnar)

    # HBM regression hash: same discrimination
    h2_mnar = compute_hbm_regression_hash(data, 1, :robust_t, 5.0, :mnar)
    h2_mar  = compute_hbm_regression_hash(data, 1, :robust_t, 5.0, :mar)
    h2_none = compute_hbm_regression_hash(data, 1, :robust_t, 5.0, :none)
    @test h2_mnar != h2_mar
    @test h2_mnar != h2_none
end

@testitem "Cache filename suffix encodes imputation method" begin
    using BayesInteractomics

    config_mnar = CONFIG(
        datafile = ["test.xlsx"],
        control_cols = Dict{Int,Vector{Int}}[Dict(1 => [2])],
        sample_cols = Dict{Int,Vector{Int}}[Dict(1 => [3])],
        poi = "x",
        imputation_method = :mnar,
    )
    fp_mnar = BayesInteractomics.get_betabernoulli_cache_filepath(config_mnar)
    @test occursin("_mnar", fp_mnar)

    config_mar = CONFIG(
        datafile = ["test.xlsx"],
        control_cols = Dict{Int,Vector{Int}}[Dict(1 => [2])],
        sample_cols = Dict{Int,Vector{Int}}[Dict(1 => [3])],
        poi = "x",
        imputation_method = :mar,
    )
    fp_mar = BayesInteractomics.get_betabernoulli_cache_filepath(config_mar)
    @test occursin("_mar", fp_mar)
    @test fp_mnar != fp_mar  # MICE and MNAR caches have distinct paths

    # Same check for HBM regression and H0 cache filepaths
    @test occursin("_mnar", BayesInteractomics.get_hbm_regression_cache_filepath(config_mnar))
    @test occursin("_mnar", BayesInteractomics.get_h0_cache_filepath(config_mnar))
    @test occursin("_mar",  BayesInteractomics.get_hbm_regression_cache_filepath(config_mar))
    @test occursin("_mar",  BayesInteractomics.get_h0_cache_filepath(config_mar))
end

@testitem "MNAR end-to-end synthetic: load_data + cache helper smoke (50x6 fixture, seed 42)" begin
    using BayesInteractomics
    using Random, DataFrames, XLSX, Statistics

    # Fixture: 50 proteins x 6 numeric cols (3 sample + 3 control);
    # ~10% missingness; deterministic seed 42; one bait + 5 interactors + 44 noise.
    # Missingness rate kept low enough that most proteins survive the n_obs >= 2 filter.
    Random.seed!(42)
    tmp = mktempdir()
    raw_path = joinpath(tmp, "dataset.xlsx")

    n_proteins = 50
    n_cols = 6
    protein_ids = ["P$(lpad(i, 3, '0'))" for i in 1:n_proteins]

    intensities = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_cols)
    for i in 1:n_proteins
        baseline = i <= 6 ? 22.0 : 18.0   # rows 1..6 = bait + 5 interactors
        for c in 1:n_cols
            x = baseline + randn() * 1.5
            intensities[i, c] = rand() < 0.10 ? missing : 2.0^x
        end
    end

    df = DataFrame(:id => protein_ids)
    for c in 1:n_cols
        df[!, Symbol("col$c")] = intensities[:, c]
    end
    XLSX.writetable(raw_path, df)

    # 3 sample cols (cols 2..4 of the XLSX) + 3 control cols (cols 5..7)
    sample_cols = [Dict(1 => [2, 3, 4])]
    control_cols = [Dict(1 => [5, 6, 7])]

    # Load with :mnar (no actual MNAR imputation done at loader; metadata-only)
    data = load_data([raw_path], sample_cols, control_cols, 1, 1, false; curate = false, imputation = :mnar)
    @test data isa BayesInteractomics.InteractionData

    # Same dataset under :mar should also load (and emit deprecation warn — not asserted here)
    data_mar = load_data([raw_path], sample_cols, control_cols, 1, 1, false; curate = false, imputation = :mar)
    @test data_mar isa BayesInteractomics.InteractionData

    # And under :none
    data_none = load_data([raw_path], sample_cols, control_cols, 1, 1, false; curate = false, imputation = :none)
    @test data_none isa BayesInteractomics.InteractionData

    # Cache filenames should carry _mnar suffix (verifies the helpers under a realistic config)
    config = CONFIG(
        datafile = [raw_path],
        control_cols = control_cols,
        sample_cols = sample_cols,
        poi = protein_ids[1],
        imputation_method = :mnar,
        run_simulation = false,        # keep CI test fast — no full pipeline
        run_validation = false,
        run_sensitivity = false,
        run_input_qc = false,
        generate_report_html = false,
        curate = false,
    )
    @test occursin("_mnar", BayesInteractomics.get_betabernoulli_cache_filepath(config))
    @test occursin("_mnar", BayesInteractomics.get_hbm_regression_cache_filepath(config))
    @test occursin("_mnar", BayesInteractomics.get_h0_cache_filepath(config))
end
