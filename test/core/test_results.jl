using BayesInteractomics
using Test
using DataFrames
using Dates
using JLD2
using Copulas
using Random
using Distributions

@testset "AnalysisResult Caching System" begin

    # Helper function to create mock AnalysisResult
    function create_mock_result(;
        n_proteins=10,
        config_hash=UInt64(12345),
        data_hash=UInt64(67890),
        timestamp=now(),
        package_version="0.1.0"
    )
        # Create mock copula results
        copula_results = DataFrame(
            Protein = ["Protein$i" for i in 1:n_proteins],
            BF = rand(n_proteins) .* 100,
            posterior_prob = rand(n_proteins),
            q = rand(n_proteins),
            mean_log2FC = randn(n_proteins),
            bf_enrichment = rand(n_proteins) .* 10,
            bf_correlation = rand(n_proteins) .* 10,
            bf_detected = rand(n_proteins) .* 10
        )

        # Create mock hierarchical results
        df_hierarchical = DataFrame(
            Protein = ["Protein$i" for i in 1:n_proteins],
            BF_log2FC = rand(n_proteins) .* 10,
            bf_slope = rand(n_proteins) .* 10,
            mean_log2FC = randn(n_proteins)
        )

        # Create mock EM result
        logs_df = DataFrame(
            iter = 1:10,
            loglikelihood = cumsum(randn(10)),
            π0 = rand(10),
            π1 = rand(10)
        )

        # Create valid SklarDist with marginals matching copula dimension
        marginals = (Normal(0, 1), Normal(0, 1), Normal(0, 1))
        copula = GaussianCopula([1.0 0.5 0.3; 0.5 1.0 0.4; 0.3 0.4 1.0])
        joint_dist = SklarDist(copula, marginals)

        em = BayesInteractomics.EMResult(
            0.3, 0.7,
            joint_dist,
            logs_df,
            true
        )

        # Create mock copula distributions
        joint_H0 = joint_dist
        joint_H1 = joint_dist

        return BayesInteractomics.AnalysisResult(
            copula_results,
            df_hierarchical,
            em,
            joint_H0,
            joint_H1,
            nothing,  # latent_class_result
            :copula,  # combination_method
            nothing,  # em_diagnostics
            nothing,  # em_diagnostics_summary
            config_hash,
            data_hash,
            timestamp,
            package_version,
            nothing,  # bait_protein
            nothing,  # bait_index
            nothing   # sensitivity
        )
    end

    @testset "AnalysisResult Construction" begin
        result = create_mock_result(n_proteins=5)

        @test result isa BayesInteractomics.AnalysisResult
        @test nrow(result.copula_results) == 5
        @test nrow(result.df_hierarchical) == 5
        @test result.em.has_converged == true
        @test result.config_hash == UInt64(12345)
        @test result.data_hash == UInt64(67890)
        @test result.package_version == "0.1.0"
    end

    @testset "Iterator Interface" begin
        result = create_mock_result(n_proteins=5)

        # Test length
        @test length(result) == 5

        # Test iteration
        count = 0
        for (protein, row) in result
            count += 1
            @test protein isa String
            @test row isa DataFrameRow
            @test haskey(row, :BF)
            @test haskey(row, :posterior_prob)
        end
        @test count == 5

        # Test collect
        collected = collect(result)
        @test length(collected) == 5
        @test collected[1][1] == "Protein1"
    end

    @testset "Indexing" begin
        result = create_mock_result(n_proteins=5)

        # Integer indexing
        row1 = result[1]
        @test row1.Protein == "Protein1"
        @test haskey(row1, :BF)

        # String indexing
        row_p3 = result["Protein3"]
        @test row_p3.Protein == "Protein3"

        # Out of bounds
        @test_throws BoundsError result[10]
    end

    @testset "Convenience Accessors" begin
        result = create_mock_result(n_proteins=5)

        proteins = BayesInteractomics.getProteins(result)
        @test proteins == ["Protein$i" for i in 1:5]

        bfs = BayesInteractomics.getBayesFactors(result)
        @test length(bfs) == 5
        @test all(bfs .>= 0)

        probs = BayesInteractomics.getPosteriorProbabilities(result)
        @test length(probs) == 5
        @test all(0 .<= probs .<= 1)

        q_vals = BayesInteractomics.getQValues(result)
        @test length(q_vals) == 5

        # Test new accessors
        log2fc = BayesInteractomics.getMeanLog2FC(result)
        @test length(log2fc) == 5

        bait = BayesInteractomics.getBaitProtein(result)
        @test isnothing(bait)

        # Test alias
        probs_alias = BayesInteractomics.getPosteriorProbs(result)
        @test probs_alias == probs
    end

    @testset "Property Accessor for results" begin
        result = create_mock_result(n_proteins=5)

        # Test that .results returns copula_results
        @test result.results === result.copula_results

        # Test propertynames includes results
        props = propertynames(result)
        @test :results in props
        @test :copula_results in props
    end

    @testset "Abstract Type Hierarchy" begin
        result = create_mock_result(n_proteins=5)

        # Test that AnalysisResult is a subtype of AbstractAnalysisResult
        @test result isa BayesInteractomics.AbstractAnalysisResult
        @test result isa BayesInteractomics.AnalysisResult
    end

    @testset "Bait Protein Fields" begin
        # Create result with bait info
        mock = create_mock_result()
        result_with_bait = BayesInteractomics.AnalysisResult(
            mock.copula_results,
            mock.df_hierarchical,
            mock.em,
            mock.joint_H0,
            mock.joint_H1,
            nothing,      # latent_class_result
            :copula,      # combination_method
            nothing,      # em_diagnostics
            nothing,      # em_diagnostics_summary
            UInt64(12345),
            UInt64(67890),
            now(),
            "0.1.0",
            "MYC",
            1,
            nothing       # sensitivity
        )

        @test result_with_bait.bait_protein == "MYC"
        @test result_with_bait.bait_index == 1
        @test BayesInteractomics.getBaitProtein(result_with_bait) == "MYC"
    end

    @testset "Serialization Round-trip" begin
        mktempdir() do tmpdir
            filepath = joinpath(tmpdir, "test_cache.jld2")
            original = create_mock_result(n_proteins=10)

            # Save
            BayesInteractomics.save_result(original, filepath)
            @test isfile(filepath)

            # Load
            loaded = BayesInteractomics.load_result(filepath)
            @test !isnothing(loaded)

            # Verify fields match
            @test loaded.copula_results == original.copula_results
            @test loaded.df_hierarchical == original.df_hierarchical
            @test loaded.config_hash == original.config_hash
            @test loaded.data_hash == original.data_hash
            @test loaded.timestamp == original.timestamp
            @test loaded.package_version == original.package_version
            @test loaded.em.has_converged == original.em.has_converged
            @test loaded.bait_protein == original.bait_protein
            @test loaded.bait_index == original.bait_index
        end
    end

    @testset "Old Cache Version Rejected" begin
        mktempdir() do tmpdir
            filepath = joinpath(tmpdir, "old_cache.jld2")

            # Save old-style cache with outdated version
            result = create_mock_result(n_proteins=5)
            jldsave(filepath; compress=true,
                cache_version = 1,
                copula_results = result.copula_results,
                df_hierarchical = result.df_hierarchical,
                em = result.em,
                joint_H0 = result.joint_H0,
                joint_H1 = result.joint_H1,
                config_hash = result.config_hash,
                data_hash = result.data_hash,
                timestamp = result.timestamp,
                package_version = result.package_version
            )

            # Old cache version should be rejected
            loaded = BayesInteractomics.load_result(filepath)
            @test isnothing(loaded)
        end
    end

    @testset "Load Invalid Cache" begin
        mktempdir() do tmpdir
            # Non-existent file
            result = BayesInteractomics.load_result(joinpath(tmpdir, "nonexistent.jld2"))
            @test isnothing(result)

            # Invalid cache version
            filepath = joinpath(tmpdir, "invalid_version.jld2")
            jldsave(filepath; cache_version=999, copula_results=DataFrame())
            result = BayesInteractomics.load_result(filepath)
            @test isnothing(result)

            # Corrupted file
            filepath2 = joinpath(tmpdir, "corrupted.jld2")
            write(filepath2, "not a jld2 file")
            result = BayesInteractomics.load_result(filepath2)
            @test isnothing(result)
        end
    end

    @testset "Config Hash Computation" begin
        # Create two configs that differ only in non-computational fields
        config1 = BayesInteractomics.CONFIG(
            datafile = ["data.xlsx"],
            control_cols = [Dict(1 => [1,2,3])],
            sample_cols = [Dict(1 => [4,5,6])],
            poi = "ProteinA",
            normalise_protocols = true,
            H0_file = "H0.xlsx",
            n_controls = 3,
            n_samples = 3,
            refID = 1,
            results_file = "results1.xlsx",
            volcano_file = "volcano1.png"
        )

        config2 = BayesInteractomics.CONFIG(
            datafile = ["data.xlsx"],
            control_cols = [Dict(1 => [1,2,3])],
            sample_cols = [Dict(1 => [4,5,6])],
            poi = "ProteinB",  # Different POI - should NOT affect hash
            normalise_protocols = true,
            H0_file = "H0.xlsx",
            n_controls = 3,
            n_samples = 3,
            refID = 1,
            results_file = "results2.xlsx",  # Different output file
            volcano_file = "volcano2.png"    # Different plot file
        )

        # Hashes should be equal (non-computational fields excluded)
        @test BayesInteractomics.compute_config_hash(config1) ==
              BayesInteractomics.compute_config_hash(config2)

        # Change a computational field
        config3 = BayesInteractomics.CONFIG(
            datafile = ["data.xlsx"],
            control_cols = [Dict(1 => [1,2,3])],
            sample_cols = [Dict(1 => [4,5,6])],
            poi = "ProteinA",
            normalise_protocols = true,
            H0_file = "H0.xlsx",
            n_controls = 4,  # Different n_controls - SHOULD affect hash
            n_samples = 3,
            refID = 1
        )

        @test BayesInteractomics.compute_config_hash(config1) !=
              BayesInteractomics.compute_config_hash(config3)
    end

    @testset "Data Hash Consistency" begin
        # Create minimal mock InteractionData
        function create_mock_data(n_proteins=5, seed=123)
            Random.seed!(seed)

            protein_ids = ["P$i" for i in 1:n_proteins]
            protein_names = ["Protein$i" for i in 1:n_proteins]

            # Create sample and control protocols (matrices must be Union{Missing,Float64})
            sample_data = Dict(1 => Matrix{Union{Missing,Float64}}(randn(n_proteins, 3)))
            control_data = Dict(1 => Matrix{Union{Missing,Float64}}(randn(n_proteins, 3)))

            sample_protocol = BayesInteractomics.Protocol(
                1, protein_ids, sample_data
            )
            control_protocol = BayesInteractomics.Protocol(
                1, protein_ids, control_data
            )

            no_experiments = Dict(1 => 1)
            protocol_pos, exp_pos, matched_pos = BayesInteractomics.getPositions(no_experiments, 3)

            return BayesInteractomics.InteractionData(
                protein_ids, protein_names,
                Dict(1 => sample_protocol),
                Dict(1 => control_protocol),
                1, no_experiments, 3, 3,
                protocol_pos, exp_pos, matched_pos
            )
        end

        # Same seed should produce same hash
        data1 = create_mock_data(5, 123)
        data2 = create_mock_data(5, 123)
        @test BayesInteractomics.compute_data_hash(data1) ==
              BayesInteractomics.compute_data_hash(data2)

        # Different seed should produce different hash
        data3 = create_mock_data(5, 456)
        @test BayesInteractomics.compute_data_hash(data1) !=
              BayesInteractomics.compute_data_hash(data3)
    end

    @testset "Cache Validation" begin
        mktempdir() do tmpdir
            cache_file = joinpath(tmpdir, "cache.jld2")

            config = BayesInteractomics.CONFIG(
                datafile = ["data.xlsx"],
                control_cols = [Dict(1 => [1,2,3])],
                sample_cols = [Dict(1 => [4,5,6])],
                poi = "ProteinA",
                n_controls = 3,
                n_samples = 3,
                refID = 1
            )

            # Create mock data
            protein_ids = ["P1", "P2", "P3"]
            protein_names = ["Protein1", "Protein2", "Protein3"]
            sample_protocol = BayesInteractomics.Protocol(
                1, protein_ids, Dict(1 => Matrix{Union{Missing,Float64}}(randn(3, 3)))
            )
            control_protocol = BayesInteractomics.Protocol(
                1, protein_ids, Dict(1 => Matrix{Union{Missing,Float64}}(randn(3, 3)))
            )
            no_experiments = Dict(1 => 1)
            protocol_pos, exp_pos, matched_pos = BayesInteractomics.getPositions(no_experiments, 3)

            data = BayesInteractomics.InteractionData(
                protein_ids, protein_names,
                Dict(1 => sample_protocol),
                Dict(1 => control_protocol),
                1, no_experiments, 3, 3,
                protocol_pos, exp_pos, matched_pos
            )

            # Test CACHE_MISS_NO_FILE
            status, cached = BayesInteractomics.check_cache(cache_file, config, data)
            @test status == BayesInteractomics.CACHE_MISS_NO_FILE
            @test isnothing(cached)

            # Create and save a result
            result = create_mock_result(
                n_proteins = 3,
                config_hash = BayesInteractomics.compute_config_hash(config),
                data_hash = BayesInteractomics.compute_data_hash(data)
            )
            BayesInteractomics.save_result(result, cache_file)

            # Test CACHE_HIT
            status, cached = BayesInteractomics.check_cache(cache_file, config, data)
            @test status == BayesInteractomics.CACHE_HIT
            @test !isnothing(cached)
            @test cached.config_hash == result.config_hash
            @test cached.data_hash == result.data_hash

            # Test CACHE_MISS_CONFIG (change config)
            config_changed = deepcopy(config)
            config_changed.n_controls = 4
            status, cached = BayesInteractomics.check_cache(cache_file, config_changed, data)
            @test status == BayesInteractomics.CACHE_MISS_CONFIG
            @test isnothing(cached)

            # Test CACHE_MISS_DATA (change data)
            data_changed = BayesInteractomics.InteractionData(
                protein_ids, protein_names,
                Dict(1 => BayesInteractomics.Protocol(1, protein_ids, Dict(1 => Matrix{Union{Missing,Float64}}(randn(3, 3))))),
                Dict(1 => BayesInteractomics.Protocol(1, protein_ids, Dict(1 => Matrix{Union{Missing,Float64}}(randn(3, 3))))),
                1, no_experiments, 3, 3,
                protocol_pos, exp_pos, matched_pos
            )
            status, cached = BayesInteractomics.check_cache(cache_file, config, data_changed)
            @test status == BayesInteractomics.CACHE_MISS_DATA
            @test isnothing(cached)
        end
    end

    @testset "get_cache_filepath" begin
        mktempdir() do tmpdir
            config = BayesInteractomics.CONFIG(
                datafile = [joinpath(tmpdir, "data.xlsx")],
                control_cols = [Dict(1 => [1,2,3])],
                sample_cols = [Dict(1 => [4,5,6])],
                poi = "ProteinA",
                n_controls = 3,
                n_samples = 3,
                refID = 1
            )

            cache_path = BayesInteractomics.get_cache_filepath(config)

            # Check path structure
            @test dirname(cache_path) == joinpath(tmpdir, ".bayesinteractomics_cache")
            @test endswith(basename(cache_path), ".jld2")
            @test occursin("analysis_cache", basename(cache_path))

            # Cache directory should be created
            @test isdir(dirname(cache_path))
        end
    end

    @testset "Display Method" begin
        result = create_mock_result(n_proteins=20)
        io = IOBuffer()
        show(io, result)
        output = String(take!(io))

        @test occursin("AnalysisResult", output)
        @test occursin("Proteins analyzed", output)
        @test occursin("20", output)
        @test occursin("Timestamp", output)
        @test occursin("EM converged", output)
        @test occursin("true", output)
    end
end
