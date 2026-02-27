@testitem "Latent Class Model" begin
    using BayesInteractomics
    using Random
    using Distributions
    using Statistics

    @testset "Latent class model on synthetic data" begin
        # Set random seed for reproducibility
        Random.seed!(42)

        # Generate synthetic data
        n_bg = 180
        n_int = 20
        n = n_bg + n_int

        # Background proteins: log-BF ≈ 0, moderate variance
        enrich_bg = randn(n_bg) .* 1.2
        corr_bg = randn(n_bg) .* 1.0
        pres_bg = randn(n_bg) .* 1.1

        # Interaction proteins: log-BF clearly positive
        enrich_int = randn(n_int) .* 0.8 .+ 4.0
        corr_int = randn(n_int) .* 0.9 .+ 3.0
        pres_int = randn(n_int) .* 0.7 .+ 3.5

        # Combine (background first, then interactions)
        log_bf_enrich = vcat(enrich_bg, enrich_int)
        log_bf_corr = vcat(corr_bg, corr_int)
        log_bf_pres = vcat(pres_bg, pres_int)

        # Convert to BFs (exp of log-BFs)
        bf_enrich = exp.(log_bf_enrich)
        bf_corr = exp.(log_bf_corr)
        bf_pres = exp.(log_bf_pres)

        # Test prepare_lc_scores returns 6-tuple
        result = BayesInteractomics.prepare_lc_scores(bf_enrich, bf_corr, bf_pres; log_transform=true)

        @test length(result) == 6
        y_e_win, y_c_win, y_p_win, y_e_orig, y_c_orig, y_p_orig = result

        @test length(y_e_win) == n
        @test length(y_c_win) == n
        @test length(y_p_win) == n
        @test length(y_e_orig) == n
        @test length(y_c_orig) == n
        @test length(y_p_orig) == n
        @test all(isfinite.(y_e_win))
        @test all(isfinite.(y_c_win))
        @test all(isfinite.(y_p_win))
        @test all(isfinite.(y_e_orig))

        # Original values should be preserved
        @test y_e_orig ≈ log.(max.(bf_enrich, 1e-300))

        # Winsorized values should be bounded
        @test maximum(y_e_win) <= maximum(y_e_orig)
        @test minimum(y_e_win) >= minimum(y_e_orig)
    end

    @testset "prepare_lc_scores winsorization disabled" begin
        Random.seed!(42)
        n = 100
        bf = exp.(randn(n) .* 2.0)

        # With winsorization disabled, winsorized == original
        result = BayesInteractomics.prepare_lc_scores(bf, bf, bf;
            log_transform=true, winsorize=false)
        y_e_win, _, _, y_e_orig, _, _ = result
        @test y_e_win ≈ y_e_orig
    end

    @testset "prepare_lc_scores custom quantiles" begin
        Random.seed!(42)
        n = 200
        bf = exp.(randn(n) .* 3.0)

        # Narrow quantiles should clip more
        r_narrow = BayesInteractomics.prepare_lc_scores(bf, bf, bf;
            log_transform=true, winsorize=true, winsorize_quantiles=(0.05, 0.95))
        r_wide = BayesInteractomics.prepare_lc_scores(bf, bf, bf;
            log_transform=true, winsorize=true, winsorize_quantiles=(0.01, 0.99))

        y_narrow = r_narrow[1]
        y_wide = r_wide[1]

        # Narrower quantiles should produce a tighter range
        @test (maximum(y_narrow) - minimum(y_narrow)) <= (maximum(y_wide) - minimum(y_wide))
    end

    @testset "combined_BF_latent_class main entry point" begin
        Random.seed!(123)

        # Generate smaller synthetic dataset
        n_bg = 50
        n_int = 10
        n = n_bg + n_int

        # Create BF triplet
        bf_enrich = exp.(vcat(randn(n_bg) .* 1.0, randn(n_int) .* 0.8 .+ 3.5))
        bf_corr = exp.(vcat(randn(n_bg) .* 0.9, randn(n_int) .* 0.7 .+ 2.5))
        bf_detect = exp.(vcat(randn(n_bg) .* 1.1, randn(n_int) .* 0.6 .+ 3.0))

        bf_triplet = BayesInteractomics.BayesFactorTriplet(bf_enrich, bf_corr, bf_detect)

        # Run latent class combination
        lc_result = BayesInteractomics.combined_BF_latent_class(
            bf_triplet, 1;
            n_iterations = 50,
            alpha_prior = [10.0, 1.0],
            convergence_tol = 1e-4,
            verbose = false
        )

        # Check result type
        @test isa(lc_result, BayesInteractomics.LatentClassResult)

        # Check fields
        @test length(lc_result.bf) == n
        @test length(lc_result.posterior_prob) == n
        @test length(lc_result.mixing_weights) == 2
        @test length(lc_result.free_energy) >= 1  # At least one iteration
        @test isa(lc_result.converged, Bool)
        @test lc_result.n_iterations >= 1 && lc_result.n_iterations <= 50

        # Check value ranges
        @test all(lc_result.bf .>= 0)
        @test all(0 .<= lc_result.posterior_prob .<= 1)
        @test sum(lc_result.mixing_weights) ≈ 1.0

        # Check bait protein handling (should have max probability)
        max_prob = maximum(lc_result.posterior_prob)
        @test lc_result.posterior_prob[1] == max_prob

        # Check class parameters
        @test haskey(lc_result.class_parameters, "background")
        @test haskey(lc_result.class_parameters, "interaction")
    end

    @testset "Label ordering constraint" begin
        Random.seed!(789)

        n_bg = 100
        n_int = 20
        n = n_bg + n_int

        bf_enrich = exp.(vcat(randn(n_bg) .* 1.2, randn(n_int) .* 0.8 .+ 4.0))
        bf_corr = exp.(vcat(randn(n_bg) .* 1.0, randn(n_int) .* 0.9 .+ 3.0))
        bf_detect = exp.(vcat(randn(n_bg) .* 1.1, randn(n_int) .* 0.7 .+ 3.5))

        bf_triplet = BayesInteractomics.BayesFactorTriplet(bf_enrich, bf_corr, bf_detect)

        lc_result = BayesInteractomics.combined_BF_latent_class(
            bf_triplet, 1;
            n_iterations = 100,
            verbose = false
        )

        # Interaction class should have higher mean than background
        @test lc_result.class_parameters["interaction"].mu > lc_result.class_parameters["background"].mu
    end

    @testset "Ranking correctness with extreme outliers" begin
        Random.seed!(999)

        n_bg = 150
        n_moderate = 15
        n_extreme = 5
        n = n_bg + n_moderate + n_extreme

        # Background: weak evidence (log-BF ~ 0)
        enrich_bg = randn(n_bg) .* 1.2
        corr_bg = randn(n_bg) .* 1.0
        pres_bg = randn(n_bg) .* 1.1

        # Moderate interactors: clear positive evidence
        enrich_mod = randn(n_moderate) .* 0.8 .+ 4.0
        corr_mod = randn(n_moderate) .* 0.9 .+ 3.0
        pres_mod = randn(n_moderate) .* 0.7 .+ 3.5

        # Extreme outliers: very strong evidence (like TNRC6B)
        enrich_ext = randn(n_extreme) .* 0.5 .+ 12.0
        corr_ext = randn(n_extreme) .* 0.5 .+ 20.0
        pres_ext = randn(n_extreme) .* 0.3 .+ 5.0

        bf_enrich = exp.(vcat(enrich_bg, enrich_mod, enrich_ext))
        bf_corr = exp.(vcat(corr_bg, corr_mod, corr_ext))
        bf_detect = exp.(vcat(pres_bg, pres_mod, pres_ext))

        bf_triplet = BayesInteractomics.BayesFactorTriplet(bf_enrich, bf_corr, bf_detect)

        lc_result = BayesInteractomics.combined_BF_latent_class(
            bf_triplet, 1;  # bait = index 1
            n_iterations = 200,
            verbose = false,
            winsorize = true,
            winsorize_quantiles = (0.01, 0.99)
        )

        # Key test: extreme outlier proteins must get BF > 1 and posterior_prob > 0.5
        extreme_indices = (n_bg + n_moderate + 1):n
        for idx in extreme_indices
            @test lc_result.bf[idx] > 1.0
            @test lc_result.posterior_prob[idx] > 0.5
        end

        # Extreme outliers should rank higher than median background protein
        median_bg_bf = median(lc_result.bf[1:n_bg])
        for idx in extreme_indices
            @test lc_result.bf[idx] > median_bg_bf
        end

        # Moderate interactors should also rank above average background
        moderate_indices = (n_bg + 1):(n_bg + n_moderate)
        mean_moderate_pp = mean(lc_result.posterior_prob[moderate_indices])
        mean_bg_pp = mean(lc_result.posterior_prob[1:n_bg])
        @test mean_moderate_pp > mean_bg_pp
    end

    @testset "plot_lc_convergence" begin
        Random.seed!(456)

        # Create minimal dataset
        n = 30
        bf_enrich = exp.(randn(n) .+ 1.0)
        bf_corr = exp.(randn(n) .+ 0.5)
        bf_detect = exp.(randn(n) .+ 1.5)

        bf_triplet = BayesInteractomics.BayesFactorTriplet(bf_enrich, bf_corr, bf_detect)

        lc_result = BayesInteractomics.combined_BF_latent_class(
            bf_triplet, 1;
            n_iterations = 30,
            verbose = false
        )

        # Test plot generation
        plt = BayesInteractomics.plot_lc_convergence(lc_result)
        @test !isnothing(plt)
    end

    @testset "LatentClassResult show method" begin
        # Create dummy result
        lc_result = BayesInteractomics.LatentClassResult(
            [1.0, 2.0, 3.0],
            [0.5, 0.7, 0.9],
            Dict(
                "background" => (mu=0.0, sigma=1.0, precision=1.0),
                "interaction" => (mu=3.0, sigma=0.5, precision=4.0)
            ),
            [0.9, 0.1],
            [100.0, 90.0, 85.0],
            true,
            3
        )

        # Test show doesn't error
        io = IOBuffer()
        show(io, lc_result)
        output = String(take!(io))

        @test contains(output, "LatentClassResult")
        @test contains(output, "Converged: true")
        @test contains(output, "Iterations: 3")
    end
end
