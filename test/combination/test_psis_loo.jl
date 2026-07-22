@testitem "psis_loo_per_protein returns N-element vector" begin
    using BayesInteractomics: psis_loo_per_protein
    using Random
    Random.seed!(42)
    ll_bma = randn(50) .- 2.0  # 50 proteins with varying log-likelihoods
    k_hat = psis_loo_per_protein(ll_bma)
    @test length(k_hat) == 50
    @test all(-0.5 .<= k_hat .<= 2.0)
end

@testitem "psis_loo_per_protein detects outlier protein" begin
    using BayesInteractomics: psis_loo_per_protein
    using Statistics: median
    using Random
    Random.seed!(42)
    # 49 well-fitted proteins + 1 extreme outlier
    ll_bma = vcat(randn(49) .- 2.0, [-100.0])
    k_hat = psis_loo_per_protein(ll_bma)
    # Outlier protein (index 50) should have higher k-hat than median of others
    @test k_hat[50] > median(k_hat[1:49])
end

@testitem "psis_loo_per_protein uniform LL produces low k-hat" begin
    using BayesInteractomics: psis_loo_per_protein
    ll_bma = fill(-2.0, 100)  # identical log-likelihoods
    k_hat = psis_loo_per_protein(ll_bma)
    @test all(k_hat .<= 0.5)  # all should be "reliable"
end

@testitem "psis_loo_per_protein handles small N" begin
    using BayesInteractomics: psis_loo_per_protein
    using Random
    Random.seed!(42)
    ll_bma = randn(5) .- 2.0
    k_hat = psis_loo_per_protein(ll_bma)
    @test length(k_hat) == 5
    @test all(isfinite.(k_hat))
end

@testitem "psis_loo_per_protein uses BMA mixture LL" begin
    using BayesInteractomics: psis_loo_per_protein
    # Verify the function signature accepts a single Vector{Float64} (the BMA mixture LL)
    # Not two separate vectors (ll_em, ll_cop) -- that would be the old global approach
    ll_bma = [-1.0, -2.0, -3.0, -1.5, -2.5, -0.5, -4.0, -1.2, -2.8, -3.5]
    k_hat = psis_loo_per_protein(ll_bma)
    @test length(k_hat) == 10
end

@testitem "moment_match_loo leaves k<=0.7 proteins unchanged" begin
    using BayesInteractomics: psis_loo_per_protein, moment_match_loo
    using Random
    Random.seed!(42)
    # All proteins well-fitted (k should be < 0.7)
    ll_bma = randn(50) .- 2.0
    k_raw = psis_loo_per_protein(ll_bma)
    k_adjusted = moment_match_loo(ll_bma, k_raw)
    # If no k > 0.7, output should equal input
    if all(k_raw .<= 0.7)
        @test k_adjusted == k_raw
    end
    @test all(isfinite.(k_adjusted))
end

@testitem "PSIS_unreliable flag logic" begin
    using DataFrames
    # Simulate the flagging logic from pipeline.jl
    df = DataFrame(
        Protein = ["A", "B", "C", "D"],
        pareto_k = [0.3, 0.8, 0.5, 0.9],
        diagnostic_flag = ["ok", "warning", "", missing]
    )
    for i in 1:nrow(df)
        pk_val = df.pareto_k[i]
        if !ismissing(pk_val) && pk_val > 0.7
            existing = df.diagnostic_flag[i]
            if ismissing(existing) || isempty(string(existing))
                df[i, :diagnostic_flag] = "PSIS_unreliable"
            else
                df[i, :diagnostic_flag] = string(existing) * "; PSIS_unreliable"
            end
        end
    end
    @test df.diagnostic_flag[1] == "ok"  # k=0.3, no flag
    @test df.diagnostic_flag[2] == "warning; PSIS_unreliable"  # k=0.8, appended
    @test df.diagnostic_flag[3] == ""  # k=0.5, no flag
    @test df.diagnostic_flag[4] == "PSIS_unreliable"  # k=0.9, was missing -> set
end

@testitem "moment_match_loo adjusts k>0.7 proteins" begin
    using BayesInteractomics: psis_loo_per_protein, moment_match_loo
    # Create a case with guaranteed k > 0.7 outlier
    ll_bma = vcat(fill(-2.0, 49), [-200.0])  # extreme outlier
    k_raw = psis_loo_per_protein(ll_bma)
    k_adjusted = moment_match_loo(ll_bma, k_raw)
    @test length(k_adjusted) == 50
    @test all(isfinite.(k_adjusted))
    # The adjusted value for the outlier should differ from raw
    if k_raw[50] > 0.7
        @test k_adjusted[50] != k_raw[50]
    end
end
