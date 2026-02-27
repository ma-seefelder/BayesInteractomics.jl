#=
# All standard imports are already loaded by the main module
# This file includes additional functionality for Bayes Factor Data Analysis

using LaTeXStrings
using RxInfer

include("utils.jl")
include("data_loading.jl")
include("model_fitting.jl")
include("model_evaluation.jl")
include("model_visualization.jl")
include("mulitple_imputation.jl")
include("betabernoulli.jl")
include("pca.jl")

modelBetaBernoulli() | (
    x = RxInfer.DeferredDataHandler(),
    y = RxInfer.DeferredDataHandler(),
    n = RxInfer.DeferredDataHandler()
) 


likelihood_g1 = Bernoulli(0.7) # control
likelihood_g2 = Bernoulli(0.5) # treatment

inference_function(x,y, n) = infer(
    model           = modelBetaBernoulli(n = n), 
    data            = (x = x, y = y),
    returnvars      = KeepLast()
)

function data_generator(n, prior = false) 
    prior ? (x = fill(missing, n), y = fill(missing), n) : rand(Bernoulli(0.3), n), rand(Bernoulli(0.7), n), n
end

function bf(posterior, prior_odds = 1.0) 
    θx, θy = posterior.posteriors[:θx], posterior.posteriors[:θy]

    # sample from posterior
    nsamples = 1_000_000
    samples_x = rand(θx, nsamples)
    samples_y = rand(θy, nsamples)

    p = count(samples_x .< samples_y) / nsamples
    posterior_odds =  p / (1 - p)

    return posterior_odds / prior_odds 
end

lbound, ubound, n = 3, 250, 5000
BF_limit, prior_odds = 10.0, 1.0
=#

function sample_size(
    inference_function::Function, data_generator::Function, bf::Function, 
    lbound::I = 3, ubound::I = 50, power = 0.9; 
    n::I = 5000, BF_limit::F = 10.0, prior_odds::F = 1.0, max_iter::I = 20
    ) where {I <: Int, F <: Real}

    nmid = div(lbound + ubound, 2)

    function compute_power(sample_size, n, prior_odds, BF_limit)
        ThreadsX.map(1:n) do _
            bf(inference_function(data_generator(sample_size)...), prior_odds) >= BF_limit
        end |> x -> count(x) / n
    end

    p_low  = compute_power(lbound, n, prior_odds, BF_limit)
    p_high = compute_power(ubound, n, prior_odds, BF_limit)
    p_mid = compute_power(nmid, n, prior_odds, BF_limit)

    if p_low > power 
        return "With $lbound samples, the power is $(p_low * 100)%. The desired power is already achieved at the minimum sample size."
    elseif p_high < power
         return "The desired power is not possible. With $ubound samples, the power is $(p_high * 100)%. The desired power is not possible."
    end


    # create container for results i, powers
    results = DataFrame(n = Int64[], power = Float64[])
    push!(results, [lbound, p_low])
    push!(results, [ubound, p_high])
    push!(results, [nmid, p_mid])

    iter = 1
    while iter ≤ max_iter
        nmid = div(lbound + ubound, 2)
        p_mid = compute_power(nmid, n, prior_odds, BF_limit)
        push!(results, (nmid, p_mid))
        @info "Iteration $iter: n = $nmid, power = $(round(p_mid*100, digits=2))%"
        
        isapprox(p_mid, power; atol = 1e-3) && break
        p_mid ≥ power ? ubound = nmid : lbound = nmid
        iter += 1
    end
        
    sort!(results, [:n])

    plt = plot(
        results.n, results.power,
        xlabel = "Sample Size", ylabel = "Power",
        title = "Power vs Sample Size (BF > $BF_limit)",
        legend = false, xlim = (lbound, ubound), ylim = (0.0, 1.0)
    )    

    return results, plt
end

#x = sample_size(
#    inference_function, data_generator, bf, 3,100, 0.9; 
#    n = 5000, BF_limit = 10.0
#    )

