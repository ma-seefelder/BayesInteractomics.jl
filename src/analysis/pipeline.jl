# Additional imports needed for analysis pipeline
import DataFrames: innerjoin, leftjoin
import Flux
using LogExpFunctions

# Note: DNN and metalearner functionality will be added separately
# include("../dnn/generate_dataset.jl")
# include("../dnn/model.jl")
# include("../ml/metalearner.jl")
# include("../visualization/plotting.jl")

    """
    Updates the posterior probability in the dataframe based on the meta-learner predictions.

    Args:
        df: DataFrame with the original results.
        df_meta: DataFrame with the meta-learner predictions.

    Returns:
        DataFrame with the updated posterior probability.
    """
function update_posterior_prob!(df::AbstractDataFrame, df_meta::AbstractDataFrame)
    if ("Protein" ∉ names(df_meta))
        rename!(df_meta, :preferred_name => :Protein)
    end
    
    df_meta = df_meta[:, 2:end]
    # join DataFrames
    df = leftjoin(df, df_meta, on = :Protein)
    # find proteins with BF = 0
    bf_is_zero = findall(x -> x == 0, df.BF)
    # update posterior probability
    prior_odds = df.MetaClassifier ./ (1 .- df.MetaClassifier)
    posterior_odds = prior_odds .* df.BF
    df.posterior_prob = posterior_odds ./ (1 .+ posterior_odds)

    df.posterior_prob[bf_is_zero] .= 0
    # sort dataframe by decreasing posterior probability
    df = df[sortperm(df.posterior_prob, rev = true), :]
    return df
end

"""
    analyse(data, H0_file="copula_H0.xlsx"; kwargs...)

Performs the main Bayesian analysis pipeline on the provided proteomics data.

This function integrates results from three different models:
1. A Beta-Bernoulli model for detection probabilities.
2. A hierarchical Bayesian model for protein enrichment (log2 fold change).
3. A Bayesian linear regression model for dose-response correlation.

The Bayes factors from these models are combined using a copula to calculate a final,
joint Bayes factor and posterior probability for each protein. The analysis is
parallelized across proteins.

# Arguments
- `data`: The input data, typically from `load_data`, containing protein quantification data.
- `H0_file::String`: Path to the H0 file containing precomputed Bayes factors for the null hypothesis. If the file does not exist, it will be computed.

# Keywords
## Basic Analysis Parameters
- `n_controls::Int=0`: Number of controls in the dataset.
- `n_samples::Int=0`: Number of samples in the dataset.
- `refID::Int=1`: The reference ID for the main analysis function, typically referring to a reference condition.
- `plotHBMdists::Bool=false`: If `true`, generates and saves plots of the hierarchical Bayesian model distributions.
- `plotlog2fc::Bool=false`: If `true`, generates and saves plots of the log2 fold changes.
- `plotregr::Bool=false`: If `true`, generates and saves plots of the regression model.
- `plotbayesrange::Bool=false`: If `true`, generates and saves plots of the Bayes factor ranges.
- `verbose::Bool=false`: If `true`, prints detailed progress and debugging information.

## Caching Parameters
- `temp_result_file::String="temp_results.xlsx"`: Path for temporary results file.
- `use_intermediate_cache::Bool=true`: If `true`, enables caching of intermediate results (Beta-Bernoulli and HBM+Regression).
- `betabernoulli_cache_file::String=""`: Path to cache file for Beta-Bernoulli results. Empty string disables caching for this step.
- `hbm_regression_cache_file::String=""`: Path to cache file for HBM and regression results. Empty string disables caching for this step.

## Copula-EM Parameters
- `prior::Union{Symbol, NamedTuple}=:default`: Prior specification for the EM algorithm. Use `:default` for automatic prior or provide a NamedTuple with `(α=..., β=...)` for custom Beta prior parameters.
- `n_restarts::Int=20`: Number of random restarts for the EM algorithm to avoid local optima.
- `copula_criterion::Symbol=:BIC`: Model selection criterion for copula fitting. Options: `:BIC`, `:AIC`.
- `h1_refitting::Bool=true`: If `true`, refits the H1 (alternative hypothesis) distribution after EM convergence.
- `burn_in::Int=10`: Number of initial EM iterations to discard before convergence checking.

## Diagnostics Parameters
- `run_em_diagnostics::Bool=true`: If `true` and `n_restarts > 1`, runs diagnostic analysis of EM restart stability and convergence.

# Returns
- `NamedTuple`: A named tuple containing the analysis results with the following fields:
    - `copula_results::DataFrame`: DataFrame with combined Bayes factors, posterior probabilities, q-values, and other key metrics for each protein.
    - `df_hierarchical::DataFrame`: DataFrame with detailed results from the hierarchical and regression models.
    - `convergence_plt`: A plot diagnosing the convergence of the EM algorithm.
    - `em`: The fitted Expectation-Maximization model object.
    - `joint_H0`: The estimated joint distribution under the null hypothesis (H0).
    - `joint_H1`: The estimated joint distribution under the alternative hypothesis (H1).
    - `em_diagnostics`: Detailed diagnostics from EM restart analysis (if `run_em_diagnostics=true` and `n_restarts > 1`).
    - `em_diagnostics_summary`: Summary statistics of EM diagnostics (if `run_em_diagnostics=true` and `n_restarts > 1`).
"""
function analyse(
    data, H0_file = "copula_H0.xlsx";
    n_controls = 0, n_samples = 0, refID = 1,
    plotHBMdists = false, plotlog2fc = false, plotregr = false,
    plotbayesrange = false,
    verbose = false,
    temp_result_file = "temp_results.xlsx",
    use_intermediate_cache::Bool = true,
    betabernoulli_cache_file::String = "",
    hbm_regression_cache_file::String = "",
    # Copula-EM parameters
    prior::Union{Symbol, NamedTuple} = :default,
    n_restarts::Int = 20,
    copula_criterion::Symbol = :BIC,
    h1_refitting::Bool = true,
    burn_in::Int = 10,
    # Diagnostics
    run_em_diagnostics::Bool = true,
    # Evidence combination method
    combination_method::Symbol = :bma,
    # Latent class parameters
    lc_n_iterations::Int = 100,
    lc_alpha_prior::Vector{Float64} = [10.0, 1.0],
    lc_convergence_tol::Float64 = 1e-6,
    lc_winsorize::Bool = true,
    lc_winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99),
    # Robust regression parameters
    regression_likelihood::Symbol = :normal,
    student_t_nu::Float64 = 5.0
)

    # generate cache folder
    ispath("cache") && rm("cache", recursive = true)
    mkpath("cache")
    # get number of proteins
    n_proteins = length(getIDs(data))

    # load H0 file or recompute it (skip for latent_class-only mode; copula and bma need it)
    if combination_method in (:copula, :bma)
        if isnothing(H0_file) || !isfile(H0_file)
            H0 = computeH0_BayesFactors(
                data,
                n_controls = n_controls, n_samples = n_samples,
                refID = refID, savefile = H0_file,
                regression_likelihood = regression_likelihood,
                student_t_nu = student_t_nu
                )
        else
            H0 = DataFrame(readtable(H0_file, "Sheet1", first_row = 1))
        end
    end

    # ------------------------------------ #
    # Beta-Bernoulli model
    # ------------------------------------ #
    bf_detected = zeros(Float64, n_proteins)
    bb_cache_used = false

    # Check Beta-Bernoulli cache
    if use_intermediate_cache && !isempty(betabernoulli_cache_file)
        bb_status, bb_cached = check_betabernoulli_cache(betabernoulli_cache_file, data, n_controls, n_samples)
        if bb_status == INTERMEDIATE_CACHE_HIT
            bf_detected = bb_cached.bf_detected
            bb_cache_used = true
        end
    end

    # Compute if not cached
    if !bb_cache_used
        p = Progress(
            n_proteins, desc="Step 1: Computing Beta-Bernoulli Bayes factors...",
            showspeed=true,
            barglyphs=BarGlyphs('|','█', [' ' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|'),
            barlen = 20
            )

        Threads.@threads for i in 1:n_proteins
            b, _, _ = betabernoulli(data, i, n_controls, n_samples)
            ismissing(b) ? (bf_detected[i]) = 0.0 : (bf_detected[i] = b)
            ProgressMeter.next!(p)
        end
        finish!(p)

        # Save to cache
        if use_intermediate_cache && !isempty(betabernoulli_cache_file)
            try
                bb_cache = BetaBernoulliCache(
                    bf_detected,
                    getIDs(data),
                    n_controls,
                    n_samples,
                    compute_data_hash(data),
                    now(),
                    string(pkgversion(@__MODULE__))
                )
                save_betabernoulli_cache(bb_cache, betabernoulli_cache_file)
                @info "Saved Beta-Bernoulli results to cache: $betabernoulli_cache_file"
            catch e
                @warn "Failed to save Beta-Bernoulli cache: $e"
            end
        end
    end

    # ------------------------------------ #
    # hierarchical & regression model
    # ------------------------------------ #
    df = nothing
    bf_enrichment = Float64[]
    bf_correlation = Float64[]
    hbm_cache_used = false

    # Check HBM+Regression cache
    if use_intermediate_cache && !isempty(hbm_regression_cache_file)
        hbm_status, hbm_cached = check_hbm_regression_cache(hbm_regression_cache_file, data, refID, regression_likelihood, student_t_nu)
        if hbm_status == INTERMEDIATE_CACHE_HIT
            df = hbm_cached.df_hierarchical
            bf_enrichment = hbm_cached.bf_enrichment
            bf_correlation = hbm_cached.bf_correlation
            hbm_cache_used = true
        end
    end

    # Compute if not cached
    if !hbm_cache_used
        τ_dist = τ0(data)
        a_0, b_0 = τ_dist.α, τ_dist.θ
        μ_0, σ_0 = μ0(data)

        # Precompute priors once (they only depend on hyperparameters, not individual proteins)
        @info "Precomputing prior distributions..."
        # Compute τ_base for robust regression (Empirical Bayes)
        robust_tau_base = NaN
        if regression_likelihood == :robust_t
            robust_tau_base = estimate_regression_tau_base(data, refID)
            @info "Estimated τ_base = $(round(robust_tau_base, digits=4)) for robust regression"
        end

        if getNoProtocols(data) == 1
            cached_hbm_prior = precompute_HBM_single_protocol_prior(data, μ_0=μ_0, σ_0=σ_0, a_0=a_0, b_0=b_0)
            if regression_likelihood == :robust_t
                cached_regression_prior = precompute_regression_one_protocol_robust_prior(data, refID, μ_0, σ_0; nu=student_t_nu, τ_base=robust_tau_base)
            else
                cached_regression_prior = precompute_regression_one_protocol_prior(data, refID, μ_0, σ_0)
            end
        else
            cached_hbm_prior = precompute_HBM_prior(data, μ_0=μ_0, σ_0=σ_0, a_0=a_0, b_0=b_0)
            if regression_likelihood == :robust_t
                cached_regression_prior = precompute_regression_multi_protocol_robust_prior(data, refID, μ_0, σ_0; nu=student_t_nu, τ_base=robust_tau_base)
            else
                cached_regression_prior = precompute_regression_multi_protocol_prior(data, refID, μ_0, σ_0)
            end
        end

        p = Progress(
            n_proteins, desc="Step 2: Computing hierarchical and regression Bayes factors...",
            showspeed=true,
            barglyphs=BarGlyphs('|','█', [' ' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen = 20, dt = 10
            )

        Threads.@threads for i in 1:n_proteins
            try
                _ = main(
                    data, i, refID,
                    plotHBMdists = plotHBMdists, plotlog2fc = plotlog2fc,
                    plotregr = plotregr, plotbayesrange = plotbayesrange,
                    csv_file = "cache/results_$(Threads.threadid()).csv", writecsv = true,
                    verbose = verbose, computeHBM = true,
                    μ_0 = μ_0, σ_0 = σ_0, a_0 = a_0, b_0 = b_0,
                    cached_hbm_prior = cached_hbm_prior,
                    cached_regression_prior = cached_regression_prior,
                    regression_likelihood = regression_likelihood,
                    student_t_nu = student_t_nu,
                    robust_tau_base = robust_tau_base
                    )
            catch e
                open("log.txt", "a") do f
                    println(f, "Error in protein $i: $e")
                end
            end
            ProgressMeter.next!(p)
        end
        finish!(p)

        @info "Finished computing hierarchical and regression Bayes factors"
        @info "Errors are logged in log.txt"
        lst_files = string.(readdir("cache", join = true))
        lst_files = filter(x -> occursin(".csv", x), lst_files)
        convert_to_xlsx_filename(file) = replace(file, ".csv" => "_hierarchical.xlsx")

        df = [clean_result(data, file, convert_to_xlsx_filename(file)) for file in lst_files]
        df = reduce(vcat, df)

        # remove all proteins where no BF could be computed
        protein_names = getIDs(data)
        protein_in_dataset = [protein_names[i] in df.Protein for i ∈ 1:length(protein_names)]
        protein_names = protein_names[protein_in_dataset]

        # sort dataframe
        new_order = [findfirst(x -> x == p, df.Protein) for p ∈ protein_names]
        df = df[new_order, :]

        # add BF for detection to the dataframe
        bf_detection = DataFrame(Protein = getIDs(data), bf_detected = bf_detected)
        df = innerjoin(df, bf_detection, on = :Protein)

        # parse and convert to Float64 vectors
        bf_enrichment = Vector{Float64}(undef, size(df, 1))
        bf_correlation = Vector{Float64}(undef, size(df, 1))

        for i in 1:size(df,1)
            # Handle BF_log2FC
            if typeof(df.BF_log2FC[i]) == Float64
                bf_enrichment[i] = df.BF_log2FC[i]
            else
                bf_enrichment[i] = (df.BF_log2FC[i] == "NA" || ismissing(df.BF_log2FC[i])) ? 0.0 : parse(Float64, String(df.BF_log2FC[i]))
            end

            # Handle bf_slope
            if typeof(df.bf_slope[i]) == Float64
                bf_correlation[i] = df.bf_slope[i]
            else
                bf_correlation[i] = (df.bf_slope[i] == "NA" || ismissing(df.bf_slope[i])) ? 0.0 : parse(Float64, String(df.bf_slope[i]))
            end
        end

        # Save to cache
        if use_intermediate_cache && !isempty(hbm_regression_cache_file)
            try
                hbm_cache = HBMRegressionCache(
                    df,
                    bf_enrichment,
                    bf_correlation,
                    getIDs(data),
                    refID,
                    regression_likelihood,
                    student_t_nu,
                    compute_data_hash(data),
                    now(),
                    string(pkgversion(@__MODULE__))
                )
                save_hbm_regression_cache(hbm_cache, hbm_regression_cache_file)
                @info "Saved HBM+Regression results to cache: $hbm_regression_cache_file"
            catch e
                @warn "Failed to save HBM+Regression cache: $e"
            end
        end
    end

    # Update bf_detected from df (either from cache or freshly computed)
    bf_detected = Float64.(df.bf_detected)

    # ------------------------------------ #
    # Evidence combination
    # ------------------------------------ #
    combined_bf = Float64[]
    posterior_prob = Float64[]
    convergence_plt = nothing
    em_result = nothing
    joint_H0 = nothing
    joint_H1 = nothing
    latent_class_result = nothing
    bma_result = nothing
    em_diagnostics = nothing
    em_diagnostics_summary = nothing

    bf_triplet = BayesFactorTriplet(bf_enrichment, bf_correlation, bf_detected)

    if combination_method == :copula
        @info "Starting copula-EM combination"

        combinedResult = combined_BF(
            bf_triplet, refID,
            max_iter = 5_000, H0_file = H0_file,
            prior = prior,
            n_restarts = n_restarts,
            copula_criterion = copula_criterion,
            h1_refitting = h1_refitting,
            burn_in = burn_in,
            verbose = verbose
        )

        combined_bf = combinedResult.bf
        posterior_prob = combinedResult.posterior_prob
        em_result = combinedResult.em_result
        joint_H0 = combinedResult.joint_H0
        joint_H1 = combinedResult.joint_H1
        convergence_plt = EMconvergenceDiagnosticPlot(em_result)

        # Process EM diagnostics
        em_diagnostics = combinedResult.em_diagnostics
        if run_em_diagnostics && !isnothing(em_diagnostics)
            @info "Summarizing EM restart diagnostics..."
            em_diagnostics_summary = summarize_em_diagnostics(em_diagnostics)
        end

    elseif combination_method == :latent_class
        @info "Starting latent class (VMP) combination"

        latent_class_result = combined_BF_latent_class(
            bf_triplet, refID,
            n_iterations = lc_n_iterations,
            alpha_prior = lc_alpha_prior,
            convergence_tol = lc_convergence_tol,
            verbose = verbose,
            winsorize = lc_winsorize,
            winsorize_quantiles = lc_winsorize_quantiles
        )

        combined_bf = latent_class_result.bf
        posterior_prob = latent_class_result.posterior_prob
        convergence_plt = plot_lc_convergence(latent_class_result)

    elseif combination_method == :bma
        @info "Starting Bayesian Model Averaging (BMA) combination"

        bma_result = combined_BF_bma(
            bf_triplet, refID;
            H0_file = H0_file,
            prior = prior,
            n_restarts = n_restarts,
            copula_criterion = copula_criterion,
            h1_refitting = h1_refitting,
            burn_in = burn_in,
            lc_n_iterations = lc_n_iterations,
            lc_alpha_prior = lc_alpha_prior,
            lc_convergence_tol = lc_convergence_tol,
            lc_winsorize = lc_winsorize,
            lc_winsorize_quantiles = lc_winsorize_quantiles,
            verbose = verbose
        )

        combined_bf = bma_result.bf
        posterior_prob = bma_result.posterior_prob
        # Store sub-model results for caching
        em_result = bma_result.copula_result.em_result
        joint_H0 = bma_result.copula_result.joint_H0
        joint_H1 = bma_result.copula_result.joint_H1
        latent_class_result = bma_result.latent_class_result
        em_diagnostics = bma_result.copula_result.em_diagnostics
        if run_em_diagnostics && !isnothing(em_diagnostics)
            @info "Summarizing EM restart diagnostics..."
            em_diagnostics_summary = summarize_em_diagnostics(em_diagnostics)
        end
        convergence_plt = EMconvergenceDiagnosticPlot(em_result)

    else
        error("Unknown combination_method: $combination_method. Must be :copula, :latent_class, or :bma")
    end

    q_values = q(combined_bf)

    # ------------------------------------ #
    # generate output file
    # ------------------------------------ #
    copula_df = DataFrame(
        Protein = df.Protein,
        BF = combined_bf,
        posterior_prob = posterior_prob,
        q = q_values,
        mean_log2FC = df.mean_log2FC,
        bf_enrichment = bf_enrichment,
        bf_correlation = bf_correlation,
        bf_detected = bf_detected
    )

    writetable(
        temp_result_file,
        "hierarchical" => df,
        "copula" => copula_df
        )

    # delete logs
    rm("cache", recursive = true)


    return (
        copula_results      = copula_df,
        df_hierarchical     = df,
        convergence_plt     = convergence_plt,
        em                  = em_result,
        joint_H0            = joint_H0,
        joint_H1            = joint_H1,
        latent_class_result = latent_class_result,
        bma_result          = bma_result,
        combination_method  = combination_method,
        em_diagnostics      = em_diagnostics,
        em_diagnostics_summary = em_diagnostics_summary
        )
end

"""
    analyse(imputed_data, raw_data, H0_file="copula_H0.xlsx"; kwargs...)

Performs the main Bayesian analysis pipeline on the provided proteomics data with multiple imputation.

This function integrates results from three different models:
1. A Beta-Bernoulli model for detection probabilities.
2. A hierarchical Bayesian model for protein enrichment (log2 fold change).
3. A Bayesian linear regression model for dose-response correlation.

The Bayes factors from these models are combined using a copula to calculate a final,
joint Bayes factor and posterior probability for each protein. The analysis is
parallelized across proteins.

# Arguments
- `imputed_data::Vector{InteractionData}`: The multiple imputed data set loaded as a vector of `InteractionData` objects. Used for HBM and regression models.
- `raw_data::InteractionData`: The non-imputed data. This dataset is used for the computation of the BF-detection (Beta-Bernoulli model).
- `H0_file::String`: Path to the H0 file containing precomputed Bayes factors for the null hypothesis. If the file does not exist, it will be computed.

# Keywords
## Basic Analysis Parameters
- `n_controls::Int=0`: Number of controls in the dataset.
- `n_samples::Int=0`: Number of samples in the dataset.
- `refID::Int=1`: The reference ID for the main analysis function, typically referring to a reference condition.
- `plotHBMdists::Bool=false`: If `true`, generates and saves plots of the hierarchical Bayesian model distributions.
- `plotlog2fc::Bool=false`: If `true`, generates and saves plots of the log2 fold changes.
- `plotregr::Bool=false`: If `true`, generates and saves plots of the regression model.
- `plotbayesrange::Bool=false`: If `true`, generates and saves plots of the Bayes factor ranges.
- `verbose::Bool=false`: If `true`, prints detailed progress and debugging information.

## Caching Parameters
- `use_intermediate_cache::Bool=true`: If `true`, enables caching of intermediate results (Beta-Bernoulli and HBM+Regression).
- `betabernoulli_cache_file::String=""`: Path to cache file for Beta-Bernoulli results. Empty string disables caching for this step.
- `hbm_regression_cache_file::String=""`: Path to cache file for HBM and regression results. Empty string disables caching for this step.

## Copula-EM Parameters
- `prior::Union{Symbol, NamedTuple}=:default`: Prior specification for the EM algorithm. Use `:default` for automatic prior or provide a NamedTuple with `(α=..., β=...)` for custom Beta prior parameters.
- `n_restarts::Int=20`: Number of random restarts for the EM algorithm to avoid local optima.
- `copula_criterion::Symbol=:BIC`: Model selection criterion for copula fitting. Options: `:BIC`, `:AIC`.
- `h1_refitting::Bool=true`: If `true`, refits the H1 (alternative hypothesis) distribution after EM convergence.
- `burn_in::Int=10`: Number of initial EM iterations to discard before convergence checking.

## Diagnostics Parameters
- `run_em_diagnostics::Bool=true`: If `true` and `n_restarts > 1`, runs diagnostic analysis of EM restart stability and convergence.

# Returns
- `NamedTuple`: A named tuple containing the analysis results with the following fields:
    - `copula_results::DataFrame`: DataFrame with combined Bayes factors, posterior probabilities, q-values, and other key metrics for each protein.
    - `df_hierarchical::DataFrame`: DataFrame with detailed results from the hierarchical and regression models.
    - `convergence_plt`: A plot diagnosing the convergence of the EM algorithm.
    - `em`: The fitted Expectation-Maximization model object.
    - `joint_H0`: The estimated joint distribution under the null hypothesis (H0).
    - `joint_H1`: The estimated joint distribution under the alternative hypothesis (H1).
    - `em_diagnostics`: Detailed diagnostics from EM restart analysis (if `run_em_diagnostics=true` and `n_restarts > 1`).
    - `em_diagnostics_summary`: Summary statistics of EM diagnostics (if `run_em_diagnostics=true` and `n_restarts > 1`).
"""
function analyse(
    imputed_data::Vector{InteractionData},
    raw_data::InteractionData,
    H0_file = "copula_H0.xlsx";
    n_controls = 0, n_samples = 0, refID = 1,
    plotHBMdists = false, plotlog2fc = false, plotregr = false,
    plotbayesrange = false,
    verbose = false,
    use_intermediate_cache::Bool = true,
    betabernoulli_cache_file::String = "",
    hbm_regression_cache_file::String = "",
    # Copula-EM parameters
    prior::Union{Symbol, NamedTuple} = :default,
    n_restarts::Int = 20,
    copula_criterion::Symbol = :BIC,
    h1_refitting::Bool = true,
    burn_in::Int = 10,
    # Diagnostics
    run_em_diagnostics::Bool = true,
    # Evidence combination method
    combination_method::Symbol = :bma,
    # Latent class parameters
    lc_n_iterations::Int = 100,
    lc_alpha_prior::Vector{Float64} = [10.0, 1.0],
    lc_convergence_tol::Float64 = 1e-6,
    lc_winsorize::Bool = true,
    lc_winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99),
    # Robust regression parameters
    regression_likelihood::Symbol = :normal,
    student_t_nu::Float64 = 5.0
)

    n_imputed = length(imputed_data)

    # generate cache folder
    ispath("cache") && rm("cache", recursive = true)
    mkpath("cache")
    # get number of proteins
    n_proteins = length(getIDs(raw_data))

    # load H0 file or recompute it (skip for latent_class-only mode; copula and bma need it)
    if combination_method in (:copula, :bma)
        if isnothing(H0_file) || !isfile(H0_file)
            H0 = computeH0_BayesFactors(
                imputed_data[1],
                n_controls = n_controls, n_samples = n_samples,
                refID = refID, savefile = H0_file
                )
        else
            H0 = DataFrame(readtable(H0_file, "Sheet1", first_row = 1))
        end
    end

    # ------------------------------------ #
    # Beta-Bernoulli model (uses raw_data)
    # ------------------------------------ #
    bf_detected = zeros(Float64, n_proteins)
    bb_cache_used = false

    # Check Beta-Bernoulli cache (uses raw_data for hash)
    if use_intermediate_cache && !isempty(betabernoulli_cache_file)
        bb_status, bb_cached = check_betabernoulli_cache(betabernoulli_cache_file, raw_data, n_controls, n_samples)
        if bb_status == INTERMEDIATE_CACHE_HIT
            bf_detected = bb_cached.bf_detected
            bb_cache_used = true
        end
    end

    # Compute if not cached
    if !bb_cache_used
        p = Progress(
            n_proteins, desc="Step 1: Computing Beta-Bernoulli Bayes factors...",
            showspeed=true,
            barglyphs=BarGlyphs('|','█', [' ' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|'),
            barlen = 20
            )

        Threads.@threads for i in 1:n_proteins
            bf_detected[i], _, _ = betabernoulli(raw_data, i, n_controls, n_samples)
            ProgressMeter.next!(p)
        end
        finish!(p)

        # Save to cache
        if use_intermediate_cache && !isempty(betabernoulli_cache_file)
            try
                bb_cache = BetaBernoulliCache(
                    bf_detected,
                    getIDs(raw_data),
                    n_controls,
                    n_samples,
                    compute_data_hash(raw_data),
                    now(),
                    string(pkgversion(@__MODULE__))
                )
                save_betabernoulli_cache(bb_cache, betabernoulli_cache_file)
                @info "Saved Beta-Bernoulli results to cache: $betabernoulli_cache_file"
            catch e
                @warn "Failed to save Beta-Bernoulli cache: $e"
            end
        end
    end

    bf_detection = DataFrame(Protein = getIDs(raw_data), bf_detected = bf_detected)
    writetable("cache/bf_detection.xlsx", "bf_detection" => bf_detection)

    # ------------------------------------ #
    # hierarchical & regression model (uses imputed_data)
    # ------------------------------------ #
    df = nothing
    bf_enrichment = Float64[]
    bf_correlation = Float64[]
    hbm_cache_used = false

    # Check HBM+Regression cache (uses combined hash of imputed + raw data)
    if use_intermediate_cache && !isempty(hbm_regression_cache_file)
        hbm_status, hbm_cached = check_hbm_regression_cache(hbm_regression_cache_file, (imputed_data, raw_data), refID, regression_likelihood, student_t_nu)
        if hbm_status == INTERMEDIATE_CACHE_HIT
            df = hbm_cached.df_hierarchical
            bf_enrichment = hbm_cached.bf_enrichment
            bf_correlation = hbm_cached.bf_correlation
            hbm_cache_used = true
        end
    end

    # Compute if not cached
    if !hbm_cache_used
        μ_0 = zeros(Float64, n_imputed)
        σ_0 = zeros(Float64, n_imputed)
        a_0 = zeros(Float64, n_imputed)
        b_0 = zeros(Float64, n_imputed)

        τ_dist = τ0.(imputed_data)
        a_0 = [τ.α for τ in τ_dist]
        b_0 = [τ.θ for τ in τ_dist]

        μ_dist = μ0.(imputed_data)
        μ_0 = [μ[1] for μ in μ_dist]
        σ_0 = [μ[2] for μ in μ_dist]

        # Compute τ_base for robust regression (Empirical Bayes) using first imputed dataset
        robust_tau_base = NaN
        if regression_likelihood == :robust_t
            robust_tau_base = estimate_regression_tau_base(imputed_data[1], refID)
            @info "Estimated τ_base = $(round(robust_tau_base, digits=4)) for robust regression"
        end

        p = Progress(
            n_proteins, desc="Step 2: Computing hierarchical and regression Bayes factors...",
            showspeed=true,
            barglyphs=BarGlyphs('|','█', [' ' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen = 20, dt = 10
            )

        Threads.@threads for i in 1:n_proteins
            try
                results = BayesResult[]
                for j in 1:n_imputed
                    result = main(
                        imputed_data[j], i, refID,
                        plotHBMdists = plotHBMdists, plotlog2fc = plotlog2fc,
                        plotregr = plotregr, plotbayesrange = plotbayesrange,
                        csv_file = "cache/results_$(Threads.threadid()).csv",
                        writecsv = false, verbose = verbose, computeHBM = true,
                        μ_0 = μ_0[j], σ_0 = σ_0[j], a_0 = a_0[j], b_0 = b_0[j],
                        regression_likelihood = regression_likelihood,
                        student_t_nu = student_t_nu,
                        robust_tau_base = robust_tau_base
                    )
                    push!(results, result)
                end
                mI = evaluate_imputed_fc_posteriors(
                    results, getProtocolPositions(imputed_data[1]),
                    writecsv = true, plotlog2fc = plotlog2fc,
                    plotbayesrange = plotbayesrange,
                    csv_file = "cache/results_$(Threads.threadid()).csv"
                    )

            catch e
                open("log.txt", "a") do f
                    println(f, "Error in protein $i: $e")
                end
            end
            ProgressMeter.next!(p)
        end
        finish!(p)

        @info "Finished computing hierarchical and regression Bayes factors"
        @info "Errors are logged in log.txt"
        lst_files = string.(readdir("cache", join = true))
        lst_files = filter(x -> occursin(".csv", x), lst_files)
        convert_to_xlsx_filename(file) = replace(file, ".csv" => "_hierarchical.xlsx")

        df = [clean_result(imputed_data[1], file, convert_to_xlsx_filename(file)) for file in lst_files]
        df = reduce(vcat, df)

        # remove all proteins where no BF could be computed
        protein_names = getIDs(raw_data)
        protein_in_dataset = [protein_names[i] in df.Protein for i ∈ 1:length(protein_names)]
        protein_names = protein_names[protein_in_dataset]

        # sort dataframe
        new_order = [findfirst(x -> x == p, df.Protein) for p ∈ protein_names]
        df = df[new_order, :]

        # add BF for detection to the dataframe
        df = innerjoin(df, bf_detection, on = :Protein)

        # parse
        for i in 1:size(df,1)
            if typeof(df.BF_log2FC[i]) != Float64
                df.BF_log2FC[i] == "NA" ? df.BF_log2FC[i] = 0.0 : df.bf_slope[i] = parse(Float64, df.bf_slope[i])
            end
            if typeof(df.bf_slope[i]) != Float64
                df.bf_slope[i] == "NA" ? df.bf_slope[i] = 0.0 : df.bf_slope[i] = parse(Float64, df.bf_slope[i])
            end
        end

        # extract Bayes factors
        bf_correlation = df.bf_slope
        bf_enrichment  = df.BF_log2FC

        # Save to cache
        if use_intermediate_cache && !isempty(hbm_regression_cache_file)
            try
                hbm_cache = HBMRegressionCache(
                    df,
                    bf_enrichment,
                    bf_correlation,
                    getIDs(raw_data),
                    refID,
                    regression_likelihood,
                    student_t_nu,
                    compute_data_hash(imputed_data, raw_data),
                    now(),
                    string(pkgversion(@__MODULE__))
                )
                save_hbm_regression_cache(hbm_cache, hbm_regression_cache_file)
                @info "Saved HBM+Regression results to cache: $hbm_regression_cache_file"
            catch e
                @warn "Failed to save HBM+Regression cache: $e"
            end
        end
    end

    # Update bf_detected from df (either from cache or freshly computed)
    bf_detected = df.bf_detected

    # ------------------------------------ #
    # Evidence combination
    # ------------------------------------ #
    combined_bf = Float64[]
    posterior_prob = Float64[]
    convergence_plt = nothing
    em_result = nothing
    joint_H0 = nothing
    joint_H1 = nothing
    latent_class_result = nothing
    bma_result = nothing
    em_diagnostics = nothing
    em_diagnostics_summary = nothing

    bf_triplet = BayesFactorTriplet(bf_enrichment, bf_correlation, bf_detected)

    if combination_method == :copula
        @info "Starting copula-EM combination"

        combinedResult = combined_BF(
            bf_triplet, refID,
            max_iter = 5000, H0_file = H0_file,
            prior = prior,
            n_restarts = n_restarts,
            copula_criterion = copula_criterion,
            h1_refitting = h1_refitting,
            burn_in = burn_in,
            verbose = verbose
        )

        combined_bf = combinedResult.bf
        posterior_prob = combinedResult.posterior_prob
        em_result = combinedResult.em_result
        joint_H0 = combinedResult.joint_H0
        joint_H1 = combinedResult.joint_H1
        convergence_plt = EMconvergenceDiagnosticPlot(em_result)

        # Process EM diagnostics
        em_diagnostics = combinedResult.em_diagnostics
        if run_em_diagnostics && !isnothing(em_diagnostics)
            @info "Summarizing EM restart diagnostics..."
            em_diagnostics_summary = summarize_em_diagnostics(em_diagnostics)
        end

    elseif combination_method == :latent_class
        @info "Starting latent class (VMP) combination"

        latent_class_result = combined_BF_latent_class(
            bf_triplet, refID,
            n_iterations = lc_n_iterations,
            alpha_prior = lc_alpha_prior,
            convergence_tol = lc_convergence_tol,
            verbose = verbose,
            winsorize = lc_winsorize,
            winsorize_quantiles = lc_winsorize_quantiles
        )

        combined_bf = latent_class_result.bf
        posterior_prob = latent_class_result.posterior_prob
        convergence_plt = plot_lc_convergence(latent_class_result)

    elseif combination_method == :bma
        @info "Starting Bayesian Model Averaging (BMA) combination"

        bma_result = combined_BF_bma(
            bf_triplet, refID;
            H0_file = H0_file,
            prior = prior,
            n_restarts = n_restarts,
            copula_criterion = copula_criterion,
            h1_refitting = h1_refitting,
            burn_in = burn_in,
            lc_n_iterations = lc_n_iterations,
            lc_alpha_prior = lc_alpha_prior,
            lc_convergence_tol = lc_convergence_tol,
            lc_winsorize = lc_winsorize,
            lc_winsorize_quantiles = lc_winsorize_quantiles,
            verbose = verbose
        )

        combined_bf = bma_result.bf
        posterior_prob = bma_result.posterior_prob
        em_result = bma_result.copula_result.em_result
        joint_H0 = bma_result.copula_result.joint_H0
        joint_H1 = bma_result.copula_result.joint_H1
        latent_class_result = bma_result.latent_class_result
        em_diagnostics = bma_result.copula_result.em_diagnostics
        if run_em_diagnostics && !isnothing(em_diagnostics)
            @info "Summarizing EM restart diagnostics..."
            em_diagnostics_summary = summarize_em_diagnostics(em_diagnostics)
        end
        convergence_plt = EMconvergenceDiagnosticPlot(em_result)

    else
        error("Unknown combination_method: $combination_method. Must be :copula, :latent_class, or :bma")
    end

    q_values = q(combined_bf)

    # ------------------------------------ #
    # generate output file
    # ------------------------------------ #
    copula_df = DataFrame(
        Protein = df.Protein,
        BF = combined_bf,
        posterior_prob = posterior_prob,
        q = q_values,
        mean_log2FC = df.mean_log2FC,
        bf_enrichment = bf_enrichment,
        bf_correlation = bf_correlation,
        bf_detected = bf_detected
    )

    writetable(
        "results.xlsx",
        "hierarchical" => df,
        "copula" => copula_df
        )

    # delete logs
    rm("cache", recursive = true)

    return (
        copula_results      = copula_df,
        df_hierarchical     = df,
        convergence_plt     = convergence_plt,
        em                  = em_result,
        joint_H0            = joint_H0,
        joint_H1            = joint_H1,
        latent_class_result = latent_class_result,
        bma_result          = bma_result,
        combination_method  = combination_method,
        em_diagnostics      = em_diagnostics,
        em_diagnostics_summary = em_diagnostics_summary
        )
end


"""
    OutputFiles

Struct holding all output file paths for the analysis pipeline.

Construct with `OutputFiles(basedir)` to auto-generate all paths under a single directory,
or `OutputFiles(basedir; image_ext=".svg")` to change image format.

Individual paths can be overridden after construction since the struct is mutable.

# Fields
- `basedir::String`: Base directory for all output files.
- `H0_file::String`: Path to the null hypothesis Bayes factors file.
- `results_file::String`: Path for the final results Excel file.
- `volcano_file::String`: Path for the volcano plot image.
- `convergence_file::String`: Path for the EM convergence diagnostic plot.
- `evidence_file::String`: Path for the evidence plot.
- `dnn_file::String`: Path for the DNN/metalearner results Excel file.
- `rank_rank_file::String`: Path for the rank-rank plot image.
- `prior_file::String`: Path for the prior Excel file.
- `em_diagnostics_file::String`: Path for the EM diagnostics plot.
- `lc_convergence_file::String`: Path for the latent class convergence plot.
- `sensitivity_report_file::String`: Path for the sensitivity analysis report.
- `sensitivity_tornado_file::String`: Path for the sensitivity tornado plot.
- `sensitivity_heatmap_file::String`: Path for the sensitivity heatmap plot.
- `sensitivity_rankcorr_file::String`: Path for the sensitivity rank correlation plot.
- `pit_histogram_file::String`: Path for the PIT histogram plot.
- `nu_optimization_file::String`: Path for the Student-t ν optimization plot.
"""
Base.@kwdef mutable struct OutputFiles
    basedir::String
    H0_file::String
    results_file::String
    volcano_file::String
    convergence_file::String
    evidence_file::String
    dnn_file::String
    rank_rank_file::String
    prior_file::String
    em_diagnostics_file::String
    lc_convergence_file::String
    sensitivity_report_file::String
    sensitivity_tornado_file::String
    sensitivity_heatmap_file::String
    sensitivity_rankcorr_file::String
    sensitivity_table_file::String
    diagnostics_report_file::String
    ppc_histogram_file::String
    qq_plot_file::String
    regression_qq_plot_file::String
    calibration_plot_file::String
    calibration_comparison_file::String
    pit_histogram_file::String
    scale_location_hbm_file::String
    scale_location_regression_file::String
    nu_optimization_file::String
    report_file::String
    report_methods_file::String
end

"""
    OutputFiles(basedir::String; image_ext::String=".png")

Create an `OutputFiles` with all paths auto-generated under `basedir`.
"""
function OutputFiles(basedir::String; image_ext::String=".png")
    OutputFiles(
        basedir             = basedir,
        H0_file             = joinpath(basedir, "copula_H0.xlsx"),
        results_file        = joinpath(basedir, "final_results.xlsx"),
        volcano_file        = joinpath(basedir, "volcano_plot" * image_ext),
        convergence_file    = joinpath(basedir, "convergence" * image_ext),
        evidence_file       = joinpath(basedir, "evidence" * image_ext),
        dnn_file            = joinpath(basedir, "dnn_results.xlsx"),
        rank_rank_file      = joinpath(basedir, "rank_rank_plot" * image_ext),
        prior_file          = joinpath(basedir, "prior.xlsx"),
        em_diagnostics_file = joinpath(basedir, "em_diagnostics" * image_ext),
        lc_convergence_file = joinpath(basedir, "lc_convergence" * image_ext),
        sensitivity_report_file = joinpath(basedir, "sensitivity_report.md"),
        sensitivity_tornado_file = joinpath(basedir, "sensitivity_tornado" * image_ext),
        sensitivity_heatmap_file = joinpath(basedir, "sensitivity_heatmap" * image_ext),
        sensitivity_rankcorr_file = joinpath(basedir, "sensitivity_rankcorr" * image_ext),
        sensitivity_table_file = joinpath(basedir, "sensitivity_table.xlsx"),
        diagnostics_report_file = joinpath(basedir, "diagnostics_report.md"),
        ppc_histogram_file = joinpath(basedir, "ppc_histogram" * image_ext),
        qq_plot_file = joinpath(basedir, "residual_qq_hbm" * image_ext),
        regression_qq_plot_file = joinpath(basedir, "residual_qq_regression" * image_ext),
        calibration_plot_file = joinpath(basedir, "calibration" * image_ext),
        calibration_comparison_file = joinpath(basedir, "calibration_comparison" * image_ext),
        pit_histogram_file = joinpath(basedir, "pit_histogram" * image_ext),
        scale_location_hbm_file = joinpath(basedir, "scale_location_hbm" * image_ext),
        scale_location_regression_file = joinpath(basedir, "scale_location_regression" * image_ext),
        nu_optimization_file = joinpath(basedir, "nu_optimization" * image_ext),
        report_file = joinpath(basedir, "interactive_report.html"),
        report_methods_file = joinpath(basedir, "methods.md"),
    )
end

"""
    CONFIG

A struct to hold all configuration parameters for the analysis pipeline.

This struct uses `Base.@kwdef` to allow initialization by keyword arguments.

# Required Fields
- `datafile::Vector{String}`: Paths to the input data files (e.g., Excel files).
- `control_cols::Vector{Dict{Int,Vector{Int}}}`: Control columns for each data file.
- `sample_cols::Vector{Dict{Int,Vector{Int}}}`: Sample columns for each data file.
- `poi::String`: Identifier for the protein of interest (bait protein), used for the meta-learner.

# Output
- `output::OutputFiles = OutputFiles(".")`: Output file paths. Construct with `OutputFiles(basedir)` for auto-generated paths.

# Analysis Parameters
- `normalise_protocols::Bool = true`: Normalize data across different experimental protocols.
- `n_controls::Int = 0`: Number of control experiments.
- `n_samples::Int = 0`: Number of sample experiments.
- `refID::Int = 1`: The ID of the reference (bait) protein or reference condition.
- `plotHBMdists::Bool = false`: Plot HBM posterior distributions per protein.
- `plotlog2fc::Bool = false`: Plot log2 fold-change distributions.
- `plotregr::Bool = false`: Plot regression model fits.
- `plotbayesrange::Bool = false`: Plot Bayes factor range plots.
- `verbose::Bool = false`: Enable verbose logging output.
- `vc_legend_pos::Symbol = :topleft`: Legend position in the volcano plot.
- `metalearner_path::String`: Path to the metalearner model file.

# Copula-EM Parameters
- `em_prior::Union{Symbol, NamedTuple{(:α, :β), ...}} = :default`: Prior for EM algorithm (`:default` or named tuple).
- `em_n_restarts::Int = 20`: Number of random restarts for EM.
- `copula_criterion::Symbol = :BIC`: Copula model selection criterion (`:BIC` or `:AIC`).
- `h1_refitting::Bool = true`: Re-fit H1 copula after initial EM.
- `em_burn_in::Int = 10`: EM burn-in iterations.
- `run_em_diagnostics::Bool = true`: Generate EM convergence diagnostics.

# Evidence Combination
- `combination_method::Symbol = :bma`: Evidence combination method (`:copula`, `:latent_class`, or `:bma` for Bayesian Model Averaging).

# Latent Class Parameters (used when `combination_method = :latent_class`)
- `lc_n_iterations::Int = 100`: Maximum EM iterations for latent class model.
- `lc_alpha_prior::Vector{Float64} = [10.0, 1.0]`: Dirichlet prior on class proportions.
- `lc_convergence_tol::Float64 = 1e-6`: Convergence tolerance.
- `lc_winsorize::Bool = true`: Winsorize extreme Bayes factors.
- `lc_winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99)`: Winsorization quantiles.

# Prior Sensitivity Analysis
- `run_sensitivity::Bool = true`: Run prior sensitivity analysis. Evaluates how robust posterior
  probabilities are to different prior specifications. Per-protein sensitivity metrics
  (std, min, max, range of posteriors; classification stability fractions) are merged into
  `final_results.xlsx` when diagnostics are also enabled.
- `sensitivity_config::SensitivityConfig = SensitivityConfig()`: Configuration for sensitivity analysis
  (prior grids for Beta-Bernoulli, EM, and latent class models).

# Posterior Predictive Checks & Model Diagnostics
- `run_diagnostics::Bool = false`: Run posterior predictive checks and model diagnostics.
  When enabled, computes per-protein diagnostic flags (observation counts, residual outliers,
  low-data warnings) for ALL proteins and merges them as columns into `final_results.xlsx`.
  Extended PPC statistics (p-values for skewness, kurtosis, IQR ratio) are available for the
  subset of proteins selected for PPC. Set `diagnostics_config.n_proteins_to_check` to control
  how many proteins undergo full PPC (default 50; set to total protein count for exhaustive checks).
- `diagnostics_config::DiagnosticsConfig = DiagnosticsConfig()`: Configuration for diagnostics
  (number of PPC draws, protein selection strategy, residual model, calibration bins, etc.).

# Regression Model Comparison
- `run_model_comparison::Bool = true`: Run both Normal and robust regression models for all proteins and compare via WAIC. Refits regression posteriors for all proteins to compute pointwise WAIC.
- `regression_likelihood::Symbol = :robust_t`: Likelihood for regression models (`:normal` or `:robust_t`).
- `student_t_nu::Float64 = 5.0`: Degrees of freedom for Student-t distribution.
- `optimize_nu::Bool = run_diagnostics`: Optimize ν over [3, 50] via Brent's method minimizing WAIC. When `true`, automatically sets `student_t_nu` to the optimal value. Implies `run_model_comparison` (Normal WAIC is computed as baseline). Defaults to `true` when `run_diagnostics` is enabled.
"""
Base.@kwdef mutable struct CONFIG
    datafile::Vector{String}
    control_cols::Vector{Dict{Int,Vector{Int}}}
    sample_cols::Vector{Dict{Int,Vector{Int}}}
    poi::String

    # output file paths
    output::OutputFiles         = OutputFiles(".")

    # analysis parameters
    normalise_protocols::Bool   = true
    n_controls::Int         = 0
    n_samples::Int          = 0
    refID::Int                  = 1
    plotHBMdists::Bool          = false
    plotlog2fc::Bool            = false
    plotregr::Bool              = false
    plotbayesrange::Bool        = false
    verbose::Bool               = false
    vc_legend_pos::Symbol       = :topleft
    metalearner_path::String    = "metalearners/HistGradientBoosting_tune.jld2"

    # Copula-EM parameters
    em_prior::Union{Symbol, NamedTuple{(:α, :β), Tuple{Float64, Float64}}} = :default
    em_n_restarts::Int          = 20
    copula_criterion::Symbol    = :BIC
    h1_refitting::Bool          = true
    em_burn_in::Int             = 10
    run_em_diagnostics::Bool    = true

    # Evidence combination method
    combination_method::Symbol  = :bma  # :copula, :latent_class, or :bma

    # Latent class parameters (used when combination_method = :latent_class)
    lc_n_iterations::Int        = 100
    lc_alpha_prior::Vector{Float64} = [10.0, 1.0]
    lc_convergence_tol::Float64 = 1e-6
    lc_winsorize::Bool          = true
    lc_winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99)

    # Prior sensitivity analysis
    run_sensitivity::Bool                   = true
    sensitivity_config::SensitivityConfig   = SensitivityConfig()

    # Posterior predictive checks & model diagnostics
    run_diagnostics::Bool                   = false
    diagnostics_config::DiagnosticsConfig   = DiagnosticsConfig()

    # Interactive HTML report
    generate_report_html::Bool              = true

    # Robust regression (Student-t via scale mixture)
    regression_likelihood::Symbol           = :robust_t    # :normal or :robust_t
    student_t_nu::Float64                   = 5.0        # degrees of freedom for Student-t
    run_model_comparison::Bool              = true        # run both models + WAIC comparison
    optimize_nu::Bool                       = true        # optimize ν via Brent's method (WAIC-based); follows run_diagnostics by default

    # Data curation (protein group splitting, synonym resolution, merging)
    curate::Bool                            = true         # enable protein curation (default: true)
    species::Int                            = 9606         # NCBI taxonomy ID (9606 = human)
    curate_interactive::Bool                = true         # prompt user for merge confirmation
    curate_merge_strategy::Symbol           = :max         # :max or :mean for merging duplicate rows
    bait_name::Union{Nothing, String}       = nothing      # bait protein name for refID tracking through curation
    curate_replay::Union{Nothing, String}   = nothing      # path to saved CurationReport JLD2 for replay
    curate_remove_contaminants::Bool        = true         # remove CON__/REV__ entries
    curate_delimiter::String                = ";"          # delimiter for protein group splitting
    curate_auto_approve::Int                = 0            # auto-approve merges with shared prefix length (0 = always ask)
end


"""
    _run_model_comparison(data, config) -> ModelComparisonResult

Run both Normal and robust regression models on all proteins and compare via WAIC.
Refits regression posteriors for all proteins to compute pointwise WAIC.
"""
function _run_model_comparison(data::InteractionData, config::CONFIG)
    refID = config.refID
    n_proteins = length(getIDs(data))
    protein_names = getNames(data)
    nu = config.student_t_nu

    # Compute hyperparameters
    μ_0, σ_0 = μ0(data)

    # Compute τ_base for robust regression (Empirical Bayes)
    τ_base = estimate_regression_tau_base(data, refID)
    @info "  Estimated τ_base = $(round(τ_base, digits=4)) for WAIC model comparison"

    # Precompute priors for both models
    @info "  Precomputing Normal + Robust regression priors..."
    if getNoProtocols(data) == 1
        normal_prior = precompute_regression_one_protocol_prior(data, refID, μ_0, σ_0)
        robust_prior = precompute_regression_one_protocol_robust_prior(data, refID, μ_0, σ_0; nu=nu, τ_base=τ_base)
    else
        normal_prior = precompute_regression_multi_protocol_prior(data, refID, μ_0, σ_0)
        robust_prior = precompute_regression_multi_protocol_robust_prior(data, refID, μ_0, σ_0; nu=nu, τ_base=τ_base)
    end

    # Build name → idx mapping
    name_to_idx = Dict(protein_names[i] => i for i in 1:n_proteins)

    # Fit both models for all proteins
    normal_results = Dict{String, RegressionResult}()
    robust_results = Dict{String, RobustRegressionResult}()

    p = Progress(
        n_proteins, desc="WAIC: Fitting Normal + Robust regression models...",
        showspeed=true,
        barglyphs=BarGlyphs('|','█', [' ' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|'),
        barlen=20, dt=10
    )

    # Use thread-local storage for thread-safe collection
    normal_thread_results = [Dict{String, RegressionResult}() for _ in 1:Threads.nthreads()]
    robust_thread_results = [Dict{String, RobustRegressionResult}() for _ in 1:Threads.nthreads()]

    Threads.@threads for i in 1:n_proteins
        tid = Threads.threadid()
        pname = protein_names[i]
        try
            # Normal model
            if getNoProtocols(data) == 1
                nr = RegressionModel_one_protocol(data, i, refID, μ_0, σ_0; cached_prior=normal_prior)
                normal_thread_results[tid][pname] = nr
            else
                nr = RegressionModel(data, i, refID, μ_0, σ_0; cached_prior=normal_prior)
                normal_thread_results[tid][pname] = nr
            end

            # Robust model
            if getNoProtocols(data) == 1
                rr = RegressionModel_one_protocol_robust(data, i, refID, μ_0, σ_0; nu=nu, τ_base=τ_base, cached_prior=robust_prior)
                robust_thread_results[tid][pname] = rr
            else
                rr = RegressionModelRobust(data, i, refID, μ_0, σ_0; nu=nu, τ_base=τ_base, cached_prior=robust_prior)
                robust_thread_results[tid][pname] = rr
            end
        catch e
            config.verbose && @warn "WAIC regression failed for protein $i ($pname): $e"
        end
        ProgressMeter.next!(p)
    end
    finish!(p)

    # Merge thread-local results
    for tid in 1:Threads.nthreads()
        merge!(normal_results, normal_thread_results[tid])
        merge!(robust_results, robust_thread_results[tid])
    end

    @info "  Fitted $(length(normal_results)) Normal and $(length(robust_results)) Robust regression models"

    # Compute WAIC comparison
    return compare_regression_models(data, normal_results, robust_results, name_to_idx, refID)
end


"""
    _build_curation_lookup(config::CONFIG) -> Union{DataFrame, Nothing}

Load curation reports for all data files and build a lookup DataFrame with
original protein names and STRING IDs. Returns `nothing` if curation was
disabled or no reports are found.

Returned columns: `Protein` (canonical name), `original_name`, `string_id`.
For merged proteins the original names are joined with `";"`.
"""
function _build_curation_lookup(config::CONFIG)::Union{DataFrame, Nothing}
    !config.curate && return nothing

    entries = CurationEntry[]
    for file in config.datafile
        cache_dir   = joinpath(dirname(abspath(file)), ".bayesinteractomics_cache")
        report_base = joinpath(cache_dir, splitext(basename(file))[1])
        report = load_curation_report(report_base * "_curation_report.jld2")
        isnothing(report) && continue
        append!(entries, report.entries)
    end

    isempty(entries) && return nothing

    # Removed proteins are not present in final results — skip them
    filter!(e -> e.action != CURATE_REMOVE, entries)

    # Group by canonical_name: collect unique original names; take the first non-empty string_id
    lookup = Dict{String, Tuple{Vector{String}, String}}()
    for e in entries
        orig_names, sid = get!(lookup, e.canonical_name, (String[], ""))
        push!(orig_names, e.original_name)
        new_sid = isempty(sid) ? e.canonical_id : sid
        lookup[e.canonical_name] = (orig_names, new_sid)
    end

    proteins = collect(keys(lookup))
    DataFrame(
        Protein       = proteins,
        original_name = [join(unique(lookup[p][1]), ";") for p in proteins],
        string_id     = [lookup[p][2] for p in proteins]
    )
end


"""
    run_analysis(config::CONFIG; use_cache=true, cache_file="", temp_result_file="temp_results.xlsx")

Runs the complete analysis pipeline from data loading to final result generation with intelligent caching.

This function serves as a high-level wrapper that orchestrates the entire
analysis workflow. It includes hash-based caching to avoid redundant computation
when config and data haven't changed. The meta-learner is always re-run even
when using cached results.

# Arguments
- `config::CONFIG`: Configuration struct containing all analysis parameters

# Keywords
- `use_cache::Bool=true`: Enable cache checking and saving
- `cache_file::String=""`: Custom cache file path (empty = auto-generate based on config)
- `temp_result_file::String="temp_results.xlsx"`: Temporary results file during analysis

# Returns
- `Tuple{DataFrame, AnalysisResult}`: A tuple containing:
  - `final_results::DataFrame`: Final DataFrame with posterior probabilities updated by meta-learner
  - `analysis_result::AnalysisResult`: Complete analysis results with caching metadata

# Cache Behavior
- On cache hit: Loads copula results, EM results, and distributions from cache
- Meta-learner always runs (uses `config.poi` which may change between runs)
- Plots are always regenerated
- Cache location: `.bayesinteractomics_cache/` directory next to first data file

# Side Effects
- Creates cache file if `use_cache=true` and analysis runs
- Creates final results Excel file at `config.output.results_file`
- Generates and saves plots (volcano, convergence, rank-rank, evidence)
- Creates and deletes temporary "cache" directory for intermediate files
- Creates "log.txt" file for logging errors during analysis

# Examples
```julia
config = CONFIG(
    datafile=["data.xlsx"],
    control_cols=[Dict(1=>[1,2,3])],
    sample_cols=[Dict(1=>[4,5,6])],
    poi="MyBaitProtein",
    n_controls=3,
    n_samples=3,
    refID=1
)

# First run - performs full analysis and caches results
final_df, result = run_analysis(config)

# Second run with same config/data - uses cache (fast!)
final_df, result = run_analysis(config)

# Disable caching
final_df, result = run_analysis(config, use_cache=false)

# Custom cache location
final_df, result = run_analysis(config, cache_file="my_cache.jld2")
```

See also: [`AnalysisResult`](@ref), [`check_cache`](@ref), [`analyse`](@ref)
"""
function run_analysis(config::CONFIG; use_cache::Bool=true, cache_file::String="", temp_result_file::String="temp_results.xlsx", use_intermediate_cache::Bool=true)
    # Load data (needed for cache validation and analysis)
    data = load_data(
        config.datafile, config.sample_cols, config.control_cols,
        normalise_protocols = config.normalise_protocols,
        curate = config.curate,
        species = config.species,
        curate_interactive = config.curate_interactive,
        curate_merge_strategy = config.curate_merge_strategy,
        bait_name = config.bait_name,
        curate_replay = config.curate_replay,
        curate_remove_contaminants = config.curate_remove_contaminants,
        curate_delimiter = config.curate_delimiter,
        curate_auto_approve = config.curate_auto_approve
    )

    # Update refID if bait tracking found a new position
    if config.curate && !isnothing(config.bait_name) && data isa Tuple
        # load_data returns (InteractionData, bait_idx) when curate=true and bait_name is set
        data, new_bait_idx = data
        if !isnothing(new_bait_idx) && new_bait_idx != config.refID
            @info "Updating refID from $(config.refID) to $new_bait_idx (bait protein relocated by curation)"
            config.refID = new_bait_idx
        end
    end

    # update DiagnosticsConfig
    config.diagnostics_config.n_proteins_to_check = length(data.protein_names)

    # Check cache if enabled
    cache_path = isempty(cache_file) ? get_cache_filepath(config) : cache_file
    analysis_result = nothing

    if use_cache
        status, cached = check_cache(cache_path, config, data)
        if status == CACHE_HIT
            # Use cached results
            analysis_result = cached
            @info "Using cached analysis results"
        elseif status == CACHE_MISS_CONFIG
            @info "Config changed, running full analysis"
        elseif status == CACHE_MISS_DATA
            @info "Data changed, running full analysis"
        else  # CACHE_MISS_NO_FILE
            @info "No cache found, running full analysis"
        end
    end

    # Run analysis if not cached
    if isnothing(analysis_result)
        # Generate intermediate cache file paths
        bb_cache_path = use_intermediate_cache ? get_betabernoulli_cache_filepath(config) : ""
        hbm_cache_path = use_intermediate_cache ? get_hbm_regression_cache_filepath(config) : ""

        results = analyse(
            data,
            config.output.H0_file,
            n_controls      = config.n_controls,
            n_samples       = config.n_samples,
            refID           = config.refID,
            plotHBMdists    = config.plotHBMdists,
            plotlog2fc      = config.plotlog2fc,
            plotregr        = config.plotregr,
            plotbayesrange  = config.plotbayesrange,
            verbose         = config.verbose,
            temp_result_file = temp_result_file,
            use_intermediate_cache = use_intermediate_cache,
            betabernoulli_cache_file = bb_cache_path,
            hbm_regression_cache_file = hbm_cache_path,
            # Copula-EM parameters
            prior           = config.em_prior,
            n_restarts      = config.em_n_restarts,
            copula_criterion = config.copula_criterion,
            h1_refitting    = config.h1_refitting,
            burn_in         = config.em_burn_in,
            run_em_diagnostics = config.run_em_diagnostics,
            # Evidence combination
            combination_method = config.combination_method,
            lc_n_iterations = config.lc_n_iterations,
            lc_alpha_prior  = config.lc_alpha_prior,
            lc_convergence_tol = config.lc_convergence_tol,
            lc_winsorize    = config.lc_winsorize,
            lc_winsorize_quantiles = config.lc_winsorize_quantiles,
            # Robust regression
            regression_likelihood = config.regression_likelihood,
            student_t_nu = config.student_t_nu
        )

        # Create and cache AnalysisResult
        analysis_result = AnalysisResult(
            results.copula_results,
            results.df_hierarchical,
            results.em,
            results.joint_H0,
            results.joint_H1,
            results.latent_class_result,
            results.bma_result,
            results.combination_method,
            results.em_diagnostics,
            results.em_diagnostics_summary,
            compute_config_hash(config),
            compute_data_hash(data),
            now(),
            string(pkgversion(@__MODULE__)),
            nothing,  # bait_protein - can be set by user later
            nothing,  # bait_index
            nothing,  # sensitivity
            nothing   # diagnostics
        )

        # Save to cache
        if use_cache
            try
                save_result(analysis_result, cache_path)
                @info "Saved results to cache: $cache_path"
            catch e
                @warn "Failed to save cache: $e"
            end
        end

        # Save convergence plot
        if config.combination_method in (:copula, :bma)
            StatsPlots.savefig(results.convergence_plt, config.output.convergence_file)
        elseif config.combination_method == :latent_class
            StatsPlots.savefig(results.convergence_plt, config.output.lc_convergence_file)
        end

        # Save EM diagnostics plot if available (copula or bma)
        if config.combination_method in (:copula, :bma) && config.run_em_diagnostics && results.em_diagnostics !== nothing
            try
                diag_plt = plot_em_diagnostics(results.em_diagnostics)
                StatsPlots.savefig(diag_plt, config.output.em_diagnostics_file)
                @info "Saved EM diagnostics plot to: $(config.output.em_diagnostics_file)"
            catch e
                @warn "Failed to save EM diagnostics plot: $e"
            end
        end

        # For BMA: also save latent class convergence plot
        if config.combination_method == :bma && !isnothing(results.latent_class_result)
            try
                lc_plt = plot_lc_convergence(results.latent_class_result)
                StatsPlots.savefig(lc_plt, config.output.lc_convergence_file)
                @info "Saved latent class convergence plot to: $(config.output.lc_convergence_file)"
            catch e
                @warn "Failed to save latent class convergence plot: $e"
            end
        end
    end

    # Generate plots even when loading from cache
    # (plots are lightweight to regenerate and may have different filenames)
    if analysis_result.combination_method in (:copula, :bma) && !isnothing(analysis_result.em_diagnostics)
        try
            diag_plt = plot_em_diagnostics(analysis_result.em_diagnostics)
            StatsPlots.savefig(diag_plt, config.output.em_diagnostics_file)
            @info "Saved EM diagnostics plot to: $(config.output.em_diagnostics_file)"
        catch e
            @warn "Failed to save EM diagnostics plot: $e"
        end
    end

    # Prior sensitivity analysis
    if config.run_sensitivity
        try
            @info "Running prior sensitivity analysis..."
            sr = sensitivity_analysis(
                analysis_result, data;
                config = config.sensitivity_config,
                n_controls = config.n_controls,
                n_samples = config.n_samples,
                refID = config.refID,
                H0_file = config.output.H0_file,
                combination_method = config.combination_method,
                lc_n_iterations = config.lc_n_iterations,
                lc_convergence_tol = config.lc_convergence_tol,
                verbose = config.verbose
            )
            analysis_result.sensitivity = sr

            # Generate sensitivity plots
            try
                sensitivity_tornado_plot(sr; file = config.output.sensitivity_tornado_file)
                @info "Saved sensitivity tornado plot to: $(config.output.sensitivity_tornado_file)"
            catch e
                @warn "Failed to save sensitivity tornado plot: $e"
            end
            try
                sensitivity_heatmap(sr; file = config.output.sensitivity_heatmap_file)
                @info "Saved sensitivity heatmap to: $(config.output.sensitivity_heatmap_file)"
            catch e
                @warn "Failed to save sensitivity heatmap: $e"
            end
            try
                sensitivity_rank_correlation(sr; file = config.output.sensitivity_rankcorr_file)
                @info "Saved sensitivity rank correlation plot to: $(config.output.sensitivity_rankcorr_file)"
            catch e
                @warn "Failed to save sensitivity rank correlation plot: $e"
            end

            generate_sensitivity_report(sr;
                filename = config.output.sensitivity_report_file,
                tornado_file = config.output.sensitivity_tornado_file,
                heatmap_file = config.output.sensitivity_heatmap_file,
                rankcorr_file = config.output.sensitivity_rankcorr_file
            )
            @info "Sensitivity report saved to: $(config.output.sensitivity_report_file)"

            try
                writetable(config.output.sensitivity_table_file, "sensitivity" => sr.summary)
                @info "Sensitivity table saved to: $(config.output.sensitivity_table_file)"
            catch e
                @warn "Failed to save sensitivity table: $e"
            end
        catch e
            @warn "Prior sensitivity analysis failed: $e"
        end
    end

    # Posterior predictive checks & model diagnostics
    if config.run_diagnostics
        try
            @info "Running posterior predictive checks & model diagnostics..."
            # Compute τ_base for robust regression diagnostics (Empirical Bayes)
            diag_tau_base = config.regression_likelihood == :robust_t ? estimate_regression_tau_base(data, config.refID) : NaN
            dr = model_diagnostics(
                analysis_result, data;
                config = config.diagnostics_config,
                n_controls = config.n_controls,
                n_samples = config.n_samples,
                refID = config.refID,
                verbose = config.verbose,
                regression_likelihood = config.regression_likelihood,
                student_t_nu = config.student_t_nu,
                robust_tau_base = diag_tau_base
            )
            analysis_result.diagnostics = dr

            # Generate diagnostic plots
            try
                ppc_plt = ppc_pvalue_histogram(dr; file = config.output.ppc_histogram_file)
                @info "Saved PPC histogram to: $(config.output.ppc_histogram_file)"
            catch e
                @warn "Failed to save PPC histogram: $e"
            end
            if !isnothing(dr.hbm_residuals)
                try
                    residual_qq_plot(dr.hbm_residuals; file = config.output.qq_plot_file)
                    @info "Saved HBM Q-Q plot to: $(config.output.qq_plot_file)"
                catch e
                    @warn "Failed to save HBM Q-Q plot: $e"
                end
            end
            if !isnothing(dr.regression_residuals)
                try
                    residual_qq_plot(dr.regression_residuals; file = config.output.regression_qq_plot_file)
                    @info "Saved regression Q-Q plot to: $(config.output.regression_qq_plot_file)"
                catch e
                    @warn "Failed to save regression Q-Q plot: $e"
                end
            end
            if !isnothing(dr.hbm_residuals)
                try
                    scale_location_plot(dr.hbm_residuals; file = config.output.scale_location_hbm_file)
                    @info "Saved HBM scale-location plot to: $(config.output.scale_location_hbm_file)"
                catch e
                    @warn "Failed to save HBM scale-location plot: $e"
                end
            end
            if !isnothing(dr.regression_residuals)
                try
                    scale_location_plot(dr.regression_residuals; file = config.output.scale_location_regression_file)
                    @info "Saved regression scale-location plot to: $(config.output.scale_location_regression_file)"
                catch e
                    @warn "Failed to save regression scale-location plot: $e"
                end
            end
            if !isnothing(dr.calibration)
                try
                    calibration_plot(dr.calibration; file = config.output.calibration_plot_file)
                    @info "Saved calibration plot to: $(config.output.calibration_plot_file)"
                catch e
                    @warn "Failed to save calibration plot: $e"
                end
            end
            try
                calibration_comparison_plot(dr; file = config.output.calibration_comparison_file)
                @info "Saved calibration comparison plot to: $(config.output.calibration_comparison_file)"
            catch e
                @warn "Failed to save calibration comparison plot: $e"
            end

            # Save PIT histogram when enhanced residuals are present
            pit_values = Float64[]
            if !isnothing(dr.enhanced_hbm_residuals)
                append!(pit_values, dr.enhanced_hbm_residuals.pit_values)
            end
            if !isnothing(dr.enhanced_regression_residuals)
                append!(pit_values, dr.enhanced_regression_residuals.pit_values)
            end
            if !isempty(pit_values)
                try
                    pit_histogram_plot(pit_values; file = config.output.pit_histogram_file)
                    @info "Saved PIT histogram to: $(config.output.pit_histogram_file)"
                catch e
                    @warn "Failed to save PIT histogram: $e"
                end
            end

        catch e
            @warn "Model diagnostics failed: $e"
        end
    end

    # WAIC model comparison (Normal vs. robust regression)
    # Skip when optimize_nu is enabled — the optimization already computes Normal WAIC as baseline
    if config.run_model_comparison && !config.optimize_nu
        try
            @info "Running WAIC model comparison (Normal vs. robust regression)..."
            model_cmp = _run_model_comparison(data, config)
            # Attach to diagnostics if available
            if !isnothing(analysis_result.diagnostics)
                dr = analysis_result.diagnostics
                analysis_result.diagnostics = DiagnosticsResult(
                    dr.config, dr.protein_ppcs, dr.bb_ppcs,
                    dr.hbm_residuals, dr.regression_residuals,
                    dr.calibration, dr.calibration_relaxed, dr.calibration_enrichment_only,
                    dr.enhanced_hbm_residuals, dr.enhanced_regression_residuals,
                    dr.ppc_extended, dr.protein_flags,
                    model_cmp, dr.nu_optimization,
                    dr.summary, dr.timestamp
                )
            end
            @info "WAIC comparison complete: preferred model = $(model_cmp.preferred_model), ΔWAIC = $(round(model_cmp.delta_waic, digits=1)) ± $(round(model_cmp.delta_se, digits=1))"
        catch e
            @warn "WAIC model comparison failed: $e"
        end
    end

    # ν optimization (runs BEFORE meta-learner; updates config.student_t_nu for future calls)
    if config.optimize_nu && config.regression_likelihood == :robust_t
        try
            @info "Running Student-t ν optimization via Brent's method..."
            nu_opt = optimize_nu(data, config)
            config.student_t_nu = nu_opt.optimal_nu
            @info "Optimal ν = $(round(nu_opt.optimal_nu, digits=2)), WAIC = $(round(nu_opt.optimal_waic.waic, digits=1))"

            # Save ν optimization plot
            try
                nu_optimization_plot(nu_opt; file = config.output.nu_optimization_file)
                @info "Saved ν optimization plot to: $(config.output.nu_optimization_file)"
            catch e
                @warn "Failed to save ν optimization plot: $e"
            end

            # Build ModelComparisonResult from nu_opt for backward compat
            model_cmp_from_nu = ModelComparisonResult(
                nu_opt.normal_waic, nu_opt.optimal_waic,
                nu_opt.delta_waic, nu_opt.delta_se,
                nu_opt.delta_waic > 0 ? :robust : :normal
            )

            # Attach to diagnostics if available
            if !isnothing(analysis_result.diagnostics)
                dr = analysis_result.diagnostics
                analysis_result.diagnostics = DiagnosticsResult(
                    dr.config, dr.protein_ppcs, dr.bb_ppcs,
                    dr.hbm_residuals, dr.regression_residuals,
                    dr.calibration, dr.calibration_relaxed, dr.calibration_enrichment_only,
                    dr.enhanced_hbm_residuals, dr.enhanced_regression_residuals,
                    dr.ppc_extended, dr.protein_flags,
                    model_cmp_from_nu, nu_opt,
                    dr.summary, dr.timestamp
                )
            end
        catch e
            @warn "Student-t ν optimization failed: $e"
        end
    end

    # Generate diagnostics report AFTER model comparison and ν optimization
    # so the report includes all computed results
    if config.run_diagnostics && !isnothing(analysis_result.diagnostics)
        try
            generate_diagnostics_report(analysis_result.diagnostics;
                filename = config.output.diagnostics_report_file,
                ppc_histogram_file = config.output.ppc_histogram_file,
                qq_plot_file = config.output.qq_plot_file,
                regression_qq_plot_file = config.output.regression_qq_plot_file,
                calibration_plot_file = config.output.calibration_plot_file,
                calibration_comparison_file = config.output.calibration_comparison_file,
                pit_histogram_file = config.output.pit_histogram_file,
                scale_location_hbm_file = config.output.scale_location_hbm_file,
                scale_location_regression_file = config.output.scale_location_regression_file,
                nu_optimization_file = config.output.nu_optimization_file
            )
            @info "Diagnostics report saved to: $(config.output.diagnostics_report_file)"
        catch e
            @warn "Failed to generate diagnostics report: $e"
        end
    end

    # Always run meta-learner (poi may change between runs)
    meta_data, _ = predict_metalearner(config.poi, output_file = config.output.prior_file, metalearner_file = config.metalearner_path)

    # Validate metalearner prediction succeeded
    if isnothing(meta_data)
        error("Metalearner prediction failed. Check errors above for details.")
    end

    # Update posterior probabilities with meta-learner predictions
    final_results = copy(analysis_result.copula_results)  # Copy to avoid mutating cached data
    final_results = update_posterior_prob!(final_results, meta_data)

    # Update q-values
    final_results.q = q(final_results.posterior_prob, isBF = false)

    # Sort dataframe by Bayesian FDR (q-value) and log2FC
    sort!(final_results, [:q, :mean_log2FC], rev = [false, false])

    # Merge diagnostic and sensitivity columns into final results
    if !isnothing(analysis_result.diagnostics) || !isnothing(analysis_result.sensitivity)
        final_results = _merge_diagnostics_to_results(
            final_results, analysis_result.diagnostics;
            sensitivity = analysis_result.sensitivity
        )
    end

    # Add curation metadata (original names and STRING IDs) as columns 2–3
    curation_lookup = _build_curation_lookup(config)
    if !isnothing(curation_lookup)
        final_results = leftjoin(final_results, curation_lookup, on = :Protein)
        other_cols = setdiff(names(final_results), ["Protein", "original_name", "string_id"])
        select!(final_results, "Protein", "original_name", "string_id", other_cols...)
    end

    # Update AnalysisResult with metalearner-updated results (includes diagnostic columns)
    # This ensures ar.copula_results (accessed via ar.results) has final posterior probabilities
    analysis_result.copula_results = final_results

    # Save final results
    writetable(config.output.results_file, "df" => final_results)

    # Generate plots (always regenerate for current run)
    volcano_plt = volcano_plot(final_results)
    StatsPlots.savefig(volcano_plt, config.output.volcano_file)

    rrp = rank_rank_plot(final_results)
    StatsPlots.savefig(rrp, config.output.rank_rank_file)

    plot_results(final_results)
    evd_plot = evidence_plot(final_results)
    StatsPlots.savefig(evd_plot, config.output.evidence_file)

    # Clean up temporary files
    isfile(temp_result_file) && rm(temp_result_file, force = true)

    # Generate interactive HTML report
    if config.generate_report_html
        try
            generate_report(final_results, config)
        catch e
            @warn "Failed to generate interactive HTML report: $e"
        end
    end

    return final_results, analysis_result
end

# ---- function for multiple imputed data ---- #
"""
    run_analysis(config::CONFIG, imputed_data::Vector{InteractionData}, raw_data::InteractionData;
                 use_cache=true, cache_file="")

Runs analysis pipeline with multiple imputation and intelligent caching.

Similar to single-dataset version but uses multiple imputed datasets for HBM/regression
and raw data for Beta-Bernoulli model. Cache validation uses combined hash of all datasets.

# Arguments
- `config::CONFIG`: Configuration struct
- `imputed_data::Vector{InteractionData}`: Vector of imputed datasets
- `raw_data::InteractionData`: Original non-imputed data

# Keywords
- `use_cache::Bool=true`: Enable cache checking and saving
- `cache_file::String=""`: Custom cache file path (empty = auto-generate)

# Returns
- `Tuple{DataFrame, AnalysisResult}`: Final results and cached analysis result

See also: [`run_analysis(::CONFIG)`](@ref), [`AnalysisResult`](@ref)
"""
function run_analysis(config::CONFIG, imputed_data::Vector{InteractionData}, raw_data::InteractionData;
                     use_cache::Bool=true, cache_file::String="", use_intermediate_cache::Bool=true)
    # Check cache with combined hash of imputed + raw data
    cache_path = isempty(cache_file) ? get_cache_filepath(config) : cache_file
    analysis_result = nothing

    if use_cache
        status, cached = check_cache(cache_path, config, (imputed_data, raw_data))
        if status == CACHE_HIT
            analysis_result = cached
            @info "Using cached analysis results"
        elseif status == CACHE_MISS_CONFIG
            @info "Config changed, running full analysis"
        elseif status == CACHE_MISS_DATA
            @info "Data changed, running full analysis"
        else  # CACHE_MISS_NO_FILE
            @info "No cache found, running full analysis"
        end
    end

    # Run analysis if not cached
    if isnothing(analysis_result)
        # Generate intermediate cache file paths
        bb_cache_path = use_intermediate_cache ? get_betabernoulli_cache_filepath(config) : ""
        hbm_cache_path = use_intermediate_cache ? get_hbm_regression_cache_filepath(config) : ""

        results = analyse(
            imputed_data,
            raw_data,
            config.output.H0_file,
            n_controls      = config.n_controls,
            n_samples       = config.n_samples,
            refID           = config.refID,
            plotHBMdists    = config.plotHBMdists,
            plotlog2fc      = config.plotlog2fc,
            plotregr        = config.plotregr,
            plotbayesrange  = config.plotbayesrange,
            verbose         = config.verbose,
            use_intermediate_cache = use_intermediate_cache,
            betabernoulli_cache_file = bb_cache_path,
            hbm_regression_cache_file = hbm_cache_path,
            # Copula-EM parameters
            prior           = config.em_prior,
            n_restarts      = config.em_n_restarts,
            copula_criterion = config.copula_criterion,
            h1_refitting    = config.h1_refitting,
            burn_in         = config.em_burn_in,
            run_em_diagnostics = config.run_em_diagnostics,
            # Evidence combination
            combination_method = config.combination_method,
            lc_n_iterations = config.lc_n_iterations,
            lc_alpha_prior  = config.lc_alpha_prior,
            lc_convergence_tol = config.lc_convergence_tol,
            lc_winsorize    = config.lc_winsorize,
            lc_winsorize_quantiles = config.lc_winsorize_quantiles,
            # Robust regression
            regression_likelihood = config.regression_likelihood,
            student_t_nu = config.student_t_nu
        )

        # Create and cache AnalysisResult
        analysis_result = AnalysisResult(
            results.copula_results,
            results.df_hierarchical,
            results.em,
            results.joint_H0,
            results.joint_H1,
            results.latent_class_result,
            results.bma_result,
            results.combination_method,
            results.em_diagnostics,
            results.em_diagnostics_summary,
            compute_config_hash(config),
            compute_data_hash(imputed_data, raw_data),
            now(),
            string(pkgversion(@__MODULE__)),
            nothing,  # bait_protein - can be set by user later
            nothing,  # bait_index
            nothing,  # sensitivity
            nothing   # diagnostics
        )

        # Save to cache
        if use_cache
            try
                save_result(analysis_result, cache_path)
                @info "Saved results to cache: $cache_path"
            catch e
                @warn "Failed to save cache: $e"
            end
        end

        # Save convergence plot
        if config.combination_method in (:copula, :bma)
            StatsPlots.savefig(results.convergence_plt, config.output.convergence_file)
        elseif config.combination_method == :latent_class
            StatsPlots.savefig(results.convergence_plt, config.output.lc_convergence_file)
        end

        # Save EM diagnostics plot if available (copula or bma)
        if config.combination_method in (:copula, :bma) && config.run_em_diagnostics && results.em_diagnostics !== nothing
            try
                diag_plt = plot_em_diagnostics(results.em_diagnostics)
                StatsPlots.savefig(diag_plt, config.output.em_diagnostics_file)
                @info "Saved EM diagnostics plot to: $(config.output.em_diagnostics_file)"
            catch e
                @warn "Failed to save EM diagnostics plot: $e"
            end
        end

        # For BMA: also save latent class convergence plot
        if config.combination_method == :bma && !isnothing(results.latent_class_result)
            try
                lc_plt = plot_lc_convergence(results.latent_class_result)
                StatsPlots.savefig(lc_plt, config.output.lc_convergence_file)
                @info "Saved latent class convergence plot to: $(config.output.lc_convergence_file)"
            catch e
                @warn "Failed to save latent class convergence plot: $e"
            end
        end
    end

    # Generate plots even when loading from cache
    # (plots are lightweight to regenerate and may have different filenames)
    if !isnothing(analysis_result.em_diagnostics)
        try
            diag_plt = plot_em_diagnostics(analysis_result.em_diagnostics)
            StatsPlots.savefig(diag_plt, config.output.em_diagnostics_file)
            @info "Saved EM diagnostics plot to: $(config.output.em_diagnostics_file)"
        catch e
            @warn "Failed to save EM diagnostics plot: $e"
        end
    end

    # Prior sensitivity analysis
    if config.run_sensitivity
        try
            @info "Running prior sensitivity analysis..."
            sr = sensitivity_analysis(
                analysis_result, raw_data;
                config = config.sensitivity_config,
                n_controls = config.n_controls,
                n_samples = config.n_samples,
                refID = config.refID,
                H0_file = config.output.H0_file,
                combination_method = config.combination_method,
                lc_n_iterations = config.lc_n_iterations,
                lc_convergence_tol = config.lc_convergence_tol,
                verbose = config.verbose
            )
            analysis_result.sensitivity = sr

            # Generate sensitivity plots
            try
                sensitivity_tornado_plot(sr; file = config.output.sensitivity_tornado_file)
                @info "Saved sensitivity tornado plot to: $(config.output.sensitivity_tornado_file)"
            catch e
                @warn "Failed to save sensitivity tornado plot: $e"
            end
            try
                sensitivity_heatmap(sr; file = config.output.sensitivity_heatmap_file)
                @info "Saved sensitivity heatmap to: $(config.output.sensitivity_heatmap_file)"
            catch e
                @warn "Failed to save sensitivity heatmap: $e"
            end
            try
                sensitivity_rank_correlation(sr; file = config.output.sensitivity_rankcorr_file)
                @info "Saved sensitivity rank correlation plot to: $(config.output.sensitivity_rankcorr_file)"
            catch e
                @warn "Failed to save sensitivity rank correlation plot: $e"
            end

            generate_sensitivity_report(sr;
                filename = config.output.sensitivity_report_file,
                tornado_file = config.output.sensitivity_tornado_file,
                heatmap_file = config.output.sensitivity_heatmap_file,
                rankcorr_file = config.output.sensitivity_rankcorr_file
            )
            @info "Sensitivity report saved to: $(config.output.sensitivity_report_file)"

            try
                writetable(config.output.sensitivity_table_file, "sensitivity" => sr.summary)
                @info "Sensitivity table saved to: $(config.output.sensitivity_table_file)"
            catch e
                @warn "Failed to save sensitivity table: $e"
            end
        catch e
            @warn "Prior sensitivity analysis failed: $e"
        end
    end

    # Posterior predictive checks & model diagnostics
    if config.run_diagnostics
        try
            @info "Running posterior predictive checks & model diagnostics..."
            # Compute τ_base for robust regression diagnostics (Empirical Bayes)
            diag_tau_base = config.regression_likelihood == :robust_t ? estimate_regression_tau_base(raw_data, config.refID) : NaN
            dr = model_diagnostics(
                analysis_result, raw_data;
                config = config.diagnostics_config,
                n_controls = config.n_controls,
                n_samples = config.n_samples,
                refID = config.refID,
                verbose = config.verbose,
                regression_likelihood = config.regression_likelihood,
                student_t_nu = config.student_t_nu,
                robust_tau_base = diag_tau_base
            )
            analysis_result.diagnostics = dr

            # Generate diagnostic plots
            try
                ppc_plt = ppc_pvalue_histogram(dr; file = config.output.ppc_histogram_file)
                @info "Saved PPC histogram to: $(config.output.ppc_histogram_file)"
            catch e
                @warn "Failed to save PPC histogram: $e"
            end
            if !isnothing(dr.hbm_residuals)
                try
                    residual_qq_plot(dr.hbm_residuals; file = config.output.qq_plot_file)
                    @info "Saved HBM Q-Q plot to: $(config.output.qq_plot_file)"
                catch e
                    @warn "Failed to save HBM Q-Q plot: $e"
                end
            end
            if !isnothing(dr.regression_residuals)
                try
                    residual_qq_plot(dr.regression_residuals; file = config.output.regression_qq_plot_file)
                    @info "Saved regression Q-Q plot to: $(config.output.regression_qq_plot_file)"
                catch e
                    @warn "Failed to save regression Q-Q plot: $e"
                end
            end
            if !isnothing(dr.hbm_residuals)
                try
                    scale_location_plot(dr.hbm_residuals; file = config.output.scale_location_hbm_file)
                    @info "Saved HBM scale-location plot to: $(config.output.scale_location_hbm_file)"
                catch e
                    @warn "Failed to save HBM scale-location plot: $e"
                end
            end
            if !isnothing(dr.regression_residuals)
                try
                    scale_location_plot(dr.regression_residuals; file = config.output.scale_location_regression_file)
                    @info "Saved regression scale-location plot to: $(config.output.scale_location_regression_file)"
                catch e
                    @warn "Failed to save regression scale-location plot: $e"
                end
            end
            if !isnothing(dr.calibration)
                try
                    calibration_plot(dr.calibration; file = config.output.calibration_plot_file)
                    @info "Saved calibration plot to: $(config.output.calibration_plot_file)"
                catch e
                    @warn "Failed to save calibration plot: $e"
                end
            end
            try
                calibration_comparison_plot(dr; file = config.output.calibration_comparison_file)
                @info "Saved calibration comparison plot to: $(config.output.calibration_comparison_file)"
            catch e
                @warn "Failed to save calibration comparison plot: $e"
            end

            # Save PIT histogram when enhanced residuals are present
            pit_values = Float64[]
            if !isnothing(dr.enhanced_hbm_residuals)
                append!(pit_values, dr.enhanced_hbm_residuals.pit_values)
            end
            if !isnothing(dr.enhanced_regression_residuals)
                append!(pit_values, dr.enhanced_regression_residuals.pit_values)
            end
            if !isempty(pit_values)
                try
                    pit_histogram_plot(pit_values; file = config.output.pit_histogram_file)
                    @info "Saved PIT histogram to: $(config.output.pit_histogram_file)"
                catch e
                    @warn "Failed to save PIT histogram: $e"
                end
            end

        catch e
            @warn "Model diagnostics failed: $e"
        end
    end

    # WAIC model comparison (Normal vs. robust regression) — uses first imputed dataset
    # Skip when optimize_nu is enabled — the optimization already computes Normal WAIC as baseline
    if config.run_model_comparison && !config.optimize_nu
        try
            @info "Running WAIC model comparison (Normal vs. robust regression)..."
            model_cmp = _run_model_comparison(imputed_data[1], config)
            if !isnothing(analysis_result.diagnostics)
                dr = analysis_result.diagnostics
                analysis_result.diagnostics = DiagnosticsResult(
                    dr.config, dr.protein_ppcs, dr.bb_ppcs,
                    dr.hbm_residuals, dr.regression_residuals,
                    dr.calibration, dr.calibration_relaxed, dr.calibration_enrichment_only,
                    dr.enhanced_hbm_residuals, dr.enhanced_regression_residuals,
                    dr.ppc_extended, dr.protein_flags,
                    model_cmp, dr.nu_optimization,
                    dr.summary, dr.timestamp
                )
            end
            @info "WAIC comparison complete: preferred model = $(model_cmp.preferred_model), ΔWAIC = $(round(model_cmp.delta_waic, digits=1)) ± $(round(model_cmp.delta_se, digits=1))"
        catch e
            @warn "WAIC model comparison failed: $e"
        end
    end

    # ν optimization (uses first imputed dataset, consistent with _run_model_comparison)
    if config.optimize_nu && config.regression_likelihood == :robust_t
        try
            @info "Running Student-t ν optimization via Brent's method..."
            nu_opt = optimize_nu(imputed_data[1], config)
            config.student_t_nu = nu_opt.optimal_nu
            @info "Optimal ν = $(round(nu_opt.optimal_nu, digits=2)), WAIC = $(round(nu_opt.optimal_waic.waic, digits=1))"

            # Save ν optimization plot
            try
                nu_optimization_plot(nu_opt; file = config.output.nu_optimization_file)
                @info "Saved ν optimization plot to: $(config.output.nu_optimization_file)"
            catch e
                @warn "Failed to save ν optimization plot: $e"
            end

            # Build ModelComparisonResult from nu_opt for backward compat
            model_cmp_from_nu = ModelComparisonResult(
                nu_opt.normal_waic, nu_opt.optimal_waic,
                nu_opt.delta_waic, nu_opt.delta_se,
                nu_opt.delta_waic > 0 ? :robust : :normal
            )

            # Attach to diagnostics if available
            if !isnothing(analysis_result.diagnostics)
                dr = analysis_result.diagnostics
                analysis_result.diagnostics = DiagnosticsResult(
                    dr.config, dr.protein_ppcs, dr.bb_ppcs,
                    dr.hbm_residuals, dr.regression_residuals,
                    dr.calibration, dr.calibration_relaxed, dr.calibration_enrichment_only,
                    dr.enhanced_hbm_residuals, dr.enhanced_regression_residuals,
                    dr.ppc_extended, dr.protein_flags,
                    model_cmp_from_nu, nu_opt,
                    dr.summary, dr.timestamp
                )
            end
        catch e
            @warn "Student-t ν optimization failed: $e"
        end
    end

    # Generate diagnostics report AFTER model comparison and ν optimization
    # so the report includes all computed results
    if config.run_diagnostics && !isnothing(analysis_result.diagnostics)
        try
            generate_diagnostics_report(analysis_result.diagnostics;
                filename = config.output.diagnostics_report_file,
                ppc_histogram_file = config.output.ppc_histogram_file,
                qq_plot_file = config.output.qq_plot_file,
                regression_qq_plot_file = config.output.regression_qq_plot_file,
                calibration_plot_file = config.output.calibration_plot_file,
                calibration_comparison_file = config.output.calibration_comparison_file,
                pit_histogram_file = config.output.pit_histogram_file,
                scale_location_hbm_file = config.output.scale_location_hbm_file,
                scale_location_regression_file = config.output.scale_location_regression_file,
                nu_optimization_file = config.output.nu_optimization_file
            )
            @info "Diagnostics report saved to: $(config.output.diagnostics_report_file)"
        catch e
            @warn "Failed to generate diagnostics report: $e"
        end
    end

    # Always run meta-learner (poi may change between runs)
    meta_data, _ = predict_metalearner(config.poi, output_file = config.output.prior_file, metalearner_file = config.metalearner_path)

    # Validate metalearner prediction succeeded
    if isnothing(meta_data)
        error("Metalearner prediction failed. Check errors above for details.")
    end

    # Update posterior probabilities with meta-learner predictions
    final_results = copy(analysis_result.copula_results)  # Copy to avoid mutating cached data
    final_results = update_posterior_prob!(final_results, meta_data)

    # Update q-values
    final_results.q = q(final_results.posterior_prob, isBF = false)

    # Sort dataframe by Bayesian FDR (q-value) and log2FC
    sort!(final_results, [:q, :mean_log2FC], rev = [false, false])

    # Merge diagnostic and sensitivity columns into final results
    if !isnothing(analysis_result.diagnostics) || !isnothing(analysis_result.sensitivity)
        final_results = _merge_diagnostics_to_results(
            final_results, analysis_result.diagnostics;
            sensitivity = analysis_result.sensitivity
        )
    end

    # Add curation metadata (original names and STRING IDs) as columns 2–3
    curation_lookup = _build_curation_lookup(config)
    if !isnothing(curation_lookup)
        final_results = leftjoin(final_results, curation_lookup, on = :Protein)
        other_cols = setdiff(names(final_results), ["Protein", "original_name", "string_id"])
        select!(final_results, "Protein", "original_name", "string_id", other_cols...)
    end

    # Update AnalysisResult with metalearner-updated results (includes diagnostic columns)
    # This ensures ar.copula_results (accessed via ar.results) has final posterior probabilities
    analysis_result.copula_results = final_results

    # Save final results
    writetable(config.output.results_file, "df" => final_results)

    # Generate plots (always regenerate for current run)
    volcano_plt = volcano_plot(final_results, legend_pos = config.vc_legend_pos)
    StatsPlots.savefig(volcano_plt, config.output.volcano_file)

    rrp = rank_rank_plot(final_results)
    StatsPlots.savefig(rrp, config.output.rank_rank_file)

    evd_plot = evidence_plot(final_results)
    StatsPlots.savefig(evd_plot, config.output.evidence_file)

    # Generate interactive HTML report
    if config.generate_report_html
        try
            generate_report(final_results, config)
        catch e
            @warn "Failed to generate interactive HTML report: $e"
        end
    end

    return final_results, analysis_result
end
