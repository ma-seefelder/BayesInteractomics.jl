# Additional imports needed for analysis pipeline
import DataFrames: innerjoin, leftjoin
# removed `import Flux` — unused in this file (verified by grep "Flux\." returns 0 hits).
# Flux is a [weakdeps] dep; the eager Flux import previously here broke standalone
# `using BayesInteractomics` precompilation when Flux was not in the active loadpath.
using LogExpFunctions

# Note: DNN and metalearner functionality will be added separately
# include("../dnn/generate_dataset.jl")
# include("../dnn/model.jl")
# include("../ml/metalearner.jl")
# include("../visualization/plotting.jl")

"""
    check_bait_detected(data::InteractionData, refID::Integer)

Check that the bait protein (at index `refID`) was detected in at least one sample.
Throws an `ErrorException` if the bait is not detected, which would make
dose-response correlation analysis meaningless.
"""
function check_bait_detected(data::InteractionData, refID::Integer)
    if !data.detected[refID]
        error("Bait protein at index $refID ($(data.protein_IDs[refID])) was not detected in any sample. " *
              "Correlation tests require bait dose-response signal. Aborting analysis.")
    end
end

"""
    _safe_predict_metalearner(config) -> (meta_data, model, embedding_matrix, status::Symbol)

Variante B fallback wrapper around `predict_metalearner`.

When the metalearner extension (`BayesInteractomicsMetalearnerExt`) is not
loaded the stub function has zero methods and any call raises a `MethodError`.
This helper catches that error, emits a one-shot `@warn` pointing the user at
the trigger packages, and returns the sentinel value `:extension_not_loaded`.
Any other exception inside `predict_metalearner` is logged and translated to
`:prediction_failed`. A successful call that returns `nothing` (POI likely
renamed by STRING curation) also maps to `:prediction_failed`.

expanded to a 4-tuple to carry the
`embedding_matrix::Matrix{Float32}` constructed inside `predict_metalearner`.
The embedding matrix is consumed by `_safe_compute_mc_prior!` for K-pass
MC-Dropout uncertainty quantification on the DNN prior; reusing it avoids a
second 1.5 GB HDF5 + ~6 s cold load. `nothing` in this slot on any fallback path.

Returns a tuple `(predictions::Union{DataFrame,Nothing}, model::Any,
embedding_matrix::Union{Matrix{Float32},Nothing}, status::Symbol)` where
`status ∈ METALEARNER_STATUS_VALUES` (`:loaded`, `:extension_not_loaded`,
`:prediction_failed`).
"""
function _safe_predict_metalearner(config)
    if config.metalearner_use_mc_dropout
        @warn("CONFIG.metalearner_use_mc_dropout = true is deprecated as of v1.2.1. " *
              "The non-MC :tr_ddi schema outperforms :tr_ddi_mc on multi-species data " *
              "(AUC +0.019, MCC +0.024). Set `metalearner_use_mc_dropout = false` (the new default) " *
              "or omit the field. MC-Dropout is retained for non-human-focused use only.",
              maxlog=1)
    end
    local meta_data, model, embedding_matrix
    try
        # derive the per-species STRING input filenames
        # from config.species and forward them explicitly so a non-human bait reads
        # real per-species channels. config.species == 9606 reproduces the legacy
        # human literals byte-for-byte. String interpolation into the same
        # fixed `encodings/<sp>.<suffix>` templates; the typed Int species prevents
        # path injection (threat T-77.2-05-PATH).
        sp = string(config.species)
        meta_data, model, embedding_matrix = predict_metalearner(config.poi;
                                               species = config.species,
                                               embeddings_seq = "encodings/$(sp).protein.sequence.embeddings.v12.0.h5",
                                               embeddings_net = "encodings/$(sp).protein.network.embeddings.v12.0.h5",
                                               links = "encodings/$(sp).protein.links.detailed.v12.0.onlyAB.txt",
                                               protein_info = "encodings/$(sp).protein.info.v12.0.txt",
                                               output_file = config.output.prior_file,
                                               metalearner_file = config.metalearner_path,
                                               use_mc_dropout = config.metalearner_use_mc_dropout,
                                               # forward the per-species staging dir so a NON-HUMAN
                                               # bait featurises real TR/DDI channels instead of zero-imputing. For
                                               # human (sp=9606) predict_metalearner gates this out (byte-identical);
                                               # for a non-human bait without staged files, featurise_pairs_onthefly
                                               # throws → caught → graceful human-lookup/zero fallback.
                                               feature_source_dir = "encodings")
    catch e
        # (contract change): the MC-Dropout hard-error guard
        # was removed (MC rides the metalearner's existing DNN dependency). A residual
        # ArgumentError here can now only originate from a genuine schema-config error
        # (e.g. an invalid `schema_tag` or a schema-column mismatch on a mis-pinned
        # `metalearner_path`). Such real misconfiguration MUST propagate rather than be
        # silently swallowed by Variante B's :prediction_failed sentinel.
        if e isa ArgumentError
            rethrow(e)
        end
        # Detect "stub has zero methods" MethodError. With keyword arguments the
        # MethodError's .f is Core.kwcall and the underlying function appears in
        # e.args[2]; without kwargs it would be e.f directly. Match both forms.
        is_stub_methoderror = e isa MethodError && (
            e.f === predict_metalearner ||
            (length(e.args) >= 2 && e.args[2] === predict_metalearner)
        )
        if is_stub_methoderror
            @warn "Metalearner extension (BayesInteractomicsMetalearnerExt) not loaded; falling back to BF-derived posterior_prob. To enable ML-adjusted posteriors, run `using Flux, MLJ, MLJScikitLearnInterface, HDF5` before `run_analysis`." maxlog=1
            return (nothing, nothing, nothing, :extension_not_loaded)
        else
            @warn "Metalearner prediction failed; falling back to BF-derived posterior_prob." exception=(e, catch_backtrace()) maxlog=1
            return (nothing, nothing, nothing, :prediction_failed)
        end
    end
    if isnothing(meta_data)
        # Extension loaded but returned nothing (e.g., POI renamed by STRING curation)
        @warn "Metalearner returned no predictions (POI likely renamed by STRING curation); using BF-derived posterior_prob." maxlog=1
        return (nothing, model, nothing, :prediction_failed)
    end
    return (meta_data, model, embedding_matrix, :loaded)
end

"""
    _safe_compute_mc_prior!(df, embedding_matrix, config) -> Symbol

Variante-B fallback wrapper around `compute_mc_prior!`.

Mirrors `_safe_predict_metalearner` Variante-B discipline.
Returns one of `METALEARNER_STATUS_VALUES`-style sentinels:

- `:skipped` — `config.run_dnn_prior_mc_dropout == false` OR
  `embedding_matrix === nothing` (upstream metalearner failed). 5 NaN columns
  populated; no `@warn` emitted (opt-out path is silent).
- `:extension_not_loaded` — `compute_mc_prior!` stub has zero methods (extension
  not loaded). One `@warn maxlog=1` emitted pointing the user at the trigger
  packages; 5 NaN columns populated; `posterior_prob` untouched.
- `:compute_failed` — extension loaded but `compute_mc_prior!` threw a non-stub
  exception. One `@warn maxlog=1` with the captured exception; 5 NaN columns
  populated; `posterior_prob` untouched.
- `:loaded` — extension loaded AND `compute_mc_prior!` populated the 5 prior
  columns with finite values.

The 5 columns written on every path are: `prior_mc_mean`, `prior_mc_std`,
`prior_mc_ci_low`, `prior_mc_ci_high`, `prior_contribution`. Schema uniformity
is preserved across all four return sentinels per byte-equality contract
(posterior_prob never mutated by this wrapper).
"""
function _safe_compute_mc_prior!(df, embedding_matrix, config; protein_names = nothing)
    if !config.run_dnn_prior_mc_dropout
        _populate_mc_prior_nan_columns!(df)
        return :skipped
    end
    if isnothing(embedding_matrix)
        _populate_mc_prior_nan_columns!(df)
        return :skipped
    end
    try
        compute_mc_prior!(df, embedding_matrix, config;
                          K = config.dnn_prior_mc_k,
                          batch_size = config.dnn_prior_mc_batch_size,
                          protein_names = protein_names)
        return :loaded
    catch e
        # Mirror the kwcall MethodError detection from _safe_predict_metalearner lines 70-73:
        # with keyword arguments, MethodError.f is Core.kwcall and the underlying function
        # appears in e.args[2]; without kwargs it would be e.f directly.
        is_stub_methoderror = e isa MethodError && (
            e.f === compute_mc_prior! ||
            (length(e.args) >= 2 && e.args[2] === compute_mc_prior!)
        )
        if is_stub_methoderror
            @warn "Metalearner extension (BayesInteractomicsMetalearnerExt) not loaded; MC-Dropout prior columns will be NaN. Run `using Flux, MLJ, MLJScikitLearnInterface, HDF5` to enable." maxlog=1
            _populate_mc_prior_nan_columns!(df)
            return :extension_not_loaded
        else
            @warn "MC-Dropout prior computation failed; columns will be NaN." exception=(e, catch_backtrace()) maxlog=1
            _populate_mc_prior_nan_columns!(df)
            return :compute_failed
        end
    end
end

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
    # find proteins with BF = 0 or missing (treat missing BF as zero-like)
    bf_is_zero = findall(x -> coalesce(x == 0, true), df.BF)
    # update posterior probability — coalesce missing BF and MetaClassifier to safe defaults
    bf_safe = coalesce.(df.BF, 0.0)
    mc_safe = coalesce.(df.MetaClassifier, 0.5)
    prior_odds = mc_safe ./ (1 .- mc_safe)
    posterior_odds = prior_odds .* bf_safe
    new_pp = posterior_odds ./ (1 .+ posterior_odds)
    new_pp[bf_is_zero] .= 0.0
    # Convert to Union{Missing, Float64} so non-detected proteins can hold missing
    pp_col = Vector{Union{Missing, Float64}}(new_pp)
    if hasproperty(df, :is_detected)
        for i in 1:nrow(df)
            if !coalesce(df.is_detected[i], false)
                pp_col[i] = missing
            end
        end
    end
    df.posterior_prob = pp_col
    # sort dataframe by decreasing posterior probability (missing sorts last)
    df = df[sortperm(df.posterior_prob, rev = true, lt = (a, b) -> coalesce(a < b, false)), :]
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

# ----- helper: build a single (proteins × columns) intensity matrix --- #
# from an InteractionData by concatenating sample + control columns across all
# protocols/experiments. Used by the :inflation arm of analyse(...) to derive
# per-protein missingness masks and per-column σ̂². Missings (and short rows from
# protocols with fewer replicates) are represented as NaN so the downstream
# `isnan` filter inside _compute_inflation_factor_protein handles them uniformly.
#
# IMPORTANT (anti-pattern from): dropout fit
# was run on the SAME canonical column layout (per-column missingness across the
# analysis matrix); the helper here MUST mirror column ordering or
# the per-column σ̂² ↔ (ρ̂_c, ζ̂_c) alignment is broken. This v1 helper concatenates
# protocol-by-protocol, sample-block then control-block, experiments in order,
# replicates left-to-right — same order as `fit_dropout_curves`.
function _build_intensity_matrix_for_inflation(data::InteractionData)
    n_proteins = length(getIDs(data))
    # Count total columns: for each protocol p, sum 2 * (n_experiments[p]) * replicate_count
    # (samples + controls, replicates from each experiment matrix).
    col_count = 0
    for p in 1:data.no_protocols
        sp = data.samples[p]
        cp = data.controls[p]
        for e in 1:sp.no_experiments
            col_count += size(sp.data[e], 2)
        end
        for e in 1:cp.no_experiments
            col_count += size(cp.data[e], 2)
        end
    end

    M = fill(NaN, n_proteins, col_count)
    col = 0
    for p in 1:data.no_protocols
        sp = data.samples[p]
        for e in 1:sp.no_experiments
            mat = sp.data[e]
            for r in 1:size(mat, 2)
                col += 1
                for i in 1:n_proteins
                    v = mat[i, r]
                    M[i, col] = ismissing(v) ? NaN : Float64(v)
                end
            end
        end
        cp = data.controls[p]
        for e in 1:cp.no_experiments
            mat = cp.data[e]
            for r in 1:size(mat, 2)
                col += 1
                for i in 1:n_proteins
                    v = mat[i, r]
                    M[i, col] = ismissing(v) ? NaN : Float64(v)
                end
            end
        end
    end
    return M
end

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
    h0_cache_file::String = "",
    # Copula-EM parameters
    prior::Union{Symbol, NamedTuple} = :default,
    n_restarts::Int = 20,
    copula_criterion::Symbol = :BIC,
    copula_family::Union{Nothing, Type} = nothing,
    h1_copula_family::Union{Nothing, Type} = nothing,
    streams::Vector{Symbol} = [:enrichment, :correlation, :detection],
    h1_refitting::Bool = true,
    burn_in::Int = 10,
    # Diagnostics
    run_em_diagnostics::Bool = true,
    # Evidence combination method
    combination_method::Symbol = :bma,
    # Latent class parameters
    lc_n_iterations::Int = 100,
    lc_alpha_prior::Union{Symbol, Vector{Float64}} = :auto,
    lc_convergence_tol::Float64 = 1e-6,
    lc_winsorize::Bool = true,
    lc_winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99),
    # Robust regression parameters
    regression_likelihood::Symbol = :normal,
    student_t_nu::Float64 = 5.0,
    regression_bf_threshold::Float64 = 0.1,
    # JZS prior
    jzs_r_scale::Float64 = 0.0,
    # Regression posterior variance floor
    regression_min_posterior_var::Float64 = 0.0,
    # Imputation method (forwarded to cache validation/construction)
    imputation::Symbol = :mnar,
    # optional MNAR-driven post-hoc variance inflation
    variance_recovery::Symbol = :off,
    dropout_fit::Union{Nothing, DropoutFit} = nothing,
    inflation_max::Float64 = 3.0,
    inflation_override::Union{Nothing, Float64} = nothing,
    # bb_mnar_codriven diagnostic thresholds (must be passed
    # through from run_analysis; the inner `analyse` has no `config` in scope).
    bb_mnar_codriven::BBMnarCodrivenConfig = BBMnarCodrivenConfig(),
    # NEW: mask-aware regression flag + raw_data handle.
    # `raw_data` is the pre-imputation data passed by the multi-imputation overload of
    # run_analysis; the single-imputation path leaves it `nothing` (algebraic-collapse
    # path through v2b wrappers; is_imputed is all-false).
    mask_aware_regression::Bool = true,
    raw_data::Union{Nothing, InteractionData} = nothing,
)

    # generate cache folder
    ispath("cache") && rm("cache", recursive = true)
    mkpath("cache")
    # get number of proteins
    n_proteins = length(getIDs(data))

    # Check bait detection (hard error — bait must be present for correlation tests)
    check_bait_detected(data, refID)

    # Compute detection indices for downstream filtering
    detected = data.detected
    detected_indices = findall(detected)
    n_detected = length(detected_indices)
    @info "Excluded $(n_proteins - n_detected)/$n_proteins proteins (not detected in any sample)"

    # load H0 file or recompute it (skip for latent_class-only mode; copula and bma need it)
    if combination_method in (:copula, :bma)
        H0 = nothing
        # Fall back to legacy XLSX file
        if isnothing(H0) && !isnothing(H0_file) && isfile(H0_file)
            if endswith(H0_file, ".xlsx")
                @warn "Loading H0 from legacy XLSX file '$H0_file'. Consider deleting it to use the new JLD2 cache with parameter validation."
            end
            H0 = DataFrame(readtable(H0_file, "Sheet1", first_row = 1))
        end

        # Compute from scratch
        if isnothing(H0)
            H0 = computeH0_BayesFactors(
                data,
                n_controls = n_controls, n_samples = n_samples,
                refID = refID,
                regression_likelihood = regression_likelihood,
                student_t_nu = student_t_nu,
                regression_bf_threshold = regression_bf_threshold,
                jzs_r_scale = jzs_r_scale,
                regression_min_posterior_var = regression_min_posterior_var,
                detected_mask = data.detected
            )
        end
    end

    # ------------------------------------ #
    # Beta-Bernoulli model
    # ------------------------------------ #
    bf_detected = zeros(Float64, n_proteins)
    bb_cache_used = false

    # Check Beta-Bernoulli cache
    if use_intermediate_cache && !isempty(betabernoulli_cache_file)
        bb_status, bb_cached = check_betabernoulli_cache(betabernoulli_cache_file, data, n_controls, n_samples, imputation)
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
            if !detected[i]
                bf_detected[i] = 0.0
                ProgressMeter.next!(p)
                continue
            end
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
                    string(pkgversion(@__MODULE__)),
                    imputation
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
        hbm_status, hbm_cached = check_hbm_regression_cache(hbm_regression_cache_file, data, refID, regression_likelihood, student_t_nu, regression_bf_threshold, imputation)
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

        cached_hbm_prior = precompute_enrichment_prior(data; μ_0=μ_0, σ_0=σ_0, a_0=a_0, b_0=b_0)
        if getNoProtocols(data) == 1
            if regression_likelihood == :robust_t
                if jzs_r_scale > 0.0
                    cached_regression_prior = precompute_regression_one_protocol_robust_jzs_prior(data, refID, μ_0, σ_0; nu=student_t_nu, τ_base=robust_tau_base, jzs_r_scale=jzs_r_scale)
                else
                    cached_regression_prior = precompute_regression_one_protocol_robust_prior(data, refID, μ_0, σ_0; nu=student_t_nu, τ_base=robust_tau_base)
                end
            else
                cached_regression_prior = precompute_regression_one_protocol_prior(data, refID, μ_0, σ_0)
            end
        else
            if regression_likelihood == :robust_t
                if jzs_r_scale > 0.0
                    cached_regression_prior = precompute_regression_multi_protocol_robust_jzs_prior(data, refID, μ_0, σ_0; nu=student_t_nu, τ_base=robust_tau_base, jzs_r_scale=jzs_r_scale)
                else
                    cached_regression_prior = precompute_regression_multi_protocol_robust_prior(data, refID, μ_0, σ_0; nu=student_t_nu, τ_base=robust_tau_base)
                end
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
            if !detected[i]
                ProgressMeter.next!(p)
                continue
            end
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
                    robust_tau_base = robust_tau_base,
                    regression_bf_threshold = regression_bf_threshold,
                    jzs_r_scale = jzs_r_scale,
                    global_tau_base = robust_tau_base,
                    regression_min_posterior_var = regression_min_posterior_var,
                    # NEW: mask-aware regression dispatch
                    mask_aware_regression = mask_aware_regression,
                    raw_data = raw_data,
                    dropout_fit = dropout_fit,
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

        # --------------------------------------------------------------------- #
        # Optional post-hoc MNAR variance inflation
        # Widens the per-protein log2FC σ (and 95% CI bounds) in `df` BEFORE the
        # BF/PEP/BFDR columns are recomputed downstream. HBM/RxInfer factor graph
        # is NEVER touched (hard-lock).
        # --------------------------------------------------------------------- #
        if variance_recovery == :off
            # No-op; default behaviour
        elseif variance_recovery == :inflation
            dropout_fit === nothing && throw(ArgumentError(
                "variance_recovery = :inflation requires a non-nothing `dropout_fit` " *
                "kwarg. (run_analysis loads it from imputed_data/dropout_curves.json; " *
                "direct callers of analyse() must pass it explicitly.)"
            ))
            # Reconstruct the intensity matrix from the data (use the first protocol's
            # combined sample+control columns — is per-column missingness
            # across the analysis matrix, not per-replicate).
            intensity_matrix = _build_intensity_matrix_for_inflation(data)
            n_cols = size(intensity_matrix, 2)
            @assert n_cols == length(dropout_fit.rho) "intensity matrix has $n_cols cols but dropout_fit has $(length(dropout_fit.rho)) — the dropout fit was run on a different dataset shape; re-run fit_dropout_curves"
            # Per-column σ̂² across all proteins, computed once
            sigma_sq_per_column = Vector{Float64}(undef, n_cols)
            for c in 1:n_cols
                col_vals = collect(skipmissing(intensity_matrix[:, c]))
                col_vals = filter(x -> !(x isa Number && isnan(Float64(x))), col_vals)
                sigma_sq_per_column[c] = length(col_vals) >= 2 ? var(Float64.(col_vals)) : 1.0
            end
            rho_zeta = [(dropout_fit.rho[c], dropout_fit.zeta[c]) for c in 1:n_cols]
            protein_ids = getIDs(data)
            inflated_factors = Float64[]
            for row_idx in 1:nrow(df)
                protein = df.Protein[row_idx]
                i = findfirst(==(protein), protein_ids)
                i === nothing && continue
                factor = if inflation_override !== nothing
                    max(1.0, Float64(inflation_override))
                else
                    row = view(intensity_matrix, i, :)
                    mm = [ismissing(x) || (x isa Number && isnan(Float64(x))) for x in row]
                    _compute_inflation_factor_protein(mm, rho_zeta, sigma_sq_per_column, inflation_max)
                end
                push!(inflated_factors, factor)
                sqrt_f = sqrt(factor)
                # σ widening — WARNING: MUST throw if no σ column matches (no silent no-op)
                # Note: the running codebase emits :sd_log2FC from clean_result()
                # (src/inference/models.jl). The textbook list (:log2FC_std, :σ, :log2FC_sigma)
                # is kept for forward-compat per the plan's acceptance criterion.
                σ_col_matched = nothing
                for σ_col in (:sd_log2FC, :log2FC_std, :σ, :log2FC_sigma)
                    if hasproperty(df, σ_col)
                        df[row_idx, σ_col] = df[row_idx, σ_col] * sqrt_f
                        σ_col_matched = σ_col
                        break
                    end
                end
                if σ_col_matched === nothing
                    error(":inflation expected one of [:log2FC_std, :σ, :log2FC_sigma, :sd_log2FC] in clean_result output; got columns: $(propertynames(df))")
                end
                # CI re-derivation
                μ_col = hasproperty(df, :log2FC_mean) ? :log2FC_mean :
                        (hasproperty(df, :mean_log2FC) ? :mean_log2FC :
                         (hasproperty(df, :μ) ? :μ : nothing))
                lo_col = hasproperty(df, :log2FC_CI_lower) ? :log2FC_CI_lower : nothing
                hi_col = hasproperty(df, :log2FC_CI_upper) ? :log2FC_CI_upper : nothing
                if μ_col !== nothing && lo_col !== nothing && hi_col !== nothing
                    μv = df[row_idx, μ_col]; σv = df[row_idx, σ_col_matched]
                    df[row_idx, lo_col] = μv - 1.96 * σv
                    df[row_idx, hi_col] = μv + 1.96 * σv
                end
            end
            # Diagnostics — captured for the Methods-tab report (reads this)
            if !isempty(inflated_factors)
                n_capped = count(==(inflation_max), inflated_factors)
                @info ":inflation applied" mode=variance_recovery n_proteins=length(inflated_factors) median_factor=median(inflated_factors) p95_factor=quantile(inflated_factors, 0.95) max_factor=maximum(inflated_factors) n_capped=n_capped
                if n_capped >= 0.10 * length(inflated_factors)
                    @warn maxlog=1 "$(n_capped)/$(length(inflated_factors)) proteins hit mnar_inflation_max cap = $inflation_max. Consider raising mnar_inflation_max or reviewing dropout curves."
                end
            end
        elseif variance_recovery == :multi_impute
            throw(ArgumentError(
                "variance_recovery = :multi_impute should be dispatched at run_analysis level " *
                "into the Vector{InteractionData} overload. Reaching here is a bug."
            ))
        else
            # BLOCKER 3: terminal else — unknown symbols MUST throw, never silently no-op
            throw(ArgumentError(
                "variance_recovery must be one of (:off, :inflation, :multi_impute), got :$variance_recovery"
            ))
        end

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
                    regression_bf_threshold,
                    compute_data_hash(data),
                    now(),
                    string(pkgversion(@__MODULE__)),
                    imputation
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
    # Remap refID to detected-only df index
    # df only contains detected proteins (non-detected were skipped in HBM loop and excluded by innerjoin)
    # Find position of the bait protein name in df.Protein
    # ------------------------------------ #
    bait_protein_name = getIDs(data)[refID]
    refID_detected = findfirst(==(bait_protein_name), df.Protein)
    if isnothing(refID_detected)
        error("Bait protein '$bait_protein_name' (refID=$refID) not found in HBM results DataFrame. " *
              "This should not happen if check_bait_detected passed. Please report this bug.")
    end

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

    # bf_enrichment, bf_correlation, bf_detected are already detected-only (from df, which excludes non-detected)
    bf_triplet = BayesFactorTriplet(bf_enrichment, bf_correlation, bf_detected)

    # ALWAYS run stage-1 EM first (all combination methods share this)
    @info "Running stage-1 3-component EM..."
    phase1_result = combined_BF_latent_class(bf_triplet, refID_detected;
        use_3c = true,
        alpha_prior = lc_alpha_prior,
        n_iterations = lc_n_iterations,
        convergence_tol = lc_convergence_tol,
        verbose = verbose,
        winsorize = lc_winsorize,
        winsorize_quantiles = lc_winsorize_quantiles,
        protein_names = String.(df.Protein))

    if combination_method == :copula
        @info "Starting copula-EM combination"

        combinedResult = combined_BF(
            bf_triplet, refID_detected;
            phase1_result = phase1_result,
            n_restarts = n_restarts,
            copula_criterion = copula_criterion,
            copula_family = copula_family,
            h1_copula_family = h1_copula_family,
            streams = streams,
            burn_in = burn_in,
            verbose = verbose
        )

        combined_bf = combinedResult.bf
        posterior_prob = combinedResult.posterior_prob
        em_result = combinedResult.em_result
        joint_H0 = combinedResult.joint_H0
        joint_H1 = combinedResult.joint_H1
        latent_class_result = phase1_result
        convergence_plt = EMconvergenceDiagnosticPlot(em_result)

        # Process EM diagnostics
        em_diagnostics = combinedResult.em_diagnostics
        if run_em_diagnostics && !isnothing(em_diagnostics)
            @info "Summarizing EM restart diagnostics..."
            em_diagnostics_summary = summarize_em_diagnostics(em_diagnostics)
        end

    elseif combination_method == :latent_class
        @info "Using stage-1 latent class result directly"

        latent_class_result = phase1_result

        combined_bf = latent_class_result.bf
        posterior_prob = latent_class_result.posterior_prob
        convergence_plt = plot_lc_convergence(latent_class_result)

        # Process EM diagnostics (latent class path)
        em_diagnostics = latent_class_result.em_diagnostics
        if run_em_diagnostics && !isnothing(em_diagnostics)
            @info "Summarizing EM restart diagnostics..."
            em_diagnostics_summary = summarize_em_diagnostics(em_diagnostics)
        end

    elseif combination_method == :bma
        @info "Starting Bayesian Model Averaging (BMA) combination..."

        bma_result = combined_BF_bma(
            bf_triplet, refID_detected;
            phase1_result = phase1_result,
            n_restarts = n_restarts,
            copula_criterion = copula_criterion,
            copula_family = copula_family,
            h1_copula_family = h1_copula_family,
            streams = streams,
            burn_in = burn_in,
            lc_n_iterations = lc_n_iterations,
            lc_alpha_prior = lc_alpha_prior,
            lc_convergence_tol = lc_convergence_tol,
            lc_winsorize = lc_winsorize,
            lc_winsorize_quantiles = lc_winsorize_quantiles,
            verbose = verbose,
            protein_names = String.(df.Protein)
        )

        combined_bf = bma_result.bf
        posterior_prob = bma_result.posterior_prob
        # Store sub-model results for caching
        em_result = bma_result.copula_result.em_result
        joint_H0 = bma_result.copula_result.joint_H0
        joint_H1 = bma_result.copula_result.joint_H1
        latent_class_result = bma_result.em3c_result
        em_diagnostics = bma_result.copula_result.em_diagnostics
        if run_em_diagnostics && !isnothing(em_diagnostics)
            @info "Summarizing EM restart diagnostics..."
            em_diagnostics_summary = summarize_em_diagnostics(em_diagnostics)
        end
        convergence_plt = EMconvergenceDiagnosticPlot(em_result)

    else
        error("Unknown combination_method: $combination_method. Must be :copula, :latent_class, or :bma")
    end

    bfdr_values = bfdr(posterior_prob, isBF=false)
    pep_values = pep(posterior_prob)

    # ------------------------------------ #
    # Scatter combination results back to full-protein list
    # ------------------------------------ #
    # df only contains detected proteins (length n_detected).
    # We need to scatter combined_bf, posterior_prob, bfdr_values and Component/P_* columns
    # back into full-length vectors (length n_proteins) with missing for non-detected rows.
    all_protein_names = getIDs(data)
    detected_protein_names = String.(df.Protein)

    # Build scatter index: for each detected protein, find its position in all_protein_names
    detected_full_indices = [findfirst(==(p), all_protein_names) for p in detected_protein_names]

    combined_bf_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    posterior_prob_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    pep_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    bfdr_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    for (k, idx) in enumerate(detected_full_indices)
        if !isnothing(idx)
            combined_bf_full[idx] = combined_bf[k]
            posterior_prob_full[idx] = posterior_prob[k]
            pep_full[idx] = pep_values[k]
            bfdr_full[idx] = bfdr_values[k]
        end
    end

    # ------------------------------------ #
    # generate output file (full protein list with missing for non-detected)
    # ------------------------------------ #

    # Build detected-only lookup for mean_log2FC and BF columns from df
    # (df only contains detected proteins; n_proteins total, n_detected in df)
    mean_log2FC_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    sd_log2FC_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    bf_enrichment_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    bf_correlation_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    bf_detected_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    for (k, idx) in enumerate(detected_full_indices)
        if !isnothing(idx)
            mean_log2FC_full[idx] = coalesce(df.mean_log2FC[k], missing)
            # thread per-protein σ for Laplace omnibus consumers.
            # Defensive on legacy cached HBM_stats DataFrames that may pre-date σ propagation.
            sd_log2FC_full[idx] = hasproperty(df, :sd_log2FC) ? coalesce(df.sd_log2FC[k], missing) : missing
            bf_enrichment_full[idx] = bf_enrichment[k]
            bf_correlation_full[idx] = bf_correlation[k]
            bf_detected_full[idx] = bf_detected[k]
        end
    end

    # per-protein missingness from raw (pre-imputation)
    # InteractionData. Single-data path uses `data` directly.
    missing_frac_full = _compute_per_protein_missingness(data)

    # bb_mnar_codriven flag using current BMA combined BF, BB
    # detection BF, and pre-imputation missingness. Strict `>` comparison.
    bb_mnar_codriven_full = _compute_bb_mnar_codriven(
        bf_detected_full, combined_bf_full, missing_frac_full, bb_mnar_codriven,
    )

    # (Branch A): bb_mnar_codriven column lineage is
    # ALREADY emitted in final_results.xlsx — spike P8 verified the column
    # present at position 17 in all 4 production xlsx outputs (wtHTT, mHTT,
    # HAP40_Strep, GST_HAP40). The column propagates from this `copula_df` builder
    # through _merge_diagnostics_to_results into `final_results`, then survives to
    # the writetable call below. Regression test:
    # test/analysis/test_bb_mnar_codriven_xlsx.jl.
    copula_df = DataFrame(
        Protein = all_protein_names,
        is_detected = Vector{Bool}(detected),
        BF = combined_bf_full,
        posterior_prob = posterior_prob_full,
        pep = pep_full,                                # canonical lowercase
        BFDR = bfdr_full,
        mean_log2FC = mean_log2FC_full,
        sd_log2FC = sd_log2FC_full,                     # per-protein σ for Laplace omnibus
        bf_enrichment = bf_enrichment_full,
        bf_correlation = bf_correlation_full,
        bf_detected = bf_detected_full,
        missing_fraction = missing_frac_full,
        bb_mnar_codriven = bb_mnar_codriven_full,       # ; verified emitted in xlsx per spike P8
    )
    copula_df.PEP = copula_df.pep                       # silent mirror (same Vector reference)

    # Add model_disagreement column if BMA was used
    # Scatter from detected-only length back to full length
    if combination_method == :bma && !isnothing(bma_result)
        model_disagreement_full = Vector{Union{Missing, Bool}}(fill(missing, n_proteins))
        for (k, idx) in enumerate(detected_full_indices)
            if !isnothing(idx)
                model_disagreement_full[idx] = Bool(bma_result.model_disagreement[k])
            end
        end
        copula_df.model_disagreement = model_disagreement_full

        # Add sub-model BF columns for BMA transparency
        bf_em_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
        bf_copula_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
        for (k, idx) in enumerate(detected_full_indices)
            if !isnothing(idx)
                bf_em_full[idx] = bma_result.em3c_result.bf[k]
                bf_copula_full[idx] = bma_result.copula_result.bf[k]
            end
        end
        copula_df.bf_em = bf_em_full
        copula_df.bf_copula = bf_copula_full

        # Add stacking weight columns — constant per-analysis, same value every row
        w_em_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
        w_copula_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
        for (k, idx) in enumerate(detected_full_indices)
            if !isnothing(idx)
                w_em_full[idx] = bma_result.em_weight
                w_copula_full[idx] = bma_result.copula_weight
            end
        end
        copula_df.w_em = w_em_full
        copula_df.w_copula = w_copula_full

        # Add per-protein Pareto k-hat column
        if bma_result.pareto_k !== nothing && length(bma_result.pareto_k) > 1
            pk_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
            for (k, idx) in enumerate(detected_full_indices)
                if !isnothing(idx) && k <= length(bma_result.pareto_k)
                    pk_full[idx] = bma_result.pareto_k[k]
                end
            end
            copula_df.pareto_k = pk_full
        end
    end

    # Add 3-component mixture model columns if available
    # Scatter from detected-only length back to full length
    if latent_class_result !== nothing &&
       latent_class_result.responsibilities !== nothing &&
       size(latent_class_result.responsibilities, 2) == 3
        resp = latent_class_result.responsibilities
        labels = ["H0", "agnostic", "H1"]
        threshold = 0.7

        component_full = Vector{Union{Missing, String}}(fill(missing, n_proteins))
        P_H0_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
        P_agnostic_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
        P_H1_full = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))

        for (k, idx) in enumerate(detected_full_indices)
            if !isnothing(idx) && k <= size(resp, 1)
                max_idx = argmax(resp[k, :])
                component_full[idx] = resp[k, max_idx] > threshold ? labels[max_idx] : "Uncertain"
                P_H0_full[idx] = resp[k, 1]
                P_agnostic_full[idx] = resp[k, 2]
                P_H1_full[idx] = resp[k, 3]
            end
        end
        copula_df.Component = component_full
        copula_df.P_H0 = P_H0_full
        copula_df.P_agnostic = P_agnostic_full
        copula_df.P_H1 = P_H1_full
    end

    writetable(
        temp_result_file,
        "hierarchical" => df,
        "copula" => copula_df;
        overwrite = true,
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
    h0_cache_file::String = "",
    # Copula-EM parameters
    prior::Union{Symbol, NamedTuple} = :default,
    n_restarts::Int = 20,
    copula_criterion::Symbol = :BIC,
    copula_family::Union{Nothing, Type} = nothing,
    h1_copula_family::Union{Nothing, Type} = nothing,
    streams::Vector{Symbol} = [:enrichment, :correlation, :detection],
    h1_refitting::Bool = true,
    burn_in::Int = 10,
    # Diagnostics
    run_em_diagnostics::Bool = true,
    # Evidence combination method
    combination_method::Symbol = :bma,
    # Latent class parameters
    lc_n_iterations::Int = 100,
    lc_alpha_prior::Union{Symbol, Vector{Float64}} = :auto,
    lc_convergence_tol::Float64 = 1e-6,
    lc_winsorize::Bool = true,
    lc_winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99),
    # Robust regression parameters
    regression_likelihood::Symbol = :normal,
    student_t_nu::Float64 = 5.0,
    regression_bf_threshold::Float64 = 0.1,
    # JZS prior
    jzs_r_scale::Float64 = 0.0,
    # Regression posterior variance floor
    regression_min_posterior_var::Float64 = 0.0,
    # Imputation method (forwarded to cache validation/construction)
    imputation::Symbol = :mnar,
    # bb_mnar_codriven diagnostic thresholds (must be passed
    # through from run_analysis; the inner `analyse` has no `config` in scope).
    bb_mnar_codriven::BBMnarCodrivenConfig = BBMnarCodrivenConfig(),
    # NEW: mask-aware regression flag + dropout_fit handle.
    mask_aware_regression::Bool = true,
    dropout_fit::Union{Nothing, DropoutFit} = nothing,
)

    n_imputed = length(imputed_data)

    # generate cache folder
    ispath("cache") && rm("cache", recursive = true)
    mkpath("cache")
    # get number of proteins
    n_proteins = length(getIDs(raw_data))

    # Check bait detection (hard error — bait must be present for correlation tests)
    check_bait_detected(raw_data, refID)

    # Compute detection indices for downstream filtering (use raw_data.detected)
    detected = raw_data.detected
    detected_indices = findall(detected)
    n_detected = length(detected_indices)
    @info "Excluded $(n_proteins - n_detected)/$n_proteins proteins (not detected in any sample)"

    # load H0 file or recompute it (skip for latent_class-only mode; copula and bma need it)
    if combination_method in (:copula, :bma)
        H0 = nothing

        # Fall back to legacy XLSX file
        if isnothing(H0) && !isnothing(H0_file) && isfile(H0_file)
            if endswith(H0_file, ".xlsx")
                @warn "Loading H0 from legacy XLSX file '$H0_file'. Consider deleting it to use the new JLD2 cache with parameter validation."
            end
            H0 = DataFrame(readtable(H0_file, "Sheet1", first_row = 1))
        end

        # Compute from scratch
        if isnothing(H0)
            H0 = computeH0_BayesFactors(
                imputed_data[1],
                n_controls = n_controls, n_samples = n_samples,
                refID = refID,
                regression_likelihood = regression_likelihood,
                student_t_nu = student_t_nu,
                regression_bf_threshold = regression_bf_threshold,
                jzs_r_scale = jzs_r_scale,
                regression_min_posterior_var = regression_min_posterior_var,
                detected_mask = raw_data.detected
            )
        end
    end

    # ------------------------------------ #
    # Beta-Bernoulli model (uses raw_data)
    # ------------------------------------ #
    bf_detected = zeros(Float64, n_proteins)
    bb_cache_used = false

    # Check Beta-Bernoulli cache (uses raw_data for hash)
    if use_intermediate_cache && !isempty(betabernoulli_cache_file)
        bb_status, bb_cached = check_betabernoulli_cache(betabernoulli_cache_file, raw_data, n_controls, n_samples, imputation)
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
            if !detected[i]
                bf_detected[i] = 0.0
                ProgressMeter.next!(p)
                continue
            end
            b, _, _ = betabernoulli(raw_data, i, n_controls, n_samples)
            ismissing(b) ? (bf_detected[i] = 0.0) : (bf_detected[i] = b)
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
                    string(pkgversion(@__MODULE__)),
                    imputation
                )
                save_betabernoulli_cache(bb_cache, betabernoulli_cache_file)
                @info "Saved Beta-Bernoulli results to cache: $betabernoulli_cache_file"
            catch e
                @warn "Failed to save Beta-Bernoulli cache: $e"
            end
        end
    end

    bf_detection = DataFrame(Protein = getIDs(raw_data), bf_detected = bf_detected)
    writetable("cache/bf_detection.xlsx", "bf_detection" => bf_detection; overwrite = true)

    # ------------------------------------ #
    # hierarchical & regression model (uses imputed_data)
    # ------------------------------------ #
    df = nothing
    bf_enrichment = Float64[]
    bf_correlation = Float64[]
    hbm_cache_used = false

    # Check HBM+Regression cache (uses combined hash of imputed + raw data)
    if use_intermediate_cache && !isempty(hbm_regression_cache_file)
        hbm_status, hbm_cached = check_hbm_regression_cache(hbm_regression_cache_file, (imputed_data, raw_data), refID, regression_likelihood, student_t_nu, regression_bf_threshold, imputation)
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

        # integration completion: build the per-replicate-column σ²_imp
        # lookup ONCE per imputed dataset, keyed by the exact (p,e,s) coordinates of the
        # regression input. Without this, `main` forwarded `column_imputation_sigma_sq = nothing`
        # and the v2b wrapper saw σ²_imp = 0 at every cell — collapsing the mask-aware
        # variance-additive observation factor to the legacy form and leaving `bf_correlation`
        # saturated on MNAR-imputed data (the problem the v2b model was meant to fix).
        # The per-protein `is_imputed` mask (from raw_data) gates which cells receive the variance.
        sigma_sq_lookups = if mask_aware_regression && regression_likelihood == :robust_t && jzs_r_scale > 0.0
            lks = [_build_column_imputation_sigma_sq_from_data(imputed_data[j]) for j in 1:n_imputed]
            @info "mask-aware regression: built σ²_imp lookup" coords_per_imputation=length(lks[1]) n_imputations=n_imputed
            lks
        else
            nothing
        end

        p = Progress(
            n_proteins, desc="Step 2: Computing hierarchical and regression Bayes factors...",
            showspeed=true,
            barglyphs=BarGlyphs('|','█', [' ' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen = 20, dt = 10
            )

        Threads.@threads for i in 1:n_proteins
            if !detected[i]
                ProgressMeter.next!(p)
                continue
            end
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
                        robust_tau_base = robust_tau_base,
                        regression_bf_threshold = regression_bf_threshold,
                        jzs_r_scale = jzs_r_scale,
                        global_tau_base = robust_tau_base,
                        regression_min_posterior_var = regression_min_posterior_var,
                        # NEW: mask-aware regression dispatch.
                        # `raw_data` is the pre-imputation handle (this overload has it directly in scope);
                        # `dropout_fit` is forwarded from run_analysis (loaded from imputed_data/dropout_curves.json).
                        mask_aware_regression = mask_aware_regression,
                        raw_data = raw_data,
                        dropout_fit = dropout_fit,
                        # integration completion: pre-built per-(p,e,s)
                        # σ²_imp lookup for this imputation. nothing → v2b sees σ²_imp = 0 (legacy).
                        column_imputation_sigma_sq = sigma_sq_lookups === nothing ? nothing : sigma_sq_lookups[j],
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
                    regression_bf_threshold,
                    compute_data_hash(imputed_data, raw_data),
                    now(),
                    string(pkgversion(@__MODULE__)),
                    imputation
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
    # Remap refID to detected-only df index
    # df only contains detected proteins (non-detected were skipped in HBM loop and excluded by innerjoin)
    # ------------------------------------ #
    bait_protein_name_imp = getIDs(raw_data)[refID]
    refID_detected_imp = findfirst(==(bait_protein_name_imp), df.Protein)
    if isnothing(refID_detected_imp)
        error("Bait protein '$bait_protein_name_imp' (refID=$refID) not found in HBM results DataFrame (imputed path). " *
              "This should not happen if check_bait_detected passed. Please report this bug.")
    end

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

    # bf_enrichment, bf_correlation, bf_detected are already detected-only (from df)
    bf_triplet = BayesFactorTriplet(bf_enrichment, bf_correlation, bf_detected)

    # ALWAYS run stage-1 EM first (all combination methods share this)
    @info "Running stage-1 3-component EM..."
    phase1_result = combined_BF_latent_class(bf_triplet, refID_detected_imp;
        use_3c = true,
        alpha_prior = lc_alpha_prior,
        n_iterations = lc_n_iterations,
        convergence_tol = lc_convergence_tol,
        verbose = verbose,
        winsorize = lc_winsorize,
        winsorize_quantiles = lc_winsorize_quantiles,
        protein_names = String.(df.Protein))

    if combination_method == :copula
        @info "Starting copula-EM combination"

        combinedResult = combined_BF(
            bf_triplet, refID_detected_imp;
            phase1_result = phase1_result,
            n_restarts = n_restarts,
            copula_criterion = copula_criterion,
            copula_family = copula_family,
            h1_copula_family = h1_copula_family,
            streams = streams,
            burn_in = burn_in,
            verbose = verbose
        )

        combined_bf = combinedResult.bf
        posterior_prob = combinedResult.posterior_prob
        em_result = combinedResult.em_result
        joint_H0 = combinedResult.joint_H0
        joint_H1 = combinedResult.joint_H1
        latent_class_result = phase1_result
        convergence_plt = EMconvergenceDiagnosticPlot(em_result)

        # Process EM diagnostics
        em_diagnostics = combinedResult.em_diagnostics
        if run_em_diagnostics && !isnothing(em_diagnostics)
            @info "Summarizing EM restart diagnostics..."
            em_diagnostics_summary = summarize_em_diagnostics(em_diagnostics)
        end

    elseif combination_method == :latent_class
        @info "Using stage-1 latent class result directly"

        latent_class_result = phase1_result

        combined_bf = latent_class_result.bf
        posterior_prob = latent_class_result.posterior_prob
        convergence_plt = plot_lc_convergence(latent_class_result)

        # Process EM diagnostics (latent class path)
        em_diagnostics = latent_class_result.em_diagnostics
        if run_em_diagnostics && !isnothing(em_diagnostics)
            @info "Summarizing EM restart diagnostics..."
            em_diagnostics_summary = summarize_em_diagnostics(em_diagnostics)
        end

    elseif combination_method == :bma
        @info "Starting Bayesian Model Averaging (BMA) combination..."

        bma_result = combined_BF_bma(
            bf_triplet, refID_detected_imp;
            phase1_result = phase1_result,
            n_restarts = n_restarts,
            copula_criterion = copula_criterion,
            copula_family = copula_family,
            h1_copula_family = h1_copula_family,
            streams = streams,
            burn_in = burn_in,
            lc_n_iterations = lc_n_iterations,
            lc_alpha_prior = lc_alpha_prior,
            lc_convergence_tol = lc_convergence_tol,
            lc_winsorize = lc_winsorize,
            lc_winsorize_quantiles = lc_winsorize_quantiles,
            verbose = verbose,
            protein_names = String.(df.Protein)
        )

        combined_bf = bma_result.bf
        posterior_prob = bma_result.posterior_prob
        em_result = bma_result.copula_result.em_result
        joint_H0 = bma_result.copula_result.joint_H0
        joint_H1 = bma_result.copula_result.joint_H1
        latent_class_result = bma_result.em3c_result
        em_diagnostics = bma_result.copula_result.em_diagnostics
        if run_em_diagnostics && !isnothing(em_diagnostics)
            @info "Summarizing EM restart diagnostics..."
            em_diagnostics_summary = summarize_em_diagnostics(em_diagnostics)
        end
        convergence_plt = EMconvergenceDiagnosticPlot(em_result)

    else
        error("Unknown combination_method: $combination_method. Must be :copula, :latent_class, or :bma")
    end

    bfdr_values = bfdr(posterior_prob, isBF=false)
    pep_values = pep(posterior_prob)

    # ------------------------------------ #
    # Scatter combination results back to full-protein list (imputed path)
    # ------------------------------------ #
    all_protein_names_imp = getIDs(raw_data)
    detected_protein_names_imp = String.(df.Protein)
    detected_full_indices_imp = [findfirst(==(p), all_protein_names_imp) for p in detected_protein_names_imp]

    combined_bf_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    posterior_prob_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    pep_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    bfdr_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    for (k, idx) in enumerate(detected_full_indices_imp)
        if !isnothing(idx)
            combined_bf_full_imp[idx] = combined_bf[k]
            posterior_prob_full_imp[idx] = posterior_prob[k]
            pep_full_imp[idx] = pep_values[k]
            bfdr_full_imp[idx] = bfdr_values[k]
        end
    end

    # ------------------------------------ #
    # generate output file (full protein list with missing for non-detected)
    # ------------------------------------ #
    mean_log2FC_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    sd_log2FC_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    bf_enrichment_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    bf_correlation_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    bf_detected_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
    for (k, idx) in enumerate(detected_full_indices_imp)
        if !isnothing(idx)
            mean_log2FC_full_imp[idx] = coalesce(df.mean_log2FC[k], missing)
            # thread per-protein σ for Laplace omnibus consumers.
            # Defensive on legacy cached HBM_stats DataFrames that may pre-date σ propagation.
            sd_log2FC_full_imp[idx] = hasproperty(df, :sd_log2FC) ? coalesce(df.sd_log2FC[k], missing) : missing
            bf_enrichment_full_imp[idx] = bf_enrichment[k]
            bf_correlation_full_imp[idx] = bf_correlation[k]
            bf_detected_full_imp[idx] = bf_detected[k]
        end
    end

    # missingness from raw (unimputed) data.
    # The imputed-vector path computes from `raw_data`, NOT from any imputed dataset.
    missing_frac_full_imp = _compute_per_protein_missingness(raw_data)

    bb_mnar_codriven_full_imp = _compute_bb_mnar_codriven(
        bf_detected_full_imp, combined_bf_full_imp, missing_frac_full_imp, bb_mnar_codriven,
    )

    copula_df = DataFrame(
        Protein = all_protein_names_imp,
        is_detected = Vector{Bool}(detected),
        BF = combined_bf_full_imp,
        posterior_prob = posterior_prob_full_imp,
        pep = pep_full_imp,                               # canonical lowercase
        BFDR = bfdr_full_imp,
        mean_log2FC = mean_log2FC_full_imp,
        sd_log2FC = sd_log2FC_full_imp,                   # per-protein σ for Laplace omnibus
        bf_enrichment = bf_enrichment_full_imp,
        bf_correlation = bf_correlation_full_imp,
        bf_detected = bf_detected_full_imp,
        missing_fraction = missing_frac_full_imp,
        bb_mnar_codriven = bb_mnar_codriven_full_imp,
    )
    copula_df.PEP = copula_df.pep                         # silent mirror

    # Add model_disagreement column if BMA was used (scatter to full length)
    if combination_method == :bma && !isnothing(bma_result)
        model_disagreement_full_imp = Vector{Union{Missing, Bool}}(fill(missing, n_proteins))
        for (k, idx) in enumerate(detected_full_indices_imp)
            if !isnothing(idx)
                model_disagreement_full_imp[idx] = Bool(bma_result.model_disagreement[k])
            end
        end
        copula_df.model_disagreement = model_disagreement_full_imp

        # Add sub-model BF columns for BMA transparency
        bf_em_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
        bf_copula_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
        for (k, idx) in enumerate(detected_full_indices_imp)
            if !isnothing(idx)
                bf_em_full_imp[idx] = bma_result.em3c_result.bf[k]
                bf_copula_full_imp[idx] = bma_result.copula_result.bf[k]
            end
        end
        copula_df.bf_em = bf_em_full_imp
        copula_df.bf_copula = bf_copula_full_imp

        # Add stacking weight columns
        w_em_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
        w_copula_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
        for (k, idx) in enumerate(detected_full_indices_imp)
            if !isnothing(idx)
                w_em_full_imp[idx] = bma_result.em_weight
                w_copula_full_imp[idx] = bma_result.copula_weight
            end
        end
        copula_df.w_em = w_em_full_imp
        copula_df.w_copula = w_copula_full_imp
    end

    # Add 3-component mixture model columns if available (scatter to full length)
    if latent_class_result !== nothing &&
       latent_class_result.responsibilities !== nothing &&
       size(latent_class_result.responsibilities, 2) == 3
        resp = latent_class_result.responsibilities
        labels = ["H0", "agnostic", "H1"]
        threshold = 0.7

        component_full_imp = Vector{Union{Missing, String}}(fill(missing, n_proteins))
        P_H0_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
        P_agnostic_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))
        P_H1_full_imp = Vector{Union{Missing, Float64}}(fill(missing, n_proteins))

        for (k, idx) in enumerate(detected_full_indices_imp)
            if !isnothing(idx) && k <= size(resp, 1)
                max_idx = argmax(resp[k, :])
                component_full_imp[idx] = resp[k, max_idx] > threshold ? labels[max_idx] : "Uncertain"
                P_H0_full_imp[idx] = resp[k, 1]
                P_agnostic_full_imp[idx] = resp[k, 2]
                P_H1_full_imp[idx] = resp[k, 3]
            end
        end
        copula_df.Component = component_full_imp
        copula_df.P_H0 = P_H0_full_imp
        copula_df.P_agnostic = P_agnostic_full_imp
        copula_df.P_H1 = P_H1_full_imp
    end

    writetable(
        "results.xlsx",
        "hierarchical" => df,
        "copula" => copula_df;
        overwrite = true,
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
- `sidecar_file::String`: Path for the JSON sidecar file containing report data for merging.
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
    pit_histogram_file::String
    scale_location_hbm_file::String
    scale_location_regression_file::String
    nu_optimization_file::String
    report_file::String
    report_methods_file::String
    sidecar_file::String
    h0_marginals_file::String
    h1_marginals_file::String
    bma_weights_file::String
    # Copula diagnostics
    kl_divergence_file::String
    within_class_corr_file::String
    agnostic_zone_file::String
    copula_bootstrap_file::String
    discordant_proteins_file::String
    copula_diagnostics_summary_file::String
    # Simulation (parametric bootstrap FDR calibration)
    simulation_file::String
end

"""
    OutputFiles(basedir::String; image_ext::String=".png")

Create an `OutputFiles` with all paths auto-generated under `basedir`.
"""
function OutputFiles(basedir::String; image_ext::String=".png")
    OutputFiles(
        basedir             = basedir,
        H0_file             = joinpath(basedir, "copula_H0.xlsx"),  # legacy fallback (read-only, no longer written)
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
        pit_histogram_file = joinpath(basedir, "pit_histogram" * image_ext),
        scale_location_hbm_file = joinpath(basedir, "scale_location_hbm" * image_ext),
        scale_location_regression_file = joinpath(basedir, "scale_location_regression" * image_ext),
        nu_optimization_file = joinpath(basedir, "nu_optimization" * image_ext),
        report_file = joinpath(basedir, "interactive_report.html"),
        report_methods_file = joinpath(basedir, "methods.md"),
        sidecar_file = joinpath(basedir, "interactive_report_data.json"),
        h0_marginals_file = joinpath(basedir, "h0_marginals" * image_ext),
        h1_marginals_file = joinpath(basedir, "h1_marginals" * image_ext),
        bma_weights_file = joinpath(basedir, "bma_weights" * image_ext),
        # Copula diagnostics
        kl_divergence_file = joinpath(basedir, "kl_divergence" * image_ext),
        within_class_corr_file = joinpath(basedir, "within_class_correlation" * image_ext),
        agnostic_zone_file = joinpath(basedir, "agnostic_zone" * image_ext),
        copula_bootstrap_file = joinpath(basedir, "copula_bootstrap" * image_ext),
        discordant_proteins_file = joinpath(basedir, "discordant_proteins" * image_ext),
        copula_diagnostics_summary_file = joinpath(basedir, "copula_diagnostics_summary.md"),
        simulation_file = joinpath(basedir, "simulation_cache.jld2"),
    )
end

"""
    calibration_cache_path(config::CONFIG) -> String

Resolve the calibration-cache JLD2 path lazily from `(basedir, imputation_method)`.
The filename gains an `_<imputation>` suffix so MICE and MNAR runs in the same
`OutputFiles.basedir` produce distinct cache files instead of silently overwriting
one another (CAL-05 independent invalidation + MNAR-default coexistence).
"""
calibration_cache_path(config) =
    joinpath(config.output.basedir, "calibration_cache_$(config.imputation_method).jld2")

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
- `normalise_protocols::Bool = false`: Normalize data across different experimental protocols.
   Back-compat alias for `normalisation_method` (true->:row_center, false->:none). Kept working
   via `_resolve_normalisation_method`; superseded by `normalisation_method` when that selector is
   set to anything other than `:none`.
- `normalisation_method::Symbol = :auto`: Normalisation selector. One of
   `:none | :row_center | :median_of_ratios | :both | :auto`. `:none` is byte-identical to
   `normalise_protocols=false`; `:row_center` is byte-identical to `normalise_protocols=true`
   (the existing `normalize()`). `:auto` is currently a no-op (=:none) pending the
   multi-protocol auto-detector.
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
- `lc_alpha_prior::Union{Symbol, Vector{Float64}} = :auto`: Dirichlet prior on class proportions. `:auto` triggers Empirical Bayes estimation + grid marginalization; explicit vector (e.g. `[5.0, 2.0, 1.0]`) uses that vector directly.
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
- `regression_bf_threshold::Float64 = 0.1`: Slope threshold for the regression Bayes factor. H1: slope > threshold. Default 0.1 filters out very weak correlation signals. Set to 0.0 to test for any positive correlation, or 0.3 for minimum effect size testing.
- `optimize_nu::Bool = run_diagnostics`: Optimize ν over [5, 50] via Brent's method minimizing WAIC. When `true`, automatically sets `student_t_nu` to the optimal value and runs BEFORE the main analysis. Implies `run_model_comparison` (Normal WAIC is computed as baseline). Defaults to `true` when `run_diagnostics` is enabled.
- `jzs_r_scale::Float64 = 0.354`: JZS Cauchy r-scale for the regression slope prior. `0.0` falls back to a Normal prior; `>0` uses JZS via a Normal-Gamma scale mixture. Default `0.354` is the JASP convention (√2/4).
- `regression_min_posterior_var::Float64 = 0.01`: Minimum variance floor for regression posterior distributions. VMP posteriors can be over-confident (extremely narrow sigma), causing P(slope>0) to saturate at 0 or 1 and regression BFs to hit the +-max_bf clamp. Setting to 0.01 ensures min posterior std = 0.1, restoring BF gradation. Set to 0.0 to disable.

# Imputation Method
- `imputation_method::Symbol = :mnar`: Metadata tag identifying how the input data was imputed. One of `:mnar` (default, MNAR-aware tilted-Gaussian imputation), `:mar` (deprecated MICE / multiple imputation; v1.3 removal), or `:none` (raw data with missings preserved). Propagates into every cache parameter-hash so MICE and MNAR caches coexist on disk. The `load_data` `imputation` kwarg overrides this CONFIG default.

# Variance Recovery
- `mnar_variance_recovery::Symbol = :off`: Post-hoc variance-recovery mode for the single MNAR draw. One of `:off` (default; behaviour bit-for-bit), `:inflation` (post-hoc per-protein posterior log2FC widening), or `:multi_impute` (in-process `m` MNAR draws + Rubin's-rules pooling). Both `:inflation` and `:multi_impute` require `imputation_method == :mnar`; otherwise `run_analysis` throws `ArgumentError`.
- `mnar_m::Int = 3`: Number of MNAR imputations for `:multi_impute` (Rubin's MI literature default). Validated `2 ≤ mnar_m ≤ 10` at `run_analysis` start.
- `mnar_inflation_max::Float64 = 3.0`: Upper bound on the per-protein variance-inflation factor used by `:inflation`; prevents pathological tilts.
- `mnar_inflation_factor::Union{Nothing, Float64} = nothing`: Optional scalar override for the auto-derived per-protein inflation factor. When `nothing`, the factor is derived from per-column dropout curves × per-protein missingness.
- `mnar_base_seed::Int = 42`: Deterministic seed for `:multi_impute`; per-imputation seed = `mnar_base_seed * 1_000_003 + i`.

# Input Data Quality Control
- `run_input_qc::Bool = true`: Run the five-check input QC system (v1.1.5) on the loaded data — scale detection, replicate correlation, missingness asymmetry, intensity shape, PCA separation. Produces per-check `:ok`/`:warning`/`:fail` flags that are surfaced in the interactive HTML report.

# Automated Validation Gates
- `run_validation::Bool = true`: Run mixture-model quality gates after evidence combination — KS marginal goodness-of-fit, KL contamination between H0 and H1, component separation, within-class correlation. A failing gate is flagged in the report; results are still produced.

# Copula Diagnostics
- `run_copula_diagnostics::Bool = true`: Compute copula-model diagnostics (KL divergence between fitted and empirical joints, within-class correlation, agnostic-zone analysis) for the BMA Copula sub-model. Plots written to `OutputFiles.kl_divergence_file` / `within_class_corr_file` / `agnostic_zone_file`.

# Simulation & Calibration
- `run_simulation::Bool = true`: Run the parametric simulation engine (5×5 grid × 10 replicates → 250 synthetic datasets) to generate ground truth for Platt-scaling calibration. Required for posterior recalibration.
- `sim_n_synthetic::Int = 10_000`: Total number of synthetic proteins drawn across the simulation grid.

# Interactive HTML Report
- `generate_report_html::Bool = true`: Auto-generate `OutputFiles.report_file` (single-file interactive HTML) at the end of `run_analysis`. Combines volcano, calibration, sensitivity, mixture, methods, and data-quality tabs in one self-contained document.

# Data Curation (STRING-API protein-group splitting, synonym resolution, merging)
- `curate::Bool = true`: Enable protein curation (group splitting + synonym resolution + contaminant removal). Disabling turns `load_data` into a pure file-reading pass.
- `species::Int = 9606`: NCBI taxonomy ID used for STRING queries (9606 = human; 10090 = mouse).
- `bait_name::Union{Nothing, String} = nothing`: Bait protein name. When set, `load_data` tracks the bait through curation and returns its post-curation row index, even if curation re-orders or splits rows.
- `curate_interactive::Bool = true`: Prompt the user to confirm each ambiguous merge. Set `false` for non-interactive runs.
- `curate_merge_strategy::Symbol = :max`: Strategy when collapsing duplicate rows after synonym resolution (`:max` or `:mean`).
- `curate_replay::Union{Nothing, String} = nothing`: Path to a saved `CurationReport` JLD2; when set, curation decisions are replayed deterministically instead of re-querying STRING.
- `curate_remove_contaminants::Bool = true`: Drop standard MaxQuant contaminant entries (`CON__*`, `REV__*`).
- `curate_delimiter::String = ";"`: Delimiter used to split protein-group strings (`P12345;Q67890`) before per-ID synonym resolution.
- `curate_auto_approve::Int = 0`: Auto-approve merges whose ID strings share at least this many leading characters; `0` always prompts.

# Docking Integration (optional post-analysis step)
- `run_docking::Bool = false`: Enable AlphaFold-Server docking integration. When `true`, generates JSON requests for high-confidence MS hits, expects user-uploaded result ZIPs, parses them, and applies a two-stage Bayesian update combining MS evidence with docking evidence.
- `docking_config::Union{DockingConfig, Nothing} = nothing`: Required when `run_docking = true`. Holds `pep_threshold`, scoring tier selection (`:iptm`, `:pdockq`, `:c2qscore`), and quality-gate parameters.
- `bait_sequence::String = ""`: Bait protein sequence (required when `run_docking = true`; can be auto-fetched via `bait_uniprot`).
- `bait_uniprot::String = ""`: UniProt accession used to auto-fetch `bait_sequence` from UniProt if the sequence isn't supplied directly.

# DNN Prior + MC-Dropout
- `run_dnn_prior_mc_dropout::Bool = true`: enable per-pair MC-Dropout uncertainty quantification on
  the DNN prior. Adds 5 columns to the results DataFrame: `prior_mc_mean`, `prior_mc_std`,
  `prior_mc_ci_low`, `prior_mc_ci_high`, `prior_contribution = posterior_prob − prior_mc_mean`.
  Latency on K=30 / 50k-pair / CPU: ~5–10 min added; GPU: <1 min. Falls back to NaN
  columns when the metalearner extension is not loaded (Variante B). Requires
  `using Flux, MLJ, MLJScikitLearnInterface, HDF5` to populate columns.
- `dnn_prior_mc_k::Int = 30`: number of MC-Dropout forward passes per pair.
  K=30 produces a stable variance estimate. Lowering reduces latency but increases CI noise;
  raising marginally improves CI stability at proportional latency cost.
- `dnn_prior_mc_batch_size::Int = 256`: mini-batch size for the K-pass forward inference.
  Matches the production `predict_metalearner` default. Increase for GPU; decrease if memory-bound.
"""
Base.@kwdef mutable struct CONFIG
    datafile::Vector{String}
    control_cols::Vector{Dict{Int,Vector{Int}}}
    sample_cols::Vector{Dict{Int,Vector{Int}}}
    poi::String

    # output file paths
    output::OutputFiles         = OutputFiles(".")

    # analysis parameters
    normalise_protocols::Bool   = false
    # normalisation method selector. Allowed values:
    # :none (no normalisation; byte-identical to normalise_protocols=false),
    # :row_center (per-protein per-(protocol,exp) row-centering; byte-identical to
    # normalise_protocols=true / the existing normalize()), :median_of_ratios (DESeq
    # size factors), :both (column-scale then row-center), :auto (auto-detect; resolution
    # wired in — currently a no-op = :none in). When the selector is left
    # at :none the legacy normalise_protocols::Bool wins (true->:row_center, false->:none);
    # any other selector value is authoritative over the bool.
    normalisation_method::Symbol = :auto
    n_controls::Int         = 0
    n_samples::Int          = 0
    refID::Int                  = 1
    plotHBMdists::Bool          = false
    plotlog2fc::Bool            = false
    plotregr::Bool              = false
    plotbayesrange::Bool        = false
    verbose::Bool               = false
    vc_legend_pos::Symbol       = :topleft
    # (Pitfall 6): widened from `::String` to `::Union{Nothing, String}`
    # with default `nothing`. When left as `nothing`, the metalearner extension resolves
    # to the schema-matching default artefact based on `metalearner_use_mc_dropout`
    # (see `resolve_metalearner_path(::Nothing; use_mc_dropout)` overload). An explicit
    # `String` path bypasses the schema-aware default selection.
    metalearner_path::Union{Nothing, String} = nothing
    # Default is `false` (non-MC `:tr_ddi` 14-feat schema). Setting `true` enables
    # MC-Dropout (`:tr_ddi_mc` 15-feat) but is DEPRECATED as of 2026-06-13:
    # species-agnostic spike reruns show `:tr_ddi` beats `:tr_ddi_mc` overall
    # (AUC +0.019, MCC +0.024 on n=1862 multi-species pairs).
    # MC-Dropout is retained as an explicit opt-in for non-human-focused workflows;
    # a one-time @warn is emitted when `true` is set explicitly.
    metalearner_use_mc_dropout::Bool = false

    # Copula-EM parameters
    em_prior::Union{Symbol, NamedTuple{(:α, :β), Tuple{Float64, Float64}}} = :default
    em_n_restarts::Int          = 20
    copula_criterion::Symbol    = :BIC
    # force a fixed copula family instead of BIC selection.
    # `nothing` = BIC over LOGBF_COPULA_FAMILIES (default, byte-identical). When set
    # (e.g. `FrankCopula`), the family is forced at the H0 and H1 fits and at both
    # EM-loop refit sites, threaded through the :bma default path AND the :copula path.
    # BMA sub-model naming stays "Copula" / "3c-EM" (never "vine copula") per the
    # terminology lock.
    copula_family::Union{Nothing, Type}    = nothing
    h1_copula_family::Union{Nothing, Type} = nothing
    h1_refitting::Bool          = true
    em_burn_in::Int             = 10
    run_em_diagnostics::Bool    = true

    # Evidence combination method
    combination_method::Symbol  = :bma  # :copula, :latent_class, or :bma

    # Latent class parameters (used when combination_method = :latent_class)
    lc_n_iterations::Int        = 100
    lc_alpha_prior::Union{Symbol, Vector{Float64}} = :auto
    lc_convergence_tol::Float64 = 1e-6
    lc_winsorize::Bool          = true
    lc_winsorize_quantiles::Tuple{Float64,Float64} = (0.01, 0.99)

    # Prior sensitivity analysis
    run_sensitivity::Bool                   = true
    sensitivity_config::SensitivityConfig   = SensitivityConfig()

    # Posterior predictive checks & model diagnostics
    run_diagnostics::Bool                   = false
    diagnostics_config::DiagnosticsConfig   = DiagnosticsConfig()

    # Similarity & Embeddings. Drives sample/protein
    # UMAP/t-SNE + condition-pair similarity. Default EmbeddingsConfig() uses
    # method=:umap, supervised=false, n_neighbors=15, min_dist=0.1,
    # top_k_jaccard=50, seed=42, run_embeddings=true. Validated early in
    # run_analysis via _validate_embeddings_config. Partial cache
    # invalidation via _should_recompute_embeddings(ar, cfg).
    embeddings_config::EmbeddingsConfig     = EmbeddingsConfig()

    # bb_mnar_codriven diagnostic thresholds.
    # Per-protein flag is raised when Beta-Bernoulli detection BF, post-MNAR
    # BMA combined BF, and pre-imputation missing_fraction all exceed their
    # respective thresholds (defaults 10.0, 10.0, 0.5 per §7 spec).
    bb_mnar_codriven::BBMnarCodrivenConfig  = BBMnarCodrivenConfig()

    # Copula diagnostics (KL divergence, within-class correlation, etc.)
    run_copula_diagnostics::Bool            = true

    # Input data quality control
    run_input_qc::Bool                      = true

    # Automated validation gates (quality gate matrix + KL contamination + consistency checks)
    run_validation::Bool                    = true

    # Interactive HTML report
    generate_report_html::Bool              = true

    # Robust regression (Student-t via scale mixture)
    regression_likelihood::Symbol           = :robust_t    # :normal or :robust_t
    student_t_nu::Float64                   = 5.0        # degrees of freedom for Student-t
    regression_bf_threshold::Float64        = 0.1        # slope threshold for regression BF: H1: slope > threshold
    run_model_comparison::Bool              = true        # run both models + WAIC comparison
    optimize_nu::Bool                       = true        # optimize ν via Brent's method (WAIC-based); follows run_diagnostics by default
    jzs_r_scale::Float64                    = 0.354      # JZS Cauchy r-scale for regression slope prior (0 = Normal prior, >0 = JZS)
    regression_min_posterior_var::Float64    = 0.01       # min variance floor for regression posterior (0.0 = no floor); prevents BF saturation from over-confident VMP posteriors

    # NEW: Mask-aware regression (v2b — variance-additive σ inflation).
    # When true, regression() dispatches to RegressionModel_*_robust_jzs_v2b wrappers and threads
    # the per-cell σ²_imp + is_imputed mask through. When false, the legacy pre-spike
    # RegressionModel_*_robust_jzs wrappers are invoked verbatim (byte-identical on non-MNAR data).
    mask_aware_regression::Bool             = true

    # Variance recovery (OFF by default; requires imputation_method == :mnar)
    mnar_variance_recovery::Symbol                       = :off
    mnar_m::Int                                          = 3
    mnar_inflation_max::Float64                          = 3.0
    mnar_inflation_factor::Union{Nothing, Float64}       = nothing
    mnar_base_seed::Int                                  = 42

    # Imputation method (metadata tag — propagates to cache hashes; load_data kwarg overrides)
    imputation_method::Symbol               = :mnar       # :mnar (default), :mar (deprecated), or :none

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

    # Docking (optional post-analysis step)
    run_docking::Bool                       = false
    docking_config::Union{DockingConfig, Nothing} = nothing
    bait_sequence::String                   = ""           # Required when run_docking = true
    bait_uniprot::String                    = ""           # For sequence auto-fetch

    # DNN Prior + MC-Dropout uncertainty
    # When true, compute_mc_prior! runs after predict_metalearner and populates 5
    # result-DataFrame columns: prior_mc_mean, prior_mc_std, prior_mc_ci_low,
    # prior_mc_ci_high, prior_contribution. K=30 / 50k-pair / CPU adds ~5–10 min
    # over baseline; GPU <1 min. Opt out via flag = false. Requires the
    # metalearner extension trigger: `using Flux, MLJ, MLJScikitLearnInterface, HDF5`.
    # See (uncertainty signal
    # validated, calibration improvement invalidated) and
    # (metalearner is the load-bearing
    # calibration step).
    run_dnn_prior_mc_dropout::Bool          = true
    dnn_prior_mc_k::Int                     = 30
    dnn_prior_mc_batch_size::Int            = 256
    # (ABL-P2): which evidence streams enter the joint copula.
    # Default = all three (current behaviour, byte-identical 3-D copula). Drop one for an
    # ablation variant (drop_detection / drop_correlation / drop_enrichment). Set membership
    # (NOT list order) drives copula dimensionality; the copula build canonicalises order.
    # A <2-stream config is rejected loudly via @assert in precompute_h0 / combined_BF.
    evidence_streams::Vector{Symbol}        = [:detection, :correlation, :enrichment]
    # (ABL-P1): apply the metalearner DNN-prior posterior update.
    # Default = true (current behaviour). When false, suppress the metalearner DNN-prior
    # update at BOTH the single- and multi-protocol guard sites EVEN IF the metalearner
    # extension is loaded; the result keeps the upstream BF-derived posterior_prob fallback
    # (no recomputation needed). Lets an ablation variant exclude the learned prior cleanly.
    use_metalearner_prior::Bool             = true

    # Simulation (parametric bootstrap FDR calibration)
    run_simulation::Bool                    = true
    sim_n_synthetic::Int                    = 10_000
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

# ─── Validation pipeline step ────────────────────────────────────────────────

"""
    _print_quality_gate_summary(qg::QualityGateResult)

Print a formatted 3x3 quality gate summary to the logger.
"""
function _print_quality_gate_summary(qg::QualityGateResult)
    marginal_labels = ["Enrichment", "Correlation", "Detection"]
    component_labels = ["H0", "Agnostic", "H1"]
    lines = String[]
    push!(lines, "Quality Gate Matrix:")
    push!(lines, "           $(rpad("H0", 15))$(rpad("Agnostic", 15))H1")
    for i in 1:3
        row_parts = String[]
        for j in 1:3
            c = qg.cells[i, j]
            s = c.status == :pass ? "PASS" : c.status == :warn ? "WARN" : "FAIL"
            push!(row_parts, rpad("$s($(round(c.ks_statistic, digits=3)))", 15))
        end
        push!(lines, "  $(rpad(marginal_labels[i], 12))$(join(row_parts))")
    end
    push!(lines, "  Overall: $(qg.overall_status)")
    if !isempty(qg.remediation_details)
        for d in qg.remediation_details
            push!(lines, "  Remediation: $d")
        end
    end
    @info join(lines, "\n")
end

"""
    _run_validation(bf, lc_result, results_df, config) -> ValidationResult

Run automated validation: quality gates, KL contamination, and internal consistency checks.
"""
function _run_validation(
    bf::BayesFactorTriplet,
    lc_result::Union{Nothing, LatentClassResult},
    results_df::DataFrame,
    config::CONFIG
)
    quality_gates = nothing
    kl_contamination = nothing
    consistency = Dict{String, Bool}()

    # 1. Quality gates (requires 3-component LatentClassResult)
    if lc_result !== nothing && lc_result.responsibilities !== nothing && size(lc_result.responsibilities, 2) == 3
        try
            quality_gates = run_quality_gates(bf, lc_result)
            consistency["all_ks_pass"] = quality_gates.overall_status == :pass
            if config.verbose
                _print_quality_gate_summary(quality_gates)
            end
        catch e
            @warn "Quality gate computation failed" exception=e
            consistency["all_ks_pass"] = false
        end
    end

    # 2. KL contamination
    if lc_result !== nothing && lc_result.responsibilities !== nothing && size(lc_result.responsibilities, 2) == 3
        try
            kl_contamination = compute_kl_contamination(bf, lc_result)
            consistency["kl_pass"] = kl_contamination.per_stream_pass
            if config.verbose
                @info "KL contamination" enrichment=round(kl_contamination.kl_enrichment, digits=3) correlation=round(kl_contamination.kl_correlation, digits=3) detection=round(kl_contamination.kl_detection, digits=3) joint=round(kl_contamination.kl_joint, digits=3) pass=kl_contamination.per_stream_pass
            end
        catch e
            @warn "KL contamination computation failed" exception=e
            consistency["kl_pass"] = false
        end
    end

    # 3. Internal consistency checks
    # H1 component size (< 200 proteins)
    if lc_result !== nothing && lc_result.responsibilities !== nothing && size(lc_result.responsibilities, 2) == 3
        h1_count = sum(lc_result.responsibilities[:, 3] .> 0.5)
        consistency["h1_lt_200"] = h1_count < 200
        if config.verbose
            @info "H1 component size: $h1_count proteins" pass=(h1_count < 200)
        end
    end

    # Check known anchors if present in results (only for HAP40-like data)
    if hasproperty(results_df, :Protein) && hasproperty(results_df, :posterior_prob)
        proteins = Vector{String}(results_df.Protein)
        posteriors = Vector{Float64}(coalesce.(results_df.posterior_prob, 0.0))

        f8a1_idx = findfirst(p -> occursin("F8A1", p) || occursin("ENSP00000337401", p), proteins)
        if f8a1_idx !== nothing
            consistency["F8A1_P1"] = posteriors[f8a1_idx] >= 0.999
        end

        htt_idx = findfirst(p -> occursin("HTT", p) || occursin("ENSP00000355072", p), proteins)
        if htt_idx !== nothing
            consistency["HTT_P099"] = posteriors[htt_idx] > 0.99
        end
    end

    overall_pass = isempty(consistency) ? true : all(values(consistency))

    return ValidationResult(
        quality_gates,
        kl_contamination,
        nothing,  # sensitivity_crossings filled by sensitivity_analysis separately
        consistency,
        overall_pass,
        Dates.now()
    )
end

# ============================================================================
# QC warning emitter (private helper)
# ============================================================================

"""
    _emit_qc_warnings(qc::InputQCResult)

Emit `@warn` messages with `[QC]` prefix for each individual QC issue found.
Called after `run_input_qc()` completes successfully.
"""
function _emit_qc_warnings(qc::InputQCResult)
    # Scale warnings
    if !isnothing(qc.scale)
        for p in qc.scale.protocols
            if p.flag != :ok
                @warn "[QC] Protocol $(p.protocol_index): data may be on linear scale (max value = $(p.max_value), expected log2 range 10-35)"
            end
        end
    end

    # Replicate correlation warnings
    if !isnothing(qc.replicate_correlation)
        for c in qc.replicate_correlation.checks
            if c.flag != :ok
                @warn "[QC] Protocol $(c.protocol_index), Experiment $(c.experiment_index), $(c.group): low replicate correlation (min Spearman = $(round(c.min_correlation, digits=3)), threshold = 0.80)"
            end
        end
    end

    # Missingness warnings
    if !isnothing(qc.missingness)
        for c in qc.missingness.checks
            if c.flag != :ok
                @warn "[QC] Protocol $(c.protocol_index), Experiment $(c.experiment_index), $(c.group): missingness asymmetry detected (max ratio = $(round(c.max_ratio, digits=1))x median)"
            end
        end
    end

    # Intensity shape warnings
    if !isnothing(qc.intensity_shape)
        for c in qc.intensity_shape.checks
            if c.flag != :ok
                parts = String[]
                c.bimodality_flag != :ok && push!(parts, "possible bimodality (kurtosis = $(round(c.excess_kurtosis, digits=2)))")
                c.spike_flag != :ok && push!(parts, "spike at zero/min ($(round(100*c.spike_fraction, digits=1))%)")
                c.tail_flag != :ok && push!(parts, "heavy tails (kurtosis = $(round(c.excess_kurtosis, digits=2)))")
                detail = isempty(parts) ? "shape anomaly" : join(parts, "; ")
                @warn "[QC] Protocol $(c.protocol_index), Experiment $(c.experiment_index), $(c.group), Replicate $(c.replicate_index): $detail"
            end
        end
    end
end

"""
    _validate_variance_recovery_config(config::CONFIG) -> Nothing

early-validation guard. Throws an `ArgumentError` BEFORE any
expensive work begins when the user has set `config.mnar_variance_recovery`
to `:inflation` or `:multi_impute` without also setting `config.imputation_method
= :mnar`. Also validates `2 ≤ mnar_m ≤ 10` and that
`mnar_variance_recovery` is one of `:off | :inflation | :multi_impute`.

Called as the first executable statement of both `run_analysis(config::CONFIG)`
and `run_analysis(config, imputed_data::Vector{InteractionData}, raw_data;
...)`. Mirrors the loud-ArgumentError pattern from
`src/data/imputation_stubs.jl:_require_imputation_extension`.
"""
function _validate_variance_recovery_config(config::CONFIG)
    valid_modes = (:off, :inflation, :multi_impute)
    if !(config.mnar_variance_recovery in valid_modes)
        throw(ArgumentError(
            "CONFIG.mnar_variance_recovery = :$(config.mnar_variance_recovery) is not a valid mode. " *
            "Valid values: :off (default), :inflation, :multi_impute."
        ))
    end

    # mnar_m range check fires regardless of mode (covers misconfigured :off with future toggles too)
    if !(2 <= config.mnar_m <= 10)
        throw(ArgumentError(
            "CONFIG.mnar_m = $(config.mnar_m) is out of range. " *
            "Required: 2 ≤ mnar_m ≤ 10 (Rubin's MI literature default = 3)."
        ))
    end

    if config.mnar_variance_recovery in (:inflation, :multi_impute) &&
       config.imputation_method !== :mnar
        throw(ArgumentError(
            "CONFIG.mnar_variance_recovery = :$(config.mnar_variance_recovery) requires " *
            "CONFIG.imputation_method = :mnar (got :$(config.imputation_method)). Either set imputation_method = :mnar " *
            "(the v1.2.0 default), or set mnar_variance_recovery = :off."
        ))
    end

    return nothing
end

"""
    _should_recompute_embeddings(ar::AnalysisResult, cfg::EmbeddingsConfig) -> Bool

partial cache invalidation gate. Returns `true` if the cached `ar.embeddings` is
`nothing` OR if its `config_snapshot` differs from the current `cfg`. Returns `false`
when the cached embeddings still match the requested config (cache hit).

Callers (e.g. `run_analysis`) recompute ONLY the `embeddings` field on mismatch; the
rest of the result (posteriors, calibration, sensitivity, diagnostics) is reused.
"""
function _should_recompute_embeddings(ar, cfg::EmbeddingsConfig)
    ar.embeddings === nothing && return true
    return ar.embeddings.config_snapshot != _config_snapshot(cfg)
end

"""
    _compute_embeddings(data, ar, cfg::EmbeddingsConfig) -> EmbeddingsResult

orchestrator that calls `_compute_sample_embedding` and
`_compute_protein_embedding`, then assembles the results into a fully
populated `EmbeddingsResult`. Reads `ar.copula_results` and `ar.latent_class_result`
(single-bait class labels source).

Returns a populated EmbeddingsResult; `config_snapshot` is always set so future calls
can detect re-enablement / config drift via `_should_recompute_embeddings`.
"""
function _compute_embeddings(data, ar, cfg::EmbeddingsConfig)
    s = _compute_sample_embedding(data, cfg)
    lc_result = hasproperty(ar, :latent_class_result) ? ar.latent_class_result : nothing
    p = _compute_protein_embedding(ar.copula_results, lc_result, cfg)
    return EmbeddingsResult(
        s.sample_pca_scores,
        s.sample_pca_var_explained,
        s.sample_labels,
        s.sample_filter_level,
        s.sample_umap_coords,
        s.sample_tsne_coords,
        p.protein_umap_coords,
        p.protein_classes,
        p.protein_ids,
        _config_snapshot(cfg),
    )
end

"""
    _resolve_dropout_curves_path(config::CONFIG) -> String

(BLOCKER 4): single source of truth for `dropout_curves.json`
discovery. Both `:inflation` and `:multi_impute` dispatch
paths call THIS helper to resolve the canonical path — eliminating drift risk
where one mode could resolve a different file than the other for the same
CONFIG (would violate determinism).

Searches 3 candidate paths in order; returns the first `isfile`-true match.
Throws `ArgumentError` if none resolves, pointing the user at the dropout-curve fit step.
"""
function _resolve_dropout_curves_path(config::CONFIG)::String
    candidate_paths = String[
        joinpath(dirname(first(config.datafile)), "..", "imputed_data", "dropout_curves.json"),
        joinpath(config.output.basedir, "..", "imputed_data", "dropout_curves.json"),
        joinpath(config.output.basedir, "imputed_data", "dropout_curves.json"),
    ]
    chosen = findfirst(isfile, candidate_paths)
    if chosen === nothing
        throw(ArgumentError(
            "Variance recovery requires dropout_curves.json. " *
            "Searched: " * join(candidate_paths, ", ") *
            ". Run `fit_dropout_curves` (needs `using GLM`) to generate it. " *
            "(not found in any of the candidate paths)"
        ))
    end
    return candidate_paths[chosen]
end

"""
    _load_dropout_fit_for_inflation(path::String) -> DropoutFit

core-loaded reader for `dropout_curves.json` (schema).
Used by the `:inflation` arm of `mnar_variance_recovery`. Does NOT depend on
GLM / BayesInteractomicsImputationExt — the official `load_dropout_fit` does
(because the imputation extension owns the writer), but the reader is pure
JSON3 + a struct construction so the inflation path stays core-loaded.

**WARNING — schema co-versioning note:** this reader duplicates the read
logic in `ext/BayesInteractomicsImputationExt/dropout.jl::load_dropout_fit`.
Both readers consume the SAME schema; any future schema change
(adding/removing/renaming keys) MUST update BOTH readers in lockstep, or
`:inflation` and the multi-impute path will diverge on identical input.
See `ext/BayesInteractomicsImputationExt/dropout.jl::load_dropout_fit` for
the canonical writer + extension-loaded reader.

Throws `ArgumentError` with actionable guidance if the file is missing or
the schema is malformed.
"""
function _load_dropout_fit_for_inflation(path::String)::DropoutFit
    isfile(path) || throw(ArgumentError(
        "dropout_curves.json not found at `$path`. " *
        "`:inflation` mode requires a dropout-curve fit; " *
        "run `fit_dropout_curves(...)` (needs `using GLM`) or set " *
        "mnar_variance_recovery = :off."
    ))
    local raw
    try
        raw = JSON3.read(Base.read(path, String))
    catch e
        throw(ArgumentError(
            "Failed to parse `$path` as JSON: $e. " *
            "Re-run `fit_dropout_curves` to regenerate."
        ))
    end
    required_keys = (:rho, :zeta, :column_names, :n_proteins,
                     :n_detections_per_column, :fit_timestamp,
                     :software_version, :dataset_hash)
    for k in required_keys
        haskey(raw, k) || throw(ArgumentError(
            "dropout_curves.json at `$path` is missing required key `:$k`. " *
            "Re-run `fit_dropout_curves` with v1.1.6+ to regenerate."
        ))
    end
    return DropoutFit(
        Float64.(raw[:rho]),
        Float64.(raw[:zeta]),
        String.(raw[:column_names]),
        Int(raw[:n_proteins]),
        Int.(raw[:n_detections_per_column]),
        String(raw[:fit_timestamp]),
        String(raw[:software_version]),
        String(raw[:dataset_hash]),
    )
end

"""
    _wrap_matrix_into_interaction_data(imputed_matrix::AbstractMatrix,
                                       raw_data::InteractionData)::InteractionData

inverse of `_build_intensity_matrix_for_inflation`. Takes a
`(n_proteins × n_columns)` intensity matrix (typically the output of
`impute_mnar`) and a template `raw_data::InteractionData` whose schema
(protein IDs, protocol layout, experiment counts, replicate widths, detection
mask, position vectors) defines the target shape, then constructs a fresh
`InteractionData` whose `.samples` and `.controls` carry the columns of the
input matrix in the EXACT order that `_build_intensity_matrix_for_inflation`
emits — i.e. protocol-by-protocol, samples-first then controls, experiments
in order, replicates left-to-right.

**Missing-handling convention:** values that are `NaN` in `imputed_matrix` map
back to `missing` in the output `Matrix{Union{Missing,F}}`; finite Float64
values pass through unchanged. This makes the round-trip
`_wrap_matrix_into_interaction_data(_build_intensity_matrix_for_inflation(raw),
raw)` an identity on `.samples` / `.controls` (the no-imputation invariant —
WARNING in the plan body). For `impute_mnar` output (no NaN — all cells
filled), every cell becomes a non-missing Float64, which is the intended
behaviour for multi-impute (the imputed datasets are treated as fully
observed by downstream BetaBernoulli / HBM / regression inference).

All non-intensity fields (`protein_IDs`, `protein_names`, `no_protocols`,
`no_experiments`, `no_parameters_HBM`, `no_parameters_Regression`,
`protocol_positions`, `experiment_positions`, `matched_positions`,
`detected`) are copied verbatim from `raw_data` via `deepcopy` — imputation
only mutates intensities, not the schema.
"""
function _wrap_matrix_into_interaction_data(
    imputed_matrix::AbstractMatrix,
    raw_data::InteractionData,
)::InteractionData
    n_proteins = length(getIDs(raw_data))
    @assert size(imputed_matrix, 1) == n_proteins (
        "imputed_matrix has $(size(imputed_matrix, 1)) rows, expected $n_proteins (one per protein)")

    # Walk the same protocol/experiment/replicate order as _build_intensity_matrix_for_inflation
    # and rebuild per-protocol Protocol{F,I} structs whose .data dicts carry fresh matrices.
    F = Float64
    new_samples  = Dict{Int, Protocol{F, Int}}()
    new_controls = Dict{Int, Protocol{F, Int}}()

    col = 0
    for p in 1:raw_data.no_protocols
        sp_raw = raw_data.samples[p]
        cp_raw = raw_data.controls[p]

        new_samples_data = Dict{Int, Matrix{Union{Missing, F}}}()
        for e in 1:sp_raw.no_experiments
            mat_raw = sp_raw.data[e]
            nrows, ncols = size(mat_raw)
            mat_new = Matrix{Union{Missing, F}}(undef, nrows, ncols)
            for r in 1:ncols
                col += 1
                for i in 1:nrows
                    v = imputed_matrix[i, col]
                    # NaN-sentinel -> missing (round-trip identity); finite -> Float64
                    mat_new[i, r] = (isa(v, Real) && isnan(v)) ? missing : F(v)
                end
            end
            new_samples_data[e] = mat_new
        end
        new_samples[p] = Protocol{F, Int}(
            sp_raw.no_experiments,
            copy(sp_raw.protein_ids),
            new_samples_data,
        )

        new_controls_data = Dict{Int, Matrix{Union{Missing, F}}}()
        for e in 1:cp_raw.no_experiments
            mat_raw = cp_raw.data[e]
            nrows, ncols = size(mat_raw)
            mat_new = Matrix{Union{Missing, F}}(undef, nrows, ncols)
            for r in 1:ncols
                col += 1
                for i in 1:nrows
                    v = imputed_matrix[i, col]
                    mat_new[i, r] = (isa(v, Real) && isnan(v)) ? missing : F(v)
                end
            end
            new_controls_data[e] = mat_new
        end
        new_controls[p] = Protocol{F, Int}(
            cp_raw.no_experiments,
            copy(cp_raw.protein_ids),
            new_controls_data,
        )
    end

    @assert col == size(imputed_matrix, 2) (
        "consumed $col matrix columns but matrix has $(size(imputed_matrix, 2)) — " *
        "schema mismatch between raw_data and imputed_matrix")

    return InteractionData{F, Int}(
        copy(raw_data.protein_IDs),
        copy(raw_data.protein_names),
        new_samples,
        new_controls,
        raw_data.no_protocols,
        copy(raw_data.no_experiments),
        raw_data.no_parameters_HBM,
        raw_data.no_parameters_Regression,
        copy(raw_data.protocol_positions),
        copy(raw_data.experiment_positions),
        copy(raw_data.matched_positions),
        copy(raw_data.detected),
    )
end

"""
    _generate_multi_impute_data(raw_data::InteractionData, dropout_fit::DropoutFit,
                                config::CONFIG) -> Vector{InteractionData}

build `config.mnar_m` MNAR-imputed `InteractionData`
objects in-process by calling `impute_mnar` serially with deterministic seeds
`seed_i = config.mnar_base_seed * 1_000_003 + i` for `i ∈ 1..config.mnar_m`.

Serial execution by design: nested threading over `m` imputations with the
existing per-protein `Threads.@threads` inside `analyse(...)` triggers
segfaults on Julia 1.12 / Windows (MEMORY.md note). Each `impute_mnar` call
is sub-second on HD, so the serial cost is negligible.

Writes the reproducibility manifest to
`<output.basedir>/imputed_data/dataset_mnar_multi_impute_manifest.json` per
the schema (extended with `mnar_m` and per-imputation seeds).

Requires `using GLM` (BayesInteractomicsImputationExt). Errors loudly via
`_require_imputation_extension(:mnar)` if the extension is absent — matches
the loud-error precedent.
"""
function _generate_multi_impute_data(
    raw_data::InteractionData,
    dropout_fit::DropoutFit,
    config::CONFIG,
)::Vector{InteractionData}
    # loud error BEFORE any imputation work if GLM/extension not loaded
    if !_imputation_extension_loaded()
        _require_imputation_extension(:mnar)
    end

    m = config.mnar_m
    base_seed = config.mnar_base_seed
    seeds = [Int(base_seed) * 1_000_003 + i for i in 1:m]

    # Extract the raw intensity matrix from raw_data (concatenated samples+controls
    # in the canonical column order — must match dropout_fit.column_names).
    # `_build_intensity_matrix_for_inflation` emits Float64-with-NaN; impute_mnar
    # consumes `Matrix{Union{Missing, Float64}}` — convert NaN -> missing here.
    raw_matrix_nan = _build_intensity_matrix_for_inflation(raw_data)
    @assert size(raw_matrix_nan, 2) == length(dropout_fit.rho) (
        "intensity matrix has $(size(raw_matrix_nan, 2)) cols but dropout_fit has " *
        "$(length(dropout_fit.rho)) — the dropout fit was run on a different dataset shape")
    raw_matrix_for_impute = Matrix{Union{Missing, Float64}}(undef, size(raw_matrix_nan)...)
    @inbounds for i in eachindex(raw_matrix_nan)
        v = raw_matrix_nan[i]
        raw_matrix_for_impute[i] = isnan(v) ? missing : v
    end

    # Convert DropoutFit -> per-column curves Dict (impute_mnar's expected signature).
    curves = Dict{Int, Tuple{Float64, Float64}}()
    for c in 1:length(dropout_fit.rho)
        curves[c] = (dropout_fit.rho[c], dropout_fit.zeta[c])
    end

    imputed_vec = Vector{InteractionData}(undef, m)
    per_imp_meta = Vector{Dict{String, Any}}(undef, m)

    for i in 1:m
        seed_i = seeds[i]
        # SERIAL — do NOT parallelise (MEMORY.md Julia 1.12 Windows segfault note)
        imputed_matrix, meta = impute_mnar(raw_matrix_for_impute, curves; seed = seed_i)
        per_imp_meta[i] = Dict{String, Any}(string(k) => v for (k, v) in pairs(meta))
        per_imp_meta[i]["seed"] = seed_i
        # Wrap imputed_matrix back into an InteractionData with the SAME schema as raw_data
        # (same protein IDs, same protocol layout, same detection mask — only intensities differ).
        imputed_vec[i] = _wrap_matrix_into_interaction_data(imputed_matrix, raw_data)
    end

    # Write the manifest (schema, extended with mnar_m + per-imputation seeds).
    manifest_dir = joinpath(config.output.basedir, "imputed_data")
    mkpath(manifest_dir)
    manifest_path = joinpath(manifest_dir, "dataset_mnar_multi_impute_manifest.json")
    raw_hash_hex = string(compute_data_hash(raw_data); base=16)
    manifest = Dict{String, Any}(
        "mnar_base_seed" => base_seed,
        "mnar_m" => m,
        "seeds" => seeds,
        "raw_dataset_hash" => "ui64:" * raw_hash_hex,
        "dropout_curves_dataset_hash" => dropout_fit.dataset_hash,
        "software_version" => string(pkgversion(@__MODULE__)),
        "timestamp" => string(Dates.now()),
        "imputation_method" => "mnar",
        "per_imputation_metadata" => per_imp_meta,
    )
    open(manifest_path, "w") do io
        JSON3.write(io, manifest)
    end
    @info ":multi_impute manifest written" path=manifest_path m=m seeds=seeds

    return imputed_vec
end

"""
    _input_for_pipeline_imputation(data::InteractionData, config::CONFIG) -> InteractionData

ordering seam. The SINGLE function boundary through
which the (already-normalised) observed `InteractionData` flows into the pipeline-driven
MNAR multi-imputation generator (`_generate_multi_impute_data`).

`data` reaches this point having ALREADY passed through `apply_normalisation` inside
`load_data` (see the load_data call at the top of `run_analysis` + the
apply site in `src/data/loading.jl`). This boundary makes the normalise-BEFORE-impute
ordering invariant explicit and hookable: a test can assert that the value returned here
equals `apply_normalisation(raw, resolved_method)` and NOT the raw, un-normalised data,
without reaching into `load_data`'s internals.

Currently the identity (no transform) — the normalisation already happened upstream and must
NOT be repeated here (double-apply would re-scale already-scaled data). The function exists
purely as a named, testable ordering seam; future ordering changes route through it.
"""
function _input_for_pipeline_imputation(data::InteractionData, config::CONFIG)::InteractionData
    return data
end

"""
    normalise_then_impute(raw_data::InteractionData, dropout_fit::DropoutFit;
                          normalisation_method::Symbol = :auto, refID::Int = 1) -> InteractionData

Correct-order entry point for the **user-pre-imputes** workflow:
normalise the RAW observed `InteractionData` FIRST, then MNAR-impute in-process.

Use this INSTEAD of `impute_mnar_from_paths` (which writes `dataset_mnar.xlsx`) followed by
`load_data(...; normalisation_method=...)`. That two-step file path normalises AFTER imputation
— the WRONG order: the MNAR dropout curve `σ(ρ_c + ζ_c·ȳ_i)` is intensity-scale-sensitive, so
imputing before normalising contaminates the size factors (imputation accuracy is
higher norm→impute for ALL normalisers; catastrophic for `vsn`, mild for `median_of_ratios`).

Steps:
1. Resolve `normalisation_method` (`:auto` → `detect_protocol_scale_mismatch` → `:both` or `:none`,
   identical to the `load_data` apply site).
2. `apply_normalisation(raw_data, resolved)` on the OBSERVED data.
3. Extract the post-normalisation intensity matrix, MNAR-impute once via the extension's
   `impute_mnar`, and round-trip back into a complete `InteractionData` mirroring `raw_data`'s
   schema.

Requires `using GLM` (BayesInteractomicsImputationExt); errors loudly via
`_require_imputation_extension(:mnar)` when the extension is absent.

Returns a complete imputed `InteractionData` (same protein IDs / protocol layout / detection
mask as `raw_data`, only intensities filled).
"""
function normalise_then_impute(
    raw_data::InteractionData,
    dropout_fit::DropoutFit;
    normalisation_method::Symbol = :auto,
    refID::Int = 1,
)::InteractionData
    # Loud error BEFORE any work if the imputation extension (GLM) is not loaded.
    if !_imputation_extension_loaded()
        _require_imputation_extension(:mnar)
    end

    # 1. Resolve the normalisation method — mirror the load_data :auto resolution EXACTLY.
    resolved = _resolve_normalisation_method(normalisation_method, false)
    if resolved === :auto
        if detect_protocol_scale_mismatch(raw_data; refID=refID)
            resolved = :both
            @info "normalise_then_impute: normalisation_method=:auto auto-applied :both " *
                  "(median_of_ratios + row-centering) — multi-protocol scale mismatch detected " *
                  "(pooled residual SD > 2.5 log2)."
        else
            resolved = :none
        end
    end

    # 2. NORMALISE FIRST (before imputation).
    normalised = apply_normalisation(raw_data, resolved)

    # 3. Impute in-process on the normalised data, then round-trip to InteractionData.
    raw_matrix_nan = _build_intensity_matrix_for_inflation(normalised)
    @assert size(raw_matrix_nan, 2) == length(dropout_fit.rho) (
        "intensity matrix has $(size(raw_matrix_nan, 2)) cols but dropout_fit has " *
        "$(length(dropout_fit.rho)) — dropout curves were fit on a different dataset shape")
    matrix_for_impute = Matrix{Union{Missing, Float64}}(undef, size(raw_matrix_nan)...)
    @inbounds for i in eachindex(raw_matrix_nan)
        v = raw_matrix_nan[i]
        matrix_for_impute[i] = isnan(v) ? missing : v
    end

    curves = Dict{Int, Tuple{Float64, Float64}}()
    for c in 1:length(dropout_fit.rho)
        curves[c] = (dropout_fit.rho[c], dropout_fit.zeta[c])
    end

    imputed_matrix, _meta = impute_mnar(matrix_for_impute, curves)
    return _wrap_matrix_into_interaction_data(imputed_matrix, normalised)
end

function run_analysis(config::CONFIG; use_cache::Bool=true, cache_file::String="", temp_result_file::String="temp_results.xlsx", use_intermediate_cache::Bool=true)
    # validate variance-recovery preconditions BEFORE any expensive work
    _validate_variance_recovery_config(config)
    # validate embeddings preconditions BEFORE any expensive work
    _validate_embeddings_config(config.embeddings_config)

    # Load data (needed for cache validation and analysis)
    data = load_data(
        config.datafile, config.sample_cols, config.control_cols,
        normalise_protocols = config.normalise_protocols,
        normalisation_method = config.normalisation_method,
        # thread refID into the :auto scale-mismatch detector.
        refID = config.refID,
        curate = config.curate,
        species = config.species,
        curate_interactive = config.curate_interactive,
        curate_merge_strategy = config.curate_merge_strategy,
        bait_name = config.bait_name,
        curate_replay = config.curate_replay,
        curate_remove_contaminants = config.curate_remove_contaminants,
        curate_delimiter = config.curate_delimiter,
        curate_auto_approve = config.curate_auto_approve,
        imputation = config.imputation_method
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

    # ------------------------------------------------------------------ #
    # :multi_impute early dispatch
    # When user has set mnar_variance_recovery = :multi_impute, build m MNAR
    # imputations in-process and forward into the existing Vector{InteractionData}
    # overload. Rubin pooling (compute_mixture / evaluate_imputed_fc_posteriors)
    # is unchanged (src/data/imputation.jl:35-206 untouched).
    # BLOCKER 4: dropout_curves.json path resolution goes through
    # shared `_resolve_dropout_curves_path(config)` — single source of truth.
    # ------------------------------------------------------------------ #
    if config.mnar_variance_recovery == :multi_impute
        # Resolve dropout_curves.json path via the shared helper (BLOCKER 4)
        dropout_path = _resolve_dropout_curves_path(config)
        dropout_fit_for_mi = _load_dropout_fit_for_inflation(dropout_path)
        @info ":multi_impute — loaded dropout_curves from $dropout_path"
        # ──────────────────────────────────────────────────────────────── #
        # ORDERING INVARIANT:
        #   Normalisation MUST run on the observed data BEFORE MNAR imputation.
        #   The MNAR dropout curve σ(ρ_c + ζ_c·ȳ_i) is intensity-scale-sensitive,
        # so imputing first contaminates the size factors (accuracy
        #   higher norm→impute for ALL normalisers).
        #
        #   This is GUARANTEED structurally: `data` was already produced by
        #   `load_data(...)` above, which applies `apply_normalisation` at its END
        #   (including the Task-1 :auto → :both resolution). `_input_for_pipeline_imputation`
        #   is the SINGLE, hookable boundary through which the normalised data flows
        #   into `_generate_multi_impute_data` — the multi-impute generator imputes
        #   in-process from this already-normalised InteractionData and NEVER re-reads
        #   or re-normalises (no double-apply).
        # ──────────────────────────────────────────────────────────────── #
        impute_input = _input_for_pipeline_imputation(data, config)
        # Build the Vector{InteractionData} via m serial impute_mnar calls
        imputed_vec = _generate_multi_impute_data(impute_input, dropout_fit_for_mi, config)
        @info ":multi_impute — generated $(length(imputed_vec)) imputed datasets, forwarding to Vector overload"
        # Forward to the existing multi-imputation overload
        return run_analysis(config, imputed_vec, data;
                            use_cache = use_cache,
                            cache_file = cache_file,
                            use_intermediate_cache = use_intermediate_cache)
    end

    # update DiagnosticsConfig
    config.diagnostics_config.n_proteins_to_check = length(data.protein_names)

    # Check bait detection (hard error — bait must be present for correlation tests)
    check_bait_detected(data, config.refID)

    # Run input data quality checks
    input_qc_result = nothing
    if config.run_input_qc
        try
            input_qc_result = run_input_qc(data)
            _emit_qc_warnings(input_qc_result)
        catch e
            @warn "[QC] Input quality check failed, continuing with analysis" exception=(e, catch_backtrace())
        end
    end

    # Check cache if enabled
    cache_path = isempty(cache_file) ? get_cache_filepath(config) : cache_file
    analysis_result = nothing

    # Restore optimized ν from cache before hash check (avoids cache invalidation
    # when optimize_nu=true changes student_t_nu after the cache was saved)
    if use_cache && config.optimize_nu && config.regression_likelihood == :robust_t && isfile(cache_path)
        try
            cached_nu = JLD2.load(cache_path, "optimal_nu")
            if !isnothing(cached_nu)
                config.student_t_nu = cached_nu
                @info "Restored optimized ν=$(round(cached_nu, digits=2)) from cache"
            end
        catch
            # Cache doesn't have optimal_nu (older format) — proceed normally
        end
    end

    if use_cache
        status, cached = check_cache(cache_path, config, data)
        if status == CACHE_HIT
            # Use cached results
            analysis_result = cached
            @info "Using cached analysis results"
            if !isnothing(input_qc_result)
                analysis_result.input_qc = input_qc_result
            end
        elseif status == CACHE_MISS_CONFIG
            @info "Config changed, running full analysis"
        elseif status == CACHE_MISS_DATA
            @info "Data changed, running full analysis"
        else  # CACHE_MISS_NO_FILE
            @info "No cache found, running full analysis"
        end
    end

    # ν optimization (runs BEFORE main analysis so the optimal ν is used for BF computation)
    nu_opt = nothing
    if isnothing(analysis_result) && config.optimize_nu && config.regression_likelihood == :robust_t
        nu_cache_path = get_nu_cache_filepath(config)
        cached_nu = load_nu_cache(nu_cache_path;
            data_hash         = compute_data_hash(data),
            likelihood        = config.regression_likelihood,
            imputation_method = config.imputation_method)
        if !isnothing(cached_nu)
            config.student_t_nu = cached_nu.optimal_nu
            @info "Loaded ν from cache" path=nu_cache_path nu=cached_nu.optimal_nu waic=cached_nu.optimal_waic
        else
            try
                @info "Running Student-t ν optimization via Brent's method..."
                nu_opt = optimize_nu(data, config)
                config.student_t_nu = nu_opt.optimal_nu
                @info "Optimal ν = $(round(nu_opt.optimal_nu, digits=2)), WAIC = $(round(nu_opt.optimal_waic.waic, digits=1))"

                # Persist ν cache for fast restart
                try
                    save_nu_cache(nu_cache_path;
                        data_hash         = compute_data_hash(data),
                        optimal_nu        = nu_opt.optimal_nu,
                        optimal_waic      = nu_opt.optimal_waic.waic,
                        normal_waic       = nu_opt.normal_waic.waic,
                        n_evaluations     = length(nu_opt.nu_trace),
                        likelihood        = config.regression_likelihood,
                        imputation_method = config.imputation_method)
                    @info "Saved ν cache to: $nu_cache_path"
                catch e
                    @warn "Failed to save ν cache: $e"
                end

                # Save ν optimization plot
                try
                    nu_optimization_plot(nu_opt; file = config.output.nu_optimization_file)
                    @info "Saved ν optimization plot to: $(config.output.nu_optimization_file)"
                catch e
                    @warn "Failed to save ν optimization plot: $e"
                end
            catch e
                @warn "Student-t ν optimization failed: $e"
            end
        end
    end

    # Generate intermediate cache file paths (defined outside the `if isnothing(analysis_result)`
    # block so that downstream sensitivity / diagnostic stages can reuse `h0_cache_path` even
    # when the AnalysisResult was loaded from cache).
    bb_cache_path = use_intermediate_cache ? get_betabernoulli_cache_filepath(config) : ""
    hbm_cache_path = use_intermediate_cache ? get_hbm_regression_cache_filepath(config) : ""
    h0_cache_path = use_intermediate_cache ? get_h0_cache_filepath(config) : ""

    # Run analysis if not cached
    if isnothing(analysis_result)
        # load dropout curves once if :inflation is active
        dropout_fit_for_inflation = nothing
        if config.mnar_variance_recovery == :inflation
            # BLOCKER 4: single source of truth for path resolution; reuses the same helper.
            dropout_path = _resolve_dropout_curves_path(config)
            dropout_fit_for_inflation = _load_dropout_fit_for_inflation(dropout_path)
            @info ":inflation — loaded dropout_curves from $dropout_path"
        end

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
            h0_cache_file   = h0_cache_path,
            # Copula-EM parameters
            prior           = config.em_prior,
            n_restarts      = config.em_n_restarts,
            copula_criterion = config.copula_criterion,
            copula_family   = config.copula_family,
            h1_copula_family = config.h1_copula_family,
            streams         = config.evidence_streams,
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
            student_t_nu = config.student_t_nu,
            regression_bf_threshold = config.regression_bf_threshold,
            jzs_r_scale = config.jzs_r_scale,
            regression_min_posterior_var = config.regression_min_posterior_var,
            imputation = config.imputation_method,
            # variance recovery
            variance_recovery = config.mnar_variance_recovery,
            dropout_fit = dropout_fit_for_inflation,
            inflation_max = config.mnar_inflation_max,
            inflation_override = config.mnar_inflation_factor,
            # bb_mnar_codriven thresholds
            bb_mnar_codriven = config.bb_mnar_codriven,
            # NEW: mask-aware regression flag (CONFIG → analyse → main → regression)
            mask_aware_regression = config.mask_aware_regression,
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
            nothing,  # diagnostics
            nothing,  # input_qc
            :extension_not_loaded  # metalearner_status (overwritten by _safe_predict_metalearner later)
        )

        if !isnothing(input_qc_result)
            analysis_result.input_qc = input_qc_result
        end

        # Save to cache
        if use_cache
            try
                save_result(analysis_result, cache_path)
                # Store optimized ν alongside cache so it can be restored before hash check
                if config.optimize_nu && config.regression_likelihood == :robust_t
                    jldopen(cache_path, "a+") do f
                        f["optimal_nu"] = config.student_t_nu
                    end
                end
                @info "Saved results to cache: $cache_path"
            catch e
                @warn "Failed to save cache: $e"
            end
        end

        # Save convergence plot (non-fatal: a degenerate Plots layout must not abort the
        # analysis, whose results are already cached above — mirrors the EM/LC plot guards below)
        try
            if config.combination_method in (:copula, :bma)
                StatsPlots.savefig(results.convergence_plt, config.output.convergence_file)
            elseif config.combination_method == :latent_class
                StatsPlots.savefig(results.convergence_plt, config.output.lc_convergence_file)
            end
        catch e
            @warn "Failed to save convergence plot (continuing): $e"
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


        # BMA weights diagnostic plot (for bma only)
        if config.combination_method == :bma && !isnothing(results.bma_result)
            try
                bma_weights_plot(results.bma_result; file = config.output.bma_weights_file)
                @info "Saved BMA weights plot to: $(config.output.bma_weights_file)"
            catch e
                @warn "Failed to save BMA weights plot: $e"
            end
        end

        # --- Copula Diagnostics ---
        if config.run_copula_diagnostics && config.combination_method in (:copula, :bma)
            config.verbose && @info "Running copula diagnostics..."
            try
                # Filter to detected proteins only (non-detected have missing BFs)
                detected_mask_cd = if hasproperty(results.copula_results, :is_detected)
                    coalesce.(results.copula_results.is_detected, false)
                else
                    trues(nrow(results.copula_results))
                end
                # Also exclude rows where any BF column is still missing
                detected_mask_cd .&= .!ismissing.(results.copula_results.bf_enrichment)
                detected_mask_cd .&= .!ismissing.(results.copula_results.bf_correlation)
                detected_mask_cd .&= .!ismissing.(results.copula_results.bf_detected)
                bf_triplet = BayesFactorTriplet(
                    Float64.(results.copula_results.bf_enrichment[detected_mask_cd]),
                    Float64.(results.copula_results.bf_correlation[detected_mask_cd]),
                    Float64.(results.copula_results.bf_detected[detected_mask_cd])
                )
                ppt = posterior_probability_from_bayes_factor(bf_triplet)
                ppt_sq = squeeze(ppt, ϵ = 1e-6)

                # Determine CombinedBayesResult to use (filter to detected for Float64 vectors)
                cbr = if !isnothing(results.bma_result)
                    results.bma_result.copula_result
                elseif !isnothing(results.em)
                    det_bf = Float64.(results.copula_results.combined_BF[detected_mask_cd])
                    det_pp = Float64.(results.copula_results.posterior_prob[detected_mask_cd])
                    CombinedBayesResult(
                        det_bf, det_pp,
                        results.joint_H0, results.joint_H1,
                        results.em, results.em_diagnostics)
                else
                    nothing
                end

                if cbr !== nothing
                    # 1. KL divergence (use new log-BF signature if LC result available)
                    try
                        if !isnothing(results.latent_class_result) && results.latent_class_result.responsibilities !== nothing
                            kl_result = kl_h1_divergence(bf_triplet, results.latent_class_result)
                            kl_divergence_plot(kl_result, bf_triplet; file = config.output.kl_divergence_file)
                        else
                            kl_result = kl_h1_divergence(cbr, ppt_sq)
                            kl_divergence_plot(kl_result, bf_triplet; file = config.output.kl_divergence_file)
                        end
                        @info "Saved KL divergence plot to: $(config.output.kl_divergence_file)"
                    catch e
                        @warn "Failed KL divergence diagnostic: $e"
                    end

                    # 2. Within-class correlation (only if LC result exists)
                    if !isnothing(results.latent_class_result)
                        try
                            wc_result = within_class_correlation(bf_triplet, results.latent_class_result)
                            within_class_correlation_plot(wc_result, bf_triplet; file = config.output.within_class_corr_file)
                            @info "Saved within-class correlation plot to: $(config.output.within_class_corr_file)"
                        catch e
                            @warn "Failed within-class correlation diagnostic: $e"
                        end
                    end

                    # 3. Agnostic zone
                    try
                        az_result = agnostic_zone_analysis(bf_triplet, ppt_sq, cbr)
                        agnostic_zone_plot(az_result, ppt_sq, cbr; file = config.output.agnostic_zone_file)
                        @info "Saved agnostic zone plot to: $(config.output.agnostic_zone_file)"
                    catch e
                        @warn "Failed agnostic zone diagnostic: $e"
                    end

                    # 4. Copula bootstrap CI
                    try
                        boot_result = copula_bootstrap_ci(ppt_sq, cbr)
                        copula_bootstrap_plot(boot_result; file = config.output.copula_bootstrap_file)
                        @info "Saved copula bootstrap plot to: $(config.output.copula_bootstrap_file)"
                    catch e
                        @warn "Failed copula bootstrap diagnostic: $e"
                    end

                    # 5. Discordant proteins
                    try
                        disc_result = discordant_protein_analysis(bf_triplet, cbr, ppt_sq)
                        discordant_protein_plot(disc_result; file = config.output.discordant_proteins_file)
                        @info "Saved discordant proteins plot to: $(config.output.discordant_proteins_file)"
                    catch e
                        @warn "Failed discordant protein diagnostic: $e"
                    end
                end
            catch e
                @warn "Failed to run copula diagnostics" exception=(e, catch_backtrace())
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
                h0_cache_file = h0_cache_path,
                combination_method = config.combination_method,
                lc_n_iterations = config.lc_n_iterations,
                lc_convergence_tol = config.lc_convergence_tol,
                verbose = config.verbose
            )
            analysis_result.sensitivity = sr

            # Generate sensitivity plots
            try
                sensitivity_rank_correlation(sr; file = config.output.sensitivity_rankcorr_file)
                @info "Saved sensitivity rank correlation plot to: $(config.output.sensitivity_rankcorr_file)"
            catch e
                @warn "Failed to save sensitivity rank correlation plot: $e"
            end

            generate_sensitivity_report(sr;
                filename = config.output.sensitivity_report_file,
                rankcorr_file = config.output.sensitivity_rankcorr_file
            )
            @info "Sensitivity report saved to: $(config.output.sensitivity_report_file)"

            try
                writetable(config.output.sensitivity_table_file, "sensitivity" => sr.summary; overwrite = true)
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

    # Attach ν optimization results to diagnostics (optimization ran before analyse())
    if !isnothing(nu_opt) && !isnothing(analysis_result.diagnostics)
        try
            model_cmp_from_nu = ModelComparisonResult(
                nu_opt.normal_waic, nu_opt.optimal_waic,
                nu_opt.delta_waic, nu_opt.delta_se,
                nu_opt.delta_waic > 0 ? :robust : :normal
            )
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
        catch e
            @warn "Failed to attach ν optimization results to diagnostics: $e"
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
                pit_histogram_file = config.output.pit_histogram_file,
                scale_location_hbm_file = config.output.scale_location_hbm_file,
                scale_location_regression_file = config.output.scale_location_regression_file,
                nu_optimization_file = config.output.nu_optimization_file
            )
            @info "Diagnostics report saved to: $(config.output.diagnostics_report_file)"
        catch e
            @warn "Failed to generate diagnostics report" exception=(e, catch_backtrace())
        end
    end

    # compute embeddings if config requests them.
    # AFTER calibration + sensitivity + diagnostics, BEFORE metalearner / clean_result.
    if config.embeddings_config.run_embeddings &&
       _should_recompute_embeddings(analysis_result, config.embeddings_config)
        try
            analysis_result.embeddings = _compute_embeddings(data, analysis_result, config.embeddings_config)
        catch e
            @warn "[Embeddings] computation failed: $e; report will render PCA only" maxlog=1 exception=(e, catch_backtrace())
            analysis_result.embeddings = nothing
        end
    end

    # Variante B graceful fallback when metalearner extension not loaded
    # 4-tuple destructure captures embedding_matrix for MC-Dropout reuse
    meta_data, _, embedding_matrix, ml_status = _safe_predict_metalearner(config)
    analysis_result.metalearner_status = ml_status

    # Update posterior probabilities with meta-learner predictions when available;
    # otherwise fall back to the BF-derived posterior already in analysis_result.
    final_results = copy(analysis_result.copula_results)  # Copy to avoid mutating cached data
    if config.use_metalearner_prior && ml_status === :loaded && !isnothing(meta_data)
        final_results = update_posterior_prob!(final_results, meta_data)
    end
    # When ml_status ∈ (:extension_not_loaded, :prediction_failed), final_results already
    # holds the BF-derived posterior_prob from the upstream EM/copula stage. No action needed —
    # _safe_predict_metalearner has already emitted the one-shot @warn.

    # DNN Prior + MC-Dropout uncertainty (Variante B fallback applies).
    # Must run BEFORE the PEP/BFDR write (Pitfall 3) AND BEFORE the
    # `analysis_result.copula_results = final_results` mutation sink (Pitfall 4).
    # The metalearner scores only the feature-available protein subset, so the MC
    # stats (one per embedding_matrix column = one per meta_data row) must be name-
    # aligned onto final_results; pass meta_data's protein names in column order.
    _mc_names = (ml_status === :loaded && !isnothing(meta_data) && hasproperty(meta_data, :Protein)) ?
                    string.(meta_data.Protein) : nothing
    _safe_compute_mc_prior!(final_results, embedding_matrix, config; protein_names = _mc_names)

    # PEP/BFDR write MOVED to after the calibration block (~line 3435)
    # so it can consume coalesce(posterior_calibrated, posterior_prob). The early write
    # is retained as :pep canonical + :PEP mirror in a placeholder form (raw posterior)
    # so downstream merges/sort below see a populated column; the calibrated rewrite
    # at the end of the calibration block overwrites with the coalesced quantity.
    final_results.pep = pep(final_results.posterior_prob)
    final_results.PEP = final_results.pep                    # silent mirror
    final_results.BFDR = bfdr(final_results.posterior_prob, isBF = false)

    # Sort dataframe by Bayesian FDR (BFDR) and log2FC
    sort!(final_results, [:BFDR, :mean_log2FC], rev = [false, false])

    # Merge diagnostic and sensitivity columns into final results
    if !isnothing(analysis_result.diagnostics) || !isnothing(analysis_result.sensitivity)
        final_results = _merge_diagnostics_to_results(
            final_results, analysis_result.diagnostics;
            sensitivity = analysis_result.sensitivity
        )
    end

    # Append PSIS_unreliable flag for proteins with Pareto k > 0.7
    if hasproperty(final_results, :pareto_k)
        if !hasproperty(final_results, :diagnostic_flag)
            final_results.diagnostic_flag = fill("", nrow(final_results))
        end
        for i in 1:nrow(final_results)
            pk_val = final_results.pareto_k[i]
            if !ismissing(pk_val) && pk_val > 0.7
                existing = final_results.diagnostic_flag[i]
                if ismissing(existing) || isempty(string(existing))
                    final_results[i, :diagnostic_flag] = "PSIS_unreliable"
                else
                    final_results[i, :diagnostic_flag] = string(existing) * "; PSIS_unreliable"
                end
            end
        end
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

    # Generate plots (always regenerate for current run)
    volcano_plt = volcano_plot(final_results)
    StatsPlots.savefig(volcano_plt, config.output.volcano_file)

    rrp = rank_rank_plot(final_results)
    StatsPlots.savefig(rrp, config.output.rank_rank_file)

    # Diagnostic plot needs the metalearner columns; skip when absent (e.g. synthetic /
    # no-metalearner runs) so the calibration block downstream is still reached. No-op when
    # the columns exist (any metalearner run), so HTT/HAP40/real-world output is unchanged.
    if hasproperty(final_results, :MetaClassifier) && hasproperty(final_results, :DNN)
        plot_results(final_results)
    end
    evd_plot = evidence_plot(final_results)
    StatsPlots.savefig(evd_plot, config.output.evidence_file)

    # Clean up temporary files
    isfile(temp_result_file) && rm(temp_result_file, force = true)

    # Generate docking requests (optional post-analysis step)
    if config.run_docking && !isempty(config.bait_sequence)
        try
            dc = something(config.docking_config, DockingConfig())
            dc.species = config.species
            if isempty(dc.cache_dir) && !isempty(config.datafile)
                dc.cache_dir = joinpath(dirname(abspath(config.datafile[1])), ".bayesinteractomics_cache", "docking")
            end
            batch = generate_docking_requests(
                final_results, config.bait_sequence;
                bait_name = config.poi,
                config = dc,
            )
            @info "Docking requests generated: $(batch.n_requests) requests in $(batch.n_batches) batches"
        catch e
            @warn "Docking request generation failed" exception=e
        end
    end

    # Automated validation gates
    validation_result = nothing
    if config.run_validation
        if config.verbose
            @info "Running automated validation..."
        end
        lc_result = nothing
        if hasproperty(analysis_result, :latent_class_result)
            lc_result = analysis_result.latent_class_result
        elseif hasproperty(analysis_result, :bma_result) && analysis_result.bma_result !== nothing
            lc_result = analysis_result.bma_result.em3c_result
        end

        # Filter to proteins that survived inference — lc_result.responsibilities
        # has one row per inferred protein. `is_detected` reflects data-level
        # detection (which can include proteins that failed HBM/regression and
        # therefore have missing BFs). Combine `is_detected` AND `!ismissing(bf_*)`
        # so bf_triplet length matches responsibilities exactly.
        is_detected_col = hasproperty(final_results, :is_detected) ?
            coalesce.(final_results.is_detected, false) : trues(nrow(final_results))
        non_missing_bf = .!ismissing.(final_results.bf_enrichment) .&
                         .!ismissing.(final_results.bf_correlation) .&
                         .!ismissing.(final_results.bf_detected)
        detected_mask_val = BitVector(is_detected_col .& non_missing_bf)
        detected_results_val = final_results[detected_mask_val, :]

        bf_triplet = BayesFactorTriplet(
            Vector{Float64}(detected_results_val.bf_enrichment),
            Vector{Float64}(detected_results_val.bf_correlation),
            Vector{Float64}(detected_results_val.bf_detected),
        )

        try
            validation_result = _run_validation(bf_triplet, lc_result, final_results, config)
        catch e
            @warn "Validation step failed" exception=e
        end
    end

    # Parametric simulation (optional)
    simulation_result = nothing
    if config.run_simulation
        lc_for_sim = nothing
        if hasproperty(analysis_result, :latent_class_result) && analysis_result.latent_class_result !== nothing
            lc_for_sim = analysis_result.latent_class_result
        elseif hasproperty(analysis_result, :bma_result) && analysis_result.bma_result !== nothing
            lc_for_sim = analysis_result.bma_result.em3c_result
        end
        if !isnothing(lc_for_sim)
            try
                simulation_result = run_simulation(lc_for_sim;
                    n_synthetic = config.sim_n_synthetic,
                    cache_file  = config.output.simulation_file)
                config.verbose && @info "Simulation complete: $(length(simulation_result.scenarios)) scenarios"
            catch e
                @warn "Simulation failed" exception=e
            end
        else
            @warn "Cannot run simulation: no LatentClassResult available"
        end
    end

    # Load/save calibration cache separately (CAL-05: independent invalidation)
    if !isnothing(simulation_result)
        cal_cached = load_calibration_cache(calibration_cache_path(config); imputation_method=config.imputation_method)
        if !isnothing(cal_cached)
            (cal_m, fdr_m, cv_m) = cal_cached
            simulation_result = SimulationResult(
                simulation_result.scenarios,
                simulation_result.pi_h1_grid,
                simulation_result.effect_grid,
                simulation_result.n_synthetic,
                simulation_result.n_replicates,
                simulation_result.h1_enrichment_family,
                simulation_result.fdr_at_p95_range,
                cal_m,
                fdr_m,
                cv_m,
                simulation_result.fdr_curve_empirical,
                simulation_result.fdr_curve_declared_bfdr
            )
            config.verbose && @info "Calibration loaded from separate cache: $(calibration_cache_path(config))"
        elseif !isnothing(simulation_result.calibration_model)
            try
                save_calibration_cache(simulation_result, calibration_cache_path(config); imputation_method=config.imputation_method)
                config.verbose && @info "Calibration cache saved to $(calibration_cache_path(config))"
            catch e
                @warn "Failed to save calibration cache" exception=e
            end
        end
    end

    # Apply isotonic calibration to real posteriors (guarded)
    # `should_calibrate` lifted out of the try block so it survives
    # for the post-calibration PEP/BFDR rewrite + is_calibrated wire below.
    should_calibrate = false
    if !isnothing(simulation_result) && !isnothing(simulation_result.calibration_model)
        try
            cal_model = simulation_result.calibration_model
            fdr_model = simulation_result.fdr_calibration_model
            epsilon   = 1e-6
            n_rows    = nrow(final_results)

            # ECE guard — skip calibration if it does not improve ECE
            cal_ece = simulation_result.calibration_cv !== nothing ?
                      simulation_result.calibration_cv.posterior_ece_mean : Inf
            should_calibrate = cal_ece < 0.10  # Only calibrate if CV ECE is reasonable

            if !should_calibrate
                @warn "Calibration skipped: CV ECE=$(round(cal_ece, digits=4)) >= 0.10 threshold; keeping raw posteriors"
            else
                post_cal = Vector{Union{Missing, Float64}}(missing, n_rows)
                fdr_cal  = Vector{Union{Missing, Float64}}(missing, n_rows)

                for i in 1:n_rows
                    p = final_results.posterior_prob[i]
                    if !ismissing(p)
                        post_cal[i] = _apply_calibration(Float64(p), cal_model; epsilon=epsilon)
                        if !isnothing(fdr_model)
                            fdr_cal[i] = _apply_fdr_calibration(Float64(p), fdr_model)
                        end
                    end
                end

                # Insert after identifier columns — check for existing columns first (re-run safety)
                if "posterior_calibrated" in names(final_results)
                    final_results.posterior_calibrated = post_cal
                    final_results.fdr_calibrated = fdr_cal
                else
                    # Find insertion point: after identifier columns (Protein, original_name, string_id)
                    id_cols = intersect(["Protein", "original_name", "string_id"], names(final_results))
                    insert_pos = length(id_cols) + 1
                    insertcols!(final_results, insert_pos, :posterior_calibrated => post_cal)
                    insertcols!(final_results, insert_pos + 1, :fdr_calibrated => fdr_cal)
                end

                config.verbose && @info "Calibration applied: ECE=$(round(cal_ece, digits=4))"
            end
        catch e
            @warn "Failed to apply isotonic calibration to results" exception=e
            should_calibrate = false  # ensure flag reflects actual outcome on error
        end
    end

    # stash dataset-level ECE-gate provenance on the AR struct
    # (AnalysisResult is mutable; field added downstream).
    analysis_result.is_calibrated = should_calibrate

    # PEP/BFDR rewrite using coalesced posterior (calibrated when
    # available, raw otherwise). Overwrites the placeholder pep/BFDR populated
    # before the calibration block.
    # T-70-11-01 mitigation: when posterior_calibrated is present, its missing-row
    # pattern must mirror posterior_prob exactly. Per-row missing in
    # posterior_calibrated where posterior_prob is non-missing indicates a
    # regression in the calibration application loop — fail loudly.
    if hasproperty(final_results, :posterior_calibrated)
        for i in 1:nrow(final_results)
            if !ismissing(final_results.posterior_prob[i]) &&
               ismissing(final_results.posterior_calibrated[i])
                @warn "T-70-11-01: posterior_calibrated is missing at row $i while posterior_prob is non-missing; calibration may have regressed"
                break
            end
        end
        post_for_pep = coalesce.(final_results.posterior_calibrated, final_results.posterior_prob)
    else
        post_for_pep = final_results.posterior_prob
    end
    final_results.pep = pep(post_for_pep)                    # canonical lowercase
    final_results.PEP = final_results.pep                    # silent mirror (same Vector reference)
    final_results.BFDR = bfdr(post_for_pep, isBF = false)

    # Re-sort by the (possibly calibrated) BFDR + log2FC
    sort!(final_results, [:BFDR, :mean_log2FC], rev = [false, false])

    # Re-publish to the AR struct so downstream consumers see the calibrated quantities.
    analysis_result.copula_results = final_results

    # Save final results (after calibration so calibrated columns are included).
    # final_results carries `bb_mnar_codriven::Bool`
    # propagated from the copula_df builder above + _merge_diagnostics_to_results.
    # spike P8 confirmed the column is present at position 17 across all 4
    # production xlsx outputs. Regression locked in
    # test/analysis/test_bb_mnar_codriven_xlsx.jl.
    writetable(config.output.results_file, "df" => final_results; overwrite = true)

    # Generate interactive HTML report
    if config.generate_report_html
        try
            sr = hasproperty(analysis_result, :sensitivity) ? analysis_result.sensitivity : nothing
            generate_report(final_results, config; analysis_result=analysis_result,
                            validation_result=validation_result,
                            sensitivity_result=sr,
                            simulation_result=simulation_result)
        catch e
            @warn "Failed to generate interactive HTML report" exception=(e, catch_backtrace())
        end
    end

    # (Pitfalls 2 + 4): expose per-bait simulation result + CONFIG on the
    # returned AnalysisResult so the differential report's per-condition tabs can iterate
    # diff.analyses[i].simulation_result and diff.analyses[i].config. AnalysisResult is
    # `mutable struct` (precedent — metalearner_status mutated above), so
    # post-construction field assignment is legal.
    analysis_result.simulation_result = simulation_result  # may be nothing when run_simulation=false
    analysis_result.config = config

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
    # validate variance-recovery preconditions BEFORE any expensive work
    _validate_variance_recovery_config(config)
    # validate embeddings preconditions BEFORE any expensive work
    _validate_embeddings_config(config.embeddings_config)

    # fix: HARD guard against raw↔imputed protein index misalignment.
    # The multi-imputation `analyse(imputed, raw, …)` loop indexes imputed_data[j] AND
    # raw_data by the SAME protein index i, so their ID vectors MUST be identical and
    # identically ordered. The classic cause of misalignment is loading raw with the
    # default n_obs<2 exclusion while the filled imputed file keeps those proteins — this
    # silently pairs mismatched proteins (wrong is_imputed mask) and drops the difference.
    let raw_ids = getIDs(raw_data)
        for (j, imp) in enumerate(imputed_data)
            imp_ids = getIDs(imp)
            if imp_ids != raw_ids
                nmin = min(length(imp_ids), length(raw_ids))
                nmatch = count(i -> imp_ids[i] == raw_ids[i], 1:nmin)
                throw(ArgumentError(
                    "run_analysis(config, imputed, raw): imputed_data[$j] ($(length(imp_ids)) proteins) " *
                    "is NOT index-aligned with raw_data ($(length(raw_ids)) proteins) — only " *
                    "$nmatch/$nmin positions match. The multi-imputation pipeline indexes both by the " *
                    "same protein index, so misalignment silently pairs mismatched proteins and produces " *
                    "a garbage is_imputed mask. Fix: load BOTH raw and imputed with " *
                    "`load_data(...; filter_insufficient_obs=false)` so neither drops sparse proteins."
                ))
            end
        end
    end

    # Check cache with combined hash of imputed + raw data
    cache_path = isempty(cache_file) ? get_cache_filepath(config) : cache_file
    analysis_result = nothing

    # Restore optimized ν from cache before hash check
    if use_cache && config.optimize_nu && config.regression_likelihood == :robust_t && isfile(cache_path)
        try
            cached_nu = JLD2.load(cache_path, "optimal_nu")
            if !isnothing(cached_nu)
                config.student_t_nu = cached_nu
                @info "Restored optimized ν=$(round(cached_nu, digits=2)) from cache"
            end
        catch
        end
    end

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

    # Run input data quality checks (on raw data, not imputed)
    input_qc_result = nothing
    if config.run_input_qc
        try
            input_qc_result = run_input_qc(raw_data)
            _emit_qc_warnings(input_qc_result)
        catch e
            @warn "[QC] Input quality check failed, continuing with analysis" exception=(e, catch_backtrace())
        end
    end

    # Attach QC to cached result if cache hit
    if !isnothing(analysis_result) && !isnothing(input_qc_result)
        analysis_result.input_qc = input_qc_result
    end

    # ν optimization (runs BEFORE main analysis so the optimal ν is used for BF computation)
    nu_opt = nothing
    if isnothing(analysis_result) && config.optimize_nu && config.regression_likelihood == :robust_t
        nu_cache_path = get_nu_cache_filepath(config)
        cached_nu = load_nu_cache(nu_cache_path;
            data_hash         = compute_data_hash(imputed_data[1]),
            likelihood        = config.regression_likelihood,
            imputation_method = config.imputation_method)
        if !isnothing(cached_nu)
            config.student_t_nu = cached_nu.optimal_nu
            @info "Loaded ν from cache (multi-imp)" path=nu_cache_path nu=cached_nu.optimal_nu waic=cached_nu.optimal_waic
        else
            try
                @info "Running Student-t ν optimization via Brent's method (using first imputed dataset)..."
                nu_opt = optimize_nu(imputed_data[1], config)
                config.student_t_nu = nu_opt.optimal_nu
                @info "Optimal ν = $(round(nu_opt.optimal_nu, digits=2)), WAIC = $(round(nu_opt.optimal_waic.waic, digits=1))"

                # Persist ν cache for fast restart
                try
                    save_nu_cache(nu_cache_path;
                        data_hash         = compute_data_hash(imputed_data[1]),
                        optimal_nu        = nu_opt.optimal_nu,
                        optimal_waic      = nu_opt.optimal_waic.waic,
                        normal_waic       = nu_opt.normal_waic.waic,
                        n_evaluations     = length(nu_opt.nu_trace),
                        likelihood        = config.regression_likelihood,
                        imputation_method = config.imputation_method)
                    @info "Saved ν cache to: $nu_cache_path"
                catch e
                    @warn "Failed to save ν cache: $e"
                end

                # Save ν optimization plot
                try
                    nu_optimization_plot(nu_opt; file = config.output.nu_optimization_file)
                    @info "Saved ν optimization plot to: $(config.output.nu_optimization_file)"
                catch e
                    @warn "Failed to save ν optimization plot: $e"
                end
            catch e
                @warn "Student-t ν optimization failed: $e"
            end
        end
    end

    # Generate intermediate cache file paths (defined outside the `if isnothing(analysis_result)`
    # block so that downstream sensitivity / diagnostic stages can reuse `h0_cache_path` even
    # when the AnalysisResult was loaded from cache).
    bb_cache_path = use_intermediate_cache ? get_betabernoulli_cache_filepath(config) : ""
    hbm_cache_path = use_intermediate_cache ? get_hbm_regression_cache_filepath(config) : ""
    h0_cache_path = use_intermediate_cache ? get_h0_cache_filepath(config) : ""

    # Run analysis if not cached
    if isnothing(analysis_result)
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
            h0_cache_file   = h0_cache_path,
            # Copula-EM parameters
            prior           = config.em_prior,
            n_restarts      = config.em_n_restarts,
            copula_criterion = config.copula_criterion,
            copula_family   = config.copula_family,
            h1_copula_family = config.h1_copula_family,
            streams         = config.evidence_streams,
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
            student_t_nu = config.student_t_nu,
            regression_bf_threshold = config.regression_bf_threshold,
            jzs_r_scale = config.jzs_r_scale,
            regression_min_posterior_var = config.regression_min_posterior_var,
            imputation = config.imputation_method,
            # bb_mnar_codriven thresholds
            bb_mnar_codriven = config.bb_mnar_codriven,
            # NEW: mask-aware regression flag (CONFIG → analyse → main → regression).
            # dropout_fit defaults to nothing; the dropout_fit load wiring is deferred —
            # for now this overload accepts the kwarg but does not load the fit
            # automatically (callers may pre-load via `using GLM; fit_dropout_curves(...)`).
            mask_aware_regression = config.mask_aware_regression,
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
            nothing,  # diagnostics
            nothing,  # input_qc
            :extension_not_loaded  # metalearner_status (overwritten by _safe_predict_metalearner later)
        )

        if !isnothing(input_qc_result)
            analysis_result.input_qc = input_qc_result
        end

        # Save to cache
        if use_cache
            try
                save_result(analysis_result, cache_path)
                # Store optimized ν alongside cache so it can be restored before hash check
                if config.optimize_nu && config.regression_likelihood == :robust_t
                    jldopen(cache_path, "a+") do f
                        f["optimal_nu"] = config.student_t_nu
                    end
                end
                @info "Saved results to cache: $cache_path"
            catch e
                @warn "Failed to save cache: $e"
            end
        end

        # Save convergence plot (non-fatal: a degenerate Plots layout must not abort the
        # analysis, whose results are already cached above — mirrors the EM/LC plot guards below)
        try
            if config.combination_method in (:copula, :bma)
                StatsPlots.savefig(results.convergence_plt, config.output.convergence_file)
            elseif config.combination_method == :latent_class
                StatsPlots.savefig(results.convergence_plt, config.output.lc_convergence_file)
            end
        catch e
            @warn "Failed to save convergence plot (continuing): $e"
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


        # BMA weights diagnostic plot (for bma only)
        if config.combination_method == :bma && !isnothing(results.bma_result)
            try
                bma_weights_plot(results.bma_result; file = config.output.bma_weights_file)
                @info "Saved BMA weights plot to: $(config.output.bma_weights_file)"
            catch e
                @warn "Failed to save BMA weights plot: $e"
            end
        end

        # --- Copula Diagnostics ---
        if config.run_copula_diagnostics && config.combination_method in (:copula, :bma)
            config.verbose && @info "Running copula diagnostics..."
            try
                # Filter to detected proteins only (non-detected have missing BFs)
                detected_mask_cd = if hasproperty(results.copula_results, :is_detected)
                    coalesce.(results.copula_results.is_detected, false)
                else
                    trues(nrow(results.copula_results))
                end
                # Also exclude rows where any BF column is still missing
                detected_mask_cd .&= .!ismissing.(results.copula_results.bf_enrichment)
                detected_mask_cd .&= .!ismissing.(results.copula_results.bf_correlation)
                detected_mask_cd .&= .!ismissing.(results.copula_results.bf_detected)
                bf_triplet = BayesFactorTriplet(
                    Float64.(results.copula_results.bf_enrichment[detected_mask_cd]),
                    Float64.(results.copula_results.bf_correlation[detected_mask_cd]),
                    Float64.(results.copula_results.bf_detected[detected_mask_cd])
                )
                ppt = posterior_probability_from_bayes_factor(bf_triplet)
                ppt_sq = squeeze(ppt, ϵ = 1e-6)

                cbr = if !isnothing(results.bma_result)
                    results.bma_result.copula_result
                elseif !isnothing(results.em)
                    det_bf = Float64.(results.copula_results.combined_BF[detected_mask_cd])
                    det_pp = Float64.(results.copula_results.posterior_prob[detected_mask_cd])
                    CombinedBayesResult(
                        det_bf, det_pp,
                        results.joint_H0, results.joint_H1,
                        results.em, results.em_diagnostics)
                else
                    nothing
                end

                if cbr !== nothing
                    try
                        kl_result = kl_h1_divergence(cbr, ppt_sq)
                        kl_divergence_plot(kl_result, ppt_sq; file = config.output.kl_divergence_file)
                        @info "Saved KL divergence plot to: $(config.output.kl_divergence_file)"
                    catch e
                        @warn "Failed KL divergence diagnostic: $e"
                    end

                    if !isnothing(results.latent_class_result)
                        try
                            wc_result = within_class_correlation(bf_triplet, results.latent_class_result)
                            within_class_correlation_plot(wc_result, bf_triplet; file = config.output.within_class_corr_file)
                            @info "Saved within-class correlation plot to: $(config.output.within_class_corr_file)"
                        catch e
                            @warn "Failed within-class correlation diagnostic: $e"
                        end
                    end

                    try
                        az_result = agnostic_zone_analysis(bf_triplet, ppt_sq, cbr)
                        agnostic_zone_plot(az_result, ppt_sq, cbr; file = config.output.agnostic_zone_file)
                        @info "Saved agnostic zone plot to: $(config.output.agnostic_zone_file)"
                    catch e
                        @warn "Failed agnostic zone diagnostic: $e"
                    end

                    try
                        boot_result = copula_bootstrap_ci(ppt_sq, cbr)
                        copula_bootstrap_plot(boot_result; file = config.output.copula_bootstrap_file)
                        @info "Saved copula bootstrap plot to: $(config.output.copula_bootstrap_file)"
                    catch e
                        @warn "Failed copula bootstrap diagnostic: $e"
                    end

                    try
                        disc_result = discordant_protein_analysis(bf_triplet, cbr, ppt_sq)
                        discordant_protein_plot(disc_result; file = config.output.discordant_proteins_file)
                        @info "Saved discordant proteins plot to: $(config.output.discordant_proteins_file)"
                    catch e
                        @warn "Failed discordant protein diagnostic: $e"
                    end
                end
            catch e
                @warn "Failed to run copula diagnostics" exception=(e, catch_backtrace())
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
                h0_cache_file = h0_cache_path,
                combination_method = config.combination_method,
                lc_n_iterations = config.lc_n_iterations,
                lc_convergence_tol = config.lc_convergence_tol,
                verbose = config.verbose
            )
            analysis_result.sensitivity = sr

            # Generate sensitivity plots
            try
                sensitivity_rank_correlation(sr; file = config.output.sensitivity_rankcorr_file)
                @info "Saved sensitivity rank correlation plot to: $(config.output.sensitivity_rankcorr_file)"
            catch e
                @warn "Failed to save sensitivity rank correlation plot: $e"
            end

            generate_sensitivity_report(sr;
                filename = config.output.sensitivity_report_file,
                rankcorr_file = config.output.sensitivity_rankcorr_file
            )
            @info "Sensitivity report saved to: $(config.output.sensitivity_report_file)"

            try
                writetable(config.output.sensitivity_table_file, "sensitivity" => sr.summary; overwrite = true)
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

    # Attach ν optimization results to diagnostics (optimization ran before analyse())
    if !isnothing(nu_opt) && !isnothing(analysis_result.diagnostics)
        try
            model_cmp_from_nu = ModelComparisonResult(
                nu_opt.normal_waic, nu_opt.optimal_waic,
                nu_opt.delta_waic, nu_opt.delta_se,
                nu_opt.delta_waic > 0 ? :robust : :normal
            )
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
        catch e
            @warn "Failed to attach ν optimization results to diagnostics: $e"
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
                pit_histogram_file = config.output.pit_histogram_file,
                scale_location_hbm_file = config.output.scale_location_hbm_file,
                scale_location_regression_file = config.output.scale_location_regression_file,
                nu_optimization_file = config.output.nu_optimization_file
            )
            @info "Diagnostics report saved to: $(config.output.diagnostics_report_file)"
        catch e
            @warn "Failed to generate diagnostics report" exception=(e, catch_backtrace())
        end
    end

    # compute embeddings if config requests them.
    # AFTER calibration + sensitivity + diagnostics, BEFORE metalearner / clean_result.
    # pool the M imputed datasets element-wise and feed the pooled-mean
    # InteractionData into sample PCA/UMAP per ("post-imputation log-intensity matrix").
    # Without imputation, fall back to raw_data. NOTE: raw-vs-imputed contract applies to
    # missing_fraction + bb_mnar_codriven (which MEASURE original dropout), NOT to embeddings —
    # the previous comment here misapplied scope.
    if config.embeddings_config.run_embeddings &&
       _should_recompute_embeddings(analysis_result, config.embeddings_config)
        try
            embedding_input = isempty(imputed_data) ? raw_data : _pool_imputed_matrix(imputed_data)
            analysis_result.embeddings = _compute_embeddings(embedding_input, analysis_result, config.embeddings_config)
        catch e
            @warn "[Embeddings] computation failed: $e; report will render PCA only" maxlog=1 exception=(e, catch_backtrace())
            analysis_result.embeddings = nothing
        end
    end

    # Variante B graceful fallback when metalearner extension not loaded
    # 4-tuple destructure captures embedding_matrix for MC-Dropout reuse
    meta_data, _, embedding_matrix, ml_status = _safe_predict_metalearner(config)
    analysis_result.metalearner_status = ml_status

    # Update posterior probabilities with meta-learner predictions when available;
    # otherwise fall back to the BF-derived posterior already in analysis_result.
    final_results = copy(analysis_result.copula_results)  # Copy to avoid mutating cached data
    if config.use_metalearner_prior && ml_status === :loaded && !isnothing(meta_data)
        final_results = update_posterior_prob!(final_results, meta_data)
    end
    # When ml_status ∈ (:extension_not_loaded, :prediction_failed), final_results already
    # holds the BF-derived posterior_prob from the upstream EM/copula stage. No action needed —
    # _safe_predict_metalearner has already emitted the one-shot @warn.

    # DNN Prior + MC-Dropout uncertainty (Variante B fallback applies).
    # Must run BEFORE the PEP/BFDR write (Pitfall 3) AND BEFORE the
    # `analysis_result.copula_results = final_results` mutation sink (Pitfall 4).
    # The metalearner scores only the feature-available protein subset, so the MC
    # stats (one per embedding_matrix column = one per meta_data row) must be name-
    # aligned onto final_results; pass meta_data's protein names in column order.
    _mc_names = (ml_status === :loaded && !isnothing(meta_data) && hasproperty(meta_data, :Protein)) ?
                    string.(meta_data.Protein) : nothing
    _safe_compute_mc_prior!(final_results, embedding_matrix, config; protein_names = _mc_names)

    # placeholder PEP/BFDR write — coalesce-based rewrite happens
    # AFTER the calibration block at the end of this overload (imputed-vector path).
    final_results.pep = pep(final_results.posterior_prob)
    final_results.PEP = final_results.pep                    # silent mirror
    final_results.BFDR = bfdr(final_results.posterior_prob, isBF = false)

    # Sort dataframe by Bayesian FDR (BFDR) and log2FC
    sort!(final_results, [:BFDR, :mean_log2FC], rev = [false, false])

    # Merge diagnostic and sensitivity columns into final results
    if !isnothing(analysis_result.diagnostics) || !isnothing(analysis_result.sensitivity)
        final_results = _merge_diagnostics_to_results(
            final_results, analysis_result.diagnostics;
            sensitivity = analysis_result.sensitivity
        )
    end

    # Append PSIS_unreliable flag for proteins with Pareto k > 0.7
    if hasproperty(final_results, :pareto_k)
        if !hasproperty(final_results, :diagnostic_flag)
            final_results.diagnostic_flag = fill("", nrow(final_results))
        end
        for i in 1:nrow(final_results)
            pk_val = final_results.pareto_k[i]
            if !ismissing(pk_val) && pk_val > 0.7
                existing = final_results.diagnostic_flag[i]
                if ismissing(existing) || isempty(string(existing))
                    final_results[i, :diagnostic_flag] = "PSIS_unreliable"
                else
                    final_results[i, :diagnostic_flag] = string(existing) * "; PSIS_unreliable"
                end
            end
        end
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

    # Generate plots (always regenerate for current run; non-fatal — a degenerate Plots
    # layout from a very small sample must not abort the already-completed analysis)
    try
        volcano_plt = volcano_plot(final_results, legend_pos = config.vc_legend_pos)
        StatsPlots.savefig(volcano_plt, config.output.volcano_file)
    catch e
        @warn "Failed to save volcano plot (continuing): $e" maxlog=1
    end
    try
        rrp = rank_rank_plot(final_results)
        StatsPlots.savefig(rrp, config.output.rank_rank_file)
    catch e
        @warn "Failed to save rank-rank plot (continuing): $e" maxlog=1
    end
    try
        evd_plot = evidence_plot(final_results)
        StatsPlots.savefig(evd_plot, config.output.evidence_file)
    catch e
        @warn "Failed to save evidence plot (continuing): $e" maxlog=1
    end

    # Generate docking requests (optional post-analysis step)
    if config.run_docking && !isempty(config.bait_sequence)
        try
            dc = something(config.docking_config, DockingConfig())
            dc.species = config.species
            if isempty(dc.cache_dir) && !isempty(config.datafile)
                dc.cache_dir = joinpath(dirname(abspath(config.datafile[1])), ".bayesinteractomics_cache", "docking")
            end
            batch = generate_docking_requests(
                final_results, config.bait_sequence;
                bait_name = config.poi,
                config = dc,
            )
            @info "Docking requests generated: $(batch.n_requests) requests in $(batch.n_batches) batches"
        catch e
            @warn "Docking request generation failed" exception=e
        end
    end

    # Automated validation gates
    validation_result = nothing
    if config.run_validation
        if config.verbose
            @info "Running automated validation..."
        end
        lc_result = nothing
        if hasproperty(analysis_result, :latent_class_result)
            lc_result = analysis_result.latent_class_result
        elseif hasproperty(analysis_result, :bma_result) && analysis_result.bma_result !== nothing
            lc_result = analysis_result.bma_result.em3c_result
        end

        # Filter to proteins that survived inference — lc_result.responsibilities
        # has one row per inferred protein. `is_detected` reflects data-level
        # detection (which can include proteins that failed HBM/regression and
        # therefore have missing BFs). Combine `is_detected` AND `!ismissing(bf_*)`
        # so bf_triplet length matches responsibilities exactly.
        is_detected_col = hasproperty(final_results, :is_detected) ?
            coalesce.(final_results.is_detected, false) : trues(nrow(final_results))
        non_missing_bf = .!ismissing.(final_results.bf_enrichment) .&
                         .!ismissing.(final_results.bf_correlation) .&
                         .!ismissing.(final_results.bf_detected)
        detected_mask_val = BitVector(is_detected_col .& non_missing_bf)
        detected_results_val = final_results[detected_mask_val, :]

        bf_triplet = BayesFactorTriplet(
            Vector{Float64}(detected_results_val.bf_enrichment),
            Vector{Float64}(detected_results_val.bf_correlation),
            Vector{Float64}(detected_results_val.bf_detected),
        )

        try
            validation_result = _run_validation(bf_triplet, lc_result, final_results, config)
        catch e
            @warn "Validation step failed" exception=e
        end
    end

    # Parametric simulation (optional)
    simulation_result = nothing
    if config.run_simulation
        lc_for_sim = nothing
        if hasproperty(analysis_result, :latent_class_result) && analysis_result.latent_class_result !== nothing
            lc_for_sim = analysis_result.latent_class_result
        elseif hasproperty(analysis_result, :bma_result) && analysis_result.bma_result !== nothing
            lc_for_sim = analysis_result.bma_result.em3c_result
        end
        if !isnothing(lc_for_sim)
            try
                simulation_result = run_simulation(lc_for_sim;
                    n_synthetic = config.sim_n_synthetic,
                    cache_file  = config.output.simulation_file)
                config.verbose && @info "Simulation complete: $(length(simulation_result.scenarios)) scenarios"
            catch e
                @warn "Simulation failed" exception=e
            end
        else
            @warn "Cannot run simulation: no LatentClassResult available"
        end
    end

    # Load/save calibration cache separately — imputed path (CAL-05: independent invalidation)
    if !isnothing(simulation_result)
        cal_cached = load_calibration_cache(calibration_cache_path(config); imputation_method=config.imputation_method)
        if !isnothing(cal_cached)
            (cal_m, fdr_m, cv_m) = cal_cached
            simulation_result = SimulationResult(
                simulation_result.scenarios,
                simulation_result.pi_h1_grid,
                simulation_result.effect_grid,
                simulation_result.n_synthetic,
                simulation_result.n_replicates,
                simulation_result.h1_enrichment_family,
                simulation_result.fdr_at_p95_range,
                cal_m,
                fdr_m,
                cv_m,
                simulation_result.fdr_curve_empirical,
                simulation_result.fdr_curve_declared_bfdr
            )
            config.verbose && @info "Calibration loaded from separate cache: $(calibration_cache_path(config))"
        elseif !isnothing(simulation_result.calibration_model)
            try
                save_calibration_cache(simulation_result, calibration_cache_path(config); imputation_method=config.imputation_method)
                config.verbose && @info "Calibration cache saved to $(calibration_cache_path(config))"
            catch e
                @warn "Failed to save calibration cache" exception=e
            end
        end
    end

    # Apply isotonic calibration to real posteriors — imputed path (guarded)
    # `should_calibrate` lifted out of the try block (imputed-vector path).
    should_calibrate = false
    if !isnothing(simulation_result) && !isnothing(simulation_result.calibration_model)
        try
            cal_model = simulation_result.calibration_model
            fdr_model = simulation_result.fdr_calibration_model
            epsilon   = 1e-6
            n_rows    = nrow(final_results)

            # ECE guard — skip calibration if it does not improve ECE
            cal_ece = simulation_result.calibration_cv !== nothing ?
                      simulation_result.calibration_cv.posterior_ece_mean : Inf
            should_calibrate = cal_ece < 0.10  # Only calibrate if CV ECE is reasonable

            if !should_calibrate
                @warn "Calibration skipped: CV ECE=$(round(cal_ece, digits=4)) >= 0.10 threshold; keeping raw posteriors"
            else
                post_cal = Vector{Union{Missing, Float64}}(missing, n_rows)
                fdr_cal  = Vector{Union{Missing, Float64}}(missing, n_rows)

                for i in 1:n_rows
                    p = final_results.posterior_prob[i]
                    if !ismissing(p)
                        post_cal[i] = _apply_calibration(Float64(p), cal_model; epsilon=epsilon)
                        if !isnothing(fdr_model)
                            fdr_cal[i] = _apply_fdr_calibration(Float64(p), fdr_model)
                        end
                    end
                end

                # Insert after identifier columns — check for existing columns first (re-run safety)
                if "posterior_calibrated" in names(final_results)
                    final_results.posterior_calibrated = post_cal
                    final_results.fdr_calibrated = fdr_cal
                else
                    # Find insertion point: after identifier columns (Protein, original_name, string_id)
                    id_cols = intersect(["Protein", "original_name", "string_id"], names(final_results))
                    insert_pos = length(id_cols) + 1
                    insertcols!(final_results, insert_pos, :posterior_calibrated => post_cal)
                    insertcols!(final_results, insert_pos + 1, :fdr_calibrated => fdr_cal)
                end

                config.verbose && @info "Calibration applied: ECE=$(round(cal_ece, digits=4))"
            end
        catch e
            @warn "Failed to apply isotonic calibration to results" exception=e
            should_calibrate = false  # ensure flag reflects actual outcome on error
        end
    end

    # stash dataset-level ECE-gate provenance on the AR struct
    # (imputed-vector path mirror of single-data write).
    analysis_result.is_calibrated = should_calibrate

    # PEP/BFDR rewrite using coalesced posterior (imputed-vector path).
    # T-70-11-01 mitigation: assert posterior_calibrated missingness mirrors posterior_prob.
    if hasproperty(final_results, :posterior_calibrated)
        for i in 1:nrow(final_results)
            if !ismissing(final_results.posterior_prob[i]) &&
               ismissing(final_results.posterior_calibrated[i])
                @warn "T-70-11-01: posterior_calibrated is missing at row $i while posterior_prob is non-missing; calibration may have regressed"
                break
            end
        end
        post_for_pep = coalesce.(final_results.posterior_calibrated, final_results.posterior_prob)
    else
        post_for_pep = final_results.posterior_prob
    end
    final_results.pep = pep(post_for_pep)                    # canonical lowercase
    final_results.PEP = final_results.pep                    # silent mirror (same Vector reference)
    final_results.BFDR = bfdr(post_for_pep, isBF = false)

    # Re-sort by the (possibly calibrated) BFDR + log2FC
    sort!(final_results, [:BFDR, :mean_log2FC], rev = [false, false])

    analysis_result.copula_results = final_results

    # Save final results (after calibration so calibrated columns are included).
    # imputed-data path; `bb_mnar_codriven` flows
    # through the parallel imputed-pipeline copula_df builder (see L1455-1472)
    # and survives to this writer. Same xlsx-position guarantee as the single-
    # data path above. Regression: test/analysis/test_bb_mnar_codriven_xlsx.jl.
    writetable(config.output.results_file, "df" => final_results; overwrite = true)

    # Generate interactive HTML report
    if config.generate_report_html
        try
            sr = hasproperty(analysis_result, :sensitivity) ? analysis_result.sensitivity : nothing
            generate_report(final_results, config; analysis_result=analysis_result,
                            validation_result=validation_result,
                            sensitivity_result=sr,
                            simulation_result=simulation_result)
        catch e
            @warn "Failed to generate interactive HTML report" exception=(e, catch_backtrace())
        end
    end

    # (Pitfalls 2 + 4): expose per-bait simulation result + CONFIG on the
    # returned AnalysisResult so the differential report's per-condition tabs can iterate
    # diff.analyses[i].simulation_result and diff.analyses[i].config. AnalysisResult is
    # `mutable struct` (precedent — metalearner_status mutated above), so
    # post-construction field assignment is legal.
    analysis_result.simulation_result = simulation_result  # may be nothing when run_simulation=false
    analysis_result.config = config

    return final_results, analysis_result
end
