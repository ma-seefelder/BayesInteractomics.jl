# Differential Interaction Analysis — Core Logic
# Compare interaction profiles between two experimental conditions

import DataFrames: DataFrame, innerjoin, nrow, rename!
using Dates

# ----------------------- Main Function ----------------------- #

"""
    differential_analysis(result_A, result_B; condition_A, condition_B, config)

Compare interaction profiles between two experimental conditions.

Takes two `AnalysisResult` objects (one per condition) and computes differential
Bayes factors, posterior probabilities, and interaction classifications for each
protein present in either condition.

# Arguments
- `result_A::AbstractAnalysisResult`: Results from condition A (e.g., wild-type)
- `result_B::AbstractAnalysisResult`: Results from condition B (e.g., mutant)

# Keywords
- `condition_A::String = "Condition_A"`: Human-readable label for condition A
- `condition_B::String = "Condition_B"`: Human-readable label for condition B
- `config::DifferentialConfig = DifferentialConfig()`: Analysis configuration

# Returns
- `DifferentialResult`: Complete differential analysis results

# Statistical Methodology

For each protein present in both conditions (inner join on Protein ID):

1. **Differential Bayes Factor (dBF)**: `BF_A / BF_B` (log-space subtraction).
   Positive `log10(dBF)` means stronger evidence for interaction in condition A.

2. **Per-evidence differential**: Same ratio for enrichment, correlation, and
   detection BFs separately — diagnoses which evidence drives the signal.

3. **Effect size**: `delta_log2FC = mean_log2FC_A - mean_log2FC_B`.

4. **Differential posterior**: `P(diff | data) = |dBF| / (1 + |dBF|)`.
   Direction-agnostic measure of evidence for any difference.

5. **Multiple testing**: Bayesian FDR q-values on differential posteriors.

6. **Classification**: `GAINED`, `REDUCED`, `UNCHANGED`, or `BOTH_NEGATIVE` based on config method.

Proteins found in only one condition are appended as `CONDITION_A_SPECIFIC` or
`CONDITION_B_SPECIFIC` with `NaN` fill for the missing condition.

# Examples
```julia
result_wt = run_analysis(config_wt)[2]
result_mut = run_analysis(config_mut)[2]

diff = differential_analysis(result_wt, result_mut,
    condition_A = "WT",
    condition_B = "Mutant",
    config = DifferentialConfig(q_threshold = 0.01))

println(diff)
gained = gained_interactions(diff)
```

See also: [`DifferentialResult`](@ref), [`DifferentialConfig`](@ref)
"""
function differential_analysis(
    result_A::AbstractAnalysisResult,
    result_B::AbstractAnalysisResult;
    condition_A::String = "Condition_A",
    condition_B::String = "Condition_B",
    config::DifferentialConfig = DifferentialConfig()
)
    df_A = _extract_copula_df(result_A)
    df_B = _extract_copula_df(result_B)

    # Deduplicate: keep first occurrence of each protein (avoids cartesian product in join)
    df_A = _deduplicate_proteins(df_A)
    df_B = _deduplicate_proteins(df_B)

    n_proteins_A = nrow(df_A)
    n_proteins_B = nrow(df_B)

    # Inner join on Protein ID
    df_A_renamed = _rename_columns(df_A, :A)
    df_B_renamed = _rename_columns(df_B, :B)
    df_shared = innerjoin(df_A_renamed, df_B_renamed, on = :Protein)
    n_shared = nrow(df_shared)

    # Identify condition-specific proteins
    proteins_A = Set(df_A.Protein)
    proteins_B = Set(df_B.Protein)
    only_A = setdiff(proteins_A, proteins_B)
    only_B = setdiff(proteins_B, proteins_A)

    # Compute differential statistics for shared proteins
    if n_shared > 0
        results_df = _compute_differential_statistics(df_shared, config)
    else
        results_df = _empty_results_dataframe()
    end

    # Append condition-specific proteins
    _append_condition_specific!(results_df, df_A, df_B, only_A, only_B)

    # Compute summary counts
    n_gained       = count(==(GAINED),        results_df.classification)
    n_reduced      = count(==(REDUCED),       results_df.classification)
    n_unchanged    = count(==(UNCHANGED),     results_df.classification)
    n_both_negative = count(==(BOTH_NEGATIVE), results_df.classification)

    return DifferentialResult(
        results_df,
        condition_A, condition_B,
        config,
        n_proteins_A, n_proteins_B, n_shared,
        length(only_A), length(only_B),
        now(),
        n_gained, n_reduced, n_unchanged, n_both_negative
    )
end

# ----------------------- Pipeline Method ----------------------- #

"""
    differential_analysis(config_A, config_B; condition_A, condition_B, config, scatter_metric)

End-to-end differential interaction analysis pipeline.

Runs `run_analysis` on both `CONFIG` objects, compares the resulting interaction
profiles, generates diagnostic plots (volcano, evidence, scatter), and exports
results to an Excel file.

# Arguments
- `config_A::CONFIG`: Analysis configuration for condition A
- `config_B::CONFIG`: Analysis configuration for condition B

# Keywords
- `condition_A::String = "Condition_A"`: Human-readable label for condition A
- `condition_B::String = "Condition_B"`: Human-readable label for condition B
- `config::DifferentialConfig = DifferentialConfig()`: Differential analysis parameters
  and output paths (see [`DifferentialConfig`](@ref))
- `scatter_metric::Symbol = :posterior_prob`: Metric for the scatter plot
  (`:posterior_prob`, `:bf`, or `:log2fc`)

# Returns
- `DifferentialResult`: Complete differential analysis results

# Side Effects
- Saves volcano plot to `config.volcano_file`
- Saves evidence plot to `config.evidence_file`
- Saves scatter plot to `config.scatter_file`
- Saves results Excel to `config.results_file`

# Examples
```julia
config_wt = CONFIG(datafile=["wt.xlsx"], ...)
config_mut = CONFIG(datafile=["mut.xlsx"], ...)

diff = differential_analysis(config_wt, config_mut,
    condition_A = "WT",
    condition_B = "Mutant",
    config = DifferentialConfig(
        q_threshold = 0.01,
        volcano_file = "results/volcano.svg",
        results_file = "results/differential.xlsx"
    ))
```

See also: [`DifferentialConfig`](@ref), [`DifferentialResult`](@ref), [`run_analysis`](@ref)
"""
function differential_analysis(
    config_A::CONFIG,
    config_B::CONFIG;
    condition_A::String = "Condition_A",
    condition_B::String = "Condition_B",
    config::DifferentialConfig = DifferentialConfig(),
    scatter_metric::Symbol = :posterior_prob
)
    # Run both analyses
    _, result_A = run_analysis(config_A)
    _, result_B = run_analysis(config_B)

    # Differential analysis
    diff = differential_analysis(
        result_A, result_B,
        condition_A = condition_A,
        condition_B = condition_B,
        config = config
    )

    # Plotting
    plt = differential_volcano_plot(diff)
    StatsPlots.savefig(plt, config.volcano_file)

    plt = differential_evidence_plot(diff)
    StatsPlots.savefig(plt, config.evidence_file)

    plt = differential_scatter_plot(diff, metric = scatter_metric)
    StatsPlots.savefig(plt, config.scatter_file)

    plt = differential_classification_plot(diff)
    StatsPlots.savefig(plt, config.classification_file)

    plt = differential_ma_plot(diff)
    StatsPlots.savefig(plt, config.ma_file)

    # Export results
    export_differential(diff, config.results_file)

    # Generate interactive HTML report
    if config.generate_report_html
        generate_differential_report(diff)
    end

    return diff
end

# ----------------------- Pipeline Method (Multiple Imputation) ----------------------- #

"""
    differential_analysis(config_A, config_B, imputed_data_A, raw_data_A, imputed_data_B, raw_data_B; ...)

End-to-end differential interaction analysis pipeline with multiple imputation support.

Runs `run_analysis` with imputed datasets on both conditions, compares the resulting
interaction profiles, generates diagnostic plots (volcano, evidence, scatter), and
exports results to an Excel file.

# Arguments
- `config_A::CONFIG`: Analysis configuration for condition A
- `config_B::CONFIG`: Analysis configuration for condition B
- `imputed_data_A::Vector{InteractionData}`: Imputed datasets for condition A
- `raw_data_A::InteractionData`: Raw (non-imputed) data for condition A (used for Beta-Bernoulli)
- `imputed_data_B::Vector{InteractionData}`: Imputed datasets for condition B
- `raw_data_B::InteractionData`: Raw (non-imputed) data for condition B (used for Beta-Bernoulli)

# Keywords
- `condition_A::String = "Condition_A"`: Human-readable label for condition A
- `condition_B::String = "Condition_B"`: Human-readable label for condition B
- `config::DifferentialConfig = DifferentialConfig()`: Differential analysis parameters
  and output paths (see [`DifferentialConfig`](@ref))
- `scatter_metric::Symbol = :posterior_prob`: Metric for the scatter plot
  (`:posterior_prob`, `:bf`, or `:log2fc`)

# Returns
- `DifferentialResult`: Complete differential analysis results

# Side Effects
- Saves individual analysis outputs as configured in `config_A` and `config_B`
- Saves volcano plot to `config.volcano_file`
- Saves evidence plot to `config.evidence_file`
- Saves scatter plot to `config.scatter_file`
- Saves results Excel to `config.results_file`

# Examples
```julia
config_wt  = CONFIG(datafile=["wt.xlsx"], ...)
config_mut = CONFIG(datafile=["mut.xlsx"], ...)

diff = differential_analysis(config_wt, config_mut,
    wt_imputed, wt_raw, mut_imputed, mut_raw,
    condition_A = "WT",
    condition_B = "Mutant",
    config = DifferentialConfig(
        q_threshold = 0.01,
        volcano_file = "results/volcano.svg",
        results_file = "results/differential.xlsx"
    ))
```

See also: [`DifferentialConfig`](@ref), [`DifferentialResult`](@ref), [`run_analysis`](@ref)
"""
function differential_analysis(
    config_A::CONFIG,
    config_B::CONFIG,
    imputed_data_A::Vector{InteractionData},
    raw_data_A::InteractionData,
    imputed_data_B::Vector{InteractionData},
    raw_data_B::InteractionData;
    condition_A::String = "Condition_A",
    condition_B::String = "Condition_B",
    config::DifferentialConfig = DifferentialConfig(),
    scatter_metric::Symbol = :posterior_prob
)
    # Run both analyses with multiple imputation
    _, result_A = run_analysis(config_A, imputed_data_A, raw_data_A)
    _, result_B = run_analysis(config_B, imputed_data_B, raw_data_B)

    # Differential analysis
    diff = differential_analysis(
        result_A, result_B,
        condition_A = condition_A,
        condition_B = condition_B,
        config = config
    )

    # Plotting
    plt = differential_volcano_plot(diff)
    StatsPlots.savefig(plt, config.volcano_file)

    plt = differential_evidence_plot(diff)
    StatsPlots.savefig(plt, config.evidence_file)

    plt = differential_scatter_plot(diff, metric = scatter_metric)
    StatsPlots.savefig(plt, config.scatter_file)

    plt = differential_classification_plot(diff)
    StatsPlots.savefig(plt, config.classification_file)

    plt = differential_ma_plot(diff)
    StatsPlots.savefig(plt, config.ma_file)

    # Export results
    export_differential(diff, config.results_file)

    # Generate interactive HTML report
    if config.generate_report_html
        generate_differential_report(diff)
    end

    return diff
end

# ----------------------- Internal Helpers ----------------------- #

"""Extract copula results DataFrame from an AnalysisResult."""
function _extract_copula_df(result::AbstractAnalysisResult)
    return result.copula_results
end

"""Deduplicate proteins: keep first occurrence of each protein name."""
function _deduplicate_proteins(df::DataFrame)
    seen = Set{String}()
    keep = trues(nrow(df))
    for i in 1:nrow(df)
        p = df.Protein[i]
        if p in seen
            keep[i] = false
        else
            push!(seen, p)
        end
    end
    return df[keep, :]
end

"""Rename all non-Protein columns by appending _A or _B suffix."""
function _rename_columns(df::DataFrame, suffix::Symbol)
    new_df = copy(df)
    suffix_str = String(suffix)
    for col in names(new_df)
        col == "Protein" && continue
        rename!(new_df, col => "$(col)_$(suffix_str)")
    end
    return new_df
end

"""Compute log10 with protection against zero, negative, and missing values.
Clamped to [-8, 8] (representing BF range 1e-8 to 1e8) to prevent extreme
floor/ceiling values from distorting differential Bayes factors."""
const _LOG10_MAX = 8.0
_safe_log10(x::Real) = x > 0 ? clamp(log10(Float64(x)), -_LOG10_MAX, _LOG10_MAX) : -_LOG10_MAX
_safe_log10(::Missing) = -_LOG10_MAX

"""Safely divide a by b, returning ratio clamped to [1e-8, 1e8]. Missing treated as floor/ceiling."""
_safe_ratio(a::Real, b::Real) = clamp(Float64(a) / max(Float64(b), eps(Float64)), 1e-8, 1e8)
_safe_ratio(::Missing, b::Real) = 1e-8
_safe_ratio(a::Real, ::Missing) = min(Float64(a) / eps(Float64), 1e8)
_safe_ratio(::Missing, ::Missing) = 1.0

"""Convert to Float64, treating Missing as NaN."""
_to_float(x::Real) = Float64(x)
_to_float(::Missing) = NaN

"""Create empty results DataFrame with correct schema."""
function _empty_results_dataframe()
    return DataFrame(
        Protein              = String[],
        bf_A                 = Float64[],
        bf_B                 = Float64[],
        dbf                  = Float64[],
        log10_dbf            = Float64[],
        posterior_A          = Float64[],
        posterior_B          = Float64[],
        delta_posterior      = Float64[],
        q_A                 = Union{Missing,Float64}[],
        q_B                 = Union{Missing,Float64}[],
        log2fc_A             = Float64[],
        log2fc_B             = Float64[],
        delta_log2fc         = Float64[],
        bf_enrichment_A      = Float64[],
        bf_enrichment_B      = Float64[],
        dbf_enrichment       = Float64[],
        bf_correlation_A     = Float64[],
        bf_correlation_B     = Float64[],
        dbf_correlation      = Float64[],
        bf_detected_A        = Float64[],
        bf_detected_B        = Float64[],
        dbf_detected         = Float64[],
        differential_posterior = Float64[],
        differential_q       = Union{Missing,Float64}[],
        classification       = InteractionClass[],
        diagnostic_flag_A    = String[],
        diagnostic_flag_B    = String[],
        sensitivity_range_A  = Float64[],
        sensitivity_range_B  = Float64[],
    )
end

"""
Core computation: differential BFs, posteriors, q-values, classifications
for proteins shared between both conditions.
"""
function _compute_differential_statistics(
    df_shared::DataFrame,
    config::DifferentialConfig
)
    n = nrow(df_shared)

    # Combined differential BF
    log10_bf_A = _safe_log10.(df_shared.BF_A)
    log10_bf_B = _safe_log10.(df_shared.BF_B)
    log10_dbf = clamp.(log10_bf_A .- log10_bf_B, -_LOG10_MAX, _LOG10_MAX)
    dbf = 10.0 .^ log10_dbf

    # Per-evidence differential BFs
    dbf_enrichment  = _safe_ratio.(df_shared.bf_enrichment_A, df_shared.bf_enrichment_B)
    dbf_correlation = _safe_ratio.(df_shared.bf_correlation_A, df_shared.bf_correlation_B)
    dbf_detected    = _safe_ratio.(df_shared.bf_detected_A, df_shared.bf_detected_B)

    # Effect size
    delta_log2fc = _to_float.(df_shared.mean_log2FC_A) .- _to_float.(df_shared.mean_log2FC_B)

    # Delta posterior (directional)
    delta_posterior = _to_float.(df_shared.posterior_prob_A) .- _to_float.(df_shared.posterior_prob_B)

    # Differential posterior: P(diff|data) from |dBF|
    abs_dbf = abs.(dbf)
    differential_posterior = abs_dbf ./ (1.0 .+ abs_dbf)

    # Multiple testing correction via Bayesian FDR
    differential_q = q(differential_posterior, isBF = false)

    # Classification
    classification = _classify_interactions(
        df_shared, log10_dbf, delta_log2fc, differential_q, config
    )

    # Optional columns: diagnostic_flag and sensitivity_range (present only when diagnostics ran)
    diag_A = hasproperty(df_shared, :diagnostic_flag_A) ?
        coalesce.(string.(df_shared.diagnostic_flag_A), "") : fill("", n)
    diag_B = hasproperty(df_shared, :diagnostic_flag_B) ?
        coalesce.(string.(df_shared.diagnostic_flag_B), "") : fill("", n)
    sens_A = hasproperty(df_shared, :sensitivity_range_A) ?
        _to_float.(df_shared.sensitivity_range_A) : fill(NaN, n)
    sens_B = hasproperty(df_shared, :sensitivity_range_B) ?
        _to_float.(df_shared.sensitivity_range_B) : fill(NaN, n)

    return DataFrame(
        Protein              = df_shared.Protein,
        bf_A                 = _to_float.(df_shared.BF_A),
        bf_B                 = _to_float.(df_shared.BF_B),
        dbf                  = dbf,
        log10_dbf            = log10_dbf,
        posterior_A          = _to_float.(df_shared.posterior_prob_A),
        posterior_B          = _to_float.(df_shared.posterior_prob_B),
        delta_posterior      = delta_posterior,
        q_A                 = df_shared.q_A,
        q_B                 = df_shared.q_B,
        log2fc_A             = _to_float.(df_shared.mean_log2FC_A),
        log2fc_B             = _to_float.(df_shared.mean_log2FC_B),
        delta_log2fc         = delta_log2fc,
        bf_enrichment_A      = _to_float.(df_shared.bf_enrichment_A),
        bf_enrichment_B      = _to_float.(df_shared.bf_enrichment_B),
        dbf_enrichment       = dbf_enrichment,
        bf_correlation_A     = _to_float.(df_shared.bf_correlation_A),
        bf_correlation_B     = _to_float.(df_shared.bf_correlation_B),
        dbf_correlation      = dbf_correlation,
        bf_detected_A        = _to_float.(df_shared.bf_detected_A),
        bf_detected_B        = _to_float.(df_shared.bf_detected_B),
        dbf_detected         = dbf_detected,
        differential_posterior = differential_posterior,
        differential_q       = differential_q,
        classification       = classification,
        diagnostic_flag_A    = diag_A,
        diagnostic_flag_B    = diag_B,
        sensitivity_range_A  = sens_A,
        sensitivity_range_B  = sens_B,
    )
end

"""
Classify each protein as GAINED, REDUCED, UNCHANGED, or BOTH_NEGATIVE.

## Methods
- `:posterior`: Interactor in A but not B AND Δlog₂FC ≥ 0 → GAINED; reverse → REDUCED.
  When both are interactors, uses delta_log2fc threshold. When neither is an interactor
  but differential q is significant → BOTH_NEGATIVE.
- `:dbf`: |log10(dBF)| exceeds threshold → GAINED or REDUCED by sign.
- `:combined`: Both posterior and dBF criteria must hold.
"""
function _classify_interactions(
    df_shared::DataFrame,
    log10_dbf::Vector{Float64},
    delta_log2fc::Vector{Float64},
    differential_q,
    config::DifferentialConfig
)
    n = nrow(df_shared)
    classification = fill(UNCHANGED, n)

    for i in 1:n
        q_val = ismissing(differential_q[i]) ? 1.0 : Float64(differential_q[i])
        is_significant = q_val < config.q_threshold

        if config.classification_method == :posterior
            pp_A = ismissing(df_shared.posterior_prob_A[i]) ? 0.0 : Float64(df_shared.posterior_prob_A[i])
            pp_B = ismissing(df_shared.posterior_prob_B[i]) ? 0.0 : Float64(df_shared.posterior_prob_B[i])
            is_interactor_A = pp_A > config.posterior_threshold
            is_interactor_B = pp_B > config.posterior_threshold

            if is_significant
                if is_interactor_A && !is_interactor_B
                    # Require positive Δlog₂FC for GAINED — otherwise evidence is contradictory
                    if delta_log2fc[i] >= 0
                        classification[i] = GAINED
                    end
                elseif !is_interactor_A && is_interactor_B
                    # Require negative Δlog₂FC for REDUCED — otherwise evidence is contradictory
                    if delta_log2fc[i] <= 0
                        classification[i] = REDUCED
                    end
                elseif is_interactor_A && is_interactor_B
                    if delta_log2fc[i] > config.delta_log2fc_threshold
                        classification[i] = GAINED
                    elseif delta_log2fc[i] < -config.delta_log2fc_threshold
                        classification[i] = REDUCED
                    end
                else
                    # Neither condition shows interaction above threshold
                    classification[i] = BOTH_NEGATIVE
                end
            end

        elseif config.classification_method == :dbf
            if is_significant
                if log10_dbf[i] > config.dbf_threshold
                    classification[i] = GAINED
                elseif log10_dbf[i] < -config.dbf_threshold
                    classification[i] = REDUCED
                end
            end

        elseif config.classification_method == :combined
            pp_A = ismissing(df_shared.posterior_prob_A[i]) ? 0.0 : Float64(df_shared.posterior_prob_A[i])
            pp_B = ismissing(df_shared.posterior_prob_B[i]) ? 0.0 : Float64(df_shared.posterior_prob_B[i])
            is_interactor_A = pp_A > config.posterior_threshold
            is_interactor_B = pp_B > config.posterior_threshold

            if is_significant && abs(log10_dbf[i]) > config.dbf_threshold
                if is_interactor_A && !is_interactor_B && log10_dbf[i] > 0
                    classification[i] = GAINED
                elseif !is_interactor_A && is_interactor_B && log10_dbf[i] < 0
                    classification[i] = REDUCED
                elseif is_interactor_A && is_interactor_B
                    if log10_dbf[i] > config.dbf_threshold && delta_log2fc[i] > config.delta_log2fc_threshold
                        classification[i] = GAINED
                    elseif log10_dbf[i] < -config.dbf_threshold && delta_log2fc[i] < -config.delta_log2fc_threshold
                        classification[i] = REDUCED
                    end
                end
            end
        end
    end

    return classification
end

"""Append condition-specific proteins to results with NaN fill."""
function _append_condition_specific!(
    results_df::DataFrame,
    df_A::DataFrame,
    df_B::DataFrame,
    only_A::Set{<:AbstractString},
    only_B::Set{<:AbstractString}
)
    for protein in only_A
        idx = findfirst(==(protein), df_A.Protein)
        row_A = df_A[idx, :]
        push!(results_df, _make_condition_specific_row(protein, row_A, :A); promote = true)
    end

    for protein in only_B
        idx = findfirst(==(protein), df_B.Protein)
        row_B = df_B[idx, :]
        push!(results_df, _make_condition_specific_row(protein, row_B, :B); promote = true)
    end

    return results_df
end

"""Build a result row for a condition-specific protein."""
function _make_condition_specific_row(protein::AbstractString, row, condition::Symbol)
    nan = NaN
    diag  = hasproperty(row, :diagnostic_flag) ?
                coalesce(string(row.diagnostic_flag), "") : ""
    sens  = hasproperty(row, :sensitivity_range) ? _to_float(row.sensitivity_range) : nan
    if condition == :A
        return (
            Protein = protein,
            bf_A = _to_float(row.BF), bf_B = nan,
            dbf = nan, log10_dbf = nan,
            posterior_A = _to_float(row.posterior_prob), posterior_B = nan,
            delta_posterior = nan,
            q_A = row.q, q_B = missing,
            log2fc_A = _to_float(row.mean_log2FC), log2fc_B = nan,
            delta_log2fc = nan,
            bf_enrichment_A = _to_float(row.bf_enrichment), bf_enrichment_B = nan,
            dbf_enrichment = nan,
            bf_correlation_A = _to_float(row.bf_correlation), bf_correlation_B = nan,
            dbf_correlation = nan,
            bf_detected_A = _to_float(row.bf_detected), bf_detected_B = nan,
            dbf_detected = nan,
            differential_posterior = nan,
            differential_q = missing,
            classification = CONDITION_A_SPECIFIC,
            diagnostic_flag_A = diag, diagnostic_flag_B = "",
            sensitivity_range_A = sens, sensitivity_range_B = nan,
        )
    else
        return (
            Protein = protein,
            bf_A = nan, bf_B = _to_float(row.BF),
            dbf = nan, log10_dbf = nan,
            posterior_A = nan, posterior_B = _to_float(row.posterior_prob),
            delta_posterior = nan,
            q_A = missing, q_B = row.q,
            log2fc_A = nan, log2fc_B = _to_float(row.mean_log2FC),
            delta_log2fc = nan,
            bf_enrichment_A = nan, bf_enrichment_B = _to_float(row.bf_enrichment),
            dbf_enrichment = nan,
            bf_correlation_A = nan, bf_correlation_B = _to_float(row.bf_correlation),
            dbf_correlation = nan,
            bf_detected_A = nan, bf_detected_B = _to_float(row.bf_detected),
            dbf_detected = nan,
            differential_posterior = nan,
            differential_q = missing,
            classification = CONDITION_B_SPECIFIC,
            diagnostic_flag_A = "", diagnostic_flag_B = diag,
            sensitivity_range_A = nan, sensitivity_range_B = sens,
        )
    end
end

# ----------------------- Convenience Functions ----------------------- #

"""
    gained_interactions(diff::DifferentialResult) -> DataFrame

Return only gained interactions (stronger in condition A).
"""
function gained_interactions(diff::DifferentialResult)
    return diff.results[diff.results.classification .== GAINED, :]
end

"""
    lost_interactions(diff::DifferentialResult) -> DataFrame

Return only reduced interactions (stronger in condition B).
"""
function lost_interactions(diff::DifferentialResult)
    return diff.results[diff.results.classification .== REDUCED, :]
end

"""
    unchanged_interactions(diff::DifferentialResult) -> DataFrame

Return only unchanged interactions.
"""
function unchanged_interactions(diff::DifferentialResult)
    return diff.results[diff.results.classification .== UNCHANGED, :]
end

"""
    significant_differential(diff::DifferentialResult; q_threshold=0.05) -> DataFrame

Return all proteins with significant differential interaction evidence.
"""
function significant_differential(diff::DifferentialResult; q_threshold::Float64 = 0.05)
    valid_idx = findall(x -> !ismissing(x) && x < q_threshold, diff.results.differential_q)
    return diff.results[valid_idx, :]
end

"""
    export_differential(diff::DifferentialResult, filepath::String)

Export differential analysis results to an Excel file.

Creates two sheets:
- `"differential"`: Full results DataFrame (classification as strings)
- `"summary"`: Summary statistics and configuration parameters
"""
function export_differential(diff::DifferentialResult, filepath::String)
    summary_df = DataFrame(
        Metric = [
            "Condition A", "Condition B",
            "Proteins in A", "Proteins in B",
            "Shared proteins",
            "A-specific", "B-specific",
            "Gained", "Reduced", "Unchanged", "Both negative",
            "Posterior threshold", "Q threshold",
            "Delta log2FC threshold", "dBF threshold",
            "Classification method"
        ],
        Value = [
            diff.condition_A, diff.condition_B,
            string(diff.n_proteins_A), string(diff.n_proteins_B),
            string(diff.n_shared),
            string(diff.n_condition_A_specific), string(diff.n_condition_B_specific),
            string(diff.n_gained), string(diff.n_reduced), string(diff.n_unchanged), string(diff.n_both_negative),
            string(diff.config.posterior_threshold), string(diff.config.q_threshold),
            string(diff.config.delta_log2fc_threshold), string(diff.config.dbf_threshold),
            string(diff.config.classification_method)
        ]
    )

    # Convert classification enum to strings for Excel compatibility
    export_df = copy(diff.results)
    export_df.classification = string.(export_df.classification)

    writetable(filepath,
        "differential" => export_df,
        "summary" => summary_df
    )
end
