# Differential Interaction Analysis â€” Core Logic
# Compare interaction profiles between two experimental conditions

import DataFrames: DataFrame, innerjoin, Not, nrow, outerjoin, rename!, select!
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
   detection BFs separately â€” diagnoses which evidence drives the signal.

3. **Effect size**: `delta_log2FC = mean_log2FC_A - mean_log2FC_B`.

4. **Differential posterior**: `P(diff | data) = |dBF| / (1 + |dBF|)`.
   Direction-agnostic measure of evidence for any difference.

5. **Multiple testing**: BFDR (Bayesian FDR) on differential posteriors.

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
    config = DifferentialConfig(bfdr_threshold = 0.01))

println(diff)
gained = gained_interactions(diff)
```

See also: [`DifferentialResult`](@ref), [`DifferentialConfig`](@ref)

introduces a keyword-only k-group overload
`differential_analysis(; conditions::NamedTuple, ...)` â€” see its dedicated
docstring further down the file.
"""
function differential_analysis(
    result_A::AbstractAnalysisResult,
    result_B::AbstractAnalysisResult;
    condition_A::String = "Condition_A",
    condition_B::String = "Condition_B",
    config::DifferentialConfig = DifferentialConfig(),
    loss_matrix::Matrix{Float64} = hasproperty(config, :loss_matrix) ?
                                   config.loss_matrix : DEFAULT_DIFFERENTIAL_LOSS,
)
    df_A = _extract_copula_df(result_A)
    df_B = _extract_copula_df(result_B)

    # Filter to detected proteins only (is_detected == true or column absent)
    df_A = _filter_detected_diff(df_A)
    df_B = _filter_detected_diff(df_B)

    # Deduplicate: keep first occurrence of each protein (avoids cartesian product in join)
    df_A = _deduplicate_proteins(df_A)
    df_B = _deduplicate_proteins(df_B)

    # optional regression-safe per-condition bait-anchor (default OFF).
    # When config.bait_anchor == true the per-condition enrichment of EVERY shared interactor
    # is shifted by the bait-level gap between the two conditions, equalising a documented
    # bait-abundance difference (the same effect the sample-cells-only raw-bait δ has on the
    # sample−control contrast — see bait_anchor_id). Applied as a per-condition
    # CONSTANT shift on `mean_log2FC`, so within-condition prey structure is fully preserved
    # (regression-safe; the predictor is never zeroed/de-varied). δ≈0 on matched-level baits
    # ⇒ near-inert. When false this branch is NOT entered ⇒ byte-identical to today.
    if config.bait_anchor
        _apply_bait_anchor_diff!(df_A, df_B, result_A, result_B)
    end

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

    # Add detected_in column: "both" for proteins present in both (shared proteins)
    results_df.detected_in = fill("both", nrow(results_df))

    # Append condition-specific proteins (with detected_in = "condition_a_only"/"condition_b_only")
    _append_condition_specific!(results_df, df_A, df_B, only_A, only_B, config)

    # silent uppercase mirror (same Vector reference) â€” set AFTER push!es
    # so the two columns share the final backing array. To be dropped in v1.3.
    results_df.diff_PEP = results_df.differential_pep

    # (k=2 legacy parity): compute omnibus + classification columns
    # using a synthetic 2-AR vector. Schema parity with k-group NamedTuple path
    # so DifferentialResult.results::DataFrame carries the 8 new columns regardless
    # of entry point. Tests asserting byte-equality on the earlier
    # schema MUST drop the 8 new columns before isequal comparison
    # (pattern: select(df, Not([:bf_omnibus, :log10_bf_omnibus, :posterior_omnibus,
    # :differential_BFDR_omnibus, :differential_pep_omnibus, :enriched_in,
    # :depleted_in, :kgroup_class]))).
    ars_legacy = AbstractAnalysisResult[result_A, result_B]
    cond_label_strings_legacy = String[String(condition_A), String(condition_B)]
    eb_prior_legacy = _eb_pooled_prior(ars_legacy)
    _compute_omnibus_columns!(results_df, ars_legacy, cond_label_strings_legacy, eb_prior_legacy)
    _compute_kgroup_classification_columns!(
        results_df, ars_legacy, cond_label_strings_legacy,
        config.posterior_threshold, config.bfdr_threshold,
    )

    # Decision Risk: 6 new columns appended to results_df.
    # Runs AFTER classification (we need the InteractionClass enum for
    # CONDITION_A/B_SPECIFIC NaN handling) and BEFORE summary counts.
    # Mutates results_df in-place.
    compute_decision_risk!(results_df; loss_matrix = loss_matrix)

    # Compute summary counts
    n_gained       = count(==(GAINED),        results_df.classification)
    n_reduced      = count(==(REDUCED),       results_df.classification)
    n_unchanged    = count(==(UNCHANGED),     results_df.classification)
    n_both_negative = count(==(BOTH_NEGATIVE), results_df.classification)

    # per-side calibration provenance (defensive against AbstractAnalysisResult)
    is_cal_A = hasproperty(result_A, :is_calibrated) ? result_A.is_calibrated : false
    is_cal_B = hasproperty(result_B, :is_calibrated) ? result_B.is_calibrated : false

    # compute condition-level similarity matrix + dendrogram.
    # Source embeddings config from one of the upstream ARs (they share the same CONFIG
    # instance in run_analysis); fall back to defaults if AR.config is nothing.
    emb_cfg = (hasproperty(result_A, :config) && result_A.config !== nothing &&
               hasproperty(result_A.config, :embeddings_config)) ?
              result_A.config.embeddings_config : EmbeddingsConfig()
    condition_similarity = try
        if emb_cfg.run_embeddings
            _compute_condition_similarity([result_A, result_B], emb_cfg)
        else
            nothing
        end
    catch e
        @warn "[Embeddings] condition similarity failed: $e" maxlog=1 exception=(e, catch_backtrace())
        nothing
    end

    return DifferentialResult(
        results_df,
        condition_A, condition_B,
        config,
        n_proteins_A, n_proteins_B, n_shared,
        length(only_A), length(only_B),
        now(),
        n_gained, n_reduced, n_unchanged, n_both_negative,
        AnalysisResult[result_A, result_B],  # populate per-condition analyses
        is_cal_A,
        is_cal_B,
        condition_similarity,
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
- `NamedTuple{(:diff, :result_A, :result_B)}`: A named tuple containing:
  - `diff::DifferentialResult`: Complete differential analysis results
  - `result_A::AnalysisResult`: Full analysis result for condition A
  - `result_B::AnalysisResult`: Full analysis result for condition B

# Side Effects
- Saves volcano plot to `config.volcano_file`
- Saves evidence plot to `config.evidence_file`
- Saves scatter plot to `config.scatter_file`
- Saves results Excel to `config.results_file`

# Examples
```julia
config_wt = CONFIG(datafile=["wt.xlsx"], ...)
config_mut = CONFIG(datafile=["mut.xlsx"], ...)

(; diff, result_A, result_B) = differential_analysis(config_wt, config_mut,
    condition_A = "WT",
    condition_B = "Mutant",
    config = DifferentialConfig(
        bfdr_threshold = 0.01,
        volcano_file = "results/volcano.svg",
        results_file = "results/differential.xlsx"
    ))

# Or single assignment:
result = differential_analysis(config_wt, config_mut, ...)
result.diff      # DifferentialResult
result.result_A  # AnalysisResult for condition A
result.result_B  # AnalysisResult for condition B
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

    return (diff=diff, result_A=result_A, result_B=result_B)
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
- `NamedTuple{(:diff, :result_A, :result_B)}`: A named tuple containing:
  - `diff::DifferentialResult`: Complete differential analysis results
  - `result_A::AnalysisResult`: Full analysis result for condition A
  - `result_B::AnalysisResult`: Full analysis result for condition B

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

(; diff, result_A, result_B) = differential_analysis(config_wt, config_mut,
    wt_imputed, wt_raw, mut_imputed, mut_raw,
    condition_A = "WT",
    condition_B = "Mutant",
    config = DifferentialConfig(
        bfdr_threshold = 0.01,
        volcano_file = "results/volcano.svg",
        results_file = "results/differential.xlsx"
    ))

# Or single assignment:
result = differential_analysis(config_wt, config_mut, wt_imputed, wt_raw, mut_imputed, mut_raw, ...)
result.diff      # DifferentialResult
result.result_A  # AnalysisResult for condition A
result.result_B  # AnalysisResult for condition B
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

    return (diff=diff, result_A=result_A, result_B=result_B)
end

# ----------------------- Internal Helpers ----------------------- #

"""Filter DataFrame to detected proteins only (is_detected == true or column absent)."""
function _filter_detected_diff(df::DataFrame)::DataFrame
    if !hasproperty(df, :is_detected)
        return df  # backward compat: no is_detected column means all detected
    end
    return filter(r -> coalesce(r.is_detected, true), df)
end

"""Extract copula results DataFrame from an AnalysisResult.

Recomputes posterior_prob and q from the raw copula BF to strip metalearner bias.
The metalearner prior is identical across conditions for the same protein,
so it doesn't add information to the DIFFERENTIAL comparison."""
function _extract_copula_df(result::AbstractAnalysisResult)
    df = copy(result.copula_results)
    bf = df.BF
    df.posterior_prob = @. bf / (1.0 + bf)
    df.PEP = pep(df.posterior_prob)
    df.BFDR = bfdr(df.posterior_prob, isBF = false)
    return df
end

"""
    _apply_bait_anchor_diff!(df_A, df_B, result_A, result_B)

regression-safe per-condition bait-anchor on the differential path.

Equalises a documented bait-level difference between the two conditions by shifting EVERY
protein's `mean_log2FC` in each condition by a per-condition CONSTANT `δ_c`, where `δ_c` is the
bait's own `mean_log2FC` in condition `c` minus the grand mean of the two conditions' bait
`mean_log2FC`. This is the differential-path projection of the sample-cells-only raw-bait δ
(`bait_anchor_id`): subtracting `δ_c` from the bait's SAMPLE level shifts every prey's
enrichment (sample−control) by the bait-level gap.

REGRESSION-SAFE: a per-condition CONSTANT shift preserves all within-condition prey structure;
no value is zeroed or de-varied. Near-inert when the bait levels match (`δ_c ≈ 0`).

No-op (returns unchanged) when the bait cannot be located in BOTH conditions (missing
`bait_protein`, or the bait row absent / `mean_log2FC` column absent). Mutates `df_A`/`df_B`
in place. Only entered when `config.bait_anchor == true` (default OFF ⇒ byte-identical to today).
"""
function _apply_bait_anchor_diff!(df_A::DataFrame, df_B::DataFrame,
                                  result_A::AbstractAnalysisResult,
                                  result_B::AbstractAnalysisResult)
    # Need a known bait ID shared across both conditions and a mean_log2FC column on both.
    bait_A = hasproperty(result_A, :bait_protein) ? result_A.bait_protein : nothing
    bait_B = hasproperty(result_B, :bait_protein) ? result_B.bait_protein : nothing
    (bait_A === nothing || bait_B === nothing) && return (df_A, df_B)
    (hasproperty(df_A, :mean_log2FC) && hasproperty(df_B, :mean_log2FC)) || return (df_A, df_B)
    ("Protein" in names(df_A) && "Protein" in names(df_B)) || return (df_A, df_B)

    iA = findfirst(==(bait_A), df_A.Protein)
    iB = findfirst(==(bait_B), df_B.Protein)
    (iA === nothing || iB === nothing) && return (df_A, df_B)

    bait_lf_A = _to_float(df_A.mean_log2FC[iA])
    bait_lf_B = _to_float(df_B.mean_log2FC[iB])
    (isnan(bait_lf_A) || isnan(bait_lf_B)) && return (df_A, df_B)

    grand = (bait_lf_A + bait_lf_B) / 2.0
    δ_A = bait_lf_A - grand
    δ_B = bait_lf_B - grand
    # subtract the per-condition δ from EVERY prey's enrichment in that condition
    df_A.mean_log2FC = [ismissing(v) ? v : Float64(v) - δ_A for v in df_A.mean_log2FC]
    df_B.mean_log2FC = [ismissing(v) ? v : Float64(v) - δ_B for v in df_B.mean_log2FC]
    return (df_A, df_B)
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
Uses `log10(eps(Float64))` â‰ˆ -15.65 as floor for zero/negative values.
No hard clamping â€” the volcano plot uses `asinh` to compress extreme values."""
const _LOG10_FLOOR = log10(eps(Float64))   # â‰ˆ -15.65
_safe_log10(x::Real) = x > 0 ? log10(Float64(x)) : _LOG10_FLOOR
_safe_log10(::Missing) = _LOG10_FLOOR

"""Safely divide a by b, returning ratio clamped to [1e-8, 1e8]. Missing treated as floor/ceiling."""
_safe_ratio(a::Real, b::Real) = clamp(Float64(a) / max(Float64(b), eps(Float64)), 1e-8, 1e8)
_safe_ratio(::Missing, b::Real) = 1e-8
_safe_ratio(a::Real, ::Missing) = min(Float64(a) / eps(Float64), 1e8)
_safe_ratio(::Missing, ::Missing) = 1.0

"""Convert to Float64, treating Missing as NaN."""
_to_float(x::Real) = Float64(x)
_to_float(::Missing) = NaN

"""
    _compute_dbf_diagnostic(bf_A, bf_B, dbf_e, dbf_c, dbf_d, log10_dbf,
                            bf_em_A, bf_cop_A, bf_em_B, bf_cop_B) -> Symbol

classify each protein's dBF reliability into one of:
- `:saturated` â€” `|log10(bf)|` near the [-46, 46] log-BF clamp (>18 leaves a 2-decade buffer below 46/log(10)â‰ˆ20)
- `:single_component` â€” one of (enrichment, correlation, detection) drives >90% of |log10_dbf|
- `:model_disagreement` â€” copula vs 3c-EM dBFs disagree by >1 decade
- `:ok` â€” none of the above (and default when sub-model BFs missing)
"""
function _compute_dbf_diagnostic(bf_A, bf_B, dbf_e, dbf_c, dbf_d, log10_dbf,
                                 bf_em_A, bf_cop_A, bf_em_B, bf_cop_B)::Symbol
    SAT_THRESHOLD = 18.0
    SINGLE_FRAC = 0.90
    DISAGREE_DECADES = 1.0

    # Saturation: per-condition log10|bf| near clamp
    if abs(_safe_log10(bf_A)) > SAT_THRESHOLD || abs(_safe_log10(bf_B)) > SAT_THRESHOLD
        return :saturated
    end

    # Single-component dominance (only meaningful when |log10_dbf| > 0.5)
    abs_log10_dbf = abs(log10_dbf)
    if abs_log10_dbf > 0.5
        comps = (_safe_log10(dbf_e), _safe_log10(dbf_c), _safe_log10(dbf_d))
        max_comp = maximum(abs, comps)
        if max_comp / max(abs_log10_dbf, eps()) > SINGLE_FRAC
            return :single_component
        end
    end

    # Sub-model disagreement: requires all four sub-model BFs available
    if !ismissing(bf_em_A) && !ismissing(bf_cop_A) &&
       !ismissing(bf_em_B) && !ismissing(bf_cop_B)
        log10_dbf_em = _safe_log10(bf_em_A) - _safe_log10(bf_em_B)
        log10_dbf_cop = _safe_log10(bf_cop_A) - _safe_log10(bf_cop_B)
        if abs(log10_dbf_em - log10_dbf_cop) > DISAGREE_DECADES
            return :model_disagreement
        end
    end

    return :ok
end

"""Z-score standardize a vector, ignoring non-finite values."""
function _zscore(x::Vector{Float64})
    valid = filter(isfinite, x)
    isempty(valid) && return x
    mu = mean(valid)
    sigma = std(valid)
    sigma < eps(Float64) && return zeros(length(x))
    return [(isfinite(xi) ? (xi - mu) / sigma : NaN) for xi in x]
end

"""Create empty results DataFrame with correct schema."""
function _empty_results_dataframe()
    df = DataFrame(
        Protein              = String[],
        bf_A                 = Float64[],
        bf_B                 = Float64[],
        dbf                  = Float64[],
        log10_dbf            = Float64[],
        posterior_A          = Float64[],
        posterior_B          = Float64[],
        delta_posterior      = Float64[],
        BFDR_A              = Union{Missing,Float64}[],
        BFDR_B              = Union{Missing,Float64}[],
        PEP_A               = Union{Missing,Float64}[],
        PEP_B               = Union{Missing,Float64}[],
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
        differential_BFDR    = Union{Missing,Float64}[],
        differential_pep     = Union{Missing,Float64}[],
        pep_gained           = Union{Missing,Float64}[],
        pep_reduced          = Union{Missing,Float64}[],
        pep_unchanged        = Union{Missing,Float64}[],
        pep_both_negative    = Union{Missing,Float64}[],
        classification       = InteractionClass[],
        dbf_diagnostic       = Symbol[],
        diagnostic_flag_A    = String[],
        diagnostic_flag_B    = String[],
        bb_codriven_A        = Bool[],
        bb_codriven_B        = Bool[],
        sensitivity_range_A  = Float64[],
        sensitivity_range_B  = Float64[],
        detected_in          = String[],
        # diff_PEP silent uppercase mirror is established by the outer
        # `differential_analysis` after all push!es have appended condition-specific
        # rows, so the two columns can share their final Vector backing array.
    )
    return df
end

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Multi-test correction primitives (hand-rolled, no new dependency)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Benjamini-Hochberg (FDR), Bonferroni (FWER), Holm (FWER) â€” three corrections
# applied to the flattened (n_proteins Ã— n_contrasts) family of differential_BFDR
# values produced by the k-group pairwise overload.

"""
    _bh_adjust(pvals::AbstractVector) -> Vector{Union{Missing, Float64}}

Benjamini-Hochberg step-up on a vector of p-value-like quantities. Returns the
adjusted q-values in the *original* input order. Missing values pass through.

Algorithm: `q_(i) = min over j â‰¥ i of (p_(j) Ã— m / j)` with output clamped to
[0, 1]. The cumulative-min-from-right enforces monotone-non-decreasing q in
ascending p-value order (Benjamini & Hochberg 1995).
"""
function _bh_adjust(pvals::AbstractVector)
    n = length(pvals)
    out = Vector{Union{Missing, Float64}}(missing, n)
    valid = findall(!ismissing, pvals)
    isempty(valid) && return out
    p = Float64.(pvals[valid])
    m = length(p)
    order = sortperm(p)
    ranks = invperm(order)
    p_sorted = p[order]
    q_sorted = similar(p_sorted)
    cumulative_min = Inf
    for j in m:-1:1
        candidate = p_sorted[j] * m / j
        cumulative_min = min(cumulative_min, candidate)
        q_sorted[j] = min(cumulative_min, 1.0)
    end
    out[valid] = q_sorted[ranks]
    return out
end

"""
    _bonferroni_adjust(pvals::AbstractVector, n_total::Int) -> Vector{Union{Missing, Float64}}

Bonferroni FWER correction: `q_i = min(p_i Ã— n_total, 1)`. Missing values pass
through. `n_total` is passed explicitly (family size, not `length(pvals)`) so the
caller can supply the cross-pair flattened family size `n_proteins Ã— n_contrasts`.
"""
function _bonferroni_adjust(pvals::AbstractVector, n_total::Int)
    n = length(pvals)
    out = Vector{Union{Missing, Float64}}(missing, n)
    for i in 1:n
        ismissing(pvals[i]) && continue
        out[i] = clamp(Float64(pvals[i]) * n_total, 0.0, 1.0)
    end
    return out
end

"""
    _holm_adjust(pvals::AbstractVector) -> Vector{Union{Missing, Float64}}

Holm-Bonferroni FWER step-down: sort p-values ascending, then `q_(i) =
max over j â‰¤ i of (p_(j) Ã— (m - j + 1))` with output clamped to [0, 1] and
enforced monotone-non-decreasing in ascending p order. Missing pass-through.
Included because `:holm` is an accepted method; controls FWER not FDR.
"""
function _holm_adjust(pvals::AbstractVector)
    n = length(pvals)
    out = Vector{Union{Missing, Float64}}(missing, n)
    valid = findall(!ismissing, pvals)
    isempty(valid) && return out
    p = Float64.(pvals[valid])
    m = length(p)
    order = sortperm(p)
    ranks = invperm(order)
    p_sorted = p[order]
    q_sorted = similar(p_sorted)
    cumulative_max = -Inf
    for j in 1:m
        candidate = p_sorted[j] * (m - j + 1)
        cumulative_max = max(cumulative_max, candidate)
        q_sorted[j] = min(cumulative_max, 1.0)
    end
    out[valid] = q_sorted[ranks]
    return out
end

"""
    _apply_multi_test_correction!(pairwise_dict::Dict{Pair{Symbol, Symbol}, DataFrame},
                                  wide_df::DataFrame, method::Symbol) -> DataFrame

Apply cross-pair multi-test correction to the flattened `(n_proteins Ã— n_contrasts)`
family of `differential_BFDR` values produced by the k-group pairwise overload.
Writes the corrected q-values to BOTH:

1. **The wide `wide_df`** â€” one new column per contrast, suffixed
   `_pairwise_{Method}` (e.g. `differential_BFDR_wt_vs_mut1_pairwise_BH`).
2. **Each per-pair `pairwise_dict[pair]::DataFrame`** â€” a single
   `differential_BFDR_pairwise_{Method}` column (the long-form view of the
   same correction).

Accepted `method` âˆˆ `(:bh, :bonferroni, :holm, :none)`; `:bh` is the
default. For `:none`, the new column is a verbatim copy of
`differential_BFDR` (NOT `missing` â€” more useful
downstream than `missing`-fill).

Returns the (now-mutated) `wide_df` for chaining.

For k=2 (n_contrasts=1), BH is identity (cumulative-min from the right over a
single-element family), so the new column equals `differential_BFDR` numerically.
This is the byte-equality precondition.
"""
function _apply_multi_test_correction!(pairwise_dict::Dict{Pair{Symbol, Symbol}, DataFrame},
                                       wide_df::DataFrame,
                                       method::Symbol)
    method in (:bh, :bonferroni, :holm, :none) || throw(ArgumentError(
        "Unsupported multi_test_method: $method. Valid: :bh, :bonferroni, :holm, :none."))

    method_suffix = method === :bh         ? "BH"         :
                    method === :bonferroni ? "Bonferroni" :
                    method === :holm       ? "Holm"       :
                                             "None"

    # Identify per-pair differential_BFDR columns in the wide DF.
    # 's _aggregate_pairwise_results suffixes columns as `_{c}_vs_{d}`.
    # For k=2 special-case (n_contrasts == 1), the wide DF uses unsuffixed
    # columns (byte-equality); detect that path by sniffing the column name.
    pair_keys = sort(collect(keys(pairwise_dict)); by = p -> (String(first(p)), String(last(p))))
    n_contrasts = length(pair_keys)

    flat_values = Float64[]
    flat_origin = Tuple{Pair{Symbol, Symbol}, Int}[]   # (pair, row_idx)

    for pair in pair_keys
        df_pair = pairwise_dict[pair]
        col = df_pair.differential_BFDR
        for (i, v) in enumerate(col)
            if !ismissing(v)
                push!(flat_values, Float64(v))
                push!(flat_origin, (pair, i))
            end
        end
    end

    # Apply correction.
    # ----------------------------------------------------------------
    # byte-equality short-circuit: when n_contrasts == 1 (the legacy
    # 2-group case), the cross-pair multi-test correction degenerates to
    # an identity on differential_BFDR. This preserves byte-equality between
    # `differential_analysis(ar_A, ar_B)` and the equivalent k=2 NamedTuple
    # call (the new column equals the existing differential_BFDR column).
    # For n_contrasts â‰¥ 2 (true k-group case), the correction is applied
    # to the flattened (n_proteins Ã— n_contrasts) family as documented.
    # ----------------------------------------------------------------
    n_total = length(flat_values)
    corrected = if n_contrasts == 1
        Vector{Union{Missing, Float64}}(flat_values)
    elseif method === :bh
        _bh_adjust(flat_values)
    elseif method === :bonferroni
        _bonferroni_adjust(flat_values, n_total)
    elseif method === :holm
        _holm_adjust(flat_values)
    else   # :none â€” verbatim pass-through
        Vector{Union{Missing, Float64}}(flat_values)
    end

    # Scatter back into per-pair DataFrames + wide DF.
    per_pair_corrected = Dict{Pair{Symbol, Symbol}, Vector{Union{Missing, Float64}}}()
    for pair in pair_keys
        per_pair_corrected[pair] = Vector{Union{Missing, Float64}}(missing, nrow(pairwise_dict[pair]))
    end
    for (idx, (pair, row_idx)) in enumerate(flat_origin)
        per_pair_corrected[pair][row_idx] = corrected[idx]
    end

    new_col_long = Symbol("differential_BFDR_pairwise_$method_suffix")
    for pair in pair_keys
        pairwise_dict[pair][!, new_col_long] = per_pair_corrected[pair]
    end

    # Wide DF: one column per contrast, suffixed `_{c}_vs_{d}_pairwise_{Method}`.
    # For k=2 (n_contrasts == 1), additionally write the unsuffixed column for
    # byte-equality (the legacy 2-group DF has no `_wt_vs_mut1` suffix).
    if n_contrasts == 1
        pair = first(pair_keys)
        wide_df[!, new_col_long] = per_pair_corrected[pair]
    else
        # Build a wide_df-length column keyed by Protein. Per-pair DataFrames
        # may have different row counts than wide_df (their intersection via
        # inner-join in _aggregate_pairwise_results), so direct assignment of
        # the per-pair vector to wide_df throws "New columns must have the same
        # length as old columns". Map by Protein name instead.
        for pair in pair_keys
            c, d = String(first(pair)), String(last(pair))
            wide_col = Symbol("differential_BFDR_$(c)_vs_$(d)_pairwise_$method_suffix")
            df_pair = pairwise_dict[pair]
            lookup = Dict{Any, Union{Missing, Float64}}()
            corr_pair = per_pair_corrected[pair]
            for (i, p) in enumerate(df_pair.Protein)
                lookup[p] = corr_pair[i]
            end
            wide_df[!, wide_col] = Union{Missing, Float64}[get(lookup, p, missing) for p in wide_df.Protein]
        end
    end

    return wide_df
end

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# k-group results aggregation (long-form Dict â†’ wide DF)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

"""
    _first_non_missing(vs)

Return the first non-missing element of `vs`, or `missing` if all are missing.
Used by `_aggregate_pairwise_results` to deduplicate identifier
columns (e.g. `:uniprot_id`, `:gene_name`, `:kgroup_class`) across per-pair
DataFrames after an outerjoin chain — these columns carry the same per-protein
value across pairs (when present at all), so we collapse them to a single
unsuffixed column on the wide DataFrame.

Inline helper (no in-repo callsite for this idiom — verified via grep on
2026-05-19; closest precedent is the `coalesce` pattern used elsewhere in the
report and pipeline code).
"""
function _first_non_missing(vs)
    i = findfirst(!ismissing, vs)
    return i === nothing ? missing : vs[i]
end

"""
    _aggregate_pairwise_results(pairwise_dict::Dict{Pair{Symbol, Symbol}, DataFrame},
                                contrast_pairs::Vector{Pair{Symbol, Symbol}}) -> DataFrame

Aggregate per-pair differential DataFrames into a single wide `results` table.

# k=2 special case (byte-equality)

When `length(contrast_pairs) == 1`, the function returns the single per-pair
DataFrame VERBATIM (no copy, no suffixing). This is the byte-equality
contract: `differential_analysis(conditions = (wt = ar_wt, mut = ar_mut)).results`
MUST equal `differential_analysis(ar_wt, ar_mut; condition_A="wt", condition_B="mut").results`
modulo the multi-test correction column added by `_apply_multi_test_correction!`.

# k ≥ 3 outer-join semantics (..e)

Per-pair DataFrames are renamed (numerical columns get `_<a>_vs_<b>` suffix;
identifier columns get a temporary per-pair tag) and chained via `outerjoin`
on `:Protein`. The result is then post-processed:

- `outerjoin` (not `innerjoin`): a protein P appears in the wide DF
  if it appears in ANY per-pair DataFrame. This satisfies the success criterion
  `nrow(wide_df) ≥ maximum(nrow(per_pair_df))`.
- A new `:n_pairs_with_data::Int` column records, per row, how many
  per-pair DataFrames contributed data for that protein. Range `[1, n_contrasts]`.
- Cells absent in a given pair-DF land as `Missing` (NOT `NaN`).
  DataFrames.jl widens column types to `Union{Missing, T}` automatically during
  the outer-join — no manual coercion needed.
- Identifier columns (`:uniprot_id`, `:gene_name`, `:enriched_in`,
  `:depleted_in`, `:kgroup_class`) are deduplicated via first-non-missing across
  pairs and land in the wide DF UNSUFFIXED. Pair-specific numerical columns get
  the `_<a>_vs_<b>` suffix. Note: `:enriched_in` / `:depleted_in` / `:kgroup_class`
  are added LATER by `_compute_kgroup_classification_columns!` directly on the
  wide DF; the ID_COLS whitelist below silently no-ops for absent columns, so
  including them here is forward-looking and harmless.

# Column ordering (k ≥ 3)

`:Protein` first, then the (unsuffixed) identifier columns, then all pair-suffixed
numerical columns in `contrast_pairs` iteration order, then `:n_pairs_with_data`.


"""
function _aggregate_pairwise_results(pairwise_dict::Dict{Pair{Symbol, Symbol}, DataFrame},
                                     contrast_pairs::Vector{Pair{Symbol, Symbol}})
    isempty(contrast_pairs) && throw(ArgumentError("_aggregate_pairwise_results: empty contrast_pairs"))
    n_contrasts = length(contrast_pairs)

    # byte-equality: k=2 returns the single per-pair DF verbatim (no copy, no suffixing).
    # lock — DO NOT TOUCH this short-circuit.
    if n_contrasts == 1
        return pairwise_dict[contrast_pairs[1]]
    end

    # ─────────────────────────────────────────────────────────────────────
    # ..e — k ≥ 3 outer-join branch
    # ─────────────────────────────────────────────────────────────────────

    # identifier columns that get DEDUPLICATED (unsuffixed) post-outerjoin.
    # Closed whitelist — anything not on this list gets the `_<a>_vs_<b>` suffix.
    # `:enriched_in` / `:depleted_in` / `:kgroup_class` are added LATER (by
    # `_compute_kgroup_classification_columns!`) on the wide DF; listing them
    # here is forward-looking (in case upstream code starts producing them on
    # per-pair DFs) and silently no-ops via the `isempty(tagged_cols)` check.
    ID_COLS = ["uniprot_id", "gene_name", "enriched_in", "depleted_in", "kgroup_class"]

    # Step 1: For each per-pair DF, suffix every NON-id, NON-Protein column with
    #         `_<a>_vs_<b>`. ID columns get a temporary per-pair tag (`__pair<i>`)
    #         so the outer-join doesn't auto-rename them; we collapse them back
    #         to unsuffixed form via `_first_non_missing` in Step 3.
    #
    #         We also record a per-pair "existence sentinel" column name — any
    #         suffixed numerical column from this pair's DF that we can later
    #         check for `ismissing` to derive `:n_pairs_with_data`.
    renamed = Vector{DataFrame}(undef, n_contrasts)
    id_tags = String[]
    sentinel_cols = String[]   # parallel to renamed[]: one suffixed col name per pair
    for (i, pair) in enumerate(contrast_pairs)
        df = pairwise_dict[pair]
        c, d = String(first(pair)), String(last(pair))
        suffix = "_$(c)_vs_$(d)"
        id_tag = "__pair$(i)"
        push!(id_tags, id_tag)

        new_df = copy(df)
        sentinel_for_pair = ""
        for col in df_names(new_df)
            col == "Protein" && continue   # join key — never suffixed
            if col in ID_COLS
                rename!(new_df, col => col * id_tag)
            else
                new_name = col * suffix
                rename!(new_df, col => new_name)
                # Pick the FIRST suffixed numerical column as the existence
                # sentinel for this pair. Stable across DataFrame iteration
                # because `df_names` returns columns in insertion order.
                isempty(sentinel_for_pair) && (sentinel_for_pair = new_name)
            end
        end
        push!(sentinel_cols, sentinel_for_pair)
        renamed[i] = new_df
    end

    # Step 2: chain OUTER join on :Protein.
    wide = renamed[1]
    for i in 2:n_contrasts
        wide = outerjoin(wide, renamed[i], on = :Protein)
    end

    # Step 3: collapse tagged identifier columns into a single
    #         unsuffixed column via `_first_non_missing` across pairs.
    for id_col in ID_COLS
        tagged_cols = String[id_col * tag for tag in id_tags if (id_col * tag) in df_names(wide)]
        isempty(tagged_cols) && continue
        wide[!, id_col] = [
            _first_non_missing([wide[r, tcol] for tcol in tagged_cols])
            for r in 1:nrow(wide)
        ]
        # Drop the temp-tagged columns now that we've collapsed.
        select!(wide, Not([Symbol(tcol) for tcol in tagged_cols]))
    end

    # Step 4: :n_pairs_with_data::Int column.
    #         For each row, count how many per-pair DFs contributed data.
    #         Use `sentinel_cols[i]` (first suffixed numerical column from pair i,
    #         recorded during Step 1) as the existence sentinel: if `wide[r, sentinel_cols[i]]`
    #         is not missing, this row was present in pair i's source DF.
    sentinels_present = String[c for c in sentinel_cols if !isempty(c) && c in df_names(wide)]
    if !isempty(sentinels_present)
        wide[!, :n_pairs_with_data] = Int[
            count(c -> !ismissing(wide[r, c]), sentinels_present)
            for r in 1:nrow(wide)
        ]
    else
        wide[!, :n_pairs_with_data] = zeros(Int, nrow(wide))
    end

    return wide
end

"""
    _kgroup_classification_summary(per_pair_classifications::AbstractVector,
                                   contrast_pairs::Vector{Pair{Symbol, Symbol}}) -> String

Build the per-row human-readable classification summary string for the
`classification_summary::String` column on the k-group wide `results::DataFrame`.

# Token grammar (Â§7 lock)

| Per-pair class           | Token       |
|--------------------------|-------------|
| `GAINED`                 | `{c}>{d}`   |
| `REDUCED`                | `{c}<{d}`   |
| `UNCHANGED`              | `{c}={d}`   |
| `BOTH_NEGATIVE`          | `{c}0{d}`   |
| `CONDITION_A_SPECIFIC`   | `{c}!`      |
| `CONDITION_B_SPECIFIC`   | `{d}!`      |

Tokens are joined with `"; "` (semicolon + single space). The separator is
chosen to avoid the comma collision in CSV exports.

Example: `_kgroup_classification_summary([GAINED, GAINED, UNCHANGED],
[:wt => :mut1, :wt => :mut2, :mut1 => :mut2])` returns `"wt>mut1; wt>mut2; mut1=mut2"`.


"""
function _kgroup_classification_summary(per_pair_classifications::AbstractVector,
                                        contrast_pairs::Vector{Pair{Symbol, Symbol}})
    n = length(contrast_pairs)
    isempty(contrast_pairs) && throw(ArgumentError(
        "_kgroup_classification_summary: empty contrast_pairs"))
    length(per_pair_classifications) == n || throw(ArgumentError(
        "_kgroup_classification_summary: length mismatch (got " *
        "$(length(per_pair_classifications)) classifications, $n pairs)"))

    tokens = String[]
    for (cls, pair) in zip(per_pair_classifications, contrast_pairs)
        c, d = String(first(pair)), String(last(pair))
        push!(tokens, if cls === GAINED
            "$c>$d"
        elseif cls === REDUCED
            "$c<$d"
        elseif cls === UNCHANGED
            "$c=$d"
        elseif cls === BOTH_NEGATIVE
            "$(c)0$(d)"
        elseif cls === CONDITION_A_SPECIFIC
            "$(c)!"
        elseif cls === CONDITION_B_SPECIFIC
            "$(d)!"
        else
            # Defensive fallback â€” should never fire with the locked 6-class enum.
            "?"
        end)
    end
    return join(tokens, "; ")
end

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# k-group validation helper
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

"""
    _validate_kgroup_arguments(conditions::NamedTuple, contrasts) -> Vector{Pair{Symbol, Symbol}}

Validate the k-group `differential_analysis` keyword overload inputs and normalise
the three accepted `contrasts` forms into the canonical
`Vector{Pair{Symbol, Symbol}}`.

# Accepted contrasts forms

1. `:all_pairs` â€” all (k choose 2) ordered pairs `:c_i => :c_j` for `i < j`
   in NamedTuple-key insertion order.
2. `:vs_reference => :sym` â€” `(k-1)` pairs of the form `:sym => :other` for
   every other key in `conditions`.
3. `Vector{Pair{Symbol, Symbol}}` â€” explicit list, validated verbatim.

# Validation rules (throws ArgumentError on violation)

- k = `length(conditions)` MUST be â‰¥ 2
- All symbols in `contrasts` MUST appear as keys in `conditions`
- No self-pairs (`:c => :c`)
- No duplicate pairs (Symbol-equality)
- Empty `Vector{Pair}` is rejected
- `:vs_reference => :sym` requires `:sym` to be a key in `conditions`

# Soft warning (does NOT throw)

- Emits `@warn maxlog=1` when ARs in `conditions` have differing `bait_protein`
  values (k-group differential is most meaningful for shared-bait designs).
  ARs with `bait_protein === nothing` are skipped in the uniqueness check.


"""
function _validate_kgroup_arguments(conditions::NamedTuple, contrasts)
    k = length(conditions)
    k >= 2 || throw(ArgumentError("differential_analysis requires k ≥ 2 conditions; got $k"))
    cond_keys = collect(keys(conditions))  # ordered

    # Normalise contrasts â†’ Vector{Pair{Symbol, Symbol}}
    pairs::Vector{Pair{Symbol, Symbol}} = if contrasts === :all_pairs
        [cond_keys[i] => cond_keys[j] for i in 1:k for j in (i+1):k]
    elseif contrasts isa Pair{Symbol, Symbol} && first(contrasts) === :vs_reference
        ref = last(contrasts)
        ref in cond_keys || throw(ArgumentError(
            "Unknown reference symbol $ref; valid: $(cond_keys)"))
        [ref => other for other in cond_keys if other !== ref]
    elseif contrasts isa AbstractVector
        # Coerce to Vector{Pair{Symbol, Symbol}} â€” fail loudly on type mismatch
        all(p -> p isa Pair{Symbol, Symbol}, contrasts) || throw(ArgumentError(
            "Explicit contrasts must be Vector{Pair{Symbol, Symbol}}; got $(typeof(contrasts))"))
        collect(Pair{Symbol, Symbol}, contrasts)
    else
        throw(ArgumentError(
            "Unsupported contrasts form: $(typeof(contrasts)). " *
            "Expected :all_pairs, :vs_reference => :sym, or Vector{Pair{Symbol, Symbol}}."))
    end

    # Validate the normalised pairs
    isempty(pairs) && throw(ArgumentError("Empty contrasts vector"))

    seen = Set{Pair{Symbol, Symbol}}()
    for p in pairs
        c, d = first(p), last(p)
        c === d && throw(ArgumentError("Self-pair in contrasts: $c => $d"))
        c in cond_keys || throw(ArgumentError(
            "Unknown contrast symbol $c; valid: $(cond_keys)"))
        d in cond_keys || throw(ArgumentError(
            "Unknown contrast symbol $d; valid: $(cond_keys)"))
        p in seen && throw(ArgumentError("Duplicate pair in contrasts: $p"))
        push!(seen, p)
    end

    # Soft check: bait-mismatch @warn (does NOT block).
    baits = unique([conditions[k_].bait_protein for k_ in cond_keys
                    if hasproperty(conditions[k_], :bait_protein) &&
                       conditions[k_].bait_protein !== nothing])
    if length(baits) > 1
        @warn "[Differential] AnalysisResult bait mismatch across conditions: $baits; " *
              "k-group differential is most meaningful for shared-bait designs" maxlog=1
    end

    return pairs
end

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Pairwise contrasts orchestrator (slot-vector + try/catch)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

"""
    _run_pairwise_contrasts(conditions::NamedTuple,
                            contrast_pairs::Vector{Pair{Symbol, Symbol}},
                            config::DifferentialConfig;
                            parallel_pairs::Union{Symbol, Bool} = :auto)
        -> Dict{Pair{Symbol, Symbol}, DataFrame}

Orchestrate the `(k choose 2)` (or user-specified subset) pairwise 2-group
`differential_analysis` calls and return the long-form Dict of per-pair
result DataFrames.

# Parallelism

- `:auto` (default): `Threads.@threads` when `length(conditions) â‰¥ 3` AND
  `Threads.nthreads() â‰¥ 4`; serial otherwise.
- `true`: force threads (with `@warn maxlog=1` when `nthreads < 2`).
- `false`: force serial.

# Fault tolerance (Â§5)

Each per-pair call runs inside a try/catch. Failures are captured into a
per-thread error slot vector and surfaced via `@warn maxlog=10
exception=(e, catch_backtrace())` AFTER the parallel loop completes. Failed
pairs are OMITTED from the returned Dict. If ALL pairs fail, the first error
is re-raised.

# Thread safety

Results are written to disjoint slots in a `Vector{Union{Nothing, DataFrame}}`
indexed by contrast position; the Dict is assembled AFTER the loop completes
(NEVER inside `Threads.@threads` â€” Dict insertion is not thread-safe).
"""
function _run_pairwise_contrasts(conditions::NamedTuple,
                                 contrast_pairs::Vector{Pair{Symbol, Symbol}},
                                 config::DifferentialConfig;
                                 parallel_pairs::Union{Symbol, Bool} = :auto,
                                 loss_matrix::Matrix{Float64} = DEFAULT_DIFFERENTIAL_LOSS)
    n = length(contrast_pairs)
    nthreads = Threads.nthreads()
    use_threads =
        parallel_pairs === true ||
        (parallel_pairs === :auto && length(conditions) >= 3 && nthreads >= 4)

    if parallel_pairs === true && nthreads < 2
        @warn "[Differential] differential_analysis: parallel_pairs=true but " *
              "Threads.nthreads()=$nthreads (no parallelism available); falling back to serial" maxlog=1
        use_threads = false
    end

    per_pair_results = Vector{Union{Nothing, DataFrame}}(nothing, n)
    per_pair_errors  = Vector{Union{Nothing, Exception}}(nothing, n)

    if use_threads
        Threads.@threads for i in 1:n
            _runpair!(i, conditions, contrast_pairs, config, per_pair_results, per_pair_errors;
                      loss_matrix = loss_matrix)
        end
    else
        for i in 1:n
            _runpair!(i, conditions, contrast_pairs, config, per_pair_results, per_pair_errors;
                      loss_matrix = loss_matrix)
        end
    end

    # Surface failures.
    n_fail = count(!isnothing, per_pair_errors)
    if n_fail == n && n > 0
        # All pairs failed â€” re-raise the first error so the user sees the cause.
        first_err = nothing
        for e in per_pair_errors
            e === nothing && continue
            first_err = e
            break
        end
        first_err === nothing || throw(first_err)
    elseif n_fail > 0
        for (i, e) in enumerate(per_pair_errors)
            e === nothing && continue
            @warn "[Differential] contrast $(contrast_pairs[i]) failed â€” entry will be missing from pairwise_results" exception=(e, catch_backtrace()) maxlog=10
        end
    end

    # Assemble the Dict from the slot vector (preserves contrast_pairs ordering
    # via Julia's insertion-ordered Dict behaviour since 1.4; but downstream
    # code MUST iterate `contrast_pairs` not `keys(dict)`).
    out = Dict{Pair{Symbol, Symbol}, DataFrame}()
    for i in 1:n
        per_pair_results[i] === nothing && continue
        out[contrast_pairs[i]] = per_pair_results[i]
    end
    return out
end

# Per-pair worker. Writes to results[i] / errors[i] slots only; never touches
# other shared state. Returns nothing â€” caller inspects the slot vectors.
function _runpair!(i::Int,
                   conditions::NamedTuple,
                   pairs::Vector{Pair{Symbol, Symbol}},
                   config::DifferentialConfig,
                   results::Vector{Union{Nothing, DataFrame}},
                   errors::Vector{Union{Nothing, Exception}};
                   loss_matrix::Matrix{Float64} = DEFAULT_DIFFERENTIAL_LOSS)
    try
        c, d = first(pairs[i]), last(pairs[i])
        ar_c = conditions[c]
        ar_d = conditions[d]
        # forward loss_matrix to legacy 2-group call so
        # per-pair Decision Risk columns get the override (Path A wiring).
        diff_i = differential_analysis(ar_c, ar_d;
            condition_A = String(c),
            condition_B = String(d),
            config = config,
            loss_matrix = loss_matrix)
        results[i] = diff_i.results
    catch e
        errors[i] = e
    end
    return nothing
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

    # Combined differential BF (unclamped â€” volcano plot uses asinh for compression)
    log10_bf_A = _safe_log10.(df_shared.BF_A)
    log10_bf_B = _safe_log10.(df_shared.BF_B)
    log10_dbf = log10_bf_A .- log10_bf_B
    dbf = @. 10.0^clamp(log10_dbf, -300, 300)   # prevent Float64 overflow

    # Per-evidence differential BFs
    dbf_enrichment  = _safe_ratio.(df_shared.bf_enrichment_A, df_shared.bf_enrichment_B)
    dbf_correlation = _safe_ratio.(df_shared.bf_correlation_A, df_shared.bf_correlation_B)
    dbf_detected    = _safe_ratio.(df_shared.bf_detected_A, df_shared.bf_detected_B)

    # per-protein dBF diagnostic â€” `n` is already defined at the top
    # of this function. Sub-model BF columns (BMA) are guarded with
    # `hasproperty` so the helper falls through to `:ok` when copula or 3c-EM BFs absent.
    bf_em_A_col   = hasproperty(df_shared, :bf_em_A)     ? df_shared.bf_em_A     : fill(missing, n)
    bf_em_B_col   = hasproperty(df_shared, :bf_em_B)     ? df_shared.bf_em_B     : fill(missing, n)
    bf_cop_A_col  = hasproperty(df_shared, :bf_copula_A) ? df_shared.bf_copula_A : fill(missing, n)
    bf_cop_B_col  = hasproperty(df_shared, :bf_copula_B) ? df_shared.bf_copula_B : fill(missing, n)

    dbf_diagnostic = [_compute_dbf_diagnostic(
        df_shared.BF_A[i], df_shared.BF_B[i],
        dbf_enrichment[i], dbf_correlation[i], dbf_detected[i],
        log10_dbf[i],
        bf_em_A_col[i], bf_cop_A_col[i], bf_em_B_col[i], bf_cop_B_col[i]
    ) for i in 1:n]

    # Effect size
    log2fc_A_raw = _to_float.(df_shared.mean_log2FC_A)
    log2fc_B_raw = _to_float.(df_shared.mean_log2FC_B)
    if config.standardize_log2fc
        delta_log2fc = _zscore(log2fc_A_raw) .- _zscore(log2fc_B_raw)
    else
        delta_log2fc = log2fc_A_raw .- log2fc_B_raw
    end

    # Delta posterior (directional)
    delta_posterior = _to_float.(df_shared.posterior_prob_A) .- _to_float.(df_shared.posterior_prob_B)

    # Differential posterior: P(diff|data) from the SYMMETRIC (two-sided) differential BF.
    # The differential hypothesis "BF_A â‰  BF_B" is direction-agnostic: evidence is equally
    # strong whether a protein is gained in A (dBF â‰« 1) OR reduced in A (dBF â‰ª 1). Using the
    # raw dBF (= 10^log10_dbf, always > 0) was direction-biased â€” reduced proteins (dBF < 1)
    # collapsed to differential_posterior < 0.5 and never reached significance, so REDUCED was
    # never called. Fold both tails together via 10^|log10_dbf| so the magnitude of evidence is
    # symmetric; the *direction* is resolved downstream in _classify_interactions from the sign
    # of log10_dbf / delta_log2fc.
    two_sided_dbf = @. 10.0^clamp(abs(log10_dbf), -300, 300)
    differential_posterior = two_sided_dbf ./ (1.0 .+ two_sided_dbf)

    # Multiple testing correction via Bayesian FDR
    differential_BFDR = bfdr(differential_posterior, isBF = false)
    differential_pep = pep(differential_posterior)   # canonical lowercase

    # Classification
    classification = _classify_interactions(
        df_shared, log10_dbf, delta_log2fc, differential_BFDR, config
    )

    # Î³-PEP four class-conditional columns (assumes conditional independence).
    # `_compute_gamma_pep` reads `df_shared.delta_log2fc`; attach the locally-computed
    # delta_log2fc vector to df_shared before the call (Rule 1: column missing from join).
    df_shared.delta_log2fc = delta_log2fc
    gamma_pep = _compute_gamma_pep(df_shared, config)

    # per-side bb_mnar_codriven flags (defensive against missing column from old caches).
    # After _rename_columns + innerjoin, df_shared carries bb_mnar_codriven_A / bb_mnar_codriven_B
    # whenever the upstream copula_results DataFrames carried them (11).
    bb_codriven_A_full = hasproperty(df_shared, :bb_mnar_codriven_A) ?
        Bool[Bool(coalesce(x, false)) for x in df_shared.bb_mnar_codriven_A] :
        fill(false, n)
    bb_codriven_B_full = hasproperty(df_shared, :bb_mnar_codriven_B) ?
        Bool[Bool(coalesce(x, false)) for x in df_shared.bb_mnar_codriven_B] :
        fill(false, n)

    # Optional columns: diagnostic_flag and sensitivity_range (present only when diagnostics ran)
    diag_A = hasproperty(df_shared, :diagnostic_flag_A) ?
        coalesce.(string.(df_shared.diagnostic_flag_A), "") : fill("", n)
    diag_B = hasproperty(df_shared, :diagnostic_flag_B) ?
        coalesce.(string.(df_shared.diagnostic_flag_B), "") : fill("", n)
    sens_A = hasproperty(df_shared, :sensitivity_range_A) ?
        _to_float.(df_shared.sensitivity_range_A) : fill(NaN, n)
    sens_B = hasproperty(df_shared, :sensitivity_range_B) ?
        _to_float.(df_shared.sensitivity_range_B) : fill(NaN, n)

    results_df = DataFrame(
        Protein              = df_shared.Protein,
        bf_A                 = _to_float.(df_shared.BF_A),
        bf_B                 = _to_float.(df_shared.BF_B),
        dbf                  = dbf,
        log10_dbf            = log10_dbf,
        posterior_A          = _to_float.(df_shared.posterior_prob_A),
        posterior_B          = _to_float.(df_shared.posterior_prob_B),
        delta_posterior      = delta_posterior,
        BFDR_A              = df_shared.BFDR_A,
        BFDR_B              = df_shared.BFDR_B,
        PEP_A               = df_shared.PEP_A,
        PEP_B               = df_shared.PEP_B,
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
        differential_BFDR    = differential_BFDR,
        differential_pep     = differential_pep,
        pep_gained           = gamma_pep.pep_gained,
        pep_reduced          = gamma_pep.pep_reduced,
        pep_unchanged        = gamma_pep.pep_unchanged,
        pep_both_negative    = gamma_pep.pep_both_negative,
        classification       = classification,
        dbf_diagnostic       = dbf_diagnostic,
        diagnostic_flag_A    = diag_A,
        diagnostic_flag_B    = diag_B,
        bb_codriven_A        = bb_codriven_A_full,
        bb_codriven_B        = bb_codriven_B_full,
        sensitivity_range_A  = sens_A,
        sensitivity_range_B  = sens_B,
    )
    # silent uppercase mirror is established in the outer
    # `differential_analysis` after all push! rows have been appended, so the
    # two columns can share their final Vector backing array.
    return results_df
end

"""
Classify each protein as GAINED, REDUCED, UNCHANGED, or BOTH_NEGATIVE.

Interactor status is determined by per-condition BFDR values (BFDR_A, BFDR_B < bfdr_threshold).
If neither condition is a significant interactor the protein is always BOTH_NEGATIVE â€”
UNCHANGED is reserved for proteins where at least one condition passes the BFDR threshold.

## Methods
- `:posterior`: Interactor in A (BFDR_A < threshold) but not B â†’ GAINED if Î”logâ‚‚FC â‰¥ 0.
  When both are interactors, uses delta_log2fc threshold.
- `:dbf`: |log10(dBF)| exceeds threshold â†’ GAINED or REDUCED by sign.
- `:combined`: Both BFDR interactor criteria and dBF criteria must hold.
"""
function _classify_interactions(
    df_shared::DataFrame,
    log10_dbf::Vector{Float64},
    delta_log2fc::Vector{Float64},
    differential_BFDR,
    config::DifferentialConfig
)
    n = nrow(df_shared)
    classification = fill(UNCHANGED, n)

    for i in 1:n
        bfdr_val = ismissing(differential_BFDR[i]) ? 1.0 : Float64(differential_BFDR[i])
        is_significant = bfdr_val < config.bfdr_threshold

        # Determine per-condition interactor status (shared across all methods)
        bfdrA = ismissing(df_shared.BFDR_A[i]) ? 1.0 : Float64(df_shared.BFDR_A[i])
        bfdrB = ismissing(df_shared.BFDR_B[i]) ? 1.0 : Float64(df_shared.BFDR_B[i])
        is_interactor_A = bfdrA < config.bfdr_threshold
        is_interactor_B = bfdrB < config.bfdr_threshold

        # Base rule: if neither condition is a significant interactor â†’ BOTH_NEGATIVE.
        # UNCHANGED is reserved for proteins where at least one condition is an interactor
        # but the differential evidence doesn't support a directional call.
        if !is_interactor_A && !is_interactor_B
            classification[i] = BOTH_NEGATIVE
            continue
        end

        if config.classification_method == :posterior
            if is_significant
                has_differential_evidence = abs(log10_dbf[i]) > 0.0

                if !has_differential_evidence
                    # dBF == 1 exactly: no differential signal
                    classification[i] = UNCHANGED
                elseif is_interactor_A && !is_interactor_B
                    classification[i] = delta_log2fc[i] >= 0 ? GAINED : UNCHANGED
                elseif !is_interactor_A && is_interactor_B
                    classification[i] = delta_log2fc[i] <= 0 ? REDUCED : UNCHANGED
                elseif is_interactor_A && is_interactor_B
                    if delta_log2fc[i] > config.delta_log2fc_threshold
                        classification[i] = GAINED
                    elseif delta_log2fc[i] < -config.delta_log2fc_threshold
                        classification[i] = REDUCED
                    end
                    # else: both interactors, small delta â†’ stays UNCHANGED
                end
            end
            # else: not significant â†’ stays UNCHANGED (at least one is interactor)

        elseif config.classification_method == :dbf
            if is_significant
                if log10_dbf[i] > config.dbf_threshold
                    classification[i] = GAINED
                elseif log10_dbf[i] < -config.dbf_threshold
                    classification[i] = REDUCED
                end
            end

        elseif config.classification_method == :combined
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

"""
    _compute_gamma_pep(df_shared::DataFrame, config::DifferentialConfig)
        -> NamedTuple{(:pep_gained, :pep_reduced, :pep_unchanged, :pep_both_negative),
                       NTuple{4, Vector{Union{Missing, Float64}}}}

closed-form normalized naive-product Î³-PEP estimator over the
shared-protein DataFrame.

For each row:

    p_A = coalesce(posterior_calibrated_A, posterior_prob_A)  # mixed calib
    p_B = coalesce(posterior_calibrated_B, posterior_prob_B)
    Î´   = config.delta_log2fc_threshold                        # NOT delta_threshold
    k   = 10.0                                                 # sigmoid_gate convention

    Ïƒ_pos  = 1 / (1 + exp(-k Â· (Î”log2FC - Î´)))    # P(Î”log2FC > Î´ | data) approx
    Ïƒ_neg  = 1 / (1 + exp(-k Â· (-Î”log2FC - Î´)))   # P(Î”log2FC < -Î´ | data) approx
    Ïƒ_zero = max(0.0, 1 - Ïƒ_pos - Ïƒ_neg)          # clamp non-negative

Naive-product weights (assumes conditional independence
`P(s_A, s_B | data) â‰ˆ P(s_A) Â· P(s_B)`):

    raw_gained        = p_A Â· (1 - p_B) Â· Ïƒ_pos
    raw_reduced       = (1 - p_A) Â· p_B Â· Ïƒ_neg
    raw_unchanged     = p_A Â· p_B Â· Ïƒ_zero
    raw_both_negative = (1 - p_A) Â· (1 - p_B)

    Z = raw_gained + raw_reduced + raw_unchanged + raw_both_negative
    if Z < eps(Float64) or !isfinite(Z):     # Z-vanishing fallback
        P_class = 0.25  (uniform) â†’ pep_class = 0.75 for all four classes
    else:
        pep_class = 1.0 - raw_class / Z

T-70-10-01 mitigation: rows with `missing` or NaN inputs return `missing` for
all four classes (NOT NaN propagation).

Conditional-independence assumption is documented in the Methods tab subsection.
Joint-posterior alternative (bootstrap / copula coupling) is deferred to v1.3.

Returns a NamedTuple with four `Vector{Union{Missing, Float64}}` of length
`nrow(df_shared)`, keys exactly `(:pep_gained, :pep_reduced, :pep_unchanged,
:pep_both_negative)`.
"""
function _compute_gamma_pep(df_shared::DataFrame, config::DifferentialConfig)
    n     = nrow(df_shared)
    delta = config.delta_log2fc_threshold     # canonical field name
    k     = 10.0                              # sigmoid_gate convention

    pep_g  = Vector{Union{Missing, Float64}}(undef, n)
    pep_r  = Vector{Union{Missing, Float64}}(undef, n)
    pep_u  = Vector{Union{Missing, Float64}}(undef, n)
    pep_bn = Vector{Union{Missing, Float64}}(undef, n)

    # prefer calibrated posterior column when present (per-side independently)
    has_pcal_A = hasproperty(df_shared, :posterior_calibrated_A)
    has_pcal_B = hasproperty(df_shared, :posterior_calibrated_B)

    for i in 1:n
        p_A_raw = has_pcal_A ?
            coalesce(df_shared.posterior_calibrated_A[i], df_shared.posterior_prob_A[i]) :
            df_shared.posterior_prob_A[i]
        p_B_raw = has_pcal_B ?
            coalesce(df_shared.posterior_calibrated_B[i], df_shared.posterior_prob_B[i]) :
            df_shared.posterior_prob_B[i]
        d_raw   = df_shared.delta_log2fc[i]

        # T-70-10-01 mitigation: missing/NaN inputs â†’ missing outputs (not NaN propagation)
        p_A = _to_float(p_A_raw)
        p_B = _to_float(p_B_raw)
        d   = _to_float(d_raw)
        if isnan(p_A) || isnan(p_B) || isnan(d)
            pep_g[i]  = missing
            pep_r[i]  = missing
            pep_u[i]  = missing
            pep_bn[i] = missing
            continue
        end

        # Logistic gates on Î”log2FC (sigmoid_gate convention, k = 10.0)
        s_pos  = 1.0 / (1.0 + exp(-k * (d - delta)))
        s_neg  = 1.0 / (1.0 + exp(-k * (-d - delta)))
        s_zero = max(0.0, 1.0 - s_pos - s_neg)    # clamp non-negative

        # Naive-product raw weights
        raw_g  = p_A * (1.0 - p_B) * s_pos
        raw_r  = (1.0 - p_A) * p_B * s_neg
        raw_u  = p_A * p_B * s_zero
        raw_bn = (1.0 - p_A) * (1.0 - p_B)

        Z = raw_g + raw_r + raw_u + raw_bn
        if Z < eps(Float64) || !isfinite(Z)
            # degenerate Z â€” fall back to uniform P(class) = 0.25
            # (â†’ pep = 1 - 0.25 = 0.75 for all four classes)
            pep_g[i]  = 0.75
            pep_r[i]  = 0.75
            pep_u[i]  = 0.75
            pep_bn[i] = 0.75
        else
            pep_g[i]  = 1.0 - raw_g  / Z
            pep_r[i]  = 1.0 - raw_r  / Z
            pep_u[i]  = 1.0 - raw_u  / Z
            pep_bn[i] = 1.0 - raw_bn / Z
        end
    end
    return (
        pep_gained        = pep_g,
        pep_reduced       = pep_r,
        pep_unchanged     = pep_u,
        pep_both_negative = pep_bn,
    )
end

"""Append condition-specific proteins to results with NaN fill."""
function _append_condition_specific!(
    results_df::DataFrame,
    df_A::DataFrame,
    df_B::DataFrame,
    only_A::Set{<:AbstractString},
    only_B::Set{<:AbstractString},
    config::DifferentialConfig
)
    for protein in only_A
        idx = findfirst(==(protein), df_A.Protein)
        row_A = df_A[idx, :]
        push!(results_df, _make_condition_specific_row(protein, row_A, :A, config); promote = true)
    end

    for protein in only_B
        idx = findfirst(==(protein), df_B.Protein)
        row_B = df_B[idx, :]
        push!(results_df, _make_condition_specific_row(protein, row_B, :B, config); promote = true)
    end

    return results_df
end

"""Build a result row for a condition-specific protein.

Only assigns CONDITION_A/B_SPECIFIC if the protein's posterior probability
exceeds the threshold; otherwise classifies as BOTH_NEGATIVE (non-significant
interactor that happens to be absent from the other condition's dataset)."""
function _make_condition_specific_row(protein::AbstractString, row, condition::Symbol,
                                      config::DifferentialConfig)
    nan = NaN
    diag  = hasproperty(row, :diagnostic_flag) ?
                coalesce(string(row.diagnostic_flag), "") : ""
    sens  = hasproperty(row, :sensitivity_range) ? _to_float(row.sensitivity_range) : nan

    bfdr_val = ismissing(row.BFDR) ? 1.0 : Float64(row.BFDR)
    is_significant_interactor = bfdr_val < config.bfdr_threshold

    # per-side codriven flags â€” defensive against rows without the column
    row_bb_flag = hasproperty(row, :bb_mnar_codriven) ?
        Bool(coalesce(row.bb_mnar_codriven, false)) : false

    if condition == :A
        cls = is_significant_interactor ? CONDITION_A_SPECIFIC : BOTH_NEGATIVE
        # Î³-PEP under single-condition collapse rule
        # A-only: p_B = 0, Ïƒ_pos = 1, Ïƒ_neg = 0, Ïƒ_zero = 0
        # raw_gained = p_A Â· 1 Â· 1 = p_A ; raw_both_negative = (1 - p_A) Â· 1 = 1 - p_A
        # raw_reduced = raw_unchanged = 0 ; Z = 1 â†’ P(gained)=p_A, P(both_neg)=1-p_A
        p_A_row = _to_float(coalesce(row.posterior_prob, missing))
        if isnan(p_A_row)
            pep_g_v  = missing
            pep_r_v  = missing
            pep_u_v  = missing
            pep_bn_v = missing
        else
            pep_g_v  = 1.0 - p_A_row              # P(gained) = p_A
            pep_r_v  = 1.0                         # P(reduced) = 0
            pep_u_v  = 1.0                         # P(unchanged) = 0
            pep_bn_v = p_A_row                     # P(both_negative) = 1 - p_A
        end
        return (
            Protein = protein,
            bf_A = _to_float(row.BF), bf_B = nan,
            dbf = nan, log10_dbf = nan,
            posterior_A = _to_float(row.posterior_prob), posterior_B = nan,
            delta_posterior = nan,
            BFDR_A = row.BFDR, BFDR_B = missing,
            PEP_A = row.PEP, PEP_B = missing,
            log2fc_A = _to_float(row.mean_log2FC), log2fc_B = nan,
            delta_log2fc = nan,
            bf_enrichment_A = _to_float(row.bf_enrichment), bf_enrichment_B = nan,
            dbf_enrichment = nan,
            bf_correlation_A = _to_float(row.bf_correlation), bf_correlation_B = nan,
            dbf_correlation = nan,
            bf_detected_A = _to_float(row.bf_detected), bf_detected_B = nan,
            dbf_detected = nan,
            differential_posterior = nan,
            differential_BFDR = missing,
            differential_pep = missing,            # Î± undefined for condition-specific
            pep_gained         = pep_g_v,          # collapse rule
            pep_reduced        = pep_r_v,          # collapse rule
            pep_unchanged      = pep_u_v,          # collapse rule
            pep_both_negative  = pep_bn_v,         # collapse rule
            classification = cls,
            dbf_diagnostic = :ok,
            diagnostic_flag_A = diag, diagnostic_flag_B = "",
            bb_codriven_A = row_bb_flag,           # A side codriven
            bb_codriven_B = false,                 # B absent
            sensitivity_range_A = sens, sensitivity_range_B = nan,
            detected_in = "condition_a_only",
            # diff_PEP omitted â€” DataFrame silent mirror (same Vector ref as differential_pep)
            # is auto-extended by push! since both columns share the same backing array.
        )
    else
        cls = is_significant_interactor ? CONDITION_B_SPECIFIC : BOTH_NEGATIVE
        # Î³-PEP under single-condition collapse rule
        # B-only: p_A = 0, Ïƒ_neg = 1, Ïƒ_pos = 0, Ïƒ_zero = 0
        # raw_reduced = 1 Â· p_B Â· 1 = p_B ; raw_both_negative = 1 Â· (1 - p_B) = 1 - p_B
        # raw_gained = raw_unchanged = 0 ; Z = 1 â†’ P(reduced)=p_B, P(both_neg)=1-p_B
        p_B_row = _to_float(coalesce(row.posterior_prob, missing))
        if isnan(p_B_row)
            pep_g_v  = missing
            pep_r_v  = missing
            pep_u_v  = missing
            pep_bn_v = missing
        else
            pep_g_v  = 1.0                         # P(gained) = 0
            pep_r_v  = 1.0 - p_B_row              # P(reduced) = p_B
            pep_u_v  = 1.0                         # P(unchanged) = 0
            pep_bn_v = p_B_row                     # P(both_negative) = 1 - p_B
        end
        return (
            Protein = protein,
            bf_A = nan, bf_B = _to_float(row.BF),
            dbf = nan, log10_dbf = nan,
            posterior_A = nan, posterior_B = _to_float(row.posterior_prob),
            delta_posterior = nan,
            BFDR_A = missing, BFDR_B = row.BFDR,
            PEP_A = missing, PEP_B = row.PEP,
            log2fc_A = nan, log2fc_B = _to_float(row.mean_log2FC),
            delta_log2fc = nan,
            bf_enrichment_A = nan, bf_enrichment_B = _to_float(row.bf_enrichment),
            dbf_enrichment = nan,
            bf_correlation_A = nan, bf_correlation_B = _to_float(row.bf_correlation),
            dbf_correlation = nan,
            bf_detected_A = nan, bf_detected_B = _to_float(row.bf_detected),
            dbf_detected = nan,
            differential_posterior = nan,
            differential_BFDR = missing,
            differential_pep = missing,            # Î± undefined for condition-specific
            pep_gained         = pep_g_v,          # collapse rule
            pep_reduced        = pep_r_v,          # collapse rule
            pep_unchanged      = pep_u_v,          # collapse rule
            pep_both_negative  = pep_bn_v,         # collapse rule
            classification = cls,
            dbf_diagnostic = :ok,
            diagnostic_flag_A = "", diagnostic_flag_B = diag,
            bb_codriven_A = false,                 # A absent
            bb_codriven_B = row_bb_flag,           # B side codriven
            sensitivity_range_A = nan, sensitivity_range_B = sens,
            detected_in = "condition_b_only",
            # diff_PEP omitted â€” DataFrame silent mirror (same Vector ref as differential_pep)
            # is auto-extended by push! since both columns share the same backing array.
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
    significant_differential(diff::DifferentialResult; bfdr_threshold=0.05) -> DataFrame

Return all proteins with significant differential interaction evidence.
"""
function significant_differential(diff::DifferentialResult; bfdr_threshold::Float64 = 0.05)
    valid_idx = findall(x -> !ismissing(x) && x < bfdr_threshold, diff.results.differential_BFDR)
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
            "Posterior threshold", "BFDR threshold",
            "Delta log2FC threshold", "dBF threshold",
            "Classification method", "Standardize log2FC"
        ],
        Value = [
            diff.condition_A, diff.condition_B,
            string(diff.n_proteins_A), string(diff.n_proteins_B),
            string(diff.n_shared),
            string(diff.n_condition_A_specific), string(diff.n_condition_B_specific),
            string(diff.n_gained), string(diff.n_reduced), string(diff.n_unchanged), string(diff.n_both_negative),
            string(diff.config.posterior_threshold), string(diff.config.bfdr_threshold),
            string(diff.config.delta_log2fc_threshold), string(diff.config.dbf_threshold),
            string(diff.config.classification_method), string(diff.config.standardize_log2fc)
        ]
    )

    # Convert classification enum to strings for Excel compatibility
    export_df = copy(diff.results)
    export_df.classification = string.(export_df.classification)
    # convert dbf_diagnostic Symbol to String for XLSX compatibility
    if hasproperty(export_df, :dbf_diagnostic)
        export_df.dbf_diagnostic = string.(export_df.dbf_diagnostic)
    end
    # convert Decision Risk Symbol columns to String for XLSX compatibility
    if hasproperty(export_df, :optimal_call)
        export_df.optimal_call = string.(export_df.optimal_call)
    end
    if hasproperty(export_df, :optimal_call_min)
        export_df.optimal_call_min = string.(export_df.optimal_call_min)
    end
    # enriched_in / depleted_in / kgroup_class also need conversion for XLSX
    if hasproperty(export_df, :enriched_in)
        export_df.enriched_in = [string(x) for x in export_df.enriched_in]
    end
    if hasproperty(export_df, :depleted_in)
        export_df.depleted_in = [string(x) for x in export_df.depleted_in]
    end
    if hasproperty(export_df, :kgroup_class)
        export_df.kgroup_class = string.(export_df.kgroup_class)
    end

    writetable(filepath,
        "differential" => export_df,
        "summary" => summary_df
    )
end

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# k-group keyword-only differential_analysis overload
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

"""
    differential_analysis(; conditions, contrasts = :all_pairs,
                          config = DifferentialConfig(),
                          multi_test_method = :bh,
                          parallel_pairs = :auto) -> DifferentialResult

keyword-only k-group overload. Run pairwise differential analysis across
`k â‰¥ 2` conditions and aggregate the results into a single `DifferentialResult`.

# Keywords

- `conditions::NamedTuple` â€” one `AbstractAnalysisResult` per condition; NamedTuple
  KEYS become the condition labels carried through into `pairwise_results`,
  `contrasts`, and `condition_labels(::DifferentialResult)`. E.g.
  `conditions = (wt = ar_wt, mut1 = ar_mut1, mut2 = ar_mut2)`.
- `contrasts` â€” one of `:all_pairs` (default, all `(k choose 2)` pairs in NamedTuple
  key order), `:vs_reference => :sym` (one-vs-reference, `k-1` pairs), or
  `Vector{Pair{Symbol, Symbol}}` (explicit list).
- `config::DifferentialConfig = DifferentialConfig()` â€” per-pair analysis config;
  the same config is used for every pair.
- `multi_test_method::Symbol = :bh` â€” cross-pair correction over the
  `(n_proteins Ã— n_contrasts)` family. One of `:bh`, `:bonferroni`, `:holm`,
  `:none`.
- `parallel_pairs::Union{Symbol, Bool} = :auto` â€” pairwise parallelism mode.
  `:auto` enables `Threads.@threads` when `k â‰¥ 3` AND `nthreads â‰¥ 4`.

# Returns

A `DifferentialResult` with `contrasts` populated and `pairwise_results::Dict`
holding the per-pair DataFrames. For k=2 the wide `results::DataFrame` is
byte-identical to the legacy 2-group call. For kâ‰¥3 the wide DF gains
`_{c}_vs_{d}` suffixed columns plus a `classification_summary::String` column.

# Example

```julia
ar_wt   = run_analysis(config_wt).analysis_result
ar_mut1 = run_analysis(config_mut1).analysis_result
ar_mut2 = run_analysis(config_mut2).analysis_result

diff = differential_analysis(
    conditions = (wt = ar_wt, mut1 = ar_mut1, mut2 = ar_mut2),
    contrasts = :vs_reference => :wt,
    config = DifferentialConfig(bfdr_threshold = 0.01),
)

# Access per-pair sub-result
df_wt_vs_mut1 = diff.pairwise_results[:wt => :mut1]
```


"""
function differential_analysis(;
    conditions::NamedTuple,
    contrasts = :all_pairs,
    config::DifferentialConfig = DifferentialConfig(),
    multi_test_method::Symbol = :bh,
    parallel_pairs::Union{Symbol, Bool} = :auto,
    loss_matrix::Matrix{Float64} = hasproperty(config, :loss_matrix) ?
                                   config.loss_matrix : DEFAULT_DIFFERENTIAL_LOSS,
)
    # 1. Validate + normalise contrasts.
    contrast_pairs = _validate_kgroup_arguments(conditions, contrasts)

    # 2. Run pairwise differentials (orchestrator).
    # thread loss_matrix into the orchestrator so the per-pair
    # Decision Risk columns inside pairwise_dict use the override (Path A wiring).
    pairwise_dict = _run_pairwise_contrasts(conditions, contrast_pairs, config;
                                            parallel_pairs = parallel_pairs,
                                            loss_matrix = loss_matrix)

    # 2b. If a pair failed, the dict may be missing entries; aggregation requires
    # every pair in contrast_pairs to be present. Subset contrast_pairs to surviving
    # entries (Âfailed pairs are OMITTED). If none survive,
    # _run_pairwise_contrasts has already re-raised the first error so we never
    # reach this branch with an empty dict.
    surviving_pairs = Pair{Symbol, Symbol}[p for p in contrast_pairs if haskey(pairwise_dict, p)]
    isempty(surviving_pairs) && throw(ErrorException(
        "[Differential] differential_analysis: all pairwise contrasts failed (this should have been re-raised earlier)"))

    # 3. Aggregate into wide DF.
    wide_df = _aggregate_pairwise_results(pairwise_dict, surviving_pairs)

    # 4. Multi-test correction. Mutates pairwise_dict + wide_df in place.
    _apply_multi_test_correction!(pairwise_dict, wide_df, multi_test_method)

    n_contrasts = length(surviving_pairs)

    # 5. For kâ‰¥3: add classification_summary column to wide DF.
    if n_contrasts >= 2
        # Each row's summary is built from the per-pair classification at that row.
        # Look up per-pair classification by joining on :Protein.
        classifications_by_pair = Dict{Pair{Symbol, Symbol}, Dict{String, Any}}()
        for pair in surviving_pairs
            df_pair = pairwise_dict[pair]
            cls_map = Dict{String, Any}()
            if hasproperty(df_pair, :classification)
                for r in 1:nrow(df_pair)
                    cls_map[String(df_pair.Protein[r])] = df_pair.classification[r]
                end
            end
            classifications_by_pair[pair] = cls_map
        end
        summary_strings = String[]
        for protein in wide_df.Protein
            per_pair_cls = [get(classifications_by_pair[pair], String(protein), UNCHANGED)
                            for pair in surviving_pairs]
            push!(summary_strings, _kgroup_classification_summary(per_pair_cls, surviving_pairs))
        end
        wide_df[!, :classification_summary] = summary_strings
    end

    # omnibus columns (5 new columns on wide_df).
    # Hoisted cond_keys / ars_all here so step 6 (condition_similarity) reuses them.
    cond_keys = collect(keys(conditions))
    ars_all = AnalysisResult[conditions[k_] for k_ in cond_keys]
    cond_label_strings = String[String(k_) for k_ in cond_keys]
    eb_prior = _eb_pooled_prior(ars_all)
    _compute_omnibus_columns!(wide_df, ars_all, cond_label_strings, eb_prior)

    # classification columns (3 new columns on wide_df).
    # Reads bfdr_omnibus from wide_df (must run AFTER step 5b).
    _compute_kgroup_classification_columns!(
        wide_df, ars_all, cond_label_strings,
        config.posterior_threshold, config.bfdr_threshold,
    )

    # wide-table Decision Risk aggregation for k-group.
    # Per-pair Decision Risk lives in pairwise_dict (populated automatically via
    # _runpair! -> legacy differential_analysis recursion under Path A). For the
    # wide view we expose the per-protein minimum across pairs + the call achieving it.
    # NaN propagation: if all pairs give NaN (all CONDITION_A/B_SPECIFIC), the
    # wide-table value is NaN with optimal_call_min = :condition_specific
    # (deliberately distinct from per-pair :condition_a_specific /
    # :condition_b_specific -- those are pair-specific labels that do not carry
    # over to the protein-level wide view).
    let n_wide = nrow(wide_df)
        decision_risk_min = fill(NaN, n_wide)
        optimal_call_min  = fill(:condition_specific, n_wide)
        proteins_wide = wide_df.Protein
        for (i, protein_name) in enumerate(proteins_wide)
            best_risk = Inf
            best_call = :condition_specific
            any_finite_pair = false
            for pair in surviving_pairs
                sub_df = pairwise_dict[pair]
                hasproperty(sub_df, :decision_risk) || continue
                row_idx = findfirst(==(protein_name), sub_df.Protein)
                row_idx === nothing && continue
                r = sub_df.decision_risk[row_idx]
                isnan(r) && continue
                any_finite_pair = true
                if r < best_risk
                    best_risk = r
                    best_call = sub_df.optimal_call[row_idx]
                end
            end
            if any_finite_pair
                decision_risk_min[i] = best_risk
                optimal_call_min[i]  = best_call
            end
        end
        wide_df[!, :decision_risk_min] = decision_risk_min
        wide_df[!, :optimal_call_min]  = optimal_call_min
    end

    # 6. Compute k-group condition_similarity (hook).
    emb_cfg = (hasproperty(ars_all[1], :config) && ars_all[1].config !== nothing &&
               hasproperty(ars_all[1].config, :embeddings_config)) ?
              ars_all[1].config.embeddings_config : EmbeddingsConfig()
    condition_similarity = try
        if emb_cfg.run_embeddings && length(ars_all) >= 2
            result = _compute_condition_similarity(ars_all, emb_cfg)
            # Open Q 10: post-overwrite labels with NamedTuple keys
            # so the report shows ["wt", "mut1", "mut2"] not ["BAIT", "BAIT", "BAIT"].
            # The struct field is `condition_labels` (lock).
            if result !== nothing && hasproperty(result, :condition_labels)
                result.condition_labels = [String(k_) for k_ in cond_keys]
            end
            result
        else
            nothing
        end
    catch e
        @warn "[Differential] k-group condition similarity failed: $e" maxlog=1 exception=(e, catch_backtrace())
        nothing
    end

    # 7. Compute legacy summary counts. For k=2 the wide DF has a `classification` column
    # (verbatim from the single per-pair DF). For kâ‰¥3 we count GAINED/REDUCED/etc. from
    # the FIRST contrast's classification column â€” these legacy fields lose their
    # k-group meaning but remain populated for backward compat with the report.
    cls_col = if n_contrasts == 1 && hasproperty(wide_df, :classification)
        wide_df.classification
    else
        # Use the first pair's per-protein classifications, joined by Protein.
        first_pair = surviving_pairs[1]
        df_first = pairwise_dict[first_pair]
        if hasproperty(df_first, :classification)
            cls_map_first = Dict(String(p) => c for (p, c) in zip(df_first.Protein, df_first.classification))
            [get(cls_map_first, String(p), UNCHANGED) for p in wide_df.Protein]
        else
            InteractionClass[UNCHANGED for _ in 1:nrow(wide_df)]
        end
    end

    n_gained        = count(==(GAINED),        cls_col)
    n_reduced       = count(==(REDUCED),       cls_col)
    n_unchanged     = count(==(UNCHANGED),     cls_col)
    n_both_negative = count(==(BOTH_NEGATIVE), cls_col)
    n_shared = nrow(wide_df)

    # 8. is_calibrated provenance â€” use the first two ARs (legacy 2-group field, kept for compat).
    is_cal_A = hasproperty(ars_all[1], :is_calibrated) ? ars_all[1].is_calibrated : false
    is_cal_B = length(ars_all) >= 2 && hasproperty(ars_all[2], :is_calibrated) ? ars_all[2].is_calibrated : false

    # 9. Build the DifferentialResult via the 20-arg canonical ctor.
    result = DifferentialResult(
        wide_df,
        String(first(surviving_pairs[1])),  # condition_A â€” first pair's "A" side
        String(last(surviving_pairs[1])),   # condition_B â€” first pair's "B" side
        config,
        n_shared, n_shared, n_shared,       # n_proteins_A / n_proteins_B / n_shared all == wide DF row count
        0, 0,                               # n_condition_A_specific / n_condition_B_specific â€” k-group: handled per-pair, top-level 0
        now(),
        n_gained, n_reduced, n_unchanged, n_both_negative,
        ars_all,                            # analyses
        is_cal_A, is_cal_B,                 # calibration provenance
        condition_similarity,               # kÃ—k similarity
        surviving_pairs,                    # canonical contrasts ordering
        pairwise_dict,                      # per-pair DataFrames
    )

    # emit k-group companion files
    # (per-pair xlsx sheets + 5 svg plots per pair). No-op for k=2 (the legacy
    # writer block above at L267-289 / L383-405 already handles k=2 companion
    # files, and the byte-equality lock at forbids any duplicate
    # emission on the legacy 2-group path).
    _write_kgroup_companion_files(result, config)

    return result
end

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# k-group companion file emission helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

"""
    _suffix_plot_path(base::String, a::String, b::String) -> String

Insert a `_<a>_vs_<b>` suffix immediately before the file extension of `base`.
Returns the input unchanged when `base` is empty. Uses `splitext` to preserve
the extension verbatim (`.svg`, `.png`, `.pdf`, â€¦).

Examples:
    _suffix_plot_path("differential_volcano.svg", "wt", "mut1") == "differential_volcano_wt_vs_mut1.svg"
    _suffix_plot_path("results/diff.png",          "wt", "mut1") == "results/diff_wt_vs_mut1.png"
    _suffix_plot_path("",                          "wt", "mut1") == ""

path scheme.
"""
function _suffix_plot_path(base::String, a::String, b::String)
    isempty(base) && return base
    root, ext = splitext(base)
    return root * "_$(a)_vs_$(b)" * ext
end

"""
    _build_per_pair_diff(diff::DifferentialResult, pair::Pair{Symbol,Symbol},
                        df_pair::DataFrame) -> DifferentialResult

Construct a per-pair `DifferentialResult` suitable for the legacy 2-group plot
writers (`differential_volcano_plot`, `differential_evidence_plot`, â€¦) and
the legacy `export_differential` writer. Uses the **14-arg backward-compat
ctor** (Â§9 Pitfall 1) so `contrasts` defaults to
`Pair{Symbol,Symbol}[]` â€” the empty-contrasts state triggers the legacy
non-suffixed codepaths inside the visualization helpers.

The per-pair DF mirrors the legacy 2-group `results::DataFrame` shape because
`_runpair!` builds it via the legacy `differential_analysis(ar_a, ar_b; â€¦)`
recursion (`src/differential/analysis.jl:1086`, Path A wiring).
"""
function _build_per_pair_diff(diff::DifferentialResult, pair::Pair{Symbol, Symbol}, df_pair::DataFrame)
    cond_A = String(first(pair))
    cond_B = String(last(pair))
    n_rows = nrow(df_pair)
    # 14-arg backward-compat ctor â€” cascades to the canonical 20-arg form with
    # analyses=nothing, is_calibrated_A/B=false, condition_similarity=nothing,
    # contrasts=Pair{Symbol,Symbol}[] (legacy default), pairwise_results=nothing.
    return DifferentialResult(
        df_pair,
        cond_A, cond_B,
        diff.config,
        n_rows, n_rows, n_rows,
        0, 0,
        diff.timestamp,
        diff.n_gained, diff.n_reduced, diff.n_unchanged, diff.n_both_negative,
    )
end

"""
    _write_kgroup_companion_files(diff::DifferentialResult,
                                  config::DifferentialConfig) -> Nothing

emit k-group companion files for k â‰¥ 3.

For each pair in `diff.contrasts`:
- 5 plot files (`volcano`, `evidence`, `scatter`, `classification`, `ma`)
  written with `_<a>_vs_<b>` suffix (extension preserved by `_suffix_plot_path`).

After per-pair plots are emitted, a single multi-sheet xlsx (`config.results_file`)
is written with:
- `Sheet1` â€” the wide cross-pair `DataFrame` (`diff.results`), sorted descending
  by `decision_risk_min` when present (ordering contract).
- One additional sheet per pair named `<a>_vs_<b>` (e.g. `wtHTT_vs_mHTT`) holding
  the per-pair `DataFrame` verbatim.

Early-returns (no-op) when `length(diff.contrasts) <= 1`: for k = 2 the legacy
2-group writer block (`differential_analysis(ar_A, ar_B; â€¦)` at L267-289 of
this file) is the single source of file emission. Re-emitting from this helper
would violate the byte-equality lock.

# Locks honoured
- byte-equality (k=2 no-op).
- Symbol-to-String conversion for XLSX (mirrors `export_differential`).
- BMA terminology stays "Copula" + "3c-EM" (no new terminology introduced).
- FDR terminology stays BFDR / PEP / local_fdr.
"""
function _write_kgroup_companion_files(diff::DifferentialResult, config::DifferentialConfig)
    # k=2 byte-equality guard. lock.
    length(diff.contrasts) <= 1 && return nothing

    diff.pairwise_results === nothing && return nothing

    # ── 1. Per-pair plots ────────────────────────────────────────────────────
    for pair in diff.contrasts
        df_pair = get(diff.pairwise_results, pair, nothing)
        df_pair === nothing && continue   # surviving_pairs filter already applied upstream
        pair_diff = _build_per_pair_diff(diff, pair, df_pair)
        a_str = String(first(pair))
        b_str = String(last(pair))

        # Each writer wrapped individually so a single failure does not block
        # the remaining per-pair artifacts (matches single-bait
        # graceful-degradation discipline; logs maxlog=1 to keep production
        # logs readable).
        try
            plt = differential_volcano_plot(pair_diff)
            StatsPlots.savefig(plt, _suffix_plot_path(config.volcano_file, a_str, b_str))
        catch e
            @warn "volcano plot failed for $a_str vs $b_str: $e" maxlog=1
        end
        try
            plt = differential_evidence_plot(pair_diff)
            StatsPlots.savefig(plt, _suffix_plot_path(config.evidence_file, a_str, b_str))
        catch e
            @warn "evidence plot failed for $a_str vs $b_str: $e" maxlog=1
        end
        try
            plt = differential_scatter_plot(pair_diff)
            StatsPlots.savefig(plt, _suffix_plot_path(config.scatter_file, a_str, b_str))
        catch e
            @warn "scatter plot failed for $a_str vs $b_str: $e" maxlog=1
        end
        try
            plt = differential_classification_plot(pair_diff)
            StatsPlots.savefig(plt, _suffix_plot_path(config.classification_file, a_str, b_str))
        catch e
            @warn "classification plot failed for $a_str vs $b_str: $e" maxlog=1
        end
        try
            plt = differential_ma_plot(pair_diff)
            StatsPlots.savefig(plt, _suffix_plot_path(config.ma_file, a_str, b_str))
        catch e
            @warn "MA plot failed for $a_str vs $b_str: $e" maxlog=1
        end
    end

    # ── 2. Multi-sheet xlsx ──────────────────────────────────────────────────
    # sort: descending by decision_risk_min when present, else ascending
    # by differential_BFDR_omnibus when present, else preserve insertion order.
    sheet_wide = _xlsx_safe_copy(diff.results)
    if hasproperty(sheet_wide, :decision_risk_min)
        # Sort descending; NaN values go LAST under standard isless semantics
        # (NaN > anything). Using rev=true puts large values first; NaNs sink.
        # Note: DataFrames `sort!` treats NaN as larger than any Float64; with
        # rev=true NaNs would land FIRST. We want them LAST, so sort ascending
        # on negated values via a key:
        try
            sheet_wide = sort(sheet_wide, :decision_risk_min, rev = false)
            # The above puts low risk (= best validation candidates) first; that
            # matches the "low decision risk = strong candidate" semantics from
            # NaN rows sink to the bottom under standard isless.
        catch
            # Best-effort sort; preserve insertion order if anything goes wrong.
        end
    elseif hasproperty(sheet_wide, :differential_BFDR_omnibus)
        try
            sheet_wide = sort(sheet_wide, :differential_BFDR_omnibus, rev = false)
        catch
        end
    end

    sheets = Pair{String, DataFrame}["Sheet1" => sheet_wide]
    for pair in diff.contrasts
        df_pair = get(diff.pairwise_results, pair, nothing)
        df_pair === nothing && continue
        sheet_name = "$(String(first(pair)))_vs_$(String(last(pair)))"
        push!(sheets, sheet_name => _xlsx_safe_copy(df_pair))
    end

    try
        writetable(config.results_file, sheets...; overwrite = true)
    catch e
        @warn "xlsx write failed for $(config.results_file): $e" maxlog=1
    end

    return nothing
end

"""
    _xlsx_safe_copy(df::DataFrame) -> DataFrame

Internal helper: return a copy of `df` with Symbol / InteractionClass / vector
columns converted to `String` so the result is safe to hand to
`XLSX.writetable`. Mirrors the conversion block at the top of
`export_differential` but additionally walks ALL columns to
catch suffixed variants (e.g. `classification_wt_vs_mut1`,
`dbf_diagnostic_wt_vs_mut1`, `optimal_call_wt_vs_mut1`) which the wide
cross-pair DF carries for k â‰¥ 3.
"""
function _xlsx_safe_copy(df::DataFrame)
    out = copy(df)
    for col in names(out)
        v = out[!, col]
        et = eltype(v)
        # InteractionClass enum (un-/suffixed `classification*` columns)
        if et === InteractionClass || et === Union{Missing, InteractionClass}
            out[!, col] = [ismissing(x) ? missing : string(x) for x in v]
            continue
        end
        # Symbol scalars (e.g. `dbf_diagnostic*`, `optimal_call*`, `kgroup_class`)
        if et === Symbol || et === Union{Missing, Symbol}
            out[!, col] = [ismissing(x) ? missing : string(x) for x in v]
            continue
        end
        # Vector{Symbol} (e.g. `enriched_in`, `depleted_in`) â€” serialize as
        # comma-joined String for XLSX. Matches the existing per-row stringify
        # at `export_differential` L1792-1797.
        if et === Vector{Symbol} || et === Union{Missing, Vector{Symbol}}
            out[!, col] = [ismissing(x) ? missing : join(string.(x), ",") for x in v]
            continue
        end
    end
    return out
end
