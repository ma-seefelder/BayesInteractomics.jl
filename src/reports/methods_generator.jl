# src/reports/methods_generator.jl
# Auto-generates manuscript-ready methods text and parameter tables.

"""
    _html_esc(s) -> String

T-70-14-01 mitigation: HTML-escape a string for safe interpolation
into Methods HTML fragments. Mirrors the JS `esc()` helper in the report templates.
Used by `_methods_differential_block` to defang user-supplied `condition_A` /
`condition_B` labels (XSS path).
"""
function _html_esc(s)::String
    str = string(s)
    isempty(str) && return ""
    out = IOBuffer()
    for ch in str
        if ch == '&'
            print(out, "&amp;")
        elseif ch == '<'
            print(out, "&lt;")
        elseif ch == '>'
            print(out, "&gt;")
        elseif ch == '"'
            print(out, "&quot;")
        elseif ch == '\''
            print(out, "&#39;")
        else
            print(out, ch)
        end
    end
    return String(take!(out))
end

# short alias used in `_methods_differential_block`.
const esc = _html_esc

"""
    generate_methods_text(config::CONFIG, results::DataFrame) -> String

Generate a manuscript-ready methods paragraph from the analysis configuration.
"""
function generate_methods_text(config::CONFIG, results::DataFrame)::String
    n_proteins  = nrow(results)
    n_sig       = sum(skipmissing(results.BFDR) .≤ 0.05)
    n_strong    = sum(skipmissing(results.BFDR) .≤ 0.01)
    method_str  = config.combination_method == :copula ? "copula-based" : "latent class"
    regr_str    = config.regression_likelihood == :robust_t ?
                  "robust (Student-t, ν = $(round(config.student_t_nu, digits=1)))" : "normal"
    pkg_version = _report_pkg_version()

    return """
AP-MS data were analyzed using BayesInteractomics v$(pkg_version) (Julia v$(VERSION)). \
A total of $(n_proteins) proteins were evaluated for interaction with the '$(config.poi)' bait protein \
using $(config.n_controls) control experiment(s) and $(config.n_samples) sample experiment(s).

Protein-protein interactions were scored by integrating evidence from three independent Bayesian models: \
(1) a Beta-Bernoulli model for detection probability across replicates, \
(2) a Hierarchical Bayesian Model (HBM) for log₂ fold-change enrichment, and \
(3) a Bayesian $(regr_str) linear regression model for dose-response correlation. \
Individual Bayes factors from the three models were combined using a $(method_str) mixture model \
fit by expectation-maximization ($(config.em_n_restarts) restarts). \
Significant interactors were defined at Bayesian FDR (q) ≤ 0.05; $(n_sig) proteins met this \
threshold ($(n_strong) at q ≤ 0.01).

Software: BayesInteractomics v$(pkg_version) (Julia v$(VERSION)).
""" |> strip
end

"""
    generate_methods_parameters(config::CONFIG) -> Vector{Pair{String,String}}

Return a list of (parameter name, value) pairs covering **all** CONFIG fields,
suitable for display in the report's Analysis Parameters table.
"""
function generate_methods_parameters(config::CONFIG)::Vector{Pair{String,String}}
    # Fields to skip (complex nested structs serialised separately or too large for a table).
    # The four ablation knobs (evidence_streams, copula_family, h1_copula_family,
    # use_metalearner_prior) are skip-listed so report.html stays byte-identical on the full
    # default path; their non-default values surface in the run via the ablation harness, not
    # the standard parameters table.
    skip = Set([:output, :sensitivity_config, :diagnostics_config,
                :control_cols, :sample_cols, :em_prior,
                :evidence_streams, :copula_family, :h1_copula_family, :use_metalearner_prior])
    pairs = Pair{String,String}[]
    for fname in fieldnames(CONFIG)
        fname in skip && continue
        val = getfield(config, fname)
        s = if val isa Vector
                isempty(val) ? "[]" : "[" * join(basename.(string.(val)), ", ") * "]"
            elseif val isa AbstractFloat
                string(round(val, digits=4))
            else
                string(val)
            end
        push!(pairs, string(fname) => s)
    end
    return pairs
end

"""
    generate_reproducibility_block(config::CONFIG) -> String

Generate a complete reproducibility information block covering all CONFIG parameters.
"""
function generate_reproducibility_block(config::CONFIG)::String
    pkg_version = _report_pkg_version()
    lines = String[
        "Generated:         $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))",
        "Package:           BayesInteractomics v$(pkg_version)",
        "Julia:             $(VERSION)",
        "Output directory:  $(config.output.basedir)",
        "",
        "Input files:",
    ]
    for f in config.datafile
        push!(lines, "  $(f)")
    end
    push!(lines, "H0 file: $(config.output.H0_file)")
    push!(lines, "")
    push!(lines, "All configuration parameters:")

    # Same skip-list as generate_methods_parameters: the four ablation knobs
    # are excluded so the reproducibility block stays byte-identical on the full default path.
    skip = Set([:output, :sensitivity_config, :diagnostics_config,
                :control_cols, :sample_cols, :em_prior,
                :evidence_streams, :copula_family, :h1_copula_family, :use_metalearner_prior])
    for fname in fieldnames(CONFIG)
        fname in skip && continue
        val = getfield(config, fname)
        s = if val isa Vector
                isempty(val) ? "[]" : "[" * join(string.(val), ", ") * "]"
            elseif val isa AbstractFloat
                string(round(val, digits=6))
            else
                string(val)
            end
        push!(lines, "  $(lpad(string(fname), 30)) = $s")
    end
    return join(lines, "\n")
end

"""
    _report_pkg_version() -> String

Return BayesInteractomics package version string, or "?" if unavailable.
"""
function _report_pkg_version()::String
    try
        v = pkgversion(@__MODULE__)
        isnothing(v) ? "?" : string(v)
    catch
        "?"
    end
end

"""
    _metalearner_tuning_measure() -> String

Return the human-readable inner-CV tuning measure used to select the shipped
production metalearner. The shipped `MLJ.Stack` artefacts were tuned on the
Brier score (the production default), so this is a constant `"Brier"` for the
Methods-tab subsection.
"""
_metalearner_tuning_measure()::String = "Brier"

"""
    _methods_metalearner_status(status::Symbol) -> String

Return the "Metalearner Status" subsection HTML for the
Methods tab based on the `AnalysisResult.metalearner_status` sentinel. Returns
an empty string for unknown statuses so the Methods tab degrades gracefully.

Sentinels (from `src/ml/metalearner_stubs.jl::METALEARNER_STATUS_VALUES`):
- `:loaded` — metalearner extension active, posteriors are ML-adjusted
- `:extension_not_loaded` — Variante B fallback (BF-derived posteriors)
- `:prediction_failed` — extension loaded but `predict_metalearner` returned
   no usable predictions (commonly: bait renamed by STRING curation)

the `:loaded` branch was extended IN PLACE (no
parallel `_v77` helper — this exact function is wired into both
`templates/report.html` and `templates/differential_report.html`). The
extended body documents the production TR+DDI schema, the 6-candidate pool, the
inner-CV tuning measure, and the MC-Dropout opt-in.
This subsection carries WHAT the shipping system does now.
"""
function _methods_metalearner_status(status::Symbol)::String
    if status === :loaded
        measure = _metalearner_tuning_measure()
        return """
        <section class="methods-subsection mb-4">
        <h5>Metalearner Status</h5>
        <p>The metalearner extension (<code>BayesInteractomicsMetalearnerExt</code>) is loaded.
           Posterior probabilities in the Results table are <strong>metalearner-adjusted</strong>:
           the BF-derived posterior <code>bf/(1+bf)</code> from the EM/copula stage was further
           refined by a stacked meta-learner trained on STRING + DNN features.</p>
        <h6>Production schema</h6>
        <p>TR+DDI (14 features) &mdash; 7 STRING in-species channels + 1 DNN score
           + 4 STRING transferred channels + 2 Pfam DDI counts. Source:
           <code>metalearners/metalearner_tr_ddi.jld2</code>.</p>
        <h6>Candidate pool</h6>
        <p>Six base learners: HistGradientBoostingClassifier, EvoTreesClassifier,
           LogisticClassifier, KNNClassifier, RandomForestClassifier,
           ExtraTreesClassifier. Stacked via an <code>LR_L2</code> blender
           (Sill-2009 feature-weighted stacking &mdash; the original features are
           fed at both level-1 base training AND level-2 blender training).
           AdaBoost + GaussianNB were evaluated but dropped for structural
           mis-calibration (ECE 0.12 and 0.22 respectively) that survives
           hyperparameter tuning.</p>
        <h6>Tuning measure</h6>
        <p>Inner-CV tuning measure: <strong>$(measure)</strong> (selected on the
           ΔECE / ΔMCC trade-off between Brier and log-loss tuning).</p>
        <h6>MC-Dropout (deprecated opt-in)</h6>
        <p>The production <strong>default</strong> is the 14-feature
           <code>:tr_ddi</code> schema
           (<code>metalearners/metalearner_tr_ddi.jld2</code>). MC-Dropout &mdash;
           the 15-feature <code>:tr_ddi_mc</code> schema, which appends a per-pair
           MC-Dropout standard deviation <code>mc_std</code> reconstructed via K=30
           stochastic forward passes through the <em>same</em> DNN (model-473)
           already used to compute the <code>DNN</code> feature &mdash; is available
           as an <strong>opt-in</strong>: set
           <code>config.metalearner_use_mc_dropout = true</code>. This is
           <strong>deprecated</strong> as of v1.2.1. A species-agnostic benchmark
           shows <code>:tr_ddi</code> AUC 0.8381 vs <code>:tr_ddi_mc</code> 0.8194 and
           MCC 0.5319 vs 0.5075 on n=1862 multi-species pairs &mdash; the non-MC
           schema wins overall. MC-Dropout is retained for non-human-focused
           workflows where the K=30 cost may still provide lift on smaller, sparser
           species. There is <strong>no hard error</strong>: setting <code>true</code>
           emits a one-time deprecation <code>@warn</code> and loads
           <code>metalearners/metalearner_tr_ddi_mc.jld2</code>; if the extension and
           model are absent the whole metalearner falls back to <code>bf/(1+bf)</code>
           (Variante B), exactly as for the 14-feature schema. The underlying TR+DDI
           feature set is shared by both schemas.</p>
        </section>
        """
    elseif status === :extension_not_loaded
        return """
        <h5>Metalearner Status</h5>
        <p>The metalearner extension (<code>BayesInteractomicsMetalearnerExt</code>) is <strong>not loaded</strong>.
           Posterior probabilities are <strong>BF-derived</strong>:
           <code>posterior_prob = bf / (1 + bf)</code>, where <code>bf</code> is the BMA-combined
           Bayes factor from the EM/copula stage. To enable ML-adjusted posteriors, run
           <code>using Flux, MLJ, MLJScikitLearnInterface, HDF5</code>
           before <code>run_analysis(cfg)</code>. The fallback is documented as <em>Variante B</em>.</p>
        """
    elseif status === :prediction_failed
        return """
        <h5>Metalearner Status</h5>
        <p>The metalearner extension is loaded but <code>predict_metalearner</code> failed
           (most commonly: the bait was renamed by STRING curation and no longer matches
           the metalearner's training-data identifiers). Posterior probabilities fall back to
           BF-derived (<code>bf/(1+bf)</code>). See the Julia logs for the exception trace.</p>
        """
    else
        return ""  # unknown status — emit nothing (graceful degradation)
    end
end

"""
    _methods_dnn_prior_block(config) -> String

Render the "DNN Prior + MC-Dropout" subsection HTML for the
Methods tab. Adjacent helper to `_methods_metalearner_status` (line 161). Body
content: model-473 identity, MC-Dropout mechanism (K + BN
frozen + `_collect_dropouts!` walker), 95% CI definition (empirical quantile,
NOT Beta — KS p95 = 0.293 caveat),
performance budget, high-std caveat.

Two-branch dispatch:
- `!config.run_dnn_prior_mc_dropout` → explicit "disabled" message + the five
  new column names.
- Opt-in → four-paragraph subsection: model identity & mechanism, CI
  definition, calibration responsibility, and
  caveat + performance budget.

Verbatim numerical citations:
- K from `config.dnn_prior_mc_k`
- model-473 path: `encodings/model-473-0.5414302830201915.jld2`
- baseline ECE 0.023; with MC-Dropout ECE 0.026
- ρ_Spearman = 0.526 (MC variance vs absolute prediction error)
- KS p95 = 0.293 (Beta-fit invalidation)
"""
function _methods_dnn_prior_block(config)::String
    if !config.run_dnn_prior_mc_dropout
        return """
        <h4>DNN Prior + MC-Dropout</h4>
        <p>MC-Dropout uncertainty quantification is <strong>disabled</strong> for this run
           (<code>run_dnn_prior_mc_dropout = false</code>). The DNN Prior tab renders the
           empty-state alert. Set the flag to <code>true</code> to populate the
           five new columns: <code>prior_mc_mean</code>, <code>prior_mc_std</code>,
           <code>prior_mc_ci_low</code>, <code>prior_mc_ci_high</code>, and
           <code>prior_contribution</code>.</p>
        """
    end
    return """
    <h4>DNN Prior + MC-Dropout</h4>
    <p>Per-pair MC-Dropout uncertainty was computed with K=$(config.dnn_prior_mc_k)
       forward passes through model-473 (<code>encodings/model-473-0.5414302830201915.jld2</code>;
       11-layer Chain, 3072 input = CLS sequence + STRING network embedding, MCC 0.541,
       baseline ECE 0.023). BatchNorm runs in test mode (frozen running statistics);
       only Dropout (and AlphaDropout) layers stay active at inference, located via the
       recursive <code>_collect_dropouts!</code> struct walker — <code>Flux.fmap</code>
       alone does not visit Dropout layers (intermediate nodes, not leaves).</p>
    <p>The 95% CI is computed from the empirical K-sample distribution
       (<code>quantile(samples, [0.025, 0.975])</code>), NOT from a parametric
       Beta-of-Moments fit. A KS test (p95 = 0.293) invalidates Beta summarisability.</p>
    <p><strong>Calibration responsibility:</strong> the metalearner is the
       load-bearing calibration step; MC-Dropout is a <strong>transparency</strong>
       signal, NOT a calibration improvement. MC-Dropout slightly
       worsens ECE on model-473 (0.026 vs baseline 0.023) but ρ_Spearman between
       MC variance and absolute error is 0.526 — informative per-pair confidence.</p>
    <p><strong>Caveat:</strong> the DNN prior reflects sequence (CLS) and STRING
       network only — NOT experiment-specific signal. High <code>prior_mc_std</code>
       pairs are candidates for manual validation or pairs where the prior has
       little to contribute. <strong>Performance:</strong>
       K=$(config.dnn_prior_mc_k) on a 50k-pair proteome adds ~5–10 min on CPU
       (single-threaded inference); &lt;1 min on GPU. Opt out via
       <code>run_dnn_prior_mc_dropout = false</code>.</p>
    """
end

"""
    _methods_variance_recovery_block(config::CONFIG) -> String

render the Methods-tab `Variance Recovery` block.
Returns empty string when `config.mnar_variance_recovery == :off` (block is
HIDDEN — conditional rendering). For `:inflation` and `:multi_impute`,
returns an HTML fragment with mode, parameters, and summary stats.

Adjacent helper to `_methods_metalearner_status`; both
land inside the Methods tab via dedicated card divs in
`templates/report.html` and `templates/differential_report.html`.

seed contract for `:multi_impute`: derived seeds are
`base_seed * 1_000_003 + i` for `i ∈ 1..m`. With `base_seed=42, m=3` this
yields `[42_000_126, 42_000_127, 42_000_128]`.

No external placeholder-organisation URL; until a docs page
lands, the block has NO footer link (a broken URL is
worse than no URL).
"""
function _methods_variance_recovery_block(config::CONFIG)::String
    mode = config.mnar_variance_recovery
    mode === :off && return ""

    rows = String[]
    push!(rows, "<h5>Variance Recovery</h5>")
    push!(rows, "<table class=\"table table-sm\">")
    push!(rows, "<tr><th>Mode</th><td><code>:$(mode)</code></td></tr>")

    if mode === :inflation
        override_text = config.mnar_inflation_factor === nothing ?
                        "&mdash; (auto from dropout curves)" :
                        string(config.mnar_inflation_factor)
        push!(rows, "<tr><th>Inflation cap (<code>mnar_inflation_max</code>)</th><td>$(config.mnar_inflation_max)</td></tr>")
        push!(rows, "<tr><th>Override scalar (<code>mnar_inflation_factor</code>)</th><td>$override_text</td></tr>")
        push!(rows, "<tr><th>Per-protein inflation summary</th><td>see runtime <code>@info \":inflation applied\"</code> log line</td></tr>")
    elseif mode === :multi_impute
        m = config.mnar_m
        base_seed = config.mnar_base_seed
        # seed contract: seed_i = base_seed * 1_000_003 + i for i ∈ 1..m
        # base_seed=42, m=3 → [42_000_126, 42_000_127, 42_000_128]
        seeds = [Int(base_seed) * 1_000_003 + i for i in 1:m]
        push!(rows, "<tr><th>Imputations (<code>mnar_m</code>)</th><td>$m</td></tr>")
        push!(rows, "<tr><th>Base seed (<code>mnar_base_seed</code>)</th><td>$base_seed</td></tr>")
        push!(rows, "<tr><th>Per-imputation seeds</th><td><code>$(seeds)</code></td></tr>")
        push!(rows, "<tr><th>Rubin B/W ratio</th><td>see <code>rubin_bw_ratio</code> column when exposed (D-CD)</td></tr>")
    end

    push!(rows, "</table>")
    # NO footer link — a later docs page will cover this.
    # Until then, omit the link rather than ship a broken placeholder URL.
    return join(rows, "\n")
end

"""
    _methods_mask_aware_regression_block(config::CONFIG) -> String

render the Methods-tab `Mask-aware regression
(v2b)` subsection. Returns the empty string when `config.mask_aware_regression
== false` (silent suppression — no orphan anchor in the rendered HTML).

The block documents:
- The variance-additive observation factor:
  `var(data[cell]) = 1/τ[cell] + σ²_imp[cell] · is_imputed[cell]`.
- The σ_imp source: per-MS-run `DropoutFit` (the
  imputation extension exposes `column_imputation_sigma`).
- The CONFIG opt-out `mask_aware_regression = false`.

Adjacency: lives next to `_methods_variance_recovery_block` — both
helpers document imputation-related model details. Until a docs page
lands, the in-report block is the authoritative
reference (no external link, following the placeholder-URL convention).
"""
function _methods_mask_aware_regression_block(config::CONFIG)::String
    config.mask_aware_regression || return ""  # silent when opted-out
    nu_disp = round(config.student_t_nu, digits=1)
    return """
    <section class="methods-subsection mb-4">
      <h4 data-bs-toggle="tooltip"
          title="Variance-additive σ inflation prevents bf_correlation saturation on MNAR-imputed cells. Each imputed cell contributes an additional variance σ²_imp from the per-MS-run DropoutFit, preserving the biological τ[cell] precision for non-imputed cells.">
        Mask-aware regression (v2b)
      </h4>
      <p>
        For each cell in the regression input we compute an effective variance
        <code>var(data[cell]) = 1/τ[cell] + σ²_imp[cell] · is_imputed[cell]</code>,
        where <code>τ[cell] ~ Gamma(shape=ν/2, scale=2·τ_base/ν)</code> is the
        per-cell Empirical-Bayes precision (Student-t scale mixture,
        ν = $(nu_disp)), and <code>σ²_imp[cell]</code> is the per-MS-run
        tilted-Gaussian variance read from the dropout-curve fit
        (<code>DropoutFit</code>). The <code>is_imputed</code> mask is set
        cell-wise to <code>true</code> when the raw data was missing at that
        position prior to MNAR imputation.
      </p>
      <p>
        The added variance on imputed cells prevents the dose-response Bayes
        factor (<code>bf_correlation</code>) from saturating on MNAR-imputed
        data: imputed values carry the dropout-model uncertainty rather than
        being treated as exactly observed, so the regression evidence stays
        calibrated.
      </p>
      <p>
        Opt-out: set <code>mask_aware_regression = false</code> in your
        CONFIG to revert to the <code>precision=τ[cell]</code> observation
        factor (identical behaviour on raw / non-imputed data).
      </p>
    </section>
    """
end

"""
    _methods_normalisation_block(config::CONFIG) -> String

render the Methods-tab `Normalisation` subsection. Unlike
the opt-out mask-aware block, this block is rendered UNCONDITIONALLY — every
analysis applies a (possibly identity) normalisation, so the block always
documents what the pipeline did.

The block documents WHAT the system does now (per the project's "Methods tab
describes current behaviour, not the derivation journey" convention — no spike
narration):

- The `normalisation_method` selector and its five values
  (`:none`, `:row_center`, `:median_of_ratios`, `:both`, `:auto` — the default).
- `:auto` auto-applies `:both` (median_of_ratios + per-protein cross-protocol
  row-centering) on multi-protocol scale-disparate loads, and `:none` otherwise.
- `median_of_ratios` is DESeq size-factor SAMPLE normalisation; row-centering
  removes the per-protein cross-protocol baseline OFFSET — orthogonal axes.
- Normalisation runs BEFORE MNAR imputation.
- Row-centering preserves log2FC (it cancels in the sample − control contrast).
- The `bait_anchor` differential flag (per-condition raw-bait correction on
  sample cells only; default on).

The displayed effective/requested method reflects `config.normalisation_method`
verbatim; for `:auto` the block explains the multi-protocol-driven resolution
(the concrete flip is decided at load time by `detect_protocol_scale_mismatch`).

BMA terminology ("Copula" / "3c-EM") and FDR terminology (BFDR / PEP /
local_fdr) are untouched — this block is BMA-neutral and FDR-neutral.

No external placeholder URL — the Methods tab is the
authoritative in-report reference until a long-form docs page lands; a broken
placeholder link is worse than no link.
"""
function _methods_normalisation_block(config::CONFIG)::String
    method = config.normalisation_method
    return """
    <section class="methods-subsection mb-4">
      <h4 data-bs-toggle="tooltip"
          title="AP-MS intensities are normalised on the log2 scale before MNAR imputation. median_of_ratios equalises per-sample loading (DESeq size factors); per-protein cross-protocol row-centering removes the per-protein baseline offset across protocols. The two axes are orthogonal and compose under :both.">
        Normalisation
      </h4>
      <p>
        Active selector: <code>normalisation_method = :$(method)</code>. The
        selector accepts five values, all operating on the log2 intensity scale:
      </p>
      <ul>
        <li><code>:none</code> — no normalisation (byte-identical to the legacy
          <code>normalise_protocols = false</code>).</li>
        <li><code>:row_center</code> — per-protein cross-protocol row-centering
          only (byte-identical to the legacy <code>normalise_protocols = true</code>);
          removes the per-protein baseline offset between protocols.</li>
        <li><code>:median_of_ratios</code> — DESeq size-factor SAMPLE
          normalisation (per-protein geometric mean &rarr; per-sample median of
          ratios &rarr; divide), missing-aware, equalising per-sample loading.</li>
        <li><code>:both</code> — <code>:median_of_ratios</code> followed by
          per-protein cross-protocol row-centering. The two corrections are
          ORTHOGONAL axes: column-scaling fixes per-sample loading, row-centering
          removes the per-protein cross-protocol offset. Neither substitutes for
          the other.</li>
        <li><code>:auto</code> (default) — on a multi-protocol load with a
          detected per-protein cross-protocol scale mismatch, automatically
          applies <code>:both</code>; otherwise applies <code>:none</code>.
          Single-protocol loads always resolve to <code>:none</code>.</li>
      </ul>
      <p>
        Normalisation is applied BEFORE MNAR imputation — the imputation dropout
        curve is intensity-scale-sensitive, so size factors are computed on the
        pre-imputation data. Per-protein row-centering preserves enrichment /
        log2FC because the per-protein constant cancels in the
        <code>sample &minus; control</code> contrast (it is differential-neutral
        on the enrichment axis while removing the cross-protocol offset that
        otherwise inflates the dose-response evidence).
      </p>
      <p>
        Differential analyses additionally apply a per-condition
        <code>bait_anchor</code> correction (<code>DifferentialConfig.bait_anchor</code>,
        default <code>true</code>): a per-condition shift derived from the RAW
        bait abundance, applied to SAMPLE cells only (controls untouched) so
        that within-condition bait variation — the regression dose axis — is
        preserved. It corrects bait-abundance differences between conditions and
        is near-inert when bait levels are matched; set
        <code>bait_anchor = false</code> to disable it.
      </p>
    </section>
    """
end

"""
    _mask_aware_chip_html(pct) -> String

render the per-condition `pct_imputed_cells`
Bootstrap chip for the Data Quality tab. Local helper — `pct_imputed_cells`
is computed at the report-generator call site (or threaded as a kwarg) from
the `is_imputed` mask that `prepare_regression_data` already exposes. The
`AnalysisResult` struct is NOT modified.

Inputs:
- `pct::Float64` — single-condition path; renders one chip.
- `pct::AbstractDict{Symbol, <:Real}` — differential path; renders one chip
  per condition, in iteration order.
- `pct::Nothing` — returns the empty string (chip suppressed because the
  mask is genuinely unknown — distinct from the `0.00%` chip).

Empty data path: when `pct == 0.0`, the chip renders `0.00% imputed` rather
than being suppressed. Its presence is part of the v2b transparency contract
(the reader knows the regression saw raw, non-imputed data).
"""
function _mask_aware_chip_html(pct::Float64)::String
    return string(
        "<span class=\"badge text-bg-info\" data-bs-toggle=\"tooltip\" ",
        "title=\"Percentage of cells in the regression input that were imputed ",
        "(mask-aware regression v2b). 0% means raw, non-MNAR data.\">",
        round(pct, digits=2),
        "% imputed</span>",
    )
end

_mask_aware_chip_html(pct::Real)::String = _mask_aware_chip_html(Float64(pct))

function _mask_aware_chip_html(pct::AbstractDict{Symbol, <:Real})::String
    parts = String[]
    for (cond, p) in pct
        push!(parts, string(
            "<span class=\"badge text-bg-info me-1\" data-bs-toggle=\"tooltip\" ",
            "title=\"Condition '",
            String(cond),
            "': percentage of cells in the regression input that were imputed ",
            "(mask-aware regression v2b).\">",
            String(cond),
            ": ",
            round(Float64(p), digits=2),
            "% imputed</span>",
        ))
    end
    return join(parts, " ")
end

_mask_aware_chip_html(::Nothing)::String = ""

"""
    _methods_differential_block(diff::DifferentialResult) -> String

HTML fragment for the "Differential Analysis" subsection of the
Methods tab in `differential_report.html`. Adjacent to `_methods_variance_recovery_block`
per the established adjacent-helper convention.

Covers:
1. Δlog2FC definition + 6-class classification rules (GAINED / REDUCED / UNCHANGED /
   BOTH_NEGATIVE / CONDITION_A_SPECIFIC / CONDITION_B_SPECIFIC — verbatim enum value
   names from `src/differential/types.jl::InteractionClass`)
2. dBF formula `BF_A / BF_B` + per-component breakdown (`dbf_enrichment`,
   `dbf_correlation`, `dbf_detected`)
3. dbf_diagnostic legend — one paragraph per Symbol value (`:ok`, `:saturated`,
   `:single_component`, `:model_disagreement`)
4. BFDR semantics for differential (Storey monotone step-down on differential posteriors)
5. Explicit "differential is MS-only" note (mirrors the footer in tab-results)
6. Five <h6> subsections covering:
   - PEP general (α-PEP definition; distinction from BFDR and local_fdr)
   - α-PEP vs γ-PEP (class-conditional naive-product estimator)
   - Conditional-independence caveat
   - Calibration application gate (per-condition is_calibrated_A / is_calibrated_B)
   - bb_mnar_codriven rule + defaults
   - §7a marginal-KDE deviation note

XSS mitigation: every dynamic string (e.g., `diff.condition_A`,
`diff.condition_B`) routes through the module-level `esc(...)` helper
(`_html_esc`) to defang user-supplied condition labels.

No external placeholder URLs.
BMA terminology: "Copula" and "3c-EM" (locked v1.1.6).
FDR terminology: BFDR / PEP / local_fdr.
"""
function _methods_differential_block(diff::DifferentialResult)::String
    rows = String[]
    push!(rows, "<h5>Differential Analysis</h5>")

    # T-70-14-01: defang user-supplied condition labels via HTML escape.
    condA_esc = esc(string(diff.condition_A))
    condB_esc = esc(string(diff.condition_B))

    # 1. Δlog2FC + 6-class classification (verbatim enum value names)
    push!(rows, "<h6>Δlog2FC and Classification</h6>")
    push!(rows, "<p>For each shared protein, Δlog2FC is computed as " *
                "<code>log2FC(" * condA_esc * ") &minus; log2FC(" *
                condB_esc * ")</code>. Proteins are classified into one of " *
                "six mutually exclusive classes (verbatim enum value names from " *
                "<code>InteractionClass</code> in <code>src/differential/types.jl</code>):</p>")
    push!(rows, "<ul>")
    push!(rows, "<li><strong>GAINED</strong> — interaction newly enriched in " *
                condA_esc * " relative to " * condB_esc *
                " (Δlog2FC &gt; +threshold; differential posterior &gt; posterior_threshold).</li>")
    push!(rows, "<li><strong>REDUCED</strong> — interaction lost (or weakened) in " *
                condB_esc * " relative to " * condA_esc *
                " (Δlog2FC &lt; &minus;threshold; differential posterior &gt; posterior_threshold).</li>")
    push!(rows, "<li><strong>UNCHANGED</strong> — no significant Δlog2FC between " *
                "conditions (|Δlog2FC| &lt; threshold).</li>")
    push!(rows, "<li><strong>BOTH_NEGATIVE</strong> — neither condition shows enrichment " *
                "(BFs in both conditions below the bait-level cutoff).</li>")
    push!(rows, "<li><strong>CONDITION_A_SPECIFIC</strong> — detected/enriched only in " *
                condA_esc * " (the protein is absent from " *
                condB_esc * "&apos;s results after detection-based filtering).</li>")
    push!(rows, "<li><strong>CONDITION_B_SPECIFIC</strong> — detected/enriched only in " *
                condB_esc * " (the protein is absent from " *
                condA_esc * "&apos;s results after detection-based filtering).</li>")
    push!(rows, "</ul>")

    # 2. dBF formula + per-component breakdown
    push!(rows, "<h6>Differential Bayes Factor (dBF)</h6>")
    push!(rows, "<p>The differential Bayes factor compares evidence between conditions:</p>")
    push!(rows, "<p><code>dBF = BF(" * condA_esc * ") / BF(" *
                condB_esc * ")</code></p>")
    push!(rows, "<p>Per-component breakdown (computed from each condition&apos;s HBM, " *
                "regression, and Beta-Bernoulli sub-Bayes-factors):</p>")
    push!(rows, "<ul>")
    push!(rows, "<li><code>dbf_enrichment</code> — HBM (log2 fold change) evidence ratio " *
                "<code>BF_enrichment(A) / BF_enrichment(B)</code></li>")
    push!(rows, "<li><code>dbf_correlation</code> — dose-response regression evidence ratio " *
                "<code>BF_correlation(A) / BF_correlation(B)</code></li>")
    push!(rows, "<li><code>dbf_detected</code> — Beta-Bernoulli detection-probability " *
                "evidence ratio <code>BF_detected(A) / BF_detected(B)</code></li>")
    push!(rows, "</ul>")
    push!(rows, "<p>When <code>combination_method = :bma</code>, both <strong>Copula</strong> " *
                "and <strong>3c-EM</strong> sub-models contribute to each condition&apos;s BF " *
                "via LOO-stacking-weighted linear pooling; the columns " *
                "<code>bf_em_A</code>, <code>bf_copula_A</code>, <code>bf_em_B</code>, " *
                "<code>bf_copula_B</code> expose the sub-model BFs in the results table.</p>")

    # 3. dbf_diagnostic legend (4 Symbol values)
    push!(rows, "<h6>dBF Diagnostic Traffic-Light</h6>")
    push!(rows, "<p>The <code>dbf_diagnostic::Symbol</code> column flags reliability concerns " *
                "in each protein&apos;s dBF call:</p>")
    push!(rows, "<ul>")
    push!(rows, "<li><strong><code>:ok</code></strong> — none of the diagnostic conditions " *
                "below triggered; the dBF magnitude and rank are both reliable.</li>")
    push!(rows, "<li><strong><code>:saturated</code></strong> — at least one condition&apos;s " *
                "<code>|log10(BF)| &gt; 18</code>, near the [&minus;46, +46] log-BF clamp; " *
                "the dBF magnitude is artificially extreme. Treat the protein&apos;s " *
                "<em>rank</em> as more reliable than the absolute dBF value.</li>")
    push!(rows, "<li><strong><code>:single_component</code></strong> — one of the three " *
                "sub-evidence ratios (<code>dbf_enrichment</code>, <code>dbf_correlation</code>, " *
                "<code>dbf_detected</code>) drives more than 90% of the <code>log10(dBF)</code> " *
                "magnitude. The differential signal is not multi-evidence supported; inspect " *
                "the per-component breakdown before reporting.</li>")
    push!(rows, "<li><strong><code>:model_disagreement</code></strong> — the Copula and 3c-EM " *
                "sub-models give per-condition log-dBFs differing by more than one decade " *
                "(<code>|log10(dbf_em) &minus; log10(dbf_copula)| &gt; 1.0</code>). The dBF is " *
                "sensitive to BMA stacking weights; consult the per-condition Mixture Model " *
                "tab to diagnose which model dominates.</li>")
    push!(rows, "</ul>")
    push!(rows, "<p>The dedicated <strong>dBF Diagnostics</strong> tab shows the distribution " *
                "and per-protein detail for each diagnostic value.</p>")

    # 4. BFDR semantics
    push!(rows, "<h6>BFDR (Bayesian False Discovery Rate)</h6>")
    push!(rows, "<p>The differential <code>BFDR</code> column applies the Storey monotone " *
                "step-down procedure to the differential posterior probabilities, controlling " *
                "the expected fraction of false discoveries among proteins called significant " *
                "at a given threshold. <code>BFDR</code> is a <em>global</em> error rate; the " *
                "per-protein analogues <code>PEP = 1 &minus; differential_posterior</code> " *
                "(Posterior Error Probability) and <code>local_fdr</code> ship in the results " *
                "column set as well.</p>")

    # 5. MS-only note (mirrors the footer wording)
    push!(rows, "<h6>Differential Analysis is MS-Only</h6>")
    push!(rows, "<p>This differential analysis uses <strong>MS evidence only</strong>. " *
                "Docking-adjusted posteriors and structural-evidence two-stage Bayesian " *
                "updates (AlphaFold Server predictions) live in the per-condition single-bait " *
                "reports. See the footer note in the <strong>Results</strong> tab for navigation " *
                "to the per-condition reports for each of " * condA_esc * " and " *
                condB_esc * ".</p>")

    # ── (PEP α + γ definitions + caveats subsection) ──

    # 6a. PEP α + γ definitions
    push!(rows, "<h6>PEP — Posterior Error Probability</h6>")
    push!(rows, "<p><strong>PEP</strong> (per-protein) is the complement of the posterior " *
                "probability of a true interaction: <code>pep = 1 &minus; P(H1 | data)</code>. " *
                "Distinct from <strong>BFDR</strong> (Bayesian False Discovery Rate, " *
                "cumulative across rank-sorted hits; Storey monotone step-down) and " *
                "<strong>local_fdr</strong> (per-protein local FDR; mathematically identical " *
                "to PEP under the binary H0/H1 model). When the ECE-gate calibration is " *
                "applied (<code>cal_ece &lt; 0.10</code>), PEP uses the Platt-calibrated " *
                "posterior; otherwise it uses the raw posterior.</p>")

    push!(rows, "<h6>α-PEP vs γ-PEP (class-conditional)</h6>")
    push!(rows, "<p>The differential <code>differential_pep</code> column is the α variant: " *
                "<code>differential_pep = 1 &minus; differential_posterior</code>. The four " *
                "γ-class-conditional columns <code>pep_gained</code>, <code>pep_reduced</code>, " *
                "<code>pep_unchanged</code>, <code>pep_both_negative</code> are derived from " *
                "the per-condition posteriors and Δlog2FC via a normalized naive-product " *
                "estimator. Logistic gate steepness " *
                "<code>k = 10.0</code>; threshold <code>δ = config.delta_log2fc_threshold</code>.</p>")

    # 6b. Conditional-independence caveat
    push!(rows, "<p><strong>Conditional-independence caveat:</strong> The two condition " *
                "posteriors <code>p_A</code> and <code>p_B</code> are treated as conditionally " *
                "independent given the underlying interactor state. This approximation is " *
                "reasonable when conditions share replicates / protocols only at the " *
                "pre-processing stage; it underweights correlated posterior uncertainty " *
                "otherwise. A joint-posterior γ-PEP (bootstrap or copula-coupled) is " *
                "logged for v1.3.</p>")

    # 6c. Calibration application gate
    push!(rows, "<h6>Calibration application</h6>")
    isCalA = diff.is_calibrated_A
    isCalB = diff.is_calibrated_B
    push!(rows, "<p>Calibration is a dataset-level decision (per <code>AnalysisResult</code>) " *
                "gated by the Expected Calibration Error safety guard " *
                "(<code>cal_ece &lt; 0.10</code>). For this differential analysis: " *
                "<strong>" * condA_esc * "</strong> is " *
                (isCalA ? "calibrated" : "<em>not</em> calibrated") * "; " *
                "<strong>" * condB_esc * "</strong> is " *
                (isCalB ? "calibrated" : "<em>not</em> calibrated") * ".</p>")

    # 6d. bb_mnar_codriven rule
    push!(rows, "<h6>bb_mnar_codriven diagnostic flag</h6>")
    push!(rows, "<p>The <code>bb_mnar_codriven</code> flag is true when ALL THREE conditions " *
                "hold strictly: " *
                "<code>bf_detected &gt; cfg.bb_bf_threshold</code> AND " *
                "<code>BF &gt; cfg.hbm_bf_threshold</code> AND " *
                "<code>missing_fraction &gt; cfg.missing_fraction_threshold</code>. " *
                "Defaults: <code>10.0, 10.0, 0.5</code>. <code>BF</code> here is the post-MNAR " *
                "BMA combined Bayes factor (not the standalone HBM enrichment). The flag " *
                "warns that when Beta-Bernoulli detection and post-MNAR HBM evidence are " *
                "both strong AND most replicates were originally missing, the two pieces " *
                "of evidence may be reinforcing an MNAR-imputation artefact rather than " *
                "corroborating independent biological signal. The differential report " *
                "renders <code>bb_codriven_A</code> and <code>bb_codriven_B</code> as paired " *
                "warning icons per protein.</p>")

    # 6e. §7a marginal-KDE deviation note
    push!(rows, "<p><em>Volcano §7a note:</em> the differential volcano implements the hue " *
                "× saturation contract (y-axis = &minus;log&#x2081;&#x2080;(differential_pep); " *
                "hue = classification, saturation = 1 &minus; pep, linear, " *
                "clamped to [0.25, 1.0]). The marginal histograms are continuous kernel-" *
                "density estimates per classification class rather than PEP-bin stacked " *
                "histograms — the main scatter's per-marker opacity already encodes per-" *
                "protein confidence continuously, so the marginals answer a complementary " *
                "question (class distribution along each axis). This is a deliberate " *
                "interpretation of §7a's 'coloured by classification, stacked by PEP-bin' " *
                "wording. A runtime toggle to switch to PEP-bin stacked histograms is " *
                "logged for v1.3.</p>")

    # k-group omnibus + multi-condition
    # subsection. Rendered only when the differential is k>=2 (any NamedTuple call OR
    # legacy 2-group emits an empty contrasts vector, so this gate naturally hides the
    # block for legacy entries while exposing it for k-group calls). Zero external URLs
    # BMA names locked to "Copula" + "3c-EM"; FDR
    # names locked to BFDR / PEP / local_fdr.
    if length(diff.contrasts) >= 2
        push!(rows, "<h6>k-Group Omnibus + Multi-Condition</h6>")

        # A. Laplace omnibus formula + bullet list
        push!(rows, "<p>For k&thinsp;&ge;&thinsp;2 conditions the report computes a " *
                    "closed-form Gaussian omnibus Bayes factor over the per-condition " *
                    "posterior summaries (&mu;<sub>c</sub>, &sigma;<sup>2</sup><sub>c</sub>) " *
                    "extracted from each <code>AnalysisResult.results.mean_log2FC</code> and " *
                    "<code>sd_log2FC</code> column.</p>")
        push!(rows, "<ul>")
        push!(rows, "<li><strong>M<sub>0</sub> (null):</strong> all conditions share a " *
                    "single mean &mdash; a precision-weighted shared estimate " *
                    "&mu;&#x0302;<sub>shared</sub>.</li>")
        push!(rows, "<li><strong>M<sub>1</sub> (omnibus alternative):</strong> the conditions " *
                    "have heterogeneous means; the test statistic is " *
                    "S&thinsp;=&thinsp;&Sigma;<sub>c</sub>&thinsp;(&mu;<sub>c</sub>&thinsp;" *
                    "&minus;&thinsp;&mu;&#x0302;<sub>shared</sub>)<sup>2</sup>&thinsp;/&thinsp;" *
                    "&sigma;<sup>2</sup><sub>c</sub>, with log&thinsp;BF&thinsp;=&thinsp;S&thinsp;/&thinsp;2 " *
                    "(Kass &amp; Raftery 1995 &sect;3.2; Wagenmakers 2007 &sect;2 on closed-form " *
                    "BF derivations).</li>")
        push!(rows, "</ul>")
        push!(rows, "<p>The five new columns emitted on <code>diff.results</code> are " *
                    "<code>bf_omnibus</code>, <code>log10_bf_omnibus</code>, " *
                    "<code>posterior_omnibus</code>, <code>differential_BFDR_omnibus</code>, " *
                    "and <code>differential_pep_omnibus</code>.</p>")

        # B. EB pooled prior
        push!(rows, "<p><em>Empirical-Bayes pooled prior:</em> a robust " *
                    "(&mu;<sub>pool</sub>, &tau;<sup>2</sup><sub>pool</sub>) tuple is " *
                    "computed once per <code>differential_analysis(; conditions, &hellip;)</code> " *
                    "call as &mu;<sub>pool</sub>&thinsp;=&thinsp;median(finite&thinsp;&mu;) and " *
                    "&tau;<sup>2</sup><sub>pool</sub>&thinsp;=&thinsp;max((1.4826&thinsp;&times;&thinsp;MAD)<sup>2</sup>,&thinsp;0.01). " *
                    "The prior is exposed on the omnibus helper signature for downstream " *
                    "consumers (Methods documentation; report banner) but does not currently " *
                    "enter the test statistic itself &mdash; the simplified heterogeneity-LR " *
                    "form above satisfies the locked BF&thinsp;=&thinsp;1 (identical means) and " *
                    "BF&thinsp;&rarr;&thinsp;&infin; (4&sigma; shift) sanity contracts.</p>")

        # D. Multi-condition view thresholds
        push!(rows, "<h6>Multi-Condition View Thresholds</h6>")
        push!(rows, "<p>The Multi-Condition tab is rendered when " *
                    "<code>length(diff.contrasts)&thinsp;&ge;&thinsp;2</code> AND " *
                    "<code>length(condition_labels(diff))&thinsp;&ge;&thinsp;3</code>. For " *
                    "k&thinsp;=&thinsp;3 (three contrasts) the small-multiples grid renders " *
                    "inline. For k&thinsp;&ge;&thinsp;4 (six or more contrasts under the " *
                    "default <code>:all_pairs</code> heuristic) a dropdown widget switches " *
                    "between the matrix view (k&thinsp;&times;&thinsp;k posterior-median heatmap " *
                    "alongside the small-multiples grid) and the per-pair detail view " *
                    "(enlarged volcano with a pair selector).</p>")

        # E. Generalised classification + 5-class enum
        push!(rows, "<h6>Generalised Classification (<code>kgroup_class</code>)</h6>")
        push!(rows, "<p>For each protein, <code>enriched_in::Vector{Symbol}</code> and " *
                    "<code>depleted_in::Vector{Symbol}</code> record the conditions for which " *
                    "the per-condition posterior crosses the configured " *
                    "<code>posterior_threshold</code> (default 0.8) or its complement " *
                    "(1&thinsp;&minus;&thinsp;<code>posterior_threshold</code>, default 0.2) " *
                    "respectively. The <code>kgroup_class::Symbol</code> column collapses the " *
                    "two vectors plus the omnibus BFDR into a 5-class enum:</p>")
        push!(rows, "<ul>")
        push!(rows, "<li><code>:omnibus_null</code> &mdash; omnibus BFDR exceeds " *
                    "<code>bfdr_threshold</code>; insufficient evidence for any " *
                    "heterogeneity.</li>")
        push!(rows, "<li><code>:none_enriched</code> &mdash; omnibus is significant but no " *
                    "single condition crosses the posterior threshold.</li>")
        push!(rows, "<li><code>:condition_specific</code> &mdash; enriched in a strict subset " *
                    "of the conditions.</li>")
        push!(rows, "<li><code>:all_enriched</code> &mdash; enriched in every condition.</li>")
        push!(rows, "<li><code>:fully_resolved</code> &mdash; every condition is either " *
                    "enriched or depleted; no abstentions.</li>")
        push!(rows, "</ul>")
        push!(rows, "<p>The string-typed <code>classification_summary</code> column " *
                    "remains alongside the new structured columns for backward compatibility.</p>")

        # F. Dual FDR families
        push!(rows, "<h6>Dual FDR Families</h6>")
        push!(rows, "<p>Two BFDR families coexist on the differential result and answer " *
                    "different scientific questions:</p>")
        push!(rows, "<ul>")
        push!(rows, "<li><code>differential_BFDR_pairwise_BH</code> &mdash; cross-pair " *
                    "Benjamini&ndash;Hochberg adjustment applied across the " *
                    "(n<sub>proteins</sub>&thinsp;&times;&thinsp;n<sub>contrasts</sub>) family " *
                    "of per-pair <code>differential_BFDR</code> values.</li>")
        push!(rows, "<li><code>differential_BFDR_omnibus</code> &mdash; within-omnibus " *
                    "Storey monotone step-down on the <code>posterior_omnibus</code> column.</li>")
        push!(rows, "</ul>")
        push!(rows, "<p>The two families are NOT cross-corrected &mdash; they answer " *
                    "different scientific questions and should be filtered independently. " *
                    "<code>PEP</code> (per-protein) and <code>local_fdr</code> keep their existing semantics.</p>")

        # G. BMA terminology reminder
        push!(rows, "<p><em>BMA sub-model note:</em> when " *
                    "<code>combination_method&thinsp;=&thinsp;:bma</code>, both the " *
                    "<strong>Copula</strong> and <strong>3c-EM</strong> sub-models contribute " *
                    "via LOO-stacking-weighted linear pooling on the BF scale. " *
                    "The omnibus statistic operates on the per-condition posterior " *
                    "summaries and is therefore independent of the BMA sub-model choice.</p>")
    end

    # append Decision Risk subsection.
    # Authoritative in-report reference until a docs page lands.
    # The worked-example numbers are byte-equal to
    # what `compute_decision_risk` produces for the fixture γ-PEPs; a
    # spec-drift testitem in `test/reports/test_decision_risk_report.jl` pins
    # the rendered values via `string(round(x; digits=4))` (Julia strips
    # trailing zeros, so 3.870967741935484 renders as "3.871" — pinned form).
    push!(rows, _methods_decision_risk_block())

    return join(rows, "\n")
end

"""
    _methods_decision_risk_block() -> String

HTML fragment for the "Decision Risk" subsection of
the Methods tab in `differential_report.html`. Adjacent helper to
`_methods_differential_block` per the established adjacent-helper
convention.

Covers:
1. Per-cell loss-matrix justification table (verbatim rationale).
2. TWO worked examples back-to-back; arithmetic is
   pulled from the implemented `compute_decision_risk` helper and rendered via
   `string(round(x; digits=4))`. Note: Julia's `string(Float64)` strips
   trailing zeros, so `string(round(3.870967741935484; digits=4)) == "3.871"`
   (NOT the four-decimal trailing-zero form). The spec-drift testitem asserts
   this single pinned form.
3. Override instructions (Julia code snippet showing
   `DifferentialConfig(; loss_matrix = my_matrix)` and the per-call kwarg).
4. Coverage subsection documenting `CONDITION_A/B_SPECIFIC` → NaN exclusion.
5. Dual FDR family disclosure (omnibus pre-filter for Validation Candidates vs
   pairwise BH for cross-pair correction — the dual-FDR lock).
6. Top-N configurability (`validation_candidates_top_n` field).
7. k-group output description (`decision_risk_min` + `optimal_call_min` columns).

The `<h4>` "Decision Risk" subheading carries the tooltip text wired via
Bootstrap's `data-bs-toggle="tooltip"` attribute (this is the Methods
subheading tooltip; column-header tooltips are wired into the template
separately).

No external placeholder URLs; until a docs page lands,
the Methods tab is the authoritative in-report reference.
"""
function _methods_decision_risk_block()::String
    return """
    <section class="methods-subsection mb-4">
      <h4 data-bs-toggle="tooltip" title="The N × N matrix L[a, k] giving the cost of action a when truth = k. Customise via DifferentialConfig.loss_matrix or the loss_matrix= kwarg to differential_analysis. Default values: see table above.">
        Decision Risk
      </h4>

      <p>
        For each protein × pairwise contrast we compute the Bayes-optimal call
        minimising the posterior expected loss under a user-overrideable 4×4
        loss matrix. The default matrix encodes an asymmetric cost structure
        where reversing a direction (calling <code>gained</code> when truth is
        <code>reduced</code> or vice versa) is twice as costly as missing a
        real hit — see the per-cell justification below.
      </p>

      <h5>Default loss matrix <code>DEFAULT_DIFFERENTIAL_LOSS</code></h5>
      <table class="table table-sm table-bordered" style="width: auto;">
        <thead>
          <tr>
            <th>action ↓ / truth →</th>
            <th>gained</th><th>reduced</th><th>unchanged</th><th>both_negative</th>
          </tr>
        </thead>
        <tbody>
          <tr><th>gained</th>
              <td>0</td><td>10</td><td>3</td><td>3</td></tr>
          <tr><th>reduced</th>
              <td>10</td><td>0</td><td>3</td><td>3</td></tr>
          <tr><th>unchanged</th>
              <td>5</td><td>5</td><td>0</td><td>1</td></tr>
          <tr><th>both_negative</th>
              <td>5</td><td>5</td><td>1</td><td>0</td></tr>
        </tbody>
      </table>

      <h6>Per-cell justification</h6>
      <ul>
        <li><strong>Diagonal = 0:</strong> a correct call costs nothing.</li>
        <li><strong>Direction-flip = 10 (gained ↔ reduced):</strong>
          the most expensive error — it sends the wet-lab investigator in
          exactly the wrong direction on a strong signal.
          <em>10 was chosen as 2× missed-hit because reversing a published
          direction is more costly than missing one entirely.</em></li>
        <li><strong>Over-claim = 3 (calling enriched when truth is unchanged):</strong>
          wastes follow-up resources but doesn&apos;t actively mislead.
          <em>3 was chosen as conservatively low to reflect that false-positive
          enrichment claims are detected quickly in confirmatory MS runs.</em></li>
        <li><strong>Missed-hit = 5 (calling unchanged/both_negative when truth is enriched):</strong>
          permanently buries a real signal.
          <em>5 was chosen above over-claim (3) because missed hits are detected
          only via independent re-analysis, whereas over-claims surface in
          routine follow-up.</em></li>
        <li><strong>Conservative-default = 1 (unchanged ↔ both_negative):</strong>
          within-quadrant slip in the no-interaction region.
          <em>1 was chosen as the floor of the spec range [1, 2] to reflect
          that these calls are functionally interchangeable in the
          no-interaction quadrant.</em></li>
      </ul>

      <h5>Posterior input</h5>
      <p>
        The risk integral is computed against the renormalised γ-PEP posterior:
        <code>P(state = k) = (1 − pep_k) / Σ_j (1 − pep_j)</code>
        for <code>k, j ∈ {gained, reduced, unchanged, both_negative}</code>.
        When <code>Σ_j (1 − pep_j) &lt; 1e-12</code> the posterior degenerates
        to a uniform <code>0.25</code> over the four states and a one-shot
        <code>@warn</code> is emitted (subsequent rows in the same analysis
        are silenced via <code>maxlog=1</code>).
      </p>

      <h5>Worked examples</h5>

      <h6>Example 1: MAP equals Optimal (no badge)</h6>
      <p>Protein X (synthetic) carries γ-PEP posteriors
        <code>[pep_gained=0.05, pep_reduced=0.60, pep_unchanged=0.85, pep_both_negative=0.95]</code>.
      </p>
      <p>
        Σ(1 − pep) = 1.55, giving normalised posterior
        <code>P = [0.6129, 0.2581, 0.0968, 0.0323]</code>.
      </p>
      <p>Expected loss of each action:</p>
      <ul>
        <li><code>risk_gained        = 0 · 0.6129 + 10 · 0.2581 + 3 · 0.0968 + 3 · 0.0323 = 2.9677</code></li>
        <li><code>risk_reduced       = 10 · 0.6129 + 0 · 0.2581 + 3 · 0.0968 + 3 · 0.0323 = 6.5161</code></li>
        <li><code>risk_unchanged     = 5 · 0.6129 + 5 · 0.2581 + 0 · 0.0968 + 1 · 0.0323 = 4.3871</code></li>
        <li><code>risk_both_negative = 5 · 0.6129 + 5 · 0.2581 + 1 · 0.0968 + 0 · 0.0323 = 4.4516</code></li>
      </ul>
      <p>
        <strong>Optimal call:</strong> <code>:gained</code> (lowest risk = 2.9677).
        <strong>MAP:</strong> <code>:gained</code> (argmin γ-PEP).
        MAP == Optimal → no recommended-call badge fires.
      </p>

      <h6>Example 2: MAP differs from Optimal (badge fires)</h6>
      <p>Protein Y (synthetic) carries γ-PEP posteriors
        <code>[pep_gained=0.40, pep_reduced=0.55, pep_unchanged=0.60, pep_both_negative=0.90]</code>.
      </p>
      <p>
        Σ(1 − pep) = 1.55, giving normalised posterior
        <code>P = [0.3871, 0.2903, 0.2581, 0.0645]</code>.
      </p>
      <p>Expected loss of each action (note: <code>risk_gained</code> rounds to
        <code>3.871</code> — Julia&apos;s <code>string(round(x; digits=4))</code>
        strips trailing zeros, so the rendered value has three decimals here):</p>
      <ul>
        <li><code>risk_gained        = 10 · 0.2903 + 3 · 0.2581 + 3 · 0.0645 = 3.871</code></li>
        <li><code>risk_reduced       = 10 · 0.3871 + 3 · 0.2581 + 3 · 0.0645 = 4.8387</code></li>
        <li><code>risk_unchanged     = 5 · 0.3871 + 5 · 0.2903 + 1 · 0.0645 = 3.4516</code></li>
        <li><code>risk_both_negative = 5 · 0.3871 + 5 · 0.2903 + 1 · 0.2581 = 3.6452</code></li>
      </ul>
      <p>
        <strong>Optimal call:</strong> <code>:unchanged</code> (lowest risk = 3.4516).
        <strong>MAP:</strong> <code>:gained</code> (argmin γ-PEP = γ-PEP for
        <code>gained</code> is smallest, equivalently P(gained) is largest).
        MAP ≠ Optimal → recommended-call badge fires. The asymmetric loss
        matrix penalises the direction-flip risk (10 for
        <code>gained ↔ reduced</code>) enough that recommending the
        conservative <code>unchanged</code> call costs less in expectation
        than committing to <code>gained</code> when P(gained) leads only
        narrowly over P(reduced).
      </p>

      <h5>Override instructions</h5>
      <p>The default 4×4 matrix is a published baseline. Override per-analysis via either:</p>
      <pre><code>using BayesInteractomics

# Option A: bake the override into the config
cfg = DifferentialConfig(
    loss_matrix = Float64[
        0   5  2  2;   # less penalty on direction-flip
        5   0  2  2;
        3   3  0  1;
        3   3  1  0;
    ],
)
diff = differential_analysis(ar_wt, ar_mut; config = cfg)

# Option B: per-call kwarg override (shadows the config field for this call only)
diff = differential_analysis(ar_wt, ar_mut; loss_matrix = my_matrix)
</code></pre>
      <p>
        The <code>loss_matrix_default::Bool</code> column on
        <code>diff.results</code> flips to <code>false</code> for every row
        when the active matrix differs element-wise from
        <code>DEFAULT_DIFFERENTIAL_LOSS</code> — this is the per-row
        provenance flag.
      </p>

      <h5>Coverage</h5>
      <p>
        Proteins classified as <code>CONDITION_A_SPECIFIC</code> or
        <code>CONDITION_B_SPECIFIC</code> (detected only in one of the two
        conditions of a pair) fall outside the 4-action / 4-truth-state loss
        matrix. For these rows the helper emits:
      </p>
      <ul>
        <li><code>optimal_call = :condition_a_specific</code> (or
            <code>:condition_b_specific</code>) — preserving the MAP
            <code>InteractionClass</code> as a <code>Symbol</code>.</li>
        <li><code>decision_risk = NaN</code> and the four <code>risk_&lt;class&gt;</code>
            columns set to <code>NaN</code>. <code>NaN</code> rather than
            <code>missing</code> keeps the column <code>eltype</code> simple
            (<code>Float64</code>) for downstream tools.</li>
        <li>Validation Candidates pre-filter skips these rows (NaN sorts to
            the end ascending).</li>
      </ul>
      <p>
        Native subset-action support (an enriched-in / depleted-in subset for
        each protein under arbitrary <code>2^k − 1</code> subsets — DRISK-K)
        is deferred to v1.3.0.
      </p>

      <h5>Dual FDR families</h5>
      <p>
        For k-group analyses, the Validation Candidates pre-filter applies
        <strong><code>differential_BFDR_omnibus &lt;= config.bfdr_threshold</code></strong>
        (the &quot;any-difference across all conditions&quot; family).
        The cross-pair <code>differential_BFDR_pairwise_BH</code> family is
        independent and is NOT used as a Validation Candidates pre-filter
        because it controls FDR at the contrast-pair level, not the
        any-difference level. The two families are not cross-corrected; each
        answers a different scientific question. See the
        omnibus + multi-condition subsections above for the dual-FDR
        contract.
      </p>

      <h5>Top-N configurability</h5>
      <p>
        The Validation Candidates view defaults to top-20 ranked rows. Override via:
      </p>
      <pre><code>cfg = DifferentialConfig(; validation_candidates_top_n = 50)
diff = differential_analysis(ar_wt, ar_mut; config = cfg)
</code></pre>

      <h5>k-group output (k ≥ 3)</h5>
      <p>
        For k ≥ 3 conditions the helper runs once per pairwise contrast — the
        per-pair Decision Risk lives inside
        <code>diff.pairwise_results[pair]</code>. The wide table
        <code>diff.results</code> carries the aggregated
        <code>decision_risk_min::Float64</code> column (per-protein minimum
        across pairs) plus <code>optimal_call_min::Symbol</code> (the call
        achieving the min). The Validation Candidates ranking and the
        Multi-Condition Decision Risk heatmap both use
        <code>decision_risk_min</code> as the sort key.
      </p>
    </section>
    """
end

"""
    _methods_embeddings_block(config_or_emb) -> String

HTML fragment for the Methods tab "Embeddings & Similarity" subsection.

Accepts EITHER a `CONFIG` (will read `config.embeddings_config`) OR an
`EmbeddingsConfig` directly (so a test can pass `EmbeddingsConfig()`
without needing to construct a full CONFIG with a real datafile).

Returns `""` when the resolved `EmbeddingsConfig.run_embeddings === false`. Otherwise
emits three `<h6>` subsections (Sample, Protein, Condition) with rationale.

Citations are inline text only (no external URLs / no `<a href>`).
"""
function _methods_embeddings_block(config_or_emb)::String
    # Resolve to an EmbeddingsConfig regardless of which type the caller passed.
    cfg = if config_or_emb isa EmbeddingsConfig
        config_or_emb
    elseif hasproperty(config_or_emb, :embeddings_config)
        config_or_emb.embeddings_config
    else
        return ""
    end
    !cfg.run_embeddings && return ""
    method_str = string(cfg.method)

    rows = String[]
    push!(rows, "<h5>Embeddings &amp; Similarity</h5>")

    push!(rows, "<h6>Sample similarity (PCA + UMAP/t-SNE)</h6>")
    push!(rows, "<p>PCA on the post-imputation log-intensity matrix surfaces the dominant axes of " *
                "replicate-to-replicate variance. UMAP (default) preserves local neighbourhood " *
                "structure while resolving global clusters; t-SNE (opt-in via " *
                "<code>CONFIG.embeddings_config.method = :tsne</code>) preserves local structure only " *
                "and should not be interpreted globally &mdash; distances between well-separated " *
                "clusters in t-SNE are not meaningful. Both non-linear methods are seeded with " *
                "<code>CONFIG.embeddings_config.seed</code> (default 42) for reproducibility. " *
                "(McInnes et al. 2018; van der Maaten 2014.)</p>")
    push!(rows, "<p>When multiple imputation is enabled, the sample-level PCA / UMAP runs on the " *
                "pooled-mean post-imputation log-intensity matrix (element-wise mean across the " *
                "M imputations per cell); the cascading 100/80/50&percnt; non-missing filter is " *
                "bypassed because the pooled matrix is dense by construction. Without imputation " *
                "the embedding consumes the raw matrix via the cascade. The pooled-mean view loses " *
                "between-imputation variance by design; multi-imputation-aware uncertainty " *
                "visualisations are deferred to a later release.</p>")

    push!(rows, "<h6>Protein similarity (UMAP on posterior feature vector)</h6>")
    push!(rows, "<p>For each protein, a 5-dimensional feature vector is constructed from " *
                "<code>[log10(bf_enrichment), log10(bf_correlation), log10(bf_detected), " *
                "posterior_prob, log2FC_mean]</code> with the numerical safety clamp " *
                "<code>log10(max(bf, 1e-12))</code>. Features are z-scored before UMAP. " *
                "Points are coloured by classification (H0 / Agnostic / H1 for single-bait, or the " *
                "4-class differential enum: GAINED / REDUCED / UNCHANGED / BOTH_NEGATIVE).</p>")

    push!(rows, "<h6>Condition similarity (multi-condition only)</h6>")
    push!(rows, "<p>The primary metric is Spearman rank correlation on " *
                "<code>log10(BF)</code> across the pairwise intersection of detected proteins. " *
                "Secondary togglable views: Pearson on log2FC, Pearson on posterior_prob. " *
                "<strong>Jaccard@Top-$(cfg.top_k_jaccard)</strong> surfaces set overlap among the " *
                "strongest posterior_prob hits per condition. The k&times;k matrix is clustered " *
                "with average linkage on D = 1 &minus; &rho; (UPGMA); the dendrogram leaf order " *
                "drives the heatmap row/column order so visual grouping is preserved. Diagonal " *
                "cells are rendered in light grey to mark the trivial &rho;[i,i] = 1.0 anchor.</p>")

    push!(rows, "<table class=\"table table-sm\">")
    push!(rows, "<tr><th>Method</th><td><code>$(method_str)</code></td></tr>")
    push!(rows, "<tr><th>Seed</th><td>$(cfg.seed)</td></tr>")
    push!(rows, "<tr><th>n_neighbors</th><td>$(cfg.n_neighbors)</td></tr>")
    push!(rows, "<tr><th>min_dist</th><td>$(cfg.min_dist)</td></tr>")
    push!(rows, "<tr><th>top_k_jaccard</th><td>$(cfg.top_k_jaccard)</td></tr>")
    push!(rows, "<tr><th>supervised</th><td>$(cfg.supervised) <em>(no-op in v1.2.0; " *
                "UMAP.jl 0.1.x has no y= kwarg)</em></td></tr>")
    push!(rows, "</table>")
    return join(rows, "\n")
end

"""
    _build_structured_methods_data(config::CONFIG, results::DataFrame, metalearner_status::Symbol=:loaded) -> String

Build a structured JSON string with per-model data objects and a consolidated
priors array.  Consumed by the HTML report template (Methods tab).

Top-level keys: overview, detection, enrichment, correlation, combination,
calibration, priors, metalearner_status, metalearner_status_html.

`metalearner_status` (Symbol — `:loaded`,
`:extension_not_loaded`, `:prediction_failed`) is rendered into a
`metalearner_status_html` subsection describing Variante B fallback semantics.
"""
function _build_structured_methods_data(config::CONFIG, results::DataFrame,
                                        metalearner_status::Symbol = :loaded)::String
    # --- overview ---
    n_sig   = count(x -> coalesce(x <= 0.05, false), results.BFDR)
    n_strong = count(x -> coalesce(x <= 0.01, false), results.BFDR)

    overview = json_object(
        "n_proteins"         => json_number(nrow(results)),
        "n_sig"              => json_number(n_sig),
        "n_strong"           => json_number(n_strong),
        "pkg_version"        => json_string(_report_pkg_version()),
        "julia_version"      => json_string(string(VERSION)),
        "bait"               => json_string(config.poi),
        "n_controls"         => json_number(config.n_controls),
        "n_samples"          => json_number(config.n_samples),
        "combination_method" => json_string(string(config.combination_method)),
    )

    # --- detection ---
    detection = json_object(
        "prior_alpha" => json_number(3.0),
        "prior_beta"  => json_number(3.0),
    )

    # --- enrichment ---
    enrichment = json_object(
        "mu_0"    => json_number(25.0),
        "sigma_0" => json_number(1.0),
        "a_0"     => json_number(1.0),
        "b_0"     => json_number(1.0),
    )

    # --- correlation ---
    slope_prior_var = round((0.3 / 1.96)^2, sigdigits=5)
    correlation = json_object(
        "regression_likelihood" => json_string(string(config.regression_likelihood)),
        "student_t_nu"          => json_number(config.student_t_nu),
        "jzs_r_scale"           => json_number(config.jzs_r_scale),
        "bf_threshold"          => json_number(config.regression_bf_threshold),
        "slope_prior_var"       => json_number(slope_prior_var),
        "jzs_cauchy_scale"      => json_number(config.jzs_r_scale),
    )

    # --- combination ---
    lc_wq = "($(config.lc_winsorize_quantiles[1]), $(config.lc_winsorize_quantiles[2]))"
    combination = json_object(
        "method"                  => json_string(string(config.combination_method)),
        "em_n_restarts"           => json_number(config.em_n_restarts),
        "copula_criterion"        => json_string(string(config.copula_criterion)),
        "lc_alpha_prior"          => if config.lc_alpha_prior === :auto
            json_string("auto")
        else
            json_array([json_number(x) for x in config.lc_alpha_prior])
        end,
        "lc_winsorize_quantiles"  => json_string(lc_wq),
        "bma_weight_floor"        => json_number(0.05),
    )

    # --- calibration (null when simulation not run) ---
    calibration = if config.run_simulation
        json_object(
            "run_simulation"  => json_bool(config.run_simulation),
            "sim_n_synthetic" => json_number(config.sim_n_synthetic),
        )
    else
        "null"
    end

    # --- priors array ---
    slope_var_str = string(round((0.3 / 1.96)^2, sigdigits=4))
    prior_rows = String[]

    # Row 1: detection
    push!(prior_rows, json_object(
        "key"          => json_string("detection_theta"),
        "model"        => json_string("Detection (Beta-Bernoulli)"),
        "parameter"    => json_string("theta (detection rate)"),
        "distribution" => json_string("Beta(alpha, beta)"),
        "values"       => json_string("alpha = 3.0, beta = 3.0"),
    ))

    # Row 2: enrichment mu
    push!(prior_rows, json_object(
        "key"          => json_string("enrichment_mu"),
        "model"        => json_string("Enrichment (HBM)"),
        "parameter"    => json_string("mu (intensity mean)"),
        "distribution" => json_string("Normal(mu, sigma)"),
        "values"       => json_string("mu = 25.0, sigma = 1.0"),
    ))

    # Row 3: enrichment sigma (tau)
    push!(prior_rows, json_object(
        "key"          => json_string("enrichment_sigma"),
        "model"        => json_string("Enrichment (HBM)"),
        "parameter"    => json_string("tau (intensity precision)"),
        "distribution" => json_string("Gamma(shape, scale)"),
        "values"       => json_string("shape = 1.0, scale = 1.0"),
    ))

    # Row 4: correlation slope
    push!(prior_rows, json_object(
        "key"          => json_string("correlation_slope"),
        "model"        => json_string("Correlation (Regression)"),
        "parameter"    => json_string("alpha (slope)"),
        "distribution" => json_string("Normal(0, sigma^2)"),
        "values"       => json_string("sigma^2 = $slope_var_str"),
    ))

    # Row 5: JZS (conditional)
    if config.jzs_r_scale > 0
        push!(prior_rows, json_object(
            "key"          => json_string("correlation_slope_jzs"),
            "model"        => json_string("Correlation (Regression)"),
            "parameter"    => json_string("alpha (slope, JZS)"),
            "distribution" => json_string("Cauchy(0, r)"),
            "values"       => json_string("r = $(config.jzs_r_scale)"),
        ))
    end

    # Row 6: robust nu (conditional)
    if config.regression_likelihood == :robust_t
        push!(prior_rows, json_object(
            "key"          => json_string("correlation_robust_nu"),
            "model"        => json_string("Correlation (Regression)"),
            "parameter"    => json_string("nu (degrees of freedom)"),
            "distribution" => json_string("Fixed"),
            "values"       => json_string("nu = $(config.student_t_nu)"),
        ))
    end

    # Row 7: combination EM prior
    alpha_str = if config.lc_alpha_prior === :auto
        "auto (Empirical Bayes)"
    else
        join(string.(config.lc_alpha_prior), ", ")
    end
    push!(prior_rows, json_object(
        "key"          => json_string("combination_em_prior"),
        "model"        => json_string("Evidence Combination"),
        "parameter"    => json_string("pi (mixture weights)"),
        "distribution" => json_string("Dirichlet-like"),
        "values"       => json_string("alpha = [$alpha_str]"),
    ))

    # Row 8: BMA weights
    push!(prior_rows, json_object(
        "key"          => json_string("combination_bma_weights"),
        "model"        => json_string("Evidence Combination (BMA)"),
        "parameter"    => json_string("w (stacking weights)"),
        "distribution" => json_string("LOO stacking (Yao et al. 2018)"),
        "values"       => json_string("floor = 0.05"),
    ))

    priors = json_array(prior_rows)

    # Metalearner Status subsection HTML + status string.
    metalearner_status_html = _methods_metalearner_status(metalearner_status)

    # DNN Prior + MC-Dropout subsection HTML.
    # Adjacent to the metalearner status block — same Methods-tab card grouping
    # (model-level transparency / calibration responsibility). 2-branch dispatch
    # on `config.run_dnn_prior_mc_dropout`; opt-out branch renders the
    # "disabled" message + the five new column names.
    dnn_prior_block_html = _methods_dnn_prior_block(config)

    return json_object(
        "overview"    => overview,
        "detection"   => detection,
        "enrichment"  => enrichment,
        "correlation" => correlation,
        "combination" => combination,
        "calibration" => calibration,
        "priors"      => priors,

        "metalearner_status"      => json_string(string(metalearner_status)),
        "metalearner_status_html" => json_string(metalearner_status_html),
        # DNN Prior + MC-Dropout subsection
        # HTML. The template injects this into the Methods tab immediately after
        # the metalearner-status card via the dnn-prior-block card div.
        "dnn_prior_block_html"    => json_string(dnn_prior_block_html),
    )
end

# ────────────────────────────────────────────────────────────────────────────
# Methods cascade fallback helpers.
#
# When the per-condition `AnalysisResult` carries no in-memory `CONFIG`
# (e.g. it was rehydrated from JLD2 via `load_result`, or constructed from a
# path-only fixture), the differential report's Methods tab still needs to
# render *something* per condition. These helpers implement levels 2-4 of
# the Methods cascade:
#
#     Level 1 (existing) : ar.config !== nothing → _build_methods_json(ar.config, ...)
#     Level 2 (new)      : sidecar JSON at <basedir>/interactive_report_data.json → data.methods.text
#     Level 3 (new)      : raw markdown at <basedir>/methods.md → wrapped in card-body.markdown-body
#     Level 4 (new)      : placeholder card + single @warn per condition (maxlog=1 discipline)
#
# Each helper returns `nothing` when its source is unavailable, so the cascade
# in `_build_diff_methods_json` is expressed as cascading `if`/`elseif` calls.
# ────────────────────────────────────────────────────────────────────────────

"""
    _basedir_for_ar(ar) -> Union{String, Nothing}

helper. Best-effort extraction of the
per-condition output base directory. Tries (in order):

1. `ar.output.basedir` — duck-typed accessor used by test fixtures that wrap a
   bare `OutputFiles` on the AR-shaped object.
2. `ar.config.output.basedir` — the canonical production path when the AR has
   a fully-populated `CONFIG`.

Returns `nothing` when neither path resolves. Defensive against missing fields
and `nothing` slots.
"""
function _basedir_for_ar(ar)::Union{String, Nothing}
    # Path 1: duck-typed `ar.output.basedir` (test fixtures, future AR variants).
    if hasproperty(ar, :output)
        out = getproperty(ar, :output)
        if out !== nothing && hasproperty(out, :basedir)
            bd = getproperty(out, :basedir)
            if bd isa AbstractString && !isempty(bd)
                return String(bd)
            end
        end
    end
    # Path 2: `ar.config.output.basedir` — production AR with a CONFIG attached.
    if hasproperty(ar, :config)
        cfg = getproperty(ar, :config)
        if cfg !== nothing && hasproperty(cfg, :output)
            out = getproperty(cfg, :output)
            if out !== nothing && hasproperty(out, :basedir)
                bd = getproperty(out, :basedir)
                if bd isa AbstractString && !isempty(bd)
                    return String(bd)
                end
            end
        end
    end
    return nothing
end

"""
    _try_locate_sidecar(ar) -> Union{String, Nothing}

helper. Return the absolute path
to `<basedir>/interactive_report_data.json` when the file exists; otherwise
`nothing`. Used as the cascade-level-2 source by `_build_diff_methods_json`.
"""
function _try_locate_sidecar(ar)::Union{String, Nothing}
    bd = _basedir_for_ar(ar)
    bd === nothing && return nothing
    path = joinpath(bd, "interactive_report_data.json")
    return isfile(path) ? path : nothing
end

"""
    _try_locate_methods_md(ar) -> Union{String, Nothing}

helper. Return the absolute path
to `<basedir>/methods.md` when the file exists; otherwise `nothing`. Used as
the cascade-level-3 source by `_build_diff_methods_json`.
"""
function _try_locate_methods_md(ar)::Union{String, Nothing}
    bd = _basedir_for_ar(ar)
    bd === nothing && return nothing
    path = joinpath(bd, "methods.md")
    return isfile(path) ? path : nothing
end

"""
    _read_methods_from_sidecar_json(path::AbstractString) -> Union{String, Nothing}

helper. Read a per-condition sidecar
`interactive_report_data.json` and return its `methods.text` payload as a plain
string (the same prose `generate_methods_text` produces). The sidecar key
structure was sampled from `/home/mseefelder/Schreibtisch/HTT_meta/wtHTT/interactive_report_data.json`
and matches the writer in `_build_methods_json` (top-level `methods` key with
sub-keys `text`, `reproducibility`, `parameters`, `structured`, ...).

Returns `nothing` on JSON parse error or when the `methods.text` key is missing
(emits a single `@warn` with `maxlog=1` on parse error so the placeholder
branch can fire cleanly).

Uses `JSON3.read` (already imported into the `Reports` submodule), per
RESEARCH.md §11 row "Methods sidecar JSON read" — do NOT hand-roll a parser.
"""
function _read_methods_from_sidecar_json(path::AbstractString)::Union{String, Nothing}
    try
        parsed = JSON3.read(Base.read(path, String))
        # Sidecar uses top-level `methods` key (NOT `data.methods`); verified
        # against the wtHTT/mHTT/HAP40_Strep/GST_HAP40 evidence-base samples.
        methods_obj = if haskey(parsed, :methods)
            parsed[:methods]
        elseif haskey(parsed, "methods")
            parsed["methods"]
        elseif haskey(parsed, :data) && haskey(parsed[:data], :methods)
            parsed[:data][:methods]
        else
            return nothing
        end
        text = if haskey(methods_obj, :text)
            methods_obj[:text]
        elseif haskey(methods_obj, "text")
            methods_obj["text"]
        elseif haskey(methods_obj, :html)
            methods_obj[:html]
        else
            return nothing
        end
        return text === nothing ? nothing : String(text)
    catch err
        @warn "[Methods] failed to parse sidecar JSON at $(path); falling through cascade" exception=(err, catch_backtrace()) maxlog=1
        return nothing
    end
end

"""
    _methods_placeholder_card(label::AbstractString) -> String

helper. Render the placeholder
Bootstrap card for the rare case where neither a sidecar JSON nor a
`methods.md` is available alongside the per-condition `AnalysisResult`. The
HTML hint enumerates *both* paths the user might populate so the message is
actionable (the level-4 placeholder contract).

The consumer wraps the returned string into `json_object("html" => json_string(...))`.
"""
function _methods_placeholder_card(label::AbstractString)::String
    safe_label = _html_esc(string(label))
    return """<div class="card-body"><h5>Methods unavailable for $(safe_label)</h5><p>No methods payload found for this condition. The differential report tried (in order): in-memory <code>CONFIG</code>, the sidecar JSON, then the raw markdown. Provide one of the following alongside the per-condition <code>AnalysisResult</code> so this tab renders the full Methods block:</p><ul><li><code>&lt;basedir&gt;/interactive_report_data.json</code> (preferred &mdash; carries the structured methods payload produced by <code>generate_report</code>)</li><li><code>&lt;basedir&gt;/methods.md</code> (manuscript-ready prose written by <code>generate_methods_text</code>)</li></ul></div>"""
end
