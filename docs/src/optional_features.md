# Optional Features and Extensions

BayesInteractomics keeps the default `using BayesInteractomics` fast by deferring its heavy optional dependencies to package extensions. This page documents what each extension provides and how to activate it.

The package ships three extensions and four organisational submodules. Loading **only** `using BayesInteractomics` gives you the core Bayesian pipeline (Beta-Bernoulli, HBM, regression, BMA, Storey BFDR, the interactive report, curation, differential, and docking). Loading additional trigger packages activates extra capabilities.

## TL;DR — Activating the full pipeline

For a "batteries-included" analysis with metalearner-adjusted posteriors and MNAR-aware imputation:

```julia
using Flux, MLJ, MLJScikitLearnInterface, HDF5    # activates the Metalearner extension
using GLM                                          # activates the Imputation extension
using BayesInteractomics

config = CONFIG(
    datafile     = ["path/to/data.xlsx"],
    control_cols = [Dict(1 => [2,3,4])],
    sample_cols  = [Dict(1 => [5,6,7])],
    poi          = "BAIT_UNIPROT_ID",
    n_controls   = 3, n_samples = 3, refID = 1,
    output       = OutputFiles("results"),
    imputation_method = :mnar,                     # MNAR-aware imputation (requires GLM)
)
results, ar = run_analysis(config)
println(ar.metalearner_status)  # :loaded
```

The metalearner activates automatically when the trigger packages (`Flux, MLJ, MLJScikitLearnInterface, HDF5`) are loaded — there is no separate enable flag.

Without `using Flux, MLJ, MLJScikitLearnInterface, HDF5`, `posterior_prob` falls back to `bf / (1 + bf)` (BF-derived). The interactive HTML report renders a status banner explaining this — see [Why this matters (TTFX)](#why-this-matters-ttfx) for the speed/feature trade-off.

Without `using GLM`, attempting `imputation_method ∈ (:mar, :mnar)` raises an `ArgumentError`. With `imputation_method = :none` (the default), BayesInteractomics works without GLM.

## Metalearner extension (Variante B graceful fallback)

**Trigger:** `using Flux, MLJ, MLJScikitLearnInterface, HDF5`

**Package:** `BayesInteractomicsMetalearnerExt` (in `ext/BayesInteractomicsMetalearnerExt/`)

The metalearner extension provides the DNN architecture (`dnn_model.jl`) and the self-contained `MLJ.Stack` metalearner (`metalearner.jl`) that adjusts the per-protein BMA posterior. When the trigger packages are loaded, `run_analysis` automatically detects them at runtime and the metalearner replaces the BF-derived posterior with a calibrated probability conditional on the sub-model BFs, the DNN prior score, and STRING / Pfam-DDI evidence channels.

The production metalearner uses the 14-feature `:tr_ddi` schema by default, with a bundled isotonic calibrator. MC-Dropout (the 15-feature `:tr_ddi_mc` schema) is an **opt-in**, not the default — set `metalearner_use_mc_dropout = true` to enable it (this emits a deprecation warning). For the full prior-engine architecture — feature schema, the six base learners, calibration, and the MC-Dropout opt-in — see the dedicated [Metalearner](metalearner.md) page.

### When the trigger packages are NOT loaded

The pipeline still runs end-to-end via a graceful fallback path (Variante B):

1. The empty `predict_metalearner` stub catches `MethodError`.
2. `run_analysis` emits exactly one `@warn` per session.
3. `AnalysisResult.metalearner_status` becomes `:extension_not_loaded`.
4. `posterior_prob` falls back to `bf / (1 + bf)` from the EM/copula stage — already populated upstream.
5. The interactive HTML report renders a status banner (`.alert.alert-warning`) pointing you at the trigger command. The Methods tab adds a "Metalearner Status" subsection. The `posterior_prob` column gets a tooltip distinguishing metalearner-adjusted from BF-derived posteriors.

### Status sentinels

`AnalysisResult.metalearner_status` is always one of three symbols (defined in `METALEARNER_STATUS_VALUES`):

| Sentinel                  | Meaning                                                                            |
|---------------------------|------------------------------------------------------------------------------------|
| `:loaded`                 | Extension active AND prediction succeeded. Posteriors are metalearner-adjusted.    |
| `:extension_not_loaded`   | Trigger packages not loaded. BF-derived fallback is used. Banner rendered.         |
| `:prediction_failed`      | Extension active but `predict_metalearner` errored or returned nothing.            |

The interactive report renders one of three banner / Methods-tab messages depending on which sentinel is set.

### DNN Prior uncertainty (MC-Dropout transparency signal)

When the metalearner extension is loaded **and** `run_dnn_prior_mc_dropout = true` (the default), the pipeline runs **K = 30 MC-Dropout forward passes** through the production DNN for every protein-pair after `predict_metalearner` returns. This is a **transparency signal** (per-pair prior uncertainty), not a calibration improvement — the load-bearing calibration is the metalearner's bundled isotonic calibrator. Five new columns are persisted on the per-condition results DataFrame:

| Column | Meaning |
|--------|---------|
| `prior_mc_mean` | Posterior-mean over K MC-Dropout draws (the per-pair DNN prior probability). |
| `prior_mc_std`  | Standard deviation of the K MC-Dropout draws (uncertainty signal). |
| `prior_mc_ci_low`, `prior_mc_ci_high` | Empirical 2.5 % / 97.5 % quantiles of the K-sample distribution (95 % CI). |
| `prior_contribution` | `posterior_prob − prior_mc_mean`. Positive = AP-MS evidence supports the interaction beyond the sequence/network prior; negative = AP-MS contradicts the prior. |

The interactive report surfaces these via a **"DNN Prior"** tab (searchable + sortable DataTable + Plotly scatter of `prior_mc_mean` × `posterior_prob` coloured by `prior_mc_std`) and a **`Prior Δ`** column in the main Results DataTable.

| CONFIG field | Default | Meaning |
|-------|---------|---------|
| `run_dnn_prior_mc_dropout::Bool` | `true` | Enable MC-Dropout uncertainty quantification. Set to `false` to skip the K forward passes and render the empty-state alert in the report tab. |
| `dnn_prior_mc_k::Int` | `30` | Number of MC-Dropout forward passes per pair. |
| `dnn_prior_mc_batch_size::Int` | `256` | Mini-batch size for the K-pass inference loop. |

On a 50 000-pair proteome at `K = 30`, expect **5–10 min added latency on CPU** over the no-MC baseline; on a CUDA-capable GPU the same workload completes in **< 1 min**. For time-constrained CPU-only runs, set `run_dnn_prior_mc_dropout = false`.

When the metalearner trigger packages are not loaded, the pipeline emits one `@warn` per session and populates the five prior columns with `NaN`; the "DNN Prior" tab renders an empty-state alert. Downstream `BFDR`, `PEP`, and `posterior_prob` are **not** affected. Inside `differential_analysis(; conditions=...)`, each per-condition `analyse()` call computes its own MC-Dropout priors (per-condition baits can differ), switchable via the report's per-condition dropdown; the wide cross-pair DataFrame does not carry the prior columns.

> **Note:** `run_dnn_prior_mc_dropout` (the DNN Prior transparency tab, default `true`) is distinct from `metalearner_use_mc_dropout` (the 15-feature metalearner schema opt-in, default `false`). See the [Metalearner](metalearner.md) page for the schema distinction.

## Imputation extension (explicit-error path)

**Trigger:** `using GLM`

**Package:** `BayesInteractomicsImputationExt` (in `ext/BayesInteractomicsImputationExt/`)

The imputation extension provides per-column logistic dropout-curve fitting (`dropout.jl`) and tilted-Gaussian MNAR sampling (`imputation_mnar.jl`). These are needed only when you request **dropout-aware** imputation; the simpler mean-imputation wrapper in `src/data/imputation.jl` stays in core and does not depend on GLM.

### Behaviour when GLM is NOT loaded

Unlike the metalearner, the imputation extension takes a **loud, explicit-error** path because imputation is something you actively opted in to:

| `imputation_method` value | Behaviour without GLM                                                                                |
|---------------------------|------------------------------------------------------------------------------------------------------|
| `:none` (default)         | Stubs not invoked. Raw data flows through silently. Works without GLM.                                |
| `:mar`                    | `_require_imputation_extension(:mar)` throws `ArgumentError("... requires using GLM ...")`            |
| `:mnar`                   | `_require_imputation_extension(:mnar)` throws `ArgumentError("... requires using GLM ...")`           |

The error fires **before** any expensive work begins, so you do not waste time on a half-completed run.

### When GLM IS loaded

`fit_dropout_curves`, `impute_mnar`, `impute_mnar_from_paths`, `save_dropout_fit`, and `load_dropout_fit` all resolve to their extension methods. The `DropoutFit` struct itself is in core (constructible), but the only producer of fitted dropout curves is the extension.

## Mask-aware regression (v2b)

When `config.mask_aware_regression = true` (the default), the Bayesian
regression observation factor is **variance-aware of which cells were
MNAR-imputed**. This down-weights imputed cells so that a dose-response slope
is not driven by sampler-injected values.

Two factor-graph topologies implement the contract:

- **Single-protocol (production):** the `(slope, intercept)` block is modelled
  as a single joint `θ ~ MvNormal` node (`regression_one_protocol_robust_structured`)
  so RxInfer propagates the exact joint covariance. The MNAR down-weighting is
  folded into the per-cell precision τ **prior mean** (`imputed → Gamma centred
  at 1/(1/τ_base + σ²_imp)`).
- **Multi-protocol:** an additive-variance observation factor
  `var(data[cell]) = 1/τ[cell] + σ²_imp[cell] · is_imputed[cell]`.

In both paths the per-cell Gamma τ (Student-t robustness) is **retained** — the
imputed-cell down-weighting is layered on top of it, not a replacement.

**Source of `σ²_imp`:** per-source-column tilted-Gaussian variance computed
from the post-imputation intensity matrix; uses the per-MS-run `DropoutFit`
produced by the imputation extension (`ext/BayesInteractomicsImputationExt/`).
The exported helper is `column_imputation_sigma(fit::DropoutFit, col::Int, intensity_matrix)`.

**Opt-out:** set `config.mask_aware_regression = false` to revert to the plain
`precision = τ[cell]` observation factor (byte-identical on raw, non-MNAR data):

```julia
config = CONFIG(
    # ... your standard config ...
    mask_aware_regression = false,    # plain precision = τ[cell] observation factor
)
```

**Cache invalidation:** the observation-factor change bumped
`HBM_REGRESSION_CACHE_VERSION` (in `src/core/intermediate_cache.jl`). Old
regression-cache files emit a "regression model changed; recompute" warning
and trigger explicit recomputation. BetaBernoulli, H0, and Platt-calibration
caches are unaffected (each cache carries an independent version constant).

**Multi-imputation pooling:** when `run_analysis(::Vector{InteractionData}, raw_data, config)`
is called, the regression runs once per imputed draw and the μ_α slope
posteriors are pooled via the existing `miconvertRegression` pattern in
`src/data/imputation.jl`. M = 1 collapses to identity.

**Methods-tab visibility:** the generated `report.html` and
`differential_report.html` include a `<h4>Mask-aware regression (v2b)</h4>`
subsection in the Methods tab when this feature is active, plus a per-condition
`pct_imputed_cells` chip in the Data Quality tab. See
[`docs/src/imputation.md`](imputation.md) for the dropout-curve methodology and
σ_imp derivation.

## Network analysis extension

**Trigger:** `using Graphs, SimpleWeightedGraphs, GraphPlot, Compose` (and `Cairo` if you want PNG export)

**Package:** `BayesInteractomicsNetworkExt` (in `ext/BayesInteractomicsNetworkExt/`).

See [Network Analysis](network_analysis.md) for the full API: `build_network`, `centrality_measures`, `detect_communities`, `plot_network`, `export_graphml`, etc.

## Organisational submodules (no dependency change)

Four areas of `src/` are wrapped in submodules. There is **no user-facing change** — `using BayesInteractomics` still re-exports every public symbol — but downstream code can now reach the submodule directly if it wants a cleaner namespace.

| Submodule                          | Wraps                                                       | Public API entry points                                                                                  |
|------------------------------------|-------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `BayesInteractomics.Reports`       | `src/reports/*.jl`                                          | `generate_report`, `generate_differential_report`                                                        |
| `BayesInteractomics.Curation`      | `src/data/{curation, curation_types, string_api}.jl`        | `curate_proteins`, `CurationReport`, splitting / merging API                                             |
| `BayesInteractomics.Differential`  | `src/differential/*.jl`                                     | `differential_analysis`, `DifferentialConfig`, `DifferentialResult`, classification enums                |
| `BayesInteractomics.Docking`       | `src/docking/*.jl`                                          | `DockingConfig`, `DockingResult`, `apply_docking_update`, two-stage Bayesian update API                  |

## Why this matters (TTFX)

The motivation for splitting these features into extensions is **time-to-first-X** (TTFX) — how long `using BayesInteractomics` takes at the REPL before you can run anything. Each extension adds load time when its trigger packages are imported; users who do not need a feature should not pay for its load cost.

Splitting the optional features into extensions delivered a large cold-load speed-up against the pre-split baseline. The measured deltas are:

| Metric                  | Pre-split baseline | With extensions split | Delta            |
|-------------------------|--------------------|-----------------------|------------------|
| Cold median (5 runs, s) | 50.06              | 29.27                 | **−41.52 %**     |
| Warm median (5 runs, s) | 13.91              | 5.52                  | **−60.30 %**     |

Measured on Julia 1.12.5, Windows, `--threads=4`. Cold = fresh `julia` subprocess with `~/.julia/compiled/v1.12/BayesInteractomics/` deleted. Warm = fresh subprocess with precompile cache preserved (measures per-process load cost: `__init__`, extension trigger resolution, eager `using` traversals).

Reproduction script: `julia --project=. --threads=4 scripts/measure_ttfx.jl --n 5`.

If you load all three extensions, you pay roughly the pre-split cost — that is by design. The win is that users who do not need the metalearner or MNAR imputation get a much faster REPL.

## Example scripts

All four scripts under `examples/` activate the full stack at the top of the file:

```julia
using Flux, MLJ, MLJScikitLearnInterface, HDF5    # activates BayesInteractomicsMetalearnerExt
using GLM                                          # activates BayesInteractomicsImputationExt
using BayesInteractomics
```

`examples/Project.toml` pins all five trigger packages (`Flux`, `MLJ`, `MLJScikitLearnInterface`, `HDF5`, `GLM`) plus `Graphs`, `SimpleWeightedGraphs`, `GraphPlot`, `Compose`, `Cairo` for the network extension, so the example scripts always produce a full report with `metalearner_status == :loaded`.

If you fork an example and remove some of these `using` lines, you will see the corresponding fallback path activate in your output (a `:warning` banner in the report, or an `ArgumentError` if you set `imputation_method ∈ (:mar, :mnar)` without `using GLM`).
