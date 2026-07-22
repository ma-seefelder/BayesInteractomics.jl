# Configuration

The [`CONFIG`](@ref) struct centralises every analysis parameter for [`run_analysis`](@ref). Most fields carry sensible defaults; this page documents the fields most likely to be tuned — the metalearner, normalisation, evidence-stream, and copula-family selectors. For the full list of `CONFIG` fields, see the [`CONFIG`](@ref) docstring in the [API Reference](api.md).

The metalearner is the load-bearing posterior-calibration step: when the metalearner extension is loaded (`using Flux, MLJ, MLJScikitLearnInterface, HDF5`), it refines the BF-derived posterior `bf / (1 + bf)` from the EM/copula stage into a calibrated probability conditional on STRING + DNN features. See [Optional Features and Extensions](@ref) for the extension-activation mechanics and the graceful fallback when the trigger packages are not loaded.

## Metalearner configuration

### `metalearner_path::Union{Nothing, String}` (default `nothing`)

This field is `::Union{Nothing, String}` with a default of `nothing`. The default triggers **schema-matching default resolution** at runtime: the metalearner extension selects the production artefact that matches the requested feature schema.

- When `metalearner_path === nothing` AND `metalearner_use_mc_dropout == false` (the **default**), the extension loads `metalearners/metalearner_tr_ddi.jld2` (the 14-feature TR+DDI schema — the production default).
- When `metalearner_path === nothing` AND `metalearner_use_mc_dropout == true` (the deprecated opt-in), the extension loads `metalearners/metalearner_tr_ddi_mc.jld2` (the 15-feature schema, including the `mc_std` MC-Dropout column).
- An explicit `String` path bypasses the schema-aware default selection and loads exactly that file.

**Backward compatibility.** Set `metalearner_path = "metalearners/HistGradientBoosting_tune.jld2"` to load the legacy 8-feature artefact. It loads with `schema_tag = :legacy_8feat` and predicts byte-identically to the earlier pipeline. The other legacy artefacts on disk (`LogisticClassifier_tune.jld2`, `GaussianNBC_tune.jld2`, `ensemble.jld2`) likewise continue to load and work.

```julia
# (1) Auto-resolve to the 14-feature TR+DDI artefact (the default).
cfg = CONFIG(
    # ... required fields ...
    metalearner_path           = nothing,   # → metalearners/metalearner_tr_ddi.jld2
    metalearner_use_mc_dropout = false,     # default
)

# (2) Opt in to MC-Dropout → 15-feature artefact (deprecated; adds the K=30 cost).
cfg = CONFIG(
    # ... required fields ...
    metalearner_path           = nothing,   # → metalearners/metalearner_tr_ddi_mc.jld2
    metalearner_use_mc_dropout = true,
)

# (3) Pin the legacy 8-feature artefact explicitly (earlier pipeline behaviour).
cfg = CONFIG(
    # ... required fields ...
    metalearner_path = "metalearners/HistGradientBoosting_tune.jld2",
)
```

### `metalearner_use_mc_dropout::Bool` (default `false`)

**The 14-feature `:tr_ddi` schema is the production default; MC-Dropout is a deprecated opt-in.** When `false` (the default), the metalearner loads the 14-feature `metalearners/metalearner_tr_ddi.jld2` artefact. When `true`, the metalearner inference path appends a per-pair **MC-Dropout standard-deviation** column (`mc_std`) to the feature matrix and loads the 15-feature `metalearners/metalearner_tr_ddi_mc.jld2` artefact (unless `metalearner_path` is pinned explicitly). The `mc_std` column is reconstructed via K=30 stochastic forward passes through the **same** DNN model that the metalearner already uses to produce the `DNN` feature.

**Why the 14-feature schema is the default.** On multi-species data the non-MC `:tr_ddi` schema outperforms the 15-feature `:tr_ddi_mc` schema (AUC +0.019, MCC +0.024). Setting `metalearner_use_mc_dropout = true` emits a one-shot deprecation warning to that effect; MC-Dropout is retained for non-human-focused use only.

**No hard error — MC rides the existing DNN dependency.** Both the 14-feature (`:tr_ddi`) and 15-feature (`:tr_ddi_mc`) schemas already call `predict_DNN(...)` to compute the `DNN` feature, so the DNN model is a hard dependency of *every* metalearner schema. MC-Dropout is just K=30 extra passes through that same model — it introduces no new dependency. Consequently there is **no** state where the 14-feature path works but MC cannot:

- If the metalearner extension (`using Flux, MLJ, MLJScikitLearnInterface, HDF5`) and the DNN model are available, the metalearner runs and (when opted in) MC-Dropout runs with it.
- If the extension or model is absent, the **whole** metalearner falls back to the BF-derived posterior `bf / (1 + bf)` — identical behaviour for both schemas, no MC-specific error.

```julia
cfg = CONFIG(
    # ... required fields ...
    metalearner_use_mc_dropout = false,   # default → 14-feature metalearner_tr_ddi.jld2
)
results, ar = run_analysis(cfg)
```

## Candidate pool and tuning measure

The production metalearner stacks **six** base learners — HistGradientBoostingClassifier, EvoTreesClassifier, LogisticClassifier, KNNClassifier, RandomForestClassifier, and ExtraTreesClassifier — via an `LR_L2` blender (Sill-2009 feature-weighted stacking, where the original features are fed at both level-1 base training and level-2 blender training). **AdaBoost and GaussianNB were dropped**, because both exhibited structural mis-calibration (ECE 0.12 and 0.22 respectively) that survived hyperparameter tuning.

The inner-CV tuning measure is **Brier** by default; LogLoss was evaluated but did not clear the promotion threshold.

## Shipped metalearner artefacts

Two production artefacts ship alongside the retained legacy ones:

| Artefact | Schema | Features | Selected when |
|----------|--------|----------|---------------|
| `metalearners/metalearner_tr_ddi.jld2` | `:tr_ddi` | 14 | `metalearner_path === nothing`, `metalearner_use_mc_dropout == false` (**default**) |
| `metalearners/metalearner_tr_ddi_mc.jld2` | `:tr_ddi_mc` | 15 (incl. `mc_std`) | `metalearner_path === nothing`, `metalearner_use_mc_dropout == true` (deprecated opt-in) |
| `metalearners/HistGradientBoosting_tune.jld2` | `:legacy_8feat` | 8 | explicit `metalearner_path` (back-compat baseline) |

The legacy artefacts (`HistGradientBoosting_tune.jld2`, `LogisticClassifier_tune.jld2`, `GaussianNBC_tune.jld2`, `ensemble.jld2`) remain on disk and continue to load and predict byte-identically to the earlier pipeline — no migration is required for existing configurations that pin an explicit `metalearner_path`.

## Plotting toggles

Per-protein and per-analysis diagnostic plots are off by default — they add substantial wall-time on large proteomes. Enable individual plots as needed:

- **`plotHBMdists::Bool` (default `false`)** — generate and save per-protein hierarchical Bayesian model (HBM) enrichment posterior distribution plots.
- **`plotlog2fc::Bool` (default `false`)** — generate and save per-protein log2 fold-change distribution plots.
- **`plotregr::Bool` (default `false`)** — generate and save per-protein regression model fit plots (dose-response correlation).
- **`plotbayesrange::Bool` (default `false`)** — generate and save Bayes factor range plots (sensitivity of the BF to the prior).
- **`vc_legend_pos::Symbol` (default `:topleft`)** — legend position in the volcano plot. Accepts any Plots.jl legend-position symbol (e.g. `:topright`, `:bottomleft`).

## Evidence-combination knobs

### `evidence_streams::Vector{Symbol}` (default `[:detection, :correlation, :enrichment]`)

Which evidence streams enter the joint copula. The default includes all three streams (the standard 3-D copula). Drop one stream — e.g. `[:correlation, :enrichment]` — for an ablation variant. Set membership (not list order) drives copula dimensionality; the copula build canonicalises the order internally. A configuration with fewer than two streams is rejected loudly with an `@assert` during H0 precomputation / evidence combination. The default is byte-identical to the standard pipeline.

### `use_metalearner_prior::Bool` (default `true`)

Apply the metalearner DNN-prior posterior update. When `true` (the default), the learned prior refines the posterior at both the single- and multi-protocol guard sites. When `false`, the metalearner DNN-prior update is suppressed at both sites **even if** the metalearner extension is loaded; the result keeps the BF-derived `posterior_prob` fallback from the EM/copula stage (no recomputation needed). This lets an ablation variant exclude the learned prior cleanly.

### `copula_family::Union{Nothing, Type}` (default `nothing`) and `h1_copula_family::Union{Nothing, Type}` (default `nothing`)

Force a fixed copula family instead of per-fit BIC selection. When `nothing` (the default), the family is chosen by BIC over the candidate set (byte-identical to the standard pipeline). When set — e.g. `copula_family = FrankCopula` — the family is forced at the H0 and H1 fits and at both EM-loop refit sites, threaded through both the `:bma` default path and the `:copula` path. `h1_copula_family` independently forces the H1-component family. The BMA sub-model names stay **"Copula"** and **"3c-EM"** regardless of the chosen family.

## EM and latent-class fields

### `em_burn_in::Int` (default `10`)

Number of EM burn-in iterations discarded before convergence diagnostics are collected for the Copula / BMA evidence-combination stage.

### `run_em_diagnostics::Bool` (default `true`)

When `true` **and** `n_restarts > 1`, runs the EM restart stability and convergence diagnostics (per-restart log-likelihood trace, π₀ initial/final, convergence flags). The diagnostics are skipped automatically when only a single restart is requested.

### `lc_convergence_tol::Float64` (default `1e-6`)

Convergence tolerance for the latent-class (3c-EM) EM loop — the EM iteration stops once the change in log-likelihood falls below this threshold.

## Regression Bayes factor

### `regression_bf_threshold::Float64` (default `0.1`)

Slope threshold for the regression (dose-response correlation) Bayes factor. The alternative hypothesis is `H1: slope > threshold`. The default `0.1` filters out very weak correlation signals; set it to `0.0` to test for *any* positive correlation, or to `0.3` to require a minimum effect size.

## Imputation and mask-aware regression

### `imputation_method::Symbol` (default `:mnar`)

Selects how input intensities are imputed. `:mnar` (the default) applies the detection-limit-calibrated MNAR-aware tilted-Gaussian imputation; `:none` flows missings through to the downstream models as latent variables. See [Imputation](imputation.md) for the full flow, the `mnar_variance_recovery` variance-recovery modes, and the activation contract (`using GLM`).

### `mask_aware_regression::Bool` (default `true`)

When `true` (the default), the Bayesian regression observation factor is made variance-aware of MNAR-imputed cells. Set it to `false` to fall back to the `precision = τ[cell]` observation factor (byte-identical on raw, non-imputed data). See the [Mask-aware regression (v2b)](imputation.md#mask-aware-regression-v2b) section of the Imputation page for the single-protocol vs multi-protocol mechanism.

## Embeddings and MNAR variance-inflation

### `embeddings_config::EmbeddingsConfig` (default `EmbeddingsConfig()`)

Configuration for the sample-level + protein-level embeddings and condition-level similarity subsystem (consumed by the volcano background layer and the [Differential Analysis](differential_analysis.md) Multi-Condition tab). Embeddings are configured entirely through this nested `EmbeddingsConfig` — there is no top-level `CONFIG` embedding-method field. The default `EmbeddingsConfig()` enables UMAP embeddings with deterministic seeding. Sub-fields:

- **`method::Symbol` (default `:umap`)** — non-linear projection method; one of `:umap`, `:tsne`, `:none`. `:none` skips the non-linear projection and produces `nothing` coordinates.
- **`seed::Int` (default `42`)** — `Random.seed!(seed)` is invoked before each UMAP / t-SNE call (UMAP.jl 0.1.10 has no `seed=` kwarg, so determinism is injected via the global RNG state).
- **`supervised::Bool` (default `false`)** — supervised UMAP using protein classes. UMAP.jl 0.1.x has no `y=` kwarg, so when `true` the extension emits one `@warn` and proceeds unsupervised.
- **`n_neighbors::Int` (default `15`)** — UMAP `n_neighbors`; clamped to `max(2, min(n_neighbors, n-1))` for small sample sizes.
- **`min_dist::Float64` (default `0.1`)** — UMAP `min_dist`.
- **`top_k_jaccard::Int` (default `50`)** — Top-K used for the Jaccard@Top-K condition-similarity secondary metric.
- **`run_embeddings::Bool` (default `true`)** — master toggle; `false` skips all embedding computation.

### `mnar_inflation_factor::Union{Nothing, Float64}` (default `nothing`)

Optional scalar override for the auto-derived per-protein variance-inflation factor used by the `mnar_variance_recovery = :inflation` path. When `nothing` (the default), the factor is derived per-protein from the per-column dropout curves × per-protein missingness; supplying a `Float64` pins a single factor for every protein. It is the sibling of `mnar_inflation_max::Float64` (default `3.0`), which caps the auto-derived factor. See the `mnar_variance_recovery` section of [Imputation](imputation.md) for the full variance-recovery flow.

## Normalisation

BayesInteractomics normalises AP-MS intensities on the log2 scale, **before** MNAR imputation, via the `CONFIG.normalisation_method` selector. Normalisation has two orthogonal axes: per-sample loading correction (DESeq size factors / `median_of_ratios`) and per-protein cross-protocol offset correction (row-centering). The two compose under `:both`; neither substitutes for the other.

### `normalisation_method::Symbol` (default `:auto`)

Selects the normalisation applied at the end of [`load_data`](@ref) (before imputation). All five values operate on the log2 scale:

- **`:none`** — no normalisation. Byte-identical to the legacy `normalise_protocols = false`.
- **`:row_center`** — per-protein cross-protocol row-centering only (subtracts each protein's cross-protocol baseline offset). Byte-identical to the legacy `normalise_protocols = true` (the existing `normalize()`).
- **`:median_of_ratios`** — DESeq size-factor **sample** normalisation: per-protein geometric mean across samples → per-sample median of ratios → divide. Linear-scale internally (`2^x → … → log2`), missing-aware (geometric mean over proteins observed in common). Equalises per-sample loading and composition bias.
- **`:both`** — `:median_of_ratios` followed by per-protein cross-protocol row-centering. This is the recommended combination for multi-protocol differential interactomics: column-scaling fixes per-sample loading, row-centering removes the per-protein cross-protocol offset that otherwise inflates the dose-response evidence.
- **`:auto`** (default) — on a multi-protocol load where a per-protein cross-protocol scale mismatch is detected (`detect_protocol_scale_mismatch`), automatically applies `:both`; otherwise applies `:none`. Single-protocol loads always resolve to `:none`. **This is a breaking change for existing multi-protocol users** — see the `CHANGELOG` normalisation entry.

```julia
# (1) Default — auto-detect multi-protocol scale mismatch and apply :both when needed.
cfg = CONFIG(
    # ... required fields ...
    normalisation_method = :auto,           # default
)

# (2) Force the full recipe (column-scaling + row-centering) unconditionally.
cfg = CONFIG(
    # ... required fields ...
    normalisation_method = :both,
)

# (3) Disable normalisation entirely.
cfg = CONFIG(
    # ... required fields ...
    normalisation_method = :none,
)
```

**Backward compatibility.** The legacy `normalise_protocols::Bool` CONFIG field and `load_data` keyword still work. The effective method is resolved by `_resolve_normalisation_method(normalisation_method, normalise_protocols)`: the `normalisation_method` selector wins when it is anything other than `:none`; otherwise the boolean is mapped (`true → :row_center`, `false → :none`). So `normalisation_method = :none, normalise_protocols = true` resolves to `:row_center`, and existing call sites that set only `normalise_protocols` keep working byte-identically.

**Ordering.** Normalisation runs BEFORE MNAR imputation — the imputation dropout curve is intensity-scale-sensitive, so size factors are computed on the pre-imputation data. For the file-based pre-imputation workflow (imputed data written to disk, then re-loaded), use [`normalise_then_impute`](@ref) to guarantee the correct order; `load_data` emits a warning if it reads an already-imputed file with a non-`:none` normalisation requested.

### `DifferentialConfig.bait_anchor::Bool` (default `false`)

A conditional, regression-safe bait-level correction for differential analysis. When `true`, a per-condition correction derived from the **raw bait abundance** is applied to **sample cells only** (controls untouched), so within-condition bait variation — the regression dose axis — is preserved and the predictor is never zeroed. Enable it only when conditions have a documented bait-abundance difference; it is near-inert when bait levels are matched across conditions.

```julia
cfg = DifferentialConfig(bait_anchor = true)
diff = differential_analysis(ar_wt, ar_mut; config = cfg)
# or per call:
diff = differential_analysis(ar_wt, ar_mut; bait_anchor = true)
```

The default (`false`) leaves the differential path byte-identical to the pre-bait-anchor behaviour.

### API

```@docs
normalise_then_impute
```
