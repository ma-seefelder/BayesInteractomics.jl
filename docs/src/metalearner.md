# Prior Engine (Metalearner)

The **prior engine** is BayesInteractomics' data-driven prior on each candidate
protein–protein interaction. Where the [evidence engine](model_evaluation.md) weighs the
experimental data through detection, enrichment, and correlation Bayes factors, the prior
engine brings in orthogonal, sequence- and network-level knowledge: it predicts whether two
proteins physically interact *before* the mass-spectrometry evidence is consulted. Its output
enters the pipeline as the per-interaction prior probability that the calibrated posterior is
built on.

The prior engine has two stages: a deep neural network that scores direct interaction, and a
metalearner stack that blends that score with additional evidence channels into a single
calibrated prior.

## Deep neural network (structural-contact prior)

A feed-forward deep neural network (DNN) predicts direct physical interaction between a protein
pair from protein sequence and STRING network embeddings. The trained network produces one
scalar per pair — a probability of direct interaction — which becomes a fixed input column
(`DNN`) to the metalearner. The DNN is trained once; all downstream prior-engine behaviour is
reported conditional on this single trained network.

The DNN lives in the `BayesInteractomicsMetalearnerExt` extension
(`ext/BayesInteractomicsMetalearnerExt/dnn_model.jl`) and is only active when the extension's
trigger packages are loaded (see [Extension activation](#Extension-activation)).

## Metalearner stack

The metalearner is a **self-contained `MLJ.Stack`**: a single serialised machine that bundles
six fixed-hyperparameter base learners, an L2-regularised logistic-regression blender, and a
post-hoc calibrator. The base learners are histogram gradient boosting, EvoTrees, L2-regularised
logistic regression, ``k``-nearest neighbours, random forests, and extra trees. Their
out-of-fold predictions are blended by the L2 logistic layer, and a post-hoc **isotonic
calibrator** — fitted on a validation slice the stack itself never saw — maps the blended score
to a calibrated probability. Because the whole stack is self-contained, loading the artefact and
calling `predict` in a fresh process reproduces the trained pipeline exactly.

Tuning uses Brier loss under 5-fold group-disjoint cross-validation inside the MLJ.jl framework.
The calibrated test-set Expected Calibration Error (ECE) sits at or below the ``0.035``
acceptance threshold adopted for downstream use.

### The `:tr_ddi` feature schema (production default)

The production schema is `:tr_ddi` — a **14-feature** input grouped into four families:

| Family | Count | Features |
|--------|-------|----------|
| In-species STRING channels | 7 | neighborhood, fusion, phylogenetic, coexpression, experimental, database, textmining |
| Transferred STRING channels | 4 | the same evidence types mapped across orthologous proteins |
| Pfam domain-interaction (DDI) | 2 | whether the pair's Pfam domain pair has any known interaction, and how many |
| Deep-learning prior | 1 | the `DNN` score |

The transferred-STRING and Pfam features are computed per species from each organism's own STRING
and Pfam resources, so the metalearner scores a non-human bait on species-correct evidence rather
than on human-derived or zero-filled features. This 14-feature `:tr_ddi` schema is the headline
default and the recommended production configuration.

### MC-Dropout (`:tr_ddi_mc`) — opt-in only

An uncertainty-augmented variant, `:tr_ddi_mc`, adds a **15th** feature: the Monte-Carlo-Dropout
standard deviation obtained from ``K = 30`` stochastic forward passes through the *same* DNN the
schema already loads for the `DNN` column. This exposes the network's per-pair predictive
uncertainty to the blender.

MC-Dropout is a **deprecated opt-in**, not the recommended path. On multi-species data the
14-feature `:tr_ddi` schema outperforms `:tr_ddi_mc` (higher AUC and MCC), so `:tr_ddi` is the
default. The switch is `CONFIG.metalearner_use_mc_dropout::Bool = false` (default `false` → the
14-feature schema); setting it to `true` selects the 15-feature schema and emits a runtime
deprecation warning that names the replacement, so a user who enables it is not left stranded.
Leave it unset (or `false`) for the recommended configuration.

## Extension activation

The prior engine is packaged as an extension so that a bare `using BayesInteractomics` stays fast
at the REPL. Load the trigger packages to activate it:

```julia
using Flux, MLJ, MLJScikitLearnInterface, HDF5   # activates the metalearner extension
using BayesInteractomics

config = CONFIG( #= ... =# )
results, ar = run_analysis(config)

ar.metalearner_status   # :loaded when the extension is active and prediction succeeded
```

When the extension is loaded and prediction succeeds, `posterior_prob` is metalearner-adjusted:
the calibrated prior enters the posterior for each interaction.

### Status sentinels

`AnalysisResult.metalearner_status` is always one of three symbols:

| Status | Meaning |
|--------|---------|
| `:loaded` | Extension active **and** prediction succeeded. Posteriors are metalearner-adjusted. |
| `:extension_not_loaded` | Extension inactive. Posteriors fall back to the Bayes-factor-derived value (see below). |
| `:prediction_failed` | Extension active but prediction errored or returned nothing. |

### Graceful fallback (extension not loaded)

The pipeline still runs end-to-end without the extension. When the trigger packages are absent,
`run_analysis` emits exactly one warning per session, sets
`metalearner_status = :extension_not_loaded`, and derives `posterior_prob` directly from the
combined Bayes factor as `bf / (1 + bf)` — the value already produced by the evidence engine, so
no recomputation is needed. The interactive HTML report surfaces this state with a status banner,
a "Metalearner Status" subsection in the Methods tab, and a tooltip on the `posterior_prob`
column distinguishing metalearner-adjusted from Bayes-factor-derived posteriors.

## Configuration reference

| Field | Default | Meaning |
|-------|---------|---------|
| `metalearner_use_mc_dropout::Bool` | `false` | Keep `false` for the recommended 14-feature `:tr_ddi` schema. `true` selects the 15-feature `:tr_ddi_mc` schema (deprecated opt-in; emits a warning). |
| `metalearner_path::Union{Nothing,String}` | `nothing` | Path to the stacked metalearner artefact. When `nothing`, the default artefact for the active schema is resolved automatically. |

See [Configuration](configuration.md) for the full `CONFIG` reference and
[Optional Features and Extensions](optional_features.md) for the extension trigger set, the
imputation extension, and REPL start-up cost figures.
