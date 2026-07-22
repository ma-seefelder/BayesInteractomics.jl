# Imputation

## Overview

AP-MS intensity matrices are sparsely populated: any given protein is detected
in only a subset of MS runs, and the rest of the row is `missing`. In AP-MS /
DDA data that missingness is predominantly **MNAR (missing not at random)** —
the probability of non-detection depends on the underlying intensity through a
detection-limit effect: low-intensity peptides are systematically less likely
to be detected. Imputing MNAR cells as though the missingness were unrelated to
intensity produces values that are too high and too confident, which feeds
through to the `bf_correlation` term.

BayesInteractomics ships **detection-limit-calibrated MNAR-aware imputation**
as the default (`imputation_method = :mnar`). A per-MS-run logistic dropout
curve is fit to the observed detection pattern, and a per-cell tilted-Gaussian
sampler draws values from a Gaussian shifted downward by the column's dropout
slope. The per-column variance produced by that sampler (`σ_imp`) is then made
available to the regression observation factor — see
[Mask-aware regression (v2b)](#mask-aware-regression-v2b) — so the inference
downstream of imputation is aware of which cells were imputed and how
confidently.

The MNAR pipeline is delivered by the `BayesInteractomicsImputationExt`
package extension. Load it with `using GLM` before `using BayesInteractomics`
(see [`docs/src/optional_features.md`](optional_features.md) for the full
extension activation contract).

## Quick start

```julia
using GLM                        # activates BayesInteractomicsImputationExt
using BayesInteractomics

cfg = CONFIG(
    datafile          = ["path/to/data.xlsx"],
    control_cols      = [Dict(1 => [2, 3, 4])],
    sample_cols       = [Dict(1 => [5, 6, 7])],
    poi               = "BAIT_UNIPROT_ID",
    n_controls        = 3, n_samples = 3, refID = 1,
    output            = OutputFiles("results"),
    imputation_method = :mnar,                # default
)

results, ar = run_analysis(cfg)
```

The supported modes:

| `imputation_method` | Behaviour                                                                                     | Requires `using GLM`? |
| ------------------- | --------------------------------------------------------------------------------------------- | --------------------- |
| `:mnar` (default)   | Per-column logistic dropout fit + tilted-Gaussian MNAR sampler.                               | yes                   |
| `:none`             | No imputation. Missings flow through to the downstream HBM / regression as latent variables.  | no                    |

`imputation_method = :mar` selects a deprecated legacy imputation path retained
only for backward compatibility with older cached datasets. It is not a
documented workflow; selecting it emits a one-shot deprecation warning whose
message explicitly names `:mnar` as the replacement, so a user who hits the
warning is directed to the supported path. It will be removed in a future
release.

The `load_data(...; imputation = :mnar)` kwarg accepts the same values and
**takes precedence over `CONFIG.imputation_method`** when both are supplied
(see [`docs/src/data_loading.md`](data_loading.md)).

## Per-column dropout curve fit

For each MS run column `c` the imputation extension fits a logistic
detection model

```
o_{i,c} ~ Bernoulli(σ(ρ_c + ζ_c · ȳ_i))
```

where `o_{i,c}` is the detection indicator for protein `i` in column
`c`, and `ȳ_i` is the global mean of observed (log-)intensities for
that protein across all columns where it was detected.

- `ρ̂_c` — the intercept of the column's dropout curve. The
  **asymptotic detection probability** at high `ȳ` is `σ(ρ_c)`; large
  positive `ρ` indicates the column detects nearly every protein
  it sees, large negative `ρ` indicates the column never reaches full
  detection even for abundant proteins.
- `ζ̂_c` — the slope of the column's dropout curve. The
  **inflection-point intensity** at which detection probability is 50 %
  is `−ρ_c / ζ_c`; large `ζ` indicates a sharp detection threshold,
  small `ζ` a soft one. A healthy MS run has `ζ̂_c ∈ [0.5, 3]`. Negative
  `ζ̂_c` is a quality signal that the column does not behave like a
  detection-limit-driven sampling channel (e.g., heavy contaminant load)
  and surfaces in the `SANITY.md` report.

The fit is produced by `fit_dropout_curves` in
`ext/BayesInteractomicsImputationExt/dropout.jl` and persisted as JSON via
`save_dropout_fit` / `load_dropout_fit`. The JSON schema carries a locked
`version` field and stores `ρ`, `ζ`, `n_detections`, `excluded` per column
plus a SHA-256 hash of the intensity matrix so cross-language consumers
(e.g., the R companion script) can verify they are looking at the same data.

Three diagnostic plots are emitted to `diagnostics_dir/`:

1. Per-column decile-binned scatter + fitted sigmoid overlay.
2. All-sigmoids overlay grid, panelled by protocol.
3. Histogram of `ζ̂_c` with vertical lines at 0.5 and 3.

A free-text `SANITY.md` report carries five fit-quality metrics
(well-fit fraction, negative-`ζ̂` fraction, excluded fraction,
per-protocol mean ± SD, outlier columns).

## Tilted-Gaussian MNAR sampler

For each `(protein i, column c)` cell with `missing` intensity, the
sampler draws a value from a Gaussian centred at the per-protein mean
`μ̂_i` and shifted left by the column's dropout slope. The
unnormalised density is

```
f(y) ∝ φ(y; μ̂_i, σ̂_i²) · (1 − σ(ρ_c + ζ_c · y))
```

— a standard Normal multiplied by the column's per-cell *non*-detection
probability. The implementation uses an asymmetric grid
`[μ − k_low·σ, μ + k_high·σ]` (with `k_low = 5`, `k_high = 2` by
default) and inverts the discrete CDF with midpoint quadrature for
unbiased sampling under modest `n_grid`. Per-protein moments
`(μ̂_i, σ̂_i²)` are estimated by `_estimate_per_protein_moments` with
a pooled-σ² fallback for proteins with fewer than `n_obs_threshold`
detections.

The sampler lives in `impute_mnar` in
`ext/BayesInteractomicsImputationExt/imputation_mnar.jl`. It is invoked by
`load_data` when `imputation == :mnar` (or by the pipeline when
`CONFIG.imputation_method == :mnar`); single-call, single-draw imputation is
the default. Multi-draw imputation is covered in
[Optional variance recovery](#optional-variance-recovery).

The companion helper

```julia
column_imputation_sigma(fit::DropoutFit, col::Int, intensity_matrix) -> Float64
```

returns the empirical `σ` of the **post-imputation** intensities in
column `col`. This is the same per-column value the v2b mask-aware
regression consumes — see [Mask-aware regression (v2b)](#mask-aware-regression-v2b)
below. `fit` is consulted for boundscheck only; the actual σ value
comes from the post-imputation matrix to match the empirical pattern
the downstream observation factor expects.

## Optional variance recovery

A single-draw MNAR sampler produces tight downstream credible intervals
that may understate the true uncertainty introduced by imputation. Two
escape hatches are available behind the `mnar_variance_recovery` CONFIG
field:

| `mnar_variance_recovery` | Behaviour                                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `:off` (default)         | Single-draw MNAR. Recommended for the reference workflow.                                                                        |
| `:inflation`             | Variance-inflation post-hoc. Per-protein HBM σ is widened by a factor derived from the per-protein imputed-cell count; the factor is capped by `mnar_inflation_max`. |
| `:multi_impute`          | Full multi-draw MNAR with `m = mnar_m` imputations (default 2) and Rubin's within + between pooling at the regression-posterior step. |

Both opt-in modes leave `:off` as the byte-equality reference path —
turning the flag back to `:off` reproduces the single-draw posterior CIs
exactly. The reproducibility manifest at
`<output.basedir>/imputed_data/dataset_mnar_multi_impute_manifest.json`
carries nine keys including the explicit `seeds` array for `:multi_impute`
runs.

Multi-draw MNAR uses a deterministic seed schedule for reproducibility.
For `mnar_base_seed = base_seed` and `mnar_m = m`, the per-draw seeds are

```julia
seeds = [Int(base_seed) * 1_000_003 + i for i in 1:m]   # i ∈ 1..m
```

so the m-draw imputation is byte-reproducible under a fixed `base_seed`.
The recommendation is to leave the flag at `:off` and only flip to
`:inflation` if the resulting CIs look implausibly tight on a real
dataset where MNAR is known to dominate.

## Mask-aware regression (v2b)

The Bayesian regression observation factor is made **variance-aware of
MNAR-imputed cells** when `mask_aware_regression = true` (the default).
Two regression code paths implement this contract through different
factor-graph topologies.

**Single-protocol (production path).** The `(slope, intercept)` block is
modelled as a single joint `MvNormal` node `θ`, so the variational
message passing propagates the exact joint covariance between slope and
intercept rather than a mean-field split. The per-column MNAR variance
`σ²_imp` is folded into the per-cell precision `τ` **prior mean**: an
imputed cell receives a `Gamma` precision prior centred at
`1 / (1/τ_base + σ²_imp)` (a conjugate down-weighting). The per-cell
Empirical-Bayes Gamma precision — the Student-t scale mixture that gives
the regression its heavy-tailed robustness — is retained. The slope
posterior prior is a scale-matched `Normal(0, jzs_r_scale²)`; the JZS
Cauchy remains the Bayes factor's analytical null reference. Model
function: `regression_one_protocol_robust_structured`.

**Multi-protocol.** The multi-protocol path
(`RegressionModel_multi_protocol_robust_jzs_v2b`) implements the
variance-aware contract as an **additive** variance term on imputed
cells:

```
var(data[cell]) = 1/τ[cell] + σ²_imp[cell] · is_imputed[cell]
```

realised via a latent biological-noise chain
`y_bio[cell] ~ Normal(predicted_value, precision = τ[cell])` +
`data[cell] ~ Normal(y_bio[cell], variance = σ²_imp[cell] + ε_floor)`.
The structured joint-covariance treatment used on the single-protocol
path is a tracked follow-up for the multi-protocol path.

In both paths the terms are:

- `τ[cell]` — the per-cell Empirical-Bayes precision (Student-t scale
  mixture; heavy-tail / robust-regression behaviour is intact).
- `σ²_imp[cell]` — per-source-column tilted-Gaussian variance from the
  post-imputation intensity matrix via
  `column_imputation_sigma(fit, col, intensity_matrix)`. Computed once
  per analysis and passed into the regression wrappers as a
  `Dict{Tuple{Int,Int,Int}, Float64}` lookup.
- `is_imputed[cell] ∈ {0, 1}` — built at `prepare_regression_data` time
  by comparing the raw and imputed cells. All-`false` when the
  regression is called without a raw-data reference (the
  backward-compatible default).

The effect in contract terms is the same on both paths: cells imputed
under the dropout-limit regime carry extra observation uncertainty, so
the regression posterior on the slope reflects that the value was
inferred rather than measured. This makes the `bf_correlation` evidence
aware of which cells were imputed; it does not, on its own, guarantee
any particular saturation profile for that term on real data.

Opt out by setting

```julia
cfg = CONFIG(..., mask_aware_regression = false)
```

This reverts to the `precision = τ[cell]` observation factor; the
resulting Bayes factors are byte-identical on raw, non-imputed data.

### Multi-imputation pooling

When `run_analysis(::Vector{InteractionData}, raw_data, config)` is
called, v2b regression runs once per imputed draw and the `μ_α` slope
posteriors are pooled via `miconvertRegression(::Vector{BayesResult})`
in `src/data/imputation.jl` — a `MixtureModel{Normal}` of length `m`,
moment-matched to a pooled Normal via `to_normal`. This is equivalent
to a Rubin within+between pooled Normal without the Barnard-Rubin df
correction; `m = 1` collapses to identity.

## Cache invalidation

`src/core/intermediate_cache.jl` carries four per-cache version
constants:

- `BB_CACHE_VERSION` — Beta-Bernoulli (detection) cache.
- `HBM_REGRESSION_CACHE_VERSION` — HBM + regression posteriors.
- `H0_CACHE_VERSION` — null hypothesis Bayes-factor cache.
- `CALIBRATION_CACHE_VERSION` — Platt scaling / FDR calibration cache.

Each cache file's parameter hash includes the active `imputation_method`,
so caches produced under different imputation modes coexist on disk for
the same dataset; switching the mode does not invalidate an existing
cache because the two artefacts are looked up under different parameter
hashes.

The v2b mask-aware-regression change and the normalisation-pipeline
integration each bumped `HBM_REGRESSION_CACHE_VERSION`; the normalisation
change also bumped `H0_CACHE_VERSION` and `CALIBRATION_CACHE_VERSION`
because the normalised data scale those artefacts depend on changed.
`BB_CACHE_VERSION` stayed put because detection is presence/absence-only
and is independent of the intensity scale.

Stale regression caches emit

```
@warn "regression model changed; recompute"
```

at load time and the loader returns `nothing` — there is no silent
stale read.

## Worked example

```julia
using GLM                                    # activates BayesInteractomicsImputationExt
using BayesInteractomics

cfg = CONFIG(
    datafile          = ["path/to/data.xlsx"],
    control_cols      = [Dict(1 => [2, 3, 4])],
    sample_cols       = [Dict(1 => [5, 6, 7])],
    poi               = "BAIT_UNIPROT_ID",
    n_controls        = 3, n_samples = 3, refID = 1,
    output            = OutputFiles("results"),
    imputation_method        = :mnar,        # default
    mnar_variance_recovery   = :off,         # default
    mask_aware_regression    = true,         # default; consumes σ²_imp downstream
)

results, ar = run_analysis(cfg)

# After the run, inspect the dropout fit:
fit = load_dropout_fit(joinpath("results", "dropout_curves.json"))
display(fit)
# DropoutFit (<timestamp>)
# ────────────────────────────────────
#   columns:        6  (fit: 6, excluded: 0)
#   proteins:       5029
#   ζ̂ ∈ [0.5, 3]:   5 / 6
#   software:       1.2.1
#   dataset_hash:   sha256:d46d1bc4270c0232024ba353…

# Per-column σ_imp lookup is computed once per analysis from the
# post-imputation intensity matrix; the same value feeds the v2b
# mask-aware regression observation factor downstream:
σ_imp_col3 = column_imputation_sigma(fit, 3, ar.intensity_matrix)
```

Diagnostic plots and the `SANITY.md` fit-quality report are emitted to
`<output.basedir>/dropout_diagnostics/` whenever `fit_dropout_curves`
is invoked with a `diagnostics_dir` argument (the pipeline does this by
default). The cross-language JSON contract at
`<output.basedir>/dropout_curves.json` is the same file consumed by the
R companion script.

The pipeline also exposes the `mask_aware_regression` flag — set it
to `false` to fall back to the `precision = τ[cell]` observation factor
(byte-identical on raw, non-imputed data); see also
[`docs/src/differential_analysis.md`](differential_analysis.md) for how
the downstream differential pipeline consumes the same `bf_correlation`
term that the v2b mask-aware regression makes imputation-aware.

## Activation contract

The MNAR pipeline lives behind the `using GLM` trigger because the
per-column logistic dropout fit calls into GLM.jl. The activation
contract is **explicit-error**, not graceful-fallback (in contrast to the
metalearner extension):

- `imputation_method = :none` — silent. No GLM required, missings flow
  through to the HBM / regression as latent variables.
- `imputation_method ∈ (:mnar, :mar)` without `using GLM` — throws

  ```
  ArgumentError: imputation_method = :mnar requires `using GLM` before `using BayesInteractomics`.
                 Alternatively, set `imputation_method = :none` to skip imputation.
  ```

  fired by `_require_imputation_extension(method)` in
  `src/data/imputation_stubs.jl` **before** any expensive work runs.

The full extension contract — TTFX impact, trigger-package list, status
sentinels for the metalearner / imputation pair — lives in
[`docs/src/optional_features.md`](optional_features.md). For the
CONFIG-field reference (including all `mnar_*` fields) see
[`docs/src/configuration.md`](configuration.md).
