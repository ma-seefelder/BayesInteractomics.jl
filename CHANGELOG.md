# Changelog

All notable changes to BayesInteractomics are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased — v1.2.0 in flight] Dropout-Aware Imputation & Differential Refinement

The v1.2.0 milestone lands MNAR-aware imputation and differential-analysis refinement.

### v1.2.0 — Normalisation pipeline integration (median_of_ratios + cross-protocol row-centering) (2026-05-25)

Wires the normalisation recipe into the production pipeline: a `normalisation_method` selector, multi-protocol `:auto` scale-mismatch auto-flip, normalise-before-impute ordering, a conditional `bait_anchor` differential flag, and a cache-version bump.

**Added**
- `CONFIG.normalisation_method::Symbol` selector (default `:auto`) with five values: `:none`, `:row_center`, `:median_of_ratios`, `:both`, `:auto`. `:none`/`:row_center` are byte-identical to the legacy `normalise_protocols=false`/`true`.
- `median_of_ratios` (DESeq size-factor) sample normalisation operating directly on `InteractionData` — linear-scale internally (`2^x → … → log2`), missing-aware. New exported helpers `apply_normalisation`, `build_run_matrix`, `matrix_to_interactiondata`, `norm_median_of_ratios_id`.
- `detect_protocol_scale_mismatch(data; refID, threshold, n_sample)::Bool` (exported) — a pooled-OLS-residual guard turned into a boolean detector driving the `:auto` resolution. Threshold 2.5 log2 (calibrated): fires on disparate multi-protocol loads, stays silent on matched-level (~0.1) offsets.
- `normalise_then_impute(raw_data, dropout_fit; normalisation_method, refID)` (exported) — correct-order entry point (normalise RAW first, then MNAR-impute in-process) for the file-based pre-imputation workflow.
- `DifferentialConfig.bait_anchor::Bool` field (default `false`) + the exported `bait_anchor_id(data::InteractionData)` helper — a regression-safe per-condition raw-bait correction applied to SAMPLE cells only (controls untouched), default OFF and byte-identical when off.
- Methods-tab `Normalisation` subsection rendered in BOTH `report.html` and `differential_report.html` via `_methods_normalisation_block(config)`. `docs/src/configuration.md` gains a `## Normalisation` section.

**Changed**
- Normalisation now runs BEFORE MNAR imputation. The pipeline-driven multi-impute path already held this invariant; it is now made observable via a named seam, and the file-based pre-imputation path gets a loud order warning plus the `normalise_then_impute` correct-order helper.

**Changed (BREAKING)**
- **Multi-protocol scale-disparate results CHANGE** under the new `:auto` default. On a multi-protocol load where `detect_protocol_scale_mismatch` fires, `:auto` now auto-applies `:both` (median_of_ratios + per-protein cross-protocol row-centering) — eliminating the cross-protocol `bf_correlation` saturation. Existing multi-protocol users will see different `bf_correlation`, BMA posteriors, and downstream FDR/classification on scale-disparate data. Single-protocol behaviour is unchanged. **Opt out** with `normalisation_method = :none` (or `:row_center`) to recover the pre-v1.2.0 result.
- **Cache versions bumped: `HBM_REGRESSION_CACHE_VERSION` 18 → 19, `H0_CACHE_VERSION` 16 → 17, `CALIBRATION_CACHE_VERSION` 16 → 17.** Stale caches computed on the old un-normalised scale are rejected loudly (`return nothing` from the loaders → recompute). `BB_CACHE_VERSION` is **unchanged** (16) because Beta-Bernoulli detection is presence/absence and is invariant under row-centering + column-scaling. Users with cached results MUST delete `cache/` and re-run after upgrading.
- **`:none`/`:row_center` byte-identical guarantee**: `normalisation_method = :none` is byte-identical to `normalise_protocols = false`, and `:row_center` is byte-identical to `normalise_protocols = true`. The legacy `normalise_protocols::Bool` CONFIG field and `load_data` keyword continue to work (resolved via `_resolve_normalisation_method`; the new selector wins unless it is `:none`).

**Migration notes**
- Existing `cache/` directories MUST be wiped post-upgrade: `rm -rf cache/` (Unix) or `Remove-Item -Recurse -Force cache` (PowerShell). The cache-version mismatch handlers refuse stale caches loudly.
- **Re-run the HD capstone after upgrading** — multi-protocol HD results will change under `:auto`. The capstone is autonomous (user-run on real HD data).
- Single-protocol pipelines and any pipeline that explicitly sets `normalisation_method = :none` / `:row_center` (or only the legacy `normalise_protocols` bool) are unaffected.

**Threats mitigated**
- Undocumented breaking change: this **BREAKING** entry + the Methods-tab `Normalisation` subsection in both reports + the capstone re-run note.
- Stale docs misrepresenting behaviour: `docs/src/configuration.md` + the Methods tab describe the current behaviour; BMA ("Copula" / "3c-EM") and FDR (BFDR / PEP / local_fdr) terminology preserved.

---

### v1.2.0 — PEP / local FDR layer + bb_mnar_codriven diagnostic (2026-05-14)

Per-protein Posterior Error Probability layer + `bb_mnar_codriven` provenance flag.

**Added**
- `pep` column on `BayesResult.copula_results` (canonical lowercase; populated from the calibrated posterior when the ECE gate has been passed, otherwise from the raw posterior; coalesce in pipeline ensures non-NaN downstream).
- `differential_pep` column on `DifferentialResult.results` (α direct complement; `1 − differential_posterior`).
- Four γ class-conditional columns on `DifferentialResult.results`: `pep_gained`, `pep_reduced`, `pep_unchanged`, `pep_both_negative`. Normalized naive-product estimator under a documented conditional-independence approximation; sum-to-one within 1e-9 per row. Joint-posterior γ-PEP is deferred to v1.3+.
- `missing_fraction::Float64` column on `BayesResult.copula_results` (pre-imputation per-protein missingness; identical across `imputation_method ∈ (:mnar, :mar, :none)`).
- `bb_mnar_codriven::Bool` flag on bait `BayesResult.copula_results`; per-side `bb_codriven_A`/`bb_codriven_B` on `DifferentialResult.results`.
- `AnalysisResult.is_calibrated::Bool` struct field; `DifferentialResult.is_calibrated_A`/`is_calibrated_B` per-side fields.
- `BBMnarCodrivenConfig` struct (`CONFIG.bb_mnar_codriven::BBMnarCodrivenConfig`) with defaults `bb_bf_threshold=10.0`, `hbm_bf_threshold=10.0`, `missing_fraction_threshold=0.5`.
- New exported accessors: `getPEP(ar::AnalysisResult)`, `getDifferentialPEP(diff::DifferentialResult; class=:alpha)` (`class ∈ (:alpha, :gained, :reduced, :unchanged, :both_negative)` — invalid class throws `ArgumentError`), `isCalibrated(ar::AnalysisResult)` / `isCalibrated(diff::DifferentialResult; side=:A)`.
- Differential report Methods tab gains four new subsections: (1) PEP α/γ definitions, (2) γ-PEP conditional-independence caveat, (3) calibration gate semantics + `is_calibrated_A/_B` flags, (4) `bb_mnar_codriven` rule with threshold defaults.
- Bait + differential interactive reports gain a BB×MNAR codriven warning icon ⚠ in the Results table; differential renders per-side `bb_codriven_A`/`bb_codriven_B`.
- New testitems covering the PEP layer, the γ-PEP columns, the `bb_mnar_codriven` diagnostic, and the report changes.

**Changed (BREAKING)**
- **CACHE_VERSION bumped 22 → 23.** All pre-existing `analysis_cache.jld2` files force recomputation; the `CACHE_VERSION` mismatch path returns `nothing` from `load_result` for stale caches (loud-fail by design — no silent data corruption risk). Users with cached v1.1.x or earlier v1.2.0 results MUST delete `cache/` and re-run `run_analysis(config)`.
- **Differential volcano colour palette swapped**: GAINED is now red (`#d62728`), REDUCED is now blue (`#1f77b4`), UNCHANGED is mid-grey (`#7f7f7f`), BOTH_NEGATIVE is light-grey (`#cccccc`). This is a deliberate visual-semantics change vs v1.1.x where the GAINED/REDUCED colour assignment was inverted. Rationale: red conventionally means "increased / up" in biology dashboards; blue means "decreased / down". Cached v1.1.x report HTMLs retain the old palette; regenerate with v1.2.0 for the new contract.
- **Differential volcano y-axis switched** from `asinh(log10_dbf)` to `−log10(differential_pep)`. The y-axis now encodes statistical confidence (higher = more confident discovery); dBF magnitude is still surfaced on the dBF Diagnostics tab and in the volcano hover `customdata`.
- DataFrame columns `df.PEP` and `df.diff_PEP` become silent uppercase mirrors of `df.pep` and `df.differential_pep` (same `Vector` reference — `df.PEP === df.pep`). The uppercase aliases are retained as a v1.2.0 grace-window compatibility shim and will be dropped in v1.3.

**Deprecated**
- `getDifferentialQValues` retains its deprecation warning; users should migrate to `getDifferentialBFDR` (recommended) or the new `getDifferentialPEP` (per-protein PEP variant).

**Deferred to v1.3+**
- Joint-posterior γ-PEP that drops the conditional-independence approximation (currently the naive-product estimator with the caveat documented in Methods).
- k-group γ-PEP action set (`2^k − 1` subsets) — required for k≥3 differential overloads.
- PEP-bin stacked marginal histograms as an alternative to the continuous KDE per-class marginals shipped in v1.2.0 (both surface the same hue × saturation contract).
- Drop the uppercase-`PEP` / `diff_PEP` DataFrame mirror columns.

**Migration notes**
- Code reading `df.PEP` continues to work — the uppercase mirror is reference-identical to `df.pep`.
- Code reading `df.diff_PEP` continues to work — same mirror pattern over `df.differential_pep`.
- Tooling consuming JSON exports gets new fields: `pep_gained`, `pep_reduced`, `pep_unchanged`, `pep_both_negative`, `bb_codriven_A`, `bb_codriven_B`, `is_calibrated_A`, `is_calibrated_B`. Existing JSON consumers can ignore them or read them — both are safe.
- Existing `cache/` directories MUST be wiped post-upgrade: `rm -rf cache/` (Unix) or `Remove-Item -Recurse -Force cache` (PowerShell). The CACHE_VERSION mismatch handler will refuse stale caches loudly.

**Threats mitigated**
- NaN propagation in γ-PEP: σ_zero clamp + sum-to-one validation per row.
- Silent fallthrough between calibrated/raw posterior: coalesce semantics + `is_calibrated` flag exposed via `isCalibrated()` accessor + Methods subsection.
- Opacity NaN on the volcano: `marker.opacity ∈ [0.25, 1.0]` clamp enforced.
- XSS via condition labels in Methods HTML: condition labels esc()'d in the Methods template.

---

### v1.2.0 — Differential report parity (2026-05-13)

Differential report reaches tab-for-tab parity with the single-bait report.

**Added** Calibration / Sensitivity / Mixture Model / Methods / Data Quality tabs + dBF Diagnostics tab on `differential_report.html`. New `dbf_diagnostic::Symbol` column on `DifferentialResult.results` (`:ok` / `:saturated` / `:single_component` / `:model_disagreement`). An MS-only footer note replaces a separate Structural Evidence tab. `test/reports/test_report_parity.jl` enforces tab parity going forward.

---

### v1.2.0 — Optional MNAR variance recovery (2026-05-12)

Post-hoc variance inflation library + multi-imputation Rubin's MI dispatch for MNAR runs.

---

### v1.2.0 — Extension modularisation (2026-05-12)

Moved Flux / HDF5 / MLJ / MLJScikitLearnInterface / GLM from `[deps]` to `[weakdeps]` via two new package extensions (`BayesInteractomicsMetalearnerExt` + `BayesInteractomicsImputationExt`). Wrapped Reports / Curation / Differential / Docking into organisational submodules. TTFX cold-load **−41.52 %** (50.06 s → 29.27 s); warm-load **−60.30 %** (13.91 s → 5.52 s) — N=5 medians, Julia 1.12.5, Windows, `--threads=4`.

---

### v1.2.0 — Dropout-aware MNAR imputation (2026-05-09 → 2026-05-12)

Per-column dropout-curve fit (`p_detect = sigmoid(ρ_c + ζ_c · y)`); tilted-Gaussian MNAR sampler driven by the fitted curves; `CONFIG.imputation_method::Symbol = :mnar` default; all four caches (H0Cache + BetaBernoulli + HBMRegression + calibration) gain an `imputation_method` parameter-hash.

---

## 1.1.6 — BMA Evidence Combination Fix + Docking Integration (2026-05-09)

Released. Highlights: BMA linear BF pooling fix, AlphaFold Server docking integration, C2Qscore 4-metric AF3 scorer, `BFDR` / `PEP` / `local_fdr` terminology rename, documentation refresh.

## 1.1.5 — Input Data Quality Control (2026-04-11)

Five-check input-QC system (scale detection, replicate correlation, missingness asymmetry, intensity shape, PCA separation); `:ok` / `:warning` / `:fail` flag surfaced in the report header.

## 1.1.4 — Bugfixes and Interactive Report (2026-04-10)

Single-file interactive HTML report; Plotly.js + DataTables.js CDN integration; report.html + differential_report.html templates.

## 1.1.3 — Prior Sensitivity Reduction (2026-04-09)

Joint LC + copula sensitivity sweep; classification stability traffic-light per protein.

## 1.1.2 — Reorganise Plots (2026-04-05)

OutputFiles auto-population; volcano / convergence / evidence plot reorganisation.

## 1.1.1 — LatentClass Fix (2026-03-24)

3-component EM step-halving monotonicity; Student-t H0 anchoring; BIC-selected H1 family.

## 1.1 — Plotting, Fitting & Simulation (2026-03-20)

Simulation engine (5×5 grid × 10 replicates); Platt calibration; quality gates; sensitivity analysis.

## 1.0 — MVP (2026-03-10)

Initial release. Beta-Bernoulli detection + HBM enrichment + Bayesian regression correlation; Copula-EM evidence combination; CSV/XLSX data loading; per-protein parallel inference.
