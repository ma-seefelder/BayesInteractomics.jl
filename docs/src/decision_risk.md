# Bayesian Decision Risk

## Overview

Bayesian Decision Risk is a **distinct axis** in BayesInteractomics.
It augments the per-protein output of [`differential_analysis`](differential_analysis.md) with a
posterior **expected loss** under a user-overrideable 4×4 loss matrix, and surfaces the Bayes-optimal
call for each protein × pairwise contrast.

Decision Risk is conceptually distinct from the FDR-family columns documented in
[differential_analysis.md](differential_analysis.md):

| Quantity                | Type              | Scale            | Interpretation                                                    |
|-------------------------|-------------------|------------------|-------------------------------------------------------------------|
| `posterior_prob`        | Probability       | `[0, 1]`         | "How likely is this protein an interactor?"                       |
| `BFDR`, `PEP`, `local_fdr` | Probability    | `[0, 1]`         | "What fraction of calls at this cutoff are wrong?"                |
| **`decision_risk`**     | **Expected loss** | **`[0, ∞)`**     | **"How much would I lose, in loss units, if I act on this row?"** |

The user-facing benefit is direct: when ranking proteins for downstream wet-lab validation,
**the cheapest validation candidates surface first** — i.e. the rows where calling the protein
according to the Bayes-optimal action carries the least expected experimental cost. The
[Validation Candidates pill](#validation-candidates) in the interactive `differential_report.html`
uses this ranking out of the box.

Decision Risk is computed by `src/differential/decision_risk.jl` (the `compute_decision_risk!`
helper) and is invoked automatically inside both the legacy 2-group `differential_analysis(ar_a, ar_b; ...)`
signature and the k-group `differential_analysis(; conditions, ...)` overload — no opt-in needed.

## Statistical framing

For each protein × pairwise contrast (one row of `diff.results` in the legacy 2-group call,
or one row inside each entry of `diff.pairwise_results::Dict` for k≥3):

**Step 1 — Renormalise the γ-PEP columns into a 4-state posterior.**

The upstream pipeline emits four per-class γ-PEP columns:
`pep_gained`, `pep_reduced`, `pep_unchanged`, `pep_both_negative`.
Each γ-PEP is the posterior error probability for the corresponding class — `1 − pep_k` is the
posterior probability the true state is `k`. Decision Risk renormalises the four `(1 − pep_k)`
values into a proper categorical posterior:

```
P(state = k) = (1 − pep_k) / Σ_j (1 − pep_j)    for k, j ∈ {gained, reduced, unchanged, both_negative}
```

The back-reference for the γ-PEP definitions and the class-conditional construction lives in
[`differential_analysis.md`](differential_analysis.md). Decision Risk consumes those columns
as-is — it does not refit any EM or copula model.

**Step 2 — Compute the expected loss of every candidate action.**

For each action `a ∈ {gained, reduced, unchanged, both_negative}`, integrate the loss against the
posterior:

```
r_a = Σ_k P(state = k) · L[a, k]
```

where `L` is the active 4×4 loss matrix (either [`DEFAULT_DIFFERENTIAL_LOSS`](#default-differential-loss)
or a user-supplied override, see [Overriding the loss matrix](#overriding-the-loss-matrix)). The four
risks land in the columns `risk_gained`, `risk_reduced`, `risk_unchanged`, `risk_both_negative`.

**Step 3 — Pick the Bayes-optimal call.**

```
optimal_call    = argmin_a r_a
decision_risk   = r_{optimal_call}     # the expected loss of the chosen action
```

The chosen action lands in the `optimal_call::Symbol` column; its expected loss lands in the
`decision_risk::Float64` column.

**Degenerate-posterior fallback.** When the denominator `Σ_j (1 − pep_j) < 1e-12` (all four classes
near certainty against — typically a numerical artefact in extremely high-quality rows), Decision
Risk falls back to a **uniform posterior** `P(state = k) = 0.25` for each `k`, emits one
`@warn` per analysis (`maxlog = 1` discipline), and continues. The fallback is conservative: a
uniform posterior makes `optimal_call` track the row of `L` with the smallest unweighted sum, which
under `DEFAULT_DIFFERENTIAL_LOSS` is `:unchanged` (sum = 10) tied with `:both_negative` (sum = 10).

**Out-of-matrix rows.** Two rows from the 6-class `InteractionClass` enum sit outside the
4-action / 4-truth-state matrix: `CONDITION_A_SPECIFIC` and `CONDITION_B_SPECIFIC` describe one-sided
detection-only signals that the loss matrix does not model. For these rows the four `risk_<class>`
columns and `decision_risk` are set to `NaN`, and `optimal_call` is set to `:condition_a_specific`
or `:condition_b_specific` respectively. The `NaN` sentinel is
chosen over `Missing` so the column eltype stays `Float64`, matching the omnibus column
NaN pattern.

## DEFAULT_DIFFERENTIAL_LOSS

The default 4×4 asymmetric loss matrix (rows = action, columns = truth state, zero diagonal):

```
                truth:gained   truth:reduced   truth:unchanged   truth:bothneg
action:gained         0             10               3                3
action:reduced       10              0               3                3
action:unchanged      5              5               0                1
action:bothneg        5              5               1                0
```

Per-cell justification:

| Cell           | Value | Rationale                                                                                          |
|----------------|-------|----------------------------------------------------------------------------------------------------|
| Diagonal       | 0     | Correct call costs nothing.                                                                        |
| Direction-flip | 10    | Calling `gained` when truth is `reduced` (or vice versa) is the most expensive error — 2× missed-hit because reversing a published direction triggers downstream zero-information follow-ups.|
| Over-claim     | 3     | Calling `gained` / `reduced` when truth is `unchanged` (or `both_negative`) wastes one validation slot but does not actively mislead the literature.|
| Missed-hit     | 5     | Calling `unchanged` / `both_negative` when truth is `gained` / `reduced` permanently buries a real signal — the protein never enters the validation queue at all.|
| Conservative-default | 1 | `unchanged ↔ both_negative` confusion: both calls treat the protein as a non-interactor downstream — cheap.|

The matrix is exported as a Julia constant:

```julia
using BayesInteractomics
DEFAULT_DIFFERENTIAL_LOSS    # 4×4 Matrix{Float64}, zero diagonal
DECISION_RISK_ACTIONS        # [:gained, :reduced, :unchanged, :both_negative]
```

## Output columns

Decision Risk appends **six columns** to every per-pair result table (the legacy 2-group
`results` DataFrame, and each entry of `pairwise_results::Dict` for k≥3):

| Column                  | Type      | Description                                                                                          |
|-------------------------|-----------|------------------------------------------------------------------------------------------------------|
| `optimal_call`          | `Symbol`  | The Bayes-optimal action. One of `:gained`, `:reduced`, `:unchanged`, `:both_negative`, OR `:condition_a_specific` / `:condition_b_specific` for the out-of-matrix rows. |
| `decision_risk`         | `Float64` | Expected loss of `optimal_call`. `NaN` for `CONDITION_A_SPECIFIC` / `CONDITION_B_SPECIFIC` rows.     |
| `risk_gained`           | `Float64` | Expected loss of calling `:gained`. `NaN` for out-of-matrix rows.                                    |
| `risk_reduced`          | `Float64` | Expected loss of calling `:reduced`. `NaN` for out-of-matrix rows.                                   |
| `risk_unchanged`        | `Float64` | Expected loss of calling `:unchanged`. `NaN` for out-of-matrix rows.                                 |
| `risk_both_negative`    | `Float64` | Expected loss of calling `:both_negative`. `NaN` for out-of-matrix rows.                             |

The eltype of the four `risk_<class>` columns is plain `Float64` (NOT `Union{Missing, Float64}`):
`NaN` is the chosen sentinel for out-of-matrix rows, matching the omnibus column pattern
and keeping the schema flat for downstream filtering.

One additional **provenance column** is also emitted on the wide `results` DataFrame:

| Column                  | Type      | Description                                                                                          |
|-------------------------|-----------|------------------------------------------------------------------------------------------------------|
| `loss_matrix_default`   | `Bool`    | Per-row flag. `true` when the active loss matrix equals `DEFAULT_DIFFERENTIAL_LOSS` element-wise; `false` when the user supplied a custom matrix via `DifferentialConfig.loss_matrix` or the `loss_matrix=` kwarg. |

`loss_matrix_default` lives as a **column on `results`**, not as a field on `DifferentialResult` —
this preserves the byte-equality contract and avoids cascading another
backward-compat constructor variant onto the 20-arg `DifferentialResult` ctor.

**k≥3 wide-table aggregate columns.** For k-group `differential_analysis(; conditions, ...)` calls,
the per-pair Decision Risk lives inside `pairwise_results::Dict` entries (six columns each, as
above). The wide `results::DataFrame` additionally carries two aggregate columns summarising the
per-pair confusion across all contrasts:

| Column                  | Type      | Description                                                                                          |
|-------------------------|-----------|------------------------------------------------------------------------------------------------------|
| `decision_risk_min`     | `Float64` | `minimum(decision_risk across pairs)` per protein. The **worst per-pair confusion** — i.e. the cheapest pair to validate. |
| `optimal_call_min`      | `Symbol`  | The call achieving the min. For k=2 (degenerate k-group entry) `decision_risk_min == decision_risk` element-wise (schema uniformity). |

For legacy 2-group calls the wide `results` table IS the per-pair table, so all six per-pair
columns + `loss_matrix_default` appear directly on it, and `pairwise_results` is `nothing`.

## Overriding the loss matrix

You can supply a custom 4×4 loss matrix in two ways:

```julia
using BayesInteractomics

# (a) Via DifferentialConfig
cfg = DifferentialConfig(; loss_matrix = my_4x4)
diff = differential_analysis(ar_wt, ar_mut; config = cfg)

# (b) Per-call kwarg (overrides DifferentialConfig if both are set)
diff = differential_analysis(ar_wt, ar_mut; loss_matrix = my_4x4)

# (c) Use the package default (DEFAULT_DIFFERENTIAL_LOSS)
diff = differential_analysis(ar_wt, ar_mut)
@assert diff.results.loss_matrix_default[1] == true
```

The default top-N for the [Validation Candidates pane](#validation-candidates) is also
configurable via `DifferentialConfig.validation_candidates_top_n::Int = 20`.

**Validation rules** (raised as `ArgumentError`):

- Must be 4×4.
- Diagonal must be exactly zero.
- All entries must be `>= 0`.
- All entries must be `< Inf` (finite).

The same validation runs at both entry points (`DifferentialConfig` constructor AND the
`loss_matrix=` kwarg path) so a per-call override cannot bypass the constructor check.

## Validation Candidates

The interactive `differential_report.html` ships a dedicated **Validation Candidates** pane —
a second-level Bootstrap pill **inside the Results tab** sitting next to the Full Table pill:

```
Results  →  [Full Table]  [Validation Candidates]
```

The pane applies three transformations to the wide `results::DataFrame`:

1. **Pre-filter on omnibus significance.** Only rows where
   `differential_BFDR_omnibus <= config.bfdr_threshold` are kept (the "any-difference
   across all k conditions" family). For legacy 2-group output where `differential_BFDR_omnibus`
   is identity over `differential_BFDR` (k=2 degenerate case), this is equivalent to filtering
   on `differential_BFDR <= bfdr_threshold`.
2. **Rank by ascending Decision Risk.** Sort by `decision_risk` for legacy 2-group calls, or by
   `decision_risk_min` for k≥3 calls. Lower = cheaper to validate.
3. **Render the top-N as cards above a filtered DataTable.** The top-N (default 20, configurable
   via `DifferentialConfig.validation_candidates_top_n`) appears as a card row at the top of the
   pane; the full ranked list lives in the DataTable below.

**`BOTH_NEGATIVE` exclusion.** Rows where the (existing 6-class) `classification` column equals
`BOTH_NEGATIVE` are excluded from BOTH the top-N card grid AND the default-shown DataTable.
They remain available in a collapsible `<details>` panel inside the
pane so they are auditable but do not dilute the validation queue.

**MAP class column.** For k≥3 calls, the MAP class column in the Validation Candidates pane is
populated from `kgroup_class::Symbol` (the 5-class coarse enum). For legacy 2-group calls
it is populated from the existing 6-class `classification` column.

## Multi-Condition Decision Risk heatmap (k≥3 only)

For k-group calls (k≥3 conditions with k(k−1)/2 ≥ 2 pairwise contrasts), the **Multi-Condition tab**
of `differential_report.html` carries a Plotly heatmap that visualises the top-20 cheapest-to-validate
proteins across every pair:

| Axis     | Encoding                                                                                                  |
|----------|-----------------------------------------------------------------------------------------------------------|
| Rows     | Top-20 proteins by `decision_risk_min`.                                                                   |
| Columns  | Pairs in `diff.contrasts` order.                                                                          |
| Colorscale | **Reversed viridis** — low values bright, high values dark. Bright = cheapest pair to validate.         |

When no proteins pass the omnibus BFDR pre-filter, the heatmap is suppressed and an info alert
appears in its place:

```html
<div class="alert alert-info">
  No proteins pass the omnibus BFDR pre-filter — Decision Risk heatmap suppressed.
</div>
```

The heatmap is suppressed for k=2 calls (single pair → a one-column heatmap is not informative).
Its palette is deliberately disjoint from the §7a classification-colour map (`CLS_COLOR`)
so the two visualisations do not interfere when displayed side-by-side.

## Recommended-call badge

The DataTable in both the Results tab and the Validation Candidates pane renders `optimal_call`
with a recommended-call badge whenever the Bayes-optimal call disagrees with the MAP classification:

- **When `optimal_call != map_class`** (where `map_class = lowercase(classification)`) — the cell
  renders as a Bootstrap `.badge .text-bg-warning` with a `data-bs-toggle="tooltip"` carrying
  the text `"MAP: <map>, Optimal: <opt> (risk saved: <delta>)"`. The `<delta>` is the difference
  between `risk_<map_class>` and `decision_risk` — i.e. how much expected loss the user would
  incur by following the MAP call instead of the Bayes-optimal call.
- **When `optimal_call == map_class`** — the cell renders as plain text (no badge).

This UI reuses the existing report tooltip wiring; no new JS plumbing was added.

## FDR family interaction

Decision Risk reads **one** FDR family for the [Validation Candidates pre-filter](#validation-candidates):
`differential_BFDR_omnibus` — the "any-difference across all k conditions" family.

The `differential_BFDR_pairwise_BH` family is **independent** of `differential_BFDR_omnibus`
and is NOT cross-corrected against it. Each FDR family answers a different scientific question:

| FDR family                       | Question                                                                                                         |
|----------------------------------|------------------------------------------------------------------------------------------------------------------|
| `differential_BFDR_pairwise_BH`  | "Is this protein different in any one specified pair (controlling FDR across `n_proteins × n_contrasts`)?"      |
| `differential_BFDR_omnibus`      | "Is this protein different across all k conditions, i.e. any-difference test (controlling FDR across `n_proteins`)?" |

Decision Risk picks `differential_BFDR_omnibus` for the Validation Candidates pre-filter because the
omnibus is the natural any-difference triage step — once a protein clears the omnibus, Decision
Risk ranks _within_ the survivors. The pairwise BH family remains the right choice for
"which specific contrast drives the signal" downstream questions (per-pair known-interactor
recovery reads it).

## Caveats and reproducibility

**Results are CONDITIONAL on the chosen loss matrix.** Different labs with different validation
costs would rationally pick different `optimal_call` values. The `optimal_call` column is NOT
a model-independent probability — it is a decision rule under a stated loss structure.

For full reproducibility, **report the loss matrix used**. The `loss_matrix_default::Bool` column
makes this auditable on a per-row basis: rows where `loss_matrix_default == true` were computed
under [`DEFAULT_DIFFERENTIAL_LOSS`](#default-differential-loss), and rows where
`loss_matrix_default == false` were computed under a user-supplied override. When publishing
Decision Risk results based on a custom loss matrix, include the matrix entries in the methods
section.

**Optimal-call distribution sanity check.** If a Decision Risk run is dominated by
`optimal_call = :unchanged` (or `:both_negative`) across most of the survivors, the loss matrix
may be too conservative on the missed-hit / over-claim ratio. Revisit the off-diagonal entries —
in particular the missed-hit cost (default 5) and the over-claim cost (default 3). The default
ratio of 5:3 reflects a moderate preference for false-positive over false-negative; raising the
missed-hit cost relative to over-claim will push the optimal-call distribution toward more
`:gained` / `:reduced` calls.

## Worked example

```julia
using BayesInteractomics, DataFrames

# Standard per-condition Bayesian analysis (omitted for brevity — see the
# Tutorial / Analysis Pipeline pages for the load_data + run_analysis path).
results_wt,  ar_wt  = run_analysis(cfg_wt)
results_mut, ar_mut = run_analysis(cfg_mut)

# (1) Run the legacy 2-group differential analysis under the default loss matrix.
diff = differential_analysis(ar_wt, ar_mut)
@assert all(diff.results.loss_matrix_default)   # all rows under DEFAULT_DIFFERENTIAL_LOSS

# (2) Inspect the new Decision Risk columns.
first(diff.results[:, [:protein, :classification, :optimal_call, :decision_risk,
                       :risk_gained, :risk_reduced, :risk_unchanged, :risk_both_negative,
                       :loss_matrix_default]], 10)

# (3) Filter to the top-20 validation candidates (mirrors the report UI).
validated = filter(:differential_BFDR_omnibus => <=(0.01), diff.results)
sort!(validated, :decision_risk)
top20 = first(validated, 20)

# (4) Rebuild the loss matrix with a 2× direction-flip penalty (publication-cost lab)
#     and rerun. Watch the optimal_call distribution shift toward more conservative calls.
expensive_flip = copy(BayesInteractomics.DEFAULT_DIFFERENTIAL_LOSS)
expensive_flip[1, 2] = 20.0      # action:gained, truth:reduced
expensive_flip[2, 1] = 20.0      # action:reduced, truth:gained

diff_strict = differential_analysis(ar_wt, ar_mut; loss_matrix = expensive_flip)
@assert all(.!diff_strict.results.loss_matrix_default)   # all rows under override

# Compare optimal_call distributions.
combine(groupby(diff.results,        :optimal_call), nrow => :default_count)
combine(groupby(diff_strict.results, :optimal_call), nrow => :strict_count)
```

For k≥3 conditions the worked example is structurally identical — pass `conditions = (...)` to
`differential_analysis` and inspect `diff.pairwise_results[:wt => :mut1]` for per-pair Decision
Risk; the wide `diff.results` table additionally carries `decision_risk_min` and `optimal_call_min`
aggregating across the pairs.

## Deferred (v1.3.0+)

BayesInteractomics ships per-pairwise-contrast Decision Risk. The following extensions are
planned for a future release:

- **DRISK-K — Native k-group action set.** v1.2.0 computes Decision Risk per pairwise contrast
  (k(k−1)/2 pairs). A native action set over `2^k − 1` enriched subsets (e.g. "enriched in
  {wt, mut1} ∧ depleted in {mut2}") requires a richer loss function with subset semantics and is
  deferred to v1.3.
- **Loss-matrix elicitation UI.** An interactive 4×4 matrix editor in the report HTML so users
  can rebuild the matrix without editing Julia. Out of scope for v1.2.0; users override via the
  `DifferentialConfig.loss_matrix` field or the `loss_matrix=` kwarg.
- **Per-BMA-model Decision Risk.** Computing risk under Copula and 3c-EM separately and applying
  BMA stacking weights to the **risks** themselves. Out of scope; the existing BMA posterior is
  the input to Decision Risk via γ-PEP renormalisation, so the BMA averaging is already baked in
  upstream.

## API Reference

```@docs
compute_decision_risk!
compute_decision_risk
DEFAULT_DIFFERENTIAL_LOSS
DECISION_RISK_ACTIONS
```
