# Quality-Gate-Matrix im Interactive Report

## Überblick

Die Quality-Gate-Matrix ist ein 3×3-Diagnosewerkzeug im **Validation-Tab** des interaktiven HTML-Reports. Sie überprüft, ob die vom EM-Algorithmus geschätzten Mischungskomponenten die beobachteten Bayes-Faktor-Verteilungen korrekt beschreiben. Die Matrix wird von der Funktion `run_quality_gates()` in `src/diagnostics/copula_diagnostics.jl` berechnet.

---

## Aufbau der 3×3-Matrix

|                    | **H0 (Hintergrund)** | **Agnostisch** | **H1 (Interaktion)** |
|--------------------|-----------------------|----------------|----------------------|
| **Enrichment**     | Zelle (1,1)           | Zelle (1,2)    | Zelle (1,3)          |
| **Correlation**    | Zelle (2,1)           | Zelle (2,2)    | Zelle (2,3)          |
| **Detection**      | Zelle (3,1)           | Zelle (3,2)    | Zelle (3,3)          |

### Zeilen: Die drei Evidenz-Ströme (Marginals)

Jede Zeile entspricht einem der drei statistischen Modelle, die in BayesInteractomics pro Protein einen individuellen Bayes-Faktor berechnen:

1. **Enrichment** (Anreicherung) — stammt aus dem **Hierarchical Bayesian Model (HBM)**
   - Misst, ob ein Protein in den Proben gegenüber den Kontrollen angereichert ist (log2 Fold Change)
   - Berechnet in `src/model_fitting.jl`
   - Bayes-Faktor: Vergleich der Posteriori-Wahrscheinlichkeit für |log2FC| > Schwellenwert unter H1 vs. H0

2. **Correlation** (Korrelation) — stammt aus dem **Bayesianischen linearen Regressionsmodell**
   - Misst, ob ein Protein eine Dosis-Wirkungs-Beziehung zeigt (Korrelation zwischen Dosis und Abundanz)
   - Berechnet in `src/model_fitting.jl` (Regressionsmodell mit JZS-Prior)
   - Bayes-Faktor: Vergleich eines Modells mit Steigung ≠ 0 vs. Steigung = 0

3. **Detection** (Detektion) — stammt aus dem **Beta-Bernoulli-Modell**
   - Misst, ob ein Protein häufiger in den Proben als in den Kontrollen detektiert wird
   - Berechnet in `src/betabernoulli.jl`
   - Bayes-Faktor: Vergleich der Detektionswahrscheinlichkeiten (θ_sample vs. θ_control)

### Spalten: Die drei Mischungskomponenten

Die Spalten stammen aus dem **3-Komponenten Latent-Class EM-Modell** (`LatentClassResult` in `src/combination/latent_class.jl`), das alle Proteine in drei Klassen einteilt:

1. **H0 (Hintergrund/Background)** — Proteine, die keine echte Interaktion mit dem Köderprotein haben
   - Typischerweise niedrige Bayes-Faktoren über alle drei Evidenz-Ströme
   - Größte Komponente (oft 70–90% aller Proteine)
   - Mixing weight: `π₀`

2. **Agnostisch** — Proteine mit ambivalenter Evidenz
   - Übergangskomponente zwischen H0 und H1
   - Fängt Proteine auf, die in einem oder zwei Evidenz-Strömen Hinweise auf Interaktion zeigen, aber nicht konsistent über alle drei
   - Mixing weight: `π_ag`

3. **H1 (Interaktion)** — Proteine, die als echte Interaktionspartner klassifiziert werden
   - Typischerweise hohe Bayes-Faktoren über alle drei Evidenz-Ströme
   - Kleinste Komponente (oft < 10% der Proteine)
   - Mixing weight: `π₁`

Die Zuordnung eines Proteins zu einer Komponente erfolgt über die **Responsibility-Matrix** (n_Proteine × 3), die der EM-Algorithmus schätzt. Jede Zeile gibt die Wahrscheinlichkeit an, mit der ein Protein zu H0, Agnostisch oder H1 gehört.

---

## Was wird in jeder Zelle gemessen?

Jede der 9 Zellen enthält einen **Kolmogorov-Smirnov (KS) Goodness-of-Fit-Test**. Dieser prüft, ob die Verteilung der log-Bayes-Faktoren einer bestimmten Evidenz-Dimension (Zeile), eingeschränkt auf die Proteine einer bestimmten Komponente (Spalte), gut durch eine parametrische Verteilung beschrieben wird.

### Berechnungsschritte pro Zelle

1. **Protein-Selektion**: Proteine mit Responsibility > 0.1 für die jeweilige Komponente werden ausgewählt (Fallback: Argmax-Zuweisung, falls < 5 Proteine)

2. **Log-Transformation**: Die Bayes-Faktoren werden log-transformiert: `log(max(BF, 1e-300))`

3. **Normalverteilungs-Fit**: Eine Normalverteilung wird an die log-BF-Werte der selektierten Proteine angepasst: `Normal(μ̂, σ̂)`

4. **KS-Statistik**: Die maximale Abweichung zwischen empirischer und theoretischer Verteilungsfunktion wird berechnet:
   ```
   D = max_i |F_n(x_i) - F(x_i)|
   ```
   wobei `F_n` die empirische CDF und `F` die gefittete CDF ist.

5. **Status-Bewertung**:
   - **PASS** (grün): KS < 0.10 — Die Normalverteilung beschreibt die Daten gut
   - **WARN** (gelb): 0.10 ≤ KS < 0.15 — Marginale Abweichungen, aber noch akzeptabel
   - **FAIL** (rot): KS ≥ 0.15 — Signifikante Abweichung von der Normalverteilung

6. **Auto-Remediation**: Wenn KS ≥ 0.15 (FAIL), wird automatisch eine **t-Location-Scale-Verteilung** angepasst (schwerere Schwänze als die Normalverteilung). Wenn die t-Verteilung eine bessere Anpassung liefert (KS sinkt), wird sie stattdessen verwendet. Im Report werden remediierte Zellen mit einem **Stern (*)** markiert.

### Effektive Stichprobengröße (n_effective)

Jede Zelle zeigt auch `n_effective` an — die Summe der Responsibilities für die jeweilige Komponente. Dies gibt an, wie viele Proteine "effektiv" zu dieser Komponente gehören. Bei weniger als 5 effektiven Proteinen wird der KS-Test übersprungen und die Zelle automatisch als PASS markiert.

---

## Interpretation der Ergebnisse

### Beispiel-Lesung

```
PASS (KS=0.052)   →  Die Normalverteilung beschreibt die log-BF-Verteilung
                      dieser Proteingruppe in dieser Evidenz-Dimension sehr gut.

WARN (KS=0.128)   →  Leichte Abweichungen. Eventuell gibt es Ausreißer oder
                      die Verteilung hat schwerere Schwänze als erwartet.

FAIL (KS=0.183) * →  Signifikante Abweichung. Die Auto-Remediation wurde
                      angewandt (Stern), aber die t-Verteilung konnte den
                      KS-Wert nicht unter 0.15 drücken.
```

### Gesamtstatus (Overall)

Der Gesamtstatus entspricht dem **schlechtesten** Zellenstatus:
- Wenn eine einzige Zelle FAIL ist → Overall = FAIL
- Wenn keine FAIL, aber mindestens eine WARN → Overall = WARN
- Nur wenn alle 9 Zellen PASS → Overall = PASS

### Was bedeutet ein Fehler?

| Situation | Mögliche Ursachen | Empfohlene Maßnahmen |
|-----------|-------------------|---------------------|
| H0-Spalte FAIL | Die Hintergrundverteilung ist multimodal oder stark schiefverteilt | Überprüfung der H0-Simulation; eventuell mehr Permutationen (`n_seed` erhöhen) |
| Agnostisch-Spalte FAIL | Zu wenige Proteine in der Übergangskomponente | Kann harmlos sein; überprüfen, ob die H0/Agnostisch-Merging korrekt erfolgt ist |
| H1-Spalte FAIL | H1-Komponente ist heterogen (enthält sowohl starke als auch schwache Interaktoren) | KL-Kontamination prüfen; eventuell strengere Schwellenwerte verwenden |
| Enrichment-Zeile FAIL | HBM liefert ungewöhnlich verteilte log-BF-Werte | Prüfung auf Batch-Effekte, fehlende Normalisierung oder unzureichende Daten |
| Correlation-Zeile FAIL | Regressionsmodell zeigt systematische Abweichungen | Prüfung der Dosis-Struktur; weniger als 3 Dosispunkte können problematisch sein |
| Detection-Zeile FAIL | Beta-Bernoulli-BFs sind nicht normalverteilt auf log-Skala | Kann bei vielen Proteinen mit identischer Detektion (0 oder 100%) auftreten |

---

## Ergänzende Diagnostiken im Validation-Tab

### KL-Kontamination (Balkendiagramm)

Neben der Quality-Gate-Matrix zeigt der Report ein **KL-Divergenz-Balkendiagramm**, das misst, wie stark die H1-Komponente durch Nicht-Interaktoren "verschmutzt" ist.

**Berechnung** (`compute_kl_contamination()` in `copula_diagnostics.jl`):

1. **Pure H1**: Proteine mit Responsibility > 0.95 für die H1-Komponente (hochkonfidente Interaktoren)
2. **Full H1**: Alle Proteine mit Responsibility > 0.5 für H1
3. **Für jeden Evidenz-Strom**: Normalverteilungen werden an die log-BF-Werte beider Gruppen angepasst
4. **Monte-Carlo-KL-Divergenz**: `KL(pure || full) = E_pure[log p_pure(x) - log p_full(x)]`
5. **Schwellenwert**: KL < 0.5 pro Strom → PASS (grün), KL ≥ 0.5 → FAIL (rot)
6. **Joint KL**: Summe der drei Per-Strom-Werte

**Interpretation**:
- Niedrige KL (< 0.5): Die H1-Komponente enthält hauptsächlich echte Interaktoren
- Hohe KL (≥ 0.5): Proteine mit niedrigen Bayes-Faktoren wurden der H1-Komponente zugeordnet — die Mischungsmodellseparation ist möglicherweise unzureichend

### Consistency Checklist

Eine Checkliste mit booleschen Prüfungen:

| Check | Beschreibung | Quelle |
|-------|-------------|--------|
| `all_ks_pass` | Alle 9 KS-Tests in der Quality-Gate-Matrix bestehen | Quality-Gate-Matrix (Overall = PASS) |
| `kl_pass` | Alle drei KL-Divergenzen sind < 0.5 | KL-Kontamination |
| `h1_lt_200` | Die H1-Komponente enthält weniger als 200 Proteine (Responsibility > 0.5) | LatentClassResult Responsibilities |
| `F8A1_P1` | F8A1 (bekannter HAP40-Interaktor) hat Posterior ≥ 0.999 | Ergebnisse (nur bei HAP40-Daten) |
| `HTT_P099` | HTT (Huntingtin) hat Posterior > 0.99 | Ergebnisse (nur bei HAP40-Daten) |

Die letzten beiden Checks sind **datenspezifische Anker-Validierungen** für den HAP40-Strep-Datensatz. Bei anderen Datensätzen fehlen sie in der Checkliste.

---

## Datenfluss: Von den Modellen zur Matrix

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Individuelle Modelle                            │
│                                                                        │
│  Beta-Bernoulli  →  BF_detection  (Vektor, ein Wert pro Protein)      │
│  HBM             →  BF_enrichment (Vektor, ein Wert pro Protein)      │
│  Regression      →  BF_correlation(Vektor, ein Wert pro Protein)      │
│                                                                        │
│  → Zusammengefasst als BayesFactorTriplet                             │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  3-Komponenten Latent-Class EM                         │
│                                                                        │
│  Eingabe: BayesFactorTriplet (log-transformiert)                      │
│  Ausgabe: LatentClassResult                                           │
│    - mixing_weights: [π₀, π_ag, π₁]                                  │
│    - responsibilities: Matrix (n_Proteine × 3)                        │
│    - class_parameters: μ, σ pro Komponente und Dimension              │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Quality-Gate-Matrix                                 │
│                                                                        │
│  Für jede der 9 Kombinationen (3 Marginals × 3 Komponenten):         │
│    1. Selektiere Proteine mit hoher Responsibility (> 0.1)            │
│    2. Extrahiere deren log(BF) für die jeweilige Dimension            │
│    3. Fitte Normalverteilung → berechne KS-Statistik                  │
│    4. Falls KS ≥ 0.15 → Auto-Remediation mit t-Verteilung            │
│    5. Bewerte: PASS / WARN / FAIL                                     │
│                                                                        │
│  Gesamtstatus = worst(alle 9 Zellen)                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quellcode-Referenzen

| Datei | Funktion | Beschreibung |
|-------|----------|-------------|
| `src/diagnostics/copula_diagnostics.jl:17` | `_fit_with_remediation()` | Normal-Fit + t-Location-Scale Fallback |
| `src/diagnostics/copula_diagnostics.jl:49` | `run_quality_gates()` | Berechnung der 3×3-Matrix |
| `src/diagnostics/copula_diagnostics.jl:125` | `compute_kl_contamination()` | KL-Divergenz zwischen pure/full H1 |
| `src/diagnostics/types.jl:360` | `QualityGateCell` | Typ-Definition einer einzelnen Zelle |
| `src/diagnostics/types.jl:376` | `QualityGateResult` | Typ-Definition des Gesamtergebnisses |
| `src/diagnostics/types.jl:388` | `KLContaminationResult` | Typ-Definition der KL-Kontamination |
| `src/diagnostics/types.jl:415` | `ValidationResult` | Aggregierter Validation-Typ |
| `src/combination/copula.jl:960` | `_ks_statistic()` | KS-Statistik-Berechnung |
| `src/analysis/pipeline.jl:1763` | `_run_validation()` | Orchestrierung aller Validierungsschritte |
| `src/reports/report_generator.jl:701` | `_build_validation_json()` | Serialisierung für HTML-Report |
| `src/reports/templates/report.html:3541` | `initValidationTab()` | JavaScript-Rendering im Browser |

---

## Zusammenfassung

Die Quality-Gate-Matrix prüft die **interne Konsistenz** des Mischungsmodells: Wenn die EM-basierte Klassifikation der Proteine in H0/Agnostisch/H1 korrekt ist, sollten die log-Bayes-Faktoren innerhalb jeder Komponente für jede Evidenz-Dimension annähernd normalverteilt sein. Ein PASS in allen 9 Zellen bedeutet, dass das Mischungsmodell die Daten kohärent beschreibt und die resultierenden Posterior-Wahrscheinlichkeiten vertrauenswürdig sind. Abweichungen (WARN/FAIL) deuten auf potenzielle Modellverletzungen hin, die bei der Interpretation der Ergebnisse berücksichtigt werden sollten.
