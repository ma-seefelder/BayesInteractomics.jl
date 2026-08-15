# CLI + YAML Interface for BayesInteractomics

## Motivation

BayesInteractomics.jl currently requires Julia knowledge.
A standalone command-line tool with YAML configuration would:

- Remove the Julia barrier entirely (install binary, write YAML, run)
- Attract Python/R users who are comfortable with CLI workflows
- Integrate into existing bioinformatics pipelines (Snakemake, Nextflow)
- Work on HPC clusters (sbatch scripts)
- Enable distribution via **Bioconda** (primary discovery channel for proteomics tools)

## Usage

```bash
bayesinteractomics run config.yaml
bayesinteractomics validate config.yaml   # check config without running
bayesinteractomics --help
```

## YAML Structure

The YAML mirrors the `@interactomics` macro DSL.

### Single analysis

```yaml
type: single

protocols:
  - file: data.xlsx
    experiments:
      - { samples: [3,4,5], controls: [6,7,8] }

bait: HTT
output: ./results
method: bma
```

### Meta-analysis (multiple protocols)

```yaml
type: single

protocols:
  - file: dataset.xlsx
    experiments:
      - { samples: [2,3,4],  controls: [14,15,16] }
      - { samples: [5,6,7],  controls: [17,18,19] }
  - file: dataset.xlsx
    experiments:
      - { samples: [29,30,31], controls: [26,27,28] }

bait: ENSP00000347184
bait_id: 237
output: ./results
metalearner_path: metalearners/HistGradientBoosting_tune.jld2
normalise_protocols: false
mnar_variance_recovery: multi_impute
mnar_m: 3
```

### Differential analysis

```yaml
type: differential

conditions:
  wtHTT:
    protocols:
      - file: dataset.xlsx
        experiments:
          - { samples: [2,3,4],       controls: [14,15,16] }
          - { samples: [5,6,7],       controls: [17,18,19] }
      - file: dataset.xlsx
        experiments:
          - { samples: [29,30,31],    controls: [26,27,28] }
      - file: dataset.xlsx
        experiments:
          - { samples: [36,37,38,39], controls: [32,33,34,35] }
    bait: ENSP00000347184
    bait_id: 237
    output: ./results/wtHTT
    mnar_variance_recovery: multi_impute
    mnar_m: 3

  mHTT:
    protocols:
      - file: dataset.xlsx
        experiments:
          - { samples: [8,9,10],      controls: [14,15,16] }
          - { samples: [11,12,13],    controls: [17,18,19] }
      - file: dataset.xlsx
        experiments:
          - { samples: [],            controls: [26,27,28] }
    bait: ENSP00000347184
    bait_id: 237
    output: ./results/mHTT
    mnar_variance_recovery: multi_impute
    mnar_m: 3

compare:
  conditions: [wtHTT, mHTT]
  results_file: ./results/differential_results.xlsx
  volcano_file: ./results/differential_volcano.svg
```

### k-group differential

```yaml
type: differential

conditions:
  wt:
    protocols:
      - file: wt.xlsx
        experiments:
          - { samples: [3,4,5], controls: [6,7,8] }
    bait: HTT
    output: ./results/wt

  mut1:
    protocols:
      - file: mut1.xlsx
        experiments:
          - { samples: [3,4,5], controls: [6,7,8] }
    bait: HTT
    output: ./results/mut1

  mut2:
    protocols:
      - file: mut2.xlsx
        experiments:
          - { samples: [3,4,5], controls: [6,7,8] }
    bait: HTT
    output: ./results/mut2

compare:
  conditions: [wt, mut1, mut2]
  contrasts: all_pairs
```

## Implementation Plan

### 1. YAML parser → CONFIG mapping (~1 day)

- Add YAML.jl dependency
- `src/cli/config.jl`: parse YAML → call existing macro runtime helpers
  (`_pad_protocol_dicts!`, `_count_real_replicates`) → construct `CONFIG`
- Reuse keyword alias map (`bait` → `poi`, `bait_id` → `refID`, etc.)
- Validate required fields, emit clear error messages

### 2. CLI entry point (~0.5 day)

- `src/cli/main.jl`: argument parsing (run / validate / help)
- Subcommands: `run`, `validate`, `version`
- Exit codes: 0 success, 1 config error, 2 runtime error

### 3. PackageCompiler.jl standalone binary (~1 day)

- `build/build.jl`: PackageCompiler script
- Collect precompile statements from example runs
- Target: single binary, no Julia installation required
- Expected size: ~300-500 MB (Julia runtime + dependencies)

### 4. Distribution (~0.5 day)

- Bioconda recipe (primary target audience)
- GitHub Releases with prebuilt binaries (Linux x86_64, macOS ARM/x86)
- Docker image as fallback

### Total estimate: ~3 days

## Design Notes

- The YAML→CONFIG path shares logic with the `@interactomics` macro:
  same padding, same counting, same keyword aliases.
  Factor shared logic into `src/config/common.jl` to avoid duplication.
- All CONFIG fields are valid YAML keys (snake_case).
  Symbols (`:bma`, `:multi_impute`) become unquoted YAML strings.
- Auto-padding (missing experiments, width matching) works identically
  to the macro path — column index 0 sentinel for missing data.
