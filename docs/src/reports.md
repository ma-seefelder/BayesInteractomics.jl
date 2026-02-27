# Interactive HTML Reports

## Overview

BayesInteractomics can generate self-contained, interactive HTML reports that combine results tables, diagnostic plots, and analysis metadata into a single file for easy sharing and review.

Two report types are available:

1. **Standard analysis report**: Summary of a single `run_analysis` run
2. **Differential analysis report**: Summary of a `differential_analysis` comparison

## Standard Analysis Report

Generated automatically when `generate_report_html = true` in `CONFIG` (default):

```julia
config = CONFIG(
    # ... analysis parameters ...
    generate_report_html = true,
    output = OutputFiles("/path/to/results")
)

final_df, result = run_analysis(config)
# Report saved to config.output.report_file (default: interactive_report.html)
```

### Manual Generation

```julia
generate_report(final_results_df, config)

# Custom output path
generate_report(final_results_df, config, output = "my_report.html")
```

### Report Contents

The standard report includes:

- **Results table**: Sortable, filterable table of all proteins with posterior probabilities, Bayes factors, q-values, and log₂FC values
- **Volcano plot**: Interactive version with hover information
- **Convergence plot**: EM algorithm convergence diagnostics
- **Evidence plot**: Distribution of individual and combined Bayes factors
- **Methods section**: Auto-generated description of the statistical methods used
- **Diagnostic flags**: Per-protein quality indicators (when diagnostics are enabled)
- **Sensitivity metrics**: Prior sensitivity ranges (when sensitivity analysis is enabled)

## Differential Analysis Report

Generated automatically by the `differential_analysis` pipeline when `generate_report_html = true` in `DifferentialConfig`:

```julia
diff = differential_analysis(config_A, config_B,
    condition_A = "WT",
    condition_B = "Mutant",
    config = DifferentialConfig(generate_report_html = true)
)
```

### Manual Generation

```julia
generate_differential_report(diff)

# Custom output path
generate_differential_report(diff, output = "diff_report.html")
```

### Report Contents

The differential report includes:

- **Differential results table**: All proteins with dBF, Δlog₂FC, classifications
- **Volcano plot**: Differential Bayes factors vs. significance
- **Evidence plot**: Per-evidence-type differential Bayes factors
- **Classification summary**: Bar chart of gained/lost/unchanged interactions
- **Condition comparison**: Side-by-side scatter plots

## Customization

Reports are generated as self-contained HTML files with embedded CSS and JavaScript. Table filtering and plot-selection synchronization are built in.

### Output Paths

Output paths are controlled through the `OutputFiles` struct:

```julia
output = OutputFiles("/path/to/results")
output.report_file          # Standard report path
output.report_methods_file  # Methods markdown path
```

## API Reference

```@docs
generate_report
generate_differential_report
```
