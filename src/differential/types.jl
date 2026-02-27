# Differential Interaction Analysis — Type Definitions
# Compare interaction profiles between two experimental conditions

import DataFrames: DataFrame, nrow
using Dates

# ----------------------- InteractionClass Enum ----------------------- #

"""
    InteractionClass

Enum classifying the differential interaction status of a protein.

# Values
- `GAINED`: Interaction significantly stronger in condition A than B (PP_A ≥ threshold, PP_B < threshold, Δlog₂FC ≥ 0)
- `REDUCED`: Interaction significantly stronger in condition B than A (PP_B ≥ threshold, PP_A < threshold, Δlog₂FC ≤ 0)
- `UNCHANGED`: Both conditions show interaction; no significant directional difference, or ambiguous evidence
- `BOTH_NEGATIVE`: Neither condition detects interaction (PP below threshold), but differential evidence is significant
- `CONDITION_A_SPECIFIC`: Protein detected only in condition A results
- `CONDITION_B_SPECIFIC`: Protein detected only in condition B results
"""
@enum InteractionClass begin
    GAINED
    REDUCED
    UNCHANGED
    BOTH_NEGATIVE
    CONDITION_A_SPECIFIC
    CONDITION_B_SPECIFIC
end

# ----------------------- DifferentialConfig ----------------------- #

"""
    DifferentialConfig

Configuration for differential interaction analysis.

# Statistical Thresholds
- `posterior_threshold::Float64`: Minimum posterior probability to consider a protein
  as a true interactor in a given condition (default: 0.8)
- `q_threshold::Float64`: Maximum q-value (BFDR) for significance in differential
  analysis (default: 0.05)
- `delta_log2fc_threshold::Float64`: Minimum absolute difference in mean_log2FC
  to classify as gained/lost when both conditions show interaction (default: 1.0)
- `dbf_threshold::Float64`: Minimum absolute log10 differential Bayes factor to
  classify as gained/lost (default: 1.0, i.e., 10-fold BF difference)
- `classification_method::Symbol`: Method for classifying interactions:
  - `:posterior` (default): use posterior probability thresholds
  - `:dbf`: use differential Bayes factor thresholds
  - `:combined`: require both posterior and dBF criteria

# Output Paths (used by the pipeline method `differential_analysis(config_a, config_b)`)
- `results_file::String`: Path for the differential results Excel file (default: `"differential_results.xlsx"`)
- `volcano_file::String`: Path for the volcano plot image (default: `"differential_volcano.png"`)
- `evidence_file::String`: Path for the evidence plot image (default: `"differential_evidence.png"`)
- `scatter_file::String`: Path for the scatter plot image (default: `"differential_scatter.png"`)
- `classification_file::String`: Path for the classification bar chart (default: `"differential_classification.png"`)
- `ma_file::String`: Path for the MA plot (default: `"differential_ma.png"`)
- `generate_report_html::Bool`: Automatically generate an interactive HTML report after analysis (default: `true`)

# Examples
```julia
# Default configuration
config = DifferentialConfig()

# Stringent thresholds with custom output paths
config = DifferentialConfig(
    posterior_threshold = 0.95,
    q_threshold = 0.01,
    classification_method = :combined,
    results_file = "results/diff.xlsx",
    volcano_file = "results/volcano.svg"
)
```
"""
Base.@kwdef struct DifferentialConfig
    # Statistical thresholds
    posterior_threshold::Float64     = 0.8
    q_threshold::Float64            = 0.05
    delta_log2fc_threshold::Float64 = 1.0
    dbf_threshold::Float64          = 1.0
    classification_method::Symbol   = :posterior

    # Output paths
    results_file::String            = "differential_results.xlsx"
    volcano_file::String            = "differential_volcano.png"
    evidence_file::String           = "differential_evidence.png"
    scatter_file::String            = "differential_scatter.png"
    classification_file::String     = "differential_classification.png"
    ma_file::String                 = "differential_ma.png"

    # Report generation
    generate_report_html::Bool      = true

    function DifferentialConfig(posterior_threshold, q_threshold,
                                delta_log2fc_threshold, dbf_threshold,
                                classification_method,
                                results_file, volcano_file,
                                evidence_file, scatter_file,
                                classification_file, ma_file,
                                generate_report_html)
        0.0 <= posterior_threshold <= 1.0 || throw(ArgumentError(
            "posterior_threshold must be in [0, 1], got $posterior_threshold"))
        0.0 < q_threshold <= 1.0 || throw(ArgumentError(
            "q_threshold must be in (0, 1], got $q_threshold"))
        delta_log2fc_threshold >= 0.0 || throw(ArgumentError(
            "delta_log2fc_threshold must be non-negative, got $delta_log2fc_threshold"))
        dbf_threshold >= 0.0 || throw(ArgumentError(
            "dbf_threshold must be non-negative, got $dbf_threshold"))
        classification_method in (:posterior, :dbf, :combined) || throw(ArgumentError(
            "classification_method must be :posterior, :dbf, or :combined, got :$classification_method"))
        new(posterior_threshold, q_threshold, delta_log2fc_threshold,
            dbf_threshold, classification_method,
            results_file, volcano_file, evidence_file, scatter_file,
            classification_file, ma_file, generate_report_html)
    end
end

function Base.show(io::IO, c::DifferentialConfig)
    println(io, "DifferentialConfig")
    println(io, "  posterior_threshold     : $(c.posterior_threshold)")
    println(io, "  q_threshold            : $(c.q_threshold)")
    println(io, "  delta_log2fc_threshold : $(c.delta_log2fc_threshold)")
    println(io, "  dbf_threshold          : $(c.dbf_threshold)")
    println(io, "  classification_method  : $(c.classification_method)")
    println(io, "  results_file           : $(c.results_file)")
    println(io, "  volcano_file           : $(c.volcano_file)")
    println(io, "  evidence_file          : $(c.evidence_file)")
    println(io, "  scatter_file           : $(c.scatter_file)")
    println(io, "  classification_file    : $(c.classification_file)")
    println(io, "  ma_file                : $(c.ma_file)")
    println(io, "  generate_report_html   : $(c.generate_report_html)")
end

# ----------------------- DifferentialResult ----------------------- #

"""
    DifferentialResult

Complete results from a differential interaction analysis comparing two conditions.

# Fields
## Result DataFrame
- `results::DataFrame`: Per-protein differential statistics. Columns:
  `Protein`, `bf_A`, `bf_B`, `dbf`, `log10_dbf`, `posterior_A`, `posterior_B`,
  `delta_posterior`, `q_A`, `q_B`, `log2fc_A`, `log2fc_B`, `delta_log2fc`,
  `bf_enrichment_A`, `bf_enrichment_B`, `dbf_enrichment`,
  `bf_correlation_A`, `bf_correlation_B`, `dbf_correlation`,
  `bf_detected_A`, `bf_detected_B`, `dbf_detected`,
  `differential_posterior`, `differential_q`, `classification`

## Condition labels
- `condition_A::String`: Label for condition A
- `condition_B::String`: Label for condition B

## Configuration
- `config::DifferentialConfig`: Configuration used for this analysis

## Metadata
- `n_proteins_A::Int`: Number of proteins in condition A
- `n_proteins_B::Int`: Number of proteins in condition B
- `n_shared::Int`: Number of shared proteins (inner join)
- `n_condition_A_specific::Int`: Proteins only in condition A
- `n_condition_B_specific::Int`: Proteins only in condition B
- `timestamp::DateTime`: When analysis was performed

## Summary counts
- `n_gained::Int`: Interactions gained (stronger in condition A)
- `n_reduced::Int`: Interactions reduced (stronger in condition B)
- `n_unchanged::Int`: Interactions unchanged between conditions
- `n_both_negative::Int`: Proteins not detected in either condition but with significant differential evidence

# Iterator Interface
Iterates over `(protein_name, row_data)` tuples.

# Indexing
- `diff[i]`: Get row i from results (Integer indexing)
- `diff[protein]`: Get row for specific protein (String indexing)

# Examples
```julia
diff = differential_analysis(result_wt, result_mut,
    condition_A = "WT", condition_B = "Mutant")

# Summary
println(diff)

# Iterate
for (protein, row) in diff
    if row.classification == GAINED
        println("\$protein gained in WT")
    end
end

# Index
row = diff["MYC"]
println("dBF: ", row.dbf)
```

See also: [`differential_analysis`](@ref), [`DifferentialConfig`](@ref), [`InteractionClass`](@ref)
"""
struct DifferentialResult
    results::DataFrame
    condition_A::String
    condition_B::String
    config::DifferentialConfig
    n_proteins_A::Int
    n_proteins_B::Int
    n_shared::Int
    n_condition_A_specific::Int
    n_condition_B_specific::Int
    timestamp::DateTime
    n_gained::Int
    n_reduced::Int
    n_unchanged::Int
    n_both_negative::Int
end

# ----------------------- Accessors ----------------------- #

"""
    getProteins(r::DifferentialResult) -> Vector{String}

Get vector of protein names from differential results.
"""
getProteins(r::DifferentialResult) = r.results.Protein

"""
    getDifferentialBayesFactors(r::DifferentialResult) -> Vector{Float64}

Get vector of differential Bayes factors (BF_A / BF_B).
"""
getDifferentialBayesFactors(r::DifferentialResult) = r.results.dbf

"""
    getDifferentialPosteriors(r::DifferentialResult) -> Vector{Float64}

Get vector of differential posterior probabilities.
"""
getDifferentialPosteriors(r::DifferentialResult) = r.results.differential_posterior

"""
    getDifferentialQValues(r::DifferentialResult)

Get vector of differential q-values (Bayesian FDR).
"""
getDifferentialQValues(r::DifferentialResult) = r.results.differential_q

"""
    getClassifications(r::DifferentialResult) -> Vector{InteractionClass}

Get vector of interaction classifications for all proteins.
"""
getClassifications(r::DifferentialResult) = r.results.classification

"""
    getDeltaLog2FC(r::DifferentialResult) -> Vector{Float64}

Get vector of delta log2 fold changes (log2FC_A - log2FC_B).
"""
getDeltaLog2FC(r::DifferentialResult) = r.results.delta_log2fc

# ----------------------- Iterator Interface ----------------------- #

Base.length(r::DifferentialResult) = nrow(r.results)

Base.iterate(r::DifferentialResult, state=1) = state > length(r) ? nothing :
    ((r.results.Protein[state], r.results[state, :]), state + 1)

Base.getindex(r::DifferentialResult, protein::String) =
    r.results[findfirst(==(protein), r.results.Protein), :]

Base.getindex(r::DifferentialResult, i::Integer) = r.results[i, :]

# ----------------------- Display ----------------------- #

function Base.show(io::IO, r::DifferentialResult)
    println(io, "DifferentialResult")
    println(io, String(repeat(Char(0x2500), 40)))
    println(io, " Condition A           : $(r.condition_A)")
    println(io, " Condition B           : $(r.condition_B)")
    println(io, " Shared proteins       : $(r.n_shared)")
    println(io, " Condition A specific  : $(r.n_condition_A_specific)")
    println(io, " Condition B specific  : $(r.n_condition_B_specific)")
    println(io, " Gained interactions   : $(r.n_gained)")
    println(io, " Reduced interactions  : $(r.n_reduced)")
    println(io, " Unchanged             : $(r.n_unchanged)")
    println(io, " Both negative         : $(r.n_both_negative)")
    println(io, " Timestamp             : $(r.timestamp)")
end
