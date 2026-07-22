# =============================================================================
# diff_6condition_from_cache.jl — build the 6-condition k-group differential
# PURELY from the six cached AnalysisResults (no data loading / curation /
# imputation), then generate the differential report into diff_6condition/.
#
# Low-allocation by design: it only deserialises the six AR caches and runs
# differential_analysis + generate_differential_report — staying under the
# Julia-1.12/Windows GC threshold that the full capstone trips.
#
# Prereq: all six <OUTBASE>/<cond>/ar_cache.jld2 exist (the four baits from prior
# runs; the two mutants from examples/compute_hap40_mutants.jl).
#
# Env: OUTBASE (default BASEPATH default HTT_meta).
# =============================================================================
import Pkg
Pkg.activate(@__DIR__)

using Flux, MLJ, MLJScikitLearnInterface, HDF5
using GLM
using UMAP, Clustering                             # embeddings ext (condition similarity)
using BayesInteractomics
using Graphs, SimpleWeightedGraphs, GraphPlot, Compose, Cairo

const BASEPATH = get(ENV, "BASEPATH", "C:/Users/Manuel/Desktop/HTT_meta/")
const OUTBASE  = get(ENV, "OUTBASE", BASEPATH)
@info "diff_6condition_from_cache" OUTBASE threads=Threads.nthreads()

function _load_ar(label)
    f = joinpath(OUTBASE, label, "ar_cache.jld2")
    ar = load_result(f)
    ar === nothing && error("Missing/invalid AR cache for $label at $f — run the per-condition compute first.")
    @info "loaded AR" label f
    return ar
end

ar_wtHTT       = _load_ar("wtHTT")
ar_mHTT        = _load_ar("mHTT")
ar_hap40_strep = _load_ar("HAP40_Strep")
ar_hap40_gst   = _load_ar("GST_HAP40")
ar_hap40_delta = _load_ar("HAP40_delta")
ar_hap40_k4    = _load_ar("HAP40_K4")

diff_base = joinpath(OUTBASE, "diff_6condition")
mkpath(diff_base)

@info "6-condition k-group differential analysis"
diff6 = differential_analysis(;
    conditions = (
        wtHTT       = ar_wtHTT,
        mHTT        = ar_mHTT,
        HAP40_Strep = ar_hap40_strep,
        GST_HAP40   = ar_hap40_gst,
        HAP40_delta = ar_hap40_delta,
        HAP40_K4    = ar_hap40_k4,
    ),
    contrasts = :all_pairs,
    config = DifferentialConfig(
        results_file        = joinpath(diff_base, "differential_results.xlsx"),
        volcano_file        = joinpath(diff_base, "differential_volcano.svg"),
        evidence_file       = joinpath(diff_base, "differential_evidence.svg"),
        scatter_file        = joinpath(diff_base, "differential_scatter.svg"),
        classification_file = joinpath(diff_base, "differential_classification.svg"),
        ma_file             = joinpath(diff_base, "differential_ma.svg"),
    ),
    multi_test_method = :bh,
    parallel_pairs    = :auto,
)

@info "Generating 6-condition differential report"
generate_differential_report(diff6; output = joinpath(diff_base, "differential_report.html"))
@info "6-condition differential complete" outdir = diff_base
