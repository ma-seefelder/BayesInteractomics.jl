include("src/BayesInteractomics.jl")
using .BayesInteractomics

println("\n=== Testing EM Diagnostics Caching ===\n")

basepath = "C:/Users/Manuel/Desktop/HAP40_interactome_enrichment/wtHAP40"

wtHAP40_config = BayesInteractomics.CONFIG(
    datafile = ["data/GST_HAP40.xlsx","data/HAP40_Strep.xlsx"],
    control_cols = [
        Dict(1 => [2,3,4], 2 => [5,6,7], 3 => [8,9,10]),
        Dict(1 => [2,3,4], 2 => [5], 3 => [6,7])
    ],
    sample_cols = [
        Dict(1 => [11,12,13], 2 => [14,15,16], 3 => [17,18,19]),
        Dict(1 => [8,9,10], 2 => [11,12,13], 3 => [14,15])
    ],
    poi = "9606.ENSP00000479624",
    normalise_protocols = true,
    output = BayesInteractomics.OutputFiles(basepath, image_ext=".svg"),
    n_controls = 15,
    n_samples = 17,
    refID = 1,
    plotHBMdists = false,
    plotlog2fc = false,
    plotregr = false,
    plotbayesrange = false,
    verbose = false,
    vc_legend_pos = :topleft,
    metalearner_path = "metalearners/HistGradientBoosting_tune.jld2"
)

# Run analysis (should load from cache since we've run it before)
println("Running analysis (may load from cache)...")
wtHAP40 = BayesInteractomics.run_analysis(wtHAP40_config)

# Extract analysis result
final_results, analysis_result = wtHAP40

println("\n=== EM Result Information ===")
println("EM Converged: ", analysis_result.em.has_converged)
println("Final π₀: ", round(analysis_result.em.π0, digits=4))
println("Final π₁: ", round(analysis_result.em.π1, digits=4))
println("Number of EM iterations: ", size(analysis_result.em.logs, 1))

println("\n=== EM Diagnostics Information ===")
if isnothing(analysis_result.em_diagnostics)
    println("❌ em_diagnostics is nothing (not cached)")
else
    println("✓ em_diagnostics is available")
    println("  Diagnostics DataFrame size: ", size(analysis_result.em_diagnostics))
    println("  Columns: ", names(analysis_result.em_diagnostics))
end

if isnothing(analysis_result.em_diagnostics_summary)
    println("❌ em_diagnostics_summary is nothing (not cached)")
else
    println("✓ em_diagnostics_summary is available")
    println("  Summary keys: ", keys(analysis_result.em_diagnostics_summary))
    println("\nDiagnostics Summary:")
    for (k, v) in pairs(analysis_result.em_diagnostics_summary)
        println("  $k: $v")
    end
end

# Check if plot file exists
plot_file = wtHAP40_config.output.em_diagnostics_file
if isfile(plot_file)
    println("\n✓ EM diagnostics plot saved to: $plot_file")
else
    println("\n❌ EM diagnostics plot NOT found at: $plot_file")
end

println("\n=== Test Complete ===\n")
