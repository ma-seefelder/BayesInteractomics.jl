println("Starting test...")
flush(stdout)

println("Loading package...")
flush(stdout)
include("src/BayesInteractomics.jl")
using .BayesInteractomics

println("Package loaded successfully!")
flush(stdout)

basepath = "C:/Users/Manuel/Desktop/HAP40_interactome_enrichment/wtHAP40"

println("Creating config...")
flush(stdout)

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
    verbose = true,  # Enable verbose output
    vc_legend_pos = :topleft,
    metalearner_path = "metalearners/HistGradientBoosting_tune.jld2",
    run_em_diagnostics = true,
    em_n_restarts = 3  # Reduce restarts for faster testing
)

println("Running analysis...")
flush(stdout)

wtHAP40 = BayesInteractomics.run_analysis(wtHAP40_config)

println("\n=== Analysis Complete ===")
flush(stdout)

final_results, analysis_result = wtHAP40

println("\n=== EM Diagnostics Check ===")
if isnothing(analysis_result.em_diagnostics)
    println("❌ em_diagnostics is nothing")
else
    println("✅ em_diagnostics is available!")
    println("   Size: ", size(analysis_result.em_diagnostics))
end

if isnothing(analysis_result.em_diagnostics_summary)
    println("❌ em_diagnostics_summary is nothing")
else
    println("✅ em_diagnostics_summary is available!")
    println("   Keys: ", keys(analysis_result.em_diagnostics_summary))
end

plot_file = wtHAP40_config.output.em_diagnostics_file
if isfile(plot_file)
    println("✅ Plot saved: $plot_file")
else
    println("❌ Plot NOT found: $plot_file")
end

println("\nTest complete!")
