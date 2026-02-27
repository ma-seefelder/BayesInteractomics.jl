# src/reports/methods_generator.jl
# Auto-generates manuscript-ready methods text and parameter tables.

"""
    generate_methods_text(config::CONFIG, results::DataFrame) -> String

Generate a manuscript-ready methods paragraph from the analysis configuration.
"""
function generate_methods_text(config::CONFIG, results::DataFrame)::String
    n_proteins  = nrow(results)
    n_sig       = sum(skipmissing(results.q) .≤ 0.05)
    n_strong    = sum(skipmissing(results.q) .≤ 0.01)
    method_str  = config.combination_method == :copula ? "copula-based" : "latent class"
    regr_str    = config.regression_likelihood == :robust_t ?
                  "robust (Student-t, ν = $(round(config.student_t_nu, digits=1)))" : "normal"
    pkg_version = _report_pkg_version()

    return """
AP-MS data were analyzed using BayesInteractomics v$(pkg_version) (Julia v$(VERSION)). \
A total of $(n_proteins) proteins were evaluated for interaction with the '$(config.poi)' bait protein \
using $(config.n_controls) control experiment(s) and $(config.n_samples) sample experiment(s).

Protein-protein interactions were scored by integrating evidence from three independent Bayesian models: \
(1) a Beta-Bernoulli model for detection probability across replicates, \
(2) a Hierarchical Bayesian Model (HBM) for log₂ fold-change enrichment, and \
(3) a Bayesian $(regr_str) linear regression model for dose-response correlation. \
Individual Bayes factors from the three models were combined using a $(method_str) mixture model \
fit by expectation-maximization ($(config.em_n_restarts) restarts). \
Significant interactors were defined at Bayesian FDR (q) ≤ 0.05; $(n_sig) proteins met this \
threshold ($(n_strong) at q ≤ 0.01).

Software: BayesInteractomics v$(pkg_version) (Julia v$(VERSION)).
""" |> strip
end

"""
    generate_methods_parameters(config::CONFIG) -> Vector{Pair{String,String}}

Return a list of (parameter name, value) pairs covering **all** CONFIG fields,
suitable for display in the report's Analysis Parameters table.
"""
function generate_methods_parameters(config::CONFIG)::Vector{Pair{String,String}}
    # Fields to skip (complex nested structs serialised separately or too large for a table)
    skip = Set([:output, :sensitivity_config, :diagnostics_config,
                :control_cols, :sample_cols, :em_prior])
    pairs = Pair{String,String}[]
    for fname in fieldnames(CONFIG)
        fname in skip && continue
        val = getfield(config, fname)
        s = if val isa Vector
                isempty(val) ? "[]" : "[" * join(basename.(string.(val)), ", ") * "]"
            elseif val isa AbstractFloat
                string(round(val, digits=4))
            else
                string(val)
            end
        push!(pairs, string(fname) => s)
    end
    return pairs
end

"""
    generate_reproducibility_block(config::CONFIG) -> String

Generate a complete reproducibility information block covering all CONFIG parameters.
"""
function generate_reproducibility_block(config::CONFIG)::String
    pkg_version = _report_pkg_version()
    lines = String[
        "Generated:         $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))",
        "Package:           BayesInteractomics v$(pkg_version)",
        "Julia:             $(VERSION)",
        "Output directory:  $(config.output.basedir)",
        "",
        "Input files:",
    ]
    for f in config.datafile
        push!(lines, "  $(f)")
    end
    push!(lines, "H0 file: $(config.output.H0_file)")
    push!(lines, "")
    push!(lines, "All configuration parameters:")

    skip = Set([:output, :sensitivity_config, :diagnostics_config,
                :control_cols, :sample_cols, :em_prior])
    for fname in fieldnames(CONFIG)
        fname in skip && continue
        val = getfield(config, fname)
        s = if val isa Vector
                isempty(val) ? "[]" : "[" * join(string.(val), ", ") * "]"
            elseif val isa AbstractFloat
                string(round(val, digits=6))
            else
                string(val)
            end
        push!(lines, "  $(lpad(string(fname), 30)) = $s")
    end
    return join(lines, "\n")
end

"""
    _report_pkg_version() -> String

Return BayesInteractomics package version string, or "?" if unavailable.
"""
function _report_pkg_version()::String
    try
        v = pkgversion(@__MODULE__)
        isnothing(v) ? "?" : string(v)
    catch
        "?"
    end
end
