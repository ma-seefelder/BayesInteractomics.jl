module BayesInteractomicsNetworkExt

using BayesInteractomics
using Graphs
using SimpleWeightedGraphs
using GraphPlot
using Compose
using DataFrames
using Statistics
using Colors
using Cairo
using CSV

# Stdlib imports for PPI enrichment
using Downloads
using SHA
using Dates

# Cache serialization
using JLD2

# Include extension modules
include("types.jl")
include("construction.jl")
include("ppi_query.jl")
include("ppi_enrichment.jl")
include("statistics.jl")
include("centrality.jl")
include("community.jl")
include("visualization.jl")
include("export.jl")
include("pipeline.jl")

end # module
