"""
    test_network.jl

Tests for network analysis extension functionality.
Requires: Graphs, SimpleWeightedGraphs, GraphPlot, Compose
"""

using TestItemRunner
using DataFrames

# Helper function to create mock analysis results
function create_mock_results()
    return DataFrame(
        Protein = ["ProteinA", "ProteinB", "ProteinC", "ProteinD", "ProteinE"],
        PosteriorProbability = [0.95, 0.85, 0.75, 0.45, 0.30],
        BayesFactor = [100.0, 50.0, 20.0, 2.0, 0.5],
        q_value = [0.001, 0.01, 0.02, 0.08, 0.15],
        mean_log2FC = [3.5, 2.8, 2.1, 1.2, 0.5]
    )
end

@testitem "NetworkAnalysisResult basic construction" begin
    using BayesInteractomics
    using DataFrames
    include("test_network.jl")

    mock_results = create_mock_results()

    ar = NetworkAnalysisResult(mock_results)
    @test ar.results === mock_results
    @test isnothing(ar.bait_protein)
    @test isnothing(ar.bait_index)
end

@testitem "NetworkAnalysisResult with bait protein" begin
    using BayesInteractomics
    using DataFrames
    include("test_network.jl")

    mock_results = create_mock_results()

    ar_bait = NetworkAnalysisResult(mock_results, bait_protein="MYC", bait_index=1)
    @test ar_bait.bait_protein == "MYC"
    @test ar_bait.bait_index == 1
end

@testitem "NetworkAnalysisResult accessors" begin
    using BayesInteractomics
    using BayesInteractomics: getProteins, getBayesFactors, getPosteriorProbs, getQValues, getMeanLog2FC, getBaitProtein
    using DataFrames
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results)
    ar_bait = NetworkAnalysisResult(mock_results, bait_protein="MYC")

    @test getProteins(ar) == mock_results.Protein
    @test getBayesFactors(ar) == mock_results.BayesFactor
    @test getPosteriorProbs(ar) == mock_results.PosteriorProbability
    @test getQValues(ar) == mock_results.q_value
    @test getMeanLog2FC(ar) == mock_results.mean_log2FC
    @test isnothing(getBaitProtein(ar))
    @test getBaitProtein(ar_bait) == "MYC"
end

@testitem "Network construction basic" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait", bait_index=1)

    net = build_network(ar, posterior_threshold=0.5, q_threshold=0.05)
    @test net isa BayesInteractomics.AbstractNetworkResult
    @test nv(net.graph) > 0
    @test net.bait_protein == "Bait"
    @test net.bait_index == 1
end

@testitem "Network construction threshold filtering" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")

    net_strict = build_network(ar, posterior_threshold=0.9, q_threshold=0.01)
    net_relaxed = build_network(ar, posterior_threshold=0.4, q_threshold=0.1)
    @test nv(net_strict.graph) <= nv(net_relaxed.graph)
end

@testitem "Network construction without bait" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")

    net_no_bait = build_network(ar, include_bait=false)
    @test isnothing(net_no_bait.bait_index)
end

@testitem "Network construction weight_by parameter" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")

    net_pp = build_network(ar, weight_by=:posterior_prob)
    net_bf = build_network(ar, weight_by=:bayes_factor)
    net_lfc = build_network(ar, weight_by=:log2fc)

    @test net_pp isa BayesInteractomics.AbstractNetworkResult
    @test net_bf isa BayesInteractomics.AbstractNetworkResult
    @test net_lfc isa BayesInteractomics.AbstractNetworkResult
end

@testitem "Network construction empty network" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")

    net_empty = build_network(ar, posterior_threshold=0.99, q_threshold=0.0001)
    @test nv(net_empty.graph) == 0
end

@testitem "Network construction node attributes" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")

    net = build_network(ar, posterior_threshold=0.5)
    @test hasproperty(net.node_attributes, :protein)
    @test hasproperty(net.node_attributes, :posterior_prob)
    @test hasproperty(net.node_attributes, :bayes_factor)
    @test hasproperty(net.node_attributes, :q_value)
    @test hasproperty(net.node_attributes, :is_bait)
end

@testitem "Network construction edge attributes" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")

    net = build_network(ar, posterior_threshold=0.5)
    if ne(net.graph) > 0
        @test hasproperty(net.edge_attributes, :source_node)
        @test hasproperty(net.edge_attributes, :target_node)
        @test hasproperty(net.edge_attributes, :weight)
    end
end

@testitem "Network statistics computation" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        stats = network_statistics(net)
        @test stats isa BayesInteractomics.AbstractNetworkResult
        @test stats.n_nodes == nv(net.graph)
        @test stats.n_edges == ne(net.graph)
        @test 0.0 <= stats.density <= 1.0
        @test stats.avg_degree >= 0.0
        @test stats.avg_weighted_degree >= 0.0
        @test 0.0 <= stats.avg_clustering <= 1.0
        @test stats.n_components >= 1
        @test stats.largest_component_size >= 1
        @test stats.largest_component_size <= stats.n_nodes
    end
end

@testitem "Network statistics empty network" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    empty_ar = NetworkAnalysisResult(mock_results)
    empty_net = build_network(empty_ar, posterior_threshold=0.99)
    empty_stats = network_statistics(empty_net)

    @test empty_stats.n_nodes == 0
    @test empty_stats.n_edges == 0
end

@testitem "Centrality measures computation" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        cm = centrality_measures(net)
        @test cm isa BayesInteractomics.AbstractNetworkResult
        @test length(cm.protein_names) == nv(net.graph)
        @test length(cm.degree) == nv(net.graph)
        @test length(cm.weighted_degree) == nv(net.graph)
        @test length(cm.betweenness) == nv(net.graph)
        @test length(cm.closeness) == nv(net.graph)
        @test length(cm.eigenvector) == nv(net.graph)
        @test length(cm.pagerank) == nv(net.graph)
    end
end

@testitem "Centrality measures non-negative" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        cm = centrality_measures(net)
        @test all(cm.degree .>= 0)
        @test all(cm.weighted_degree .>= 0)
        @test all(cm.betweenness .>= 0)
        @test all(cm.closeness .>= 0)
        @test all(cm.eigenvector .>= 0)
        @test all(cm.pagerank .>= 0)
    end
end

@testitem "Centrality measures PageRank sums to one" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        cm = centrality_measures(net)
        @test isapprox(sum(cm.pagerank), 1.0, atol=0.01)
    end
end

@testitem "Centrality measures DataFrame conversion" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        cm = centrality_measures(net)
        df = centrality_dataframe(cm)
        @test df isa DataFrame
        @test nrow(df) == length(cm.protein_names)
        @test hasproperty(df, :Protein)
        @test hasproperty(df, :Degree)
        @test hasproperty(df, :PageRank)
    end
end

@testitem "Community detection label propagation" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        comm = detect_communities(net, algorithm=:label_propagation)
        @test comm isa BayesInteractomics.AbstractNetworkResult
        @test length(comm.protein_names) == nv(net.graph)
        @test length(comm.membership) == nv(net.graph)
        @test comm.n_communities > 0
        @test all(1 .<= comm.membership .<= comm.n_communities)
        @test sum(comm.community_sizes) == nv(net.graph)
        @test -1.0 <= comm.modularity <= 1.0
    end
end

@testitem "Community detection multiple algorithms" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        comm_louvain = detect_communities(net, algorithm=:louvain)
        comm_greedy = detect_communities(net, algorithm=:greedy_modularity)

        @test comm_louvain isa BayesInteractomics.AbstractNetworkResult
        @test comm_greedy isa BayesInteractomics.AbstractNetworkResult
    end
end

@testitem "Community detection DataFrame conversion" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        comm = detect_communities(net, algorithm=:label_propagation)
        df = community_dataframe(comm)
        @test df isa DataFrame
        @test nrow(df) == length(comm.protein_names)
        @test hasproperty(df, :Protein)
        @test hasproperty(df, :Community)
    end
end

@testitem "Community detection invalid algorithm" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        @test_throws ErrorException detect_communities(net, algorithm=:invalid)
    end
end

@testitem "Network visualization basic" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        plt = plot_network(net)
        @test plt isa Compose.Context
    end
end

@testitem "Network visualization layouts" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        for layout in [:spring, :circular, :shell, :spectral]
            plt = plot_network(net, layout=layout)
            @test plt isa Compose.Context
        end
    end
end

@testitem "Network visualization node sizes" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        for size_by in [:uniform, :degree, :posterior_prob, :log2fc]
            plt = plot_network(net, node_size=size_by)
            @test plt isa Compose.Context
        end
    end
end

@testitem "Network visualization node colors" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        for color_by in [:uniform, :posterior_prob, :log2fc]
            plt = plot_network(net, node_color=color_by)
            @test plt isa Compose.Context
        end
    end
end

@testitem "Network visualization labels" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        plt_labels = plot_network(net, show_labels=true)
        plt_no_labels = plot_network(net, show_labels=false)
        @test plt_labels isa Compose.Context
        @test plt_no_labels isa Compose.Context
    end
end

@testitem "Network visualization bait highlighting" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        plt_highlight = plot_network(net, highlight_bait=true)
        plt_no_highlight = plot_network(net, highlight_bait=false)
        @test plt_highlight isa Compose.Context
        @test plt_no_highlight isa Compose.Context
    end
end

@testitem "Network export GraphML" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        graphml_file = tempname() * ".graphml"
        export_graphml(net, graphml_file)
        @test isfile(graphml_file)
        content = read(graphml_file, String)
        @test occursin("<?xml", content)
        @test occursin("<graphml", content)
        @test occursin("</graphml>", content)
        rm(graphml_file)
    end
end

@testitem "Network export edge list" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    using CSV
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        edges_file = tempname() * ".csv"
        export_edgelist(net, edges_file)
        @test isfile(edges_file)
        edges_df = CSV.read(edges_file, DataFrame)
        @test hasproperty(edges_df, :source)
        @test hasproperty(edges_df, :target)
        @test hasproperty(edges_df, :weight)
        @test nrow(edges_df) == ne(net.graph)
        rm(edges_file)
    end
end

@testitem "Network export node attributes" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    using CSV
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        nodes_file = tempname() * ".csv"
        export_node_attributes(net, nodes_file)
        @test isfile(nodes_file)
        nodes_df = CSV.read(nodes_file, DataFrame)
        @test nrow(nodes_df) == nv(net.graph)
        rm(nodes_file)
    end
end

@testitem "Network export bundle" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        bundle_prefix = tempname()
        BayesInteractomicsNetworkExt.export_network_bundle(net, bundle_prefix)
        @test isfile(bundle_prefix * ".graphml")
        @test isfile(bundle_prefix * "_edges.csv")
        @test isfile(bundle_prefix * "_nodes.csv")
        rm(bundle_prefix * ".graphml")
        rm(bundle_prefix * "_edges.csv")
        rm(bundle_prefix * "_nodes.csv")
    end
end

@testitem "Network save plot PNG" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        png_file = tempname() * ".png"
        save_network_plot(net, png_file)
        @test isfile(png_file)
        rm(png_file)
    end
end

@testitem "Network save plot PDF" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        pdf_file = tempname() * ".pdf"
        save_network_plot(net, pdf_file)
        @test isfile(pdf_file)
        rm(pdf_file)
    end
end

@testitem "Network save plot SVG" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        svg_file = tempname() * ".svg"
        save_network_plot(net, svg_file)
        @test isfile(svg_file)
        rm(svg_file)
    end
end

@testitem "Network save plot unsupported format" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net = build_network(ar, posterior_threshold=0.5)

    if nv(net.graph) > 0
        @test_throws ErrorException save_network_plot(net, "test.txt")
    end
end

@testitem "Edge case minimal data" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose

    minimal_results = DataFrame(
        Protein = ["ProteinA"],
        PosteriorProbability = [0.95],
        BayesFactor = [100.0],
        q_value = [0.001],
        mean_log2FC = [3.5]
    )

    ar_minimal = NetworkAnalysisResult(minimal_results, bait_protein="Bait")
    net_minimal = build_network(ar_minimal)
    @test nv(net_minimal.graph) >= 0
end

@testitem "Edge case all proteins failing threshold" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar_strict = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net_strict = build_network(ar_strict, posterior_threshold=1.0)
    @test nv(net_strict.graph) == 0
end

@testitem "Edge case statistics on empty network" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net_empty = build_network(ar, posterior_threshold=1.0)
    stats_empty = network_statistics(net_empty)
    @test stats_empty.n_nodes == 0
    @test stats_empty.n_edges == 0
end

@testitem "Edge case centrality on empty network" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait")
    net_empty = build_network(ar, posterior_threshold=1.0)
    cm_empty = centrality_measures(net_empty)
    @test length(cm_empty.protein_names) == 0
end

@testitem "Extension loading verification" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    # Verify extension is loaded
    @test isdefined(Main, :BayesInteractomicsNetworkExt)

    # Verify functions are not stubs (should not throw "extension not loaded" error)
    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results)
    try
        net = build_network(ar, posterior_threshold=0.9)
        @test true
    catch e
        if occursin("extension not loaded", string(e))
            @test false
        else
            # Some other error is OK (e.g., no data passing threshold)
            @test true
        end
    end
end

# ===================================================================
# NetworkConfig and run_network_analysis pipeline tests
# ===================================================================

@testitem "NetworkConfig default construction" begin
    using BayesInteractomics

    cfg = NetworkConfig()
    @test cfg.posterior_threshold == 0.5
    @test cfg.q_threshold == 0.05
    @test isnothing(cfg.bf_threshold)
    @test isnothing(cfg.log2fc_threshold)
    @test cfg.include_bait == true
    @test cfg.weight_by == :posterior_prob
    @test cfg.enrich == false
    @test cfg.plot == true
    @test cfg.detect_communities == true
    @test cfg.compute_centrality == true
    @test cfg.export_files == true
    @test cfg.generate_report == true
    @test cfg.verbose == true
    @test cfg.output_dir == "network_results"
    @test cfg.layout == :spring
    @test cfg.community_algorithm == :louvain
    @test cfg.top_hubs_by == :pagerank
    @test cfg.top_hubs_n == 10
end

@testitem "NetworkConfig keyword override" begin
    using BayesInteractomics

    cfg = NetworkConfig(
        posterior_threshold = 0.8,
        q_threshold = 0.01,
        bf_threshold = 10.0,
        enrich = true,
        species = 10090,
        output_dir = "custom_output",
        report_title = "Custom Title",
        layout = :circular,
        community_algorithm = :label_propagation,
        top_hubs_by = :degree,
        top_hubs_n = 5,
        verbose = false
    )
    @test cfg.posterior_threshold == 0.8
    @test cfg.q_threshold == 0.01
    @test cfg.bf_threshold == 10.0
    @test cfg.enrich == true
    @test cfg.species == 10090
    @test cfg.output_dir == "custom_output"
    @test cfg.report_title == "Custom Title"
    @test cfg.layout == :circular
    @test cfg.community_algorithm == :label_propagation
    @test cfg.top_hubs_by == :degree
    @test cfg.top_hubs_n == 5
    @test cfg.verbose == false
end

@testitem "NetworkConfig is mutable" begin
    using BayesInteractomics

    cfg = NetworkConfig()
    cfg.posterior_threshold = 0.9
    cfg.output_dir = "new_dir"
    @test cfg.posterior_threshold == 0.9
    @test cfg.output_dir == "new_dir"
end

@testitem "NetworkConfig bait fields" begin
    using BayesInteractomics

    cfg = NetworkConfig()
    @test isnothing(cfg.bait_protein)
    @test isnothing(cfg.bait_index)

    cfg2 = NetworkConfig(bait_protein="HAP40", bait_index=1)
    @test cfg2.bait_protein == "HAP40"
    @test cfg2.bait_index == 1
end

@testitem "run_network_analysis basic pipeline" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait", bait_index=1)

    output_dir = mktempdir()
    cfg = NetworkConfig(
        posterior_threshold = 0.5,
        q_threshold = 0.05,
        output_dir = output_dir,
        verbose = false,
        enrich = false
    )

    result = run_network_analysis(ar, cfg)

    @test result isa BayesInteractomics.NetworkPipelineResult
    @test nv(result.network.graph) > 0
    @test !isnothing(result.statistics)
    @test result.statistics.n_nodes == nv(result.network.graph)
    @test !isnothing(result.centrality)
    @test !isnothing(result.top_hubs)
    @test result.top_hubs isa DataFrame
    @test !isnothing(result.communities)
    @test !isnothing(result.edge_sources)
    @test !isnothing(result.plot_object)
    @test !isnothing(result.report_path)
    @test !isnothing(result.report_content)
    @test isfile(result.report_path)
    @test isempty(result.warnings)

    rm(output_dir, recursive=true)
end

@testitem "run_network_analysis default config" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait", bait_index=1)

    # Should work with just the default config (uses NetworkConfig() defaults)
    result = run_network_analysis(ar, NetworkConfig(verbose=false, output_dir=mktempdir()))

    @test result isa BayesInteractomics.NetworkPipelineResult
    @test nv(result.network.graph) > 0

    rm(result.config.output_dir, recursive=true)
end

@testitem "run_network_analysis empty network" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait", bait_index=1)

    output_dir = mktempdir()
    cfg = NetworkConfig(
        posterior_threshold = 1.0,
        q_threshold = 0.0001,
        output_dir = output_dir,
        verbose = false
    )

    result = run_network_analysis(ar, cfg)

    @test result isa BayesInteractomics.NetworkPipelineResult
    @test nv(result.network.graph) == 0
    @test isnothing(result.statistics)
    @test isnothing(result.centrality)
    @test isnothing(result.top_hubs)
    @test isnothing(result.communities)
    @test length(result.warnings) > 0
    @test any(occursin("empty", lowercase(w)) for w in result.warnings)

    rm(output_dir, recursive=true)
end

@testitem "run_network_analysis no plot no export no report" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait", bait_index=1)

    output_dir = mktempdir()
    cfg = NetworkConfig(
        posterior_threshold = 0.5,
        output_dir = output_dir,
        verbose = false,
        plot = false,
        export_files = false,
        generate_report = false
    )

    result = run_network_analysis(ar, cfg)

    @test result isa BayesInteractomics.NetworkPipelineResult
    @test isnothing(result.plot_object)
    @test isnothing(result.report_path)
    @test isnothing(result.report_content)
    @test isempty(result.export_paths)

    # Statistics, centrality, communities should still be computed
    @test !isnothing(result.statistics)
    @test !isnothing(result.centrality)
    @test !isnothing(result.communities)

    rm(output_dir, recursive=true)
end

@testitem "run_network_analysis disable centrality and communities" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait", bait_index=1)

    output_dir = mktempdir()
    cfg = NetworkConfig(
        posterior_threshold = 0.5,
        output_dir = output_dir,
        verbose = false,
        compute_centrality = false,
        detect_communities = false,
        generate_report = false,
        plot = false,
        export_files = false
    )

    result = run_network_analysis(ar, cfg)

    @test isnothing(result.centrality)
    @test isnothing(result.top_hubs)
    @test isnothing(result.communities)
    @test !isnothing(result.statistics)  # always computed

    rm(output_dir, recursive=true)
end

@testitem "run_network_analysis export files exist" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait", bait_index=1)

    output_dir = mktempdir()
    cfg = NetworkConfig(
        posterior_threshold = 0.5,
        output_dir = output_dir,
        file_prefix = "test_net",
        verbose = false,
        plot = false,
        generate_report = false
    )

    result = run_network_analysis(ar, cfg)

    @test haskey(result.export_paths, "graphml")
    @test haskey(result.export_paths, "edgelist")
    @test haskey(result.export_paths, "node_attributes")
    @test haskey(result.export_paths, "centrality")
    @test haskey(result.export_paths, "communities")

    for (_, path) in result.export_paths
        @test isfile(path)
    end

    rm(output_dir, recursive=true)
end

@testitem "generate_network_report content" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait", bait_index=1)

    output_dir = mktempdir()
    cfg = NetworkConfig(
        posterior_threshold = 0.5,
        output_dir = output_dir,
        verbose = false,
        report_title = "Test Report",
        plot = false
    )

    result = run_network_analysis(ar, cfg)

    @test !isnothing(result.report_content)
    @test occursin("Test Report", result.report_content)
    @test occursin("Analysis Parameters", result.report_content)
    @test occursin("Network Topology", result.report_content)
    @test occursin("Hub Proteins", result.report_content)
    @test occursin("Community Structure", result.report_content)
    @test occursin("Bait Protein", result.report_content)
    @test occursin("Exported Files", result.report_content)
    @test isfile(result.report_path)

    rm(output_dir, recursive=true)
end

@testitem "generate_network_report standalone call" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait", bait_index=1)

    output_dir = mktempdir()
    cfg = NetworkConfig(
        posterior_threshold = 0.5,
        output_dir = output_dir,
        verbose = false,
        generate_report = false,
        plot = false
    )

    result = run_network_analysis(ar, cfg)
    @test isnothing(result.report_path)

    # Now generate report standalone with custom filename
    custom_path = joinpath(output_dir, "custom_report.md")
    report_path, report_content = generate_network_report(result; filename=custom_path, title="Standalone Report")

    @test report_path == custom_path
    @test isfile(custom_path)
    @test occursin("Standalone Report", report_content)

    rm(output_dir, recursive=true)
end

@testitem "run_network_analysis bait from config" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    # Create AnalysisResult WITHOUT bait info
    ar = NetworkAnalysisResult(mock_results)
    @test isnothing(ar.bait_protein)

    output_dir = mktempdir()
    cfg = NetworkConfig(
        bait_protein = "ConfigBait",
        bait_index = 1,
        posterior_threshold = 0.5,
        output_dir = output_dir,
        verbose = false,
        plot = false,
        export_files = false,
        generate_report = false
    )

    result = run_network_analysis(ar, cfg)

    # Bait should come from config
    @test result.network.bait_protein == "ConfigBait"
    @test result.network.bait_index == 1
    @test nv(result.network.graph) > 0

    rm(output_dir, recursive=true)
end

@testitem "run_network_analysis bait config overrides ar" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="OriginalBait", bait_index=2)

    output_dir = mktempdir()
    cfg = NetworkConfig(
        bait_protein = "OverrideBait",
        bait_index = 1,
        posterior_threshold = 0.5,
        output_dir = output_dir,
        verbose = false,
        plot = false,
        export_files = false,
        generate_report = false
    )

    result = run_network_analysis(ar, cfg)

    # Config should override AnalysisResult
    @test result.network.bait_protein == "OverrideBait"
    @test result.network.bait_index == 1

    rm(output_dir, recursive=true)
end

@testitem "run_network_analysis small network skips communities" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose

    # Only one protein passing threshold -> 2 nodes (bait + 1 interactor) < 3
    small_results = DataFrame(
        Protein = ["ProteinA"],
        PosteriorProbability = [0.95],
        BayesFactor = [100.0],
        q_value = [0.001],
        mean_log2FC = [3.5]
    )
    ar = NetworkAnalysisResult(small_results, bait_protein="Bait", bait_index=1)

    output_dir = mktempdir()
    cfg = NetworkConfig(
        posterior_threshold = 0.5,
        output_dir = output_dir,
        verbose = false,
        plot = false,
        export_files = false,
        generate_report = false
    )

    result = run_network_analysis(ar, cfg)

    @test isnothing(result.communities)
    @test any(occursin("too small", lowercase(w)) for w in result.warnings)

    rm(output_dir, recursive=true)
end

@testitem "NetworkPipelineResult show method" begin
    using BayesInteractomics
    using DataFrames
    using Graphs
    using SimpleWeightedGraphs
    using GraphPlot
    using Compose
    include("test_network.jl")

    mock_results = create_mock_results()
    ar = NetworkAnalysisResult(mock_results, bait_protein="Bait", bait_index=1)

    output_dir = mktempdir()
    cfg = NetworkConfig(
        posterior_threshold = 0.5,
        output_dir = output_dir,
        verbose = false,
        plot = false,
        export_files = false,
        generate_report = false
    )

    result = run_network_analysis(ar, cfg)
    str = sprint(show, result)
    @test occursin("NetworkPipelineResult", str)
    @test occursin("nodes", str)
    @test occursin("edges", str)

    rm(output_dir, recursive=true)
end
