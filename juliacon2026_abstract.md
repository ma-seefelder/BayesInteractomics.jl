# JuliaConf 2026 Submission

## Title

BayesInteractomics.jl: When One Bayes Factor Isn't Enough

## Abstract (max 500 characters)

Identifying genuine protein-protein interactions from mass spectrometry data requires disentangling real biology from experimental noise. BayesInteractomics.jl tackles this by fitting three complementary Bayesian models (detection, enrichment, and dose-response) and combining their evidence through copula mixture models. Built on RxInfer.jl and Copulas.jl, it leverages Julia's type system, multiple dispatch, and threading to analyze thousands of proteins in minutes.

## Description

Mass spectrometry-based interactomics experiments produce lists of hundreds to thousands of candidate protein-protein interactions, but a large fraction of these are non-specific contaminants or experimental artifacts. Existing tools typically apply a single statistical test and threshold -- a t-test on fold changes, or a simple scoring scheme -- discarding the rich multi-dimensional structure of the data. A protein might be modestly enriched but detected with striking consistency, or show a clear dose-response trend that a fold-change filter would miss entirely.

BayesInteractomics.jl takes a different approach. For each candidate interaction, the package computes Bayes factors from three independent statistical models that each capture a distinct aspect of the data:

1. A Beta-Bernoulli model that evaluates whether a protein is detected more consistently in bait samples than in controls.
2. A hierarchical Bayesian model (via RxInfer.jl) that estimates quantitative enrichment (log2 fold change) while sharing information across experimental protocols.
3. A Bayesian linear regression that tests for dose-response correlation between prey and bait abundance.

Rather than multiplying these Bayes factors under a naive independence assumption, the package uses Copulas.jl to model the dependency structure between evidence sources. This matters because enrichment and detection evidence are positively correlated under both hypotheses -- a genuinely enriched protein is also more likely to be consistently detected -- so treating them as independent inflates the combined evidence and drives up false discovery rates. A MAP-EM algorithm fits a three-component mixture (H0, an agnostic component, and H1) that accounts for this correlation, selecting among four single-parameter copula families -- Clayton, Frank, Gumbel, and Gaussian -- via model comparison. Bayesian model averaging then stacks this Copula model against a three-component latent-class model (3c-EM), producing a joint Bayes factor for every protein that properly reflects the shared information content across evidence types.

On top of the evidence engine, a deep-learning prior sharpens the ranking: a neural network predicts direct protein-protein interactions from sequence and network features, and a metalearner turns that DNN score together with orthogonal database evidence into a calibrated per-interaction prior. Prior odds and the combined Bayes factor multiply into a posterior probability, controlled at a Bayesian false discovery rate (BFDR) with per-protein posterior error probabilities (PEP).

This talk will cover the statistical design choices, how the Julia ecosystem made them practical, and lessons learned building a research-grade Bayesian analysis package. Specific topics include:

- How RxInfer.jl's reactive message-passing enables fast variational inference for the hierarchical enrichment model, and why this matters when you need to fit the same model thousands of times.
- Using Copulas.jl for method-of-moments fitting of Archimedean copulas, and how multiple dispatch made it straightforward to support four copula families through a single interface.
- Automated data curation via the STRING database API, including protein group resolution and synonym mapping
- Package extensions for optional functionality: a network analysis extension (Graphs.jl, GraphPlot.jl) for building and visualizing interaction networks, and an experimental structural docking extension (BioStructures.jl) -- both loaded only when the user imports the relevant packages.
- Self-contained HTML reports (client-side Plotly.js and DataTables.js, no additional Julia dependencies) so that collaborators who do not have Julia installed can explore results interactively in a browser.

The talk is aimed at Julia users interested in Bayesian statistics, scientific computing, or computational biology. No proteomics background is assumed and I will demonstrate a complete analysis from raw data to interactive report on a real dataset.

## Biography

Manuel Seefelder is a postdoctoral researcher in the Department of Gene Therapy at Ulm University Hospital, Germany. His background is in molecular medicine, with a doctorate on the huntingtin-associated protein 40 and its role in Huntington's disease. His current work sits at the intersection of wet-lab research, proteomics and computational method development: he builds Bayesian and deep-learning pipelines for analyzing protein interactome data from mass spectrometry experiments. Julia is his primary research language, and BayesInteractomics.jl grew directly out of the need to rigorously quantify interaction evidence in his own experiments. He also developed ProteinCoLoc, a Bayesian tool for colocalization analysis in fluorescence microscopy (Scientific Reports, 2024), and teaches a workshop on applied Bayesian statistics for PhD students at Ulm University.
