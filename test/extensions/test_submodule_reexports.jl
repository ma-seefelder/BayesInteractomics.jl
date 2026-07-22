# Submodule re-export contract tests.
#
# The submodule wrap grouped src/reports/, src/data/curation/, src/differential/,
# src/docking/ into four organisational submodules (Reports / Curation / Differential
# / Docking) inside BayesInteractomics. Each submodule's public surface MUST remain
# resolvable at the parent's top-level namespace (`using BayesInteractomics;
# differential_analysis(...)`) so existing user code keeps working unchanged.
#
# These tests grep-resistant-ly enforce that contract: for each submodule we
# verify (a) the submodule itself is a Module, (b) every documented public
# symbol resolves via the parent namespace.

@testitem "Reports submodule: generate_report and generate_differential_report resolve via parent namespace" begin
    using BayesInteractomics
    using Test

    @test isa(BayesInteractomics.Reports, Module)
    @test isdefined(BayesInteractomics, :generate_report)
    @test isdefined(BayesInteractomics, :generate_differential_report)

    # generate_report is re-exported via `using .Reports: generate_report` — must be
    # the SAME function object as the submodule's internal binding.
    @test generate_report === BayesInteractomics.Reports.generate_report

    # generate_differential_report is the parent-stub-then-extend pattern:
    # the empty generic is declared at parent scope BEFORE module Differential, and
    # both Reports + Differential extend that same parent binding. The Reports
    # submodule imports the parent stub with `import ..BayesInteractomics: generate_differential_report`
    # so its method extends the parent binding rather than creating a local shadow.
    # The parent-level `generate_differential_report` must therefore be the same
    # binding accessible through BayesInteractomics.
    @test generate_differential_report === BayesInteractomics.generate_differential_report
end

@testitem "Curation submodule: 16 public symbols resolve via parent namespace" begin
    using BayesInteractomics
    using Test

    @test isa(BayesInteractomics.Curation, Module)
    for sym in (:curate_proteins, :CurationReport, :CurationActionType, :CurationEntry,
                :MergeCandidate, :MergeDecision, :CurationCache, :CurationAPIError,
                :split_protein_groups, :resolve_to_string_ids,
                :merge_protein_rows, :confirm_merges_interactive,
                :save_curation_report, :load_curation_report,
                :remove_contaminants, :parse_protein_id)
        @test isdefined(BayesInteractomics, sym)
    end
end

@testitem "Differential submodule: 25 public symbols resolve via parent namespace" begin
    using BayesInteractomics
    using Test

    @test isa(BayesInteractomics.Differential, Module)
    for sym in (:DifferentialConfig, :DifferentialResult, :InteractionClass,
                :GAINED, :REDUCED, :UNCHANGED, :BOTH_NEGATIVE,
                :CONDITION_A_SPECIFIC, :CONDITION_B_SPECIFIC,
                :differential_analysis,
                :differential_volcano_plot, :differential_evidence_plot,
                :differential_scatter_plot, :differential_classification_plot,
                :differential_ma_plot,
                :gained_interactions, :lost_interactions, :unchanged_interactions,
                :significant_differential, :export_differential,
                :getDifferentialBayesFactors, :getDifferentialPosteriors,
                :getDifferentialQValues, :getDifferentialBFDR,
                :getClassifications, :getDeltaLog2FC)
        @test isdefined(BayesInteractomics, sym)
    end
end

@testitem "Docking submodule: 16 public symbols resolve via parent namespace" begin
    using BayesInteractomics
    using Test

    @test isa(BayesInteractomics.Docking, Module)
    for sym in (:DockingConfig, :DockingCalibration, :DockingPairResult,
                :DockingResult, :DockingRequestBatch,
                :compute_bf_dock, :apply_docking_update, :default_calibration,
                :compute_pdockq, :compute_bf_from_pdockq, :compute_bf_from_iptm,
                :compute_bf_from_c2qscore, :docking_cache_key, :compute_c2qscore,
                :generate_docking_requests, :import_docking_results)
        @test isdefined(BayesInteractomics, sym)
    end
end
