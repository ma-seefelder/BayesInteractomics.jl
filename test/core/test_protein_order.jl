"""
    test_protein_order.jl

Tests that the protein reordering and join logic in the analysis pipeline
preserves the correct mapping between protein identifiers and their computed
Bayes factors. A bug here would silently associate BFs with the wrong proteins.
"""

@testitem "Reordering + join logic preserves protein-to-BF mapping" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, getIDs, getPositions
    using DataFrames
    import DataFrames: innerjoin

    # --- Setup: 8 proteins with distinct IDs vs names ---
    protein_ids   = ["ID_01", "ID_02", "ID_03", "ID_04", "ID_05", "ID_06", "ID_07", "ID_08"]
    protein_names = ["Gene_A", "Gene_B", "Gene_C", "Gene_D", "Gene_E", "Gene_F", "Gene_G", "Gene_H"]

    # Minimal InteractionData (only need protein_IDs for the reordering logic)
    mat = Union{Missing, Float64}[rand(8, 3);]
    proto = Protocol(1, protein_ids, Dict(1 => mat))
    no_experiments_dict = Dict(1 => 1)
    no_parameters_HBM = 1 + 1 + 1  # intercept + 1 protocol + 1 experiment
    no_parameters_Regression = 1 + 1
    protocol_positions, experiment_positions, matched_positions =
        getPositions(no_experiments_dict, no_parameters_HBM)

    data = InteractionData(
        protein_ids, protein_names,
        Dict(1 => proto), Dict(1 => proto),
        1, no_experiments_dict,
        no_parameters_HBM, no_parameters_Regression,
        experiment_positions, protocol_positions, matched_positions
    )

    # --- Mock df: scrambled order, 2 proteins removed (ID_03, ID_06) ---
    kept_ids = ["ID_08", "ID_05", "ID_01", "ID_07", "ID_04", "ID_02"]  # scrambled
    mock_bf_enrichment = Dict(
        "ID_01" => 1.1, "ID_02" => 2.2, "ID_04" => 4.4,
        "ID_05" => 5.5, "ID_07" => 7.7, "ID_08" => 8.8
    )
    df = DataFrame(
        Protein    = kept_ids,
        BF_log2FC  = [mock_bf_enrichment[id] for id in kept_ids],
        mean_log2FC = [10.0 * i for i in 1:length(kept_ids)],
        bf_slope   = [0.1 * i for i in 1:length(kept_ids)]
    )

    # Known per-protein bf_detected (length 8, one per protein in original order)
    bf_detected = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

    # --- Run the exact reordering logic from pipeline.jl:283-301 ---
    pnames = getIDs(data)
    protein_in_dataset = [pnames[i] in df.Protein for i in 1:length(pnames)]
    pnames_filtered = pnames[protein_in_dataset]

    new_order = [findfirst(x -> x == p, df.Protein) for p in pnames_filtered]
    df_sorted = df[new_order, :]

    bf_detection = DataFrame(Protein = getIDs(data), bf_detected = bf_detected)
    df_final = innerjoin(df_sorted, bf_detection, on = :Protein)

    # --- Assertions ---
    # 1. Output proteins match the original ID order (minus removed ones)
    expected_order = ["ID_01", "ID_02", "ID_04", "ID_05", "ID_07", "ID_08"]
    @test df_final.Protein == expected_order

    # 2. bf_detected values match the per-protein values for each ID
    expected_bf_detected = Dict(
        "ID_01" => 10.0, "ID_02" => 20.0, "ID_04" => 40.0,
        "ID_05" => 50.0, "ID_07" => 70.0, "ID_08" => 80.0
    )
    for row in eachrow(df_final)
        @test row.bf_detected == expected_bf_detected[row.Protein]
    end

    # 3. BF_log2FC values match the per-protein mock enrichment values
    for row in eachrow(df_final)
        @test row.BF_log2FC == mock_bf_enrichment[row.Protein]
    end

    # 4. Relative order is strictly increasing when mapped back to original indices
    original_positions = [findfirst(==(p), protein_ids) for p in df_final.Protein]
    @test issorted(original_positions)
end


@testitem "Full analyse() pipeline preserves protein order and signal" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData, getIDs, getNames, getPositions, analyse
    using DataFrames
    using Random

    Random.seed!(42)

    # --- Build synthetic InteractionData ---
    # n_experiments must equal n_replicates for HierarchicalBayesianModelSingle
    # (the model uses size(samples, 2) to determine hierarchy depth).
    # Need ≥50 proteins so the latent class EM can reliably fit a 2-component mixture
    # without component collapse. ~20% enriched gives well-separated classes.
    n_proteins = 50
    n_experiments = 3
    n_replicates = 3
    enriched_idx = 37  # protein with strongest enrichment (not near start/end)
    bait_idx = 1       # bait protein

    protein_ids   = ["ID_$(lpad(i, 3, '0'))" for i in 1:n_proteins]
    protein_names = ["Gene_$(lpad(i, 3, '0'))" for i in 1:n_proteins]

    # Define enriched proteins (~20% of total) with varying fold changes
    enriched_set = Dict{Int,Float64}(
        bait_idx => 3.0,
        enriched_idx => 5.0,    # strongest signal
        5 => 2.5, 12 => 2.0, 18 => 1.8, 24 => 2.2, 30 => 1.5,
        35 => 1.7, 42 => 2.8, 48 => 2.3
    )

    control_dict = Dict{Int, Matrix{Union{Missing, Float64}}}()
    sample_dict  = Dict{Int, Matrix{Union{Missing, Float64}}}()

    # Per-protein baseline intensities (deterministic, varying)
    baselines = 20.0 .+ collect(1:n_proteins) .* 0.05

    for exp in 1:n_experiments
        ctrl = zeros(Union{Missing, Float64}, n_proteins, n_replicates)
        samp = zeros(Union{Missing, Float64}, n_proteins, n_replicates)

        for p in 1:n_proteins
            ctrl[p, :] .= baselines[p] .+ randn(n_replicates) .* 0.4
            samp[p, :] .= baselines[p] .+ randn(n_replicates) .* 0.4
            if haskey(enriched_set, p)
                samp[p, :] .+= enriched_set[p]
            end
        end

        control_dict[exp] = ctrl
        sample_dict[exp]  = samp
    end

    control_proto = Protocol(n_experiments, protein_ids, control_dict)
    sample_proto  = Protocol(n_experiments, protein_ids, sample_dict)

    no_experiments_dict = Dict(1 => n_experiments)
    no_parameters_HBM = 1 + 1 + n_experiments  # intercept + 1 protocol + experiments
    no_parameters_Regression = 1 + 1

    protocol_positions, experiment_positions, matched_positions =
        getPositions(no_experiments_dict, no_parameters_HBM)

    data = InteractionData(
        protein_ids, protein_names,
        Dict(1 => sample_proto), Dict(1 => control_proto),
        1, no_experiments_dict,
        no_parameters_HBM, no_parameters_Regression,
        experiment_positions, protocol_positions, matched_positions
    )

    # Verify our data has distinct IDs vs names
    @test getIDs(data) != getNames(data)

    # --- Run analyse() in a temp directory to isolate side effects ---
    results = mktempdir() do tmpdir
        cd(tmpdir) do
            analyse(
                data, nothing;
                combination_method = :latent_class,
                n_controls = n_replicates,
                n_samples = n_replicates,
                refID = bait_idx,
                use_intermediate_cache = false,
                temp_result_file = joinpath(tmpdir, "temp_results.xlsx")
            )
        end
    end

    copula_df = results.copula_results

    # --- Order assertions ---
    # 1. Output uses protein IDs (not names)
    @test all(id -> id in protein_ids, copula_df.Protein)
    @test !any(name -> name in protein_names, copula_df.Protein)

    # 2. Output order matches the original protein_ids order (for proteins present)
    original_positions = [findfirst(==(p), protein_ids) for p in copula_df.Protein]
    @test issorted(original_positions)

    # --- Signal integrity ---
    # 3. The protein with strongest enrichment (enriched_idx=37, fc=5.0) should have
    #    bf_enrichment well above the median — catches value-to-name misalignment
    enriched_id = protein_ids[enriched_idx]
    enriched_row = findfirst(==(enriched_id), copula_df.Protein)
    @test !isnothing(enriched_row)

    if !isnothing(enriched_row)
        median_bf = sort(copula_df.bf_enrichment)[div(nrow(copula_df), 2)]
        @test copula_df.bf_enrichment[enriched_row] > median_bf
    end

    # 4. Completeness: most proteins should be present (some may fail inference)
    @test nrow(copula_df) >= n_proteins - 10
end
