# test/embeddings/test_pool_imputed_matrix.jl
# RED scaffold: locks the pooled-mean InteractionData helper contract.
# These tests are expected to fail with UndefVarError until the helper lands
# `BayesInteractomics._pool_imputed_matrix`.

@testitem "_pool_imputed_matrix correctness on synthetic M=3" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData

    # Build M=3 synthetic imputed InteractionData with identical scaffolding.
    # Per-experiment matrix shape: 3 proteins × 2 replicates. One protocol, one experiment.
    protein_ids   = ["P1", "P2", "P3"]
    protein_names = ["P1", "P2", "P3"]
    base = Union{Missing, Float64}[1.0 2.0; 3.0 4.0; 5.0 6.0]

    function make_id(offset::Float64)
        mat = Union{Missing, Float64}[1.0+offset 2.0+offset; 3.0+offset 4.0+offset; 5.0+offset 6.0+offset]
        sample_proto  = Protocol(1, protein_ids, Dict(1 => mat))
        # Controls share shape; we populate samples only for this @testitem.
        control_mat   = Union{Missing, Float64}[0.0 0.0; 0.0 0.0; 0.0 0.0]
        control_proto = Protocol(1, protein_ids, Dict(1 => control_mat))
        return InteractionData(
            protein_ids, protein_names,
            Dict(1 => sample_proto), Dict(1 => control_proto),
            1, Dict(1 => 1),
            3, 2,
            [2], [3], [1],
            trues(3),
        )
    end

    imputed = [make_id(0.0), make_id(1.0), make_id(2.0)]

    pooled = BayesInteractomics._pool_imputed_matrix(imputed)

    # Element-wise mean: (base + (base+1) + (base+2)) / 3 = base + 1
    expected = Union{Missing, Float64}[2.0 3.0; 4.0 5.0; 6.0 7.0]
    @test pooled.samples[1].data[1] ≈ expected
end

@testitem "_pool_imputed_matrix shape + eltype preservation" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData

    protein_ids   = ["P1", "P2", "P3"]
    protein_names = ["P1", "P2", "P3"]

    function make_id(offset::Float64)
        mat = Union{Missing, Float64}[1.0+offset 2.0+offset; 3.0+offset 4.0+offset; 5.0+offset 6.0+offset]
        sample_proto  = Protocol(1, protein_ids, Dict(1 => mat))
        control_mat   = Union{Missing, Float64}[0.0 0.0; 0.0 0.0; 0.0 0.0]
        control_proto = Protocol(1, protein_ids, Dict(1 => control_mat))
        return InteractionData(
            protein_ids, protein_names,
            Dict(1 => sample_proto), Dict(1 => control_proto),
            1, Dict(1 => 1),
            3, 2,
            [2], [3], [1],
            trues(3),
        )
    end

    imputed = [make_id(0.0), make_id(1.0), make_id(2.0)]
    pooled = BayesInteractomics._pool_imputed_matrix(imputed)

    # Scaffolding preservation
    @test pooled.no_protocols         == imputed[1].no_protocols
    @test pooled.no_experiments       == imputed[1].no_experiments
    @test pooled.protein_IDs          == imputed[1].protein_IDs
    @test pooled.protocol_positions   == imputed[1].protocol_positions
    @test pooled.experiment_positions == imputed[1].experiment_positions
    @test pooled.matched_positions    == imputed[1].matched_positions

    # LANDMINE (RESEARCH.md Critical Pitfall #1):
    # Protocol.data is Dict{I, Matrix{Union{Missing, F}}} — eltype is invariant,
    # the pooled matrix must carry Union{Missing, Float64} even though no missings remain.
    @test eltype(pooled.samples[1].data[1]) === Union{Missing, Float64}
    @test !any(ismissing, pooled.samples[1].data[1])
end

@testitem "_pool_imputed_matrix controls block pooled identically to samples block" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData

    protein_ids   = ["P1", "P2", "P3"]
    protein_names = ["P1", "P2", "P3"]

    function make_id(offset::Float64)
        sample_mat    = Union{Missing, Float64}[0.0 0.0; 0.0 0.0; 0.0 0.0]
        sample_proto  = Protocol(1, protein_ids, Dict(1 => sample_mat))
        # Populate CONTROLS this time; samples stay zero.
        control_mat   = Union{Missing, Float64}[10.0+offset 20.0+offset; 30.0+offset 40.0+offset; 50.0+offset 60.0+offset]
        control_proto = Protocol(1, protein_ids, Dict(1 => control_mat))
        return InteractionData(
            protein_ids, protein_names,
            Dict(1 => sample_proto), Dict(1 => control_proto),
            1, Dict(1 => 1),
            3, 2,
            [2], [3], [1],
            trues(3),
        )
    end

    imputed = [make_id(0.0), make_id(1.0), make_id(2.0)]
    pooled = BayesInteractomics._pool_imputed_matrix(imputed)

    expected_controls = Union{Missing, Float64}[11.0 21.0; 31.0 41.0; 51.0 61.0]
    @test pooled.controls[1].data[1] ≈ expected_controls
    @test eltype(pooled.controls[1].data[1]) === Union{Missing, Float64}
end

@testitem "_pool_imputed_matrix accepts abstract Vector{InteractionData} container" begin
    using BayesInteractomics
    using BayesInteractomics: Protocol, InteractionData

    # Regression test: `load_data` returns a `Vector{InteractionData}` with abstract
    # element type (no F/I parameterisation), and the
    # `run_analysis(::CONFIG, ::Vector{InteractionData}, ::InteractionData{F,I})`
    # overload forwards that container verbatim to `_pool_imputed_matrix`.
    # An earlier signature `_pool_imputed_matrix(::Vector{InteractionData{F,I}}) where {F,I}`
    # did NOT match the abstract container and raised a MethodError at runtime
    # (HAP40_Strep smoke run, observed via the pipeline.jl:4174 try/catch falling
    # into `@warn "[Embeddings] computation failed: MethodError(...)"`).
    # This test locks the abstract-container call path.

    protein_ids   = ["P1", "P2"]
    protein_names = ["P1", "P2"]

    function make_id(offset::Float64)
        sample_mat    = Union{Missing, Float64}[1.0+offset 2.0+offset; 3.0+offset 4.0+offset]
        sample_proto  = Protocol(1, protein_ids, Dict(1 => sample_mat))
        control_mat   = Union{Missing, Float64}[0.5 0.5; 0.5 0.5]
        control_proto = Protocol(1, protein_ids, Dict(1 => control_mat))
        return InteractionData(
            protein_ids, protein_names,
            Dict(1 => sample_proto), Dict(1 => control_proto),
            1, Dict(1 => 1),
            2, 2,
            [2], [2], [1],
            trues(2),
        )
    end

    # Construct the abstract container that mirrors load_data's return type.
    imputed_abstract = InteractionData[make_id(0.0), make_id(1.0), make_id(2.0)]
    @test imputed_abstract isa Vector{InteractionData}
    @test !(imputed_abstract isa Vector{InteractionData{Float64,Int64}})

    # The call MUST NOT throw MethodError — this is the regression contract.
    pooled = BayesInteractomics._pool_imputed_matrix(imputed_abstract)

    # And it must produce the same pooled mean as the parameterised path.
    expected_samples = Union{Missing, Float64}[2.0 3.0; 4.0 5.0]
    @test pooled.samples[1].data[1] ≈ expected_samples
end

