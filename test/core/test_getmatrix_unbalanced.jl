# Regression test for the getMatrix heap-corruption fix (src/core/types.jl).
#
# getMatrix used to size dim-3 from getMaxSamples(protein) (the SAMPLE replicate
# max) for BOTH the sample and control matrices. When a condition had more
# control replicates per experiment than samples (e.g. a single-experiment
# pulldown: 6 EGFP controls, 3 mutant samples), building the control matrix
# wrote a 6-element vector into a size-3 dim under @inbounds → heap corruption
# (a Julia-1.12/Windows GC EXCEPTION_ACCESS_VIOLATION downstream). The fix sizes
# dim-2/dim-3 to the max over BOTH samples and controls.

using TestItemRunner

@testitem "getMatrix: dim-3 covers controls > samples (heap-corruption regression)" begin
    using BayesInteractomics

    F = Union{Missing, Float64}
    # Single protocol, single experiment: 3 samples, 6 controls — the HAP40-mutant shape.
    p = BayesInteractomics.Protein{F, Int}(
        "F8A1", "F8A1",
        [Dict(1 => F[1.0, 2.0, 3.0])],                            # samples: 3 replicates
        [Dict(1 => F[10.0, 11.0, 12.0, 13.0, 14.0, 15.0])],       # controls: 6 replicates
    )

    sm = BayesInteractomics.getSampleMatrix(p)
    cm = BayesInteractomics.getControlMatrix(p)

    # dim-3 must cover the LONGER (control) replicate vector, not the sample max.
    @test size(sm, 3) == 6
    @test size(cm, 3) == 6
    # Sample and control matrices MUST share dims 1 and 2 — the regression cats
    # them with dims=2 (models.jl), so a mismatch would error there.
    @test size(sm)[1:2] == size(cm)[1:2]
    @test size(cat(sm, cm; dims = 2)) == (1, 2, 6)

    # All six control values are present (none dropped / no out-of-bounds write).
    @test sort(collect(skipmissing(cm[1, 1, :]))) == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    @test sort(collect(skipmissing(sm[1, 1, :]))) == [1.0, 2.0, 3.0]
    # The 3 unfilled sample slots are padded with missing (not garbage).
    @test count(ismissing, sm[1, 1, :]) == 3
end

@testitem "getMatrix: balanced controls == samples unchanged" begin
    using BayesInteractomics
    F = Union{Missing, Float64}
    # 1 protocol, 2 experiments, 3 replicates each on both sides (balanced).
    p = BayesInteractomics.Protein{F, Int}(
        "P", "P",
        [Dict(1 => F[1.0, 2.0, 3.0], 2 => F[4.0, 5.0, 6.0])],
        [Dict(1 => F[7.0, 8.0, 9.0], 2 => F[10.0, 11.0, 12.0])],
    )
    sm = BayesInteractomics.getSampleMatrix(p)
    cm = BayesInteractomics.getControlMatrix(p)
    @test size(sm) == (1, 2, 3)          # (protocols, experiments, replicates)
    @test size(cm) == (1, 2, 3)
    @test sort(collect(skipmissing(sm[1, 2, :]))) == [4.0, 5.0, 6.0]
end
