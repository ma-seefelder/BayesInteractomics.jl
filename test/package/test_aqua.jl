"""
    test_aqua.jl

Package quality checks using Aqua.jl
"""

@testitem "Aqua quality checks" begin
    using BayesInteractomics
    using Aqua

    Aqua.test_all(BayesInteractomics)
end
