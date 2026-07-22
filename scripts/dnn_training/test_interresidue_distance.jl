using Test
using HTTP
using BioStructures

include("interresidue_distance.jl")

@testset "readPDB Tests" begin

    @testset "Valid PDB IDs" begin
        # Test with a known small protein PDB ID
        pdb_id_1crn = "1crn"
        structure_1crn = readPDB(pdb_id_1crn)
        @test structure_1crn isa BioStructures.MolecularStructure
        # Test with another valid PDB ID
        pdb_id_6ez8 = "6Ez8"
        structure_6ez8 = readPDB(pdb_id_6ez8)
        @test structure_6ez8 isa BioStructures.MolecularStructure
    end

    @testset "Invalid PDB IDs" begin
        # Test with a PDB ID that is unlikely to exist (should result in 404)
        invalid_pdb_id = "XXXX"
        @test readPDB(invalid_pdb_id) === nothing
        # Test with a PDB ID that is too short / malformed for RCSB URL structure
        malformed_pdb_id = "X"
        @test readPDB(malformed_pdb_id) === nothing
    end

    @testset "Edge Cases" begin
        # Test with an empty string PDB ID
        # This should lead to an error in HTTP.get or a non-200 response,
        # caught by the try-catch block.
        empty_pdb_id = ""
        @test readPDB(empty_pdb_id) === nothing
    end
end