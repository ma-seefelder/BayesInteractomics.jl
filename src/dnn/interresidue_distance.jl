##########################################################################################################
# Calculate interresidue distance betweeen proteins in a PDB strcuture                                 ###
##########################################################################################################
# Dr. rer. nat. Manuel Seelelder, 12th Mai 2025
# julia version 1.8

import BioStructures, BioSequences, BioAlignments
import CSV
import DataFrames: DataFrame
import HTTP
import ThreadsX
import CodecZlib

using ProgressMeter

# --------------------------------------------------------------------------- #
# Functions to download and read the PDB file                                 #
# --------------------------------------------------------------------------- #
"""
    readPDB(structure_id::String, mail_address::String; verbose::Bool = true)

Downloads a protein structure from the Protein Data Bank (PDB) and parses it into a `BioStructures.MolecularStructure`.
It also extracts protein names and species information associated with the structure.

# Arguments
- `structure_id::String`: The PDB identifier for the protein structure (e.g., "5A5B"). The ID will be converted to uppercase.
- `mail_address::String`: The email address for logging purposes. An email addess is required to query the UniProt API to convert the protein names into Gene Symbols. 

# Returns
- `structure::BioStructures.MolecularStructure`: The parsed molecular structure.
- `protein_names::Dict{String, String}`: A dictionary mapping chain IDs to their corresponding protein names.
- `species::Int`: The NCBI taxonomy ID of the organism.
- `verbose::Bool`: A flag to control verbosity.
Returns `(nothing, nothing, nothing)` if the download fails or an error occurs during processing.

# Details
The function constructs a URL to `https://files.rcsb.org/download/` using the provided `structure_id`.
It then uses `HTTP.get` to fetch the PDB file. If the download is successful (HTTP status 200),
the PDB data is parsed using `BioStructures.read`. The internal `mapProteinNames` function is
then called with the HTTP response and the parsed structure to extract chain-to-protein name mappings
and the species ID from the PDB file content.

Error handling is in place:
- A warning (`@warn`) is logged if the PDB file download fails (e.g., non-200 status code).
- An error (`@error`) is logged for other issues during the PDB fetch or parsing process (e.g., network issues, parsing errors).
In both error scenarios, the function returns `(nothing, nothing, nothing)`.

# Examples
```julia
structure, names, species_id = readPDB("6EZ8")
if structure !== nothing
    println("Successfully read PDB ID: 1AKE for species ID: \$species_id")
    for (chain, name) in names
        println("Chain \$chain maps to protein: \$name")
    end
else
    println("Failed to retrieve or parse PDB ID: 6EZ8")
end
```
"""
function readPDB(structure_id::String, mail_address::String; verbose::Bool = true)
    # construct the URL for the PDB file
    url = "https://files.rcsb.org/download/$(uppercase(structure_id)).pdb" 
    try 
        response = HTTP.get(url)
        if response.status == 200
            structure = BioStructures.read(IOBuffer(response.body), BioStructures.PDBFormat)::BioStructures.MolecularStructure
            # map chain IDs to protein names 
            protein_names, species = mapProteinNames(response, structure, mail_address)
            return structure::BioStructures.MolecularStructure, protein_names, species
        else
            @warn "Failed to download PDB file. Status code: $(response.status)"
            return nothing, nothing, nothing
        end 
    catch e   
        @error "Error during PDB fetch or parse for $structure_id: $e"
        return nothing, nothing, nothing
    end   
end 

"""
    mapProteinNames(response::HTTP.Messages.Response, structure::BioStructures.MolecularStructure, mail_address::String; verbose::Bool = false)

Extracts and refines protein names and determines the species ID from PDB data.

This function parses the raw PDB file content (from an HTTP response) to identify
the organism's taxonomy ID and initial protein molecule names associated with specific
chains. It then queries the UniProt database for each protein and uses sequence
alignment against the actual chain sequences from the `structure` object to
disambiguate and select the most appropriate gene name.

# Arguments
- `response::HTTP.Messages.Response`: The HTTP response object containing the PDB file
  content in its body. This is used to parse `ORGANISM_TAXID` and `COMPND` records.
- `structure::BioStructures.MolecularStructure`: The parsed molecular structure (e.g.,
  from `BioStructures.read`). This is used to extract the amino acid sequence of
  each chain for alignment purposes.
- `mail_address::String`: The user's email address, required for querying the
  UniProt API via the internal `queryPDB` function.

# Keyword Arguments
- `verbose::Bool`: A flag to control verbosity. If `true`, warnings are displayed.

# Returns
- `protein_names::Dict{String, String}`: A dictionary mapping chain identifiers (e.g., "A")
   to their refined protein names. This name is typically a gene name obtained from UniProt
  if a suitable UniProt entry is found and matched. If UniProt mapping fails or no
  suitable entry is found, the original molecule name extracted from the PDB file is used
  as a fallback.
- `species::Int`: The NCBI taxonomy ID for the organism, extracted from the
  `ORGANISM_TAXID` record in the PDB file.

# Details
The process involves several steps:
1.  **Species ID Extraction**: The `ORGANISM_TAXID` is parsed from the PDB file content.
2.  **Initial Protein Name Extraction**: `COMPND` records in the PDB file are parsed to
    create an initial mapping of chain IDs to molecule names.
3.  **UniProt Query and Disambiguation**: For each chain and its initial molecule name:
    a.  The `queryPDB` function is called to retrieve potential UniProt entries.
    b.  The chain's sequence is extracted from `structure`.
    c.  UniProt entries are filtered by length against the chain's sequence.
    d.  If multiple candidates remain, local sequence alignment (`BioAlignments.pairalign`
        with `BLOSUM62`) determines the best match. The STRING identifier from this match is used.
        If only one candidate remains after length filtering, its gene name is used.

# Error Handling and Edge Cases
- The function relies on the presence and correct formatting of `ORGANISM_TAXID` and
  `COMPND` records in the PDB file content. Errors in parsing these records can lead
  to failures in extracting the species ID or initial protein names.
- For protein name refinement using UniProt:
    - If `queryPDB` (the UniProt query function) returns `nothing` or an empty DataFrame.
    - If no UniProt entries pass the sequence length filter against the chain's sequence.
    - If no UniProt entry yields a sufficiently good alignment score to be considered a match.
  In these scenarios, a warning (`@warn`) is logged, and the function falls back to
  using the original protein name extracted from the PDB `COMPND` records for that chain.
- Parsing of UniProt sequences (retrieved as strings) into `BioSequences.LongAA` can fail
  if a sequence string is malformed. Such cases are logged with a warning, and the
  problematic UniProt entry is skipped during the alignment process for that chain.

# See Also
- `readPDB`: The typical caller of this function.
- `queryPDB`: Used internally to fetch data from UniProt.
"""
function mapProteinNames(response::HTTP.Messages.Response, structure::BioStructures.MolecularStructure, mail_address::String; verbose::Bool = false) 
    body = String(response.body)
    lines = split(body, '\n')

    # ---------------------------------------------------------------- #
    # get species ID from PDB header 
    # ---------------------------------------------------------------- #
    local species::Int # Ensure species is defined for the function scope
    species_id_str = ""
    for line in lines
        if occursin("ORGANISM_TAXID", line)
            parts = split(line, "ORGANISM_TAXID:")
            if length(parts) > 1
                species_id_str = strip(parts[2], [' ', ';']) # Strip spaces and semicolons
                break
            end
        end
    end

    if isempty(species_id_str)
        @error "ORGANISM_TAXID not found or malformed in PDB header." # Consider adding structure_id to log if available
        return nothing, nothing
    end
    parsed_species = tryparse(Int64, species_id_str)

    if parsed_species === nothing
        @error "Failed to parse ORGANISM_TAXID: '$species_id_str'"
        return nothing, nothing
    end

    species = parsed_species

    # ---------------------------------------------------------------- #
    # get chain names from header
    # ---------------------------------------------------------------- # 
    # 1. Keep only lines starting with COMPND
    filter!(line -> startswith(line, "COMPND"), lines)

    # 2. Extract MOL_ID and SYNONYM lines into a dictionary
    protein_names = Dict{String, String}()
    current_chain_str = ""
    current_molecule_name = ""

    for line in lines
        # extract CHAIN_ID
        if occursin("CHAIN", line)
            parts = split(line, "CHAIN:")
            if length(parts) == 2
                current_chain_str = strip(parts[2], [' ', ';'])
            else
                current_chain_str = "" # Invalidate if malformed
            end
        elseif occursin("MOLECULE", line) 
            parts = split(line, "MOLECULE:")
            if length(parts) == 2
                current_molecule_name = strip(parts[2], [' ', ';'])
                current_molecule_name = replace(current_molecule_name, r"[\/\\\(\)\?\*\[\]]" => "")
            else
                current_molecule_name = "" # Invalidate if malformed
            end
        end

        # If both a chain string and a molecule name have been identified for the current "block"
        if !isempty(current_chain_str) && !isempty(current_molecule_name)
            for chain_id in split(current_chain_str, ',')
                actual_chain_id = strip(chain_id)
                if !isempty(actual_chain_id)
                    protein_names[actual_chain_id] = current_molecule_name
                end
            end
            current_chain_str = ""    # Reset for the next pair
            current_molecule_name = "" # Reset for the next pair
        end
    end

    # remove all non-protein chains from the dictionary, i.e., containng 5'- or 3'-
    keys_to_delete = String[]
    for key in keys(protein_names)
        if occursin("5'", protein_names[key]) || occursin("3'", protein_names[key])
            push!(keys_to_delete, key)
        end
    end

    for key in keys_to_delete
        delete!(protein_names, key)
    end

    length(protein_names) == 0 && return nothing, nothing

    # 3. Query PDB for protein names 
    scoremodel = BioAlignments.AffineGapScoreModel(BioAlignments.BLOSUM62, gap_open=-5, gap_extend=-1)

    for key in keys(protein_names)
        original_pdb_name = protein_names[key] 
        uniprot_df = queryPDB(original_pdb_name, species, mail_address)

        # If no UniProt entries found, keep original PDB name
        if uniprot_df === nothing || isempty(uniprot_df)
            verbose && @warn "No UniProt entries found for PDB name '$(original_pdb_name)' (chain $key, species $species). Keeping original PDB name."
            continue # Skip to the next chain
        end

        protein_sequence = BioSequences.LongAA(structure[1][key], BioStructures.standardselector, gaps = false)

        # ---------------------------------------------------------------------------- #
        # remove all proteins from uniprot_df that are shorter than protein_sequence
        # Ensure the sequence column contains valid sequences for length check
        # ---------------------------------------------------------------------------- #
        valid_indices = Int[]
        for i in axes(uniprot_df, 1)
            seq_candidate = uniprot_df[i, 7] 
            if (typeof(seq_candidate) <: BioSequences.LongSequence || typeof(seq_candidate) <: AbstractString) &&
               length(protein_sequence) <= length(seq_candidate)
                push!(valid_indices, i)
            end
        end

        if isempty(valid_indices)
            verbose && @warn "No UniProt entries for PDB name '$(original_pdb_name)' (chain $key) matched length criteria. Keeping original PDB name."
            continue
        end
        uniprot_df = uniprot_df[valid_indices, :]

        # remove columns where STRING IDs are missing
        filter!(row -> ismissing(row.STRING) == false, uniprot_df)

        if isempty(uniprot_df)
            verbose && @warn "No UniProt entries for PDB name '$(original_pdb_name)' (chain $key) with available STRING IDs. Keeping original PDB name."
            continue
        end

        # ---------------------------------------------------------------------------- #
        # Select the UniProt entry with the highest sequence similarity
        # ---------------------------------------------------------------------------- #
        best_uniprot_row_idx = 0      
        
        if size(uniprot_df,1) == 1
            protein_names[key] = uniprot_df[1, 8]
        elseif size(uniprot_df,1) > 1
            best_score = -Inf

            for i in axes(uniprot_df, 1)
                uniprot_sequence_str = uniprot_df[i, 7]
                local uniprot_aa_sequence::BioSequences.LongAA
                try
                    uniprot_aa_sequence = BioSequences.LongAA(uniprot_sequence_str)
                catch e
                    verbose && @warn "Could not parse UniProt sequence for chain $key, entry $i ('$(uniprot_sequence_str)'). Skipping. Error: $e"
                    continue
                end
                
                alignment = BioAlignments.pairalign(
                    BioAlignments.LocalAlignment(),
                    protein_sequence, uniprot_aa_sequence,
                    scoremodel
                    )

                score = BioAlignments.score(alignment)
                if score > best_score
                    best_score = score
                    best_uniprot_row_idx = i
                end
            end 
            
            if best_uniprot_row_idx == 0
                verbose && @warn "No suitable UniProt entry found for PDB name '$(original_pdb_name)' (chain $key). Keeping original PDB name."
                continue
            end
            
            gene_names_str = String(uniprot_df[best_uniprot_row_idx, 8])
            protein_names[key] = isempty(strip(gene_names_str)) ? original_pdb_name : strip(gene_names_str, ';')
        else
            verbose && @warn "Could not find a suitable UniProt match by alignment for PDB name '$(original_pdb_name)' (chain $key). Keeping original PDB name."
        end
    end
    return protein_names, species
end

"""
    queryPDB(protein_name::String, species::Int, mail_address::String)

Queries the UniProt API to retrieve protein information based on a protein name and species ID.

The function constructs a request to the UniProtKB stream endpoint, requesting specific
protein attributes in TSV format. It handles potential gzip compression of the response
and parses the TSV data into a `DataFrame`.

# Arguments
- `protein_name::String`: The name of the protein to search for (e.g., "Hemoglobin subunit alpha").
  Spaces in the name are automatically URL-encoded.
- `species::Int`: The NCBI taxonomy ID of the organism to filter the search (e.g., `9606` for Homo sapiens).
- `mail_address::String`: The user's email address. This is required by UniProt's fair
  use policy and is sent as the `User-Agent` in the HTTP request.

# Returns
- `DataFrame`: A DataFrame containing the UniProt entries matching the query.
  The columns typically include: `Entry` (accession), `Reviewed` (status), `Entry Name` (ID),
  `Protein names`, `Gene Names`, `Organism`, `Sequence`, and `Cross-reference (string)`.
  The exact columns are determined by the `fields` parameter in the UniProt query URL.
- `nothing`: If the HTTP request to UniProt fails (e.g., network error, non-200 status code),
  or if an error occurs during the process. An error message is logged in such cases.

# Details
The function performs the following steps:
1.  URL-encodes spaces in the `protein_name`.
2.  Constructs a query URL for the UniProt API (`https://rest.uniprot.org/uniprotkb/stream`)
    requesting data in TSV format for fields: `accession`, `reviewed`, `id`,
    `protein_name`, `gene_names`, `organism_name`, and `sequence`. The query filters
    by the provided `protein_name` and `species` ID, and also includes `xref_string`.
3.  Sends an HTTP GET request with headers:
    - `Accept: text/tab-separated-values`
    - `User-Agent: [mail_address]`
4.  If the request is successful (HTTP status 200):
    - Checks for `Content-Encoding: gzip` and decompresses the response body if necessary
      using `CodecZlib.GzipDecompressorStream`.
    - Parses the (potentially decompressed) TSV string into a DataFrame using the
      `parse_uniprot_tsv_to_dataframe` function.
5.  If the HTTP request encounters an error or returns a non-200 status, an error is
    logged using `@error`, and `nothing` is returned.
6.  Note that the `xref_string` field is included in the query and is used by the calling
        function (`mapProteinNames`).
# See Also
- `mapProteinNames`: The primary function that typically calls `queryPDB`.
- `parse_uniprot_tsv_to_dataframe`: Helper function used to convert the TSV response.
"""
function queryPDB(protein_name::String, species::Int, mail_address::String)
    protein_name = replace(protein_name, " " => "%20")
    # construct the URL
    url = "https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Csequence%2Cxref_string&format=tsv&query=%28$(protein_name)+AND+%28taxonomy_id%3A$(species)%29%29"
    # define header
    headers = Dict(
        "Accept" => "text/tab-separated-values", 
        "User-Agent" => mail_address
        )

    try 
        response = HTTP.get(url, headers, verbose = 0)
        if response.status == 200
            body_bytes = response.body
            content_encoding = HTTP.header(response, "Content-Encoding", "")

            if content_encoding == "gzip"
                # Decompress if gzipped
                io = IOBuffer(body_bytes)
                gz_stream = CodecZlib.GzipDecompressorStream(io)
                decompressed_body = read(gz_stream, String)
                close(gz_stream)
                return parse_uniprot_tsv_to_dataframe(decompressed_body)
            else
                return parse_uniprot_tsv_to_dataframe(String(body_bytes)) 
            end
        end
    catch e   
        @warn "Error during UniProt fetch for $protein_name: $e"
        return nothing
    end
end


"""
    parse_uniprot_tsv_to_dataframe(tsv_string::String) -> DataFrame

Converts a tab-separated string (typically from UniProt API) into a DataFrame.
Assumes the TSV string has a header row.
"""
function parse_uniprot_tsv_to_dataframe(tsv_string::String)::DataFrame
    if isempty(tsv_string)
        @warn "Input TSV string is empty. Returning an empty DataFrame."
        # Return an empty DataFrame to achieve type stability.
        return DataFrame() 
    end
    try
        return CSV.File(IOBuffer(tsv_string), delim='\t') |> DataFrame
    catch e
        @error "Error parsing TSV string to DataFrame: $e"
        return DataFrame() # Return an empty DataFrame on error
    end
end


# --------------------------------------------------------------------------- #
# Functions to compute the interresidue distance between Cα atoms             #
# --------------------------------------------------------------------------- #
pairing_matrix(size::Int) = [j > i for i in 1:size, j in 1:size]

"""
    calculate_protein_complex(structure::BioStructures.MolecularStructure, protein_names::Dict{String, String})


Calculates and tabulates inter-chain distances for unique pairs of protein chains within a given molecular structure.

The function determines distances between every unique pair of chains in the provided `structure`.
The `BioStructures.distance` function is used for this calculation, which typically computes
the minimum distance between any two atoms of the respective chains.

# Arguments
- `structure::BioStructures.MolecularStructure`: The molecular structure object, typically obtained
  from parsing a PDB file (e.g., via `readPDB`), containing multiple protein chains.
- `protein_names::Dict{String, String}`: A dictionary mapping chain identifiers (e.g., "A") to
   their corresponding protein names (ENSEMBL IDs).

# Returns
- `DataFrame`: A DataFrame where each row represents a unique pair of interacting protein chains
  and their calculated distance. The DataFrame includes the following columns:
    - `Protein1::String`: The name of the first protein in the pair (obtained using the chain ID from `chain_names` and the `protein_names` dictionary).
    - `Protein2::String`: The name of the second protein in the pair (obtained using the chain ID from `chain_names` and the `protein_names` dictionary).
    - `Distance::Float16`: The calculated distance between the two protein chains. This is the
      minimum distance between any two atoms of the respective chains, as computed by
      `BioStructures.distance(chain1, chain2)`.
  Each row corresponds to a pair of chains `(chain_names[i], chain_names[j])` where `j > i`
  and the calculated distance is non-zero.

# Details
The function first retrieves all chain identifiers from the `structure`. It then utilizes a
helper function, `pairing_matrix`, to generate a boolean matrix that ensures distances
are calculated only for the upper triangle of the chain-pair matrix (i.e., for `j > i`),
avoiding redundant calculations and self-comparisons. The distances are stored as `Float16`
to conserve memory. Finally the function creates a DataFrame and returns it.

# Example
```julia
structure, protein_names_map, species_id = readPDB("5A5B", "your_email@example.com")
if structure !== nothing
    inter_chain_distances_df = calculate_protein_complex(structure, protein_names_map)
    println("Calculated inter-chain distance matrix for PDB ID 5A5B:")
    display(inter_chain_distances_df)
end
```
"""
function calculate_protein_complex(structure::BioStructures.MolecularStructure, protein_names::Dict{String, String})::DataFrame
    chain_names = BioStructures.chainids(structure)
    pairing = pairing_matrix(length(chain_names))

    result_array = zeros(Float16,length(chain_names), length(chain_names))
    for i in eachindex(chain_names), j in eachindex(chain_names)
        if pairing[i,j] == true
            result_array[i,j] = BioStructures.distance(structure[chain_names[i]],structure[chain_names[j]])
        end
    end
    return generateIRoutput(result_array, protein_names, chain_names)
end

"""
    generateIRoutput(distance_matrix::Matrix{Float16}, protein_names::Dict{String, String}, chain_names::Vector{String}) -> DataFrame

Converts an inter-chain distance matrix into a DataFrame of protein pairs and distances.

This function takes a matrix containing distances between protein chains, along with
dictionaries mapping chain identifiers to protein names. It extracts the non-zero
distances from the upper triangle of the distance matrix and formats them into a
tabular DataFrame suitable for further analysis or output.

# Arguments
- `distance_matrix::Matrix{Float16}`: An `N x N` matrix containing inter-chain distances,
  typically computed by `calculate_protein_complex`, with non-zero values expected
  only in the upper triangle (`j > i`).
- `protein_names::Dict{String, String}`: A dictionary mapping chain identifiers
  (corresponding to indices in `chain_names`) to their refined protein names.
- `chain_names::Vector{String}`: A vector of chain identifiers (e.g., "A", "B"),
  ordered such that `chain_names[i]` corresponds to the i-th row/column of the
  `distance_matrix`.

# Returns
- `DataFrame`: A DataFrame with three columns:
    - `Protein1::String`: The name of the first protein in the pair.
    - `Protein2::String`: The name of the second protein in the pair.
    - `Distance::Float16`: The calculated distance between the two protein chains.
  Each row corresponds to a unique pair of chains `(chain_names[i], chain_names[j])`
  where `j > i` and the distance `distance_matrix[i,j]` is non-zero.

# See Also
- `calculate_protein_complex`: The function that typically calls `generateIRoutput`
  and produces the `distance_matrix`.

# Example
```julia
# Assuming 'distance_matrix', 'protein_names_dict', and 'chain_ids_vec' are available
# from a call to calculate_protein_complex:
# distance_matrix = calculate_protein_complex(structure, protein_names_map)
# protein_names_dict = protein_names_map # from readPDB
# chain_ids_vec = BioStructures.chainids(structure)

# Example dummy data for illustration:
distance_matrix = Float16[0.0 10.5 0.0; 0.0 0.0 5.2; 0.0 0.0 0.0]
protein_names_dict = Dict("A" => "ProteinX", "B" => "ProteinY", "C" => "ProteinZ")
chain_ids_vec = ["A", "B", "C"]

result_df = generateIRoutput(distance_matrix, protein_names_dict, chain_ids_vec)

println("Generated DataFrame:")
display(result_df)
```
"""
function generateIRoutput(distance::Matrix{Float16}, protein_names::Dict{String, String}, chain_names::Vector{String})::DataFrame
    npairs = sum(distance .!= 0) 
    names_1 = String[];     sizehint!(names_1, npairs)
    names_2 = String[];     sizehint!(names_2, npairs)
    distances = Float16[];  sizehint!(distances, npairs)

    for j ∈ 1:size(distance,2)
        for i in 1:(j-1)
            if haskey(protein_names, chain_names[i]) && haskey(protein_names, chain_names[j])
                push!(names_1, protein_names[chain_names[i]])
                push!(names_2, protein_names[chain_names[j]])
                push!(distances, distance[i,j])
            end
        end
    end
    return DataFrame(Protein1 = names_1, Protein2 = names_2, Distance = distances)
end

function getPDBData(accession_file::String, mail_address::String)::Union{Nothing, DataFrame}
    # check files
    if !isfile(accession_file)
        @error "File $accession_file does not exist."
        return nothing
    end

    if isfile("interresidue_distance.csv")
        @error "File interresidue_distance.csv already exists. Please delete it if you want to overwrite it."
        return nothing
    end

    accession_list = CSV.read(accession_file, DataFrame, stringtype = String, types = String, delim = ';')[:,2]
    df = DataFrame()
    
    # initialise ProgressMeter
    p = Progress(
        size(accession_list,1), desc="Computing interresidue distances...",
        showspeed=true,
        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
        barlen = 50, dt = 10
        )

    # loop over accessions
    for accession in accession_list
        structure, protein_names, _ = readPDB(accession, mail_address, verbose = false)

        # if no structure found
        if structure === nothing || protein_names === nothing
            open("log_pdb.txt", "a") do f
                write(f, "No structure found for $accession\n")
            end
            ProgressMeter.next!(p)
            continue
        end
        
        # calculate interresidue distance
        distance = calculate_protein_complex(structure, protein_names)

        # filter non conerved names 
        regex_pattern = r"^\d+\.([A-Z]+\d+)+$"
        mask = occursin.(regex_pattern, distance.Protein1) .& occursin.(regex_pattern, distance.Protein2)
        distance = distance[mask, :]

        if isempty(distance)
            open("log_pdb.txt", "a") do f
                write(f, "None of the chain names match the regular expression for $accession\n") # Changed $ac to $accession and added newline
                end
            ProgressMeter.next!(p)
            continue
        end

        # append to output file 
        CSV.write("interresidue_distance.csv", distance, append = true)
        df = vcat(df, distance)
        ProgressMeter.next!(p)
    end
    ProgressMeter.finish!(p)
    return df
end

# --------------------------------------------------------------------------- #
# Run script                                                                  #
# --------------------------------------------------------------------------- #
#mail_address = "manuel.seefelder@uni-ulm.de"
#structure_id::String = "6EZ8"
#accession_file::String = "encodings/pdb_codes.csv"

#structure, protein_names, species = readPDB(structure_id, mail_address)
#distance = calculate_protein_complex(structure, protein_names)

#distance = getPDBData(accession_file, mail_address)
