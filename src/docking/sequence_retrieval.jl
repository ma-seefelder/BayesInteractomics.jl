# ═══════════════════════════════════════════════════════════════════════════════
# Sequence retrieval: UniProt FASTA fetching + local FASTA support
# ═══════════════════════════════════════════════════════════════════════════════

"""
    canonical_uniprot_id(id) -> String

Strip any isoform suffix (e.g. `-2`) from a UniProt accession so that subsequent
lookups resolve to the canonical sequence. Whitespace and surrounding version
tokens are also removed. Returns an empty string for empty/missing input.
"""
function canonical_uniprot_id(id::AbstractString)::String
    s = strip(String(id))
    isempty(s) && return ""
    # Drop an explicit isoform suffix like "-2" on "P12345-2".
    # Match only trailing digit suffixes so "O00-DUMMY" style IDs (rare) are left alone.
    return replace(s, r"-\d+$" => "")
end

"""
    fetch_uniprot_sequence(uniprot_id; timeout=30) -> String

Fetch the canonical protein sequence from the UniProt REST API.

Any isoform suffix (e.g. `P12345-2`) is stripped before the request so the
endpoint always returns the canonical isoform. Returns an empty string on
failure.
"""
function fetch_uniprot_sequence(uniprot_id::AbstractString; timeout::Int = 30)::String
    accession = canonical_uniprot_id(uniprot_id)
    isempty(accession) && return ""
    url = "https://rest.uniprot.org/uniprotkb/$(accession).fasta"
    try
        buf = IOBuffer()
        Downloads.download(url, buf; timeout=timeout)
        fasta = String(take!(buf))
        # Parse FASTA: skip header line(s), join sequence lines
        lines = split(fasta, '\n')
        seq_lines = filter(l -> !startswith(l, '>') && !isempty(strip(l)), lines)
        return uppercase(join(strip.(seq_lines)))
    catch e
        @warn "Failed to fetch canonical sequence for $uniprot_id (accession $accession)" exception=e
        return ""
    end
end

"""
    read_fasta_file(path) -> Dict{String, String}

Read a FASTA file and return a dictionary mapping header identifiers to sequences.
Recognizes UniProt accessions (sp|ACCESSION|NAME or tr|ACCESSION|NAME format)
and plain identifiers (first word of header line).
"""
function read_fasta_file(path::AbstractString)::Dict{String, String}
    sequences = Dict{String, String}()
    !isfile(path) && return sequences

    current_id = ""
    current_seq = String[]

    for line in eachline(path)
        line = strip(line)
        isempty(line) && continue

        if startswith(line, '>')
            # Store previous sequence
            if !isempty(current_id) && !isempty(current_seq)
                sequences[current_id] = uppercase(join(current_seq))
            end
            # Parse header
            header = line[2:end]
            # Try sp|P12345|NAME or tr|P12345|NAME format
            m = match(r"^(?:sp|tr)\|([A-Za-z0-9_]+)\|", header)
            if m !== nothing
                current_id = uppercase(m.captures[1])
            else
                # Use first word
                current_id = uppercase(split(header)[1])
            end
            current_seq = String[]
        else
            push!(current_seq, line)
        end
    end
    # Store last sequence
    if !isempty(current_id) && !isempty(current_seq)
        sequences[current_id] = uppercase(join(current_seq))
    end

    return sequences
end

"""
    resolve_prey_sequences(results, config; fasta_file="") -> Dict{String, String}

Resolve prey protein sequences. Uses local FASTA if provided, otherwise
fetches from UniProt. Returns Dict mapping protein name → amino acid sequence.
"""
function resolve_prey_sequences(results::DataFrame, config::DockingConfig;
                                fasta_file::String = "")::Dict{String, String}
    sequences = Dict{String, String}()

    # Load from FASTA file if provided
    if !isempty(fasta_file) && isfile(fasta_file)
        fasta_seqs = read_fasta_file(fasta_file)
        @info "  Loaded $(length(fasta_seqs)) sequences from FASTA file"
    else
        fasta_seqs = Dict{String, String}()
    end

    # Identify proteins to fetch
    prog = Progress(nrow(results); desc="Resolving prey sequences", enabled=nrow(results) > 1)
    for row in eachrow(results)
        protein = row.Protein

        # Check FASTA first (by protein name or UniProt ID)
        if haskey(fasta_seqs, uppercase(protein))
            sequences[protein] = fasta_seqs[uppercase(protein)]
            ProgressMeter.next!(prog)
            continue
        end

        # Try UniProt ID if available in results
        uniprot_id = ""
        if hasproperty(results, :uniprot_id) && !ismissing(row.uniprot_id)
            uniprot_id = string(row.uniprot_id)
        elseif hasproperty(results, :UniProt) && !ismissing(row.UniProt)
            uniprot_id = string(row.UniProt)
        end

        # Check FASTA by UniProt ID (try the as-provided ID first, then the
        # canonical accession so that local FASTAs with canonical-only entries
        # still resolve when the results carry an isoform-suffixed ID).
        if !isempty(uniprot_id)
            if haskey(fasta_seqs, uppercase(uniprot_id))
                sequences[protein] = fasta_seqs[uppercase(uniprot_id)]
                ProgressMeter.next!(prog)
                continue
            end
            canonical = canonical_uniprot_id(uniprot_id)
            if !isempty(canonical) && haskey(fasta_seqs, uppercase(canonical))
                sequences[protein] = fasta_seqs[uppercase(canonical)]
                ProgressMeter.next!(prog)
                continue
            end
        end

        # Fetch the canonical sequence from UniProt if we have an ID
        if !isempty(uniprot_id)
            seq = fetch_uniprot_sequence(uniprot_id)
            if !isempty(seq)
                sequences[protein] = seq
                ProgressMeter.next!(prog)
                continue
            end
        end

        # Skip proteins without sequences
        config.verbose && @warn "  No sequence found for $protein"
        ProgressMeter.next!(prog)
    end
    ProgressMeter.finish!(prog)

    return sequences
end
