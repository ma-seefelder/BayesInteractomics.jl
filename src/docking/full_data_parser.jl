# ═══════════════════════════════════════════════════════════════════════════════
# Full data parser: Extract pDockQ, mean pLDDT, interface contacts from
# AlphaFold Server full_data JSON files (~40MB each).
# ═══════════════════════════════════════════════════════════════════════════════

"""
Parsed scores from a single model's full_data JSON.
"""
struct FullDataScores
    pdockq::Float64
    mean_plddt_a::Float64
    mean_plddt_b::Float64
    n_interface_contacts::Int
    avg_interface_plddt::Float64
    ipae::Float64                  # Mean cross-chain PAE (NaN if not computed)
end

"""
    parse_full_data_json(json_bytes; contact_threshold=0.2) -> FullDataScores

Parse a full_data JSON (as bytes or string) and compute pDockQ + per-chain pLDDT.

Fields used from full_data:
- `contact_probs`: N×N matrix of predicted contact probabilities
- `atom_plddts`: per-atom pLDDT values
- `atom_chain_ids`: per-atom chain assignment ("A" or "B")
- `token_chain_ids`: per-residue chain assignment
"""
function parse_full_data_json(json_bytes::Union{Vector{UInt8}, String};
                              contact_threshold::Float64 = 0.2)::FullDataScores
    data = JSON3.read(json_bytes)

    token_chain_ids = data[:token_chain_ids]
    n_residues = length(token_chain_ids)

    # Identify chain boundaries at residue level
    chain_a_residues = findall(==(String("A")), String.(token_chain_ids))
    chain_b_residues = findall(==(String("B")), String.(token_chain_ids))

    # Per-chain mean pLDDT from atom-level data
    atom_plddts = Float64.(data[:atom_plddts])
    atom_chain_ids = String.(data[:atom_chain_ids])

    plddt_a = Float64[]
    plddt_b = Float64[]
    for (plddt, chain) in zip(atom_plddts, atom_chain_ids)
        if chain == "A"
            push!(plddt_a, plddt)
        elseif chain == "B"
            push!(plddt_b, plddt)
        end
    end
    mean_plddt_a = isempty(plddt_a) ? NaN : mean(plddt_a)
    mean_plddt_b = isempty(plddt_b) ? NaN : mean(plddt_b)

    # Interface contacts from contact_probs matrix
    contact_probs = data[:contact_probs]
    interface_residue_pairs = Tuple{Int, Int}[]
    interface_plddts = Float64[]

    # We need residue-level pLDDT. Approximate from token_res_ids mapping.
    # For simplicity, compute per-residue pLDDT by averaging atom pLDDTs per chain.
    # Use a simple mapping: spread atom pLDDTs evenly across residues per chain.
    residue_plddt = _compute_residue_plddt(data)

    for i in chain_a_residues
        row = contact_probs[i]
        for j in chain_b_residues
            prob = Float64(row[j])
            if prob > contact_threshold
                push!(interface_residue_pairs, (i, j))
                push!(interface_plddts, residue_plddt[i])
                push!(interface_plddts, residue_plddt[j])
            end
        end
    end

    n_contacts = length(interface_residue_pairs)
    avg_interface_plddt = isempty(interface_plddts) ? 0.0 : mean(interface_plddts)

    pdockq = compute_pdockq(avg_interface_plddt, n_contacts)

    # Compute iPAE (mean cross-chain PAE)
    ipae = _compute_ipae(data)

    return FullDataScores(pdockq, mean_plddt_a, mean_plddt_b, n_contacts, avg_interface_plddt, ipae)
end

"""
    _compute_ipae(data) -> Float64

Compute interface PAE (iPAE) as the mean PAE over all cross-chain residue pairs
(A->B and B->A). Returns NaN if no cross-chain pairs exist or PAE matrix not found.

Per Genz et al. 2025: iPAE is the mean predicted aligned error between all
inter-chain residue pairs.
"""
function _compute_ipae(data)::Float64
    # Get chain assignments
    if !haskey(data, :token_chain_ids)
        return NaN
    end
    token_chain_ids = String.(data[:token_chain_ids])

    chain_a_idx = findall(==("A"), token_chain_ids)
    chain_b_idx = findall(==("B"), token_chain_ids)

    # No cross-chain pairs possible
    (isempty(chain_a_idx) || isempty(chain_b_idx)) && return NaN

    # Get PAE matrix — check both possible keys
    pae_matrix = nothing
    if haskey(data, :pae)
        pae_matrix = data[:pae]
    elseif haskey(data, :predicted_aligned_error)
        pae_matrix = data[:predicted_aligned_error]
    end
    pae_matrix === nothing && return NaN

    # Compute mean PAE over all cross-chain pairs (A->B and B->A)
    total = 0.0
    count = 0
    for i in chain_a_idx
        row = pae_matrix[i]
        for j in chain_b_idx
            total += Float64(row[j])
            count += 1
        end
    end
    for i in chain_b_idx
        row = pae_matrix[i]
        for j in chain_a_idx
            total += Float64(row[j])
            count += 1
        end
    end

    count == 0 && return NaN
    return total / count
end

"""
Compute per-residue pLDDT from atom-level data. Uses token_res_ids to group atoms.
"""
function _compute_residue_plddt(data)::Vector{Float64}
    atom_plddts = Float64.(data[:atom_plddts])
    atom_chain_ids = String.(data[:atom_chain_ids])
    token_chain_ids = String.(data[:token_chain_ids])
    n_residues = length(token_chain_ids)

    # Approximate: distribute atoms evenly across residues per chain
    # Count atoms per chain
    n_atoms_a = count(==("A"), atom_chain_ids)
    n_atoms_b = count(==("B"), atom_chain_ids)
    n_res_a = count(==("A"), token_chain_ids)
    n_res_b = count(==("B"), token_chain_ids)

    residue_plddt = zeros(Float64, n_residues)

    # Chain A residues
    if n_res_a > 0 && n_atoms_a > 0
        a_atoms = [atom_plddts[i] for i in 1:length(atom_chain_ids) if atom_chain_ids[i] == "A"]
        atoms_per_res = n_atoms_a / n_res_a
        for (ri, res_idx) in enumerate(findall(==(String("A")), String.(token_chain_ids)))
            start_atom = round(Int, (ri - 1) * atoms_per_res) + 1
            end_atom = min(round(Int, ri * atoms_per_res), n_atoms_a)
            if start_atom <= end_atom
                residue_plddt[res_idx] = mean(a_atoms[start_atom:end_atom])
            end
        end
    end

    # Chain B residues
    if n_res_b > 0 && n_atoms_b > 0
        b_atoms = [atom_plddts[i] for i in 1:length(atom_chain_ids) if atom_chain_ids[i] == "B"]
        atoms_per_res = n_atoms_b / n_res_b
        for (ri, res_idx) in enumerate(findall(==(String("B")), String.(token_chain_ids)))
            start_atom = round(Int, (ri - 1) * atoms_per_res) + 1
            end_atom = min(round(Int, ri * atoms_per_res), n_atoms_b)
            if start_atom <= end_atom
                residue_plddt[res_idx] = mean(b_atoms[start_atom:end_atom])
            end
        end
    end

    return residue_plddt
end
