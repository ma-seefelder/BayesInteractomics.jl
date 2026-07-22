# ═══════════════════════════════════════════════════════════════════════════════
# Result parser: Parse AlphaFold Server output ZIPs and directories
# ═══════════════════════════════════════════════════════════════════════════════

"""
    BayesInteractomics.import_docking_results(results_dir, results; config) -> DockingResult

Parse AlphaFold Server result ZIPs/directories and cache scores.

Scans `results_dir` for ZIP files and extracted directories, parses
`summary_confidences` JSONs, optionally parses `full_data` JSONs for pDockQ,
caches per-pair results, and computes BF_dock for each pair.
"""
function import_docking_results(
        results_dir::String,
        results::DataFrame;
        config::DockingConfig = DockingConfig())::DockingResult

    @info "Importing docking results from $results_dir..."
    !isdir(results_dir) && error("Results directory does not exist: $results_dir")

    cache_dir = _resolve_cache_dir(config)

    # Find all result sources (ZIPs and directories)
    sources = _find_result_sources(results_dir)
    config.verbose && @info "  Found $(length(sources)) result source(s)"

    # Parse each source
    pairs = DockingPairResult[]
    n_cached = 0

    for source in sources
        # Extract job name and parse
        try
            pair_result, was_cached = _parse_single_result(source, config, cache_dir)
            if pair_result !== nothing
                push!(pairs, pair_result)
                was_cached && (n_cached += 1)
            end
        catch e
            @warn "Failed to parse result source" source=source exception=e
        end
    end

    # Deduplicate pairs — the same bait-prey pair can appear multiple times
    # if e.g. a ZIP and its extracted directory coexist, or a result was
    # resubmitted. Keep the best representative per canonical pair key
    # (successful status wins, then highest ipTM).
    pairs = _deduplicate_pairs(pairs)

    # Count statistics
    n_docked = count(p -> p.status == :success, pairs)
    n_disordered = count(p -> p.status == :disordered || p.calibration_tier == "disordered", pairs)
    n_too_large_in_pairs = count(p -> p.status == :too_large, pairs)
    n_total = length(pairs)

    # Count pairs that were skipped for token budget during request generation.
    # These never produced a pair in `pairs`, so we read them from
    # `skipped_too_large.csv` in the request output directory if available.
    n_skipped_too_large = _count_skipped_too_large(config.request_output_dir)
    n_too_large = n_too_large_in_pairs + n_skipped_too_large

    # Pending = proteins that passed the PEP/posterior candidate filter,
    # minus those already docked and those excluded (e.g. too many tokens).
    n_candidates = _count_pep_candidates(results, config)
    n_pending = max(0, n_candidates - n_total - n_skipped_too_large)

    @info "Docking results imported" n_total=n_total n_docked=n_docked n_cached=n_cached

    return DockingResult(
        pairs, config, n_total, n_docked, n_cached, n_pending,
        n_too_large, n_disordered, now(),
    )
end

# ─── Source discovery ─────────────────────────────────────────────────────────

"""
A result source: either a ZIP file or an already-extracted directory.
"""
struct ResultSource
    path::String
    is_zip::Bool
end

"""Find all ZIP files and result directories in the results directory."""
function _find_result_sources(results_dir::String)::Vector{ResultSource}
    sources = ResultSource[]

    for entry in readdir(results_dir)
        full_path = joinpath(results_dir, entry)

        if isfile(full_path) && endswith(lowercase(entry), ".zip")
            push!(sources, ResultSource(full_path, true))
        elseif isdir(full_path)
            # Check if it looks like an AF Server result directory
            files = readdir(full_path)
            if any(f -> contains(f, "summary_confidences"), files)
                push!(sources, ResultSource(full_path, false))
            end
        end
    end

    return sources
end

# ─── Single result parsing ────────────────────────────────────────────────────

"""
Parse a single result source (ZIP or directory). Returns (DockingPairResult, was_cached).
"""
function _parse_single_result(source::ResultSource, config::DockingConfig,
                              cache_dir::String)::Tuple{Union{DockingPairResult, Nothing}, Bool}
    if source.is_zip
        return _parse_zip_result(source.path, config, cache_dir)
    else
        return _parse_dir_result(source.path, config, cache_dir)
    end
end

"""Parse a ZIP file containing AlphaFold Server results."""
function _parse_zip_result(zip_path::String, config::DockingConfig,
                           cache_dir::String)::Tuple{Union{DockingPairResult, Nothing}, Bool}
    # Read ZIP contents
    zip_data = Base.read(zip_path)

    # Use a temporary directory to extract
    mktempdir() do tmpdir
        _extract_zip(zip_data, tmpdir)

        # Find the result subdirectory (there may be one job dir inside)
        subdirs = filter(d -> isdir(joinpath(tmpdir, d)), readdir(tmpdir))
        result_dir = if !isempty(subdirs)
            # Check if subdirectory has summary_confidences
            subdir = joinpath(tmpdir, first(subdirs))
            if any(f -> contains(f, "summary_confidences"), readdir(subdir))
                subdir
            else
                tmpdir
            end
        else
            tmpdir
        end

        return _parse_dir_result(result_dir, config, cache_dir)
    end
end

"""Extract a ZIP archive (basic implementation using shell)."""
function _extract_zip(zip_data::Vector{UInt8}, dest_dir::String)
    zip_path = joinpath(dest_dir, "_temp.zip")
    Base.write(zip_path, zip_data)

    if Sys.iswindows()
        run(pipeline(`powershell -Command "Expand-Archive -Path '$zip_path' -DestinationPath '$dest_dir' -Force"`, devnull))
    else
        run(pipeline(`unzip -o -q $zip_path -d $dest_dir`, devnull))
    end
    rm(zip_path, force=true)
end

"""Parse an extracted AlphaFold Server result directory."""
function _parse_dir_result(dir_path::String, config::DockingConfig,
                           cache_dir::String)::Tuple{Union{DockingPairResult, Nothing}, Bool}
    files = readdir(dir_path)

    # Find summary_confidences files
    conf_files = sort(filter(f -> contains(f, "summary_confidences") && endswith(f, ".json"), files))
    isempty(conf_files) && return (nothing, false)

    # Parse job request to identify the pair
    req_files = filter(f -> contains(f, "job_request") && endswith(f, ".json"), files)
    protein_a, protein_b, uniprot_a, uniprot_b, token_count = _parse_job_identity(
        dir_path, req_files)

    # Check cache
    pair_key = docking_cache_key(protein_a, protein_b)
    cached = _load_cached_pair(cache_dir, pair_key)
    if cached !== nothing
        return (cached, true)
    end

    # Parse all model summary_confidences
    model_scores = _parse_all_summary_confidences(dir_path, conf_files)
    isempty(model_scores) && return (nothing, false)

    # Select best model (max ipTM)
    best_idx = argmax([s.iptm for s in model_scores])
    best = model_scores[best_idx]

    iptm_all = [s.iptm for s in model_scores]
    iptm_best = maximum(iptm_all)
    iptm_std_val = length(iptm_all) > 1 ? std(iptm_all) : 0.0
    frac_disordered = minimum([s.fraction_disordered for s in model_scores])

    # Parse full_data for pDockQ if configured and ipTM suggests plausible interaction
    pdockq = NaN
    mean_plddt_a = NaN
    mean_plddt_b = NaN
    n_interface_contacts = 0

    # C2Qscore fields (computed from full_data when available)
    c2qscore_val = NaN
    ipae_raw = NaN
    iplddt_interface_val = NaN

    if config.parse_full_data
        full_data_scores = _try_parse_full_data(dir_path, files, best_idx)
        if full_data_scores !== nothing
            pdockq = full_data_scores.pdockq
            mean_plddt_a = full_data_scores.mean_plddt_a
            mean_plddt_b = full_data_scores.mean_plddt_b
            n_interface_contacts = full_data_scores.n_interface_contacts
            ipae_raw = full_data_scores.ipae

            # Compute C2Qscore when iPAE is available
            if !isnan(full_data_scores.ipae)
                iplddt_interface_val = full_data_scores.avg_interface_plddt / 100.0
                ipae_norm = 1.0 - (full_data_scores.ipae / 31.75)
                c2qscore_val = compute_c2qscore(iplddt_interface_val, ipae_norm, best.ptm, iptm_best)
            end
        end
    end

    # Compute BF
    mean_plddt_min_val = NaN
    if !isnan(mean_plddt_a) && !isnan(mean_plddt_b)
        mean_plddt_min_val = min(mean_plddt_a, mean_plddt_b)
    end

    bf_dock, calibration_tier = compute_bf_dock(
        iptm_best, frac_disordered;
        pdockq = pdockq,
        c2qscore = c2qscore_val,
        iptm_std = iptm_std_val,
        chain_pair_pae_min = best.chain_pair_pae_min,
        mean_plddt_min = mean_plddt_min_val,
    )

    status = if calibration_tier == "disordered"
        :disordered
    else
        :success
    end

    pair = DockingPairResult(
        protein_a, protein_b, uniprot_a, uniprot_b,
        iptm_best, iptm_all, iptm_std_val,
        best.ranking_score, frac_disordered,
        best.chain_pair_iptm, best.chain_pair_pae_min,
        pdockq, mean_plddt_a, mean_plddt_b, n_interface_contacts,
        bf_dock, calibration_tier, status, token_count,
        c2qscore_val, ipae_raw, iplddt_interface_val,
    )

    # Cache result
    _save_cached_pair(cache_dir, pair_key, pair)

    # Save best model CIF to structures directory for report 3D viewer
    _save_best_model_cif(dir_path, files, best_idx, cache_dir, pair_key)

    return (pair, false)
end

"""Copy the best model CIF file to `{cache_dir}/structures/{pair_key}.cif` for the report viewer."""
function _save_best_model_cif(dir_path::String, files::Vector{String},
                              best_idx::Int, cache_dir::String, pair_key::String)
    model_idx_0based = best_idx - 1
    cif_candidates = filter(f -> contains(f, "model_$(model_idx_0based)") && endswith(f, ".cif"), files)
    isempty(cif_candidates) && return nothing

    src = joinpath(dir_path, first(cif_candidates))
    !isfile(src) && return nothing

    structures_dir = joinpath(cache_dir, "structures")
    mkpath(structures_dir)
    dest = joinpath(structures_dir, pair_key * ".cif")
    cp(src, dest; force=true)
    return dest
end

# ─── Summary confidences parsing ─────────────────────────────────────────────

"""Parsed scores from a single model's summary_confidences JSON."""
struct ModelSummary
    iptm::Float64
    ptm::Float64
    ranking_score::Float64
    fraction_disordered::Float64
    has_clash::Float64
    chain_pair_iptm::Float64       # Off-diagonal [1,2]
    chain_pair_pae_min::Float64    # Off-diagonal [1,2]
end

"""Parse all summary_confidences files in a directory."""
function _parse_all_summary_confidences(dir_path::String,
                                        conf_files::Vector{String})::Vector{ModelSummary}
    summaries = ModelSummary[]
    for f in conf_files
        try
            json_str = String(Base.read(joinpath(dir_path, f)))
            data = JSON3.read(json_str)

            # Extract off-diagonal chain_pair values
            cp_iptm = _get_off_diagonal(data, :chain_pair_iptm)
            cp_pae = _get_off_diagonal(data, :chain_pair_pae_min)

            push!(summaries, ModelSummary(
                Float64(data[:iptm]),
                Float64(data[:ptm]),
                Float64(data[:ranking_score]),
                Float64(data[:fraction_disordered]),
                Float64(data[:has_clash]),
                cp_iptm,
                cp_pae,
            ))
        catch e
            @warn "Failed to parse summary_confidences" file=f exception=e
        end
    end
    return summaries
end

"""Extract off-diagonal element from a 2×2 chain_pair matrix (row 1, col 2)."""
function _get_off_diagonal(data, field::Symbol)::Float64
    if haskey(data, field)
        matrix = data[field]
        if length(matrix) >= 1 && length(matrix[1]) >= 2
            return Float64(matrix[1][2])
        end
    end
    return NaN
end

# ─── Job identity resolution ─────────────────────────────────────────────────

"""Parse job_request.json to extract pair identity and token count."""
function _parse_job_identity(dir_path::String, req_files::Vector{String})
    protein_a = "UNKNOWN_A"
    protein_b = "UNKNOWN_B"
    uniprot_a = ""
    uniprot_b = ""
    token_count = 0

    if !isempty(req_files)
        try
            json_str = String(Base.read(joinpath(dir_path, first(req_files))))
            data = JSON3.read(json_str)

            # AlphaFold Server format: array with one job dict
            job = data isa AbstractVector ? data[1] : data

            # Parse name: "BAIT:PREY" format
            if haskey(job, :name)
                name = String(job[:name])
                parts = split(name, ':')
                if length(parts) >= 2
                    protein_a = String(parts[1])
                    protein_b = String(parts[2])
                end
            end

            # Count tokens from sequences
            if haskey(job, :sequences)
                for seq_entry in job[:sequences]
                    if haskey(seq_entry, :proteinChain)
                        chain = seq_entry[:proteinChain]
                        if haskey(chain, :sequence)
                            token_count += length(String(chain[:sequence]))
                        end
                    end
                end
            end
        catch e
            @warn "Failed to parse job_request" exception=e
        end
    end

    # Fallback: try to extract from directory name
    if protein_a == "UNKNOWN_A"
        dir_name = basename(dir_path)
        parts = split(dir_name, '_')
        if length(parts) >= 2
            protein_a = uppercase(String(parts[1]))
            protein_b = uppercase(String(parts[end]))
        end
    end

    return protein_a, protein_b, uniprot_a, uniprot_b, token_count
end

# ─── Full data parsing ────────────────────────────────────────────────────────

"""Try to parse the full_data JSON for the best model. Returns FullDataScores or nothing."""
function _try_parse_full_data(dir_path::String, files::Vector{String},
                              best_model_idx::Int)::Union{FullDataScores, Nothing}
    # Look for full_data file matching best model index (0-based in filenames)
    model_idx_0based = best_model_idx - 1  # Julia 1-based → AF 0-based
    target = "full_data_$(model_idx_0based).json"

    full_data_file = findfirst(f -> contains(f, target), files)
    if full_data_file === nothing
        # Try any full_data file
        full_data_file = findfirst(f -> contains(f, "full_data") && endswith(f, ".json"), files)
    end

    full_data_file === nothing && return nothing

    try
        json_bytes = Base.read(joinpath(dir_path, files[full_data_file]))
        return parse_full_data_json(json_bytes)
    catch e
        @warn "Failed to parse full_data" file=files[full_data_file] exception=e
        return nothing
    end
end

# ─── Deduplication ───────────────────────────────────────────────────────────

"""
Return the better of two `DockingPairResult` values for the same pair key.
Preference: `:success` over anything else; then the higher ipTM wins.
"""
function _prefer_pair(a::DockingPairResult, b::DockingPairResult)::DockingPairResult
    a_success = a.status == :success
    b_success = b.status == :success
    a_success && !b_success && return a
    b_success && !a_success && return b
    iptm_a = isfinite(a.iptm_best) ? a.iptm_best : -Inf
    iptm_b = isfinite(b.iptm_best) ? b.iptm_best : -Inf
    return iptm_a >= iptm_b ? a : b
end

"""
Collapse duplicate pairs sharing the same canonical bait-prey key, keeping
the best representative per `_prefer_pair`.
"""
function _deduplicate_pairs(pairs::Vector{DockingPairResult})::Vector{DockingPairResult}
    isempty(pairs) && return pairs
    by_key = Dict{String, DockingPairResult}()
    order = String[]
    for p in pairs
        key = docking_cache_key(p.protein_a, p.protein_b)
        if haskey(by_key, key)
            by_key[key] = _prefer_pair(by_key[key], p)
        else
            by_key[key] = p
            push!(order, key)
        end
    end
    return [by_key[k] for k in order]
end

# ─── Candidate / skipped counting ────────────────────────────────────────────

"""Count proteins in `results` that pass the PEP and posterior thresholds in `config`."""
function _count_pep_candidates(results::DataFrame, config::DockingConfig)::Int
    nrow(results) == 0 && return 0
    mask = trues(nrow(results))

    if hasproperty(results, :is_detected)
        mask .&= coalesce.(results.is_detected .=== true, false)
    end
    if hasproperty(results, :posterior_prob)
        mask .&= coalesce.(results.posterior_prob .>= config.posterior_threshold, false)
    end
    if hasproperty(results, :PEP)
        mask .&= coalesce.(results.PEP .<= config.pep_threshold, false)
    elseif hasproperty(results, :posterior_prob)
        mask .&= coalesce.((1.0 .- results.posterior_prob) .<= config.pep_threshold, false)
    end
    if hasproperty(results, :diagnostic_flag)
        mask .&= _diagnostic_flag_ok.(results.diagnostic_flag)
    end

    return count(mask)
end

"""Read `skipped_too_large.csv` from the request output directory. Returns 0 when absent."""
function _count_skipped_too_large(request_output_dir::String)::Int
    isempty(request_output_dir) && return 0
    path = joinpath(request_output_dir, "skipped_too_large.csv")
    isfile(path) || return 0
    try
        lines = eachline(path)
        n = 0
        for (i, line) in enumerate(lines)
            i == 1 && continue  # skip header
            isempty(strip(line)) && continue
            n += 1
        end
        return n
    catch e
        @warn "Failed to read skipped_too_large.csv" path=path exception=e
        return 0
    end
end
