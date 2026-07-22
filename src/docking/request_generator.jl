# ═══════════════════════════════════════════════════════════════════════════════
# Request generation: AlphaFold Server JSON files + batch organization
# ═══════════════════════════════════════════════════════════════════════════════

"""
    BayesInteractomics.generate_docking_requests(results, bait_sequence; kwargs...) -> DockingRequestBatch

Generate AlphaFold Server JSON request files for high-confidence candidates.

Creates one JSON file per bait-prey pair, organized into daily batches of
≤30 jobs. Skips pairs already in cache and pairs exceeding 5000 tokens.
Writes an `upload_guide.txt` with step-by-step instructions.

# Arguments
- `results::DataFrame`: Output from `analyse()` or `run_analysis()`
- `bait_sequence::String`: Amino acid sequence of the bait protein

# Keywords
- `bait_name::String = "BAIT"`: Name/gene symbol for the bait
- `output_dir::String = "docking_requests"`: Output directory
- `fasta_file::String = ""`: Path to local FASTA with prey sequences
- `config::DockingConfig = DockingConfig()`: Configuration
"""
function generate_docking_requests(
        results::DataFrame,
        bait_sequence::String;
        bait_name::String = "BAIT",
        output_dir::String = "docking_requests",
        fasta_file::String = "",
        config::DockingConfig = DockingConfig())::DockingRequestBatch

    @info "Generating docking requests..."
    bait_sequence = _clean_sequence(bait_sequence)
    isempty(bait_sequence) && error("Bait sequence is empty")
    bait_len = length(bait_sequence)

    # Step 1: Filter candidates
    candidates = _filter_candidates(results, config)
    config.verbose && @info "  $(nrow(candidates)) candidates pass posterior/PEP filters"

    # Step 2: Resolve prey sequences
    prey_seqs = resolve_prey_sequences(candidates, config; fasta_file=fasta_file)
    config.verbose && @info "  Resolved $(length(prey_seqs))/$(nrow(candidates)) prey sequences"

    # Step 3: Load cache index for deduplication
    cache_dir = _resolve_cache_dir(config)
    cached_keys = _load_cache_index(cache_dir)

    # Step 4: Classify pairs
    requests = Dict{String, Any}[]  # Pairs to submit
    skipped_cached = String[]
    skipped_too_large = Tuple{String, Int}[]

    for row in eachrow(candidates)
        protein = row.Protein
        haskey(prey_seqs, protein) || continue

        prey_seq = prey_seqs[protein]
        token_count = bait_len + length(prey_seq)

        # Deduplication
        uniprot_prey = _get_uniprot(row)
        pair_key = docking_cache_key(bait_name, protein)

        if pair_key in cached_keys
            push!(skipped_cached, protein)
            continue
        end

        # Token budget
        if token_count > config.max_tokens_per_job
            push!(skipped_too_large, (protein, token_count))
            continue
        end

        push!(requests, Dict(
            "protein" => protein,
            "prey_seq" => prey_seq,
            "uniprot_prey" => uniprot_prey,
            "token_count" => token_count,
            "composite_score" => hasproperty(row, :composite_score) ? row.composite_score : NaN,
            "mean_plddt_monomer" => hasproperty(row, :mean_plddt_monomer) ? row.mean_plddt_monomer : NaN,
            "frac_disordered_monomer" => hasproperty(row, :frac_disordered_monomer) ? row.frac_disordered_monomer : NaN,
        ))

        length(requests) >= config.max_pairs && break
    end

    config.verbose && @info "  $(length(requests)) pairs to submit, " *
        "$(length(skipped_cached)) cached, $(length(skipped_too_large)) too large"

    # Step 5: Create output directory structure
    mkpath(output_dir)

    # Organize into batches
    n_batches = max(1, ceil(Int, length(requests) / config.max_jobs_per_batch))
    batch_dirs = String[]

    for b in 1:n_batches
        batch_dir = joinpath(output_dir, "batch_$b")
        mkpath(batch_dir)
        push!(batch_dirs, batch_dir)

        batch_start = (b - 1) * config.max_jobs_per_batch + 1
        batch_end = min(b * config.max_jobs_per_batch, length(requests))

        batch_jobs = Dict{String, Any}[]
        for i in batch_start:batch_end
            req = requests[i]
            filename = "$(bait_name)_$(req["protein"]).json"
            filepath = joinpath(batch_dir, filename)

            json_content = _build_af_request_json(
                bait_name, bait_sequence,
                req["protein"], req["prey_seq"]
            )
            Base.write(filepath, json_content)

            # Collect job for combined batch file
            push!(batch_jobs, _build_af_request_dict(
                bait_name, bait_sequence,
                req["protein"], req["prey_seq"]
            ))
        end

        # Write combined batch file with all jobs as a JSON array
        combined_path = joinpath(batch_dir, "batch_$(b)_combined.json")
        Base.write(combined_path, JSON3.write(batch_jobs))
    end

    # Step 6: Write skipped files
    if !isempty(skipped_cached)
        Base.write(joinpath(output_dir, "skipped_cached.csv"),
            "protein\n" * join(skipped_cached, "\n") * "\n")
    end
    if !isempty(skipped_too_large)
        lines = ["protein,token_count"]
        for (p, tc) in skipped_too_large
            push!(lines, "$p,$tc")
        end
        Base.write(joinpath(output_dir, "skipped_too_large.csv"), join(lines, "\n") * "\n")
    end

    # Step 7: Write manifest
    manifest = Dict(
        "bait_name" => bait_name,
        "bait_length" => bait_len,
        "n_requests" => length(requests),
        "n_batches" => n_batches,
        "n_skipped_cached" => length(skipped_cached),
        "n_skipped_too_large" => length(skipped_too_large),
        "pairs" => [Dict(
            "protein" => r["protein"],
            "tokens" => r["token_count"],
            "composite_score" => get(r, "composite_score", NaN),
            "mean_plddt_monomer" => get(r, "mean_plddt_monomer", NaN),
            "frac_disordered_monomer" => get(r, "frac_disordered_monomer", NaN),
        ) for r in requests],
    )
    manifest_path = joinpath(output_dir, "request_manifest.json")
    Base.write(manifest_path, JSON3.write(manifest))

    # Step 8: Write upload guide
    guide_path = joinpath(output_dir, "upload_guide.txt")
    _write_upload_guide(guide_path, bait_name, requests, n_batches, config,
                        skipped_cached, skipped_too_large)

    @info "Docking requests generated" n_requests=length(requests) n_batches=n_batches path=output_dir

    return DockingRequestBatch(
        batch_dirs, length(requests), n_batches,
        length(skipped_cached), length(skipped_too_large),
        manifest_path, guide_path,
    )
end

# ─── Helpers ──────────────────────────────────────────────────────────────────

"""Clean amino acid sequence: uppercase, remove non-standard characters."""
function _clean_sequence(seq::AbstractString)::String
    valid_aa = Set("ACDEFGHIKLMNPQRSTVWY")
    return String(filter(c -> uppercase(c) in valid_aa, uppercase(strip(String(seq)))))
end

"""Filter results DataFrame to docking candidates."""
function _filter_candidates(results::DataFrame, config::DockingConfig)::DataFrame
    mask = trues(nrow(results))

    # Exclude non-detected proteins (they have missing analytics and no dose-response signal)
    if hasproperty(results, :is_detected)
        mask .&= coalesce.(results.is_detected .=== true, false)
    end

    if hasproperty(results, :posterior_prob)
        mask .&= coalesce.(results.posterior_prob .>= config.posterior_threshold, false)
    end
    if hasproperty(results, :PEP)
        mask .&= coalesce.(results.PEP .<= config.pep_threshold, false)
    elseif hasproperty(results, :posterior_prob)
        # Derive PEP on the fly if the column is missing
        mask .&= coalesce.((1.0 .- results.posterior_prob) .<= config.pep_threshold, false)
    end

    # Only dock proteins whose diagnostic_flag is "ok" (when diagnostics were run).
    # If the column is absent, diagnostics were disabled — do not filter.
    if hasproperty(results, :diagnostic_flag)
        mask .&= _diagnostic_flag_ok.(results.diagnostic_flag)
    end

    candidates = results[mask, :]

    # Fetch structural dockability scores from AlphaFold DB (also resolves UniProt IDs).
    # Cache uses hybrid TTL + version check: fresh entries (<plddt_cache_max_age_days)
    # served immediately; stale entries trigger a cheap entryId comparison and only
    # refetch per-residue scores if AF DB has released a new version.
    cb = _resolve_cache_base(config)
    dockability_scores, uniprot_map = fetch_dockability_scores(candidates;
        species=config.species,
        cache_base=cb,
        cache_max_age_days=config.plddt_cache_max_age_days,
        force_refresh=config.plddt_force_refresh)

    # Compute composite score: w * dockability + (1-w) * posterior
    w = config.dockability_weight
    composite = map(eachrow(candidates)) do row
        posterior = hasproperty(row, :posterior_prob) ? coalesce(row.posterior_prob, 0.0) : 0.0
        dockability = get(dockability_scores, row.Protein, 0.5)  # neutral fallback
        (1.0 - w) * Float64(posterior) + w * Float64(dockability)
    end

    candidates = copy(candidates)  # avoid modifying input

    # Add UniProt ID column so downstream _get_uniprot() finds resolved accessions
    candidates.uniprot_id = map(row -> get(uniprot_map, row.Protein, ""), eachrow(candidates))
    candidates.composite_score = composite
    candidates.mean_plddt_monomer = map(eachrow(candidates)) do row
        uid = _get_uniprot(row)
        isempty(uid) && return NaN
        result = _load_plddt_cache(uid; cache_base=cb)
        result === nothing ? NaN : result.mean_plddt
    end
    candidates.frac_disordered_monomer = map(eachrow(candidates)) do row
        uid = _get_uniprot(row)
        isempty(uid) && return NaN
        result = _load_plddt_cache(uid; cache_base=cb)
        result === nothing ? NaN : result.frac_disordered
    end

    # Sort by composite score descending (replaces posterior-only sort)
    candidates = sort(candidates, :composite_score, rev=true)

    return candidates
end

"""
Return true when `diagnostic_flag` represents an OK baseline.
Accepts only the exact string "ok" (matching the report UI's green-check rule).
Missing or empty values are treated as not-OK so proteins without diagnostics
are excluded from docking when the column is present.
"""
function _diagnostic_flag_ok(flag)::Bool
    ismissing(flag) && return false
    return strip(string(flag)) == "ok"
end

"""Get UniProt ID from a results row, if available."""
function _get_uniprot(row)::String
    for col in (:uniprot_id, :UniProt, :uniprot)
        if hasproperty(row, col)
            val = getproperty(row, col)
            !ismissing(val) && return string(val)
        end
    end
    return ""
end

"""Build AlphaFold Server request Dict for a bait-prey pair."""
function _build_af_request_dict(bait_name::String, bait_seq::String,
                                prey_name::String, prey_seq::String)::Dict{String, Any}
    return Dict{String, Any}(
        "name" => "$(bait_name):$(prey_name)",
        "modelSeeds" => Int[],
        "sequences" => [
            Dict("proteinChain" => Dict(
                "sequence" => bait_seq,
                "count" => 1,
            )),
            Dict("proteinChain" => Dict(
                "sequence" => prey_seq,
                "count" => 1,
            )),
        ],
        "dialect" => "alphafoldserver",
        "version" => 1,
    )
end

"""Build AlphaFold Server JSON request for a bait-prey pair (single-job array)."""
function _build_af_request_json(bait_name::String, bait_seq::String,
                                prey_name::String, prey_seq::String)::String
    return JSON3.write([_build_af_request_dict(bait_name, bait_seq, prey_name, prey_seq)])
end

"""Write the upload guide text file."""
function _write_upload_guide(path::String, bait_name::String,
                             requests::Vector, n_batches::Int, config::DockingConfig,
                             skipped_cached, skipped_too_large)
    lines = String[]
    push!(lines, "=== BayesInteractomics Docking Request Guide ===")
    push!(lines, "")
    push!(lines, "Generated: $(format(now(), "yyyy-mm-ddTHH:MM:SS"))")
    push!(lines, "Total pairs to dock: $(length(requests))")
    push!(lines, "Batches: $n_batches (≤$(config.max_jobs_per_batch) per day)")
    push!(lines, "Skipped (cached): $(length(skipped_cached))")
    push!(lines, "Skipped (too large): $(length(skipped_too_large))")
    push!(lines, "")
    push!(lines, "INSTRUCTIONS:")
    push!(lines, "1. Go to https://alphafoldserver.com")
    push!(lines, "2. Sign in with your Google account")
    push!(lines, "3. For each batch directory (batch_1/, batch_2/, ...):")
    push!(lines, "   a. Click \"New fold\" or use the upload feature")
    push!(lines, "   b. Upload the combined file (batch_N_combined.json) to submit all jobs at once,")
    push!(lines, "      or upload individual .json files one by one")
    push!(lines, "   c. Wait for all jobs to complete (~30 min each)")
    push!(lines, "   d. Download all result ZIP files")
    push!(lines, "4. Place all downloaded ZIP files in a single directory")
    push!(lines, "5. Run:")
    push!(lines, "   julia> docking = import_docking_results(\"path/to/zips/\", results)")
    push!(lines, "   julia> updated = apply_docking_update(results, docking)")
    push!(lines, "")

    for b in 1:n_batches
        batch_start = (b - 1) * config.max_jobs_per_batch + 1
        batch_end = min(b * config.max_jobs_per_batch, length(requests))
        push!(lines, "BATCH $b ($(batch_end - batch_start + 1) jobs, upload on day $b):")
        for i in batch_start:batch_end
            r = requests[i]
            push!(lines, "  $(bait_name)_$(r["protein"]).json  ($(r["token_count"]) tokens)")
        end
        push!(lines, "")
    end

    push!(lines, "NOTE: Do NOT re-upload pairs listed in skipped_cached.csv —")
    push!(lines, "      these already have results from previous runs.")
    push!(lines, "")
    push!(lines, "IMPORTANT: AlphaFold Server output is for non-commercial use only.")
    push!(lines, "See https://alphafoldserver.com for Terms of Use.")

    Base.write(path, join(lines, "\n") * "\n")
end
