# ═══════════════════════════════════════════════════════════════════════════════
# Cache management: JLD2 per-pair docking result cache
# ═══════════════════════════════════════════════════════════════════════════════

"""Resolve the `.bayesinteractomics_cache` base directory from DockingConfig.
Returns empty string when cache_dir is unset (functions fall back to CWD-relative)."""
function _resolve_cache_base(config::DockingConfig)::String
    isempty(config.cache_dir) && return ""
    # cache_dir points to .bayesinteractomics_cache/docking — parent is the base
    return dirname(config.cache_dir)
end

"""Resolve the cache directory path."""
function _resolve_cache_dir(config::DockingConfig)::String
    dir = if isempty(config.cache_dir)
        joinpath(".bayesinteractomics_cache", "docking")
    else
        config.cache_dir
    end
    mkpath(dir)
    return dir
end

"""Load the set of cached pair keys from the cache directory."""
function _load_cache_index(cache_dir::String)::Set{String}
    keys = Set{String}()
    !isdir(cache_dir) && return keys

    for f in readdir(cache_dir)
        if endswith(f, ".jld2")
            push!(keys, replace(f, ".jld2" => ""))
        end
    end
    return keys
end

"""Load a cached DockingPairResult, or return nothing if not cached."""
function _load_cached_pair(cache_dir::String, pair_key::String)::Union{DockingPairResult, Nothing}
    cache_path = joinpath(cache_dir, "$(pair_key).jld2")
    !isfile(cache_path) && return nothing

    try
        data = JLD2.load(cache_path)
        haskey(data, "pair") || return nothing

        d = data["pair"]
        # Reconstruct DockingPairResult from saved Dict
        return DockingPairResult(
            String(d["protein_a"]), String(d["protein_b"]),
            String(d["uniprot_a"]), String(d["uniprot_b"]),
            Float64(d["iptm_best"]),
            Float64.(d["iptm_all"]),
            Float64(d["iptm_std"]),
            Float64(d["ranking_score"]),
            Float64(d["fraction_disordered"]),
            Float64(d["chain_pair_iptm"]),
            Float64(d["chain_pair_pae_min"]),
            Float64(d["pdockq"]),
            Float64(d["mean_plddt_a"]),
            Float64(d["mean_plddt_b"]),
            Int(d["n_interface_contacts"]),
            Float64(d["bf_dock"]),
            String(d["calibration_tier"]),
            Symbol(d["status"]),
            Int(d["token_count"]),
            Float64(get(d, "c2qscore", NaN)),
            Float64(get(d, "ipae", NaN)),
            Float64(get(d, "iplddt_interface", NaN)),
        )
    catch e
        @warn "Failed to load cached pair" pair_key=pair_key exception=e
        return nothing
    end
end

"""Save a DockingPairResult to the cache."""
function _save_cached_pair(cache_dir::String, pair_key::String, pair::DockingPairResult)
    mkpath(cache_dir)
    cache_path = joinpath(cache_dir, "$(pair_key).jld2")

    d = Dict{String, Any}(
        "protein_a"           => pair.protein_a,
        "protein_b"           => pair.protein_b,
        "uniprot_a"           => pair.uniprot_a,
        "uniprot_b"           => pair.uniprot_b,
        "iptm_best"           => pair.iptm_best,
        "iptm_all"            => pair.iptm_all,
        "iptm_std"            => pair.iptm_std,
        "ranking_score"       => pair.ranking_score,
        "fraction_disordered" => pair.fraction_disordered,
        "chain_pair_iptm"     => pair.chain_pair_iptm,
        "chain_pair_pae_min"  => pair.chain_pair_pae_min,
        "pdockq"              => pair.pdockq,
        "mean_plddt_a"        => pair.mean_plddt_a,
        "mean_plddt_b"        => pair.mean_plddt_b,
        "n_interface_contacts"=> pair.n_interface_contacts,
        "bf_dock"             => pair.bf_dock,
        "calibration_tier"    => pair.calibration_tier,
        "status"              => string(pair.status),
        "token_count"         => pair.token_count,
        "c2qscore"            => pair.c2qscore,
        "ipae"                => pair.ipae,
        "iplddt_interface"    => pair.iplddt_interface,
        "timestamp"           => format(now(), "yyyy-mm-ddTHH:MM:SS"),
    )

    JLD2.jldsave(cache_path; pair=d)
end
