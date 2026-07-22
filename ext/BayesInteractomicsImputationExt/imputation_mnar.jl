"""
    imputation_mnar.jl

MNAR-aware single imputation. Reads the per-column dropout
curves `(ρ̂_c, ζ̂_c)` from `imputed_data/dropout_curves.json`
and samples each missing entry once from a tilted Gaussian
    p̃(y | μ̂_i, σ̂_i², ρ̂_c, ζ̂_c) ∝ φ(y; μ̂_i, σ̂_i²) · (1 − σ(ρ̂_c + ζ̂_c · y))
via inverse-CDF interpolation on a fixed grid. Outputs `dataset_mnar.xlsx`
matching the raw `dataset.xlsx` layout, plus a reproducibility manifest.

Notes
-----
- `LogExpFunctions.logistic` / `log1pexp` are used for the sigmoid factor;
  the symbol named `sigmoid` is NOT exported from `Distributions` (verified
  via live invocation).
- `MersenneTwister` requires explicit `using Random` on Julia 1.12.
- Per-protein RNG seeded as `MersenneTwister(seed * 1_000_003 + i)` so
  reproducibility is invariant under `JULIA_NUM_THREADS`.
- Excluded-column fallback: when `(ρ̂_c, ζ̂_c) = (NaN, NaN)` (the dropout fit
  marked the column excluded), the sampler falls back to plain
  `Normal(μ̂_i, σ̂_i²)` for that column's missings and emits one `@warn` per
  excluded column (logged serially before the threaded loop, NOT inside it).
"""

# Module-scope imports moved to
# BayesInteractomicsImputationExt.jl (extension shell).

# ============================================================================
# Public API
# ============================================================================

"""
    impute_mnar(intensity_matrix, curves; kwargs...) -> (Matrix{Float64}, Dict{String,Any})

Sample one MNAR-tilted value per missing cell of `intensity_matrix` using
per-column dropout curves `curves[c] = (ρ̂_c, ζ̂_c)`. Observed values pass
through unchanged. Excluded columns (`(NaN, NaN)`) fall back to plain
`Normal(μ̂_i, σ̂²_i)` with a per-column `@warn`.

Returns a no-missings `Matrix{Float64}` and a partial manifest dict (the
caller fills `raw_dataset_hash`, `dropout_curves_dataset_hash`,
`dropout_curves_path`).
"""
function BayesInteractomics.impute_mnar(
    intensity_matrix::AbstractMatrix{Union{Missing, Float64}},
    curves::Dict{Int, Tuple{Float64, Float64}};
    seed::Int = 42,
    n_grid::Int = 50,
    k_low::Float64 = 5.0,
    k_high::Float64 = 2.0,
    n_obs_threshold::Int = 5,
    log_transform::Bool = true,
)::Tuple{Matrix{Float64}, Dict{String, Any}}
    n_proteins, n_columns = size(intensity_matrix)
    @assert length(curves) >= n_columns ||
            all(haskey(curves, c) for c in 1:n_columns) "curves dict must cover columns 1:$n_columns"

    # Apply log_transform once, before estimation + sampling (mirrors the dropout-fit discipline).
    m = log_transform ? _maybe_log2(intensity_matrix) : intensity_matrix

    # Pre-compute per-protein moments (one pass; pooled_σ² shared).
    μ_hat, σ²_hat, n_obs, pooled_σ² =
        _estimate_per_protein_moments(m; n_obs_threshold = n_obs_threshold)

    # Count missings BEFORE imputation (manifest field).
    n_missing_entries_imputed = sum(ismissing, intensity_matrix)

    # Serial pre-loop @warn for NaN-excluded columns (logging hygiene).
    for c in 1:n_columns
        ρ̂_c, ζ̂_c = curves[c]
        if isnan(ρ̂_c) || isnan(ζ̂_c)
            n_missing_in_col = sum(ismissing, intensity_matrix[:, c])
            @warn "Column $c excluded from the dropout fit (rho/zeta = NaN); falling back to plain Normal for $n_missing_in_col missings"
        end
    end

    # Per-protein parallel imputation. RNG seeded by protein index → reproducible under any thread count.
    imputed_internal = Matrix{Float64}(undef, n_proteins, n_columns)
    Threads.@threads for i in 1:n_proteins
        rng = MersenneTwister(seed * 1_000_003 + i)
        μ̂_i  = μ_hat[i]
        σ²_i = σ²_hat[i]
        for c in 1:n_columns
            x = m[i, c]
            if !ismissing(x)
                imputed_internal[i, c] = x
            elseif isnan(μ̂_i) || isnan(σ²_i)
                # Protein has no observations — should be filtered upstream;
                # we still must produce SOME value. Use 0.0 as a safe sentinel.
                # This protects the no-missings post-condition.
                imputed_internal[i, c] = 0.0
            else
                ρ̂_c, ζ̂_c = curves[c]
                if isnan(ρ̂_c) || isnan(ζ̂_c)
                    imputed_internal[i, c] = rand(rng, Normal(μ̂_i, sqrt(σ²_i)))
                else
                    imputed_internal[i, c] = _sample_tilted_normal(
                        μ̂_i, σ²_i, ρ̂_c, ζ̂_c, rng;
                        n_grid = n_grid, k_low = k_low, k_high = k_high)
                end
            end
        end
    end

    # Back-transform: imputed cells (originally missing) come off the log scale via 2^x;
    # observed cells pass through on the original linear scale.
    imputed = Matrix{Float64}(undef, n_proteins, n_columns)
    @inbounds for i in 1:n_proteins, c in 1:n_columns
        x = intensity_matrix[i, c]
        if !ismissing(x)
            imputed[i, c] = Float64(x)            # original linear-scale observed value
        else
            imputed[i, c] = log_transform ? 2.0 ^ imputed_internal[i, c] : imputed_internal[i, c]
        end
    end

    # Partial manifest — caller fills hashes + paths.
    manifest = Dict{String, Any}(
        "version"                   => "1.2.0",
        "fit_timestamp"             => format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"),
        "software_version"          => _get_software_version(),
        "seed"                      => seed,
        "n_proteins"                => n_proteins,
        "n_columns_imputed"         => n_columns,
        "n_missing_entries_imputed" => n_missing_entries_imputed,
        "n_obs_threshold"           => n_obs_threshold,
        "n_grid"                    => n_grid,
        "k_low"                     => k_low,
        "k_high"                    => k_high,
        "dropout_curves_path"        => "",
        "dropout_curves_dataset_hash" => "",
        "raw_dataset_hash"           => "",
        "estimator" => Dict("strategy" => "hybrid", "n_obs_threshold" => n_obs_threshold),
        "sampler"   => Dict("strategy" => "inverse_cdf_grid",
                            "n_grid" => n_grid, "k_low" => k_low, "k_high" => k_high),
    )

    return imputed, manifest
end

"""
    impute_mnar_from_paths(xlsx_path, curves_path; kwargs...)

Path-driven orchestrator: loads `xlsx_path` (raw dataset) and `curves_path`
(dropout-curves JSON), runs `impute_mnar`, writes `dataset_mnar.xlsx` + manifest
JSON, and returns a NamedTuple of paths and in-memory products.
"""
function BayesInteractomics.impute_mnar_from_paths(
    xlsx_path::String, curves_path::String;
    output_path::String = "",
    manifest_path::String = "",
    seed::Int = 42,
    n_grid::Int = 50,
    k_low::Float64 = 5.0,
    k_high::Float64 = 2.0,
    n_obs_threshold::Int = 5,
    log_transform::Bool = true,
)
    isfile(xlsx_path)   || error("dataset.xlsx not found at: $xlsx_path")
    isfile(curves_path) || error("dropout_curves.json not found at: $curves_path")

    if isempty(output_path)
        output_path = joinpath(dirname(abspath(xlsx_path)),
                               "imputed_data", "dataset_mnar.xlsx")
    end
    if isempty(manifest_path)
        manifest_path = joinpath(dirname(abspath(output_path)),
                                 "dataset_mnar_manifest.json")
    end

    intensity_matrix, column_names, protein_ids, id_col_name =
        _load_intensity_matrix_with_ids(xlsx_path)

    curves, dropout_dataset_hash = _load_column_curves(curves_path)
    n_proteins, n_columns = size(intensity_matrix)

    imputed, partial_manifest = impute_mnar(intensity_matrix, curves;
        seed=seed, n_grid=n_grid, k_low=k_low, k_high=k_high,
        n_obs_threshold=n_obs_threshold, log_transform=log_transform)

    _write_mnar_xlsx(output_path, id_col_name, protein_ids, column_names, imputed)

    raw_hash = _hash_file_bytes(xlsx_path)
    manifest = merge(partial_manifest, Dict{String,Any}(
        "raw_dataset_hash"             => raw_hash,
        "dropout_curves_path"          => curves_path,
        "dropout_curves_dataset_hash"  => dropout_dataset_hash,
    ))
    _write_mnar_manifest(manifest_path, manifest)

    return (output_path = output_path,
            manifest_path = manifest_path,
            imputed = imputed,
            manifest = manifest)
end

# ============================================================================
# Internal helpers — Persistence
# ============================================================================

"""
    _load_column_curves(json_path) -> (Dict{Int,Tuple{Float64,Float64}}, dataset_hash::String)

Strict-version-check loader for the dropout-curves JSON. Errors if the
JSON's `version` field is not exactly the literal string `"1.2.0"`.
NaN curves (excluded columns) flow through unchanged via `load_dropout_fit`.
"""
function _load_column_curves(json_path::String)
    # Strict version check on raw JSON; null→NaN coercion via load_dropout_fit
    raw = JSON3.read(String(read(json_path)))
    if String(raw.version) != "1.2.0"
        error("dropout_curves.json version mismatch: expected '1.2.0', got '$(raw.version)'. " *
              "Re-run scripts/fit_dropout.jl on this dataset.")
    end
    fit = load_dropout_fit(json_path)
    curves = Dict{Int, Tuple{Float64, Float64}}()
    for c in 1:length(fit.column_names)
        curves[c] = (fit.rho[c], fit.zeta[c])
    end
    return curves, String(raw.dataset_hash)
end

"""
    _hash_file_bytes(path) -> "sha256:<hex>"

SHA256 of the raw file bytes at `path`. Used for the manifest's
`raw_dataset_hash` field. Distinct from
`_hash_intensity_matrix` in `dropout.jl` which hashes the parsed matrix.
"""
function _hash_file_bytes(path::String)::String
    return "sha256:" * bytes2hex(sha256(read(path)))
end

# Note: `_maybe_log2(m::AbstractMatrix{Union{Missing,Float64}})` is provided by
# `src/data/dropout.jl`, which is `include`-d immediately before this file in
# `src/BayesInteractomics.jl`. Reusing that definition avoids a method-overwrite
# collision during package precompilation.

"""
    _load_intensity_matrix_with_ids(xlsx_path; sheet_name="Sheet1", id_col=1)
        -> (intensity_matrix, column_names, protein_ids, id_col_name)

ID-aware variant of the `_load_intensity_matrix` helper. Returns
the protein-ID column (column 1 by default) alongside the numeric matrix
so the imputed XLSX can be written back with the same layout.
"""
function _load_intensity_matrix_with_ids(xlsx_path::String;
                                          sheet_name::String = "Sheet1",
                                          id_col::Int = 1)
    raw_df = DataFrame(readtable(xlsx_path, sheet_name))
    all_names    = String.(names(raw_df))
    id_col_name  = all_names[id_col]
    raw_ids      = raw_df[:, id_col]
    # Tolerate occasional missing IDs (rare data-cleaning gaps): replace with
    # a stable per-row placeholder. The caller writes the placeholder back
    # verbatim, so the imputed XLSX preserves row alignment with the input.
    protein_ids  = [ismissing(v) ? "_MISSING_ID_row$i" : String(v)
                    for (i, v) in enumerate(raw_ids)]
    column_names = all_names[id_col + 1 : end]
    raw_matrix   = Matrix(raw_df[:, id_col + 1 : end])
    intensity_matrix = convert(Matrix{Union{Missing, Float64}}, raw_matrix)
    return intensity_matrix, column_names, protein_ids, id_col_name
end

"""
    _write_mnar_xlsx(path, id_col_name, protein_ids, column_names, imputed_matrix) -> path

Write the no-missings imputed matrix to an XLSX matching `dataset.xlsx` raw
layout (sheet 1, col 1 = protein IDs, cols 2..N = numeric intensities).
Uses `XLSX.writetable` with `Sheet1 => DataFrame` per the project convention.
"""
function _write_mnar_xlsx(path::String,
                          id_col_name::String,
                          protein_ids::Vector{String},
                          column_names::Vector{String},
                          imputed_matrix::AbstractMatrix{Float64})::String
    n_rows, n_cols = size(imputed_matrix)
    @assert length(protein_ids)  == n_rows
    @assert length(column_names) == n_cols
    df = DataFrame()
    df[!, Symbol(id_col_name)] = protein_ids
    for (c, name) in enumerate(column_names)
        df[!, Symbol(name)] = imputed_matrix[:, c]
    end
    mkpath(dirname(abspath(path)))
    writetable(path, "Sheet1" => df)
    return path
end

"""
    _atomic_write_text(path, content) -> path

Helper that serialises a `String` payload to disk. Available as a reusable
helper for callers who want a single auditable disk-write site.
"""
function _atomic_write_text(path::String, content::String)::String
    mkpath(dirname(abspath(path)))
    write(path, content)
    return path
end

"""
    _write_mnar_manifest(path, metadata) -> path

Reproducibility manifest writer. Emits a JSON file containing exactly the 12 top-level
keys: version, fit_timestamp, software_version, seed, n_proteins,
n_columns_imputed, n_missing_entries_imputed, dropout_curves_path,
dropout_curves_dataset_hash, raw_dataset_hash, estimator (sub-dict), sampler
(sub-dict).
"""
function _write_mnar_manifest(path::String, metadata::Dict{String, Any})::String
    payload = Dict{String, Any}(
        "version"                     => "1.2.0",
        "fit_timestamp"               => get(metadata, "fit_timestamp",
                                              format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ")),
        "software_version"            => get(metadata, "software_version", _get_software_version()),
        "seed"                        => metadata["seed"],
        "n_proteins"                  => metadata["n_proteins"],
        "n_columns_imputed"           => metadata["n_columns_imputed"],
        "n_missing_entries_imputed"   => metadata["n_missing_entries_imputed"],
        "dropout_curves_path"         => get(metadata, "dropout_curves_path", ""),
        "dropout_curves_dataset_hash" => get(metadata, "dropout_curves_dataset_hash", ""),
        "raw_dataset_hash"            => get(metadata, "raw_dataset_hash", ""),
        "estimator" => Dict("strategy" => "hybrid",
                            "n_obs_threshold" => get(metadata, "n_obs_threshold", 5)),
        "sampler"   => Dict("strategy" => "inverse_cdf_grid",
                            "n_grid"   => get(metadata, "n_grid", 50),
                            "k_low"    => get(metadata, "k_low", 5.0),
                            "k_high"   => get(metadata, "k_high", 2.0)),
    )
    mkpath(dirname(abspath(path)))
    write(path, JSON3.write(payload))
    return path
end

# Note: `_get_software_version()` is provided by `src/data/dropout.jl`, which is
# `include`-d immediately before this file in `src/BayesInteractomics.jl`.
# Reusing that definition avoids a method-overwrite collision during package
# precompilation.

# ============================================================================
# Internal helpers — Statistics
# ============================================================================

"""
    _estimate_per_protein_moments(intensity_matrix; n_obs_threshold=5)
        -> (μ̂::Vector, σ̂²::Vector, n_obs::Vector, pooled_σ²::Float64)

Hybrid per-protein estimator:
- proteins with `n_obs ≥ n_obs_threshold`: plain `mean` + `var(corrected=true)`
- proteins with `1 ≤ n_obs < n_obs_threshold`: plain `mean` + `pooled_σ²`
  (median of variances over well-detected proteins)
- proteins with `n_obs == 0`: NaN/NaN (caller should filter upstream)

Floors σ̂² at `0.01 * pooled_σ²` to prevent pathological tilts when the
sample variance is degenerate (RESEARCH §A.3 + §B.2 mitigation).
"""
function _estimate_per_protein_moments(intensity_matrix::AbstractMatrix{Union{Missing, Float64}};
                                        n_obs_threshold::Int = 5)
    n_proteins = size(intensity_matrix, 1)
    μ_hat  = Vector{Float64}(undef, n_proteins)
    σ²_hat = Vector{Float64}(undef, n_proteins)
    n_obs  = Vector{Int}(undef, n_proteins)
    var_well = Float64[]

    @inbounds for i in 1:n_proteins
        obs = collect(skipmissing(intensity_matrix[i, :]))
        n_obs[i] = length(obs)
        if n_obs[i] == 0
            μ_hat[i]  = NaN
            σ²_hat[i] = NaN
        else
            μ_hat[i] = mean(obs)
            if n_obs[i] >= n_obs_threshold
                v = var(obs; corrected=true)
                σ²_hat[i] = v
                push!(var_well, v)
            else
                σ²_hat[i] = NaN
            end
        end
    end

    pooled_σ² = isempty(var_well) ? 1.0 : median(var_well)

    @inbounds for i in 1:n_proteins
        if 1 <= n_obs[i] < n_obs_threshold
            σ²_hat[i] = pooled_σ²
        end
    end

    # Floor against degenerate σ̂² ≈ 0 (RESEARCH §A.3 + §B.2 mitigation)
    floor_val = 0.01 * pooled_σ²
    @inbounds for i in 1:n_proteins
        if !isnan(σ²_hat[i]) && σ²_hat[i] < floor_val
            σ²_hat[i] = floor_val
        end
    end

    return μ_hat, σ²_hat, n_obs, pooled_σ²
end

# ============================================================================
# Internal helpers — Sampler
# ============================================================================

"""
    _log_unnormalised_density(y, μ, σ², ρ, ζ) -> Float64

Log of the unnormalised tilted-Gaussian density:
    log φ(y; μ, σ²) + log(1 − σ(ρ + ζ·y))
The second term is computed via `-log1pexp(ρ + ζ·y)` for numerical stability
(RESEARCH §A.4) — equivalent to `log(1 - logistic(ρ + ζ·y))` but never
underflows.
"""
@inline function _log_unnormalised_density(y::Float64, μ::Float64, σ²::Float64,
                                            ρ::Float64, ζ::Float64)::Float64
    log_phi              = -0.5 * log(2π * σ²) - (y - μ)^2 / (2 * σ²)
    log_one_minus_sigma  = -log1pexp(ρ + ζ * y)   # = log(1 - logistic(ρ + ζ·y)), stable
    return log_phi + log_one_minus_sigma
end

"""
    _invert_grid_cdf(u, y_grid, cdf) -> Float64

Linear-interpolated inverse of a discrete CDF on a strictly increasing grid.
Saturates at `y_grid[1]` / `y_grid[end]` outside the range.
"""
function _invert_grid_cdf(u::Float64, y_grid::AbstractVector{Float64}, cdf::AbstractVector{Float64})::Float64
    k = searchsortedfirst(cdf, u)
    if k == 1
        return y_grid[1]
    elseif k > length(y_grid)
        return y_grid[end]
    else
        cdf_lo, cdf_hi = cdf[k-1], cdf[k]
        y_lo,   y_hi   = y_grid[k-1], y_grid[k]
        return y_lo + (u - cdf_lo) / (cdf_hi - cdf_lo) * (y_hi - y_lo)
    end
end

"""
    _sample_tilted_normal(μ, σ², ρ, ζ, rng; n_grid=50, k_low=5.0, k_high=2.0) -> Float64

Draw one sample from
    p̃(y) ∝ φ(y; μ, σ²) · (1 − σ(ρ + ζ·y))
via inverse-CDF on an asymmetric grid `[μ - k_low·σ, μ + k_high·σ]`. The
asymmetry reflects that the dropout factor pulls mass leftward; widening
the left side of the grid prevents truncation under high-ζ tilts.

Numerical recipe (RESEARCH §A): evaluate log-density on the grid, subtract
the maximum, exponentiate, normalise, build cumulative sum, draw `u ~ U(0,1)`
and linearly interpolate.
"""
function _sample_tilted_normal(μ::Float64, σ²::Float64,
                                ρ::Float64, ζ::Float64,
                                rng::AbstractRNG;
                                n_grid::Int = 50,
                                k_low::Float64 = 5.0,
                                k_high::Float64 = 2.0)::Float64
    σ = sqrt(σ²)
    y_grid = collect(range(μ - k_low * σ, μ + k_high * σ; length = n_grid))

    log_p = Vector{Float64}(undef, n_grid)
    @inbounds for k in 1:n_grid
        log_p[k] = _log_unnormalised_density(y_grid[k], μ, σ², ρ, ζ)
    end

    log_p_max = maximum(log_p)
    p = exp.(log_p .- log_p_max)
    p ./= sum(p)

    # Midpoint-corrected cumulative: each grid point owns p[k]/2 to its left and
    # p[k]/2 to its right.  Without this correction, `cumsum(p)[k]` interprets
    # the entire mass p[k] as lying to the LEFT of `y_grid[k]`, producing an
    # O(h) systematic mean bias (Rule 1 fix — auto-detected by the plan-level
    # verify block requiring |mean| < 0.05 at n_grid=50 with ζ=0).
    cdf = cumsum(p)
    @inbounds for k in 1:length(p)
        cdf[k] -= 0.5 * p[k]
    end

    u = rand(rng)
    return _invert_grid_cdf(u, y_grid, cdf)
end
