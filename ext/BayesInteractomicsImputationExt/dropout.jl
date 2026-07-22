"""
    dropout.jl

Per-column logistic dropout-curve fitting for MNAR-aware imputation.

Fits `o_{i,c} ~ Bernoulli(σ(ρ_c + ζ_c · ȳ_i))` per column `c` of the raw
intensity matrix, where `o_{i,c}` is the detection indicator and `ȳ_i` is the
mean observed log-intensity for protein `i` across all columns where it was
detected. Output is a `DropoutFit` struct that can be persisted to JSON for
consumption by the downstream R imputation script.

Notes
-----
- `ConvergenceException` is imported explicitly from `GLM` — `using GLM`
  alone does not bring it into scope (the symbol is owned by StatsBase and
  re-exported into GLM's namespace).
- The per-column GLM loop is intentionally serial — ~88 fits × ~1 ms each
  is dominated by parallelism overhead; `@warn` ordering
  also matters for the SANITY.md report.
"""

# Module-scope imports moved to
# BayesInteractomicsImputationExt.jl (extension shell). The `struct DropoutFit`
# block was promoted to `src/data/imputation_stubs.jl` as a concrete struct so
# the symbol resolves at compile time even when this extension is not loaded.
# Producer + persistence helpers (`fit_dropout_curves`, `save_dropout_fit`,
# `load_dropout_fit`) stay here.

function Base.show(io::IO, f::DropoutFit)
    n = length(f.column_names)
    n_excl = Base.count(isnan, f.rho)
    z_fit = filter(!isnan, f.zeta)
    n_well = Base.count(z -> 0.5 ≤ z ≤ 3.0, z_fit)
    n_fit = n - n_excl
    println(io, "DropoutFit ($(f.fit_timestamp))")
    println(io, "────────────────────────────────────")
    println(io, "  columns:        $n  (fit: $n_fit, excluded: $n_excl)")
    println(io, "  proteins:       $(f.n_proteins)")
    println(io, "  ζ̂ ∈ [0.5, 3]:   $n_well / $n_fit")
    println(io, "  software:       $(f.software_version)")
    if length(f.dataset_hash) >= 24
        println(io, "  dataset_hash:   $(first(f.dataset_hash, 24))…")
    else
        println(io, "  dataset_hash:   $(f.dataset_hash)")
    end
end

"""
    fit_dropout_curves(intensity_matrix; column_names, log_transform=true,
                       min_detections_per_column=5, output_path=nothing,
                       diagnostics_dir=nothing, protocol_assignment=nothing) -> DropoutFit

Fit per-column logistic dropout curves `p_detect = σ(ρ_c + ζ_c · ȳ_i)`.

# Arguments
- `intensity_matrix`: rows = proteins, columns = MS runs. Missing entries indicate
  non-detection. Pre-log if `log_transform=false`; raw if `log_transform=true`.
- `column_names`: length = `size(intensity_matrix, 2)`. Defaults to `"c1", "c2", …`.
- `log_transform`: if `true`, take `log2` of observed values before computing ȳ_i.
  Set to `false` if the matrix is already on the log scale.
- `min_detections_per_column`: columns with fewer detections are excluded
  (rho/zeta = NaN). Default 5.
- `output_path`: if non-`nothing`, calls `save_dropout_fit(fit, output_path)` as a
  side effect.
- `diagnostics_dir`: if non-`nothing`, generates the three diagnostic plots and
  the SANITY.md report under this directory.
- `protocol_assignment`: optional length-n_columns vector mapping each column to
  a protocol index (used for the all-sigmoids overlay grid; default = single
  protocol).

# Returns
- `DropoutFit` — see struct docstring.

# Algorithm
1. `log_transform` if requested.
2. Compute `ȳ_i` ONCE globally as `mean(skipmissing(intensity_matrix[i, :]))`
   per protein. Exclude proteins with zero global detections.
3. Build per-column detection mask `o_{i,c} = .!ismissing.(intensity_matrix[:, c])`.
4. For each column `c`:
   - Skip if `n_detections_c < min_detections_per_column` → record NaN.
   - Else: fit `glm(@formula(detected ~ mean_intensity), df, Bernoulli(), LogitLink())`.
   - Wrap in try/catch: on `ConvergenceException` (or any other GLM error),
     record NaN + log warning, continue.
5. Build the `DropoutFit` with timestamp, software version, dataset_hash.
6. If `output_path` set: call `save_dropout_fit(fit, output_path)`.
7. If `diagnostics_dir` set: call `_emit_diagnostics(fit, ...)`.

# See also
- `save_dropout_fit`, `load_dropout_fit`, `DropoutFit`
"""
function BayesInteractomics.fit_dropout_curves(
    intensity_matrix::AbstractMatrix{Union{Missing, Float64}};
    column_names::Vector{String} = String["c$c" for c in 1:size(intensity_matrix, 2)],
    log_transform::Bool = true,
    min_detections_per_column::Int = 5,
    output_path::Union{Nothing, String} = nothing,
    diagnostics_dir::Union{Nothing, String} = nothing,
    protocol_assignment::Union{Nothing, Vector{Int}} = nothing,
)::DropoutFit
    n_proteins_raw, n_cols = size(intensity_matrix)
    @assert length(column_names) == n_cols "column_names length must equal number of columns ($(length(column_names)) != $n_cols)"

    # 1. Apply log_transform if requested
    m = log_transform ? _maybe_log2(intensity_matrix) : intensity_matrix

    # 2. Compute ȳ_i once globally; filter proteins with zero detections
    ybar_all, keep = _compute_ybar(m)
    m_kept = m[keep, :]
    ybar = ybar_all[keep]
    n_proteins = sum(keep)
    if n_proteins == 0
        throw(ArgumentError("No proteins with at least one detection — cannot fit"))
    end

    # 3. Per-column detection mask
    detected_mat = .!ismissing.(m_kept)

    # 4. Per-column GLM fit (serial — see RESEARCH §E.4 anti-pattern note)
    rho = fill(NaN, n_cols)
    zeta = fill(NaN, n_cols)
    n_det = zeros(Int, n_cols)
    for c in 1:n_cols
        det_c = collect(detected_mat[:, c])
        n_det[c] = sum(det_c)
        if n_det[c] < min_detections_per_column
            @warn "Column $c ($(column_names[c])) excluded: only $(n_det[c]) detections (< $min_detections_per_column)"
            continue
        end
        try
            df_col = DataFrame(detected = Int.(det_c), mean_intensity = ybar)
            fit_obj = glm(@formula(detected ~ mean_intensity), df_col, Bernoulli(), LogitLink())
            cf = coef(fit_obj)
            rho[c] = cf[1]
            zeta[c] = cf[2]
        catch e
            if e isa ConvergenceException
                @warn "Column $c ($(column_names[c])): GLM did not converge (separation likely)" exception=e
            else
                @warn "Column $c ($(column_names[c])): unexpected GLM error" exception=e
            end
            # rho[c] / zeta[c] stay NaN
        end
    end

    # 5. Construct DropoutFit
    fit = DropoutFit(
        rho,
        zeta,
        column_names,
        n_proteins,
        n_det,
        format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"),
        _get_software_version(),
        _hash_intensity_matrix(m),
    )

    # 6. Optional persistence
    if !isnothing(output_path)
        save_dropout_fit(fit, output_path)
    end

    # 7. Optional diagnostics (full implementation lives below, alongside plotting helpers)
    if !isnothing(diagnostics_dir)
        _emit_diagnostics(fit, m_kept, ybar, detected_mat;
                          diagnostics_dir = diagnostics_dir,
                          protocol_assignment = protocol_assignment,
                          min_detections_per_column = min_detections_per_column)
    end

    return fit
end

# ============================================================================
# Internal helpers
# ============================================================================

"""
    _compute_ybar(intensity_matrix) -> (ybar, keep_mask)

Per-protein mean of observed (non-missing) log-intensities. Returns a length-n_proteins
vector and a `BitVector` marking proteins with ≥1 detection (rows with zero detections
are excluded — their `ȳ_i` is undefined per Pitfall 5).
"""
function _compute_ybar(intensity_matrix::AbstractMatrix{Union{Missing, Float64}})
    n_proteins, _ = size(intensity_matrix)
    ybar = Vector{Float64}(undef, n_proteins)
    keep = trues(n_proteins)
    for i in 1:n_proteins
        row = intensity_matrix[i, :]
        obs = collect(skipmissing(row))
        if isempty(obs)
            ybar[i] = NaN
            keep[i] = false
        else
            ybar[i] = mean(obs)
        end
    end
    return ybar, keep
end

"""
    _hash_intensity_matrix(m) -> "sha256:<hex>"

Deterministic SHA256 of the matrix bytes (column-major Float64 dump,
`missing → NaN`). Used as `DropoutFit.dataset_hash` for cross-language
reproducibility (the cross-language JSON contract requires a stable string
that survives Julia version bumps).
"""
function _hash_intensity_matrix(m::AbstractMatrix{Union{Missing, Float64}})::String
    io = IOBuffer()
    n_rows, n_cols = size(m)
    write(io, UInt32(n_rows))
    write(io, UInt32(n_cols))
    for j in 1:n_cols, i in 1:n_rows
        v = ismissing(m[i, j]) ? NaN : m[i, j]
        write(io, v)  # 8-byte Float64 little-endian on x86
    end
    return "sha256:" * bytes2hex(sha256(take!(io)))
end

"""
    _maybe_log2(m) -> Matrix{Union{Missing, Float64}}

Element-wise log2 transform that preserves `missing` entries.
"""
function _maybe_log2(m::AbstractMatrix{Union{Missing, Float64}})
    out = similar(m)
    for i in eachindex(m)
        out[i] = ismissing(m[i]) ? missing : log2(m[i])
    end
    return out
end

"""
    _get_software_version() -> String

Read the BayesInteractomics version from the active project's `Project.toml`
at runtime. Falls back to `"unknown"` if the lookup fails (e.g., when this
file is `include`-d outside the package context).
"""
function _get_software_version()::String
    try
        project = Base.active_project()
        if !isnothing(project) && isfile(project)
            candidates = String[project, joinpath(dirname(project), "Project.toml")]
            for path in candidates
                isfile(path) || continue
                for line in eachline(path)
                    matched = match(r"^version\s*=\s*\"([^\"]+)\"", line)
                    if matched !== nothing
                        return String(matched.captures[1])
                    end
                end
            end
        end
    catch
        # fall through to "unknown"
    end
    return "unknown"
end

# ============================================================================
# JSON persistence (schema — cross-language contract with the R imputation script)
# ============================================================================

"""
    save_dropout_fit(fit, path) -> path

Persist a `DropoutFit` to JSON at `path` using the locked schema.
The schema is a cross-language contract — the R imputation script reads the same
file via `jsonlite::fromJSON()`. Excluded columns are written with `null` for
`rho`/`zeta` and `excluded: true`.
"""
function BayesInteractomics.save_dropout_fit(fit::DropoutFit, path::String)::String
    n = length(fit.column_names)
    payload = Dict(
        "version"          => "1.2.0",  # locked schema version; reflects v1.2.0 milestone
        "fit_timestamp"    => fit.fit_timestamp,
        "dataset_hash"     => fit.dataset_hash,
        "n_proteins"       => fit.n_proteins,
        "software_version" => fit.software_version,
        "columns" => [
            Dict(
                "index"        => c,
                "name"         => fit.column_names[c],
                "rho"          => isnan(fit.rho[c]) ? nothing : fit.rho[c],
                "zeta"         => isnan(fit.zeta[c]) ? nothing : fit.zeta[c],
                "n_detections" => fit.n_detections_per_column[c],
                "excluded"     => isnan(fit.rho[c]),
            )
            for c in 1:n
        ],
    )
    mkpath(dirname(abspath(path)))
    write(path, JSON3.write(payload))
    return path
end

"""
    load_dropout_fit(path) -> DropoutFit

Load a `DropoutFit` from JSON written by `save_dropout_fit`. Excluded columns
(`rho: null` or `zeta: null` in the JSON) are reconstructed with `NaN`.
"""
function BayesInteractomics.load_dropout_fit(path::String)::DropoutFit
    data = JSON3.read(String(read(path)))
    cols = data.columns
    n = length(cols)
    rho = Vector{Float64}(undef, n)
    zeta = Vector{Float64}(undef, n)
    column_names = Vector{String}(undef, n)
    n_det = Vector{Int}(undef, n)
    for c in 1:n
        col = cols[c]
        rho[c]          = isnothing(col.rho)  ? NaN : Float64(col.rho)
        zeta[c]         = isnothing(col.zeta) ? NaN : Float64(col.zeta)
        column_names[c] = String(col.name)
        n_det[c]        = Int(col.n_detections)
    end
    # software_version is a top-level field added by save; tolerate older files via fallback.
    sv = haskey(data, :software_version) ? String(data.software_version) :
         (haskey(data, :version) ? String(data.version) : "unknown")
    return DropoutFit(
        rho, zeta, column_names, Int(data.n_proteins),
        n_det, String(data.fit_timestamp),
        sv, String(data.dataset_hash),
    )
end

# ============================================================================
# Diagnostic plot helpers
# ============================================================================

"""
    _plot_column_fit(ybar, detected, ρ, ζ, col_name, file)

Per-column scatter (decile-binned detection rate) + fitted sigmoid overlay.
Output: PNG at `file`, size (800, 600). Plot 1 of 3.
"""
function _plot_column_fit(ybar::Vector{Float64}, detected::AbstractVector{Bool},
                          ρ::Float64, ζ::Float64, col_name::String, file::String)
    # Decile binning of ybar
    edges = quantile(ybar, range(0, 1; length = 11))
    centers = Float64[]
    rates = Float64[]
    for k in 1:10
        lo, hi = edges[k], edges[k+1]
        idx = if k == 10
            (ybar .>= lo) .& (ybar .<= hi)
        else
            (ybar .>= lo) .& (ybar .< hi)
        end
        n = sum(idx)
        n == 0 && continue
        push!(centers, mean(ybar[idx]))
        push!(rates, sum(detected[idx]) / n)
    end

    # Smooth fitted sigmoid
    x_grid = range(minimum(ybar), maximum(ybar); length = 200)
    y_grid = 1.0 ./ (1.0 .+ exp.(.-(ρ .+ ζ .* x_grid)))

    plt = StatsPlots.plot(centers, rates;
        seriestype = :scatter, markersize = 4, label = "binned detection rate",
        xlabel = "ȳᵢ (mean log-intensity)", ylabel = "P(detected)",
        ylims = (-0.02, 1.02), title = col_name, size = (800, 600),
    )
    StatsPlots.plot!(plt, x_grid, y_grid;
        label = "σ(ρ + ζ·ȳ), ρ=$(round(ρ; digits=2)), ζ=$(round(ζ; digits=2))",
        linewidth = 2, color = :red,
    )
    mkpath(dirname(abspath(file)))
    StatsPlots.savefig(plt, file)
    return file
end

"""
    _plot_all_sigmoids(fit, protocol_assignment, file)

All-sigmoids overlay grid, panels by protocol. Output: PNG at `file`,
size scales with n_protocols. Plot 2 of 3.

`protocol_assignment` is a length-n_columns Vector{Int} mapping each column
to a protocol index. If absent (`nothing`), all columns are plotted in a
single panel.
"""
function _plot_all_sigmoids(fit::DropoutFit,
                            protocol_assignment::Union{Nothing, Vector{Int}},
                            file::String)
    n_cols = length(fit.column_names)
    assignment = isnothing(protocol_assignment) ? ones(Int, n_cols) : protocol_assignment
    n_protocols = maximum(assignment)
    x_grid = range(-3.0, 3.0; length = 200)
    panels = Any[]
    for p in 1:n_protocols
        cols_p = findall(assignment .== p)
        plt_p = StatsPlots.plot(; xlabel = "ȳ", ylabel = "P(det)",
                                  title = "Protocol $p", ylims = (-0.02, 1.02),
                                  legend = false)
        for c in cols_p
            isnan(fit.rho[c]) && continue
            y_grid = 1.0 ./ (1.0 .+ exp.(.-(fit.rho[c] .+ fit.zeta[c] .* x_grid)))
            StatsPlots.plot!(plt_p, x_grid, y_grid;
                             label = nothing, linewidth = 1, alpha = 0.6)
        end
        push!(panels, plt_p)
    end
    layout_shape = n_protocols <= 3 ? (1, n_protocols) : (2, ceil(Int, n_protocols/2))
    plt = StatsPlots.plot(panels...; layout = layout_shape, size = (1200, 800))
    mkpath(dirname(abspath(file)))
    StatsPlots.savefig(plt, file)
    return file
end

"""
    _plot_zeta_distribution(fit, file)

Histogram of `ζ̂_c` with vertical lines at 0.5 and 3.0 marking the §6 expectation
band. Plot 3 of 3.
"""
function _plot_zeta_distribution(fit::DropoutFit, file::String)
    z = filter(!isnan, fit.zeta)
    plt = StatsPlots.histogram(z;
        xlabel = "ζ̂_c", ylabel = "count", bins = 30, label = nothing,
        title = "Per-column dropout slope (n=$(length(z)))", size = (800, 600),
    )
    StatsPlots.vline!(plt, [0.5, 3.0]; label = nothing,
                      color = :black, linestyle = :dash, linewidth = 0.8)
    mkpath(dirname(abspath(file)))
    StatsPlots.savefig(plt, file)
    return file
end

# ============================================================================
# SANITY.md writer
# ============================================================================

"""
    _write_sanity_report(fit, protocol_assignment, file; min_detections=5,
                         expected_band=(0.5, 3.0))

Write a free-text Markdown sanity-check report covering the five fit-quality metrics:
  1. Fraction of columns with `ζ̂_c` in the expectation band [0.5, 3]
  2. Fraction of columns with `ζ̂_c < 0` (target 0%)
  3. Fraction excluded for `n_detections < min_detections`
  4. Per-protocol mean ± SD of `ζ̂_c`
  5. List of outlier column indices (`ζ̂ > 3` or `ζ̂ < 0.3`)

Returns `file`. Always written to disk via `Base.write` (shadowing convention).
"""
function _write_sanity_report(fit::DropoutFit,
                              protocol_assignment::Union{Nothing, Vector{Int}},
                              file::String;
                              min_detections::Int = 5,
                              expected_band::Tuple{Float64, Float64} = (0.5, 3.0))
    n_cols = length(fit.column_names)
    excluded = isnan.(fit.rho)
    n_excl = sum(excluded)
    z_fit = fit.zeta[.!excluded]
    n_well = sum(expected_band[1] .≤ z_fit .≤ expected_band[2])
    n_neg = sum(z_fit .< 0)
    outliers = Int[]
    for c in 1:n_cols
        isnan(fit.zeta[c]) && continue
        (fit.zeta[c] > expected_band[2] || fit.zeta[c] < 0.3) && push!(outliers, c)
    end

    io = IOBuffer()
    println(io, "# Dropout-Curve Fit Sanity Check")
    println(io, "")
    println(io, "- Generated: ", fit.fit_timestamp)
    println(io, "- Software:  ", fit.software_version)
    println(io, "- Dataset:   ", fit.dataset_hash)
    println(io, "- Columns:   ", n_cols, "  (fit: ", n_cols - n_excl, ", excluded: ", n_excl, ")")
    println(io, "- Proteins:  ", fit.n_proteins)
    println(io, "")
    println(io, "## Fit-Quality Metrics")
    println(io, "")
    n_fit = max(n_cols - n_excl, 1)
    pct_well = round(100 * n_well / n_fit; digits = 1)
    pct_neg  = round(100 * n_neg / n_fit; digits = 1)
    pct_excl = round(100 * n_excl / n_cols; digits = 1)
    println(io, "| Metric | Value | Target |")
    println(io, "|--------|-------|--------|")
    println(io, "| ζ̂_c ∈ [", expected_band[1], ", ", expected_band[2], "] | $n_well / $n_fit ($(pct_well)%) | > 70% |")
    println(io, "| ζ̂_c < 0 | $n_neg / $n_fit ($(pct_neg)%) | 0% |")
    println(io, "| Excluded (n_det < $min_detections or GLM failure) | $n_excl / $n_cols ($(pct_excl)%) | informational |")
    println(io, "")

    if !isnothing(protocol_assignment)
        println(io, "## Per-Protocol ζ̂_c Statistics")
        println(io, "")
        println(io, "| Protocol | n columns | ζ̂ mean | ζ̂ SD | n excluded |")
        println(io, "|----------|-----------|--------|------|------------|")
        n_protocols = maximum(protocol_assignment)
        for p in 1:n_protocols
            idx = findall(protocol_assignment .== p)
            z_p = fit.zeta[idx]
            z_p_fit = z_p[.!isnan.(z_p)]
            mu = isempty(z_p_fit) ? NaN : mean(z_p_fit)
            sd = length(z_p_fit) >= 2 ? std(z_p_fit) : NaN
            n_excl_p = sum(isnan.(z_p))
            println(io, "| $p | $(length(idx)) | $(round(mu; digits=3)) | $(round(sd; digits=3)) | $n_excl_p |")
        end
        println(io, "")
    end

    if !isempty(outliers)
        println(io, "## Outlier Columns (ζ̂ > $(expected_band[2]) or ζ̂ < 0.3)")
        println(io, "")
        for c in outliers
            println(io, "- col $c (", fit.column_names[c], "): ζ̂ = ",
                    round(fit.zeta[c]; digits = 3))
        end
        println(io, "")
    end

    if pct_well < 70.0
        println(io, "**WARNING:** Less than 70% of columns are well-fit. Consider revisiting the `condition_groups` grouping.")
        println(io, "")
    end

    mkpath(dirname(abspath(file)))
    Base.write(file, String(take!(io)))
    return file
end

# ============================================================================
# Diagnostics orchestrator (replaces the Task-1 stub)
# ============================================================================

"""
    _emit_diagnostics(fit, m_kept, ybar, detected_mat; diagnostics_dir,
                      protocol_assignment, min_detections_per_column)

Write the three diagnostic plots and the SANITY.md report under `diagnostics_dir`.
"""
function _emit_diagnostics(fit::DropoutFit, m_kept, ybar, detected_mat;
                           diagnostics_dir::String,
                           protocol_assignment = nothing,
                           min_detections_per_column::Int = 5)
    mkpath(diagnostics_dir)

    # 1. Per-column scatter+sigmoid plots
    n_cols = length(fit.column_names)
    for c in 1:n_cols
        isnan(fit.rho[c]) && continue
        safe_name = replace(fit.column_names[c], r"[^A-Za-z0-9_-]" => "_")
        file = joinpath(diagnostics_dir,
                        "col_" * lpad(c, 2, '0') * "_" * safe_name * ".png")
        det_c = collect(detected_mat[:, c])
        _plot_column_fit(ybar, det_c, fit.rho[c], fit.zeta[c],
                         fit.column_names[c], file)
    end

    # 2. All-sigmoids overlay grid
    _plot_all_sigmoids(fit, protocol_assignment,
                       joinpath(diagnostics_dir, "all_sigmoids.png"))

    # 3. ζ̂ distribution histogram
    _plot_zeta_distribution(fit,
                            joinpath(diagnostics_dir, "zeta_distribution.png"))

    # 4. SANITY.md report
    _write_sanity_report(fit, protocol_assignment,
                         joinpath(diagnostics_dir, "SANITY.md");
                         min_detections = min_detections_per_column)

    return diagnostics_dir
end

# ============================================================================
# Non-exported intensity-matrix loader (used by the imputation CLI script)
# ============================================================================

"""
    _load_intensity_matrix(xlsx_path; sheet_name="Sheet1", id_col=1) -> (matrix, column_names)

Load an intensity matrix from `dataset.xlsx` without going through `load_data`.
Returns `(Matrix{Union{Missing, Float64}}, Vector{String})`.

The first `id_col` columns are treated as protein identifiers and discarded;
all remaining columns are intensity columns. CSV/XLSX returns `InlineString`
column names — these are coerced to `String` per the project memory note.
"""
function _load_intensity_matrix(xlsx_path::String;
                                sheet_name::String = "Sheet1",
                                id_col::Int = 1)
    raw_df = DataFrame(readtable(xlsx_path, sheet_name))
    all_names = String.(names(raw_df))
    column_names = all_names[id_col + 1 : end]
    raw_matrix = Matrix(raw_df[:, id_col + 1 : end])
    intensity_matrix = convert(Matrix{Union{Missing, Float64}}, raw_matrix)
    return intensity_matrix, column_names
end

"""
    column_imputation_sigma(fit::DropoutFit, col::Int,
                            intensity_matrix::AbstractMatrix) -> Float64

Empirical σ (sqrt-variance) of the finite-numeric values in column `col` of
the post-imputation `intensity_matrix`. Returns 0.0 when fewer than 2 finite
values are available.

Used by v2b mask-aware regression to source σ_imp per source-
matrix column. `fit` is consulted for boundscheck only (col ∈ 1:length(fit.rho));
the actual σ value comes from the intensity matrix to match the
post-imputation variance pattern (src/analysis/pipeline.jl:513-518).
"""
function BayesInteractomics.column_imputation_sigma(
    fit::DropoutFit,
    col::Int,
    intensity_matrix::AbstractMatrix,
)::Float64
    @boundscheck (1 <= col <= length(fit.rho)) || throw(BoundsError(fit, col))
    raw_col = view(intensity_matrix, :, col)
    finite_vals = Float64[]
    for x in raw_col
        ismissing(x) && continue
        xf = Float64(x)
        isnan(xf) && continue
        push!(finite_vals, xf)
    end
    return length(finite_vals) >= 2 ? sqrt(var(finite_vals)) : 0.0
end
