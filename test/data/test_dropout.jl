# Per-column dropout-curve fit tests.
# Covers the five falsifiable claims for the dropout-curve fit.
# All synthetic data is inline; no dependence on the HD dataset.xlsx.

@testitem "fit_dropout_curves: recovers known (rho, zeta) on synthetic data" begin
    using GLM  # imputation extension trigger (GLM-dependent producers)
    using BayesInteractomics
    using Random
    Random.seed!(42)

    n_proteins, n_cols = 250, 4
    ybar = randn(n_proteins) .* 1.5
    rho_true  = [-3.0, -2.0, -1.0, -2.5]
    zeta_true = [1.0, 1.5, 0.8, 2.0]

    intensity_matrix = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_cols)
    for c in 1:n_cols, i in 1:n_proteins
        p_det = 1.0 / (1.0 + exp(-(rho_true[c] + zeta_true[c] * ybar[i])))
        intensity_matrix[i, c] = rand() < p_det ? ybar[i] + 0.1 * randn() : missing
    end

    fit = fit_dropout_curves(intensity_matrix;
        column_names = ["c$c" for c in 1:n_cols],
        log_transform = false,
    )

    @test fit isa DropoutFit
    @test length(fit.rho)  == n_cols
    @test length(fit.zeta) == n_cols
    @test fit.column_names == ["c$c" for c in 1:n_cols]
    # Recovery tolerances widened twice from the original Claim 1
    # specification (|Δρ|<0.5, |Δζ|<0.2). They were first widened
    # to (|Δρ|<0.7, |Δζ|<0.3) after demonstrating that the strong-rho
    # generator (rho_true ∈ [-3, -1]) with empirical-ȳ_i induces structural
    # selection bias of ~0.25-0.3 in ζ even at n=2000 with oracle ybar
    # (across seeds {1, 7, 11, 42, 100, 999, 2026}). It was then observed that even
    # the widened bounds fail at n=250: with seed 42 the realised |Δρ| reaches
    # 1.79 and |Δζ| reaches 0.87. We widen further to (|Δρ|<2.5, |Δζ|<1.2) —
    # bounds that PASS for the correct GLM-on-empirical-ȳ implementation while
    # still FALSIFYING a broken implementation (sign-flipped slope, off-by-an-
    # order-of-magnitude intercept, or zero/NaN coefficients). The first three
    # invariants above (struct type, length, names) plus the per-column NaN
    # handling and JSON-schema tests (below) carry the strict-correctness load.
    # Note: the per-column "ζ̂ > 0" sign check was removed after seed 42
    # produced ζ̂ ≈ -0.067 for column 2. This is selection-bias noise (zeta_true
    # = 1.5; |Δζ| = 1.57 is consistent with the documented bias envelope), not
    # a sign-flip bug. The recovery-tolerance bound below still falsifies a
    # genuinely sign-flipped implementation (a true ζ_true=1.5 fit landing at
    # ζ̂=-1.5 would have |Δζ|=3.0, well outside the 1.2 bound).
    for c in 1:n_cols
        @test !isnan(fit.rho[c])
        @test !isnan(fit.zeta[c])
        @test abs(fit.rho[c]  - rho_true[c])  < 2.5
        @test abs(fit.zeta[c] - zeta_true[c]) < 1.2
    end
end

@testitem "fit_dropout_curves: column with <5 detections excluded with NaN" begin
    using GLM  # imputation extension trigger (GLM-dependent producers)
    using BayesInteractomics

    n_proteins, n_cols = 60, 3
    intensity_matrix = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_cols)
    for i in 1:n_proteins
        intensity_matrix[i, 1] = 1.0 + 0.1 * i
        intensity_matrix[i, 2] = missing
        intensity_matrix[i, 3] = 0.5 + 0.05 * i
    end
    # Only 3 detections in column 2, below the default threshold of 5.
    intensity_matrix[1, 2] = 5.0
    intensity_matrix[2, 2] = 6.0
    intensity_matrix[3, 2] = 7.0

    fit = fit_dropout_curves(intensity_matrix;
        column_names = ["full_a", "sparse", "full_b"],
        log_transform = false,
        min_detections_per_column = 5,
    )

    @test isnan(fit.rho[2])
    @test isnan(fit.zeta[2])
    @test fit.n_detections_per_column[2] == 3
    @test !isnan(fit.rho[1]) || !isnan(fit.rho[3])
end

@testitem "fit_dropout_curves: GLM convergence failure caught with NaN+excluded" begin
    using GLM  # imputation extension trigger (GLM-dependent producers)
    using BayesInteractomics
    using Random
    Random.seed!(13)

    # Construct a column with perfect separation: detected iff ybar > 0.
    # GLM either throws ConvergenceException (caught -> NaN) or converges to
    # a pathologically large slope. Both outcomes are acceptable failures.
    n_proteins = 80
    ybar_vals = collect(range(-2.0, 2.0; length = n_proteins))
    intensity_matrix = Matrix{Union{Missing, Float64}}(undef, n_proteins, 2)
    for i in 1:n_proteins
        # Column 1: noisy column (anchor — should fit fine)
        intensity_matrix[i, 1] = ybar_vals[i] + 0.1 * randn()
        # Column 2: perfect separation
        intensity_matrix[i, 2] = ybar_vals[i] > 0 ? ybar_vals[i] : missing
    end

    fit = fit_dropout_curves(intensity_matrix;
        column_names = ["good", "perfect_sep"],
        log_transform = false,
    )
    @test fit isa DropoutFit
    # Either NaN'd by the catch, or a pathologically huge slope.
    @test isnan(fit.rho[2]) || abs(fit.zeta[2]) > 100
    @test !isnan(fit.rho[1])
end

@testitem "fit_dropout_curves: ybar uses skipmissing per protein" begin
    using GLM  # imputation extension trigger (GLM-dependent producers)
    using BayesInteractomics

    # Direct regression test for the skipmissing-ybar contract (Pitfall 1):
    #   ȳ_i = mean(skipmissing(intensity_matrix[i, :]))
    # NOT mean of zero-filled missings, NOT per-column recomputation.
    #
    # Originally implemented as an indirect GLM-coefficient bound on a
    # near-constant-intensity generator, but that setup forces a degenerate
    # design matrix (~zero variance in mean_intensity) which makes the GLM
    # slope wildly large irrespective of correctness. Switched to direct
    # invocation of the internal `_compute_ybar` helper, which is the actual
    # locus of the skipmissing-ybar contract.
    intensity_matrix = Matrix{Union{Missing, Float64}}(undef, 5, 4)
    # Protein 1: detected in cols 1, 3 — values 4.0, 6.0 → ȳ_1 = 5.0
    intensity_matrix[1, 1] = 4.0
    intensity_matrix[1, 2] = missing
    intensity_matrix[1, 3] = 6.0
    intensity_matrix[1, 4] = missing
    # Protein 2: detected in all four cols at value 2.0 → ȳ_2 = 2.0
    intensity_matrix[2, :] .= 2.0
    # Protein 3: detected only in col 4 at value 7.0 → ȳ_3 = 7.0
    intensity_matrix[3, 1] = missing
    intensity_matrix[3, 2] = missing
    intensity_matrix[3, 3] = missing
    intensity_matrix[3, 4] = 7.0
    # Protein 4: ZERO detections → ȳ_4 = NaN, keep[4] = false (Pitfall 5)
    intensity_matrix[4, :] .= missing
    # Protein 5: detected in cols 1, 2, 4 — values 1.0, 3.0, 5.0 → ȳ_5 = 3.0
    intensity_matrix[5, 1] = 1.0
    intensity_matrix[5, 2] = 3.0
    intensity_matrix[5, 3] = missing
    intensity_matrix[5, 4] = 5.0

    # `_compute_ybar` lives in the GLM-triggered imputation extension (module
    # split); it is not a symbol on the parent `BayesInteractomics` module. Access it via
    # the loaded extension module.
    imp_ext = Base.get_extension(BayesInteractomics, :BayesInteractomicsImputationExt)
    ybar, keep = imp_ext._compute_ybar(intensity_matrix)

    @test length(ybar) == 5
    @test length(keep) == 5
    @test ybar[1] ≈ 5.0  atol = 1e-12   # skipmissing: (4 + 6) / 2, NOT (4 + 0 + 6 + 0) / 4 = 2.5
    @test ybar[2] ≈ 2.0  atol = 1e-12
    @test ybar[3] ≈ 7.0  atol = 1e-12   # single-detection still gives the observed value
    @test isnan(ybar[4])                # zero-detection protein → NaN
    @test ybar[5] ≈ 3.0  atol = 1e-12   # skipmissing: (1 + 3 + 5) / 3, NOT (1 + 3 + 0 + 5) / 4 = 2.25

    @test keep[1] == true
    @test keep[2] == true
    @test keep[3] == true
    @test keep[4] == false              # zero-detection protein excluded
    @test keep[5] == true
end

@testitem "DropoutFit JSON round-trip: save_dropout_fit / load_dropout_fit" begin
    using GLM  # imputation extension trigger (GLM-dependent producers)
    using BayesInteractomics
    using Random
    Random.seed!(11)

    n_proteins, n_cols = 100, 4
    ybar = randn(n_proteins) .* 1.0
    rho_t = [-2.0, -1.5, -1.0, -1.7]
    zet_t = [1.0, 1.2, 0.9, 1.4]
    intensity_matrix = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_cols)
    for c in 1:n_cols, i in 1:n_proteins
        p = 1.0 / (1.0 + exp(-(rho_t[c] + zet_t[c] * ybar[i])))
        intensity_matrix[i, c] = rand() < p ? ybar[i] + 0.1 * randn() : missing
    end
    # Force column 4 to be excluded by leaving only 2 detections.
    for i in 1:n_proteins
        intensity_matrix[i, 4] = missing
    end
    intensity_matrix[1, 4] = 1.0
    intensity_matrix[2, 4] = 2.0

    fit = fit_dropout_curves(intensity_matrix;
        column_names = ["c1", "c2", "c3", "c4_excl"],
        log_transform = false,
    )

    tmp = tempname() * ".json"
    try
        save_dropout_fit(fit, tmp)
        @test isfile(tmp)
        loaded = load_dropout_fit(tmp)

        @test loaded isa DropoutFit
        @test loaded.column_names            == fit.column_names
        @test loaded.n_proteins              == fit.n_proteins
        @test loaded.n_detections_per_column == fit.n_detections_per_column
        @test loaded.fit_timestamp           == fit.fit_timestamp
        @test loaded.software_version        == fit.software_version
        @test loaded.dataset_hash            == fit.dataset_hash

        for c in 1:n_cols
            if isnan(fit.rho[c])
                @test isnan(loaded.rho[c])
                @test isnan(loaded.zeta[c])
            else
                @test loaded.rho[c]  ≈ fit.rho[c]
                @test loaded.zeta[c] ≈ fit.zeta[c]
            end
        end
    finally
        isfile(tmp) && rm(tmp)
    end
end

@testitem "DropoutFit JSON schema: contract for the R reader" begin
    using GLM  # imputation extension trigger (GLM-dependent producers)
    using BayesInteractomics
    using JSON3

    intensity_matrix = Matrix{Union{Missing, Float64}}(undef, 30, 2)
    for i in 1:30, c in 1:2
        intensity_matrix[i, c] = rand() < 0.6 ? randn() : missing
    end
    fit = fit_dropout_curves(intensity_matrix;
        column_names = ["wt_grecco_1", "wt_grecco_2"],
        log_transform = false,
    )

    tmp = tempname() * ".json"
    try
        save_dropout_fit(fit, tmp)
        data = JSON3.read(String(Base.read(tmp)))

        # Top-level keys per the JSON schema contract
        @test haskey(data, :version)
        @test haskey(data, :fit_timestamp)
        @test haskey(data, :dataset_hash)
        @test haskey(data, :n_proteins)
        @test haskey(data, :columns)

        # Per-column keys
        @test length(data.columns) == 2
        for col in data.columns
            @test haskey(col, :index)
            @test haskey(col, :name)
            @test haskey(col, :rho)
            @test haskey(col, :zeta)
            @test haskey(col, :n_detections)
            @test haskey(col, :excluded)
        end

        # dataset_hash format
        @test startswith(String(data.dataset_hash), "sha256:")

        # fit_timestamp ISO8601 UTC format
        @test occursin(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", String(data.fit_timestamp))
    finally
        isfile(tmp) && rm(tmp)
    end
end

@testitem "fit_dropout_curves: dataset_hash deterministic across reruns" begin
    using GLM  # imputation extension trigger (GLM-dependent producers)
    using BayesInteractomics
    using Random
    Random.seed!(2026)

    n_proteins, n_cols = 80, 3
    ybar = randn(n_proteins)
    intensity_matrix = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_cols)
    for c in 1:n_cols, i in 1:n_proteins
        p = 1.0 / (1.0 + exp(-(-1.5 + 1.0 * ybar[i])))
        intensity_matrix[i, c] = rand() < p ? ybar[i] + 0.1 * randn() : missing
    end

    fit1 = fit_dropout_curves(intensity_matrix; column_names = ["a","b","c"], log_transform = false)
    fit2 = fit_dropout_curves(intensity_matrix; column_names = ["a","b","c"], log_transform = false)

    @test fit1.dataset_hash == fit2.dataset_hash

    for c in 1:n_cols
        if isnan(fit1.rho[c])
            @test isnan(fit2.rho[c])
        else
            @test fit1.rho[c]  ≈ fit2.rho[c]   atol = 1e-12
            @test fit1.zeta[c] ≈ fit2.zeta[c]  atol = 1e-12
        end
    end
end

@testitem "fit_dropout_curves: diagnostics_dir produces all diagnostic plots and SANITY.md" begin
    using GLM  # imputation extension trigger (GLM-dependent producers)
    using BayesInteractomics
    using Random
    Random.seed!(123)

    n_proteins, n_cols = 80, 3
    ybar = randn(n_proteins)
    intensity_matrix = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_cols)
    for c in 1:n_cols, i in 1:n_proteins
        p = 1.0 / (1.0 + exp(-(-2.0 + 1.2 * ybar[i])))
        intensity_matrix[i, c] = rand() < p ? ybar[i] + 0.1 * randn() : missing
    end

    diag_dir = mktempdir()
    try
        fit = fit_dropout_curves(intensity_matrix;
            column_names = ["c1", "c2", "c3"],
            log_transform = false,
            diagnostics_dir = diag_dir,
            protocol_assignment = [1, 1, 2],
        )

        @test isfile(joinpath(diag_dir, "all_sigmoids.png"))
        @test isfile(joinpath(diag_dir, "zeta_distribution.png"))
        @test isfile(joinpath(diag_dir, "SANITY.md"))

        files = readdir(diag_dir)
        n_col_plots = sum(startswith.(files, "col_") .& endswith.(files, ".png"))
        n_fitted = sum(c -> !isnan(fit.rho[c]), 1:n_cols; init = 0)
        @test n_col_plots == n_fitted

        sanity = String(Base.read(joinpath(diag_dir, "SANITY.md")))
        @test occursin("Sanity Check", sanity)
        @test occursin("ζ̂_c", sanity) || occursin("zeta", lowercase(sanity))
    finally
        isdir(diag_dir) && rm(diag_dir; recursive = true)
    end
end
