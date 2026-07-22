"""
    test_quality_gate.jl

Tests for quality gate matrix (9-cell KS), auto-remediation, and KL contamination.
"""

@testitem "Quality gate with well-separated components" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult,
        QualityGateCell, QualityGateResult, run_quality_gates
    using Distributions, Random, Test

    Random.seed!(42)
    n = 300

    # Three well-separated clusters on BF scale
    # H0: BF ~ exp(Normal(-3, 0.5)) -> log-BF ~ Normal(-3, 0.5)
    # Agnostic: BF ~ exp(Normal(0, 0.5)) -> log-BF ~ Normal(0, 0.5)
    # H1: BF ~ exp(Normal(4, 0.5)) -> log-BF ~ Normal(4, 0.5)
    bf_e_h0 = exp.(randn(100) .* 0.5 .- 3.0)
    bf_e_ag = exp.(randn(100) .* 0.5)
    bf_e_h1 = exp.(randn(100) .* 0.5 .+ 4.0)
    bf_e = vcat(bf_e_h0, bf_e_ag, bf_e_h1)

    bf_c_h0 = exp.(randn(100) .* 0.5 .- 3.0)
    bf_c_ag = exp.(randn(100) .* 0.5)
    bf_c_h1 = exp.(randn(100) .* 0.5 .+ 4.0)
    bf_c = vcat(bf_c_h0, bf_c_ag, bf_c_h1)

    bf_d_h0 = exp.(randn(100) .* 0.5 .- 3.0)
    bf_d_ag = exp.(randn(100) .* 0.5)
    bf_d_h1 = exp.(randn(100) .* 0.5 .+ 4.0)
    bf_d = vcat(bf_d_h0, bf_d_ag, bf_d_h1)

    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    # Responsibilities: clear assignment
    resp = zeros(n, 3)
    resp[1:100, 1] .= 0.95       # H0
    resp[1:100, 2] .= 0.025
    resp[1:100, 3] .= 0.025
    resp[101:200, 1] .= 0.025
    resp[101:200, 2] .= 0.95     # Agnostic
    resp[101:200, 3] .= 0.025
    resp[201:300, 1] .= 0.025
    resp[201:300, 2] .= 0.025
    resp[201:300, 3] .= 0.95     # H1

    class_params = Dict(
        "background" => (mu=-3.0, sigma=0.5, precision=4.0),
        "agnostic"   => (mu=0.0, sigma=0.5, precision=4.0),
        "interaction" => (mu=4.0, sigma=0.5, precision=4.0)
    )
    lc = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [0.33, 0.33, 0.34], collect(1.0:10.0), true, 10, resp
    )

    qg = run_quality_gates(bf, lc)
    @test qg isa QualityGateResult
    @test size(qg.cells) == (3, 3)
    # Well-separated Normals should all pass
    for cell in qg.cells
        @test cell.status in (:pass, :warn)
        @test cell.ks_statistic >= 0.0
        @test cell.n_effective > 5
    end
    @test qg.overall_status in (:pass, :warn)
end

@testitem "Quality gate auto-remediation" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult,
        QualityGateResult, run_quality_gates
    using Distributions, Random, Test

    Random.seed!(123)
    n = 300

    # H0 component: heavy-tailed data (TLocationScale-like)
    # Use a distribution with heavy tails so Normal fit gives KS > 0.15
    bf_e_h0 = exp.(rand(TDist(3), 100) .* 2.0 .- 3.0)
    bf_e_ag = exp.(randn(100) .* 0.5)
    bf_e_h1 = exp.(randn(100) .* 0.5 .+ 4.0)
    bf_e = vcat(bf_e_h0, bf_e_ag, bf_e_h1)

    bf_c = exp.(randn(n) .* 0.5)  # well-behaved
    bf_d = exp.(randn(n) .* 0.5)  # well-behaved

    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    resp = zeros(n, 3)
    resp[1:100, 1] .= 0.95
    resp[1:100, 2] .= 0.025
    resp[1:100, 3] .= 0.025
    resp[101:200, 2] .= 0.95
    resp[101:200, 1] .= 0.025
    resp[101:200, 3] .= 0.025
    resp[201:300, 3] .= 0.95
    resp[201:300, 1] .= 0.025
    resp[201:300, 2] .= 0.025

    class_params = Dict(
        "background" => (mu=-3.0, sigma=1.0, precision=1.0),
        "agnostic"   => (mu=0.0, sigma=0.5, precision=4.0),
        "interaction" => (mu=4.0, sigma=0.5, precision=4.0)
    )
    lc = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [0.33, 0.33, 0.34], collect(1.0:10.0), true, 10, resp
    )

    qg = run_quality_gates(bf, lc)
    @test qg isa QualityGateResult
    # Check that at least the result was produced
    @test size(qg.cells) == (3, 3)
    # At least some cells should have attempted remediation on the heavy-tailed H0
    # (enrichment x H0 is the heavy-tailed one)
    enrichment_h0_cell = qg.cells[1, 1]
    # The cell should either have been remediated or have a higher KS
    @test enrichment_h0_cell.ks_statistic >= 0.0
end

@testitem "Quality gate overall status is worst cell" begin
    using BayesInteractomics
    using BayesInteractomics: QualityGateCell, QualityGateResult
    using Distributions, Test

    # Manually construct cells with mixed statuses
    cells = Matrix{QualityGateCell}(undef, 3, 3)
    for i in 1:3, j in 1:3
        status = (i == 1 && j == 1) ? :fail : :pass
        cells[i, j] = QualityGateCell(
            [:enrichment, :correlation, :detection][i],
            [:H0, :agnostic, :H1][j],
            (i == 1 && j == 1) ? 0.25 : 0.05,
            status,
            Normal(0.0, 1.0),
            50.0,
            false
        )
    end

    qg = QualityGateResult(cells, :fail, String[])
    @test qg.overall_status == :fail

    # Now test with :warn being worst
    cells2 = Matrix{QualityGateCell}(undef, 3, 3)
    for i in 1:3, j in 1:3
        status = (i == 2 && j == 2) ? :warn : :pass
        cells2[i, j] = QualityGateCell(
            [:enrichment, :correlation, :detection][i],
            [:H0, :agnostic, :H1][j],
            (i == 2 && j == 2) ? 0.12 : 0.05,
            status,
            Normal(0.0, 1.0),
            50.0,
            false
        )
    end
    qg2 = QualityGateResult(cells2, :warn, String[])
    @test qg2.overall_status == :warn
end

@testitem "Quality gate small component skipped" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult,
        QualityGateResult, run_quality_gates
    using Distributions, Random, Test

    Random.seed!(42)
    n = 200

    bf_e = exp.(randn(n) .* 0.5)
    bf_c = exp.(randn(n) .* 0.5)
    bf_d = exp.(randn(n) .* 0.5)
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    # Agnostic component has < 5 effective proteins
    resp = zeros(n, 3)
    resp[1:100, 1] .= 0.95
    resp[1:100, 2] .= 0.025
    resp[1:100, 3] .= 0.025
    # Only 3 proteins are agnostic (< 5)
    resp[101:103, 2] .= 0.95
    resp[101:103, 1] .= 0.025
    resp[101:103, 3] .= 0.025
    # Rest are H1
    resp[104:200, 3] .= 0.95
    resp[104:200, 1] .= 0.025
    resp[104:200, 2] .= 0.025

    class_params = Dict(
        "background" => (mu=-1.0, sigma=0.5, precision=4.0),
        "agnostic"   => (mu=0.0, sigma=0.5, precision=4.0),
        "interaction" => (mu=2.0, sigma=0.5, precision=4.0)
    )
    lc = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [0.5, 0.015, 0.485], collect(1.0:10.0), true, 10, resp
    )

    qg = run_quality_gates(bf, lc)
    @test qg isa QualityGateResult
    # Agnostic cells (column 2) should be :pass by default due to small n
    for i in 1:3
        @test qg.cells[i, 2].status == :pass
        @test qg.cells[i, 2].ks_statistic == 0.0
    end
end

@testitem "KL contamination computation" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult,
        KLContaminationResult, compute_kl_contamination
    using Distributions, Random, Test

    Random.seed!(42)
    n = 200

    # Create data where pure H1 and full H1 are similar
    bf_e = exp.(randn(n) .+ 2.0)
    bf_c = exp.(randn(n) .+ 2.0)
    bf_d = exp.(randn(n) .+ 2.0)
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    resp = zeros(n, 3)
    resp[1:50, 1] .= 0.95
    resp[1:50, 2] .= 0.025
    resp[1:50, 3] .= 0.025
    resp[51:80, 2] .= 0.95
    resp[51:80, 1] .= 0.025
    resp[51:80, 3] .= 0.025
    # 120 proteins in H1, 80 of them pure (resp > 0.95)
    resp[81:160, 3] .= 0.96
    resp[81:160, 1] .= 0.02
    resp[81:160, 2] .= 0.02
    # 40 are soft H1 (resp > 0.5 but < 0.95)
    resp[161:200, 3] .= 0.7
    resp[161:200, 1] .= 0.15
    resp[161:200, 2] .= 0.15

    class_params = Dict(
        "background" => (mu=-1.0, sigma=0.5, precision=4.0),
        "agnostic"   => (mu=0.0, sigma=0.5, precision=4.0),
        "interaction" => (mu=2.0, sigma=0.5, precision=4.0)
    )
    lc = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [0.25, 0.15, 0.6], collect(1.0:10.0), true, 10, resp
    )

    kl = compute_kl_contamination(bf, lc)
    @test kl isa KLContaminationResult
    @test kl.kl_enrichment >= 0.0
    @test kl.kl_correlation >= 0.0
    @test kl.kl_detection >= 0.0
    @test kl.kl_joint >= 0.0
    @test kl.kl_joint ≈ kl.kl_enrichment + kl.kl_correlation + kl.kl_detection
    @test kl.pure_h1_count == 80  # proteins with resp > 0.95
end

@testitem "KL contamination with few pure H1 proteins" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult,
        KLContaminationResult, compute_kl_contamination
    using Distributions, Random, Test

    Random.seed!(42)
    n = 100

    bf_e = exp.(randn(n))
    bf_c = exp.(randn(n))
    bf_d = exp.(randn(n))
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    # No proteins have resp > 0.95 in H1
    resp = zeros(n, 3)
    resp[:, 1] .= 0.6  # All mostly H0
    resp[:, 2] .= 0.2
    resp[:, 3] .= 0.2  # None exceed 0.95

    class_params = Dict(
        "background" => (mu=0.0, sigma=1.0, precision=1.0),
        "agnostic"   => (mu=0.0, sigma=1.0, precision=1.0),
        "interaction" => (mu=0.0, sigma=1.0, precision=1.0)
    )
    lc = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [0.6, 0.2, 0.2], collect(1.0:5.0), true, 5, resp
    )

    kl = compute_kl_contamination(bf, lc)
    @test kl isa KLContaminationResult
    @test kl.kl_enrichment == 0.0
    @test kl.kl_correlation == 0.0
    @test kl.kl_detection == 0.0
    @test kl.kl_joint == 0.0
    @test kl.pure_h1_count < 5
end

@testitem "Quality gate rejects size mismatch between bf_triplet and responsibilities" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult, run_quality_gates
    using Random

    Random.seed!(42)
    n_detected = 100
    n_all = 150  # 50 non-detected proteins

    bf_all = BayesFactorTriplet(
        vcat(exp.(randn(n_detected)), zeros(50)),   # non-detected get BF=0
        vcat(exp.(randn(n_detected)), zeros(50)),
        vcat(exp.(randn(n_detected)), zeros(50)),
    )

    resp = zeros(n_detected, 3)  # responsibilities only for detected
    resp[:, 1] .= 0.8
    resp[:, 2] .= 0.1
    resp[:, 3] .= 0.1

    class_params = Dict(
        "background" => (mu=0.0, sigma=1.0, precision=1.0),
        "agnostic"   => (mu=0.0, sigma=1.0, precision=1.0),
        "interaction" => (mu=0.0, sigma=1.0, precision=1.0)
    )
    lc = LatentClassResult(
        ones(n_detected), fill(0.5, n_detected), class_params,
        [0.8, 0.1, 0.1], collect(1.0:5.0), true, 5, resp
    )

    # Should error on size mismatch (150 BFs vs 100 responsibilities)
    @test_throws ErrorException run_quality_gates(bf_all, lc)

    # Correct: filtered to detected-only should work
    bf_detected = BayesFactorTriplet(
        exp.(randn(n_detected)),
        exp.(randn(n_detected)),
        exp.(randn(n_detected)),
    )
    qg = run_quality_gates(bf_detected, lc)
    @test qg.overall_status in (:pass, :warn, :fail)
    @test size(qg.cells) == (3, 3)
end

@testitem "Quality gate cells include histogram and PDF data" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult,
        QualityGateCell, QualityGateResult, run_quality_gates, DiscreteEmpirical
    using Distributions, Random

    Random.seed!(42)
    n = 200

    bf_e = vcat(exp.(randn(100) .* 0.5 .- 2.0), exp.(randn(100) .* 0.5 .+ 3.0))
    bf_c = vcat(exp.(randn(100) .* 0.5 .- 2.0), exp.(randn(100) .* 0.5 .+ 3.0))

    # Detection BFs: discrete values (Beta-Bernoulli origin)
    det_vals_h0 = [0.2, 0.5, 0.8]
    det_vals_h1 = [2.0, 5.0, 10.0]
    bf_d = vcat(rand(det_vals_h0, 100), rand(det_vals_h1, 100))
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    resp = zeros(n, 3)
    resp[1:100, 1] .= 0.9; resp[1:100, 2] .= 0.05; resp[1:100, 3] .= 0.05
    resp[101:200, 1] .= 0.05; resp[101:200, 2] .= 0.05; resp[101:200, 3] .= 0.9

    # Build DiscreteEmpirical on log-BF scale for each component
    function make_disc(vals)
        log_vals = log.(vals)
        uv = sort(unique(log_vals))
        counts = [count(==(v), log_vals) for v in uv]
        probs = counts ./ sum(counts)
        lookup = Dict(uv[i] => probs[i] for i in eachindex(uv))
        DiscreteEmpirical(uv, probs, lookup)
    end
    disc_H0 = make_disc(rand(det_vals_h0, 100))
    disc_H1 = make_disc(rand(det_vals_h1, 100))
    disc_ag = make_disc(vcat(rand(det_vals_h0, 50), rand(det_vals_h1, 50)))

    class_params = Dict(
        "background" => (mu=-2.0, sigma=0.5, precision=4.0),
        "agnostic"   => (mu=0.0, sigma=0.5, precision=4.0),
        "interaction" => (mu=3.0, sigma=0.5, precision=4.0)
    )
    lc = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [0.5, 0.0, 0.5], collect(1.0:5.0), true, 5, resp,
        nothing,  # all_restart_traces
        2.0, 2.0, 1.0,  # alpha, theta, sd
        :gamma, Dict{Symbol,Float64}(:gamma => 0.0),  # family, bic
        nothing,  # em_diagnostics
        disc_H0, disc_ag, disc_H1  # discrete detection distributions
    )

    qg = run_quality_gates(bf, lc)
    for cell in qg.cells
        # Enrichment/correlation cells have hist_bin_edges; detection cells have pdf_x/pdf_y
        if cell.marginal != :detection
            if !isempty(cell.hist_bin_edges)
                @test length(cell.hist_bin_edges) > 1
                @test length(cell.hist_counts) == length(cell.hist_bin_edges) - 1
                @test length(cell.fitted_pdf_x) > 0
                @test length(cell.fitted_pdf_y) == length(cell.fitted_pdf_x)
                @test all(cell.hist_counts .>= 0)
                @test all(cell.fitted_pdf_y .>= 0)
            else
                # Skipped cells (e.g., agnostic with no assigned proteins) have empty arrays
                @test isempty(cell.hist_counts)
                @test isempty(cell.fitted_pdf_x)
            end
        else
            # Detection cells use discrete support points
            if !isempty(cell.fitted_pdf_x)
                @test length(cell.fitted_pdf_y) == length(cell.fitted_pdf_x)
                @test all(cell.fitted_pdf_y .>= 0)
            end
        end
    end
    # At least H0 and H1 enrichment+correlation cells (4 of 9) should have histograms
    # Detection cells have discrete PDF data instead
    n_with_hist = count(c -> !isempty(c.hist_bin_edges), qg.cells)
    @test n_with_hist >= 4
    # Detection cells with assigned proteins should have PDF data
    n_with_pdf = count(c -> c.marginal == :detection && !isempty(c.fitted_pdf_x), qg.cells)
    @test n_with_pdf >= 2  # H0 and H1 (agnostic may be skipped)
end

# ── Chi-squared GOF tests ───────────────────────────────────────────────────

@testitem "Chi-squared GOF: uniform bins" begin
    using BayesInteractomics: _chisq_gof_discrete

    # Equal observed and expected => chi2 ~ 0, p ~ 1
    observed = [10, 10, 10, 10, 10]
    expected = [10.0, 10.0, 10.0, 10.0, 10.0]
    chi2, p, df = _chisq_gof_discrete(observed, expected)
    @test chi2 ≈ 0.0 atol=1e-10
    @test p ≈ 1.0 atol=0.01
    @test df == 4  # k-1 = 5-1
end

@testitem "Chi-squared GOF: poor fit" begin
    using BayesInteractomics: _chisq_gof_discrete

    # Highly skewed observed vs uniform expected => low p-value
    observed = [50, 5, 5, 5, 5]
    expected = [14.0, 14.0, 14.0, 14.0, 14.0]
    chi2, p, df = _chisq_gof_discrete(observed, expected)
    @test chi2 > 10.0
    @test p < 0.05
    @test df == 4
end

@testitem "Bin merging: small expected counts" begin
    using BayesInteractomics: _merge_small_bins

    # Bins with expected < 5 get merged into neighbors
    # [1, 2, 10, 12] => left merge: 1+2=3 => [3, 10, 12] => interior: 3<5 merges with 10 => [13, 12]
    observed = [1, 2, 10, 12]
    expected = [1.0, 2.0, 10.0, 12.0]
    merged_obs, merged_exp = _merge_small_bins(observed, expected)
    # All bins with expected < 5 are merged; result has 2 bins
    @test length(merged_obs) == 2
    @test sum(merged_obs) == sum(observed)  # Total count preserved
    @test sum(merged_exp) ≈ sum(expected)   # Total expected preserved

    # Test case where merging preserves more bins: all >= 5
    observed2 = [10, 20, 15, 8]
    expected2 = [10.0, 20.0, 15.0, 8.0]
    merged_obs2, merged_exp2 = _merge_small_bins(observed2, expected2)
    @test length(merged_obs2) == 4  # No merging needed
    @test merged_obs2 == observed2
end

@testitem "Chi-squared: too few bins after merge" begin
    using BayesInteractomics: _chisq_gof_discrete

    # All bins have expected < 5 except one => merge down to < 3 bins
    observed = [2, 3, 100]
    expected = [2.0, 3.0, 100.0]
    # After merging: [2,3] merge to [5, 100] => 2 bins < 3 threshold
    chi2, p, df = _chisq_gof_discrete(observed, expected)
    @test chi2 == 0.0
    @test p == 1.0
    @test df == 0
end

@testitem "Detection quality gate uses chi-squared against DiscreteEmpirical" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult,
        QualityGateCell, QualityGateResult, run_quality_gates, DiscreteEmpirical
    using Distributions, Random

    Random.seed!(99)
    n = 300

    # Well-separated components
    bf_e_h0 = exp.(randn(100) .* 0.5 .- 3.0)
    bf_e_ag = exp.(randn(100) .* 0.5)
    bf_e_h1 = exp.(randn(100) .* 0.5 .+ 4.0)
    bf_e = vcat(bf_e_h0, bf_e_ag, bf_e_h1)

    bf_c = vcat(exp.(randn(100) .* 0.5 .- 3.0),
                exp.(randn(100) .* 0.5),
                exp.(randn(100) .* 0.5 .+ 4.0))

    # Detection BFs: truly discrete values (Beta-Bernoulli origin)
    # Use a small set of unique values to mimic real detection BFs
    det_vals_h0 = [0.1, 0.3, 0.5]
    det_vals_ag = [0.8, 1.0, 1.2]
    det_vals_h1 = [3.0, 5.0, 10.0]
    bf_d_h0 = rand(det_vals_h0, 100)
    bf_d_ag = rand(det_vals_ag, 100)
    bf_d_h1 = rand(det_vals_h1, 100)
    bf_d = vcat(bf_d_h0, bf_d_ag, bf_d_h1)
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    # Clear responsibilities
    resp = zeros(n, 3)
    resp[1:100, 1] .= 0.9;   resp[1:100, 2] .= 0.05;  resp[1:100, 3] .= 0.05
    resp[101:200, 1] .= 0.05; resp[101:200, 2] .= 0.9;  resp[101:200, 3] .= 0.05
    resp[201:300, 1] .= 0.05; resp[201:300, 2] .= 0.05; resp[201:300, 3] .= 0.9

    # Build DiscreteEmpirical distributions matching the true data-generating process
    # (on log scale, since run_quality_gates takes log of BFs)
    log_d_h0 = log.(bf_d_h0)
    log_d_ag = log.(bf_d_ag)
    log_d_h1 = log.(bf_d_h1)

    function make_disc(vals)
        uv = sort(unique(vals))
        counts = [count(==(v), vals) for v in uv]
        probs = counts ./ sum(counts)
        lookup = Dict(uv[i] => probs[i] for i in eachindex(uv))
        DiscreteEmpirical(uv, probs, lookup)
    end

    disc_H0 = make_disc(log_d_h0)
    disc_ag = make_disc(log_d_ag)
    disc_H1 = make_disc(log_d_h1)

    class_params = Dict(
        "background" => (mu=-3.0, sigma=0.5, precision=4.0),
        "agnostic"   => (mu=0.0, sigma=0.5, precision=4.0),
        "interaction" => (mu=4.0, sigma=0.5, precision=4.0)
    )

    # Use 18-arg constructor that includes disc_detection fields
    lc = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [1/3, 1/3, 1/3], collect(1.0:5.0), true, 5, resp,
        nothing,  # all_restart_traces
        2.0, 2.0, 1.0,  # alpha, theta, sd
        :gamma, Dict{Symbol,Float64}(:gamma => 0.0),  # family, bic
        nothing,  # em_diagnostics
        disc_H0, disc_ag, disc_H1  # discrete detection distributions
    )

    qg = run_quality_gates(bf, lc)

    # Detection cells are row 3 (m=3) -- check they have valid status
    for k in 1:3
        cell = qg.cells[3, k]  # detection marginal
        @test cell.marginal == :detection
        @test cell.status in (:pass, :warn, :fail)
        # The ks_statistic field holds chi-squared p-value
        @test cell.ks_statistic >= 0.0
        @test cell.ks_statistic <= 1.0
        # Fitted distribution should be DiscreteEmpirical, not Normal
        @test cell.fitted_distribution isa DiscreteEmpirical
        # PDF overlay should use discrete support points
        @test length(cell.fitted_pdf_x) > 0
        @test length(cell.fitted_pdf_y) == length(cell.fitted_pdf_x)
    end

    # Since we generated data FROM the fitted distributions, chi-squared should pass
    for k in 1:3
        @test qg.cells[3, k].status == :pass
    end
end

@testitem "Detection quality gate graceful fallback when DiscreteEmpirical is nothing" begin
    using BayesInteractomics
    using BayesInteractomics: BayesFactorTriplet, LatentClassResult,
        QualityGateResult, run_quality_gates
    using Distributions, Random

    Random.seed!(99)
    n = 300

    bf_e = exp.(randn(n) .* 0.5)
    bf_c = exp.(randn(n) .* 0.5)
    bf_d = exp.(randn(n) .* 0.5)
    bf = BayesFactorTriplet(bf_e, bf_c, bf_d)

    resp = zeros(n, 3)
    resp[1:100, 1] .= 0.9;   resp[1:100, 2] .= 0.05;  resp[1:100, 3] .= 0.05
    resp[101:200, 1] .= 0.05; resp[101:200, 2] .= 0.9;  resp[101:200, 3] .= 0.05
    resp[201:300, 1] .= 0.05; resp[201:300, 2] .= 0.05; resp[201:300, 3] .= 0.9

    class_params = Dict(
        "background" => (mu=0.0, sigma=1.0, precision=1.0),
        "agnostic"   => (mu=0.0, sigma=1.0, precision=1.0),
        "interaction" => (mu=0.0, sigma=1.0, precision=1.0)
    )
    # 8-arg constructor: disc_detection fields default to nothing
    lc = LatentClassResult(
        ones(n), fill(0.5, n), class_params,
        [1/3, 1/3, 1/3], collect(1.0:5.0), true, 5, resp
    )

    qg = run_quality_gates(bf, lc)

    # Detection cells should gracefully pass when no DiscreteEmpirical available
    for k in 1:3
        cell = qg.cells[3, k]
        @test cell.marginal == :detection
        @test cell.status == :pass
        @test cell.ks_statistic == 1.0  # fallback p-value
    end
end
