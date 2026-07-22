# scale-mismatch guard. `estimate_regression_tau_base` emits a one-time
# @warn (recommending normalise_protocols=true) when a MULTI-protocol analysis has an
# implausibly large pooled regression residual SD (sqrt(1/τ_base) > 2.5 log2), the
# hallmark of un-normalised cross-protocol/experiment intensity baselines that inflate
# the regression slope MEAN and saturate bf_correlation.
#
# maxlog=1 makes the @warn itself awkward to assert across a shared test process, so we
# test the discriminating SIGNAL the guard keys on (the residual SD via τ_base) plus the
# single-protocol no-fire branch.

@testitem "scale-mismatch guard: residual-SD signal discriminates" tags=[:mask_aware] begin
    using BayesInteractomics
    using BayesInteractomics: estimate_regression_tau_base, InteractionData, Protocol,
        getPositions, getNoProtocols
    using Random

    # Build an n_protocols-protocol InteractionData where protein p's cells either TRACK
    # the reference (protein 1) cells (matched → small residuals) or are independent
    # large-spread noise (mismatch → residual SD ≫ 2.5).
    function build(; n_protocols, mismatch::Bool, n_proteins=8, n_exp=2, n_rep=3, seed=123)
        Random.seed!(seed)
        protein_ids   = ["P$i" for i in 1:n_proteins]
        protein_names = ["Protein_$i" for i in 1:n_proteins]
        function mk()
            dd = Dict{Int, Matrix{Union{Missing, Float64}}}()
            for e in 1:n_exp
                m = Matrix{Union{Missing, Float64}}(undef, n_proteins, n_rep)
                for r in 1:n_rep
                    x = 10.0 + 3.0 * randn()          # reference (protein 1) cell value
                    m[1, r] = x
                    for p in 2:n_proteins
                        m[p, r] = mismatch ? (10.0 + 5.0 * randn()) :   # independent → big residuals
                                             (x + 0.3 * randn())          # tracks x → small residuals
                    end
                end
                dd[e] = m
            end
            Protocol(n_exp, protein_ids, dd)
        end
        samples_dict  = Dict(p => mk() for p in 1:n_protocols)
        controls_dict = Dict(p => mk() for p in 1:n_protocols)
        no_exp_dict = Dict(p => n_exp for p in 1:n_protocols)
        no_hbm = 1 + n_protocols + n_protocols * n_exp
        no_reg = 1 + n_protocols
        pp, ep, mp = getPositions(no_exp_dict, no_hbm)
        # 5th positional is `no_protocols` — must be n_protocols (the guard reads data.no_protocols).
        InteractionData(protein_ids, protein_names, samples_dict, controls_dict,
            n_protocols, no_exp_dict, no_hbm, no_reg, ep, pp, mp, trues(n_proteins))
    end

    # multi-protocol mismatch → residual SD large → guard condition TRUE
    d_mismatch = build(n_protocols=2, mismatch=true)
    @test getNoProtocols(d_mismatch) == 2
    τ_mismatch = estimate_regression_tau_base(d_mismatch, 1)
    res_sd_mismatch = sqrt(1.0 / τ_mismatch)
    @test res_sd_mismatch > 2.5            # guard fires here

    # multi-protocol matched → residual SD small → guard silent
    d_matched = build(n_protocols=2, mismatch=false)
    τ_matched = estimate_regression_tau_base(d_matched, 1)
    res_sd_matched = sqrt(1.0 / τ_matched)
    @test res_sd_matched < 2.5             # guard does NOT fire

    # single-protocol mismatch → guard gated off (getNoProtocols == 1), never fires
    d_single = build(n_protocols=1, mismatch=true)
    @test getNoProtocols(d_single) == 1
end
