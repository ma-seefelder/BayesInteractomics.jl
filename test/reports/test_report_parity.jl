# Parity audit gate.
#
# Reads both report templates and regex-parses <li class="nav-item" id="tab-X-li"> patterns
# to build tab-id sets S_single and S_diff. Asserts every single-bait tab without a
# differential counterpart is either present in differential OR explicitly allowlisted
# with a documented reason. Reverse check (informational only) lists differential-only
# tabs.
#
# Updates to PARITY_ALLOWLIST MUST be accompanied by a comment explaining why the
# single-bait tab has no differential counterpart.

@testitem "report parity audit" begin
    using BayesInteractomics  # for pkgdir

    # ── Allowlist: single-bait tabs intentionally absent from differential ──
    const PARITY_ALLOWLIST = Dict(
        "tab-dock-li" => "docking is bait-level / condition-invariant; differential report points to single-bait reports via footer note",
        "tab-diff-li" => "differential viewing itself is meaningless inside a differential report",
    )

    template_dir = joinpath(pkgdir(BayesInteractomics), "src", "reports", "templates")
    single_path  = joinpath(template_dir, "report.html")
    diff_path    = joinpath(template_dir, "differential_report.html")
    @assert isfile(single_path) "Missing template: $single_path"
    @assert isfile(diff_path)   "Missing template: $diff_path"

    single_html = read(single_path, String)
    diff_html   = read(diff_path,   String)

    # Stable pattern across both templates: <li class="nav-item" ... id="tab-X-li" ...>
    # \s+ tolerates multi-whitespace; [^>]* tolerates attribute reordering.
    pattern = r"<li\s+class=\"nav-item\"[^>]*id=\"(tab-[\w-]+-li)\""

    s_single = Set{String}(m.captures[1] for m in eachmatch(pattern, single_html))
    s_diff   = Set{String}(m.captures[1] for m in eachmatch(pattern, diff_html))

    # Sanity: regex didn't silently break (Pitfall 7)
    @test !isempty(s_single)   # currently 6 ids in single-bait
    @test !isempty(s_diff)     # currently 6+

    # Forward direction (the parity contract):
    # every single-bait tab without a differential counterpart MUST be allowlisted.
    missing_in_diff = setdiff(s_single, s_diff)
    not_allowlisted = setdiff(missing_in_diff, keys(PARITY_ALLOWLIST))

    if !isempty(not_allowlisted)
        for id in sort(collect(not_allowlisted))
            @info "single-bait template has tab '$id' with no counterpart in differential and no allowlist entry — add to differential_report.html or update PARITY_ALLOWLIST in test_report_parity.jl"
        end
    end
    @test isempty(not_allowlisted)

    # Reverse direction (informational only): differential-only tabs are intentional.
    diff_only = setdiff(s_diff, s_single)
    if !isempty(diff_only)
        diff_only_str = join(sort(collect(diff_only)), ", ")
        @info "Differential-only tabs (intentional): $(diff_only_str)"
    end
end
