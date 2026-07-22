#!/usr/bin/env julia
# TTFX measurement script
#
# Usage:
#     julia --project=. scripts/measure_ttfx.jl [--n 5]
#
# Default: 5 cold runs + 5 warm runs of `using BayesInteractomics`,
# reports median wall-clock seconds. Cold runs delete the BayesInteractomics
# precompile cache at ~/.julia/compiled/v<major>.<minor>/BayesInteractomics/
# before each run. Warm runs preserve the precompile cache (the package is
# already compiled; what we measure is the per-process load cost: __init__,
# extension trigger checks, eager `using` traversals).
#
# Each measurement spawns a fresh `julia` subprocess so parent-process state
# (already-loaded modules, hot caches) does NOT contaminate the measurement.
#
# Output emits a final CSV line "CSV: ..." that the verification step parses.

using Statistics
using SHA
using Dates

const N_DEFAULT = 5
const PRECOMPILE_DIR = joinpath(homedir(), ".julia", "compiled",
                                "v$(VERSION.major).$(VERSION.minor)",
                                "BayesInteractomics")

"""
    delete_precompile_cache()

Remove `~/.julia/compiled/v<major>.<minor>/BayesInteractomics/` so the next
`using BayesInteractomics` triggers a full precompile.

Wrapped in a 3-attempt retry loop with 1s sleep between attempts: Julia 1.12
on Windows can briefly hold open file handles in the precompile dir (AV scan,
filesystem flushing), and a single `rm(...; force=true)` can raise an IOError
even with `force=true`. The retry loop tolerates the transient.
"""
function delete_precompile_cache()
    if !isdir(PRECOMPILE_DIR)
        return
    end
    local last_err = nothing
    for attempt in 1:3
        try
            rm(PRECOMPILE_DIR; recursive=true, force=true)
            return
        catch e
            last_err = e
            if attempt < 3
                sleep(1)
            end
        end
    end
    @warn "Failed to delete precompile cache after 3 attempts" path=PRECOMPILE_DIR err=last_err
    rethrow(last_err)
end

"""
    measure_one_run(; cold::Bool)

Spawn a fresh `julia --project=<proj> --threads=4 -e "using BayesInteractomics"`
subprocess and return the wall-clock elapsed time (seconds). When `cold=true`,
the BayesInteractomics precompile cache is deleted first.
"""
function measure_one_run(; cold::Bool)
    proj = dirname(@__DIR__)
    if cold
        delete_precompile_cache()
    end
    t0 = time_ns()
    # --threads=4 explicitly (--threads=auto can segfault on
    # Julia 1.12 / Windows). Stderr captured so precompile log lines don't
    # spam the console.
    run(pipeline(`julia --project=$proj --threads=4 -e "using BayesInteractomics"`;
                 stdout=devnull, stderr=devnull))
    t1 = time_ns()
    return (t1 - t0) / 1e9   # seconds
end

"""
    measure_cohort(; cold::Bool, n::Int=N_DEFAULT)

Run `n` measurements of `measure_one_run(; cold=cold)` and return the raw
vector of times (seconds).
"""
function measure_cohort(; cold::Bool, n::Int=N_DEFAULT)
    times = Float64[]
    for i in 1:n
        t = measure_one_run(; cold=cold)
        push!(times, t)
        println("  run $i/$n: $(round(t, digits=2)) s")
        flush(stdout)
    end
    return times
end

"""
    project_sha()

Return the SHA-256 hex digest of the active Project.toml. Used to verify
the dependency-graph state is identical between baseline and HEAD runs
(determinism guard).
"""
function project_sha()
    bytes = Base.read(joinpath(dirname(@__DIR__), "Project.toml"))
    return bytes2hex(sha256(bytes))
end

function main()
    n = N_DEFAULT
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--n" && i < length(ARGS)
            n = parse(Int, ARGS[i+1])
            i += 2
        else
            i += 1
        end
    end

    proj_sha = project_sha()
    timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")
    println("=" ^ 70)
    println("TTFX measurement")
    println("Date:           $timestamp")
    println("Julia version:  $VERSION")
    println("Project SHA256: $proj_sha")
    println("N per cohort:   $n")
    println("Precompile dir: $PRECOMPILE_DIR")
    println("=" ^ 70)

    println("\nCOLD runs (precompile cache cleared before each)…")
    cold_times = measure_cohort(; cold=true, n=n)
    cold_median = median(cold_times)

    println("\nWARM runs (precompile cache preserved)…")
    warm_times = measure_cohort(; cold=false, n=n)
    warm_median = median(warm_times)

    println("\n" * "=" ^ 70)
    println("RESULTS")
    println("Cold median: $(round(cold_median, digits=2)) s   (raw: $(round.(cold_times, digits=2)))")
    println("Warm median: $(round(warm_median, digits=2)) s   (raw: $(round.(warm_times, digits=2)))")
    println("=" ^ 70)

    # Machine-readable summary lines for downstream parsing by the
    # 67.1-VERIFICATION.md edit step.
    println("\nCSV: timestamp,julia_version,project_sha,cold_median_s,warm_median_s,n")
    println("CSV: $timestamp,$VERSION,$proj_sha,$cold_median,$warm_median,$n")
    println("COLD_RAW: $(join(cold_times, ','))")
    println("WARM_RAW: $(join(warm_times, ','))")
end

main()
