# test/static/test_no_extra_compute_embeddings_callsites.jl
# Static guard against drift in the number of
# `_compute_embeddings` call sites under src/. The fix at the imputed-vector
# `analyse()` branch relies on there being exactly two call sites
# (single-data + imputed-vector). A future change that legitimately adds a
# third call site must update this test in lock-step.
#
# The regex-based docstring tracker below is BEST-EFFORT.
# Known limitations:
#   1. Does NOT handle `raw"""..."""` string blocks — triple quotes inside a
#      raw string are still counted by the regex.
#   2. Does NOT handle escaped triple quotes `\"\"\"` inside non-raw strings.
#   3. A single-line docstring of the form `"""... _compute_embeddings(...) ..."""`
#      would be (incorrectly) counted as a call site because the even-count
#      branch falls through to the `occursin` check on the SAME line.
# The current src/ tree has none of these patterns, so the test passes today.
# Any future RED on this test caused by edits to docstrings (rather than real
# code) MUST be triaged before adjusting the count contract — verify by hand
# that the additional callsite entry is in fact a docstring and not a real
# call. A proper AST-based replacement (JuliaSyntax) is deferred.

@testitem "exactly two _compute_embeddings call sites in src/ (both in analysis/pipeline.jl)" begin
    using BayesInteractomics

    src_root = joinpath(pkgdir(BayesInteractomics), "src")
    @assert isdir(src_root) "Could not locate src/ under $(pkgdir(BayesInteractomics))"

    callsites = String[]
    for (root, _dirs, files) in walkdir(src_root)
        for f in files
            endswith(f, ".jl") || continue
            path = joinpath(root, f)
            in_docstring = false
            for (lineno, line) in enumerate(eachline(path))
                # Track triple-quoted docstring blocks; skip lines inside them.
                # Count un-escaped triple quotes on the line; an odd count toggles state.
                triple_quote_count = length(collect(eachmatch(r"\"\"\"", line)))
                line_has_triple_quote = triple_quote_count > 0
                # If we're already inside a docstring, skip the line entirely
                # (even if it contains _compute_embeddings(); these are docstring signatures).
                if in_docstring
                    if isodd(triple_quote_count)
                        in_docstring = false
                    end
                    continue
                end
                # If a docstring opens on this line, decide if it also closes on the same line.
                if line_has_triple_quote && !in_docstring
                    if isodd(triple_quote_count)
                        in_docstring = true
                        continue
                    end
                    # Even count on a single line — docstring opened and closed; treat line as code.
                end
                # Filter out the function definition line: only lines that contain
                # `_compute_embeddings(` AND do NOT start with `function ` (ignoring leading whitespace).
                occursin(r"_compute_embeddings\(", line) || continue
                stripped = lstrip(line)
                startswith(stripped, "function ") && continue
                push!(callsites, "$(relpath(path, src_root)):$(lineno)")
            end
        end
    end

    # Exactly two call sites expected.
    @test length(callsites) == 2
    # Both must reside in src/analysis/pipeline.jl.
    @test all(s -> startswith(s, joinpath("analysis", "pipeline.jl")) || startswith(s, "analysis/pipeline.jl"), callsites)
end
