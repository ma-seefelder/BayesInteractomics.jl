"""
    Input data quality control orchestrator.

Runs all QC checks with error isolation so that one failing check
cannot prevent others from running (T-50-05 mitigation).
"""

"""
    run_input_qc(data::InteractionData) -> InputQCResult

Run all input data quality checks and return aggregated results.

Each check is wrapped in its own try/catch block so that a failure in one
check does not prevent the others from running. Failed checks produce a
`@warn` message and contribute `nothing` to the result.

If all checks fail, overall_flag defaults to `:warning`.
"""
function run_input_qc(data::InteractionData)::InputQCResult
    # Run each check with error isolation (T-50-05)
    scale_result = try
        check_scale(data)
    catch e
        @warn "[QC] Scale check failed" exception = (e, catch_backtrace())
        nothing
    end

    corr_result = try
        check_replicate_correlation(data)
    catch e
        @warn "[QC] Replicate correlation check failed" exception = (e, catch_backtrace())
        nothing
    end

    miss_result = try
        check_missingness(data)
    catch e
        @warn "[QC] Missingness check failed" exception = (e, catch_backtrace())
        nothing
    end

    shape_result = try
        check_intensity_shape(data)
    catch e
        @warn "[QC] Intensity shape check failed" exception = (e, catch_backtrace())
        nothing
    end

    # PCA separation analysis
    pca_result = try
        run_pca_separation(data)
    catch e
        @warn "[QC] PCA separation analysis failed" exception = (e, catch_backtrace())
        nothing
    end

    # Compute overall flag from non-nothing results
    result_flags = Symbol[]
    for r in (scale_result, corr_result, miss_result, shape_result, pca_result)
        if !isnothing(r)
            push!(result_flags, r.flag)
        end
    end

    overall_flag = isempty(result_flags) ? :warning : worst_flag(result_flags...)

    return InputQCResult(scale_result, corr_result, miss_result, shape_result, pca_result, overall_flag)
end
