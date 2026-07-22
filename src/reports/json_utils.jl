# src/reports/json_utils.jl
# Minimal JSON serialization for report generation (no external dependencies).
#
# note: the report payload `variance_recovery` block is assembled
# in `src/reports/report_generator.jl::_build_report_json` using the helpers in
# this file (`json_object`, `json_string`, `json_number`, `json_array`). The
# `variance_recovery` top-level key carries:
#   - mode ::String  ("off" | "inflation" | "multi_impute" | "unknown")
#   - m ::Union{Int, null}  (Int when mode == "multi_impute"; null otherwise)
#   - seeds ::Union{Vector{Int}, null}  (base_seed * 1_000_003 + i for i ∈ 1..m)
#   - inflation_max ::Union{Float64, null}
#   - inflation_override ::Union{Float64, null}
#   - block_html ::String  (rendered HTML fragment from
#     methods_generator.jl::_methods_variance_recovery_block; empty for :off)
# See `_build_report_json` for the producer side and the templates'
# `#methods-variance-recovery-card` div for the consumer side.

"""
    json_number(x) -> String

Serialize a numeric value to a JSON literal.
Returns `"null"` for `missing`, `NaN`, and `Inf`.
"""
function json_number(x)::String
    (ismissing(x) || (x isa AbstractFloat && !isfinite(x))) && return "null"
    v = Float64(x)
    v = round(v; sigdigits=5)
    # Emit integer literal when the value is a whole number
    if isinteger(v) && abs(v) < 9.007199254740992e15   # safe Int64 range
        return string(Int64(v))
    end
    return string(v)
end
json_number(x::Integer) = string(x)

"""
    json_string(s) -> String

Serialize a string to a JSON string literal with proper escaping.
"""
function json_string(s::AbstractString)::String
    buf = IOBuffer()
    Base.write(buf, '"')
    for c in String(s)
        if c == '"';      Base.write(buf, "\\\"")
        elseif c == '\\'; Base.write(buf, "\\\\")
        elseif c == '\n'; Base.write(buf, "\\n")
        elseif c == '\r'; Base.write(buf, "\\r")
        elseif c == '\t'; Base.write(buf, "\\t")
        elseif c < '\x20'; Base.write(buf, "\\u$(lpad(string(Int(c), base=16), 4, '0'))")
        else; Base.write(buf, c)
        end
    end
    Base.write(buf, '"')
    return String(take!(buf))
end
json_string(x) = json_string(string(x))

"""
    json_bool(b) -> String

Serialize a Bool to a JSON literal.
"""
json_bool(b::Bool) = b ? "true" : "false"

"""
    json_array(items) -> String

Serialize a vector of already-serialized JSON strings to a JSON array.
"""
function json_array(items::AbstractVector{<:AbstractString})::String
    return "[" * join(items, ",") * "]"
end

"""
    json_object(pairs...) -> String

Build a JSON object from alternating (key, already-serialized-value) pairs.
Keys are auto-quoted; values must already be valid JSON strings.
"""
function json_object(pairs::Pair{<:AbstractString, <:AbstractString}...)::String
    parts = [json_string(k) * ":" * v for (k, v) in pairs]
    return "{" * join(parts, ",") * "}"
end

"""
    json_number_nan_safe(x) -> String

named alias for json_number(x) -- exposes the NaN/missing/Inf -> null
behaviour explicitly at call-sites that construct per-row Decision Risk JSON.
This is a thin wrapper for self-documenting call-sites; behaviour is identical
to json_number (above).
"""
json_number_nan_safe(x) = json_number(x)

"""
    json_symbol_or_string(x) -> String

named alias for json_string(x) -- Symbol input is auto-stringified
via the fallback json_string(x) = json_string(string(x)) (above). Used at
per-row JSON construction sites for optimal_call::Symbol and
optimal_call_min::Symbol columns.
"""
json_symbol_or_string(x) = json_string(x)

"""
    encode_png_file(filepath) -> String

Read a PNG/image file and return a `data:image/png;base64,...` URI string
suitable for embedding in HTML.  Returns `""` if the file does not exist.
"""
function encode_png_file(filepath::AbstractString)::String
    isfile(filepath) || return ""
    ext = lowercase(splitext(filepath)[2])
    mime = ext == ".svg" ? "image/svg+xml" : "image/png"
    encoded = Base64.base64encode(Base.read(filepath))
    return "data:$(mime);base64,$(encoded)"
end
