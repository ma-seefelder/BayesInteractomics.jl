# src/reports/json_utils.jl
# Minimal JSON serialization for report generation (no external dependencies).

"""
    json_number(x) -> String

Serialize a numeric value to a JSON literal.
Returns `"null"` for `missing`, `NaN`, and `Inf`.
"""
function json_number(x)::String
    (ismissing(x) || (x isa AbstractFloat && !isfinite(x))) && return "null"
    v = Float64(x)
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
    write(buf, '"')
    for c in String(s)
        if c == '"';      write(buf, "\\\"")
        elseif c == '\\'; write(buf, "\\\\")
        elseif c == '\n'; write(buf, "\\n")
        elseif c == '\r'; write(buf, "\\r")
        elseif c == '\t'; write(buf, "\\t")
        elseif c < '\x20'; write(buf, "\\u$(lpad(string(Int(c), base=16), 4, '0'))")
        else; write(buf, c)
        end
    end
    write(buf, '"')
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
