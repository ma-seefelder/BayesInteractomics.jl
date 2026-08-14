# ============================================================================
# macro.jl — @interactomics DSL for simplified BayesInteractomics workflows
# ============================================================================
#
# Submacros (inside @interactomics):
#   @protocol   — data file + column layout
#   @experiment — per-experiment columns (inside @protocol)
#   @condition  — named analysis condition (for differential)
#   @compare    — trigger differential between conditions
#
# Standalone:
#   @protocol   — returns (file, sample_cols, control_cols, n_samples, n_controls)
# ============================================================================

# ── Runtime helpers (called by macro-generated code) ─────────────────────────

function _take_dummy(dummy::Vector{Int}, n::Int)
    isempty(dummy) && throw(ArgumentError(
        "dummy columns required for protocol padding; add dummy=[col,...] to @protocol"))
    return [dummy[mod1(i, length(dummy))] for i in 1:n]
end

function _pad_protocol_dicts!(
    sd::Vector{Dict{Int,Vector{Int}}},
    cd::Vector{Dict{Int,Vector{Int}}},
    dummy::Vector{Int}
)
    isempty(sd) && return (sd, cd)
    max_exp = maximum(
        maximum(keys(d); init=0) for d in Iterators.flatten((sd, cd))
    )
    max_exp == 0 && return (sd, cd)

    function _max_width(dicts, exp)
        w = 0
        for d in dicts
            haskey(d, exp) && (w = max(w, length(d[exp])))
        end
        return w > 0 ? w : length(dummy)
    end

    # Pre-compute target widths before any mutation
    s_widths = Dict(exp => _max_width(sd, exp) for exp in 1:max_exp)
    c_widths = Dict(exp => _max_width(cd, exp) for exp in 1:max_exp)

    for i in eachindex(sd), exp in 1:max_exp
        # Fill missing experiments
        if !haskey(sd[i], exp)
            sd[i][exp] = _take_dummy(dummy, s_widths[exp])
        elseif length(sd[i][exp]) < s_widths[exp]
            append!(sd[i][exp], _take_dummy(dummy, s_widths[exp] - length(sd[i][exp])))
        end
        if !haskey(cd[i], exp)
            cd[i][exp] = _take_dummy(dummy, c_widths[exp])
        elseif length(cd[i][exp]) < c_widths[exp]
            append!(cd[i][exp], _take_dummy(dummy, c_widths[exp] - length(cd[i][exp])))
        end
    end
    return (sd, cd)
end

function _count_real_replicates(dicts::Vector{Dict{Int,Vector{Int}}}, dummy_set::Set{Int})
    n = 0
    for d in dicts, (_, cols) in d
        n += count(c -> c ∉ dummy_set, cols)
    end
    return n
end

# ── AST helpers (macro-expansion time) ───────────────────────────────────────

function _mc_name(ex)
    ex isa Expr && ex.head === :macrocall && !isempty(ex.args) || return nothing
    s = ex.args[1]
    return s isa GlobalRef ? s.name : (s isa Symbol ? s : nothing)
end

_is_mc(ex, name::Symbol) = _mc_name(ex) === Symbol("@", name)

function _mc_args(ex)
    filter(a -> !(a isa LineNumberNode), ex.args[2:end])
end

# ── Internal parsed representations ──────────────────────────────────────────

struct _Exp
    samples::Any
    controls::Any
end

struct _Proto
    file::Any
    exps::Vector{_Exp}
    dummy::Any
end

struct _Cond
    name::Symbol
    protos::Vector{_Proto}
    kws::Vector{Pair{Symbol,Any}}
end

struct _Cmp
    names::Vector{Symbol}
    kws::Vector{Pair{Symbol,Any}}
end

# ── Parsing ──────────────────────────────────────────────────────────────────

function _parse_experiment_args(args)
    s = c = nothing
    for a in args
        a isa Expr && a.head === :(=) || continue
        k = a.args[1]
        k === :samples  && (s = a.args[2]; continue)
        k === :controls && (c = a.args[2]; continue)
        throw(ArgumentError("@experiment: unknown keyword :$k (expected: samples, controls)"))
    end
    isnothing(s) && throw(ArgumentError("@experiment: samples=... required"))
    isnothing(c) && throw(ArgumentError("@experiment: controls=... required"))
    return _Exp(s, c)
end

function _parse_protocol(ex)
    args = _mc_args(ex)
    file = dummy = block = s_kw = c_kw = nothing

    for a in args
        if a isa Expr && a.head === :block
            block = a
        elseif a isa Expr && a.head === :(=)
            k, v = a.args[1], a.args[2]
            k === :samples  && (s_kw = v; continue)
            k === :controls && (c_kw = v; continue)
            k === :dummy    && (dummy = v; continue)
            throw(ArgumentError("@protocol: unknown keyword :$k (expected: samples, controls, dummy)"))
        else
            isnothing(file) ? (file = a) :
                throw(ArgumentError("@protocol: unexpected argument: $a"))
        end
    end
    isnothing(file) && throw(ArgumentError("@protocol: file path required"))

    if !isnothing(block)
        exps = _Exp[]
        for st in block.args
            st isa LineNumberNode && continue
            _is_mc(st, :experiment) || throw(ArgumentError(
                "@protocol block: expected @experiment, got: $st"))
            push!(exps, _parse_experiment_args(_mc_args(st)))
        end
        isempty(exps) && throw(ArgumentError("@protocol: at least one @experiment required"))
        return _Proto(file, exps, dummy)
    elseif !isnothing(s_kw) && !isnothing(c_kw)
        return _Proto(file, [_Exp(s_kw, c_kw)], dummy)
    else
        throw(ArgumentError(
            "@protocol: provide (samples=..., controls=...) or begin @experiment ... end"))
    end
end

function _parse_condition(ex)
    args = _mc_args(ex)
    length(args) >= 2 || throw(ArgumentError("@condition: need name + begin...end block"))
    name = args[1]
    name isa Symbol || throw(ArgumentError("@condition: name must be a symbol, got: $name"))
    blk = args[end]
    blk isa Expr && blk.head === :block ||
        throw(ArgumentError("@condition $name: expected begin...end block"))

    protos = _Proto[]
    kws = Pair{Symbol,Any}[]
    for st in blk.args
        st isa LineNumberNode && continue
        if _is_mc(st, :protocol)
            push!(protos, _parse_protocol(st))
        elseif st isa Expr && st.head === :(=) && st.args[1] isa Symbol
            push!(kws, st.args[1] => st.args[2])
        else
            throw(ArgumentError("@condition $name: unexpected: $st"))
        end
    end
    isempty(protos) && throw(ArgumentError("@condition $name: at least one @protocol required"))
    return _Cond(name, protos, kws)
end

function _parse_compare(ex)
    args = _mc_args(ex)
    names = Symbol[]
    kws = Pair{Symbol,Any}[]
    for a in args
        if a isa Symbol
            push!(names, a)
        elseif a isa Expr && a.head === :(=)
            push!(kws, a.args[1] => a.args[2])
        end
    end
    length(names) >= 2 || throw(ArgumentError("@compare: at least two condition names required"))
    return _Cmp(names, kws)
end

# ── Code generation ──────────────────────────────────────────────────────────

const _BI_MOD = @__MODULE__

const _KW_ALIAS = Dict{Symbol,Symbol}(
    :bait   => :poi,
    :method => :combination_method,
    :bait_id => :refID,
)

function _proto_dict_expr(exps::Vector{_Exp}, field::Symbol)
    pairs = Any[]
    for (i, e) in enumerate(exps)
        val = field === :samples ? e.samples : e.controls
        push!(pairs, :($(i) => collect(Int, $(val))))
    end
    return :(Dict{Int,Vector{Int}}($(pairs...)))
end

function _gen_config_stmts(protos::Vector{_Proto}, kws::Vector{Pair{Symbol,Any}})
    output_expr = nothing
    image_ext = QuoteNode(".png")
    bait = bait_id = imp_data = raw_data = global_dummy = nothing
    config_kws = Pair{Symbol,Any}[]

    for (k, v) in kws
        if     k === :output;       output_expr = v
        elseif k === :image_ext;    image_ext = v
        elseif k === :bait;         bait = v
        elseif k === :bait_id;      bait_id = v
        elseif k === :imputed_data; imp_data = v
        elseif k === :raw_data;     raw_data = v
        elseif k === :dummy;        global_dummy = v
        else   push!(config_kws, get(_KW_ALIAS, k, k) => v)
        end
    end
    isnothing(bait) && throw(ArgumentError("@interactomics: 'bait' is required"))

    stmts = Any[]

    # Files
    file_exprs = [p.file for p in protos]
    push!(stmts, :(_bi_files = String[$(file_exprs...)]))

    # Column dicts
    s_exprs = [_proto_dict_expr(p.exps, :samples) for p in protos]
    c_exprs = [_proto_dict_expr(p.exps, :controls) for p in protos]
    push!(stmts, :(_bi_scols = Dict{Int,Vector{Int}}[$(s_exprs...)]))
    push!(stmts, :(_bi_ccols = Dict{Int,Vector{Int}}[$(c_exprs...)]))

    # Padding + replicate counting
    has_dummy = !isnothing(global_dummy) || any(p -> !isnothing(p.dummy), protos)
    if has_dummy
        dummy_expr = !isnothing(global_dummy) ? global_dummy :
            first(p.dummy for p in protos if !isnothing(p.dummy))
        pad_ref = GlobalRef(_BI_MOD, :_pad_protocol_dicts!)
        cnt_ref = GlobalRef(_BI_MOD, :_count_real_replicates)
        push!(stmts, :(_bi_dummy = collect(Int, $(dummy_expr))))
        push!(stmts, :($(pad_ref)(_bi_scols, _bi_ccols, _bi_dummy)))
        push!(stmts, :(_bi_ns = $(cnt_ref)(_bi_scols, Set(_bi_dummy))))
        push!(stmts, :(_bi_nc = $(cnt_ref)(_bi_ccols, Set(_bi_dummy))))
    else
        push!(stmts, :(_bi_ns = sum(sum(length(v) for v in values(d)) for d in _bi_scols)))
        push!(stmts, :(_bi_nc = sum(sum(length(v) for v in values(d)) for d in _bi_ccols)))
    end

    # Output
    OF = GlobalRef(_BI_MOD, :OutputFiles)
    if !isnothing(output_expr)
        push!(stmts, :(_bi_out_raw = $(output_expr)))
        push!(stmts, :(_bi_output = _bi_out_raw isa $(OF) ?
                        _bi_out_raw : $(OF)(string(_bi_out_raw); image_ext=$(image_ext))))
    else
        push!(stmts, :(_bi_output = $(OF)(".")))
    end

    # CONFIG constructor kwargs
    cfg_kw = Any[
        Expr(:kw, :datafile,     :_bi_files),
        Expr(:kw, :sample_cols,  :_bi_scols),
        Expr(:kw, :control_cols, :_bi_ccols),
        Expr(:kw, :poi,          bait),
        Expr(:kw, :n_samples,    :_bi_ns),
        Expr(:kw, :n_controls,   :_bi_nc),
        Expr(:kw, :output,       :_bi_output),
    ]
    !isnothing(bait_id) && push!(cfg_kw, Expr(:kw, :refID, bait_id))
    for (k, v) in config_kws
        push!(cfg_kw, Expr(:kw, k, v))
    end

    CFG = GlobalRef(_BI_MOD, :CONFIG)
    push!(stmts, :(_bi_config = $(Expr(:call, CFG, Expr(:parameters, cfg_kw...)))))

    return (stmts=stmts, imp=imp_data, raw=raw_data)
end

function _gen_single(protos, kws)
    r = _gen_config_stmts(protos, kws)
    RA = GlobalRef(_BI_MOD, :run_analysis)

    if !isnothing(r.imp) && !isnothing(r.raw)
        push!(r.stmts, :($(RA)(_bi_config, $(r.imp), $(r.raw))))
    else
        push!(r.stmts, :($(RA)(_bi_config)))
    end
    return Expr(:let, Expr(:block), Expr(:block, r.stmts...))
end

function _gen_differential(conds, cmp, global_kws)
    stmts = Any[]
    cfg_syms = Symbol[]

    for cond in conds
        merged = copy(global_kws)
        for (k, v) in cond.kws
            idx = findfirst(p -> p.first === k, merged)
            isnothing(idx) ? push!(merged, k => v) : (merged[idx] = k => v)
        end

        r = _gen_config_stmts(cond.protos, merged)
        cfg_sym = Symbol("_bi_cfg_", cond.name)
        push!(cfg_syms, cfg_sym)

        let_body = Expr(:block, r.stmts..., :(_bi_config))
        push!(stmts, :($cfg_sym = $(Expr(:let, Expr(:block), let_body))))
    end

    DA = GlobalRef(_BI_MOD, :differential_analysis)

    if length(conds) == 2
        diff_kw = Any[
            Expr(:kw, :condition_A, string(conds[1].name)),
            Expr(:kw, :condition_B, string(conds[2].name)),
        ]
        !isnothing(cmp) && append!(diff_kw, [Expr(:kw, k, v) for (k, v) in cmp.kws])
        push!(stmts, Expr(:call, DA, Expr(:parameters, diff_kw...), cfg_syms[1], cfg_syms[2]))
    else
        nt_pairs = [Expr(:(=), c.name, s) for (c, s) in zip(conds, cfg_syms)]
        diff_kw = Any[Expr(:kw, :conditions, Expr(:tuple, nt_pairs...))]
        !isnothing(cmp) && append!(diff_kw, [Expr(:kw, k, v) for (k, v) in cmp.kws])
        push!(stmts, Expr(:call, DA, Expr(:parameters, diff_kw...)))
    end

    return Expr(:let, Expr(:block), Expr(:block, stmts...))
end

# ── Macro definitions ────────────────────────────────────────────────────────

"""
    @protocol "file.xlsx" samples=[cols] controls=[cols]
    @protocol "file.xlsx" begin
        @experiment samples=[...] controls=[...]
        ...
    end

Define a protocol's data file and column layout.

Returns a `NamedTuple` with fields `file`, `sample_cols`, `control_cols`,
`n_samples`, `n_controls`.

Columns accept arrays (`[3,4,5]`) or ranges (`3:5`).

# Single experiment
```julia
p = @protocol "data.xlsx" samples=3:5 controls=6:8
```

# Multiple experiments
```julia
p = @protocol "data.xlsx" begin
    @experiment samples=[8,9,10] controls=[2,3,4]
    @experiment samples=[11,12,13] controls=[5]
    @experiment samples=[14,15] controls=[6,7]
end
```
"""
macro protocol(args...)
    mc = Expr(:macrocall, Symbol("@protocol"), LineNumberNode(0, :none), args...)
    p = _parse_protocol(mc)

    s_expr = _proto_dict_expr(p.exps, :samples)
    c_expr = _proto_dict_expr(p.exps, :controls)

    result = Expr(:let, Expr(:block), Expr(:block,
        :(_bi_s = Dict{Int,Vector{Int}}[$(s_expr)]),
        :(_bi_c = Dict{Int,Vector{Int}}[$(c_expr)]),
        :(_bi_ns = sum(sum(length(v) for v in values(d)) for d in _bi_s)),
        :(_bi_nc = sum(sum(length(v) for v in values(d)) for d in _bi_c)),
        :((file=$(p.file), sample_cols=_bi_s, control_cols=_bi_c,
           n_samples=_bi_ns, n_controls=_bi_nc)),
    ))
    return esc(result)
end

"""
    @interactomics begin ... end

Simplified DSL for BayesInteractomics analysis pipelines.

# Submacros

- `@protocol "file" samples=... controls=...` — single-experiment protocol
- `@protocol "file" begin @experiment ... end` — multi-experiment protocol
- `@experiment samples=... controls=...` — experiment definition (inside `@protocol`)
- `@condition name begin ... end` — named analysis condition (for differential)
- `@compare A B [kw=val ...]` — trigger differential analysis

# Keywords

Any `CONFIG` field can be set as `key = value`. Aliases:

| Alias      | CONFIG field           |
|------------|------------------------|
| `bait`     | `poi`                  |
| `method`   | `combination_method`   |
| `bait_id`  | `refID`                |

Special keywords: `output` (directory path or `OutputFiles`), `image_ext`
(default `".png"`), `imputed_data` / `raw_data` (for pre-loaded imputation).

# Examples

## Single analysis
```julia
results, ar = @interactomics begin
    @protocol "data.xlsx" samples=[3,4,5] controls=[6,7,8]
    bait = "HTT"
    output = "./results"
    method = :bma
end
```

## Multi-experiment protocol
```julia
results, ar = @interactomics begin
    @protocol "data.xlsx" begin
        @experiment samples=[8,9,10] controls=[2,3,4]
        @experiment samples=[11,12,13] controls=[5]
        @experiment samples=[14,15] controls=[6,7]
    end
    bait = "HAP40"
    output = "./results"
end
```

## Multi-protocol meta-analysis with dummy padding
```julia
results, ar = @interactomics begin
    @protocol "dataset.xlsx" dummy=[162,163,164,165] begin
        @experiment samples=[2,3,4] controls=[14,15,16]
        @experiment samples=[5,6,7] controls=[17,18,19]
    end
    @protocol "dataset.xlsx" dummy=[162,163,164,165] begin
        @experiment samples=[29,30,31] controls=[26,27,28]
    end
    bait = "HTT"
    bait_id = 237
    output = "./results"
end
```

## Differential analysis (2 conditions)
```julia
(; diff, result_A, result_B) = @interactomics begin
    @condition A begin
        @protocol "cond_a.xlsx" samples=[3,4,5] controls=[6,7,8]
        bait = "HTT"
        output = "./results/A"
    end
    @condition B begin
        @protocol "cond_b.xlsx" samples=[3,4,5] controls=[6,7,8]
        bait = "HTT"
        output = "./results/B"
    end
    @compare A B config=DifferentialConfig(results_file="diff.xlsx")
end
```

## k-group differential
```julia
result = @interactomics begin
    @condition wt begin
        @protocol "wt.xlsx" samples=3:5 controls=6:8
        bait = "HTT"
        output = "./results/wt"
    end
    @condition mut1 begin
        @protocol "mut1.xlsx" samples=3:5 controls=6:8
        bait = "HTT"
        output = "./results/mut1"
    end
    @condition mut2 begin
        @protocol "mut2.xlsx" samples=3:5 controls=6:8
        bait = "HTT"
        output = "./results/mut2"
    end
    @compare wt mut1 mut2 contrasts=:all_pairs
end
```

## Pre-loaded imputed data
```julia
results, ar = @interactomics begin
    @protocol "data.xlsx" samples=3:5 controls=6:8
    bait = "HTT"
    output = "./results"
    imputed_data = my_imputed_vec
    raw_data = my_raw
end
```
"""
macro interactomics(block)
    block isa Expr && block.head === :block ||
        throw(ArgumentError("@interactomics requires a begin...end block"))

    protos = _Proto[]
    conds  = _Cond[]
    cmp    = nothing
    kws    = Pair{Symbol,Any}[]

    for st in block.args
        st isa LineNumberNode && continue
        if _is_mc(st, :protocol)
            push!(protos, _parse_protocol(st))
        elseif _is_mc(st, :condition)
            push!(conds, _parse_condition(st))
        elseif _is_mc(st, :compare)
            cmp = _parse_compare(st)
        elseif st isa Expr && st.head === :(=) && st.args[1] isa Symbol
            push!(kws, st.args[1] => st.args[2])
        else
            throw(ArgumentError("@interactomics: unexpected expression: $st"))
        end
    end

    if !isempty(conds)
        !isempty(protos) && throw(ArgumentError(
            "@interactomics: use @protocol inside @condition for differential analysis"))
        return esc(_gen_differential(conds, cmp, kws))
    elseif !isempty(protos)
        return esc(_gen_single(protos, kws))
    else
        throw(ArgumentError("@interactomics: at least one @protocol or @condition required"))
    end
end
