_map(f, x...) = map(f, x...)
_map(f, x::Dict, ys...) = Dict(k => f(v, (y[k] for y in ys)...) for (k, v) in x)

function children_(x::T) where T
    fn = fieldnames(T)
    if length(fn) !=0 && fn[1] isa Symbol
        return NamedTuple((f,getfield(x, f)) for f in fn)
    else
        return Functors.children(x)
    end
end

function update_cache!(cx, Δx, x)
    if cx.cache === nothing
        return nothing
    end
    walk = (recurse, x, ys...) -> _map(recurse, children_(x))
    x = fmap(x->x, x; walk)
    fmap(Δx, x) do Δxi, xi
        if (xi, vals) in cache(cx)
            if vals === nothing
                cache(cx)[xi] = Δxi
            end
        end
    end
    return cx
end


import Base.+
+(::Nothing, ::Nothing) = nothing # Fix for summing nothing, TODO: Remove this

function fmap_sum(a, b)
    fmap(a, b) do x, y
        try
            return x .+ y
        catch
            return x
        end
    end
end

function add_context!(cx::Zygote.Context, cx2::Vector{Any}, params; factor=1)
    for (i, key) in enumerate(params)
        val = cx2[i]
        add_context_key!(cx, key, val; factor)
    end
end

function add_context!(cx::Zygote.Context, cx2::Zygote.Context{true}; factor=1)
    for (key, val) in cache(cx2)
        add_context_key!(cx, key, val; factor)
    end
end

function add_context_key!(cx, key, val; factor=1)
    if val !== nothing
        val = mul_nothing(val, factor)
    end
    if !(key in keys(cache(cx))) || cache(cx)[key] === nothing
        cache(cx)[key] = val
    elseif val isa Base.RefValue
    elseif val !== nothing && cache(cx)[key] !== nothing # Add the gradients
        cache(cx)[key] = fmap_sum(cache(cx)[key], val)
    end
end

function add_context!(cx, contexts::Union{Vector{Vector{Any}}, Vector{Zygote.Context{true}}}, args...; factor=[1 for _ in contexts])
    for (cxi, fac) in zip(contexts, factor)
        add_context!(cx, cxi, args...; factor=fac)
    end
end


function cache_to_vector(cx, params)
    cx_p = Vector{Any}(undef, length(params))
    for (i, p) in enumerate(params)
        cx_p[i] = cache(cx)[p]
    end
    return cx_p
end

function find_vars(f::Function)
    o = []
    for s in fieldnames(typeof(f))
        o = vcat(o, find_vars(getfield(f, s)))
    end
    return o
end
find_vars(f) = f
find_params(f::Function) = Flux.params(find_vars(f))