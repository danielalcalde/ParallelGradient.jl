dtmap(f, args...; tmap_function=tmap) = tmap(f, args...)
function ∇dtmap(cx, f::F, args::Vararg{Any, N}; tmap_function=tmap) where {F, N}
    cx_type = typeof(cx).parameters[1]

    function fw(args_i...)
        cxi = Zygote.Context{cx_type}(nothing)
        y, back = Zygote._pullback(cxi, f, args_i...)
        return y, back, cxi
    end

    ys_and_backs_and_contexts = tmap_function(fw, args...)

    ys = map(x -> x[1], ys_and_backs_and_contexts)
    backs = map(x -> x[2], ys_and_backs_and_contexts)
    contexts = map(x -> x[3], ys_and_backs_and_contexts)

    arg_ax = map(_tryaxes, args)
    
    function map_back(Δ)
        function bw(back, δ, cxi)
            for p in keys(cache(cx))
                cache(cxi)[p] = nothing
            end
            return back(δ)
        end
        Δf_and_args_zipped = tmap_function(bw, backs, Δ, contexts)
        
        Δf_and_args = _unzip(Δf_and_args_zipped, Val(N + 1))
        Δf = reduce(accum, Δf_and_args[1]; init=nothing)
        Δargs = map(_restore, Δf_and_args[2:end], arg_ax)
        
        # Merge the context gotten from the different threads into the context of the main thread
        if cx_type
            add_context!(cx, contexts)
        end

        (Δf, Δargs...)
      end
    
    map_back(::Nothing) = nothing
    return ys, map_back
end

@adjoint function dtmap(f, args::Union{AbstractArray,Tuple}...; kwargs...)
    ∇dtmap(__context__, f, args...; kwargs...)
end