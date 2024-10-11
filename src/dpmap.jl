function get_pull_dict_and_delete!(key, i)
    val = Main.dpmap_pull_dict[key][i]
    delete!(Main.dpmap_pull_dict[key], i)
    if isempty(Main.dpmap_pull_dict[key])
        delete!(Main.dpmap_pull_dict, key)
    end
    return val
end

function set_pull_dict!(key, i, val)
    @eval Main begin
        if !isdefined(Main, :dpmap_pull_dict)
            dpmap_pull_dict = Dict()
        end
    end
    if !(key in keys(Main.dpmap_pull_dict))
        Main.dpmap_pull_dict[key] = Dict()
    end
    Main.dpmap_pull_dict[key][i] = val
end

dpmap(f, args...; pmap_function=pmap, pmap_function_workerid=nothing) = pmap_function(f, args...)
function ∇dpmap(cx, f::F, args::Vararg{Any, N}; pmap_function=pmap, pmap_function_workerid=nothing) where {F, N}
    if pmap_function_workerid === nothing
        pmap_function_workerid = pmap_function_workerid_[pmap_function]
    end

    key = Random.randstring(100)
    cx_type = typeof(cx).parameters[1]

    params = Flux.params()
    if cx_type # Assuming that the context should be filled with the parameters of the function
        params = find_params(f)
    end
    function fw(i, args_i...)
        cxi = Zygote.Context{cx_type}(nothing)
        y, back = Zygote._pullback(cxi, f, args_i...)

        set_pull_dict!(key, i, (back, cxi, params))
        
        return y, myid()
    end
    i_s = 1:length(args[1])

    ys_and_ids = pmap_function(fw, i_s, args...)

    ys = map(x -> x[1], ys_and_ids)
    worker_ids = map(x -> x[2], ys_and_ids)
    arg_ax = map(_tryaxes, args)
    
    function map_back(Δ)
        function bw(i, δ)
            back, cxi, params = get_pull_dict_and_delete!(key, i)
            for p in params
                cache(cxi)[p] = nothing
            end
            b = back(δ) # Writes to cxi

            cx_vec = cache_to_vector(cxi, params)
            return (b, cx_vec)
        end
        local Δf_and_args_zipped_and_context
        if pmap_function_workerid !== nothing
            Δf_and_args_zipped_and_context = pmap_function_workerid(bw, worker_ids, i_s, Δ)
        else
            # If the pmap function does not have a workerid argument, we assume that it is consistent with placing the same args on the same worker
            Δf_and_args_zipped_and_context = pmap_function(bw, i_s, Δ)
        end

        Δf_and_args_zipped = map(x -> x[1], Δf_and_args_zipped_and_context)
        
        Δf_and_args = _unzip(Δf_and_args_zipped, Val(N + 1))
        Δf = reduce(accum, Δf_and_args[1]; init=nothing)
        Δargs = map(_restore, Δf_and_args[2:end], arg_ax)
        
        # Merge the context gotten from the different workers
        if cx_type
            contexts = map(x->x[2], Δf_and_args_zipped_and_context)
            update_context!(cx, contexts, params)
        end

        (Δf, Δargs...)
      end
    
    map_back(::Nothing) = nothing
    return ys, map_back
end

@adjoint function dpmap(f, args::Union{AbstractArray,Tuple}...; kwargs...)
    ∇dpmap(__context__, f, args...; kwargs...)
end


mul_nothing(x, y::Nothing) = nothing
mul_nothing(x::Nothing, y) = nothing
mul_nothing(x, y) = x .* y

dpmap_scalar(f, iter) = dpmap(f, iter)
function ∇dpmap_scalar(cx, f::F, args::Vararg{Any, N}) where {F, N}   
    ys_Δf = @distributed_map for args_i = args[1]
        y, back = Zygote._pullback(cx, f, args_i)
        y, back(1)
    end
    ys = [y for (y, Δf) in ys_Δf]
    Δf_and_args_zipped = [Δf for (y, Δf) in ys_Δf]
    
    arg_ax = map(_tryaxes, args)
    function map_back(Δ)
        # Multiply Δ with Δf_and_args_zipped
        Δf_and_args_zipped = map((Δi, y) -> fmap(z->mul_nothing(z,Δi), y), Δ, Δf_and_args_zipped)
        Δf_and_args = _unzip(Δf_and_args_zipped, Val(N + 1))
        Δf = reduce(accum, Δf_and_args[1]; init=nothing)
        Δargs = map(_restore, Δf_and_args[2:end], arg_ax)
        
        # TODO: Fix this to update the context cache properly
        for s in keys(Δf)
            if isdefined(f, s)
                update_cache!(cx, getfield(Δf, s), getfield(f, s))
            end
        end
        
        (Δf, Δargs...)
      end
    
    map_back(::Nothing) = nothing
    return ys, map_back
end

@adjoint function dpmap_scalar(f, args::Union{AbstractArray,Tuple}...)
    ∇dpmap_scalar(__context__, f, args...)
end

mapsum(func, iterator, args...) = sum(map(iter -> func(iter, args...), iterator))
pmapsum(func, iterator, args...) = sum(dpmap(iter -> func(iter, args...), iterator))
pmapsum_scalar(func, iterator, args...) = sum(dpmap_scalar(iter -> func(iter, args...), iterator))