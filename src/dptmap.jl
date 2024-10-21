dptmap_scalar(f, iter...; pmap_function=pmap_chunked, tmap_function=tmap, batch_size=nothing) = ptmap(f, iter...; pmap_function=pmap_function, tmap_function=tmap_function, batch_size=batch_size)
function ∇dptmap_scalar(cx, f::F, args::Vararg{Any, N}; pmap_function=pmap_chunked, tmap_function=tmap, batch_size=nothing) where {F, N}
    cx_type = typeof(cx).parameters[1]

    params = Flux.params()
    if cx_type # Bad fix for not knowing which parameters are needed. We assume that the context should be filled with the parameters of the function
        params = find_params(f)
    end

    nr_threads = get_nrthreads(; batch_size)
    chunks = Zygote.@ignore_derivatives splitrange(Int(firstindex(args[1])), Int(lastindex(args[1])), nr_threads)
    args_chunks = map(c -> cunkit(args, c), chunks)
    function fw_and_bw(args_i)
        params2 = params
        # Define a function that will be called on each worker
        function fw_and_bw_threaded(args_j...)
            cxi = Zygote.Context{cx_type}(nothing)
            y, back = Zygote._pullback(cxi, f, args_j...)
            for p in params2
                cache(cxi)[p] = nothing
            end

            local b
            if y isa Number
                b = back(1)
            elseif y isa Tuple
                o = (1, (nothing for _ in y[2:end])...) # 
                b = back(o)
            else
                error("The output of the function should be a number or a tuple, where the gradients should only be of the first element. Got $(typeof(y))")
            end

            return y, b, cache_to_vector(cxi, params2)
        end

        ys_and_Δf_and_context = tmap_function(fw_and_bw_threaded, args_i...)
        
        ys = [i[1] for i in  ys_and_Δf_and_context]
        Δf_and_args_zipped = [i[2] for i in  ys_and_Δf_and_context]
        context = [i[3] for i in  ys_and_Δf_and_context]

        return ys, Δf_and_args_zipped, context
    end
    ys_and_Δf_and_context = pmap_function(fw_and_bw, args_chunks)

    ys = vcat((i[1] for i in  ys_and_Δf_and_context)...)
    Δf_and_args_zipped = vcat((i[2] for i in  ys_and_Δf_and_context)...)
    context = vcat((i[3] for i in  ys_and_Δf_and_context)...)

    arg_ax = map(_tryaxes, args)
    function map_back(Δs)
        @assert length(Δs) == length(ys) "The length of the gradients should be the same as the length of the outputs"
        Δs = map(zip(ys, Δs)) do (y, Δ)
            if y isa Tuple
                @assert length(Δ) == length(y) "The length of the gradients should be the same as the length of the outputs"
                @assert all(is_nothing_or_zero.(Δ[2:end])) "If the output is a tuple, the gradients should only be taken of the first element. Got $(Δ[2:end])"
                return Δ[1]
            end
            return Δ
        end

        # Multiply Δs with Δf_and_args_zipped
        Δf_and_args_zipped = map((Δ, y) -> fmap(z->mul_nothing(z, Δ), y), Δs, Δf_and_args_zipped)
        Δf_and_args = _unzip(Δf_and_args_zipped, Val(N + 1))
        Δf = reduce(accum, Δf_and_args[1]; init=nothing)
        Δargs = map(_restore, Δf_and_args[2:end], arg_ax)
        
        # Merge the context gotten from the different workers
        if cx_type
            add_context!(cx, context, params; factor=Δs)
        end
        (Δf, Δargs...)
      end
    
    map_back(::Nothing) = nothing
    return ys, map_back
end

@adjoint function dptmap_scalar(f, args::Union{AbstractArray,Tuple}...)
    ∇dptmap_scalar(__context__, f, args...)
end