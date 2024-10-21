macro distributed_map(loop)
    if !isa(loop,Expr) || loop.head !== :for
        error("malformed @distributed loop")
    end
    var = loop.args[1].args[1]
    r = loop.args[1].args[2]
    body = loop.args[2]
    if Meta.isexpr(body, :block) && body.args[end] isa LineNumberNode
        resize!(body.args, length(body.args) - 1)
    end
       
    return :(pmap_chunked($(make_preduce_body_map(var, body)), $(esc(r)); input_chunking=false))

end

function make_preduce_body_map(var, body)
    quote
        function (R)
            accum = Vector{Any}(undef, length(R))
            for (i, $(esc(var))) in enumerate(R)
                accum[i] = $(esc(body))
            end
            accum
        end
    end
end