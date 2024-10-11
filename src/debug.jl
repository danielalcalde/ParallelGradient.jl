
print_context(x) = x
Zygote.@adjoint function print_context(x; id="")
    cx = __context__
    function b(Δ)
        println("context_$id: ", length(cache(cx))) 
        global saved_context = cx
        return (Δ,)
    end

    return x, b
end