"""
Sama as `vcat(o...)` but with no error differentiating.
"""
function vcat_vec(vs::Vector{Vector{T}}; dims=length.(vs)) where T  
    v = Vector{T}(undef, sum(dims))
    j = 1
    for (d, vi) in zip(dims, vs)
       v[j:j+d-1] = vi
       j += d
    end
    return v
end

function vcat_vec(vs::Vector{Vector}; dims=length.(vs))
    v = Vector{Any}(undef, sum(dims))
    j = 1
    for (d, vi) in zip(dims, vs)
       v[j:j+d-1] = vi
       j += d
    end
    return v
end

function vcat_vec(vs::NTuple{N, Vector{T}}; dims=length.(vs)) where {N, T}  
    v = Vector{T}(undef, sum(dims))
    j = 1
    for (d, vi) in zip(dims, vs)
       v[j:j+d-1] = vi
       j += d
    end
    return v
end

Zygote.@adjoint function vcat_vec(vs::Vector{Vector{T}}) where T 
    dims = length.(vs)
    function ∇vcat_vec(∇)
        ∇s = Vector{Vector{T}}(undef, length(dims))
        local j = 1
        function f(d)
            ∇i = ∇[j:j+d-1]
            j = j + d
            return ∇i
        end
        ∇s = [f(d) for d in dims]
        return (∇s, )
    end
   return  vcat_vec(vs; dims), ∇vcat_vec
end

Zygote.@adjoint function vcat_vec(vs::NTuple{N, Vector{T}}) where {N, T}
    dims = length.(vs)
    function ∇vcat_vec(∇)
        ∇s = Vector{Vector{T}}(undef, length(dims))
        local j = 1
        function f(d)
            ∇i = ∇[j:j+d-1]
            j = j + d
            return ∇i
        end
        ∇s = Tuple(f(d) for d in dims)
        
        return (∇s, )
    end
   return  vcat_vec(vs; dims), ∇vcat_vec
end


function hcat_vec(V)
    height = length(V[1])
    @Zygote.ignore_derivatives begin
        for j in 2:length(V)
            if length(V[j]) != height
                throw(DimensionMismatch("vectors must have same lengths"))
            end
        end
    end
    return [ V[j][i] for i=1:length(V[1]), j=1:length(V) ]
end