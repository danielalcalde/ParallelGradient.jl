using ParallelGradient
using Distributed
using Zygote
using Flux
using LinearAlgebra
using DataStructures

using Test

nr_procs, nr_threads = 2, 3
addprocs(nr_procs, exeflags="-t $nr_threads")

@everywhere begin
    using Zygote
    using Flux
    using ParallelGradient
    using LinearAlgebra
    BLAS.set_num_threads(1)
end
import Base.isapprox
function Base.isapprox(g1::Zygote.Grads, g2::Zygote.Grads)
    for k in g1.params
        if !(k in g2.params)
            return false
        end
        if !isapprox(g1[k], g2[k])
            return false
        end
    end
    return true
end

for file in readlines(joinpath(@__DIR__, "testgroups"))
    @testset "$file" begin
        include(file * ".jl")
    end
end