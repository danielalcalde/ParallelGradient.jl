module ParallelGradient
using Random
using Distributed
using Distributed: splitrange

using Zygote: @adjoint, _pullback, _tryaxes, last_or_nothing, _tryreverse, _unzip, accum,  _restore, cache
using Zygote
using Functors
using Flux

include("debug.jl")
include("context.jl")
include("maps.jl")
include("distributed_macro.jl")
include("dpmap.jl")
include("dtmap.jl")



export dpmap, dpmap_scalar, tmap, dtmap, ptmap

end # module ParallelGradient
