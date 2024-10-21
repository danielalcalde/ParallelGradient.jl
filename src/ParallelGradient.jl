module ParallelGradient
using Random
using Distributed
using Distributed: splitrange

using Zygote: @adjoint, _pullback, _tryaxes, last_or_nothing, _tryreverse, _unzip, accum,  _restore, cache
using Zygote
using Functors
using Flux

include("misc.jl")
include("debug.jl")
include("context.jl")
include("maps.jl")
include("distributed_macro.jl")
include("dpmap.jl")
include("dtmap.jl")
include("dptmap.jl")
include("start_processes.jl")

export tmap, pmap_chunked
export dpmap, dpmap_scalar, dtmap, ptmap, dptmap_scalar

end # module ParallelGradient
