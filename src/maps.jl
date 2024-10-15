pmap_w(f, workers::Vector{Int}, args...; kwargs...) = pmap_chuncked(f, args...; workers_=workers, kwargs...)
const pmap_function_workerid_ = Dict{Any, Any}(pmap => pmap_w)


cunkit(c, interval::UnitRange) = map(x->x[interval], c)
iterate_(f, args...) = [f(args_i...) for args_i in zip(args...)]
     
function pmap_chuncked(f, c1, c...; input_chunking=true, workers_::Vector{Int}=workers())
    f_orig = f
    chunks = splitrange(Int(firstindex(c1)), Int(lastindex(c1)), length(workers_))
    all_w = workers_[1:length(chunks)]
    
    if input_chunking
        f = (x...) -> iterate_(f_orig, x...)
    end

    w_exec = Task[]
    for (chunk, pid) in zip(chunks, all_w)
        b, e = first(chunk), last(chunk)
        t = Task(() -> remotecall_fetch(f, pid, c1[b:e], cunkit(c, b:e)...))
        schedule(t)
        push!(w_exec, t)
    end
    return vcat(fetch.(w_exec)...)
end

pmap_function_workerid_[pmap_chuncked] = nothing


function tmap_simple(f, args...)
    tasks = map(args...) do (args_i...)
        Threads.@spawn f(args_i...)
    end
    return fetch.(tasks)
end

function tmap(f, c1, c...)
    f_orig = f
    nr_threads = Threads.nthreads()
    chunks = splitrange(Int(firstindex(c1)), Int(lastindex(c1)), nr_threads)
    f = (x...) -> iterate_(f_orig, x...)
    w_exec = Task[]
    for chunk in chunks
        b, e = first(chunk), last(chunk)
        t = Threads.@spawn f(c1[b:e], cunkit(c, b:e)...)
        push!(w_exec, t)
    end
    return vcat(fetch.(w_exec)...)
end

function get_nrthreads(; batch_size=nothing)
    local nr_threads
    if batch_size === nothing
        nr_threads = pmap_chuncked((_)->Threads.nthreads(), workers(); workers_=workers())
    elseif isinteger(batch_size)
        nr_threads = Int[batch_size for _ in workers()]
    elseif batch_size isa Vector{Int}
        nr_threads = batch_size
    else
        error("batch_size should be an integer or a vector of integers")
    end
    return nr_threads
end

"""
ptmap combines pmap and tmap, it is a parallel map that uses calls threads on the workers for parallelism.
The workers should be started with addprocs(nr_procs, exeflags="-t nr_threads")
"""
function ptmap(f, args...; batch_size=nothing, pmap_function=pmap_chuncked, tmap_function=tmap, kwargs...)
    # We need to split the arguments into chunks for each worker
    nr_threads = Zygote.@ignore_derivatives get_nrthreads(; batch_size)
    chunks = Zygote.@ignore_derivatives splitrange(Int(firstindex(args[1])), Int(lastindex(args[1])), nr_threads)
    args_chunks = map(c -> cunkit(args, c), chunks)

    # Define a function that will be called on each worker
    f_threaded(args_i) = dtmap(f, args_i...; tmap_function)
    out = dpmap(f_threaded, args_chunks; pmap_function, kwargs...) # Vector{Vector{T}}
    return vcat_vec(out) # Vector{T}
end



function splitbuckets(total, np::Vector{Int})
    buckets = zeros(Int, length(np))
    while true
        for (i, n) in enumerate(np)
            if total == 0
                return buckets
            elseif total >= n
                buckets[i] += n
                total -= n
            elseif total < n
                buckets[i] += total
                total = 0
                return buckets
            end
        end
    end
end


function Distributed.splitrange(firstIndex::Int, lastIndex::Int, np::Vector{Int})
    total = lastIndex-firstIndex+1
    buckets = splitbuckets(total, np)
    chunks = UnitRange{Int}[]
    lo = firstIndex
    for (i, b) in enumerate(buckets)
        if b != 0
            hi = lo + b - 1
            push!(chunks, lo:hi)
            lo = hi+1
        end
    end
    return chunks
end
