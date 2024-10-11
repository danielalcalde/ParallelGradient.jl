"""
pmapw asks for specific workers for each job, other than that it is the same as pmap.
"""
function pmap_w(f, workers_::Vector{Int}, c; distributed=true, batch_size=1, on_error=nothing,
                                           retry_delays=[], retry_check=nothing)
    nr_workers = length(workers_)
    f_orig = f
    # Don't do remote calls if there are no workers.
    if (length(workers_) == 0) || (length(workers_) == 1)
        distributed = false
    end

    # Don't do batching if not doing remote calls.
    if !distributed
        batch_size = 1
    end

    # If not batching, do simple remote call.
    if batch_size == 1
        if on_error !== nothing
            f = wrap_on_error(f, on_error)
        end

        if distributed
            #f = remote(p, f)
            f = (id, x) -> remotecall_fetch(f_orig, workers_[mod1(id, nr_workers)], x)
        end

        if length(retry_delays) > 0
            f = wrap_retry(f, retry_delays, retry_check)
        end

        return asyncmap(f, 1:length(c), c; ntasks=()->nr_workers)
    else
        # During batch processing, We need to ensure that if on_error is set, it is called
        # for each element in error, and that we return as many elements as the original list.
        # retry, if set, has to be called element wise and we will do a best-effort
        # to ensure that we do not call mapped function on the same element more than length(retry_delays).
        # This guarantee is not possible in case of worker death / network errors, wherein
        # we will retry the entire batch on a new worker.

        handle_errors = ((on_error !== nothing) || (length(retry_delays) > 0))

        # Unlike the non-batch case, in batch mode, we trap all errors and the on_error hook (if present)
        # is processed later in non-batch mode.
        if handle_errors
            f = wrap_on_error(f, (x,e)->BatchProcessingError(x,e); capture_data=true)
        end

        f = wrap_batch(f, p, handle_errors)
        results = asyncmap(f, c; ntasks=()->nworkers(p), batch_size=batch_size)

        # process errors if any.
        if handle_errors
            process_batch_errors!(p, f_orig, results, on_error, retry_delays, retry_check)
        end

        return results
    end
end


pmap_w(f, p::Vector{Int}, c1, c...; kwargs...) = pmap_w(a->f(a...), p, zip(c1, c...); kwargs...)

const pmap_function_workerid_ = Dict{Any, Any}(pmap => pmap_w)


cunkit(c, interval::UnitRange) = map(x->x[interval], c)


function iterate_(f, args...)
    R = zip(args...)
    accum = Vector{Any}(undef, length(R))
    for (i, args_i) in enumerate(R)
        accum[i] = f(args_i...)
    end
    accum
end
     
function pmap_chuncks(f, c1, c...; input_chunking=true)
    f_orig = f
    chunks = splitrange(Int(firstindex(c1)), Int(lastindex(c1)), nworkers())
    all_w = workers()[1:length(chunks)]
    
    w_exec = Task[]
    for (chunk, pid) in zip(chunks, all_w)
        b, e = first(chunk), last(chunk)
        if input_chunking
            f = (x...) -> iterate_(f_orig, x...)
        end
        
        t = Task(() -> remotecall_fetch(f, pid, c1[b:e], cunkit(c, b:e)...))
        schedule(t)
        push!(w_exec, t)
    end
    a = vcat(fetch.(w_exec)...)
    return a 
end
pmap_function_workerid_[pmap_chuncks] = nothing


function tmap(f, args...)
    tasks = map(args...) do (args_i...)
        Threads.@spawn f(args_i...)
        #Threads.@spawn Base.invokelatest(f, args_i...)
    end
    fetch.(tasks)
end

function tmap_chunked(f, args...)
    nr_threads = Threads.nthreads()
    chunks = splitrange(Int(firstindex(c1)), Int(lastindex(c1)), nr_threads)
    tasks = map(args...) do (args_i...)
        Threads.@spawn f(args_i...)
        #Threads.@spawn Base.invokelatest(f, args_i...)
    end
    fetch.(tasks)
end

function get_nrthreads(; batch_size=nothing)
    local nr_threads
    if batch_size === nothing
        nr_threads = pmap_w((_)->Threads.nthreads(), workers(), workers())
    elseif isinteger(batch_size)
        nr_threads = [batch_size for _ in workers()]
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
function ptmap(f, args...; batch_size=nothing, kwargs...)
    # We need to split the arguments into chunks for each worker
    nr_threads = Zygote.@ignore get_nrthreads(; batch_size)
    chunks = Zygote.@ignore splitrange(Int(firstindex(args[1])), Int(lastindex(args[1])), nr_threads)
    args_chunks = map(c -> cunkit(args, c), chunks)

    # Define a function that will be called on each worker
    f_threaded(args_i) = dtmap(f, args_i...)
    out = dpmap(f_threaded, args_chunks; kwargs...) # Vector{Vector{T}}
    return vcat(out...) # Vector{T}
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
