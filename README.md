# ParallelGradient
Fast automatic differentiation thorugh pmap and tmap functions. The standard chain rule for the pmap function copys over the pullback functions tp the mmaster process from the workers, instead in this package the pullback functions are stored and computed on the workers and only the gradients are copied over to the master process. Also it adds support for Flux models, by defining and compying the 'Context' objects around. Fot this the following functions are provided:
- dpmap: parallel distributed map with automatic differentiation
- dtmap: parallel threaded map with automatic differentiation
- dpmap_scalar: parallel distributed map with automatic differentiation for scalar functions, this is faster as it only need to call the process only once.
- dptmap: parallel distributed map over both threads and proccesses
- dptmap_scalar: parallel distributed map over both threads and proccesses for scalar functions

## Example dpmap
```julia
addprocs(nr_procs)
function f(x, y)
    return sum(x .* y)
end
g = gradient(x, y) do
    return sum(dpmap_scalar(f, x, y))
end
```

## Example dptmap
```julia
addprocs(nr_procs, exeflags="-t $nr_threads")
g = gradient(x, y) do
    return sum(dptmap(f, x, y))
end
```