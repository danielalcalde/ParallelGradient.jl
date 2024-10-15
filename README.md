# ParallelGradient

**ParallelGradient** provides fast automatic differentiation using parallel processing via `pmap` and `tmap`. Unlike the standard chain rule in `pmap`, which transfers pullback functions from worker processes to the master process, this package retains the pullback functions on the workers. This allows only the computed gradients to be transferred back to the master, leading to a more efficient gradient computation. Additionally, ParallelGradient supports Flux models by defining and compiling context objects for them.

## Key Features
- **`dpmap`**: Parallel distributed map with automatic differentiation.
- **`dtmap`**: Parallel threaded map with automatic differentiation.
- **`dpmap_scalar`**: A faster version of `dpmap` for scalar functions, reducing communication overhead by only calling the process once.
- **`dptmap`**: Parallel distributed map over both threads and processes.
- **`dptmap_scalar`**: A combination of threads and processes for scalar functions, optimized for performance.

## Installation

You can add ParallelGradient to your Julia environment with:

```julia
using Pkg
Pkg.add("ParallelGradient")
```

## Usage Examples

### 1. Distributed Parallel Map (`dpmap`)

Here’s an example of using `dpmap` for parallel gradient computation:

```julia
using ParallelGradient
addprocs(nr_procs) # Add worker processes

function f(x, y)
    return sum(x .* y)
end

g = gradient(x, y) do
    return sum(dpmap_scalar(f, x, y))
end
```

In this example, the `dpmap_scalar` function computes gradients by distributing the function `f` across worker processes.

### 2. Distributed and Threaded Parallel Map (`dptmap`)

For more advanced parallelism over both threads and processes:

```julia
using ParallelGradient
addprocs(nr_procs, exeflags="-t $nr_threads") # Add worker processes and threads

function f(x, y)
    return sum(x .* y)
end

g = gradient(x, y) do
    return sum(dptmap(f, x, y))
end
```

This example utilizes `dptmap` to distribute and parallelize the function `f` over both processes and threads, enhancing performance for large-scale computations.

## Where to Use

These functions can be used in any scenario where the standard `map` function applies, making them flexible and easy to integrate into existing workflows.
