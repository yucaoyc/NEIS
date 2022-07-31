using Distributed
num_threads = Base.Sys.CPU_THREADS

if !(@isdefined to_train)
  # if to train
  # then we always add more cpu
  to_add_procs = true
end

if !(@isdefined to_add_procs)
  # by default, not add proc.
  to_add_procs = false
end

if to_add_procs
    if num_threads <= 32
        if nprocs() < (num_threads - 2)
            addprocs(num_threads-2-nprocs())
        end
    else
        if nprocs() < num_threads - 8
            addprocs(num_threads-8-nprocs())
        end
    end
end

@everywhere push!(LOAD_PATH, "../src")
@everywhere push!(LOAD_PATH, "../../src")
@everywhere push!(LOAD_PATH, "./")
@everywhere using SharedArrays

using LinearAlgebra
using Printf
using Flux
using JLD2, FileIO
using ModTestCase
using ModDyn: init_dyn_nn
using ModUtil: get_ndims
@everywhere using ModUtil: domain_ball
using Random
using ModOpt

Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)
