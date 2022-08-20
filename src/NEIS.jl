module NEIS

using Distributed
using LinearAlgebra
using Printf
using Random
using Plots
using ForwardDiff
using Documenter
using FFTW
using SparseArrays
using SpecialFunctions
using BasicInterpolators
using Tullio
using Reexport: @reexport
using Statistics: mean, std
using Flux: glorot_normal, glorot_uniform

export Potential
export Dyn

# Utility function
include("util/nn.jl")
include("util/util.jl")
include("util/quadrature.jl")
include("util/grid.jl")
include("util/ode_solver.jl")
include("util/traj.jl")

# todo: 
# 1. vectorized implementation of funnel.jl
#
# Potential functions
abstract type Potential{T} end
include("potential/mixture_potential.jl")
include("potential/restricted.jl")
include("potential/gaussian.jl")
include("potential/gaussian_stat.jl")
include("potential/funnel.jl")
include("potential/testcase.jl")
include("potential/query.jl")
#include("potential/poisson.jl") # solvers of Poisson eq.
#include("potential/loggaussiancox.jl")


# Implementation of classical methods
include("classical_methods/vanilla.jl") # vanilla IS.
include("classical_methods/mala.jl")
include("classical_methods/smc.jl") # AIS only. Its SMC version is not implemented yet.

# Flow dynamics
abstract type Dyn end
# a generic functor.
#function (flow::Dyn)(x::Array)
#    flow.f(x, flow.para_list...)
#end

include("dyn/dyn_fix.jl") # a fixed dynamics
include("dyn/dyn_generic_two.jl") # a generic two-layer nn-flow.
#include("dyn/dyn_generic_one.jl")
#include("dyn/dyn_grad_two.jl")
#include("dyn/dyn_funnelexpansatz.jl")
#include("dyn/dyn_grad_flow.jl")
#include("dyn/dyn_ul.jl")



# Generator
#include("dyn/ode_flow.jl")
#include("opt/generator.jl")
#include("opt/opt.jl")
#include("opt/opt_ode.jl")
#include("opt/opt_int.jl")

end # module
