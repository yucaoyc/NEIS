module NEIS

using Distributed
using LinearAlgebra
using Printf
using Random
using SharedArrays
using ForwardDiff
using Documenter
using FFTW
using SparseArrays
using SpecialFunctions
using BasicInterpolators
using Tullio
using Reexport: @reexport
using Statistics: mean, std, median
using Flux: glorot_normal, glorot_uniform

import Humanize: datasize
using Plots
using Plots.PlotMeasures
using LaTeXStrings
using ProgressMeter
using FileIO, JLD2

export Potential
export Dyn, DynTrain

# abstract types and a generic functor.
abstract type Dyn{T} end
abstract type DynTrain{T} <: Dyn{T} end
function (flow::DynTrain{T})(x::Array{T}) where T<:AbstractFloat
    flow.f(x, flow.para_list...)
end
abstract type Potential{T} end

########################################
# Utility function
include("util/nn.jl")
include("util/util.jl")
include("util/quadrature.jl")
include("util/grid.jl")
include("util/ode_solver.jl")
include("util/traj.jl")

# todo:
# 1. vectorized implementation of funnel.jl
# Potential functions
include("potential/query.jl")
include("potential/mixture_potential.jl")
include("potential/restricted.jl")
include("potential/gaussian.jl")
include("potential/gaussian_stat.jl")
include("potential/funnel.jl")
include("potential/testcase.jl")
include("potential/explicit.jl")
#include("potential/loggaussiancox.jl")
function (V::Potential{T})(x::Array{T}) where T<:AbstractFloat
    return U(V, x)
end

# Implementation of classical methods
include("classical_methods/vanilla.jl") # vanilla IS.
include("classical_methods/mala.jl")
include("classical_methods/ais_neal.jl")

# Flow dynamics
include("dyn/dyn_fix.jl") # a fixed dynamics
include("dyn/dyn_generic_two.jl") # a generic two-layer nn-flow.
include("dyn/dyn_generic_one.jl")
include("dyn/dyn_grad_two.jl")
include("dyn/dyn_funnelexpansatz.jl")
DynNNGeneric=Union{DynNNGenericOne, DynNNGenericTwo, DynNNGradTwo}
include("dyn/dyn_util.jl")
include("dyn/init_dyn.jl")
#include("dyn/dyn_grad_flow.jl")

# training optimal flows
include("opt/phi.jl")
include("opt/opt.jl")
include("opt/opt_int.jl")
include("opt/opt_ode.jl")
include("opt/train.jl")
include("opt/test_zero_var.jl")

# budget plan and evaluate Z
include("opt/budget.jl")
include("evaluate/evaluate.jl")

# Generator
include("generator/generator.jl")
include("generator/solve_poisson_torus.jl") # solvers of Poisson eq. on 2d torus
include("generator/solve_poisson_neumann.jl")

end # module
