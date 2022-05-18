module ModPotential

using LinearAlgebra
using Printf

export Potential,
    FixedPotential,
    generate_extended_Potential,
    mixedPotential_Potential,
    mixedPotential_Grad,
    mixedPotential_Hess,
    mixedPotential_Laplace,
    convert_to_bounded_potential

# Data structure

abstract type Potential end
TypeCovMatrix = Union{Array{Float64,2}, Diagonal{Float64, Vector{Float64}}}

mutable struct FixedPotential <: Potential
    dim::Int64
    U
    gradU
    HessU
    LaplaceU
    fixpara::AbstractArray{} # fixed parameters
    sampler
end

function generate_extended_Potential(U::FixedPotential, a::Float64)
    # from U(q) to generate U(q) + a |p|^2/2
    #
    # Details:
    # (1) we assume U does not have gradU_para, or say gradU_para = [nothing];
    # (2) we assume q, p have the same dimension.

    U_ext(x) = U.U(x[1:U.dim]) + a*norm(x[(U.dim+1):(2*U.dim)])^2/2
    gradU_ext(x) = vcat(U.gradU(x[1:U.dim]), a*x[(U.dim+1):(2*U.dim)])
    HessU_ext(x) = hcat( vcat(U.HessU(x[1:U.dim]),zeros(U.dim,U.dim)),
                        vcat(zeros(U.dim,U.dim),a*Matrix(I,U.dim,U.dim)) )
    LaplaceU_ext = nothing

    if U.sampler == nothing
        return FixedPotential(2*U.dim, U_ext, gradU_ext, HessU_ext, LaplaceU_ext, [U,a], nothing)
    else
        sampler = (z) -> vcat(U.sampler(z), randn(U.dim)/sqrt(a))
        return FixedPotential(2*U.dim, U_ext, gradU_ext, HessU_ext, LaplaceU_ext, [U,a], sampler)
    end
end

function mixedPotential_Potential(x::Array{},
        Ulist::Array{FixedPotential,1}, weightlist::Array{Float64,1}, β::Float64)

    v = 0;
    for j = 1:length(weightlist)
        v += weightlist[j]*exp(-Ulist[j].U(x))
    end
    return min(-log(v)*β,750) # avoid infinity.

end

function mixedPotential_Grad(x::Array{},
        Ulist::Array{FixedPotential,1}, weightlist::Array{Float64,1}, β::Float64)

    dim = length(x)
    weight_potential = [weightlist[j]*exp(-Ulist[j].U(x)) for j = 1:length(weightlist)]

    # avoid extreme cases.
    if sum(weight_potential) < 1.0e-300
        weight_potential = zeros(length(weightlist))
    else
        weight_potential = weight_potential/sum(weight_potential)
    end
    grad = zeros(dim)
    for j = 1:length(weightlist)
        grad += weight_potential[j]*Ulist[j].gradU(x)
    end
    return grad*β

end

function mixedPotential_Hess(x::Array{},
        Ulist::Array{FixedPotential,1}, weightlist::Array{Float64,1}, β::Float64; option="hess")

    dim = length(x)
    weight_potential = [weightlist[j]*exp(-Ulist[j].U(x)) for j = 1:length(weightlist)]

    # avoid extreme cases.
    if sum(weight_potential) < 1.0e-300
        weight_potential = zeros(length(weightlist))
    else
        weight_potential = weight_potential/sum(weight_potential)
    end

    grad = zeros(dim)
    hess = zeros(dim,dim)
    for j = 1:length(weightlist)
        vec = Ulist[j].gradU(x)
        grad += weight_potential[j]*vec
        hess -= weight_potential[j]*vec*vec'
        hess += weight_potential[j]*Ulist[j].HessU(x)
    end
    hess += grad*grad'

    if option == "hess"
        return hess*β
    else
        return grad*β, hess*β
    end

end

function mixedPotential_Laplace(x::Array{Float64,1},
        Ulist::Array{FixedPotential,1}, weightlist::Array{Float64,1}, β::Float64; option="laplace")

    dim = length(x)
    weight_potential = [weightlist[j]*exp(-Ulist[j].U(x)) for j = 1:length(weightlist)]

    grad = zeros(dim)
    laplace = 0.0
    # avoid extreme cases.
    if sum(weight_potential) > 1.0e-300
        weight_potential = weight_potential/sum(weight_potential)
        for j = 1:length(weightlist)
            vec = Ulist[j].gradU(x)
            grad += weight_potential[j]*vec
            laplace -= weight_potential[j]*dot(vec,vec)
            laplace += weight_potential[j]*Ulist[j].LaplaceU(x)
        end
        laplace += dot(grad,grad)
    end
    if option == "laplace"
        return laplace*β
    else
        return grad*β, laplace*β
    end

end


function convert_to_bounded_potential(V::FixedPotential, Ω, cutoff=Inf)
    # Ω(x) is a function that returns 1 if x is inside the domain
    # and returns 0 if not.
    #
    # We want to return a fixed potential that V₁ such that
    # e^{-V₁} = Ω(x) e^{-V}.

    dim = V.dim
    Poten(x) =  Ω(x) ? V.U(x) : cutoff
    gradU(x) = Ω(x) ? V.gradU(x) : zeros(dim)
    HessU(x) = Ω(x) ? V.HessU(x) : zeros(dim, dim)
    LaplaceU(x) = Ω(x) ? V.LaplaceU(x) : 0.0
    fixpara = V.fixpara
    sampler = nothing
    return FixedPotential(dim, Poten, gradU, HessU, LaplaceU, fixpara, sampler)
end

#################################
include("potential/gaussian.jl")
include("potential/funnel.jl")
include("potential/cauchy.jl")
include("potential/loggaussiancox.jl")

end # end of module
