export MixturePotential, U, ∇U, HessU, LaplaceU,
    mixedPotential_Potential,
    mixedPotential_Grad,
    mixedPotential_Hess,
    mixedPotential_Laplace

"""
A mixture from a list of potentials and weights.

The potential ``U^{\\beta}(x) = β U``,
where ``\\exp(-U(x)) = \\sum_{j=1}^n w_j \\exp(-U_j(x))``.
"""
mutable struct MixturePotential{T<:AbstractFloat} <: Potential{T}
    dim::Int
    Ulist::Array{}
    weightlist::Array{T,1}
    β::T
    query_u::UInt128
    query_gradu::UInt128
    query_hessu::UInt128
    query_laplaceu::UInt128
end

function MixturePotential(dim::Int, Ul::Array{}, wl::Array{T,1}) where T<: AbstractFloat
    return MixturePotential(dim, Ul, wl, T(1.0), UInt128(0), UInt128(0), UInt128(0), UInt128(0))
end

function U(p::MixturePotential{T}, x::Array{T}) where T<:AbstractFloat
    p.query_u += size(x,2)
    return mixedPotential_Potential(x, p.Ulist, p.weightlist, p.β) 
end

function ∇U(p::MixturePotential{T}, x::Array{T}) where T<:AbstractFloat
    p.query_gradu += size(x,2)
    return mixedPotential_Grad(x, p.Ulist, p.weightlist, p.β)
end

function HessU(p::MixturePotential{T}, x::Array{T}) where T<:AbstractFloat
    p.query_hessu += size(x,2)
    return mixedPotential_Hess(x, p.Ulist, p.weightlist, p.β)
end

function LaplaceU(p::MixturePotential{T}, x::Array{T}) where T<:AbstractFloat
    p.query_laplaceu += size(x,2)
    return mixedPotential_Laplace(x, p.Ulist, p.weightlist, p.β)
end

function mixedPotential_Potential(x::Array{T,1},
        Ulist::Array{}, weightlist::Array{T,1}, β::T; 
        ub = 85) where T <: AbstractFloat

    v = 0
    for j = 1:length(weightlist)
        v += weightlist[j]*exp.(-U(Ulist[j],x))
    end

    return min(-log(v)*β, ub) # avoid infinity.
end

function mixedPotential_Potential(x::Array{T,2},
        Ulist::Array{}, 
        weightlist::Array{T,1}, β::T;
        ub = 85) where T <: AbstractFloat

    num_particle = size(x,2)
    v = zeros(T, num_particle)

    for j = 1:length(weightlist)
        v .+= weightlist[j]*exp.(-U(Ulist[j], x))
    end
    return min.(-log.(v)*β, ub) # avoid infinity.
end

function mixedPotential_Grad(x::Array{T,1},
        Ulist::Array{}, 
        weightlist::Array{T,1}, β::T; eps=1.0e-40) where T <: AbstractFloat

    dim = length(x)
    weight_potential = [weightlist[j]*exp(-U(Ulist[j],x)) for j = 1:length(weightlist)]
    # avoid extreme cases.
    if sum(weight_potential) < eps
        weight_potential .= T(0.0)
    else
        weight_potential .= weight_potential/sum(weight_potential)
    end
    grad = zeros(T, dim)
    for j = 1:length(weightlist)
        grad .+= weight_potential[j]*∇U(Ulist[j], x)
    end
    return grad*β
end

function mixedPotential_Grad(x::Array{T,2},
        Ulist::Array{}, 
        weightlist::Array{T,1}, β::T; eps=1.0e-40) where T <: AbstractFloat

    wlen = length(weightlist)
    dim = size(x,1)
    num_particle = size(x,2)
    weight_potential = zeros(T, wlen, num_particle)
    for j = 1:wlen
        weight_potential[j,:] .= weightlist[j]*exp.(-U(Ulist[j],x))
    end
    # avoid extreme cases.
    weight_sum = vec(sum(weight_potential, dims=1))
    weight_idx = weight_sum .> eps
    weight_potential[:,@.(!weight_idx)] .= T(0.0)
    # normalize.
    weight_potential[:,weight_idx] .= 
        divide_col(weight_potential[:,weight_idx], weight_sum[weight_idx])
    grad = zeros(T,size(x))
    for j = 1:wlen
        grad .+= multiply_col(∇U(Ulist[j],x), weight_potential[j,:])
    end
    return grad*β
end

function mixedPotential_Hess(x::Array{T,1},
        Ulist::Array{}, 
        weightlist::Array{T,1}, 
        β::T; 
        option="hess", eps=1.0e-40) where T <: AbstractFloat
    
    dim = length(x)
    weight_potential = [weightlist[j]*exp(-U(Ulist[j],x)) for j = 1:length(weightlist)]
    # avoid extreme cases.
    if sum(weight_potential) < eps
        weight_potential = zeros(T, length(weightlist))
    else
        weight_potential = weight_potential/sum(weight_potential)
    end
    grad = zeros(T, dim)
    hess = zeros(T, dim,dim)
    for j = 1:length(weightlist)
        vec = ∇U(Ulist[j],x)
        grad += weight_potential[j]*vec
        hess -= weight_potential[j]*vec*vec'
        hess += weight_potential[j]*HessU(Ulist[j],x)
    end
    hess += grad*grad'
    if option == "hess"
        return hess*β
    else
        return grad*β, hess*β
    end
end

function mixedPotential_Laplace(x::Array{T,1},
        Ulist::Array{}, 
        weightlist::Array{T,1}, β::T; 
        option="laplace", eps=1.0e-40) where T <: AbstractFloat
    dim = length(x)
    weight_potential = [weightlist[j]*exp(-U(Ulist[j],x)) for j = 1:length(weightlist)]

    grad = zeros(T, dim)
    laplace = T(0.0)
    # avoid extreme cases.
    if sum(weight_potential) > eps
        weight_potential = weight_potential/sum(weight_potential)
        for j = 1:length(weightlist)
            vec = ∇U(Ulist[j],x)
            grad += weight_potential[j]*vec
            laplace -= weight_potential[j]*dot(vec,vec)
            laplace += weight_potential[j]*LaplaceU(Ulist[j],x)
        end
        laplace += dot(grad,grad)
    end
    if option == "laplace"
        return laplace*β
    else
        return grad*β, laplace*β
    end
end
