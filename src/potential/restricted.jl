"""
Given a potential U, 
implement a new potential V such that 
e^{-V(x)} = Ω(x) e^{-U(x)} where Ω is an indicator function:
    - Ω(x) = true if x is inside the domain;
    - Ω(x) = false if x is outside of the domain.
"""
struct RestrictedPotential{T<:AbstractFloat} <: Potential{T}
    dim::Int
    U::Potential{T}
    Ω::Function
    cutoff::T
end

"""
A default constructor. By default, let cutoff=Inf.
"""
function RestrictedPotential(dim::Int, U::Potential{T}, Ω::Function) where T<:AbstractFloat
    RestrictedPotential(dim, U, Ω, T(Inf))
end

function U(p::RestrictedPotential{T}, x::Vector{T}) where T<:AbstractFloat
    p.Ω(x) ? U(p.U, x) : p.cutoff
end

function U(p::RestrictedPotential{T}, x::Matrix{T}) where T<:AbstractFloat
    idx = p.Ω(x)
    v = U(p.U, x)
    v[@.(!idx)] .= p.cutoff
    return v
end

function ∇U(p::RestrictedPotential{T}, x::Vector{T}) where T<:AbstractFloat
    p.Ω(x) ? ∇U(p.U, x) : zeros(T, p.dim)
end

function HessU(p::RestrictedPotential{T}, x::Vector{T}) where T<:AbstractFloat
    p.Ω(x) ? HessU(p.U, x) : zeros(T, p.dim, p.dim)
end

function LaplaceU(p::RestrictedPotential{T}, x::Vector{T}) where T<:AbstractFloat
    p.Ω(x) ? LaplaceU(p.U, x) : T(0.0)
end
