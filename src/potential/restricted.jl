export RestrictedPotential

"""
Given a potential U, 
implement a new potential V such that 
e^{-V(x)} = Ω(x) e^{-U(x)} where Ω is an indicator function:
    - Ω(x) = true if x is inside the domain;
    - Ω(x) = false if x is outside of the domain.
"""
mutable struct RestrictedPotential{T<:AbstractFloat} <: Potential{T}
    dim::Int
    U::Potential{T}
    Ω::Function
    cutoff::T
    query_u::QueryNumber
    query_gradu::QueryNumber
    query_hessu::QueryNumber
    query_laplaceu::QueryNumber
    count_mode::Symbol
end

"""
A default constructor. By default, let cutoff=Inf.
"""
function RestrictedPotential(dim::Int, U::Potential{T}, Ω::Function; 
        count_mode=:unsafe_count) where T<:AbstractFloat

    safe = get_safe_mode(count_mode)
    RestrictedPotential(dim, U, Ω, T(Inf), 
                        set_query_number(0, safe=safe), set_query_number(0, safe=safe), 
                        set_query_number(0, safe=safe), set_query_number(0, safe=safe),
                        count_mode)
end

function _U(p::RestrictedPotential{T}, x::Vector{T}) where T<:AbstractFloat
    p.Ω(x) ? U(p.U, x) : p.cutoff
end

function _U(p::RestrictedPotential{T}, x::Matrix{T}) where T<:AbstractFloat
    idx = p.Ω(x)
    v = U(p.U, x)
    v[@.(!idx)] .= p.cutoff
    return v
end

function _∇U(p::RestrictedPotential{T}, x::Vector{T}) where T<:AbstractFloat
    p.Ω(x) ? ∇U(p.U, x) : zeros(T, p.dim)
end

function _HessU(p::RestrictedPotential{T}, x::Vector{T}) where T<:AbstractFloat
    p.Ω(x) ? HessU(p.U, x) : zeros(T, p.dim, p.dim)
end

function _LaplaceU(p::RestrictedPotential{T}, x::Vector{T}) where T<:AbstractFloat
    p.Ω(x) ? LaplaceU(p.U, x) : T(0.0)
end
