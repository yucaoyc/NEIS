export DynFix, DynFixWithDivg

"""
This is a simplest flow 𝐛: ℜ ᵈ → ℜ ᵈ.
The function f contains the implementation of 𝐛.
"""
struct DynFix <: Dyn
    dim::Int
    f::Function
end

function (b::DynFix)(x::Array{T}) where T<:AbstractFloat
    return b.f(x)
end

"""
By default, we use ForwardDiff to take the gradient.
"""
function ∇b(b::DynFix, x::Array{T}) where T<:AbstractFloat
    ForwardDiff.jacobian(z -> b.f(z), x)
end

function divg_b(b::DynFix, x::Array{T}) where T<:AbstractFloat
    tr(∇b(b,x))
end

"""
In case the divergence is know,
we can speed up the code by using divergence directly.
"""
mutable struct DynFixWithDivg <: Dyn
    dim::Int
    f::Function
#    para_list::Array{}
    divg_b::Function
end

function divg_b(flow::DynFixWithDivg, x::Array{T}) where T <: AbstractFloat
    flow.divg_b(x)
end
