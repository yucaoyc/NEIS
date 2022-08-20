export Funnel

# todo: vectorized version of Funnel

mutable struct Funnel{T<:AbstractFloat} <: Potential{T}
    dim::Int
    σf::T
    cutoff::T # used to avoid Inf, -Inf etc.
    query_u::UInt128
    query_gradu::UInt128
    query_hessu::UInt128
    query_laplaceu::UInt128
end

function Funnel(dim::Int, σf::T) where T<:AbstractFloat 
    if T==Float32
        cutoff = T(-80)
    elseif T==Float64
        cutoff = T(-200)
    else
        error("Either use Float32 or Float64!")
    end
    Funnel(dim, σf, cutoff, UInt128(0), UInt128(0), UInt128(0), UInt128(0)) 
end

function U(p::Funnel{T}, x::Vector{T}) where T<:AbstractFloat
    p.query_u += 1
    n = p.dim
    σf = p.σf
    cutoff = p.cutoff
    x₁ = x[1]
    return x₁^2/(2*σf^2) + x₁*(n-1)/2 + exp(-max(x₁,cutoff))/2*(dot(x,x)-x₁^2)
end

function ∇U(p::Funnel{T}, x::Vector{T}) where T<:AbstractFloat
    p.query_gradu += 1
    n = p.dim
    σf = p.σf
    cutoff = p.cutoff
    x₁ = x[1]
    v = zeros(T, n)
    v[1] = x₁/σf^2 + (n-1)/2 - exp(-max(x₁,cutoff))/2*(dot(x,x)-x₁^2)
    v[2:end] = exp(-max(x₁,cutoff)).*x[2:end]
    return v
end

function LaplaceU(p::Funnel, x::Vector{T}) where T<:AbstractFloat
    p.query_laplaceu += 1
    n = p.dim
    σf = p.σf
    cutoff = p.cutoff
    x₁ = x[1]
    return exp(-x₁)*(n-1) + 1/σf^2 + 1/2*exp(-x₁)*dot(x[2:end],x[2:end])
end

function HessU(p::Funnel, x::Vector{T}) where T<:AbstractFloat
    p.query_hessu += 1
    n = p.dim
    σf = p.σf
    cutoff = p.cutoff
    x₁ = x[1]
    row1 = vcat(1/σf^2 + 1/2*exp(-x₁)*dot(x[2:end],x[2:end]), -exp(-x₁)*x[2:end])
    row2 = hcat(-exp(-x₁)*x[2:end], exp(-x₁)*T.(Matrix(1.0I,n-1,n-1)))
    return vcat(row1', row2)
end
