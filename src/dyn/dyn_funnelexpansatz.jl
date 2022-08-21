export funnelexpansatz, init_funnelexpansatz

mutable struct funnelexpansatz{T} <: DynTrain{T}
    dim::Int
    para_list::AbstractArray{Vector{T}}
    f::Function
    num_para::Array{Int}
    total_num_para::Int
    Ω::Function
end

function ∇b(b::funnelexpansatz{T}, x::Vector{T}) where T<:AbstractFloat
    n = b.dim
    if b.Ω(x)
        β, a = b.para_list[1]
        return vcat(vcat(zeros(T,1,n)), hcat(zeros(T,n-1,1), -a*Matrix{T}(1.0I,n-1,n-1)))
    else
        return zeros(T, n, n)
    end
end

function divg_b(b::funnelexpansatz{T}, x::Vector{T}) where T<:AbstractFloat
    if b.Ω(x)
        n = b.dim
        β, a = b.para_list[1]
        return -a*(n-1)
    else
        return T(0.0)
    end
end

function grad_divg_b(b::funnelexpansatz{T}, x::Vector{T}) where T<:AbstractFloat
   return zeros(T, b.dim)
end

function grad_b_wrt_para(b::funnelexpansatz{T}, x::Vector{T}) where T<:AbstractFloat
    n = b.dim
    if b.Ω(x)
        β, a = b.para_list[1]
        v1 = vcat([-1], zeros(T,n-1))
        v2 = vcat([0], -x[2:end])
        return hcat(v1, v2)
    else
        return zeros(T,n,2)
    end
end

function grad_divg_wrt_para(b::funnelexpansatz{T}, x::Vector{T}) where T<:AbstractFloat
    if b.Ω(x)
        n = b.dim
        β, a = b.para_list[1]
        return [T(0.0), -(n-1)]
    else
        return zeros(T,2)
    end
end

function init_funnelexpansatz(n::Int, β::T, a::T, Ω::Function) where T<:AbstractFloat
    function funnelexpansatz_f(x, args...)
        β, a = args[1]
        if Ω(x)
            return vcat([-β], -a*x[2:end])
        else
            return zeros(T,n)
        end
    end

    para_list = [[β, a]]
    num_para = [2]
    total_num_para = 2
    return funnelexpansatz(n, para_list, funnelexpansatz_f, 
                           num_para, total_num_para, Ω);
end
