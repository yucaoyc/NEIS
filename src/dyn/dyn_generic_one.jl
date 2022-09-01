# A linear flow

export DynNNGenericOne, init_DynNNGenericOne, init_random_DynNNGenericOne

"""
A Generic linear parameterization of flow
"""
mutable struct DynNNGenericOne{T} <: DynTrain{T}
    dim::Int # dimension
    para_list::Array{Array{T}} # an array of parameters
#    train_para_idx::Array{Int}
    f::Function
    num_para::Array{Int}
    total_num_para::Int
end

function init_DynNNGenericOne(dim::Int, W::Array{T,2}, b::Array{T,1}) where T <: AbstractFloat
    para_list = [W, b]
    f(x, W, b) = W*x .+ b
    return DynNNGenericOne(dim, para_list, f, [dim^2, dim], dim^2+dim)
end

function init_random_DynNNGenericOne(dim::Int; convert=x->Float32.(x),
        init=glorot_uniform, seed::Int=-1)
    seed > 0 ? Random.seed!(seed) : nothing
    W = convert(init(dim,dim))
    b = convert(init(dim))
    return init_DynNNGenericOne(dim, W, b)
end

function ∇b(flow::DynNNGenericOne{T}, x::Vector{T}) where T <: AbstractFloat
   return flow.para_list[1]
end

function divg_b(flow::DynNNGenericOne{T}, x::Vector{T}) where T <: AbstractFloat
    return tr(flow.para_list[1])
end

function grad_divg_b(flow::DynNNGenericOne{T}, x::Vector{T}) where T <: AbstractFloat
   return zeros(T, flow.dim)
end

function grad_b_wrt_para(flow::DynNNGenericOne{T}, x::Vector{T}) where T <: AbstractFloat
    n = flow.dim
    Id = Matrix{T}(1.0I,n,n)
    Fw = @tullio Fw[i,l,k] := Id[i,l]*x[k]
    Fw = reshape(Fw, (n, n*n))
    Fb = Id
    return hcat(Fw, Fb)
end

function grad_divg_wrt_para(flow::DynNNGenericOne{T}, x::Vector{T}) where T <: AbstractFloat
    n = flow.dim
    Fb = zeros(T, n)
    Fw = vec(Matrix{T}(1.0I,n,n))
    return vcat(Fw, Fb)
end

function grad_divg_wrt_para_part(flow::DynNNGenericOne{T}, x::Vector{T}) where T <: AbstractFloat
    n = flow.dim
    Fb = zeros(T,n)
    Fw = vec(Matrix{T}(1.0I,n,n))
    return Fw, Fb
end
