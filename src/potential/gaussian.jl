export Gaussian, sampler,
    generate_mixGaussian

struct Gaussian{T<:AbstractFloat} <: Potential{T}
    dim::Int
    μ::Vector{T}
    cov::Union{T,AbstractMatrix{T},Diagonal{T}}
    covinv::Union{T,AbstractMatrix{T},Diagonal{T}} # for computing U, etc. 
    covhalf::Union{T,AbstractMatrix{T},Diagonal{T}} # for sampling
    laplace::T
end

function Gaussian(dim::Int,μ::Vector{T},cov::Union{T,Matrix{T},Diagonal{T}}) where T<:AbstractFloat 
    return Gaussian(dim, μ, cov, inv(cov), cov^(T(1/2)), get_laplace_gaussian(cov,dim))
end

function get_laplace_gaussian(cov::Union{Matrix{T},Diagonal{T}}, dim::Int) where T<:AbstractFloat
    return tr(inv(cov))
end

function get_laplace_gaussian(cov::T, dim::Int) where T<:AbstractFloat
    return dim*inv(cov)
end

function gaussU(x::Matrix{T}, μ::Vector{T}, σsqinv::T) where T<:AbstractFloat
    return vec(sum((x .- μ).^2, dims=1).*σsqinv./2)
end

function gaussU(x::Matrix{T}, μ::Vector{T}, 
        Σinv::Union{Matrix{T},Diagonal{T}}) where T<:AbstractFloat
    return vec(T(0.5)*sum((x .- μ) .* (Σinv*(x .- μ)), dims=1))
end

function gaussU(x::Vector{T}, μ::Vector{T}, σsqinv::T) where T<:AbstractFloat
    return norm(x-μ)^2*σsqinv/2
end

function gaussU(x::Vector{T}, μ::Vector{T}, 
        Σinv::Union{Matrix{T},Diagonal{T}}) where T<:AbstractFloat
    return T(0.5)*dot(x-μ, Σinv*(x-μ))
end

function U(p::Gaussian{T}, x::Array{T}) where T<:AbstractFloat
    return gaussU(x, p.μ, p.covinv)
end

function ∇U(p::Gaussian{T}, x::Array{T}) where T<:AbstractFloat
    return p.covinv*(x.-p.μ)
end

function HessU(p::Gaussian{T}, x::Vector{T}) where T<:AbstractFloat
    #return p.covinv*Matrix(T(1.0)I,p.dim,p.dim)
    return get_hess_gaussian(p.covinv, p.dim)
end

function get_hess_gaussian(covinv::Union{Matrix{T},Diagonal{T}}, dim::Int) where T<:AbstractFloat
    return covinv
end

function get_hess_gaussian(covinv::T, dim::Int) where T<:AbstractFloat
    return covinv*Diagonal(ones(T,dim))
end

function LaplaceU(p::Gaussian{T}, x::Vector{T}) where T<:AbstractFloat
    return p.laplace
end

function LaplaceU(p::Gaussian{T}, x::Matrix{T}) where T<:AbstractFloat
    return p.laplace*ones(T,size(x,2))
end

"""
Generate samples according to Gaussian.
"""
function sampler(p::Gaussian{T}, num_particle::Int; func=nothing, args=nothing) where T<:AbstractFloat
    if func==nothing
        # no override function
        return p.μ .+ p.covhalf*randn(T,p.dim,num_particle)
    else
        return func(args)
    end
end

"""
Generate Gaussian mixture.
"""
function generate_mixGaussian(dim::Int, 
        μlist::Array{Array{T,1},1}, 
        Σlist::Array{}, 
        weightlist::Vector{T}) where T<:AbstractFloat

    m = length(weightlist)
    if length(Σlist) != m
        error("Length does not match in generate_mixGaussian.")
    elseif length(μlist) != m
        error("Length does not match in generate_mixGaussian.")
    end
    Ulist = [Gaussian(dim,μlist[j],Σlist[j]) for j=1:m]
    return MixturePotential(dim, Ulist, weightlist)
end
