export ais_neal, linear_scheme

"""
we use the usual interpolation
Uₖ(x) = (1-βₖ) U₀ + βₖ U₁
"""
function linear_scheme(a::Array{T}, b::Array{T}, β::T) where T <: AbstractFloat
    return (1-β)*a .+ β*b
end

function linear_scheme(a::T, b::T, β::T) where T <: AbstractFloat
    return (1-β)*a .+ β*b
end

"""
This function implements the Neal's AIS.
This returns a single estimator for the chain.
"""
function ais_neal(x::Array{T}, n::Int64, 
        U₀::Potential, U₁::Potential, 
        K::Int64, βlist::Array{T}, τ::T; 
        scheme::Function=linear_scheme) where T <: AbstractFloat
    
    #u(x, β) = scheme(U₀.U(x), U₁.U(x), β)
    u(x, β) = scheme(U(U₀,x), U(U₁,x), β)
    #gradu(x, β) = scheme(U₀.gradU(x), U₁.gradU(x), β)
    gradu(x, β) = scheme(∇U(U₀,x), ∇U(U₁,x), β)
    
    accept = zeros(T,K)
    G = T(1.0)
    for ℓ = 1:K
        G *= exp(-u(x, βlist[ℓ+1]) + u(x, βlist[ℓ]))
        x, accept[ℓ] = MALA_OLD(n, u, gradu, τ, x, βlist[ℓ+1])
    end
    
    return G, accept
end
