export ais_neal, linear_scheme, get_data_err_var_ais_neal

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
function ais_neal(x::Array{T}, n::Int,
        U₀::Potential{T}, U₁::Potential{T},
        K::Int, βlist::Array{T}, τ::T;
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


function get_data_err_var_ais_neal(U₀::Potential{T},
        U₁::Potential{T},
        K::Int, numsample::Int;
        τ::Union{T,Nothing}=nothing,
        fixed_sampler_func=nothing) where T<:AbstractFloat

    if τ == nothing
        τ = T(1/K)
    end

    n = U₀.dim
    βlist = Array(range(0, stop=T(1.0), length=K+1))
    if fixed_sampler_func == nothing
        init_func = j->sampler(U₀,1)[:]
    else
        init_func = j->fixed_sampler_func(j)
    end
    data = zeros(T, numsample)
    Threads.@threads for j = 1:numsample
        data[j] = ais_neal(init_func(j), n, U₀, U₁, K, βlist, τ)[1]
    end
    m = mean(data)
    msec = mean(data.^2)
    return m, msec - m^2
end
