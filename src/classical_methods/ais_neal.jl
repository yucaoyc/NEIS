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
function ais_neal(x::Array{T}, n::Int,
        U₀::Potential{T}, U₁::Potential{T},
        K::Int, βlist::Array{T}, τ::T;
        scheme::Function=linear_scheme) where T <: AbstractFloat

    #u(x, β) = scheme(U₀(x), U₁(x), β)
    #gradu(x, β) = scheme(∇U(U₀,x), ∇U(U₁,x), β)

    accept = zeros(T,K)
    G = T(1.0)

    # initial preparation.
    u1x = U₁(x)
    dU1x = ∇U(U₁, x)

    for ℓ = 1:K
        u0x = U₀(x)
        u_bef = scheme(u0x, u1x, βlist[ℓ])
        u_aft = scheme(u0x, u1x, βlist[ℓ+1])
        G *= exp(-u_aft + u_bef)
        # x, accept[ℓ] = MALA_OLD(n, u, gradu, τ, x, args=βlist[ℓ+1], uxargs=u_aft)

        # we replace the MALA_OLD via more specialized codes to reduce query costs.
        args = βlist[ℓ+1]; uxargs = u_aft;
        ξ = randn(T,n)
        xnew = x - τ*scheme(∇U(U₀,x), dU1x, args) + sqrt(2*τ)*ξ
        u1x_new = U₁(xnew); dU1x_new = ∇U(U₁, xnew) # queries to ∇U₁ and U₁ herein.
        r = exp(-scheme(U₀(xnew), u1x_new, args) + uxargs)
        r *= exp(-norm(x - (xnew - τ*scheme(∇U(U₀,xnew), dU1x_new, args)))^2/(4*τ))/exp(-norm(ξ)^2/2)
        a = min(T(1.0), r)

        if rand() < a
            # accept
            x = xnew; accept[ℓ] = 1
            u1x = u1x_new; dU1x = dU1x_new;
        else
            # reject
            accept[ℓ] = 0
        end
    end

    return G, accept
end

function ais_neal(U₀::Potential{T},
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
    return data, m, msec - m^2
end
