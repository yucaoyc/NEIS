export MALA_OLD, MALA_OLD_chain

"""
Metroplis-Adjusted Langevin algorithm associated with overdamped Langevin dynamics.
This function returns the new state and acceptance decision.

Input:
n: dimension
u: potential function
gradu: gradient function
τ: time step
x: initial state (a single input state only)
args: arguments inside potential and gradient functions.

Reference: 
- https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm
"""
function MALA_OLD(n::Int64, u::Function, gradu::Function, 
        τ::T, x::Array{T,1}, args...) where T <: AbstractFloat
    
    ξ = randn(T,n)
    xnew = x - τ*gradu(x, args...) + sqrt(2*τ)*ξ
    
    r = exp(-u(xnew, args...) + u(x, args...))
    #r *= exp(-1/(4*τ)*norm(x - (xnew - τ*gradu(xnew, args...)))^2)/exp(-norm(ξ)^2/2)
    r *= exp(-norm(x - (xnew - τ*gradu(xnew, args...)))^2/(4*τ))/exp(-norm(ξ)^2/2)
    a = min(T(1.0), r)
    
    if rand() < a
        # accept
        return xnew, 1
    else
        # reject
        return x, 0
    end
end

"""
Generate a chain by MALA.
"""
function MALA_OLD_chain(n::Int64, V::Potential, 
        τ::T, chain_length::Int64, x0::Array{T,1}) where T <: AbstractFloat
    
    state = Array{Any}(undef, chain_length)
    decision = Array{Any}(undef, chain_length)
    
    state[1] = x0
    decision[1] = 1
   
    u = x->U(V,x)
    gradu = x->∇U(V,x)
    for i = 2:chain_length
        state[i], decision[i] = MALA_OLD(n, u, gradu, τ, state[i-1])
    end
    return state, decision
end
