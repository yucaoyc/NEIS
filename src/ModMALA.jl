module ModMALA

using ModPotential
using LinearAlgebra

export MALA_OLD, MALA_OLD_chain

function MALA_OLD(n, u, gradu, τ::Float64, x, args...)
    # Metroplis-Adjusted Langevin algorithm associated with overdamped Langevin dynamics
    # return the new state and acceptance decision
    # τ is the time step
    
    ξ = randn(n)
    xnew = x - τ*gradu(x, args...) + sqrt(2*τ)*ξ
    
    r = exp(-u(xnew, args...) + u(x, args...))
    r *= exp(-1/(4*τ)*norm(x - (xnew - τ*gradu(xnew, args...)))^2)/exp(-norm(ξ)^2/2)
    a = min(1.0, r)
    
    if rand() < a
        # accept
        return xnew, 1
    else
        return x, 0
    end
end


function MALA_OLD_chain(n::Int64, U::Potential, τ::Float64, chain_length::Int64, x0)
    
    state = Array{Any}(undef, chain_length)
    decision = Array{Any}(undef, chain_length)
    
    state[1] = x0
    decision[1] = 1
    
    for i = 2:chain_length
        state[i], decision[i] = MALA_OLD(n, U.U, U.gradU, τ, state[i-1])
    end
    return state, decision
end

end
