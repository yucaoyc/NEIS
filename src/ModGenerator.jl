module ModGenerator

using Documenter
using ModODEIntegrate
using ModQuadratureScheme
using ModDyn

export integrate_F_half_traj, findT

"""
Compute ``\\int \\rho_k(X_t(x)) J_t(x)\\ \\mathsf{d} t`` on integers ``(-\\infty,0]`` or ``[0,\\infty)``.
"""
function integrate_F_half_traj(x₀::Array, maxT::Union{Float64,Int64}, N::Int64, direction, flow::Dyn, ρ₀, ρ₁, ρdiff)
    
    h = maxT/N
    n = flow.dim

    test_func = [(x,t,p)->ρ₀(x[1:n])*x[n+1], (x,t,p)->ρ₁(x[1:n])*x[n+1]]
    test_func_para = Array{Any}(undef,2)
    test_func_para[1:2] = [flow, flow]
    for i = 1:(n+1)
        push!(test_func, (x,t,p)->x[i])
        push!(test_func_para, nothing)
    end
    if direction == :forward
        ode = (x,t,p)->ode_flow_with_jaco(x,t,p,ρdiff)
    else
       ode = (x,t,p)->(-1)*ode_flow_with_jaco(x,t,p,ρdiff)
    end
    func_v, state = time_integrate(ode, flow, vcat(x₀, 1.0), 0.0, maxT, N, RK4, test_func, test_func_para)
    f₀, f₁ = Trapezoidal(func_v[1:2,:], h)
    return f₀, f₁, func_v, state
end

"""
For a zero-variance dynamics stored in flow, we find the time map T 
such that ``X_{T(x)}(x)`` is a push-forward map for ``\\rho_0`` to ``\\rho_1``.
"""
function findT(maxT::Union{Float64,Int64}, N::Int64, x₀::Array, flow::Dyn, ρ₀, ρ₁, ρdiff; full_info=false, printwarn=false)
    
    n = flow.dim    
    f₀, f₁, _, _ = integrate_F_half_traj(x₀, maxT, N, :forward, flow, ρ₀, ρ₁, ρdiff)
    fb₀, fb₁, _, _ = integrate_F_half_traj(x₀, maxT, N, :backward, flow, ρ₀, ρ₁, ρdiff)
    int₀ = f₀ + fb₀
    int₁ = f₁ + fb₁
    
    err = abs(int₁/int₀ - 1)
    if err > 1.0e-2 && printwarn
        printstyled("Time length may not be long enough in findT function.\n"; color = :red)
    end

    h = maxT/N
    
    if fb₀ <= fb₁
        direction = :backward
        v = fb₁ - fb₀
        ode = (x,t,p)->(-1)*ode_flow_with_jaco(x,t,p,ρdiff) 
    else
        direction = :forward
        v = fb₀ - fb₁
        ode = (x,t,p)->ode_flow_with_jaco(x,t,p,ρdiff) 
    end
    
    xinit = vcat(x₀, 1.0)
    acc_v = 0.0 # accumulated value
    Nmax = N
    T = maxT
    
    for i = 1:Nmax
        xinit = RK4(ode, xinit, (i-1)*h, h, flow)
        acc_v += ρ₁(xinit[1:n])*xinit[n+1]*h
        if acc_v >= v
            T = i*h
            break
        end
    end

    if direction == :backward
        T *= (-1)
    end

    if full_info == false
        return T, xinit[1:n], xinit[n+1]
    else
        return err, T, xinit[1:n], xinit[n+1]
    end
end


end
