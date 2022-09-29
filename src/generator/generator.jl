export integrate_F_half_traj
export find_τ

"""
    Suppose p is a fixed flow with ∇ ⋅ 𝐛 = ρdiff ≡ ρ₁ - ρ₀.
    This function gives the dynamics to run Xₜ and Jₜ.
"""
function ode_flow_with_jaco(x::Array{T}, t::T, p::DynFix, ρdiff::Function) where T<:AbstractFloat
    n = p.dim
    v = zeros(T,n+1)
    v[1:n] = p.f(x[1:n])
    v[n+1] = ρdiff(x[1:n])*x[n+1]
    return v
end


"""
    Suppose p is a generic fixed flow.
    This function gives the dynamics to run Xₜ and Jₜ.
"""
function ode_flow_with_jaco(x::Array{T}, t::T, p::Dyn) where T<:AbstractFloat
    n = p.dim
    v = zeros(n+1)
    v[1:n] = p.f(x[1:n])
    v[n+1] = divg_b(p, x[1:n])*x[n+1]
    return v
end

"""
Compute ∫ ρₖ(Xₜ(x)) Jₜ(x) dt on t∈ (-∞,0] or [0,∞) for k = 0, 1.
x₀: the initial state at time t=0.
maxT: the time for truncation (to avoid ∞)
N: the number of time discretization grid points
direction: :forward or :backward
    - :forward means integral on the interval [0,∞).
    - :backward means integral on the interval (-∞,0].
flow: the flow for Xₜ.

! Remark: we assume flow 𝐛 solves ∇ ⋅ 𝐛 = ρ₁ - ρ₀ = ρdiff !

"""
function integrate_F_half_traj(x₀::Array{T}, maxT::Union{T,Int}, N::Int,
        direction::Symbol, flow::Dyn,
        ρ₀::Function, ρ₁::Function, ρdiff::Function) where T <: AbstractFloat

    h = maxT/N
    n = flow.dim

    test_func = [(x,t,p)->ρ₀(x[1:n])*x[n+1], (x,t,p)->ρ₁(x[1:n])*x[n+1]] # Remark x[n+1]=Jₜ in ODE form.
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
For a zero-variance dynamics stored in flow, we find the time map τ
such that 𝐓(x) := X_{τ(x)}(x) is a push-forward map from ρ₀ to ρ₁.

Return the time τ we find as well as X_τ, J_τ, i.e., τ, X_τ, J_τ as a tuple.
When full_info = true, an error arising from discretization and the choice of maxT will be reported:
i.e., we return error, τ, X_τ, J_τ.

Remark: this implementation might be sensitive to the extent that the flow is a zero-variance dynamics.
If the estimate about the flow (an almost zero-variance dynamics) is not accurate,
then this function is longer applicable.
"""
function find_τ(maxT::Number, N::Int, x₀::Array, flow::Dyn,
        ρ₀::Function, ρ₁::Function, ρdiff::Function;
        full_info=false, # whether to include error information.
        printwarn=false)

    n = flow.dim
    f₀, f₁, _, _ = integrate_F_half_traj(x₀, maxT, N, :forward, flow, ρ₀, ρ₁, ρdiff)
    fb₀, fb₁, _, _ = integrate_F_half_traj(x₀, maxT, N, :backward, flow, ρ₀, ρ₁, ρdiff)
    int₀ = f₀ + fb₀
    int₁ = f₁ + fb₁

    err = abs(int₁/int₀ - 1)
    if err > 1.0e-2 && printwarn
        printstyled("Time length may not be long enough in findT function.\n"; color = :yellow)
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
    τ = maxT

    for i = 1:Nmax
        xinit = RK4(ode, xinit, (i-1)*h, h, flow)
        acc_v += ρ₁(xinit[1:n])*xinit[n+1]*h
        if acc_v >= v
            τ = i*h
            break
        end
    end

    if direction == :backward
        τ *= (-1)
    end

    if full_info == false
        return τ, xinit[1:n], xinit[n+1]
    else
        return err, τ, xinit[1:n], xinit[n+1]
    end
end
