export ode_flow_with_jaco

function ode_flow_with_jaco(x::VecType,t::Float64,p::DynFix, ρdiff)
    n = p.dim
    v = zeros(n+1)
    v[1:n] = p.f(x[1:n])
    v[n+1] = ρdiff(x[1:n])*x[n+1]
    return v
end

function ode_flow_with_jaco(x::VecType, t::Float64, p::DynNN)
    n = p.dim
    v = zeros(n+1)
    v[1:n] = p.f(x[1:n], p.para_list...)
    v[n+1] = divg_b(p, x[1:n])*x[n+1]
    return v
end
