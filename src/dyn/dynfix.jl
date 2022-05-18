export DynFix, DynFixWithDivg, generate_gradient_flow

struct DynFix <: Dyn
    dim
    f 
end

function ∇b(b::DynFix, x::VecType)
    ForwardDiff.jacobian(z -> b.f(z), x)
end

function divg_b(b::DynFix, x::VecType)
    tr(∇b(b,x))
end

mutable struct DynFixWithDivg <: Dyn
    dim
    f
    para_list
    divg_b
end

function divg_b(flow::DynFixWithDivg, x)
    flow.divg_b(x)
end

    
function generate_gradient_flow(U::Potential, T::Float64, Ω)
    n = U.dim
    grad_flow(x) = Ω(x) ? (-T)*U.gradU(x) : zeros(n)
    divg_gd(x) =  Ω(x) ? (-T)*U.LaplaceU(x) : 0.0
    flow_gd = DynFixWithDivg(n, grad_flow, [], divg_gd)
    return flow_gd
end
