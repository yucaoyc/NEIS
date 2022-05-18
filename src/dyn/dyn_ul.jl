export DynUL, init_DynUL

mutable struct DynUL <: Dyn
    # must-have entries
    dim
    para_list
    train_para_idx
    f
    dyn_type
    num_para
    total_num_para
    # optional part
    Ω
    U
    dimhalf
    model
end

"""
    Generate a ULD ansatz for training
    
    dqₜ = a pₜ
    dpₜ = -m ∇U(qₜ) - γ pₜ
    

    x = [qₜ, pₜ]
    m represents the inverse mass
    
"""
function init_DynUL(U, γ, Ω; a=1.0, m=1.0, train_all=true)
    
    if U.gradU == nothing || U.HessU == nothing
        warn("U must have nontrivial fields gradU and HessU.") 
        return
    end
    
    n = U.dim
    dim = 2*n
    para_list = [[γ], [a], [m]]
    
    if train_all 
        train_para_idx = [1, 2, 3]
    else
        train_para_idx = [1]
    end
    
    function f(x, args...)
        if Ω(x)
            γ = args[1][1]
            a = args[2][1]
            m = args[3][1]

            q = x[1:n]
            p = x[(n+1):end]
            return vcat(a*p, -m*U.gradU(q) - γ*p)
        else
           return zeros(dim) 
        end
    end
    
    dyn_type = 4
    
    if train_all 
        num_para = [1, 1, 1]
        total_num_para = 3
    else
        num_para = [1]
        total_num_para = 1
    end
    
    model(x) = f(x, para_list...)
    
    return DynUL(dim, para_list, train_para_idx, f, dyn_type, num_para, total_num_para, Ω, U, n, model)
    
end

function ∇b(b::DynUL, x::Array)
    if b.Ω(x)
        d = b.dimhalf
        γ = b.para_list[1][1]
        a = b.para_list[2][1]
        m = b.para_list[3][1]
        q = x[1:d]
        return vcat(hcat(zeros(d,d), a*Matrix(1.0I,d,d)), hcat(-m*b.U.HessU(q), -γ*Matrix(1.0I,d,d)))
    else
        return zeros(b.dim, b.dim)
    end
end

function divg_b(b::DynUL, x::Array)
    if b.Ω(x)
        γ = b.para_list[1][1]
        return -γ*b.dimhalf
    else
        return 0.0
    end
end

function grad_divg_b(b::DynUL, x::Array)
   return zeros(b.dim)
end

function grad_b_wrt_para(b::DynUL, x::Array, i::Int64)

    v = zeros(b.dim,1)
    if b.Ω(x)
        n = b.dimhalf
        if i == 1
            v[(n+1):end,1] = -x[(n+1):end]
        elseif i == 2
            v[1:n,1] = x[(n+1):end]
        elseif i == 3
            v[(n+1):end,1] = -b.U.gradU(x[1:n])
        else
            error("i must be 1, 2, or 3.")
        end
    end
    return v
end

function grad_b_wrt_para(b::DynUL, x::Array)
    v1 = grad_b_wrt_para(b, x, 1)
    v2 = grad_b_wrt_para(b, x, 2)
    v3 = grad_b_wrt_para(b, x, 3)
    return hcat(v1, v2, v3)
end

function grad_divg_wrt_para(b::DynUL, x::Array, i::Int64)
    if i == 1 && b.Ω(x)
        return [-b.dimhalf]
    else
        return [0.0]
    end
end

function grad_divg_wrt_para(b::DynUL, x::Array)
    if b.Ω(x)
        return [-b.dimhalf, 0.0, 0.0]
    else
        return zeros(3)
    end
end
