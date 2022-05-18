export funnelexpansatz, init_funnelexpansatz

mutable struct funnelexpansatz <: Dyn
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
    model
end

function ∇b(b::funnelexpansatz, x)
    n = b.dim
    if b.Ω(x)
        T, a = b.para_list[1]
        return vcat(vcat(zeros(1,n)), hcat(zeros(n-1,1), -a*Matrix(1.0I,n-1,n-1)))
    else
        return zeros(n, n)
    end
end


function divg_b(b::funnelexpansatz, x)
    if b.Ω(x)
        n = b.dim
        T, a = b.para_list[1]
        return -a*(n-1)
    else
        return 0.0
    end
end

function grad_divg_b(b::funnelexpansatz, x)
   return zeros(b.dim)
end


function grad_b_wrt_para(b::funnelexpansatz, x)
    n = b.dim
    if b.Ω(x)
        T, a = b.para_list[1]
        v1 = vcat([-1], zeros(n-1))
        v2 = vcat([0], -x[2:end])
        return hcat(v1, v2)
    else
        return zeros(n,2)
    end
end

function grad_divg_wrt_para(b::funnelexpansatz, x)
    if b.Ω(x)
        n = b.dim
        T, a = b.para_list[1]
        return [0.0, -(n-1)]
    else
        return zeros(2)
    end
end

function grad_b_wrt_para(b::funnelexpansatz, x, i)
    if i == 1
        return grad_b_wrt_para(b, x)
    else
        error("error")
    end
end

function grad_divg_wrt_para(b::funnelexpansatz, x, i)
   if i == 1
        return grad_divg_wrt_para(b, x)
    else
        error("error")
    end
end

function init_funnelexpansatz(n, T, a, Ω)

    ## Define customized gradient functions
    function funnelexpansatz_f(x, args...)
        T, a = args[1]
        if Ω(x)
            return vcat([-T], -a*x[2:end])
        else
            return zeros(n)
        end
    end

    para_list = [[T, a]]
    train_para_idx = [1]
    dyn_type = 5
    num_para = [2]
    total_num_para = 2
    model(x) = funnelexpansatz_f(x, para_list...) 

    flow = funnelexpansatz(n, para_list, train_para_idx, funnelexpansatz_f, 
                           dyn_type, num_para, total_num_para, Ω, model);
    return flow
end
