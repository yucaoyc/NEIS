module ModOptODE

using LinearAlgebra
using Distributed
using ModDyn
using ModODEIntegrate
using ModPotential
using ModUtil
using Statistics
using Printf
using ModOpt

export backward_aug,
       compute_estimator,
       backward_aug_ODE_for_grad,
       backward_aug_for_grad,
       ODE_for_estimator_and_grad,
       compute_estimator_and_gradient,
       get_data_err_var,
       get_rela_err_var,
       stat_optimize_dyn_b,
       train_NN

function backward_aug(flow, x0, U₀, N; solver=RK4)
    f = (x,t,p) -> (-p.f(x[1], p.para_list...), -divg_b(p, x[1])*x[2], exp(-U₀.U(x[1]))*x[2])
    
    result = time_integrate(f, flow, (x0, 1.0, 0.0), 0.0, 1.0, N, solver)
    return result
end

function compute_estimator(flow, x0, U₀, U₁, N; solver=RK4)
    
    aug_result = backward_aug(flow, x0, U₀, N, solver=RK4)
    f = (x,t,p) -> (p.f(x[1],p.para_list...),
        divg_b(p, x[1])*x[2],
        p.f(x[3], p.para_list...),
        divg_b(p, x[3])*x[4],
        exp(-U₀.U(x[1]))*x[2] - exp(-U₀.U(x[3]))*x[4],
        exp(-U₁.U(x[1]))*x[2]/x[5])
    result = time_integrate(f, flow, (x0, 1.0, aug_result[1], aug_result[2], aug_result[3], 0.0), 
        0.0, 1.0, N, solver)
    
    return result[6]
end

function backward_aug_ODE_for_grad(x::Tuple, p::Dyn, U₀::Potential)
    X, J, B, g, H, L, Y = x

    newX = (-1)*p.f(X, p.para_list...)
    newJ = (-1)*divg_b(p, X)*J
    newB = exp(-U₀.U(X))*J
    newg = newB*(-(U₀.gradU(X)'*Y)' .+ H .+ L)
    nabla_divg_b = grad_divg_b(p, X) #∇ (∇⋅b)
    newH = (-1)*(nabla_divg_b'*Y)'
    newL = (-1)*grad_divg_wrt_para(p, X)
    newY = (-1)*((∇b(p, X))*Y .+ grad_b_wrt_para(p, X))

    return (newX, newJ, newB, newg, newH, newL, newY)

end

function backward_aug_for_grad(flow, x0, U₀, N; solver=RK4)

    f = (x,t,p) -> backward_aug_ODE_for_grad(x, p, U₀)
    m = flow.total_num_para
    state = (x0, 1.0, 0.0, zeros(m), zeros(m), zeros(m), zeros(flow.dim, m))
    result = time_integrate(f, flow, state, 0.0, 1.0, N, solver)
    return result
end


function ODE_for_estimator_and_grad(x::Tuple, p::Dyn, U₀::Potential, U₁::Potential)
    X, J, Xlag, Jlag, B, A, D, g, L, Llag, H, Hlag, Y, Ylag = x

    ρ0 = exp(-U₀.U(X))
    ρ1 = exp(-U₁.U(X))
    ρ0lag = exp(-U₀.U(Xlag))

    newX = p.f(X, p.para_list...)
    newJ = divg_b(p, X)*J

    newXlag =  p.f(Xlag, p.para_list...)
    newJlag = divg_b(p, Xlag)*Jlag

    newB = ρ0*J - ρ0lag*Jlag
    newA = ρ1*J/B

    newD = ρ1*J*(-(U₁.gradU(X)'*Y)' .+ H .+ L)/B - ρ1*J*g/B^2

    newg = ρ0*J*(-(U₀.gradU(X)'*Y)' .+ H .+ L) - ρ0lag*Jlag*(-(U₀.gradU(Xlag)'*Ylag)' .+ Hlag .+ Llag)

    newL = grad_divg_wrt_para(p, X)
    newLlag = grad_divg_wrt_para(p, Xlag)

    nabla_divg_b = grad_divg_b(p, X)
    newH = (nabla_divg_b'*Y)'

    nabla_divg_b_lag = grad_divg_b(p, Xlag)
    newHlag = (nabla_divg_b_lag'*Ylag)'

    newY = (∇b(p, X))*Y .+ grad_b_wrt_para(p, X)
    newYlag = (∇b(p, Xlag))*Ylag .+ grad_b_wrt_para(p, Xlag)

    return (newX, newJ, newXlag, newJlag, newB, newA, newD, newg, newL, newLlag, newH, newHlag, newY, newYlag)
end

function compute_estimator_and_gradient(flow, x0, U₀, U₁, N; solver=RK4)

    backward_result = backward_aug_for_grad(flow, x0, U₀, N, solver=solver)
    Xlag, Jlag, Blag, glag, Hlag, Llag, Ylag = backward_result
    m = flow.total_num_para

    state = (x0, 1.0, Xlag, Jlag, Blag, 0.0,
        zeros(m), glag, # D, g
        zeros(m), Llag, # L
        zeros(m), Hlag, # H
        zeros(flow.dim, m), Ylag) # Y
    f = (x,t,p) -> ODE_for_estimator_and_grad(x, p, U₀, U₁)
    result = time_integrate(f, flow, state, 0.0, 1.0, N, solver)
    X, J, _, _, _, A, D, _, _, _, _, _, _, _ = result
    return X, J, A, D
end


function get_data_err_var(U₀::Potential, U₁::Potential, flow::Dyn, N::Int64, numsample::Int64; solver=RK4)
    
    data = pmap(j->compute_estimator(flow, U₀.sampler(j), U₀, U₁, N; solver=solver), 1:numsample)
    data, _ = remove_nan(data) # remove NaN
    
    m = mean(data)
    m2 = mean(data.^2)
    return data, m, m2 - m^2
end

function get_rela_err_var(U₀::Potential, U₁::Potential, flow::Dyn, 
        N::Int64, numsample::Int64, exact_mean::Float64; solver=RK4)
    
    _, m, var = get_data_err_var(U₀, U₁, flow, N, numsample, solver=solver)
    return abs(m/exact_mean - 1), abs(var/exact_mean^2)

end

function stat_optimize_dyn_b(U₀, U₁, flow, N, numsample; solver=RK4)

    data = pmap(j->compute_estimator_and_gradient(flow, U₀.sampler(j), U₀, U₁, N, solver=solver)[3:4], 1:numsample)
    V = [vcat(item[1], 2*item[1]*item[2]) for item in data]
    fst_m = mean(V)
    sec_m = mean(map(x->x.^2, V))
    return fst_m, sec_m

end


"""
    train_NN
"""

function train_NN(U₀::Potential, U₁::Potential, flow::Dyn,
        N::Int64, numsample_max::Int64,
        train_step::Int64, 
        h::Float64, decay::Float64,
        gpts, gpts_sampler;
        biased_func=nothing,
        solver=RK4,
        ref_value = 1.0, 
        seed::Int64=-1, verbose=false, printpara=false, 
        ρ=0.5, c=0.5, max_search_iter=10, sample_num_repeat=1, 
        test_data=false, numsample_min=nothing, savepara=false)

    stat_opt_func(m) = stat_optimize_dyn_b(U₀, U₁, flow, N, m, solver=solver)
    loss_func(m) = get_data_err_var(U₀, U₁, flow, N, m, solver=solver)[3]

    return train_NN_template(stat_opt_func, loss_func,
                      flow, numsample_max, train_step,
                      h, decay, gpts, gpts_sampler, 
                      biased_func=biased_func,
                      ref_value=ref_value, seed=seed,
                      verbose=verbose, printpara=printpara,
                      ρ=ρ, c=c, max_search_iter=max_search_iter, 
                      sample_num_repeat=sample_num_repeat, 
                      test_data=test_data,
                      numsample_min=numsample_min, savepara=savepara)
end


end
