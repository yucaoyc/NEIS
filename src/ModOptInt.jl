module ModOptInt

using Distributed
using LinearAlgebra
using Statistics
using Printf
using Random

using ModPotential: Potential
using ModODEIntegrate: RK4, time_integrate
using ModQuadratureScheme: Trapezoidal
using ModUtil: remove_nan, get_relative_stats, integrate_over_time
using ModDyn: Dyn,
    DynNN,
    vectorize,
    ∇b, divg_b, grad_divg_b,
    grad_b_wrt_para, grad_divg_wrt_para,
    update_flow_para!
using ModOpt

export one_path_dyn_b,
       stat_optimize_dyn_b,
       train_NN,
       optimize_b_dyn,
       one_path_optimize_dyn_b,
       test_func_train

"""
    This function generates a sample of the estimator.

    Remarks:
    * T₋  = -offset*h
"""

function one_path_dyn_b(n::Int64, para::Dyn,
    test_func::AbstractArray{}, test_func_para::AbstractArray{},
    N::Int64, offset::Int64, init_func, init_func_arg, solver; verbose=false, shiftprotect=true)

    X₀ = init_func(init_func_arg)
    verbose ? println(X₀) : nothing

    h = 1/N
    Vf, _ = time_integrate((x,t,p)-> p.f(x,p.para_list...), para,
                X₀, 0.0, 1.0, N, solver, test_func, test_func_para)
    Vb, _ = time_integrate((x,t,p)->(-1)*p.f(x,p.para_list...), para,
                X₀, 0.0, 1.0, N, solver, test_func, test_func_para)
    Value = hcat((reverse(Vb[1:2,:], dims=2))[:,1:N], Vf[1:2,:])

    Jaco = integrate_over_time(N, h, Vf[3,:], Vb[3,:])
    if ~shiftprotect
        F₀ = exp.(Value[1,:] .+ Jaco)
        F₁ = exp.(Value[2,:] .+ Jaco)
    else
        F₀ = Value[1,:] .+ Jaco
        F₁ = Value[2,:] .+ Jaco
        shift = maximum(F₀) #sum(F₀)/length(F₀)
        F₀ = F₀ .- shift
        F₁ = F₁ .- shift
        F₀ = exp.(F₀)
        F₁ = exp.(F₁)
    end

    # todo: a small speed-up is possible for computing B. Todo later.
    # trapezoidal rule
    B = [Trapezoidal(F₀[(j+offset-N):(j+offset)],h)
         for j in (N+1-offset):(2*N+1-offset)]
    F₁B = F₁[(N+1-offset):(2*N+1-offset)]./B

    return Trapezoidal(F₁B, h), F₀, F₁, Jaco

end

"""
    Return relative error and variance for statistics.
"""

function get_data_err_var(n::Int64, U₀::Potential, U₁::Potential,
        para::Dyn, N::Int64, tm::Float64, numsample::Int64; solver=RK4, shiftprotect=true)

    offset = Int64(abs(round(N*tm)))

    test_func = [(x,t,p)->-U₀.U(x), (x,t,p)->-U₁.U(x), (x,t,p)-> divg_b(p, x)]
    test_func_para = [nothing nothing para]

    init_func = U₀.sampler
    data = pmap(j->one_path_dyn_b(n, para, test_func, test_func_para, N, offset, init_func, j, solver, shiftprotect=shiftprotect)[1],
                    1:numsample)
    data, _ = remove_nan(data) # remove NaN

    m = mean(data)
    m2 = mean(data.^2)
    return data, m, m2 - m^2
end


"""
    optimize_b_dyn(XY::Array{Float64}, flow::DynNN, p_idx::Int64)

    This function implements ̇the following dyanmics:
        Xₜ = b(Xₜ), ̇Y_t = ∇b(X_t) Y_t + δb(X_t), δb = db/dθ.

    Input:
    * XY is the state of [Xₜ, Y₁ₜ, Y₂ₜ, ⋯, Yₖₜ], where k is the number of parameters;
    * flow is the neural network representation of b(X);
    * p_idx is the index of the parameter to train.

    Output:
    * the time derivative [̇X_t, ̇Y₁ₜ, ⋯, ̇Yₖₜ].
"""

function optimize_b_dyn(XY::Array{Float64}, flow::Dyn, num_para, M)

    dim = flow.dim
    x = XY[:,1]
    dXY = zeros(dim, 1 + M)

    # first column
    dXY[:,1] = flow.f(x, flow.para_list...)

    # other columns
    grad_b = ∇b(flow, x)
    for k = 2:(1 + M)
        dXY[:,k] = grad_b*XY[:,k]
    end

    left_idx = 1
    right_idx = 1
    for i = 1:length(num_para)
        left_idx = right_idx + 1
        right_idx += num_para[i]
            # todo: this should be upgraded.
        dXY[:, left_idx:right_idx] .+= grad_b_wrt_para(flow, x, flow.train_para_idx[i])
    end
    return dXY
end

"""
    one_path_optimize_dyn_b

    It gives a single trajectory estimation for Z₁/Z₀ and
        the derivatives of the secomd moment with respect to parameters.

"""

function one_path_optimize_dyn_b(flow::Dyn, test_func, test_func_para,
        N::Int64, offset::Int64, U₀::Potential, init_sampler_arg, num_para, M, solver; verbose=false, shiftprotect=true)

    n = flow.dim
    h = 1/N
    
    # initialize a particle.
    X₀ = U₀.sampler(init_sampler_arg)
    verbose ? println(X₀) : nothing
    
    Vf,_ = time_integrate((x,t,p) -> optimize_b_dyn(x,p,num_para,M), flow,
        hcat(X₀,zeros(n, M)),
        0.0, 1.0, N, solver,
        test_func, test_func_para, 3*(1+M))
    Vb,_ = time_integrate((x,t,p) -> (-1)*optimize_b_dyn(x,p,num_para,M), flow,
        hcat(X₀,zeros(n, M)),
        0.0, 1.0, N, solver,
        test_func, test_func_para, 3*(1+M))
    Value = hcat((reverse(Vb, dims=2))[:,1:N], Vf)

    # Update F
    Jaco = integrate_over_time(N, h, Vf[3,:], Vb[3,:])
    if ~shiftprotect
        F₀ = exp.(Value[1,:] .+ Jaco)
        F₁ = exp.(Value[2,:] .+ Jaco)
    else
        F₀ = Value[1,:] .+ Jaco
        F₁ = Value[2,:] .+ Jaco
        shift = maximum(F₀) #sum(F₀)/length(F₀)
        F₀ = F₀ .- shift
        F₁ = F₁ .- shift
        F₀ = exp.(F₀)
        F₁ = exp.(F₁)
    end

    # update G₀ and G₁.
    G₀ = zeros(M, 2*N+1)
    G₁ = zeros(M, 2*N+1)
    for i = 1:M
        J = integrate_over_time(N, h, Vf[3+3*i,:], Vb[3+3*i,:])
        G₀[i,:] = Value[3*i+1,:] + J
        G₁[i,:] = Value[3*i+2,:] + J
        G₀[i,:] = G₀[i,:].*F₀
        G₁[i,:] = G₁[i,:].*F₁
    end

    # get estimator
    TmTp = (N+1-offset):(2*N+1-offset) # range for T₋ to T₊
    B = [Trapezoidal(F₀[(j+offset-N):(j+offset)],h) for j in TmTp]
    estimator = Trapezoidal(F₁[TmTp]./B, h)

    # derivatives
    deri = zeros(M)
    for i = 1:M
        Int_G₀ = [Trapezoidal(G₀[i,(j+offset-N):(j+offset)], h) for j in TmTp]
        tmp₁ = Trapezoidal(G₁[i,TmTp]./B, h)
        tmp₂ = Trapezoidal(F₁[TmTp].*Int_G₀./(B.^2), h)
        deri[i] = 2*estimator*( tmp₁ - tmp₂ )
    end

    return vcat(estimator, deri)
end


"""
    test_func_train(...) is a generic template for generating test functions during training.
"""

function test_func_train(XY, p, U₀, U₁, num_para, M)

    x = XY[:,1]
    ϕ = zeros(3*(1 + M))
    ϕ[1:3] = [-U₀.U(x), -U₁.U(x), divg_b(p, x)]

    ∇U₀ = U₀.gradU(x)
    ∇U₁ = U₁.gradU(x)
    graddivgb = grad_divg_b(p, x) # the term ∇(∇⋅b)

    left_idx = 1
    # todo: this should be upgraded.
    for k = 1:length(num_para)
        divgbdθ = grad_divg_wrt_para(p, x, p.train_para_idx[k]) # the term ∇⋅(δb)
        for i = 1:num_para[k]
            ϕ[(3*(i+left_idx)-2):(3*(i+left_idx))] = [-dot(∇U₀, XY[:,left_idx+i]),
                                                      -dot(∇U₁, XY[:,left_idx+i]),
                                                dot(graddivgb, XY[:,left_idx+i]) + divgbdθ[i] ]
        end
        left_idx += num_para[k]
    end
    return ϕ

end

"""
    stat_optimize_dyn_b(U₀::Potential, U₁::Potential, flow::DynNN, N::Int64,
        p_idx::Int64, offset::Int64, numsample::Int64; verbose=false)

    This function returns first and second order moment of estimator and derivative with respect to
        the parameter flow.para_list[p_idx].

    Input:
        * U₀ and U₁ are potential functions for proposal and target.
        * flow is a data structure to represent deterministic dynamics b.
        * N is the number of grid points in time
        * p_idx is the index for the variable
        * offset encodes the starting time T₋: the relation is that T₋ = -offset*(1/N)
        * numsample is the sample size
        * init_func is a function to generate a random variable according to ρ₀

    Output: (fst_m, sec_m) where
        * fst_m: first moment of A, ∂(A^2)/∂θ₁, ∂(A^2)/∂θ₂, ⋯,  ∂(A^2)/∂θₗ where l is the number of parameters
        * sec_m: second moment of the above
        * A ≡ A^(∞) is the estimator herein (see paper)
"""

function stat_optimize_dyn_b(U₀::Potential, U₁::Potential, flow::Dyn, N::Int64,
        offset::Int64, numsample::Int64, solver; verbose=false)

    num_para = [length(θ) for θ in flow.para_list[flow.train_para_idx]]
    M = sum(num_para)

    
    test_func = (x,t,p) -> test_func_train(x, p, U₀, U₁, num_para, M)
    test_func_para = flow

    data = pmap(j-> one_path_optimize_dyn_b(flow, test_func, test_func_para, N,
                                        offset, U₀, j, num_para, M, solver, verbose=verbose),
        1:numsample)

    # todo: add nan protection
    fst_m = zeros(1 + M)
    sec_m = zeros(1 + M)
    for j = 1:numsample
        fst_m .+= data[j]
        sec_m .+= data[j].^2
    end
    fst_m /= numsample
    sec_m /= numsample

    if verbose
        println(fst_m)
        println(sec_m)
    end
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
        offset::Int64=0,
        biased_func=nothing,
        solver=RK4,
        ref_value::Float64=1.0,
        seed::Int64=-1, verbose=false, printpara=false,
        ρ=0.5, c=0.5, max_search_iter=10, sample_num_repeat=1, 
        test_data=false, numsample_min=nothing, savepara=false)

    (offset < 0 || offset > N) ? error("Incorrect offset") : nothing

    tm = offset/N
    loss_func(m)=get_data_err_var(flow.dim, U₀, U₁, flow, N, tm, m, solver=solver)[3]
    stat_opt_func(m) = stat_optimize_dyn_b(U₀, U₁, flow, N, offset, m, solver)

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
