export backward_aug,
       compute_estimator,
       backward_aug_ODE_for_grad,
       backward_aug_for_grad,
       ODE_for_estimator_and_grad,
       compute_estimator_and_gradient,
       get_data_err_var,
#       get_rela_err_var,
       stat_optimize_dyn_b,
       train_NN_ode

function backward_aug(flow::Dyn, x0::Vector{T}, U₀::Potential{T}, N::Int;
        solver=RK4) where T<:AbstractFloat

    f = (x,t,p) -> (-p(x[1]), -divg_b(p, x[1])*x[2], exp(-U₀(x[1]))*x[2])

    result = time_integrate(f, flow,
                (x0, T(1.0), T(0.0)),
                T(0.0), T(1.0), N, solver)
    return result
end

function compute_estimator(flow::Dyn, x0::Vector{T}, U₀::Potential{T}, U₁::Potential{T},
        N::Int; solver=RK4) where T<:AbstractFloat

    aug_result = backward_aug(flow, x0, U₀, N, solver=RK4)
    f = (x,t,p) -> (p(x[1]),
        divg_b(p, x[1])*x[2],
        p(x[3]),
        divg_b(p, x[3])*x[4],
        exp(-U₀(x[1]))*x[2] - exp(-U₀(x[3]))*x[4],
        exp(-U₁(x[1]))*x[2]/x[5])
    result = time_integrate(f, flow,
                            (x0, T(1.0), aug_result[1], aug_result[2], aug_result[3], T(0.0)),
                            T(0.0), T(1.0), N, solver)
    return result[6]
end

function backward_aug_ODE_for_grad(x::Tuple, p::Dyn,
        U₀::Potential{T}) where T<:AbstractFloat

    X, J, B, g, H, L, Y = x

    newX = (-1)*p(X)
    newJ = (-1)*divg_b(p, X)*J
    newB = exp(-U₀(X))*J
    newg = newB*(-(∇U(U₀,X)'*Y)' .+ H .+ L)
    nabla_divg_b = grad_divg_b(p, X) #∇ (∇⋅b)
    newH = (-1)*(nabla_divg_b'*Y)'
    newL = (-1)*grad_divg_wrt_para(p, X)
    newY = (-1)*((∇b(p, X))*Y .+ grad_b_wrt_para(p, X))

    return (newX, newJ, newB, newg, newH, newL, newY)

end

function backward_aug_for_grad(flow::Dyn, x0::Vector{T},
        U₀::Potential{T}, N::Int; solver=RK4) where T<:AbstractFloat

    f = (x,t,p) -> backward_aug_ODE_for_grad(x, p, U₀)
    m = flow.total_num_para
    state = (x0, T(1.0), T(0.0), zeros(T, m), zeros(T, m), zeros(T, m), zeros(T, flow.dim, m))
    result = time_integrate(f, flow, state, T(0.0), T(1.0), N, solver)
    return result
end


function ODE_for_estimator_and_grad(x::Tuple, p::Dyn,
        U₀::Potential{T}, U₁::Potential{T}) where T<:AbstractFloat

    X, J, Xlag, Jlag, B, A, D, g, L, Llag, H, Hlag, Y, Ylag = x

    ρ0 = exp(-U₀(X))
    ρ1 = exp(-U₁(X))
    ρ0lag = exp(-U₀(Xlag))

    newX = p(X)
    newJ = divg_b(p, X)*J

    newXlag =  p(Xlag)
    newJlag = divg_b(p, Xlag)*Jlag

    newB = ρ0*J - ρ0lag*Jlag
    newA = ρ1*J/B

    newD = ρ1*J*(-(∇U(U₁,X)'*Y)' .+ H .+ L)/B - ρ1*J*g/B^2

    newg = ρ0*J*(-(∇U(U₀,X)'*Y)' .+ H .+ L) - ρ0lag*Jlag*(-(∇U(U₀,Xlag)'*Ylag)' .+ Hlag .+ Llag)

    newL = grad_divg_wrt_para(p, X)
    newLlag = grad_divg_wrt_para(p, Xlag)

    nabla_divg_b = grad_divg_b(p, X)
    newH = (nabla_divg_b'*Y)'

    nabla_divg_b_lag = grad_divg_b(p, Xlag)
    newHlag = (nabla_divg_b_lag'*Ylag)'

    newY = (∇b(p, X))*Y .+ grad_b_wrt_para(p, X)
    newYlag = (∇b(p, Xlag))*Ylag .+ grad_b_wrt_para(p, Xlag)

    return (newX, newJ, newXlag, newJlag, newB, newA, newD,
            newg, newL, newLlag, newH, newHlag, newY, newYlag)
end

function compute_estimator_and_gradient(flow::Dyn, x0::Vector{T},
        U₀::Potential{T}, U₁::Potential{T}, N::Int; solver=RK4) where T<:AbstractFloat

    backward_result = backward_aug_for_grad(flow, x0, U₀, N, solver=solver)
    Xlag, Jlag, Blag, glag, Hlag, Llag, Ylag = backward_result
    m = flow.total_num_para

    state = (x0, T(1.0), Xlag, Jlag, Blag, T(0.0),
        zeros(T,m), glag, # D, g
        zeros(T,m), Llag, # L
        zeros(T,m), Hlag, # H
        zeros(T, flow.dim, m), Ylag) # Y
    f = (x,t,p) -> ODE_for_estimator_and_grad(x, p, U₀, U₁)
    result = time_integrate(f, flow, state, T(0.0), T(1.0), N, solver)
    X, J, _, _, _, A, D, _, _, _, _, _, _, _ = result
    return X, J, A, D
end


function get_data_err_var(U₀::Potential{T}, U₁::Potential{T},
        flow::Dyn, N::Int, numsample::Int;
        ϕname="msq", ϕϵ=T(1.0e-3),
        fixed_sampler_func=nothing,
        solver=RK4,
        ptype="threads") where T<:AbstractFloat

    ϕ = get_Φ(ϕname, ϕϵ)

    if fixed_sampler_func == nothing
        # the default way to generate sample
        init_func = j->sampler(U₀,1)[:]
    else
        # if one wants to use a fix collection of samples, e.g., denoted as pts,
        # then we can simply choose fixed_sampler_func = j-> pts[j]
        init_func = j->fixed_sampler_func(j)
    end

    if ptype == "pmap"
        data = pmap(j->compute_estimator(flow, init_func(j), U₀, U₁,
                                         N; solver=solver), 1:numsample)
    else
        data = zeros(T, numsample)
        Threads.@threads for j = 1:numsample
            data[j] = compute_estimator(flow, init_func(j), U₀, U₁, N; solver=solver)
        end
    end
    data, percent = remove_nan(data) # remove NaN
    if percent > 1.0e-4
        @warn(@sprintf("%10.2E percent of data is removed due to NaN", percent))
    end

    m = mean(data)
    #m2 = mean(data.^2)
    m2 = mean(ϕ.f(data))
    #return data, m, m2 - m^2
    return data, m, m2 - ϕ.f(m)
end

#function get_rela_err_var(U₀::Potential{T},
#        U₁::Potential{T}, flow::Dyn,
#        N::Int64, numsample::Int, exact_mean::T;
#        fixed_sampler_func=nothing,
#        solver=RK4,
#        ptype="threads") where T<:AbstractFloat
#
#    _, m, var = get_data_err_var(U₀, U₁, flow, N, numsample,
#                                fixed_sampler_func=fixed_sampler_func,
#                                solver=solver,
#                                ptype=ptype)
#    return abs(m/exact_mean - 1), abs(var/exact_mean^2)
#end

function stat_optimize_dyn_b(U₀::Potential{T},
        U₁::Potential{T}, flow::Dyn,
        N::Int, numsample::Int;
        ϕname="msq", ϕϵ::T=T(1.0e-3),
        fixed_sampler_func=nothing,
        solver=RK4,
        ptype="threads") where T<:AbstractFloat

    ϕ = get_Φ(ϕname, ϕϵ)

    if fixed_sampler_func == nothing
        # the default way to generate sample
        init_func = j->sampler(U₀,1)[:]
    else
        # if one wants to use a fix collection of samples, e.g., denoted as pts,
        # then we can simply choose fixed_sampler_func = j-> pts[j]
        init_func = j->fixed_sampler_func(j)
    end

    if ptype=="pmap"
        data = pmap(j->compute_estimator_and_gradient(flow,
                  init_func(j), U₀, U₁, N, solver=solver)[3:4], 1:numsample)
        # V = [vcat(item[1], 2*item[1]*item[2]) for item in data]
        V = [vcat(item[1], ϕ.fderi(item[1])*item[2]) for item in data]
    else
        data = [zeros(T,1+flow.total_num_para) for j=1:numsample]
        Threads.@threads for j = 1:numsample
            item = compute_estimator_and_gradient(flow,
                init_func(j), U₀, U₁, N, solver=solver)[3:4]
            # data[j] = vcat(item[1], 2*item[1]*item[2])
            data[j] = vcat(item[1], ϕ.fderi(item[1])*item[2])
        end
    end
    # return get_mean_sec_moment(data)
    return get_mean_entropy(data, ϕ)
end


"""
    train_NN
"""
function train_NN_ode(U₀::Potential{T}, U₁::Potential{T},
        flow::DynTrain{T},
        N::Int, numsample_max::Int,
        train_step::Int,
        h::T, decay::T,
        gpts::AbstractMatrix{T}, gpts_sampler::Union{Function,Nothing};
        ϕname="msq", ϕϵ=T(1.0e-3),
        biased_func::Union{Function,Nothing}=nothing,
        solver::Function=RK4,
        ref_value::T=T(1.0),
        seed::Int=-1,
        verbose::Bool=false,
        printpara::Bool=false,
        ρ::T=T(0.5), c::T=T(0.5),
        to_normalize::Bool=true,
        max_search_iter::Int=10,
        sample_num_repeat::Int=1,
        print_sample::Bool=false,
        numsample_min::Int=-1,
        savepara::Bool=false,
        ptype="threads",
        showprogress=true,
        compute_rela_divg=true) where T<:AbstractFloat

    ϕ = get_Φ(ϕname, ϕϵ)
    fixed_sampler_func = j->gpts[:,j]
    stat_opt_func(m) = stat_optimize_dyn_b(U₀, U₁, flow, N, m,
                            ϕname=ϕname, ϕϵ=ϕϵ,
                            fixed_sampler_func=fixed_sampler_func,
                            solver=solver, ptype=ptype)
    loss_func(m) = get_data_err_var(U₀, U₁, flow, N, m,
                            ϕname=ϕname, ϕϵ=ϕϵ,
                            fixed_sampler_func=fixed_sampler_func,
                            solver=solver, ptype=ptype)[3]

    return train_NN_template(stat_opt_func, loss_func,
                      flow, numsample_max, train_step,
                      h, decay, gpts, gpts_sampler, ϕ,
                      biased_func=biased_func,
                      ref_value=ref_value, seed=seed,
                      verbose=verbose, printpara=printpara,
                      ρ=ρ, c=c,
                      to_normalize=to_normalize,
                      max_search_iter=max_search_iter,
                      sample_num_repeat=sample_num_repeat,
                      print_sample=print_sample,
                      numsample_min=numsample_min,
                      savepara=savepara,
                      showprogress=showprogress,
                      compute_rela_divg=compute_rela_divg)
end
