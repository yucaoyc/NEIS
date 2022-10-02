export estimate_offset,
       one_path_dyn_b,
       get_data_err_var,
       optimize_b_dyn,
       test_func_train,
       one_path_optimize_dyn_b,
       stat_optimize_dyn_b,
       train_NN_int

# Remarks: T₋  = -offset*h
estimate_offset(tm::AbstractFloat, N::Int) = convert(typeof(N), abs(round(N*tm)))

"""
Suppose d/dt J_t = V_t. Given values of V_t, we return J_t by midpoint rule.
"""
function integrate_over_time(N::Int, h::T,
        Vf::Vector{T}, Vb::Vector{T}) where T<:AbstractFloat
    Jf = zeros(T, N+1); Jb = zeros(T, N+1)
    for k = 2:(N+1)
        Jf[k] = Jf[k-1] + (Vf[k] + Vf[k-1])/2*h
        Jb[k] = Jb[k-1] - (Vb[k] + Vb[k-1])/2*h
    end
    return vcat(reverse(Jb)[1:N], Jf)
end

"""
This function generates a sample of the estimator.
"""
function one_path_dyn_b(para::Dyn{T},
    test_func::AbstractArray{},
    test_func_para::AbstractArray{},
    N::Int, offset::Int,
    init_func::Function, init_func_arg,
    solver::Function;
#    T = Float64,
    verbose=false, shiftprotect=true) where T<:AbstractFloat

    X₀ = init_func(init_func_arg)
    verbose ? println(X₀) : nothing

    h = T(1/N)
    Vf, _ = time_integrate((x,t,p)-> p(x), para,
                           X₀, T(0.0), T(1.0), N, solver, test_func, test_func_para)
    Vb, _ = time_integrate((x,t,p)->(-1)*p(x), para,
                           X₀, T(0.0), T(1.0), N, solver, test_func, test_func_para)
    Value = hcat((reverse(Vb[1:2,:], dims=2))[:,1:N], Vf[1:2,:])

    Jaco = integrate_over_time(N, h, Vf[3,:], Vb[3,:])
    if ~shiftprotect
        F₀ = exp.(Value[1,:] .+ Jaco)
        F₁ = exp.(Value[2,:] .+ Jaco)
    else
        # as we only need the ratio of F₁/F₀,
        # we can divide the numerator and denominator
        # by the same value without affecting the conclusion
        F₀ = Value[1,:] .+ Jaco
        F₁ = Value[2,:] .+ Jaco
        shift = maximum(F₀)
        F₀ = F₀ .- shift
        F₁ = F₁ .- shift
        F₀ = exp.(F₀)
        F₁ = exp.(F₁)
    end

    # todo: a minor speed-up is possible for computing B. Todo later.
    # trapezoidal rule
    B = [Trapezoidal(F₀[(j+offset-N):(j+offset)],h)
         for j in (N+1-offset):(2*N+1-offset)]
    F₁B = F₁[(N+1-offset):(2*N+1-offset)]./B

    return Trapezoidal(F₁B, h), F₀, F₁, Jaco
end

"""
Return relative error and variance for statistics.
"""
function get_data_err_var(U₀::Potential{T}, U₁::Potential{T},
        para::Dyn,
        N::Int64,
        offset::Int,
        numsample::Int;
        ϕname="msq", ϕϵ=T(1.0e-3),
        fixed_sampler_func=nothing,
        solver=RK4,
        shiftprotect=true,
        ptype="threads") where T<:AbstractFloat

    ϕ = get_Φ(ϕname, ϕϵ)

    test_func = [(x,t,p)->-U₀(x), (x,t,p)->-U₁(x), (x,t,p)-> divg_b(p, x)]
    test_func_para = [nothing nothing para]

    if fixed_sampler_func == nothing
        # the default way to generate sample
        init_func = j->sampler(U₀,1)[:]
    else
        # if one wants to use a fix collection of samples, e.g., denoted as pts,
        # then we can simply choose fixed_sampler_func = j-> pts[j]
        init_func = j->fixed_sampler_func(j)
    end

    if ptype == "pmap"
        data = pmap(j->one_path_dyn_b(para, test_func, test_func_para,
                                  N, offset, init_func, j, solver,
                                  T=T, shiftprotect=shiftprotect)[1],
                    1:numsample)
    else
        data = zeros(T, numsample)
        Threads.@threads for j = 1:numsample
            data[j] = one_path_dyn_b(para, test_func, test_func_para,
                            N, offset, init_func, j, solver,
                            T=T, shiftprotect=shiftprotect)[1];
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

"""
This function implements the following dyanmics:
- dXₜ = b(Xₜ),
- dY_t = ∇b(X_t) Y_t + δb(X_t), where δb = db/dθ.

Input:
- XY is the state of [X, Y₁, Y₂, ⋯, Yₘ], where m is the number of parameters;
- flow is the neural network representation of 𝐛;

Output:
- the time derivative [̇dX, dY₁,⋯, dYₘ].
"""
function optimize_b_dyn(XY::Array{T}, flow::DynTrain{T}) where T<:AbstractFloat
    M = flow.total_num_para
    x = XY[:,1]
    dXY = zeros(T, size(XY))
    dXY[:,1] = flow(x)
    grad_b = ∇b(flow, x)
    dXY[:,2:(1+M)] = grad_b*XY[:,2:(1+M)] + grad_b_wrt_para(flow, x)
    return dXY
end

"""
test_func_train(...) is a generic template to generate test functions.
"""
function test_func_train(XY::AbstractMatrix{T}, p::DynTrain{T},
        U₀::Potential{T}, U₁::Potential{T}) where T<:AbstractFloat

    num_para = p.num_para
    M = p.total_num_para

    x = XY[:,1]
    ϕ = zeros(T, 3*(1 + M))
    ϕ[1:3] = [-U₀(x), -U₁(x), divg_b(p, x)]

    ∇U₀ = ∇U(U₀, x)
    ∇U₁ = ∇U(U₁, x)
    graddivgb = grad_divg_b(p, x) # the term ∇(∇⋅b)

    divgbdθ = grad_divg_wrt_para(p,x)
    for k = 1:p.total_num_para
        ϕ[3*k+1] = -dot(∇U₀, XY[:,k+1])
        ϕ[3*k+2] = -dot(∇U₁, XY[:,k+1])
        ϕ[3*k+3] = dot(graddivgb, XY[:,k+1]) + divgbdθ[k]
    end

    return ϕ
end

"""
It gives a single trajectory estimation for Z₁/Z₀ and
the derivatives of the secomd moment with respect to parameters.
"""
function one_path_optimize_dyn_b(flow::DynTrain{T},
    test_func::Function,
    test_func_para,
    N::Int, offset::Int,
    init_func::Function,
    init_func_arg,
    solver::Function, ϕ::Φ{T};
    verbose::Bool=false,
    shiftprotect::Bool=true) where T<:AbstractFloat

    n = flow.dim
    h = T(1/N)
    M = flow.total_num_para

    # initialize a particle.
    X₀ = init_func(init_func_arg)
    verbose ? println(X₀) : nothing

    Vf,_ = time_integrate((x,t,p) -> optimize_b_dyn(x,p), flow,
        hcat(X₀,zeros(T, n, M)),
        T(0.0), T(1.0), N, solver,
        test_func, test_func_para, 3*(1+M))
    Vb,_ = time_integrate((x,t,p) -> (-1)*optimize_b_dyn(x,p), flow,
        hcat(X₀,zeros(T, n, M)),
        T(0.0), T(1.0), N, solver,
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
        shift = maximum(F₀)
        F₀ = F₀ .- shift
        F₁ = F₁ .- shift
        F₀ = exp.(F₀)
        F₁ = exp.(F₁)
    end

    # update G₀ and G₁.
    G₀ = zeros(T, M, 2*N+1)
    G₁ = zeros(T, M, 2*N+1)
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
    deri = zeros(T, M)
    for i = 1:M
        Int_G₀ = [Trapezoidal(G₀[i,(j+offset-N):(j+offset)], h) for j in TmTp]
        tmp₁ = Trapezoidal(G₁[i,TmTp]./B, h)
        tmp₂ = Trapezoidal(F₁[TmTp].*Int_G₀./(B.^2), h)
        #deri[i] = 2*estimator*( tmp₁ - tmp₂ )
        deri[i] = ϕ.fderi(estimator)*( tmp₁ - tmp₂ )
    end

    return vcat(estimator, deri)
end


"""
This function returns first and second order moment of estimator and derivative with respect to
    the parameter (stored inside flow.para_list).

Input:
- N is the number of grid points in time
- offset encodes the starting time T₋: the relation is that T₋ = -offset*(1/N)
- numsample is the sample size

Output: (fst_m, sec_m) where
- fst_m: first moment of A, ∂(A^2)/∂θ₁, ∂(A^2)/∂θ₂, ⋯,  ∂(A^2)/∂θₗ
where l is the number of parameters where A is the estimator herein (see paper)
- sec_m: second moment of the above

"""
function stat_optimize_dyn_b(U₀::Potential{T}, U₁::Potential{T},
        flow::DynTrain{T}, N::Int,
        offset::Int, numsample::Int;
        ϕname="msq", ϕϵ::T=T(1.0e-3),
        fixed_sampler_func=nothing,
        solver::Function=RK4,
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

    num_para = flow.num_para
    M = flow.total_num_para
    test_func = (x,t,p) -> test_func_train(x, p, U₀, U₁)
    test_func_para = flow

    if ptype == "pmap"
        data = pmap(j-> one_path_optimize_dyn_b(flow, test_func, test_func_para, N,
                                        offset, init_func, j, solver, ϕ),
            1:numsample)
    else
        data = [zeros(T,1+flow.total_num_para) for j=1:numsample]
        Threads.@threads for j = 1:numsample
            data[j] = one_path_optimize_dyn_b(flow, test_func, test_func_para, N,
                                              offset, init_func, j, solver, ϕ)
        end
    end

    #return get_mean_sec_moment(data)
    return get_mean_entropy(data, ϕ)
end


"""
train_NN_int
"""
function train_NN_int(U₀::Potential{T}, U₁::Potential{T},
        flow::DynTrain{T},
        N::Int, numsample_max::Int,
        train_step::Int,
        h::T, decay::T,
        gpts::AbstractMatrix{T}, gpts_sampler::Union{Function,Nothing};
        ϕname="msq", ϕϵ=T(1.0e-3),
        offset::Int=0,
        biased_func=nothing,
        solver::Function=RK4,
        ref_value::T=T(1.0),
        seed::Int64=-1,
        verbose::Bool=false, printpara::Bool=false,
        ρ=T(0.5), c=T(0.5),
        to_normalize::Bool=true,
        max_search_iter::Int=10,
        sample_num_repeat::Int=1,
        print_sample::Bool=false,
        numsample_min::Int=-1,
        savepara::Bool=false,
        ptype="threads",
        showprogress=true,
        compute_rela_divg=true,
        quiet::Bool=true) where T<:AbstractFloat

    ϕ = get_Φ(ϕname, ϕϵ)

    (offset < 0 || offset > N) ? error("Incorrect offset") : nothing

    fixed_sampler_func = j->gpts[:,j]
    loss_func(m)=get_data_err_var(U₀, U₁, flow, N, offset, m,
                                  ϕname=ϕname, ϕϵ=ϕϵ,
                                  fixed_sampler_func=fixed_sampler_func,
                                  solver=solver, ptype=ptype)[3]
    stat_opt_func(m) = stat_optimize_dyn_b(U₀, U₁, flow, N, offset, m,
                                  ϕname=ϕname, ϕϵ=ϕϵ,
                                  fixed_sampler_func=fixed_sampler_func,
                                  solver=solver, ptype=ptype)

    return train_NN_template(stat_opt_func, loss_func,
                      flow, numsample_max, train_step,
                      h, decay, gpts, gpts_sampler, ϕ,
                      biased_func=biased_func,
                      ref_value=ref_value, seed=seed,
                      verbose=verbose, printpara=printpara,
                      ρ=ρ, c=c, to_normalize=to_normalize,
                      max_search_iter=max_search_iter,
                      sample_num_repeat=sample_num_repeat,
                      print_sample=print_sample,
                      numsample_min=numsample_min, savepara=savepara,
                      showprogress=showprogress,
                      compute_rela_divg=compute_rela_divg,
                      quiet=quiet)
end
