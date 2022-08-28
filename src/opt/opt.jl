export armijo_line_search, train_NN_template, biased_GD_func

function armijo_line_search(flow::DynTrain{T}, loss_func::Function, numsample::Int, 
        grad_normalized::Vector{}, # grad_normalized = ∇f/norm(∇f)
        grad_norm::T, 
        h₀::T, 
        loss_bef::T;
        ρ= T(0.5), c=T(0.5), 
        max_search_iter::Int=10, 
        sample_num_repeat::Int=1) where T<:AbstractFloat
    
    new_numsample = numsample*sample_num_repeat
    h = h₀
    update_flow_para!(flow, grad_normalized, h)

    if max_search_iter == 0
        # if we don't do line search, it reduces to gradient descend algorithm
        return 0
    else
        loss_aft = loss_func(new_numsample)
        search_iter = 0
        while (loss_aft > loss_bef - c*h*grad_norm) && (search_iter < max_search_iter)
            update_flow_para!(flow, grad_normalized, (ρ-1)*h)
            h = h*ρ
            search_iter += 1
            loss_aft = loss_func(new_numsample)
        end

        if search_iter >= max_search_iter
            @warn "Iteration in line search exceeds maximum steps allowed."
        end
        return search_iter
    end
end

"""
- stat\\_opt\\_func: returns the first and second moment for the derivative of loss function 
    with respect to parameters
- loss\\_func: the loss function
- flow: the dynamics to train
- numsample_max: the largest sample size used
- h: learning rate
- decay: decay rate
- gpts: a matrix that contains samples
- gpts\\_sampler: if it is nothing, then we use fixed samples during training;
    otherwise, we refresh data stored in "gpts" using gpts_sampler
- biased\\_func: if it is nothing, we don't use assisted training method;
    otherwise, the sample data in "gpts" is modified according to the gradient dynamics of U₁
- ref\\_value: if the exact value of Z is known, then we can let ref\\_value = Z;
    by default, it is 1. 
    The ref\\_value only scales the mean and variance and it won't matter
    in terms of finding optimized parameters.
- seed: determine whether and how we change the random seed; if seed = -1, we take no action.
- verbose: whether we print something.
- printpara: whether we print parameters. 
- ρ, c: parameters used in Armijo line search.
- max\\_search_iter: the largest line searching step.
- sample\\_num_repeat: determines the amount of samples used during line search.
- print\\_sample: whether we print sample or not.
- numsample\\_min: if the value is -1, we use fixed amount of samples during training;
    otherwise, we linearly increase the sample from [numsample\\_min, numsample\\_max]
- savepara: whether we save parameters during each training step.
- allow\\_early\\_terminate: determine whether we terminate the training when the gradient of loss function is smaller than terminate\\_value.
- terminate\\_value: the criterion to terminate the training.
"""
function train_NN_template(stat_opt_func::Function, 
        loss_func::Function,
        flow::DynTrain{T},
        numsample_max::Int,
        train_step::Int, 
        h::T, decay::T,
        gpts::AbstractMatrix{T}, gpts_sampler::Union{Function,Nothing};
        # optimal parameters
        biased_func::Union{Function,Nothing} = nothing,
        ref_value = T(1.0), 
        seed::Int=-1, 
        verbose::Bool=false, 
        printpara::Bool=false, 
        ρ=T(0.5), c=T(0.5), 
        max_search_iter::Int=10, 
        sample_num_repeat::Int=1, 
        print_sample::Bool=false, 
        numsample_min::Int=-1, 
        savepara::Bool=false, 
        allow_early_terminate::Bool=false,
        terminate_value::T=T(1.0e-6)) where T<:AbstractFloat

    (seed >= 0) ? Random.seed!(seed) : nothing

    est_fst_m = zeros(T,train_step+1)
    est_sec_m = zeros(T,train_step+1)
    est_grad_norm = zeros(T,train_step+1)
    est_para = Array{Any}(undef, train_step+1)

    # when numsample_min == -1
    # sample size is assumed to be a constant during the training
    # otherwise, sample size is linearly increasing.
    if numsample_min == (-1)
        numsample_min = numsample_max
    end
    num_of_para = length(flow.train_para_idx)

    train_iter = 1
    while train_iter <= train_step + 1

        numsample = Int64(ceil((train_iter-1)/train_step*(numsample_max - numsample_min) 
                               + numsample_min))
        printstyled("Train step $(train_iter) samplesize $(numsample)\n", color=:blue)
        
        # generate data
        if gpts_sampler != nothing
            for j = 1:numsample
                gpts[:,j] = gpts_sampler()
                if biased_func != nothing
                    gpts[:,j] = biased_func(gpts[:,j], train_iter)
                end
            end
            print_sample ? println(gpts) : nothing
        end
        
        # evaluate
        @time fst_m, sec_m = stat_opt_func(numsample)
        est_fst_m[train_iter] = fst_m[1]
        est_sec_m[train_iter] = sec_m[1]
        savepara ? est_para[train_iter] = deepcopy(flow.para_list) : nothing
        vec_deri = reshape_deri(flow, fst_m[2:end])
        loss_bef = (est_sec_m[train_iter] - est_fst_m[train_iter]^2)
            
        if verbose # print information
            rela_mean = est_fst_m[train_iter]/ref_value
            rela_var = loss_bef/ref_value^2
            info = @sprintf("Relative mean %.2E, Relative variance %.2E\n", 
                            rela_mean, rela_var)
            printstyled(info, color=:green)
        end

        # normalize derivative
        grad_norm = norm(fst_m[2:end])
        if allow_early_terminate
            if abs(grad_norm/ref_value) < terminate_value
                @warn("The training step terminates earlier!")
                break
            end
        end

        for i = 1:num_of_para
            vec_deri[i] /= grad_norm
        end
        est_grad_norm[train_iter] = grad_norm

        ##########
        # update parameters via line searching.
        if train_iter < train_step + 1
            newh = h/(1+decay*train_iter)
            @time search_iter = armijo_line_search(flow, loss_func, numsample,
                vec_deri, grad_norm, newh, loss_bef,
                ρ=ρ, c=c, 
                max_search_iter=max_search_iter, 
                sample_num_repeat=sample_num_repeat)
            if verbose
                printstyled("Search iter: "*string(search_iter)*"\n", color=:green)
            end
        end

        train_iter += 1
        printpara ? println(flow.para_list) : nothing
        verbose ? @printf("Grad norm: %.2E\n", grad_norm) : nothing
        flush(stdout)
    end

    return est_fst_m, est_sec_m, est_grad_norm, est_para
end

"""
    A function that alters the sample x0 by x₁ where
    x₁ is the state obtained by running gradient dx = -∇U₁(x), x₀ = x0.
"""
function biased_GD_func(x0::Array{T}, iter::Int, train_step::Int,
        percent::T, U₁::Potential{T}, N::Int;
        solver=RK4, s = T(1.0), υ=T(0.5)) where T<:AbstractFloat

    slope = percent/(train_step*υ)
    p = max(T(0.0), percent - slope*iter)
    if rand() < p
        # run the ode till a time
        return time_integrate((z,t,para)->-s*gradU(U₁,z), nothing, 
                              x0, T(0.0), T(1.0), N, solver)
    else
        return x0
    end
end
