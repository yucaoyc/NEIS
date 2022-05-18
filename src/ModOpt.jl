module ModOpt

using LinearAlgebra
using Distributed
using ModDyn
using ModODEIntegrate
using ModPotential
using ModUtil
using Statistics
using Printf

export armijo_line_search, train_NN_template, biased_GD_func

function armijo_line_search(flow::Dyn, loss_func, numsample::Int64, 
        grad_normalized, # grad_normalized = ∇f/norm(∇f)
        grad_norm::Float64, 
        h₀::Float64, 
        loss_bef;
        ρ=0.5, c=0.5, max_search_iter=10, sample_num_repeat=1)
    
    new_numsample = numsample*sample_num_repeat
    h = h₀
    update_flow_para!(flow, grad_normalized, length(flow.train_para_idx), h)
    loss_aft = loss_func(new_numsample)

    search_iter = 0
    while (loss_aft > loss_bef - c*h*grad_norm) && (search_iter <= max_search_iter)
        update_flow_para!(flow, grad_normalized, length(flow.train_para_idx), (ρ-1)*h)
        h = h*ρ
        search_iter += 1
        loss_aft = loss_func(new_numsample)
    end

    if search_iter > max_search_iter
        @warn "Iteration in line search exceeds maximum steps allowed."
    end
    return search_iter
end

function train_NN_template(stat_opt_func, loss_func,
        flow::Dyn,
        numsample_max::Int64,
        train_step::Int64, 
        h::Float64, decay::Float64,
        gpts, gpts_sampler;
        biased_func = nothing,
        ref_value = 1.0, 
        seed::Int64=-1, verbose=false, printpara=false, 
        ρ=0.5, c=0.5, max_search_iter=10, sample_num_repeat=1, 
        test_data=false, numsample_min=nothing, savepara=false)

    (seed >= 0) ? Random.seed!(seed) : nothing

    est_fst_m = zeros(train_step+1)
    est_sec_m = zeros(train_step+1)
    est_grad_norm = zeros(train_step+1)
    est_para = Array{Any}(undef, train_step+1)

    if numsample_min == nothing
        numsample_min = numsample_max
    end
    num_of_para = length(flow.train_para_idx)

    train_iter = 1
    while train_iter <= train_step + 1

        numsample = Int64(ceil((train_iter-1)/train_step*(numsample_max - numsample_min)+numsample_min))
        printstyled("Train step $(train_iter) samplesize $(numsample)\n", color=:blue)
        
        # generate data
        if gpts_sampler != nothing
            for j = 1:numsample
                gpts[:,j] = gpts_sampler()
                if biased_func != nothing
                    gpts[:,j] = biased_func(gpts[:,j], train_iter)
                end
            end
            test_data ? println(gpts) : nothing
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
        if abs(grad_norm/ref_value) < 1.0e-10
            @warn("The training step terminates earlier!")
            break
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


function biased_GD_func(x0::Array, iter::Int64, train_step::Int64,
        percent::Float64, U₁::Potential, N::Int64;
        solver=RK4, s = 1.0, υ=0.5)

    slope = percent/(train_step*υ)
    p = max(0.0, percent - slope*iter)
    if rand() < p
        # run the ode till a time
        return time_integrate((z,t,para)->-s*U₁.gradU(z), nothing, x0, 0.0, 1.0, N, solver)
    else
        return x0
    end
end


end
