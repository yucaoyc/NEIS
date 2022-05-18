function plot_error(m, ℓ, model_num, seed_list, exact_mean, folder; 
        start_idx = 1, fig_type=:var, yscale=:log10, xlabel="", ylabel="", title="", linewidth=2, nolabel=false)
    
    fig = plot(yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title)
    for i in 1:length(seed_list)
        seed = seed_list[i]
        
        casename =  folder*@sprintf("case_%d_model_%d_%d_%d_%d_%d_%d_%d_training_data.jld2", 
            testcasenum, model_num, n, N, numsample_max, m, ℓ, seed)
        est_fst_m, est_sec_m, est_grad_norm = load(casename, "est_fst_m", "est_sec_m", "est_grad_norm")
        
        time = 0:1:train_step
        if fig_type == :err
            data = abs.(est_fst_m .- exact_mean)./exact_mean
        elseif fig_type == :var
            data = (est_sec_m .- est_fst_m.^2)/exact_mean.^2
        elseif fig_type == :grad
            data = est_grad_norm
        end
        
        if nolabel 
            plot!(fig, time[start_idx:end], data[start_idx:end], linewidth=linewidth, label="")
        else
           plot!(fig, time[start_idx:end], data[start_idx:end], linewidth=linewidth, label="trial $(i)")
        end
    end
    return fig
end


function plot_traj(num_particle, traj_gpts, flow, t_vec, unit_step, idx1 = 1, idx2=2, color=:white, lw=1.0)
    for idx = 1:num_particle
        traj_state, traj_time, _, _ = traj_dyn_b(traj_gpts[idx], flow, t_vec, unit_step, 
            (x,t,p)->flow.f(x, flow.para_list...));
        traj_x = traj_state[idx1,:]
        traj_y = traj_state[idx2,:]
        locx(x) = LinearInterpolator(traj_time, traj_x)(x)
        locy(x) = LinearInterpolator(traj_time, traj_y)(x)
        Plots.plot!(x->locx(x), x->locy(x), traj_time[1], traj_time[end], label="", color=color, linewidth=lw)
    end
end