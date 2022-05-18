module ModTraj

using ModPotential: Potential
using Plots
using Printf
using Random
using ModODEIntegrate: RK4

export traj_dyn_b, traj_up_to_time

function traj_dyn_b(X₀::Array{Float64}, para, 
        t_windows::Array{Float64,1}, unit_step::Int64, dyn)
    # Return trajectory and time for a particular initial choice X₀.
    #
    # - t_windows gives the time points for which we need to find the state
    # - unit_step gives the number of steps required
    #       for the time interval [t_windows[k-1],t_windows[k]]

    dim = length(X₀)
    state_forward = zeros(dim, length(t_windows))
    state_forward[:,1] = copy(X₀)

    ode_dyn_forward = (x,t,p)->dyn(x,t,p)
    for j = 2:length(t_windows)
        a = t_windows[j-1]
        b = t_windows[j]
        h = (b-a)/unit_step
        Xtmp = copy(state_forward[:,j-1])
        for k = 1:unit_step
            Xtmp = RK4(ode_dyn_forward,Xtmp,a+(k-1)*h,h,para)
        end
        state_forward[:,j] = Xtmp
    end

    state_backward = zeros(dim, length(t_windows))
    state_backward[:,1] = copy(X₀)

    ode_dyn_backward = (x,t,p)->(-1)dyn(x,t,p)
    for j = 2:length(t_windows)
        a = t_windows[j-1]
        b = t_windows[j]
        h = (b-a)/unit_step
        Xtmp = copy(state_backward[:,j-1])
        for k = 1:unit_step
            Xtmp = RK4(ode_dyn_backward,Xtmp,a+(k-1)*h,h,para)
        end
        state_backward[:,j] = Xtmp
    end

    traj_state = hcat((reverse(state_backward, dims=2))[:,1:(length(t_windows)-1)], state_forward)
    traj_time = vcat((reverse(-1*t_windows))[1:(length(t_windows)-1)], t_windows)

    return traj_state, traj_time, state_forward, state_backward

end

function traj_up_to_time(k::Int64, num_particle::Int64,
        traj_state::Array{}, traj_time::Array{},
        contour_potential, xgrid::Array{Float64,1}, ygrid::Array{Float64,1}, 
        color_grad, 
        figsize, 
        sp_x, sp_y, sp_color, sp_size; ct_color=:tofino, plot_marker_pts=true, linewidth=2, clevels=40)

    fig = contour(xgrid, ygrid, contour_potential, fill=true, size=figsize,
                xlim = (minimum(xgrid),maximum(xgrid)),
                ylim = (minimum(ygrid),maximum(ygrid)), color=ct_color, clevels=clevels)
    for m = 1:num_particle
        if plot_marker_pts 
            scatter!(traj_state[m,1,1:k], traj_state[m,2,1:k], color=color_grad[m], label="")
        end
        for j = 1:(k-1)
            x1 = traj_state[m,1,j]
            x2 = traj_state[m,1,j+1]
            y1 = traj_state[m,2,j]
            y2 = traj_state[m,2,j+1]
            #time_title = @sprintf("t = %.2f", traj_time[k])
            plot!([x1,x2],[y1,y2], color=color_grad[m],label="", linewidth=linewidth)
        end
    end
    if length(sp_x) > 0
        scatter!(sp_x, sp_y, color=sp_color, markersize=sp_size, label="")
    end
    return fig
end

#function plot_particle_traj(dim, para,
#        seed::Int64, num_particle::Int64,
#        t_windows::Array{Float64,1}, unit_step::Int64,
#        contour_potential, xgrid::Array{Float64,1}, ygrid::Array{Float64,1};
#        folder::String="", traj_name::String="traj",
#        color_grad=nothing, fps::Int64=3, figsize=(900,800),
#        sp_x=[], sp_y=[], sp_color=:white, sp_size=3, dyn=nothing)
#
#    Random.seed!(seed)
#
#    # we assume initially X₀ follows standard normal random variable
#    X0 = randn(num_particle,dim)
#
#    # set color gradient
#    if color_grad == nothing
#        color_grad = cgrad(:heat, num_particle)
#    end
#
#    traj_state = zeros(num_particle, dim, 2*length(t_windows)-1)
#    traj_forward = zeros(num_particle, dim, length(t_windows))
#    traj_backward = zeros(num_particle, dim, length(t_windows))
#    traj_time = vcat((reverse(-1*t_windows))[1:(length(t_windows)-1)], t_windows)
#
#    for j = 1:num_particle
#        traj_state[j,:,:],_,traj_forward[j,:,:], traj_backward[j,:,:] =
#            traj_dyn_b(X0[j,:], para, t_windows, unit_step, dyn)
#    end
#
#    # create animation
#    anim1 = @animate for i = 1:length(t_windows)
#       traj_up_to_time(i, num_particle, traj_forward, t_windows,
#                       contour_potential, xgrid, ygrid, color_grad, figsize,
#                       sp_x, sp_y, sp_color, sp_size)
#    end
#
#    if folder != ""
#        fname = @sprintf("%s/%s_forward_%d.gif", folder, traj_name, seed)
#        gif(anim1, fname, fps = fps)
#    end
#
#    anim2 = @animate for i = 1:length(traj_time)
#       traj_up_to_time(i, num_particle, traj_state, traj_time,
#                       contour_potential, xgrid, ygrid, color_grad, figsize,
#                       sp_x, sp_y, sp_color, sp_size)
#    end
#
#    if folder != ""
#        fname = @sprintf("%s/%s_all_%d.gif", folder, traj_name, seed)
#        gif(anim2, fname, fps = fps)
#    end
#
#    return anim1, anim2
#end
#
end
