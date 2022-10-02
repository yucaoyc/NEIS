export traj_dyn_b, plot_traj, plot_traj_torus, plot_rho_and_flow

"""
Return the trajectory and the time grid points for a particular initial choice X₀.

- X₀: initial state at time 0.
- para: parameter for ODE.
- t_windows gives the time grid points
at which we need to find the state (for forward direction only).
- unit_step gives the number of steps required
for the time interval [t_windows[k-1],t_windows[k]].
- dyn: the function for ODE dynamics.
"""
function traj_dyn_b(X₀::Array{T}, para::Any,
        t_windows::Array{T,1}, unit_step::Int,
        dyn::Function) where T <: AbstractFloat

    dim = length(X₀)
    state_forward = zeros(T, dim, length(t_windows))
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

    state_backward = zeros(T, dim, length(t_windows))
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

"""
    In a given picture using Plots,
    we add trajectories of the flow.
    e.g., t_vec is an array that contains positive times, t_vec = [0, 0.1, 0.2, 0.3]
    It will automatically include trajectories for  [-0.3, -0.2, -0.1, 0.0].
"""
function plot_traj(num_particle::Int, traj_gpts::Array{},
        flow::Dyn{T}, t_vec::Vector{T}, unit_step::Int;
        idx1::Int = 1, idx2::Int = 2,
        color = :white, lw = 1.0) where T <: AbstractFloat

    for idx = 1:num_particle
        traj_state, traj_time, _, _ = traj_dyn_b(traj_gpts[idx], flow, t_vec, unit_step,
            (x,t,p)->flow(x))
        traj_x = traj_state[idx1,:]
        traj_y = traj_state[idx2,:]
        locx(x) = LinearInterpolator(traj_time, traj_x)(x)
        locy(x) = LinearInterpolator(traj_time, traj_y)(x)
        Plots.plot!(x->locx(x), x->locy(x), traj_time[1], traj_time[end],
                    label="", color=color, linewidth=lw)
    end
end


function plot_traj_torus(num_particle::Int, traj_gpts::Array{},
        flow::Dyn{T}, t_vec::Vector{T}, unit_step::Int;
        Ω₁::Function=(@. x->mod(x,T(1.0))), Ω₂::Function = (@. x->mod(x,T(1.0))),
        idx1::Int = 1, idx2::Int = 2,
        color = :white, lw = 1.0,
        dx1 = 0.6, dx2 = 0.6) where T <: AbstractFloat

    for i = 1:num_particle
        traj_state, traj_time, _, _ = traj_dyn_b(traj_gpts[i], flow, t_vec, unit_step,
            (x,t,p)->flow(x))
        # move each point into the torus.
        traj_x = Ω₁(traj_state[idx1,:])
        traj_y = Ω₂(traj_state[idx2,:])
        t_idx = []
        # find time where the trajectory passes the boundary of a tours.
        for k = 1:(length(traj_time)-1)
            if abs(traj_x[k+1] - traj_x[k]) > dx1
                push!(t_idx, k)
            elseif abs(traj_y[k+1] - traj_y[k]) > dx2
                push!(t_idx, k)
            end
        end
        push!(t_idx, length(traj_time))

        # plot each connected part one by one.
        # we skip the segment where the trajectory passes the boundary
        # if t_vec is refined enough,
        # such a missing part should be negligible.
        l_idx = 1
        for k = 1:length(t_idx)
            r_idx = t_idx[k]
            # if r_idx - l_idx > 1
            if r_idx - l_idx > 0
                locx(x) = LinearInterpolator(traj_time[l_idx:r_idx], traj_x[l_idx:r_idx])(x)
                locy(x) = LinearInterpolator(traj_time[l_idx:r_idx], traj_y[l_idx:r_idx])(x)
                plot!(locx, locy, traj_time[l_idx], traj_time[r_idx],
                      label="", color=color, linewidth=lw)
            end
            l_idx = r_idx + 1
        end
    end
end


function plot_rho_and_flow(xmin, xmax, ymin, ymax,
        gpts, flow, ρ, figsize;
        unit_step=5, color=:tofino, margin=20px,
        Nx=100, Ny=100, Tscale=1.0, xlabel="", ylabel="", t_step=100,
        plot_torus = false,
        traj_paras::Dict{Any,Any}=Dict{Any,Any}())

    xc = range(xmin, stop=xmax, length=Nx)
    yc = range(ymin, stop=ymax, length=Ny)

    if size(gpts, 2) == 1
        traj_gpts = gpts
        num_particle = length(traj_gpts)
    else
        num_particle = size(gpts, 2)
        traj_gpts = [gpts[:,i] for i in 1:num_particle]
    end

    # e.g., rho = U₁(vcat([x,y],zeros(n-2)))
    fig_flow = contour(xc, yc, (x,y)-> ρ(x,y), fill=true,
        color=color, size=figsize,
        xlims=(xmin,xmax), ylims=(ymin,ymax),
        margin=margin, xlabel=xlabel, ylabel=ylabel)

    t_vec = Array(range(0,stop=Tscale,length=Int64(ceil(Tscale*t_step))+1))

    if !plot_torus
        plot_traj(num_particle, traj_gpts, flow, t_vec, unit_step, traj_paras...)
    else
        plot_traj_torus(num_particle, traj_gpts, flow, t_vec, unit_step, traj_paras...)
    end

    return fig_flow
end
