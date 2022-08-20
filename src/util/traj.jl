export traj_dyn_b

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
