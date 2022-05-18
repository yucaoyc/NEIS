module ModODEIntegrate

using ModQuadratureScheme: QuadratureScheme1D
using LinearAlgebra

export RK4, 
       MM,
    time_integrate, 
    path_integrate

function RK4(f, x0::Union{Tuple,Array}, t::Float64, h::Float64, para; detect_nan=false)
    # Run Runge-Kutta-4 for dynamics f(x,t) at position x0 and time t, to the next time t+h

    k1 = f(x0, t, para)
    k2 = f(x0 .+ h/2 .* k1, t+h/2, para)
    k3 = f(x0 .+ h/2 .* k2, t+h/2, para)
    k4 = f(x0 .+ h .* k3, t+h, para)

    if detect_nan
        if sum(isnan.(k1)) > 1.0e-10
            println(x0)
            error("NaN appears in k1.")
        end
        if sum(isnan.(k2)) > 1.0e-10
            println(x0)
            println(k1)
            println(x0+h*k1/2)
            error("NaN appears in k2.")
        end
        if sum(isnan.(k3)) > 1.0e-10
            println(x0)
            println(k1)
            println(k2)
            println(x0+h*k2/2)
            error("NaN appears in k3.")
        end
        if sum(isnan.(k4)) > 1.0e-10
            println(x0)
            println(k1)
            println(k2)
            println(k3)
            println(x0+h*k3)
            error("NaN appears in k4.")
        end
    end

    return x0 .+ (1/6*h) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)

end

function MM(f, x0::Union{Tuple,Array}, t::Float64, h::Float64, para; detect_nan=false)
    # midpoint rule
    k1 = f(x0, t, para)
    k2 = f(x0 .+ (0.5*h).*k1, t+0.5*h, para)
    return x0 .+ (h.*k2)
end

"""
    This is a generic template to estimate ϕ(Xₜ)
    where X_t follows the ode_dyn(x,t,para),
    numerically solved by ode_solver on time interval [a,b]
    with initial condition Xₐ at time a.

    N is the total number of steps
    We support a list of test functions.


    Version 1: 
    ∘ testfunc(x, t, test_func_para) -> Real number, is a list of generic functions.

    Version 2: 
    ∘ testfunc(x, t, test_func_para) -> a vector of dimension m.
"""

function time_integrate(ode_dyn, para, Xₐ::Union{Array,Tuple},
    a::Float64, b::Float64, N::Int64,
    ode_solver,
    test_func::AbstractArray{},
    test_func_para::AbstractArray{})


    m = length(test_func)
    func_values = zeros(m, N+1)
    h = (b-a)/N
    X = copy.(Xₐ)

    for r = 1:m
        func_values[r,1] = test_func[r](X, a, test_func_para[r])
    end

    for j = 2:(N+1)
        X = ode_solver(ode_dyn, X, a+(j-2)*h, h, para)
        for r = 1:m
            func_values[r,j] = test_func[r](X, a + (j-1)*h, test_func_para[r])
        end
    end

    return (func_values, copy.(X))
end

function time_integrate(ode_dyn, para, Xₐ::Array{Float64},
    a::Float64, b::Float64, N::Int64,
    ode_solver, test_func, test_func_para, m::Int64)

    func_values = zeros(m, N+1)
    h = (b-a)/N
    X = copy.(Xₐ)

    func_values[:,1] = test_func(X, a, test_func_para)

    for j = 2:(N+1)
        X = ode_solver(ode_dyn, X, a+(j-2)*h, h, para)
        func_values[:,j] = test_func(X, a + (j-1)*h, test_func_para)
    end

    return (func_values, copy.(X))
end


#################################################

function time_integrate(ode_dyn, para, Xₐ::Union{Tuple, Array},
    a::Float64, b::Float64, N::Int64, ode_solver)

    h = (b-a)/N
    X = copy.(Xₐ)

    for j = 2:(N+1)
        X = ode_solver(ode_dyn, X, a+(j-2)*h, h, para)
    end

    return X
end

"""
    This is a generic template to estimate int_{a}^{b} ϕ(X_t) dt,
    where X_t follows the ode_dyn(x,t,para), numerically solved by ode_solver
    with initial condition Xₐ.

    Inputs:
    * h means the time step.
    * we support a list of test functions;
    * qs is the quadrature scheme used to estimate int_{c}^{c+h} ϕ(X_t) dt
    * stopcriterion is an array of functions aiming at stopping the estimation earlier than the time b.
"""
function path_integrate(ode_dyn, para, Xₐ::Array{Float64},
    a::Float64, b::Float64, h::Float64,
    ode_solver,
    test_func::AbstractArray{},
    test_func_para::AbstractArray{},
    qs::QuadratureScheme1D,
    stopcriterion::AbstractArray{})


    n = length(Xₐ)
    m = length(test_func)
    func_values = zeros(m)
    num_quad_pts = length(qs.ξ)
    to_run = [true for j = 1:m]
    N = Int64(floor((b-a)/h))
    X = copy.(Xₐ)

    for j = 1:N
        st_time = a + (j-1)*h
        final_time = a + j*h
        quadrature_time = (final_time-st_time)/2*qs.ξ .+ (st_time + final_time)/2

        state = zeros(n, num_quad_pts)
        values = zeros(m, num_quad_pts)
        state[:,1] = X
        k = 1
        for r = 1:m
            if to_run[r]
                values[r,k] = test_func[r](state[:,k], quadrature_time[k], test_func_para[r])
            end
        end
        for k = 2:num_quad_pts
           state[:,k] = ode_solver(ode_dyn, state[:,k-1], quadrature_time[k-1], quadrature_time[k]-quadrature_time[k-1], para)
            for r = 1:m
                if to_run[r]
                    values[r,k] = test_func[r](state[:,k], quadrature_time[k], test_func_para[r])
                end
            end
        end

        newvalues = [dot(values[r,:],qs.w)*(final_time-st_time)/2 for r=1:m]
        func_values = func_values .+ newvalues
        X = state[:,num_quad_pts]

        # a NaN handler.
#         if sum(isnan.(X)) > 1.0e-10
#             println(state)
#             error("NaN appears in the simulation.")
#         end

        # decide whether a particular test function needs to run
        if length(stopcriterion) > 0
            for r = 1:m
                if to_run[r]
                    if stopcriterion[r](X,final_time,newvalues[r],func_values[r])
                        to_run[r] = false
                    end
                end
            end
        end
        # decide whether to abort the dynamics
        if !any(to_run)
            return (func_values, final_time, copy.(X))
        end
    end

    return (func_values, a+h*N, copy.(X))
end


end
