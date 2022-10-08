export RK4,
       MM,
       Euler,
       time_integrate
"""
Run Runge-Kutta-4 for dynamics f(x,t) at position x0 and time t, to the next time t+h
"""
function RK4(f::Function, x0::Union{Tuple,Array},
        t::T, h::T, para::Any) where T<:AbstractFloat
    k1 = f(x0, t, para)
    k2 = f(x0 .+ h/2 .* k1, t+h/2, para)
    k3 = f(x0 .+ h/2 .* k2, t+h/2, para)
    k4 = f(x0 .+ h .* k3, t+h, para)
    return x0 .+ T(1/6*h) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

"""
    midpoint rule
"""
function MM(f, x0::Union{Tuple,Array}, t::T, h::T, para::Any) where T<:AbstractFloat
    k1 = f(x0, t, para)
    k2 = f(x0 .+ T(0.5*h).*k1, t+ T(0.5)*h, para)
    return x0 .+ (h.*k2)
end

function Euler(f, x0::Union{Tuple,Array}, t::T, h::T, para::Any) where T<:AbstractFloat
    k1 = f(x0, t, para)
    return x0 .+ (h.*k1)
end

"""
This is a generic template to estimate ϕ(X_t)
where X_t follows the ode_dyn(x,t,para),
numerically solved by ode_solver on time interval [a,b]
with initial condition Xₐ at time a.

N is the total number of steps
We support a list of test functions.

Version 1:
- testfunc(x, t, test_func_para) -> Real number, is a list of generic functions.

Version 2:
- testfunc(x, t, test_func_para) -> a vector of dimension m.

Version 3:
- No test function and only propagate the ODE.
"""
function time_integrate(ode_dyn::Function, para::Any,
    Xₐ::Union{Array,Tuple},
    a::T, b::T, N::Int,
    ode_solver::Function,
    test_func::AbstractArray{Function},
    test_func_para::AbstractArray{}) where T<:AbstractFloat

    m = length(test_func)
    func_values = zeros(T, m, N+1)
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

function time_integrate(ode_dyn::Function, para::Any, Xₐ::Array{T},
    a::T, b::T, N::Int,
    ode_solver::Function,
    test_func,
    test_func_para,
    m::Int) where T<:AbstractFloat

    func_values = zeros(T, m, N+1)
    h = (b-a)/N
    X = copy.(Xₐ)

    func_values[:,1] = test_func(X, a, test_func_para)

    for j = 2:(N+1)
        X = ode_solver(ode_dyn, X, a+(j-2)*h, h, para)
        func_values[:,j] = test_func(X, a + (j-1)*h, test_func_para)
    end

    return (func_values, copy.(X))
end


function time_integrate(ode_dyn::Function, para::Any,
        Xₐ::Union{Tuple, Array},
        a::T, b::T, N::Int, ode_solver) where T<:AbstractFloat

    h = (b-a)/N
    X = copy.(Xₐ)

    for j = 2:(N+1)
        X = ode_solver(ode_dyn, X, a+(j-2)*h, h, para)
    end

    return X
end
