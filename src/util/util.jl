export get_resource, 
       get_ndims, 
       plot_setup,
       randHermitian, 
       remove_nan, 
       get_relative_stats, 
       integrate_over_time,
       domain_ball,
       domain_rectangle,
#       prod_domain,
       print_stat_name,
       add_procs,
       repeat_experiment

export divide_col, multiply_col
"""
Divide each column of a by b, i.e., return a[:,i]/b[i] as column i.
"""
function divide_col(a::Array{T,2}, b::Array{T}) where T<: AbstractFloat
    return a ./ b'
end

function divide_col(a::Array{T,1}, b::T) where T <: AbstractFloat
    return a ./ b
end

function multiply_col(a::Array{T,2}, b::Array{T}) where T <: AbstractFloat
    return a .* b'
end

function multiply_col(a::Array{T,1}, b::T) where T <: AbstractFloat
    return b .* a
end


"""
Get CPU information and number of processors used.
"""
function get_resource()
    return (Sys.cpu_info()[1].model, nprocs())
end

"""
Get the deep nn layer structure for an ℓ-layer nn with input and output dimension as n
and each inner layer having width m.
"""
get_ndims(n,m,ℓ) = vcat([n],m*ones(Int64,ℓ-1),[n])

"""
A default plot setup.
"""
function plot_setup()
    gr()
    default(titlefont = (12, "times"), legendfontsize=8, 
        legend_font_family="times", guidefont = (11, "times"), 
        fg_legend = :transparent);
end

"""
Return a n-by-n random Hermitian matrix with spectrum between [σmin, σmax]
"""
function randHermitian(n, σmin, σmax)
    Q , _ = qr(randn(n,n))
    D = Diagonal(rand(n)*(σmax-σmin) .+ σmin)
    return Q*D*transpose(Q)
end

"""
Remove NaN from data.
"""
function remove_nan(data::Array{},verbose=false)
    ndata = length(data)
    data = filter(x-> !isnan(x), data)
    percent = (ndata - length(data))/ndata
    if verbose
        @printf("Percentage of NaN: %.3f\n", percent)
    end
    return data, percent
end

"""
Return relative error and variance for a collection of sample data.
exact_mean is the reference value.
"""
function get_relative_stats(data::Array{T}, exact_mean::T; verbose=false) where T <: AbstractFloat
    rela_err = abs(mean(data)-exact_mean)/exact_mean
    rela_var = (mean(data.^2) - (mean(data))^2)/exact_mean^2
    if verbose
        @printf("Relative error: %.3f\n", rela_err)
        @printf("Relative variance: %.3f\n", rela_var)
    end
    return rela_err, rela_var
end

"""
Return an indicator function for a ball with a given radius.
"""
function domain_ball(radius::T) where T<:AbstractFloat
    Ω(x) = norm(x) < radius ? true : false
    return Ω
end

"""
An indicator function for a ball with a given radius.
"""
function domain_ball(x::Vector{T}, radius::T) where T<:AbstractFloat
    return norm(x) < radius
end

"""
The same as the last function but for multiple input states.
"""
function domain_ball(x::Matrix{T}, radius::T) where T<:AbstractFloat
    return sum(x.^2, dims=1) .< (radius^2)
end

"""
An indicator function for a rectangular domain specified by lb (lower bound)
and ub (upper bound).
"""
function domain_rectangle(x::Vector{T}, 
        lb::Union{Vector{T},T}, ub::Union{T,Vector{T}}) where T<:AbstractFloat
    r = (x .<= ub) .& (x .>= lb)
    return prod(r)
end

#function prod_domain(Ω₁::Function, Ω₂::Function, n₁::Int, n₂::Int)
#    Ω(x) = Ω₁(x[1:n₁]) && Ω₂(x[(n₁+1):(n₁ + n₂)])
#    return Ω
#end

"""
A function to print mean and variance information.
"""
function print_stat_name(name::String, m::T, var::T, exact_mean::T) where T<:AbstractFloat
    @printf("%10s: mean %6.2f variance %6.2f\n", name, m/exact_mean, var/exact_mean^2)
    @printf("%s\n", repeat("-", 40))
end

"""
Add threads according to system specifications.
"""
function add_procs(n1::Int=2, n2::Int=8)
    num_threads = Base.Sys.CPU_THREADS
    if num_threads <= 32
        if nprocs() < (num_threads - n1)
            addprocs(num_threads - n1 -nprocs())
        end
    else
        if nprocs() < (num_threads - n2)
            addprocs(num_threads -n2 - nprocs())
        end
    end
end

"""
Repeat the experiment specified by fun.
"""
function repeat_experiment(fun::Function, numsample::Int, numrepeat::Int, 
        gpts::Array{T}, gpts_sampler::Function) where T<:AbstractFloat
    v = zeros(T, numrepeat)
    test_update = gpts[:,1]
    for i = 1:numrepeat
        # refresh data
        for j = 1:numsample
            gpts[:,j] = gpts_sampler()
        end
        if norm(gpts[:,1] - test_update) < 1.0e-4
            @warn "The sample points may not be appropriately updated! Make sure this is what you want!"
        end
        v[i] = fun(numsample) 
    end
    return v
end
