export get_resource,
       get_train_stat,
       print_train_stat,
       get_ndims,
       plot_setup,
       randHermitian,
       remove_nan,
       get_relative_stats,
       integrate_over_time,
       domain_ball,
       domain_rectangle,
       prod_domain,
       print_stat_name,
       add_procs,
       repeat_experiment,
       get_mean_sec_moment
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
Get CPU information, number of processors used, and number of threads.
"""
function get_resource()
    return Dict{Symbol,Any}(:cpu=>Sys.cpu_info()[1].model,
                :nprocs=>nprocs(),
                :nthreads=>Threads.nthreads())
end

"""
Pack resources, runtime and query statistics into a Dictionary.
"""
function get_train_stat(train_time, U₁)
    # get training resources
    info = get_resource()
    # train time
    info[:runtime] = train_time
    # queries
    info[:query] = get_query_stat(U₁)

    return info
end

function print_train_stat(info; type=:nonzero)
    @printf("cpu is %s\n", info[:cpu])
    @printf("nprocs=%d, nthreads=%d\n", info[:nprocs], info[:nthreads])
    @printf("runtime %.2f (seconds)\n", info[:runtime])
    a, b, c, d = info[:query]
    if type == :full
        @printf("query (U): %10s\n", datasize(a))
        @printf("query (∇U): %9s\n", datasize(b))
        @printf("query (∇²U): %8s\n", datasize(c))
        @printf("query (ΔU): %9s\n", datasize(d))
    elseif type == :nonzero
        a > 0 ? @printf("query (U): %10s\n", datasize(a)) : nothing
        b > 0 ? @printf("query (∇U): %9s\n", datasize(b)) : nothing
        c > 0 ? @printf("query (∇²U): %8s\n", datasize(c)) : nothing
        d > 0 ? @printf("query (ΔU): %9s\n", datasize(d)) : nothing
    end
end

"""
Get the deep nn layer structure for an ℓ-layer nn with input and output dimension as n
and each inner layer having width m.
"""
get_ndims(n,m,ℓ) = vcat([n],m*ones(Int64,ℓ-1),[n])

"""
A default plot setup.
"""
function plot_setup(; titlefontsize=12, legendfontsize=9, guifontsize=11)
    default(titlefont = (titlefontsize, "times"), legendfontsize=legendfontsize,
        legend_font_family="times", guidefont = (guifontsize, "times"),
        fg_legend = :transparent);
end

"""
Return a n-by-n random Hermitian matrix with spectrum between [σmin, σmax]
"""
function randHermitian(n, σmin::T, σmax::T) where T<:AbstractFloat
    trial_num = 1
    while (trial_num < 10)
        Q , _ = qr(rand(T, n, n))
        D = Diagonal(rand(T, n)*(σmax-σmin) .+ σmin)
        #M = Q*D*transpose(Q)
        M = Q*D*Q'
        if typeof(M) <: AbstractMatrix{T}
            return M
        else
            trial_num += 1
            continue
        end
    end
    if trial_num == 10
        @error("Problem with randHermitian function.")
    end
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

function prod_domain(Ω₁::Function, Ω₂::Function, n₁::Int, n₂::Int)
    Ω(x) = Ω₁(x[1:n₁]) && Ω₂(x[(n₁+1):(n₁ + n₂)])
    return Ω
end

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
        gpts::AbstractMatrix{T}, gpts_sampler::Function) where T<:AbstractFloat
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

# todo: add NaN protection
function get_mean_sec_moment(data::Vector{Vector{T}}) where T<:AbstractFloat
    numsample = length(data)
    L = length(data[1])
    fst_m = zeros(T, L)
    sec_m = zeros(T, L)
    for j = 1:numsample
        fst_m .+= data[j]
        sec_m .+= data[j].^2
    end
    fst_m /= numsample
    sec_m /= numsample

    return fst_m, sec_m
end

function pretty_float(f::AbstractFloat)
    if abs(f) > 1.0e3 || abs(f) < 1.0e-3
        @sprintf("%1.3E", f)
    else
        @sprintf("%1.3f", f)
    end
end
