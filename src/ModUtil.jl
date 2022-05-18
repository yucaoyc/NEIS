module ModUtil

using Plots
using Distributed
using Random
using LinearAlgebra
using Statistics
using Printf

export get_ndims, 
       plot_setup,
       randHermitian, 
       remove_nan, 
       get_relative_stats, 
       integrate_over_time,
       meshgrid,
       get_grids, 
       get_sparse_grids,
       get_dirichlet_rescale,
       domain_ball,
       domain_rectangle,
       prod_domain,
	   print_stat_name,
       add_procs,
       repeat_experiment

get_ndims(n,m,ℓ) = vcat([n],m*ones(Int64,ℓ-1),[n])

function plot_setup()
    gr()
    default(titlefont = (12, "times"), legendfontsize=8, 
        legend_font_family="times", guidefont = (11, "times"), 
        fg_legend = :transparent);
end

function randHermitian(n, σmin, σmax)
    # return a n-by-n random Hermitian matrix with spectrum between [σmin, σmax]

    Q , _ = qr(randn(n,n))
    D = Diagonal(rand(n)*(σmax-σmin) .+ σmin)
    return Q*D*transpose(Q)

end

function remove_nan(data::Array{},verbose=false)

    ndata = length(data)
    data = filter(x-> !isnan(x), data)
    percent = (ndata - length(data))/ndata
    if verbose
        @printf("Percentage of NaN: %.3f\n", percent)
    end
    return data, percent

end

function get_relative_stats(data::Array{Float64}, exact_mean::Float64; verbose=false)
    
    rela_err = abs(mean(data)-exact_mean)/exact_mean
    rela_var = (mean(data.^2) - (mean(data))^2)/exact_mean^2
    if verbose
        @printf("Relative error: %.3f\n", rela_err)
        @printf("Relative variance: %.3f\n", rela_var)
    end
    return rela_err, rela_var
    
end

function integrate_over_time(N::Int64, h::Float64, Vf::Array{}, Vb::Array{})
    
    Jf = zeros(N+1); Jb = zeros(N+1)
    for k = 2:(N+1)
        Jf[k] = Jf[k-1] + (Vf[k] + Vf[k-1])/2*h
        Jb[k] = Jb[k-1] - (Vb[k] + Vb[k-1])/2*h
    end
    return vcat(reverse(Jb)[1:N], Jf)
    
end

"""
    return the meshgrid
"""
function meshgrid(xgrid, ygrid)
    Nx = length(xgrid)
    Ny = length(ygrid)
    CXX = zeros(Nx, Ny)
    CYY = zeros(Nx, Ny)
    for i = 1:Nx
        for j = 1:Ny
            CXX[i,j] = xgrid[i]
            CYY[i,j] = ygrid[j]
        end
    end
    return CXX, CYY
end

# grid points for training.
function get_grids(Δx, Δy, xmin, xmax, ymin, ymax)
    ΔA = (Δx*Δy)
    xx = Vector(xmin:Δx:xmax)
    yy = Vector(ymin:Δy:ymax)
    grid_pts = []
    for i = 1:length(xx)
        for j = 1:length(yy)
            push!(grid_pts, [xx[i],yy[j]])
        end
    end
    return ΔA, xx, yy, grid_pts
end

function get_sparse_grids(xmin, xmax, ymin, ymax; sp=12)
    # sparse points for plotting vector fields.
    sparse_xx = Vector(xmin:(xmax-xmin)/sp:xmax)
    sparse_yy = Vector(ymin:(ymax-ymin)/sp:ymax)
    xxb = [x for x in sparse_xx for y in sparse_yy]
    yyb = [y for x in sparse_xx for y in sparse_yy]
    return xxb, yyb
end

function get_dirichlet_rescale(Ω, n, a=1.0)
    α(x) = Ω(x) ? a : 0.0
    αgrad(x) = zeros(n)
    αhess(x) = zeros(n, n)
    return α, αgrad, αhess
end

function domain_ball(radius)
    Ω(x) = norm(x) < radius ? true : false
    return Ω
end

function domain_rectangle(x, lb, ub)
    r = (x .<= ub) .& (x .>= lb)
    prod(r)
end

function prod_domain(Ω₁, Ω₂, n₁, n₂)
    Ω(x) = Ω₁(x[1:n₁]) && Ω₂(x[(n₁+1):(n₁ + n₂)])
    return Ω
end

function print_stat_name(name, m, var, exact_mean)
    @printf("%10s: mean %6.2f variance %6.2f\n", name, m/exact_mean, var/exact_mean^2)
    @printf("%s\n", repeat("-", 40))
end

function add_procs(n1=2, n2=8)
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
function repeat_experiment(fun, numsample, numrepeat, gpts, gpts_sampler)

    v = zeros(numrepeat)
    for i = 1:numrepeat
        # refresh data
        for j = 1:numsample
            gpts[:,j] = gpts_sampler()
        end
        v[i] = fun(numsample) 
    end
    return v
end

end
