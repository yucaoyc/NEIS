using Distributed
addprocs(4)
@everywhere push!(LOAD_PATH,"../src")

@everywhere begin
    using ModOpt
    using ModOptInt
    using ModDyn
    using ModPotential
    using FFTW
    using LinearAlgebra
    using Plots
    using SparseArrays
    using BenchmarkTools
    using Test
    using LaTeXStrings
    using Printf
    using Plots.PlotMeasures
    using ModTraj
    using ModUtil
    using ModPoisson
    using ModGenerator
    using BasicInterpolators
    using ModODEIntegrate: RK4
    plot_setup();
    folder="./assets_gen/";
end

~isdir(folder) ? mkdir(folder) : nothing

###################################
# set-up the model
###################################

N = 2^9
λ = 0.2
a = 2.0
mode = 3
prior="uniform"

_, xgrid, q₀, ρ₀, q₁, ρ₁, ρdiff, exact_mean = load_torus_eg(N, λ, a, mode=mode, prior=prior);

ρρ₀ = (x,y)->ρ₀([x,y])
ρρ₁ = (x,y)->ρ₁([x,y]);

figsize =  (300, 200);

xmin = 0
xmax = 1
ymin = 0
ymax = 1
Nx = N
Ny = N

@time Δx, Δy, xgrid, ygrid, CXX, CYY, Φ = solve_poisson(xmin, xmax, ymin, ymax, Nx, Ny, ρρ₀, ρρ₁);
gradϕx, gradϕy, Divgϕ = get_grad_phi(Nx, Ny, Φ, Δx, Δy);

@everywhere begin
    V_interp = BilinearInterpolator($xgrid, $ygrid, $Φ)
    bx = BilinearInterpolator($xgrid, $ygrid, $gradϕx)
    by = BilinearInterpolator($xgrid, $ygrid, $gradϕy)

    function b_interp(x)
        return [bx(x[1],x[2]), by(x[1],x[2])]
    end

    function b_interp(x,y)
        return b_interp([x,y])
    end
end

# ϵ = 1.0e-8

# @testset "poision solver" begin
#     for iter = 1:10
#         i = Int64(ceil(rand()*Nx))
#         j = Int64(ceil(rand()*Ny))

#         @test abs((Φ[i,j] - V_interp(xgrid[i], ygrid[j]))/Φ[i,j]) < ϵ
#         @test abs((Divgϕ[i,j] - ρdiff([xgrid[i],ygrid[j]]))/Divgϕ[i,j]) < ϵ
#     end
# end

# @test norm([norm(gradϕx[1,:]), norm(gradϕx[Nx,:]), norm(gradϕy[:,1]), norm(gradϕy[:,Ny])]) < ϵ

###################################
# test zero-variance dynamics
###################################

numsample = 10^4
accept_rate, gpts, new_pts_bool = MH_generate_data_from_uniform(numsample, q₀, 2)
new_pts_idx = findall(>(0), new_pts_bool)

T = 50.0
function flowb(x)
    v = b_interp([x[1], x[2]])
    return T*v
end

U₀ = FixedPotential(2, x->-log(q₀(x)), nothing, nothing, nothing, [], (j)->gpts[j])
U₁ = FixedPotential(2, x->-log(q₁(x)), nothing, nothing, nothing, [], nothing)
flow = DynNN(2, x->flowb(x), [], [], (x,args...)->flowb(x), 4, [], 0, []);

# divg_b(p, x) = f(x) in this case.
test_func = [(x,t,p)->-U₀.U(x), (x,t,p)->-U₁.U(x), (x,t,p)-> T*ρdiff(x)]
test_func_para = [nothing nothing flow]

Nt = Int64(200*T)
offset = Int64(ceil(Nt/2));

function get_data(j)
    v, F₀, F₁, _ = one_path_dyn_b(2, flow, test_func, test_func_para, Nt, offset, U₀.sampler, j, RK4)
    return v, sum(F₁)/sum(F₀)
end

new_samplesize = 500
@time data = map(j->get_data(j), 1:new_samplesize);

println(length(findall(<(new_samplesize), new_pts_idx)))
@testset "test-zero-variance dynamics" begin
    @test abs(maximum([item[1] for item in data])/exact_mean - 1.0) < 1.0E-2
    @test abs(minimum([item[1] for item in data])/exact_mean - 1.0) < 1.0E-2
    @test abs(maximum([item[2] for item in data])/exact_mean - 1.0) < 1.0E-2
    @test abs(minimum([item[2] for item in data])/exact_mean - 1.0) < 1.0E-2
end

###################################
# sample trajectories
###################################

t_vec = Array(range(0,stop=1.0,length=Int64(ceil(T*50))+1))
unit_step = 1

gr()
num_particle = 100

fig_traj = contour(xgrid, ygrid, (x1,x2)-> V_interp(x1, x2), color=:tofino, 
    fill=true,  
    size=(400,300), 
    left_margin = 20px, right_margin=20px)

for i = 1:num_particle
    idx = new_pts_idx[i]
    traj_state, traj_time, _, _ = traj_dyn_b(gpts[idx], flow, t_vec, unit_step, (x,t,p)->flowb(x));
    
    traj_x = traj_state[1,:]
    traj_y = traj_state[2,:]
    
    locx(x) = LinearInterpolator(traj_time, traj_x)(x)
    locy(x) = LinearInterpolator(traj_time, traj_y)(x)
    plot!(x->locx(x), x->locy(x), traj_time[1], traj_time[end], label="", color=:white, linewidth=1.0)
end
savefig(fig_traj, folder*"neumann_"*prior*"_flow_traj_"*string(mode)*".pdf")

# ###################################
# # Time map
# ###################################

# flow_exact = DynFix(2, b_interp)

# maxT = 30.0
# Nperstep = 50
# N = Int64(ceil(maxT*Nperstep))

# Nx = 50; Ny = Nx;
# xgrid = range(0.0, stop=1.0, length=Nx+1)
# ygrid = xgrid
# err_mat = zeros(Nx+1, Ny+1)
# CXX = zeros(Nx+1, Ny+1)
# CYY = zeros(Nx+1, Ny+1)
# Tvalue = zeros(Nx+1, Ny+1);

# # we do not consider the boundary points

# @time for i = 2:(Nx)
#     for j = 2:(Ny)
#         x₀ = [xgrid[i],xgrid[j]]
#         CXX[i,j] = x₀[1]
#         CYY[i,j] = x₀[2]
#         err_mat[i,j], Tvalue[i,j], _, _ = findT(maxT, N, x₀, flow_exact, ρ₀, ρ₁, ρdiff, full_info=true)
#     end
# end

# T_interp = BilinearInterpolator(xgrid[2:Nx], ygrid[2:Ny], Tvalue[2:Nx,2:Ny]);

# @printf("error %.2E\n", maximum(err_mat[2:Nx, 2:Ny]))

# ########################################
# # time map
# ########################################
# fig_mapT = contour(xgrid[2:Nx], xgrid[2:Ny], (z,w)-> T_interp(z,w), 
#     fill=true, color=:tofino, size=figsize, levels=50, 
#     right_margin=20px, left_margin=20px)
# savefig(fig_mapT, folder*"neumann_"*prior*"_timemap_"*string(mode)*".pdf")

# ########################################
# # empirical distribution
# ########################################
# numsample = 10^5

# if prior == "uniform"
#     gpts = [rand(2) for i = 1:numsample]
# end
# @time sample_data_ρ₁ = pmap(x₀->findT(maxT, N, x₀, flow_exact, ρ₀, ρ₁, ρdiff)[2], gpts);

# sdx = [item[1] for item in sample_data_ρ₁]
# sdy = [item[2] for item in sample_data_ρ₁]

# fig_emp_rho_1 = histogram2d(sdx, sdy, size=figsize, fill=true, color=:tofino, 
#     left_margin=20px, right_margin=20px, normed=true, nbinsx=50, nbinsy=50)
# savefig(fig_emp_rho_1, folder*"neumann_"*prior*"_emp_rho_1_"*string(mode)*".pdf")

# fig_rho_1 = contour(xgrid, xgrid, (z,w)->ρ₁([z,w]), size=figsize, fill=true, color=:tofino, 
#     left_margin=20px, right_margin=20px)

# savefig(fig_traj, folder*"neumann_"*prior*"_flow_traj_"*string(mode)*".pdf")
