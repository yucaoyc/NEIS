using Distributed
addprocs(4);
@everywhere push!(LOAD_PATH,"../src")
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
using BasicInterpolators
using ModTraj
using ModUtil
using ModPoisson;
using ModGenerator
using ModODEIntegrate: RK4
@everywhere using ModGenerator

plot_setup();
folder = "./assets_gen/"
~isdir(folder) ? mkdir(folder) : nothing

N = 2^9
λ = 0.2
a = 2.0
mode= 3

#prior="nonuniform"
prior="uniform"

kmat, xgrid, q₀, ρ₀, q₁, ρ₁, ρdiff, exact_mean = load_torus_eg(N, λ, a, mode=mode, prior=prior);
_, V_interp, b_interp, flow_exact = solve_poisson_2dtorus_fft(N, xgrid, ρdiff, kmat);


figsize =  (300, 200)

Nc = 100
xc = range(0, stop=1.0, length=Nc+1)
fig_rho_diff = contour(xc, xc, (x₁,x₂)->ρdiff([x₁,x₂]), fill=true, color=:tofino, 
    size=figsize, 
    left_margin = 20px, right_margin=20px)
fig_rho_0 = contour(xc, xc, (a₁, a₂)-> ρ₀([a₁, a₂]), 
    size=figsize, color=:tofino, fill=true, right_margin=20px, left_margin=20px)
fig_rho_1 = contour(xc, xc, (a₁, a₂)-> ρ₁([a₁, a₂]), 
    size=figsize, color=:tofino, fill=true, right_margin=20px, left_margin=20px)
plot(fig_rho_0, fig_rho_1, size=(800,300))

######################################
# Test zero-variance dynamics
######################################

numsample = 10^5
accept_rate, gpts, new_pts_bool = MH_generate_data_from_uniform(numsample, q₀, 2)
new_pts_idx = findall(>(0), new_pts_bool)
@printf("acceptane rate %.2f\n", accept_rate);
#fig_init_data = histogram2d([item[1] for item in gpts], [item[2] for item in gpts], 
#    nbins=20, size=(400, 300), normed=true)
#plot(fig_init_data, fig_rho_0, size=(800,300))

T = 50.0
Ω = x->mod(x,1.0)

function flowb(x)
    v = b_interp([Ω(x[1]), Ω(x[2])])
    return T*v
end

U₀ = FixedPotential(2, x->-log(q₀(x)), nothing, nothing, nothing, [], (j)->gpts[j])
U₁ = FixedPotential(2, x->-log(q₁(x)), nothing, nothing, nothing, [], nothing)
flow_neis = DynNN(2, x->flowb(x), [], [], (x,args...)->flowb(x), 4, [], 0, []);

# divg_b(p, x) = f(x) in this case.
test_func = [(x,t,p)->-U₀.U(x), (x,t,p)->-U₁.U(x), (x,t,p)-> T*ρdiff(x)]
test_func_para = [nothing nothing flow_neis]

Nt = Int64(200*T)
offset = Int64(ceil(Nt/2));

function get_data(j)
    v, F₀, F₁, _ = one_path_dyn_b(2, flow_neis, test_func, test_func_para, Nt, offset, U₀.sampler, j, RK4)
    return v, sum(F₁)/sum(F₀)
end

new_samplesize=500
@time data = map(j->get_data(j), 1:new_samplesize);
@printf("%d\n", length(findall(<(new_samplesize), new_pts_idx)))

@testset "test-zero-variance dynamics" begin
    @test abs(maximum([item[1] for item in data])/exact_mean - 1.0) < 1.0E-2
    @test abs(minimum([item[1] for item in data])/exact_mean - 1.0) < 1.0E-2
    @test abs(maximum([item[2] for item in data])/exact_mean - 1.0) < 1.0E-2
    @test abs(minimum([item[2] for item in data])/exact_mean - 1.0) < 1.0E-2
end

######################################
# Sample trajectories
######################################

t_vec = Array(range(0,stop=1.0,length=Int64(ceil(T*20))+1))
unit_step = 1
num_particle = 100

fig_traj = contour(xc, xc, (x1,x2)-> V_interp(x1, x2), color=:tofino, 
    fill=true,  
    size=(400,300), 
    left_margin = 20px, right_margin=20px)

for i = 1:num_particle
    idx = new_pts_idx[i]
    traj_state, traj_time, _, _ = traj_dyn_b(gpts[idx], flow_neis, t_vec, unit_step, (x,t,p)->flowb(x));
    
    traj_x = map(Ω, traj_state[1,:])
    traj_y = map(Ω, traj_state[2,:])
    
    t_idx = []
    for k = 1:(length(traj_time)-1)
        if abs(traj_x[k+1] - traj_x[k]) > 0.9
            push!(t_idx, k)
        elseif abs(traj_y[k+1] - traj_y[k]) > 0.9
            push!(t_idx, k)
        end
    end
    push!(t_idx, length(traj_time))
    l_idx = 1
    for k = 1:length(t_idx)
        r_idx = t_idx[k]
        if r_idx - l_idx > 1
            locx(x) = LinearInterpolator(traj_time[l_idx:r_idx], traj_x[l_idx:r_idx])(x)
            locy(x) = LinearInterpolator(traj_time[l_idx:r_idx], traj_y[l_idx:r_idx])(x)
            plot!(locx, locy, traj_time[l_idx], traj_time[r_idx], label="", color=:white, linewidth=1.0)
        end
        l_idx = r_idx + 1
    end
end

#################################
# Time map
#################################

maxT = 30.0
Nperstep = 50
N = Int64(ceil(maxT*Nperstep))

Nx = 50; Ny = Nx;
xgrid = range(0.0, stop=1.0, length=Nx+1)
ygrid = xgrid
CXX = zeros(Nx+1, Ny+1)
CYY = zeros(Nx+1, Ny+1)
Tvalue = zeros(Nx+1, Ny+1)
@time for i = 1:(Nx+1)
    for j = 1:(Ny+1)
        x₀ = [xgrid[i],xgrid[j]]
        CXX[i,j] = x₀[1]
        CYY[i,j] = x₀[2]
        Tvalue[i,j], _, _ = findT(maxT, N, x₀, flow_exact, ρ₀, ρ₁, ρdiff)
    end
end
T_interp = BilinearInterpolator(xgrid, ygrid, Tvalue);
fig_mapT = contour(xgrid, xgrid, (z,w)-> T_interp(z,w), fill=true, color=:tofino, size=figsize, levels=50, 
    right_margin=20px, left_margin=20px)

#################################
# Empirical distribution
#################################

numsample = 10^5
if prior == "uniform"
    gpts = [rand(2) for i = 1:numsample]
end
@time sample_data_ρ₁ = pmap(x₀->findT(maxT, N, x₀, flow_exact, ρ₀, ρ₁, ρdiff)[2], gpts);

sdx = [mod(item[1],1) for item in sample_data_ρ₁]
sdy = [mod(item[2],1) for item in sample_data_ρ₁]
fig_emp_rho_1 = histogram2d(sdx, sdy, size=figsize, fill=true, color=:tofino, 
    left_margin=20px, right_margin=20px, normed=true, nbinsx=30, nbinsy=30)
fig_rho_1 = contour(xgrid, xgrid, (z,w)->ρ₁([z,w]), size=figsize, fill=true, color=:tofino, 
    left_margin=20px, right_margin=20px)

#################################
# Save results
#################################

savefig(fig_traj, folder*"torus_"*prior*"_flow_traj_"*string(mode)*".pdf")
savefig(fig_mapT, folder*"torus_"*prior*"_timemap_"*string(mode)*".pdf")
savefig(fig_emp_rho_1, folder*"torus_"*prior*"_emp_rho_1_"*string(mode)*".pdf")
savefig(fig_rho_1, folder*"torus_"*prior*"_rho_1_"*string(mode)*".pdf")
