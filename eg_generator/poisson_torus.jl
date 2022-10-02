##############################
# Three-mode mixture with periodic bc
##############################
using LinearAlgebra, BasicInterpolators, Printf
using Plots, LaTeXStrings, Plots.PlotMeasures
using NEIS

figsize =  (350, 300)
plot_setup()
folder = "./assets/"
~isdir(folder) ? mkdir(folder) : nothing

##############################
# Load the model and solution of Poisson eq.
##############################
n = 2;
N = 2^9; λ = 0.2; a = 2.0; mode= 3; prior="uniform"
kmat, xgrid, q₀, ρ₀, q₁, ρ₁, ρdiff, U₀, U₁, exact_mean =
    load_torus_eg(N, λ, a, mode=mode, prior=prior)
# solve b and flow; this b_interp can already map R^2 into torus [0,1]^2.
_, V_interp, b_interp, flow = solve_poisson_2dtorus_fft(N, xgrid, ρdiff, kmat)

##############################
# plot ρ₀, ρ₁, and ρ₁ - ρ₀.
##############################
Nc = 100
xc = range(0, stop=1.0, length=Nc+1)
fig_rho_1 = contour(xc, xc, (a₁, a₂)-> ρ₁([a₁, a₂]),
    size=figsize, color=:tofino, fill=true, margin=20px)

##############################
## Test zero-variance dynamics
##############################
numsample = 10^5
accept_rate, gpts, _ = MH_generate_data_from_uniform(numsample, q₀, 2)
accept_rate < 0.95 ? @warn("acceptance rate too low! double check the code!") : nothing

Tscale = 50.0
num_sample_to_test = 200
sample_to_test = hcat(gpts[1:num_sample_to_test]...)
@time test_zero_var_dyn_for_infty_time(U₀, U₁, sample_to_test, flow, Tscale,
                                       exact_mean=exact_mean, ρdiff=ρdiff, ϵ=0.01)

##############################
## Sample trajectories
##############################
xmin = 0; xmax = 1; ymin = 0; ymax = 1;
fig_traj = plot_rho_and_flow(xmin, xmax, ymin, ymax, sample_to_test, flow, V_interp, figsize,
                             Tscale=20.0, plot_torus=true)

##############################
## Time map
##############################
maxT = 30.0; Nperstep = 50; N = Int64(ceil(maxT*Nperstep))
global count_large_T = Threads.Atomic{UInt128}(0)

Nx = 50; Ny = Nx;
xgrid = Array(range(0.0, stop=1.0, length=Nx+1))
ygrid = xgrid
CXX, CYY = meshgrid(xgrid, ygrid)
Tvalue = zeros(Nx+1, Ny+1)
@time Threads.@threads for i = 1:(Nx+1)
    for j = 1:(Ny+1)
        x₀ = [xgrid[i], ygrid[j]]
        Tvalue[i,j], _, _ = find_τ(maxT, N, x₀, flow, ρ₀, ρ₁, ρdiff)
        if abs.(Tvalue[i,j]) > maxT - 1
            query_add!(count_large_T, 1)
        end
    end
end
T_interp = BilinearInterpolator(xgrid, ygrid, Tvalue);
fig_mapT = contour(xgrid, xgrid, (z,w)-> T_interp(z,w), fill=true, color=:tofino,
                   size=figsize, levels=50, margin=20px)
@printf("Percent of large T is %.2f\n", Int64(count_large_T[])/(Nx+1)/(Ny+1))

##################################
## Empirical distribution
##################################
sample_data_ρ₁ = [zeros(n) for i = 1:numsample]
@time Threads.@threads for i = 1:numsample
    _, sample_data_ρ₁[i], _ = find_τ(maxT, N, gpts[i], flow, ρ₀, ρ₁, ρdiff)
end
sdx = [mod(item[1],1) for item in sample_data_ρ₁]
sdy = [mod(item[2],1) for item in sample_data_ρ₁]
fig_emp_rho_1 = histogram2d(sdx, sdy, size=figsize, fill=true, color=:tofino,
    margin=20px, normed=true, nbinsx=30, nbinsy=30)

##################################
## Save results
##################################
savefig(fig_traj, folder*"torus_"*prior*"_flow_traj_"*string(mode)*".pdf")
savefig(fig_mapT, folder*"torus_"*prior*"_timemap_"*string(mode)*".pdf")
savefig(fig_emp_rho_1, folder*"torus_"*prior*"_emp_rho_1_"*string(mode)*".pdf")
savefig(fig_rho_1, folder*"torus_"*prior*"_rho_1_"*string(mode)*".pdf")
