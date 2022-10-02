##############################
# Three-mode mixture with neumann bc
##############################
using LinearAlgebra, BasicInterpolators, Printf
using Plots, LaTeXStrings, Plots.PlotMeasures
using NEIS

figsize = (350, 300)
plot_setup()
folder="./assets/"
~isdir(folder) ? mkdir(folder) : nothing

##############################
# Load the model and solution of Poisson eq.
##############################
n = 2
N = 2^9; λ = 0.2; a = 2.0; mode = 3; prior="uniform"
_, xgrid, q₀, ρ₀, q₁, ρ₁, ρdiff, U₀, U₁, exact_mean =
    load_torus_eg(N, λ, a, mode=mode, prior=prior)

xmin = 0; xmax = 1; ymin = 0; ymax = 1; Nx = N; Ny = N
@time sol = solve_2d_poisson_neumann(xmin, xmax, ymin, ymax, Nx, Ny, (x,y)->ρ₀([x,y]), (x,y)->ρ₁([x,y]))
V = sol["V"]
flow = DynFix{Float64}(n, x->sol["b"](x))

# No need to plot ρ₀, ρ₁; same model as poisson_torus.jl

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

####################################
## sample trajectories
####################################
fig_traj = plot_rho_and_flow(xmin, xmax, ymin, ymax, sample_to_test, flow, V, figsize,
                             Tscale=20.0)
savefig(fig_traj, folder*"neumann_"*prior*"_flow_traj_"*string(mode)*".pdf")
