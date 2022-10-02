##############################
# The example in Appendix G.2.3 of arXiv:2206.09908v1
##############################
using LinearAlgebra, BasicInterpolators, Printf, Test
using Plots, LaTeXStrings, Plots.PlotMeasures
using NEIS

figsize = (350, 300)
plot_setup()
folder="./assets/"
~isdir(folder) ? mkdir(folder) : nothing

##############################
# Load the model and solution of Poisson eq.
##############################
n = 2 # dimension

k₁ = 1; k₂ = 1; c = 0.9
γ = c/(4*π^2)/(k₁^2 + k₂^2)

V = (x)-> γ*cos(2*π*k₁*x[1])*cos(2*π*k₂*x[2])
b = (x) -> γ*[-2*π*k₁*sin(2*π*k₁*x[1])*cos(2*π*k₂*x[2]), -2*π*k₂*cos(2*π*k₁*x[1])*sin(2*π*k₂*x[2])]
ΔV = (x) -> V(x)*(-4*π^2)*(k₁^2 + k₂^2)

ρ₀ = (x)->1
ρ₁(x) = ΔV(x) + ρ₀(x)
ρdiff(x) = ΔV(x)
exact_mean = 1.0

U₀ = ExplicitPotential(Float64, 2, x->-log(ρ₀(x)))
U₁ = ExplicitPotential(Float64, 2, x->-log(ρ₁(x)))
flow = DynFix{Float64}(n, x -> b(x))

xmin = 0; xmax = 1; ymin = 0; ymax = 1
Nx = 100; Ny = Nx;

# Remark: V automatically solves the Poisson's eq.; No need to solve Poisson's eq anymore.

##############################
## plot the model
##############################
xgrid = Array(range(xmin, stop=xmax, length=(Nx+1)))
ygrid = Array(range(ymin, stop=ymax, length=(Ny+1)))
fig_rho_1 = contour(xgrid, ygrid, (a,b)-> ρ₁([a,b]), fill=true, color=:tofino, margin=20px, size=figsize)

# test partition
Z₁ = sum([ρ₁([xgrid[i], ygrid[j]]) for i = 1:Nx for j = 1:Ny])/Nx/Ny
@test abs(Z₁ - exact_mean) < 1.0e-3

##############################
## test zero-variance dynamics
##############################
numsample = 10^5
gpts = [rand(2) for i = 1:numsample]

Tscale = 50.0
num_sample_to_test = 200
sample_to_test = hcat(gpts[1:num_sample_to_test]...)
@time test_zero_var_dyn_for_infty_time(U₀, U₁, sample_to_test, flow, Tscale,
        exact_mean=exact_mean, ρdiff=ρdiff, ϵ=0.01)

#############################################
## sample trajectories
#############################################
fig_traj = plot_rho_and_flow(xmin, xmax, ymin, ymax, sample_to_test, flow,
                             (x,y)->V([x,y]), figsize, Tscale=20.0, margin=30px)

#############################################
## time map
#############################################
maxT = 30.0; Nperstep = 50; N = Int64(ceil(maxT*Nperstep))
global count_large_T = Threads.Atomic{UInt128}(0)
CXX, CYY = meshgrid(xgrid, ygrid)
Tvalue = zeros(Nx+1, Ny+1)
err_mat = zeros(Nx+1, Ny+1)
# we do not consider the boundary points
# as it can be numerically difficult to compute τ for boundary points.
@time Threads.@threads for i = 2:Nx
    for j = 2:Ny
        x₀ = [xgrid[i], ygrid[j]]
        err_mat[i,j], Tvalue[i,j], _, _ = find_τ(maxT, N, x₀, flow, ρ₀, ρ₁, ρdiff, full_info=true)
        if abs.(Tvalue[i,j]) > maxT - 1
            query_add!(count_large_T, 1)
        end
    end
end

ϕ(z) = sign(z)*log(1+norm(z))
T_interp = BilinearInterpolator(xgrid[2:Nx], ygrid[2:Ny], Tvalue[2:Nx,2:Ny])
fig_mapT = contour(xgrid[2:Nx], ygrid[2:Ny], (z,w)-> ϕ(T_interp(z,w)), fill=true, color=:tofino,
                   size=figsize, levels=50, margin=20px)

@printf("Percent of large T is %.2f\n", Int64(count_large_T[])/(Nx-1)/(Ny-1))

@printf("Maximum value of error in time map %10.2f\n", maximum(err_mat[2:Nx, 2:Ny]))
fig_err = contour(xgrid, ygrid, log10.(err_mat), color=:tofino, fill=true,
    size=figsize, title="error (log10)", margin=20px)

#############################################
## empirical distributions
#############################################
sample_data_ρ₁ = [zeros(n) for i = 1:numsample]
@time Threads.@threads for i = 1:numsample
    _, sample_data_ρ₁[i], _ = find_τ(maxT, N, gpts[i], flow, ρ₀, ρ₁, ρdiff)
end

sdx = [item[1] for item in sample_data_ρ₁]
sdy = [item[2] for item in sample_data_ρ₁]
fig_emp_rho_1 = histogram2d(sdx, sdy, size=figsize, fill=true, color=:tofino,
    margin=20px, normed=true, nbinsx=30, nbinsy=30)

#############################################
## save figures
#############################################
savefig(fig_traj, folder*"neumann_kmode_"*string(k₁)*"_"*string(k₂)*"_flow_traj.pdf")
savefig(fig_err, folder*"neumann_kmode_"*string(k₁)*"_"*string(k₂)*"_err.pdf")
savefig(fig_mapT, folder*"neumann_kmode_"*string(k₁)*"_"*string(k₂)*"_mapT.pdf")
savefig(fig_emp_rho_1, folder*"neumann_kmode_"*string(k₁)*"_"*string(k₂)*"_emp_rho_1.pdf")
savefig(fig_rho_1, folder*"neumann_kmode_"*string(k₁)*"_"*string(k₂)*"_rho_1.pdf");
