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

@everywhere begin
    
    k₁ = 1
    k₂ = 1
    c = 0.9
    γ = c/(4*π^2)/(k₁^2 + k₂^2)

    V = (x)-> γ*cos(2*π*k₁*x[1])*cos(2*π*k₂*x[2])
    b = (x) -> γ*[-2*π*k₁*sin(2*π*k₁*x[1])*cos(2*π*k₂*x[2]), -2*π*k₂*cos(2*π*k₁*x[1])*sin(2*π*k₂*x[2])]
    ΔV = (x) -> V(x)*(-4*π^2)*(k₁^2 + k₂^2)

    ρ₀ = (x)->1
    ρ₁(x) = ΔV(x) + ρ₀(x)
    ρdiff(x) = ΔV(x)
    exact_mean = 1.0

    ρρ₀ = (x,y)->ρ₀([x,y])
    ρρ₁ = (x,y)->ρ₁([x,y]);

    figsize =  (300, 200);

    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1

end

############################################
# plot the model
############################################

Nx = 100; Ny = 100;
xgrid = range(xmin, stop=xmax, length=Nx)
ygrid = range(ymin, stop=ymax, length=Ny);

f1 = contour(xgrid, ygrid, (a,b)-> ρ₀([a,b]), fill=true, color=:tofino, title="ρ₀")
f2 = contour(xgrid, ygrid, (a,b)-> ρ₁([a,b]), fill=true, color=:tofino, title="ρ₁")
f3 = contour(xgrid, ygrid, (a,b)-> V([a,b]), fill=true, color=:tofino, title="V")
f4 = contour(xgrid, ygrid, (a,b)-> ΔV([a,b]), fill=true, color=:tofino, title="ΔV")

plot(f1, f2, f3, f4, left_margin=10px, right_margin=10px, upper_margin=10px, bottom_margin=10px,
    layout=(@layout grid(2,2)), size=(600,500))

############################################
# test partition
############################################

Z₁ = sum([ρ₁([xgrid[i], ygrid[j]]) for i = 1:Nx for j = 1:Ny])/Nx/Ny
@test abs(Z₁ - 1.0) < 1.0e-3;

############################################
# test zero-variance dynamics
############################################

numsample = 10^5
gpts = [rand(2) for i = 1:numsample]

T = 50.0
function flowb(x)
    v = b([x[1], x[2]])
    return T*v
end

U₀ = FixedPotential(2, x->-log(ρ₀(x)), nothing, nothing, nothing, [], (j)->gpts[j])
U₁ = FixedPotential(2, x->-log(ρ₁(x)), nothing, nothing, nothing, [], nothing)
flow = DynNN(2, x->flowb(x), [], [], (x,args...)->flowb(x), 4, [], 0, []);

test_func = [(x,t,p)->-U₀.U(x), (x,t,p)->-U₁.U(x), (x,t,p)-> T*ρdiff(x)]
test_func_para = [nothing nothing flow]

Nt = Int64(200*T)
offset = Int64(ceil(Nt/2));

function get_data(j)
    v, F₀, F₁, _ = one_path_dyn_b(2, flow, test_func, test_func_para, Nt, offset, U₀.sampler, j, RK4)
    return v, sum(F₁)/sum(F₀)
end

new_samplesize=20
@time data = map(j->get_data(j), 1:new_samplesize);

@testset "zero-variance dynamics" begin
    @test abs(maximum([item[1] for item in data]) - 1.0) < 1.0e-2
    @test abs(minimum([item[1] for item in data]) - 1.0) < 1.0e-2
    @test abs(maximum([item[2] for item in data]) - 1.0) < 1.0e-2
    @test abs(minimum([item[2] for item in data]) - 1.0) < 1.0e-2
end

############################################
# sample trajectories
############################################

t_vec = Array(range(0,stop=1.0,length=Int64(ceil(T*50))+1))
unit_step = 1

gr()
num_particle = 200

fig_traj = contour(xgrid, ygrid, (x1,x2)-> V([x1, x2]), color=:tofino, 
    fill=true,  
    size = figsize,
    left_margin = 20px, right_margin=35px)

@time for i = 1:num_particle
    idx = i
    traj_state, traj_time, _, _ = traj_dyn_b(gpts[idx], flow, t_vec, unit_step, (x,t,p)->flowb(x));
    
    traj_x = traj_state[1,:]
    traj_y = traj_state[2,:]
    
    locx(x) = LinearInterpolator(traj_time, traj_x)(x)
    locy(x) = LinearInterpolator(traj_time, traj_y)(x)
    plot!(x->locx(x), x->locy(x), traj_time[1], traj_time[end], label="", color=:white, linewidth=1.0)
end


############################################
# time map
############################################

flow_exact = DynFix(2, b)
maxT = 50.0
Nperstep = 50
N = Int64(ceil(maxT*Nperstep))
Nx = 50; Ny = Nx;
xgrid = range(0.0, stop=1.0, length=Nx+1)
ygrid = xgrid
err_mat = zeros(Nx+1, Ny+1)
Tvalue = zeros(Nx+1, Ny+1)
CXX, CYY = ModUtil.meshgrid(xgrid, ygrid)

# we do not consider the boundary points
@time for i = 2:(Nx)
    for j = 2:(Ny)
        x₀ = [xgrid[i],xgrid[j]]
        err_mat[i,j], Tvalue[i,j], _, _ = findT(maxT, N, x₀, flow_exact, ρ₀, ρ₁, ρdiff, full_info=true)
    end
end
T_interp = BilinearInterpolator(xgrid[2:Nx], ygrid[2:Ny], Tvalue[2:Nx,2:Ny]);

@printf("maximum error in time map %10.2f\n", maximum(err_mat[2:Nx, 2:Ny]))
fig_err = contour(xgrid, ygrid, log10.(err_mat), color=:tofino, fill=true, 
    size=figsize, title="error (log10)", left_margin=20px, right_margin=20px)

ϕ(z) = sign(z)*log(1+norm(z))
fig_mapT = contour(xgrid[2:Nx], xgrid[2:Ny], (z,w)-> ϕ.(T_interp(z,w)), 
    fill=true, color=:tofino, size=figsize, levels=50, 
    right_margin=20px, left_margin=20px)


############################################
# empirical distributions
############################################

numsample = 10^5
gpts = [rand(2) for i = 1:numsample]
@time sample_data_ρ₁ = pmap(x₀->findT(maxT, N, x₀, flow_exact, ρ₀, ρ₁, ρdiff)[2], gpts);
sdx = [item[1] for item in sample_data_ρ₁]
sdy = [item[2] for item in sample_data_ρ₁]
fig_emp_rho_1 = histogram2d(sdx, sdy, size=figsize, fill=true, color=:tofino, 
    left_margin=20px, right_margin=20px, normed=true, nbinsx=30, nbinsy=30)
fig_rho_1 = contour(xgrid, xgrid, (z,w)->ρ₁([z,w]), size=figsize, fill=true, color=:tofino, 
    left_margin=20px, right_margin=20px)


############################################
# save figures
############################################
savefig(fig_traj, folder*"neumann_kmode_"*string(k₁)*"_"*string(k₂)*"_flow_traj.pdf")
savefig(fig_err, folder*"neumann_kmode_"*string(k₁)*"_"*string(k₂)*"_err.pdf")
savefig(fig_mapT, folder*"neumann_kmode_"*string(k₁)*"_"*string(k₂)*"_mapT.pdf")
savefig(fig_emp_rho_1, folder*"neumann_kmode_"*string(k₁)*"_"*string(k₂)*"_emp_rho_1.pdf")
savefig(fig_rho_1, folder*"neumann_kmode_"*string(k₁)*"_"*string(k₂)*"_rho_1.pdf");
