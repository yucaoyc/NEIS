to_train = false
include("../example/test2/test2_nn.jl")
folder = "../example/test2/";
push!(LOAD_PATH,"../src")
using ModDyn
using ModUtil: repeat_experiment
using ModTraj
using BasicInterpolators
using ModSMC
using Statistics
using StatsPlots, DataFrames

using Plots
using LaTeXStrings
using Plots.PlotMeasures

gr()

default(titlefont = (12), legendfontsize=8, 
    guidefont = (11), 
    fg_legend = :transparent)

include("plot_training.jl");

figsize = (350, 250)

#######################################
# Training result
#######################################

start_idx = 100
linewidth= 1.2

#fig_type = :err
fig_type = :var
#fig_type = :grad
#yscale=:log10
yscale=:identity

ℓm_list = [(ℓ, m) for ℓ in ℓ_list for m in m_list]
fig  = Array{Any}(undef, length(ℓm_list), length(model_list))
ansatz_list = ["generic", "gradient", "divergence-free"]

for i = 1:length(ℓm_list) 
    for j = 1:length(model_list)
        ℓ, m = ℓm_list[i]
        model_num = model_list[j]
        if j == 1 
            ylabel = L"{\bf \ell = %$(ℓ), m = %$(m)}"
        else 
            ylabel = "" 
        end
        if i == 1 
            title = ansatz_list[j] 
        else 
            title = "" 
        end
        xlabel= ""
        
        fig[i,j] = plot_error(m, ℓ, model_num, seed_list, exact_mean, folder, 
                fig_type=fig_type, start_idx = start_idx, xlabel=xlabel, ylabel=ylabel, title=title, 
                yscale=yscale, linewidth=linewidth, nolabel=false)
    end 
end    

xtic = [100,150,200]
fig_simu = plot(fig[1,1], fig[1,2], fig[1,3], 
    fig[2,1], fig[2,2], fig[2,3], 
    fig[3,1], fig[3,2], fig[3,3], 
    fig[4,1], fig[4,2], fig[4,3], 
    size=(600,650), layout=(@layout grid(4,3)),  
    xticks=(xtic,xtic))
savefig(fig_simu, "assets/test2_simu_"*string(fig_type)*".pdf")

#######################################
# Flow lines and sample distributions for a particular example
#######################################


xmin = -3
xmax = 8
ymin = -8
ymax = 3

m = 20
ℓ = 3
model_num = 2
seed = 1

xc = range(xmin, stop=xmax, length=10^2)
yc = range(ymin, stop=ymax, length=10^2)
casename =  folder*@sprintf("case_%d_model_%d_%d_%d_%d_%d_%d_%d_training_data.jld2", 
        testcasenum, model_num, n, N, numsample_max, m, ℓ, seed)

flow = load(casename, "flow")

Random.seed!(seed)
gpts = SharedArray(randn(n, numsample_max))
U₀.sampler = (j)->gpts[:,j]

T = 1.0
t_vec = Array(range(0,stop=1.0,length=Int64(ceil(T*50))+1))
unit_step = 5
num_particle = 100
traj_gpts = [U₀.sampler(i) for i in 1:num_particle]

#xb = range(xmin, stop=xmax, length=14)
#yb = range(ymin, stop=ymax, length=14)
#xxb = [x for x in xb for y in yb]
#yyb = [y for x in xb for y in yb]
#df = (x,y)->flow.f([x,y], flow.para_list...)
#scale = 1.0/maximum(map(norm, map(df, xxb, yyb)))
f1 = Plots.contour(xc, yc, (x,y)-> U₁.U([x,y]), fill=true, 
    color=:tofino, size=figsize, xlims=(xmin,xmax), ylims=(ymin,ymax),
    left_margin=20px,right_margin=20px)
plot_traj(num_particle, traj_gpts, flow, t_vec, unit_step)
#Plots.quiver!(xxb, yyb, quiver=(x,y)->scale*df(x,y), 
#    xlims=(xmin,xmax), ylims=(ymin, ymax), color=:orangered3)

data, emp_m, emp_var = get_data_err_var(U₀, U₁, flow, N, numsample_max)
f2 = histogram(data/exact_mean, nbins=40, label="", normed=true, size=figsize,
    left_margin=20px,right_margin=20px)
@printf("%.2E %.2E\n", emp_m/exact_mean, emp_var/exact_mean^2)
savefig(f1, @sprintf("assets/test2_flow_%d_%d_%d_%d.pdf",model_num,ℓ,m,seed))
savefig(f2, @sprintf("assets/test2_sample_%d_%d_%d_%d.pdf",model_num,ℓ,m,seed));

#######################################
# Comparison
#######################################

Random.seed!(1)
numsample = 10^3
numrepeat = 10;
U₀.sampler = j->gpts[:,j]

K = 10
βlist = Array(range(0, stop=1.0, length=K+1))
τ = 0.1
fun_ais_10 = numsample->mean(map(j->ais_neal(U₀.sampler(j), n, U₀, U₁, K, βlist, τ)[1], 1:numsample))
@time data_ais_10_rep = repeat_experiment(fun_ais_10, numsample, numrepeat, gpts, gpts_sampler);

K = 100
βlist = Array(range(0, stop=1.0, length=K+1))
τ = 0.1
fun_ais_100 = numsample->mean(map(j->ais_neal(U₀.sampler(j), n, U₀, U₁, K, βlist, τ)[1], 1:numsample))
@time data_ais_100_rep = repeat_experiment(fun_ais_100, numsample, numrepeat, gpts, gpts_sampler);

Ngd = 2*N # time-discretization; for gradient-flow only
Ωq = domain_ball(30);
flow_gd_1 = generate_gradient_flow(U₁, 1.0, Ωq);
fun_gd_1 = numsample-> get_data_err_var(U₀, U₁, flow_gd_1, Ngd, numsample)[2]
@time data_gd_rep_1 = repeat_experiment(fun_gd_1, numsample, numrepeat, gpts, gpts_sampler);

fun_nn = numsample -> get_data_err_var(U₀, U₁, flow, N, numsample)[2]
@time data_nn_rep = repeat_experiment(fun_nn, numsample, numrepeat, gpts, gpts_sampler);

# boxplot
df = DataFrame(ais10=data_ais_10_rep/exact_mean, 
    ais100=data_ais_100_rep/exact_mean, 
    gd1=data_gd_rep_1/exact_mean, 
    nn=data_nn_rep/exact_mean)
fig = @df df boxplot([:ais10, :ais100, :gd1, :nn], 
    xticks = (1:1:5, ["AIS-10", "AIS-100", "Direct GD", "NN"]),
    fillalpha = 0.75, linewidth=2, label="", size=(600,250), outliers=false)
plot!(0:1:5, ones(6), linewidth=2, label="", color=:red, linestyle=:dash)
plot!(xlim=(0.2, 4.8))
savefig(fig, "assets/test2_compare.pdf")

println(df);
