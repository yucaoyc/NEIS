to_train = false
include("../example/test3/test3_nn.jl")
folder = "../example/test3/";
include("plot_training.jl");
using ModUtil: repeat_experiment
using ModTraj
using BasicInterpolators
using ModSMC
using Statistics
using StatsPlots, DataFrames
using ModDyn

using Plots
using LaTeXStrings
using Plots.PlotMeasures

gr()

default(titlefont = (12), legendfontsize=8, 
    guidefont = (11), 
    fg_legend = :transparent)

figsize=(350,250)

############################################################
# Plot training
############################################################

model_num = 2

start_idx = Int64(ceil(υ*train_step))
linewidth= 1.2

#fig_type = :err
fig_type = :var
#fig_type = :grad
yscale=:log10
#yscale=:identity
new_m_list = [20]

ℓm_list = [(ℓ, m) for ℓ in ℓ_list for m in new_m_list]
fig_grad  = Array{Any}(undef, length(ℓm_list))

for i = 1:length(ℓm_list) 
    ℓ, m = ℓm_list[i]
    ylabel = L"{\bf \ell = %$(ℓ), m = %$(m)}"
    if i == 1
        title = "gradient ansatz"
    else
        title = ""
    end
    fig_grad[i] = plot_error(m, ℓ, model_num, seed_list, exact_mean, folder, 
            fig_type=fig_type, start_idx = start_idx, ylabel=ylabel, title=title, 
            yscale=yscale, linewidth=linewidth)
end

xtic = [150,175,200]
fig_simu_grad = plot(fig_grad..., 
    size=(250,150), layout=(@layout grid(2,1)), 
    xticks=(xtic,xtic), left_margin=20px, right_margin=20px)
savefig(fig_simu_grad, "assets/test3_simu_grad_ansatz_"*string(fig_type)*".pdf")


#############################################################
# Flow lines
#############################################################

xmin = -10
xmax = 10
ymin = -10
ymax = 10

m = 20
ℓ = 2
model_num = 2
seed = 1

xc = range(xmin, stop=xmax, length=10^2)
yc = range(ymin, stop=ymax, length=10^2)
#xb = range(xmin, stop=xmax, length=20)
#yb = range(ymin, stop=ymax, length=20)
#xxb = [x for x in xb for y in yb]
#yyb = [y for x in xb for y in yb]
casename =  folder*@sprintf("case_%d_model_%d_%d_%d_%d_%d_%d_%d_training_data.jld2", 
        testcasenum, model_num, n, N, numsample_max, m, ℓ, seed)

flow = load(casename,  "flow")

T = 1.0
t_vec = Array(range(0,stop=1.0,length=Int64(ceil(T*50))+1))
unit_step = 5
num_particle = 100
traj_gpts = [U₀.sampler(i) for i in 1:num_particle]

#df = (x,y)->flow.f(vcat([x,y], zeros(n-2)), flow.para_list...)[1:2]
#scale = 0.8/maximum(map(norm, map(df, xxb, yyb)))
fig_v = contour(xc, yc, (x,y)-> U₁.U(vcat([x,y], zeros(n-2))), fill=true, 
                color=:tofino, size=figsize, xlims=(xmin,xmax), ylims=(ymin,ymax),
                left_margin=20px, right_margin=20px)
plot_traj(num_particle, traj_gpts, flow, t_vec, unit_step)
#quiver!(xxb, yyb, quiver=(x,y)->scale*df(x,y), 
#    xlims=(xmin,xmax), ylims=(ymin, ymax), color=:orangered3)
savefig(fig_v, @sprintf("assets/test3_%d_%d_%d_%d.pdf",model_num,ℓ,m,seed))


###########################################################
# Comparison
###########################################################

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

Ngd = 5*N # time-discretization; for gradient-flow only
GDT = 5.0
Ωq = domain_ball(30);
flow_gd = generate_gradient_flow(U₁, GDT, Ωq);
fun_gd = numsample-> get_data_err_var(U₀, U₁, flow_gd, Ngd, numsample)[2]
@time data_gd_rep = repeat_experiment(fun_gd, numsample, numrepeat, gpts, gpts_sampler);

fun_nn = numsample -> get_data_err_var(U₀, U₁, flow, N, numsample)[2]
@time data_nn_rep = repeat_experiment(fun_nn, numsample, numrepeat, gpts, gpts_sampler);

df = DataFrame(ais10 = data_ais_10_rep/exact_mean, 
    ais100 = data_ais_100_rep/exact_mean, 
    gd = data_gd_rep/exact_mean, 
    nn = data_nn_rep/exact_mean)
fig = @df df boxplot([:ais10, :ais100, :gd, :nn], 
    xticks = (1:1:5, ["AIS-10", "AIS-100", "Direct GD", "NN"]),
    fillalpha = 0.75, linewidth=2, label="", size=(600,250), outliers=false)
plot!(0:1:5, ones(6), linewidth=2, label="", color=:red, linestyle=:dash)
plot!(xlim=(0.2, 4.8))
savefig(fig, "assets/test3_compare.pdf")

println(df);
