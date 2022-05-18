to_train_ln = false
push!(LOAD_PATH,"../src")
push!(LOAD_PATH,"../../src")
include("../example/test4_funnel/funnel_train.jl")
folder = "../example/test4_funnel/";

using ModDyn
using Printf
using ModUtil: repeat_experiment

include("plot_training.jl");

using Plots
using LaTeXStrings
using Plots.PlotMeasures

gr()

default(titlefont = (12), legendfontsize=8, 
    guidefont = (11), 
    fg_legend = :transparent)

figsize=(350,250)

#####################################
# Load the model
#####################################

tmname = "sym"
if tmname == "sym"
    tm = -1/2
else
    tm = -0.0
end

model_num = 5
casename_ln = @sprintf("case_%d_model_%d_%d_%d_%d_%d_%d_%s_%s",
             testcasenum, model_num, σ₀^2,
             n, N, numsample_max, seed, train_method, tmname)
est_fst_m, est_sec_m, est_grad_norm, est_para, flow_ln, flow_ln_bef = load(folder*casename_ln*"_ln_data.jld2", 
    "est_fst_m", "est_sec_m", "est_grad_norm", "est_para", "flow_ln", "flow_ln_bef");


#####################################
# Plot training result
#####################################

fig1 = Plots.plot(0:1:train_step, abs.(est_fst_m/exact_mean.-1),
    label="", title="error")
fig2 = Plots.plot(0:1:train_step, (est_sec_m .- est_fst_m.^2)/exact_mean^2,
    label="", yscale=:log10, title="var")
fig3 = Plots.plot(0:1:train_step, [item[1][1] for item in est_para],
    label="β", title="parameters")
Plots.plot!(0:1:train_step, [item[1][2] for item in est_para],
    label="α")
fig = Plots.plot(fig1, fig2, fig3, size=(800,200), layout =@layout Plots.grid(1,3))
Plots.savefig(fig, "assets/test4_train_"*tmname*".pdf")


#####################################
# Comparison
#####################################

numrepeat = 10
numsample = 10^3

Random.seed!(seed)

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

flow_gd = generate_gradient_flow(UU₁, 2.0, Ωq);
fun_gd = numsample-> ModOptInt.get_data_err_var(n, U₀, U₁, flow_gd, N, tm, numsample, solver=solver)[2]
@time data_gd_rep = repeat_experiment(fun_gd, numsample, numrepeat, gpts, gpts_sampler);

fun_ln_bef = numsample->ModOptInt.get_data_err_var(n, U₀, U₁, flow_ln_bef, N, tm, numsample, solver=solver)[2]
@time data_ln_bef_rep = repeat_experiment(fun_ln_bef, numsample, numrepeat, gpts, gpts_sampler);

fun_ln = numsample->ModOptInt.get_data_err_var(n, U₀, U₁, flow_ln, N, tm, numsample, solver=solver)[2]
@time data_ln_rep = repeat_experiment(fun_ln, numsample, numrepeat, gpts, gpts_sampler);


using StatsPlots, DataFrames

df = DataFrame(ais10=data_ais_10_rep/exact_mean, 
    ais100=data_ais_100_rep/exact_mean, 
    gd=data_gd_rep/exact_mean,
    lnbef=data_ln_bef_rep/exact_mean,
    ln=data_ln_rep/exact_mean)
fig = @df df StatsPlots.boxplot([:ais10, :ais100, :gd, :lnbef, :ln], 
    xticks = (1:1:5, ["AIS-10", "AIS-100", "Direct GD", "Linear (before)", "Linear (after)"]),
    fillalpha = 0.75, linewidth=2, label="", size=(700, 250), outliers=false)
plot!(0:1:6, ones(7), linewidth=2, label="", color=:red, linestyle=:dash)
plot!(xlim=(0.2, 5.8))
Plots.savefig(fig, "assets/test4_compare"*tmname*".pdf")


########################################
# Flows
########################################

using ModTraj
using BasicInterpolators

T = 1.0
t_vec = Array(range(0,stop=1.0,length=Int64(ceil(T*50))+1))
unit_step = 5
num_particle = 100
traj_gpts = [σ₀*randn(n) for i = 1:num_particle]

flow = flow_ln
#flow_bef = flow_ln_bef

xmin = -10
xmax = 5
ymin = -3
ymax = 3

xc = range(xmin, stop=xmax, length=10^2)
yc = range(ymin, stop=ymax, length=10^2)

φ = x -> log10.(x)
Z₁  = sqrt(2*π)^n*σf*reduce_percent
ρ₁(x,y) = exp(-U₁.U(vcat([x,y],zeros(n-2))))/Z₁

# f1 = Plots.contour(xc, yc, (x,y)-> φ(ρ₁(x,y)), fill=true, 
#     color=:tofino, 
#     size=(500,400), xlim=(xmin,xmax),ylim=(ymin,ymax), title="before", left_margin=20px, right_margin=20px)
# plot_traj(num_particle, traj_gpts, flow_bef, t_vec, unit_step)

# f2 = Plots.contour(xc, yc, (x,y)-> φ(ρ₁(x,y)), fill=true, 
#     color=:tofino, 
#     size=(500,400), xlim=(xmin,xmax),ylim=(ymin,ymax), title="after", left_margin=20px, right_margin=20px)
# plot_traj(num_particle, traj_gpts, flow, t_vec, unit_step)
# Plots.plot(f1, f2,size=(900,350), left_margin=20px, right_margin=20px)

fig_ln_aft = Plots.contour(xc, yc, (x,y)-> φ(ρ₁(x,y)), fill=true, 
    color=:tofino, 
    size=figsize, xlim=(xmin,xmax),ylim=(ymin,ymax), left_margin=20px, right_margin=20px)
plot_traj(num_particle, traj_gpts, flow, t_vec, unit_step)
Plots.savefig(fig_ln_aft, @sprintf("assets/test4_ln_flow_%s.pdf", tmname))



########################################
# Extend time interval
########################################

Tscale = 2.0;
flow_T2 = deepcopy(flow_ln);
flow_T2.para_list[1] .*= Tscale;
fun_ln_T2 = numsample->ModOptInt.get_data_err_var(n, U₀, U₁, flow_T2, Int64(ceil(Tscale*N)), tm, numsample)[2]
@time data_ln_T2 = repeat_experiment(fun_ln_T2, numsample, numrepeat, gpts, gpts_sampler);
dflongT = DataFrame(ln=data_ln_rep/exact_mean, ln2=data_ln_T2/exact_mean);

Tscale = 4.0;
flow_T4 = deepcopy(flow_ln);
flow_T4.para_list[1] .*= Tscale;
fun_ln_T4 = numsample->ModOptInt.get_data_err_var(n, U₀, U₁, flow_T4, Int64(ceil(Tscale*N)), tm, numsample)[2]
@time data_ln_T4 = repeat_experiment(fun_ln_T4, numsample, numrepeat, gpts, gpts_sampler);
dflongT[!,:ln4] = data_ln_T4/exact_mean;

fig_longerT = @df dflongT StatsPlots.boxplot([:ln, :ln2, :ln4], 
    xticks = (1:3, ["Linear", "Linear (2x longer time)", "Linear (4x longer time)"]),
    fillalpha = 0.75, linewidth=2, label="", size=(500,250), outliers=false)
plot!(0:1:4, ones(5), linewidth=2, label="", color=:red, linestyle=:dash)
plot!(xlim=(0.2, 3.8))
Plots.savefig(fig_longerT, "assets/test4_longerT.pdf")

flow_T8 = deepcopy(flow_ln);
flow_T8.para_list[1] .*= 8.0;

@time flow_T4_sample = ModOptInt.get_data_err_var(n, U₀, U₁, flow_T4, Int64(ceil(4*N)), tm, numsample)[1];
@time flow_T8_sample = ModOptInt.get_data_err_var(n, U₀, U₁, flow_T8, Int64(ceil(8*N)), tm, numsample)[1];

df_sample = DataFrame(ln4 = flow_T4_sample/exact_mean, ln8 = flow_T8_sample/exact_mean)
fig_sample = @df df_sample StatsPlots.boxplot([:ln4, :ln8], xlims=(0, 3))
