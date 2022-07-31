using Distributed

if !(@isdefined to_train_ln)
    to_train_ln = true
end

push!(LOAD_PATH, "../../src")
using ModUtil: add_procs
if to_train_ln
    add_procs()
end

@everywhere push!(LOAD_PATH, "../../src")

using JLD2, FileIO
using LinearAlgebra
using ModDyn
using ModTestCase
using ModPotential
using ModOpt
using ModOptInt
using ModOptODE
using ModODEIntegrate
using SharedArrays
@everywhere using ModUtil: domain_ball, prod_domain
using ModUtil: print_stat_name
using Plots
using PyPlot
using ModDyn
using Random
using Statistics
using ModVanilla
using ModSMC
using Printf

###################################################
# Load the model
###################################################

testcasenum = 4
n = 10
σf = 3.0
σ₀ = 1.0 

@everywhere Ωq = domain_ball(25)
U₀, Uext₀, UU₁, UUext₁, exact_mean = load_funnel(n, σf, σ₀ = σ₀, reducedsystemonly=false)
U₁ = convert_to_bounded_potential(UU₁, Ωq)
@everywhere Ωp = domain_ball(20)
@everywhere Ω = prod_domain(Ωq, Ωp, $n, $n) 
Uext₁ = convert_to_bounded_potential(UUext₁, Ω)

##################################################
# Correct the exact value
# Count the effect of domain into consideration
##################################################

function partition_reduction_percent(Ωq, numsample_domain, n, σf)
    v = zeros(numsample_domain)
    for i = 1:numsample_domain
        x0 = randn()*σf
        x1 = randn(n-1)*exp(x0/2)
        v[i] = Ωq(vcat(x0, x1))
    end
    return sum(v)/numsample_domain
end

percent_vec = map((j)->partition_reduction_percent(Ωq, 10^6, n, σf), 1:10)
if std(percent_vec)/mean(percent_vec) < 1.0e-3
    reduce_percent = mean(percent_vec)
else
    error("Not accurate enough estimates of the exact value.")
end

exact_mean *= reduce_percent

###################################################
# training parameters
###################################################

N = 200
numsample = 3*10^3
numsample_max = numsample
numsample_min = numsample

#train_step = 2
train_step = 100
max_search_iter = 15
ϵ = 0.5
decay = (1/ϵ - 1)/train_step

gpts = σ₀*SharedArray(randn(n, numsample_max))
gpts_sampler = ()->σ₀*randn(n)
U₀.sampler = (j)->gpts[:,j]

seed = 1

#train_method = "ode"
train_method = "int"
tmname = "sym"
#tmname = "asym"
if tmname == "sym"
	tm = -1/2
else
	tm = -0.0
end
offset = Int64(ceil(abs(tm*N)))
solver = RK4
train_paras = Dict(
            :ref_value => exact_mean,
            :verbose => true,
            :max_search_iter => max_search_iter,
            :numsample_min => numsample_min,
            :printpara => true, 
            :savepara => true)

###################################################
# Train linear Dynamics
###################################################

if to_train_ln
    model_num = 5
    Random.seed!(seed)
    h = 0.2

    T₀ = 2.0
    α₀ = 2.0
    flow_ln = init_funnelexpansatz(n, T₀, α₀, Ωq)
    flow_ln_bef = deepcopy(flow_ln);
    casename = @sprintf("case_%d_model_%d_%d_%d_%d_%d_%d_%s_%s",
                 testcasenum, model_num, σ₀^2, 
                 n, N, numsample_max, seed, train_method, tmname)

    if train_method == "ode"
        est_fst_m, est_sec_m, est_grad_norm, est_para =
            ModOptODE.train_NN(U₀, U₁, flow_ln, N, numsample_max,
                train_step, h, decay, gpts, gpts_sampler; train_paras...)
    else
        est_fst_m, est_sec_m, est_grad_norm, est_para =
            ModOptInt.train_NN(U₀, U₁, flow_ln, N, numsample_max,
                train_step, h, decay, gpts, gpts_sampler;
                offset = offset, train_paras...)
    end

    fig1 = Plots.plot(0:1:train_step, abs.(est_fst_m/exact_mean.-1),
        label="", title="error")
    fig2 = Plots.plot(0:1:train_step, (est_sec_m .- est_fst_m.^2)/exact_mean^2,
        label="", yscale=:log10, title="var")
    fig3 = Plots.plot(0:1:train_step, est_grad_norm,
        label="", yscale=:log10, title="grad norm")
    fig4 = Plots.plot(0:1:train_step, [item[1][1] for item in est_para],
        label="T", title="parameters")
    Plots.plot!(0:1:train_step, [item[1][2] for item in est_para],
        label="a")
    fig = Plots.plot(fig1, fig2, fig3, fig4,
        size=(500,300))
    Plots.savefig(fig, casename*"_ln_train.pdf")

    @save casename*"_ln_data.jld2" est_fst_m est_sec_m est_grad_norm est_para flow_ln flow_ln_bef exact_mean train_step
end

