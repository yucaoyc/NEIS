using NEIS
using LinearAlgebra
using SharedArrays
using JLD2, FileIO
using Printf
using Random
using Plots

if !(@isdefined to_train) to_train = true end

# Load the model
testcasenum = 3
n = 10
σf = 3.0
U₀, U₁, exact_mean = load_eg3(n, σf)

# training parameters
N = 200
numsample = 500
numsample_max = numsample
numsample_min = numsample

train_step = 100
max_search_iter = 0

h = 0.2
ϵ = 0.5
decay = (1/ϵ - 1)/train_step

gpts = SharedArray(randn(n, numsample_max))
gpts_sampler = ()->randn(n)

#train_method = "ode"
train_method = "int"
tmname = "sym"
#tmname = "asym"
if tmname == "sym"
    tm = -1/2
else
    tm = -0.0
end
offset = estimate_offset(tm, N)
train_paras = Dict(
            :ref_value => exact_mean,
            :verbose => true,
            :max_search_iter => max_search_iter,
            :numsample_min => numsample_min,
            :printpara => true,
            :savepara => true)

if to_train
    seed = 1
    model_num = 0 # linear
    Random.seed!(seed)

    β = 2.0
    α = 2.0
    flow = init_funnelexpansatz(n, β, α, U₁.Ω)
    flow_bef = deepcopy(flow);
    casename = @sprintf("case_%d_model_%d_%d_%d_%d_%d_%s_%s",
                 testcasenum, model_num,
                 n, N, numsample_max, seed, train_method, tmname)
    #if train_method == "ode"
    #    train_time = @elapsed est_fst_m, est_sec_m, est_grad_norm, est_para =
    #        train_NN_ode(U₀, U₁, flow, N, numsample_max,
    #            train_step, h, decay, gpts, gpts_sampler; train_paras...)
    #else
        train_time = @elapsed est_fst_m, est_sec_m, est_grad_norm, est_para =
            train_NN_int(U₀, U₁, flow, N, numsample_max,
                train_step, h, decay, gpts, gpts_sampler;
                offset = offset, train_paras...)
    #end
#    train_stat = get_train_stat(train_time, U₁)
#    print_train_stat(train_stat)

    fig1 = Plots.plot(0:1:train_step, abs.(est_fst_m/exact_mean.-1),
        label="", title="error")
    fig2 = Plots.plot(0:1:train_step, (est_sec_m .- est_fst_m.^2)/exact_mean^2,
        label="", yscale=:log10, title="var")
    fig3 = Plots.plot(0:1:train_step, est_grad_norm,
        label="", yscale=:log10, title="grad norm")
    fig4 = Plots.plot(0:1:train_step, [item[1][1] for item in est_para],
        label="β", title="parameters")
    Plots.plot!(0:1:train_step, [item[1][2] for item in est_para],
        label="α")
    fig = Plots.plot(fig1, fig2, fig3, fig4,
        size=(500,300))
    Plots.savefig(fig, casename*"_ln_train.pdf")

    @save(casename*"_ln_data.jld2", est_fst_m, est_sec_m, est_grad_norm, est_para,
          flow, flow_bef, exact_mean, train_step, train_stat)

end
