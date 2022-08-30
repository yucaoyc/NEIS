using NEIS
using LinearAlgebra
using SharedArrays
using FileIO, JLD2
using Printf
using Random

biased = true
if !(@isdefined to_train) to_train = true end

# model
testcasenum = 2
#m_list = [10, 20]
#seed_list = [1, 2]
#model_list = [1, 2]
m_list = [20]
seed_list = [2]
model_list = [1]
ℓ = 2

T = Float64

n = 10
λ = T(5.0)
U₀, U₁, exact_mean = load_eg2(n, λ) # see potential/testcase.jl

# model for dynamics
N = 100
scale = T(5.0)

numsample_max = 500
numsample_min = 500
train_step = 50

η = T(0.5)
ϵ = T(0.2)
decay = (1/ϵ - 1)/train_step

gpts = SharedArray(randn(T, n, numsample_max))
gpts_sampler = ()->randn(T, n)

# add bias into the training.
if biased
    biased_percent = T(0.1)
    biased_scale = T(1.0)
    υ = T(0.5)
    biased_func = (x,i)->biased_GD_func(x, i,
                train_step, biased_percent, U₁, N, s = biased_scale, υ=υ)
else
    biased_func = nothing
end

convert = x->T.(x)

train_paras = Dict(:biased_func => biased_func, :ref_value => exact_mean,
                   :verbose => true, :max_search_iter => 0,
                   :numsample_min => numsample_min)

if to_train
    for (m, model_num, seed) in Iterators.product(m_list, model_list, seed_list)
        Random.seed!(seed)
        # file names
        casename =  @sprintf("case_%d_model_%d_%d_%d_%d_%d_%d_%d",
                     testcasenum, model_num, n, N, numsample_max, m, ℓ, seed)
        # use a optimized implementation for better speed
        if model_num == 1 && ℓ == 2
            flow = init_random_DynNNGenericTwo(n, m, convert=convert, scale=scale)
        elseif model_num == 2 && ℓ == 2
            flow = init_random_DynNNGradTwo(n, m, convert=convert, scale=scale)
        else
            @error("Not implemented yet")
        end
        flow_bef = deepcopy(flow)

        # training part
        train_time =  @elapsed est_fst_m, est_sec_m, est_grad_norm, est_para =
                train_NN_ode(U₀, U₁, flow, N, numsample_max,
                train_step, η, decay, gpts, gpts_sampler;
                train_paras...)
        train_stat = get_train_stat(train_time, U₁)
        print_train_stat(train_stat)

        fn = casename*"_training_data.jld2"
        @save(fn, est_fst_m, est_sec_m, est_grad_norm, flow, flow_bef,
              exact_mean, train_step, est_para, train_stat)
    end
end
