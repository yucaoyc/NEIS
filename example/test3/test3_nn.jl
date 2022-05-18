include("../../src/Preamble.jl")
using ModOptODE

biased = true

if !(@isdefined to_train)
    to_train = true
end

# model
testcasenum = 3
activation = softplus
to_train_bL = true

m_list = [10, 20]
#m_list = [40]
ℓ_list = [2]
model_list = [2]
#model_list = [1,2,3]
seed_list = [1, 2]

n = 10
λ = 5.0
σsq₁ = 0.1
σsq₂ = 0.5
center_pts = []

num_pts = 4
for i = 1:num_pts
    θ = i*(2*pi/num_pts)
    push!(center_pts, λ*[cos(θ), sin(θ)])
end
U₀, U₁, exact_mean = load_mixGaussian(n, σsq₁, σsq₂, center_pts);

# model for dynamics
N = 100
scale = 5.0

numsample_max = 2*10^3
numsample_min = 10^3
train_step = 200
max_search_iter = 15

η = 0.5
ϵ = 0.2
decay = (1/ϵ - 1)/train_step

gpts = SharedArray(randn(n, numsample_max))
gpts_sampler = ()->randn(n)
U₀.sampler = (j)->gpts[:,j]

# add bias into the training.
if biased
    biased_percent = 0.1
    biased_scale = 1.0
    υ = 0.75
    biased_func = (x,i) -> biased_GD_func(x, i, train_step, biased_percent, U₁, N, 
                                          s = biased_scale, υ=υ)
else
    biased_func = nothing
end

if to_train
for m = m_list for ℓ = ℓ_list for model_num = model_list for seed = seed_list

    # file names
    casename =  @sprintf("case_%d_model_%d_%d_%d_%d_%d_%d_%d", 
                 testcasenum, model_num,
                 n, N, numsample_max, m, ℓ, seed)
    # initialize
    Random.seed!(seed)
    ndims = get_ndims(n,m,ℓ)
    flow, V = init_dyn_nn(ndims, model_num; 
                   scale=scale, activation=activation, to_train_bL=to_train_bL)
    flow_bef = deepcopy(flow)
    V_bef = deepcopy(V)
    # training part
    est_fst_m, est_sec_m, est_grad_norm, est_para = train_NN(U₀, U₁, flow, N, numsample_max, 
                train_step, η, decay, gpts, gpts_sampler; 
                biased_func = biased_func,
                ref_value = exact_mean, 
                verbose=true, printpara=false, max_search_iter=max_search_iter,
                numsample_min = numsample_min)
    fn = casename*"_training_data.jld2" 
    @save fn est_fst_m est_sec_m est_grad_norm flow flow_bef exact_mean train_step V V_bef est_para

end end end end
end
