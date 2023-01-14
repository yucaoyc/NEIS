using NEIS
using LinearAlgebra, Printf, Random
using Statistics
plot_setup()
push!(LOAD_PATH,"./")
using VisualizeComparison

to_train = true
to_plot = true
to_compare = true
to_compute_data = true

########################################
# model
testcasenum = 2; n = 10; λ = 5.0;
σsq₁ = 0.1; σsq₂ = 0.5
U₀, U₁, exact_mean, vanilla_var = load_eg2(n, λ, σsq₁=σsq₁, σsq₂=σsq₂)
printstyled(@sprintf("Vanilla Importance Sampling Variance = %.2E\n", vanilla_var),
            color=:yellow, bold=true)
biased = true
T = Float64

########################################
# training parameters
numsample_max = 800
train_step = 60
lr = 0.5
ϵ = 0.2
decay = (1/ϵ - 1)/train_step
discretize_method = "ode"; offset = 0;

########################################
# ansatz
#N = 100; scale = 5.0; ℓ = 2
N = 60; scale = 5.0; ℓ = 2
m_list = [30, 40]; seed_list = [1, 2];
model_list = [2]

########################################
# use assisted training
# by introducing bias in the initial stage
if biased
    biased_percent = 0.3; biased_scale = 1.0; υ = 0.75;
    biased_func = (x,i)->biased_GD_func(x, i,
                train_step, biased_percent, U₁, N,
                s = biased_scale, υ = υ)
else
    biased_func = nothing
end

train_paras = Dict{Symbol,Any}(:ref_value => exact_mean,
                   :verbose => false,
                   :max_search_iter => 0, # no line search
                   :numsample_min => numsample_max,
                   :biased_func => biased_func,
                   :quiet => true,
                   :to_normalize => true)

########################################
# loss function energy landscape
train_paras[:ϕname] = "msq"; train_paras[:ϕϵ] = 1.0e-3
ϕ = get_Φ(train_paras[:ϕname], train_paras[:ϕϵ])

ModelParas = Dict("testcasenum"=>testcasenum, "n"=>n, "N"=>N, "numsample_max"=>numsample_max,
                  "ℓ"=>ℓ, "model_num"=>0, "m"=>0, "ϕ"=>ϕ)
data_folder = "./data/";

########################################
# start training
if to_train
    for (seed, m, model_num) in Iterators.product(seed_list, m_list, model_list)
        separate_line_small()
        @printf("model number = %1d, ϕ = %s, m = %3d, seed = %2d\n", model_num, ϕ.name, m, seed)

        Random.seed!(seed)
        ModelParas["m"] = m; ModelParas["model_num"] = model_num
        filename = get_experiment_name(ModelParas, seed, data_folder)

        # use a optimized implementation for better speed
        flow = init_rand_dyn(model_num, n, m, scale, U₁)

        # do the training (statistics will be reset), data will be saved automatically.
        train_flow(U₀, U₁, flow, N, numsample_max, train_step, lr, decay,
                   discretize_method, offset, train_paras, filename=filename, save_data=true)
    end
end


########################################
# show results
if to_plot
    show_training_result_various_ansatz_and_m(model_list, m_list, seed_list,
                    data_folder, ModelParas, exact_mean, υ, train_step, "eg"*string(testcasenum),
                    vanilla_var = vanilla_var)
end

########################################
# Compare with AIS
if to_compare
    print_current_resource()
    println("="^40)
    printstyled("Start comparison\n", bold=true, color=:blue)
    println("="^40)

    xmin = -10; xmax = 10; ymin = -10; ymax = 10
    percent_train = 1/4;
    numrepeat = 10;
    solver = RK4
    loc_y = 1.35; gap = 0.1; ylim = (0.75, 1.35); outliers = false

    for (m, model_num) in Iterators.product(m_list, model_list)
        printstyled(@sprintf("Comparison for model_num=%d, m=%d\n", model_num, m),
                    bold=true, color=:blue)
        ModelParas["m"] = m; ModelParas["model_num"] = model_num
        show_comparison_for_all_random_initializations(U₀, U₁, ModelParas,
                seed_list, data_folder, xmin, xmax, ymin, ymax,
                solver, discretize_method, offset, exact_mean,
                numrepeat, percent_train, to_compute_data,
                loc_y=loc_y, gap=gap, ylim=ylim, outliers=outliers)
    end
    # end of comparison
end

print_data(m_list, model_list, seed_list, ModelParas, data_folder)
