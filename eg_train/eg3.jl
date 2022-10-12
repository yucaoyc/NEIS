using NEIS
using LinearAlgebra, Printf, Random
using Statistics
using JLD2, FileIO
using Plots
using Plots.PlotMeasures
using LaTeXStrings
plot_setup()
push!(LOAD_PATH,"./")
using VisualizeComparison
setup_folder()

to_train = true
to_plot = true
to_compare = true
to_compute_data = true
xmin = -10; xmax = 4; ymin = -7; ymax = 7;

########################################
# Load the model
testcasenum = 3
n = 10
# n = 2
σf = 3.0
U₀, U₁, exact_mean, reduce_percent = load_eg3(n, σf)
T = Float64

########################################
# training parameters
N = 100
discretize_method = "int"
tmname = "sym"; tm = -1/2;
offset = estimate_offset(tm, N)

########################################
# ansatz
scale = 3.0

for model_num in [0, 4]
    println("&"^40)

    if model_num == 0
        ℓ = 1; m_list = [n]; # linear 𝐛(x) = A x + b
        seed_list = [2, 3, 4]
        numsample_max = 10^3
        biased = true
        train_step = 200
        lr = 0.3
        ϵ = 0.2
        decay = (1/ϵ - 1)/train_step
    elseif model_num == 4
        ℓ = 1; m_list = [0] # two-parameter form
        seed_list = [1]
        numsample_max = 10^3
        biased = false
        train_step = 100
        lr = 0.3
        ϵ = 0.5
        decay = (1/ϵ - 1)/train_step
    end
    model_list = [model_num]

    ########################################
    # whether to use assisted training
    if biased
        biased_percent = 0.3; biased_scale = 1.0; υ = 0.7;
        biased_func = (x,i) -> biased_GD_func(x, i,
                    train_step, biased_percent, U₁, N,
                    s = biased_scale, υ = υ)
    else
        biased_percent = 0.0; biased_scale = 1.0; υ = 0.0
        biased_func = nothing
    end

    ########################################
    # more training parameters
    train_paras = Dict{Symbol, Any}(:ref_value => exact_mean,
                :verbose => false,
                :max_search_iter => 0,
                :numsample_min => numsample_max,
                :biased_func => biased_func,
                :printpara => false,
                :quiet => false)
    if model_num == 0
        train_paras[:savepara] = false
    elseif model_num == 4
        train_paras[:savepara] = true
    end

    ########################################
    # loss function energy landscape
    train_paras[:ϕname] = "msq"; train_paras[:ϕϵ] = 1.0e-4;
    ϕ = get_Φ(train_paras[:ϕname], train_paras[:ϕϵ])

    ModelParas = Dict("testcasenum"=>testcasenum, "n"=>n, "N"=>N, "numsample_max"=>numsample_max,
                      "ℓ"=>ℓ, "model_num"=>0, "m"=>0, "ϕ"=>ϕ)
    data_folder = "./data/"

    φ = x -> log10.(x)
    #φ = x -> x
    Z₁  = sqrt(2*π)^n*σf*reduce_percent
    ρ₁(x,y) = φ(exp(-U₁(vcat([x,y],zeros(n-2))))/Z₁)

    ########################################
    # start training
    if to_train
        for (seed, m) in Iterators.product(seed_list, m_list)
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
              data_folder, ModelParas, exact_mean, υ, train_step, "eg"*string(testcasenum))

        if model_num == 4
            ModelParas["model_num"] = model_num
            ModelParas["m"] = m_list[1]
            seed = seed_list[1]
            filename = get_experiment_name(ModelParas, seed, data_folder)
            est_para = load(filename, "est_para")
            f1, max_v, min_v = plot_error_or_divg(seed_list, data_folder, ModelParas, exact_mean,
                                                  fig_type=:err, title="error", nolabel=true, margin=20px)
            f2, max_v, min_v = plot_error_or_divg(seed_list, data_folder, ModelParas, exact_mean,
                                                  fig_type=:divg, title="var", nolabel=true, margin=20px)
            βlist = [item[1][1] for item in est_para]
            αlist = [item[1][2] for item in est_para]
            f3 = plot(1:train_step, βlist, label=L"\beta", title="parameters")
            plot!(1:train_step, αlist, label=L"\alpha", margin=20px)
            fig_train = plot(f1, f2, f3, layout=(@layout grid(1,3)), size=(800,250))
            fn = @sprintf("assets/train_case_%d_model_%d_phi_%s_with_para.pdf", ModelParas["testcasenum"],
                                                                             ModelParas["model_num"],
                                                                             ModelParas["ϕ"].name)
            savefig(fig_train, fn)
        end
    end

    ########################################
    # Compare with AIS
    if to_compare
        print_current_resource()
        println("="^40)
        printstyled("Start comparison\n", bold=true, color=:blue)
        println("="^40)

        percent_train = 1/7; numrepeat = 10;
        solver = RK4
        loc_y = 1.35; gap = 0.1; ylim = (0.5, 1.35); outliers = false

        for (m, model_num) in Iterators.product(m_list, model_list)
            printstyled(@sprintf("Comparison for model_num=%d, m=%d\n", model_num, m),
                    bold=true, color=:blue)
            ModelParas["m"] = m; ModelParas["model_num"] = model_num
            show_comparison_for_all_random_initializations(U₀, U₁, ModelParas,
                                        seed_list, data_folder,
                                        xmin, xmax, ymin, ymax,
                                        solver, discretize_method, offset, exact_mean,
                                        numrepeat, percent_train, to_compute_data,
                                        loc_y=loc_y, gap=gap, ylim=ylim, outliers=outliers, rho=ρ₁)
        end
    end # end of comparison

    print_data(m_list, model_list, seed_list, ModelParas, data_folder)

    println("&"^40)
end # end of model number
