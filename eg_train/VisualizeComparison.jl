module VisualizeComparison
"""
This module contains functions used for visualizing training and comparison with AIS.
"""

using NEIS
using Plots
using Plots.PlotMeasures
using LaTeXStrings
using FileIO, JLD2
using Statistics
using LinearAlgebra
using DataFrames
using StatsPlots
using Printf
using Random
using SharedArrays
import Humanize: datasize

export get_experiment_name, get_cmp_data_name, get_short_name, sprint_experiment_paras # get name
export plot_error_or_divg # plot training result for various initializations
export print_divg_at_end # print divergence at the end of training
export add_annotate, plot_assist_window # small plotting functions
export show_training_result_various_ansatz_and_m
export plot_flow
export generate_comparison_data
export plot_comparison
export show_comparison_for_all_random_initializations
export setup_folder
export print_data

function setup_folder()
    if ~isdir("./data")
        mkdir("./data")
    end
    if ~isdir("./assets")
        mkdir("./assets")
    end
    #if ~isdir("./assets/flows")
    #    mkdir("./assets/flows")
    #end
end

function _savefig(fig, filename)
    Plots.savefig(fig, filename)
end

"""
Get filename for training cases.
"""
function get_experiment_name(ModelParas::Dict{}, seed::Int, folder::String) where T<:AbstractFloat
    return folder*@sprintf("case_%d_model_%d_%d_%d_%d_%d_%d_%d_%s_training_data.jld2",
                ModelParas["testcasenum"],
                ModelParas["model_num"],
                ModelParas["n"],
                ModelParas["N"],
                ModelParas["numsample_max"],
                ModelParas["m"],
                ModelParas["ℓ"],
                seed,
                ModelParas["ϕ"].name)
end

"""
Get filename that stores comparison data with AIS.
"""
function get_cmp_data_name(ModelParas::Dict{}, seed::Int, folder::String) where T<:AbstractFloat
    return folder*@sprintf("case_%d_model_%d_%d_%d_%d_%d_%d_%d_%s_cmp_data.jld2",
                ModelParas["testcasenum"],
                ModelParas["model_num"],
                ModelParas["n"],
                ModelParas["N"],
                ModelParas["numsample_max"],
                ModelParas["m"],
                ModelParas["ℓ"],
                seed,
                ModelParas["ϕ"].name)
end

"""
Get a simplified name.
"""
function get_short_name(ModelParas::Dict{}, folder::String) where T<:AbstractFloat
    return folder*@sprintf("case_%d_model_%d_m_%d_phi_%s",
                                ModelParas["testcasenum"],
                                ModelParas["model_num"],
                                ModelParas["m"],
                                ModelParas["ϕ"].name)
end

function sprint_experiment_paras(ModelParas::Dict)
    text = @sprintf("Example # = %d, ansatz num = %d, ℓ = %d, m = %d\n",
            ModelParas["testcasenum"], ModelParas["model_num"], ModelParas["ℓ"], ModelParas["m"])
end

"""
Plot training result for a particular architecture with different initialization (specified by seed).
fig_type = :err (plot the error), :divg (plot the divergence and in particular the variance)
            or :grad (plot the norm of ∇L where L is the loss)
"""
function plot_error_or_divg(seed_list::Vector, folder::String,
        M::Dict{}, exact_mean::T;
        start_idx = 1, fig_type=:divg,
        yscale=:log10, xlabel="", ylabel="", title="",
        linewidth=1.2, nolabel=false,
        rescale_var=true, margin=10px) where T<:AbstractFloat

    fig = plot(yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title)

    ϕ = M["ϕ"]
    max_value = -Inf
    min_value = Inf
    for i in 1:length(seed_list)
        seed = seed_list[i]
        casename = get_experiment_name(M, seed, folder)
        est_fst_m, est_sec_m, est_grad_norm, train_step = load(casename,
                                    "est_fst_m", "est_sec_m", "est_grad_norm", "train_step")
        time = 1:train_step
        if fig_type == :err
            data = abs.(est_fst_m .- exact_mean)./exact_mean
        elseif fig_type == :divg
            data = est_sec_m .- ϕ.f(est_fst_m)
            if rescale_var
                data = data ./ ϕ.f(exact_mean)
            end
        elseif fig_type == :grad
            data = est_grad_norm
        else
            @error("Please input the correct type: :err, :divg, or :grad .")
        end

       plot!(fig, time[start_idx:end], data[start_idx:end], linewidth=linewidth, margin=margin,
            label= (nolabel ? "" : "trial $(i)"))

        max_value = max(max_value, maximum(data[start_idx:end]))
        min_value = min(min_value, minimum(data[start_idx:end]))
    end

    return fig, max_value, min_value
end


function print_divg_at_end(seed_list::Vector{}, folder::String,
        M::Dict{}, exact_mean::T;
        avg_len=5, compute_rela_value=true, verbose=true) where T <: AbstractFloat
        # M contains testcasenum::Any, model_num::Int, n::Int, N::Int, numsample_max::Int, m::Int, ℓ::Int,

    ϕ = M["ϕ"]
    verbose ? println(sprint_experiment_paras(M)) : nothing
    for i in 1:length(seed_list)
        seed = seed_list[i]
        casename = get_experiment_name(M, seed, folder)
        est_fst_m, est_sec_m = load(casename, "est_fst_m", "est_sec_m")
        data = (est_sec_m .- ϕ.f(est_fst_m))
        if compute_rela_value
            data = data ./ ϕ.f(exact_mean)
        end
        @printf("seed = %d, divergence = %.2f±%.2f\n", seed,
                mean(data[(end-avg_len):end]),
                std(data[(end-avg_len):end]))
    end
end

function add_annotate(query_U::Number, query_∇U::Number, time::Number, loc_x::Number;
        loc_y = 1.5, gap=0.1, color=:black, fontsize=10, reduced=false, showtime=false)

    count = -1
    if showtime
        count = count + 1
        text_time = @sprintf("Time: %.0fs", time)
        annotate!(loc_x, loc_y - count*gap, text(text_time, color, :top, fontsize))
    end

    count = count + 1
    text_U = @sprintf("U₁: %s", datasize(query_U))
    annotate!(loc_x, loc_y - count*gap, text(text_U, color, :top, fontsize))

    if !(reduced && abs(query_∇U - 0) < 1.0e-10)
        text_∇U = @sprintf("∇U₁: %s", datasize(query_∇U))
        count = count + 1
        annotate!(loc_x, loc_y - count*gap, text(text_∇U, color, :top, fontsize))
    end

end


"""
Plot the assisted training window
"""
function plot_assist_window(υ::Number, train_step::Int,
        min_value::Number, max_value::Number, label::String; color=:gray, fillalpha=0.3)

    AssistStep = Int64(round(υ*train_step))
    plot!(Array(1:AssistStep), (0.8*min_value)*ones(AssistStep),
        fillrange=(1.2*max_value)*ones(AssistStep),
        fillalpha=fillalpha, c=color, label=label)

end

function show_training_result_various_ansatz_and_m(model_list, m_list, seed_list,
        folder::String, ModelParas::Dict, exact_mean::T, υ::T, train_step::Int, egname::String;
        vanilla_var = Inf, fig_type=:divg,
        start_idx=1, margin=5px, yscale=:log10,
        min_max_value=0.0) where T<:AbstractFloat

    fig  = Array{Any}(undef, length(model_list), length(m_list))
    for i = 1:length(m_list) for j = 1:length(model_list)
        m = m_list[i]; model_num = model_list[j]
        ModelParas["m"] = m; ModelParas["model_num"]=model_num

        #j == 1 ?  ylabel = @sprintf("ℓ=%d, m=%d", ModelParas["ℓ"], m) : ylabel = ""
        tmp_l = ModelParas["ℓ"]; tmp_m = ModelParas["m"]
        j == 1 ?  ylabel = L"\ell=%$(tmp_l), m=%$(tmp_m)" : ylabel = ""
        i == 1 ? title = get_ansatz_name(model_num) : title = ""
        i == 1 && j == 1 ? nolabel=false : nolabel=true

        fig[j,i], max_value, min_value = plot_error_or_divg(seed_list, folder,
                ModelParas, exact_mean, fig_type = fig_type, start_idx = start_idx,
                xlabel="", ylabel=ylabel, title=title, nolabel=nolabel, margin=margin, yscale=yscale)

        if start_idx == 1 # the whole training process
            # show the variance of vanilla importance sampling as a reference.
            if ModelParas["ϕ"].name == "msq" && vanilla_var != Inf
                plot!(1:train_step, vanilla_var*ones(train_step), color=:black,
                    label=(i==1 && j == 1 ? "variance of vanilla IS" : ""))
                max_value = max(max_value, vanilla_var)
            end
            # min_max_value can be used to tune the axis-size.
            max_value = max(min_max_value, max_value)
            label = (i==1&&j==1 ? "Assisted Period" : "")
            plot_assist_window(υ, train_step, min_value, max_value, label)
        end
    end end

    fig_simu = Plots.plot(fig..., size=(300*length(model_list), 200*length(m_list)),
        layout=(@layout grid(length(m_list),length(model_list))), margin=margin)

    fn = @sprintf("assets/train_case_%d_model_%d_phi_%s_type_%s.pdf", ModelParas["testcasenum"],
                                                                     ModelParas["model_num"],
                                                                     ModelParas["ϕ"].name,
                                                                     string(fig_type))

    _savefig(fig_simu, fn)

    println("*"^40)
    printstyled("Empirical divergence for each case during training:\n", color=:blue, bold=true)
    for (m, model_num) in Iterators.product(m_list, model_list)
        ModelParas["m"] = m; ModelParas["model_num"]=model_num
        separate_line_small()
        avg_len = 5
        print_divg_at_end(seed_list, folder, ModelParas, exact_mean, avg_len=avg_len)
        if train_step*(1-υ) <  avg_len
            @warn("υ too large!")
        end
    end

    return fig_simu
end




########################################
########################################


function plot_flow(U₀, U₁, ModelParas, seed, data_folder, xmin, xmax, ymin, ymax;
        figsize=(300,250), num_particle=200,
        rho::Function=(x,y)->nothing, idx1=1, idx2=2, margin=20px)

    filename = get_experiment_name(ModelParas, seed, data_folder)
    # load the flow
    flow, train_stat = load(filename, "flow", "train_stat")

    # plot the flow
    gpts = SharedArray(sampler(U₀, num_particle))
    if rho(0,0) == nothing
        rho = (x,y)->U₁(vcat([x,y],zeros(U₁.dim-2)))
    end
    fig_flow = plot_rho_and_flow(xmin, xmax, ymin, ymax, gpts, flow,
                                 rho, figsize, margin=margin)
                                 # xlabel=L"x_{%$idx1}", ylabel=L"x_{%$idx2}")
    return fig_flow
end

function generate_comparison_data(U₀::Potential{T}, U₁::Potential{T},
        ModelParas::Dict{}, seed::Int, data_folder::String,
        solver, discretize_method::String, offset::Int, exact_mean::T,
        numrepeat::Int, percent_train::Number;
        verbose=true, stat_ais_10=nothing, stat_ais_100=nothing) where T<:AbstractFloat

    filename = get_experiment_name(ModelParas, seed, data_folder)

    # load the flow
    flow, train_stat = load(filename, "flow", "train_stat")
    #train_query = Int64(maximum(train_stat[:query]))
    train_query = Int64(sum(train_stat[:query]))
    query_budget = Int64(round(train_query/percent_train))

    if verbose
        separate_line_big()
        printstyled("Resources and queries used for training\n", bold=true)
        print_train_stat(train_stat)
        printstyled(@sprintf("Budget for each estimate is %s\n", datasize(query_budget)),color=:green)
        separate_line_small()
    end

    τ = 0.1
    if stat_ais_10 == nothing
        stat_ais_10 = estimate_Z_AIS_within_budget(U₀, U₁, 10, τ,
            query_budget, numrepeat, postfix=" (10)")
    else
        @info("we re-use stat_ais_10.")
    end

    if stat_ais_100 == nothing
        stat_ais_100 = estimate_Z_AIS_within_budget(U₀, U₁, 100, τ,
            query_budget, numrepeat, postfix=" (100)")
    else
        @info("we re-use stat_ais_100.")
    end

    stat_neis = estimate_Z_NEIS_within_budget(U₀, U₁, flow,
        query_budget - train_query, ModelParas["N"], numrepeat,
        discretize_method, offset, postfix="", solver=solver)

    df = DataFrame(ais10=stat_ais_10["data"]/exact_mean,
        ais100=stat_ais_100["data"]/exact_mean,
        neis=stat_neis["data"]/exact_mean)

    if verbose
        println(df)
    end

    stat_list = [stat_ais_10, stat_ais_100, stat_neis]
    result_dict = Dict("df"=>df, "stat_ais_10"=>stat_ais_10, "stat_ais_100"=>stat_ais_100,
                  "stat_neis"=>stat_neis, "stat_list"=>stat_list)
    @save(get_cmp_data_name(ModelParas, seed, data_folder), result_dict)

    return result_dict
end

function plot_comparison(ModelParas::Dict{}, seed::Int, data_folder::String;
        loc_y = 1.5, gap = 0.1, ylim=(0.5, 1.5),
        figsize=(650,250), outliers=true) where T<:AbstractFloat

    filename = get_cmp_data_name(ModelParas, seed, data_folder)
    result_dict = load(filename, "result_dict")
    df = result_dict["df"]

    fig_cmp = @df df boxplot([:ais10, :ais100, :neis],
        xticks = (1:1:4, ["AIS-10", "AIS-100", "NEIS"]),
        fillalpha = 0.5, linewidth = 1, label="", size=figsize, outliers=outliers)
    plot!(0:1:4, ones(5), linewidth=1.5, label="", color=:red, linestyle=:dash)
    plot!(xlim=(0.2, 3.8), ylim=ylim)

    fig_cmp_copy = deepcopy(fig_cmp)

    stat_list = result_dict["stat_list"]
    for i = 1:length(stat_list)
        add_annotate(stat_list[i]["query_u"], stat_list[i]["query_gradu"],
            stat_list[i]["time"], i, loc_y = loc_y, gap = gap)
    end

    result_dict["fig_cmp_copy"] = fig_cmp_copy
    result_dict["fig_cmp"] = fig_cmp

    return result_dict
end

function show_comparison_for_all_random_initializations(U₀::Potential{T}, U₁::Potential{T},
        ModelParas::Dict{}, seed_list::Vector{}, data_folder::String,
        xmin::Number, xmax::Number, ymin::Number, ymax::Number,
        solver, discretize_method::String, offset::Int, exact_mean::T,
        numrepeat::Int, percent_train::Number,
        to_compute_data::Bool;
        reuse_ais::Bool=true,
        s1 = (550, 250), s2 = (250, 250), s3 = (800, 250),
        loc_y = 1.5, gap=0.1, ylim=(0.5,1.5), outliers=true,
        verbose=true, rho::Function=(x,y)->nothing) where T<:AbstractFloat

    ########################################
    # Get comparison data
    result = Array{Any}(undef, length(seed_list))
    stat_ais_10_copy = nothing; stat_ais_100_copy = nothing;
    for i in 1:length(seed_list)
        seed = seed_list[i]
        fig_flow = plot_flow(U₀, U₁, ModelParas, seed, data_folder, xmin, xmax, ymin, ymax,
                             rho=rho, figsize=(300,200))
        #_savefig(fig_flow, get_short_name(ModelParas, "assets/flows/flow_")*"_"*string(seed)*".pdf")

        if to_compute_data || !(isfile(get_cmp_data_name(ModelParas, seed, data_folder)))
            # get the data again
            result[i] = generate_comparison_data(U₀, U₁, ModelParas, seed, data_folder,
                    solver, discretize_method, offset, exact_mean, numrepeat, percent_train,
                    stat_ais_10 = stat_ais_10_copy, stat_ais_100 = stat_ais_100_copy)
        else
            # if data already exists, just load the data.
            result[i] = load(get_cmp_data_name(ModelParas, seed, data_folder), "result_dict")
            if verbose
                print(result[i]["df"])
            end
        end

        if reuse_ais
            # if we can reuse AIS estimates for the previous seed, then we should save it for later use.
            stat_ais_10_copy = deepcopy(result[i]["stat_ais_10"])
            stat_ais_100_copy = deepcopy(result[i]["stat_ais_100"])
        end

        result[i] = plot_comparison(ModelParas, seed, data_folder,
                                    loc_y=loc_y, gap=gap, ylim=ylim, figsize=s1, outliers=outliers)
        result[i]["fig_flow"] = fig_flow
    end

    ########################################
    # combine pictures together
    fig = Array{Any}(undef, length(seed_list))
    for i = 1:length(seed_list)
        seed = seed_list[i]
        plot!(result[i]["fig_cmp"], ylabel=@sprintf("trial = %d", i), size=s1, margin=10px)
        plot!(result[i]["fig_flow"], legend=false, colorbar=false, size=s2)
        fig[i] = plot(result[i]["fig_cmp"], result[i]["fig_flow"],
            layout=grid(1, 2, widths=[0.65, 0.3]), size=s3, margin=10px)
        _savefig(fig[i], get_short_name(ModelParas, "assets/cmp_")*"--"*string(seed)*".pdf")
    end
    fig_cmp_all = plot(fig..., layout=grid(length(seed_list),1),
                       size=(800, 250*length(seed_list)), margin=20px)
    _savefig(fig_cmp_all, get_short_name(ModelParas, "assets/cmp_")*".pdf")

    ########################################
    # Display statistics of estimates
    printstyled("Statistics of independent estiamtes\n", color=:green, bold=true)
    printstyled(sprint_experiment_paras(ModelParas), color=:green, bold=true)
    for r in result
        @printf("AIS-10  %.4f ± %.4f\n", mean(r["df"][!,:ais10]), std(r["df"][!,:ais10]))
        @printf("AIS-100  %.4f ± %.4f\n", mean(r["df"][!,:ais100]), std(r["df"][!,:ais100]))
        @printf("NEIS  %.4f ± %.4f\n", mean(r["df"][!,:neis]), std(r["df"][!,:neis]))
        separate_line_small()
    end
end

function print_data(m_list, model_list, seed_list, ModelParas, data_folder)
    #for s in ["±","\\pm"]
    for s in ["±"]
        for (m, model_num) in Iterators.product(m_list, model_list)
            ModelParas["m"] = m; ModelParas["model_num"] = model_num
            result = Array{Any}(undef, length(seed_list))
            for i in 1:length(seed_list)
                seed = seed_list[i]
                result[i] = load(get_cmp_data_name(ModelParas, seed, data_folder), "result_dict")
            end
            printstyled(sprint_experiment_paras(ModelParas), color=:green, bold=true)
            for r in result
                @printf("AIS-10  %.3f %s %.3f -- %.3f\n", mean(r["df"][!,:ais10]), s,
                        std(r["df"][!,:ais10]), median(r["df"][!,:ais10]))
                @printf("AIS-100  %.3f %s %.3f -- %.3f\n", mean(r["df"][!,:ais100]), s,
                        std(r["df"][!,:ais100]), median(r["df"][!,:ais100]))
                @printf("NEIS  %.3f %s %.3f -- %.3f\n", mean(r["df"][!,:neis]), s,
                        std(r["df"][!,:neis]), median(r["df"][!,:neis]))
                separate_line_small()
            end
        end
        println("%"^40)
    end
end

end
