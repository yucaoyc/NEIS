export train_flow

function train_flow(U₀::Potential{T}, U₁::Potential{T}, flow::DynTrain{T},
        N::Int, numsample_max::Int, train_step::Int,
        lr::T, decay::T, discretize_method::String, offset::Int,
        train_paras::Any;
        save_data::Bool=false,
        filename::String="data.jld2",
        gpts::AbstractMatrix=Matrix{Any}(undef,0,0),
        gpts_sampler::Function=()->nothing,
        verbose=true) where T <: AbstractFloat

    flow_bef = deepcopy(flow)

    if length(gpts) == 0
        gpts = SharedArray(sampler(U₀, numsample_max))
    end
    if gpts_sampler() == nothing
        gpts_sampler = ()->sampler(U₀, 1)[:]
    end

    reset_query_stat(U₁)
    if discretize_method == "ode"
        train_time = @elapsed est_fst_m, est_sec_m, est_grad_norm, est_para =
            train_NN_ode(U₀, U₁, flow, N, numsample_max,
                train_step, lr, decay, gpts, gpts_sampler; train_paras...)
    elseif discretize_method == "int"
        train_time = @elapsed est_fst_m, est_sec_m, est_grad_norm, est_para =
            train_NN_int(U₀, U₁, flow, N, numsample_max,
                train_step, lr, decay, gpts, gpts_sampler;
                offset = offset, train_paras...)
    else
        @error("Please use correct discretize_method: ode or int.")
    end

    train_stat = get_train_stat(train_time, U₁)
    if verbose
        print_train_stat(train_stat)
    end

    if save_data
        @save(filename, est_fst_m, est_sec_m, est_grad_norm, est_para, flow, flow_bef,
          train_step, train_stat, train_paras)
    end

    return est_fst_m, est_sec_m, est_grad_norm, est_para, train_stat
end
