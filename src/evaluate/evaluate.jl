export get_cmp_stat, estimate_Z_AIS_within_budget, estimate_Z_NEIS_within_budget

"""
Given query budget constraint (as specified in budget),
we evaluate Z by AIS using K steps and time step τ in MALA.
numrepeat: number of times to repeat the evaluation

Remark: "budget" is the budget for each evaluation, not the total budget.

Return: a dictionary that contains
- "data": data of evaluations
- "query_u": query to U₁ per experiment
- "query_gradu": query to ∇U₁ per experiment
- "time": runtime per experiment
- "name": name of the method.
"""
function estimate_Z_AIS_within_budget(U₀::Potential{T}, U₁::Potential{T},
        K::Int, τ::T, budget::Int, numrepeat::Int;
        postfix::String="",
        verbose=true,
        sep_str="-") where T<:AbstractFloat

    # sep_str means a string for space separation.
    reset_query_stat(U₁)
    numsample = get_ais_samplesize(K, budget, grad_only=true)
    fun_ais = m -> ais_neal(U₀, U₁, K, m, τ = τ)[2]
    time_ais = @elapsed data_ais = map(j->fun_ais(numsample), 1:numrepeat)
    verify_budget(U₁, numrepeat*budget, grad_only=true)

    stat_ais = get_cmp_stat(U₁, numrepeat, time_ais, "AIS"*postfix, verbose=verbose, str=sep_str)
    stat_ais["data"] = data_ais

    return stat_ais

end

"""
Given query budget constraint (as specified in budget),
we evaluate Z by NEIS using the flow.
N : number of time grid point
numrepeat: number of times to repeat the evaluation
discretize_method : "ode" or "int"
offset: when discretize_method = "ode" is ued, offset does not matter;
    otherwise, offset is a value between 0 and N.

Remark: "budget" is the budget for each evaluation, not the total budget.

Return: a dictionary that contains
- "data": data of evaluations
- "query_u": query to U₁ per experiment
- "query_gradu": query to ∇U₁ per experiment
- "time": runtime per experiment
- "name": name of the method.
"""
function estimate_Z_NEIS_within_budget(U₀::Potential{T}, U₁::Potential{T},
        flow::Dyn, budget::Int, N::Int, numrepeat::Int,
        discretize_method::String, offset::Int;
        postfix::String="",
        solver::Function=RK4,
        verbose=true,
        sep_str="-") where T<:AbstractFloat

    reset_query_stat(U₁)
    if discretize_method == "ode"
        numsample = get_opt_ode_samplesize(N, budget, solver)
        fun_neis = M -> get_data_err_var(U₀, U₁, flow, N, M, solver=solver)[2]
    else
        numsample = get_opt_int_samplesize(N, budget)
        fun_neis = M -> get_data_err_var(U₀, U₁, flow, N, offset, M, solver=solver)[2]
    end

    time_neis = @elapsed data_neis = map(j->fun_neis(numsample),1:numrepeat)
    verify_budget(U₁, numrepeat*budget)

    stat_neis = get_cmp_stat(U₁, numrepeat, time_neis, "NEIS"*postfix,
                             verbose=verbose, str=sep_str)
    stat_neis["data"] = data_neis

    return stat_neis
end

"""
An subroutine used in the above two functions.
It is used to collect some query information and time.
"""
function get_cmp_stat(U₁, numrepeat, time, method_name; verbose=true, str="-")

    stat = Dict("query_u"=>get_query_stat(U₁)[1]/numrepeat,
        "query_gradu"=>get_query_stat(U₁)[2]/numrepeat,
        "time"=>time/numrepeat,
        "name"=>method_name)

    if verbose
        @printf("Queries per experiment\n")
        print_query_stat(U₁, repeat_num = numrepeat)
        @printf("Average runtime for %s with %d trials is %.3f (s)\n", method_name, numrepeat, time/numrepeat)
        @printf("Total runtime for %s with %d trials is %.3f (s)\n", method_name, numrepeat, time)
        println(str^40)
    end
    return stat

end
