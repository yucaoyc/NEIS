export get_ais_samplesize, get_opt_int_samplesize, get_opt_ode_samplesize

function get_ais_samplesize(K, budget; verbose=true)
    s = Int64(round(budget/(K+1)/2))
    verbose ? @printf("sample size = %s\n", datasize(s)) : nothing
    return s
end

function get_opt_int_samplesize(N, budget; verbose=true)
    s = Int64(round(budget/(2*(N+1))))
    verbose ? @printf("sample size = %s\n", datasize(s)) : nothing
    return s
end

function get_opt_ode_samplesize(N, budget, solver; verbose=true)
    if solver == RK4
        s = Int64(round(budget/(N*4)))
    elseif solver == MM
        s = Int64(round(budget/(N*2)))
    else
        @assert("Not implemented!")
    end
    verbose ? @printf("sample size = %s\n", datasize(s)) : nothing
    return s
end
