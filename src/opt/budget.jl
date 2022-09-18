export get_ais_samplesize, get_opt_int_samplesize, get_opt_ode_samplesize

function get_ais_samplesize(K, budget; grad_only=true)
    if grad_only
        return Int64(round(budget/(2*K)))
    else
        return Int64(round(budget/(3*K)))
    end
end

function get_opt_int_samplesize(N, budget)
    return Int64((budget/(2*(N+1))))
end

function get_opt_ode_samplesize(N, budget, solver)
    if solver == RK4
        return Int64(round(budget/(N*4)))
    elseif solver == MM
        return Int64(round(budget/(N*2)))
    else
        @assert("Not implemented!")
    end
end
