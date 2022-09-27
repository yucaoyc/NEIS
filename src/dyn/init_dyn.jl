export init_rand_dyn

function init_rand_dyn(model_num, n, m, scale, U₁)
    if model_num == 0
        flow = init_random_DynNNGenericOne(n, scale)
    elseif model_num == 1
        flow = init_random_DynNNGenericTwo(n, m, scale)
    elseif model_num == 2
        flow = init_random_DynNNGradTwo(n, m, scale)
    elseif model_num == 4
        β = 2.0; α = 2.0
        flow = init_funnelexpansatz(n, β, α, U₁.Ω)
    else
        @error("Not implemented yet!")
    end
    return flow
end
