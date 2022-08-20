using Statistics

export vanilla_importance_sampling

function vanilla_importance_sampling(U₀::Potential, U₁::Potential, numsample::Int)
    #estimator(x) = exp(-U₁.U(x))/exp(-U₀.U(x))
    estimator(x) = exp(-U(U₁,x))/exp(-U(U₀,x))
    data = map(j->estimator(sampler(U₀,1)[:]), 1:numsample)
    m = mean(data)
    m2 = mean(data.^2)
    return data, m, m2 - m^2
end
