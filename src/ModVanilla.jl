module ModVanilla

using Statistics
using ModPotential

export vanilla_importance_sampling

function vanilla_importance_sampling(U₀, U₁, numsample)
    estimator(x) = exp(-U₁.U(x))/exp(-U₀.U(x))
    data = map(j->estimator(U₀.sampler(j)), 1:numsample)
    m = mean(data)
    m2 = mean(data.^2)
    return data, m, m2 - m^2
end


end
