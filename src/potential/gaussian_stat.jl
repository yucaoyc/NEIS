export mean_msec_var, print_result

"""
Suppose we use direct importance sampling.
Assume that ρ₀ is standard Gaussian and ρ₁ is a Gaussian mixture.
This function compute the mean and second moment for Gaussian mixtures
"""
function mean_msec_var(ωlist, μlist, Σlist)
    # return Z_1/Z_0 and the second moment
    L = length(ωlist)
    n = length(μlist[1]) # dimension

    exact_mean = sum([ωlist[i]*sqrt(det(Σlist[i])) for i = 1:L])
    exact_msec = 0.0
    for i = 1:L
        for j = 1:L
            Aij = inv(Σlist[i]) + inv(Σlist[j]) - Matrix(1.0I, n, n)
            θij = inv(Aij)*(inv(Σlist[i])*μlist[i] + inv(Σlist[j])*μlist[j])
            Bij = -1/2*dot(μlist[i], inv(Σlist[i])*μlist[i]) - 1/2*dot(μlist[j], inv(Σlist[j])*μlist[j])
            Bij += 1/2*dot(θij, Aij*θij)
            exact_msec += ωlist[i]*ωlist[j]*exp(Bij)/sqrt(det(Aij))
        end
    end
    exact_var = exact_msec - exact_mean^2
    return exact_mean, exact_msec, exact_var
end

function print_result(weight, μlist, Σlist)
    m, msec, var = mean_msec_var(weight, μlist, Σlist)
    @printf "%.2f %.2E %.2E\n" m msec var
    rvar = var/m^2
    if rvar > 1.0E3
        @printf "Relative var: %.2E\n" rvar
    else
        @printf "Relative var: %.2f\n" rvar
    end
    println("="^40)
end
