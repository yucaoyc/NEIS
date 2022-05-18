export generate_Gaussian_Potential,
    generate_mixGaussian_Potential,
    mean_msec_var,
    print_result

# Generate Potentials with fixed parameter.
function generate_Gaussian_Potential(dim::Int64, μ::Array{Float64,1},
        Σ::Union{Array{Float64,2},Diagonal{}}; func_sampler=nothing)
    # return a potential U with Gaussian parameters μ, Σ

    Σinv = inv(Σ)
    U(x) = 0.5*dot(x-μ, Σinv*(x-μ))
    gradU(x) = Σinv*(x-μ)
    HessU(x) = Σinv
    LaplaceU(x) = tr(Σinv)
    fixpara = [μ, Σ, Σinv]

    if func_sampler == nothing
        Σhalf = Σ^(1/2)
        sampler = (z) -> (μ + Σhalf*randn(dim))
    else
        sampler = func_sampler
    end

    return FixedPotential(dim, U, gradU, HessU, LaplaceU, fixpara, sampler)
end

function generate_Gaussian_Potential(dim::Int64, μ::Array{Float64,1},
        σsq::Float64; func_sampler=nothing)
    # return a potential U with Gaussian parameters μ, Σ

    U(x) = norm(x-μ)^2/(2*σsq)
    gradU(x) = (x-μ)/σsq
    HessU(x) = Matrix(1.0I,dim,dim)/σsq
    LaplaceU(x) = dim/σsq
    fixpara = [μ, σsq]

    if func_sampler == nothing
        σhalf = sqrt(σsq)
        sampler = (z) -> (μ + σhalf*randn(dim))
    else
        sampler = func_sampler
    end

    return FixedPotential(dim, U, gradU, HessU, LaplaceU, fixpara, sampler)
end

function generate_mixGaussian_Potential(dim::Int64, μlist, Σlist, weightlist::Array{Float64,1})
    # return a potential U for Gaussian mixture formed by a list of μ, Σ, and weights
    # we assume \sum_{j} weightlist[j]*exp(-U_j(x)) =: e^{-U(x)}

    m = length(μlist)
    Ulist = [generate_Gaussian_Potential(dim,μlist[j],Σlist[j]) for j=1:m]

    U(x) = mixedPotential_Potential(x, Ulist, weightlist, 1.0)
    gradU(x) = mixedPotential_Grad(x, Ulist, weightlist, 1.0)
    HessU(x) = mixedPotential_Hess(x, Ulist, weightlist, 1.0)
    LaplaceU(x) = mixedPotential_Laplace(x, Ulist, weightlist, 1.0)
    fixpara = [μlist, Σlist, weightlist]

    return FixedPotential(dim, U, gradU, HessU, LaplaceU, fixpara, nothing)
end


########################################
# compute the mean and second moment for Gaussian mixtures

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
