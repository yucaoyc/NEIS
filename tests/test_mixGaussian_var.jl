using NEIS
using LinearAlgebra
using Printf
using Test

function random_mu(n)
    x = randn(n)
    x /= norm(x)
    x *= rand()
    return x
end

@testset "test mean_msec_var" begin
    for trial = 1:5
        n = 1 + Int64(round(rand()*3))
        mode = 1+Int64(round(rand()*3))
        ωlist = rand(mode)
        ωlist /= sum(ωlist)
        μlist = [random_mu(n) for i = 1:mode]
        Σlist = [randHermitian(n, 0.3, 1.0) for i = 1:mode]

        # theoretical result
        m, _, exact_var = mean_msec_var(ωlist, μlist, Σlist)

        println("="^40)

        numsample = 10^8
        # numerical result
        U₀ = Gaussian(n, zeros(n), 1.0)
        U₁ = generate_mixGaussian(n, μlist, Σlist, ωlist)
        @time _, num_m, num_var = vanilla_importance_sampling(U₀, U₁, numsample)

        @printf("Dim = %d, mode = %d\n", n, mode)
        @printf("Exact mean = %.4E Numerical mean = %.4E\n", m, num_m)
        @printf("Exact var = %.4E Numerical var = %.4E\n", exact_var, num_var)

        @test abs(m - num_m)/abs(m) < 0.05
        @test abs(exact_var - num_var)/abs(exact_var) < 0.05
    end
end
