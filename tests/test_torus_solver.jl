using FFTW
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using Test
using Printf
using BasicInterpolators
using NEIS

ϵ = 1.0e-4

# function V and b = ∇V
function V(x, coef, kmat, N)
    v = 0.0
    for i = 1:N
        for j = 1:N
            v += coef[i,j]*exp(-2*π*complex(0,1)*dot(kmat[i,j],x))
        end
    end
    if abs(imag(v)) > ϵ
        @warn("imag part is large in V")
        println(x)
        println(v)
    end

    return real(v)
end

function b(x, coef, kmat, N)
    v = complex(zeros(2))
    for i = 1:N
        for j = 1:N
            v += (-2*π*complex(0,1)*kmat[i,j])*coef[i,j].*exp(-2*π*complex(0,1)*dot(kmat[i,j],x))
        end
    end
    if norm(imag(v)) > ϵ
        @warn("imag part is large in b")
        println(x)
        println(v)
    end
    return real(v)
end


N = 2^9
λ = 0.2
a = 2.0
mode=3
kmat, xgrid, q₀, ρ₀, q₁, ρ₁, ρdiff, exact_mean = load_torus_eg(N, λ, a, mode=mode);
coef, V_interp, b_interp, flow = solve_poisson_2dtorus_fft(N, xgrid, ρdiff, kmat);

# test solving poisson's equation

h = 1.0e-3
e1 = [1.0, 0.0]
e2 = [0.0, 1.0]

# Check that the above function V is correct!

@time @testset "solve poissons" begin
    count = 0
    total_case = 100
    for i = 1:total_case
        x₀ = rand(2)*(1-2*h) .+ h
        dVxx = (V(x₀.+h*e1, coef, kmat, N) + V(x₀.-h*e1, coef, kmat, N) -2*V(x₀, coef, kmat, N))/h^2
        dVyy = (V(x₀.+h*e2, coef, kmat, N) + V(x₀.-h*e2, coef, kmat, N) -2*V(x₀, coef, kmat, N))/h^2

        v1 = dVxx + dVyy
        v2 = ρdiff(x₀)

        if abs(v2) < 1.0e-3 && abs(v1) < 1.0e-3
            smallv = true
            pass = true
            @info("Small v case")
            count += 1
        else
            smallv = false
            if abs(v1/v2 - 1.0) < 1.0e-2
                pass = true
            else
                pass = false
                println([v1, v2])
            end
        end
        @test pass
    end
    printstyled(@sprintf("Percent of exceptions is %f\n", count/total_case), color=:yellow)
end

# Check that the above function b is correct!

@time @testset "b = ∇V" begin
    for i = 1:10
        x₀ = rand(2)*(1-2*h) .+ h
        dVx = (V(x₀.+h*e1, coef, kmat, N) - V(x₀.-h*e1, coef, kmat, N))/(2*h)
        dVy = (V(x₀.+h*e2, coef, kmat, N) - V(x₀.-h*e2, coef, kmat, N))/(2*h)
        @test norm([dVx, dVy] .- b(x₀, coef, kmat, N)) < 10*h^2
    end
end

# Check that the interpolation is correct!

@time @testset "test interpolator" begin
    for i = 1:10
        x₀ = rand(2).*0.9 .+ 0.05
        #println(x₀)
        @test (abs(V_interp(x₀[1], x₀[2])/V(x₀, coef, kmat, N)-1.0) < 1.0e-2)
        @test (norm(b_interp(x₀) .- b(x₀, coef, kmat, N))/norm(b(x₀, coef, kmat, N)) < 5.0e-2)
    end
end
