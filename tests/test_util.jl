push!(LOAD_PATH,"../src")
using NEIS
using Test
using LinearAlgebra
using ForwardDiff

@testset "test mul and div" begin
    a = randn(3,10)
    b = randn(10)
    c1 = divide_col(a, b)
    d1 = multiply_col(a, b)
    c2 = zeros(3,10)
    d2 = zeros(3,10)
    for i = 1:10
        c2[:,i] = a[:,i]/b[i]
        d2[:,i] = a[:,i]*b[i]
    end

    @test norm(c2 - c1) <1.0e-8
    @test norm(d2 - d1) < 1.0e-8
end

@testset "test nn" begin
    g(x) = ForwardDiff.derivative(z->sigmoid(z), x)
    for i = 1:10
        x0 = randn(10)
        @test norm(sigmoid_deri.(x0) .- g.(x0)) < 1.0e-10
    end
    g2(x) = ForwardDiff.derivative(z->sigmoid_deri(z), x)
    for i = 1:10
        x0 = randn(10)
        @test norm(sigmoid_sec_deri.(x0) .- g2.(x0)) < 1.0e-10
    end
end
