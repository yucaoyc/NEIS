push!(LOAD_PATH,"../src")

using Test
using Random
using LinearAlgebra
using BenchmarkTools
using NEIS

ϵ = Float32(1.0e-4)
num_x = 10^2

function test_grad(H::Potential{T}, dim) where T<:AbstractFloat
    for k = 1:num_x
        x = rand(T,dim) .- T(1/2)
        a = U(H, x)
        b = ∇U(H, x)
        c = HessU(H, x)
        d = LaplaceU(H,x)
        @test abs(tr(c) - d) < ϵ
        δ = Float32(1.0e-3)
        δx = randn(T,dim); δx = δx/norm(δx);
        δx .*= δ
        newx = x .+ δx
        err =  abs(U(H,newx) - U(H,x) - dot(b, δx) - dot(δx, c*δx)/2)
        if T == Float64
            @test err < 0.1*δ^2
        else
            @test err < 1.0e-6
        end
    end
end

@testset "gaussian" begin
    dim = 3
    μ = randn(Float32,dim)
    σsq = 2.0f0
    Sigma = σsq*Matrix(Float32(1.0)I, dim, dim);

    p1 = Gaussian(dim, μ, σsq);
    p2 = Gaussian(dim, μ, Sigma);

    for i = 1:10
        x = rand(Float32,dim) .- 1.0f0/2
        # Test U
        @test norm(U(p1, x) - U(p2,x)) < 1.0e-5
        # Test ∇U
        @test norm(∇U(p1, x) .- ∇U(p2,x)) < 1.0e-5
        # Test ∇^2 U
        @test norm(HessU(p1,x) .- HessU(p2,x)) < 1.0e-5
        # Test Laplace U
        @test norm(LaplaceU(p1,x) - LaplaceU(p2,x)) < 1.0e-5
        x = rand(Float32,dim,10) .- 1.0f0/2
        # Test vectorize U
        @test norm(U(p1, x) .- U(p2, x)) < 1.0e-5
        # Test vectorize ∇U
        @test norm(∇U(p1, x) .- ∇U(p2, x)) < 1.0e-5
        # Test vectorize Laplace U.
        @test norm(LaplaceU(p1, x) .- LaplaceU(p2, x)) < 1.0e-5
    end
    test_grad(p1, dim)
end

@testset "mixed Gaussian" begin
    Random.seed!(2)
    testnum = 10
    for j = 1:testnum
	if j < testnum/2 
	    T = Float32
	else
	    T = Float64
	end
        dim = Int64(ceil(rand()*10+2))
        m = Int64(ceil(rand()*5+1))
        μlist = [randn(T,dim) for k = 1:m]
        Σlist = [T.(randHermitian(dim, 0.1+0.9*rand(), 1+rand())) for k = 1:m]
        Ulist = [Gaussian(dim, μlist[k], Σlist[k]) for k=1:m]
        weightlist = rand(T,m)

        H = generate_mixGaussian(dim, μlist, Σlist, weightlist)
        β = T(0.1 + rand())
        # test gradient implementations.
        test_grad(H, dim)

        # test the vectorized version.
        num_particle = 10
        x = randn(T, dim, num_particle)
        @test norm(U(H,x) - [U(H,x[:,j]) for j = 1:num_particle]) < 1.0e-5
        @test norm(∇U(H,x) - hcat([∇U(H,x[:,j]) for j = 1:num_particle]...))<1.0e-5
    end
end

@testset "test funnel " begin
    n = 10
    σf = 3.0
    U₁ = Funnel(n, σf);
    for i = 1:10
        x = randn(n)
        δx = randn(n)
        δx = δx/norm(δx)
        ϵ = 1.0e-4
        @test norm((U(U₁,x.+ϵ*δx) - U(U₁,x.-ϵ*δx))/(2*ϵ) - dot(∇U(U₁,x), δx)) < ϵ
        @test norm((∇U(U₁, x.+ϵ*δx) - ∇U(U₁,x.-ϵ*δx))/(2*ϵ) - HessU(U₁,x)*δx) < ϵ
        @test norm(LaplaceU(U₁,x) - tr(HessU(U₁,x))) < ϵ
    end
end
;
