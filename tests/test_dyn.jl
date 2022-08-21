push!(LOAD_PATH,"../src")
using NEIS
using Test
using LinearAlgebra

# todo: test the matrix version of gradients.

function rand_array(x::AbstractArray{T}) where T<:AbstractFloat
    r = rand(T,size(x)) .- T(1/2)
    r = r/norm(r)
    return r
end

function rand_para(θlist::AbstractArray{}) where T<:AbstractFloat
    δθ_list = [rand_array(θ) for θ in θlist]
    δθ_vec = [vec(θ) for θ in δθ_list]
    vec_δθ = vcat(δθ_vec...)
    return δθ_list, vec_δθ
end

function rand_para(θlist::AbstractArray{}, i::Int) where T<:AbstractFloat
    δθ_list = [j==i ? rand_array(θlist[j]) : zeros(T,size(θlist[j])) for j = 1:length(θlist)]
    δθ_vec = [vec(θ) for θ in δθ_list]
    vec_δθ = vcat(δθ_vec...)
    return δθ_list, vec_δθ
end

function test_flow(b::DynTrain{T}; ϵ=T(1.0e-3), err_tol = T(5.0e-3)) where T<:AbstractFloat
    n = b.dim
    
    if hasproperty(b, :V)
       @testset "test 𝐛 = ∇V" begin
            for i = 1:10
                x = randn(T,n)
                δx = rand_array(x)
                grad_FD = (b.V(x+ϵ*δx, b.para_list...) - b.V(x-ϵ*δx, b.para_list...))/(2*ϵ)
                grad = dot(b(x), δx)
                @test norm(grad_FD .- grad) < err_tol
            end
        end
    end

    @testset "test ∇𝐛" begin
        for i = 1:10
            x = randn(T,n)
            δx = rand_array(x)
            grad_FD = (b.f(x+ϵ*δx, b.para_list...) - b.f(x-ϵ*δx, b.para_list...))/(2*ϵ)
            grad = ∇b(b, x)*δx
            @test norm(grad_FD - grad) < err_tol
        end
    end
    
    @testset "test ∇⋅𝐛" begin
       for i = 1:10
            x = randn(T,n)
            @test norm(tr(∇b(b, x)) - divg_b(b, x)) < err_tol
        end
    end
    
    @testset "test ∇(∇⋅𝐛)" begin
        for i = 1:10
            x = randn(T,n)
            δx = rand_array(x)
            grad_FD = (divg_b(b,x+ϵ*δx) - divg_b(b,x-ϵ*δx))/(2*ϵ)
            grad = dot(grad_divg_b(b, x),δx)
            @test norm(grad_FD - grad) < err_tol
        end
    end
    
    @testset "test ∇_θ 𝐛_θ" begin
        for k = 1:10
            x = randn(T,n)
            δθ_list, vec_δθ = rand_para(b.para_list)
            for i = 1:length(b.num_para)
               b.para_list[i] .+= ϵ*δθ_list[i] 
            end
            grad1 = b(x)
            for i = 1:length(b.num_para)
               b.para_list[i] .-= 2*ϵ*δθ_list[i] 
            end
            grad2 = b(x)
            @test norm((grad1 - grad2)/(2*ϵ) - grad_b_wrt_para(b, x)*vec_δθ) < err_tol
        end
    end
    
    @testset "test ∇_θ (∇⋅𝐛_θ)" begin
        for k = 1:10
            x = randn(T,n)
            δθ_list, vec_δθ = rand_para(b.para_list)
            for i = 1:length(b.num_para)
               b.para_list[i] .+= ϵ*δθ_list[i] 
            end
            grad1 = divg_b(b, x)
            for i = 1:length(b.num_para)
               b.para_list[i] .-= 2*ϵ*δθ_list[i] 
            end
            grad2 = divg_b(b, x)
            grad = (grad1 - grad2)/(2*ϵ)
            
            @test norm(grad - dot(grad_divg_wrt_para(b, x),vec_δθ)) < err_tol
        end
    end
end

@testset "test dyn" begin 
    for i = 1:10
        dim = Int64(ceil(2+10*rand()))
        m = Int64(ceil(2+10*rand()))
        if i <= 5
            T = Float32
        else
            T = Float64
        end
        
        Ω = x->domain_ball(x, T(25.0))
        b_generic_two = init_random_DynNNGenericTwo(dim, m, convert=x->T.(x))
        b_generic_one = init_random_DynNNGenericOne(dim, convert=x->T.(x))
        b_grad_two = init_random_DynNNGradTwo(dim, m, convert=x->T.(x))
        b_funnel = init_funnelexpansatz(dim, T(1.0), T(1.0), Ω)
    
        test_flow(b_generic_one)
        test_flow(b_grad_two)
        test_flow(b_generic_two)
        test_flow(b_funnel)
    end
end
;
